#!/usr/bin/env python3
"""Prepare the pancreas quiz data for nnU-Net v2 without changing source files.

The source archive has labelled ``train`` and ``validation`` directories, each
split into ``subtype0``/``subtype1``/``subtype2``, plus an unlabelled ``test``
directory.  nnU-Net expects labelled cases in ``imagesTr``/``labelsTr`` and
test images in ``imagesTs``.  This script copies images, repairs floating-point
segmentation labels by validated rounding, and writes the metadata needed by
both the segmentation and classification tasks.

There are two validation layouts:

* ``imagesTr`` (default) puts train and supplied validation cases in imagesTr
  and writes a one-fold ``splits_final.json``.  This is the layout used for
  preprocessing and training.
* ``separate`` puts only the supplied training cases in imagesTr and stages the
  validation cases in imagesVal/labelsVal.  Use it to extract a fingerprint
  and plan from training data only.  After planning, rerun with ``imagesTr``
  before preprocessing; the generated plans remain in place.

The original source tree is opened read-only.  All writes are atomic and occur
below ``--output-root/DatasetXXX_Name``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NIFTI_SUFFIX = ".nii.gz"
CHANNEL_SUFFIX = "_0000.nii.gz"
CLASS_TO_LABEL: dict[str, int] = {
    "subtype0": 0,
    "subtype1": 1,
    "subtype2": 2,
}
SEGMENTATION_LABELS: dict[str, int] = {
    "background": 0,
    "pancreas": 1,
    "lesion": 2,
}
SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SAFE_DATASET_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class PreparationError(RuntimeError):
    """A user-correctable dataset or configuration error."""


@dataclass(frozen=True)
class LabeledCase:
    case_id: str
    image: Path
    label: Path
    class_name: str
    class_id: int
    split: str


@dataclass(frozen=True)
class TestCase:
    case_id: str
    image: Path


@dataclass(frozen=True)
class Inventory:
    train: tuple[LabeledCase, ...]
    validation: tuple[LabeledCase, ...]
    test: tuple[TestCase, ...]

    @property
    def labeled(self) -> tuple[LabeledCase, ...]:
        return self.train + self.validation


@dataclass(frozen=True)
class Geometry:
    shape: tuple[int, ...]
    affine: np.ndarray
    zooms: tuple[float, ...]


@dataclass(frozen=True)
class MaskAudit:
    case_id: str
    split: str
    raw_values: tuple[float, ...]
    rounded_values: tuple[int, ...]
    rounded_voxels: int
    max_rounding_error: float
    voxel_counts: dict[int, int]


@dataclass(frozen=True)
class Config:
    source: Path
    output_root: Path
    dataset_id: int
    dataset_name: str
    validation_layout: str = "imagesTr"
    dry_run: bool = False
    rounding_tolerance: float = 1e-3
    expected_train: int | None = 252
    expected_validation: int | None = 36
    expected_test: int | None = 72

    @property
    def dataset_directory_name(self) -> str:
        return f"Dataset{self.dataset_id:03d}_{self.dataset_name}"

    @property
    def destination(self) -> Path:
        return self.output_root / self.dataset_directory_name


def _strip_nifti_suffix(name: str) -> str:
    if not name.endswith(NIFTI_SUFFIX):
        raise PreparationError(f"Not a .nii.gz filename: {name}")
    return name[: -len(NIFTI_SUFFIX)]


def _validate_case_id(case_id: str, path: Path) -> None:
    if not SAFE_CASE_ID.fullmatch(case_id):
        raise PreparationError(
            f"Invalid case identifier {case_id!r} from {path}. "
            "Use only letters, digits, underscores, and hyphens."
        )
    if case_id.endswith("_0000"):
        raise PreparationError(
            f"Case identifier {case_id!r} incorrectly contains a channel suffix: {path}"
        )


def _visible_directories(path: Path) -> set[str]:
    return {
        item.name
        for item in path.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    }


def discover_labeled_split(source: Path, split: str) -> tuple[LabeledCase, ...]:
    """Discover and strictly pair image/label files in one labelled split."""
    split_root = source / split
    if not split_root.is_dir():
        raise PreparationError(f"Missing source directory: {split_root}")

    unknown_directories = _visible_directories(split_root) - set(CLASS_TO_LABEL)
    if unknown_directories:
        raise PreparationError(
            f"Unexpected class directories in {split_root}: "
            f"{sorted(unknown_directories)}"
        )

    cases: list[LabeledCase] = []
    seen: dict[str, Path] = {}
    for class_name, class_id in CLASS_TO_LABEL.items():
        class_root = split_root / class_name
        if not class_root.is_dir():
            raise PreparationError(f"Missing class directory: {class_root}")

        unsupported_files = sorted(
            item.name
            for item in class_root.iterdir()
            if item.is_file()
            and not item.name.startswith(".")
            and not item.name.endswith(NIFTI_SUFFIX)
        )
        if unsupported_files:
            raise PreparationError(
                f"Unexpected non-NIfTI files in {class_root}: {unsupported_files}"
            )

        images: dict[str, Path] = {}
        labels: dict[str, Path] = {}
        for path in sorted(class_root.glob(f"*{NIFTI_SUFFIX}")):
            if path.name.startswith("._"):
                continue
            if path.name.endswith(CHANNEL_SUFFIX):
                case_id = path.name[: -len(CHANNEL_SUFFIX)]
                target = images
            else:
                case_id = _strip_nifti_suffix(path.name)
                target = labels
            _validate_case_id(case_id, path)
            if case_id in target:
                raise PreparationError(
                    f"Duplicate {'image' if target is images else 'label'} for "
                    f"case {case_id}: {target[case_id]} and {path}"
                )
            target[case_id] = path

        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        if missing_labels or missing_images:
            details: list[str] = []
            if missing_labels:
                details.append(f"images without labels={missing_labels}")
            if missing_images:
                details.append(f"labels without images={missing_images}")
            raise PreparationError(
                f"Unpaired files in {class_root}: " + "; ".join(details)
            )

        for case_id in sorted(images):
            if case_id in seen:
                raise PreparationError(
                    f"Duplicate case identifier {case_id!r} in {seen[case_id]} and "
                    f"{images[case_id]}"
                )
            seen[case_id] = images[case_id]
            cases.append(
                LabeledCase(
                    case_id=case_id,
                    image=images[case_id],
                    label=labels[case_id],
                    class_name=class_name,
                    class_id=class_id,
                    split=split,
                )
            )

    return tuple(sorted(cases, key=lambda item: item.case_id))


def discover_test_split(source: Path) -> tuple[TestCase, ...]:
    test_root = source / "test"
    if not test_root.is_dir():
        raise PreparationError(f"Missing source directory: {test_root}")

    nested = _visible_directories(test_root)
    if nested:
        raise PreparationError(f"Unexpected directories in {test_root}: {sorted(nested)}")

    unsupported_files = sorted(
        item.name
        for item in test_root.iterdir()
        if item.is_file()
        and not item.name.startswith(".")
        and not item.name.endswith(NIFTI_SUFFIX)
    )
    if unsupported_files:
        raise PreparationError(
            f"Unexpected non-NIfTI files in {test_root}: {unsupported_files}"
        )

    cases: list[TestCase] = []
    seen: set[str] = set()
    for path in sorted(test_root.glob(f"*{NIFTI_SUFFIX}")):
        if path.name.startswith("._"):
            continue
        if not path.name.endswith(CHANNEL_SUFFIX):
            raise PreparationError(
                f"Test image must end with {CHANNEL_SUFFIX!r}: {path}"
            )
        case_id = path.name[: -len(CHANNEL_SUFFIX)]
        _validate_case_id(case_id, path)
        if case_id in seen:
            raise PreparationError(f"Duplicate test case identifier {case_id!r}")
        seen.add(case_id)
        cases.append(TestCase(case_id=case_id, image=path))
    return tuple(cases)


def discover_inventory(source: Path) -> Inventory:
    source = source.resolve()
    if not source.is_dir():
        raise PreparationError(f"Source directory does not exist: {source}")
    inventory = Inventory(
        train=discover_labeled_split(source, "train"),
        validation=discover_labeled_split(source, "validation"),
        test=discover_test_split(source),
    )
    validate_split_overlap(inventory)
    return inventory


def validate_split_overlap(inventory: Inventory) -> None:
    train_ids = {case.case_id for case in inventory.train}
    validation_ids = {case.case_id for case in inventory.validation}
    test_ids = {case.case_id for case in inventory.test}
    overlaps = {
        "train/validation": sorted(train_ids & validation_ids),
        "train/test": sorted(train_ids & test_ids),
        "validation/test": sorted(validation_ids & test_ids),
    }
    bad = {name: ids for name, ids in overlaps.items() if ids}
    if bad:
        raise PreparationError(f"Case identifiers overlap across splits: {bad}")


def validate_counts(config: Config, inventory: Inventory) -> None:
    actual = {
        "train": len(inventory.train),
        "validation": len(inventory.validation),
        "test": len(inventory.test),
    }
    expected = {
        "train": config.expected_train,
        "validation": config.expected_validation,
        "test": config.expected_test,
    }
    mismatches = {
        split: {"expected": expected[split], "actual": actual[split]}
        for split in actual
        if expected[split] is not None and actual[split] != expected[split]
    }
    if mismatches:
        raise PreparationError(f"Unexpected split counts: {mismatches}")


def validate_config(config: Config) -> None:
    if not (1 <= config.dataset_id <= 999):
        raise PreparationError("--dataset-id must be between 1 and 999")
    if not SAFE_DATASET_NAME.fullmatch(config.dataset_name):
        raise PreparationError(
            "--dataset-name must start with a letter and contain only letters, "
            "digits, and underscores"
        )
    if config.validation_layout not in {"imagesTr", "separate"}:
        raise PreparationError(
            "--validation-layout must be either 'imagesTr' or 'separate'"
        )
    if not math.isfinite(config.rounding_tolerance) or config.rounding_tolerance < 0:
        raise PreparationError("--rounding-tolerance must be a finite non-negative value")

    source = config.source.resolve()
    destination = config.destination.resolve()
    if destination == source or source in destination.parents:
        raise PreparationError(
            f"Output dataset directory must not be inside the source tree: {destination}"
        )


def _import_nibabel() -> Any:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PreparationError(
            "nibabel is required to audit and convert NIfTI files. "
            "Install the project dependencies, then rerun this command."
        ) from exc
    return nib


def _geometry(image: Any) -> Geometry:
    ndim = len(image.shape)
    return Geometry(
        shape=tuple(int(value) for value in image.shape),
        affine=np.asarray(image.affine, dtype=np.float64),
        zooms=tuple(float(value) for value in image.header.get_zooms()[:ndim]),
    )


def _validate_geometry(geometry: Geometry, path: Path) -> None:
    if len(geometry.shape) != 3:
        raise PreparationError(f"Expected a 3D NIfTI volume, got {geometry.shape}: {path}")
    if any(size <= 0 for size in geometry.shape):
        raise PreparationError(f"NIfTI has an empty dimension {geometry.shape}: {path}")
    if geometry.affine.shape != (4, 4) or not np.isfinite(geometry.affine).all():
        raise PreparationError(f"NIfTI has an invalid affine matrix: {path}")
    if len(geometry.zooms) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in geometry.zooms
    ):
        raise PreparationError(f"NIfTI has invalid voxel spacing {geometry.zooms}: {path}")


def _assert_same_geometry(
    first: Geometry,
    second: Geometry,
    first_path: Path,
    second_path: Path,
    *,
    affine_tolerance: float = 1e-5,
) -> None:
    if first.shape != second.shape:
        raise PreparationError(
            f"Geometry mismatch: shape {first.shape} for {first_path}, but "
            f"{second.shape} for {second_path}"
        )
    if not np.allclose(
        first.affine,
        second.affine,
        rtol=affine_tolerance,
        atol=affine_tolerance,
    ):
        raise PreparationError(
            f"Geometry mismatch: affine matrices differ for {first_path} and {second_path}"
        )
    if not np.allclose(
        first.zooms,
        second.zooms,
        rtol=affine_tolerance,
        atol=affine_tolerance,
    ):
        raise PreparationError(
            f"Geometry mismatch: voxel spacing differs for {first_path} and {second_path}"
        )


def round_and_validate_mask(
    data: np.ndarray,
    *,
    path: Path,
    tolerance: float,
    allowed_labels: Iterable[int] = SEGMENTATION_LABELS.values(),
) -> tuple[np.ndarray, tuple[float, ...], int, float, dict[int, int]]:
    """Validate near-integer mask data and return a uint8 mask and audit facts."""
    array = np.asanyarray(data)
    if array.ndim != 3:
        raise PreparationError(f"Expected a 3D segmentation mask, got {array.shape}: {path}")
    if not np.issubdtype(array.dtype, np.number):
        raise PreparationError(f"Segmentation mask is not numeric ({array.dtype}): {path}")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise PreparationError(f"Segmentation mask cannot be complex-valued: {path}")
    if not np.isfinite(array).all():
        raise PreparationError(f"Segmentation mask contains NaN or infinity: {path}")

    rounded = np.rint(array)
    errors = np.abs(array - rounded)
    max_error = float(errors.max(initial=0.0))
    if max_error > tolerance:
        raise PreparationError(
            f"Mask values in {path} are not within {tolerance:g} of integers; "
            f"maximum error is {max_error:g}"
        )

    unique_rounded, counts = np.unique(rounded, return_counts=True)
    rounded_labels = {int(value) for value in unique_rounded.tolist()}
    allowed = {int(value) for value in allowed_labels}
    invalid = sorted(rounded_labels - allowed)
    if invalid:
        raise PreparationError(
            f"Mask contains unsupported labels {invalid} after rounding: {path}; "
            f"allowed labels are {sorted(allowed)}"
        )
    if min(rounded_labels, default=0) < 0 or max(rounded_labels, default=0) > 255:
        raise PreparationError(f"Mask labels cannot be represented as uint8: {path}")

    raw_values = tuple(float(value) for value in np.unique(array).tolist())
    rounded_voxels = int(np.count_nonzero(errors))
    voxel_counts = {
        int(value): int(count)
        for value, count in zip(unique_rounded.tolist(), counts.tolist(), strict=True)
    }
    return (
        rounded.astype(np.uint8, copy=False),
        raw_values,
        rounded_voxels,
        max_error,
        voxel_counts,
    )


def audit_source_case(case: LabeledCase, *, tolerance: float) -> MaskAudit:
    nib = _import_nibabel()
    try:
        image = nib.load(str(case.image))
        label = nib.load(str(case.label))
    except Exception as exc:
        raise PreparationError(f"Could not read NIfTI pair for {case.case_id}: {exc}") from exc

    image_geometry = _geometry(image)
    label_geometry = _geometry(label)
    _validate_geometry(image_geometry, case.image)
    _validate_geometry(label_geometry, case.label)
    _assert_same_geometry(image_geometry, label_geometry, case.image, case.label)
    try:
        label_data = np.asanyarray(label.dataobj)
    except Exception as exc:
        raise PreparationError(f"Could not read mask voxels from {case.label}: {exc}") from exc
    _, raw_values, rounded_voxels, max_error, voxel_counts = round_and_validate_mask(
        label_data,
        path=case.label,
        tolerance=tolerance,
    )
    return MaskAudit(
        case_id=case.case_id,
        split=case.split,
        raw_values=raw_values,
        rounded_values=tuple(sorted(voxel_counts)),
        rounded_voxels=rounded_voxels,
        max_rounding_error=max_error,
        voxel_counts=voxel_counts,
    )


def audit_test_case(case: TestCase) -> None:
    nib = _import_nibabel()
    try:
        image = nib.load(str(case.image))
    except Exception as exc:
        raise PreparationError(f"Could not read test NIfTI {case.image}: {exc}") from exc
    _validate_geometry(_geometry(image), case.image)


def audit_source(
    inventory: Inventory,
    *,
    tolerance: float,
    progress: Callable[[str], None] = print,
) -> tuple[MaskAudit, ...]:
    audits: list[MaskAudit] = []
    total = len(inventory.labeled)
    for index, case in enumerate(inventory.labeled, start=1):
        audits.append(audit_source_case(case, tolerance=tolerance))
        if index == 1 or index % 25 == 0 or index == total:
            progress(f"Audited labelled geometry/masks: {index}/{total}")
    for index, case in enumerate(inventory.test, start=1):
        audit_test_case(case)
        if index == 1 or index % 25 == 0 or index == len(inventory.test):
            progress(f"Audited test geometry: {index}/{len(inventory.test)}")
    return tuple(audits)


def _expected_output_names(
    config: Config, inventory: Inventory
) -> dict[str, set[str]]:
    train_ids = {case.case_id for case in inventory.train}
    validation_ids = {case.case_id for case in inventory.validation}
    test_ids = {case.case_id for case in inventory.test}
    images_tr_ids = (
        train_ids | validation_ids
        if config.validation_layout == "imagesTr"
        else train_ids
    )
    expected = {
        "imagesTr": {f"{case_id}_0000.nii.gz" for case_id in images_tr_ids},
        "labelsTr": {f"{case_id}.nii.gz" for case_id in images_tr_ids},
        "imagesTs": {f"{case_id}_0000.nii.gz" for case_id in test_ids},
    }
    if config.validation_layout == "separate":
        expected["imagesVal"] = {
            f"{case_id}_0000.nii.gz" for case_id in validation_ids
        }
        expected["labelsVal"] = {f"{case_id}.nii.gz" for case_id in validation_ids}
    return expected


def validate_existing_output(config: Config, inventory: Inventory) -> None:
    """Refuse stale NIfTI files that could silently contaminate a split."""
    destination = config.destination
    if not destination.exists():
        return
    expected = _expected_output_names(config, inventory)
    # A prior train-only planning run may leave these safe duplicate directories.
    if config.validation_layout == "imagesTr":
        validation_ids = {case.case_id for case in inventory.validation}
        expected["imagesVal"] = {
            f"{case_id}_0000.nii.gz" for case_id in validation_ids
        }
        expected["labelsVal"] = {f"{case_id}.nii.gz" for case_id in validation_ids}

    for directory_name in ("imagesTr", "labelsTr", "imagesTs", "imagesVal", "labelsVal"):
        directory = destination / directory_name
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise PreparationError(f"Expected a directory but found a file: {directory}")
        actual = {
            item.name
            for item in directory.iterdir()
            if item.is_file() and not item.name.startswith(".")
        }
        unexpected = sorted(actual - expected.get(directory_name, set()))
        if unexpected:
            raise PreparationError(
                f"Unexpected files in {directory}; refusing to risk split contamination: "
                f"{unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}"
            )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=NIFTI_SUFFIX, dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(destination: Path, payload: Any) -> None:
    _atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_csv(destination: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise PreparationError(f"Refusing to write an empty CSV manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_uint8_mask(source: Path, destination: Path, *, tolerance: float) -> None:
    nib = _import_nibabel()
    try:
        source_image = nib.load(str(source))
        source_data = np.asanyarray(source_image.dataobj)
    except Exception as exc:
        raise PreparationError(f"Could not read source mask {source}: {exc}") from exc
    mask, _, _, _, _ = round_and_validate_mask(
        source_data,
        path=source,
        tolerance=tolerance,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=NIFTI_SUFFIX, dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        header = source_image.header.copy()
        header.set_data_dtype(np.uint8)
        output_image = source_image.__class__(mask, source_image.affine, header=header)

        # Construction can normalize qform/sform metadata. Restore both explicitly.
        qform, qform_code = source_image.get_qform(coded=True)
        sform, sform_code = source_image.get_sform(coded=True)
        if qform is not None:
            output_image.set_qform(qform, int(qform_code))
        if sform is not None:
            output_image.set_sform(sform, int(sform_code))
        output_image.header.set_data_dtype(np.uint8)
        output_image.header["cal_min"] = int(mask.min(initial=0))
        output_image.header["cal_max"] = int(mask.max(initial=0))
        nib.save(output_image, str(temporary))
        os.replace(temporary, destination)
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError(f"Could not write converted mask {destination}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _case_destination_directories(config: Config, case: LabeledCase) -> tuple[str, str]:
    if case.split == "validation" and config.validation_layout == "separate":
        return "imagesVal", "labelsVal"
    return "imagesTr", "labelsTr"


def copy_dataset(config: Config, inventory: Inventory, *, progress: Callable[[str], None] = print) -> None:
    destination = config.destination
    total = len(inventory.labeled)
    for index, case in enumerate(inventory.labeled, start=1):
        image_directory, label_directory = _case_destination_directories(config, case)
        _atomic_copy(
            case.image,
            destination / image_directory / f"{case.case_id}_0000.nii.gz",
        )
        _write_uint8_mask(
            case.label,
            destination / label_directory / f"{case.case_id}.nii.gz",
            tolerance=config.rounding_tolerance,
        )
        if index == 1 or index % 25 == 0 or index == total:
            progress(f"Prepared labelled cases: {index}/{total}")

    for index, case in enumerate(inventory.test, start=1):
        _atomic_copy(
            case.image,
            destination / "imagesTs" / f"{case.case_id}_0000.nii.gz",
        )
        if index == 1 or index % 25 == 0 or index == len(inventory.test):
            progress(f"Prepared test cases: {index}/{len(inventory.test)}")


def build_dataset_json(config: Config, inventory: Inventory) -> dict[str, Any]:
    number_training = (
        len(inventory.labeled)
        if config.validation_layout == "imagesTr"
        else len(inventory.train)
    )
    return {
        "channel_names": {"0": "CT"},
        "labels": SEGMENTATION_LABELS,
        "numTraining": number_training,
        "file_ending": NIFTI_SUFFIX,
        "name": config.dataset_name,
        "description": (
            "Pancreas and lesion segmentation with a three-class tumour "
            "subtype target; prepared non-destructively for nnU-Net v2."
        ),
    }


def build_split_manifest(config: Config, inventory: Inventory) -> dict[str, Any]:
    train_ids = [case.case_id for case in inventory.train]
    validation_ids = [case.case_id for case in inventory.validation]
    test_ids = [case.case_id for case in inventory.test]
    return {
        "schema_version": 1,
        "source_splits_preserved": True,
        "validation_layout": config.validation_layout,
        "planning_case_ids": train_ids,
        "train_case_ids": train_ids,
        "validation_case_ids": validation_ids,
        "test_case_ids": test_ids,
        "imagesTr_case_ids": (
            train_ids + validation_ids
            if config.validation_layout == "imagesTr"
            else train_ids
        ),
        "note": (
            "Use train_case_ids for fingerprint/planning statistics. Copy "
            "splits_final.json to the matching nnUNet_preprocessed dataset "
            "directory before training."
        ),
    }


def build_classification_manifest(inventory: Inventory) -> dict[str, Any]:
    cases = [
        {
            "case_id": case.case_id,
            "label": case.class_id,
            "class_name": case.class_name,
            "split": case.split,
            "image_filename": f"{case.case_id}_0000.nii.gz",
            "segmentation_filename": f"{case.case_id}.nii.gz",
        }
        for case in inventory.labeled
    ]
    return {
        "schema_version": 1,
        "class_names": {str(value): key for key, value in CLASS_TO_LABEL.items()},
        "cases": cases,
    }


def build_audit_report(
    config: Config, inventory: Inventory, audits: Sequence[MaskAudit]
) -> dict[str, Any]:
    split_class_counts: dict[str, dict[str, int]] = {}
    for split_name, cases in (
        ("train", inventory.train),
        ("validation", inventory.validation),
    ):
        counts = Counter(case.class_name for case in cases)
        split_class_counts[split_name] = {
            name: int(counts.get(name, 0)) for name in CLASS_TO_LABEL
        }

    voxel_counts: Counter[int] = Counter()
    raw_values: set[float] = set()
    for audit in audits:
        raw_values.update(audit.raw_values)
        voxel_counts.update(audit.voxel_counts)

    corrected = [audit for audit in audits if audit.rounded_voxels > 0]
    return {
        "schema_version": 1,
        "dataset": config.dataset_directory_name,
        "validation_layout": config.validation_layout,
        "counts": {
            "train": len(inventory.train),
            "validation": len(inventory.validation),
            "test": len(inventory.test),
            "imagesTr": (
                len(inventory.labeled)
                if config.validation_layout == "imagesTr"
                else len(inventory.train)
            ),
        },
        "classification_counts": split_class_counts,
        "checks": {
            "pairings": len(inventory.labeled),
            "label_geometries": len(inventory.labeled),
            "test_geometries": len(inventory.test),
            "split_overlap": False,
            "allowed_segmentation_labels": sorted(SEGMENTATION_LABELS.values()),
        },
        "mask_repair": {
            "rounding_tolerance": config.rounding_tolerance,
            "raw_unique_values": sorted(raw_values),
            "output_unique_values": sorted(voxel_counts),
            "masks_with_rounded_voxels": len(corrected),
            "total_rounded_voxels": sum(item.rounded_voxels for item in corrected),
            "maximum_rounding_error": max(
                (item.max_rounding_error for item in audits), default=0.0
            ),
            "output_voxel_counts": {
                str(label): int(voxel_counts[label]) for label in sorted(voxel_counts)
            },
        },
    }


def write_metadata(
    config: Config, inventory: Inventory, audits: Sequence[MaskAudit]
) -> None:
    destination = config.destination
    train_ids = [case.case_id for case in inventory.train]
    validation_ids = [case.case_id for case in inventory.validation]
    _atomic_write_json(destination / "dataset.json", build_dataset_json(config, inventory))
    _atomic_write_json(
        destination / "splits_final.json",
        [{"train": train_ids, "val": validation_ids}],
    )
    _atomic_write_json(
        destination / "split_manifest.json", build_split_manifest(config, inventory)
    )
    _atomic_write_json(
        destination / "classification_labels.json",
        {case.case_id: case.class_id for case in inventory.labeled},
    )
    classification_manifest = build_classification_manifest(inventory)
    _atomic_write_json(
        destination / "classification_manifest.json", classification_manifest
    )
    _atomic_write_csv(
        destination / "classification_labels.csv",
        classification_manifest["cases"],
    )
    _atomic_write_json(
        destination / "data_audit.json",
        build_audit_report(config, inventory, audits),
    )


def validate_output(config: Config, inventory: Inventory) -> None:
    """Validate output counts, names, mask dtype/labels, and copied geometry."""
    nib = _import_nibabel()
    destination = config.destination
    expected = _expected_output_names(config, inventory)
    for directory_name, expected_names in expected.items():
        directory = destination / directory_name
        actual_names = {
            item.name
            for item in directory.glob(f"*{NIFTI_SUFFIX}")
            if item.is_file()
        }
        if actual_names != expected_names:
            raise PreparationError(
                f"Output filenames differ in {directory}: missing="
                f"{sorted(expected_names - actual_names)[:10]}, unexpected="
                f"{sorted(actual_names - expected_names)[:10]}"
            )

    for case in inventory.labeled:
        image_directory, label_directory = _case_destination_directories(config, case)
        output_image_path = destination / image_directory / f"{case.case_id}_0000.nii.gz"
        output_label_path = destination / label_directory / f"{case.case_id}.nii.gz"
        source_image = nib.load(str(case.image))
        source_label = nib.load(str(case.label))
        output_image = nib.load(str(output_image_path))
        output_label = nib.load(str(output_label_path))
        _assert_same_geometry(
            _geometry(source_image), _geometry(output_image), case.image, output_image_path
        )
        _assert_same_geometry(
            _geometry(source_label), _geometry(output_label), case.label, output_label_path
        )
        if np.dtype(output_label.get_data_dtype()) != np.dtype(np.uint8):
            raise PreparationError(
                f"Converted label is not stored as uint8: {output_label_path} "
                f"({output_label.get_data_dtype()})"
            )
        round_and_validate_mask(
            np.asanyarray(output_label.dataobj),
            path=output_label_path,
            tolerance=0.0,
        )

    for case in inventory.test:
        output_path = destination / "imagesTs" / f"{case.case_id}_0000.nii.gz"
        _assert_same_geometry(
            _geometry(nib.load(str(case.image))),
            _geometry(nib.load(str(output_path))),
            case.image,
            output_path,
        )

    dataset_json = json.loads((destination / "dataset.json").read_text(encoding="utf-8"))
    if dataset_json != build_dataset_json(config, inventory):
        raise PreparationError("dataset.json did not round-trip to the expected content")

    labels = json.loads(
        (destination / "classification_labels.json").read_text(encoding="utf-8")
    )
    expected_labels = {case.case_id: case.class_id for case in inventory.labeled}
    if labels != expected_labels:
        raise PreparationError("classification_labels.json is incomplete or inconsistent")

    split = json.loads((destination / "splits_final.json").read_text(encoding="utf-8"))
    expected_split = [
        {
            "train": [case.case_id for case in inventory.train],
            "val": [case.case_id for case in inventory.validation],
        }
    ]
    if split != expected_split:
        raise PreparationError("splits_final.json is incomplete or inconsistent")


def prepare_dataset(
    config: Config,
    *,
    audit_function: Callable[..., tuple[MaskAudit, ...]] = audit_source,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    validate_config(config)
    inventory = discover_inventory(config.source)
    validate_counts(config, inventory)
    validate_existing_output(config, inventory)
    progress(
        "Discovered "
        f"{len(inventory.train)} train, {len(inventory.validation)} validation, "
        f"and {len(inventory.test)} test cases."
    )
    audits = audit_function(
        inventory, tolerance=config.rounding_tolerance, progress=progress
    )
    report = build_audit_report(config, inventory, audits)
    if config.dry_run:
        progress("Dry run complete: source validated; no directories or files were written.")
        return report

    copy_dataset(config, inventory, progress=progress)
    write_metadata(config, inventory, audits)
    progress("Validating generated counts, filenames, labels, and geometry...")
    validate_output(config, inventory)
    progress(f"Dataset preparation complete: {config.destination}")
    return report


def _optional_expected_count(value: str) -> int | None:
    if value.lower() in {"none", "skip"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected counts must be non-negative or 'none'")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Non-destructively prepare the pancreas quiz dataset for nnU-Net v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted source directory containing train/, validation/, and test/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="nnUNet_raw root below which DatasetXXX_Name will be created.",
    )
    parser.add_argument("--dataset-id", type=int, required=True, help="nnU-Net dataset ID (1-999).")
    parser.add_argument("--dataset-name", required=True, help="nnU-Net dataset name.")
    parser.add_argument(
        "--validation-layout",
        choices=("imagesTr", "separate"),
        default="imagesTr",
        help=(
            "Put validation cases into imagesTr for final preprocessing/training, "
            "or imagesVal/labelsVal so planning sees training data only."
        ),
    )
    parser.add_argument(
        "--rounding-tolerance",
        type=float,
        default=1e-3,
        help="Maximum permitted absolute distance from an integer mask label.",
    )
    parser.add_argument(
        "--expected-train", type=_optional_expected_count, default=252
    )
    parser.add_argument(
        "--expected-validation", type=_optional_expected_count, default=36
    )
    parser.add_argument("--expected-test", type=_optional_expected_count, default=72)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit the source and planned layout without writing anything.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config(
        source=args.source,
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        validation_layout=args.validation_layout,
        dry_run=args.dry_run,
        rounding_tolerance=args.rounding_tolerance,
        expected_train=args.expected_train,
        expected_validation=args.expected_validation,
        expected_test=args.expected_test,
    )
    try:
        report = prepare_dataset(config)
    except (PreparationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if config.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
