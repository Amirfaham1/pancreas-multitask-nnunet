#!/usr/bin/env python3
"""Validate the final test-result directory or ZIP before submission.

The validator is intentionally independent from inference.  It checks the
complete delivery contract: a flat archive root, exact test-case membership,
``subtype_results.csv`` schema/content, readable integer NIfTI masks limited to
labels ``{0, 1, 2}``, and shape/affine/spacing agreement with every test image.

Example::

    python scripts/validate_submission.py \
      artifacts/Amirfaham_Fallahpour_results.zip \
      --test-images data/test \
      --output-json artifacts/submission_validation.json \
      --output-csv artifacts/submission_case_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

NIFTI_SUFFIX = ".nii.gz"
CHANNEL_SUFFIX = "_0000.nii.gz"
DEFAULT_CSV_NAME = "subtype_results.csv"
ALLOWED_LABELS = {0, 1, 2}


class SubmissionValidationError(RuntimeError):
    """Raised for invalid command configuration or an unreadable submission."""


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str | None = None
    case_id: str | None = None


def _import_nibabel() -> Any:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SubmissionValidationError(
            "nibabel is required to validate NIfTI files; install project dependencies"
        ) from exc
    return nib


def _add_issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    *,
    path: Path | str | None = None,
    case_id: str | None = None,
) -> None:
    issues.append(
        ValidationIssue(
            code=code,
            message=message,
            path=str(path) if path is not None else None,
            case_id=case_id,
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_test_image_name(name: str) -> str:
    if not name.endswith(CHANNEL_SUFFIX):
        raise SubmissionValidationError(f"Test image must end in {CHANNEL_SUFFIX!r}, got {name!r}")
    case_id = name[: -len(CHANNEL_SUFFIX)]
    if not case_id or "/" in case_id or "\\" in case_id:
        raise SubmissionValidationError(f"Invalid test case identifier from {name!r}")
    return case_id


def discover_test_images(test_images: Path, *, expected_count: int) -> dict[str, Path]:
    root = test_images.resolve()
    if not root.is_dir():
        raise SubmissionValidationError(f"Test-image directory does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob(f"*{CHANNEL_SUFFIX}")):
        if path.name.startswith(".") or path.name.startswith("._"):
            continue
        case_id = _strip_test_image_name(path.name)
        if case_id in result:
            raise SubmissionValidationError(
                f"Duplicate test image for {case_id}: {result[case_id]} and {path}"
            )
        result[case_id] = path
    if len(result) != expected_count:
        raise SubmissionValidationError(
            f"Expected {expected_count} test images below {root}, found {len(result)}"
        )
    return result


def _validate_zip_members(
    archive: Path,
    issues: list[ValidationIssue],
) -> tuple[zipfile.ZipFile | None, list[zipfile.ZipInfo]]:
    try:
        handle = zipfile.ZipFile(archive, mode="r")
        members = handle.infolist()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        _add_issue(issues, "archive_unreadable", f"Could not open ZIP: {exc}", path=archive)
        return None, []

    if not members:
        _add_issue(issues, "archive_empty", "ZIP archive is empty", path=archive)
    names = [member.filename for member in members]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        _add_issue(
            issues,
            "archive_duplicate_members",
            f"ZIP contains duplicate member names: {duplicates}",
            path=archive,
        )

    casefold_duplicates = sorted(
        name for name, count in Counter(item.casefold() for item in names).items() if count > 1
    )
    if casefold_duplicates:
        _add_issue(
            issues,
            "archive_case_collisions",
            f"ZIP contains names that collide case-insensitively: {casefold_duplicates}",
            path=archive,
        )

    for member in members:
        raw_name = member.filename
        normalized = PurePosixPath(raw_name.replace("\\", "/"))
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or len(normalized.parts) != 1
            or raw_name != normalized.name
        ):
            _add_issue(
                issues,
                "archive_not_flat",
                f"Every archive member must be a file at the ZIP root, got {raw_name!r}",
                path=archive,
            )
        if member.is_dir():
            _add_issue(
                issues,
                "archive_directory_member",
                f"Directory entry is not allowed: {raw_name!r}",
                path=archive,
            )
        unix_mode = (member.external_attr >> 16) & 0xFFFF
        if unix_mode and stat.S_ISLNK(unix_mode):
            _add_issue(
                issues,
                "archive_symlink",
                f"Symbolic link is not allowed: {raw_name!r}",
                path=archive,
            )
        if member.flag_bits & 0x1:
            _add_issue(
                issues,
                "archive_encrypted_member",
                f"Encrypted member cannot be validated: {raw_name!r}",
                path=archive,
            )
    return handle, members


def _extract_validated_zip(
    archive: Path,
    destination: Path,
    issues: list[ValidationIssue],
) -> bool:
    handle, members = _validate_zip_members(archive, issues)
    if handle is None:
        return False
    unsafe_codes = {
        "archive_duplicate_members",
        "archive_not_flat",
        "archive_directory_member",
        "archive_symlink",
        "archive_encrypted_member",
    }
    if any(issue.code in unsafe_codes for issue in issues):
        handle.close()
        return False
    try:
        for member in members:
            target = destination / member.filename
            with handle.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        _add_issue(
            issues,
            "archive_extraction_failed",
            f"Could not read all ZIP members: {exc}",
            path=archive,
        )
        return False
    finally:
        handle.close()
    return True


def _geometry(image: Any) -> tuple[tuple[int, ...], np.ndarray, tuple[float, ...]]:
    shape = tuple(int(value) for value in image.shape)
    affine = np.asarray(image.affine, dtype=np.float64)
    zooms = tuple(float(value) for value in image.header.get_zooms()[: len(shape)])
    return shape, affine, zooms


def _validate_one_mask(
    mask_path: Path,
    test_image_path: Path,
    *,
    case_id: str,
    affine_tolerance: float,
    issues: list[ValidationIssue],
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "case_id": case_id,
        "mask_filename": mask_path.name,
        "mask_present": True,
        "mask_readable": False,
        "geometry_matches": False,
        "integer_dtype": False,
        "labels_valid": False,
        "observed_labels": "",
    }
    nib = _import_nibabel()
    try:
        mask_image = nib.load(str(mask_path))
        test_image = nib.load(str(test_image_path))
        mask_shape, mask_affine, mask_zooms = _geometry(mask_image)
        test_shape, test_affine, test_zooms = _geometry(test_image)
    except Exception as exc:  # noqa: BLE001 - readers may raise backend-specific exceptions
        _add_issue(
            issues,
            "nifti_unreadable",
            f"Could not read mask and matching test image: {exc}",
            path=mask_path,
            case_id=case_id,
        )
        return audit
    audit["mask_readable"] = True

    geometry_valid = True
    if len(mask_shape) != 3 or any(size <= 0 for size in mask_shape):
        geometry_valid = False
        _add_issue(
            issues,
            "mask_not_3d",
            f"Mask must be non-empty and 3D, got shape {mask_shape}",
            path=mask_path,
            case_id=case_id,
        )
    if mask_affine.shape != (4, 4) or not np.isfinite(mask_affine).all():
        geometry_valid = False
        _add_issue(
            issues,
            "mask_invalid_affine",
            "Mask affine is not a finite 4x4 matrix",
            path=mask_path,
            case_id=case_id,
        )
    if len(mask_zooms) != 3 or any(not math.isfinite(value) or value <= 0 for value in mask_zooms):
        geometry_valid = False
        _add_issue(
            issues,
            "mask_invalid_spacing",
            f"Mask has invalid voxel spacing {mask_zooms}",
            path=mask_path,
            case_id=case_id,
        )
    if mask_shape != test_shape:
        geometry_valid = False
        _add_issue(
            issues,
            "geometry_shape_mismatch",
            f"Mask shape {mask_shape} differs from test image shape {test_shape}",
            path=mask_path,
            case_id=case_id,
        )
    if mask_affine.shape != test_affine.shape or not np.allclose(
        mask_affine,
        test_affine,
        rtol=affine_tolerance,
        atol=affine_tolerance,
    ):
        geometry_valid = False
        _add_issue(
            issues,
            "geometry_affine_mismatch",
            "Mask affine differs from matching test image",
            path=mask_path,
            case_id=case_id,
        )
    if len(mask_zooms) != len(test_zooms) or not np.allclose(
        mask_zooms,
        test_zooms,
        rtol=affine_tolerance,
        atol=affine_tolerance,
    ):
        geometry_valid = False
        _add_issue(
            issues,
            "geometry_spacing_mismatch",
            f"Mask spacing {mask_zooms} differs from test image spacing {test_zooms}",
            path=mask_path,
            case_id=case_id,
        )
    audit["geometry_matches"] = geometry_valid

    storage_dtype = np.dtype(mask_image.get_data_dtype())
    integer_dtype = bool(np.issubdtype(storage_dtype, np.integer))
    audit["integer_dtype"] = integer_dtype
    if not integer_dtype:
        _add_issue(
            issues,
            "mask_noninteger_dtype",
            f"Mask storage dtype must be integer, got {storage_dtype}",
            path=mask_path,
            case_id=case_id,
        )
    try:
        data = np.asanyarray(mask_image.dataobj)
    except Exception as exc:  # noqa: BLE001 - proxy decompression errors are backend-specific
        _add_issue(
            issues,
            "mask_voxels_unreadable",
            f"Could not read mask voxels: {exc}",
            path=mask_path,
            case_id=case_id,
        )
        return audit
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.complexfloating):
        _add_issue(
            issues,
            "mask_nonnumeric",
            f"Mask voxels are not real numeric values ({data.dtype})",
            path=mask_path,
            case_id=case_id,
        )
        return audit
    if not np.isfinite(data).all():
        _add_issue(
            issues,
            "mask_nonfinite",
            "Mask contains NaN or infinity",
            path=mask_path,
            case_id=case_id,
        )
        return audit
    rounded = np.rint(data)
    if not np.array_equal(data, rounded):
        _add_issue(
            issues,
            "mask_noninteger_values",
            "Mask contains non-integer voxel values",
            path=mask_path,
            case_id=case_id,
        )
        return audit
    observed = {int(value) for value in np.unique(rounded).tolist()}
    audit["observed_labels"] = ";".join(str(value) for value in sorted(observed))
    invalid = sorted(observed - ALLOWED_LABELS)
    if invalid:
        _add_issue(
            issues,
            "mask_invalid_labels",
            f"Mask contains labels {invalid}; allowed labels are {sorted(ALLOWED_LABELS)}",
            path=mask_path,
            case_id=case_id,
        )
        return audit
    audit["labels_valid"] = True
    return audit


def _validate_csv(
    csv_path: Path,
    *,
    expected_filenames: set[str],
    mask_filenames: set[str],
    issues: list[ValidationIssue],
) -> tuple[dict[str, int], bool]:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.reader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        _add_issue(
            issues,
            "csv_unreadable",
            f"Could not read classification CSV: {exc}",
            path=csv_path,
        )
        return {}, False
    if not records:
        _add_issue(issues, "csv_empty", "Classification CSV is empty", path=csv_path)
        return {}, False
    if records[0] != ["Names", "Subtype"]:
        _add_issue(
            issues,
            "csv_header",
            f"CSV header must be exactly Names,Subtype; got {records[0]!r}",
            path=csv_path,
        )
    mapping: dict[str, int] = {}
    structural_valid = records[0] == ["Names", "Subtype"]
    for row_number, row in enumerate(records[1:], start=2):
        if len(row) != 2:
            structural_valid = False
            _add_issue(
                issues,
                "csv_row_width",
                f"Row {row_number} must have exactly two columns, got {len(row)}",
                path=csv_path,
            )
            continue
        name, raw_subtype = row
        if name != name.strip() or not name:
            structural_valid = False
            _add_issue(
                issues,
                "csv_invalid_name",
                f"Row {row_number} has an empty or whitespace-padded name {name!r}",
                path=csv_path,
            )
            continue
        if "/" in name or "\\" in name:
            structural_valid = False
            _add_issue(
                issues,
                "csv_name_is_path",
                f"Row {row_number} Names value must be a filename, got {name!r}",
                path=csv_path,
            )
            continue
        if name in mapping:
            structural_valid = False
            _add_issue(
                issues,
                "csv_duplicate_name",
                f"Duplicate CSV Names value {name!r}",
                path=csv_path,
            )
            continue
        try:
            subtype = int(raw_subtype)
        except ValueError:
            structural_valid = False
            _add_issue(
                issues,
                "csv_invalid_subtype",
                f"Subtype for {name} must be an integer in {{0,1,2}}, got {raw_subtype!r}",
                path=csv_path,
            )
            continue
        if raw_subtype not in {"0", "1", "2"} or subtype not in ALLOWED_LABELS:
            structural_valid = False
            _add_issue(
                issues,
                "csv_invalid_subtype",
                f"Subtype for {name} must be exactly 0, 1, or 2, got {raw_subtype!r}",
                path=csv_path,
            )
            continue
        mapping[name] = subtype

    if len(records) - 1 != len(expected_filenames):
        structural_valid = False
        _add_issue(
            issues,
            "csv_row_count",
            f"Expected {len(expected_filenames)} CSV data rows, found {len(records) - 1}",
            path=csv_path,
        )
    missing = sorted(expected_filenames - set(mapping))
    unexpected = sorted(set(mapping) - expected_filenames)
    if missing:
        structural_valid = False
        _add_issue(
            issues,
            "csv_missing_names",
            f"CSV is missing test filenames: {missing}",
            path=csv_path,
        )
    if unexpected:
        structural_valid = False
        _add_issue(
            issues,
            "csv_unexpected_names",
            f"CSV contains unexpected filenames: {unexpected}",
            path=csv_path,
        )
    mask_mismatch = sorted(set(mapping) ^ mask_filenames)
    if mask_mismatch:
        structural_valid = False
        _add_issue(
            issues,
            "csv_mask_name_mismatch",
            f"CSV Names and output mask filenames differ: {mask_mismatch}",
            path=csv_path,
        )
    return mapping, structural_valid


def _inspect_submission_root(
    root: Path,
    *,
    test_images: dict[str, Path],
    expected_count: int,
    csv_name: str,
    affine_tolerance: float,
    issues: list[ValidationIssue],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    expected_mask_names = {f"{case_id}{NIFTI_SUFFIX}" for case_id in test_images}
    expected_root_names = expected_mask_names | {csv_name}
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        _add_issue(issues, "submission_unreadable", str(exc), path=root)
        return [], {}

    actual_names = {entry.name for entry in entries}
    for entry in entries:
        if entry.is_dir():
            _add_issue(
                issues,
                "nested_directory",
                f"Submission root must be flat; found directory {entry.name!r}",
                path=entry,
            )
        if entry.name.startswith(".") or entry.name.startswith("._"):
            _add_issue(
                issues,
                "hidden_file",
                f"Hidden metadata is not allowed: {entry.name!r}",
                path=entry,
            )
    missing_root = sorted(expected_root_names - actual_names)
    unexpected_root = sorted(actual_names - expected_root_names)
    if missing_root:
        _add_issue(
            issues,
            "missing_root_files",
            f"Submission is missing required root files: {missing_root}",
            path=root,
        )
    if unexpected_root:
        _add_issue(
            issues,
            "unexpected_root_files",
            f"Submission contains unexpected root entries: {unexpected_root}",
            path=root,
        )

    mask_paths = {
        entry.name: entry
        for entry in entries
        if entry.is_file() and entry.name.endswith(NIFTI_SUFFIX)
    }
    if len(mask_paths) != expected_count:
        _add_issue(
            issues,
            "mask_count",
            f"Expected exactly {expected_count} mask files, found {len(mask_paths)}",
            path=root,
        )
    missing_masks = sorted(expected_mask_names - set(mask_paths))
    unexpected_masks = sorted(set(mask_paths) - expected_mask_names)
    if missing_masks:
        _add_issue(
            issues,
            "missing_masks",
            f"Missing output masks: {missing_masks}",
            path=root,
        )
    if unexpected_masks:
        _add_issue(
            issues,
            "unexpected_masks",
            f"Unexpected output mask names: {unexpected_masks}",
            path=root,
        )

    audits: list[dict[str, Any]] = []
    for case_id, test_path in sorted(test_images.items()):
        filename = f"{case_id}{NIFTI_SUFFIX}"
        if filename not in mask_paths:
            audits.append(
                {
                    "case_id": case_id,
                    "mask_filename": filename,
                    "mask_present": False,
                    "mask_readable": False,
                    "geometry_matches": False,
                    "integer_dtype": False,
                    "labels_valid": False,
                    "observed_labels": "",
                }
            )
            continue
        audits.append(
            _validate_one_mask(
                mask_paths[filename],
                test_path,
                case_id=case_id,
                affine_tolerance=affine_tolerance,
                issues=issues,
            )
        )

    csv_path = root / csv_name
    mapping: dict[str, int] = {}
    if not csv_path.is_file():
        _add_issue(
            issues,
            "csv_missing",
            f"Required classification file {csv_name!r} is missing",
            path=root,
        )
    else:
        mapping, _ = _validate_csv(
            csv_path,
            expected_filenames=expected_mask_names,
            mask_filenames=set(mask_paths),
            issues=issues,
        )
    for audit in audits:
        filename = str(audit["mask_filename"])
        audit["csv_present"] = filename in mapping
        audit["predicted_subtype"] = mapping.get(filename, "")
    return audits, mapping


def _atomic_write_text(path: Path, text: str) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_audit_csv(path: Path, audits: Sequence[dict[str, Any]]) -> None:
    if not audits:
        return
    fieldnames = [
        "case_id",
        "mask_filename",
        "mask_present",
        "mask_readable",
        "geometry_matches",
        "integer_dtype",
        "labels_valid",
        "observed_labels",
        "csv_present",
        "predicted_subtype",
    ]
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(audits)
    _atomic_write_text(path, buffer.getvalue())


def validate_submission(
    submission: Path,
    test_image_directory: Path,
    *,
    expected_count: int = 72,
    csv_name: str = DEFAULT_CSV_NAME,
    affine_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Return a full JSON-serializable validation report for a directory or ZIP."""

    if expected_count <= 0:
        raise SubmissionValidationError("expected_count must be positive")
    if not csv_name or Path(csv_name).name != csv_name or csv_name.startswith("."):
        raise SubmissionValidationError("csv_name must be a visible root filename")
    if not math.isfinite(affine_tolerance) or affine_tolerance < 0:
        raise SubmissionValidationError("affine_tolerance must be finite and non-negative")
    source = submission.resolve()
    if not source.exists():
        raise SubmissionValidationError(f"Submission path does not exist: {source}")
    if not source.is_dir() and not source.is_file():
        raise SubmissionValidationError(f"Submission must be a directory or ZIP file: {source}")
    archive_mode = source.is_file()
    if archive_mode and not zipfile.is_zipfile(source):
        raise SubmissionValidationError(f"Submission file is not a readable ZIP archive: {source}")

    test_images = discover_test_images(test_image_directory, expected_count=expected_count)
    issues: list[ValidationIssue] = []
    audits: list[dict[str, Any]] = []
    mapping: dict[str, int] = {}
    archive_sha256 = _sha256(source) if archive_mode else None

    if archive_mode:
        with tempfile.TemporaryDirectory(prefix="pancreas_submission_validation_") as temporary:
            extraction_root = Path(temporary)
            extracted = _extract_validated_zip(source, extraction_root, issues)
            if extracted:
                audits, mapping = _inspect_submission_root(
                    extraction_root,
                    test_images=test_images,
                    expected_count=expected_count,
                    csv_name=csv_name,
                    affine_tolerance=affine_tolerance,
                    issues=issues,
                )
    else:
        audits, mapping = _inspect_submission_root(
            source,
            test_images=test_images,
            expected_count=expected_count,
            csv_name=csv_name,
            affine_tolerance=affine_tolerance,
            issues=issues,
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "valid": not issues,
        "submission": str(source),
        "submission_type": "zip" if archive_mode else "directory",
        "archive_sha256": archive_sha256,
        "expected_case_count": expected_count,
        "validated_mask_count": sum(bool(row.get("mask_present")) for row in audits),
        "validated_csv_row_count": len(mapping),
        "csv_name": csv_name,
        "allowed_mask_labels": sorted(ALLOWED_LABELS),
        "geometry_policy": {
            "shape": "exact",
            "affine_and_spacing_tolerance": affine_tolerance,
        },
        "issues": [asdict(issue) for issue in issues],
        "issue_count": len(issues),
        "cases": audits,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission",
        nargs="?",
        type=Path,
        help="Submission directory or .zip file",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        help="Named alternative to the positional submission path",
    )
    parser.add_argument(
        "--test-images",
        required=True,
        type=Path,
        help="Directory containing the original *_0000.nii.gz test images",
    )
    parser.add_argument("--expected-count", type=int, default=72)
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME)
    parser.add_argument("--affine-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-json", type=Path, help="Validation report JSON path")
    parser.add_argument("--output-csv", type=Path, help="Per-case audit CSV path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.submission is not None and args.submission_path is not None:
        parser.error("provide either positional submission or --submission-path, not both")
    submission = args.submission if args.submission is not None else args.submission_path
    if submission is None:
        parser.error("a submission directory or ZIP is required")
    try:
        report = validate_submission(
            submission,
            args.test_images,
            expected_count=args.expected_count,
            csv_name=args.csv_name,
            affine_tolerance=args.affine_tolerance,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output_json is not None:
            _atomic_write_text(args.output_json, rendered)
        else:
            print(rendered, end="")
        if args.output_csv is not None:
            _write_audit_csv(args.output_csv, report["cases"])
    except (SubmissionValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if report["valid"]:
        if args.output_json is not None:
            print(
                f"VALID: {report['expected_case_count']} masks and subtype rows; "
                f"report={args.output_json.resolve()}"
            )
        return 0
    print(
        f"INVALID: found {report['issue_count']} issue(s)"
        + (f"; report={args.output_json.resolve()}" if args.output_json is not None else ""),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
