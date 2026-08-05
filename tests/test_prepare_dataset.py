from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import prepare_dataset as prep


def _make_class_directories(root: Path) -> None:
    for split in ("train", "validation"):
        for class_name in prep.CLASS_TO_LABEL:
            (root / split / class_name).mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)


def _touch_labeled_case(
    root: Path, split: str, class_name: str, case_id: str
) -> None:
    directory = root / split / class_name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{case_id}_0000.nii.gz").write_bytes(b"image")
    (directory / f"{case_id}.nii.gz").write_bytes(b"label")


def _touch_test_case(root: Path, case_id: str) -> None:
    (root / "test").mkdir(parents=True, exist_ok=True)
    (root / "test" / f"{case_id}_0000.nii.gz").write_bytes(b"test")


def _small_source(root: Path) -> prep.Inventory:
    _make_class_directories(root)
    for index, class_name in enumerate(prep.CLASS_TO_LABEL):
        _touch_labeled_case(root, "train", class_name, f"train_{index}")
        _touch_labeled_case(root, "validation", class_name, f"val_{index}")
    _touch_test_case(root, "test_001")
    _touch_test_case(root, "test_002")
    return prep.discover_inventory(root)


def _empty_audits(inventory: prep.Inventory, **_: object) -> tuple[prep.MaskAudit, ...]:
    return tuple(
        prep.MaskAudit(
            case_id=case.case_id,
            split=case.split,
            raw_values=(0.0, 1.0000153, 2.0),
            rounded_values=(0, 1, 2),
            rounded_voxels=1,
            max_rounding_error=1.53e-5,
            voxel_counts={0: 20, 1: 5, 2: 1},
        )
        for case in inventory.labeled
    )


def test_discovers_pairs_classes_and_split_manifests(tmp_path: Path) -> None:
    inventory = _small_source(tmp_path / "source")

    assert len(inventory.train) == 3
    assert len(inventory.validation) == 3
    assert len(inventory.test) == 2
    assert {case.class_id for case in inventory.train} == {0, 1, 2}

    full_config = prep.Config(
        source=tmp_path / "source",
        output_root=tmp_path / "raw",
        dataset_id=501,
        dataset_name="PancreasQuiz",
        expected_train=3,
        expected_validation=3,
        expected_test=2,
    )
    separate_config = prep.Config(
        source=full_config.source,
        output_root=full_config.output_root,
        dataset_id=501,
        dataset_name="PancreasQuiz",
        validation_layout="separate",
        expected_train=3,
        expected_validation=3,
        expected_test=2,
    )

    assert prep.build_dataset_json(full_config, inventory)["numTraining"] == 6
    assert prep.build_dataset_json(separate_config, inventory)["numTraining"] == 3
    split_manifest = prep.build_split_manifest(full_config, inventory)
    assert split_manifest["planning_case_ids"] == ["train_0", "train_1", "train_2"]
    assert split_manifest["validation_case_ids"] == ["val_0", "val_1", "val_2"]
    assert len(split_manifest["imagesTr_case_ids"]) == 6

    classification = prep.build_classification_manifest(inventory)
    labels = {row["case_id"]: row["label"] for row in classification["cases"]}
    assert labels["train_0"] == 0
    assert labels["val_2"] == 2


def test_rejects_an_image_without_a_label(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_class_directories(source)
    unmatched = source / "train" / "subtype0" / "orphan_0000.nii.gz"
    unmatched.write_bytes(b"image")

    with pytest.raises(prep.PreparationError, match="images without labels"):
        prep.discover_labeled_split(source, "train")


def test_rejects_visible_non_nifti_source_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_class_directories(source)
    (source / "train" / "subtype0" / "notes.txt").write_text("unexpected")

    with pytest.raises(prep.PreparationError, match="Unexpected non-NIfTI"):
        prep.discover_labeled_split(source, "train")


def test_rejects_case_overlap_across_supplied_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_class_directories(source)
    _touch_labeled_case(source, "train", "subtype0", "shared_case")
    _touch_labeled_case(source, "validation", "subtype1", "shared_case")

    with pytest.raises(prep.PreparationError, match="overlap across splits"):
        prep.discover_inventory(source)


def test_rounds_known_float_artifact_and_casts_uint8() -> None:
    data = np.array(
        [[[0.0, 1.0000153], [2.0, 1.0]]],
        dtype=np.float32,
    )

    converted, raw_values, rounded_voxels, max_error, counts = (
        prep.round_and_validate_mask(
            data,
            path=Path("mask.nii.gz"),
            tolerance=1e-3,
        )
    )

    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted, [[[0, 1], [2, 1]]])
    assert any(value > 1.0 for value in raw_values)
    assert rounded_voxels == 1
    assert max_error < 1e-3
    assert counts == {0: 1, 1: 2, 2: 1}


@pytest.mark.parametrize(
    ("data", "error"),
    [
        (np.array([[[1.2]]]), "not within"),
        (np.array([[[3.0]]]), "unsupported labels"),
        (np.array([[[np.nan]]]), "NaN or infinity"),
    ],
)
def test_rejects_unsafe_mask_values(data: np.ndarray, error: str) -> None:
    with pytest.raises(prep.PreparationError, match=error):
        prep.round_and_validate_mask(
            data,
            path=Path("bad_mask.nii.gz"),
            tolerance=1e-3,
        )


def test_dry_run_audits_but_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _small_source(source)
    output_root = tmp_path / "nnUNet_raw"
    config = prep.Config(
        source=source,
        output_root=output_root,
        dataset_id=501,
        dataset_name="PancreasQuiz",
        dry_run=True,
        expected_train=3,
        expected_validation=3,
        expected_test=2,
    )

    report = prep.prepare_dataset(
        config,
        audit_function=_empty_audits,
        progress=lambda _: None,
    )

    assert report["counts"] == {
        "train": 3,
        "validation": 3,
        "test": 2,
        "imagesTr": 6,
    }
    assert report["mask_repair"]["masks_with_rounded_voxels"] == 6
    assert not output_root.exists()


def test_train_only_planning_layout_maps_validation_outside_images_tr(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    inventory = _small_source(source)
    config = prep.Config(
        source=source,
        output_root=tmp_path / "raw",
        dataset_id=501,
        dataset_name="PancreasQuiz",
        validation_layout="separate",
        expected_train=3,
        expected_validation=3,
        expected_test=2,
    )

    expected = prep._expected_output_names(config, inventory)

    assert expected["imagesTr"] == {
        "train_0_0000.nii.gz",
        "train_1_0000.nii.gz",
        "train_2_0000.nii.gz",
    }
    assert expected["imagesVal"] == {
        "val_0_0000.nii.gz",
        "val_1_0000.nii.gz",
        "val_2_0000.nii.gz",
    }


def test_existing_stale_case_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    inventory = _small_source(source)
    config = prep.Config(
        source=source,
        output_root=tmp_path / "raw",
        dataset_id=501,
        dataset_name="PancreasQuiz",
        expected_train=3,
        expected_validation=3,
        expected_test=2,
    )
    stale = config.destination / "imagesTr" / "unrelated_0000.nii.gz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    with pytest.raises(prep.PreparationError, match="split contamination"):
        prep.validate_existing_output(config, inventory)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nifti_conversion_preserves_geometry_and_originals(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    source = tmp_path / "source"
    _make_class_directories(source)
    affine = np.array(
        [
            [0.8, 0.0, 0.0, -10.0],
            [0.0, 0.9, 0.0, 12.0],
            [0.0, 0.0, 2.5, 20.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image_data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    label_data = np.zeros((2, 3, 4), dtype=np.float32)
    label_data[0, 0, 0] = 1.0000153
    label_data[1, 2, 3] = 2.0
    case_id = "train_case"
    image_path = source / "train" / "subtype0" / f"{case_id}_0000.nii.gz"
    label_path = source / "train" / "subtype0" / f"{case_id}.nii.gz"
    test_path = source / "test" / "test_case_0000.nii.gz"
    nib.save(nib.Nifti1Image(image_data, affine), image_path)
    label_image = nib.Nifti1Image(label_data, affine)
    label_image.set_qform(affine, 1)
    label_image.set_sform(affine, 2)
    nib.save(label_image, label_path)
    nib.save(nib.Nifti1Image(image_data, affine), test_path)
    original_hashes = {path: _digest(path) for path in (image_path, label_path, test_path)}

    config = prep.Config(
        source=source,
        output_root=tmp_path / "raw",
        dataset_id=501,
        dataset_name="PancreasQuiz",
        expected_train=1,
        expected_validation=0,
        expected_test=1,
    )
    prep.prepare_dataset(config, progress=lambda _: None)

    output_label = nib.load(str(config.destination / "labelsTr" / f"{case_id}.nii.gz"))
    assert output_label.get_data_dtype() == np.dtype(np.uint8)
    np.testing.assert_array_equal(np.asanyarray(output_label.dataobj), np.rint(label_data))
    np.testing.assert_allclose(output_label.affine, affine)
    np.testing.assert_allclose(output_label.get_qform(), affine)
    np.testing.assert_allclose(output_label.get_sform(), affine)
    assert int(output_label.header["qform_code"]) == 1
    assert int(output_label.header["sform_code"]) == 2
    assert {path: _digest(path) for path in original_hashes} == original_hashes
