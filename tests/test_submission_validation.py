from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import validate_submission as validator


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(data, transform), str(path))


def _make_test_inputs(root: Path) -> dict[str, np.ndarray]:
    cases = {
        "quiz_001": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        "quiz_002": np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 10,
    }
    for case_id, image in cases.items():
        _save_nifti(root / f"{case_id}_0000.nii.gz", image)
    return cases


def _write_csv(path: Path, rows: list[tuple[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Names", "Subtype"])
        writer.writerows(rows)


def _make_valid_submission(root: Path) -> None:
    mask_one = np.array([[[0, 0], [1, 1]], [[0, 2], [2, 1]]], dtype=np.uint8)
    mask_two = np.zeros((2, 2, 2), dtype=np.uint8)
    _save_nifti(root / "quiz_001.nii.gz", mask_one)
    _save_nifti(root / "quiz_002.nii.gz", mask_two)
    _write_csv(
        root / "subtype_results.csv",
        [("quiz_001.nii.gz", 0), ("quiz_002.nii.gz", 2)],
    )


def _issue_codes(report: dict[str, object]) -> set[str]:
    return {str(issue["code"]) for issue in report["issues"]}  # type: ignore[index]


def test_valid_directory_checks_masks_csv_and_geometry(tmp_path: Path) -> None:
    test_images = tmp_path / "test"
    submission = tmp_path / "submission"
    test_images.mkdir()
    submission.mkdir()
    _make_test_inputs(test_images)
    _make_valid_submission(submission)

    report = validator.validate_submission(submission, test_images, expected_count=2)

    assert report["valid"] is True
    assert report["issue_count"] == 0
    assert report["validated_mask_count"] == 2
    assert report["validated_csv_row_count"] == 2
    assert all(case["geometry_matches"] for case in report["cases"])
    assert all(case["integer_dtype"] for case in report["cases"])
    assert all(case["labels_valid"] for case in report["cases"])


def test_rejects_float_dtype_invalid_labels_and_geometry_mismatch(tmp_path: Path) -> None:
    test_images = tmp_path / "test"
    submission = tmp_path / "submission"
    test_images.mkdir()
    submission.mkdir()
    _make_test_inputs(test_images)

    float_mask = np.zeros((2, 2, 2), dtype=np.float32)
    float_mask[0, 0, 0] = 3.0
    _save_nifti(submission / "quiz_001.nii.gz", float_mask)
    shifted_affine = np.eye(4)
    shifted_affine[0, 3] = 5.0
    _save_nifti(
        submission / "quiz_002.nii.gz",
        np.zeros((2, 2, 2), dtype=np.uint8),
        shifted_affine,
    )
    _write_csv(
        submission / "subtype_results.csv",
        [("quiz_001.nii.gz", 0), ("quiz_002.nii.gz", 1)],
    )

    report = validator.validate_submission(submission, test_images, expected_count=2)

    assert report["valid"] is False
    assert {
        "mask_noninteger_dtype",
        "mask_invalid_labels",
        "geometry_affine_mismatch",
    } <= _issue_codes(report)


def test_rejects_csv_contract_violations(tmp_path: Path) -> None:
    test_images = tmp_path / "test"
    submission = tmp_path / "submission"
    test_images.mkdir()
    submission.mkdir()
    _make_test_inputs(test_images)
    _make_valid_submission(submission)
    (submission / "subtype_results.csv").write_text(
        "Names,Subtype\nquiz_001.nii.gz,1\nquiz_001.nii.gz,9\n",
        encoding="utf-8",
    )

    report = validator.validate_submission(submission, test_images, expected_count=2)

    assert report["valid"] is False
    assert {
        "csv_duplicate_name",
        "csv_missing_names",
        "csv_mask_name_mismatch",
    } <= _issue_codes(report)


def test_valid_flat_zip_and_rejects_parent_directory(tmp_path: Path) -> None:
    test_images = tmp_path / "test"
    submission = tmp_path / "submission"
    test_images.mkdir()
    submission.mkdir()
    _make_test_inputs(test_images)
    _make_valid_submission(submission)

    valid_zip = tmp_path / "results.zip"
    with zipfile.ZipFile(valid_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission.iterdir()):
            archive.write(path, arcname=path.name)
    valid_report = validator.validate_submission(valid_zip, test_images, expected_count=2)
    assert valid_report["valid"] is True
    assert len(valid_report["archive_sha256"]) == 64

    nested_zip = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission.iterdir()):
            archive.write(path, arcname=f"results/{path.name}")
    nested_report = validator.validate_submission(nested_zip, test_images, expected_count=2)
    assert nested_report["valid"] is False
    assert "archive_not_flat" in _issue_codes(nested_report)


def test_rejects_unexpected_root_file(tmp_path: Path) -> None:
    test_images = tmp_path / "test"
    submission = tmp_path / "submission"
    test_images.mkdir()
    submission.mkdir()
    _make_test_inputs(test_images)
    _make_valid_submission(submission)
    (submission / ".DS_Store").write_bytes(b"metadata")

    report = validator.validate_submission(submission, test_images, expected_count=2)

    assert report["valid"] is False
    assert {"hidden_file", "unexpected_root_files"} <= _issue_codes(report)
