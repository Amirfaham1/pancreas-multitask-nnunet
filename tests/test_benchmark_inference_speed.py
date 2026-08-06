from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "benchmark_inference_speed.py"
    spec = importlib.util.spec_from_file_location("benchmark_inference_speed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime(
    *,
    batch_size: int,
    seconds_per_case: float,
    started_at: str,
    process_id: int,
) -> dict[str, object]:
    case_ids = ["case_a", "case_b"]
    return {
        "case_count": 2,
        "case_ids": case_ids,
        "case_ids_sha256": "same-case-fingerprint",
        "checkpoint": "checkpoint_final.pth",
        "checkpoint_files": [
            {"fold": "0", "sha256": "a" * 64, "size_bytes": 123}
        ],
        "cuda_runtime_version": "12.8",
        "cudnn_version": 91002,
        "device": "cuda",
        "device_capability": [7, 5],
        "device_name": "Tesla T4",
        "folds": [0],
        "gaussian_enabled": True,
        "inference_execution": {
            "joint_network_forward_calls": 16 // batch_size,
            "maximum_network_batch_size_observed": batch_size,
            "network_batch_size_histogram": {str(batch_size): 16 // batch_size},
            "network_batch_size_limit": batch_size,
            "logical_tile_batches_completed": 16 // batch_size,
            "logical_tiles_completed": 16,
            "tile_batch_oom_fallback_count": 0,
            "tile_batch_size_adaptive_limit": batch_size,
            "tile_batch_size_histogram": {str(batch_size): 16 // batch_size},
            "tile_batch_size_requested": batch_size,
            "tta_batch_oom_fallback_count": 0,
            "tta_batch_size_adaptive_limit": batch_size,
            "tta_batch_size_histogram": {str(batch_size): 16 // batch_size},
            "tta_batch_size_requested": batch_size,
            "tta_view_batches_completed": 16 // batch_size,
            "tta_views_completed": 16,
        },
        "mean_seconds_per_case": seconds_per_case,
        "overwrite": True,
        "peak_allocated_mib": 2000.0 + batch_size,
        "peak_reserved_mib": 2400.0 + batch_size,
        "process_id": process_id,
        "python_version": "3.12.10",
        "started_at_utc": started_at,
        "tile_step_size": 0.5,
        "timing_scope": "fresh_process_model_initialization_preprocessing_inference_export",
        "total_seconds": 2 * seconds_per_case,
        "torch_version": "2.8.0+cu128",
        "tta_enabled": True,
        "warmup_policy": "none_fresh_process_end_to_end",
    }


def _write_runtime(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_outputs(path: Path, *, changed_mask: bool = False, probability_shift=0.0) -> Path:
    path.mkdir()
    for index, case_id in enumerate(("case_a", "case_b")):
        array = np.full((2, 2, 2), index, dtype=np.uint8)
        if changed_mask and index == 1:
            array[0, 0, 0] = 2
        nib.save(nib.Nifti1Image(array, np.eye(4)), path / f"{case_id}.nii.gz")

    with (path / "subtype_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(
            (("Names", "Subtype"), ("case_a.nii.gz", 0), ("case_b.nii.gz", 1))
        )
    with (path / "subtype_probabilities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("Names", "Subtype", "Probability_0", "Probability_1", "Probability_2")
        )
        writer.writerow(("case_a.nii.gz", 0, 0.8 + probability_shift, 0.1, 0.1))
        writer.writerow(("case_b.nii.gz", 1, 0.1, 0.8 + probability_shift, 0.1))
    return path


def _fixture(tmp_path: Path, *, candidate_seconds: float = 8.0):
    reference_runtime = [
        _write_runtime(
            tmp_path / "r1.json",
            _runtime(
                batch_size=1,
                seconds_per_case=10.0,
                started_at="2026-08-06T15:00:00+00:00",
                process_id=101,
            ),
        ),
        _write_runtime(
            tmp_path / "r2.json",
            _runtime(
                batch_size=1,
                seconds_per_case=10.0,
                started_at="2026-08-06T15:03:00+00:00",
                process_id=104,
            ),
        ),
    ]
    candidate_runtime = [
        _write_runtime(
            tmp_path / "c1.json",
            _runtime(
                batch_size=2,
                seconds_per_case=candidate_seconds,
                started_at="2026-08-06T15:01:00+00:00",
                process_id=102,
            ),
        ),
        _write_runtime(
            tmp_path / "c2.json",
            _runtime(
                batch_size=2,
                seconds_per_case=candidate_seconds,
                started_at="2026-08-06T15:02:00+00:00",
                process_id=103,
            ),
        ),
    ]
    reference_output = [
        _write_outputs(tmp_path / "r1-output"),
        _write_outputs(tmp_path / "r2-output"),
    ]
    candidate_output = [
        _write_outputs(tmp_path / "c1-output", probability_shift=1e-6),
        _write_outputs(tmp_path / "c2-output", probability_shift=1e-6),
    ]
    return reference_runtime, candidate_runtime, reference_output, candidate_output


def test_accepts_abba_pair_only_when_mean_reduction_exceeds_ten_percent(
    tmp_path: Path,
) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)

    result = module.audit_benchmark(*paths, expected_case_count=2)

    assert result["accepted"] is True
    assert result["run_order"] == ["reference", "candidate", "candidate", "reference"]
    assert result["runtime_reduction_percent"] == pytest.approx(20.0)
    assert result["candidate_fraction_of_reference"] == pytest.approx(0.8)
    assert all(
        comparison["hard_mask_disagreeing_voxels"] == 0
        and comparison["subtype_decision_disagreements"] == 0
        and comparison["maximum_absolute_class_probability_delta"]
        <= module.MAXIMUM_CLASS_PROBABILITY_DELTA
        for comparison in result["numerical_equivalence"]
    )


def test_valid_but_insufficient_speed_is_rejected_without_redefining_rule(
    tmp_path: Path,
) -> None:
    module = _load_module()
    paths = _fixture(tmp_path, candidate_seconds=9.1)

    result = module.audit_benchmark(*paths, expected_case_count=2)

    assert result["accepted"] is False
    assert result["runtime_reduction_percent"] == pytest.approx(9.0)


def test_oom_fallback_or_configuration_change_invalidates_comparison(tmp_path: Path) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)
    candidate_payload = json.loads(paths[1][0].read_text(encoding="utf-8"))
    candidate_payload["inference_execution"]["tile_batch_oom_fallback_count"] = 1
    paths[1][0].write_text(json.dumps(candidate_payload), encoding="utf-8")

    with pytest.raises(module.BenchmarkError, match="OOM fallback"):
        module.audit_benchmark(*paths, expected_case_count=2)


def test_one_hard_mask_difference_invalidates_numerical_equivalence(tmp_path: Path) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)
    changed = tmp_path / "changed-output"
    _write_outputs(changed, changed_mask=True)
    paths[3][1] = changed

    with pytest.raises(module.BenchmarkError, match="Hard masks disagree"):
        module.audit_benchmark(*paths, expected_case_count=2)
