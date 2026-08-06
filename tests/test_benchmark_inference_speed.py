from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = "assignment_conforming_v5_neural_case_head"
TIMING_SCOPE = (
    "fresh_process_model_and_v5_head_initialization_preprocessing_"
    "feature_extraction_neural_head_offsets_export"
)
CASE_IDS = ["case_a", "case_b"]
BAG_HASHES = ["1" * 64, "2" * 64]


def _load_module():
    path = ROOT / "scripts" / "benchmark_inference_speed.py"
    spec = importlib.util.spec_from_file_location("benchmark_inference_speed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle_provenance() -> dict[str, object]:
    return {
        "classifier_pipeline": PIPELINE,
        "bundle_path": "D:/locked/neural_case_head_bundle.pt",
        "bundle_name": "neural_case_head_bundle.pt",
        "bundle_sha256": "3" * 64,
        "bundle_size_bytes": 12_345,
        "expected_bundle_sha256_verified": True,
        "numeric_train_dataset_sha256": "4" * 64,
        "selected_candidate_id": "cross_attention_mil",
        "head_parameter_count": 67_843,
        "head_in_eval_mode": True,
        "any_head_parameter_requires_grad": False,
        "head_state_sha256": "5" * 64,
        "head_state_sha256_before": "5" * 64,
        "head_state_sha256_after": "5" * 64,
        "head_state_unchanged": True,
        "class_offsets": [-0.25, 0.0, 0.25],
        "neural_lock_sha256": "6" * 64,
        "decision_lock_sha256": "7" * 64,
        "selection_audit_sha256": "8" * 64,
        "calibration_audit_sha256": "9" * 64,
        "refit_audit_sha256": "a" * 64,
        "eligible_for_official": True,
        "bundle_loaded_strictly": True,
    }


def _frozen_network_provenance() -> dict[str, object]:
    component_hashes = {
        "encoder": "324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1",
        "decoder": "b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2",
        "classification": "1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8",
    }
    return {
        "fold": 0,
        "component_hashes_before": component_hashes,
        "component_hashes_after": deepcopy(component_hashes),
        "frozen_components_unchanged": True,
        "network_in_eval_mode": True,
        "any_network_parameter_requires_grad": False,
    }


def _runtime(
    *,
    extraction_mode: str,
    seconds_per_case: float,
    started_at: str,
    process_id: int,
) -> dict[str, object]:
    input_files = [
        {"name": "case_a_0000.nii.gz", "sha256": "6" * 64, "size_bytes": 111},
        {"name": "case_b_0000.nii.gz", "sha256": "7" * 64, "size_bytes": 222},
    ]
    input_manifest_sha256 = hashlib.sha256(
        json.dumps(input_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    execution = {
        "classifier_pipeline": PIPELINE,
        "joint_network_forward_calls": 16,
        "maximum_network_batch_size_observed": 1,
        "network_batch_size_histogram": {"1": 16},
        "network_batch_size_limit": 1,
        "logical_tile_batches_completed": 16,
        "logical_tiles_completed": 16,
        "tile_batch_oom_fallback_count": 0,
        "tile_batch_size_adaptive_limit": 1,
        "tile_batch_size_histogram": {"1": 16},
        "tile_batch_size_requested": 1,
        "tta_batch_oom_fallback_count": 0,
        "tta_batch_size_adaptive_limit": 1,
        "tta_batch_size_histogram": {"1": 16},
        "tta_batch_size_requested": 1,
        "tta_view_batches_completed": 16,
        "tta_views_completed": 16,
        "speed_v3_network_batch_ceiling": 1,
        "v5_extraction_mode": extraction_mode,
        "v5_feature_extraction_executed": True,
        "v5_case_extractions_completed": len(CASE_IDS),
        "v5_neural_head_forward_calls": len(CASE_IDS),
        "v5_class_offset_applications": len(CASE_IDS),
        "v5_feature_cache_reads": 0,
        "case_identifiers_or_paths_used_as_model_inputs": False,
        "v5_neural_bag_sha256_sequence": list(BAG_HASHES),
    }
    return {
        "case_count": len(CASE_IDS),
        "case_ids": list(CASE_IDS),
        "case_ids_sha256": hashlib.sha256(
            "".join(f"{case_id}\n" for case_id in CASE_IDS).encode("utf-8")
        ).hexdigest(),
        "checkpoint": "checkpoint_classification_rescue.pth",
        "checkpoint_files": [
            {
                "fold": "0",
                "sha256": "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116",
                "size_bytes": 123_456,
            }
        ],
        "checkpoint_unchanged_during_run": True,
        "classifier_pipeline": PIPELINE,
        "class_probabilities": "v5_offset_adjusted_three_class",
        "cuda_runtime_version": "12.8",
        "cudnn_version": 91002,
        "device": "cuda",
        "device_capability": [7, 5],
        "device_name": "Tesla T4",
        "folds": [0],
        "feature_cache_policy": "disabled_online_fresh_extraction",
        "frozen_network": _frozen_network_provenance(),
        "gaussian_enabled": True,
        "input_directory": "D:/locked/test-input",
        "input_file_manifest": {
            "file_count": len(input_files),
            "files": input_files,
            "manifest_sha256": input_manifest_sha256,
        },
        "input_files_unchanged_during_run": True,
        "inference_execution": execution,
        "mean_seconds_per_case": seconds_per_case,
        "model_configuration_files": [
            {
                "name": "dataset.json",
                "sha256": "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff",
                "size_bytes": 111,
            },
            {
                "name": "plans.json",
                "sha256": "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f",
                "size_bytes": 222,
            },
        ],
        "model_configuration_unchanged_during_run": True,
        "model_directory": "D:/locked/trained-model",
        "neural_case_head_bundle": _bundle_provenance(),
        "overwrite": True,
        "peak_allocated_mib": 2_001.0,
        "peak_reserved_mib": 2_401.0,
        "process_id": process_id,
        "python_version": "3.12.10",
        "started_at_utc": started_at,
        "tile_step_size": 0.5,
        "timing_scope": TIMING_SCOPE,
        "total_seconds": len(CASE_IDS) * seconds_per_case,
        "torch_version": "2.8.0+cu128",
        "tta_enabled": True,
        "v5_extraction_mode": extraction_mode,
        "v5_implementation_files": {
            "scripts/predict_joint.py": "e" * 64,
            "src/pancreas_multitask/classification_rescue.py": "d" * 64,
            "src/pancreas_multitask/network.py": "f" * 64,
            "src/pancreas_multitask/predictor.py": "a" * 64,
            "src/pancreas_multitask/case_features.py": "b" * 64,
            "src/pancreas_multitask/neural_case_predictor.py": "0" * 64,
            "src/pancreas_multitask/case_feature_extractor.py": "1" * 64,
            "src/pancreas_multitask/neural_case_bundle.py": "2" * 64,
            "src/pancreas_multitask/neural_case_head.py": "3" * 64,
            "src/pancreas_multitask/neural_case_training.py": "4" * 64,
        },
        "v5_neural_bag_sha256_sequence": list(BAG_HASHES),
        "warmup_policy": "none_fresh_process_end_to_end",
        "case_identifiers_or_paths_used_as_model_inputs": False,
    }


def _write_runtime(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_outputs(
    path: Path,
    *,
    probability_shift: float = 0.0,
    changed_mask: bool = False,
    changed_subtype: bool = False,
) -> Path:
    path.mkdir()
    for index, case_id in enumerate(CASE_IDS):
        array = np.full((2, 2, 2), index, dtype=np.uint8)
        if changed_mask and index == 1:
            array[0, 0, 0] = 2
        nib.save(nib.Nifti1Image(array, np.eye(4)), path / f"{case_id}.nii.gz")

    subtype_b = 2 if changed_subtype else 1
    with (path / "subtype_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(
            (("Names", "Subtype"), ("case_a.nii.gz", 0), ("case_b.nii.gz", subtype_b))
        )
    with (path / "subtype_probabilities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("Names", "Subtype", "Probability_0", "Probability_1", "Probability_2")
        )
        writer.writerow(
            (
                "case_a.nii.gz",
                0,
                0.8 + probability_shift,
                0.1 - probability_shift,
                0.1,
            )
        )
        case_b_probabilities = (
            (0.1, 0.1, 0.8)
            if changed_subtype
            else (0.1, 0.8 + probability_shift, 0.1 - probability_shift)
        )
        writer.writerow(("case_b.nii.gz", subtype_b, *case_b_probabilities))
    return path


def _fixture(tmp_path: Path, *, candidate_seconds: float = 8.0):
    reference_runtime = [
        _write_runtime(
            tmp_path / "r1.json",
            _runtime(
                extraction_mode="full",
                seconds_per_case=10.0,
                started_at="2026-08-06T15:00:00+00:00",
                process_id=101,
            ),
        ),
        _write_runtime(
            tmp_path / "r2.json",
            _runtime(
                extraction_mode="full",
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
                extraction_mode="neural_only",
                seconds_per_case=candidate_seconds,
                started_at="2026-08-06T15:01:00+00:00",
                process_id=102,
            ),
        ),
        _write_runtime(
            tmp_path / "c2.json",
            _runtime(
                extraction_mode="neural_only",
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
        _write_outputs(tmp_path / "c1-output", probability_shift=5e-7),
        _write_outputs(tmp_path / "c2-output", probability_shift=5e-7),
    ]
    return reference_runtime, candidate_runtime, reference_output, candidate_output


def _mutate_runtime(path: Path, mutation) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_accepts_v3_abba_full_vs_neural_only_with_exact_v5_outputs(
    tmp_path: Path,
) -> None:
    module = _load_module()

    result = module.audit_benchmark(*_fixture(tmp_path), expected_case_count=2)

    assert result["accepted"] is True
    assert result["schema_version"] == 3
    assert result["classifier_pipeline"] == PIPELINE
    assert result["run_order"] == ["reference", "candidate", "candidate", "reference"]
    assert result["runtime_reduction_percent"] == pytest.approx(20.0)
    assert result["reference"]["v5_extraction_mode"] == "full"
    assert result["candidate"]["v5_extraction_mode"] == "neural_only"
    assert result["reference"]["tile_batch_size"] == 1
    assert result["candidate"]["tile_batch_size"] == 1
    assert result["reference"]["tta_batch_size"] == 1
    assert result["candidate"]["tta_batch_size"] == 1
    assert all(
        comparison["hard_mask_disagreeing_voxels"] == 0
        and comparison["subtype_decision_disagreements"] == 0
        and comparison["maximum_absolute_class_probability_delta"] <= 1e-6
        and comparison["passed"] is True
        for comparison in result["numerical_equivalence"]
    )


def test_valid_v5_evidence_below_speed_gate_is_rejected(tmp_path: Path) -> None:
    module = _load_module()

    result = module.audit_benchmark(
        *_fixture(tmp_path, candidate_seconds=9.1), expected_case_count=2
    )

    assert result["accepted"] is False
    assert result["timing_passed"] is False
    assert result["rejection_reasons"] == ["runtime_reduction_below_10_percent"]


@pytest.mark.parametrize(
    ("bundle_field", "replacement"),
    (
        ("bundle_sha256", "0" * 64),
        ("head_state_sha256", "1" * 64),
        ("numeric_train_dataset_sha256", "2" * 64),
    ),
)
def test_mismatched_bundle_head_or_numeric_dataset_binding_is_rejected(
    tmp_path: Path, bundle_field: str, replacement: str
) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)
    _mutate_runtime(
        paths[1][0],
        lambda payload: payload["neural_case_head_bundle"].__setitem__(
            bundle_field, replacement
        ),
    )

    with pytest.raises(module.BenchmarkError, match="neural_case_head_bundle"):
        module.audit_benchmark(*paths, expected_case_count=2)


@pytest.mark.parametrize("binding", ("frozen_network", "implementation"))
def test_mismatched_frozen_network_or_implementation_hash_is_rejected(
    tmp_path: Path, binding: str
) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)

    def mutation(payload):
        if binding == "frozen_network":
            payload["frozen_network"]["component_hashes_before"]["encoder"] = "4" * 64
            payload["frozen_network"]["component_hashes_after"]["encoder"] = "4" * 64
        else:
            payload["v5_implementation_files"][
                "src/pancreas_multitask/neural_case_predictor.py"
            ] = "4" * 64

    _mutate_runtime(paths[1][0], mutation)

    expected_field = "frozen_network" if binding == "frozen_network" else "v5_implementation_files"
    with pytest.raises(module.BenchmarkError, match=expected_field):
        module.audit_benchmark(*paths, expected_case_count=2)


def test_mismatched_exact_neural_bag_hash_sequence_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)

    def mutation(payload):
        changed = ["1" * 64, "4" * 64]
        payload["v5_neural_bag_sha256_sequence"] = changed
        payload["inference_execution"]["v5_neural_bag_sha256_sequence"] = changed

    _mutate_runtime(paths[1][0], mutation)

    with pytest.raises(module.BenchmarkError, match="v5_neural_bag_sha256_sequence"):
        module.audit_benchmark(*paths, expected_case_count=2)


def test_offset_grid_and_live_head_state_are_fail_closed(tmp_path: Path) -> None:
    module = _load_module()
    offset_root = tmp_path / "offsets"
    offset_root.mkdir()
    invalid_offsets = _fixture(offset_root)
    _mutate_runtime(
        invalid_offsets[1][0],
        lambda payload: payload["neural_case_head_bundle"].__setitem__(
            "class_offsets", [-0.25, 0.25, 0.25]
        ),
    )
    with pytest.raises(module.BenchmarkError, match="locked grid"):
        module.audit_benchmark(*invalid_offsets, expected_case_count=2)

    head_root = tmp_path / "head"
    head_root.mkdir()
    changed_head = _fixture(head_root)
    _mutate_runtime(
        changed_head[1][0],
        lambda payload: payload["neural_case_head_bundle"].__setitem__(
            "head_state_sha256_after", "b" * 64
        ),
    )
    with pytest.raises(module.BenchmarkError, match="live state"):
        module.audit_benchmark(*changed_head, expected_case_count=2)


def test_raw_input_content_manifest_must_match_across_abba(tmp_path: Path) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)

    def mutation(payload):
        manifest = payload["input_file_manifest"]
        manifest["files"][0]["sha256"] = "8" * 64
        manifest["manifest_sha256"] = module._input_manifest_sha256(manifest["files"])

    _mutate_runtime(paths[1][0], mutation)
    with pytest.raises(module.BenchmarkError, match="input_file_manifest"):
        module.audit_benchmark(*paths, expected_case_count=2)


@pytest.mark.parametrize("artifact", ("checkpoint", "plans"))
def test_locked_checkpoint_and_plans_hashes_cannot_only_match_each_other(
    tmp_path: Path,
    artifact: str,
) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)
    all_runtime_paths = [*paths[0], *paths[1]]
    for runtime_path in all_runtime_paths:
        if artifact == "checkpoint":
            _mutate_runtime(
                runtime_path,
                lambda payload: payload["checkpoint_files"][0].__setitem__(
                    "sha256", "8" * 64
                ),
            )
        else:
            _mutate_runtime(
                runtime_path,
                lambda payload: payload["model_configuration_files"][1].__setitem__(
                    "sha256", "8" * 64
                ),
            )

    message = "checkpoint" if artifact == "checkpoint" else "model-configuration"
    with pytest.raises(module.BenchmarkError, match=message):
        module.audit_benchmark(*paths, expected_case_count=2)


def test_each_abba_run_requires_a_distinct_fresh_process(tmp_path: Path) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)
    _mutate_runtime(
        paths[1][0],
        lambda payload: payload.__setitem__("process_id", 101),
    )

    with pytest.raises(module.BenchmarkError, match="distinct fresh process"):
        module.audit_benchmark(*paths, expected_case_count=2)


@pytest.mark.parametrize("location", ("top_level", "execution"))
def test_wrong_candidate_extraction_mode_is_rejected(
    tmp_path: Path, location: str
) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)

    def mutation(payload):
        if location == "top_level":
            payload["v5_extraction_mode"] = "full"
        else:
            payload["inference_execution"]["v5_extraction_mode"] = "full"

    _mutate_runtime(paths[1][0], mutation)

    with pytest.raises(module.BenchmarkError, match="extraction mode"):
        module.audit_benchmark(*paths, expected_case_count=2)


@pytest.mark.parametrize(
    "counter",
    (
        "v5_case_extractions_completed",
        "v5_neural_head_forward_calls",
        "v5_class_offset_applications",
    ),
)
def test_incomplete_v5_head_path_counter_is_rejected(
    tmp_path: Path, counter: str
) -> None:
    module = _load_module()
    paths = _fixture(tmp_path)
    _mutate_runtime(
        paths[1][0],
        lambda payload: payload["inference_execution"].__setitem__(counter, 1),
    )

    with pytest.raises(module.BenchmarkError, match=counter):
        module.audit_benchmark(*paths, expected_case_count=2)


def test_mask_and_subtype_must_be_exact_and_probability_tolerance_is_one_e_minus_six(
    tmp_path: Path,
) -> None:
    module = _load_module()
    reference = _write_outputs(tmp_path / "reference")
    within_tolerance = _write_outputs(
        tmp_path / "within-tolerance", probability_shift=9e-7
    )
    changed_mask = _write_outputs(tmp_path / "changed-mask", changed_mask=True)
    changed_subtype = _write_outputs(
        tmp_path / "changed-subtype", changed_subtype=True
    )
    excessive_probability_delta = _write_outputs(
        tmp_path / "changed-probability", probability_shift=1.1e-6
    )

    within = module._compare_outputs(
        reference, within_tolerance, comparison_name="within"
    )
    assert within["passed"] is True
    assert within["maximum_absolute_class_probability_delta"] == pytest.approx(9e-7)
    assert within["maximum_absolute_class_probability_delta"] <= 1e-6
    assert module._compare_outputs(
        reference, changed_mask, comparison_name="mask"
    )["passed"] is False
    assert module._compare_outputs(
        reference, changed_subtype, comparison_name="subtype"
    )["passed"] is False
    assert module._compare_outputs(
        reference, excessive_probability_delta, comparison_name="probability"
    )["passed"] is False


def test_each_output_requires_valid_probability_and_subtype_semantics(
    tmp_path: Path,
) -> None:
    module = _load_module()
    invalid_probability = _write_outputs(tmp_path / "invalid-probability")
    probability_path = invalid_probability / "subtype_probabilities.csv"
    probability_path.write_text(
        probability_path.read_text(encoding="utf-8").replace(
            "case_a.nii.gz,0,0.8,0.1,0.1",
            "case_a.nii.gz,0,0.4,0.5,0.1",
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.BenchmarkError, match="probability semantics"):
        module._validate_output_case_contract(invalid_probability, CASE_IDS)

    invalid_subtype = _write_outputs(tmp_path / "invalid-subtype")
    subtype_path = invalid_subtype / "subtype_results.csv"
    subtype_path.write_text(
        subtype_path.read_text(encoding="utf-8").replace(
            "case_a.nii.gz,0",
            "case_a.nii.gz,3",
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.BenchmarkError, match="Subtype must be"):
        module._validate_output_case_contract(invalid_subtype, CASE_IDS)


def test_frozen_speed_v3_lock_matches_auditor_contract() -> None:
    module = _load_module()
    lock = json.loads(
        (ROOT / "configs" / "inference_speed_benchmark_v3.json").read_text(
            encoding="utf-8"
        )
    )

    assert lock["schema_version"] == 3
    assert lock["comparison"]["batching_in_both_arms"] == {
        "tile_batch_size": 1,
        "tta_batch_size": 1,
        "maximum_network_batch_size": 1,
    }
    assert "full_locked_feature_extraction" in lock["comparison"]["reference"]
    assert "dependency_pruned_neural_only" in lock["comparison"]["candidate"]
    numerical = lock["numerical_equivalence"]
    assert numerical["hard_masks_must_match_exactly"] is True
    assert numerical["subtype_decisions_must_match_exactly"] is True
    assert numerical["required_neural_bag_arrays_must_match_exactly"] is True
    assert numerical["maximum_absolute_offset_adjusted_class_probability_delta"] == (
        module.MAXIMUM_CLASS_PROBABILITY_DELTA
    )
    assert module.MAXIMUM_HARD_MASK_DISAGREEMENT_FRACTION == 0.0
    assert module.MAXIMUM_HARD_MASK_DISAGREEING_VOXELS_PER_CASE == 0
