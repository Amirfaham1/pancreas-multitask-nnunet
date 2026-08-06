from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_timed_inference_child.py"
    spec = importlib.util.spec_from_file_location("run_timed_inference_child", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runner():
    return _load_module()


def _arguments(tmp_path: Path, *, arm: str, purpose: str = "final_benchmark"):
    label = (
        "stock_reference_1"
        if arm == "stock" and purpose == "final_benchmark"
        else "candidate_1"
        if purpose == "final_benchmark"
        else f"{arm}_functional_smoke"
    )
    return argparse.Namespace(
        execution_purpose=purpose,
        run_label=label,
        arm=arm,
        input_directory=tmp_path / "input",
        output_directory=tmp_path / "output",
        model_directory=tmp_path / "model",
        external_runtime_json=tmp_path / "external.json",
        determinism_audit_json=tmp_path / "bootstrap.json",
        process_log=tmp_path / "process.log",
        final_candidate_lock=(tmp_path / "final.json") if purpose == "final_benchmark" else None,
        expected_final_candidate_lock_sha256=("a" * 64)
        if purpose == "final_benchmark"
        else None,
        one_use_ledger=(tmp_path / "ledger.json") if purpose == "final_benchmark" else None,
        benchmark_execution_id="12345678-1234-4234-9234-123456789abc",
        candidate_runtime_json=(tmp_path / "candidate.json") if arm == "candidate" else None,
        neural_case_head_bundle=(tmp_path / "head.pth") if arm == "candidate" else None,
        expected_neural_case_head_bundle_sha256=("b" * 64)
        if arm == "candidate"
        else None,
        expected_numeric_train_dataset_sha256=("c" * 64)
        if arm == "candidate"
        else None,
        expected_case_count=72,
        timeout_seconds=100.0,
        python_executable=Path(sys.executable),
    )


def test_frozen_lock_hash_constants_match_repository(runner) -> None:
    assert runner._sha256(runner.STOCK_GATE_LOCK) == runner.STOCK_GATE_LOCK_SHA256
    assert runner._sha256(runner.DETERMINISM_LOCK) == runner.DETERMINISM_LOCK_SHA256
    assert runner._sha256(runner.STOCK_EXPORT_LOCK) == runner.STOCK_EXPORT_LOCK_SHA256


def test_stock_target_arguments_are_exact_locked_defaults(runner, tmp_path: Path) -> None:
    args = _arguments(tmp_path, arm="stock")

    target = runner._target_arguments(args)

    assert target == [
        "-i",
        str(args.input_directory.resolve()),
        "-o",
        str(args.output_directory.resolve()),
        "-d",
        "501",
        "-p",
        "nnUNetResEncUNetMPlans",
        "-tr",
        "nnUNetTrainerPancreasMultiTask",
        "-c",
        "3d_fullres",
        "-f",
        "0",
        "-step_size",
        "0.5",
        "-chk",
        "checkpoint_classification_rescue.pth",
        "-npp",
        "3",
        "-nps",
        "3",
        "-device",
        "cuda",
    ]
    assert "--disable_tta" not in target
    assert "--not_on_device" not in target
    assert "--save_probabilities" not in target
    assert "--disable_progress_bar" not in target


def test_candidate_target_is_broader_but_uses_matched_core_settings(
    runner, tmp_path: Path
) -> None:
    args = _arguments(tmp_path, arm="candidate")

    target = runner._target_arguments(args)

    assert target[:6] == [
        "--input",
        str(args.input_directory.resolve()),
        "--output",
        str(args.output_directory.resolve()),
        "--model",
        str(args.model_directory.resolve()),
    ]
    for sequence in (
        ["--folds", "0"],
        ["--checkpoint", "checkpoint_classification_rescue.pth"],
        ["--classification-mode", "neural-v5"],
        ["--v5-extraction-mode", "neural_only"],
        ["--device", "cuda"],
        ["--tile-step-size", "0.5"],
        ["--tile-batch-size", "1"],
        ["--tta-batch-size", "1"],
    ):
        index = target.index(sequence[0])
        assert target[index : index + 2] == sequence
    assert target[-1] == "--overwrite"
    assert "--disable-tta" not in target
    assert "--disable-gaussian" not in target
    assert "--results-on-cpu" not in target


def test_input_manifest_is_content_addressed_and_one_channel_only(
    runner, tmp_path: Path
) -> None:
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "case_b_0000.nii.gz").write_bytes(b"b")
    (input_directory / "case_a_0000.nii.gz").write_bytes(b"a")

    manifest, case_ids = runner._input_manifest(input_directory)

    assert case_ids == ["case_a", "case_b"]
    assert [item["relative_path"] for item in manifest["files"]] == [
        "case_a_0000.nii.gz",
        "case_b_0000.nii.gz",
    ]
    assert manifest["file_count"] == 2
    manifest_again, _ = runner._input_manifest(input_directory)
    assert manifest_again == manifest

    (input_directory / "case_c.nii.gz").write_bytes(b"invalid")
    with pytest.raises(runner.TimedRunError, match="one-channel"):
        runner._input_manifest(input_directory)


def test_nvidia_snapshot_parser_records_power_temperature_and_identity(
    runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = SimpleNamespace(
        stdout="0, NVIDIA RTX, GPU-abc, 576.02, P8, 12.5, 48\n"
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: completed)

    snapshot = runner._nvidia_smi_snapshot()

    assert snapshot["uuid"] == "GPU-abc"
    assert snapshot["driver_version"] == "576.02"
    assert snapshot["power_draw_watts"] == 12.5
    assert snapshot["temperature_celsius"] == 48.0


@pytest.mark.parametrize("property_uuid", ["abc", "GPU-abc", "ABC"])
def test_cuda_environment_normalizes_optional_gpu_uuid_prefix(
    runner, monkeypatch: pytest.MonkeyPatch, property_uuid: str
) -> None:
    properties = SimpleNamespace(uuid=property_uuid)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner.torch.cuda, "get_device_properties", lambda _index: properties)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda _index: "NVIDIA RTX")
    monkeypatch.setattr(runner.torch.cuda, "get_device_capability", lambda _index: (8, 9))
    monkeypatch.setattr(runner.torch.backends.cudnn, "version", lambda: 90100)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "2.8.1")
    snapshot = {
        "uuid": "GPU-abc",
        "driver_version": "576.02",
    }

    environment = runner._cuda_environment(snapshot, Path(sys.executable))

    assert environment["cuda_device_uuid"] == "GPU-abc"
    assert environment["nvidia_driver_version"] == "576.02"
    assert environment["nnunet_compile"] == "false"
    assert environment["cublas_workspace_config"] == ":4096:8"


def _memory_payload(*, stock: bool) -> dict[str, object]:
    return {
        "unit": "MiB",
        "collector": "torch.cuda process-local memory counters",
        "bootstrap_reset_before_target": stock,
        "before_target": {"allocated_mib": 0.0, "reserved_mib": 0.0}
        if stock
        else None,
        "after_target": {
            "allocated_mib": 10.0,
            "reserved_mib": 20.0,
            "peak_allocated_mib": 100.0,
            "peak_reserved_mib": 120.0,
        },
    }


def test_cuda_memory_contract_preserves_candidate_pre_cuda_boundary(runner) -> None:
    assert runner._validate_cuda_memory(_memory_payload(stock=True), arm="stock")
    assert runner._validate_cuda_memory(
        _memory_payload(stock=False), arm="candidate"
    )

    initialized_candidate = _memory_payload(stock=False)
    initialized_candidate["bootstrap_reset_before_target"] = True
    initialized_candidate["before_target"] = {
        "allocated_mib": 0.0,
        "reserved_mib": 0.0,
    }
    with pytest.raises(runner.TimedRunError, match="initialized CUDA"):
        runner._validate_cuda_memory(initialized_candidate, arm="candidate")


def _candidate_runtime_payload(runner, tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "output"
    output.mkdir()
    (output / "subtype_results.csv").write_text(
        "Names,Subtype\ncase_a.nii.gz,0\n", encoding="utf-8"
    )
    (output / "subtype_probabilities.csv").write_text(
        "Names,Subtype,Probability_0,Probability_1,Probability_2\n"
        "case_a.nii.gz,0,0.8,0.1,0.1\n",
        encoding="utf-8",
    )
    snapshot = dict(runner.DETERMINISTIC_SNAPSHOT)
    return {
        "case_count": 1,
        "case_ids": ["case_a"],
        "process_id": 4321,
        "input_directory": str((tmp_path / "input").resolve()),
        "model_directory": str((tmp_path / "model").resolve()),
        "v5_extraction_mode": "neural_only",
        "classifier_pipeline": "assignment_conforming_v5_neural_case_head",
        "folds": [0],
        "checkpoint": runner.CHECKPOINT_NAME,
        "device": "cuda",
        "tile_step_size": 0.5,
        "tta_enabled": True,
        "gaussian_enabled": True,
        "overwrite": True,
        "input_files_unchanged_during_run": True,
        "checkpoint_unchanged_during_run": True,
        "model_configuration_unchanged_during_run": True,
        "inference_execution": {
            "tile_batch_oom_fallback_count": 0,
            "tta_batch_oom_fallback_count": 0,
            "tile_batch_size_requested": 1,
            "tta_batch_size_requested": 1,
            "v5_case_extractions_completed": 1,
            "v5_neural_head_forward_calls": 1,
            "v5_class_offset_applications": 1,
            "v5_feature_cache_reads": 0,
            "segmentation_export_logit_dtype": "torch.float16",
            "segmentation_export_logit_dtype_sequence": ["torch.float16"],
        },
        "deterministic_execution": {
            "policy": "strict_cuda_inference_v1",
            "configured_before_cuda_initialization": True,
            "autocast_cuda_float16": True,
            "settings_unchanged": True,
            "after_initial_configuration": snapshot,
            "after_predictor_construction": snapshot,
            "after_inference": snapshot,
            "conformance_lock": {
                "sha256": runner.DETERMINISM_LOCK_SHA256,
                "unchanged_during_run": True,
            },
        },
        "stock_export_conformance": {
            "export_logit_dtype": "torch.float16",
            "case_count_verified": 1,
            "all_case_exports_verified": True,
            "conformance_lock": {
                "sha256": runner.STOCK_EXPORT_LOCK_SHA256,
                "unchanged_during_run": True,
            },
        },
        "frozen_network": {
            "component_hashes_before": dict(runner.FROZEN_COMPONENT_HASHES),
            "component_hashes_after": dict(runner.FROZEN_COMPONENT_HASHES),
            "frozen_components_unchanged": True,
            "network_in_eval_mode": True,
            "any_network_parameter_requires_grad": False,
        },
        "neural_case_head_bundle": {
            "bundle_sha256": "b" * 64,
            "numeric_train_dataset_sha256": "c" * 64,
            "expected_bundle_sha256_verified": True,
            "bundle_loaded_strictly": True,
            "eligible_for_official": True,
            "head_in_eval_mode": True,
            "any_head_parameter_requires_grad": False,
            "head_state_unchanged": True,
            "head_state_sha256": "d" * 64,
            "head_state_sha256_before": "d" * 64,
            "head_state_sha256_after": "d" * 64,
        },
    }


def test_candidate_runtime_binds_frozen_components_bundle_and_zero_fallbacks(
    runner, tmp_path: Path
) -> None:
    runtime = _candidate_runtime_payload(runner, tmp_path)

    assert (
        runner._validate_candidate_runtime(
            runtime,
            case_ids=["case_a"],
            process_id=4321,
            input_directory=tmp_path / "input",
            model_directory=tmp_path / "model",
            output_directory=tmp_path / "output",
            expected_bundle_sha256="b" * 64,
            expected_numeric_train_dataset_sha256="c" * 64,
        )
        == 0
    )

    runtime["frozen_network"]["component_hashes_after"]["decoder"] = "0" * 64
    with pytest.raises(runner.TimedRunError, match="frozen network"):
        runner._validate_candidate_runtime(
            runtime,
            case_ids=["case_a"],
            process_id=4321,
            input_directory=tmp_path / "input",
            model_directory=tmp_path / "model",
            output_directory=tmp_path / "output",
            expected_bundle_sha256="b" * 64,
            expected_numeric_train_dataset_sha256="c" * 64,
        )


def test_stock_cpu_result_fallback_marker_is_fail_closed(runner) -> None:
    assert runner._stock_cpu_fallback_detected(
        "prefix\n" + runner.STOCK_CPU_FALLBACK_MARKER + "\nsuffix"
    )
    assert not runner._stock_cpu_fallback_detected("ordinary progress output")


def test_functional_smoke_schema_has_no_reconstructable_duration(
    runner, tmp_path: Path
) -> None:
    args = _arguments(tmp_path, arm="stock", purpose="functional_smoke")
    record = runner._base_record(args)

    assert record["execution_purpose"] == "functional_smoke"
    assert record["timing_eligible"] is False
    assert record["started_at_utc"] is None
    assert record["completed_at_utc"] is None
    assert record["monotonic_start_ns"] is None
    assert record["monotonic_end_ns"] is None
    assert record["elapsed_seconds"] is None
    assert record["final_candidate_lock_before"] is None
    assert record["one_use_ledger_before"] is None
    assert record["final_candidate_lock_unchanged_during_run"] is False
    assert record["one_use_ledger_unchanged_during_run"] is False


def test_functional_smoke_never_calls_monotonic_clock(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path, arm="stock", purpose="functional_smoke")
    record = runner._base_record(args)
    monkeypatch.setattr(
        runner.time,
        "monotonic_ns",
        lambda: pytest.fail("functional smoke cannot invoke a duration clock"),
    )

    started = runner._start_external_timing(record["timing_eligible"])
    runner._complete_external_timing(record, started)

    assert started is None
    assert record["event_at_utc"] is not None
    assert record["started_at_utc"] is None
    assert record["completed_at_utc"] is None
    assert record["elapsed_seconds"] is None


def test_final_benchmark_uses_exact_monotonic_boundaries(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path, arm="stock", purpose="final_benchmark")
    record = runner._base_record(args)
    values = iter((10_000_000_000, 12_500_000_000))
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(values))

    started = runner._start_external_timing(record["timing_eligible"])
    runner._complete_external_timing(record, started)

    assert record["monotonic_start_ns"] == 10_000_000_000
    assert record["monotonic_end_ns"] == 12_500_000_000
    assert record["elapsed_seconds"] == 2.5
    assert record["started_at_utc"] is not None
    assert record["completed_at_utc"] is not None
    assert record["event_at_utc"] is None


def test_functional_smoke_redacts_nested_candidate_and_bootstrap_timing(
    runner, tmp_path: Path
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "total_seconds": 12.5,
                "mean_seconds_per_case": 6.25,
                "timing_scope": "eligible",
            }
        ),
        encoding="utf-8",
    )
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text(
        json.dumps(
            {
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "completed_at_utc": "2026-01-01T00:00:12+00:00",
            }
        ),
        encoding="utf-8",
    )

    runner._redact_smoke_timing_artifact(candidate, candidate_runtime=True)
    runner._redact_smoke_timing_artifact(bootstrap, candidate_runtime=False)

    candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
    bootstrap_payload = json.loads(bootstrap.read_text(encoding="utf-8"))
    for payload in (candidate_payload, bootstrap_payload):
        assert payload["started_at_utc"] is None
        assert payload["completed_at_utc"] is None
        assert payload["timing_eligible"] is False
        assert payload["timing_fields_redacted"] is True
    assert candidate_payload["total_seconds"] is None
    assert candidate_payload["mean_seconds_per_case"] is None
    assert candidate_payload["timing_scope"] == runner.FUNCTIONAL_SMOKE_SCOPE


def test_sha256_argument_normalizes_and_rejects_invalid_values(runner) -> None:
    assert runner._sha256_argument("A" * 64) == "a" * 64
    with pytest.raises(argparse.ArgumentTypeError):
        runner._sha256_argument("not-a-hash")


def test_existing_external_runtime_is_never_overwritten(runner, tmp_path: Path) -> None:
    args = _arguments(tmp_path, arm="stock")
    original = b"immutable prior evidence\n"
    args.external_runtime_json.write_bytes(original)

    with pytest.raises(runner.TimedRunError, match="already exists"):
        runner.run_timed_inference(args)

    assert args.external_runtime_json.read_bytes() == original


def test_one_use_ledger_is_bound_to_final_lock_execution_and_abba_order(
    runner, tmp_path: Path
) -> None:
    args = _arguments(tmp_path, arm="stock")
    args.final_candidate_lock.write_text(
        json.dumps({"run_ledger_files": {"stock_speed": "ledger.json"}}),
        encoding="utf-8",
    )
    final_record = runner._file_record(args.final_candidate_lock)
    ledger = {
        "schema_version": 1,
        "status": "started_and_consumed",
        "stage": "single_locked_stock_inference_speed_benchmark",
        "benchmark_execution_id": args.benchmark_execution_id,
        "claimed_at_utc": "2026-08-06T12:00:00+00:00",
        "orchestrator_process_id": 123,
        "work_root": str(args.output_directory.resolve().parents[1]),
        "final_candidate_lock": {
            "path": str(args.final_candidate_lock.resolve()),
            "sha256": final_record["sha256"],
        },
        "stock_gate_lock": {
            "path": str(runner.STOCK_GATE_LOCK.resolve()),
            "sha256": runner.STOCK_GATE_LOCK_SHA256,
        },
        "determinism_lock": {
            "path": str(runner.DETERMINISM_LOCK.resolve()),
            "sha256": runner.DETERMINISM_LOCK_SHA256,
        },
        "stock_export_lock": {
            "path": str(runner.STOCK_EXPORT_LOCK.resolve()),
            "sha256": runner.STOCK_EXPORT_LOCK_SHA256,
        },
        "run_order": list(runner.FINAL_LABEL_TO_ARM),
        "test_targets_or_submission_feedback_used": False,
    }
    args.one_use_ledger.write_text(json.dumps(ledger), encoding="utf-8")
    ledger_record = runner._file_record(args.one_use_ledger)

    runner._validate_one_use_ledger(
        args,
        final_candidate_record=final_record,
        ledger_record=ledger_record,
    )

    ledger["run_order"] = list(reversed(ledger["run_order"]))
    args.one_use_ledger.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(runner.TimedRunError, match="exact stock benchmark"):
        runner._validate_one_use_ledger(
            args,
            final_candidate_record=final_record,
            ledger_record=runner._file_record(args.one_use_ledger),
        )


def test_execution_id_parser_requires_canonical_uuid4(runner) -> None:
    value = "12345678-1234-4234-9234-123456789abc"
    assert runner._uuid4_argument(value) == value
    with pytest.raises(argparse.ArgumentTypeError):
        runner._uuid4_argument("12345678-1234-1234-9234-123456789abc")
