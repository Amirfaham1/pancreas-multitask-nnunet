from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import sys
import uuid
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ["case_a", "case_b"]
SNAPSHOT = {
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "cublas_workspace_config": ":4096:8",
    "nnunet_compile": "false",
}


def _load_module():
    path = ROOT / "scripts" / "benchmark_stock_inference_speed.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_stock_inference_speed", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_v3_fixtures():
    path = ROOT / "tests" / "test_benchmark_inference_speed.py"
    spec = importlib.util.spec_from_file_location("v3_speed_fixture_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, digest: str | None = None) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved) if digest is None else digest,
        "size_bytes": resolved.stat().st_size,
    }


def _manifest(root: Path) -> dict[str, object]:
    resolved = root.resolve()
    files = [
        {
            "relative_path": path.relative_to(resolved).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(
            (item for item in resolved.rglob("*") if item.is_file()), key=str
        )
    ]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "root": str(resolved),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _write_outputs(
    path: Path,
    *,
    candidate: bool,
    probability_shift: float = 0.0,
    changed_mask: bool = False,
    changed_geometry: bool = False,
    changed_dtype: bool = False,
    invalid_label: bool = False,
    changed_subtype: bool = False,
) -> None:
    path.mkdir(parents=True)
    arrays = {
        "case_a": np.asarray(
            [[[0, 0], [1, 1]], [[0, 2], [1, 0]]], dtype=np.uint8
        ),
        "case_b": np.asarray(
            [[[2, 1], [0, 0]], [[1, 2], [0, 1]]], dtype=np.uint8
        ),
    }
    for case_id in CASE_IDS:
        array = arrays[case_id].copy()
        if changed_mask and case_id == "case_b":
            array[0, 1, 1] = 1
        if invalid_label and case_id == "case_b":
            array[0, 1, 1] = 3
        if changed_dtype:
            array = array.astype(np.int16)
        affine = np.eye(4)
        if changed_geometry and case_id == "case_b":
            affine[0, 3] = 1.0
        image = nib.Nifti1Image(array, affine)
        image.header.set_xyzt_units("mm", "sec")
        nib.save(image, path / f"{case_id}.nii.gz")
    if not candidate:
        return
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
        if changed_subtype:
            writer.writerow(("case_b.nii.gz", 2, 0.1, 0.1, 0.8))
        else:
            writer.writerow(
                (
                    "case_b.nii.gz",
                    1,
                    0.1,
                    0.8 + probability_shift,
                    0.1 - probability_shift,
                )
            )


def _candidate_runtime(
    module,
    v3,
    *,
    path: Path,
    input_directory: Path,
    input_manifest: dict[str, object],
    model_directory: Path,
    bundle_path: Path,
    process_id: int,
    started_at: str,
    environment: dict[str, object],
    peak_allocated: float,
    peak_reserved: float,
) -> dict[str, object]:
    runtime = v3._runtime(
        extraction_mode="neural_only",
        seconds_per_case=5.0,
        started_at=started_at,
        process_id=process_id,
    )
    external_files = input_manifest["files"]
    internal_files = [
        {
            "name": item["relative_path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in external_files
    ]
    canonical = json.dumps(internal_files, sort_keys=True, separators=(",", ":"))
    runtime["input_directory"] = str(input_directory.resolve())
    runtime["input_file_manifest"] = {
        "file_count": len(internal_files),
        "files": internal_files,
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    runtime["model_directory"] = str(model_directory.resolve())
    runtime["process_id"] = process_id
    runtime["started_at_utc"] = started_at
    runtime["python_version"] = environment["python_version"]
    runtime["torch_version"] = environment["torch_version"]
    runtime["cuda_runtime_version"] = environment["cuda_runtime_version"]
    runtime["cudnn_version"] = environment["cudnn_version"]
    runtime["device_name"] = environment["cuda_device_name"]
    runtime["device_capability"] = environment["cuda_device_capability"]
    runtime["peak_allocated_mib"] = peak_allocated
    runtime["peak_reserved_mib"] = peak_reserved
    bundle = runtime["neural_case_head_bundle"]
    bundle["bundle_path"] = str(bundle_path.resolve())
    bundle["bundle_name"] = bundle_path.name
    bundle["bundle_sha256"] = _sha256(bundle_path)
    bundle["bundle_size_bytes"] = bundle_path.stat().st_size
    bundle["numeric_train_dataset_sha256"] = "4" * 64
    path.write_text(json.dumps(runtime), encoding="utf-8")
    return runtime


def _bootstrap_audit(
    module,
    *,
    path: Path,
    arm: str,
    target_argv: list[str],
    process_id: int,
    cuda_memory: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if arm == "stock":
        import nnunetv2.inference.predict_from_raw_data as stock_module

        source_path = Path(stock_module.__file__).resolve()
        assert _sha256(source_path) == module.STOCK_SOURCE_SHA256
        target = {
            "module": "nnunetv2.inference.predict_from_raw_data",
            "entry_point": "predict_entry_point",
            "package": "nnunetv2",
            "package_version": "2.8.1",
        }
        constructor_count = 1
        construction_snapshot = deepcopy(SNAPSHOT)
    else:
        source_path = (ROOT / "scripts" / "predict_joint.py").resolve()
        target = {
            "module": "predict_joint",
            "entry_point": "main",
            "package": "pancreas-multitask",
            "package_version": "0.1.0",
        }
        constructor_count = 0
        construction_snapshot = None
    source_manifest = {
        str(source_path): {
            "sha256": _sha256(source_path),
            "size_bytes": source_path.stat().st_size,
        }
    }
    lock = _record(ROOT / "configs" / "inference_determinism_conformance_v1.json")
    audit = {
        "schema_version": 1,
        "mode": arm,
        "target_argv": target_argv,
        "process_id": process_id,
        "status": "succeeded",
        "exit_code": 0,
        "exception": None,
        "postflight_exception": None,
        "device": "cuda",
        "autocast_cuda_float16": True,
        "cuda_memory": cuda_memory,
        "determinism_lock_before": lock,
        "determinism_lock_after": deepcopy(lock),
        "determinism_lock_unchanged": True,
        "determinism_snapshots": {
            "after_initial_configuration": deepcopy(SNAPSHOT),
            "after_predictor_construction": construction_snapshot,
            "after_inference": deepcopy(SNAPSHOT),
        },
        "stock_constructor_reassertion_count": constructor_count,
        "installed_sources_before": source_manifest,
        "installed_sources_after": deepcopy(source_manifest),
        "installed_sources_unchanged": True,
        "target": target,
    }
    path.write_text(json.dumps(audit), encoding="utf-8")
    provenance = (
        {
            **target,
            "predict_from_raw_data": {
                "path": str(source_path),
                **source_manifest[str(source_path)],
            },
        }
        if arm == "stock"
        else target
    )
    return audit, provenance


def _fixture(
    tmp_path: Path,
    module,
    *,
    candidate_seconds: float = 80.0,
    probability_delta: float = 5e-7,
    output_options: dict[int, dict[str, object]] | None = None,
) -> dict[str, object]:
    v3 = _load_v3_fixtures()
    work_root = tmp_path / "stock-speed-work"
    work_root.mkdir(parents=True)
    audit_output = work_root / "stock_speed_audit.json"
    input_directory = tmp_path / "inputs"
    input_directory.mkdir()
    for index, case_id in enumerate(CASE_IDS):
        (input_directory / f"{case_id}_0000.nii.gz").write_bytes(
            f"synthetic-input-{index}".encode()
        )
    input_manifest = _manifest(input_directory)
    input_manifest["case_ids"] = list(CASE_IDS)

    model_directory = (
        tmp_path
        / "models"
        / "Dataset501_Synthetic"
        / module.MODEL_DIRECTORY_NAME
    )
    (model_directory / "fold_0").mkdir(parents=True)
    checkpoint_path = model_directory / "fold_0" / module.CHECKPOINT_NAME
    checkpoint_path.write_bytes(b"synthetic checkpoint")
    dataset_path = model_directory / "dataset.json"
    plans_path = model_directory / "plans.json"
    dataset_path.write_text("{}", encoding="utf-8")
    plans_path.write_text("{}", encoding="utf-8")
    module.CHECKPOINT_SHA256 = _sha256(checkpoint_path)
    module.MODEL_CONFIGURATION_SHA256 = {
        "dataset.json": _sha256(dataset_path),
        "plans.json": _sha256(plans_path),
    }
    checkpoint = {
        **_record(checkpoint_path),
        "fold": "0",
        "name": module.CHECKPOINT_NAME,
    }
    model_configuration = [
        {
            **_record(dataset_path),
            "name": "dataset.json",
        },
        {
            **_record(plans_path),
            "name": "plans.json",
        },
    ]

    bundle_path = tmp_path / "frozen" / "neural_case_head_bundle.pt"
    bundle_path.parent.mkdir()
    bundle_path.write_bytes(b"synthetic frozen neural bundle")
    required_speed_paths = (
        "scripts/run_deterministic_inference.py",
        "scripts/run_timed_inference_child.py",
        "scripts/Run-StockInferenceSpeedBenchmark.ps1",
        "scripts/benchmark_stock_inference_speed.py",
    )
    final_lock_path = tmp_path / "frozen" / "final_candidate_lock.json"
    final_lock = {
        "schema_version": 1,
        "implementation_files": [
            {"path": relative, "sha256": _sha256(ROOT / relative)}
            for relative in required_speed_paths
        ],
        "stock_speed_protocol_deviations": deepcopy(
            module.REQUIRED_STOCK_LOCK_DEVIATIONS
        ),
    }
    final_lock_path.write_text(json.dumps(final_lock), encoding="utf-8")
    final_lock_record = _record(final_lock_path)
    stock_gate_record = _record(module.STOCK_GATE_LOCK_PATH)
    determinism_record = _record(module.DETERMINISM_LOCK_PATH)
    stock_export_record = _record(module.STOCK_EXPORT_LOCK_PATH)
    ledger_path = tmp_path / "frozen" / "stock_speed_one_use_ledger.json"
    execution_id = str(uuid.uuid4())
    ledger = {
        "schema_version": 1,
        "status": "started_and_consumed",
        "stage": "single_locked_stock_inference_speed_benchmark",
        "benchmark_execution_id": execution_id,
        "claimed_at_utc": "2026-08-06T11:00:00+00:00",
        "orchestrator_process_id": 9001,
        "work_root": str(work_root.resolve()),
        "intended_audit_path": str(audit_output.resolve()),
        "run_order": list(module.EXPECTED_RUN_LABELS),
        "test_targets_or_submission_feedback_used": False,
        "final_candidate_lock": {
            "path": final_lock_record["path"],
            "sha256": final_lock_record["sha256"],
        },
        "stock_gate_lock": {
            "path": stock_gate_record["path"],
            "sha256": stock_gate_record["sha256"],
        },
        "determinism_lock": {
            "path": determinism_record["path"],
            "sha256": determinism_record["sha256"],
        },
        "stock_export_lock": {
            "path": stock_export_record["path"],
            "sha256": stock_export_record["sha256"],
        },
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    ledger_record = _record(ledger_path)
    bootstrap_source_record = _record(module.DETERMINISTIC_BOOTSTRAP_PATH)
    environment = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": "2.8.0+cu128",
        "nnunetv2_version": "2.8.1",
        "cuda_runtime_version": "12.8",
        "cudnn_version": 91002,
        "cuda_device_index": 0,
        "cuda_device_name": "Tesla T4",
        "cuda_device_capability": [7, 5],
        "cuda_device_uuid": "GPU-synthetic-t4",
        "nvidia_driver_version": "570.133.20",
        "nnunet_compile": "false",
        "cublas_workspace_config": ":4096:8",
        "power_thermal_query_id": module.NVIDIA_SMI_QUERY,
    }
    base = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    starts = (0, 120, 240, 360)
    durations = (100.0, candidate_seconds, candidate_seconds, 100.0)
    external_paths: list[Path] = []
    output_paths: list[Path] = []
    internal_paths: list[Path] = []
    output_options = output_options or {}
    for index, (label, arm) in enumerate(module.EXPECTED_RUNS):
        run_root = work_root / label
        run_root.mkdir()
        output_path = run_root / "output"
        options = dict(output_options.get(index, {}))
        if arm == "candidate" and index == 2:
            options.setdefault("probability_shift", probability_delta)
        _write_outputs(output_path, candidate=arm == "candidate", **options)
        output_paths.append(output_path)
        process_id = 101 + index
        launcher_process_id = 201 + index
        started = base + timedelta(seconds=starts[index])
        completed = started + timedelta(seconds=durations[index])
        peak_allocated = 2001.0
        peak_reserved = 2401.0
        cuda_memory = {
            "unit": "MiB",
            "collector": "torch.cuda process-local memory counters",
            "bootstrap_reset_before_target": arm == "stock",
            "before_target": (
                {"allocated_mib": 0.0, "reserved_mib": 0.0}
                if arm == "stock"
                else None
            ),
            "after_target": {
                "allocated_mib": 100.0,
                "reserved_mib": 200.0,
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
            },
        }
        internal_path: Path | None = None
        internal_runtime: dict[str, object] | None = None
        if arm == "candidate":
            internal_path = run_root / "candidate_runtime.json"
            internal_runtime = _candidate_runtime(
                module,
                v3,
                path=internal_path,
                input_directory=input_directory,
                input_manifest=input_manifest,
                model_directory=model_directory,
                bundle_path=bundle_path,
                process_id=process_id,
                started_at=(started + timedelta(seconds=5)).isoformat(),
                environment=environment,
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
            )
            internal_paths.append(internal_path)
            target_argv = module._candidate_target_argv(
                input_directory.resolve(),
                output_path.resolve(),
                model_directory.resolve(),
                internal_path.resolve(),
                internal_runtime,
            )
        else:
            target_argv = module._stock_target_argv(
                input_directory.resolve(), output_path.resolve()
            )
        bootstrap_path = run_root / "determinism_audit.json"
        _, target_provenance = _bootstrap_audit(
            module,
            path=bootstrap_path,
            arm=arm,
            target_argv=target_argv,
            process_id=process_id,
            cuda_memory=cuda_memory,
        )
        process_log = run_root / "process.log"
        process_log.write_text("Synthetic successful inference process.\n", encoding="utf-8")
        power_before = {
            "query_id": module.NVIDIA_SMI_QUERY,
            "index": 0,
            "name": environment["cuda_device_name"],
            "uuid": environment["cuda_device_uuid"],
            "driver_version": environment["nvidia_driver_version"],
            "performance_state": "P8",
            "power_draw_watts": 18.0 + index,
            "temperature_celsius": 40.0 + index,
        }
        power_after = {
            **power_before,
            "performance_state": "P0",
            "power_draw_watts": 62.0 + index,
            "temperature_celsius": 48.0 + index,
        }
        monotonic_start = int((1000 + starts[index]) * 1_000_000_000)
        elapsed = durations[index]
        bundle_record = _record(bundle_path)
        external = {
            "schema_version": 1,
            "execution_purpose": "final_benchmark",
            "timing_eligible": True,
            "run_label": label,
            "arm": arm,
            "status": "succeeded",
            "exception": None,
            "command_argv": [
                environment["python_executable"],
                str(module.DETERMINISTIC_BOOTSTRAP_PATH.resolve()),
                "--mode",
                arm,
                "--determinism-audit-json",
                str(bootstrap_path.resolve()),
                "--",
                *target_argv,
            ],
            "target_argv": target_argv,
            "launcher_process_id": launcher_process_id,
            "process_id": process_id,
            "started_at_utc": started.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "event_at_utc": None,
            "monotonic_start_ns": monotonic_start,
            "monotonic_end_ns": monotonic_start + int(elapsed * 1_000_000_000),
            "elapsed_seconds": elapsed,
            "timer": "time.monotonic_ns",
            "timing_scope": module.EXTERNAL_TIMING_SCOPE,
            "fresh_process": True,
            "warmup": "none",
            "exit_code": 0,
            "timed_out": False,
            "case_count": len(CASE_IDS),
            "failed_case_count": 0,
            "oom_fallback_count": 0,
            "input_directory": str(input_directory.resolve()),
            "input_manifest_before": deepcopy(input_manifest),
            "input_manifest_after": deepcopy(input_manifest),
            "input_unchanged_during_run": True,
            "output_directory": str(output_path.resolve()),
            "output_manifest": _manifest(output_path),
            "environment": deepcopy(environment),
            "cuda_memory": cuda_memory,
            "power_and_thermal_environment": {
                "query_id": module.NVIDIA_SMI_QUERY,
                "before": power_before,
                "after": power_after,
            },
            "inference_contract": module._expected_inference_contract(arm),
            "checkpoint_before": deepcopy(checkpoint),
            "checkpoint_after": deepcopy(checkpoint),
            "checkpoint_unchanged_during_run": True,
            "model_configuration_before": deepcopy(model_configuration),
            "model_configuration_after": deepcopy(model_configuration),
            "model_configuration_unchanged_during_run": True,
            "stock_gate_lock_before": deepcopy(stock_gate_record),
            "stock_gate_lock_after": deepcopy(stock_gate_record),
            "stock_gate_lock_unchanged_during_run": True,
            "determinism_lock_before": deepcopy(determinism_record),
            "determinism_lock_after": deepcopy(determinism_record),
            "determinism_lock_unchanged_during_run": True,
            "stock_export_lock_before": deepcopy(stock_export_record),
            "stock_export_lock_after": deepcopy(stock_export_record),
            "stock_export_lock_unchanged_during_run": True,
            "determinism_bootstrap_source_before": deepcopy(bootstrap_source_record),
            "determinism_bootstrap_source_after": deepcopy(bootstrap_source_record),
            "determinism_bootstrap_source_unchanged_during_run": True,
            "final_candidate_lock_before": deepcopy(final_lock_record),
            "final_candidate_lock_after": deepcopy(final_lock_record),
            "final_candidate_lock_unchanged_during_run": True,
            "one_use_ledger_before": deepcopy(ledger_record),
            "one_use_ledger_after": deepcopy(ledger_record),
            "one_use_ledger_unchanged_during_run": True,
            "benchmark_execution_id": execution_id,
            "determinism_bootstrap_audit": {
                **_record(bootstrap_path),
                "status_verified": True,
            },
            "process_log": _record(process_log),
            "stock_cpu_result_fallback_detected": False,
            "stock_provenance": target_provenance if arm == "stock" else None,
            "candidate_internal_runtime": (
                {**_record(internal_path), "schema_validated_by_timed_runner": True}
                if internal_path is not None
                else None
            ),
            "neural_case_head_bundle_before": (
                deepcopy(bundle_record) if arm == "candidate" else None
            ),
            "neural_case_head_bundle_after": (
                deepcopy(bundle_record) if arm == "candidate" else None
            ),
            "neural_case_head_bundle_unchanged_during_run": (
                True if arm == "candidate" else None
            ),
            "test_targets_or_submission_feedback_used": False,
        }
        external_path = run_root / "external_runtime.json"
        external_path.write_text(json.dumps(external), encoding="utf-8")
        external_paths.append(external_path)
    return {
        "external": external_paths,
        "outputs": output_paths,
        "internal": internal_paths,
        "audit_output": audit_output,
        "work_root": work_root,
        "ledger": ledger_path,
        "final_lock": final_lock_path,
    }


def _audit(module, fixture: dict[str, object], *, expected_case_count: int = 2):
    return module.audit_stock_benchmark(
        fixture["external"],
        fixture["outputs"],
        fixture["internal"],
        expected_case_count=expected_case_count,
        audit_output_path=fixture["audit_output"],
    )


def _mutate_external(path: Path, mutation) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mutate_internal(fixture: dict[str, object], candidate_index: int, mutation) -> None:
    internal_path = fixture["internal"][candidate_index]
    payload = json.loads(internal_path.read_text(encoding="utf-8"))
    mutation(payload)
    internal_path.write_text(json.dumps(payload), encoding="utf-8")
    external_index = candidate_index + 1
    external_path = fixture["external"][external_index]
    external = json.loads(external_path.read_text(encoding="utf-8"))
    external["candidate_internal_runtime"] = {
        **_record(internal_path),
        "schema_validated_by_timed_runner": True,
    }
    external_path.write_text(json.dumps(external), encoding="utf-8")


def _refresh_embedded_artifact(
    external_path: Path,
    *,
    field: str,
    artifact_path: Path,
    extra: dict[str, object] | None = None,
) -> None:
    payload = json.loads(external_path.read_text(encoding="utf-8"))
    payload[field] = {**_record(artifact_path), **(extra or {})}
    external_path.write_text(json.dumps(payload), encoding="utf-8")


def _rebind_ledger(fixture: dict[str, object]) -> None:
    ledger_record = _record(fixture["ledger"])
    for external_path in fixture["external"]:
        payload = json.loads(external_path.read_text(encoding="utf-8"))
        payload["one_use_ledger_before"] = deepcopy(ledger_record)
        payload["one_use_ledger_after"] = deepcopy(ledger_record)
        external_path.write_text(json.dumps(payload), encoding="utf-8")


def _rebind_final_lock(fixture: dict[str, object]) -> None:
    lock_record = _record(fixture["final_lock"])
    ledger_path = fixture["ledger"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["final_candidate_lock"] = {
        "path": lock_record["path"],
        "sha256": lock_record["sha256"],
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    _rebind_ledger(fixture)
    for external_path in fixture["external"]:
        payload = json.loads(external_path.read_text(encoding="utf-8"))
        payload["final_candidate_lock_before"] = deepcopy(lock_record)
        payload["final_candidate_lock_after"] = deepcopy(lock_record)
        external_path.write_text(json.dumps(payload), encoding="utf-8")


def test_accepts_exact_post_repair_abba_gate_and_label_two_masks(tmp_path: Path) -> None:
    module = _load_module()
    fixture = _fixture(tmp_path, module)

    result = _audit(module, fixture)

    assert result["accepted"] is True
    assert result["runtime_reduction_percent"] == pytest.approx(20.0)
    assert result["timing_passed"] is True
    assert result["numerical_equivalence"]["passed"] is True
    assert result["execution_manifest"]["run_order"] == list(
        module.EXPECTED_RUN_LABELS
    )
    assert [
        item["launcher_process_id"] for item in result["execution_manifest"]["runs"]
    ] == [201, 202, 203, 204]
    assert all(
        comparison["hard_mask_disagreeing_voxels"] == 0
        and comparison["value_domain_mismatch_cases"] == []
        for comparison in result["numerical_equivalence"]["mask_comparisons"]
    )
    assert (
        result["post_repair_provenance"]["speed_implementation_bindings"][
            "all_speed_executables_bound_in_final_candidate_lock"
        ]
        is True
    )
    assert result["stock_lock_deviations"]["final_lock_disclosure"] == (
        module.REQUIRED_STOCK_LOCK_DEVIATIONS
    )
    assert result["claim_boundary"]["globally_fastest_model_claim_allowed"] is False


@pytest.mark.parametrize(
    ("candidate_seconds", "expected"),
    ((90.0, True), (90.000001, False)),
)
def test_applies_exact_ten_percent_complete_process_threshold(
    tmp_path: Path, candidate_seconds: float, expected: bool
) -> None:
    module = _load_module()
    fixture = _fixture(tmp_path, module, candidate_seconds=candidate_seconds)

    result = _audit(module, fixture)

    assert result["timing_passed"] is expected
    assert result["accepted"] is expected


@pytest.mark.parametrize(
    "output_options",
    (
        {1: {"changed_mask": True}},
        {1: {"changed_geometry": True}},
        {1: {"changed_dtype": True}},
        {1: {"invalid_label": True}},
    ),
    ids=("mask-values", "geometry", "dtype", "label-3"),
)
def test_rejects_nonidentical_or_out_of_domain_masks(
    tmp_path: Path, output_options: dict[int, dict[str, object]]
) -> None:
    module = _load_module()
    fixture = _fixture(tmp_path, module, output_options=output_options)

    result = _audit(module, fixture)

    assert result["accepted"] is False
    assert result["numerical_equivalence"]["passed"] is False
    assert "exact_stock_candidate_segmentation_gate_failed" in result["rejection_reasons"]


@pytest.mark.parametrize(
    ("fixture_options", "output_options"),
    (
        ({"probability_delta": 1.000001e-6}, None),
        ({}, {2: {"changed_subtype": True}}),
    ),
    ids=("probability-delta", "subtype-decision"),
)
def test_rejects_candidate_repeat_classification_drift(
    tmp_path: Path,
    fixture_options: dict[str, object],
    output_options: dict[int, dict[str, object]] | None,
) -> None:
    module = _load_module()
    fixture = _fixture(
        tmp_path, module, output_options=output_options, **fixture_options
    )

    result = _audit(module, fixture)

    assert result["accepted"] is False
    assert result["numerical_equivalence"]["candidate_repeat_classification"][
        "passed"
    ] is False
    assert "candidate_repeat_classification_gate_failed" in result["rejection_reasons"]


@pytest.mark.parametrize(
    ("index", "field", "value"),
    (
        (0, "execution_purpose", "functional_smoke"),
        (0, "timing_eligible", False),
        (0, "timing_scope", "internal_model_only"),
        (0, "run_label", "candidate_0"),
        (0, "fresh_process", False),
        (0, "elapsed_seconds", 99.0),
        (0, "failed_case_count", 1),
        (0, "oom_fallback_count", 1),
        (0, "stock_cpu_result_fallback_detected", True),
    ),
    ids=(
        "smoke-purpose",
        "timing-ineligible",
        "timing-scope",
        "abba-label",
        "not-fresh",
        "elapsed",
        "failed-case",
        "oom",
        "stock-fallback-flag",
    ),
)
def test_rejects_ineligible_or_failed_external_records(
    tmp_path: Path, index: int, field: str, value: object
) -> None:
    module = _load_module()
    fixture = _fixture(tmp_path, module)
    _mutate_external(fixture["external"][index], lambda payload: payload.__setitem__(field, value))

    with pytest.raises(module.StockBenchmarkError):
        _audit(module, fixture)


def test_rejects_duplicate_launcher_pid_and_overlapping_abba_runs(tmp_path: Path) -> None:
    module = _load_module()
    duplicate = _fixture(tmp_path / "duplicate", module)
    first = json.loads(duplicate["external"][0].read_text(encoding="utf-8"))
    _mutate_external(
        duplicate["external"][1],
        lambda payload: payload.__setitem__(
            "launcher_process_id", first["launcher_process_id"]
        ),
    )
    with pytest.raises(module.StockBenchmarkError, match="distinct launcher"):
        _audit(module, duplicate)

    overlap = _fixture(tmp_path / "overlap", module)

    def make_overlap(payload: dict[str, object]) -> None:
        payload["started_at_utc"] = "2026-08-06T12:01:30+00:00"
        payload["completed_at_utc"] = "2026-08-06T12:02:50+00:00"
        payload["monotonic_start_ns"] = 1090_000_000_000
        payload["monotonic_end_ns"] = 1170_000_000_000

    _mutate_external(overlap["external"][1], make_overlap)
    with pytest.raises(module.StockBenchmarkError, match="overlap"):
        _audit(module, overlap)


def test_rejects_duplicate_actual_child_pid(tmp_path: Path) -> None:
    module = _load_module()
    fixture = _fixture(tmp_path, module)
    first = json.loads(fixture["external"][0].read_text(encoding="utf-8"))
    fourth_path = fixture["external"][3]
    fourth = json.loads(fourth_path.read_text(encoding="utf-8"))
    fourth["process_id"] = first["process_id"]
    bootstrap_path = Path(fourth["determinism_bootstrap_audit"]["path"])
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["process_id"] = first["process_id"]
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
    fourth["determinism_bootstrap_audit"] = {
        **_record(bootstrap_path),
        "status_verified": True,
    }
    fourth_path.write_text(json.dumps(fourth), encoding="utf-8")

    with pytest.raises(module.StockBenchmarkError, match="distinct fresh process"):
        _audit(module, fixture)


def test_rejects_input_model_hardware_and_output_tampering(tmp_path: Path) -> None:
    module = _load_module()
    cases = []

    input_case = _fixture(tmp_path / "input", module)
    input_file = tmp_path / "input" / "inputs" / "case_a_0000.nii.gz"
    input_file.write_bytes(b"changed input")
    cases.append(input_case)

    model_case = _fixture(tmp_path / "model", module)
    checkpoint = next((tmp_path / "model" / "models").rglob(module.CHECKPOINT_NAME))
    checkpoint.write_bytes(b"changed checkpoint")
    cases.append(model_case)

    hardware_case = _fixture(tmp_path / "hardware", module)
    _mutate_external(
        hardware_case["external"][2],
        lambda payload: payload["environment"].__setitem__(
            "nvidia_driver_version", "tampered-driver"
        ),
    )
    cases.append(hardware_case)

    output_case = _fixture(tmp_path / "output", module)
    mask_path = output_case["outputs"][1] / "case_a.nii.gz"
    mask_path.write_bytes(mask_path.read_bytes() + b"tamper")
    cases.append(output_case)

    for fixture in cases:
        with pytest.raises(module.StockBenchmarkError):
            _audit(module, fixture)


def test_rejects_determinism_log_and_candidate_internal_tampering(tmp_path: Path) -> None:
    module = _load_module()

    deterministic = _fixture(tmp_path / "determinism", module)
    external_path = deterministic["external"][0]
    external = json.loads(external_path.read_text(encoding="utf-8"))
    bootstrap_path = Path(external["determinism_bootstrap_audit"]["path"])
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["determinism_snapshots"]["after_inference"]["cudnn_benchmark"] = True
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
    _refresh_embedded_artifact(
        external_path,
        field="determinism_bootstrap_audit",
        artifact_path=bootstrap_path,
        extra={"status_verified": True},
    )
    with pytest.raises(module.StockBenchmarkError, match="deterministic settings"):
        _audit(module, deterministic)

    source_case = _fixture(tmp_path / "source", module)
    source_external_path = source_case["external"][1]
    source_external = json.loads(source_external_path.read_text(encoding="utf-8"))
    source_audit_path = Path(source_external["determinism_bootstrap_audit"]["path"])
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    wrong_source = ROOT / "README.md"
    wrong_manifest = {
        str(wrong_source.resolve()): {
            "sha256": _sha256(wrong_source),
            "size_bytes": wrong_source.stat().st_size,
        }
    }
    source_audit["installed_sources_before"] = wrong_manifest
    source_audit["installed_sources_after"] = deepcopy(wrong_manifest)
    source_audit_path.write_text(json.dumps(source_audit), encoding="utf-8")
    _refresh_embedded_artifact(
        source_external_path,
        field="determinism_bootstrap_audit",
        artifact_path=source_audit_path,
        extra={"status_verified": True},
    )
    with pytest.raises(module.StockBenchmarkError, match="repository predict_joint"):
        _audit(module, source_case)

    fallback = _fixture(tmp_path / "fallback", module)
    fallback_external = json.loads(
        fallback["external"][0].read_text(encoding="utf-8")
    )
    log_path = Path(fallback_external["process_log"]["path"])
    log_path.write_text(module.STOCK_CPU_FALLBACK_MARKER, encoding="utf-8")
    _refresh_embedded_artifact(
        fallback["external"][0], field="process_log", artifact_path=log_path
    )
    with pytest.raises(module.StockBenchmarkError, match="silent CPU fallback"):
        _audit(module, fallback)

    internal = _fixture(tmp_path / "internal", module)
    _mutate_internal(
        internal,
        0,
        lambda payload: payload["inference_execution"].__setitem__(
            "v5_feature_cache_reads", 1
        ),
    )
    with pytest.raises(module.StockBenchmarkError, match="strict v3 validator"):
        _audit(module, internal)

    hash_case = _fixture(tmp_path / "internal_hash", module)
    runtime_path = hash_case["internal"][0]
    runtime_path.write_text(
        runtime_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(module.StockBenchmarkError, match="retained file"):
        _audit(module, hash_case)


def test_rejects_final_lock_and_one_use_ledger_tampering(tmp_path: Path) -> None:
    module = _load_module()
    final_lock_case = _fixture(tmp_path / "lock", module)
    lock_path = final_lock_case["final_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["stock_speed_protocol_deviations"][
        "original_stock_lock_literal_compliance_was_perfect"
    ] = True
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    _rebind_final_lock(final_lock_case)
    with pytest.raises(module.StockBenchmarkError, match="deviation disclosure"):
        _audit(module, final_lock_case)

    ledger_case = _fixture(tmp_path / "ledger", module)
    ledger_path = ledger_case["ledger"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["test_targets_or_submission_feedback_used"] = True
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    _rebind_ledger(ledger_case)
    with pytest.raises(module.StockBenchmarkError, match="final ABBA run"):
        _audit(module, ledger_case)


@pytest.mark.parametrize("artifact", ("final_lock", "ledger"))
def test_rejects_changed_retained_lock_or_ledger_hash(
    tmp_path: Path, artifact: str
) -> None:
    module = _load_module()
    fixture = _fixture(tmp_path, module)
    path = fixture[artifact]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(module.StockBenchmarkError, match="retained file"):
        _audit(module, fixture)
