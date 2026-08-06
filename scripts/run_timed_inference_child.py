#!/usr/bin/env python3
"""Run one locked stock-speed arm in a fresh, externally timed process.

The timer surrounds the complete deterministic-inference child process.  This
runner constructs (rather than accepts) the target command line, records all
inputs and immutable model artifacts before and after execution, and writes an
atomic success or failure record for the stock-gate auditor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import torch

ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC_BOOTSTRAP = ROOT / "scripts" / "run_deterministic_inference.py"
STOCK_GATE_LOCK = ROOT / "configs" / "inference_speed_stock_gate_v1.json"
DETERMINISM_LOCK = ROOT / "configs" / "inference_determinism_conformance_v1.json"
STOCK_EXPORT_LOCK = ROOT / "configs" / "inference_stock_export_conformance_v1.json"

SCHEMA_VERSION = 1
STOCK_GATE_LOCK_SHA256 = (
    "563d9d5e4fbe0f92653c6b7295c476d0ddf5d239c47beb1948410bbb80a7c2e2"
)
DETERMINISM_LOCK_SHA256 = (
    "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd"
)
STOCK_EXPORT_LOCK_SHA256 = (
    "bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503"
)
STOCK_SOURCE_SHA256 = (
    "c350e3202a7a67c3aef12e9206a744add442110ff8a4377c1f9640104b20a31f"
)
CHECKPOINT_SHA256 = (
    "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116"
)
MODEL_CONFIGURATION_SHA256 = {
    "dataset.json": "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff",
    "plans.json": "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f",
}
FROZEN_COMPONENT_HASHES = {
    "encoder": "324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1",
    "decoder": "b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2",
    "classification": "1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8",
}
CHECKPOINT_NAME = "checkpoint_classification_rescue.pth"
MODEL_DIRECTORY_NAME = (
    "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres"
)
FINAL_LABEL_TO_ARM = {
    "stock_reference_1": "stock",
    "candidate_1": "candidate",
    "candidate_2": "candidate",
    "stock_reference_2": "stock",
}
FUNCTIONAL_SMOKE_LABEL_TO_ARM = {
    "stock_functional_smoke": "stock",
    "candidate_functional_smoke": "candidate",
}
EXPECTED_LABEL_TO_ARM = {**FINAL_LABEL_TO_ARM, **FUNCTIONAL_SMOKE_LABEL_TO_ARM}
DETERMINISTIC_SNAPSHOT = {
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "cublas_workspace_config": ":4096:8",
    "nnunet_compile": "false",
}
NVIDIA_SMI_QUERY = (
    "index,name,uuid,driver_version,pstate,power.draw,temperature.gpu"
)
STOCK_CPU_FALLBACK_MARKER = (
    "Prediction on device was unsuccessful, probably due to a lack of memory. "
    "Moving results arrays to CPU"
)
EXTERNAL_TIMING_SCOPE = (
    "external_monotonic_wall_clock_around_complete_fresh_deterministic_"
    "inference_child_process_including_startup_model_head_initialization_"
    "preprocessing_inference_geometry_restoration_and_all_file_exports"
)
FUNCTIONAL_SMOKE_SCOPE = (
    "train_only_functional_conformance_smoke_with_all_duration_evidence_redacted"
)


class TimedRunError(RuntimeError):
    """Raised when one locked timed arm cannot produce admissible evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, include_path: bool = True) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise TimedRunError(f"Required file does not exist: {resolved}")
    record: dict[str, Any] = {
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if include_path:
        record["path"] = str(resolved)
    return record


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(resolved)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        temporary.replace(resolved)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TimedRunError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TimedRunError(f"JSON artifact must contain an object: {path}")
    return payload


def _validate_one_use_ledger(
    args: argparse.Namespace,
    *,
    final_candidate_record: dict[str, Any],
    ledger_record: dict[str, Any],
) -> None:
    final_lock = _load_object(args.final_candidate_lock.expanduser().resolve())
    ledger = _load_object(args.one_use_ledger.expanduser().resolve())
    ledgers = final_lock.get("run_ledger_files")
    stock_filename = ledgers.get("stock_speed") if isinstance(ledgers, dict) else None
    if (
        not isinstance(stock_filename, str)
        or not stock_filename.endswith(".json")
        or Path(stock_filename).name != stock_filename
        or stock_filename != stock_filename.lower()
    ):
        raise TimedRunError("Final-candidate lock lacks a bare stock_speed ledger filename")
    expected_ledger_path = args.final_candidate_lock.resolve().parent / stock_filename
    if args.one_use_ledger.resolve() != expected_ledger_path:
        raise TimedRunError("One-use ledger path differs from final-candidate lock contract")
    expected_order = list(FINAL_LABEL_TO_ARM)
    if (
        ledger.get("schema_version") != 1
        or ledger.get("status") != "started_and_consumed"
        or ledger.get("stage") != "single_locked_stock_inference_speed_benchmark"
        or ledger.get("benchmark_execution_id") != args.benchmark_execution_id
        or ledger.get("run_order") != expected_order
        or ledger.get("test_targets_or_submission_feedback_used") is not False
        or ledger.get("final_candidate_lock")
        != {
            "path": str(args.final_candidate_lock.resolve()),
            "sha256": final_candidate_record["sha256"],
        }
        or ledger.get("stock_gate_lock")
        != {"path": str(STOCK_GATE_LOCK.resolve()), "sha256": STOCK_GATE_LOCK_SHA256}
        or ledger.get("determinism_lock")
        != {"path": str(DETERMINISM_LOCK.resolve()), "sha256": DETERMINISM_LOCK_SHA256}
        or ledger.get("stock_export_lock")
        != {"path": str(STOCK_EXPORT_LOCK.resolve()), "sha256": STOCK_EXPORT_LOCK_SHA256}
    ):
        raise TimedRunError("One-use ledger does not bind the exact stock benchmark")
    process_id = ledger.get("orchestrator_process_id")
    if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
        raise TimedRunError("One-use ledger lacks a positive orchestrator process ID")
    if ledger.get("work_root") != str(args.output_directory.resolve().parents[1]):
        raise TimedRunError("One-use ledger work_root differs from the timed output root")
    claimed_at = ledger.get("claimed_at_utc")
    if not isinstance(claimed_at, str):
        raise TimedRunError("One-use ledger lacks claimed_at_utc")
    try:
        parsed = datetime.fromisoformat(claimed_at)
    except ValueError as error:
        raise TimedRunError("One-use ledger claimed_at_utc is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TimedRunError("One-use ledger claimed_at_utc must include a timezone")
    if ledger_record["sha256"] != _sha256(args.one_use_ledger.resolve()):
        raise TimedRunError("One-use ledger content changed during preflight")


def _redact_smoke_timing_artifact(path: Path, *, candidate_runtime: bool) -> None:
    """Remove reconstructable durations from a train-only functional artifact."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return
    payload = _load_object(resolved)
    payload["started_at_utc"] = None
    payload["completed_at_utc"] = None
    if candidate_runtime:
        payload["total_seconds"] = None
        payload["mean_seconds_per_case"] = None
        payload["timing_scope"] = FUNCTIONAL_SMOKE_SCOPE
    payload["timing_eligible"] = False
    payload["timing_fields_redacted"] = True
    _atomic_json(resolved, payload)


def _manifest_for_paths(paths: Sequence[Path], *, base: Path) -> dict[str, Any]:
    resolved_base = base.expanduser().resolve()
    files: list[dict[str, Any]] = []
    for path in sorted((item.expanduser().resolve() for item in paths), key=str):
        try:
            relative = path.relative_to(resolved_base).as_posix()
        except ValueError as error:
            raise TimedRunError(f"Manifest file is outside {resolved_base}: {path}") from error
        record = _file_record(path, include_path=False)
        files.append({"relative_path": relative, **record})
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "root": str(resolved_base),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _input_manifest(input_directory: Path) -> tuple[dict[str, Any], list[str]]:
    resolved = input_directory.expanduser().resolve()
    if not resolved.is_dir():
        raise TimedRunError(f"Input directory does not exist: {resolved}")
    paths = sorted(resolved.glob("*.nii.gz"), key=lambda item: item.name)
    if not paths:
        raise TimedRunError(f"No .nii.gz inputs found in {resolved}")
    invalid = [path.name for path in paths if not path.name.endswith("_0000.nii.gz")]
    if invalid:
        raise TimedRunError(f"Every input must be a one-channel *_0000.nii.gz file: {invalid}")
    case_ids = [path.name[: -len("_0000.nii.gz")] for path in paths]
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        raise TimedRunError("Input case identifiers must be nonempty and unique")
    manifest = _manifest_for_paths(paths, base=resolved)
    manifest["case_ids"] = case_ids
    return manifest, case_ids


def _output_manifest(output_directory: Path) -> dict[str, Any]:
    resolved = output_directory.expanduser().resolve()
    if not resolved.is_dir():
        raise TimedRunError(f"Output directory was not created: {resolved}")
    paths = sorted((path for path in resolved.rglob("*") if path.is_file()), key=str)
    if not paths:
        raise TimedRunError(f"Inference produced no output files: {resolved}")
    return _manifest_for_paths(paths, base=resolved)


def _model_records(model_directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = model_directory.expanduser().resolve()
    if not resolved.is_dir() or resolved.name != MODEL_DIRECTORY_NAME:
        raise TimedRunError(
            "Model directory must be the locked Dataset501 trainer/plans/configuration folder"
        )
    if not resolved.parent.name.startswith("Dataset501_"):
        raise TimedRunError("Locked stock comparison requires a Dataset501 model directory")
    checkpoint = _file_record(resolved / "fold_0" / CHECKPOINT_NAME)
    checkpoint["fold"] = "0"
    checkpoint["name"] = CHECKPOINT_NAME
    if checkpoint["sha256"] != CHECKPOINT_SHA256:
        raise TimedRunError("Fold-0 checkpoint differs from the frozen v5 checkpoint")
    configurations: list[dict[str, Any]] = []
    for name, expected_hash in MODEL_CONFIGURATION_SHA256.items():
        record = _file_record(resolved / name)
        record["name"] = name
        if record["sha256"] != expected_hash:
            raise TimedRunError(f"{name} differs from the frozen v5 artifact")
        configurations.append(record)
    return checkpoint, configurations


def _parse_nvidia_smi_number(value: str, *, field: str) -> float:
    try:
        result = float(value.strip())
    except ValueError as error:
        raise TimedRunError(f"nvidia-smi returned invalid {field}: {value!r}") from error
    if not math.isfinite(result):
        raise TimedRunError(f"nvidia-smi returned non-finite {field}")
    return result


def _stock_cpu_fallback_detected(process_output: str) -> bool:
    return STOCK_CPU_FALLBACK_MARKER in process_output


def _nvidia_smi_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--query-gpu={NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TimedRunError(f"Cannot record the required NVIDIA environment: {error}") from error
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise TimedRunError(f"Expected one NVIDIA device record, got {len(lines)}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 7:
        raise TimedRunError(f"Unexpected nvidia-smi record: {lines[0]!r}")
    index, name, uuid, driver, pstate, power, temperature = fields
    if index != "0" or not all((name, uuid, driver, pstate)):
        raise TimedRunError("nvidia-smi did not identify the locked CUDA device")
    return {
        "query_id": NVIDIA_SMI_QUERY,
        "index": 0,
        "name": name,
        "uuid": uuid,
        "driver_version": driver,
        "performance_state": pstate,
        "power_draw_watts": _parse_nvidia_smi_number(power, field="power.draw"),
        "temperature_celsius": _parse_nvidia_smi_number(
            temperature, field="temperature.gpu"
        ),
    }


def _cuda_environment(snapshot: dict[str, Any], python_executable: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise TimedRunError("The stock speed gate requires CUDA")
    if torch.cuda.device_count() != 1:
        raise TimedRunError("The stock speed gate requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    property_uuid = getattr(properties, "uuid", None)
    normalized_property_uuid = str(property_uuid).removeprefix("GPU-").lower()
    normalized_smi_uuid = str(snapshot["uuid"]).removeprefix("GPU-").lower()
    if property_uuid is not None and normalized_property_uuid != normalized_smi_uuid:
        raise TimedRunError("PyTorch and nvidia-smi report different CUDA device UUIDs")
    try:
        nnunet_version = importlib.metadata.version("nnunetv2")
    except importlib.metadata.PackageNotFoundError as error:
        raise TimedRunError("nnunetv2 is not installed") from error
    if nnunet_version != "2.8.1":
        raise TimedRunError(f"Stock comparison requires nnunetv2 2.8.1, got {nnunet_version}")
    return {
        "python_executable": str(python_executable.expanduser().resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "nnunetv2_version": nnunet_version,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_index": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_device_uuid": snapshot["uuid"],
        "nvidia_driver_version": snapshot["driver_version"],
        "nnunet_compile": "false",
        "cublas_workspace_config": ":4096:8",
        "power_thermal_query_id": NVIDIA_SMI_QUERY,
    }


def _target_arguments(args: argparse.Namespace) -> list[str]:
    input_directory = str(args.input_directory.expanduser().resolve())
    output_directory = str(args.output_directory.expanduser().resolve())
    model_directory = str(args.model_directory.expanduser().resolve())
    if args.arm == "stock":
        return [
            "-i",
            input_directory,
            "-o",
            output_directory,
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
            CHECKPOINT_NAME,
            "-npp",
            "3",
            "-nps",
            "3",
            "-device",
            "cuda",
        ]

    bundle = args.neural_case_head_bundle.expanduser().resolve()
    candidate_runtime = args.candidate_runtime_json.expanduser().resolve()
    probability_csv = args.output_directory.expanduser().resolve() / "subtype_probabilities.csv"
    return [
        "--input",
        input_directory,
        "--output",
        output_directory,
        "--model",
        model_directory,
        "--folds",
        "0",
        "--checkpoint",
        CHECKPOINT_NAME,
        "--classification-mode",
        "neural-v5",
        "--v5-extraction-mode",
        "neural_only",
        "--neural-case-head-bundle",
        str(bundle),
        "--expected-neural-case-head-bundle-sha256",
        args.expected_neural_case_head_bundle_sha256,
        "--expected-numeric-train-dataset-sha256",
        args.expected_numeric_train_dataset_sha256,
        "--runtime-json",
        str(candidate_runtime),
        "--probability-csv",
        str(probability_csv),
        "--device",
        "cuda",
        "--tile-step-size",
        "0.5",
        "--tile-batch-size",
        "1",
        "--tta-batch-size",
        "1",
        "--overwrite",
    ]


def _validate_bootstrap_audit(
    audit: dict[str, Any], *, arm: str, target_argv: list[str]
) -> dict[str, Any]:
    process_id = audit.get("process_id")
    if (
        audit.get("schema_version") != 1
        or audit.get("mode") != arm
        or audit.get("target_argv") != target_argv
        or isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        or audit.get("status") != "succeeded"
        or audit.get("exit_code") != 0
        or audit.get("exception") is not None
        or audit.get("postflight_exception") is not None
        or audit.get("device") != "cuda"
        or audit.get("autocast_cuda_float16") is not True
    ):
        raise TimedRunError("Deterministic bootstrap success record is inconsistent")
    if (
        audit.get("determinism_lock_unchanged") is not True
        or audit.get("determinism_lock_before")
        != audit.get("determinism_lock_after")
        or not isinstance(audit.get("determinism_lock_before"), dict)
        or audit["determinism_lock_before"].get("sha256")
        != DETERMINISM_LOCK_SHA256
    ):
        raise TimedRunError("Determinism lock changed or was not the frozen lock")
    snapshots = audit.get("determinism_snapshots")
    if not isinstance(snapshots, dict):
        raise TimedRunError("Deterministic bootstrap lacks settings snapshots")
    required_stages = ["after_initial_configuration", "after_inference"]
    if arm == "stock":
        required_stages.append("after_predictor_construction")
        if audit.get("stock_constructor_reassertion_count") != 1:
            raise TimedRunError("Stock predictor constructor was not reasserted exactly once")
    elif (
        audit.get("stock_constructor_reassertion_count") != 0
        or snapshots.get("after_predictor_construction") is not None
    ):
        raise TimedRunError("Candidate bootstrap unexpectedly patched a stock constructor")
    for stage in required_stages:
        if snapshots.get(stage) != DETERMINISTIC_SNAPSHOT:
            raise TimedRunError(f"Invalid deterministic settings at {stage}")
    if (
        audit.get("installed_sources_unchanged") is not True
        or audit.get("installed_sources_before") != audit.get("installed_sources_after")
        or not isinstance(audit.get("installed_sources_before"), dict)
        or not audit["installed_sources_before"]
    ):
        raise TimedRunError("Inference source changed during child execution")
    target = audit.get("target")
    if not isinstance(target, dict):
        raise TimedRunError("Deterministic bootstrap lacks target provenance")
    if arm == "stock":
        if target != {
            "module": "nnunetv2.inference.predict_from_raw_data",
            "entry_point": "predict_entry_point",
            "package": "nnunetv2",
            "package_version": "2.8.1",
        }:
            raise TimedRunError("Unexpected installed stock entry point")
        matching_sources = [
            {"path": path, **record}
            for path, record in audit["installed_sources_before"].items()
            if isinstance(record, dict) and record.get("sha256") == STOCK_SOURCE_SHA256
        ]
        if len(matching_sources) != 1:
            raise TimedRunError("Bootstrap did not bind the exact installed stock source")
        return {
            **target,
            "predict_from_raw_data": matching_sources[0],
        }
    if target != {
        "module": "predict_joint",
        "entry_point": "main",
        "package": "pancreas-multitask",
        "package_version": "0.1.0",
    }:
        raise TimedRunError("Unexpected repository candidate entry point")
    return target


def _validate_cuda_memory(payload: object, *, arm: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TimedRunError("Deterministic bootstrap lacks CUDA memory evidence")
    before = payload.get("before_target")
    after = payload.get("after_target")
    if (
        payload.get("unit") != "MiB"
        or payload.get("collector") != "torch.cuda process-local memory counters"
        or not isinstance(after, dict)
    ):
        raise TimedRunError("CUDA memory evidence has an invalid collection contract")
    if arm == "stock":
        if payload.get("bootstrap_reset_before_target") is not True or not isinstance(
            before, dict
        ):
            raise TimedRunError("Stock CUDA memory counters were not reset before target")
        containers = ((before, ("allocated_mib", "reserved_mib")),)
    else:
        if payload.get("bootstrap_reset_before_target") is not False or before is not None:
            raise TimedRunError("Candidate bootstrap initialized CUDA before candidate.run")
        containers = ()
    for container, keys in (
        *containers,
        (
            after,
            (
                "allocated_mib",
                "reserved_mib",
                "peak_allocated_mib",
                "peak_reserved_mib",
            ),
        ),
    ):
        for key in keys:
            value = container.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise TimedRunError(f"CUDA memory field {key} must be finite and nonnegative")
    if (
        after["peak_allocated_mib"] <= 0
        or after["peak_reserved_mib"] <= 0
        or after["peak_allocated_mib"] < after["allocated_mib"]
        or after["peak_reserved_mib"] < after["reserved_mib"]
    ):
        raise TimedRunError("CUDA peak-memory evidence is inconsistent")
    return payload


def _validate_candidate_runtime(
    runtime: dict[str, Any],
    *,
    case_ids: list[str],
    process_id: int,
    input_directory: Path,
    model_directory: Path,
    output_directory: Path,
    expected_bundle_sha256: str,
    expected_numeric_train_dataset_sha256: str,
) -> int:
    case_count = len(case_ids)
    execution = runtime.get("inference_execution")
    deterministic = runtime.get("deterministic_execution")
    stock_export = runtime.get("stock_export_conformance")
    frozen_network = runtime.get("frozen_network")
    bundle = runtime.get("neural_case_head_bundle")
    if (
        runtime.get("case_count") != case_count
        or runtime.get("case_ids") != case_ids
        or runtime.get("process_id") != process_id
        or runtime.get("input_directory") != str(input_directory.resolve())
        or runtime.get("model_directory") != str(model_directory.resolve())
        or runtime.get("v5_extraction_mode") != "neural_only"
        or runtime.get("classifier_pipeline")
        != "assignment_conforming_v5_neural_case_head"
        or runtime.get("folds") != [0]
        or runtime.get("checkpoint") != CHECKPOINT_NAME
        or runtime.get("device") != "cuda"
        or runtime.get("tile_step_size") != 0.5
        or runtime.get("tta_enabled") is not True
        or runtime.get("gaussian_enabled") is not True
        or runtime.get("overwrite") is not True
        or runtime.get("input_files_unchanged_during_run") is not True
        or runtime.get("checkpoint_unchanged_during_run") is not True
        or runtime.get("model_configuration_unchanged_during_run") is not True
    ):
        raise TimedRunError("Candidate runtime violates the locked inference contract")
    if not isinstance(execution, dict):
        raise TimedRunError("Candidate runtime lacks execution counters")
    oom_count = sum(
        execution.get(key, -1)
        for key in ("tile_batch_oom_fallback_count", "tta_batch_oom_fallback_count")
    )
    if (
        oom_count != 0
        or execution.get("tile_batch_size_requested") != 1
        or execution.get("tta_batch_size_requested") != 1
        or execution.get("v5_case_extractions_completed") != case_count
        or execution.get("v5_neural_head_forward_calls") != case_count
        or execution.get("v5_class_offset_applications") != case_count
        or execution.get("v5_feature_cache_reads") != 0
        or execution.get("segmentation_export_logit_dtype") != "torch.float16"
        or execution.get("segmentation_export_logit_dtype_sequence")
        != ["torch.float16"] * case_count
    ):
        raise TimedRunError("Candidate runtime reports fallback or incomplete v5 execution")
    if (
        not isinstance(deterministic, dict)
        or deterministic.get("policy") != "strict_cuda_inference_v1"
        or deterministic.get("configured_before_cuda_initialization") is not True
        or deterministic.get("autocast_cuda_float16") is not True
        or deterministic.get("settings_unchanged") is not True
        or any(
            deterministic.get(stage) != DETERMINISTIC_SNAPSHOT
            for stage in (
                "after_initial_configuration",
                "after_predictor_construction",
                "after_inference",
            )
        )
        or deterministic.get("conformance_lock", {}).get("sha256")
        != DETERMINISM_LOCK_SHA256
        or deterministic.get("conformance_lock", {}).get("unchanged_during_run")
        is not True
    ):
        raise TimedRunError("Candidate runtime lacks exact deterministic provenance")
    if (
        not isinstance(stock_export, dict)
        or stock_export.get("export_logit_dtype") != "torch.float16"
        or stock_export.get("case_count_verified") != case_count
        or stock_export.get("all_case_exports_verified") is not True
        or stock_export.get("conformance_lock", {}).get("sha256")
        != STOCK_EXPORT_LOCK_SHA256
        or stock_export.get("conformance_lock", {}).get("unchanged_during_run")
        is not True
    ):
        raise TimedRunError("Candidate runtime lacks stock-equivalent export provenance")
    if (
        not isinstance(frozen_network, dict)
        or frozen_network.get("component_hashes_before") != FROZEN_COMPONENT_HASHES
        or frozen_network.get("component_hashes_after") != FROZEN_COMPONENT_HASHES
        or frozen_network.get("frozen_components_unchanged") is not True
        or frozen_network.get("network_in_eval_mode") is not True
        or frozen_network.get("any_network_parameter_requires_grad") is not False
    ):
        raise TimedRunError("Candidate runtime does not bind the frozen network state")
    if (
        not isinstance(bundle, dict)
        or bundle.get("bundle_sha256") != expected_bundle_sha256
        or bundle.get("numeric_train_dataset_sha256")
        != expected_numeric_train_dataset_sha256
        or bundle.get("expected_bundle_sha256_verified") is not True
        or bundle.get("bundle_loaded_strictly") is not True
        or bundle.get("eligible_for_official") is not True
        or bundle.get("head_in_eval_mode") is not True
        or bundle.get("any_head_parameter_requires_grad") is not False
        or bundle.get("head_state_unchanged") is not True
        or bundle.get("head_state_sha256_before") != bundle.get("head_state_sha256")
        or bundle.get("head_state_sha256_after") != bundle.get("head_state_sha256")
    ):
        raise TimedRunError("Candidate runtime does not bind the frozen neural-v5 head")
    classification = output_directory.resolve() / "subtype_results.csv"
    probabilities = output_directory.resolve() / "subtype_probabilities.csv"
    if not classification.is_file() or not probabilities.is_file():
        raise TimedRunError("Candidate did not export both required subtype CSV files")
    return oom_count


def _base_record(args: argparse.Namespace) -> dict[str, Any]:
    timing_eligible = args.execution_purpose == "final_benchmark"
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_purpose": args.execution_purpose,
        "timing_eligible": timing_eligible,
        "run_label": args.run_label,
        "arm": args.arm,
        "status": "failed",
        "exception": None,
        "command_argv": None,
        "target_argv": None,
        "launcher_process_id": None,
        "process_id": None,
        "started_at_utc": None,
        "completed_at_utc": None,
        "event_at_utc": None,
        "monotonic_start_ns": None,
        "monotonic_end_ns": None,
        "elapsed_seconds": None,
        "timer": "time.monotonic_ns",
        "timing_scope": EXTERNAL_TIMING_SCOPE if timing_eligible else FUNCTIONAL_SMOKE_SCOPE,
        "fresh_process": True,
        "warmup": "none",
        "exit_code": None,
        "timed_out": False,
        "case_count": None,
        "failed_case_count": None,
        "oom_fallback_count": None,
        "input_directory": None,
        "input_manifest_before": None,
        "input_manifest_after": None,
        "input_unchanged_during_run": False,
        "output_directory": None,
        "output_manifest": None,
        "environment": None,
        "cuda_memory": None,
        "power_and_thermal_environment": None,
        "inference_contract": None,
        "checkpoint_before": None,
        "checkpoint_after": None,
        "checkpoint_unchanged_during_run": False,
        "model_configuration_before": None,
        "model_configuration_after": None,
        "model_configuration_unchanged_during_run": False,
        "stock_gate_lock_before": None,
        "stock_gate_lock_after": None,
        "stock_gate_lock_unchanged_during_run": False,
        "determinism_lock_before": None,
        "determinism_lock_after": None,
        "determinism_lock_unchanged_during_run": False,
        "stock_export_lock_before": None,
        "stock_export_lock_after": None,
        "stock_export_lock_unchanged_during_run": False,
        "final_candidate_lock_before": None,
        "final_candidate_lock_after": None,
        "final_candidate_lock_unchanged_during_run": False,
        "one_use_ledger_before": None,
        "one_use_ledger_after": None,
        "one_use_ledger_unchanged_during_run": False,
        "benchmark_execution_id": args.benchmark_execution_id,
        "determinism_bootstrap_audit": None,
        "determinism_bootstrap_source_before": None,
        "determinism_bootstrap_source_after": None,
        "determinism_bootstrap_source_unchanged_during_run": False,
        "process_log": None,
        "stock_cpu_result_fallback_detected": None,
        "stock_provenance": None,
        "candidate_internal_runtime": None,
        "neural_case_head_bundle_before": None,
        "neural_case_head_bundle_after": None,
        "neural_case_head_bundle_unchanged_during_run": None,
        "test_targets_or_submission_feedback_used": False,
    }


def _start_external_timing(timing_eligible: bool) -> tuple[str, int] | None:
    if not timing_eligible:
        return None
    return datetime.now(UTC).isoformat(), time.monotonic_ns()


def _complete_external_timing(
    record: dict[str, Any], started: tuple[str, int] | None
) -> None:
    if not record["timing_eligible"]:
        if started is not None:
            raise TimedRunError("Functional smoke unexpectedly started an external timer")
        record["event_at_utc"] = datetime.now(UTC).isoformat()
        return
    if started is None:
        raise TimedRunError("Final benchmark lacks an external timer start")
    started_at, start_ns = started
    end_ns = time.monotonic_ns()
    record["started_at_utc"] = started_at
    record["completed_at_utc"] = datetime.now(UTC).isoformat()
    record["monotonic_start_ns"] = start_ns
    record["monotonic_end_ns"] = end_ns
    record["elapsed_seconds"] = (end_ns - start_ns) / 1_000_000_000


def run_timed_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one locked arm and return its complete external timing record."""

    audit_path = args.external_runtime_json.expanduser().resolve()
    if audit_path.exists():
        raise TimedRunError(
            f"Fresh-run external runtime artifact already exists: {audit_path}"
        )
    record = _base_record(args)
    try:
        expected_labels = (
            FINAL_LABEL_TO_ARM
            if args.execution_purpose == "final_benchmark"
            else FUNCTIONAL_SMOKE_LABEL_TO_ARM
        )
        if expected_labels.get(args.run_label) != args.arm:
            raise TimedRunError("Run label and arm do not match the locked ABBA protocol")
        if args.expected_case_count < 1:
            raise TimedRunError("expected_case_count must be positive")
        if args.timeout_seconds <= 0:
            raise TimedRunError("timeout_seconds must be positive")
        if args.output_directory.expanduser().resolve().exists():
            raise TimedRunError("Output directory must not exist before a fresh timed run")
        for generated_path in (
            args.determinism_audit_json,
            args.candidate_runtime_json if args.arm == "candidate" else None,
        ):
            if generated_path is not None and generated_path.expanduser().resolve().exists():
                raise TimedRunError(f"Fresh-run artifact already exists: {generated_path}")
        if args.process_log.expanduser().resolve().exists():
            raise TimedRunError(f"Fresh-run process log already exists: {args.process_log}")

        stock_lock_before = _file_record(STOCK_GATE_LOCK)
        if stock_lock_before["sha256"] != STOCK_GATE_LOCK_SHA256:
            raise TimedRunError("Stock speed gate lock SHA-256 mismatch")
        determinism_lock_before = _file_record(DETERMINISM_LOCK)
        if determinism_lock_before["sha256"] != DETERMINISM_LOCK_SHA256:
            raise TimedRunError("Determinism conformance lock SHA-256 mismatch")
        stock_export_lock_before = _file_record(STOCK_EXPORT_LOCK)
        if stock_export_lock_before["sha256"] != STOCK_EXPORT_LOCK_SHA256:
            raise TimedRunError("Stock export conformance lock SHA-256 mismatch")
        deterministic_bootstrap_before = _file_record(DETERMINISTIC_BOOTSTRAP)
        if deterministic_bootstrap_before["size_bytes"] < 1:
            raise TimedRunError("Deterministic bootstrap is empty")
        record["determinism_bootstrap_source_before"] = deterministic_bootstrap_before
        record["stock_gate_lock_before"] = stock_lock_before
        record["determinism_lock_before"] = determinism_lock_before
        record["stock_export_lock_before"] = stock_export_lock_before
        final_candidate_before: dict[str, Any] | None = None
        ledger_before: dict[str, Any] | None = None
        if args.execution_purpose == "final_benchmark":
            if (
                args.final_candidate_lock is None
                or args.expected_final_candidate_lock_sha256 is None
                or args.one_use_ledger is None
            ):
                raise TimedRunError(
                    "Final benchmark requires the final-candidate lock and one-use ledger"
                )
            final_candidate_before = _file_record(args.final_candidate_lock)
            if (
                final_candidate_before["sha256"]
                != args.expected_final_candidate_lock_sha256
            ):
                raise TimedRunError("Final-candidate lock SHA-256 mismatch")
            ledger_before = _file_record(args.one_use_ledger)
            record["final_candidate_lock_before"] = final_candidate_before
            record["one_use_ledger_before"] = ledger_before
            _validate_one_use_ledger(
                args,
                final_candidate_record=final_candidate_before,
                ledger_record=ledger_before,
            )
        elif any(
            value is not None
            for value in (
                args.final_candidate_lock,
                args.expected_final_candidate_lock_sha256,
                args.one_use_ledger,
            )
        ):
            raise TimedRunError(
                "Functional smoke cannot consume or bind the final benchmark ledger"
            )

        input_before, case_ids = _input_manifest(args.input_directory)
        if len(case_ids) != args.expected_case_count:
            raise TimedRunError(
                f"Expected {args.expected_case_count} cases, discovered {len(case_ids)}"
            )
        checkpoint_before, configurations_before = _model_records(args.model_directory)
        record.update(
            {
                "case_count": len(case_ids),
                "input_directory": str(args.input_directory.expanduser().resolve()),
                "input_manifest_before": input_before,
                "output_directory": str(args.output_directory.expanduser().resolve()),
                "checkpoint_before": checkpoint_before,
                "model_configuration_before": configurations_before,
            }
        )

        bundle_before: dict[str, Any] | None = None
        if args.arm == "candidate":
            if (
                args.neural_case_head_bundle is None
                or args.candidate_runtime_json is None
                or args.expected_neural_case_head_bundle_sha256 is None
                or args.expected_numeric_train_dataset_sha256 is None
            ):
                raise TimedRunError("Candidate arm requires all neural-v5 bindings")
            bundle_before = _file_record(args.neural_case_head_bundle)
            if bundle_before["sha256"] != args.expected_neural_case_head_bundle_sha256:
                raise TimedRunError("Neural case-head bundle SHA-256 mismatch")
            record["neural_case_head_bundle_before"] = bundle_before
        elif any(
            value is not None
            for value in (
                args.neural_case_head_bundle,
                args.candidate_runtime_json,
                args.expected_neural_case_head_bundle_sha256,
                args.expected_numeric_train_dataset_sha256,
            )
        ):
            raise TimedRunError("Stock arm cannot receive candidate-only bindings")

        target_argv = _target_arguments(args)
        command = [
            str(args.python_executable.expanduser().resolve()),
            str(DETERMINISTIC_BOOTSTRAP.resolve()),
            "--mode",
            args.arm,
            "--determinism-audit-json",
            str(args.determinism_audit_json.expanduser().resolve()),
            "--",
            *target_argv,
        ]
        record["command_argv"] = command
        record["target_argv"] = target_argv
        record["inference_contract"] = {
            "dataset_id": 501,
            "trainer": "nnUNetTrainerPancreasMultiTask",
            "plans": "nnUNetResEncUNetMPlans",
            "configuration": "3d_fullres",
            "folds": [0],
            "checkpoint": CHECKPOINT_NAME,
            "device": "cuda",
            "tile_step_size": 0.5,
            "tta_enabled": True,
            "gaussian_enabled": True,
            "torch_compile": False,
            "save_segmentation_probabilities": False,
            "perform_everything_on_device": True,
            "preprocessing_processes": 3 if args.arm == "stock" else None,
            "segmentation_export_processes": 3 if args.arm == "stock" else None,
            "tile_batch_size": 1 if args.arm == "candidate" else None,
            "tta_batch_size": 1 if args.arm == "candidate" else None,
            "classification_mode": "neural-v5" if args.arm == "candidate" else None,
            "v5_extraction_mode": "neural_only" if args.arm == "candidate" else None,
            "workload": (
                "stock_segmentation_prediction_and_nifti_export"
                if args.arm == "stock"
                else "segmentation_neural_v5_subtype_nifti_and_csv_export"
            ),
            "candidate_workload_is_strictly_broader": args.arm == "candidate",
        }

        environment = os.environ.copy()
        environment["nnUNet_extTrainer"] = str((ROOT / "src").resolve())
        environment["nnUNet_compile"] = "false"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["nnUNet_results"] = str(
            args.model_directory.expanduser().resolve().parents[1]
        )
        power_before = _nvidia_smi_snapshot()
        record["environment"] = _cuda_environment(power_before, args.python_executable)
        record["power_and_thermal_environment"] = {
            "query_id": NVIDIA_SMI_QUERY,
            "before": power_before,
            "after": None,
        }

        timing_start = _start_external_timing(record["timing_eligible"])
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        record["launcher_process_id"] = process.pid
        process_output = ""
        try:
            process_output, _ = process.communicate(timeout=args.timeout_seconds)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            process.kill()
            remaining_output, _ = process.communicate()
            process_output += remaining_output
            raise TimedRunError(f"Inference child exceeded {args.timeout_seconds} seconds")
        finally:
            _complete_external_timing(record, timing_start)
            record["exit_code"] = process.returncode
            power_after = _nvidia_smi_snapshot()
            record["power_and_thermal_environment"]["after"] = power_after
            fallback_detected = _stock_cpu_fallback_detected(process_output)
            persisted_output = (
                process_output
                if record["timing_eligible"]
                else (
                    "Train-only functional-smoke process output redacted; "
                    f"stock_cpu_result_fallback_detected={fallback_detected}.\n"
                )
            )
            _atomic_text(args.process_log, persisted_output)
            record["process_log"] = _file_record(args.process_log)
            record["stock_cpu_result_fallback_detected"] = fallback_detected
        if exit_code != 0:
            raise TimedRunError(f"Inference child exited with code {exit_code}")
        if fallback_detected:
            raise TimedRunError("Stock inference silently retried result accumulation on CPU")

        bootstrap_path = args.determinism_audit_json.expanduser().resolve()
        bootstrap = _load_object(bootstrap_path)
        record["process_id"] = bootstrap.get("process_id")
        target_provenance = _validate_bootstrap_audit(
            bootstrap,
            arm=args.arm,
            target_argv=target_argv,
        )
        record["determinism_bootstrap_audit"] = {
            **_file_record(bootstrap_path),
            "status_verified": True,
        }
        record["cuda_memory"] = _validate_cuda_memory(
            bootstrap.get("cuda_memory"), arm=args.arm
        )
        if args.arm == "stock":
            record["stock_provenance"] = target_provenance
            oom_count = 0
        else:
            runtime_path = args.candidate_runtime_json.expanduser().resolve()
            runtime = _load_object(runtime_path)
            oom_count = _validate_candidate_runtime(
                runtime,
                case_ids=case_ids,
                process_id=record["process_id"],
                input_directory=args.input_directory,
                model_directory=args.model_directory,
                output_directory=args.output_directory,
                expected_bundle_sha256=args.expected_neural_case_head_bundle_sha256,
                expected_numeric_train_dataset_sha256=(
                    args.expected_numeric_train_dataset_sha256
                ),
            )
            record["candidate_internal_runtime"] = {
                **_file_record(runtime_path),
                "schema_validated_by_timed_runner": True,
            }
            internal_seconds = runtime.get("total_seconds")
            if record["timing_eligible"] and (
                isinstance(internal_seconds, bool)
                or not isinstance(internal_seconds, (int, float))
                or not math.isfinite(float(internal_seconds))
                or internal_seconds <= 0
                or internal_seconds > record["elapsed_seconds"]
            ):
                raise TimedRunError("Candidate internal timing exceeds external child timing")
            bundle_after = _file_record(args.neural_case_head_bundle)
            record["neural_case_head_bundle_after"] = bundle_after
            record["neural_case_head_bundle_unchanged_during_run"] = (
                bundle_after == bundle_before
            )
            if bundle_after != bundle_before:
                raise TimedRunError("Neural case-head bundle changed during timed inference")

        input_after, case_ids_after = _input_manifest(args.input_directory)
        checkpoint_after, configurations_after = _model_records(args.model_directory)
        stock_lock_after = _file_record(STOCK_GATE_LOCK)
        determinism_lock_after = _file_record(DETERMINISM_LOCK)
        stock_export_lock_after = _file_record(STOCK_EXPORT_LOCK)
        final_candidate_after = (
            _file_record(args.final_candidate_lock)
            if args.execution_purpose == "final_benchmark"
            else None
        )
        ledger_after = (
            _file_record(args.one_use_ledger)
            if args.execution_purpose == "final_benchmark"
            else None
        )
        deterministic_bootstrap_after = _file_record(DETERMINISTIC_BOOTSTRAP)
        record.update(
            {
                "input_manifest_after": input_after,
                "input_unchanged_during_run": input_before == input_after
                and case_ids == case_ids_after,
                "checkpoint_after": checkpoint_after,
                "checkpoint_unchanged_during_run": checkpoint_before == checkpoint_after,
                "model_configuration_after": configurations_after,
                "model_configuration_unchanged_during_run": (
                    configurations_before == configurations_after
                ),
                "stock_gate_lock_after": stock_lock_after,
                "stock_gate_lock_unchanged_during_run": (
                    stock_lock_before == stock_lock_after
                ),
                "determinism_lock_after": determinism_lock_after,
                "determinism_lock_unchanged_during_run": (
                    determinism_lock_before == determinism_lock_after
                ),
                "stock_export_lock_after": stock_export_lock_after,
                "stock_export_lock_unchanged_during_run": (
                    stock_export_lock_before == stock_export_lock_after
                ),
                "final_candidate_lock_after": final_candidate_after,
                "final_candidate_lock_unchanged_during_run": (
                    final_candidate_before == final_candidate_after
                    if args.execution_purpose == "final_benchmark"
                    else None
                ),
                "one_use_ledger_after": ledger_after,
                "one_use_ledger_unchanged_during_run": (
                    ledger_before == ledger_after
                    if args.execution_purpose == "final_benchmark"
                    else None
                ),
                "determinism_bootstrap_source_after": deterministic_bootstrap_after,
                "determinism_bootstrap_source_unchanged_during_run": (
                    deterministic_bootstrap_before == deterministic_bootstrap_after
                ),
                "output_manifest": _output_manifest(args.output_directory),
                "failed_case_count": 0,
                "oom_fallback_count": oom_count,
            }
        )
        immutable_checks = [
            record["input_unchanged_during_run"],
            record["checkpoint_unchanged_during_run"],
            record["model_configuration_unchanged_during_run"],
            record["stock_gate_lock_unchanged_during_run"],
            record["determinism_lock_unchanged_during_run"],
            record["stock_export_lock_unchanged_during_run"],
            record["determinism_bootstrap_source_unchanged_during_run"],
        ]
        if args.execution_purpose == "final_benchmark":
            immutable_checks.extend(
                (
                    record["final_candidate_lock_unchanged_during_run"],
                    record["one_use_ledger_unchanged_during_run"],
                )
            )
        if not all(immutable_checks):
            raise TimedRunError("A locked input, model artifact, or protocol changed")
        mask_names = sorted(
            path.name for path in args.output_directory.resolve().glob("*.nii.gz")
        )
        if mask_names != [f"{case_id}.nii.gz" for case_id in case_ids]:
            raise TimedRunError("Output segmentation inventory does not match input cases")
        if (
            record["power_and_thermal_environment"]["before"]["uuid"]
            != record["power_and_thermal_environment"]["after"]["uuid"]
            or record["power_and_thermal_environment"]["before"]["driver_version"]
            != record["power_and_thermal_environment"]["after"]["driver_version"]
        ):
            raise TimedRunError("CUDA device identity changed during the timed child")
        record["status"] = "succeeded"
        return record
    except BaseException as error:
        record["exception"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if args.execution_purpose == "functional_smoke":
            _redact_smoke_timing_artifact(
                args.determinism_audit_json, candidate_runtime=False
            )
            if args.candidate_runtime_json is not None:
                _redact_smoke_timing_artifact(
                    args.candidate_runtime_json, candidate_runtime=True
                )
            bootstrap_path = args.determinism_audit_json.expanduser().resolve()
            if bootstrap_path.is_file() and record["determinism_bootstrap_audit"] is not None:
                record["determinism_bootstrap_audit"] = {
                    **_file_record(bootstrap_path),
                    "status_verified": True,
                    "timing_redacted": True,
                }
            if args.candidate_runtime_json is not None:
                runtime_path = args.candidate_runtime_json.expanduser().resolve()
                if runtime_path.is_file() and record["candidate_internal_runtime"] is not None:
                    record["candidate_internal_runtime"] = {
                        **_file_record(runtime_path),
                        "schema_validated_by_timed_runner": True,
                        "timing_redacted": True,
                    }
        _atomic_json(audit_path, record)


def _sha256_argument(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise argparse.ArgumentTypeError("Expected a 64-character lowercase SHA-256")
    return normalized


def _uuid4_argument(value: str) -> str:
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a canonical UUIDv4") from error
    canonical = str(parsed)
    if parsed.version != 4 or value.lower() != canonical:
        raise argparse.ArgumentTypeError("Expected a canonical UUIDv4")
    return canonical


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-purpose",
        choices=("final_benchmark", "functional_smoke"),
        required=True,
    )
    parser.add_argument("--run-label", choices=tuple(EXPECTED_LABEL_TO_ARM), required=True)
    parser.add_argument("--arm", choices=("stock", "candidate"), required=True)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--external-runtime-json", type=Path, required=True)
    parser.add_argument("--determinism-audit-json", type=Path, required=True)
    parser.add_argument("--process-log", type=Path, required=True)
    parser.add_argument("--final-candidate-lock", type=Path)
    parser.add_argument(
        "--expected-final-candidate-lock-sha256", type=_sha256_argument
    )
    parser.add_argument("--one-use-ledger", type=Path)
    parser.add_argument("--benchmark-execution-id", type=_uuid4_argument, required=True)
    parser.add_argument("--candidate-runtime-json", type=Path)
    parser.add_argument("--neural-case-head-bundle", type=Path)
    parser.add_argument(
        "--expected-neural-case-head-bundle-sha256", type=_sha256_argument
    )
    parser.add_argument("--expected-numeric-train-dataset-sha256", type=_sha256_argument)
    parser.add_argument("--expected-case-count", type=int, default=72)
    parser.add_argument("--timeout-seconds", type=float, default=21_600.0)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_timed_inference(args)
    run_description = (
        "locked timed run"
        if args.execution_purpose == "final_benchmark"
        else "functional conformance run"
    )
    print(f"Completed {run_description}: {args.run_label}")
    print(f"External runtime: {args.external_runtime_json.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
