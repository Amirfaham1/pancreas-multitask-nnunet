#!/usr/bin/env python3
"""Audit the frozen post-repair stock-versus-v5 inference-speed gate.

The auditor never launches inference.  It accepts four externally timed
fresh-process records and their retained outputs in the frozen ABBA order,
independently validates the candidate's two internal runtime artifacts, and
applies the exact numerical and >=10 percent complete-process speed gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import mean
from typing import Any

import nibabel as nib
import numpy as np

try:
    from benchmark_inference_speed import (  # type: ignore[import-not-found]
        BenchmarkError as V3BenchmarkError,
    )
    from benchmark_inference_speed import (  # type: ignore[import-not-found]
        _validate_matched_runtime_contract as _validate_v3_matched_runtime,
    )
    from benchmark_inference_speed import (  # type: ignore[import-not-found]
        _validate_runtime as _validate_v3_runtime,
    )
except ModuleNotFoundError as error:
    if error.name != "benchmark_inference_speed":
        raise
    from scripts.benchmark_inference_speed import BenchmarkError as V3BenchmarkError
    from scripts.benchmark_inference_speed import (
        _validate_matched_runtime_contract as _validate_v3_matched_runtime,
    )
    from scripts.benchmark_inference_speed import _validate_runtime as _validate_v3_runtime

ROOT = Path(__file__).resolve().parents[1]
STOCK_GATE_LOCK_PATH = ROOT / "configs" / "inference_speed_stock_gate_v1.json"
DETERMINISM_LOCK_PATH = (
    ROOT / "configs" / "inference_determinism_conformance_v1.json"
)
STOCK_EXPORT_LOCK_PATH = (
    ROOT / "configs" / "inference_stock_export_conformance_v1.json"
)
DETERMINISTIC_BOOTSTRAP_PATH = ROOT / "scripts" / "run_deterministic_inference.py"

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
CHECKPOINT_NAME = "checkpoint_classification_rescue.pth"
MODEL_DIRECTORY_NAME = (
    "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres"
)
EXPECTED_RUNS = (
    ("stock_reference_1", "stock"),
    ("candidate_1", "candidate"),
    ("candidate_2", "candidate"),
    ("stock_reference_2", "stock"),
)
EXPECTED_RUN_LABELS = tuple(label for label, _ in EXPECTED_RUNS)
EXPECTED_ARMS = tuple(arm for _, arm in EXPECTED_RUNS)
REPEATS_PER_ARM = 2
MINIMUM_RUNTIME_REDUCTION_PERCENT = 10.0
MAXIMUM_PROBABILITY_DELTA = 1e-6
EXTERNAL_TIMING_SCOPE = (
    "external_monotonic_wall_clock_around_complete_fresh_deterministic_"
    "inference_child_process_including_startup_model_head_initialization_"
    "preprocessing_inference_geometry_restoration_and_all_file_exports"
)
NVIDIA_SMI_QUERY = (
    "index,name,uuid,driver_version,pstate,power.draw,temperature.gpu"
)
STOCK_CPU_FALLBACK_MARKER = (
    "Prediction on device was unsuccessful, probably due to a lack of memory. "
    "Moving results arrays to CPU"
)
DETERMINISTIC_SNAPSHOT = {
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "cublas_workspace_config": ":4096:8",
    "nnunet_compile": "false",
}
PROBABILITY_FILENAME = "subtype_probabilities.csv"
SUBTYPE_FILENAME = "subtype_results.csv"
REQUIRED_STOCK_LOCK_DEVIATIONS = {
    "post_stock_lock_train_only_conformance_artifacts_contained_timing_fields": True,
    "diagnostic_conformance_timings_eligible_for_final_speed_arithmetic": False,
    "candidate_implementation_changed_after_original_stock_lock": True,
    "original_stock_lock_literal_compliance_was_perfect": False,
    "repairs_were_limited_to_determinism_and_stock_export_conformance": True,
    "each_repair_was_locked_before_its_implementation_edit": True,
    "official_validation_or_test_data_used_for_repairs": False,
    "model_weights_features_head_or_offsets_changed_by_repairs": False,
}


class StockBenchmarkError(ValueError):
    """Raised when evidence violates the frozen stock-speed protocol."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise StockBenchmarkError(f"Cannot hash required artifact {path}: {error}") from error
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StockBenchmarkError(f"Cannot read {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise StockBenchmarkError(f"{description} must contain a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StockBenchmarkError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StockBenchmarkError(f"{key} must be finite")
    return result


def _parse_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise StockBenchmarkError(f"{field} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise StockBenchmarkError(f"Invalid {field}: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StockBenchmarkError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _resolved_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StockBenchmarkError(f"{field} must be a nonempty resolved path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise StockBenchmarkError(f"{field} must be absolute: {value}")
    return path.resolve()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise StockBenchmarkError(f"Required artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validate_file_record(
    value: object,
    *,
    description: str,
    expected_sha256: str | None = None,
    expected_path: Path | None = None,
    verify_current_file: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StockBenchmarkError(f"{description} must be a file record")
    path = _resolved_path(value.get("path"), field=f"{description}.path")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if (
        not _is_sha256(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise StockBenchmarkError(f"{description} has invalid SHA-256 or size")
    if expected_sha256 is not None and digest != expected_sha256:
        raise StockBenchmarkError(f"{description} differs from its frozen SHA-256")
    if expected_path is not None and path != expected_path.resolve():
        raise StockBenchmarkError(f"{description} points to the wrong artifact")
    record = {"path": str(path), "sha256": digest, "size_bytes": size}
    if verify_current_file and _file_record(path) != record:
        raise StockBenchmarkError(f"{description} no longer matches the retained file")
    return record


def _manifest_digest(files: list[dict[str, Any]]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _actual_manifest(root: Path) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise StockBenchmarkError(f"Manifest root does not exist: {resolved}")
    paths = sorted((path for path in resolved.rglob("*") if path.is_file()), key=str)
    files = [
        {
            "relative_path": path.relative_to(resolved).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {
        "root": str(resolved),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": _manifest_digest(files),
    }


def _validate_manifest(
    value: object,
    *,
    expected_root: Path,
    description: str,
    verify_current_files: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StockBenchmarkError(f"{description} must be an object")
    root = _resolved_path(value.get("root"), field=f"{description}.root")
    if root != expected_root.expanduser().resolve():
        raise StockBenchmarkError(f"{description} has the wrong root")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise StockBenchmarkError(f"{description} contains no files")
    normalized: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise StockBenchmarkError(f"{description} contains a non-object file entry")
        relative = item.get("relative_path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or "\\" in relative
            or not _is_sha256(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
        ):
            raise StockBenchmarkError(f"{description} contains an invalid file record")
        normalized.append(
            {"relative_path": relative, "sha256": digest, "size_bytes": size}
        )
    names = [item["relative_path"] for item in normalized]
    if names != sorted(set(names)):
        raise StockBenchmarkError(f"{description} file inventory is not sorted and unique")
    if (
        value.get("file_count") != len(normalized)
        or value.get("manifest_sha256") != _manifest_digest(normalized)
    ):
        raise StockBenchmarkError(f"{description} digest or file_count is inconsistent")
    result = {
        "root": str(root),
        "file_count": len(normalized),
        "files": normalized,
        "manifest_sha256": value["manifest_sha256"],
    }
    if verify_current_files and _actual_manifest(root) != result:
        raise StockBenchmarkError(f"{description} differs from retained files")
    return result


def _expected_inference_contract(arm: str) -> dict[str, Any]:
    return {
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
        "preprocessing_processes": 3 if arm == "stock" else None,
        "segmentation_export_processes": 3 if arm == "stock" else None,
        "tile_batch_size": 1 if arm == "candidate" else None,
        "tta_batch_size": 1 if arm == "candidate" else None,
        "classification_mode": "neural-v5" if arm == "candidate" else None,
        "v5_extraction_mode": "neural_only" if arm == "candidate" else None,
        "workload": (
            "stock_segmentation_prediction_and_nifti_export"
            if arm == "stock"
            else "segmentation_neural_v5_subtype_nifti_and_csv_export"
        ),
        "candidate_workload_is_strictly_broader": arm == "candidate",
    }


def _validate_environment(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StockBenchmarkError("External runtime lacks its CUDA environment")
    required_strings = (
        "python_executable",
        "python_version",
        "torch_version",
        "cuda_runtime_version",
        "cuda_device_name",
        "cuda_device_uuid",
        "nvidia_driver_version",
    )
    if any(not isinstance(value.get(key), str) or not value[key] for key in required_strings):
        raise StockBenchmarkError("CUDA environment contains an empty identity field")
    python_executable = _resolved_path(
        value["python_executable"], field="environment.python_executable"
    )
    if not python_executable.is_file():
        raise StockBenchmarkError("Recorded Python executable no longer exists")
    cudnn_version = value.get("cudnn_version")
    capability = value.get("cuda_device_capability")
    if (
        value.get("nnunetv2_version") != "2.8.1"
        or value.get("cuda_device_index") != 0
        or isinstance(cudnn_version, bool)
        or not isinstance(cudnn_version, int)
        or cudnn_version < 1
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
        or value.get("nnunet_compile") != "false"
        or value.get("cublas_workspace_config") != ":4096:8"
        or value.get("power_thermal_query_id") != NVIDIA_SMI_QUERY
    ):
        raise StockBenchmarkError("CUDA environment violates the locked execution contract")
    return dict(value)


def _validate_gpu_snapshot(
    value: object,
    *,
    environment: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StockBenchmarkError(f"{description} must be an NVIDIA snapshot")
    for key in ("name", "uuid", "driver_version", "performance_state"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise StockBenchmarkError(f"{description} lacks {key}")
    power = _require_number(value, "power_draw_watts")
    temperature = _require_number(value, "temperature_celsius")
    if (
        value.get("query_id") != NVIDIA_SMI_QUERY
        or value.get("index") != 0
        or value.get("name") != environment["cuda_device_name"]
        or value.get("uuid") != environment["cuda_device_uuid"]
        or value.get("driver_version") != environment["nvidia_driver_version"]
        or power < 0
        or not 0 <= temperature <= 120
    ):
        raise StockBenchmarkError(f"{description} is inconsistent with the CUDA device")
    return dict(value)


def _validate_power_and_thermal(
    value: object, *, environment: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("query_id") != NVIDIA_SMI_QUERY:
        raise StockBenchmarkError("Power/thermal evidence has the wrong collector")
    before = _validate_gpu_snapshot(
        value.get("before"),
        environment=environment,
        description="power_and_thermal_environment.before",
    )
    after = _validate_gpu_snapshot(
        value.get("after"),
        environment=environment,
        description="power_and_thermal_environment.after",
    )
    if (
        before["uuid"] != after["uuid"]
        or before["driver_version"] != after["driver_version"]
    ):
        raise StockBenchmarkError("CUDA identity changed during a timed child")
    return {"query_id": NVIDIA_SMI_QUERY, "before": before, "after": after}


def _validate_cuda_memory(value: object, *, arm: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StockBenchmarkError("External runtime lacks CUDA memory evidence")
    before = value.get("before_target")
    after = value.get("after_target")
    if (
        value.get("unit") != "MiB"
        or value.get("collector") != "torch.cuda process-local memory counters"
        or not isinstance(after, dict)
    ):
        raise StockBenchmarkError("CUDA memory evidence has the wrong collector")
    containers: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    if arm == "stock":
        if value.get("bootstrap_reset_before_target") is not True or not isinstance(
            before, dict
        ):
            raise StockBenchmarkError("Stock CUDA peaks were not reset before inference")
        containers.append((before, ("allocated_mib", "reserved_mib")))
    elif value.get("bootstrap_reset_before_target") is not False or before is not None:
        raise StockBenchmarkError("Candidate bootstrap initialized CUDA before candidate.run")
    containers.append(
        (
            after,
            (
                "allocated_mib",
                "reserved_mib",
                "peak_allocated_mib",
                "peak_reserved_mib",
            ),
        )
    )
    for container, keys in containers:
        for key in keys:
            number = _require_number(container, key)
            if number < 0:
                raise StockBenchmarkError(f"CUDA memory field {key} must be nonnegative")
    if (
        float(after["peak_allocated_mib"]) <= 0
        or float(after["peak_reserved_mib"]) <= 0
        or float(after["peak_allocated_mib"]) < float(after["allocated_mib"])
        or float(after["peak_reserved_mib"]) < float(after["reserved_mib"])
    ):
        raise StockBenchmarkError("CUDA peak-memory evidence is inconsistent")
    return dict(value)


def _validate_source_manifest(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise StockBenchmarkError("Bootstrap source manifest is empty")
    result: dict[str, dict[str, Any]] = {}
    for raw_path, raw_record in value.items():
        path = _resolved_path(raw_path, field="installed source path")
        if not isinstance(raw_record, dict):
            raise StockBenchmarkError("Bootstrap source manifest contains a non-object")
        digest = raw_record.get("sha256")
        size = raw_record.get("size_bytes")
        if (
            not _is_sha256(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
        ):
            raise StockBenchmarkError("Bootstrap source manifest has invalid provenance")
        actual = _file_record(path)
        if actual["sha256"] != digest or actual["size_bytes"] != size:
            raise StockBenchmarkError("Installed inference source changed after timing")
        result[str(path)] = {"sha256": digest, "size_bytes": size}
    return result


def _validate_bootstrap_audit(
    path: Path,
    embedded_record: object,
    *,
    arm: str,
    target_argv: list[str],
    process_id: int,
    cuda_memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(embedded_record, dict) or embedded_record.get(
        "status_verified"
    ) is not True:
        raise StockBenchmarkError("External runtime did not verify its bootstrap audit")
    normalized_record = _validate_file_record(
        embedded_record,
        description="determinism_bootstrap_audit",
        expected_path=path,
        verify_current_file=True,
    )
    audit = _load_json(path, description="determinism bootstrap audit")
    if (
        audit.get("schema_version") != 1
        or audit.get("mode") != arm
        or audit.get("target_argv") != target_argv
        or audit.get("process_id") != process_id
        or audit.get("status") != "succeeded"
        or audit.get("exit_code") != 0
        or audit.get("exception") is not None
        or audit.get("postflight_exception") is not None
        or audit.get("device") != "cuda"
        or audit.get("autocast_cuda_float16") is not True
        or audit.get("cuda_memory") != cuda_memory
    ):
        raise StockBenchmarkError("Determinism bootstrap record is inconsistent")
    if (
        audit.get("determinism_lock_unchanged") is not True
        or audit.get("determinism_lock_before")
        != audit.get("determinism_lock_after")
        or not isinstance(audit.get("determinism_lock_before"), dict)
        or audit["determinism_lock_before"].get("sha256")
        != DETERMINISM_LOCK_SHA256
    ):
        raise StockBenchmarkError("Bootstrap did not bind the determinism lock")
    snapshots = audit.get("determinism_snapshots")
    if not isinstance(snapshots, dict):
        raise StockBenchmarkError("Bootstrap lacks deterministic snapshots")
    required = ("after_initial_configuration", "after_inference")
    if any(snapshots.get(stage) != DETERMINISTIC_SNAPSHOT for stage in required):
        raise StockBenchmarkError("Bootstrap deterministic settings are invalid")
    if arm == "stock":
        if (
            audit.get("stock_constructor_reassertion_count") != 1
            or snapshots.get("after_predictor_construction")
            != DETERMINISTIC_SNAPSHOT
        ):
            raise StockBenchmarkError("Stock constructor override was not neutralized")
    elif (
        audit.get("stock_constructor_reassertion_count") != 0
        or snapshots.get("after_predictor_construction") is not None
    ):
        raise StockBenchmarkError("Candidate bootstrap patched a stock constructor")
    before_sources = _validate_source_manifest(audit.get("installed_sources_before"))
    after_sources = _validate_source_manifest(audit.get("installed_sources_after"))
    if audit.get("installed_sources_unchanged") is not True or before_sources != after_sources:
        raise StockBenchmarkError("Inference source changed during the timed child")
    target = audit.get("target")
    if arm == "stock":
        expected_target = {
            "module": "nnunetv2.inference.predict_from_raw_data",
            "entry_point": "predict_entry_point",
            "package": "nnunetv2",
            "package_version": "2.8.1",
        }
        matching = [
            {"path": source_path, **record}
            for source_path, record in before_sources.items()
            if record["sha256"] == STOCK_SOURCE_SHA256
        ]
        if (
            target != expected_target
            or len(matching) != 1
            or set(before_sources) != {matching[0]["path"]}
        ):
            raise StockBenchmarkError("Bootstrap did not use the exact stock source")
        provenance = {**expected_target, "predict_from_raw_data": matching[0]}
    else:
        expected_target = {
            "module": "predict_joint",
            "entry_point": "main",
            "package": "pancreas-multitask",
            "package_version": "0.1.0",
        }
        candidate_source = (ROOT / "scripts" / "predict_joint.py").resolve()
        expected_sources = {
            str(candidate_source): {
                "sha256": _sha256(candidate_source),
                "size_bytes": candidate_source.stat().st_size,
            }
        }
        if target != expected_target or before_sources != expected_sources:
            raise StockBenchmarkError("Bootstrap did not use repository predict_joint.main")
        provenance = expected_target
    return normalized_record, provenance


def _validate_locked_file_pair(
    payload: dict[str, Any],
    *,
    prefix: str,
    description: str,
    expected_sha256: str | None = None,
    expected_path: Path | None = None,
    verify_current_file: bool,
) -> dict[str, Any]:
    before = _validate_file_record(
        payload.get(f"{prefix}_before"),
        description=f"{description} before",
        expected_sha256=expected_sha256,
        expected_path=expected_path,
        verify_current_file=verify_current_file,
    )
    after = _validate_file_record(
        payload.get(f"{prefix}_after"),
        description=f"{description} after",
        expected_sha256=expected_sha256,
        expected_path=expected_path,
        verify_current_file=verify_current_file,
    )
    if payload.get(f"{prefix}_unchanged_during_run") is not True or before != after:
        raise StockBenchmarkError(f"{description} changed during the timed child")
    return before


def _validate_checkpoint(payload: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    raw_before = payload.get("checkpoint_before")
    raw_after = payload.get("checkpoint_after")
    if not isinstance(raw_before, dict) or not isinstance(raw_after, dict):
        raise StockBenchmarkError("External runtime lacks checkpoint provenance")
    before = _validate_file_record(
        raw_before,
        description="fold-0 checkpoint before",
        expected_sha256=CHECKPOINT_SHA256,
        verify_current_file=True,
    )
    after = _validate_file_record(
        raw_after,
        description="fold-0 checkpoint after",
        expected_sha256=CHECKPOINT_SHA256,
        verify_current_file=True,
    )
    if (
        raw_before.get("fold") != "0"
        or raw_before.get("name") != CHECKPOINT_NAME
        or raw_after.get("fold") != "0"
        or raw_after.get("name") != CHECKPOINT_NAME
        or payload.get("checkpoint_unchanged_during_run") is not True
        or before != after
    ):
        raise StockBenchmarkError("Fold-0 checkpoint changed or is not the frozen file")
    checkpoint_path = Path(before["path"])
    if checkpoint_path.name != CHECKPOINT_NAME or checkpoint_path.parent.name != "fold_0":
        raise StockBenchmarkError("Checkpoint path violates the fold-0 model layout")
    return {**before, "fold": "0", "name": CHECKPOINT_NAME}, checkpoint_path.parents[1]


def _validate_model_configuration(
    payload: dict[str, Any], *, model_directory: Path
) -> list[dict[str, Any]]:
    before = payload.get("model_configuration_before")
    after = payload.get("model_configuration_after")
    if not isinstance(before, list) or not isinstance(after, list) or len(before) != 2:
        raise StockBenchmarkError("External runtime lacks dataset/plans provenance")
    normalized: list[dict[str, Any]] = []
    for index, (name, expected_hash) in enumerate(MODEL_CONFIGURATION_SHA256.items()):
        raw = before[index]
        raw_after = after[index] if index < len(after) else None
        if not isinstance(raw, dict) or not isinstance(raw_after, dict):
            raise StockBenchmarkError("Model configuration entry must be an object")
        first = _validate_file_record(
            raw,
            description=f"model configuration {name} before",
            expected_sha256=expected_hash,
            expected_path=model_directory / name,
            verify_current_file=True,
        )
        second = _validate_file_record(
            raw_after,
            description=f"model configuration {name} after",
            expected_sha256=expected_hash,
            expected_path=model_directory / name,
            verify_current_file=True,
        )
        if raw.get("name") != name or raw_after.get("name") != name or first != second:
            raise StockBenchmarkError(f"Model configuration {name} changed during timing")
        normalized.append({**first, "name": name})
    if payload.get("model_configuration_unchanged_during_run") is not True:
        raise StockBenchmarkError("Model configuration was not immutable")
    return normalized


def _validate_input_inventory(
    payload: dict[str, Any], *, expected_case_count: int
) -> tuple[dict[str, Any], list[str]]:
    input_directory = _resolved_path(
        payload.get("input_directory"), field="input_directory"
    )
    before = payload.get("input_manifest_before")
    after = payload.get("input_manifest_after")
    first = _validate_manifest(
        before,
        expected_root=input_directory,
        description="input_manifest_before",
        verify_current_files=True,
    )
    second = _validate_manifest(
        after,
        expected_root=input_directory,
        description="input_manifest_after",
        verify_current_files=True,
    )
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise StockBenchmarkError("Input manifest must be an object")
    case_ids = before.get("case_ids")
    if (
        payload.get("input_unchanged_during_run") is not True
        or first != second
        or case_ids != after.get("case_ids")
        or not isinstance(case_ids, list)
        or len(case_ids) != expected_case_count
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or case_ids != sorted(set(case_ids))
    ):
        raise StockBenchmarkError("Input inventory changed or has the wrong case count")
    expected_names = [f"{case_id}_0000.nii.gz" for case_id in case_ids]
    if [item["relative_path"] for item in first["files"]] != expected_names:
        raise StockBenchmarkError("Input inventory is not one channel per expected case")
    return {**first, "case_ids": case_ids}, list(case_ids)


def _validate_process_log(value: object) -> dict[str, Any]:
    record = _validate_file_record(
        value,
        description="process_log",
        verify_current_file=True,
    )
    try:
        content = Path(record["path"]).read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise StockBenchmarkError(f"Cannot read retained process log: {error}") from error
    if STOCK_CPU_FALLBACK_MARKER in content:
        raise StockBenchmarkError("Process log contains the stock silent CPU fallback")
    return record


def _validate_external_timing(
    payload: dict[str, Any],
    *,
    expected_label: str,
    expected_arm: str,
    expected_case_count: int,
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != 1
        or payload.get("run_label") != expected_label
        or payload.get("arm") != expected_arm
        or payload.get("execution_purpose") != "final_benchmark"
        or payload.get("timing_eligible") is not True
        or payload.get("status") != "succeeded"
        or payload.get("exception") is not None
        or payload.get("fresh_process") is not True
        or payload.get("warmup") != "none"
        or payload.get("timer") != "time.monotonic_ns"
        or payload.get("timing_scope") != EXTERNAL_TIMING_SCOPE
        or payload.get("exit_code") != 0
        or payload.get("timed_out") is not False
        or payload.get("case_count") != expected_case_count
        or payload.get("failed_case_count") != 0
        or payload.get("oom_fallback_count") != 0
        or payload.get("stock_cpu_result_fallback_detected") is not False
        or payload.get("test_targets_or_submission_feedback_used") is not False
    ):
        raise StockBenchmarkError(
            f"External runtime {expected_label} is failed, diagnostic, or incomplete"
        )
    process_id = payload.get("process_id")
    launcher_process_id = payload.get("launcher_process_id")
    start_ns = payload.get("monotonic_start_ns")
    end_ns = payload.get("monotonic_end_ns")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (process_id, launcher_process_id, start_ns, end_ns)
    ) or end_ns <= start_ns:
        raise StockBenchmarkError("External timing requires positive PID and monotonic bounds")
    elapsed = _require_number(payload, "elapsed_seconds")
    exact_elapsed = (end_ns - start_ns) / 1_000_000_000
    if elapsed <= 0 or not math.isclose(
        elapsed, exact_elapsed, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise StockBenchmarkError("elapsed_seconds differs from the monotonic timer")
    started_at = _parse_timestamp(payload.get("started_at_utc"), field="started_at_utc")
    completed_at = _parse_timestamp(
        payload.get("completed_at_utc"), field="completed_at_utc"
    )
    if completed_at <= started_at:
        raise StockBenchmarkError("External wall-clock timestamps are not increasing")
    wall_elapsed = (completed_at - started_at).total_seconds()
    if abs(wall_elapsed - elapsed) > max(2.0, elapsed * 0.01):
        raise StockBenchmarkError("Wall and monotonic complete-process timings disagree")
    execution_id = payload.get("benchmark_execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        raise StockBenchmarkError("External runtime lacks benchmark_execution_id")
    try:
        uuid.UUID(execution_id)
    except ValueError as error:
        raise StockBenchmarkError("benchmark_execution_id must be a UUID") from error
    return {
        "process_id": process_id,
        "launcher_process_id": launcher_process_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "monotonic_start_ns": start_ns,
        "monotonic_end_ns": end_ns,
        "elapsed_seconds": elapsed,
        "benchmark_execution_id": execution_id,
    }


def _validate_ledger_binding(
    value: object, *, expected_record: dict[str, Any], description: str
) -> None:
    if not isinstance(value, dict):
        raise StockBenchmarkError(f"One-use ledger lacks {description}")
    if (
        value.get("path") != expected_record["path"]
        or value.get("sha256") != expected_record["sha256"]
    ):
        raise StockBenchmarkError(f"One-use ledger has the wrong {description}")


def _validate_one_use_ledger(
    record: dict[str, Any],
    *,
    benchmark_execution_id: str,
    final_candidate_lock: dict[str, Any],
    stock_gate_lock: dict[str, Any],
    determinism_lock: dict[str, Any],
    stock_export_lock: dict[str, Any],
    audit_output_path: Path,
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _load_json(Path(record["path"]), description="one-use stock-speed ledger")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "started_and_consumed"
        or payload.get("stage") != "single_locked_stock_inference_speed_benchmark"
        or payload.get("benchmark_execution_id") != benchmark_execution_id
        or payload.get("run_order") != list(EXPECTED_RUN_LABELS)
        or payload.get("test_targets_or_submission_feedback_used") is not False
    ):
        raise StockBenchmarkError("One-use ledger does not describe this final ABBA run")
    _parse_timestamp(payload.get("claimed_at_utc"), field="ledger.claimed_at_utc")
    orchestrator_pid = payload.get("orchestrator_process_id")
    if (
        isinstance(orchestrator_pid, bool)
        or not isinstance(orchestrator_pid, int)
        or orchestrator_pid <= 0
    ):
        raise StockBenchmarkError("One-use ledger lacks a positive orchestrator PID")
    work_root = _resolved_path(payload.get("work_root"), field="ledger.work_root")
    if not work_root.is_dir():
        raise StockBenchmarkError("One-use ledger work_root no longer exists")
    intended_audit = _resolved_path(
        payload.get("intended_audit_path"), field="ledger.intended_audit_path"
    )
    if intended_audit != audit_output_path.expanduser().resolve():
        raise StockBenchmarkError("One-use ledger intended_audit_path is not this audit")
    try:
        intended_audit.relative_to(work_root)
    except ValueError as error:
        raise StockBenchmarkError("Final audit path is outside the one-use work_root") from error
    _validate_ledger_binding(
        payload.get("final_candidate_lock"),
        expected_record=final_candidate_lock,
        description="final-candidate lock",
    )
    _validate_ledger_binding(
        payload.get("stock_gate_lock"),
        expected_record=stock_gate_lock,
        description="stock-gate lock",
    )
    _validate_ledger_binding(
        payload.get("determinism_lock"),
        expected_record=determinism_lock,
        description="determinism lock",
    )
    _validate_ledger_binding(
        payload.get("stock_export_lock"),
        expected_record=stock_export_lock,
        description="stock-export lock",
    )
    for context in contexts:
        label = context["run_label"]
        artifact_paths = [
            Path(context["external_runtime_artifact"]["path"]),
            Path(context["output_directory"]),
            Path(context["determinism_bootstrap_audit"]["path"]),
            Path(context["process_log"]["path"]),
        ]
        internal = context["candidate_internal_runtime_artifact"]
        if internal is not None:
            artifact_paths.append(Path(internal["path"]))
        for artifact_path in artifact_paths:
            try:
                relative = artifact_path.resolve().relative_to(work_root)
            except ValueError as error:
                raise StockBenchmarkError(
                    f"{label} artifact escapes the one-use work_root"
                ) from error
            if not relative.parts or relative.parts[0] != label:
                raise StockBenchmarkError(
                    f"{label} artifact is not under its exact run-label directory"
                )
    return {
        "artifact": record,
        "schema_version": 1,
        "status": "started_and_consumed",
        "stage": payload["stage"],
        "benchmark_execution_id": benchmark_execution_id,
        "claimed_at_utc": payload["claimed_at_utc"],
        "orchestrator_process_id": orchestrator_pid,
        "work_root": str(work_root),
        "intended_audit_path": str(intended_audit),
        "run_order": list(EXPECTED_RUN_LABELS),
    }


def _stock_target_argv(input_directory: Path, output_directory: Path) -> list[str]:
    return [
        "-i",
        str(input_directory),
        "-o",
        str(output_directory),
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


def _candidate_target_argv(
    input_directory: Path,
    output_directory: Path,
    model_directory: Path,
    internal_runtime_path: Path,
    internal_runtime: dict[str, Any],
) -> list[str]:
    bundle = internal_runtime.get("neural_case_head_bundle")
    if not isinstance(bundle, dict):
        raise StockBenchmarkError("Candidate runtime lacks neural bundle provenance")
    bundle_path = _resolved_path(
        bundle.get("bundle_path"), field="neural_case_head_bundle.bundle_path"
    )
    bundle_record = _validate_file_record(
        {
            "path": str(bundle_path),
            "sha256": bundle.get("bundle_sha256"),
            "size_bytes": bundle.get("bundle_size_bytes"),
        },
        description="neural case-head bundle",
        verify_current_file=True,
    )
    numeric_sha256 = bundle.get("numeric_train_dataset_sha256")
    if not _is_sha256(numeric_sha256):
        raise StockBenchmarkError("Candidate bundle lacks numeric-train binding")
    return [
        "--input",
        str(input_directory),
        "--output",
        str(output_directory),
        "--model",
        str(model_directory),
        "--folds",
        "0",
        "--checkpoint",
        CHECKPOINT_NAME,
        "--classification-mode",
        "neural-v5",
        "--v5-extraction-mode",
        "neural_only",
        "--neural-case-head-bundle",
        bundle_record["path"],
        "--expected-neural-case-head-bundle-sha256",
        bundle_record["sha256"],
        "--expected-numeric-train-dataset-sha256",
        numeric_sha256,
        "--runtime-json",
        str(internal_runtime_path),
        "--probability-csv",
        str(output_directory / PROBABILITY_FILENAME),
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


def _validate_command(
    payload: dict[str, Any],
    *,
    environment: dict[str, Any],
    arm: str,
    target_argv: list[str],
    bootstrap_path: Path,
) -> None:
    command = payload.get("command_argv")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise StockBenchmarkError("External runtime command_argv is invalid")
    expected = [
        environment["python_executable"],
        str(DETERMINISTIC_BOOTSTRAP_PATH.resolve()),
        "--mode",
        arm,
        "--determinism-audit-json",
        str(bootstrap_path),
        "--",
        *target_argv,
    ]
    if command != expected or payload.get("target_argv") != target_argv:
        raise StockBenchmarkError("Timed child command differs from the frozen invocation")


def _validate_candidate_internal_runtime(
    path: Path,
    embedded_record: object,
    *,
    timing: dict[str, Any],
    case_ids: list[str],
    input_manifest: dict[str, Any],
    input_directory: Path,
    model_directory: Path,
    environment: dict[str, Any],
    cuda_memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(embedded_record, dict) or embedded_record.get(
        "schema_validated_by_timed_runner"
    ) is not True:
        raise StockBenchmarkError("Timed runner did not validate candidate runtime")
    record = _validate_file_record(
        embedded_record,
        description="candidate_internal_runtime",
        expected_path=path,
        verify_current_file=True,
    )
    runtime = _load_json(path, description="candidate internal runtime")
    try:
        _validate_v3_runtime(
            runtime,
            expected_batch_size=1,
            expected_tta_batch_size=1,
            expected_extraction_mode="neural_only",
        )
    except V3BenchmarkError as error:
        raise StockBenchmarkError(
            f"Candidate internal runtime fails the strict v3 validator: {error}"
        ) from error
    if (
        runtime.get("case_count") != len(case_ids)
        or runtime.get("case_ids") != case_ids
        or runtime.get("process_id") != timing["process_id"]
        or Path(runtime.get("input_directory", "")).resolve() != input_directory
        or Path(runtime.get("model_directory", "")).resolve() != model_directory
    ):
        raise StockBenchmarkError("Candidate internal runtime is not the timed child")
    internal_started = _parse_timestamp(
        runtime.get("started_at_utc"), field="candidate.started_at_utc"
    )
    total_seconds = _require_number(runtime, "total_seconds")
    if (
        internal_started < timing["started_at"]
        or internal_started > timing["completed_at"]
        or total_seconds <= 0
        or total_seconds > timing["elapsed_seconds"]
    ):
        raise StockBenchmarkError("Candidate internal timing is outside external timing")
    expected_internal_files = [
        {
            "name": item["relative_path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in input_manifest["files"]
    ]
    internal_manifest = runtime.get("input_file_manifest")
    if (
        not isinstance(internal_manifest, dict)
        or internal_manifest.get("files") != expected_internal_files
    ):
        raise StockBenchmarkError("Candidate and external input manifests differ")
    if (
        runtime.get("python_version") != environment["python_version"]
        or runtime.get("torch_version") != environment["torch_version"]
        or runtime.get("cuda_runtime_version") != environment["cuda_runtime_version"]
        or runtime.get("cudnn_version") != environment["cudnn_version"]
        or runtime.get("device_name") != environment["cuda_device_name"]
        or runtime.get("device_capability")
        != environment["cuda_device_capability"]
    ):
        raise StockBenchmarkError("Candidate internal and external CUDA environments differ")
    after_memory = cuda_memory["after_target"]
    for internal_key, external_key in (
        ("peak_allocated_mib", "peak_allocated_mib"),
        ("peak_reserved_mib", "peak_reserved_mib"),
    ):
        internal_value = _require_number(runtime, internal_key)
        external_value = _require_number(after_memory, external_key)
        if not math.isclose(internal_value, external_value, rel_tol=0.0, abs_tol=1e-9):
            raise StockBenchmarkError("Candidate CUDA peak memory is not process-consistent")
    return record, runtime


def _validate_external_record(
    payload: dict[str, Any],
    *,
    external_runtime_path: Path,
    output_directory: Path,
    candidate_internal_path: Path | None,
    expected_label: str,
    expected_arm: str,
    expected_case_count: int,
) -> dict[str, Any]:
    timing = _validate_external_timing(
        payload,
        expected_label=expected_label,
        expected_arm=expected_arm,
        expected_case_count=expected_case_count,
    )
    if payload.get("inference_contract") != _expected_inference_contract(expected_arm):
        raise StockBenchmarkError(f"{expected_label} inference contract is not frozen")
    environment = _validate_environment(payload.get("environment"))
    power_thermal = _validate_power_and_thermal(
        payload.get("power_and_thermal_environment"), environment=environment
    )
    cuda_memory = _validate_cuda_memory(payload.get("cuda_memory"), arm=expected_arm)
    input_manifest, case_ids = _validate_input_inventory(
        payload, expected_case_count=expected_case_count
    )
    input_directory = Path(input_manifest["root"])
    checkpoint, model_directory = _validate_checkpoint(payload)
    if (
        model_directory.name != MODEL_DIRECTORY_NAME
        or not model_directory.parent.name.startswith("Dataset501_")
    ):
        raise StockBenchmarkError("External runtime used the wrong Dataset501 model folder")
    model_configuration = _validate_model_configuration(
        payload, model_directory=model_directory
    )
    stock_gate_lock = _validate_locked_file_pair(
        payload,
        prefix="stock_gate_lock",
        description="stock speed-gate lock",
        expected_sha256=STOCK_GATE_LOCK_SHA256,
        expected_path=STOCK_GATE_LOCK_PATH,
        verify_current_file=True,
    )
    determinism_lock = _validate_locked_file_pair(
        payload,
        prefix="determinism_lock",
        description="determinism conformance lock",
        expected_sha256=DETERMINISM_LOCK_SHA256,
        expected_path=DETERMINISM_LOCK_PATH,
        verify_current_file=True,
    )
    stock_export_lock = _validate_locked_file_pair(
        payload,
        prefix="stock_export_lock",
        description="stock-export conformance lock",
        expected_sha256=STOCK_EXPORT_LOCK_SHA256,
        expected_path=STOCK_EXPORT_LOCK_PATH,
        verify_current_file=True,
    )
    bootstrap_source = _validate_locked_file_pair(
        payload,
        prefix="determinism_bootstrap_source",
        description="deterministic bootstrap source",
        expected_path=DETERMINISTIC_BOOTSTRAP_PATH,
        verify_current_file=True,
    )
    final_candidate_lock = _validate_locked_file_pair(
        payload,
        prefix="final_candidate_lock",
        description="final-candidate lock",
        verify_current_file=True,
    )
    one_use_ledger = _validate_locked_file_pair(
        payload,
        prefix="one_use_ledger",
        description="one-use stock-speed ledger",
        verify_current_file=True,
    )
    resolved_output = output_directory.expanduser().resolve()
    recorded_output = _resolved_path(
        payload.get("output_directory"), field=f"{expected_label}.output_directory"
    )
    if recorded_output != resolved_output:
        raise StockBenchmarkError(f"{expected_label} output path does not match CLI input")
    output_manifest = _validate_manifest(
        payload.get("output_manifest"),
        expected_root=resolved_output,
        description=f"{expected_label}.output_manifest",
        verify_current_files=True,
    )
    expected_masks = [f"{case_id}.nii.gz" for case_id in case_ids]
    mask_names = sorted(path.name for path in resolved_output.glob("*.nii.gz"))
    manifest_nifti = sorted(
        item["relative_path"]
        for item in output_manifest["files"]
        if item["relative_path"].endswith(".nii.gz")
    )
    if mask_names != expected_masks or manifest_nifti != expected_masks:
        raise StockBenchmarkError(f"{expected_label} mask inventory is incomplete")
    subtype_path = resolved_output / SUBTYPE_FILENAME
    probability_path = resolved_output / PROBABILITY_FILENAME
    if expected_arm == "candidate":
        if not subtype_path.is_file() or not probability_path.is_file():
            raise StockBenchmarkError("Candidate output lacks required subtype CSV files")
    elif subtype_path.exists() or probability_path.exists():
        raise StockBenchmarkError("Stock output unexpectedly contains candidate subtype files")
    process_log = _validate_process_log(payload.get("process_log"))
    if payload.get("stock_cpu_result_fallback_detected") is not False:
        raise StockBenchmarkError("External runtime reports a stock CPU result fallback")

    internal_record: dict[str, Any] | None = None
    internal_runtime: dict[str, Any] | None = None
    neural_bundle: dict[str, Any] | None = None
    if expected_arm == "candidate":
        if candidate_internal_path is None:
            raise StockBenchmarkError("Candidate run lacks its internal runtime path")
        internal_record, internal_runtime = _validate_candidate_internal_runtime(
            candidate_internal_path.expanduser().resolve(),
            payload.get("candidate_internal_runtime"),
            timing=timing,
            case_ids=case_ids,
            input_manifest=input_manifest,
            input_directory=input_directory,
            model_directory=model_directory,
            environment=environment,
            cuda_memory=cuda_memory,
        )
        target_argv = _candidate_target_argv(
            input_directory,
            resolved_output,
            model_directory,
            candidate_internal_path.expanduser().resolve(),
            internal_runtime,
        )
        internal_bundle = internal_runtime["neural_case_head_bundle"]
        neural_bundle = _validate_locked_file_pair(
            payload,
            prefix="neural_case_head_bundle",
            description="neural case-head bundle",
            expected_sha256=internal_bundle["bundle_sha256"],
            expected_path=Path(internal_bundle["bundle_path"]),
            verify_current_file=True,
        )
        if payload.get("stock_provenance") is not None:
            raise StockBenchmarkError("Candidate external record contains stock provenance")
    else:
        if candidate_internal_path is not None or payload.get(
            "candidate_internal_runtime"
        ) is not None:
            raise StockBenchmarkError("Stock run contains candidate-only runtime evidence")
        if (
            payload.get("neural_case_head_bundle_before") is not None
            or payload.get("neural_case_head_bundle_after") is not None
            or payload.get("neural_case_head_bundle_unchanged_during_run") is not None
        ):
            raise StockBenchmarkError("Stock run contains neural-bundle file evidence")
        target_argv = _stock_target_argv(input_directory, resolved_output)

    bootstrap_raw = payload.get("determinism_bootstrap_audit")
    if not isinstance(bootstrap_raw, dict):
        raise StockBenchmarkError("External runtime lacks determinism bootstrap binding")
    bootstrap_path = _resolved_path(
        bootstrap_raw.get("path"), field="determinism_bootstrap_audit.path"
    )
    bootstrap_record, target_provenance = _validate_bootstrap_audit(
        bootstrap_path,
        bootstrap_raw,
        arm=expected_arm,
        target_argv=target_argv,
        process_id=timing["process_id"],
        cuda_memory=cuda_memory,
    )
    _validate_command(
        payload,
        environment=environment,
        arm=expected_arm,
        target_argv=target_argv,
        bootstrap_path=bootstrap_path,
    )
    if expected_arm == "stock" and payload.get("stock_provenance") != target_provenance:
        raise StockBenchmarkError("Stock provenance differs from bootstrap evidence")

    return {
        "run_label": expected_label,
        "arm": expected_arm,
        "external_runtime_artifact": _file_record(external_runtime_path),
        "timing": timing,
        "case_ids": case_ids,
        "input_manifest": input_manifest,
        "input_directory": str(input_directory),
        "output_directory": str(resolved_output),
        "output_manifest": output_manifest,
        "checkpoint": checkpoint,
        "model_directory": str(model_directory),
        "model_configuration": model_configuration,
        "environment": environment,
        "power_and_thermal_environment": power_thermal,
        "cuda_memory": cuda_memory,
        "stock_gate_lock": stock_gate_lock,
        "determinism_lock": determinism_lock,
        "stock_export_lock": stock_export_lock,
        "determinism_bootstrap_source": bootstrap_source,
        "final_candidate_lock": final_candidate_lock,
        "one_use_ledger": one_use_ledger,
        "determinism_bootstrap_audit": bootstrap_record,
        "target_provenance": target_provenance,
        "process_log": process_log,
        "candidate_internal_runtime_artifact": internal_record,
        "candidate_internal_runtime": internal_runtime,
        "neural_case_head_bundle": neural_bundle,
    }


def _require_shared_contract(
    contexts: list[dict[str, Any]], *, audit_output_path: Path
) -> dict[str, Any]:
    if len(contexts) != len(EXPECTED_RUNS):
        raise StockBenchmarkError("Exactly four ABBA run contexts are required")
    shared_fields = (
        "case_ids",
        "input_manifest",
        "input_directory",
        "checkpoint",
        "model_directory",
        "model_configuration",
        "environment",
        "stock_gate_lock",
        "determinism_lock",
        "stock_export_lock",
        "determinism_bootstrap_source",
        "final_candidate_lock",
        "one_use_ledger",
    )
    reference = contexts[0]
    for field in shared_fields:
        for context in contexts[1:]:
            if context[field] != reference[field]:
                raise StockBenchmarkError(f"ABBA records differ in shared field {field}")
    execution_ids = [context["timing"]["benchmark_execution_id"] for context in contexts]
    if len(set(execution_ids)) != 1:
        raise StockBenchmarkError("ABBA records do not share one benchmark execution ID")
    process_ids = [context["timing"]["process_id"] for context in contexts]
    if len(set(process_ids)) != len(process_ids):
        raise StockBenchmarkError("Every ABBA arm must use a distinct fresh process")
    launcher_process_ids = [
        context["timing"]["launcher_process_id"] for context in contexts
    ]
    if len(set(launcher_process_ids)) != len(launcher_process_ids):
        raise StockBenchmarkError("Every ABBA arm must use a distinct launcher process")
    for previous, following in pairwise(contexts):
        if (
            previous["timing"]["completed_at"]
            > following["timing"]["started_at"]
            or previous["timing"]["monotonic_end_ns"]
            > following["timing"]["monotonic_start_ns"]
        ):
            raise StockBenchmarkError("ABBA timed child processes overlap or are out of order")
    artifact_paths = [
        context["external_runtime_artifact"]["path"] for context in contexts
    ]
    output_paths = [context["output_directory"] for context in contexts]
    bootstrap_paths = [
        context["determinism_bootstrap_audit"]["path"] for context in contexts
    ]
    process_logs = [context["process_log"]["path"] for context in contexts]
    for description, values in (
        ("external runtime", artifact_paths),
        ("output directory", output_paths),
        ("bootstrap audit", bootstrap_paths),
        ("process log", process_logs),
    ):
        if len(set(values)) != len(values):
            raise StockBenchmarkError(f"ABBA reuses a {description} artifact")
    candidate_contexts = [context for context in contexts if context["arm"] == "candidate"]
    candidate_runtimes = [
        context["candidate_internal_runtime"] for context in candidate_contexts
    ]
    if any(not isinstance(runtime, dict) for runtime in candidate_runtimes):
        raise StockBenchmarkError("Candidate ABBA records lack internal runtimes")
    try:
        _validate_v3_matched_runtime(candidate_runtimes)
    except V3BenchmarkError as error:
        raise StockBenchmarkError(
            f"Candidate repeats violate the matched v3 runtime contract: {error}"
        ) from error
    if candidate_contexts[0]["neural_case_head_bundle"] != candidate_contexts[1][
        "neural_case_head_bundle"
    ]:
        raise StockBenchmarkError("Candidate repeats use different neural bundle files")
    ledger = _validate_one_use_ledger(
        reference["one_use_ledger"],
        benchmark_execution_id=execution_ids[0],
        final_candidate_lock=reference["final_candidate_lock"],
        stock_gate_lock=reference["stock_gate_lock"],
        determinism_lock=reference["determinism_lock"],
        stock_export_lock=reference["stock_export_lock"],
        audit_output_path=audit_output_path,
        contexts=contexts,
    )
    claimed_at = _parse_timestamp(ledger["claimed_at_utc"], field="ledger.claimed_at_utc")
    if claimed_at > contexts[0]["timing"]["started_at"]:
        raise StockBenchmarkError("One-use ledger was claimed after timing began")
    return {
        "benchmark_execution_id": execution_ids[0],
        "one_use_ledger": ledger,
    }


def _matrices_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.array_equal(left, right))


def _mask_geometry(image: nib.spatialimages.SpatialImage) -> dict[str, Any]:
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    return {
        "shape": tuple(int(value) for value in image.shape),
        "affine": np.asarray(image.affine),
        "zooms": tuple(float(value) for value in image.header.get_zooms()),
        "qform": qform,
        "qform_code": int(qform_code),
        "sform": sform,
        "sform_code": int(sform_code),
        "xyzt_units": tuple(image.header.get_xyzt_units()),
        "header_dtype": np.dtype(image.header.get_data_dtype()),
    }


def _geometry_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left["shape"] == right["shape"]
        and np.array_equal(left["affine"], right["affine"])
        and left["zooms"] == right["zooms"]
        and _matrices_equal(left["qform"], right["qform"])
        and left["qform_code"] == right["qform_code"]
        and _matrices_equal(left["sform"], right["sform"])
        and left["sform_code"] == right["sform_code"]
        and left["xyzt_units"] == right["xyzt_units"]
        and left["header_dtype"] == right["header_dtype"]
    )


def _compare_mask_directories(
    reference: Path,
    candidate: Path,
    *,
    case_ids: list[str],
    comparison: str,
) -> dict[str, Any]:
    geometry_mismatch_cases: list[str] = []
    dtype_mismatch_cases: list[str] = []
    value_domain_mismatch_cases: list[str] = []
    per_case_disagreements: dict[str, int] = {}
    total_voxels = 0
    disagreeing_voxels = 0
    for case_id in case_ids:
        filename = f"{case_id}.nii.gz"
        reference_image = nib.load(reference / filename)
        candidate_image = nib.load(candidate / filename)
        reference_geometry = _mask_geometry(reference_image)
        candidate_geometry = _mask_geometry(candidate_image)
        if not _geometry_equal(reference_geometry, candidate_geometry):
            geometry_mismatch_cases.append(filename)
        reference_array = np.asanyarray(reference_image.dataobj)
        candidate_array = np.asanyarray(candidate_image.dataobj)
        if reference_array.dtype != candidate_array.dtype:
            dtype_mismatch_cases.append(filename)
        for array in (reference_array, candidate_array):
            if not np.issubdtype(array.dtype, np.integer) or not np.all(
                np.isin(array, (0, 1, 2))
            ):
                value_domain_mismatch_cases.append(filename)
                break
        if reference_array.shape != candidate_array.shape:
            per_case_disagreements[filename] = -1
            continue
        count = int(np.count_nonzero(reference_array != candidate_array))
        per_case_disagreements[filename] = count
        disagreeing_voxels += count
        total_voxels += int(reference_array.size)
    passed = (
        not geometry_mismatch_cases
        and not dtype_mismatch_cases
        and not value_domain_mismatch_cases
        and total_voxels > 0
        and disagreeing_voxels == 0
        and all(value == 0 for value in per_case_disagreements.values())
    )
    return {
        "comparison": comparison,
        "case_count": len(case_ids),
        "geometry_mismatch_cases": sorted(set(geometry_mismatch_cases)),
        "dtype_mismatch_cases": sorted(set(dtype_mismatch_cases)),
        "value_domain_mismatch_cases": sorted(set(value_domain_mismatch_cases)),
        "hard_mask_total_voxels": total_voxels,
        "hard_mask_disagreeing_voxels": disagreeing_voxels,
        "maximum_hard_mask_disagreeing_voxels_in_one_case": max(
            per_case_disagreements.values(), default=0
        ),
        "per_case_hard_mask_disagreeing_voxels": per_case_disagreements,
        "passed": passed,
    }


def _read_subtypes(path: Path, *, expected_names: list[str]) -> dict[str, int]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise StockBenchmarkError(f"Cannot read subtype CSV {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Names", "Subtype"]:
            raise StockBenchmarkError(f"Invalid subtype CSV header: {path}")
        values: dict[str, int] = {}
        for row in reader:
            name = row.get("Names")
            if not isinstance(name, str) or not name or name in values:
                raise StockBenchmarkError(f"Invalid or duplicate subtype name in {path}")
            try:
                subtype = int(row["Subtype"])
            except (KeyError, TypeError, ValueError) as error:
                raise StockBenchmarkError(f"Invalid subtype for {name}: {path}") from error
            if subtype not in (0, 1, 2):
                raise StockBenchmarkError(f"Subtype must be 0, 1, or 2: {path}")
            values[name] = subtype
    if list(values) != expected_names:
        raise StockBenchmarkError(f"Subtype CSV inventory/order is wrong: {path}")
    return values


def _read_probabilities(
    path: Path, *, expected_names: list[str]
) -> dict[str, tuple[int, np.ndarray]]:
    expected_header = [
        "Names",
        "Subtype",
        "Probability_0",
        "Probability_1",
        "Probability_2",
    ]
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise StockBenchmarkError(f"Cannot read probability CSV {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise StockBenchmarkError(f"Invalid probability CSV header: {path}")
        values: dict[str, tuple[int, np.ndarray]] = {}
        for row in reader:
            name = row.get("Names")
            if not isinstance(name, str) or not name or name in values:
                raise StockBenchmarkError(f"Invalid or duplicate probability name: {path}")
            try:
                subtype = int(row["Subtype"])
                probabilities = np.asarray(
                    [float(row[f"Probability_{index}"]) for index in range(3)],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise StockBenchmarkError(f"Invalid probability row for {name}") from error
            if (
                subtype not in (0, 1, 2)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-9, rtol=1e-9)
                or int(np.argmax(probabilities)) != subtype
            ):
                raise StockBenchmarkError(f"Invalid probability semantics for {name}")
            values[name] = (subtype, probabilities)
    if list(values) != expected_names:
        raise StockBenchmarkError(f"Probability CSV inventory/order is wrong: {path}")
    return values


def _compare_candidate_classification(
    first: Path, second: Path, *, case_ids: list[str]
) -> dict[str, Any]:
    expected_names = [f"{case_id}.nii.gz" for case_id in case_ids]
    first_subtypes = _read_subtypes(first / SUBTYPE_FILENAME, expected_names=expected_names)
    second_subtypes = _read_subtypes(
        second / SUBTYPE_FILENAME, expected_names=expected_names
    )
    first_probabilities = _read_probabilities(
        first / PROBABILITY_FILENAME, expected_names=expected_names
    )
    second_probabilities = _read_probabilities(
        second / PROBABILITY_FILENAME, expected_names=expected_names
    )
    internal_disagreements = sum(
        first_probabilities[name][0] != first_subtypes[name]
        or second_probabilities[name][0] != second_subtypes[name]
        for name in expected_names
    )
    repeat_disagreements = sum(
        first_subtypes[name] != second_subtypes[name] for name in expected_names
    )
    maximum_delta = max(
        (
            float(
                np.max(
                    np.abs(
                        first_probabilities[name][1] - second_probabilities[name][1]
                    )
                )
            )
            for name in expected_names
        ),
        default=0.0,
    )
    passed = (
        internal_disagreements == 0
        and repeat_disagreements == 0
        and maximum_delta <= MAXIMUM_PROBABILITY_DELTA
    )
    return {
        "candidate_repeat_subtype_disagreements": repeat_disagreements,
        "csv_internal_subtype_disagreements": internal_disagreements,
        "maximum_absolute_probability_delta": maximum_delta,
        "maximum_allowed_probability_delta": MAXIMUM_PROBABILITY_DELTA,
        "passed": passed,
    }


def _final_lock_speed_bindings(
    final_candidate_lock: dict[str, Any], bootstrap_source: dict[str, Any]
) -> dict[str, Any]:
    payload = _load_json(
        Path(final_candidate_lock["path"]), description="final-candidate lock"
    )
    if payload.get("schema_version") != 1:
        raise StockBenchmarkError("Final-candidate lock has the wrong schema version")
    implementation = payload.get("implementation_files")
    if not isinstance(implementation, list):
        raise StockBenchmarkError("Final-candidate lock lacks implementation_files")
    deviations = payload.get("stock_speed_protocol_deviations")
    if deviations != REQUIRED_STOCK_LOCK_DEVIATIONS:
        raise StockBenchmarkError(
            "Final-candidate lock lacks the exact stock-speed deviation disclosure"
        )
    required_paths = (
        "scripts/run_deterministic_inference.py",
        "scripts/run_timed_inference_child.py",
        "scripts/Run-StockInferenceSpeedBenchmark.ps1",
        "scripts/benchmark_stock_inference_speed.py",
    )
    bindings: list[dict[str, Any]] = []
    for relative in required_paths:
        matches = [
            item
            for item in implementation
            if isinstance(item, dict) and item.get("path") == relative
        ]
        if len(matches) != 1 or not _is_sha256(matches[0].get("sha256")):
            raise StockBenchmarkError(
                f"Final-candidate lock must bind exactly one current {relative}"
            )
        current_path = ROOT / relative
        current_hash = _sha256(current_path)
        if matches[0]["sha256"] != current_hash:
            raise StockBenchmarkError(f"Final-candidate lock has stale speed file {relative}")
        bindings.append({"path": relative, "sha256": current_hash})
    if bindings[0]["sha256"] != bootstrap_source["sha256"]:
        raise StockBenchmarkError("External runs used a different deterministic bootstrap")
    return {
        "all_speed_executables_bound_in_final_candidate_lock": True,
        "implementation_files": bindings,
        "bootstrap_source": bootstrap_source,
        "stock_speed_protocol_deviations": deviations,
    }


def _execution_manifest(
    contexts: list[dict[str, Any]], shared: dict[str, Any]
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for context in contexts:
        timing = context["timing"]
        runs.append(
            {
                "run_label": context["run_label"],
                "arm": context["arm"],
                "external_runtime": context["external_runtime_artifact"],
                "candidate_internal_runtime": context[
                    "candidate_internal_runtime_artifact"
                ],
                "output_directory": context["output_directory"],
                "output_manifest_sha256": context["output_manifest"][
                    "manifest_sha256"
                ],
                "process_id": timing["process_id"],
                "launcher_process_id": timing["launcher_process_id"],
                "started_at_utc": timing["started_at"].isoformat(),
                "completed_at_utc": timing["completed_at"].isoformat(),
                "monotonic_start_ns": timing["monotonic_start_ns"],
                "monotonic_end_ns": timing["monotonic_end_ns"],
                "elapsed_seconds": timing["elapsed_seconds"],
                "determinism_bootstrap_audit": context[
                    "determinism_bootstrap_audit"
                ],
                "process_log": context["process_log"],
            }
        )
    return {
        "schema_version": 1,
        "execution_purpose": "final_benchmark",
        "timing_eligible": True,
        "benchmark_execution_id": shared["benchmark_execution_id"],
        "run_order": list(EXPECTED_RUN_LABELS),
        "fresh_process_per_run": True,
        "warmup": "none",
        "timer": "time.monotonic_ns",
        "timing_scope": EXTERNAL_TIMING_SCOPE,
        "one_use_ledger": shared["one_use_ledger"],
        "runs": runs,
    }


def _power_thermal_summary(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    snapshots = [
        context["power_and_thermal_environment"][stage]
        for context in contexts
        for stage in ("before", "after")
    ]
    return {
        "query_id": NVIDIA_SMI_QUERY,
        "sample_count": len(snapshots),
        "gpu_uuid": snapshots[0]["uuid"],
        "driver_version": snapshots[0]["driver_version"],
        "minimum_power_draw_watts": min(
            float(snapshot["power_draw_watts"]) for snapshot in snapshots
        ),
        "maximum_power_draw_watts": max(
            float(snapshot["power_draw_watts"]) for snapshot in snapshots
        ),
        "minimum_temperature_celsius": min(
            float(snapshot["temperature_celsius"]) for snapshot in snapshots
        ),
        "maximum_temperature_celsius": max(
            float(snapshot["temperature_celsius"]) for snapshot in snapshots
        ),
        "performance_states": [snapshot["performance_state"] for snapshot in snapshots],
    }


def audit_stock_benchmark(
    external_runtime_paths: Sequence[Path],
    output_directories: Sequence[Path],
    candidate_internal_runtime_paths: Sequence[Path],
    *,
    expected_case_count: int = 72,
    audit_output_path: Path,
) -> dict[str, Any]:
    """Validate and score the one final post-repair stock-speed execution."""

    if (
        len(external_runtime_paths) != len(EXPECTED_RUNS)
        or len(output_directories) != len(EXPECTED_RUNS)
        or len(candidate_internal_runtime_paths) != REPEATS_PER_ARM
    ):
        raise StockBenchmarkError(
            "Exactly four external runtimes/outputs and two candidate runtimes are required"
        )
    if expected_case_count < 1:
        raise StockBenchmarkError("expected_case_count must be positive")
    resolved_external = [path.expanduser().resolve() for path in external_runtime_paths]
    resolved_outputs = [path.expanduser().resolve() for path in output_directories]
    resolved_internal = [
        path.expanduser().resolve() for path in candidate_internal_runtime_paths
    ]
    for description, paths in (
        ("external runtime", resolved_external),
        ("output directory", resolved_outputs),
        ("candidate internal runtime", resolved_internal),
    ):
        if len(set(paths)) != len(paths):
            raise StockBenchmarkError(f"Supplied {description} paths must be unique")
    external_payloads = [
        _load_json(path, description="external timing record") for path in resolved_external
    ]
    contexts: list[dict[str, Any]] = []
    candidate_index = 0
    for index, ((label, arm), payload) in enumerate(
        zip(EXPECTED_RUNS, external_payloads, strict=True)
    ):
        internal_path = None
        if arm == "candidate":
            internal_path = resolved_internal[candidate_index]
            candidate_index += 1
        contexts.append(
            _validate_external_record(
                payload,
                external_runtime_path=resolved_external[index],
                output_directory=resolved_outputs[index],
                candidate_internal_path=internal_path,
                expected_label=label,
                expected_arm=arm,
                expected_case_count=expected_case_count,
            )
        )
    shared = _require_shared_contract(
        contexts,
        audit_output_path=audit_output_path.expanduser().resolve(),
    )
    case_ids = contexts[0]["case_ids"]
    mask_comparisons = [
        _compare_mask_directories(
            resolved_outputs[0],
            resolved_outputs[index],
            case_ids=case_ids,
            comparison=f"{EXPECTED_RUNS[0][0]}_vs_{EXPECTED_RUNS[index][0]}",
        )
        for index in range(1, len(EXPECTED_RUNS))
    ]
    candidate_classification = _compare_candidate_classification(
        resolved_outputs[1], resolved_outputs[2], case_ids=case_ids
    )
    numerical_passed = all(item["passed"] for item in mask_comparisons) and bool(
        candidate_classification["passed"]
    )

    stock_seconds = [
        contexts[index]["timing"]["elapsed_seconds"] for index in (0, 3)
    ]
    candidate_seconds = [
        contexts[index]["timing"]["elapsed_seconds"] for index in (1, 2)
    ]
    stock_mean_total = mean(stock_seconds)
    candidate_mean_total = mean(candidate_seconds)
    candidate_fraction = candidate_mean_total / stock_mean_total
    reduction_percent = (1.0 - candidate_fraction) * 100.0
    timing_passed = candidate_fraction <= 0.9
    rejection_reasons: list[str] = []
    if not timing_passed:
        rejection_reasons.append("complete_process_runtime_reduction_below_10_percent")
    if not all(item["passed"] for item in mask_comparisons):
        rejection_reasons.append("exact_stock_candidate_segmentation_gate_failed")
    if not candidate_classification["passed"]:
        rejection_reasons.append("candidate_repeat_classification_gate_failed")
    accepted = timing_passed and numerical_passed
    speed_bindings = _final_lock_speed_bindings(
        contexts[0]["final_candidate_lock"],
        contexts[0]["determinism_bootstrap_source"],
    )
    execution_manifest = _execution_manifest(contexts, shared)
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "expected_case_count": expected_case_count,
        "execution_purpose": "final_benchmark",
        "timing_eligible": True,
        "execution_manifest": execution_manifest,
        "stock_reference": {
            "repeat_complete_process_seconds": stock_seconds,
            "mean_complete_process_seconds": stock_mean_total,
            "mean_complete_process_seconds_per_case": (
                stock_mean_total / expected_case_count
            ),
            "installed_entry_point": contexts[0]["target_provenance"],
            "workload": "stock_segmentation_prediction_and_nifti_export",
            "cuda_memory": [contexts[index]["cuda_memory"] for index in (0, 3)],
        },
        "candidate": {
            "repeat_complete_process_seconds": candidate_seconds,
            "mean_complete_process_seconds": candidate_mean_total,
            "mean_complete_process_seconds_per_case": (
                candidate_mean_total / expected_case_count
            ),
            "fraction_of_stock_reference": candidate_fraction,
            "workload": "segmentation_neural_v5_subtype_nifti_and_csv_export",
            "candidate_workload_is_strictly_broader": True,
            "v5_extraction_mode": "neural_only",
            "cuda_memory": [contexts[index]["cuda_memory"] for index in (1, 2)],
        },
        "runtime_reduction_percent": reduction_percent,
        "minimum_runtime_reduction_percent": MINIMUM_RUNTIME_REDUCTION_PERCENT,
        "timing_passed": timing_passed,
        "numerical_equivalence": {
            "mask_comparisons": mask_comparisons,
            "candidate_repeat_classification": candidate_classification,
            "passed": numerical_passed,
        },
        "shared_environment": contexts[0]["environment"],
        "power_and_thermal_environment": _power_thermal_summary(contexts),
        "post_repair_provenance": {
            "stock_gate_lock": contexts[0]["stock_gate_lock"],
            "determinism_conformance_lock": contexts[0]["determinism_lock"],
            "stock_export_conformance_lock": contexts[0]["stock_export_lock"],
            "final_candidate_lock": contexts[0]["final_candidate_lock"],
            "speed_implementation_bindings": speed_bindings,
            "one_use_ledger": shared["one_use_ledger"],
            "pre_repair_timings_eligible": False,
        },
        "stock_lock_deviations": {
            "final_lock_disclosure": REQUIRED_STOCK_LOCK_DEVIATIONS,
            "train_only_timing_artifacts_after_stock_lock": (
                "Post-lock train-only conformance artifacts exposed raw or "
                "reconstructable timings despite train_only_timing_smoke_allowed=false."
            ),
            "post_lock_implementation_changes": (
                "Deterministic execution and terminal float16 export conformance code "
                "changed after the stock lock despite "
                "candidate_changes_after_this_lock_allowed=false."
            ),
            "mitigation": {
                "affected_inputs_were_train_only": True,
                (
                    "deviating_train_only_conformance_runs_used_official_validation_"
                    "or_test_inputs_or_targets"
                ): False,
                "pre_repair_timings_are_diagnostic_ineligible_and_excluded": True,
                "later_narrow_prospective_locks_bind_repairs": True,
                "model_weights_architecture_features_head_and_offsets_unchanged": True,
                "numerical_gates_tightened_or_preserved_not_relaxed": True,
            },
        },
        "acceptance_rule": (
            "candidate arithmetic mean external complete-process seconds/case <= "
            "0.90 * installed-stock mean; zero failures/OOM/stock CPU fallback; exact "
            "case names, geometry, affine, header dtype, hard-mask values, and candidate "
            "repeat subtype decisions; probability max-absolute delta <= 1e-6"
        ),
        "claim_boundary": {
            "evidence_is_post_determinism_and_stock_export_repair_only": True,
            "repair_type": (
                "post_lock_execution_determinism_and_terminal_float16_export_"
                "conformance_repair"
            ),
            "model_weights_or_architecture_changed": False,
            "neural_head_or_offsets_changed": False,
            "execution_determinism_behavior_changed": True,
            "terminal_export_dtype_behavior_changed": True,
            "numerical_gate_relaxed": False,
            "stock_comparator_claim_allowed": accepted,
            "allowed_claim_if_accepted": (
                "On this recorded single-GPU environment and locked 72-case workload, "
                "the final broader multi-task candidate was at least 10 percent faster "
                "end to end than installed nnunetv2 2.8.1 stock prediction."
            ),
            "globally_fastest_model_claim_allowed": False,
            "hardware_generalization_claim_allowed": False,
            "clinical_runtime_claim_allowed": False,
            "v3_dependency_pruning_claim_must_remain_separate": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-runtime", nargs=4, type=Path, required=True)
    parser.add_argument("--output-directory", nargs=4, type=Path, required=True)
    parser.add_argument(
        "--candidate-internal-runtime", nargs=2, type=Path, required=True
    )
    parser.add_argument("--expected-case-count", type=int, default=72)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = audit_stock_benchmark(
        args.external_runtime,
        args.output_directory,
        args.candidate_internal_runtime,
        expected_case_count=args.expected_case_count,
        audit_output_path=args.output,
    )
    _write_json_atomic(args.output, result)
    print(
        f"Complete-process runtime reduction: {result['runtime_reduction_percent']:.3f}% "
        f"({'ACCEPT' if result['accepted'] else 'REJECT'})"
    )
    print(f"Evidence: {args.output.expanduser().resolve()}")
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
