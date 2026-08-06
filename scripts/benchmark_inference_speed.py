#!/usr/bin/env python3
"""Audit a predeclared paired inference-speed benchmark.

This script does not launch inference. It accepts two fresh-process runtime
artifacts and output directories per arm, verifies that dependency pruning is
the only material configuration difference, checks ABBA execution order and
the exact speed-v3 numerical-equivalence bounds, and applies the locked >=10%
mean-runtime rule. Both arms use tile1/TTA1 scheduling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import nibabel as nib
import numpy as np

MINIMUM_RUNTIME_REDUCTION_PERCENT = 10.0
MAXIMUM_CLASS_PROBABILITY_DELTA = 1e-6
MAXIMUM_HARD_MASK_DISAGREEMENT_FRACTION = 0.0
MAXIMUM_HARD_MASK_DISAGREEING_VOXELS_PER_CASE = 0
REPEATS_PER_ARM = 2
EXPECTED_CHRONOLOGICAL_ARMS = ("reference", "candidate", "candidate", "reference")
PROBABILITY_FILENAME = "subtype_probabilities.csv"
V5_CLASSIFIER_PIPELINE = "assignment_conforming_v5_neural_case_head"
V5_TIMING_SCOPE = (
    "fresh_process_model_and_v5_head_initialization_preprocessing_"
    "feature_extraction_neural_head_offsets_export"
)
V5_MODEL_CONFIGURATION_FILENAMES = ("dataset.json", "plans.json")
V5_CHECKPOINT_NAME = "checkpoint_classification_rescue.pth"
V5_CHECKPOINT_SHA256 = "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116"
V5_MODEL_CONFIGURATION_SHA256 = {
    "dataset.json": "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff",
    "plans.json": "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f",
}
V5_FROZEN_COMPONENT_HASHES = {
    "encoder": "324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1",
    "decoder": "b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2",
    "classification": "1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8",
}
V5_OFFSET_GRID = frozenset((-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0))
V5_IMPLEMENTATION_FILES = {
    "scripts/predict_joint.py",
    "src/pancreas_multitask/classification_rescue.py",
    "src/pancreas_multitask/inference_determinism.py",
    "src/pancreas_multitask/network.py",
    "src/pancreas_multitask/predictor.py",
    "src/pancreas_multitask/case_features.py",
    "src/pancreas_multitask/case_feature_extractor.py",
    "src/pancreas_multitask/neural_case_head.py",
    "src/pancreas_multitask/neural_case_bundle.py",
    "src/pancreas_multitask/neural_case_training.py",
    "src/pancreas_multitask/neural_case_predictor.py",
}

DETERMINISTIC_INFERENCE_POLICY = "strict_cuda_inference_v1"
DETERMINISM_CONFORMANCE_LOCK_SHA256 = (
    "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd"
)
NNUNET_PREDICT_SOURCE_SHA256 = (
    "c350e3202a7a67c3aef12e9206a744add442110ff8a4377c1f9640104b20a31f"
)
STOCK_EXPORT_CONFORMANCE_LOCK_SHA256 = (
    "bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503"
)
DETERMINISTIC_INFERENCE_SNAPSHOT = {
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "cublas_workspace_config": ":4096:8",
    "nnunet_compile": "false",
}


class BenchmarkError(ValueError):
    """Raised when runtime evidence violates the locked comparison contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"Cannot read runtime JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BenchmarkError(f"Runtime JSON must contain an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BenchmarkError(f"{key} must be finite")
    return result


def _execution(payload: dict[str, Any]) -> dict[str, Any]:
    execution = payload.get("inference_execution")
    if not isinstance(execution, dict):
        raise BenchmarkError("inference_execution must be an object")
    return execution


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_deterministic_execution(payload: dict[str, Any]) -> None:
    """Require the disclosed post-lock, symmetric CUDA determinism repair."""

    execution = payload.get("deterministic_execution")
    if not isinstance(execution, dict):
        raise BenchmarkError("Runtime lacks deterministic_execution provenance")
    if execution.get("policy") != DETERMINISTIC_INFERENCE_POLICY:
        raise BenchmarkError("Runtime used an unexpected deterministic inference policy")
    if execution.get("configured_before_cuda_initialization") is not True:
        raise BenchmarkError("Determinism was not configured before CUDA initialization")
    if execution.get("settings_unchanged") is not True:
        raise BenchmarkError("Deterministic inference settings changed during the run")
    if execution.get("autocast_cuda_float16") is not True:
        raise BenchmarkError("Runtime does not disclose CUDA float16 autocast execution")
    conformance_lock = execution.get("conformance_lock")
    if (
        not isinstance(conformance_lock, dict)
        or not isinstance(conformance_lock.get("path"), str)
        or not conformance_lock["path"]
        or conformance_lock.get("sha256") != DETERMINISM_CONFORMANCE_LOCK_SHA256
        or conformance_lock.get("unchanged_during_run") is not True
    ):
        raise BenchmarkError("Runtime does not bind the exact determinism conformance lock")
    installed_source = execution.get("installed_nnunet_source")
    if not isinstance(installed_source, dict):
        raise BenchmarkError("Runtime lacks installed nnUNet prediction-source provenance")
    source_before = installed_source.get("before")
    source_after = installed_source.get("after")
    if (
        installed_source.get("unchanged_during_run") is not True
        or not isinstance(source_before, dict)
        or source_before != source_after
        or not isinstance(source_before.get("path"), str)
        or not source_before["path"]
        or source_before.get("sha256") != NNUNET_PREDICT_SOURCE_SHA256
        or isinstance(source_before.get("size_bytes"), bool)
        or not isinstance(source_before.get("size_bytes"), int)
        or source_before["size_bytes"] < 1
    ):
        raise BenchmarkError("Installed nnUNet prediction source changed or is not locked")
    for stage in (
        "after_initial_configuration",
        "after_predictor_construction",
        "after_inference",
    ):
        if execution.get(stage) != DETERMINISTIC_INFERENCE_SNAPSHOT:
            raise BenchmarkError(
                f"Deterministic inference settings are invalid at {stage}"
            )


def _validate_stock_export_conformance(
    payload: dict[str, Any], *, case_count: int
) -> None:
    conformance = payload.get("stock_export_conformance")
    if not isinstance(conformance, dict):
        raise BenchmarkError("Runtime lacks stock export-dtype conformance")
    if (
        conformance.get("export_logit_dtype") != "torch.float16"
        or conformance.get("case_count_verified") != case_count
        or conformance.get("all_case_exports_verified") is not True
    ):
        raise BenchmarkError("Runtime did not verify stock float16 export logits")
    lock = conformance.get("conformance_lock")
    if (
        not isinstance(lock, dict)
        or not isinstance(lock.get("path"), str)
        or not lock["path"]
        or lock.get("sha256") != STOCK_EXPORT_CONFORMANCE_LOCK_SHA256
        or isinstance(lock.get("size_bytes"), bool)
        or not isinstance(lock.get("size_bytes"), int)
        or lock["size_bytes"] < 1
        or lock.get("unchanged_during_run") is not True
    ):
        raise BenchmarkError("Runtime does not bind the stock export conformance lock")
    execution = _execution(payload)
    if (
        execution.get("segmentation_export_logit_dtype") != "torch.float16"
        or execution.get("segmentation_export_logit_dtype_sequence")
        != ["torch.float16"] * case_count
    ):
        raise BenchmarkError("Not every case reached export with float16 logits")


def _case_ids_sha256(case_ids: list[str]) -> str:
    canonical = "".join(f"{case_id}\n" for case_id in sorted(case_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _input_manifest_sha256(files: list[dict[str, Any]]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_v5_bundle_provenance(payload: dict[str, Any]) -> None:
    if payload.get("classifier_pipeline") != V5_CLASSIFIER_PIPELINE:
        raise BenchmarkError("Timed run did not execute the final v5 classifier pipeline")
    if payload.get("feature_cache_policy") != "disabled_online_fresh_extraction":
        raise BenchmarkError("Timed v5 inference may not read cached case features")
    if payload.get("class_probabilities") != "v5_offset_adjusted_three_class":
        raise BenchmarkError("Timed run did not export v5 offset-adjusted probabilities")
    if payload.get("case_identifiers_or_paths_used_as_model_inputs") is not False:
        raise BenchmarkError("Case identifiers or paths entered the v5 model input")

    bundle = payload.get("neural_case_head_bundle")
    if not isinstance(bundle, dict):
        raise BenchmarkError("Runtime lacks neural case-head bundle provenance")
    required_hashes = (
        "bundle_sha256",
        "numeric_train_dataset_sha256",
        "head_state_sha256",
        "neural_lock_sha256",
        "decision_lock_sha256",
        "selection_audit_sha256",
        "calibration_audit_sha256",
        "refit_audit_sha256",
    )
    if any(not _is_sha256(bundle.get(key)) for key in required_hashes):
        raise BenchmarkError("Neural bundle provenance contains an invalid SHA-256")
    if (
        bundle.get("classifier_pipeline") != V5_CLASSIFIER_PIPELINE
        or bundle.get("expected_bundle_sha256_verified") is not True
        or bundle.get("bundle_loaded_strictly") is not True
        or bundle.get("eligible_for_official") is not True
        or bundle.get("head_in_eval_mode") is not True
        or bundle.get("any_head_parameter_requires_grad") is not False
        or bundle.get("head_state_unchanged") is not True
        or not isinstance(bundle.get("bundle_path"), str)
        or not bundle.get("bundle_path")
        or not isinstance(bundle.get("bundle_name"), str)
        or not bundle.get("bundle_name")
        or isinstance(bundle.get("bundle_size_bytes"), bool)
        or not isinstance(bundle.get("bundle_size_bytes"), int)
        or bundle["bundle_size_bytes"] < 1
        or isinstance(bundle.get("head_parameter_count"), bool)
        or not isinstance(bundle.get("head_parameter_count"), int)
        or bundle["head_parameter_count"] < 1
        or not isinstance(bundle.get("selected_candidate_id"), str)
        or not bundle.get("selected_candidate_id")
    ):
        raise BenchmarkError("Neural bundle provenance is incomplete")
    if not (
        bundle.get("head_state_sha256_before") == bundle.get("head_state_sha256")
        and bundle.get("head_state_sha256_after") == bundle.get("head_state_sha256")
    ):
        raise BenchmarkError(
            "neural_case_head_bundle live state differs from its locked state"
        )
    offsets = bundle.get("class_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in offsets
        )
        or float(offsets[1]) != 0.0
        or float(offsets[0]) not in V5_OFFSET_GRID
        or float(offsets[2]) not in V5_OFFSET_GRID
    ):
        raise BenchmarkError("Neural bundle class offsets violate the locked grid")

    implementation_files = payload.get("v5_implementation_files")
    if (
        not isinstance(implementation_files, dict)
        or set(implementation_files) != V5_IMPLEMENTATION_FILES
        or any(
            not isinstance(name, str) or not name or not _is_sha256(digest)
            for name, digest in implementation_files.items()
        )
    ):
        raise BenchmarkError("Runtime lacks complete v5 implementation hashes")

    model_configuration = payload.get("model_configuration_files")
    if (
        not isinstance(model_configuration, list)
        or [item.get("name") for item in model_configuration if isinstance(item, dict)]
        != list(V5_MODEL_CONFIGURATION_FILENAMES)
        or any(
            not isinstance(item, dict)
            or not _is_sha256(item.get("sha256"))
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 1
            for item in model_configuration
        )
        or any(
            item["sha256"] != V5_MODEL_CONFIGURATION_SHA256[item["name"]]
            for item in model_configuration
            if isinstance(item, dict) and item.get("name") in V5_MODEL_CONFIGURATION_SHA256
        )
        or payload.get("model_configuration_unchanged_during_run") is not True
    ):
        raise BenchmarkError("Runtime lacks exact nnU-Net model-configuration hashes")

    input_manifest = payload.get("input_file_manifest")
    if not isinstance(input_manifest, dict):
        raise BenchmarkError("Runtime lacks a raw-input content manifest")
    input_files = input_manifest.get("files")
    if (
        payload.get("input_files_unchanged_during_run") is not True
        or not isinstance(input_files, list)
        or len(input_files) < 1
        or input_manifest.get("file_count") != len(input_files)
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"].endswith(".nii.gz")
            or not _is_sha256(item.get("sha256"))
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 1
            for item in input_files
        )
        or [item["name"] for item in input_files]
        != sorted({item["name"] for item in input_files})
        or input_manifest.get("manifest_sha256") != _input_manifest_sha256(input_files)
    ):
        raise BenchmarkError("Raw-input content manifest is invalid")

    frozen_network = payload.get("frozen_network")
    if not isinstance(frozen_network, dict):
        raise BenchmarkError("Runtime lacks frozen-network provenance")
    before = frozen_network.get("component_hashes_before")
    after = frozen_network.get("component_hashes_after")
    if (
        frozen_network.get("frozen_components_unchanged") is not True
        or frozen_network.get("network_in_eval_mode") is not True
        or frozen_network.get("any_network_parameter_requires_grad") is not False
        or not isinstance(before, dict)
        or before != V5_FROZEN_COMPONENT_HASHES
        or after != V5_FROZEN_COMPONENT_HASHES
    ):
        raise BenchmarkError("frozen_network nnU-Net component provenance is invalid")


def _validate_runtime(
    payload: dict[str, Any],
    *,
    expected_batch_size: int,
    expected_tta_batch_size: int,
    expected_extraction_mode: str,
) -> None:
    _validate_deterministic_execution(payload)
    case_count = payload.get("case_count")
    case_ids = payload.get("case_ids")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 1:
        raise BenchmarkError("case_count must be a positive integer")
    if (
        not isinstance(case_ids, list)
        or len(case_ids) != case_count
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or case_ids != sorted(set(case_ids))
    ):
        raise BenchmarkError("case_ids must be a sorted unique list matching case_count")
    if payload.get("case_ids_sha256") != _case_ids_sha256(case_ids):
        raise BenchmarkError("case_ids_sha256 does not match the ordered case IDs")
    _validate_stock_export_conformance(payload, case_count=case_count)
    input_manifest = payload.get("input_file_manifest")
    expected_input_names = [f"{case_id}_0000.nii.gz" for case_id in case_ids]
    if (
        not isinstance(input_manifest, dict)
        or input_manifest.get("file_count") != case_count
        or not isinstance(input_manifest.get("files"), list)
        or [item.get("name") for item in input_manifest["files"] if isinstance(item, dict)]
        != expected_input_names
    ):
        raise BenchmarkError("Raw-input manifest does not map one channel to every case")
    if payload.get("folds") != [0]:
        raise BenchmarkError("V5 speed inference requires exactly fold 0")
    if payload.get("checkpoint") != V5_CHECKPOINT_NAME:
        raise BenchmarkError("V5 speed inference used the wrong checkpoint name")
    checkpoint_files = payload.get("checkpoint_files")
    if (
        not isinstance(checkpoint_files, list)
        or len(checkpoint_files) != 1
        or not isinstance(checkpoint_files[0], dict)
        or checkpoint_files[0].get("fold") != "0"
        or checkpoint_files[0].get("sha256") != V5_CHECKPOINT_SHA256
        or isinstance(checkpoint_files[0].get("size_bytes"), bool)
        or not isinstance(checkpoint_files[0].get("size_bytes"), int)
        or checkpoint_files[0]["size_bytes"] < 1
    ):
        raise BenchmarkError("Runtime lacks exactly one valid fold-0 checkpoint")
    if payload.get("checkpoint_unchanged_during_run") is not True:
        raise BenchmarkError("Runtime does not prove checkpoint immutability")
    for key in ("input_directory", "model_directory"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise BenchmarkError(f"Runtime lacks a resolved {key}")
    if payload.get("overwrite") is not True:
        raise BenchmarkError("Every speed run must use --overwrite")
    if payload.get("timing_scope") != V5_TIMING_SCOPE:
        raise BenchmarkError("Unexpected timing_scope")
    if payload.get("warmup_policy") != "none_fresh_process_end_to_end":
        raise BenchmarkError("Unexpected warmup_policy")

    total_seconds = _require_number(payload, "total_seconds")
    mean_seconds_per_case = _require_number(payload, "mean_seconds_per_case")
    if total_seconds <= 0 or mean_seconds_per_case <= 0:
        raise BenchmarkError("Runtime values must be positive")
    if not math.isclose(
        mean_seconds_per_case,
        total_seconds / case_count,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise BenchmarkError("mean_seconds_per_case is inconsistent with total_seconds")

    execution = _execution(payload)
    if payload.get("v5_extraction_mode") != expected_extraction_mode:
        raise BenchmarkError("Unexpected top-level v5 extraction mode")
    if execution.get("v5_extraction_mode") != expected_extraction_mode:
        raise BenchmarkError("Execution counters contain the wrong v5 extraction mode")
    expected_network_batch_size = max(expected_batch_size, expected_tta_batch_size)
    if execution.get("network_batch_size_limit") != expected_network_batch_size:
        raise BenchmarkError("Unexpected shared network microbatch limit")
    if execution.get("maximum_network_batch_size_observed") != expected_network_batch_size:
        raise BenchmarkError("Timed run did not exercise its declared network microbatch size")
    if execution.get("tile_batch_size_requested") != expected_batch_size:
        raise BenchmarkError(
            "Unexpected tile batch size: "
            f"expected {expected_batch_size}, got "
            f"{execution.get('tile_batch_size_requested')!r}"
        )
    if execution.get("tile_batch_size_adaptive_limit") != expected_batch_size:
        raise BenchmarkError("Adaptive tile-batch limit changed during a timed run")
    if execution.get("tile_batch_oom_fallback_count") != 0:
        raise BenchmarkError("An OOM fallback disqualifies the speed result")
    if execution.get("tta_batch_size_requested") != expected_tta_batch_size:
        raise BenchmarkError(
            "Unexpected TTA batch size: "
            f"expected {expected_tta_batch_size}, got "
            f"{execution.get('tta_batch_size_requested')!r}"
        )
    if execution.get("tta_batch_size_adaptive_limit") != expected_tta_batch_size:
        raise BenchmarkError("Adaptive TTA-batch limit changed during a timed run")
    if execution.get("tta_batch_oom_fallback_count") != 0:
        raise BenchmarkError("An OOM fallback disqualifies the speed result")
    if execution.get("speed_v3_network_batch_ceiling") != 1:
        raise BenchmarkError("Unexpected speed-v3 network batch ceiling")
    if execution.get("classifier_pipeline") != V5_CLASSIFIER_PIPELINE:
        raise BenchmarkError("Execution counters do not identify the v5 classifier")
    if execution.get("v5_feature_extraction_executed") is not True:
        raise BenchmarkError("V5 feature extraction was not executed")
    for key in (
        "v5_case_extractions_completed",
        "v5_neural_head_forward_calls",
        "v5_class_offset_applications",
    ):
        if execution.get(key) != case_count:
            raise BenchmarkError(f"{key} must equal the complete case count")
    if execution.get("v5_feature_cache_reads") != 0:
        raise BenchmarkError("A cached v5 feature bag disqualifies the speed result")
    if execution.get("case_identifiers_or_paths_used_as_model_inputs") is not False:
        raise BenchmarkError("Runtime counters report identifier-bearing model input")
    bag_hashes = payload.get("v5_neural_bag_sha256_sequence")
    if (
        not isinstance(bag_hashes, list)
        or len(bag_hashes) != case_count
        or any(not _is_sha256(value) for value in bag_hashes)
        or execution.get("v5_neural_bag_sha256_sequence") != bag_hashes
    ):
        raise BenchmarkError("Runtime lacks one exact neural-bag hash per case")
    if _require_number(execution, "logical_tiles_completed") <= 0:
        raise BenchmarkError("Runtime artifact records no completed tiles")
    if _require_number(execution, "tta_views_completed") <= 0:
        raise BenchmarkError("Runtime artifact records no completed TTA views")
    if _require_number(execution, "joint_network_forward_calls") <= 0:
        raise BenchmarkError("Runtime artifact records no network forwards")
    _validate_v5_bundle_provenance(payload)


MATCHED_RUNTIME_FIELDS = (
    "case_count",
    "case_ids",
    "case_ids_sha256",
    "checkpoint",
    "checkpoint_files",
    "checkpoint_unchanged_during_run",
    "classifier_pipeline",
    "class_probabilities",
    "cuda_runtime_version",
    "cudnn_version",
    "device",
    "device_capability",
    "device_name",
    "deterministic_execution",
    "folds",
    "feature_cache_policy",
    "frozen_network",
    "gaussian_enabled",
    "input_directory",
    "input_file_manifest",
    "input_files_unchanged_during_run",
    "model_configuration_files",
    "model_configuration_unchanged_during_run",
    "model_directory",
    "python_version",
    "neural_case_head_bundle",
    "stock_export_conformance",
    "tile_step_size",
    "timing_scope",
    "torch_version",
    "tta_enabled",
    "case_identifiers_or_paths_used_as_model_inputs",
    "v5_implementation_files",
    "v5_neural_bag_sha256_sequence",
    "warmup_policy",
)


def _validate_matched_runtime_contract(payloads: list[dict[str, Any]]) -> None:
    reference = payloads[0]
    for field in MATCHED_RUNTIME_FIELDS:
        expected = reference.get(field)
        for index, payload in enumerate(payloads[1:], start=1):
            if payload.get(field) != expected:
                raise BenchmarkError(
                    f"Runtime field {field!r} differs in artifact index {index}"
                )
    if reference.get("tta_enabled") is not True:
        raise BenchmarkError("TTA must remain enabled")
    if reference.get("gaussian_enabled") is not True:
        raise BenchmarkError("Gaussian weighting must remain enabled")
    if reference.get("tile_step_size") != 0.5:
        raise BenchmarkError("tile_step_size must remain exactly 0.5")


def _parse_started_at(payload: dict[str, Any]) -> datetime:
    value = payload.get("started_at_utc")
    if not isinstance(value, str):
        raise BenchmarkError("started_at_utc must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise BenchmarkError(f"Invalid started_at_utc: {value!r}") from error
    if parsed.tzinfo is None:
        raise BenchmarkError("started_at_utc must include a timezone")
    return parsed


def _validate_abba_order(
    reference_payloads: list[dict[str, Any]], candidate_payloads: list[dict[str, Any]]
) -> list[str]:
    tagged = [
        *[("reference", _parse_started_at(payload), payload) for payload in reference_payloads],
        *[("candidate", _parse_started_at(payload), payload) for payload in candidate_payloads],
    ]
    tagged.sort(key=lambda item: item[1])
    order = [item[0] for item in tagged]
    if tuple(order) != EXPECTED_CHRONOLOGICAL_ARMS:
        raise BenchmarkError(
            f"Fresh-process run order must be ABBA {EXPECTED_CHRONOLOGICAL_ARMS}; got {order}"
        )
    timestamps = [item[1] for item in tagged]
    if len(timestamps) != len(set(timestamps)):
        raise BenchmarkError("Every fresh-process run must have a unique start time")
    process_records = [
        (item[2].get("process_id"), item[2].get("started_at_utc")) for item in tagged
    ]
    if any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        for pid, _ in process_records
    ):
        raise BenchmarkError("Every runtime must record a positive process_id")
    process_ids = [pid for pid, _ in process_records]
    if len(process_ids) != len(set(process_ids)):
        raise BenchmarkError("Every benchmark run must use a distinct fresh process")
    return order


def _read_subtypes(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Names", "Subtype"]:
            raise BenchmarkError(f"Invalid subtype CSV header: {path}")
        result: dict[str, int] = {}
        for row in reader:
            name = row["Names"]
            if name in result:
                raise BenchmarkError(f"Duplicate subtype row {name!r}: {path}")
            try:
                subtype = int(row["Subtype"])
            except (TypeError, ValueError) as error:
                raise BenchmarkError(f"Invalid subtype row {name!r}: {path}") from error
            if subtype not in (0, 1, 2):
                raise BenchmarkError(f"Subtype must be 0, 1, or 2 for {name!r}: {path}")
            result[name] = subtype
    return result


def _read_probabilities(path: Path) -> dict[str, tuple[int, np.ndarray]]:
    expected = ["Names", "Subtype", "Probability_0", "Probability_1", "Probability_2"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise BenchmarkError(f"Invalid probability CSV header: {path}")
        result: dict[str, tuple[int, np.ndarray]] = {}
        for row in reader:
            name = row["Names"]
            if name in result:
                raise BenchmarkError(f"Duplicate probability row {name!r}: {path}")
            try:
                subtype = int(row["Subtype"])
                probabilities = np.asarray(
                    [float(row[f"Probability_{index}"]) for index in range(3)],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as error:
                raise BenchmarkError(f"Invalid probability row {name!r}: {path}") from error
            if (
                subtype not in (0, 1, 2)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-9, rtol=1e-9)
                or int(np.argmax(probabilities)) != subtype
            ):
                raise BenchmarkError(f"Invalid probability semantics for {name!r}: {path}")
            result[name] = (subtype, probabilities)
    return result


def _validate_output_case_contract(path: Path, case_ids: list[str]) -> None:
    expected_names = {f"{case_id}.nii.gz" for case_id in case_ids}
    mask_names = {item.name for item in path.glob("*.nii.gz")}
    if mask_names != expected_names:
        raise BenchmarkError(f"Output mask set does not match runtime case IDs: {path}")
    subtypes = _read_subtypes(path / "subtype_results.csv")
    if set(subtypes) != expected_names:
        raise BenchmarkError(f"Subtype case set does not match runtime case IDs: {path}")
    probabilities = _read_probabilities(path / PROBABILITY_FILENAME)
    if set(probabilities) != expected_names:
        raise BenchmarkError(f"Probability case set does not match runtime case IDs: {path}")
    if any(probabilities[name][0] != subtype for name, subtype in subtypes.items()):
        raise BenchmarkError(f"Subtype and probability CSV decisions differ: {path}")


def _compare_outputs(
    reference: Path,
    candidate: Path,
    *,
    comparison_name: str,
) -> dict[str, Any]:
    reference_masks = sorted(reference.glob("*.nii.gz"))
    candidate_masks = sorted(candidate.glob("*.nii.gz"))
    reference_names = [path.name for path in reference_masks]
    candidate_names = [path.name for path in candidate_masks]
    if not reference_masks or candidate_names != reference_names:
        raise BenchmarkError("Mask filename sets differ between benchmark outputs")

    total_voxels = 0
    disagreeing_voxels = 0
    per_case_disagreeing_voxels: dict[str, int] = {}
    geometry_mismatch_cases: list[str] = []
    dtype_mismatch_cases: list[str] = []
    for reference_path, candidate_path in zip(reference_masks, candidate_masks, strict=True):
        reference_image = nib.load(reference_path)
        candidate_image = nib.load(candidate_path)
        reference_array = np.asanyarray(reference_image.dataobj)
        candidate_array = np.asanyarray(candidate_image.dataobj)
        if reference_array.shape != candidate_array.shape:
            geometry_mismatch_cases.append(reference_path.name)
            continue
        reference_zooms = reference_image.header.get_zooms()[: reference_array.ndim]
        candidate_zooms = candidate_image.header.get_zooms()[: candidate_array.ndim]
        if not np.array_equal(reference_image.affine, candidate_image.affine) or (
            reference_zooms != candidate_zooms
        ):
            geometry_mismatch_cases.append(reference_path.name)
        if reference_array.dtype != candidate_array.dtype:
            dtype_mismatch_cases.append(reference_path.name)
        total_voxels += int(reference_array.size)
        case_disagreements = int(np.count_nonzero(reference_array != candidate_array))
        per_case_disagreeing_voxels[reference_path.name] = case_disagreements
        disagreeing_voxels += case_disagreements
    if total_voxels <= 0:
        raise BenchmarkError("No shape-compatible mask voxels were available to compare")
    disagreement_fraction = disagreeing_voxels / total_voxels
    maximum_case_disagreements = max(per_case_disagreeing_voxels.values(), default=0)

    reference_subtypes = _read_subtypes(reference / "subtype_results.csv")
    candidate_subtypes = _read_subtypes(candidate / "subtype_results.csv")
    subtype_disagreements = sum(
        reference_subtypes.get(name) != candidate_subtypes.get(name)
        for name in set(reference_subtypes) | set(candidate_subtypes)
    )

    reference_probabilities = _read_probabilities(reference / PROBABILITY_FILENAME)
    candidate_probabilities = _read_probabilities(candidate / PROBABILITY_FILENAME)
    if reference_probabilities.keys() != candidate_probabilities.keys():
        raise BenchmarkError("Probability case sets differ")
    maximum_probability_delta = 0.0
    probability_subtype_disagreements = 0
    for name, (reference_subtype, reference_values) in reference_probabilities.items():
        candidate_subtype, candidate_values = candidate_probabilities[name]
        if candidate_subtype != reference_subtype:
            probability_subtype_disagreements += 1
        maximum_probability_delta = max(
            maximum_probability_delta,
            float(np.max(np.abs(reference_values - candidate_values))),
        )
    passed = (
        not geometry_mismatch_cases
        and not dtype_mismatch_cases
        and disagreement_fraction <= MAXIMUM_HARD_MASK_DISAGREEMENT_FRACTION
        and maximum_case_disagreements
        <= MAXIMUM_HARD_MASK_DISAGREEING_VOXELS_PER_CASE
        and subtype_disagreements == 0
        and probability_subtype_disagreements == 0
        and maximum_probability_delta <= MAXIMUM_CLASS_PROBABILITY_DELTA
    )

    return {
        "case_count": len(reference_masks),
        "comparison": comparison_name,
        "dtype_mismatch_cases": dtype_mismatch_cases,
        "geometry_mismatch_cases": geometry_mismatch_cases,
        "hard_mask_disagreeing_voxels": disagreeing_voxels,
        "hard_mask_disagreement_fraction": disagreement_fraction,
        "hard_mask_total_voxels": total_voxels,
        "maximum_hard_mask_disagreeing_voxels_in_one_case": maximum_case_disagreements,
        "maximum_absolute_class_probability_delta": maximum_probability_delta,
        "passed": passed,
        "per_case_hard_mask_disagreeing_voxels": per_case_disagreeing_voxels,
        "probability_csv_subtype_disagreements": probability_subtype_disagreements,
        "subtype_decision_disagreements": subtype_disagreements,
    }


def audit_benchmark(
    reference_runtime_paths: list[Path],
    candidate_runtime_paths: list[Path],
    reference_output_paths: list[Path],
    candidate_output_paths: list[Path],
    *,
    expected_case_count: int = 72,
) -> dict[str, Any]:
    lengths = {
        len(reference_runtime_paths),
        len(candidate_runtime_paths),
        len(reference_output_paths),
        len(candidate_output_paths),
    }
    if lengths != {REPEATS_PER_ARM}:
        raise BenchmarkError(
            f"Exactly {REPEATS_PER_ARM} runtimes and outputs per arm are required"
        )

    reference_payloads = [_load_json(path) for path in reference_runtime_paths]
    candidate_payloads = [_load_json(path) for path in candidate_runtime_paths]
    for payload in reference_payloads:
        _validate_runtime(
            payload,
            expected_batch_size=1,
            expected_tta_batch_size=1,
            expected_extraction_mode="full",
        )
    for payload in candidate_payloads:
        _validate_runtime(
            payload,
            expected_batch_size=1,
            expected_tta_batch_size=1,
            expected_extraction_mode="neural_only",
        )
    _validate_matched_runtime_contract([*reference_payloads, *candidate_payloads])
    if reference_payloads[0]["case_count"] != expected_case_count:
        raise BenchmarkError(
            f"Expected exactly {expected_case_count} cases, got "
            f"{reference_payloads[0]['case_count']}"
        )
    run_order = _validate_abba_order(reference_payloads, candidate_payloads)

    reference_seconds = [
        _require_number(payload, "mean_seconds_per_case") for payload in reference_payloads
    ]
    candidate_seconds = [
        _require_number(payload, "mean_seconds_per_case") for payload in candidate_payloads
    ]
    reference_mean = mean(reference_seconds)
    candidate_mean = mean(candidate_seconds)
    candidate_fraction = candidate_mean / reference_mean
    reduction_percent = (1.0 - candidate_fraction) * 100.0

    output_comparisons: list[dict[str, Any]] = []
    canonical_output = reference_output_paths[0]
    for output_path in [*reference_output_paths, *candidate_output_paths]:
        _validate_output_case_contract(output_path, reference_payloads[0]["case_ids"])
    compared_outputs = [
        ("reference_1_vs_reference_2", reference_output_paths[1]),
        ("reference_1_vs_candidate_1", candidate_output_paths[0]),
        ("reference_1_vs_candidate_2", candidate_output_paths[1]),
    ]
    for comparison_name, output_path in compared_outputs:
        output_comparisons.append(
            _compare_outputs(
                canonical_output,
                output_path,
                comparison_name=comparison_name,
            )
        )
    expected_case_count = reference_payloads[0]["case_count"]
    if any(item["case_count"] != expected_case_count for item in output_comparisons):
        raise BenchmarkError("Output mask count does not match runtime case_count")

    timing_passed = candidate_fraction <= 0.9
    numerical_equivalence_passed = all(
        item["passed"] for item in output_comparisons
    )
    rejection_reasons = []
    if not timing_passed:
        rejection_reasons.append("runtime_reduction_below_10_percent")
    if not numerical_equivalence_passed:
        rejection_reasons.append("numerical_equivalence_gate_failed")
    accepted = timing_passed and numerical_equivalence_passed
    return {
        "accepted": accepted,
        "acceptance_rule": (
            "candidate arithmetic mean end-to-end seconds/case <= 0.90 * reference; "
            "zero OOM fallbacks; exact geometry, dtype, hard masks, neural bag "
            "arrays, and subtype decisions; offset-adjusted class probability "
            "max-absolute delta <= 1e-6"
        ),
        "candidate": {
            "mean_seconds_per_case": candidate_mean,
            "peak_allocated_mib": [
                payload.get("peak_allocated_mib") for payload in candidate_payloads
            ],
            "repeat_seconds_per_case": candidate_seconds,
            "tile_batch_size": 1,
            "tta_batch_size": 1,
            "network_batch_size_limit": 1,
            "v5_extraction_mode": "neural_only",
        },
        "candidate_fraction_of_reference": candidate_fraction,
        "expected_case_count": expected_case_count,
        "minimum_runtime_reduction_percent": MINIMUM_RUNTIME_REDUCTION_PERCENT,
        "classifier_pipeline": V5_CLASSIFIER_PIPELINE,
        "neural_case_head_bundle": reference_payloads[0][
            "neural_case_head_bundle"
        ],
        "frozen_network": reference_payloads[0]["frozen_network"],
        "numerical_equivalence": output_comparisons,
        "numerical_equivalence_passed": numerical_equivalence_passed,
        "reference": {
            "mean_seconds_per_case": reference_mean,
            "peak_allocated_mib": [
                payload.get("peak_allocated_mib") for payload in reference_payloads
            ],
            "repeat_seconds_per_case": reference_seconds,
            "tile_batch_size": 1,
            "tta_batch_size": 1,
            "network_batch_size_limit": 1,
            "v5_extraction_mode": "full",
        },
        "run_order": run_order,
        "rejection_reasons": rejection_reasons,
        "runtime_reduction_percent": reduction_percent,
        "schema_version": 3,
        "timing_passed": timing_passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-runtime", nargs=2, type=Path, required=True)
    parser.add_argument("--candidate-runtime", nargs=2, type=Path, required=True)
    parser.add_argument("--reference-output", nargs=2, type=Path, required=True)
    parser.add_argument("--candidate-output", nargs=2, type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int, default=72)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = audit_benchmark(
        args.reference_runtime,
        args.candidate_runtime,
        args.reference_output,
        args.candidate_output,
        expected_case_count=args.expected_case_count,
    )
    _write_json_atomic(args.output, result)
    print(
        f"Runtime reduction: {result['runtime_reduction_percent']:.3f}% "
        f"({'ACCEPT' if result['accepted'] else 'REJECT'})"
    )
    print(f"Evidence: {args.output.resolve()}")
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
