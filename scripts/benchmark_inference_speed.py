#!/usr/bin/env python3
"""Audit a predeclared paired inference-speed benchmark.

This script does not launch inference. It accepts two fresh-process runtime
artifacts and output directories per arm, verifies that tile batch size is the
only material configuration difference, checks ABBA execution order and the
amended tight numerical-equivalence bounds, and applies the locked >=10%
mean-runtime rule.
"""

from __future__ import annotations

import argparse
import csv
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
MAXIMUM_CLASS_PROBABILITY_DELTA = 1e-4
MAXIMUM_HARD_MASK_DISAGREEMENT_FRACTION = 1e-5
MAXIMUM_HARD_MASK_DISAGREEING_VOXELS_PER_CASE = 16
REPEATS_PER_ARM = 2
EXPECTED_CHRONOLOGICAL_ARMS = ("reference", "candidate", "candidate", "reference")
PROBABILITY_FILENAME = "subtype_probabilities.csv"


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


def _validate_runtime(
    payload: dict[str, Any],
    *,
    expected_batch_size: int,
    expected_tta_batch_size: int,
) -> None:
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
    if payload.get("overwrite") is not True:
        raise BenchmarkError("Every speed run must use --overwrite")
    if payload.get("timing_scope") != (
        "fresh_process_model_initialization_preprocessing_inference_export"
    ):
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
    if _require_number(execution, "logical_tiles_completed") <= 0:
        raise BenchmarkError("Runtime artifact records no completed tiles")
    if _require_number(execution, "tta_views_completed") <= 0:
        raise BenchmarkError("Runtime artifact records no completed TTA views")
    if _require_number(execution, "joint_network_forward_calls") <= 0:
        raise BenchmarkError("Runtime artifact records no network forwards")


MATCHED_RUNTIME_FIELDS = (
    "case_count",
    "case_ids",
    "case_ids_sha256",
    "checkpoint",
    "checkpoint_files",
    "cuda_runtime_version",
    "cudnn_version",
    "device",
    "device_capability",
    "device_name",
    "folds",
    "gaussian_enabled",
    "python_version",
    "tile_step_size",
    "timing_scope",
    "torch_version",
    "tta_enabled",
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
    if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid, _ in process_records):
        raise BenchmarkError("Every runtime must record a positive process_id")
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
            result[name] = int(row["Subtype"])
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
            probabilities = np.asarray(
                [float(row[f"Probability_{index}"]) for index in range(3)],
                dtype=np.float64,
            )
            if not np.isfinite(probabilities).all():
                raise BenchmarkError(f"Non-finite probability row {name!r}: {path}")
            result[name] = (int(row["Subtype"]), probabilities)
    return result


def _validate_output_case_contract(path: Path, case_ids: list[str]) -> None:
    expected_names = {f"{case_id}.nii.gz" for case_id in case_ids}
    mask_names = {item.name for item in path.glob("*.nii.gz")}
    if mask_names != expected_names:
        raise BenchmarkError(f"Output mask set does not match runtime case IDs: {path}")
    if set(_read_subtypes(path / "subtype_results.csv")) != expected_names:
        raise BenchmarkError(f"Subtype case set does not match runtime case IDs: {path}")
    if set(_read_probabilities(path / PROBABILITY_FILENAME)) != expected_names:
        raise BenchmarkError(f"Probability case set does not match runtime case IDs: {path}")


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
        _validate_runtime(payload, expected_batch_size=1, expected_tta_batch_size=1)
    for payload in candidate_payloads:
        _validate_runtime(payload, expected_batch_size=2, expected_tta_batch_size=2)
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
            "zero OOM fallbacks; no geometry or dtype mismatch; global hard-mask "
            "disagreement fraction <= 1e-5 and <= 16 voxels/case; exact subtype "
            "decisions; class probability max-absolute delta <= 1e-4"
        ),
        "candidate": {
            "mean_seconds_per_case": candidate_mean,
            "peak_allocated_mib": [
                payload.get("peak_allocated_mib") for payload in candidate_payloads
            ],
            "repeat_seconds_per_case": candidate_seconds,
            "tile_batch_size": 2,
            "tta_batch_size": 2,
            "network_batch_size_limit": 2,
        },
        "candidate_fraction_of_reference": candidate_fraction,
        "expected_case_count": expected_case_count,
        "minimum_runtime_reduction_percent": MINIMUM_RUNTIME_REDUCTION_PERCENT,
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
        },
        "run_order": run_order,
        "rejection_reasons": rejection_reasons,
        "runtime_reduction_percent": reduction_percent,
        "schema_version": 1,
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
