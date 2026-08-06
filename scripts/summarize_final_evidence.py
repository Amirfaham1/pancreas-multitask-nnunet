#!/usr/bin/env python3
"""Validate and summarize the frozen final-evaluation evidence.

Example::

    python scripts/summarize_final_evidence.py \
      --selection <fixed_validation/checkpoint_selection.json> \
      --metrics <selected_candidate/metrics.json> \
      --case-metrics <selected_candidate/case_metrics.csv> \
      --runtime <selected_candidate/runtime.json> \
      --activation-audit <fold_0/classification_rescue_activation.json> \
      --rescue-audit <fold_0/checkpoint_classification_rescue.pth.audit.json> \
      --selected-checkpoint <fold_0/selected_checkpoint.pth> \
      --output <final_evidence_summary.json>

The command is CPU-only. It never imports PyTorch or deserializes a checkpoint;
selection-referenced checkpoints are streamed only to verify their SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METRICS_SOURCE = REPOSITORY_ROOT / "src" / "pancreas_multitask"
_METRICS_SPEC = importlib.util.spec_from_file_location(
    "pancreas_multitask_standalone_metrics", METRICS_SOURCE / "metrics.py"
)
if _METRICS_SPEC is None or _METRICS_SPEC.loader is None:
    raise RuntimeError("Could not load the standalone evaluator metric implementation")
EVALUATOR_METRICS = importlib.util.module_from_spec(_METRICS_SPEC)
_METRICS_SPEC.loader.exec_module(EVALUATOR_METRICS)
classification_metrics = EVALUATOR_METRICS.classification_metrics
summarize_values = EVALUATOR_METRICS.summarize_values

EXPECTED_CASE_COLUMNS = (
    "case_id",
    "whole_pancreas_dice",
    "lesion_dice",
    "reference_subtype",
    "predicted_subtype",
    "classification_correct",
    "whole_pancreas_predicted_voxels",
    "whole_pancreas_reference_voxels",
    "lesion_predicted_voxels",
    "lesion_reference_voxels",
    "whole_pancreas_empty_empty",
    "lesion_empty_empty",
)
ORIGINAL_CANDIDATES = {
    "checkpoint_best",
    "checkpoint_best_multitask",
    "checkpoint_final",
}
RESCUE_CANDIDATE = "checkpoint_classification_rescue"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FLOAT_REL_TOLERANCE = 1e-9
FLOAT_ABS_TOLERANCE = 1e-12
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 12345
EXPECTED_CLASSIFICATION_TRAINABLE_PARAMETER_COUNT = 496_195
CLASSIFICATION_PARAMETER_PREFIXES = (
    "classification_pool.",
    "classification_head.",
)


class EvidenceError(RuntimeError):
    """Raised when final evidence is incomplete, inconsistent, or unsafe."""


def _close(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=FLOAT_REL_TOLERANCE,
        abs_tol=FLOAT_ABS_TOLERANCE,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"Could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise EvidenceError(f"Evidence artifact does not exist or is not a file: {source}")
    return {
        "path": str(source),
        "sha256": _file_sha256(source),
        "size_bytes": source.stat().st_size,
    }


def _load_json(path: Path, *, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _artifact(path)
    try:
        payload = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Could not read {role} JSON {artifact['path']}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"{role} JSON must contain an object: {artifact['path']}")
    return payload, artifact


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{field} must be a JSON object")
    return value


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceError(f"{field} must be a JSON array")
    return value


def _integer(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise EvidenceError(f"{field} must be at least {minimum}")
    return result


def _number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise EvidenceError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise EvidenceError(f"{field} must be at most {maximum}")
    return result


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise EvidenceError(f"{field} must be a JSON boolean")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise EvidenceError(f"{field} must be a SHA-256 string")
    digest = value.strip().lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise EvidenceError(f"{field} must contain exactly 64 hexadecimal digits")
    return digest


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise EvidenceError(f"{field} must equal {expected!r}, got {actual!r}")


def _validate_interval(
    value: Any,
    *,
    field: str,
    method: str,
    expected_bounds: tuple[float, float],
) -> dict[str, Any]:
    interval = _mapping(value, field=field)
    _require_equal(interval.get("method"), method, field=f"{field}.method")
    confidence = _number(
        interval.get("confidence"), field=f"{field}.confidence", minimum=0, maximum=1
    )
    if not _close(confidence, 0.95):
        raise EvidenceError(f"{field}.confidence must equal 0.95")
    _require_equal(interval.get("samples"), BOOTSTRAP_SAMPLES, field=f"{field}.samples")
    lower = _number(interval.get("lower"), field=f"{field}.lower", minimum=0, maximum=1)
    upper = _number(interval.get("upper"), field=f"{field}.upper", minimum=0, maximum=1)
    if lower > upper:
        raise EvidenceError(f"{field}.lower exceeds {field}.upper")
    if not _close(lower, expected_bounds[0]) or not _close(upper, expected_bounds[1]):
        raise EvidenceError(f"{field} does not reproduce from the fixed-seed case bootstrap")
    return {
        "method": method,
        "confidence": confidence,
        "samples": BOOTSTRAP_SAMPLES,
        "lower": lower,
        "upper": upper,
    }


def _validate_summary(
    value: Any,
    *,
    field: str,
    case_values: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summary = _mapping(value, field=field)
    _require_equal(summary.get("n"), int(case_values.size), field=f"{field}.n")
    expected = {
        "mean": float(case_values.mean()),
        "sample_std": float(case_values.std(ddof=1)),
        "median": float(np.median(case_values)),
        "minimum": float(case_values.min()),
        "maximum": float(case_values.max()),
    }
    normalized: dict[str, Any] = {"n": int(case_values.size)}
    for name, expected_value in expected.items():
        maximum = None if name == "sample_std" else 1.0
        actual = _number(
            summary.get(name),
            field=f"{field}.{name}",
            minimum=0,
            maximum=maximum,
        )
        if not _close(actual, expected_value):
            raise EvidenceError(
                f"{field}.{name} does not match the case-level values: {actual} != {expected_value}"
            )
        normalized[name] = actual
    recomputed = summarize_values(
        case_values,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
        seed=bootstrap_seed,
    )["bootstrap_ci"]
    normalized["bootstrap_ci"] = _validate_interval(
        summary.get("bootstrap_ci"),
        field=f"{field}.bootstrap_ci",
        method="case_bootstrap_percentile_of_mean",
        expected_bounds=(recomputed["lower"], recomputed["upper"]),
    )
    return normalized


def _parse_csv_integer(value: Any, *, field: str) -> int:
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text) is None:
        raise EvidenceError(f"{field} must be an integer, got {value!r}")
    return int(text)


def _parse_csv_boolean(value: Any, *, field: str) -> bool:
    text = str(value).strip().casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise EvidenceError(f"{field} must be True or False, got {value!r}")


def _normalize_case(row: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise EvidenceError(f"{source}.case_id must be a non-empty string")
    case_id = case_id.strip()
    if Path(case_id).name != case_id:
        raise EvidenceError(f"{source}.case_id must not contain a path: {case_id!r}")

    normalized: dict[str, Any] = {"case_id": case_id}
    for name in ("whole_pancreas_dice", "lesion_dice"):
        normalized[name] = _number(row.get(name), field=f"{source}.{name}", minimum=0, maximum=1)
    for name in ("reference_subtype", "predicted_subtype"):
        value = _integer(row.get(name), field=f"{source}.{name}")
        if value not in (0, 1, 2):
            raise EvidenceError(f"{source}.{name} must be 0, 1, or 2")
        normalized[name] = value
    normalized["classification_correct"] = _boolean(
        row.get("classification_correct"), field=f"{source}.classification_correct"
    )
    for name in (
        "whole_pancreas_predicted_voxels",
        "whole_pancreas_reference_voxels",
        "lesion_predicted_voxels",
        "lesion_reference_voxels",
    ):
        normalized[name] = _integer(row.get(name), field=f"{source}.{name}", minimum=0)
    for name in ("whole_pancreas_empty_empty", "lesion_empty_empty"):
        normalized[name] = _boolean(row.get(name), field=f"{source}.{name}")

    if normalized["classification_correct"] != (
        normalized["reference_subtype"] == normalized["predicted_subtype"]
    ):
        raise EvidenceError(f"{source}.classification_correct is inconsistent")
    if normalized["whole_pancreas_reference_voxels"] <= 0:
        raise EvidenceError(f"{source} has an empty whole-pancreas reference")
    if normalized["lesion_reference_voxels"] <= 0:
        raise EvidenceError(f"{source} has an empty lesion reference")
    if normalized["whole_pancreas_reference_voxels"] < normalized["lesion_reference_voxels"]:
        raise EvidenceError(f"{source} lesion reference exceeds whole-pancreas reference")
    if normalized["whole_pancreas_predicted_voxels"] < normalized["lesion_predicted_voxels"]:
        raise EvidenceError(f"{source} predicted lesion exceeds predicted whole pancreas")
    if normalized["whole_pancreas_empty_empty"] or normalized["lesion_empty_empty"]:
        raise EvidenceError(f"{source} cannot be empty-empty with non-empty references")
    if (
        normalized["whole_pancreas_predicted_voxels"] == 0
        and normalized["whole_pancreas_dice"] != 0
    ):
        raise EvidenceError(f"{source} has non-zero whole Dice with an empty prediction")
    if normalized["lesion_predicted_voxels"] == 0 and normalized["lesion_dice"] != 0:
        raise EvidenceError(f"{source} has non-zero lesion Dice with an empty prediction")
    return normalized


def _read_case_csv(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = _artifact(path)
    try:
        with Path(artifact["path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(EXPECTED_CASE_COLUMNS):
                raise EvidenceError(
                    "Case CSV header must be exactly " + ",".join(EXPECTED_CASE_COLUMNS)
                )
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EvidenceError(f"Could not read case CSV {artifact['path']}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows, start=2):
        converted: dict[str, Any] = {"case_id": str(raw_row["case_id"])}
        for name in ("whole_pancreas_dice", "lesion_dice"):
            try:
                converted[name] = float(str(raw_row[name]).strip())
            except ValueError as exc:
                raise EvidenceError(f"case CSV row {index}.{name} is not numeric") from exc
        for name in (
            "reference_subtype",
            "predicted_subtype",
            "whole_pancreas_predicted_voxels",
            "whole_pancreas_reference_voxels",
            "lesion_predicted_voxels",
            "lesion_reference_voxels",
        ):
            converted[name] = _parse_csv_integer(
                raw_row[name], field=f"case CSV row {index}.{name}"
            )
        for name in (
            "classification_correct",
            "whole_pancreas_empty_empty",
            "lesion_empty_empty",
        ):
            converted[name] = _parse_csv_boolean(
                raw_row[name], field=f"case CSV row {index}.{name}"
            )
        rows.append(_normalize_case(converted, source=f"case CSV row {index}"))

    case_ids = [row["case_id"] for row in rows]
    if case_ids != sorted(case_ids):
        raise EvidenceError("Case CSV rows must be sorted by case_id")
    if len(case_ids) != len(set(case_ids)):
        raise EvidenceError("Case CSV contains duplicate case IDs")
    return rows, artifact


def _case_ids_sha256(case_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for case_id in sorted(case_ids):
        encoded = case_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _assert_case_rows_match(
    json_rows: Sequence[Mapping[str, Any]], csv_rows: Sequence[Mapping[str, Any]]
) -> None:
    if len(json_rows) != len(csv_rows):
        raise EvidenceError("Metrics JSON and case CSV have different row counts")
    for json_row, csv_row in zip(json_rows, csv_rows, strict=True):
        if json_row.keys() != csv_row.keys():
            raise EvidenceError(
                f"Metrics JSON/CSV fields differ for case {csv_row.get('case_id')!r}"
            )
        for key in json_row:
            left = json_row[key]
            right = csv_row[key]
            if isinstance(left, float):
                if not isinstance(right, (int, float)) or not _close(left, float(right)):
                    raise EvidenceError(f"Metrics JSON/CSV mismatch for {csv_row['case_id']}.{key}")
            elif left != right:
                raise EvidenceError(f"Metrics JSON/CSV mismatch for {csv_row['case_id']}.{key}")


def _validate_evaluation_policy(metrics: Mapping[str, Any]) -> None:
    policy = _mapping(metrics.get("evaluation_policy"), field="metrics.evaluation_policy")
    expected = {
        "whole_pancreas": "label > 0",
        "lesion": "label == 2",
        "empty_empty_dice": 1.0,
        "one_sided_empty_dice": 0.0,
        "classification_labels": [0, 1, 2],
        "classification_zero_division": 0.0,
        "confusion_matrix_rows": "reference",
        "confusion_matrix_columns": "prediction",
        "aggregation": "unweighted case mean",
        "bootstrap_seed": 12345,
    }
    for key, expected_value in expected.items():
        _require_equal(policy.get(key), expected_value, field=f"metrics.evaluation_policy.{key}")


def _validate_classification(
    value: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    expected_supports: tuple[int, int, int],
) -> dict[str, Any]:
    classification = _mapping(value, field="metrics.classification")
    case_count = len(rows)
    _require_equal(classification.get("case_count"), case_count, field="classification.case_count")
    _require_equal(classification.get("labels"), [0, 1, 2], field="classification.labels")
    _require_equal(
        classification.get("confusion_matrix_axes"),
        {"rows": "reference", "columns": "prediction"},
        field="classification.confusion_matrix_axes",
    )
    _require_equal(classification.get("zero_division"), 0.0, field="classification.zero_division")
    _require_equal(
        classification.get("unused_reference_case_count"),
        0,
        field="classification.unused_reference_case_count",
    )

    matrix_rows = _list(
        classification.get("confusion_matrix"), field="classification.confusion_matrix"
    )
    if len(matrix_rows) != 3:
        raise EvidenceError("classification.confusion_matrix must have three rows")
    matrix = np.empty((3, 3), dtype=np.int64)
    for row_index, raw_row in enumerate(matrix_rows):
        values = _list(raw_row, field=f"classification.confusion_matrix[{row_index}]")
        if len(values) != 3:
            raise EvidenceError("classification.confusion_matrix must be 3x3")
        for column_index, raw_value in enumerate(values):
            matrix[row_index, column_index] = _integer(
                raw_value,
                field=f"classification.confusion_matrix[{row_index}][{column_index}]",
                minimum=0,
            )
    if int(matrix.sum()) != case_count:
        raise EvidenceError("Classification confusion matrix does not sum to the case count")

    expected_matrix = np.zeros((3, 3), dtype=np.int64)
    for row in rows:
        expected_matrix[row["reference_subtype"], row["predicted_subtype"]] += 1
    if not np.array_equal(matrix, expected_matrix):
        raise EvidenceError("Classification confusion matrix disagrees with case-level rows")
    if tuple(int(value) for value in matrix.sum(axis=1)) != expected_supports:
        raise EvidenceError(
            "Classification supports do not match the expected validation composition"
        )

    per_class = _list(classification.get("per_class"), field="classification.per_class")
    if len(per_class) != 3:
        raise EvidenceError("classification.per_class must contain exactly three rows")
    normalized_per_class: list[dict[str, Any]] = []
    for index, raw in enumerate(per_class):
        item = _mapping(raw, field=f"classification.per_class[{index}]")
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        tn = case_count - tp - fp - fn
        support = tp + fn
        predicted_count = tp + fp
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        expected_integer_fields = {
            "label": index,
            "support": support,
            "predicted_count": predicted_count,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        }
        for name, expected_value in expected_integer_fields.items():
            _require_equal(
                item.get(name), expected_value, field=f"classification.per_class[{index}].{name}"
            )
        normalized_item = dict(expected_integer_fields)
        for name, expected_value in (("precision", precision), ("recall", recall), ("f1", f1)):
            actual = _number(
                item.get(name),
                field=f"classification.per_class[{index}].{name}",
                minimum=0,
                maximum=1,
            )
            if not _close(actual, expected_value):
                raise EvidenceError(
                    f"classification.per_class[{index}].{name} disagrees with the matrix"
                )
            normalized_item[name] = actual
        normalized_per_class.append(normalized_item)

    expected_aggregates = {
        "accuracy": float(np.trace(matrix) / case_count),
        "macro_precision": float(np.mean([row["precision"] for row in normalized_per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in normalized_per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in normalized_per_class])),
    }
    normalized: dict[str, Any] = {
        "case_count": case_count,
        "labels": [0, 1, 2],
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {"rows": "reference", "columns": "prediction"},
        "per_class": normalized_per_class,
    }
    for name, expected_value in expected_aggregates.items():
        actual = _number(
            classification.get(name), field=f"classification.{name}", minimum=0, maximum=1
        )
        if not _close(actual, expected_value):
            raise EvidenceError(f"classification.{name} disagrees with the confusion matrix")
        normalized[name] = actual

    intervals = _mapping(classification.get("bootstrap_ci"), field="classification.bootstrap_ci")
    recomputed_classification = classification_metrics(
        [row["reference_subtype"] for row in rows],
        [row["predicted_subtype"] for row in rows],
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
        seed=BOOTSTRAP_SEED,
    )["bootstrap_ci"]
    normalized["bootstrap_ci"] = {
        "macro_f1": _validate_interval(
            intervals.get("macro_f1"),
            field="classification.bootstrap_ci.macro_f1",
            method="case_bootstrap_percentile",
            expected_bounds=(
                recomputed_classification["macro_f1"]["lower"],
                recomputed_classification["macro_f1"]["upper"],
            ),
        ),
        "accuracy": _validate_interval(
            intervals.get("accuracy"),
            field="classification.bootstrap_ci.accuracy",
            method="case_bootstrap_percentile",
            expected_bounds=(
                recomputed_classification["accuracy"]["lower"],
                recomputed_classification["accuracy"]["upper"],
            ),
        ),
    }
    return normalized


def _validate_metrics(
    metrics: Mapping[str, Any],
    csv_rows: Sequence[Mapping[str, Any]],
    *,
    expected_case_count: int,
    expected_supports: tuple[int, int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_equal(metrics.get("schema_version"), 1, field="metrics.schema_version")
    _require_equal(metrics.get("case_count"), expected_case_count, field="metrics.case_count")
    if len(csv_rows) != expected_case_count:
        raise EvidenceError(
            f"Case CSV must contain {expected_case_count} cases, got {len(csv_rows)}"
        )
    if sum(expected_supports) != expected_case_count:
        raise EvidenceError("Expected class supports do not sum to the expected case count")
    _validate_evaluation_policy(metrics)

    json_cases_raw = _list(metrics.get("cases"), field="metrics.cases")
    json_rows = [
        _normalize_case(
            _mapping(row, field=f"metrics.cases[{index}]"), source=f"metrics.cases[{index}]"
        )
        for index, row in enumerate(json_cases_raw)
    ]
    json_case_ids = [row["case_id"] for row in json_rows]
    if json_case_ids != sorted(json_case_ids) or len(json_case_ids) != len(set(json_case_ids)):
        raise EvidenceError("metrics.cases must contain unique rows sorted by case_id")
    _assert_case_rows_match(json_rows, csv_rows)

    segmentation = _mapping(metrics.get("segmentation"), field="metrics.segmentation")
    _require_equal(
        segmentation.get("case_count"), expected_case_count, field="segmentation.case_count"
    )
    whole_values = np.asarray([row["whole_pancreas_dice"] for row in json_rows])
    lesion_values = np.asarray([row["lesion_dice"] for row in json_rows])
    whole_summary = _validate_summary(
        segmentation.get("whole_pancreas_dice"),
        field="segmentation.whole_pancreas_dice",
        case_values=whole_values,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    lesion_summary = _validate_summary(
        segmentation.get("lesion_dice"),
        field="segmentation.lesion_dice",
        case_values=lesion_values,
        bootstrap_seed=BOOTSTRAP_SEED + 1,
    )
    empty_cases = _mapping(segmentation.get("empty_cases"), field="segmentation.empty_cases")
    expected_empty_counts = {
        "whole_pancreas_prediction_empty": sum(
            row["whole_pancreas_predicted_voxels"] == 0 for row in json_rows
        ),
        "whole_pancreas_reference_empty": 0,
        "whole_pancreas_both_empty": 0,
        "lesion_prediction_empty": sum(row["lesion_predicted_voxels"] == 0 for row in json_rows),
        "lesion_reference_empty": 0,
        "lesion_both_empty": 0,
    }
    for name, expected_value in expected_empty_counts.items():
        _require_equal(
            empty_cases.get(name), expected_value, field=f"segmentation.empty_cases.{name}"
        )

    classification = _validate_classification(
        metrics.get("classification"), rows=json_rows, expected_supports=expected_supports
    )
    return {
        "segmentation": {
            "whole_pancreas_dice": whole_summary,
            "lesion_dice": lesion_summary,
            "empty_cases": expected_empty_counts,
        },
        "classification": classification,
    }, json_rows


def _validate_activation_window(
    value: Any,
    *,
    field: str,
    audit_epoch: int,
    hard_rule: bool,
) -> bool:
    window = _mapping(value, field=field)
    _require_equal(window.get("audit_epoch"), audit_epoch, field=f"{field}.audit_epoch")
    expected_epochs = list(range(audit_epoch - 9, audit_epoch + 1))
    _require_equal(window.get("window_epochs"), expected_epochs, field=f"{field}.window_epochs")
    losses_raw = _list(
        window.get("training_classification_ce_values"),
        field=f"{field}.training_classification_ce_values",
    )
    accuracies_raw = _list(
        window.get("training_patch_accuracy_values"),
        field=f"{field}.training_patch_accuracy_values",
    )
    if len(losses_raw) != 10 or len(accuracies_raw) != 10:
        raise EvidenceError(f"{field} must contain exactly ten loss and accuracy values")
    losses = np.asarray(
        [_number(value, field=f"{field}.loss[{index}]") for index, value in enumerate(losses_raw)]
    )
    accuracies = np.asarray(
        [
            _number(value, field=f"{field}.accuracy[{index}]", minimum=0, maximum=1)
            for index, value in enumerate(accuracies_raw)
        ]
    )
    mean_loss = _number(window.get("mean_training_classification_ce"), field=f"{field}.mean_ce")
    mean_accuracy = _number(
        window.get("mean_training_patch_accuracy"),
        field=f"{field}.mean_accuracy",
        minimum=0,
        maximum=1,
    )
    slope = _number(window.get("classification_ce_ols_slope_per_epoch"), field=f"{field}.slope")
    expected_slope = float(np.polyfit(np.arange(10, dtype=np.float64), losses, 1)[0])
    if not _close(mean_loss, float(losses.mean())):
        raise EvidenceError(f"{field}.mean_training_classification_ce is inconsistent")
    if not _close(mean_accuracy, float(accuracies.mean())):
        raise EvidenceError(f"{field}.mean_training_patch_accuracy is inconsistent")
    if not _close(slope, expected_slope):
        raise EvidenceError(f"{field}.classification_ce_ols_slope_per_epoch is inconsistent")

    conditions = _mapping(window.get("conditions"), field=f"{field}.conditions")
    if hard_rule:
        expected_conditions = {
            "ce_above_1_03": mean_loss > 1.03,
            "accuracy_below_0_45": mean_accuracy < 0.45,
        }
        expected_trigger = any(expected_conditions.values())
    else:
        expected_conditions = {
            "ce_at_least_1_05": mean_loss >= 1.05,
            "accuracy_at_most_0_42": mean_accuracy <= 0.42,
            "ce_slope_at_least_negative_0_001": slope >= -0.001,
        }
        expected_trigger = all(expected_conditions.values())
    for name, expected_value in expected_conditions.items():
        _require_equal(conditions.get(name), expected_value, field=f"{field}.conditions.{name}")
    _require_equal(window.get("triggered"), expected_trigger, field=f"{field}.triggered")
    return expected_trigger


def _validate_activation(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_equal(value.get("schema_version"), 1, field="activation.schema_version")
    _require_equal(
        value.get("source_checkpoint_name"),
        "checkpoint_final.pth",
        field="activation.source_checkpoint_name",
    )
    source_sha256 = _sha256(
        value.get("source_checkpoint_sha256"), field="activation.source_checkpoint_sha256"
    )
    _require_equal(value.get("checkpoint_current_epoch"), 200, field="activation.current_epoch")
    _require_equal(
        value.get("training_logging_epoch_count"), 200, field="activation.logging_epoch_count"
    )
    _require_equal(
        value.get("metric_scope"),
        "checkpoint_training_logging_only",
        field="activation.metric_scope",
    )
    _require_equal(value.get("validation_metrics_read"), False, field="activation.validation_read")
    _require_equal(
        value.get("validation_used_for_activation"),
        False,
        field="activation.validation_used",
    )
    epoch_40_trigger = _validate_activation_window(
        value.get("epoch_40"), field="activation.epoch_40", audit_epoch=40, hard_rule=False
    )
    epoch_50_trigger = _validate_activation_window(
        value.get("epoch_50_hard_audit"),
        field="activation.epoch_50_hard_audit",
        audit_epoch=50,
        hard_rule=True,
    )
    approved = _boolean(value.get("activation_approved"), field="activation.activation_approved")
    expected_approved = epoch_40_trigger or epoch_50_trigger
    expected_decision = 40 if epoch_40_trigger else (50 if epoch_50_trigger else None)
    _require_equal(approved, expected_approved, field="activation.activation_approved")
    _require_equal(
        value.get("decision_epoch"), expected_decision, field="activation.decision_epoch"
    )
    epoch_40 = _mapping(value["epoch_40"], field="activation.epoch_40")
    return {
        "activation_approved": approved,
        "decision_epoch": expected_decision,
        "source_checkpoint": str(value.get("source_checkpoint", "")),
        "source_checkpoint_sha256": source_sha256,
        "epoch_40": {
            "mean_training_classification_ce": epoch_40["mean_training_classification_ce"],
            "mean_training_patch_accuracy": epoch_40["mean_training_patch_accuracy"],
            "classification_ce_ols_slope_per_epoch": epoch_40[
                "classification_ce_ols_slope_per_epoch"
            ],
            "triggered": epoch_40_trigger,
        },
        "epoch_50_hard_audit_triggered": epoch_50_trigger,
    }


def _validate_component_hashes(value: Any, *, field: str) -> dict[str, str]:
    hashes = _mapping(value, field=field)
    return {
        name: _sha256(hashes.get(name), field=f"{field}.{name}")
        for name in ("encoder", "decoder", "classification")
    }


def _validate_execution_recovery(
    rescue: Mapping[str, Any],
    *,
    activation: Mapping[str, Any],
    activation_artifact_sha256: str,
) -> dict[str, Any]:
    process_launch_count = _integer(
        rescue.get("process_launch_count"), field="rescue.process_launch_count", minimum=1
    )
    zero_update_recovery_count = _integer(
        rescue.get("zero_update_recovery_count"),
        field="rescue.zero_update_recovery_count",
        minimum=0,
    )
    update_bearing_trajectory_count = _integer(
        rescue.get("update_bearing_trajectory_count"),
        field="rescue.update_bearing_trajectory_count",
        minimum=1,
    )
    if update_bearing_trajectory_count != 1:
        raise EvidenceError("Rescue must contain exactly one update-bearing trajectory")
    recovery_binding_fields = (
        "execution_recovery",
        "execution_recovery_audit",
        "execution_recovery_audit_sha256",
    )
    if (process_launch_count, zero_update_recovery_count) == (1, 0):
        fabricated = [name for name in recovery_binding_fields if name in rescue]
        if fabricated:
            raise EvidenceError(
                "Clean rescue branch must not contain recovery fields: " + ", ".join(fabricated)
            )
        source_checkpoint_raw = rescue.get("source_checkpoint")
        if not isinstance(source_checkpoint_raw, str) or not source_checkpoint_raw.strip():
            raise EvidenceError("rescue.source_checkpoint must be a non-empty path")
        canonical_recovery_audit = (
            Path(source_checkpoint_raw)
            .resolve()
            .with_name("classification_rescue_zero_update_recovery.json")
        )
        if canonical_recovery_audit.is_file():
            raise EvidenceError("Clean rescue branch conflicts with a canonical recovery artifact")
        return {
            "artifact": None,
            "failed_launch_logs": {},
            "process_launch_count": 1,
            "zero_update_recovery_count": 0,
            "update_bearing_trajectory_count": 1,
            "failed_launch_optimizer_updates": 0,
            "failed_launch_training_batches": 0,
            "rescue_process_validation_batches_consumed": 0,
        }
    if (process_launch_count, zero_update_recovery_count) != (2, 1):
        raise EvidenceError(
            "Rescue process/recovery counts must be clean 1/0 or canonical recovered 2/1"
        )
    recovery_path_raw = rescue.get("execution_recovery_audit")
    if not isinstance(recovery_path_raw, str) or not recovery_path_raw.strip():
        raise EvidenceError("rescue.execution_recovery_audit must be a non-empty path")
    recovery_payload, recovery_artifact = _load_json(
        Path(recovery_path_raw), role="execution-recovery audit"
    )
    recorded_recovery_sha = _sha256(
        rescue.get("execution_recovery_audit_sha256"),
        field="rescue.execution_recovery_audit_sha256",
    )
    if recorded_recovery_sha != recovery_artifact["sha256"]:
        raise EvidenceError("Rescue audit execution-recovery SHA-256 mismatch")
    embedded = _mapping(rescue.get("execution_recovery"), field="rescue.execution_recovery")
    if dict(embedded) != recovery_payload:
        raise EvidenceError("Rescue audit execution_recovery differs from its hash-bound artifact")

    _require_equal(recovery_payload.get("schema_version"), 1, field="recovery.schema_version")
    _require_equal(
        recovery_payload.get("event"),
        "classification_rescue_zero_update_execution_recovery",
        field="recovery.event",
    )
    _require_equal(
        recovery_payload.get("status"),
        "authorized_before_custom_joint_fixed_validation",
        field="recovery.status",
    )
    _require_equal(
        recovery_payload.get("activation_approved"), True, field="recovery.activation_approved"
    )
    _require_equal(
        _sha256(
            recovery_payload.get("source_checkpoint_sha256"),
            field="recovery.source_checkpoint_sha256",
        ),
        activation["source_checkpoint_sha256"],
        field="recovery.source_checkpoint_sha256",
    )
    _require_equal(
        _sha256(
            recovery_payload.get("activation_audit_sha256"),
            field="recovery.activation_audit_sha256",
        ),
        activation_artifact_sha256,
        field="recovery.activation_audit_sha256",
    )

    failed = _mapping(recovery_payload.get("failed_launch"), field="recovery.failed_launch")
    expected_failed = {
        "process_launch_index": 1,
        "failed_step_index": 0,
        "training_batches_consumed": 1,
        "training_samples_consumed": 2,
        "finite_loss_guard_passed": True,
        "failure_stage": "after_grad_scaler_unscale_before_gradient_clip_completion",
        "optimizer_step_reached": False,
        "optimizer_updates": 0,
        "completed_epochs": 0,
        "checkpoint_written": False,
        "rescue_audit_written": False,
        "first_step_zero_update_operator_attested": True,
    }
    for name, expected in expected_failed.items():
        _require_equal(failed.get(name), expected, field=f"recovery.failed_launch.{name}")

    recovery_root = Path(recovery_artifact["path"]).parent.resolve()
    log_artifacts: dict[str, Any] = {}
    for stream in ("stdout", "stderr"):
        descriptor = _mapping(
            failed.get(f"{stream}_artifact"),
            field=f"recovery.failed_launch.{stream}_artifact",
        )
        name = descriptor.get("name")
        if not isinstance(name, str) or not name.strip():
            raise EvidenceError(f"Recovery {stream} artifact name must be non-empty")
        log_path = (recovery_root / name).resolve()
        try:
            log_path.relative_to(recovery_root)
        except ValueError as exc:
            raise EvidenceError(f"Recovery {stream} artifact escapes its root") from exc
        artifact = _artifact(log_path)
        _require_equal(
            descriptor.get("bytes"),
            artifact["size_bytes"],
            field=f"recovery.failed_launch.{stream}_artifact.bytes",
        )
        _require_equal(
            _sha256(
                descriptor.get("sha256"),
                field=f"recovery.failed_launch.{stream}_artifact.sha256",
            ),
            artifact["sha256"],
            field=f"recovery.failed_launch.{stream}_artifact.sha256",
        )
        log_artifacts[stream] = artifact

    validation = _mapping(recovery_payload.get("validation"), field="recovery.validation")
    for name, expected in {
        "stock_nnunet_segmentation_only_validation_completed": True,
        "stock_nnunet_validation_metrics_observed_before_recovery": True,
        "stock_nnunet_mean_foreground_dice_observed_before_recovery": 0.753518646,
        "stock_nnunet_validation_used_for_recovery": False,
        "custom_joint_fixed_validation_started": False,
        "custom_joint_fixed_validation_output_existed_at_authorization": False,
        "rescue_process_validation_images_opened": False,
        "rescue_process_validation_batches_consumed": 0,
        "rescue_process_validation_used_for_recovery": False,
    }.items():
        _require_equal(validation.get(name), expected, field=f"recovery.validation.{name}")

    policy = _mapping(recovery_payload.get("recovery_policy"), field="recovery.recovery_policy")
    for name, expected in {
        "schedule_changed": False,
        "source_checkpoint_changed": False,
        "reset_seed_changed": False,
        "maximum_update_bearing_trajectories": 1,
        "maximum_zero_update_runtime_recoveries": 1,
        "process_launch_count_after_relaunch": 2,
        "no_further_recovery_allowed": True,
    }.items():
        _require_equal(policy.get(name), expected, field=f"recovery.recovery_policy.{name}")

    return {
        "artifact": recovery_artifact,
        "failed_launch_logs": log_artifacts,
        "process_launch_count": 2,
        "zero_update_recovery_count": 1,
        "update_bearing_trajectory_count": 1,
        "failed_launch_optimizer_updates": 0,
        "failed_launch_training_batches": 1,
        "rescue_process_validation_batches_consumed": 0,
    }


def _validate_rescue(
    rescue: Mapping[str, Any],
    *,
    activation: Mapping[str, Any],
    activation_artifact_sha256: str,
    expected_case_count: int,
) -> dict[str, Any]:
    _require_equal(rescue.get("schema_version"), 1, field="rescue.schema_version")
    _require_equal(rescue.get("status"), "complete", field="rescue.status")
    _require_equal(
        rescue.get("method"),
        "post_training_frozen_backbone_classification_head_rescue",
        field="rescue.method",
    )
    _require_equal(rescue.get("completed_epochs"), 30, field="rescue.completed_epochs")
    execution_recovery = _validate_execution_recovery(
        rescue,
        activation=activation,
        activation_artifact_sha256=activation_artifact_sha256,
    )
    _require_equal(rescue.get("optimizer"), "AdamW", field="rescue.optimizer")
    precision_policy = _mapping(rescue.get("precision_policy"), field="rescue.precision_policy")
    for name, expected_value in {
        "autocast_scope": "frozen_encoder_forward_only",
        "frozen_encoder_forward": "cuda_autocast_float16",
        "trainable_classification_forward": "float32",
        "classification_loss": "float32",
        "classification_backward": "float32",
        "gradient_clipping": "float32",
        "optimizer_update": "float32",
        "grad_scaler_enabled": False,
    }.items():
        _require_equal(
            precision_policy.get(name),
            expected_value,
            field=f"rescue.precision_policy.{name}",
        )
    _require_equal(
        rescue.get("successful_optimizer_updates"),
        3750,
        field="rescue.successful_optimizer_updates",
    )
    _require_equal(
        rescue.get("training_loader"),
        "single_threaded_training_split_only",
        field="rescue.training_loader",
    )
    _require_equal(rescue.get("training_batch_size"), 2, field="rescue.training_batch_size")
    parameter_names_raw = _list(
        rescue.get("classification_parameter_names"),
        field="rescue.classification_parameter_names",
    )
    if not parameter_names_raw or any(
        not isinstance(name, str) or not name for name in parameter_names_raw
    ):
        raise EvidenceError("rescue.classification_parameter_names must contain names")
    parameter_names = [str(name) for name in parameter_names_raw]
    if len(parameter_names) != len(set(parameter_names)):
        raise EvidenceError("rescue.classification_parameter_names contains duplicates")
    if any(
        not name.startswith(CLASSIFICATION_PARAMETER_PREFIXES) for name in parameter_names
    ) or not all(
        any(name.startswith(prefix) for name in parameter_names)
        for prefix in CLASSIFICATION_PARAMETER_PREFIXES
    ):
        raise EvidenceError("Rescue trainable parameters are outside the pool/head scope")
    trainable_parameter_count = _integer(
        rescue.get("classification_trainable_parameter_count"),
        field="rescue.classification_trainable_parameter_count",
        minimum=1,
    )
    _require_equal(
        trainable_parameter_count,
        EXPECTED_CLASSIFICATION_TRAINABLE_PARAMETER_COUNT,
        field="rescue.classification_trainable_parameter_count",
    )
    for field in (
        "decoder_executed_during_rescue",
        "encoder_gradient_enabled",
        "decoder_gradient_enabled",
    ):
        _require_equal(rescue.get(field), False, field=f"rescue.{field}")
    _require_equal(
        rescue.get("selection_or_stopping_metric"),
        None,
        field="rescue.selection_or_stopping_metric",
    )
    _require_equal(
        rescue.get("activation_decision_epoch"),
        activation["decision_epoch"],
        field="rescue.activation_decision_epoch",
    )
    source_sha = _sha256(
        rescue.get("source_checkpoint_sha256"), field="rescue.source_checkpoint_sha256"
    )
    if source_sha != activation["source_checkpoint_sha256"]:
        raise EvidenceError("Rescue and activation audit reference different source checkpoints")
    recorded_activation_sha = _sha256(
        rescue.get("activation_audit_sha256"), field="rescue.activation_audit_sha256"
    )
    if recorded_activation_sha != activation_artifact_sha256:
        raise EvidenceError("Rescue audit is not bound to the supplied activation audit")
    output_sha = _sha256(
        rescue.get("output_checkpoint_sha256"), field="rescue.output_checkpoint_sha256"
    )

    schedule = _mapping(rescue.get("schedule"), field="rescue.schedule")
    expected_schedule = {
        "epochs": 30,
        "iterations_per_epoch": 125,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "label_smoothing": 0.05,
        "nonlesion_patch_weight": 0.25,
        "reset_seed": 20260806,
    }
    for name, expected_value in expected_schedule.items():
        actual = schedule.get(name)
        if isinstance(expected_value, float):
            actual_number = _number(actual, field=f"rescue.schedule.{name}")
            if not _close(actual_number, expected_value):
                raise EvidenceError(f"rescue.schedule.{name} differs from the frozen schedule")
        else:
            _require_equal(actual, expected_value, field=f"rescue.schedule.{name}")

    _require_equal(
        rescue.get("training_class_counts"), [62, 106, 84], field="rescue.training_class_counts"
    )
    expected_weights = [252 / (3 * count) for count in (62, 106, 84)]
    weights = _list(rescue.get("training_class_weights"), field="rescue.training_class_weights")
    if len(weights) != 3:
        raise EvidenceError("rescue.training_class_weights must contain three values")
    for index, expected_weight in enumerate(expected_weights):
        actual_weight = _number(weights[index], field=f"rescue.training_class_weights[{index}]")
        if not math.isclose(actual_weight, expected_weight, rel_tol=1e-6, abs_tol=1e-8):
            raise EvidenceError("Rescue class weights do not match the training counts")

    split = _mapping(rescue.get("split_audit"), field="rescue.split_audit")
    expected_split = {
        "training_case_count": 252,
        "validation_case_count": expected_case_count,
        "split_disjoint": True,
        "validation_images_opened": False,
        "validation_batches_consumed": 0,
        "validation_used_for_gradients": False,
        "validation_used_for_stopping": False,
    }
    for name, expected_value in expected_split.items():
        _require_equal(split.get(name), expected_value, field=f"rescue.split_audit.{name}")
    _sha256(split.get("training_case_ids_sha256"), field="rescue.training_case_ids_sha256")
    _sha256(split.get("validation_case_ids_sha256"), field="rescue.validation_case_ids_sha256")

    source_components = _validate_component_hashes(
        rescue.get("source_component_sha256"), field="rescue.source_component_sha256"
    )
    current_components = _validate_component_hashes(
        rescue.get("current_component_sha256"), field="rescue.current_component_sha256"
    )
    for name in ("encoder", "decoder"):
        if source_components[name] != current_components[name]:
            raise EvidenceError(f"Frozen rescue component changed: {name}")
    if source_components["classification"] == current_components["classification"]:
        raise EvidenceError("Rescue classification state did not change")

    history = _list(rescue.get("training_only_history"), field="rescue.training_only_history")
    if len(history) != 30:
        raise EvidenceError("Rescue training history must contain exactly 30 epochs")
    history_seconds = 0.0
    for epoch, raw in enumerate(history):
        item = _mapping(raw, field=f"rescue.training_only_history[{epoch}]")
        _require_equal(item.get("epoch"), epoch, field=f"rescue.history[{epoch}].epoch")
        _require_equal(
            item.get("successful_optimizer_updates"),
            125,
            field=f"rescue.history[{epoch}].successful_optimizer_updates",
        )
        _require_equal(
            item.get("generalization_metric"),
            False,
            field=f"rescue.history[{epoch}].generalization_metric",
        )
        _number(item.get("training_loss_mean"), field=f"rescue.history[{epoch}].loss")
        _number(
            item.get("training_patch_accuracy"),
            field=f"rescue.history[{epoch}].accuracy",
            minimum=0,
            maximum=1,
        )
        history_seconds += _number(
            item.get("elapsed_seconds"),
            field=f"rescue.history[{epoch}].elapsed_seconds",
            minimum=0,
        )
    return {
        "status": "complete",
        "completed_head_only_epochs": 30,
        "training_patch_updates": 3750,
        "successful_optimizer_updates": 3750,
        "precision_policy": dict(precision_policy),
        "activation_decision_epoch": activation["decision_epoch"],
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint": str(rescue.get("source_checkpoint", "")),
        "output_checkpoint_sha256": output_sha,
        "output_checkpoint": str(rescue.get("output_checkpoint", "")),
        "activation_audit_sha256": recorded_activation_sha,
        "backbone_invariance_evidence_scope": "audit_recorded_component_hash_comparison",
        "audit_recorded_encoder_unchanged": True,
        "audit_recorded_decoder_unchanged": True,
        "declared_trainable_scope_validated": True,
        "declared_classification_trainable_parameter_count": trainable_parameter_count,
        "validation_batches_consumed": 0,
        "validation_case_ids_sha256": _sha256(
            split.get("validation_case_ids_sha256"),
            field="rescue.validation_case_ids_sha256",
        ),
        "summed_epoch_compute_seconds": history_seconds,
        "execution_recovery": execution_recovery,
    }


def _validate_candidate_metrics_artifact(
    path: Path,
    *,
    candidate: str,
    expected_case_count: int,
    expected_supports: tuple[int, int, int],
    expected_case_ids: Sequence[str],
    expected_metrics: Mapping[str, float],
) -> dict[str, Any]:
    payload, artifact = _load_json(path, role=f"{candidate} candidate metrics")
    raw_rows = _list(payload.get("cases"), field=f"{candidate}.metrics.cases")
    candidate_rows = [
        _normalize_case(
            _mapping(row, field=f"{candidate}.metrics.cases[{index}]"),
            source=f"{candidate}.metrics.cases[{index}]",
        )
        for index, row in enumerate(raw_rows)
    ]
    normalized, validated_rows = _validate_metrics(
        payload,
        candidate_rows,
        expected_case_count=expected_case_count,
        expected_supports=expected_supports,
    )
    candidate_case_ids = [row["case_id"] for row in validated_rows]
    if candidate_case_ids != list(expected_case_ids):
        raise EvidenceError(
            f"{candidate} was not evaluated on the exact selected validation case IDs"
        )
    discovered = {
        "whole_pancreas_dice": normalized["segmentation"]["whole_pancreas_dice"]["mean"],
        "lesion_dice": normalized["segmentation"]["lesion_dice"]["mean"],
        "macro_f1": normalized["classification"]["macro_f1"],
    }
    for name, expected_value in expected_metrics.items():
        if not _close(discovered[name], expected_value):
            raise EvidenceError(
                f"Selection metric {name} for {candidate} disagrees with its evaluator JSON"
            )
    return {**artifact, "case_ids_sha256": _case_ids_sha256(candidate_case_ids)}


def _validate_selection(
    selection: Mapping[str, Any],
    *,
    metrics_path: Path,
    metrics_summary: Mapping[str, Any],
    activation_approved: bool,
    activation_source_checkpoint: str,
    activation_source_sha256: str,
    rescue_summary: Mapping[str, Any] | None,
    expected_case_count: int,
    expected_supports: tuple[int, int, int],
    expected_case_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_equal(selection.get("schema_version"), 1, field="selection.schema_version")
    policy = _mapping(selection.get("selection_policy"), field="selection.selection_policy")
    _require_equal(policy.get("direction"), "maximize", field="selection.policy.direction")
    _require_equal(
        policy.get("metric_paths"),
        [
            "segmentation.whole_pancreas_dice.mean",
            "segmentation.lesion_dice.mean",
            "classification.macro_f1",
        ],
        field="selection.policy.metric_paths",
    )
    weights = _mapping(policy.get("metric_weights"), field="selection.policy.metric_weights")
    for name in ("whole_pancreas_dice", "lesion_dice", "macro_f1"):
        weight = _number(weights.get(name), field=f"selection.policy.metric_weights.{name}")
        if not _close(weight, 1 / 3):
            raise EvidenceError("Selection metric weights must all equal one third")
    _require_equal(
        policy.get("score"), "equal-weight arithmetic mean", field="selection.policy.score"
    )
    _require_equal(
        policy.get("tie_breaker"),
        "candidate name ascending; no secondary metric",
        field="selection.policy.tie_breaker",
    )
    expected_names = set(ORIGINAL_CANDIDATES)
    if activation_approved:
        expected_names.add(RESCUE_CANDIDATE)
    expected_count = len(expected_names)
    _require_equal(
        selection.get("candidate_count"), expected_count, field="selection.candidate_count"
    )
    ranking_raw = _list(selection.get("ranking"), field="selection.ranking")
    if len(ranking_raw) != expected_count:
        raise EvidenceError("Selection ranking length differs from candidate_count")

    ranking: list[dict[str, Any]] = []
    for index, raw in enumerate(ranking_raw, start=1):
        item = _mapping(raw, field=f"selection.ranking[{index - 1}]")
        candidate = item.get("candidate")
        if not isinstance(candidate, str) or candidate not in expected_names:
            raise EvidenceError(f"Unexpected checkpoint candidate: {candidate!r}")
        _require_equal(item.get("rank"), index, field=f"selection.{candidate}.rank")
        candidate_metrics = _mapping(item.get("metrics"), field=f"selection.{candidate}.metrics")
        normalized_metrics = {
            name: _number(
                candidate_metrics.get(name),
                field=f"selection.{candidate}.metrics.{name}",
                minimum=0,
                maximum=1,
            )
            for name in ("whole_pancreas_dice", "lesion_dice", "macro_f1")
        }
        score = _number(
            item.get("selection_score"),
            field=f"selection.{candidate}.selection_score",
            minimum=0,
            maximum=1,
        )
        expected_score = math.fsum(normalized_metrics.values()) / 3
        if not _close(score, expected_score):
            raise EvidenceError(f"Selection score is inconsistent for {candidate}")
        checkpoint_path = str(item.get("checkpoint_path", ""))
        if not checkpoint_path or Path(checkpoint_path).name != f"{candidate}.pth":
            raise EvidenceError(f"Selection checkpoint path has the wrong filename for {candidate}")
        checkpoint_artifact = _artifact(Path(checkpoint_path))
        checkpoint_sha256 = _sha256(
            item.get("checkpoint_sha256"), field=f"selection.{candidate}.checkpoint_sha256"
        )
        if checkpoint_artifact["sha256"] != checkpoint_sha256:
            raise EvidenceError(f"Selection checkpoint SHA-256 is stale for {candidate}")
        metrics_source = str(item.get("metrics_source", ""))
        if not metrics_source:
            raise EvidenceError(f"Selection metrics source is missing for {candidate}")
        metrics_artifact = _validate_candidate_metrics_artifact(
            Path(metrics_source),
            candidate=candidate,
            expected_case_count=expected_case_count,
            expected_supports=expected_supports,
            expected_case_ids=expected_case_ids,
            expected_metrics=normalized_metrics,
        )
        ranking.append(
            {
                "candidate": candidate,
                "rank": index,
                "metrics": normalized_metrics,
                "selection_score": score,
                "metrics_source": metrics_artifact["path"],
                "metrics_artifact": metrics_artifact,
                "checkpoint_path": checkpoint_artifact["path"],
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_artifact": checkpoint_artifact,
            }
        )
    if {item["candidate"] for item in ranking} != expected_names:
        raise EvidenceError("Selection ranking does not contain the exact candidate set")
    if len({Path(item["metrics_source"]).resolve() for item in ranking}) != expected_count:
        raise EvidenceError("Selection candidates must use distinct evaluator JSON files")
    if len({Path(item["checkpoint_path"]).resolve() for item in ranking}) != expected_count:
        raise EvidenceError("Selection candidates must use distinct checkpoint files")
    expected_order = sorted(ranking, key=lambda item: (-item["selection_score"], item["candidate"]))
    if [item["candidate"] for item in ranking] != [item["candidate"] for item in expected_order]:
        raise EvidenceError("Selection ranking does not follow the predeclared score/tie rule")

    selected = ranking[0]
    _require_equal(
        selection.get("selected_candidate"),
        selected["candidate"],
        field="selection.selected_candidate",
    )
    selected_score = _number(
        selection.get("selected_score"), field="selection.selected_score", minimum=0, maximum=1
    )
    if not _close(selected_score, selected["selection_score"]):
        raise EvidenceError("selection.selected_score differs from the rank-1 score")
    selected_sha = _sha256(
        selection.get("selected_checkpoint_sha256"), field="selection.selected_checkpoint_sha256"
    )
    if selected_sha != selected["checkpoint_sha256"]:
        raise EvidenceError("Selected checkpoint SHA-256 differs from the rank-1 entry")
    selected_path = str(selection.get("selected_checkpoint_path", ""))
    if (
        not selected_path
        or Path(selected_path).resolve() != Path(selected["checkpoint_path"]).resolve()
    ):
        raise EvidenceError("Selected checkpoint path differs from the rank-1 entry")
    if Path(selected["metrics_source"]).resolve() != metrics_path.resolve():
        raise EvidenceError("Supplied metrics JSON is not the rank-1 candidate metrics source")
    expected_selected_metrics = {
        "whole_pancreas_dice": metrics_summary["segmentation"]["whole_pancreas_dice"]["mean"],
        "lesion_dice": metrics_summary["segmentation"]["lesion_dice"]["mean"],
        "macro_f1": metrics_summary["classification"]["macro_f1"],
    }
    for name, expected_value in expected_selected_metrics.items():
        if not _close(selected["metrics"][name], expected_value):
            raise EvidenceError(f"Rank-1 selection metric {name} disagrees with metrics JSON")

    final_entry = next(item for item in ranking if item["candidate"] == "checkpoint_final")
    if final_entry["checkpoint_sha256"] != activation_source_sha256:
        raise EvidenceError("Activation/rescue source SHA does not match checkpoint_final ranking")
    if (
        not activation_source_checkpoint
        or Path(final_entry["checkpoint_path"]).resolve()
        != Path(activation_source_checkpoint).resolve()
    ):
        raise EvidenceError("Activation source path does not match checkpoint_final ranking")
    if rescue_summary is not None:
        rescue_entry = next(item for item in ranking if item["candidate"] == RESCUE_CANDIDATE)
        if rescue_entry["checkpoint_sha256"] != rescue_summary["output_checkpoint_sha256"]:
            raise EvidenceError("Rescue output SHA does not match the rescue ranking entry")
        if (
            not rescue_summary["output_checkpoint"]
            or Path(rescue_entry["checkpoint_path"]).resolve()
            != Path(rescue_summary["output_checkpoint"]).resolve()
        ):
            raise EvidenceError("Rescue output path does not match the rescue ranking entry")
        if (
            not rescue_summary["source_checkpoint"]
            or Path(rescue_summary["source_checkpoint"]).resolve()
            != Path(final_entry["checkpoint_path"]).resolve()
        ):
            raise EvidenceError("Rescue source path does not match checkpoint_final ranking")

    return {
        "candidate": selected["candidate"],
        "checkpoint_filename": f"{selected['candidate']}.pth",
        "checkpoint_path": selected["checkpoint_path"],
        "checkpoint_sha256": selected_sha,
        "selection_score": selected_score,
        "rank": 1,
    }, ranking


def _validate_runtime(
    runtime: Mapping[str, Any],
    *,
    selected_candidate: str,
    expected_case_count: int,
    runtime_path: Path,
    metrics_path: Path,
    case_metrics_path: Path,
) -> dict[str, Any]:
    candidate_directory = metrics_path.expanduser().resolve().parent
    if candidate_directory.name != selected_candidate:
        raise EvidenceError("Selected metrics directory does not identify the selected candidate")
    if runtime_path.expanduser().resolve().parent != candidate_directory:
        raise EvidenceError("Runtime JSON is not in the selected candidate directory")
    if case_metrics_path.expanduser().resolve().parent != candidate_directory:
        raise EvidenceError("Case metrics CSV is not in the selected candidate directory")
    _require_equal(runtime.get("case_count"), expected_case_count, field="runtime.case_count")
    _require_equal(
        runtime.get("checkpoint"), f"{selected_candidate}.pth", field="runtime.checkpoint"
    )
    device = runtime.get("device")
    if device not in ("cuda", "cpu"):
        raise EvidenceError("runtime.device must be 'cuda' or 'cpu'")
    _require_equal(runtime.get("folds"), [0], field="runtime.folds")
    _require_equal(runtime.get("gaussian_enabled"), True, field="runtime.gaussian_enabled")
    _require_equal(runtime.get("tta_enabled"), True, field="runtime.tta_enabled")
    step = _number(runtime.get("tile_step_size"), field="runtime.tile_step_size")
    if not _close(step, 0.5):
        raise EvidenceError("runtime.tile_step_size must equal 0.5")
    total = _number(runtime.get("total_seconds"), field="runtime.total_seconds", minimum=0)
    mean = _number(
        runtime.get("mean_seconds_per_case"), field="runtime.mean_seconds_per_case", minimum=0
    )
    if total <= 0 or mean <= 0 or not _close(mean, total / expected_case_count):
        raise EvidenceError("Runtime total and mean per case are inconsistent")
    if device == "cuda":
        allocated: float | None = _number(
            runtime.get("peak_allocated_mib"), field="runtime.peak_allocated_mib", minimum=0
        )
        reserved: float | None = _number(
            runtime.get("peak_reserved_mib"), field="runtime.peak_reserved_mib", minimum=0
        )
        if reserved < allocated:
            raise EvidenceError("Runtime peak reserved memory is below peak allocated memory")
    else:
        if runtime.get("peak_allocated_mib") is not None:
            raise EvidenceError("CPU runtime peak_allocated_mib must be null")
        if runtime.get("peak_reserved_mib") is not None:
            raise EvidenceError("CPU runtime peak_reserved_mib must be null")
        allocated = None
        reserved = None
    return {
        "case_count": expected_case_count,
        "checkpoint": f"{selected_candidate}.pth",
        "device": device,
        "folds": [0],
        "tta_enabled": True,
        "gaussian_enabled": True,
        "tile_step_size": step,
        "total_seconds": total,
        "mean_seconds_per_case": mean,
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
        "checkpoint_binding": "selected_candidate_directory_and_checkpoint_filename",
    }


def _qualitative_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < 4:
        raise EvidenceError("At least four cases are required for qualitative selection")
    ordered = sorted(
        rows,
        key=lambda row: (row["lesion_dice"], row["whole_pancreas_dice"], row["case_id"]),
    )

    def selected_row(role: str, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "role": role,
            "case_id": row["case_id"],
            "whole_pancreas_dice": row["whole_pancreas_dice"],
            "lesion_dice": row["lesion_dice"],
            "reference_subtype": row["reference_subtype"],
            "predicted_subtype": row["predicted_subtype"],
            "lesion_reference_voxels": row["lesion_reference_voxels"],
            "lesion_predicted_voxels": row["lesion_predicted_voxels"],
        }

    weak_1 = selected_row("weak_1_lowest", ordered[0])
    weak_2 = selected_row("weak_2_second_lowest", ordered[1])
    strong_1 = selected_row("strong_1_highest", ordered[-1])
    strong_2 = selected_row("strong_2_second_highest", ordered[-2])
    return {
        "policy": {
            "primary_sort": "lesion_dice ascending",
            "tie_breakers": ["whole_pancreas_dice ascending", "case_id ascending"],
        },
        "weak_cases": [weak_1, weak_2],
        "strong_cases": [strong_1, strong_2],
        "figure_panel_order": [weak_1, weak_2, strong_2, strong_1],
    }


def _lesion_analysis(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    sizes = np.asarray([row["lesion_reference_voxels"] for row in rows], dtype=np.float64)
    dice = np.asarray([row["lesion_dice"] for row in rows], dtype=np.float64)
    result = spearmanr(sizes, dice, nan_policy="raise", alternative="two-sided")
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(rho) or not math.isfinite(p_value):
        raise EvidenceError("Lesion-size Spearman correlation is undefined")
    zero_overlap_rows = [row for row in rows if _close(row["lesion_dice"], 0.0)]
    empty_prediction_rows = [row for row in rows if row["lesion_predicted_voxels"] == 0]
    nonempty_zero_overlap_rows = [
        row for row in zero_overlap_rows if row["lesion_predicted_voxels"] > 0
    ]
    association = {
        "method": "spearman_rank_correlation",
        "alternative": "two-sided",
        "x": "lesion_reference_voxels",
        "y": "lesion_dice",
        "n": len(rows),
        "rho": rho,
        "p_value": p_value,
        "ties_present_x": len(set(sizes.tolist())) != len(sizes),
        "ties_present_y": len(set(dice.tolist())) != len(dice),
        "interpretation_scope": "exploratory_association_not_causal",
    }
    failures = {
        "lesion_zero_overlap_case_count": len(zero_overlap_rows),
        "lesion_empty_prediction_case_count": len(empty_prediction_rows),
        "lesion_nonempty_prediction_zero_overlap_case_count": len(nonempty_zero_overlap_rows),
        "lesion_zero_overlap_case_ids": [row["case_id"] for row in zero_overlap_rows],
        "lesion_empty_prediction_case_ids": [row["case_id"] for row in empty_prediction_rows],
        "lesion_nonempty_prediction_zero_overlap_case_ids": [
            row["case_id"] for row in nonempty_zero_overlap_rows
        ],
    }
    return association, failures


def summarize_final_evidence(
    *,
    selection_path: Path,
    metrics_path: Path,
    case_metrics_path: Path,
    runtime_path: Path,
    activation_audit_path: Path,
    rescue_audit_path: Path | None,
    selected_checkpoint_path: Path | None,
    expected_case_count: int = 36,
    expected_class_supports: tuple[int, int, int] = (9, 15, 12),
) -> dict[str, Any]:
    """Return a fail-closed, report-ready summary of canonical artifacts."""

    if expected_case_count < 4:
        raise EvidenceError("expected_case_count must be at least four")
    if len(expected_class_supports) != 3 or any(value < 0 for value in expected_class_supports):
        raise EvidenceError("expected_class_supports must contain three non-negative counts")

    selection, selection_artifact = _load_json(selection_path, role="selection")
    metrics, metrics_artifact = _load_json(metrics_path, role="metrics")
    runtime, runtime_artifact = _load_json(runtime_path, role="runtime")
    activation_payload, activation_artifact = _load_json(
        activation_audit_path, role="activation audit"
    )
    csv_rows, case_metrics_artifact = _read_case_csv(case_metrics_path)

    activation = _validate_activation(activation_payload)
    rescue_artifact: dict[str, Any] | None = None
    rescue_summary: dict[str, Any] | None = None
    if activation["activation_approved"]:
        if rescue_audit_path is None:
            raise EvidenceError("An affirmative activation audit requires --rescue-audit")
        rescue_payload, rescue_artifact = _load_json(rescue_audit_path, role="rescue audit")
        rescue_summary = _validate_rescue(
            rescue_payload,
            activation=activation,
            activation_artifact_sha256=activation_artifact["sha256"],
            expected_case_count=expected_case_count,
        )
    elif rescue_audit_path is not None:
        raise EvidenceError("A negative activation audit must not be paired with a rescue audit")

    metrics_summary, normalized_rows = _validate_metrics(
        metrics,
        csv_rows,
        expected_case_count=expected_case_count,
        expected_supports=expected_class_supports,
    )
    selected, ranking = _validate_selection(
        selection,
        metrics_path=metrics_path,
        metrics_summary=metrics_summary,
        activation_approved=activation["activation_approved"],
        activation_source_checkpoint=activation["source_checkpoint"],
        activation_source_sha256=activation["source_checkpoint_sha256"],
        rescue_summary=rescue_summary,
        expected_case_count=expected_case_count,
        expected_supports=expected_class_supports,
        expected_case_ids=[row["case_id"] for row in normalized_rows],
    )
    runtime_summary = _validate_runtime(
        runtime,
        selected_candidate=selected["candidate"],
        expected_case_count=expected_case_count,
        runtime_path=runtime_path,
        metrics_path=metrics_path,
        case_metrics_path=case_metrics_path,
    )

    checkpoint_artifact: dict[str, Any] | None = None
    if selected_checkpoint_path is not None:
        checkpoint_artifact = _artifact(selected_checkpoint_path)
        if Path(checkpoint_artifact["path"]).name != selected["checkpoint_filename"]:
            raise EvidenceError("Supplied selected checkpoint has the wrong filename")
        if checkpoint_artifact["sha256"] != selected["checkpoint_sha256"]:
            raise EvidenceError("Supplied selected checkpoint SHA-256 differs from selection")
        if selected["checkpoint_path"] and (
            Path(checkpoint_artifact["path"]) != Path(selected["checkpoint_path"]).resolve()
        ):
            raise EvidenceError("Supplied selected checkpoint path differs from selection")

    qualitative = _qualitative_selection(normalized_rows)
    lesion_association, lesion_failures = _lesion_analysis(normalized_rows)
    case_ids_sha256 = _case_ids_sha256([row["case_id"] for row in normalized_rows])
    if (
        rescue_summary is not None
        and rescue_summary["validation_case_ids_sha256"] != case_ids_sha256
    ):
        raise EvidenceError("Rescue validation split IDs do not match the evaluated case IDs")
    artifacts: dict[str, Any] = {
        "selection": selection_artifact,
        "metrics": metrics_artifact,
        "case_metrics": case_metrics_artifact,
        "runtime": runtime_artifact,
        "activation_audit": activation_artifact,
    }
    if rescue_artifact is not None:
        artifacts["rescue_audit"] = rescue_artifact
    if checkpoint_artifact is not None:
        artifacts["selected_checkpoint"] = checkpoint_artifact

    targets = {
        "whole_pancreas_dice": {
            "threshold": 0.90,
            "value": metrics_summary["segmentation"]["whole_pancreas_dice"]["mean"],
        },
        "lesion_dice": {
            "threshold": 0.27,
            "value": metrics_summary["segmentation"]["lesion_dice"]["mean"],
        },
        "macro_f1": {
            "threshold": 0.60,
            "value": metrics_summary["classification"]["macro_f1"],
        },
    }
    for item in targets.values():
        item["met"] = item["value"] >= item["threshold"]

    return {
        "schema_version": 1,
        "summary_policy": {
            "case_aggregation": "unweighted",
            "selected_candidate": "rank_1_equal_weight_whole_lesion_macro_f1",
            "checkpoint_loading": (
                "prohibited; every selection-referenced checkpoint is byte-hashed only"
            ),
            "checkpoint_epochs_included": False,
        },
        "software": {"scipy": scipy.__version__},
        "expected_validation_case_count": expected_case_count,
        "expected_class_supports": list(expected_class_supports),
        "case_ids_sha256": case_ids_sha256,
        "artifacts": artifacts,
        "activation": activation,
        "rescue": rescue_summary,
        "selected_checkpoint": {
            **selected,
            "file_hash_verified": True,
            "explicit_selected_path_verified": checkpoint_artifact is not None,
        },
        "checkpoint_comparison": ranking,
        "validation": metrics_summary,
        "undergraduate_targets": targets,
        "qualitative_selection": qualitative,
        "lesion_size_association": lesion_association,
        "lesion_failure_counts": lesion_failures,
        "validation_runtime": runtime_summary,
    }


def _atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, protected: Sequence[Path]
) -> None:
    destination = path.expanduser().resolve()
    protected_paths = {item.expanduser().resolve() for item in protected}
    if destination in protected_paths:
        raise EvidenceError("--output must not overwrite an input evidence artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--case-metrics", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--activation-audit", required=True, type=Path)
    parser.add_argument(
        "--rescue-audit",
        type=Path,
        help="Required only when the train-only activation audit is affirmative",
    )
    parser.add_argument(
        "--selected-checkpoint",
        type=Path,
        help=(
            "Optional explicit rank-1 path identity check; all ranking checkpoints "
            "are already stream-hashed and never deserialized"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-case-count", type=int, default=36)
    parser.add_argument(
        "--expected-class-supports",
        type=int,
        nargs=3,
        metavar=("CLASS_0", "CLASS_1", "CLASS_2"),
        default=(9, 15, 12),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = summarize_final_evidence(
            selection_path=args.selection,
            metrics_path=args.metrics,
            case_metrics_path=args.case_metrics,
            runtime_path=args.runtime,
            activation_audit_path=args.activation_audit,
            rescue_audit_path=args.rescue_audit,
            selected_checkpoint_path=args.selected_checkpoint,
            expected_case_count=args.expected_case_count,
            expected_class_supports=tuple(args.expected_class_supports),
        )
        protected = [
            args.selection,
            args.metrics,
            args.case_metrics,
            args.runtime,
            args.activation_audit,
        ]
        if args.rescue_audit is not None:
            protected.append(args.rescue_audit)
        if args.selected_checkpoint is not None:
            protected.append(args.selected_checkpoint)
        for candidate in summary["checkpoint_comparison"]:
            protected.extend(
                [Path(candidate["metrics_source"]), Path(candidate["checkpoint_path"])]
            )
        _atomic_write_json(args.output, summary, protected=protected)
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote final evidence summary: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
