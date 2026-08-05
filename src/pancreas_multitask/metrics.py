"""Independent metrics for pancreas segmentation and subtype classification.

The functions in this module deliberately have no PyTorch, scikit-learn, or
nnU-Net dependency.  This keeps final evaluation independent from the training
implementation and makes every convention explicit:

* whole-pancreas foreground is ``label > 0``;
* lesion foreground is ``label == 2``;
* an empty prediction and empty reference receive Dice ``1.0`` by default;
* a one-sided empty mask receives Dice ``0.0``; and
* classification F1 is one-vs-rest over the fixed labels ``(0, 1, 2)`` with
  an explicit zero-division value of ``0.0``.

All public results are ordinary Python containers so they can be serialized to
JSON without a custom encoder.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

SEGMENTATION_LABELS: tuple[int, ...] = (0, 1, 2)
CLASSIFICATION_LABELS: tuple[int, ...] = (0, 1, 2)
DEFAULT_EMPTY_EMPTY_DICE = 1.0
DEFAULT_ZERO_DIVISION = 0.0


class MetricInputError(ValueError):
    """Raised when metric inputs violate the declared evaluation contract."""


def _validate_unit_interval(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise MetricInputError(f"{name} must be a finite value in [0, 1], got {value!r}")
    return result


def validate_label_array(
    values: Any,
    *,
    name: str,
    allowed_labels: Iterable[int] = SEGMENTATION_LABELS,
    integer_tolerance: float = 0.0,
) -> np.ndarray:
    """Return an integer label array after strict numeric and value checks.

    Floating-point arrays are accepted only when every value is within
    ``integer_tolerance`` of an integer.  Evaluation should normally use the
    corrected prepared labels and leave the tolerance at zero; the CLI exposes
    a tolerance for auditing the supplied near-integer source masks.
    """

    if not math.isfinite(integer_tolerance) or integer_tolerance < 0:
        raise MetricInputError("integer_tolerance must be finite and non-negative")

    array = np.asanyarray(values)
    if array.ndim < 1:
        raise MetricInputError(f"{name} must have at least one dimension, got {array.shape}")
    if array.size == 0:
        raise MetricInputError(f"{name} cannot be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise MetricInputError(f"{name} must be numeric, got dtype {array.dtype}")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise MetricInputError(f"{name} cannot be complex-valued")
    if not np.isfinite(array).all():
        raise MetricInputError(f"{name} contains NaN or infinity")

    rounded = np.rint(array)
    maximum_error = float(np.max(np.abs(array - rounded), initial=0.0))
    if maximum_error > integer_tolerance:
        raise MetricInputError(
            f"{name} contains non-integer values; maximum rounding error "
            f"{maximum_error:g} exceeds tolerance {integer_tolerance:g}"
        )

    integer_array = rounded.astype(np.int64, copy=False)
    allowed = tuple(int(label) for label in allowed_labels)
    if len(set(allowed)) != len(allowed) or not allowed:
        raise MetricInputError("allowed_labels must contain unique integer labels")
    observed = {int(value) for value in np.unique(integer_array).tolist()}
    invalid = sorted(observed - set(allowed))
    if invalid:
        raise MetricInputError(
            f"{name} contains unsupported labels {invalid}; allowed labels are {list(allowed)}"
        )
    return integer_array


def binary_dice(
    prediction: Any,
    reference: Any,
    *,
    empty_empty: float = DEFAULT_EMPTY_EMPTY_DICE,
) -> float:
    """Compute binary Dice with an explicit empty-empty convention.

    A one-sided empty input naturally returns zero because its intersection is
    empty while the denominator is positive.
    """

    empty_value = _validate_unit_interval(empty_empty, name="empty_empty")
    predicted = np.asanyarray(prediction, dtype=bool)
    target = np.asanyarray(reference, dtype=bool)
    if predicted.shape != target.shape:
        raise MetricInputError(
            f"Prediction/reference shape mismatch: {predicted.shape} versus {target.shape}"
        )
    if predicted.size == 0:
        raise MetricInputError("Dice inputs cannot be empty arrays")

    predicted_count = int(np.count_nonzero(predicted))
    reference_count = int(np.count_nonzero(target))
    denominator = predicted_count + reference_count
    if denominator == 0:
        return empty_value
    intersection = int(np.count_nonzero(predicted & target))
    return float((2.0 * intersection) / denominator)


def segmentation_case_metrics(
    prediction: Any,
    reference: Any,
    *,
    case_id: str | None = None,
    empty_empty: float = DEFAULT_EMPTY_EMPTY_DICE,
    integer_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compute whole-pancreas and lesion Dice for one 3D case."""

    predicted = validate_label_array(
        prediction,
        name="prediction",
        integer_tolerance=integer_tolerance,
    )
    target = validate_label_array(
        reference,
        name="reference",
        integer_tolerance=integer_tolerance,
    )
    if predicted.shape != target.shape:
        raise MetricInputError(
            f"Prediction/reference shape mismatch: {predicted.shape} versus {target.shape}"
        )

    predicted_whole = predicted > 0
    target_whole = target > 0
    predicted_lesion = predicted == 2
    target_lesion = target == 2

    whole_predicted_voxels = int(np.count_nonzero(predicted_whole))
    whole_reference_voxels = int(np.count_nonzero(target_whole))
    lesion_predicted_voxels = int(np.count_nonzero(predicted_lesion))
    lesion_reference_voxels = int(np.count_nonzero(target_lesion))

    result: dict[str, Any] = {
        "whole_pancreas_dice": binary_dice(predicted_whole, target_whole, empty_empty=empty_empty),
        "lesion_dice": binary_dice(predicted_lesion, target_lesion, empty_empty=empty_empty),
        "whole_pancreas_predicted_voxels": whole_predicted_voxels,
        "whole_pancreas_reference_voxels": whole_reference_voxels,
        "lesion_predicted_voxels": lesion_predicted_voxels,
        "lesion_reference_voxels": lesion_reference_voxels,
        "whole_pancreas_empty_empty": bool(
            whole_predicted_voxels == 0 and whole_reference_voxels == 0
        ),
        "lesion_empty_empty": bool(lesion_predicted_voxels == 0 and lesion_reference_voxels == 0),
    }
    if case_id is not None:
        result["case_id"] = str(case_id)
    return result


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    if samples == 0:
        return None
    if samples < 0:
        raise MetricInputError("bootstrap_samples must be non-negative")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise MetricInputError("confidence must be finite and strictly between 0 and 1")

    sample_indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence_value) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "method": "case_bootstrap_percentile_of_mean",
        "confidence": confidence_value,
        "samples": int(samples),
        "lower": float(lower),
        "upper": float(upper),
    }


def summarize_values(
    values: Sequence[float] | np.ndarray,
    *,
    bootstrap_samples: int = 0,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict[str, Any]:
    """Summarize case-level values with mean/std and an optional bootstrap CI."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise MetricInputError("Summary values must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise MetricInputError("Summary values contain NaN or infinity")
    if bootstrap_samples < 0:
        raise MetricInputError("bootstrap_samples must be non-negative")

    result: dict[str, Any] = {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }
    interval = _bootstrap_mean_interval(
        array,
        samples=int(bootstrap_samples),
        confidence=confidence,
        rng=np.random.default_rng(seed),
    )
    if interval is not None:
        result["bootstrap_ci"] = interval
    return result


def aggregate_segmentation_metrics(
    cases: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 0,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict[str, Any]:
    """Aggregate case dictionaries returned by :func:`segmentation_case_metrics`."""

    if not cases:
        raise MetricInputError("At least one segmentation case is required")
    try:
        whole = np.asarray([row["whole_pancreas_dice"] for row in cases], dtype=np.float64)
        lesion = np.asarray([row["lesion_dice"] for row in cases], dtype=np.float64)
    except KeyError as exc:
        raise MetricInputError(f"Missing per-case metric key: {exc.args[0]}") from exc

    for name, array in (("whole_pancreas_dice", whole), ("lesion_dice", lesion)):
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise MetricInputError(f"{name} values must be finite and in [0, 1]")

    def empty_count(prefix: str, side: str) -> int:
        voxel_key = f"{prefix}_{side}_voxels"
        if not all(voxel_key in row for row in cases):
            return 0
        return sum(int(row[voxel_key]) == 0 for row in cases)

    def both_empty_count(prefix: str) -> int:
        flag_key = f"{prefix}_empty_empty"
        if all(flag_key in row for row in cases):
            return sum(bool(row[flag_key]) for row in cases)
        prediction_key = f"{prefix}_predicted_voxels"
        reference_key = f"{prefix}_reference_voxels"
        if all(prediction_key in row and reference_key in row for row in cases):
            return sum(
                int(row[prediction_key]) == 0 and int(row[reference_key]) == 0 for row in cases
            )
        return 0

    return {
        "case_count": len(cases),
        "whole_pancreas_dice": summarize_values(
            whole,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        ),
        "lesion_dice": summarize_values(
            lesion,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed + 1,
        ),
        "empty_cases": {
            "whole_pancreas_prediction_empty": empty_count("whole_pancreas", "predicted"),
            "whole_pancreas_reference_empty": empty_count("whole_pancreas", "reference"),
            "whole_pancreas_both_empty": both_empty_count("whole_pancreas"),
            "lesion_prediction_empty": empty_count("lesion", "predicted"),
            "lesion_reference_empty": empty_count("lesion", "reference"),
            "lesion_both_empty": both_empty_count("lesion"),
        },
    }


def _coerce_class_vector(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
    labels: Sequence[int],
) -> np.ndarray:
    array = np.asanyarray(values)
    if array.ndim != 1 or array.size == 0:
        raise MetricInputError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.issubdtype(array.dtype, np.number):
        raise MetricInputError(f"{name} must be numeric")
    if np.issubdtype(array.dtype, np.complexfloating) or not np.isfinite(array).all():
        raise MetricInputError(f"{name} must contain finite real values")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise MetricInputError(f"{name} must contain integer class labels")
    result = rounded.astype(np.int64, copy=False)
    unsupported = sorted({int(value) for value in np.unique(result)} - set(labels))
    if unsupported:
        raise MetricInputError(
            f"{name} contains unsupported labels {unsupported}; expected {list(labels)}"
        )
    return result


def confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    labels: Sequence[int] = CLASSIFICATION_LABELS,
) -> np.ndarray:
    """Return a confusion matrix whose rows are truth and columns prediction."""

    fixed_labels = tuple(int(label) for label in labels)
    if not fixed_labels or len(set(fixed_labels)) != len(fixed_labels):
        raise MetricInputError("labels must be a non-empty sequence of unique integers")
    truth = _coerce_class_vector(y_true, name="y_true", labels=fixed_labels)
    prediction = _coerce_class_vector(y_pred, name="y_pred", labels=fixed_labels)
    if truth.shape != prediction.shape:
        raise MetricInputError(
            f"y_true/y_pred length mismatch: {truth.size} versus {prediction.size}"
        )

    index = {label: position for position, label in enumerate(fixed_labels)}
    matrix = np.zeros((len(fixed_labels), len(fixed_labels)), dtype=np.int64)
    for target, predicted in zip(truth.tolist(), prediction.tolist(), strict=True):
        matrix[index[int(target)], index[int(predicted)]] += 1
    return matrix


def _classification_from_matrix(
    matrix: np.ndarray,
    *,
    labels: tuple[int, ...],
    zero_division: float,
) -> dict[str, Any]:
    total = int(matrix.sum())
    if total == 0:
        raise MetricInputError("Classification confusion matrix cannot be empty")

    per_class: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        true_negative = total - true_positive - false_positive - false_negative
        support = true_positive + false_negative
        predicted_count = true_positive + false_positive

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else zero_division
        )
        recall = true_positive / recall_denominator if recall_denominator else zero_division
        f1_denominator = precision + recall
        f1 = 2.0 * precision * recall / f1_denominator if f1_denominator else zero_division
        per_class.append(
            {
                "label": label,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    return {
        "case_count": total,
        "labels": list(labels),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {"rows": "reference", "columns": "prediction"},
        "accuracy": float(np.trace(matrix) / total),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "zero_division": zero_division,
        "per_class": per_class,
    }


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    labels: Sequence[int] = CLASSIFICATION_LABELS,
    zero_division: float = DEFAULT_ZERO_DIVISION,
    bootstrap_samples: int = 0,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict[str, Any]:
    """Compute per-class and macro classification metrics over fixed labels."""

    fixed_labels = tuple(int(label) for label in labels)
    zero_value = _validate_unit_interval(zero_division, name="zero_division")
    truth = _coerce_class_vector(y_true, name="y_true", labels=fixed_labels)
    prediction = _coerce_class_vector(y_pred, name="y_pred", labels=fixed_labels)
    if truth.shape != prediction.shape:
        raise MetricInputError(
            f"y_true/y_pred length mismatch: {truth.size} versus {prediction.size}"
        )
    result = _classification_from_matrix(
        confusion_matrix(truth, prediction, labels=fixed_labels),
        labels=fixed_labels,
        zero_division=zero_value,
    )

    if bootstrap_samples < 0:
        raise MetricInputError("bootstrap_samples must be non-negative")
    if bootstrap_samples:
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise MetricInputError("confidence must be finite and strictly between 0 and 1")
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, truth.size, size=(bootstrap_samples, truth.size))
        macro_f1_values = np.empty(bootstrap_samples, dtype=np.float64)
        accuracy_values = np.empty(bootstrap_samples, dtype=np.float64)
        for bootstrap_index, case_indices in enumerate(indices):
            bootstrap_matrix = confusion_matrix(
                truth[case_indices], prediction[case_indices], labels=fixed_labels
            )
            bootstrap_result = _classification_from_matrix(
                bootstrap_matrix,
                labels=fixed_labels,
                zero_division=zero_value,
            )
            macro_f1_values[bootstrap_index] = bootstrap_result["macro_f1"]
            accuracy_values[bootstrap_index] = bootstrap_result["accuracy"]
        alpha = (1.0 - confidence) / 2.0

        def interval(values: np.ndarray) -> dict[str, Any]:
            lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
            return {
                "method": "case_bootstrap_percentile",
                "confidence": float(confidence),
                "samples": int(bootstrap_samples),
                "lower": float(lower),
                "upper": float(upper),
            }

        result["bootstrap_ci"] = {
            "macro_f1": interval(macro_f1_values),
            "accuracy": interval(accuracy_values),
        }
    return result


__all__ = [
    "CLASSIFICATION_LABELS",
    "DEFAULT_EMPTY_EMPTY_DICE",
    "DEFAULT_ZERO_DIVISION",
    "SEGMENTATION_LABELS",
    "MetricInputError",
    "aggregate_segmentation_metrics",
    "binary_dice",
    "classification_metrics",
    "confusion_matrix",
    "segmentation_case_metrics",
    "summarize_values",
    "validate_label_array",
]
