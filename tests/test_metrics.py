from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Import the lightweight module directly so metric tests do not require the
# training-only PyTorch/nnU-Net dependencies to be imported by package __init__.
METRICS_SOURCE = Path(__file__).resolve().parents[1] / "src" / "pancreas_multitask"
sys.path.insert(0, str(METRICS_SOURCE))

import metrics


def test_binary_dice_overlap_and_empty_policies() -> None:
    prediction = np.array([1, 1, 0, 0], dtype=bool)
    reference = np.array([1, 0, 1, 0], dtype=bool)
    assert metrics.binary_dice(prediction, reference) == pytest.approx(0.5)

    empty = np.zeros(4, dtype=bool)
    assert metrics.binary_dice(empty, empty) == 1.0
    assert metrics.binary_dice(empty, empty, empty_empty=0.0) == 0.0
    assert metrics.binary_dice(prediction, empty) == 0.0


def test_segmentation_definitions_keep_organ_overlap_when_lesion_class_is_wrong() -> None:
    reference = np.array([[[0, 1, 2, 2]]], dtype=np.uint8)
    prediction = np.array([[[0, 1, 1, 1]]], dtype=np.uint8)

    result = metrics.segmentation_case_metrics(prediction, reference, case_id="case_a")

    assert result["case_id"] == "case_a"
    assert result["whole_pancreas_dice"] == 1.0
    assert result["lesion_dice"] == 0.0
    assert result["whole_pancreas_predicted_voxels"] == 3
    assert result["lesion_reference_voxels"] == 2
    assert result["lesion_predicted_voxels"] == 0


def test_segmentation_lesion_empty_empty_is_recorded() -> None:
    reference = np.array([[[0, 1]]], dtype=np.uint8)
    prediction = reference.copy()

    result = metrics.segmentation_case_metrics(prediction, reference)

    assert result["whole_pancreas_dice"] == 1.0
    assert result["lesion_dice"] == 1.0
    assert result["lesion_empty_empty"] is True
    aggregate = metrics.aggregate_segmentation_metrics([result])
    assert aggregate["empty_cases"]["lesion_both_empty"] == 1


def test_segmentation_rejects_invalid_labels_and_shape_mismatch() -> None:
    valid = np.zeros((2, 2, 2), dtype=np.uint8)
    invalid = valid.copy()
    invalid[0, 0, 0] = 3
    with pytest.raises(metrics.MetricInputError, match="unsupported labels"):
        metrics.segmentation_case_metrics(invalid, valid)

    with pytest.raises(metrics.MetricInputError, match="shape mismatch"):
        metrics.segmentation_case_metrics(valid, np.zeros((2, 2, 1)))


def test_label_validation_tolerance_is_explicit() -> None:
    near_integer = np.array([0.0, 1.0000153, 2.0], dtype=np.float32)
    with pytest.raises(metrics.MetricInputError, match="maximum rounding error"):
        metrics.validate_label_array(near_integer, name="mask")
    repaired = metrics.validate_label_array(near_integer, name="mask", integer_tolerance=1e-3)
    np.testing.assert_array_equal(repaired, [0, 1, 2])


def test_classification_metrics_confusion_matrix_and_macro_f1() -> None:
    truth = [0, 0, 1, 1, 2, 2]
    prediction = [0, 1, 1, 1, 0, 2]

    result = metrics.classification_metrics(truth, prediction)

    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert result["confusion_matrix_axes"] == {
        "rows": "reference",
        "columns": "prediction",
    }
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert [row["f1"] for row in result["per_class"]] == pytest.approx([0.5, 0.8, 2 / 3])
    assert result["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert sum(row["support"] for row in result["per_class"]) == 6


def test_classification_keeps_all_three_classes_when_classes_are_absent() -> None:
    result = metrics.classification_metrics([0, 0], [0, 0])

    assert result["per_class"][0]["f1"] == 1.0
    assert result["per_class"][1]["f1"] == 0.0
    assert result["per_class"][2]["f1"] == 0.0
    assert result["macro_f1"] == pytest.approx(1 / 3)


def test_bootstrap_summaries_are_seeded_and_finite() -> None:
    cases = [
        {"whole_pancreas_dice": 0.8, "lesion_dice": 0.2},
        {"whole_pancreas_dice": 0.9, "lesion_dice": 0.4},
        {"whole_pancreas_dice": 1.0, "lesion_dice": 0.6},
    ]
    first = metrics.aggregate_segmentation_metrics(cases, bootstrap_samples=100, seed=7)
    second = metrics.aggregate_segmentation_metrics(cases, bootstrap_samples=100, seed=7)

    assert first == second
    assert first["whole_pancreas_dice"]["mean"] == pytest.approx(0.9)
    interval = first["whole_pancreas_dice"]["bootstrap_ci"]
    assert 0.8 <= interval["lower"] <= interval["upper"] <= 1.0

    classification = metrics.classification_metrics(
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 1, 0, 2],
        bootstrap_samples=50,
        seed=7,
    )
    assert set(classification["bootstrap_ci"]) == {"macro_f1", "accuracy"}


def test_classification_rejects_length_mismatch_and_out_of_range_labels() -> None:
    with pytest.raises(metrics.MetricInputError, match="length mismatch"):
        metrics.classification_metrics([0, 1], [0])
    with pytest.raises(metrics.MetricInputError, match="unsupported labels"):
        metrics.classification_metrics([0, 3], [0, 1])
