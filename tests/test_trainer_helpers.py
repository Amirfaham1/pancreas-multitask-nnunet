from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnunetv2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.trainer import (
    case_level_classification_metrics,
    inverse_frequency_class_weights,
    macro_f1_from_predictions,
    parse_class_from_case_id,
    resolve_case_labels,
    weighted_classification_loss,
)


def test_case_label_resolution_prefers_manifest_and_has_validated_fallback() -> None:
    assert parse_class_from_case_id("quiz_2_123") == 2
    assert resolve_case_labels(
        ["opaque_case", "quiz_1_456"],
        {"opaque_case": 0},
    ) == [0, 1]
    with pytest.raises(ValueError, match="Could not infer"):
        parse_class_from_case_id("quiz_unknown")
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        resolve_case_labels(["opaque_case"], {"opaque_case": 7})


def test_inverse_frequency_weights_match_fixed_training_split() -> None:
    labels = [0] * 62 + [1] * 106 + [2] * 84
    weights = inverse_frequency_class_weights(labels)
    np.testing.assert_allclose(
        weights,
        np.asarray([252 / (3 * 62), 252 / (3 * 106), 1.0]),
        rtol=1e-6,
    )


def test_inverse_frequency_weights_reject_missing_class() -> None:
    with pytest.raises(ValueError, match="Every classification class"):
        inverse_frequency_class_weights([0, 0, 1, 1])


def test_macro_f1_is_unweighted_across_three_classes() -> None:
    truth = np.asarray([0, 0, 1, 1, 2, 2])
    predicted = np.asarray([0, 1, 1, 1, 2, 0])
    macro_f1, per_class = macro_f1_from_predictions(truth, predicted)

    np.testing.assert_allclose(per_class, (0.5, 0.8, 2 / 3))
    assert macro_f1 == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_case_metrics_average_repeated_patch_logits_before_scoring() -> None:
    metrics = case_level_classification_metrics(
        ["case_a", "case_a", "case_b"],
        np.asarray(
            [
                [4.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
        np.asarray([0, 0, 2]),
    )

    assert metrics["case_ids"] == ["case_a", "case_b"]
    assert metrics["predictions"].tolist() == [0, 2]
    assert metrics["accuracy"] == 1.0


def test_case_metrics_reject_conflicting_targets() -> None:
    with pytest.raises(ValueError, match="Conflicting targets"):
        case_level_classification_metrics(
            ["same", "same"],
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.asarray([0, 1]),
        )


def test_classification_loss_downweights_patch_without_lesion() -> None:
    logits = torch.tensor(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        requires_grad=True,
    )
    class_targets = torch.tensor([0, 0])
    segmentation = torch.zeros((2, 1, 2, 2, 2), dtype=torch.long)
    segmentation[0, 0, 0, 0, 0] = 2

    loss, lesion_present = weighted_classification_loss(
        logits,
        class_targets,
        segmentation,
        class_weights=None,
        nonlesion_patch_weight=0.25,
        label_smoothing=0.0,
    )
    individual = torch.nn.functional.cross_entropy(
        logits,
        class_targets,
        reduction="none",
    )
    expected = (individual[0] + 0.25 * individual[1]) / 1.25

    assert lesion_present.tolist() == [True, False]
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
