from __future__ import annotations

import numpy as np

from pancreas_multitask.decision_calibration import (
    apply_class_offsets,
    evaluate_cross_fitted_offsets,
    select_offsets,
)


def _decision_lock() -> dict:
    return {
        "offset_grid": {
            "class_1_offset": 0.0,
            "class_0_offsets": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "class_2_offsets": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "candidate_count": 25,
        },
        "cross_fitted_calibration_evaluation": {
            "folds": 5,
            "repeat_seeds": [11, 12, 13],
        },
        "calibration_activation": {
            "minimum_mean_macro_f1_gain_over_plain_selected_v3": 0.01,
            "maximum_allowed_drop_in_minimum_repeat_per_class_recall": 0.02,
        },
    }


def _selection_audit() -> dict:
    labels = np.repeat(np.arange(3), 10)
    score_rows = []
    for label in labels:
        scores = np.asarray([0.0, 1.0, 0.0])
        scores[label] += 0.6
        score_rows.append(scores)
    repeats = []
    for repeat_index, seed in enumerate((11, 12, 13)):
        repeats.append(
            {
                "repeat_index": repeat_index,
                "seed": seed,
                "predictions": [
                    {
                        "case_id": f"opaque_{index:03d}",
                        "reference": int(label),
                        "prediction": int(np.argmax(scores)),
                        "log_scores": scores.tolist(),
                    }
                    for index, (label, scores) in enumerate(zip(labels, score_rows, strict=True))
                ],
            }
        )
    return {
        "selected_candidate_id": "locked_candidate",
        "candidate_results": [
            {
                "candidate_id": "locked_candidate",
                "oof_predictions": repeats,
            }
        ],
    }


def test_offset_selection_uses_zero_for_already_perfect_scores() -> None:
    references = np.repeat(np.arange(3), 4)
    scores = np.full((12, 3), -2.0)
    scores[np.arange(12), references] = 2.0

    offsets, audit = select_offsets(scores, references, _decision_lock())

    assert np.array_equal(offsets, np.zeros(3))
    assert audit["macro_f1"] == 1.0


def test_cross_fitted_offsets_recover_minority_classes_without_identifiers() -> None:
    audit = evaluate_cross_fitted_offsets(_selection_audit(), _decision_lock())

    assert audit["calibration_activated"] is True
    assert audit["mean_macro_f1_gain"] > 0.01
    assert audit["official_validation_used"] is False
    assert audit["identifiers_used_for_calibration"] is False
    assert audit["final_offsets"][1] == 0.0


def test_apply_offsets_has_fixed_three_class_contract() -> None:
    scores = np.asarray([[0.2, 0.5, 0.1], [0.1, 0.4, 0.3]])
    predictions = apply_class_offsets(scores, (0.5, 0.0, 0.5))

    assert predictions.tolist() == [0, 2]
