"""Train-only cross-fitted class-offset calibration for case classifiers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from pancreas_multitask.case_classifier_selection import canonical_content_order


def _metrics(references: np.ndarray, predictions: np.ndarray) -> tuple[float, list[float]]:
    macro_f1 = float(
        f1_score(
            references,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )
    recalls = [
        float(value)
        for value in recall_score(
            references,
            predictions,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
    ]
    return macro_f1, recalls


def _offset_candidates(lock: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    grid = lock["offset_grid"]
    if float(grid["class_1_offset"]) != 0:
        raise ValueError("Class 1 must remain the zero-offset identifiability reference")
    candidates = tuple(
        np.asarray((float(class_0), 0.0, float(class_2)), dtype=np.float64)
        for class_0 in grid["class_0_offsets"]
        for class_2 in grid["class_2_offsets"]
    )
    if len(candidates) != int(grid["candidate_count"]):
        raise ValueError("Decision offset grid count does not match its lock")
    return candidates


def select_offsets(
    log_scores: np.ndarray,
    references: np.ndarray,
    lock: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one offset using the exact deterministic v4 tie rule."""

    scores = np.asarray(log_scores, dtype=np.float64)
    targets = np.asarray(references, dtype=np.int64)
    if scores.shape != (targets.size, 3) or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a finite (cases, 3) matrix")
    evaluated: list[tuple[tuple[float, float, float, float, float], np.ndarray, dict]] = []
    for offset in _offset_candidates(lock):
        predictions = np.argmax(scores + offset[None, :], axis=1)
        macro_f1, recalls = _metrics(targets, predictions)
        key = (
            -macro_f1,
            -min(recalls),
            float(np.abs(offset).sum()),
            float(offset[0]),
            float(offset[2]),
        )
        evaluated.append(
            (
                key,
                offset,
                {
                    "offset": [float(value) for value in offset],
                    "macro_f1": macro_f1,
                    "per_class_recall": recalls,
                    "minimum_per_class_recall": min(recalls),
                },
            )
        )
    _, selected, audit = min(evaluated, key=lambda item: item[0])
    return selected.copy(), audit


def _selected_candidate_row(selection_audit: Mapping[str, Any]) -> Mapping[str, Any]:
    selected_id = selection_audit["selected_candidate_id"]
    matches = [
        row for row in selection_audit["candidate_results"] if row["candidate_id"] == selected_id
    ]
    if len(matches) != 1:
        raise ValueError("Selection audit does not contain one selected candidate row")
    return matches[0]


def _cross_fit_contract(decision_lock: Mapping[str, Any]) -> Mapping[str, Any]:
    """Support the superseded v4 diagnostic and the eligible v5 neural lock."""

    if "cross_fitted_evaluation" in decision_lock:
        return decision_lock["cross_fitted_evaluation"]
    return decision_lock["cross_fitted_calibration_evaluation"]


def _activation_contract(decision_lock: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    if "activation" in decision_lock:
        return decision_lock["activation"], (
            "minimum_mean_macro_f1_gain_over_plain_selected_neural_head"
        )
    return decision_lock["calibration_activation"], (
        "minimum_mean_macro_f1_gain_over_plain_selected_v3"
    )


def _exact_float64_content_order(
    scores: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    """Canonicalize eligible v5 scores without float32 loss or stable-sort ties."""

    values = np.asarray(scores, dtype="<f8")
    targets = np.asarray(references, dtype="<i8")
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("V5 calibration scores and references are not aligned")
    keys: list[bytes] = []
    for row, target in zip(values, targets, strict=True):
        digest = hashlib.sha256()
        digest.update(row.tobytes(order="C"))
        digest.update(target.tobytes())
        keys.append(digest.digest())
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate v5 calibration digests would make input order a split key")
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)


def _log_softmax_float64(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("V5 raw logits must be a finite (cases, 3) matrix")
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def evaluate_cross_fitted_offsets(
    selection_audit: Mapping[str, Any],
    decision_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-fit offsets on selected-candidate OOF scores, never official data."""

    selected = _selected_candidate_row(selection_audit)
    repeat_rows = selected["oof_predictions"]
    cross_fit = _cross_fit_contract(decision_lock)
    is_v5 = "cross_fitted_evaluation" in decision_lock
    if is_v5:
        score_contract = decision_lock.get("score_contract", {})
        if (
            score_contract.get("class_order") != [0, 1, 2]
            or score_contract.get("input") != "selected_neural_head_three_logits"
            or score_contract.get("normalization") != "float64_log_softmax"
            or score_contract.get("plain_rule_must_equal_argmax_raw_logits") is not True
        ):
            raise ValueError("V5 decision lock score contract is invalid")
    locked_seeds = [int(value) for value in cross_fit["repeat_seeds"]]
    if [int(row["seed"]) for row in repeat_rows] != locked_seeds:
        raise ValueError("Selected OOF repeats do not match the calibration lock")
    fold_count = int(cross_fit["folds"])
    repeat_audits: list[dict[str, Any]] = []
    plain_scores: list[float] = []
    calibrated_scores: list[float] = []
    plain_minimum_recalls: list[float] = []
    calibrated_minimum_recalls: list[float] = []
    repeat_score_matrices: list[np.ndarray] = []
    first_case_ids: tuple[str, ...] | None = None
    first_references: np.ndarray | None = None

    for repeat_index, (repeat, seed) in enumerate(zip(repeat_rows, locked_seeds, strict=True)):
        prediction_rows = repeat["predictions"]
        case_ids = tuple(str(row["case_id"]) for row in prediction_rows)
        references = np.asarray([int(row["reference"]) for row in prediction_rows], dtype=np.int64)
        recorded_predictions = np.asarray(
            [int(row["prediction"]) for row in prediction_rows], dtype=np.int64
        )
        log_scores = np.asarray([row["log_scores"] for row in prediction_rows], dtype=np.float64)
        if is_v5:
            raw_logits = np.asarray(
                [row.get("logits") for row in prediction_rows], dtype=np.float64
            )
            recomputed_log_scores = _log_softmax_float64(raw_logits)
            if not np.array_equal(log_scores, recomputed_log_scores):
                raise ValueError("Recorded v5 log-scores differ from raw-logit normalization")
        if (
            len(case_ids) != len(set(case_ids))
            or log_scores.shape != (references.size, 3)
            or not np.isfinite(log_scores).all()
            or not set(references.tolist()).issubset({0, 1, 2})
            or not np.array_equal(log_scores.argmax(axis=1), recorded_predictions)
        ):
            raise ValueError("Selected OOF rows violate the fixed three-class score contract")
        if first_case_ids is None:
            first_case_ids = case_ids
            first_references = references
        elif case_ids != first_case_ids or not np.array_equal(references, first_references):
            raise ValueError("OOF repeat rows are not aligned")
        repeat_score_matrices.append(log_scores)

        order = (
            _exact_float64_content_order(log_scores, references)
            if is_v5
            else canonical_content_order(log_scores, references)
        )
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(order.size)
        scores_ordered = log_scores[order]
        references_ordered = references[order]
        calibrated_ordered = np.full(references.shape, -1, dtype=np.int64)
        calibration_fold_ordered = np.full(references.shape, -1, dtype=np.int64)
        folds: list[dict[str, Any]] = []
        splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
        for fold_index, (fit_indices, held_indices) in enumerate(
            splitter.split(scores_ordered, references_ordered)
        ):
            offsets, fit_audit = select_offsets(
                scores_ordered[fit_indices],
                references_ordered[fit_indices],
                decision_lock,
            )
            held_predictions = np.argmax(scores_ordered[held_indices] + offsets[None, :], axis=1)
            calibrated_ordered[held_indices] = held_predictions
            calibration_fold_ordered[held_indices] = fold_index
            held_macro_f1, held_recalls = _metrics(
                references_ordered[held_indices], held_predictions
            )
            folds.append(
                {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "fold_index": fold_index,
                    "offset_fit_case_count": int(fit_indices.size),
                    "held_out_case_count": int(held_indices.size),
                    "selected_offset": [float(value) for value in offsets],
                    "offset_fit_metrics": fit_audit,
                    "held_out_macro_f1": held_macro_f1,
                    "held_out_per_class_recall": held_recalls,
                }
            )
        if np.any(calibrated_ordered < 0) or np.any(calibration_fold_ordered < 0):
            raise RuntimeError("Calibration cross-fit left cases unpredicted")
        calibrated = calibrated_ordered[inverse_order]
        calibration_fold = calibration_fold_ordered[inverse_order]
        plain = log_scores.argmax(axis=1)
        plain_macro_f1, plain_recalls = _metrics(references, plain)
        calibrated_macro_f1, calibrated_recalls = _metrics(references, calibrated)
        plain_scores.append(plain_macro_f1)
        calibrated_scores.append(calibrated_macro_f1)
        plain_minimum_recalls.append(min(plain_recalls))
        calibrated_minimum_recalls.append(min(calibrated_recalls))
        repeat_audits.append(
            {
                "repeat_index": repeat_index,
                "seed": seed,
                "plain_macro_f1": plain_macro_f1,
                "plain_per_class_recall": plain_recalls,
                "calibrated_macro_f1": calibrated_macro_f1,
                "calibrated_per_class_recall": calibrated_recalls,
                "folds": folds,
                "predictions": [
                    {
                        "case_id": case_id,
                        "reference": int(reference),
                        "plain_prediction": int(plain_prediction),
                        "cross_fitted_calibrated_prediction": int(calibrated_prediction),
                        "calibration_fold_index": int(fold_index),
                    }
                    for case_id, reference, plain_prediction, calibrated_prediction, fold_index in zip(
                        case_ids,
                        references,
                        plain,
                        calibrated,
                        calibration_fold,
                        strict=True,
                    )
                ],
            }
        )

    mean_plain = float(np.mean(plain_scores))
    mean_calibrated = float(np.mean(calibrated_scores))
    minimum_plain_recall = float(np.min(plain_minimum_recalls))
    minimum_calibrated_recall = float(np.min(calibrated_minimum_recalls))
    activation, minimum_gain_key = _activation_contract(decision_lock)
    gain_condition = mean_calibrated - mean_plain >= float(activation[minimum_gain_key])
    recall_condition = minimum_calibrated_recall >= minimum_plain_recall - float(
        activation["maximum_allowed_drop_in_minimum_repeat_per_class_recall"]
    )
    activated = bool(gain_condition and recall_condition)
    if first_references is None:
        raise RuntimeError("Calibration received no repeats")
    if activated:
        mean_oof_log_scores = np.mean(np.stack(repeat_score_matrices), axis=0)
        final_offsets, final_fit_metrics = select_offsets(
            mean_oof_log_scores,
            first_references,
            decision_lock,
        )
    else:
        final_offsets = np.zeros(3, dtype=np.float64)
        final_fit_metrics = {
            "offset": [0.0, 0.0, 0.0],
            "reason": "cross_fitted_activation_conditions_not_both_met",
        }

    return {
        "schema_version": 1,
        "scope": "selected_base_candidate_train_only_oof_scores",
        "selected_candidate_id": selected["candidate_id"],
        "repeat_audits": repeat_audits,
        "plain_repeat_macro_f1": plain_scores,
        "calibrated_repeat_macro_f1": calibrated_scores,
        "mean_plain_repeat_macro_f1": mean_plain,
        "mean_cross_fitted_calibrated_repeat_macro_f1": mean_calibrated,
        "mean_macro_f1_gain": mean_calibrated - mean_plain,
        "minimum_plain_repeat_per_class_recall": minimum_plain_recall,
        "minimum_calibrated_repeat_per_class_recall": minimum_calibrated_recall,
        "gain_condition_met": gain_condition,
        "recall_condition_met": recall_condition,
        "calibration_activated": activated,
        "final_offsets": [float(value) for value in final_offsets],
        "final_offset_fit_metrics": final_fit_metrics,
        "official_validation_used": False,
        "identifiers_used_for_calibration": False,
    }


def apply_class_offsets(log_scores: np.ndarray, offsets: Sequence[float]) -> np.ndarray:
    scores = np.asarray(log_scores, dtype=np.float64)
    offset_array = np.asarray(offsets, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 3 or offset_array.shape != (3,):
        raise ValueError("Expected (cases, 3) scores and three offsets")
    return np.argmax(scores + offset_array[None, :], axis=1).astype(np.int64)


__all__ = [
    "apply_class_offsets",
    "evaluate_cross_fitted_offsets",
    "select_offsets",
]
