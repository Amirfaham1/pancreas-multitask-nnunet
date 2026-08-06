from __future__ import annotations

import copy

import numpy as np
import pytest

from pancreas_multitask.case_classifier_selection import (
    CaseFeatureDataset,
    canonical_content_order,
    enumerate_locked_candidates,
    evaluate_locked_candidates,
    fit_selected_classifier,
    identifier_independent_dataset_sha256,
)


def _small_lock() -> dict:
    return {
        "schema_version": 2,
        "lock_status": (
            "frozen_before_inference_matched_encoder_feature_extraction_or_candidate_cv"
        ),
        "feature_extraction": {"feature_views": ["encoder", "encoder_plus_predicted_mask"]},
        "selection": {
            "splitter": "RepeatedStratifiedKFold",
            "folds": 2,
            "repeat_seeds": [13],
            "primary_metric": "mean_across_repeats_of_complete_oof_three_class_macro_f1",
            "secondary_metric": "lower_standard_deviation_of_repeat_oof_macro_f1",
            "tie_tolerance": 0.002,
            "tertiary_rule": "lower_candidate_id_lexicographic_order",
            "all_scaling_pca_and_fitting_inside_each_training_fold": True,
            "official_validation_evaluations_before_final_lock": 0,
        },
        "candidate_grid": [
            {
                "family": "balanced_multinomial_logistic_regression",
                "C": [0.1],
            },
            {"family": "balanced_linear_svm", "C": [0.1]},
            {
                "family": "balanced_rbf_svm",
                "pca_variance": [0.9],
                "C": [1.0],
                "gamma": "scale",
            },
            {
                "family": "balanced_extra_trees",
                "n_estimators": 600,
                "max_features": "sqrt",
                "min_samples_leaf": [2],
                "class_weight": "balanced",
            },
        ],
        "final_fit": {"random_seed": 13},
    }


def _dataset() -> CaseFeatureDataset:
    rng = np.random.default_rng(91)
    labels = np.repeat(np.arange(3), 10)
    encoder = rng.normal(size=(30, 8)).astype(np.float32)
    encoder[:, 0] += labels * 1.2
    combined = np.column_stack((encoder, labels + rng.normal(0, 0.2, 30))).astype(np.float32)
    return CaseFeatureDataset(
        tuple(f"opaque_{index:03d}" for index in range(30)),
        labels,
        {"encoder": encoder, "encoder_plus_predicted_mask": combined},
        {
            "encoder": tuple(f"encoder_feature_{index}" for index in range(8)),
            "encoder_plus_predicted_mask": tuple(
                [f"encoder_feature_{index}" for index in range(8)]
                + ["predicted_mask_numeric_feature"]
            ),
        },
    )


def test_locked_grid_expands_both_identifier_free_views() -> None:
    candidates = enumerate_locked_candidates(_small_lock())

    assert len(candidates) == 8
    assert {candidate.feature_view for candidate in candidates} == {
        "encoder",
        "encoder_plus_predicted_mask",
    }
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)


def test_dataset_rejects_identifier_like_feature_names() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="identifier-like"):
        CaseFeatureDataset(
            dataset.case_ids,
            dataset.labels,
            {"bad": np.ones((30, 1), dtype=np.float32)},
            {"bad": ("case_id_encoded",)},
        )


def test_content_order_and_dataset_hash_ignore_renaming_and_enumeration() -> None:
    dataset = _dataset()
    renamed = dataset.with_case_ids(tuple(f"renamed_{index}" for index in range(30)))
    permutation = np.random.default_rng(3).permutation(30)
    permuted = CaseFeatureDataset(
        tuple(dataset.case_ids[index] for index in permutation),
        dataset.labels[permutation],
        {name: matrix[permutation] for name, matrix in dataset.views.items()},
        dataset.feature_names,
    )

    assert identifier_independent_dataset_sha256(dataset) == (
        identifier_independent_dataset_sha256(renamed)
    )
    assert identifier_independent_dataset_sha256(dataset) == (
        identifier_independent_dataset_sha256(permuted)
    )
    first = canonical_content_order(dataset.views["encoder"], dataset.labels)
    second = canonical_content_order(permuted.views["encoder"], permuted.labels)
    assert np.array_equal(dataset.views["encoder"][first], permuted.views["encoder"][second])


def test_train_only_search_and_refit_are_case_rename_invariant() -> None:
    dataset = _dataset()
    lock = _small_lock()
    # Keep this unit test fast without changing the production lock.
    lock["candidate_grid"][-1]["n_estimators"] = 8
    first = evaluate_locked_candidates(dataset, lock)
    renamed = dataset.with_case_ids(tuple(f"new_{index:03d}" for index in range(30)))
    second = evaluate_locked_candidates(renamed, copy.deepcopy(lock))

    assert first["selected_candidate_id"] == second["selected_candidate_id"]
    assert first["selected_mean_repeat_oof_macro_f1"] == second["selected_mean_repeat_oof_macro_f1"]
    assert first["case_ids_used_as_model_features"] is False
    estimator, metadata = fit_selected_classifier(dataset, lock, first)
    predictions = estimator.predict(dataset.views[metadata["feature_view"]])
    assert predictions.shape == (30,)
    assert metadata["case_ids_used_as_model_features"] is False
    assert metadata["official_validation_accessed"] is False
