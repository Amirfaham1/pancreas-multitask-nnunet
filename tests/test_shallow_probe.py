from __future__ import annotations

import numpy as np

from pancreas_multitask.shallow_probe import (
    best_of_n_inflation,
    nested_cv_macro_f1,
    stage_pool,
    validation_subsample_distribution,
)


def test_stage_pool_selects_expected_channels_and_averages_space() -> None:
    spatial = np.arange(4 * 387 * 2 * 2 * 2, dtype=np.float32).reshape(4, 387, 2, 2, 2)

    stage2 = stage_pool(spatial, "stage2")
    stage3 = stage_pool(spatial, "stage3")

    assert stage2.shape == (4, 128)
    assert stage3.shape == (4, 256)
    np.testing.assert_allclose(stage2, spatial[:, :128].mean(axis=(2, 3, 4)))
    np.testing.assert_allclose(stage3, spatial[:, 128:384].mean(axis=(2, 3, 4)))


def test_nested_probe_is_deterministic_on_separable_data() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), 30)
    features = rng.normal(scale=0.15, size=(90, 6))
    features[:, :3] += np.eye(3)[labels] * 3.0

    first = nested_cv_macro_f1(features, labels, c_grid=(1.0,), seeds=(0,), folds=3)
    second = nested_cv_macro_f1(features, labels, c_grid=(1.0,), seeds=(0,), folds=3)

    assert first.mean_macro_f1 > 0.95
    assert first.mean_macro_f1 == second.mean_macro_f1
    np.testing.assert_array_equal(first.oof_probabilities, second.oof_probabilities)


def test_validation_forecast_and_look_inflation_are_seeded() -> None:
    labels = np.repeat(np.arange(3), 20)
    probabilities = np.full((60, 3), 0.05, dtype=np.float64)
    probabilities[np.arange(60), labels] = 0.9

    first = validation_subsample_distribution(
        probabilities,
        labels,
        class_counts=(3, 4, 5),
        repeats=200,
        seed=99,
    )
    second = validation_subsample_distribution(
        probabilities,
        labels,
        class_counts=(3, 4, 5),
        repeats=200,
        seed=99,
    )

    assert first == second
    assert first["validation_case_count"] == 12
    assert first["probability_at_or_above"]["0.70"] == 1.0
    assert best_of_n_inflation(0.1, 1, repeats=500, seed=2) == 0.0
    assert best_of_n_inflation(0.1, 10, repeats=500, seed=2) > 0.0
