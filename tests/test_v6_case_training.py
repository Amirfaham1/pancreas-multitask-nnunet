from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pancreas_multitask.v6_case_head import (
    LESION_PROBABILITY_CHANNEL,
    MORPHOLOGY_FEATURES,
    SPATIAL_CHANNELS,
    SPATIAL_GRID,
    SPATIAL_TOKEN_COUNT,
    V6_SPATIAL_CANDIDATE,
    V6_SPATIAL_MORPH_CANDIDATE,
    WHOLE_PROBABILITY_CHANNEL,
)
from pancreas_multitask.v6_case_training import (
    LOCKED_V6_SCHEDULE,
    V6CaseTensors,
    V6EpochEvaluation,
    V6InnerCandidateDecision,
    V6TrainingSchedule,
    balanced_v6_sample_indices,
    ensemble_v6_final_logits,
    evaluate_v6_train_only_screen,
    fit_morphology_scaler,
    make_v6_inner_split,
    make_v6_outer_splits,
    select_v6_final_family_and_epoch,
    select_v6_inner_candidate,
    select_v6_inner_epoch,
    train_v6_trajectory,
    v6_classification_loss,
    v6_learning_rate,
)


def _case_tensors(case_count: int = 9) -> V6CaseTensors:
    generator = torch.Generator().manual_seed(612)
    spatial = torch.zeros(case_count, SPATIAL_CHANNELS, *SPATIAL_GRID)
    lesion = torch.linspace(0.02, 0.70, SPATIAL_TOKEN_COUNT).reshape(1, *SPATIAL_GRID)
    lesion = lesion.expand(case_count, -1, -1, -1).clone()
    spatial[:, LESION_PROBABILITY_CHANNEL] = lesion
    spatial[:, WHOLE_PROBABILITY_CHANNEL] = (lesion + 0.20).clamp_max(1.0)
    axes = [
        (torch.arange(size, dtype=torch.float32) + 0.5) / size for size in SPATIAL_GRID
    ]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
    coordinates = coordinates.unsqueeze(0).expand(case_count, -1, -1, -1, -1).clone()
    rescue_logits = torch.randn(case_count, 3, generator=generator)
    return V6CaseTensors(
        spatial,
        coordinates,
        lesion.mean(dim=(1, 2, 3)),
        torch.randn(case_count, 320, generator=generator),
        torch.randn(case_count, 320, generator=generator),
        rescue_logits,
        torch.softmax(torch.randn(case_count, 3, generator=generator), dim=1),
        torch.randn(case_count, MORPHOLOGY_FEATURES, generator=generator),
        torch.tensor([index % 3 for index in range(case_count)], dtype=torch.long),
    )


def test_locked_training_schedule_matches_prospective_values() -> None:
    assert LOCKED_V6_SCHEDULE == V6TrainingSchedule()
    assert LOCKED_V6_SCHEDULE.folds == 5
    assert LOCKED_V6_SCHEDULE.repeat_seeds == (20260806, 20260807, 20260808)
    assert LOCKED_V6_SCHEDULE.maximum_inner_epochs == 120
    assert LOCKED_V6_SCHEDULE.batch_size == 16
    assert LOCKED_V6_SCHEDULE.samples_per_epoch == 256
    assert LOCKED_V6_SCHEDULE.learning_rate == 1e-4
    assert LOCKED_V6_SCHEDULE.weight_decay == 1e-3
    assert LOCKED_V6_SCHEDULE.gradient_clip_norm == 5.0
    assert LOCKED_V6_SCHEDULE.label_smoothing == 0.05
    assert LOCKED_V6_SCHEDULE.correction_penalty_weight == 0.01


def test_scaler_is_train_partition_only_cpu_float64_linear_then_float32() -> None:
    rows = torch.arange(10, dtype=torch.float32).unsqueeze(1)
    columns = torch.arange(MORPHOLOGY_FEATURES, dtype=torch.float32).unsqueeze(0)
    morphology = rows * 10.0 + columns
    morphology[:, 0] = 7.0
    training = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
    statistics = fit_morphology_scaler(morphology, training)

    expected_median = torch.arange(MORPHOLOGY_FEATURES, dtype=torch.float32) + 40.0
    expected_median[0] = 7.0
    expected_iqr = torch.full((MORPHOLOGY_FEATURES,), 40.0)
    expected_iqr[0] = 1e-6
    assert statistics.median.device.type == "cpu"
    assert statistics.median.dtype == torch.float32
    assert torch.equal(statistics.median, expected_median)
    assert torch.equal(statistics.iqr, expected_iqr)

    changed_outside_partition = morphology.clone()
    changed_outside_partition[1::2] = 1e6
    repeated = fit_morphology_scaler(changed_outside_partition, training)
    assert torch.equal(repeated.median, statistics.median)
    assert torch.equal(repeated.iqr, statistics.iqr)


def test_balanced_sampler_is_seeded_class_balanced_replacement() -> None:
    labels = np.asarray([0] * 2 + [1] * 5 + [2] * 3, dtype=np.int64)
    available = np.arange(labels.size)
    first = balanced_v6_sample_indices(labels, available, sample_count=9000, seed=80)
    repeated = balanced_v6_sample_indices(labels, available, sample_count=9000, seed=80)
    changed = balanced_v6_sample_indices(labels, available, sample_count=9000, seed=81)
    proportions = np.bincount(labels[first], minlength=3) / first.size

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert np.unique(first).size < first.size
    assert np.all(np.abs(proportions - 1 / 3) < 0.03)


def test_outer_splits_use_explicit_audited_order_and_cover_each_case_once_per_repeat() -> None:
    labels = np.asarray([index % 3 for index in range(30)], dtype=np.int64)
    audited_order = np.arange(labels.size, dtype=np.int64)[::-1]
    rows = make_v6_outer_splits(labels, audited_order=audited_order)

    assert len(rows) == 15
    for repeat_index in range(3):
        coverage = np.zeros(labels.size, dtype=np.int64)
        repeat = [row for row in rows if row.repeat_index == repeat_index]
        assert len(repeat) == 5
        for row in repeat:
            assert np.intersect1d(row.train_indices, row.held_indices).size == 0
            assert set(np.unique(labels[row.held_indices])) == {0, 1, 2}
            coverage[row.held_indices] += 1
        assert np.array_equal(coverage, np.ones(labels.size, dtype=np.int64))

    repeated = make_v6_outer_splits(labels, audited_order=audited_order)
    assert all(
        np.array_equal(first.held_indices, second.held_indices)
        for first, second in zip(rows, repeated, strict=True)
    )
    with pytest.raises(ValueError, match="permutation"):
        make_v6_outer_splits(labels, audited_order=np.arange(labels.size - 1))


def test_inner_split_uses_exact_seed_formula_and_stratified_80_20() -> None:
    labels = np.asarray([index % 3 for index in range(30)], dtype=np.int64)
    outer = np.arange(30, dtype=np.int64)
    train, validation, seed = make_v6_inner_split(
        labels,
        outer,
        repeat_seed=20260806,
        fold_index=3,
    )
    repeated = make_v6_inner_split(
        labels,
        outer,
        repeat_seed=20260806,
        fold_index=3,
    )

    assert seed == 20260806 + 3000 + 77
    assert train.size == 24
    assert validation.size == 6
    assert np.intersect1d(train, validation).size == 0
    assert set(np.concatenate((train, validation)).tolist()) == set(outer.tolist())
    assert np.array_equal(train, repeated[0])
    assert np.array_equal(validation, repeated[1])
    assert np.array_equal(np.bincount(labels[validation], minlength=3), [2, 2, 2])


def test_learning_rate_has_exact_warmup_and_cosine_endpoints() -> None:
    expected_warmup = [2e-5, 4e-5, 6e-5, 8e-5, 1e-4]
    assert [v6_learning_rate(epoch) for epoch in range(5)] == pytest.approx(expected_warmup)
    assert v6_learning_rate(5) == pytest.approx(1e-4)
    assert v6_learning_rate(119) == pytest.approx(0.0, abs=1e-15)
    cosine = [v6_learning_rate(epoch) for epoch in range(5, 120)]
    assert all(first >= second for first, second in pairwise(cosine))


def test_loss_is_unweighted_smoothed_ce_plus_mean_squared_bounded_correction() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.2, 0.1, -0.3]])
    labels = torch.tensor([0, 2], dtype=torch.long)
    correction = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    total, cross_entropy, penalty = v6_classification_loss(logits, labels, correction)

    expected_cross_entropy = F.cross_entropy(logits, labels, label_smoothing=0.05)
    expected_penalty = 0.01 * correction.square().mean()
    assert torch.equal(cross_entropy, expected_cross_entropy)
    assert torch.equal(penalty, expected_penalty)
    assert torch.equal(total, expected_cross_entropy + expected_penalty)


def test_inner_epoch_selection_uses_f1_then_nll_then_earlier_epoch() -> None:
    rows = [
        V6EpochEvaluation(0, 0.60, 1.1),
        V6EpochEvaluation(1, 0.61, 1.5),
        V6EpochEvaluation(2, 0.61, 1.2),
        V6EpochEvaluation(3, 0.61, 1.2),
    ]
    selected = select_v6_inner_epoch(rows)
    assert selected.epoch_index == 2
    assert selected.epoch_count == 3


def test_candidate_selection_literal_tie_order_prefers_no_morphology_before_nll() -> None:
    no_morph = V6InnerCandidateDecision(V6_SPATIAL_CANDIDATE, 20, 0.600, 9.0)
    morph = V6InnerCandidateDecision(V6_SPATIAL_MORPH_CANDIDATE, 30, 0.610, 0.1)
    selected, basis = select_v6_inner_candidate([no_morph, morph])
    assert selected == no_morph
    assert "no_morphology" in basis

    clearly_better = V6InnerCandidateDecision(
        V6_SPATIAL_MORPH_CANDIDATE, 30, 0.611, 9.0
    )
    selected, basis = select_v6_inner_candidate([no_morph, clearly_better])
    assert selected == clearly_better
    assert "higher_macro_f1" in basis


def test_final_family_and_round_half_up_epoch_use_only_winning_family_decisions() -> None:
    decisions = [
        V6InnerCandidateDecision(V6_SPATIAL_CANDIDATE, epoch, 0.6, 1.0)
        for epoch in range(1, 9)
    ]
    decisions.extend(
        V6InnerCandidateDecision(V6_SPATIAL_MORPH_CANDIDATE, 100, 0.7, 0.9)
        for _ in range(7)
    )
    family, epoch_count, audit = select_v6_final_family_and_epoch(decisions)

    assert family == V6_SPATIAL_CANDIDATE
    assert epoch_count == 5
    assert audit["median_selected_epoch"] == 4.5
    assert audit["selected_family_epoch_counts"] == list(range(1, 9))


def test_final_ensemble_averages_three_raw_logits_before_one_softmax() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.0, 0.0], [0.0, 1.0, 2.0]],
            [[0.0, 3.0, 0.0], [0.0, 2.0, 1.0]],
            [[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]],
        ]
    )
    mean_logits, probabilities = ensemble_v6_final_logits(logits)
    expected = logits.mean(dim=0)
    assert torch.equal(mean_logits, expected)
    assert torch.equal(probabilities, torch.softmax(expected, dim=1))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


def test_train_only_screen_applies_all_three_metric_gates_and_exact_coverage() -> None:
    labels = np.asarray([index % 3 for index in range(30)], dtype=np.int64)
    logits = np.zeros((3, labels.size, 3), dtype=np.float32)
    for repeat in range(3):
        logits[repeat, np.arange(labels.size), labels] = 5.0
    coverage = np.ones((3, labels.size), dtype=np.int64)
    passed = evaluate_v6_train_only_screen(
        labels,
        logits,
        coverage,
        resubstitution_macro_f1=1.0,
    )
    assert passed["passed_train_only_screen"] is True
    assert passed["validation_attempt_permitted"] is True
    assert passed["mean_across_repeat_oof_macro_f1"] == 1.0
    assert passed["minimum_repeat_per_class_recall"] == 1.0

    coverage[2, 0] = 0
    failed = evaluate_v6_train_only_screen(
        labels,
        logits,
        coverage,
        resubstitution_macro_f1=1.0,
    )
    assert failed["complete_fold_and_case_coverage"] is False
    assert failed["validation_attempt_permitted"] is False


def test_policy_neutral_caller_seeds_produce_reproducible_cpu_trajectory() -> None:
    tensors = _case_tensors()
    schedule = V6TrainingSchedule(
        folds=3,
        repeat_seeds=(7,),
        maximum_inner_epochs=7,
        batch_size=3,
        samples_per_epoch=3,
        learning_rate=1e-4,
        weight_decay=1e-3,
        warmup_epochs=5,
        gradient_clip_norm=5.0,
        label_smoothing=0.05,
        correction_penalty_weight=0.01,
        candidate_tie_band=0.01,
    )
    rows = np.arange(tensors.case_count, dtype=np.int64)
    first_model, first = train_v6_trajectory(
        V6_SPATIAL_CANDIDATE,
        tensors,
        rows,
        epochs=1,
        model_seed=701,
        sampler_seed_base=901,
        schedule=schedule,
    )
    second_model, second = train_v6_trajectory(
        V6_SPATIAL_CANDIDATE,
        tensors,
        rows,
        epochs=1,
        model_seed=701,
        sampler_seed_base=901,
        schedule=schedule,
    )

    assert first["initial_state_sha256"] == second["initial_state_sha256"]
    assert first["final_state_sha256"] == second["final_state_sha256"]
    assert first["epoch_history"] == second["epoch_history"]
    assert first["epoch_history"][0]["sampler_seed"] == 901
    assert first["caller_supplied_audited_row_order_required"] is True
    assert all(
        torch.equal(first_model.state_dict()[name], second_model.state_dict()[name])
        for name in first_model.state_dict()
    )


def test_case_tensor_batches_never_pass_morphology_to_no_morph_candidate() -> None:
    tensors = _case_tensors()
    no_morph_batch = tensors.batch([0, 1], V6_SPATIAL_CANDIDATE)
    morph_batch = tensors.batch([0, 1], V6_SPATIAL_MORPH_CANDIDATE)
    assert no_morph_batch[-2] is None
    assert morph_batch[-2] is not None
    assert morph_batch[-2].shape == (2, MORPHOLOGY_FEATURES)
