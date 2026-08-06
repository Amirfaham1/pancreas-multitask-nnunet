from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pancreas_multitask.classification_rescue import file_sha256
from pancreas_multitask.neural_case_head import (
    CHECKPOINT_SHA256,
    DATASET_JSON_SHA256,
    EXTRACTOR_IMPLEMENTATION_SHA256,
    NEURAL_ATTENTION_CANDIDATE,
    NEURAL_CANDIDATES,
    NEURAL_MEAN_CANDIDATE,
    PLANS_SHA256,
    SPEED_LOCK_SHA256,
    V3_FEATURE_LOCK_SHA256,
    V5_DECISION_LOCK_SHA256,
    V5_NEURAL_LOCK_SHA256,
    NeuralBagDataset,
    NeuralBagTensors,
    NeuralCaseBag,
    _validate_cache_binding,
    build_neural_case_head,
    materialize_neural_bags,
    neural_head_parameter_count,
)
from pancreas_multitask.neural_case_training import (
    NeuralTrainingSchedule,
    _select_candidate,
    _train_trajectory,
    _verify_materialized_tensors,
    balanced_sample_indices,
    configure_deterministic_execution,
    fit_selected_neural_head,
)

ROOT = Path(__file__).resolve().parents[1]
POST_LOCK_INFERENCE_EXTRACTOR_SHA256 = (
    "eef3eb3a8a530ea7dfa31e5eba438e8f32fae0053006ef0c44577b0230707926"
)
STOCK_EXPORT_CONFORMANCE_LOCK_SHA256 = (
    "bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503"
)


def test_historical_training_hash_and_locked_export_only_divergence_are_explicit() -> None:
    """Keep training provenance immutable while binding current inference code.

    The prospectively locked stock-export repair changed only the terminal
    logit dtype passed to resampling. It did not change any cached neural-bag
    value, so the fitted bundle must retain its historical extractor binding.
    """

    script = ROOT / "scripts" / "extract_train_case_features.py"
    specification = importlib.util.spec_from_file_location("v5_extract_train_case_features", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    assert EXTRACTOR_IMPLEMENTATION_SHA256 == (
        "68956b493c8004b86558841d830e633827f12b0c9a099d3d42f8ddab8de2c46f"
    )
    assert module._implementation_sha256() == POST_LOCK_INFERENCE_EXTRACTOR_SHA256
    assert POST_LOCK_INFERENCE_EXTRACTOR_SHA256 != EXTRACTOR_IMPLEMENTATION_SHA256
    assert file_sha256(ROOT / "configs" / "inference_speed_benchmark.json") == SPEED_LOCK_SHA256
    assert file_sha256(
        ROOT / "configs" / "inference_stock_export_conformance_v1.json"
    ) == STOCK_EXPORT_CONFORMANCE_LOCK_SHA256

    predictor_script = ROOT / "scripts" / "predict_joint.py"
    predictor_specification = importlib.util.spec_from_file_location(
        "v5_predict_joint_manifest", predictor_script
    )
    assert predictor_specification is not None and predictor_specification.loader is not None
    predictor_module = importlib.util.module_from_spec(predictor_specification)
    predictor_specification.loader.exec_module(predictor_module)
    assert "src/pancreas_multitask/case_feature_extractor.py" in (
        predictor_module.V5_IMPLEMENTATION_RELATIVE_PATHS
    )
    assert predictor_module.STOCK_EXPORT_CONFORMANCE_LOCK_SHA256 == (
        STOCK_EXPORT_CONFORMANCE_LOCK_SHA256
    )


def _synthetic_dataset(case_count: int = 9) -> NeuralBagDataset:
    generator = np.random.default_rng(420)
    bags = []
    for index in range(case_count):
        tile_count = index % 3 + 1
        lesion = generator.uniform(0.0, 0.7, size=(tile_count, 1, 4, 4, 6))
        whole = lesion + generator.uniform(0.0, 1.0, size=(tile_count, 1, 4, 4, 6)) * (1.0 - lesion)
        bags.append(
            NeuralCaseBag(
                generator.normal(size=(tile_count, 256, 4, 4, 6)).astype(np.float16),
                np.concatenate((lesion, whole), axis=1).astype(np.float16),
                np.sort(generator.uniform(size=tile_count))[::-1].copy().astype(np.float32),
                generator.normal(size=646).astype(np.float32),
            )
        )
    return NeuralBagDataset(
        tuple(f"audit_only_{index}" for index in range(case_count)),
        np.asarray([index % 3 for index in range(case_count)], dtype=np.int64),
        tuple(bags),
        {},
    )


def test_locked_heads_have_finite_three_class_gradients_and_parameter_bounds() -> None:
    dataset = _synthetic_dataset(3)
    tensors = materialize_neural_bags(dataset, torch.device("cpu"))
    batch = tensors.batch([0, 1, 2])

    assert neural_head_parameter_count(NEURAL_MEAN_CANDIDATE) == 117_263
    assert neural_head_parameter_count(NEURAL_ATTENTION_CANDIDATE) == 101_391
    for candidate_id in NEURAL_CANDIDATES:
        model = build_neural_case_head(candidate_id)
        logits = model(*batch[:-1])
        logits.square().mean().backward()

        assert logits.shape == (3, 3)
        assert torch.isfinite(logits).all()
        assert all(parameter.grad is not None for parameter in model.parameters())
        assert neural_head_parameter_count(candidate_id) <= 150_000


def test_padded_tiles_cannot_change_either_locked_head() -> None:
    dataset = _synthetic_dataset(3)
    tensors = materialize_neural_bags(dataset, torch.device("cpu"))
    stage3, predictions, masses, valid, summary, _ = tensors.batch([0])
    changed_stage3 = stage3.clone()
    changed_predictions = predictions.clone()
    changed_masses = masses.clone()
    changed_stage3[:, 1:] = 100
    changed_predictions[:, 1:] = -100
    changed_masses[:, 1:] = 100

    for candidate_id in NEURAL_CANDIDATES:
        torch.manual_seed(11)
        model = build_neural_case_head(candidate_id).eval()
        reference = model(stage3, predictions, masses, valid, summary)
        changed = model(
            changed_stage3,
            changed_predictions,
            changed_masses,
            valid,
            summary,
        )
        assert torch.equal(reference, changed)


def test_numeric_dataset_identity_and_order_ignore_case_identifiers_and_rows() -> None:
    dataset = _synthetic_dataset()
    permutation = np.asarray([8, 0, 4, 2, 7, 1, 6, 3, 5])
    renamed_permuted = NeuralBagDataset(
        tuple(f"renamed_{index}" for index in range(dataset.case_count)),
        dataset.labels[permutation],
        tuple(dataset.bags[index] for index in permutation),
        {},
    )

    original_summaries = [
        dataset.bags[index].all_tile_summary.tobytes()
        for index in dataset.content_canonical_order()
    ]
    permuted_summaries = [
        renamed_permuted.bags[index].all_tile_summary.tobytes()
        for index in renamed_permuted.content_canonical_order()
    ]
    assert dataset.content_sha256() == renamed_permuted.content_sha256()
    assert original_summaries == permuted_summaries


def test_neural_bag_rejects_invalid_prediction_semantics() -> None:
    stage3 = np.zeros((1, 256, 4, 4, 6), dtype=np.float16)
    predictions = np.zeros((1, 2, 4, 4, 6), dtype=np.float16)
    predictions[:, 0] = 0.8
    predictions[:, 1] = 0.2

    with pytest.raises(ValueError, match="probability semantics"):
        NeuralCaseBag(
            stage3,
            predictions,
            np.asarray([0.5], dtype=np.float32),
            np.zeros(646, dtype=np.float32),
        )

    predictions[:, 1] = 0.9
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        NeuralCaseBag(
            stage3,
            predictions,
            np.asarray([1.1], dtype=np.float32),
            np.zeros(646, dtype=np.float32),
        )


def test_canonical_order_fails_closed_on_duplicate_numeric_bags() -> None:
    dataset = _synthetic_dataset(3)
    duplicate = NeuralBagDataset(
        ("first", "second", "third", "fourth"),
        np.asarray([0, 0, 1, 2], dtype=np.int64),
        (dataset.bags[0], dataset.bags[0], dataset.bags[1], dataset.bags[2]),
        {},
    )

    with pytest.raises(ValueError, match="input order a split key"):
        duplicate.content_canonical_order()


def test_balanced_sampler_is_seeded_replacement_sampling() -> None:
    labels = np.asarray([0] * 2 + [1] * 5 + [2] * 3, dtype=np.int64)
    available = np.arange(labels.size)

    first = balanced_sample_indices(labels, available, sample_count=9000, seed=91)
    second = balanced_sample_indices(labels, available, sample_count=9000, seed=91)
    changed = balanced_sample_indices(labels, available, sample_count=9000, seed=92)
    proportions = np.bincount(labels[first], minlength=3) / first.size

    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert np.all(np.abs(proportions - 1 / 3) < 0.03)
    assert np.unique(first).size < first.size


def test_deterministic_execution_requires_exact_workspace_and_disables_tf32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        configure_deterministic_execution(torch.device("cpu"))

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    audit = configure_deterministic_execution(torch.device("cpu"))
    assert audit["cublas_workspace_config"] == ":4096:8"
    assert audit["torch_deterministic_algorithms"] is True
    assert audit["cudnn_benchmark"] is False
    assert audit["cudnn_deterministic"] is True
    assert audit["cuda_matmul_tf32"] is False
    assert audit["cudnn_tf32"] is False


def test_deterministic_execution_rejects_initialized_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        configure_deterministic_execution(torch.device("cuda"))


def test_exact_reference_cache_binding_rejects_any_tamper() -> None:
    binding = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "feature_lock_sha256": V3_FEATURE_LOCK_SHA256,
        "neural_lock_sha256": V5_NEURAL_LOCK_SHA256,
        "implementation_sha256": EXTRACTOR_IMPLEMENTATION_SHA256,
        "neural_decision_lock_sha256": V5_DECISION_LOCK_SHA256,
        "speed_lock_sha256": SPEED_LOCK_SHA256,
        "plans_sha256": PLANS_SHA256,
        "dataset_json_sha256": DATASET_JSON_SHA256,
        "tile_step_size": 0.5,
        "tile_batch_size": 1,
        "tta_batch_size": 1,
        "network_batch_size_limit": 1,
        "locked_network_microbatch_ceiling": 2,
        "tta_enabled": True,
        "gaussian_enabled": True,
    }
    _validate_cache_binding(binding, V5_NEURAL_LOCK_SHA256)

    tampered = dict(binding)
    tampered["tile_batch_size"] = 2
    with pytest.raises(ValueError, match="reference semantics"):
        _validate_cache_binding(tampered, V5_NEURAL_LOCK_SHA256)


def test_reduced_mean_head_trajectory_is_exactly_reproducible() -> None:
    torch.set_num_threads(1)
    configure_deterministic_execution(torch.device("cpu"))
    dataset = _synthetic_dataset()
    tensors = materialize_neural_bags(dataset, torch.device("cpu"))
    schedule = NeuralTrainingSchedule(
        folds=3,
        repeat_seeds=(7,),
        epochs=2,
        batch_size=3,
        samples_per_epoch=6,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        label_smoothing=0.05,
        final_seed=7,
        tie_band_macro_f1=0.01,
    )
    indices = np.arange(dataset.case_count)

    _, first = _train_trajectory(
        NEURAL_MEAN_CANDIDATE,
        tensors,
        indices,
        trajectory_seed=700,
        schedule=schedule,
        context={},
    )
    _, second = _train_trajectory(
        NEURAL_MEAN_CANDIDATE,
        tensors,
        indices,
        trajectory_seed=700,
        schedule=schedule,
        context={},
    )

    assert first["initial_state_sha256"] == second["initial_state_sha256"]
    assert first["final_state_sha256"] == second["final_state_sha256"]
    assert first["epoch_history"] == second["epoch_history"]
    assert first["epochs_completed"] == 2


def test_reduced_attention_trajectory_is_exactly_reproducible() -> None:
    torch.set_num_threads(1)
    configure_deterministic_execution(torch.device("cpu"))
    dataset = _synthetic_dataset()
    tensors = materialize_neural_bags(dataset, torch.device("cpu"))
    schedule = NeuralTrainingSchedule(
        folds=3,
        repeat_seeds=(8,),
        epochs=1,
        batch_size=3,
        samples_per_epoch=3,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        label_smoothing=0.05,
        final_seed=8,
        tie_band_macro_f1=0.01,
    )
    indices = np.arange(dataset.case_count)
    _, first = _train_trajectory(
        NEURAL_ATTENTION_CANDIDATE,
        tensors,
        indices,
        trajectory_seed=800,
        schedule=schedule,
        context={},
    )
    _, second = _train_trajectory(
        NEURAL_ATTENTION_CANDIDATE,
        tensors,
        indices,
        trajectory_seed=800,
        schedule=schedule,
        context={},
    )
    assert first["final_state_sha256"] == second["final_state_sha256"]
    assert first["epoch_history"] == second["epoch_history"]


def test_locked_selection_tie_rule_prefers_recall_then_simpler_mean() -> None:
    rows = [
        {
            "candidate_id": NEURAL_MEAN_CANDIDATE,
            "mean_repeat_oof_macro_f1": 0.70,
            "minimum_repeat_per_class_recall": 0.61,
        },
        {
            "candidate_id": NEURAL_ATTENTION_CANDIDATE,
            "mean_repeat_oof_macro_f1": 0.705,
            "minimum_repeat_per_class_recall": 0.63,
        },
    ]
    selected, audit = _select_candidate(rows, 0.01)
    assert selected["candidate_id"] == NEURAL_ATTENTION_CANDIDATE
    assert "minimum" in audit["decision_basis"]

    rows[1]["minimum_repeat_per_class_recall"] = 0.615
    selected, audit = _select_candidate(rows, 0.01)
    assert selected["candidate_id"] == NEURAL_MEAN_CANDIDATE
    assert "simpler" in audit["decision_basis"]

    with pytest.raises(ValueError, match="exactly the two"):
        _select_candidate([*rows, dict(rows[0])], 0.01)


def test_refit_rejects_tampered_selection_before_training() -> None:
    dataset = _synthetic_dataset()
    tensors = materialize_neural_bags(dataset, torch.device("cpu"))
    rows = [
        {
            "candidate_id": NEURAL_MEAN_CANDIDATE,
            "mean_repeat_oof_macro_f1": 0.7,
            "minimum_repeat_per_class_recall": 0.6,
        },
        {
            "candidate_id": NEURAL_ATTENTION_CANDIDATE,
            "mean_repeat_oof_macro_f1": 0.5,
            "minimum_repeat_per_class_recall": 0.4,
        },
    ]
    _, decision = _select_candidate(rows, 0.01)
    selection = {
        "status": "complete",
        "scope": "isolated_supplied_train_only",
        "candidate_count": 2,
        "candidate_results": rows,
        "selection_decision": decision,
        "selected_candidate_id": NEURAL_ATTENTION_CANDIDATE,
        "selected_mean_repeat_oof_macro_f1": 0.5,
        "selected_minimum_repeat_per_class_recall": 0.4,
        "numeric_content_dataset_sha256": dataset.content_sha256(),
    }
    lock = json.loads((ROOT / "configs" / "phd_neural_case_head_lock_v5.json").read_text())

    with pytest.raises(ValueError, match="deterministic v5 decision"):
        fit_selected_neural_head(dataset, tensors, lock, selection)


def test_materialized_tensor_binding_rejects_content_tamper() -> None:
    dataset = _synthetic_dataset()
    tensors = materialize_neural_bags(dataset, torch.device("cpu"))
    changed_summary = tensors.all_tile_summary.clone()
    changed_summary[0, 0] += 1
    tampered = NeuralBagTensors(
        tensors.stage3_maps,
        tensors.prediction_maps,
        tensors.lesion_mass,
        tensors.valid_tiles,
        changed_summary,
        tensors.labels,
    )

    with pytest.raises(ValueError, match="source bag content"):
        _verify_materialized_tensors(dataset, tampered)
