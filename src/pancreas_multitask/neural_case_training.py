"""Deterministic train-only selection and refit for the locked v5 neural heads."""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch import nn

from pancreas_multitask.neural_case_head import (
    NEURAL_CANDIDATES,
    NEURAL_MEAN_CANDIDATE,
    NeuralBagDataset,
    NeuralBagTensors,
    build_neural_case_head,
    neural_head_parameter_count,
)

TrainingEventSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class NeuralTrainingSchedule:
    folds: int
    repeat_seeds: tuple[int, ...]
    epochs: int
    batch_size: int
    samples_per_epoch: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    label_smoothing: float
    final_seed: int
    tie_band_macro_f1: float

    def __post_init__(self) -> None:
        if self.folds < 2 or not self.repeat_seeds:
            raise ValueError("Neural training requires at least two folds and one repeat")
        if self.epochs < 1 or self.batch_size < 1 or self.samples_per_epoch < 1:
            raise ValueError("Neural training counts must be positive")
        if self.samples_per_epoch % self.batch_size:
            raise ValueError("Locked samples per epoch must divide evenly into case batches")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Neural optimizer settings are invalid")
        if self.gradient_clip_norm <= 0 or not 0 <= self.label_smoothing < 1:
            raise ValueError("Neural loss or gradient settings are invalid")
        if self.tie_band_macro_f1 < 0:
            raise ValueError("Neural selection tie band must be non-negative")


def schedule_from_lock(lock: Mapping[str, Any]) -> NeuralTrainingSchedule:
    """Read the exact immutable v5 schedule into a typed value."""

    training = lock["training"]
    selection = lock["train_only_selection"]
    if training.get("sampler") != (
        "deterministic_class_balanced_sampling_with_replacement_each_epoch"
    ):
        raise ValueError("Neural lock does not declare the required balanced sampler")
    if training.get("loss") != "unweighted_cross_entropy":
        raise ValueError("Neural lock does not declare unweighted cross entropy")
    if training.get("optimizer") != "AdamW" or training.get("scheduler") != (
        "cosine_annealing_lr_to_zero_over_150_epochs"
    ):
        raise ValueError("Neural lock optimizer or scheduler differs from v5")
    if training.get("mixed_precision") is not False:
        raise ValueError("Neural head training must remain float32")
    if training.get("early_stopping") is not False:
        raise ValueError("The locked neural search prohibits early stopping")
    return NeuralTrainingSchedule(
        folds=int(training["folds"]),
        repeat_seeds=tuple(int(value) for value in training["repeat_seeds"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size_cases"]),
        samples_per_epoch=int(training["samples_per_epoch"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        label_smoothing=float(training["label_smoothing"]),
        final_seed=20260806,
        tie_band_macro_f1=float(selection["tie_band_macro_f1"]),
    )


def configure_deterministic_execution(device: torch.device) -> dict[str, Any]:
    """Enable deterministic operators before any v5 trajectory is launched."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA neural training was requested but is unavailable")
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "mixed_precision": False,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def neural_state_sha256(model: nn.Module) -> str:
    """Hash names, dtypes, shapes, and exact bytes of a neural state dict."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        encoded = name.encode("utf-8")
        value = tensor.detach().cpu().contiguous()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    values = np.asarray(labels, dtype=np.int64)
    return {str(label): int(np.sum(values == label)) for label in range(3)}


def balanced_sample_indices(
    labels: np.ndarray,
    available_indices: Sequence[int] | np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Draw with inverse-frequency weights using an isolated CPU generator."""

    targets = np.asarray(labels, dtype=np.int64)
    available = np.asarray(available_indices, dtype=np.int64)
    if available.ndim != 1 or available.size < 1:
        raise ValueError("Balanced sampling requires at least one available case")
    if np.any(available < 0) or np.any(available >= targets.size):
        raise IndexError("Balanced sampler received an out-of-range case index")
    available_targets = targets[available]
    counts = np.bincount(available_targets, minlength=3)
    if counts.shape != (3,) or np.any(counts == 0):
        raise ValueError("Every sampled training fold must contain all three classes")
    weights = 1.0 / counts[available_targets].astype(np.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected_local = torch.multinomial(
        torch.from_numpy(weights),
        int(sample_count),
        replacement=True,
        generator=generator,
    ).numpy()
    return available[selected_local].astype(np.int64, copy=False)


def _metrics(references: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    targets = np.asarray(references, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    recalls = [
        float(value)
        for value in recall_score(
            targets,
            predicted,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
    ]
    return {
        "macro_f1": float(
            f1_score(
                targets,
                predicted,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": recalls,
        "minimum_per_class_recall": float(min(recalls)),
    }


def _verify_materialized_tensors(
    dataset: NeuralBagDataset,
    tensors: NeuralBagTensors,
) -> None:
    """Rejoin every materialized tensor to its immutable numeric source bag."""

    if tensors.case_count != dataset.case_count:
        raise ValueError("Materialized neural tensor count differs from the dataset")
    stage3 = tensors.stage3_maps.detach().cpu()
    predictions = tensors.prediction_maps.detach().cpu()
    masses = tensors.lesion_mass.detach().cpu()
    valid = tensors.valid_tiles.detach().cpu()
    summaries = tensors.all_tile_summary.detach().cpu()
    labels = tensors.labels.detach().cpu().numpy().astype(np.int64, copy=False)
    if not np.array_equal(labels, dataset.labels):
        raise ValueError("Materialized neural labels differ from the verified dataset")
    for index, bag in enumerate(dataset.bags):
        count = int(bag.stage3_maps.shape[0])
        expected_valid = torch.arange(3) < count
        if (
            not torch.equal(valid[index], expected_valid)
            or not torch.equal(
                stage3[index, :count],
                torch.from_numpy(bag.stage3_maps.astype(np.float32)),
            )
            or not torch.equal(
                predictions[index, :count],
                torch.from_numpy(bag.prediction_maps.astype(np.float32)),
            )
            or not torch.equal(
                masses[index, :count],
                torch.from_numpy(bag.lesion_mass.astype(np.float32)),
            )
            or not torch.equal(
                summaries[index],
                torch.from_numpy(bag.all_tile_summary.astype(np.float32)),
            )
            or torch.count_nonzero(stage3[index, count:]).item() != 0
            or torch.count_nonzero(predictions[index, count:]).item() != 0
            or torch.count_nonzero(masses[index, count:]).item() != 0
        ):
            raise ValueError("Materialized neural tensors differ from source bag content")


def _log_softmax_float64(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError("Neural logits must be a finite (cases, 3) matrix")
    maximum = scores.max(axis=1, keepdims=True)
    shifted = scores - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


@torch.inference_mode()
def predict_neural_logits(
    model: nn.Module,
    tensors: NeuralBagTensors,
    indices: Sequence[int] | np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Predict raw logits in caller-supplied numeric order with dropout disabled."""

    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim != 1:
        raise ValueError("Neural prediction indices must be one-dimensional")
    model.eval()
    rows: list[np.ndarray] = []
    for start in range(0, requested.size, batch_size):
        batch = tensors.batch(requested[start : start + batch_size])
        logits = model(*batch[:-1])
        if logits.shape != (batch[-1].shape[0], 3) or not torch.isfinite(logits).all():
            raise FloatingPointError("Neural head emitted invalid logits")
        rows.append(logits.detach().float().cpu().numpy())
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def _train_trajectory(
    candidate_id: str,
    tensors: NeuralBagTensors,
    train_indices: np.ndarray,
    *,
    trajectory_seed: int,
    schedule: NeuralTrainingSchedule,
    context: Mapping[str, int | str],
    event_sink: TrainingEventSink | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train exactly one fold or final-refit trajectory without early stopping."""

    _seed_everything(trajectory_seed)
    model = build_neural_case_head(candidate_id)
    initial_state_hash = neural_state_sha256(model)
    model.to(tensors.device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.learning_rate,
        weight_decay=schedule.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=schedule.epochs,
        eta_min=0.0,
    )
    labels_cpu = tensors.labels.detach().cpu().numpy().astype(np.int64, copy=False)
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(schedule.epochs):
        sampler_seed = trajectory_seed + epoch
        sampled = balanced_sample_indices(
            labels_cpu,
            train_indices,
            sample_count=schedule.samples_per_epoch,
            seed=sampler_seed,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        loss_sum = 0.0
        gradient_norm_sum = 0.0
        maximum_gradient_norm = 0.0
        for start in range(0, sampled.size, schedule.batch_size):
            batch = tensors.batch(sampled[start : start + schedule.batch_size])
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:-1])
            loss = F.cross_entropy(
                logits,
                batch[-1],
                weight=None,
                label_smoothing=schedule.label_smoothing,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Neural head training produced a non-finite loss")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), schedule.gradient_clip_norm
                ).item()
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("Neural head training produced non-finite gradients")
            optimizer.step()
            batch_count = int(batch[-1].numel())
            loss_sum += float(loss.detach().item()) * batch_count
            gradient_norm_sum += gradient_norm
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        scheduler.step()
        batch_count_per_epoch = schedule.samples_per_epoch // schedule.batch_size
        epoch_row = {
            "epoch": epoch,
            "sampler_seed": sampler_seed,
            "mean_unweighted_cross_entropy": loss_sum / schedule.samples_per_epoch,
            "mean_pre_clip_gradient_norm": gradient_norm_sum / batch_count_per_epoch,
            "maximum_pre_clip_gradient_norm": maximum_gradient_norm,
            "learning_rate": learning_rate,
            "learning_rate_after_scheduler_step": float(optimizer.param_groups[0]["lr"]),
            "sampled_class_counts": _class_counts(labels_cpu[sampled]),
        }
        epoch_rows.append(epoch_row)
        if event_sink is not None:
            event_sink(
                {
                    **context,
                    "epoch": epoch,
                    "sampler_seed": sampler_seed,
                    "train/loss": epoch_row["mean_unweighted_cross_entropy"],
                    "train/learning_rate": learning_rate,
                    "train/mean_pre_clip_gradient_norm": epoch_row["mean_pre_clip_gradient_norm"],
                }
            )
    final_state_hash = neural_state_sha256(model)
    return model, {
        "trajectory_seed": trajectory_seed,
        "model_initialization_seed": trajectory_seed,
        "sampler_seed_formula": "trajectory_seed_plus_zero_based_epoch",
        "initial_state_sha256": initial_state_hash,
        "final_state_sha256": final_state_hash,
        "trainable_parameter_count": neural_head_parameter_count(candidate_id),
        "train_case_count": int(train_indices.size),
        "train_class_counts": _class_counts(labels_cpu[train_indices]),
        "epochs_completed": schedule.epochs,
        "early_stopping_used": False,
        "mixed_precision_used": False,
        "class_weighted_loss_used": False,
        "balanced_sampling_with_replacement_used": True,
        "epoch_history": epoch_rows,
    }


def _select_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
    tie_band: float,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    candidate_ids = [str(row["candidate_id"]) for row in candidate_results]
    if (
        len(candidate_results) != len(NEURAL_CANDIDATES)
        or len(candidate_ids) != len(set(candidate_ids))
        or set(candidate_ids) != set(NEURAL_CANDIDATES)
    ):
        raise ValueError("Selection requires exactly the two locked neural candidates")
    for row in candidate_results:
        metrics = (
            float(row["mean_repeat_oof_macro_f1"]),
            float(row["minimum_repeat_per_class_recall"]),
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics):
            raise ValueError("Neural candidate aggregate metrics must be finite in [0, 1]")
    by_id = {str(row["candidate_id"]): row for row in candidate_results}
    mean_row = by_id[NEURAL_MEAN_CANDIDATE]
    attention_id = next(
        candidate for candidate in NEURAL_CANDIDATES if candidate != NEURAL_MEAN_CANDIDATE
    )
    attention_row = by_id[attention_id]
    mean_difference = abs(
        float(mean_row["mean_repeat_oof_macro_f1"])
        - float(attention_row["mean_repeat_oof_macro_f1"])
    )
    recall_difference = abs(
        float(mean_row["minimum_repeat_per_class_recall"])
        - float(attention_row["minimum_repeat_per_class_recall"])
    )
    if mean_difference > tie_band:
        selected = max(
            candidate_results,
            key=lambda row: float(row["mean_repeat_oof_macro_f1"]),
        )
        basis = "higher_mean_repeat_oof_macro_f1_outside_tie_band"
    elif recall_difference > tie_band:
        selected = max(
            candidate_results,
            key=lambda row: float(row["minimum_repeat_per_class_recall"]),
        )
        basis = "higher_minimum_repeat_per_class_recall_inside_macro_f1_tie_band"
    else:
        selected = mean_row
        basis = "prospectively_simpler_mean_head_after_both_ties"
    return selected, {
        "macro_f1_absolute_difference": mean_difference,
        "minimum_recall_absolute_difference": recall_difference,
        "tie_band": tie_band,
        "decision_basis": basis,
        "selected_candidate_id": selected["candidate_id"],
    }


def evaluate_neural_candidates(
    dataset: NeuralBagDataset,
    tensors: NeuralBagTensors,
    lock: Mapping[str, Any],
    *,
    event_sink: TrainingEventSink | None = None,
) -> dict[str, Any]:
    """Run all 30 prospectively locked fold trajectories on train only."""

    schedule = schedule_from_lock(lock)
    expected_count = int(lock["development_boundary"]["case_count"])
    if dataset.case_count != expected_count or tensors.case_count != expected_count:
        raise ValueError("Neural selection requires the exact locked 252-case dataset")
    labels_original = np.asarray(dataset.labels, dtype=np.int64)
    if _class_counts(labels_original) != lock["development_boundary"]["class_counts"]:
        raise ValueError("Neural selection labels differ from the locked class counts")
    if not np.array_equal(tensors.labels.detach().cpu().numpy().astype(np.int64), labels_original):
        raise ValueError("Materialized neural labels differ from the verified dataset")
    _verify_materialized_tensors(dataset, tensors)
    canonical_order = dataset.content_canonical_order()
    labels = labels_original[canonical_order]
    candidate_results: list[dict[str, Any]] = []

    for candidate_id in NEURAL_CANDIDATES:
        repeat_macro_f1: list[float] = []
        repeat_recalls: list[list[float]] = []
        repeat_predictions: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        for repeat_index, repeat_seed in enumerate(schedule.repeat_seeds):
            splitter = StratifiedKFold(
                n_splits=schedule.folds,
                shuffle=True,
                random_state=repeat_seed,
            )
            oof_logits = np.full((dataset.case_count, 3), np.nan, dtype=np.float32)
            oof_fold_index = np.full(dataset.case_count, -1, dtype=np.int64)
            for fold_index, (train_local, held_local) in enumerate(
                splitter.split(np.zeros(dataset.case_count), labels)
            ):
                train_indices = canonical_order[train_local]
                held_indices = canonical_order[held_local]
                trajectory_seed = repeat_seed + fold_index * 1000
                model, trajectory = _train_trajectory(
                    candidate_id,
                    tensors,
                    train_indices,
                    trajectory_seed=trajectory_seed,
                    schedule=schedule,
                    context={
                        "candidate_id": candidate_id,
                        "repeat_index": repeat_index,
                        "repeat_seed": repeat_seed,
                        "fold_index": fold_index,
                        "trajectory": "cross_validation",
                    },
                    event_sink=event_sink,
                )
                held_logits = predict_neural_logits(
                    model,
                    tensors,
                    held_indices,
                    batch_size=schedule.batch_size,
                )
                held_predictions = held_logits.argmax(axis=1)
                held_metrics = _metrics(labels_original[held_indices], held_predictions)
                oof_logits[held_indices] = held_logits
                oof_fold_index[held_indices] = fold_index
                fold_rows.append(
                    {
                        "repeat_index": repeat_index,
                        "repeat_seed": repeat_seed,
                        "fold_index": fold_index,
                        "held_out_case_count": int(held_indices.size),
                        "held_out_class_counts": _class_counts(labels_original[held_indices]),
                        "held_out_metrics": held_metrics,
                        **trajectory,
                    }
                )
                del model
                if tensors.device.type == "cuda":
                    torch.cuda.empty_cache()
            if not np.isfinite(oof_logits).all() or np.any(oof_fold_index < 0):
                raise RuntimeError("Neural repeated CV left cases without OOF logits")
            oof_predictions = oof_logits.argmax(axis=1)
            repeat_metrics = _metrics(labels_original, oof_predictions)
            repeat_macro_f1.append(float(repeat_metrics["macro_f1"]))
            repeat_recalls.append(list(repeat_metrics["per_class_recall"]))
            log_scores = _log_softmax_float64(oof_logits)
            if not np.array_equal(log_scores.argmax(axis=1), oof_predictions):
                raise RuntimeError("Float64 log-softmax changed neural argmax predictions")
            repeat_predictions.append(
                {
                    "repeat_index": repeat_index,
                    "seed": repeat_seed,
                    "macro_f1": repeat_metrics["macro_f1"],
                    "per_class_recall": repeat_metrics["per_class_recall"],
                    "predictions": [
                        {
                            "case_id": case_id,
                            "reference": int(reference),
                            "prediction": int(prediction),
                            "logits": [float(value) for value in logits],
                            "log_scores": [float(value) for value in scores],
                            "held_out_fold_index": int(held_fold),
                        }
                        for case_id, reference, prediction, logits, scores, held_fold in zip(
                            dataset.case_ids,
                            labels_original,
                            oof_predictions,
                            oof_logits,
                            log_scores,
                            oof_fold_index,
                            strict=True,
                        )
                    ],
                }
            )
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "trainable_parameter_count": neural_head_parameter_count(candidate_id),
                "repeat_oof_macro_f1": repeat_macro_f1,
                "repeat_oof_per_class_recall": repeat_recalls,
                "mean_repeat_oof_macro_f1": float(np.mean(repeat_macro_f1)),
                "std_repeat_oof_macro_f1_population": float(np.std(repeat_macro_f1, ddof=0)),
                "minimum_repeat_per_class_recall": float(
                    np.min(np.asarray(repeat_recalls, dtype=np.float64))
                ),
                "fold_trajectories": fold_rows,
                "oof_predictions": repeat_predictions,
                "canonical_order_excludes_case_ids": True,
            }
        )

    selected, selection_decision = _select_candidate(candidate_results, schedule.tie_band_macro_f1)
    return {
        "schema_version": 1,
        "status": "complete",
        "scope": "isolated_supplied_train_only",
        "case_count": dataset.case_count,
        "class_counts": _class_counts(labels_original),
        "numeric_content_dataset_sha256": dataset.content_sha256(),
        "candidate_count": len(candidate_results),
        "candidate_results": candidate_results,
        "selection_decision": selection_decision,
        "selected_candidate_id": selected["candidate_id"],
        "selected_mean_repeat_oof_macro_f1": selected["mean_repeat_oof_macro_f1"],
        "selected_minimum_repeat_per_class_recall": selected["minimum_repeat_per_class_recall"],
        "case_ids_used_as_model_features_or_split_keys": False,
        "paths_filenames_or_enumeration_order_used": False,
        "ground_truth_masks_used_as_features": False,
        "official_validation_images_masks_labels_or_metrics_used": False,
        "test_data_used": False,
        "encoder_decoder_and_rescue_head_frozen": True,
        "head_oof_unbiased_end_to_end": False,
        "mandatory_disclosure": (
            "The frozen shared encoder and rescue head were previously trained on all "
            "252 subtype labels; head-only OOF comparison is not an unbiased "
            "end-to-end generalization estimate."
        ),
    }


def fit_selected_neural_head(
    dataset: NeuralBagDataset,
    tensors: NeuralBagTensors,
    lock: Mapping[str, Any],
    selection_audit: Mapping[str, Any],
    *,
    event_sink: TrainingEventSink | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Refit the selected v5 head on all 252 cases for exactly 150 epochs."""

    if (
        selection_audit.get("status") != "complete"
        or selection_audit.get("scope") != "isolated_supplied_train_only"
        or selection_audit.get("candidate_count") != 2
    ):
        raise ValueError("Final neural refit requires a complete selection audit")
    schedule = schedule_from_lock(lock)
    recomputed, recomputed_decision = _select_candidate(
        selection_audit["candidate_results"], schedule.tie_band_macro_f1
    )
    candidate_id = str(selection_audit["selected_candidate_id"])
    if candidate_id not in NEURAL_CANDIDATES:
        raise ValueError("Selection audit did not choose a locked neural candidate")
    if (
        candidate_id != recomputed["candidate_id"]
        or selection_audit.get("selection_decision") != recomputed_decision
        or selection_audit.get("selected_mean_repeat_oof_macro_f1")
        != recomputed["mean_repeat_oof_macro_f1"]
        or selection_audit.get("selected_minimum_repeat_per_class_recall")
        != recomputed["minimum_repeat_per_class_recall"]
    ):
        raise ValueError("Selection audit differs from the deterministic v5 decision")
    if selection_audit.get("numeric_content_dataset_sha256") != dataset.content_sha256():
        raise ValueError("Selection audit was produced from different neural bags")
    _verify_materialized_tensors(dataset, tensors)
    canonical_order = dataset.content_canonical_order()
    model, trajectory = _train_trajectory(
        candidate_id,
        tensors,
        canonical_order,
        trajectory_seed=schedule.final_seed,
        schedule=schedule,
        context={
            "candidate_id": candidate_id,
            "repeat_index": -1,
            "repeat_seed": schedule.final_seed,
            "fold_index": -1,
            "trajectory": "final_refit",
        },
        event_sink=event_sink,
    )
    logits = predict_neural_logits(
        model,
        tensors,
        np.arange(dataset.case_count),
        batch_size=schedule.batch_size,
    )
    resubstitution = _metrics(dataset.labels, logits.argmax(axis=1))
    return model, {
        "schema_version": 1,
        "status": "complete",
        "candidate_id": candidate_id,
        "numeric_content_dataset_sha256": dataset.content_sha256(),
        "canonical_content_order_used": True,
        "case_ids_used_as_model_features": False,
        "official_validation_or_test_used": False,
        "ground_truth_masks_used_as_features": False,
        "training_resubstitution_metrics_not_for_selection": resubstitution,
        **trajectory,
    }


__all__ = [
    "NeuralTrainingSchedule",
    "balanced_sample_indices",
    "configure_deterministic_execution",
    "evaluate_neural_candidates",
    "fit_selected_neural_head",
    "neural_state_sha256",
    "predict_neural_logits",
    "schedule_from_lock",
]
