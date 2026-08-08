"""Deterministic train-only core for the locked V6 spatial case heads.

The orchestration layer must supply an audited canonical row order, model
initialization seeds, and sampler seed bases.  This module intentionally does
not invent those prospective run identities before they are separately frozen.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import Tensor, nn

from pancreas_multitask.v6_case_head import (
    CLASS_COUNT,
    MORPHOLOGY_FEATURES,
    MORPHOLOGY_IQR_FLOOR,
    V6_CASE_CANDIDATES,
    V6_SPATIAL_CANDIDATE,
    V6_SPATIAL_MORPH_CANDIDATE,
    V6CaseHead,
    build_v6_case_head,
    v6_case_head_parameter_count,
    validate_v6_case_inputs,
)

V6_REPEAT_SEEDS: Final = (20260806, 20260807, 20260808)
V6_FINAL_SEEDS: Final = (20260806, 20260807, 20260808)
TrainingEventSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class V6TrainingSchedule:
    folds: int = 5
    repeat_seeds: tuple[int, ...] = V6_REPEAT_SEEDS
    maximum_inner_epochs: int = 120
    batch_size: int = 16
    samples_per_epoch: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    warmup_epochs: int = 5
    gradient_clip_norm: float = 5.0
    label_smoothing: float = 0.05
    correction_penalty_weight: float = 0.01
    candidate_tie_band: float = 0.01

    def __post_init__(self) -> None:
        if self.folds < 2 or not self.repeat_seeds:
            raise ValueError("V6 repeated stratification requires folds and repeat seeds")
        if self.maximum_inner_epochs < self.warmup_epochs + 2 or self.warmup_epochs < 1:
            raise ValueError("V6 schedule requires warmup followed by cosine epochs")
        if self.batch_size < 1 or self.samples_per_epoch < 1:
            raise ValueError("V6 batch and sample counts must be positive")
        if self.samples_per_epoch % self.batch_size:
            raise ValueError("V6 samples per epoch must divide evenly into batches")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("V6 optimizer settings are invalid")
        if self.gradient_clip_norm <= 0:
            raise ValueError("V6 gradient clipping must be positive")
        if not 0 <= self.label_smoothing < 1 or self.correction_penalty_weight < 0:
            raise ValueError("V6 loss settings are invalid")
        if self.candidate_tie_band != 0.01:
            raise ValueError("The prospectively locked V6 candidate tie band is exactly 0.01")


LOCKED_V6_SCHEDULE: Final = V6TrainingSchedule()


@dataclass(frozen=True, slots=True)
class MorphologyScalerStatistics:
    """CPU float32 buffers fitted only on a caller-selected training partition."""

    median: Tensor
    iqr: Tensor

    def __post_init__(self) -> None:
        if self.median.device.type != "cpu" or self.iqr.device.type != "cpu":
            raise ValueError("Stored morphology scaler buffers must reside on CPU")
        if self.median.dtype != torch.float32 or self.iqr.dtype != torch.float32:
            raise TypeError("Stored morphology scaler buffers must use float32")
        if self.median.shape != (MORPHOLOGY_FEATURES,) or self.iqr.shape != (
            MORPHOLOGY_FEATURES,
        ):
            raise ValueError("Morphology scaler buffers must each contain 46 values")
        if not torch.isfinite(self.median).all() or not torch.isfinite(self.iqr).all():
            raise ValueError("Morphology scaler buffers must be finite")
        if torch.any(self.iqr < MORPHOLOGY_IQR_FLOOR):
            raise ValueError("Stored morphology IQR must already include the 1e-6 floor")


@dataclass(frozen=True, slots=True)
class V6CaseTensors:
    """Materialized float32 whole-volume features; identifiers are absent by design."""

    spatial: Tensor
    coordinates: Tensor
    case_lesion_mass: Tensor
    stage5_global_mean: Tensor
    stage5_lesion_weighted_mean: Tensor
    rescue_mean_logits: Tensor
    rescue_mean_probabilities: Tensor
    morphology: Tensor
    labels: Tensor

    def __post_init__(self) -> None:
        validate_v6_case_inputs(
            self.spatial,
            self.coordinates,
            self.case_lesion_mass,
            self.stage5_global_mean,
            self.stage5_lesion_weighted_mean,
            self.rescue_mean_logits,
            self.rescue_mean_probabilities,
            self.morphology,
            require_morphology=True,
        )
        if self.labels.dtype != torch.long or self.labels.shape != (self.case_count,):
            raise TypeError("V6 labels must be a one-dimensional torch.long tensor")
        if self.labels.device != self.spatial.device:
            raise ValueError("V6 labels and case features must share a device")
        if bool(((self.labels < 0) | (self.labels >= CLASS_COUNT)).any().item()):
            raise ValueError("V6 labels must be in {0, 1, 2}")

    @property
    def case_count(self) -> int:
        return int(self.spatial.shape[0])

    @property
    def device(self) -> torch.device:
        return self.spatial.device

    def batch(
        self,
        indices: Sequence[int] | np.ndarray,
        candidate_id: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        if candidate_id not in V6_CASE_CANDIDATES:
            raise ValueError(f"Unknown locked V6 candidate: {candidate_id}")
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1 or requested.size < 1:
            raise ValueError("A V6 tensor batch requires one or more one-dimensional indices")
        if np.any(requested < 0) or np.any(requested >= self.case_count):
            raise IndexError("V6 tensor batch contains an out-of-range case index")
        index = torch.as_tensor(requested, dtype=torch.long, device=self.device)
        morphology = (
            self.morphology.index_select(0, index)
            if candidate_id == V6_SPATIAL_MORPH_CANDIDATE
            else None
        )
        return (
            self.spatial.index_select(0, index),
            self.coordinates.index_select(0, index),
            self.case_lesion_mass.index_select(0, index),
            self.stage5_global_mean.index_select(0, index),
            self.stage5_lesion_weighted_mean.index_select(0, index),
            self.rescue_mean_logits.index_select(0, index),
            self.rescue_mean_probabilities.index_select(0, index),
            morphology,
            self.labels.index_select(0, index),
        )


@dataclass(frozen=True, slots=True)
class V6OuterSplit:
    repeat_index: int
    repeat_seed: int
    fold_index: int
    train_indices: np.ndarray
    held_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class V6EpochEvaluation:
    epoch_index: int
    macro_f1: float
    nll: float

    def __post_init__(self) -> None:
        if self.epoch_index < 0:
            raise ValueError("Epoch indices are zero based and non-negative")
        if not math.isfinite(self.macro_f1) or not 0 <= self.macro_f1 <= 1:
            raise ValueError("Macro-F1 must be finite in [0, 1]")
        if not math.isfinite(self.nll) or self.nll < 0:
            raise ValueError("NLL must be finite and non-negative")

    @property
    def epoch_count(self) -> int:
        return self.epoch_index + 1


@dataclass(frozen=True, slots=True)
class V6InnerCandidateDecision:
    candidate_id: str
    selected_epoch_count: int
    macro_f1: float
    nll: float

    def __post_init__(self) -> None:
        if self.candidate_id not in V6_CASE_CANDIDATES:
            raise ValueError("Inner decision names an ineligible V6 candidate")
        if self.selected_epoch_count < 1:
            raise ValueError("Selected epoch count must be positive")
        if not math.isfinite(self.macro_f1) or not 0 <= self.macro_f1 <= 1:
            raise ValueError("Inner candidate macro-F1 must be finite in [0, 1]")
        if not math.isfinite(self.nll) or self.nll < 0:
            raise ValueError("Inner candidate NLL must be finite and non-negative")


def configure_v6_deterministic_execution(device: torch.device) -> dict[str, Any]:
    """Enable deterministic float32 execution before CUDA is initialized."""

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise ValueError("V6 training requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if device.type == "cuda" and torch.cuda.is_initialized():
        raise RuntimeError("V6 determinism must be configured before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA V6 training was requested but is unavailable")
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "precision": "float32",
        "mixed_precision": False,
    }


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def v6_state_sha256(model: nn.Module) -> str:
    """Hash the exact names, metadata, and bytes of model parameters and buffers."""

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


def fit_morphology_scaler(
    morphology: Tensor,
    training_indices: Sequence[int] | np.ndarray,
) -> MorphologyScalerStatistics:
    """Fit the clarified CPU-float64 linear median/IQR transform on train only."""

    if morphology.ndim != 2 or morphology.shape[1] != MORPHOLOGY_FEATURES:
        raise ValueError("Morphology matrix must have shape (cases, 46)")
    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 1 or np.unique(indices).size != indices.size:
        raise ValueError("Scaler fitting requires unique one-dimensional training indices")
    if np.any(indices < 0) or np.any(indices >= morphology.shape[0]):
        raise IndexError("Scaler fitting received an out-of-range training index")
    values = morphology.detach().cpu().to(torch.float64)[torch.from_numpy(indices)]
    if not torch.isfinite(values).all():
        raise ValueError("Morphology scaler cannot fit non-finite values")
    quantiles = torch.quantile(
        values,
        torch.tensor([0.25, 0.50, 0.75], dtype=torch.float64),
        dim=0,
        interpolation="linear",
    )
    median = quantiles[1].to(torch.float32).contiguous()
    iqr = (quantiles[2] - quantiles[0]).clamp_min(MORPHOLOGY_IQR_FLOOR)
    return MorphologyScalerStatistics(median, iqr.to(torch.float32).contiguous())


def balanced_v6_sample_indices(
    labels: np.ndarray,
    available_indices: Sequence[int] | np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Draw class-balanced cases with replacement using an isolated CPU RNG."""

    targets = np.asarray(labels, dtype=np.int64)
    available = np.asarray(available_indices, dtype=np.int64)
    if targets.ndim != 1 or available.ndim != 1 or available.size < 1:
        raise ValueError("Balanced V6 sampling requires one-dimensional labels and indices")
    if sample_count < 1:
        raise ValueError("Balanced V6 sample count must be positive")
    if np.any(available < 0) or np.any(available >= targets.size):
        raise IndexError("Balanced V6 sampler received an out-of-range index")
    available_labels = targets[available]
    counts = np.bincount(available_labels, minlength=CLASS_COUNT)
    if counts.shape != (CLASS_COUNT,) or np.any(counts == 0):
        raise ValueError("Every V6 training partition must contain all three classes")
    weights = 1.0 / counts[available_labels].astype(np.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    local = torch.multinomial(
        torch.from_numpy(weights),
        int(sample_count),
        replacement=True,
        generator=generator,
    ).numpy()
    return available[local].astype(np.int64, copy=False)


def _validate_audited_order(labels: np.ndarray, audited_order: Sequence[int]) -> np.ndarray:
    order = np.asarray(audited_order, dtype=np.int64)
    expected = np.arange(labels.size, dtype=np.int64)
    if order.shape != expected.shape or not np.array_equal(np.sort(order), expected):
        raise ValueError("Caller-supplied audited order must be a permutation of all rows")
    return order


def make_v6_outer_splits(
    labels: Sequence[int] | np.ndarray,
    *,
    audited_order: Sequence[int] | np.ndarray,
    folds: int = 5,
    repeat_seeds: Sequence[int] = V6_REPEAT_SEEDS,
) -> tuple[V6OuterSplit, ...]:
    """Build repeated folds from an explicit audited order, never identifiers."""

    targets = np.asarray(labels, dtype=np.int64)
    if targets.ndim != 1 or set(targets.tolist()) != {0, 1, 2}:
        raise ValueError("V6 outer splitting requires one-dimensional three-class labels")
    order = _validate_audited_order(targets, audited_order)
    ordered_labels = targets[order]
    rows: list[V6OuterSplit] = []
    for repeat_index, repeat_seed in enumerate(repeat_seeds):
        splitter = StratifiedKFold(
            n_splits=int(folds),
            shuffle=True,
            random_state=int(repeat_seed),
        )
        for fold_index, (train_local, held_local) in enumerate(
            splitter.split(np.zeros(order.size), ordered_labels)
        ):
            rows.append(
                V6OuterSplit(
                    repeat_index,
                    int(repeat_seed),
                    fold_index,
                    order[train_local],
                    order[held_local],
                )
            )
    if len(rows) != len(tuple(repeat_seeds)) * int(folds):
        raise RuntimeError("V6 outer splitter produced an unexpected number of folds")
    return tuple(rows)


def make_v6_inner_split(
    labels: Sequence[int] | np.ndarray,
    outer_train_indices: Sequence[int] | np.ndarray,
    *,
    repeat_seed: int,
    fold_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the clarified deterministic stratified 80/20 split."""

    targets = np.asarray(labels, dtype=np.int64)
    outer = np.asarray(outer_train_indices, dtype=np.int64)
    if targets.ndim != 1 or outer.ndim != 1 or outer.size < 1:
        raise ValueError("V6 inner splitting requires one-dimensional labels and outer rows")
    if np.unique(outer).size != outer.size:
        raise ValueError("V6 outer training indices must be unique")
    if np.any(outer < 0) or np.any(outer >= targets.size):
        raise IndexError("V6 inner split received an out-of-range outer index")
    inner_seed = int(repeat_seed) + 1000 * int(fold_index) + 77
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=inner_seed,
    )
    train_local, validation_local = next(
        splitter.split(np.zeros(outer.size), targets[outer])
    )
    return outer[train_local], outer[validation_local], inner_seed


def v6_learning_rate(
    epoch_index: int,
    *,
    maximum_epochs: int = 120,
    base_learning_rate: float = 1e-4,
    warmup_epochs: int = 5,
) -> float:
    """Return 1/5..5/5 warmup, then cosine with the last epoch exactly zero."""

    if maximum_epochs < warmup_epochs + 2 or warmup_epochs < 1:
        raise ValueError("V6 learning-rate schedule needs warmup and cosine epochs")
    if epoch_index < 0 or epoch_index >= maximum_epochs:
        raise IndexError("V6 epoch index is outside the configured trajectory")
    if base_learning_rate <= 0:
        raise ValueError("V6 base learning rate must be positive")
    if epoch_index < warmup_epochs:
        multiplier = (epoch_index + 1) / warmup_epochs
    else:
        cosine_position = (epoch_index - warmup_epochs) / (
            maximum_epochs - warmup_epochs - 1
        )
        multiplier = 0.5 * (1.0 + math.cos(math.pi * cosine_position))
    return float(base_learning_rate * multiplier)


def v6_classification_loss(
    logits: Tensor,
    labels: Tensor,
    bounded_correction: Tensor,
    *,
    label_smoothing: float = 0.05,
    correction_penalty_weight: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, unweighted smoothed CE, and bounded-correction penalty."""

    if logits.ndim != 2 or logits.shape[1] != CLASS_COUNT or labels.shape != (
        logits.shape[0],
    ):
        raise ValueError("V6 loss expects logits (batch, 3) and aligned labels")
    if bounded_correction.shape != logits.shape:
        raise ValueError("V6 bounded correction must align exactly with anchored logits")
    cross_entropy = F.cross_entropy(
        logits,
        labels,
        weight=None,
        label_smoothing=label_smoothing,
    )
    correction_penalty = correction_penalty_weight * bounded_correction.square().mean()
    return cross_entropy + correction_penalty, cross_entropy, correction_penalty


def _log_softmax_float64(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != CLASS_COUNT or not np.isfinite(scores).all():
        raise ValueError("V6 logits must be a finite (cases, 3) matrix")
    shifted = scores - scores.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def v6_prediction_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    """Compute locked three-class macro-F1, recalls, and float64 NLL."""

    targets = np.asarray(labels, dtype=np.int64)
    log_probabilities = _log_softmax_float64(logits)
    if targets.shape != (log_probabilities.shape[0],):
        raise ValueError("V6 metric labels and logits are not aligned")
    predictions = log_probabilities.argmax(axis=1)
    recalls = recall_score(
        targets,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": [float(value) for value in recalls],
        "minimum_per_class_recall": float(np.min(recalls)),
        "nll": float(-log_probabilities[np.arange(targets.size), targets].mean()),
    }


def select_v6_inner_epoch(
    evaluations: Sequence[V6EpochEvaluation],
) -> V6EpochEvaluation:
    """Select higher macro-F1, then lower NLL, then the earlier epoch."""

    if not evaluations:
        raise ValueError("V6 inner epoch selection requires at least one evaluation")
    epoch_indices = [row.epoch_index for row in evaluations]
    if len(epoch_indices) != len(set(epoch_indices)):
        raise ValueError("V6 inner epoch history contains duplicate epochs")
    return max(evaluations, key=lambda row: (row.macro_f1, -row.nll, -row.epoch_index))


def select_v6_inner_candidate(
    decisions: Sequence[V6InnerCandidateDecision],
    *,
    tie_band: float = 0.01,
) -> tuple[V6InnerCandidateDecision, str]:
    """Select macro-F1 outside 0.01; inside it, prefer locked no-morphology."""

    if tie_band != 0.01:
        raise ValueError("The locked V6 candidate tie band is exactly 0.01")
    if len(decisions) != 2 or {row.candidate_id for row in decisions} != set(
        V6_CASE_CANDIDATES
    ):
        raise ValueError("V6 inner candidate selection requires exactly both candidates")
    by_id = {row.candidate_id: row for row in decisions}
    no_morph = by_id[V6_SPATIAL_CANDIDATE]
    morph = by_id[V6_SPATIAL_MORPH_CANDIDATE]
    difference = abs(no_morph.macro_f1 - morph.macro_f1)
    outside_tie_band = difference > tie_band and not math.isclose(
        difference,
        tie_band,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    if outside_tie_band:
        selected = max(decisions, key=lambda row: row.macro_f1)
        basis = "higher_macro_f1_outside_0.01_tie_band"
    else:
        selected = no_morph
        basis = "no_morphology_preferred_within_0.01_before_nll"
    return selected, basis


def round_half_up_positive(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("V6 final epoch rounding requires a positive finite value")
    return math.floor(value + 0.5)


def select_v6_final_family_and_epoch(
    selected_inner_decisions: Sequence[V6InnerCandidateDecision],
) -> tuple[str, int, dict[str, Any]]:
    """Choose majority family over 15 decisions and its round-half-up median epoch."""

    if len(selected_inner_decisions) != 15:
        raise ValueError("V6 final selection requires exactly 15 inner fold decisions")
    counts = {
        candidate: sum(row.candidate_id == candidate for row in selected_inner_decisions)
        for candidate in V6_CASE_CANDIDATES
    }
    if counts[V6_SPATIAL_MORPH_CANDIDATE] > counts[V6_SPATIAL_CANDIDATE]:
        family = V6_SPATIAL_MORPH_CANDIDATE
    else:
        family = V6_SPATIAL_CANDIDATE
    eligible_epochs = sorted(
        row.selected_epoch_count
        for row in selected_inner_decisions
        if row.candidate_id == family
    )
    median = float(np.median(np.asarray(eligible_epochs, dtype=np.float64)))
    epoch_count = round_half_up_positive(median)
    return family, epoch_count, {
        "decision_counts": counts,
        "selected_family_epoch_counts": eligible_epochs,
        "median_selected_epoch": median,
        "rounding": "round_half_up_for_positive_epoch_counts",
    }


@torch.inference_mode()
def predict_v6_logits(
    model: V6CaseHead,
    tensors: V6CaseTensors,
    indices: Sequence[int] | np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Predict raw anchored logits in caller-supplied audited row order."""

    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim != 1:
        raise ValueError("V6 prediction indices must be one-dimensional")
    if batch_size < 1:
        raise ValueError("V6 prediction batch size must be positive")
    model.eval()
    rows: list[np.ndarray] = []
    for start in range(0, requested.size, batch_size):
        batch = tensors.batch(requested[start : start + batch_size], model.candidate_id)
        logits = model(*batch[:-1])
        if logits.shape != (batch[-1].shape[0], CLASS_COUNT) or not torch.isfinite(logits).all():
            raise FloatingPointError("V6 head emitted invalid logits")
        rows.append(logits.detach().cpu().to(torch.float32).numpy())
    if not rows:
        return np.empty((0, CLASS_COUNT), dtype=np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def train_v6_trajectory(
    candidate_id: str,
    tensors: V6CaseTensors,
    training_indices: Sequence[int] | np.ndarray,
    *,
    epochs: int,
    model_seed: int,
    sampler_seed_base: int,
    schedule: V6TrainingSchedule = LOCKED_V6_SCHEDULE,
    evaluation_indices: Sequence[int] | np.ndarray | None = None,
    event_sink: TrainingEventSink | None = None,
) -> tuple[V6CaseHead, dict[str, Any]]:
    """Train one caller-identified trajectory without early stopping.

    ``model_seed`` and ``sampler_seed_base`` are explicit orchestration inputs.
    The sampler seed for zero-based epoch ``e`` is ``sampler_seed_base + e``.
    The caller must freeze those values before any eligible training run.
    """

    if candidate_id not in V6_CASE_CANDIDATES:
        raise ValueError(f"Unknown locked V6 candidate: {candidate_id}")
    train = np.asarray(training_indices, dtype=np.int64)
    if train.ndim != 1 or train.size < 1 or np.unique(train).size != train.size:
        raise ValueError("V6 trajectory requires unique one-dimensional training rows")
    if np.any(train < 0) or np.any(train >= tensors.case_count):
        raise IndexError("V6 trajectory received an out-of-range training row")
    if epochs < 1 or epochs > schedule.maximum_inner_epochs:
        raise ValueError("V6 trajectory epochs must be within the locked maximum")
    evaluation = None
    if evaluation_indices is not None:
        evaluation = np.asarray(evaluation_indices, dtype=np.int64)
        if evaluation.ndim != 1 or evaluation.size < 1:
            raise ValueError("V6 evaluation rows must be a non-empty vector")
        if np.intersect1d(train, evaluation).size:
            raise ValueError("V6 training and evaluation partitions must be disjoint")

    scaler = None
    if candidate_id == V6_SPATIAL_MORPH_CANDIDATE:
        scaler = fit_morphology_scaler(tensors.morphology, train)
    _seed_everything(model_seed)
    model = build_v6_case_head(
        candidate_id,
        morphology_median=None if scaler is None else scaler.median,
        morphology_iqr=None if scaler is None else scaler.iqr,
    ).to(tensors.device)
    initial_state_hash = v6_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.learning_rate,
        weight_decay=schedule.weight_decay,
    )
    labels_cpu = tensors.labels.detach().cpu().numpy().astype(np.int64, copy=False)
    epoch_history: list[dict[str, Any]] = []
    for epoch_index in range(epochs):
        learning_rate = v6_learning_rate(
            epoch_index,
            maximum_epochs=schedule.maximum_inner_epochs,
            base_learning_rate=schedule.learning_rate,
            warmup_epochs=schedule.warmup_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        sampler_seed = int(sampler_seed_base) + epoch_index
        sampled = balanced_v6_sample_indices(
            labels_cpu,
            train,
            sample_count=schedule.samples_per_epoch,
            seed=sampler_seed,
        )
        model.train()
        total_sum = 0.0
        cross_entropy_sum = 0.0
        correction_penalty_sum = 0.0
        gradient_norm_sum = 0.0
        maximum_gradient_norm = 0.0
        for start in range(0, sampled.size, schedule.batch_size):
            batch = tensors.batch(sampled[start : start + schedule.batch_size], candidate_id)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_with_details(*batch[:-1])
            total, cross_entropy, correction_penalty = v6_classification_loss(
                output.logits,
                batch[-1],
                output.bounded_correction,
                label_smoothing=schedule.label_smoothing,
                correction_penalty_weight=schedule.correction_penalty_weight,
            )
            if not torch.isfinite(total):
                raise FloatingPointError("V6 training produced a non-finite loss")
            total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), schedule.gradient_clip_norm
                ).item()
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("V6 training produced non-finite gradients")
            optimizer.step()
            count = int(batch[-1].numel())
            total_sum += float(total.detach().item()) * count
            cross_entropy_sum += float(cross_entropy.detach().item()) * count
            correction_penalty_sum += float(correction_penalty.detach().item()) * count
            gradient_norm_sum += gradient_norm
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        batch_count = schedule.samples_per_epoch // schedule.batch_size
        row: dict[str, Any] = {
            "epoch_index": epoch_index,
            "epoch_count": epoch_index + 1,
            "learning_rate": learning_rate,
            "sampler_seed": sampler_seed,
            "mean_total_loss": total_sum / schedule.samples_per_epoch,
            "mean_unweighted_smoothed_cross_entropy": (
                cross_entropy_sum / schedule.samples_per_epoch
            ),
            "mean_bounded_correction_penalty": (
                correction_penalty_sum / schedule.samples_per_epoch
            ),
            "mean_pre_clip_gradient_norm": gradient_norm_sum / batch_count,
            "maximum_pre_clip_gradient_norm": maximum_gradient_norm,
            "sampled_class_counts": {
                str(label): int(np.sum(labels_cpu[sampled] == label)) for label in range(3)
            },
        }
        if evaluation is not None:
            evaluation_logits = predict_v6_logits(
                model,
                tensors,
                evaluation,
                batch_size=schedule.batch_size,
            )
            metrics = v6_prediction_metrics(labels_cpu[evaluation], evaluation_logits)
            row["evaluation"] = metrics
        epoch_history.append(row)
        if event_sink is not None:
            event_sink({"candidate_id": candidate_id, **row})

    return model, {
        "candidate_id": candidate_id,
        "model_seed": int(model_seed),
        "sampler_seed_base": int(sampler_seed_base),
        "sampler_seed_formula": "caller_seed_base_plus_zero_based_epoch",
        "caller_supplied_audited_row_order_required": True,
        "initial_state_sha256": initial_state_hash,
        "final_state_sha256": v6_state_sha256(model),
        "trainable_parameter_count": v6_case_head_parameter_count(candidate_id),
        "epochs_completed": epochs,
        "early_stopping_used": False,
        "mixed_precision_used": False,
        "class_weighted_loss_used": False,
        "focal_loss_used": False,
        "smote_used": False,
        "balanced_sampling_with_replacement_used": True,
        "morphology_scaler_fit_on_training_partition_only": scaler is not None,
        "epoch_history": epoch_history,
    }


def ensemble_v6_final_logits(final_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Average exactly three raw-logit models, then apply one softmax and zero offsets."""

    if final_logits.ndim != 3 or final_logits.shape[0] != 3 or final_logits.shape[2] != 3:
        raise ValueError("V6 final ensemble requires logits with shape (3, cases, 3)")
    if not torch.isfinite(final_logits).all():
        raise ValueError("V6 final ensemble logits must be finite")
    mean_logits = final_logits.to(torch.float32).mean(dim=0)
    return mean_logits, torch.softmax(mean_logits, dim=1)


def evaluate_v6_train_only_screen(
    labels: Sequence[int] | np.ndarray,
    repeat_oof_logits: np.ndarray,
    coverage_counts: np.ndarray,
    *,
    resubstitution_macro_f1: float,
) -> dict[str, Any]:
    """Apply the frozen pre-validation screen to three complete OOF repeats."""

    targets = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(repeat_oof_logits)
    coverage = np.asarray(coverage_counts)
    expected = (len(V6_REPEAT_SEEDS), targets.size, CLASS_COUNT)
    if logits.shape != expected or coverage.shape != expected[:2]:
        raise ValueError("V6 OOF screen requires three aligned complete repeat matrices")
    finite = bool(np.isfinite(logits).all())
    complete = bool(np.all(coverage == 1))
    if not finite:
        raise ValueError("V6 OOF screen cannot evaluate non-finite logits")
    repeat_metrics = [v6_prediction_metrics(targets, logits[index]) for index in range(3)]
    repeat_macro_f1 = [float(row["macro_f1"]) for row in repeat_metrics]
    minimum_recall = min(
        value for row in repeat_metrics for value in row["per_class_recall"]
    )
    mean_macro_f1 = float(np.mean(repeat_macro_f1))
    minimum_repeat_macro_f1 = float(min(repeat_macro_f1))
    if not math.isfinite(resubstitution_macro_f1) or not 0 <= resubstitution_macro_f1 <= 1:
        raise ValueError("V6 resubstitution macro-F1 must be finite in [0, 1]")
    passed = bool(
        mean_macro_f1 >= 0.60
        and minimum_repeat_macro_f1 >= 0.57
        and minimum_recall >= 0.50
        and finite
        and complete
    )
    return {
        "repeat_oof_metrics": repeat_metrics,
        "mean_across_repeat_oof_macro_f1": mean_macro_f1,
        "minimum_repeat_oof_macro_f1": minimum_repeat_macro_f1,
        "minimum_repeat_per_class_recall": float(minimum_recall),
        "all_predictions_and_logits_finite": finite,
        "complete_fold_and_case_coverage": complete,
        "resubstitution_macro_f1": float(resubstitution_macro_f1),
        "resubstitution_minus_mean_oof_macro_f1": float(
            resubstitution_macro_f1 - mean_macro_f1
        ),
        "passed_train_only_screen": passed,
        "validation_attempt_permitted": passed,
    }


__all__ = [
    "LOCKED_V6_SCHEDULE",
    "V6_FINAL_SEEDS",
    "V6_REPEAT_SEEDS",
    "MorphologyScalerStatistics",
    "V6CaseTensors",
    "V6EpochEvaluation",
    "V6InnerCandidateDecision",
    "V6OuterSplit",
    "V6TrainingSchedule",
    "balanced_v6_sample_indices",
    "configure_v6_deterministic_execution",
    "ensemble_v6_final_logits",
    "evaluate_v6_train_only_screen",
    "fit_morphology_scaler",
    "make_v6_inner_split",
    "make_v6_outer_splits",
    "predict_v6_logits",
    "round_half_up_positive",
    "select_v6_final_family_and_epoch",
    "select_v6_inner_candidate",
    "select_v6_inner_epoch",
    "train_v6_trajectory",
    "v6_classification_loss",
    "v6_learning_rate",
    "v6_prediction_metrics",
    "v6_state_sha256",
]
