"""Training-only utilities for an optional frozen-backbone head rescue.

The main multi-task experiment remains unchanged.  These helpers support a
strictly post-training, fixed-schedule fallback in which only the classification
pool and head are reset and optimized.  They are kept independent of nnU-Net's
data-loader implementation so the safety-critical pieces can be unit tested on
CPU with a toy network.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

CLASSIFICATION_PARAMETER_PREFIXES = (
    "classification_pool.",
    "classification_head.",
)
RESCUE_SCHEMA_VERSION = 1
ACTIVATION_AUDIT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RescueSchedule:
    """Immutable optimization choices for the conditional rescue."""

    epochs: int = 30
    iterations_per_epoch: int = 125
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    nonlesion_patch_weight: float = 0.25
    reset_seed: int = 20260806

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.iterations_per_epoch < 1:
            raise ValueError("epochs and iterations_per_epoch must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0 <= self.label_smoothing <= 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5]")
        if not 0 < self.nonlesion_patch_weight <= 1:
            raise ValueError("nonlesion_patch_weight must be in (0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RescueStepResult:
    """Training-only batch measurements; none estimate generalization."""

    loss: float
    correct: int
    sample_count: int
    lesion_patch_count: int
    gradient_norm_before_clip: float
    target_counts: tuple[int, ...]
    prediction_counts: tuple[int, ...]


def classification_activation_audit(
    classification_losses: Sequence[float],
    classification_accuracies: Sequence[float],
) -> dict[str, Any]:
    """Apply the predeclared epoch-40/50 rules to training metrics only."""

    losses = np.asarray(classification_losses, dtype=np.float64)
    accuracies = np.asarray(classification_accuracies, dtype=np.float64)
    if losses.ndim != 1 or accuracies.ndim != 1 or losses.shape != accuracies.shape:
        raise ValueError("Training CE and accuracy must be equal-length 1D series")
    if losses.size <= 50:
        raise ValueError("Activation audit requires completed training metrics through epoch 50")
    if not np.isfinite(losses[:51]).all() or not np.isfinite(accuracies[:51]).all():
        raise ValueError("Activation metrics through epoch 50 must all be finite")

    def window(audit_epoch: int) -> dict[str, Any]:
        start = audit_epoch - 9
        epoch_losses = losses[start : audit_epoch + 1]
        epoch_accuracies = accuracies[start : audit_epoch + 1]
        slope = float(np.polyfit(np.arange(10, dtype=np.float64), epoch_losses, 1)[0])
        return {
            "audit_epoch": audit_epoch,
            "window_epochs": list(range(start, audit_epoch + 1)),
            "training_classification_ce_values": [float(value) for value in epoch_losses],
            "training_patch_accuracy_values": [float(value) for value in epoch_accuracies],
            "mean_training_classification_ce": float(np.mean(epoch_losses)),
            "mean_training_patch_accuracy": float(np.mean(epoch_accuracies)),
            "classification_ce_ols_slope_per_epoch": slope,
        }

    epoch_40 = window(40)
    epoch_40_conditions = {
        "ce_at_least_1_05": epoch_40["mean_training_classification_ce"] >= 1.05,
        "accuracy_at_most_0_42": epoch_40["mean_training_patch_accuracy"] <= 0.42,
        "ce_slope_at_least_negative_0_001": (
            epoch_40["classification_ce_ols_slope_per_epoch"] >= -0.001
        ),
    }
    epoch_40_trigger = all(epoch_40_conditions.values())
    epoch_40["conditions"] = epoch_40_conditions
    epoch_40["triggered"] = epoch_40_trigger

    epoch_50 = window(50)
    epoch_50_conditions = {
        "ce_above_1_03": epoch_50["mean_training_classification_ce"] > 1.03,
        "accuracy_below_0_45": epoch_50["mean_training_patch_accuracy"] < 0.45,
    }
    epoch_50_trigger = any(epoch_50_conditions.values())
    epoch_50["conditions"] = epoch_50_conditions
    epoch_50["triggered"] = epoch_50_trigger

    activation_approved = bool(epoch_40_trigger or epoch_50_trigger)
    decision_epoch = 40 if epoch_40_trigger else (50 if epoch_50_trigger else None)
    return {
        "activation_approved": activation_approved,
        "decision_epoch": decision_epoch,
        "metric_scope": "checkpoint_training_logging_only",
        "epoch_40": epoch_40,
        "epoch_50_hard_audit": epoch_50,
        "validation_metrics_read": False,
        "validation_used_for_activation": False,
    }


def validate_activation_audit(
    audit: Mapping[str, Any],
    *,
    source_checkpoint_sha256: str,
) -> None:
    """Require an affirmative train-only audit bound to the exact source file."""

    if audit.get("schema_version") != ACTIVATION_AUDIT_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing activation-audit schema version")
    if audit.get("source_checkpoint_sha256") != source_checkpoint_sha256:
        raise ValueError("Activation audit is bound to a different source checkpoint")
    if audit.get("source_checkpoint_name") != "checkpoint_final.pth":
        raise ValueError("Activation audit source must be checkpoint_final.pth")
    if audit.get("checkpoint_current_epoch") != 200:
        raise ValueError("Activation audit does not describe a completed 200-epoch checkpoint")
    if audit.get("training_logging_epoch_count") != 200:
        raise ValueError("Activation audit does not contain exactly 200 training epochs")
    if audit.get("metric_scope") != "checkpoint_training_logging_only":
        raise ValueError("Activation audit is not restricted to checkpoint training logging")
    if audit.get("validation_used_for_activation") is not False:
        raise ValueError("Activation audit does not prohibit validation-driven activation")
    if audit.get("activation_approved") is not True:
        raise ValueError("Train-only activation audit did not approve the rescue")


def _length_prefixed(hasher: Any, value: str | bytes) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Hash a file without loading it into memory."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def case_id_sha256(case_ids: Sequence[str]) -> str:
    """Hash a sorted, duplicate-free case-ID collection unambiguously."""

    normalized = sorted(str(case_id) for case_id in case_ids)
    hasher = hashlib.sha256()
    for case_id in normalized:
        _length_prefixed(hasher, case_id)
    return hasher.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Hash tensor names, metadata, and exact bytes in a module state dict."""

    hasher = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        _length_prefixed(hasher, name)
        _length_prefixed(hasher, str(value.dtype))
        _length_prefixed(hasher, json.dumps(list(value.shape), separators=(",", ":")))
        raw_bytes = value.view(torch.uint8).numpy().tobytes()
        _length_prefixed(hasher, raw_bytes)
    return hasher.hexdigest()


def component_hashes(model: nn.Module) -> dict[str, str]:
    """Fingerprint the immutable backbone and mutable classification path."""

    for attribute in ("encoder", "decoder", "classification_pool", "classification_head"):
        if not isinstance(getattr(model, attribute, None), nn.Module):
            raise TypeError(f"model must expose an nn.Module named {attribute!r}")
    classification = nn.ModuleDict(
        {
            "pool": model.classification_pool,
            "head": model.classification_head,
        }
    )
    return {
        "encoder": module_state_sha256(model.encoder),
        "decoder": module_state_sha256(model.decoder),
        "classification": module_state_sha256(classification),
    }


def classification_parameter_items(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    """Return every and only registered pool/head parameter exactly once."""

    items = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(CLASSIFICATION_PARAMETER_PREFIXES)
    )
    if not items:
        raise RuntimeError("No registered classification parameters were found")
    parameter_ids = [id(parameter) for _, parameter in items]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("A classification parameter is registered more than once")
    return items


def freeze_for_classification_rescue(
    model: nn.Module,
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Freeze encoder/decoder and enable gradients only for pool/head."""

    classification_items = classification_parameter_items(model)
    allowed_ids = {id(parameter) for _, parameter in classification_items}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in allowed_ids)

    actual_items = tuple(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if tuple(name for name, _ in actual_items) != tuple(name for name, _ in classification_items):
        raise RuntimeError("Trainable parameters are not limited to the classification path")

    model.eval()
    model.classification_pool.train()
    model.classification_head.train()
    return actual_items


def reset_classification_parameters(model: nn.Module, seed: int) -> None:
    """Seed and reset the declared classification modules only."""

    reset = getattr(model, "reset_classification_parameters", None)
    if not callable(reset):
        raise TypeError("model must implement reset_classification_parameters()")

    devices: list[int] = []
    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            devices.append(parameter.device.index or 0)
    with torch.random.fork_rng(devices=sorted(set(devices))):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        reset()


def validate_disjoint_split(
    training_case_ids: Sequence[str],
    validation_case_ids: Sequence[str],
    *,
    expected_training_cases: int,
    expected_validation_cases: int,
) -> dict[str, int | str | bool]:
    """Validate split cardinality/disjointness without opening validation data."""

    training = [str(case_id) for case_id in training_case_ids]
    validation = [str(case_id) for case_id in validation_case_ids]
    if len(training) != len(set(training)) or len(validation) != len(set(validation)):
        raise ValueError("Training and validation case IDs must each be unique")
    if len(training) != expected_training_cases:
        raise ValueError(f"Expected {expected_training_cases} training cases, got {len(training)}")
    if len(validation) != expected_validation_cases:
        raise ValueError(
            f"Expected {expected_validation_cases} validation cases, got {len(validation)}"
        )
    overlap = set(training) & set(validation)
    if overlap:
        raise ValueError(f"Training/validation overlap detected: {sorted(overlap)}")
    return {
        "training_case_count": len(training),
        "training_case_ids_sha256": case_id_sha256(training),
        "validation_case_count": len(validation),
        "validation_case_ids_sha256": case_id_sha256(validation),
        "split_disjoint": True,
        "validation_images_opened": False,
        "validation_batches_consumed": 0,
        "validation_used_for_gradients": False,
        "validation_used_for_stopping": False,
    }


def validate_frozen_split_manifest(
    manifest: Mapping[str, Any],
    training_case_ids: Sequence[str],
    validation_case_ids: Sequence[str],
) -> dict[str, int | str | bool]:
    """Bind the live nnU-Net split to the frozen pretraining manifest.

    The source package's supplied train/validation roles are recorded before
    training in ``split_manifest.json``. Cardinality and disjointness alone are
    insufficient because a different 252/36 partition would still pass those
    checks. This function therefore requires exact case membership and records
    hashes for both the live split and the manifest lists.
    """

    if manifest.get("schema_version") != 1:
        raise ValueError("Frozen split manifest has an unsupported schema_version")
    if manifest.get("source_splits_preserved") is not True:
        raise ValueError("Frozen split manifest does not preserve the supplied splits")

    def case_list(name: str) -> list[str]:
        raw_values = manifest.get(name)
        if not isinstance(raw_values, list) or not all(
            isinstance(value, str) for value in raw_values
        ):
            raise TypeError(f"Frozen split manifest field {name!r} must be a string list")
        values = [str(value) for value in raw_values]
        if len(values) != len(set(values)):
            raise ValueError(f"Frozen split manifest field {name!r} contains duplicates")
        return values

    manifest_training = case_list("train_case_ids")
    manifest_validation = case_list("validation_case_ids")
    manifest_planning = case_list("planning_case_ids")
    live_training = [str(case_id) for case_id in training_case_ids]
    live_validation = [str(case_id) for case_id in validation_case_ids]

    if set(manifest_training) & set(manifest_validation):
        raise ValueError("Frozen split manifest training/validation lists overlap")
    if sorted(manifest_planning) != sorted(manifest_training):
        raise ValueError("Frozen split manifest planning cases differ from its training cases")
    if sorted(live_training) != sorted(manifest_training):
        missing = sorted(set(manifest_training) - set(live_training))
        unexpected = sorted(set(live_training) - set(manifest_training))
        raise ValueError(
            "Live training split differs from frozen split manifest; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if sorted(live_validation) != sorted(manifest_validation):
        missing = sorted(set(manifest_validation) - set(live_validation))
        unexpected = sorted(set(live_validation) - set(manifest_validation))
        raise ValueError(
            "Live validation split differs from frozen split manifest; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    training_hash = case_id_sha256(live_training)
    validation_hash = case_id_sha256(live_validation)
    manifest_training_hash = case_id_sha256(manifest_training)
    manifest_validation_hash = case_id_sha256(manifest_validation)
    if training_hash != manifest_training_hash or validation_hash != manifest_validation_hash:
        raise RuntimeError("Frozen split manifest hash binding failed unexpectedly")

    return {
        "frozen_split_manifest_schema_version": 1,
        "frozen_source_splits_preserved": True,
        "frozen_manifest_training_case_count": len(manifest_training),
        "frozen_manifest_training_case_ids_sha256": manifest_training_hash,
        "frozen_manifest_validation_case_count": len(manifest_validation),
        "frozen_manifest_validation_case_ids_sha256": manifest_validation_hash,
        "matches_frozen_split_manifest": True,
    }


def _classification_loss(
    logits: Tensor,
    targets: Tensor,
    segmentation_target: Tensor,
    *,
    class_weights: Tensor,
    label_smoothing: float,
    nonlesion_patch_weight: float,
) -> tuple[Tensor, Tensor]:
    per_sample = F.cross_entropy(
        logits.float(),
        targets,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    lesion_present = (segmentation_target == 2).reshape(logits.shape[0], -1).any(dim=1)
    patch_weights = torch.where(
        lesion_present,
        torch.ones_like(per_sample),
        torch.full_like(per_sample, nonlesion_patch_weight),
    )
    loss = (per_sample * patch_weights).sum() / patch_weights.sum().clamp_min(1e-8)
    return loss, lesion_present


def classification_rescue_train_step(
    model: nn.Module,
    data: Tensor,
    targets: Tensor,
    segmentation_target: Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    class_weights: Tensor,
    schedule: RescueSchedule,
    scaler: Any | None = None,
    use_amp: bool = False,
    num_classes: int = 3,
) -> RescueStepResult:
    """Run one frozen-encoder, decoder-free classification update."""

    trainable_items = classification_parameter_items(model)
    if any(not parameter.requires_grad for _, parameter in trainable_items):
        raise RuntimeError("All classification parameters must require gradients")
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("Encoder parameters must be frozen")
    if any(parameter.requires_grad for parameter in model.decoder.parameters()):
        raise RuntimeError("Decoder parameters must be frozen")

    optimizer.zero_grad(set_to_none=True)
    context = (
        torch.autocast(device_type=data.device.type, enabled=True) if use_amp else nullcontext()
    )
    with context:
        with torch.no_grad():
            bottleneck = model.encode_bottleneck(data)
        logits = model.classify_bottleneck(bottleneck)
        loss, lesion_present = _classification_loss(
            logits,
            targets,
            segmentation_target,
            class_weights=class_weights,
            label_smoothing=schedule.label_smoothing,
            nonlesion_patch_weight=schedule.nonlesion_patch_weight,
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Classification rescue produced a non-finite loss")

    trainable_parameters = [parameter for _, parameter in trainable_items]
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            schedule.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            schedule.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
    predictions = logits.detach().argmax(dim=1)
    target_counts = torch.bincount(targets.detach(), minlength=num_classes)
    prediction_counts = torch.bincount(predictions, minlength=num_classes)
    return RescueStepResult(
        loss=float(loss.detach().cpu()),
        correct=int((predictions == targets).sum().cpu()),
        sample_count=int(targets.numel()),
        lesion_patch_count=int(lesion_present.sum().cpu()),
        gradient_norm_before_clip=float(torch.as_tensor(gradient_norm).detach().cpu()),
        target_counts=tuple(int(value) for value in target_counts.cpu()),
        prediction_counts=tuple(int(value) for value in prediction_counts.cpu()),
    )


def strict_load_network_weights(model: nn.Module, checkpoint: Mapping[str, Any]) -> None:
    """Load one complete joint network state, accepting only a DDP prefix."""

    raw_state = checkpoint.get("network_weights")
    if not isinstance(raw_state, Mapping):
        raise TypeError("Checkpoint does not contain a network_weights mapping")
    expected_keys = set(model.state_dict())
    normalized: dict[str, Tensor] = {}
    for raw_name, value in raw_state.items():
        name = str(raw_name)
        if name not in expected_keys and name.startswith("module."):
            name = name[7:]
        if name in normalized:
            raise ValueError(f"Duplicate normalized state key: {name}")
        normalized[name] = value
    missing = sorted(expected_keys - set(normalized))
    unexpected = sorted(set(normalized) - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint/network state mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.load_state_dict(normalized, strict=True)


def capture_rng_state(*, include_cuda: bool) -> dict[str, Any]:
    """Capture RNG state needed by the single-threaded resumable loop."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if include_cuda:
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], *, include_cuda: bool) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if include_cuda:
        if "cuda" not in state:
            raise ValueError("Resume checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_all(seed: int, *, include_cuda: bool) -> None:
    """Seed the single-process rescue loop."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda:
        torch.cuda.manual_seed_all(seed)


def build_inference_checkpoint(
    source_checkpoint: Mapping[str, Any],
    model: nn.Module,
    rescue_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a minimal nnU-Net joint-inference checkpoint with rescue state."""

    for key in ("trainer_name", "init_args"):
        if key not in source_checkpoint:
            raise ValueError(f"Source checkpoint is missing required key {key!r}")
    checkpoint: dict[str, Any] = {
        "network_weights": model.state_dict(),
        "trainer_name": source_checkpoint["trainer_name"],
        "init_args": source_checkpoint["init_args"],
        "classification_rescue": dict(rescue_state),
    }
    if "inference_allowed_mirroring_axes" in source_checkpoint:
        checkpoint["inference_allowed_mirroring_axes"] = source_checkpoint[
            "inference_allowed_mirroring_axes"
        ]
    return checkpoint


def validate_resume_state(
    state: Mapping[str, Any],
    *,
    source_checkpoint_sha256: str,
    schedule: RescueSchedule,
) -> int:
    """Validate immutable provenance/schedule and return completed epochs."""

    if state.get("schema_version") != RESCUE_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing classification rescue schema version")
    if state.get("source_checkpoint_sha256") != source_checkpoint_sha256:
        raise ValueError("Resume checkpoint was derived from a different source checkpoint")
    if state.get("schedule") != schedule.to_dict():
        raise ValueError("Resume arguments do not match the checkpoint's frozen schedule")
    completed_epochs = int(state.get("completed_epochs", -1))
    if not 0 <= completed_epochs <= schedule.epochs:
        raise ValueError(f"Invalid completed_epochs value: {completed_epochs}")
    if "optimizer_state" not in state or "rng_state" not in state:
        raise ValueError("Resume checkpoint lacks optimizer or RNG state")
    history = state.get("training_only_history")
    if not isinstance(history, list) or len(history) != completed_epochs:
        raise ValueError("Resume history length does not match completed_epochs")
    if completed_epochs == schedule.epochs and state.get("status") != "complete":
        raise ValueError("A fully completed rescue checkpoint must have status='complete'")
    if not isinstance(state.get("current_component_sha256"), Mapping):
        raise TypeError("Resume checkpoint lacks component hashes")
    started_at = state.get("started_at_utc")
    if not isinstance(started_at, str) or not started_at.strip():
        raise ValueError("Resume checkpoint lacks its original started_at_utc")
    elapsed_seconds = float(state.get("elapsed_seconds", -1.0))
    if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("Resume checkpoint has invalid cumulative elapsed_seconds")
    resume_count = int(state.get("resume_count", -1))
    if resume_count < 0:
        raise ValueError("Resume checkpoint has invalid resume_count")
    return completed_epochs


def atomic_torch_save(payload: Any, destination: str | os.PathLike[str]) -> None:
    """Save a torch payload beside the destination, then replace atomically."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: Mapping[str, Any], destination: str | os.PathLike[str]) -> None:
    """Write a human-readable audit atomically."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ACTIVATION_AUDIT_SCHEMA_VERSION",
    "CLASSIFICATION_PARAMETER_PREFIXES",
    "RESCUE_SCHEMA_VERSION",
    "RescueSchedule",
    "RescueStepResult",
    "atomic_torch_save",
    "atomic_write_json",
    "build_inference_checkpoint",
    "capture_rng_state",
    "case_id_sha256",
    "classification_activation_audit",
    "classification_parameter_items",
    "classification_rescue_train_step",
    "component_hashes",
    "file_sha256",
    "freeze_for_classification_rescue",
    "module_state_sha256",
    "reset_classification_parameters",
    "restore_rng_state",
    "seed_all",
    "strict_load_network_weights",
    "validate_activation_audit",
    "validate_disjoint_split",
    "validate_frozen_split_manifest",
    "validate_resume_state",
]
