"""nnU-Net v2.8.1 trainer for joint segmentation and classification."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from torch import Tensor, autocast
from torch import distributed as dist

from pancreas_multitask.network import MultiTaskResEncUNet

NUM_CLASSIFICATION_CLASSES = 3
CASE_CLASS_PATTERN = re.compile(r"^quiz_([012])(?:_|$)")
DEFAULT_DEVICE = torch.device("cuda")

CUSTOM_LOG_KEYS = (
    "train_seg_losses",
    "train_cls_losses",
    "train_cls_accuracy",
    "train_lesion_patch_fraction",
    "val_seg_losses",
    "val_cls_losses",
    "val_cls_macro_f1",
    "val_cls_accuracy",
    "val_cls_f1_per_class",
    "val_cls_case_coverage",
    "val_whole_pancreas_dice",
    "val_lesion_patch_fraction",
    "val_multitask_score",
)


def parse_class_from_case_id(case_id: str) -> int:
    """Extract subtype 0/1/2 from a labelled source case identifier."""

    match = CASE_CLASS_PATTERN.match(str(case_id))
    if match is None:
        raise ValueError(
            "Could not infer a subtype from case identifier "
            f"{case_id!r}; expected a name like 'quiz_2_123'."
        )
    return int(match.group(1))


def resolve_case_labels(
    case_ids: Sequence[str],
    label_mapping: Mapping[str, int] | None = None,
) -> list[int]:
    """Resolve labels from generated metadata, with a validated ID fallback."""

    mapping = {str(key): int(value) for key, value in (label_mapping or {}).items()}
    labels: list[int] = []
    for raw_case_id in case_ids:
        case_id = str(raw_case_id)
        label = mapping.get(case_id)
        if label is None:
            label = parse_class_from_case_id(case_id)
        if not 0 <= label < NUM_CLASSIFICATION_CLASSES:
            raise ValueError(
                f"Classification label for {case_id!r} must be 0, 1, or 2; "
                f"got {label}."
            )
        labels.append(label)
    return labels


def inverse_frequency_class_weights(
    labels: Sequence[int],
    num_classes: int = NUM_CLASSIFICATION_CLASSES,
) -> np.ndarray:
    """Return ``N / (K * count[class])`` weights for weighted cross entropy."""

    array = np.asarray(labels, dtype=np.int64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if np.any((array < 0) | (array >= num_classes)):
        raise ValueError(f"labels must be in [0, {num_classes - 1}]")
    counts = np.bincount(array, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(
            "Every classification class must occur in the training split; "
            f"observed counts {counts.tolist()}."
        )
    return (array.size / (num_classes * counts)).astype(np.float32)


def macro_f1_from_predictions(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    num_classes: int = NUM_CLASSIFICATION_CLASSES,
) -> tuple[float, tuple[float, ...]]:
    """Compute unweighted per-class and macro F1 without a sklearn dependency."""

    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("targets and predictions must have the same non-empty shape")

    per_class: list[float] = []
    for class_id in range(num_classes):
        true_positive = np.sum((truth == class_id) & (predicted == class_id))
        false_positive = np.sum((truth != class_id) & (predicted == class_id))
        false_negative = np.sum((truth == class_id) & (predicted != class_id))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class.append(
            float(2 * true_positive / denominator) if denominator else 0.0
        )
    return float(np.mean(per_class)), tuple(per_class)


def case_level_classification_metrics(
    case_ids: Sequence[str],
    logits: np.ndarray,
    targets: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Average repeated patch logits by case and compute classification metrics."""

    logits_array = np.asarray(logits, dtype=np.float64)
    targets_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    if logits_array.ndim != 2 or logits_array.shape[1] != NUM_CLASSIFICATION_CLASSES:
        raise ValueError(
            "logits must have shape (samples, 3), got "
            f"{logits_array.shape}"
        )
    if len(case_ids) != logits_array.shape[0] or targets_array.size != len(case_ids):
        raise ValueError("case_ids, logits, and targets must contain the same samples")

    logits_by_case: dict[str, list[np.ndarray]] = defaultdict(list)
    target_by_case: dict[str, int] = {}
    for case_id, sample_logits, target in zip(
        (str(item) for item in case_ids),
        logits_array,
        targets_array,
        strict=True,
    ):
        existing = target_by_case.setdefault(case_id, int(target))
        if existing != int(target):
            raise ValueError(f"Conflicting targets observed for case {case_id!r}")
        logits_by_case[case_id].append(sample_logits)

    ordered_case_ids = sorted(logits_by_case)
    case_logits = np.stack(
        [np.mean(logits_by_case[case_id], axis=0) for case_id in ordered_case_ids]
    )
    case_targets = np.asarray(
        [target_by_case[case_id] for case_id in ordered_case_ids], dtype=np.int64
    )
    case_predictions = np.argmax(case_logits, axis=1)
    macro_f1, per_class_f1 = macro_f1_from_predictions(
        case_targets,
        case_predictions,
    )
    return {
        "case_ids": ordered_case_ids,
        "logits": case_logits,
        "targets": case_targets,
        "predictions": case_predictions,
        "accuracy": float(np.mean(case_predictions == case_targets)),
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }


def weighted_classification_loss(
    logits: Tensor,
    class_targets: Tensor,
    segmentation_target: Tensor,
    *,
    class_weights: Tensor | None,
    nonlesion_patch_weight: float,
    label_smoothing: float,
) -> tuple[Tensor, Tensor]:
    """Compute class-balanced CE and downweight crops without visible lesion.

    The subtype is a case label, but nnU-Net trains on patches. A crop without
    any label-2 voxel supplies weaker evidence for a lesion subtype, so it keeps
    a small non-zero contribution instead of being treated as equally reliable.
    """

    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSIFICATION_CLASSES:
        raise ValueError(f"Expected classification logits of shape (B, 3), got {logits.shape}")
    if class_targets.ndim != 1 or class_targets.shape[0] != logits.shape[0]:
        raise ValueError("class_targets must have shape (B,)")
    if segmentation_target.shape[0] != logits.shape[0]:
        raise ValueError("segmentation_target batch size must match logits")
    if not 0.0 < nonlesion_patch_weight <= 1.0:
        raise ValueError("nonlesion_patch_weight must be in (0, 1]")

    per_sample = F.cross_entropy(
        logits.float(),
        class_targets,
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


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw_value!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


class nnUNetTrainerPancreasMultiTask(nnUNetTrainer):
    """3D ResEnc-M trainer with an auxiliary three-class subtype objective."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        # Keep this explicit signature. nnU-Net records these named arguments in
        # checkpoints and uses them to reconstruct the custom trainer at inference.
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = _env_int("PANCREAS_MT_EPOCHS", 200)
        self.num_iterations_per_epoch = _env_int("PANCREAS_MT_TRAIN_ITERS", 125)
        self.num_val_iterations_per_epoch = _env_int("PANCREAS_MT_VAL_ITERS", 30)
        self.classification_loss_weight = _env_float(
            "PANCREAS_MT_CLS_LOSS_WEIGHT", 0.5, minimum=0.0
        )
        self.nonlesion_patch_weight = _env_float(
            "PANCREAS_MT_NONLESION_PATCH_WEIGHT",
            0.25,
            minimum=np.finfo(np.float32).eps,
            maximum=1.0,
        )
        self.classification_label_smoothing = _env_float(
            "PANCREAS_MT_LABEL_SMOOTHING", 0.05, minimum=0.0, maximum=0.5
        )
        self.oversample_foreground_percent = _env_float(
            "PANCREAS_MT_OVERSAMPLE_FOREGROUND", 0.5, minimum=0.0, maximum=1.0
        )

        self.classification_label_mapping = self._load_classification_label_mapping()
        self.classification_class_weights: Tensor | None = None
        self._expected_validation_cases: int | None = None
        self._best_multitask_score = float("-inf")

        for key in CUSTOM_LOG_KEYS:
            self.logger.local_logger.my_fantastic_logging.setdefault(key, [])
        self.logger.update_config(
            {
                "multitask": {
                    "classification_classes": NUM_CLASSIFICATION_CLASSES,
                    "classification_loss_weight": self.classification_loss_weight,
                    "nonlesion_patch_weight": self.nonlesion_patch_weight,
                    "classification_label_smoothing": self.classification_label_smoothing,
                    "classification_pooling": "global_average_plus_cross_attention",
                    "oversample_foreground_percent": self.oversample_foreground_percent,
                }
            }
        )

    def _load_classification_label_mapping(self) -> dict[str, int]:
        if not nnUNet_raw.is_set():
            return {}
        metadata_path = (
            Path(os.fspath(nnUNet_raw))
            / self.plans_manager.dataset_name
            / "classification_labels.json"
        )
        if not metadata_path.is_file():
            return {}
        with metadata_path.open("r", encoding="utf-8") as handle:
            raw_mapping = json.load(handle)
        if not isinstance(raw_mapping, dict):
            raise TypeError(f"Expected a JSON object in {metadata_path}")
        mapping = {str(key): int(value) for key, value in raw_mapping.items()}
        # Validate eagerly so a damaged manifest cannot silently corrupt labels.
        resolve_case_labels(tuple(mapping), mapping)
        return mapping

    def _classification_targets(self, case_ids: Sequence[str]) -> Tensor:
        labels = resolve_case_labels(case_ids, self.classification_label_mapping)
        return torch.as_tensor(labels, dtype=torch.long, device=self.device)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> MultiTaskResEncUNet:
        segmentation_network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        if segmentation_network.__class__.__name__ != "ResidualEncoderUNet":
            raise RuntimeError(
                "This trainer requires nnU-Net's ResidualEncoderUNet plans; got "
                f"{segmentation_network.__class__.__name__}. Use "
                "nnUNetPlannerResEncM / nnUNetResEncUNetMPlans."
            )
        return MultiTaskResEncUNet(segmentation_network)

    def on_train_start(self) -> None:
        super().on_train_start()
        train_keys, validation_keys = self.do_split()
        train_labels = resolve_case_labels(
            train_keys,
            self.classification_label_mapping,
        )
        weights = inverse_frequency_class_weights(train_labels)
        self.classification_class_weights = torch.as_tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )
        self._expected_validation_cases = len(validation_keys)
        counts = np.bincount(
            np.asarray(train_labels, dtype=np.int64),
            minlength=NUM_CLASSIFICATION_CLASSES,
        )
        self.logger.update_config(
            {
                "multitask_split": {
                    "training_class_counts": counts.tolist(),
                    "classification_class_weights": weights.tolist(),
                    "validation_cases": len(validation_keys),
                }
            }
        )

    @staticmethod
    def _move_segmentation_target(
        target: Tensor | list[Tensor],
        device: torch.device,
    ) -> Tensor | list[Tensor]:
        if isinstance(target, list):
            return [item.to(device, non_blocking=True) for item in target]
        return target.to(device, non_blocking=True)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._move_segmentation_target(batch["target"], self.device)
        class_target = self._classification_targets(batch["keys"])
        highest_resolution_target = target[0] if isinstance(target, list) else target

        self.optimizer.zero_grad(set_to_none=True)
        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            segmentation_output, classification_logits = self.network(
                data,
                return_classification=True,
            )
            segmentation_loss = self.loss(segmentation_output, target)
            classification_loss, lesion_present = weighted_classification_loss(
                classification_logits,
                class_target,
                highest_resolution_target,
                class_weights=self.classification_class_weights,
                nonlesion_patch_weight=self.nonlesion_patch_weight,
                label_smoothing=self.classification_label_smoothing,
            )
            total_loss = (
                segmentation_loss
                + self.classification_loss_weight * classification_loss
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        predictions = classification_logits.detach().argmax(dim=1)
        return {
            "loss": float(total_loss.detach().cpu()),
            "seg_loss": float(segmentation_loss.detach().cpu()),
            "cls_loss": float(classification_loss.detach().cpu()),
            "cls_correct": int((predictions == class_target).sum().cpu()),
            "cls_count": int(class_target.numel()),
            "lesion_patch_count": int(lesion_present.sum().cpu()),
        }

    def on_train_epoch_end(self, train_outputs: list[dict]) -> None:
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        sample_count = max(int(np.sum(outputs["cls_count"])), 1)
        self.logger.log(
            "train_seg_losses",
            float(np.mean(outputs["seg_loss"])),
            self.current_epoch,
        )
        self.logger.log(
            "train_cls_losses",
            float(np.mean(outputs["cls_loss"])),
            self.current_epoch,
        )
        self.logger.log(
            "train_cls_accuracy",
            float(np.sum(outputs["cls_correct"]) / sample_count),
            self.current_epoch,
        )
        self.logger.log(
            "train_lesion_patch_fraction",
            float(np.sum(outputs["lesion_patch_count"]) / sample_count),
            self.current_epoch,
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._move_segmentation_target(batch["target"], self.device)
        class_target = self._classification_targets(batch["keys"])
        highest_resolution_target = target[0] if isinstance(target, list) else target

        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            segmentation_output, classification_logits = self.network(
                data,
                return_classification=True,
            )
            segmentation_loss = self.loss(segmentation_output, target)
            classification_loss, lesion_present = weighted_classification_loss(
                classification_logits,
                class_target,
                highest_resolution_target,
                class_weights=self.classification_class_weights,
                nonlesion_patch_weight=self.nonlesion_patch_weight,
                label_smoothing=self.classification_label_smoothing,
            )
            total_loss = (
                segmentation_loss
                + self.classification_loss_weight * classification_loss
            )

        output_for_metrics = (
            segmentation_output[0]
            if self.enable_deep_supervision
            else segmentation_output
        )
        target_for_metrics = (
            target[0] if self.enable_deep_supervision else target
        )

        axes = [0] + list(range(2, output_for_metrics.ndim))
        if self.label_manager.has_regions:
            predicted_onehot = (torch.sigmoid(output_for_metrics) > 0.5).long()
            union_tp = union_fp = union_fn = float("nan")
        else:
            predicted_labels = output_for_metrics.argmax(1)[:, None]
            predicted_onehot = torch.zeros(
                output_for_metrics.shape,
                device=output_for_metrics.device,
                dtype=torch.float16,
            )
            predicted_onehot.scatter_(1, predicted_labels, 1)

            valid = torch.ones_like(target_for_metrics, dtype=torch.bool)
            if self.label_manager.has_ignore_label:
                valid &= target_for_metrics != self.label_manager.ignore_label
            predicted_union = (predicted_labels > 0) & valid
            target_union = (target_for_metrics > 0) & valid
            union_tp = float((predicted_union & target_union).sum().cpu())
            union_fp = float((predicted_union & ~target_union & valid).sum().cpu())
            union_fn = float((~predicted_union & target_union & valid).sum().cpu())

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_for_metrics != self.label_manager.ignore_label).float()
                target_for_metrics[target_for_metrics == self.label_manager.ignore_label] = 0
            else:
                mask = (
                    ~target_for_metrics[:, -1:]
                    if target_for_metrics.dtype == torch.bool
                    else 1 - target_for_metrics[:, -1:]
                )
                target_for_metrics = target_for_metrics[:, :-1]
        else:
            mask = None

        true_positive, false_positive, false_negative, _ = get_tp_fp_fn_tn(
            predicted_onehot,
            target_for_metrics,
            axes=axes,
            mask=mask,
        )
        tp_hard = true_positive.detach().cpu().numpy()
        fp_hard = false_positive.detach().cpu().numpy()
        fn_hard = false_negative.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": float(total_loss.detach().cpu()),
            "seg_loss": float(segmentation_loss.detach().cpu()),
            "cls_loss": float(classification_loss.detach().cpu()),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
            "cls_logits": classification_logits.detach().float().cpu().numpy(),
            "cls_targets": class_target.detach().cpu().numpy(),
            "keys": [str(key) for key in batch["keys"]],
            "lesion_patch_count": int(lesion_present.sum().cpu()),
            "cls_count": int(class_target.numel()),
            "union_tp": union_tp,
            "union_fp": union_fp,
            "union_fn": union_fn,
        }

    def _gather_validation_outputs(self, outputs: list[dict]) -> list[dict]:
        if not self.is_ddp:
            return outputs
        gathered: list[list[dict] | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, outputs)
        return [item for rank_outputs in gathered if rank_outputs for item in rank_outputs]

    def on_validation_epoch_end(self, val_outputs: list[dict]) -> None:
        # Preserve nnU-Net's native segmentation aggregation and W&B metrics.
        super().on_validation_epoch_end(val_outputs)
        outputs = self._gather_validation_outputs(val_outputs)

        all_case_ids = [key for output in outputs for key in output["keys"]]
        all_logits = np.concatenate([output["cls_logits"] for output in outputs])
        all_targets = np.concatenate([output["cls_targets"] for output in outputs])
        classification = case_level_classification_metrics(
            all_case_ids,
            all_logits,
            all_targets,
        )

        sample_count = max(int(sum(output["cls_count"] for output in outputs)), 1)
        expected_cases = self._expected_validation_cases or len(classification["case_ids"])
        union_tp = float(sum(output["union_tp"] for output in outputs))
        union_fp = float(sum(output["union_fp"] for output in outputs))
        union_fn = float(sum(output["union_fn"] for output in outputs))
        union_denominator = 2 * union_tp + union_fp + union_fn
        whole_pancreas_dice = (
            float(2 * union_tp / union_denominator)
            if union_denominator > 0 and np.isfinite(union_denominator)
            else float("nan")
        )

        self.logger.log(
            "val_seg_losses",
            float(np.mean([output["seg_loss"] for output in outputs])),
            self.current_epoch,
        )
        self.logger.log(
            "val_cls_losses",
            float(np.mean([output["cls_loss"] for output in outputs])),
            self.current_epoch,
        )
        self.logger.log(
            "val_cls_macro_f1",
            classification["macro_f1"],
            self.current_epoch,
        )
        self.logger.log(
            "val_cls_accuracy",
            classification["accuracy"],
            self.current_epoch,
        )
        self.logger.log(
            "val_cls_f1_per_class",
            list(classification["per_class_f1"]),
            self.current_epoch,
        )
        self.logger.log(
            "val_cls_case_coverage",
            float(len(classification["case_ids"]) / max(expected_cases, 1)),
            self.current_epoch,
        )
        self.logger.log(
            "val_whole_pancreas_dice",
            whole_pancreas_dice,
            self.current_epoch,
        )
        self.logger.log(
            "val_lesion_patch_fraction",
            float(sum(output["lesion_patch_count"] for output in outputs) / sample_count),
            self.current_epoch,
        )

    def on_epoch_end(self) -> None:
        segmentation_score = float(
            self.logger.get_value("mean_fg_dice", step=-1)
        )
        classification_score = float(
            self.logger.get_value("val_cls_macro_f1", step=-1)
        )
        multitask_score = float(np.mean((segmentation_score, classification_score)))
        self.logger.log("val_multitask_score", multitask_score, self.current_epoch)
        is_new_multitask_best = (
            np.isfinite(multitask_score)
            and multitask_score > self._best_multitask_score
        )
        if is_new_multitask_best:
            self._best_multitask_score = multitask_score
            self.print_to_log_file(
                f"New best multi-task score: {multitask_score:.4f}"
            )
        super().on_epoch_end()

        if is_new_multitask_best:
            # Base on_epoch_end logs the end timestamp before incrementing the
            # epoch. Save only after that so every LocalLogger series in a
            # resumable checkpoint has the same length. Temporarily restore the
            # just-finished epoch because save_checkpoint itself stores +1.
            self.current_epoch -= 1
            try:
                self.save_checkpoint(
                    os.path.join(
                        self.output_folder,
                        "checkpoint_best_multitask.pth",
                    )
                )
            finally:
                self.current_epoch += 1

    def load_checkpoint(self, checkpoint: dict | str) -> None:
        super().load_checkpoint(checkpoint)
        for key in CUSTOM_LOG_KEYS:
            self.logger.local_logger.my_fantastic_logging.setdefault(key, [])
        prior_scores = self.logger.get_value("val_multitask_score", step=None)
        finite_scores = [float(value) for value in prior_scores if np.isfinite(value)]
        self._best_multitask_score = max(finite_scores, default=float("-inf"))


__all__ = [
    "case_level_classification_metrics",
    "inverse_frequency_class_weights",
    "macro_f1_from_predictions",
    "nnUNetTrainerPancreasMultiTask",
    "parse_class_from_case_id",
    "resolve_case_labels",
    "weighted_classification_loss",
]
