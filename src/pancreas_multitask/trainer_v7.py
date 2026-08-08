"""V7 trainer: shallow classification tap with its own optimizer.

Two defects in the original joint trainer are corrected here, and nothing else
about the segmentation recipe changes.

**1. The classification head was never trainable.**
``nnUNetTrainerPancreasMultiTask`` does not override ``configure_optimizers``, so
the classification branch inherited nnU-Net's segmentation optimizer: SGD at
lr 1e-2 with momentum 0.99 and Nesterov.  Steady-state momentum amplification is
``1/(1-0.99) = 100``, i.e. an effective step of order 1.0 applied to a freshly
initialized ``LayerNorm -> Linear`` stack whose weights start at scale 0.02.  The
head saturates immediately and collapses to a constant output, and a constant
output under three-class cross-entropy sits at exactly ``ln 3 = 1.0986`` -- which
is precisely where the measured classification loss stayed for all 200 epochs.
Final validation macro-F1 was 0.1333, below the 0.19 majority-class baseline.

Note that adding a second *parameter group* does not fix this: nnU-Net's
``PolyLRScheduler.step`` overwrites ``lr`` on **every** group each epoch. A
genuinely separate optimizer is required, which is what this trainer builds.

Also corrected: ``clip_grad_norm_(self.network.parameters(), 12)`` clipped
globally across 102.7M parameters, so the 496k-parameter head's gradient was
rescaled by whatever norm the segmentation trunk happened to have. The two paths
are clipped separately here.

**2. The head read the wrong depth.**
Nested-CV linear probes over the 252 training cases score encoder stage-2 at
macro-F1 0.594, stage-3 at 0.525 and the stage-5 bottleneck at 0.399 -- against a
0.354 permuted-label control. The original head pooled the bottleneck, i.e. very
nearly noise. This trainer taps stage 2 and, because subtype is a whole-organ
property rather than a patch property, classifies whole ROIs instead of patches.

The segmentation stream is deliberately retained every step. Freezing stages 3-5
would *not* protect Dice, because the decoder consumes the stage-0/1/2 skips
directly; the live segmentation loss is what actually holds the shared trunk in
place while the shallow stages receive subtype gradient.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor, autocast

from nnunetv2.utilities.helpers import dummy_context
from pancreas_multitask.network import MultiTaskResEncUNet
from pancreas_multitask.wholevolume_dataset import pad_to_stride
from pancreas_multitask.trainer import (
    DEFAULT_DEVICE,
    nnUNetTrainerPancreasMultiTask,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

#: Encoder stage feeding the classification head. See module docstring.
CLASSIFICATION_TAP: Final = 2

#: Highest encoder stage allowed to receive subtype gradient. Stem + stages 0-2 is
#: 3,959,872 parameters, 4.4% of the 90.3M-parameter encoder -- small enough that
#: it cannot drift far, and it is exactly where the probe says the signal lives.
TRAINABLE_ENCODER_STAGE: Final = 2


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    value = float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}], got {value}")
    return value


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def anchor_penalty(
    parameters: list[Tensor],
    reference: list[Tensor],
) -> Tensor:
    """Squared drift of the fine-tuned shallow stages from their starting weights.

    A soft tether rather than a hard freeze: the shallow stages must be free to
    move for the classification objective to mean anything, but unconstrained
    drift on 252 cases is how segmentation quality gets quietly traded away.
    """

    if len(parameters) != len(reference):
        raise ValueError("Anchor reference does not match the trainable parameter set")
    total = None
    for current, initial in zip(parameters, reference):
        if current.shape != initial.shape:
            raise ValueError("Anchor reference tensor shape mismatch")
        term = (current - initial).pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Anchor penalty requires at least one trainable parameter")
    return total


class nnUNetTrainerPancreasMultiTaskV7(nnUNetTrainerPancreasMultiTask):
    """Joint trainer with a stage-2 classification tap on a separate optimizer."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.classification_learning_rate = _env_float(
            "PANCREAS_MT7_CLS_LR", 3e-4, minimum=0.0, maximum=1.0
        )
        self.encoder_learning_rate = _env_float(
            "PANCREAS_MT7_ENC_LR", 3e-5, minimum=0.0, maximum=1.0
        )
        self.classification_weight_decay = _env_float(
            "PANCREAS_MT7_CLS_WD", 1e-2, minimum=0.0, maximum=1.0
        )
        self.classification_grad_clip = _env_float(
            "PANCREAS_MT7_CLS_CLIP", 1.0, minimum=1e-6, maximum=1e3
        )
        self.anchor_weight = _env_float(
            "PANCREAS_MT7_ANCHOR", 1e-3, minimum=0.0, maximum=1e3
        )
        self.num_epochs = _env_int("PANCREAS_MT7_EPOCHS", 50, minimum=1)

        # nnU-Net sets initial_lr = 1e-2, which is its *from-scratch* learning rate, and
        # PolyLRScheduler restarts from that value every run. Applied to an already
        # converged warm-started checkpoint with SGD momentum 0.99 (effective step of
        # order 1.0), it immediately kicks the model out of its optimum: the measured
        # segmentation loss rose from -0.784 to -0.67 within five epochs and the shallow
        # encoder stages drifted regardless of anything the classification optimizer did,
        # because those stages sit in both optimizers and SGD dominates. Fine-tuning
        # needs a fine-tuning learning rate.
        self.initial_lr = _env_float("PANCREAS_MT7_SEG_LR", 1e-4, minimum=0.0, maximum=1.0)
        self.classification_batch_size = _env_int("PANCREAS_MT7_CLS_BATCH", 6, minimum=3)
        self.classification_optimizer: torch.optim.Optimizer | None = None
        self._anchor_reference: list[Tensor] | None = None
        self._volume_cache: dict[str, np.ndarray] | None = None
        self._case_pool: dict[int, list[str]] | None = None
        self._case_rng: np.random.Generator | None = None
        # Segmentation Dice is the pass/fail constraint, so a checkpoint is only
        # eligible on classification once it clears the Dice gate.
        self._dice_gate_whole = _env_float(
            "PANCREAS_MT7_DICE_GATE_WHOLE", 0.9150, minimum=0.0, maximum=1.0
        )
        self._dice_gate_lesion = _env_float(
            "PANCREAS_MT7_DICE_GATE_LESION", 0.3400, minimum=0.0, maximum=1.0
        )

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
                f"{segmentation_network.__class__.__name__}"
            )
        return MultiTaskResEncUNet(
            segmentation_network,
            classification_tap=CLASSIFICATION_TAP,
        )

    # ------------------------------------------------------------------ optimizers

    def classification_parameter_set(self) -> list[Tensor]:
        """Head parameters plus the shallow encoder stages allowed to adapt."""

        network = self.network.module if hasattr(self.network, "module") else self.network
        seen: set[int] = set()
        collected: list[Tensor] = []
        for parameter in (
            *network.classification_parameters(),
            *network.encoder_stage_parameters(TRAINABLE_ENCODER_STAGE),
        ):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                collected.append(parameter)
        return collected

    def configure_optimizers(self) -> tuple[torch.optim.Optimizer, Any]:
        """Segmentation keeps nnU-Net's SGD/PolyLR; classification gets AdamW.

        Returning only the segmentation optimizer to nnU-Net is intentional: its
        ``PolyLRScheduler`` rewrites ``lr`` on every param group it owns, so the
        classification path has to live outside that schedule entirely.
        """

        optimizer, scheduler = super().configure_optimizers()

        network = self.network.module if hasattr(self.network, "module") else self.network
        classification_ids = {id(p) for p in network.classification_parameters()}
        encoder_ids = {id(p) for p in network.encoder_stage_parameters(TRAINABLE_ENCODER_STAGE)}

        # Keep the classification head out of the SGD optimizer entirely; leave the
        # shallow encoder stages in it, because they must still receive the
        # segmentation gradient that protects Dice.
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in classification_ids]

        self.classification_optimizer = torch.optim.AdamW(
            [
                {
                    "params": network.classification_parameters(),
                    "lr": self.classification_learning_rate,
                },
                {
                    "params": network.encoder_stage_parameters(TRAINABLE_ENCODER_STAGE),
                    "lr": self.encoder_learning_rate,
                },
            ],
            weight_decay=self.classification_weight_decay,
        )
        self._anchor_reference = [
            p.detach().clone()
            for p in network.encoder_stage_parameters(TRAINABLE_ENCODER_STAGE)
        ]
        if self.anchor_weight > 0 and not encoder_ids:
            raise RuntimeError("Anchor penalty requested but no encoder stages are trainable")
        return optimizer, scheduler

    def train_step(self, batch: dict) -> dict:
        """Segmentation on patches, then classification on whole ROIs.

        The two streams are run back to back with a gradient clear between them,
        so each optimizer steps on exactly its own loss. The original trainer
        summed both losses and backpropagated once through a single global clip,
        which is how the 496k-parameter head ended up having its gradient scaled
        by the 102.7M-parameter segmentation trunk.

        The segmentation stream is retained deliberately. Freezing the deep stages
        would not protect Dice, because the decoder consumes the stage-0/1/2 skips
        that the classification objective is allowed to move; the live
        segmentation gradient is the thing actually holding them in place.
        """

        data = batch["data"].to(self.device, non_blocking=True)
        target = self._move_segmentation_target(batch["target"], self.device)

        self.optimizer.zero_grad(set_to_none=True)
        if self.classification_optimizer is not None:
            self.classification_optimizer.zero_grad(set_to_none=True)

        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            segmentation_output = self.network(data)
            segmentation_loss = self.loss(segmentation_output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(segmentation_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            segmentation_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            segmentation_loss.backward()
            segmentation_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            )
            self.optimizer.step()

        # Clear before the classification pass so the shallow encoder stages do not
        # carry segmentation gradient into the AdamW step.
        self.optimizer.zero_grad(set_to_none=True)
        if self.classification_optimizer is not None:
            self.classification_optimizer.zero_grad(set_to_none=True)

        classification = self.classification_train_step()
        return {
            "loss": float(segmentation_loss.detach().cpu()) + classification["cls_loss"],
            "seg_loss": float(segmentation_loss.detach().cpu()),
            "seg_grad_norm": segmentation_grad_norm,
            **classification,
        }

    # ------------------------------------------------------- whole-ROI classification

    def _volume_source(self) -> dict[str, np.ndarray]:
        """Lazily cache every training ROI keyed by case id.

        Loaded straight from nnU-Net's own preprocessed store, so the
        classification stream sees exactly the volumes the segmentation stream
        samples its patches from.
        """

        if getattr(self, "_volume_cache", None) is None:
            import blosc2

            folder = Path(self.preprocessed_dataset_folder)
            volumes: dict[str, np.ndarray] = {}
            for case_id in self.dataloader_train.generator._data.identifiers:  # type: ignore[attr-defined]
                array = blosc2.open(str(folder / f"{case_id}.b2nd"), mode="r")[:]
                volumes[case_id] = np.asarray(array, dtype=np.float16)
            self._volume_cache = volumes
        return self._volume_cache

    def _classification_batch(self, size: int) -> list[tuple[np.ndarray, int]]:
        """Draw a class-balanced batch of whole ROIs, with replacement.

        Balanced sampling is the single imbalance correction in use; the loss is
        deliberately left unweighted so the two corrections are not stacked.
        """

        volumes = self._volume_source()
        if getattr(self, "_case_pool", None) is None:
            pool: dict[int, list[str]] = {0: [], 1: [], 2: []}
            for case_id in volumes:
                pool[int(self.classification_label_mapping[case_id])].append(case_id)
            for label, members in pool.items():
                if not members:
                    raise RuntimeError(f"No training cases available for subtype {label}")
            self._case_pool = {label: sorted(members) for label, members in pool.items()}
            self._case_rng = np.random.default_rng(20260806)
        chosen: list[tuple[np.ndarray, int]] = []
        for index in range(int(size)):
            label = index % 3
            members = self._case_pool[label]
            case_id = members[int(self._case_rng.integers(len(members)))]
            chosen.append((volumes[case_id], label))
        return chosen

    def classification_train_step(self) -> dict[str, float]:
        """One whole-ROI classification update on its own optimizer.

        Run after the segmentation update and after gradients are cleared, so the
        two objectives never share a gradient buffer and each optimizer steps on
        exactly the loss it owns. This is also why the classification path can be
        clipped at its own scale instead of sharing the segmentation trunk's
        global norm budget.
        """

        network = self.network.module if hasattr(self.network, "module") else self.network
        optimizer = self.classification_optimizer
        if optimizer is None:
            raise RuntimeError("configure_optimizers must run before classification_train_step")

        for group, base in zip(
            optimizer.param_groups,
            (self.classification_learning_rate, self.encoder_learning_rate),
        ):
            scale = self.classification_learning_rate_at(self.current_epoch)
            group["lr"] = base * (scale / max(self.classification_learning_rate, 1e-12))

        optimizer.zero_grad(set_to_none=True)
        batch = self._classification_batch(self.classification_batch_size)
        cross_entropy_sum = 0.0
        anchor_value = 0.0
        correct = 0
        for volume, label in batch:
            work = pad_to_stride(torch.from_numpy(volume.astype(np.float32)))
            work = work[None].to(self.device, non_blocking=True)
            target = torch.tensor([label], dtype=torch.long, device=self.device)
            logits = network.classify_volume(work)
            cross_entropy = torch.nn.functional.cross_entropy(
                logits.float(), target, label_smoothing=self.classification_label_smoothing
            )
            # Report the cross-entropy on its own scale. Folding the anchor term
            # into the logged value would hide whether the head is actually
            # learning, which is the one thing this number exists to show.
            loss = cross_entropy / len(batch)
            if self.anchor_weight > 0 and self._anchor_reference is not None:
                penalty = anchor_penalty(
                    network.encoder_stage_parameters(TRAINABLE_ENCODER_STAGE),
                    self._anchor_reference,
                )
                loss = loss + (self.anchor_weight / len(batch)) * penalty
                anchor_value = float(penalty.detach())
            loss.backward()
            cross_entropy_sum += float(cross_entropy.detach())
            correct += int((logits.detach().argmax(dim=1) == target).sum())

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.classification_parameter_set(), self.classification_grad_clip
            )
        )
        optimizer.step()
        return {
            "cls_loss": cross_entropy_sum / len(batch),
            "cls_anchor": anchor_value,
            "cls_correct": correct,
            "cls_count": len(batch),
            "cls_grad_norm": grad_norm,
        }

    def classification_learning_rate_at(self, epoch: int) -> float:
        """Five-epoch linear warmup, then cosine to zero."""

        warmup = 5
        total = max(int(self.num_epochs), warmup + 1)
        index = int(epoch)
        if index < warmup:
            return float(self.classification_learning_rate * (index + 1) / warmup)
        position = (index - warmup) / max(total - warmup - 1, 1)
        return float(
            self.classification_learning_rate * 0.5 * (1.0 + np.cos(np.pi * min(position, 1.0)))
        )


__all__ = [
    "CLASSIFICATION_TAP",
    "TRAINABLE_ENCODER_STAGE",
    "anchor_penalty",
    "nnUNetTrainerPancreasMultiTaskV7",
]
