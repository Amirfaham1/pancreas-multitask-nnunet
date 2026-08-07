"""Guards for the V7 optimizer split and shallow classification tap.

The original trainer's classification head silently inherited nnU-Net's
SGD(lr=1e-2, momentum=0.99), which is an effective step of order 1.0 on a
0.02-scale head; the loss sat at ln 3 for 200 epochs as a result. These tests
exist so that regression cannot recur unnoticed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from pancreas_multitask.network import GlobalAveragePool3D, MultiTaskResEncUNet
from pancreas_multitask.trainer_v7 import (
    CLASSIFICATION_TAP,
    TRAINABLE_ENCODER_STAGE,
    anchor_penalty,
    nnUNetTrainerPancreasMultiTaskV7,
)

BUNDLE = Path(__file__).resolve().parent / "fixtures" / "v7"


def _network(tap: int = CLASSIFICATION_TAP) -> MultiTaskResEncUNet:
    plans_manager = PlansManager(str(BUNDLE / "plans.json"))
    configuration = plans_manager.get_configuration("3d_fullres")
    segmentation = get_network_from_plans(
        configuration.network_arch_class_name,
        configuration.network_arch_init_kwargs,
        configuration.network_arch_init_kwargs_req_import,
        1,
        3,
        allow_init=True,
        deep_supervision=False,
    )
    return MultiTaskResEncUNet(segmentation, classification_tap=tap)


def _trainer_stub(network: MultiTaskResEncUNet) -> nnUNetTrainerPancreasMultiTaskV7:
    """Exercise configure_optimizers without nnU-Net's full dataset machinery."""

    trainer = nnUNetTrainerPancreasMultiTaskV7.__new__(nnUNetTrainerPancreasMultiTaskV7)
    trainer.network = network
    trainer.initial_lr = 1e-2
    trainer.weight_decay = 3e-5
    trainer.num_epochs = 50
    trainer.classification_learning_rate = 3e-4
    trainer.encoder_learning_rate = 3e-5
    trainer.classification_weight_decay = 1e-2
    trainer.anchor_weight = 1e-3
    trainer.classification_optimizer = None
    trainer._anchor_reference = None
    return trainer


def test_classification_head_is_not_in_the_sgd_optimizer() -> None:
    """The defect being guarded: the head must never see SGD(momentum=0.99)."""

    network = _network()
    trainer = _trainer_stub(network)
    optimizer, _scheduler = trainer.configure_optimizers()

    head_ids = {id(p) for p in network.classification_parameters()}
    sgd_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert head_ids, "the classification branch must expose parameters"
    assert not (head_ids & sgd_ids), "classification parameters leaked into the SGD optimizer"

    adam_ids = {
        id(p)
        for group in trainer.classification_optimizer.param_groups
        for p in group["params"]
    }
    assert head_ids <= adam_ids, "classification parameters are missing from AdamW"


def test_shallow_encoder_receives_both_gradients() -> None:
    """Stages 0-2 stay in SGD too: that segmentation gradient is what protects Dice."""

    network = _network()
    trainer = _trainer_stub(network)
    optimizer, _scheduler = trainer.configure_optimizers()

    shallow_ids = {id(p) for p in network.encoder_stage_parameters(TRAINABLE_ENCODER_STAGE)}
    sgd_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    adam_ids = {
        id(p)
        for group in trainer.classification_optimizer.param_groups
        for p in group["params"]
    }
    assert shallow_ids <= sgd_ids, "shallow stages lost the segmentation gradient"
    assert shallow_ids <= adam_ids, "shallow stages lost the classification gradient"


def test_classification_optimizer_uses_distinct_learning_rates() -> None:
    network = _network()
    trainer = _trainer_stub(network)
    trainer.configure_optimizers()
    rates = [group["lr"] for group in trainer.classification_optimizer.param_groups]
    assert rates == [trainer.classification_learning_rate, trainer.encoder_learning_rate]


def test_classification_schedule_warms_up_then_decays_to_zero() -> None:
    network = _network()
    trainer = _trainer_stub(network)
    trainer.configure_optimizers()
    schedule = [trainer.classification_learning_rate_at(e) for e in range(trainer.num_epochs)]
    assert schedule[0] == pytest.approx(trainer.classification_learning_rate / 5)
    assert schedule[4] == pytest.approx(trainer.classification_learning_rate)
    assert schedule[-1] == pytest.approx(0.0, abs=1e-12)
    assert all(b >= a for a, b in zip(schedule[:5], schedule[1:5]))
    assert all(b <= a for a, b in zip(schedule[4:-1], schedule[5:]))


def test_anchor_penalty_is_zero_at_the_reference_and_positive_after_drift() -> None:
    network = _network()
    trainer = _trainer_stub(network)
    trainer.configure_optimizers()
    parameters = network.encoder_stage_parameters(TRAINABLE_ENCODER_STAGE)
    assert float(anchor_penalty(parameters, trainer._anchor_reference)) == pytest.approx(0.0)
    with torch.no_grad():
        parameters[0].add_(0.1)
    assert float(anchor_penalty(parameters, trainer._anchor_reference)) > 0.0


def test_shallow_tap_uses_plain_pooling_and_a_small_head() -> None:
    """At 252 cases every >=100k-parameter head memorized; keep this one tiny."""

    network = _network()
    assert isinstance(network.classification_pool, GlobalAveragePool3D)
    assert sum(p.numel() for p in network.classification_parameters()) < 5_000


def test_default_tap_preserves_the_original_architecture() -> None:
    """Existing checkpoints must keep loading strict=True."""

    network = _network(tap=-1)
    assert network.classification_tap == 5
    assert sum(p.numel() for p in network.classification_parameters()) == 496_195


def test_encode_to_stage_matches_the_full_encoder_skip() -> None:
    network = _network().eval()
    x = torch.zeros(1, 1, 32, 64, 64)
    with torch.no_grad():
        assert torch.equal(network.encode_to_stage(x), network.encoder(x)[CLASSIFICATION_TAP])
