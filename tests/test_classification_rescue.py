from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.classification_rescue import (
    ACTIVATION_AUDIT_SCHEMA_VERSION,
    RESCUE_SCHEMA_VERSION,
    RescueSchedule,
    build_inference_checkpoint,
    classification_activation_audit,
    classification_parameter_items,
    classification_rescue_train_step,
    component_hashes,
    freeze_for_classification_rescue,
    reset_classification_parameters,
    strict_load_network_weights,
    validate_activation_audit,
    validate_disjoint_split,
    validate_resume_state,
)
from pancreas_multitask.network import MultiTaskResEncUNet


class _ToyEncoder(nn.Module):
    output_channels = (4, 8)

    def __init__(self) -> None:
        super().__init__()
        self.high = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.low = nn.Conv3d(4, 8, kernel_size=3, padding=1)

    def forward(self, inputs):
        high = self.high(inputs)
        low = self.low(F.avg_pool3d(high, kernel_size=2))
        return [high, low]

    @staticmethod
    def compute_conv_feature_map_size(_input_size):
        return 1


class _TrackingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deep_supervision = False
        self.segmentation = nn.Conv3d(4, 3, kernel_size=1)
        self.calls = 0

    def forward(self, skips):
        self.calls += 1
        return self.segmentation(skips[0])

    @staticmethod
    def compute_conv_feature_map_size(_input_size):
        return 1


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ToyEncoder()
        self.decoder = _TrackingDecoder()


def _model() -> MultiTaskResEncUNet:
    torch.manual_seed(11)
    return MultiTaskResEncUNet(
        _ToyBackbone(),
        classification_hidden_channels=12,
        classification_dropout=0.0,
    )


def test_classification_forward_bypasses_decoder() -> None:
    model = _model().eval()
    logits = model.forward_classification(torch.randn(2, 1, 8, 8, 8))

    assert logits.shape == (2, 3)
    assert model.decoder.calls == 0


def test_reset_changes_only_classification_state() -> None:
    model = _model()
    before = component_hashes(model)

    reset_classification_parameters(model, seed=42)
    after = component_hashes(model)

    assert after["encoder"] == before["encoder"]
    assert after["decoder"] == before["decoder"]
    assert after["classification"] != before["classification"]


def test_frozen_rescue_step_updates_only_registered_classification_parameters() -> None:
    model = _model()
    reset_classification_parameters(model, seed=19)
    immutable_before = component_hashes(model)
    items = freeze_for_classification_rescue(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in items],
        lr=3e-4,
        weight_decay=1e-4,
    )
    schedule = RescueSchedule(epochs=1, iterations_per_epoch=1)
    data = torch.randn(3, 1, 8, 8, 8)
    targets = torch.tensor([0, 1, 2])
    segmentation = torch.zeros(3, 1, 8, 8, 8, dtype=torch.long)
    segmentation[1:, :, 2:4, 2:4, 2:4] = 2

    result = classification_rescue_train_step(
        model,
        data,
        targets,
        segmentation,
        optimizer=optimizer,
        class_weights=torch.ones(3),
        schedule=schedule,
    )
    after = component_hashes(model)

    assert result.sample_count == 3
    assert result.lesion_patch_count == 2
    assert sum(result.target_counts) == 3
    assert model.decoder.calls == 0
    assert after["encoder"] == immutable_before["encoder"]
    assert after["decoder"] == immutable_before["decoder"]
    assert after["classification"] != immutable_before["classification"]
    assert all(
        name.startswith(("classification_pool.", "classification_head.")) for name, _ in items
    )
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert {id(parameter) for _, parameter in classification_parameter_items(model)} == {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }


def test_inference_checkpoint_is_strictly_reloadable_without_fake_joint_optimizer() -> None:
    source_model = _model()
    rescue_state = {
        "schema_version": RESCUE_SCHEMA_VERSION,
        "source_checkpoint_sha256": "a" * 64,
        "schedule": RescueSchedule().to_dict(),
        "completed_epochs": 1,
        "optimizer_state": {"state": {}, "param_groups": []},
        "rng_state": {},
    }
    source_checkpoint = {
        "trainer_name": "nnUNetTrainerPancreasMultiTask",
        "init_args": {"configuration": "3d_fullres"},
        "inference_allowed_mirroring_axes": (0, 1, 2),
        "optimizer_state": {"unsafe": "joint state must not be copied"},
    }

    checkpoint = build_inference_checkpoint(
        source_checkpoint,
        source_model,
        rescue_state,
    )
    restored_model = _model()
    strict_load_network_weights(restored_model, checkpoint)

    assert set(checkpoint) == {
        "network_weights",
        "trainer_name",
        "init_args",
        "inference_allowed_mirroring_axes",
        "classification_rescue",
    }
    assert "optimizer_state" not in checkpoint
    assert component_hashes(restored_model) == component_hashes(source_model)


def test_resume_requires_identical_source_and_frozen_schedule() -> None:
    schedule = RescueSchedule(epochs=2, iterations_per_epoch=3)
    state = {
        "schema_version": RESCUE_SCHEMA_VERSION,
        "source_checkpoint_sha256": "b" * 64,
        "schedule": schedule.to_dict(),
        "completed_epochs": 1,
        "optimizer_state": {},
        "rng_state": {},
        "training_only_history": [{}],
        "current_component_sha256": {
            "encoder": "e",
            "decoder": "d",
            "classification": "c",
        },
    }

    assert (
        validate_resume_state(
            state,
            source_checkpoint_sha256="b" * 64,
            schedule=schedule,
        )
        == 1
    )
    with pytest.raises(ValueError, match="different source"):
        validate_resume_state(
            state,
            source_checkpoint_sha256="c" * 64,
            schedule=schedule,
        )
    with pytest.raises(ValueError, match="frozen schedule"):
        validate_resume_state(
            state,
            source_checkpoint_sha256="b" * 64,
            schedule=RescueSchedule(epochs=3, iterations_per_epoch=3),
        )


def test_split_audit_records_zero_validation_use_and_rejects_overlap() -> None:
    audit = validate_disjoint_split(
        ["quiz_0_001", "quiz_1_002"],
        ["quiz_2_003"],
        expected_training_cases=2,
        expected_validation_cases=1,
    )

    assert audit["split_disjoint"] is True
    assert audit["validation_images_opened"] is False
    assert audit["validation_batches_consumed"] == 0
    assert audit["validation_used_for_gradients"] is False
    assert audit["validation_used_for_stopping"] is False
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_split(
            ["same"],
            ["same"],
            expected_training_cases=1,
            expected_validation_cases=1,
        )


def test_activation_audit_uses_only_predeclared_training_windows() -> None:
    losses = [0.9] * 51
    accuracies = [0.5] * 51
    losses[31:41] = [1.06] * 10
    accuracies[31:41] = [0.40] * 10

    audit = classification_activation_audit(losses, accuracies)

    assert audit["activation_approved"] is True
    assert audit["decision_epoch"] == 40
    assert audit["epoch_40"]["window_epochs"] == list(range(31, 41))
    assert audit["validation_metrics_read"] is False
    assert audit["validation_used_for_activation"] is False


def test_activation_gate_is_hash_bound_and_must_be_affirmative() -> None:
    audit = {
        "schema_version": ACTIVATION_AUDIT_SCHEMA_VERSION,
        "source_checkpoint_sha256": "d" * 64,
        "source_checkpoint_name": "checkpoint_final.pth",
        "checkpoint_current_epoch": 200,
        "training_logging_epoch_count": 200,
        "metric_scope": "checkpoint_training_logging_only",
        "validation_used_for_activation": False,
        "activation_approved": True,
    }

    validate_activation_audit(audit, source_checkpoint_sha256="d" * 64)
    with pytest.raises(ValueError, match="different source"):
        validate_activation_audit(audit, source_checkpoint_sha256="e" * 64)
    audit["activation_approved"] = False
    with pytest.raises(ValueError, match="did not approve"):
        validate_activation_audit(audit, source_checkpoint_sha256="d" * 64)


def test_rescue_cli_defaults_match_predeclared_schedule() -> None:
    script_path = ROOT / "scripts" / "train_classification_rescue.py"
    spec = importlib.util.spec_from_file_location("train_classification_rescue", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "--source-checkpoint",
            "source.pth",
            "--output-checkpoint",
            "rescue.pth",
            "--activation-audit",
            "activation.json",
        ]
    )
    schedule = module._schedule_from_args(args)

    assert schedule == RescueSchedule()
    assert schedule.epochs == 30
    assert args.expected_training_cases == 252
    assert args.expected_validation_cases == 36
    assert args.resume is False
