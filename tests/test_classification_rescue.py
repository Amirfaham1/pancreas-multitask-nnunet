from __future__ import annotations

import importlib.util
import json
import subprocess
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
    validate_frozen_split_manifest,
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
    assert len(items) == 15
    assert all(
        name.startswith(("classification_pool.", "classification_head.")) for name, _ in items
    )
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
    assert result.optimizer_update_count == 1
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


def test_rescue_confines_autocast_to_frozen_encoder_and_updates_head_in_float32() -> None:
    model = _model()
    reset_classification_parameters(model, seed=23)
    items = freeze_for_classification_rescue(model)
    optimizer = torch.optim.AdamW([parameter for _, parameter in items], lr=3e-4)
    classification_input_dtypes = []
    hook = model.classification_pool.register_forward_pre_hook(
        lambda _module, inputs: classification_input_dtypes.append(inputs[0].dtype)
    )

    try:
        result = classification_rescue_train_step(
            model,
            torch.randn(2, 1, 8, 8, 8),
            torch.tensor([0, 2]),
            torch.zeros(2, 1, 8, 8, 8, dtype=torch.long),
            optimizer=optimizer,
            class_weights=torch.ones(3),
            schedule=RescueSchedule(epochs=1, iterations_per_epoch=1),
            scaler=torch.GradScaler("cpu", enabled=False),
            use_encoder_amp=True,
        )
    finally:
        hook.remove()

    assert classification_input_dtypes == [torch.float32]
    assert result.optimizer_update_count == 1
    assert all(parameter.grad is not None for _, parameter in items)
    assert all(parameter.grad.dtype == torch.float32 for _, parameter in items)
    assert all(torch.isfinite(parameter.grad).all() for _, parameter in items)


def test_rescue_rejects_enabled_gradient_scaler_before_forward() -> None:
    class _EnabledScaler:
        @staticmethod
        def is_enabled() -> bool:
            return True

    model = _model()
    items = freeze_for_classification_rescue(model)
    optimizer = torch.optim.AdamW([parameter for _, parameter in items], lr=3e-4)

    with pytest.raises(RuntimeError, match="GradScaler to remain disabled"):
        classification_rescue_train_step(
            model,
            torch.randn(2, 1, 8, 8, 8),
            torch.tensor([0, 1]),
            torch.zeros(2, 1, 8, 8, 8, dtype=torch.long),
            optimizer=optimizer,
            class_weights=torch.ones(3),
            schedule=RescueSchedule(epochs=1, iterations_per_epoch=1),
            scaler=_EnabledScaler(),
        )

    assert model.decoder.calls == 0
    assert not optimizer.state


def test_rescue_names_nonfinite_gradient_and_refuses_optimizer_update() -> None:
    model = _model()
    reset_classification_parameters(model, seed=29)
    items = freeze_for_classification_rescue(model)
    optimizer = torch.optim.AdamW([parameter for _, parameter in items], lr=3e-4)
    before = component_hashes(model)
    bad_parameter = dict(items)["classification_head.4.bias"]
    hook = bad_parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))

    try:
        with pytest.raises(
            FloatingPointError,
            match=r"classification_head\.4\.bias \(nonfinite=3",
        ):
            classification_rescue_train_step(
                model,
                torch.randn(2, 1, 8, 8, 8),
                torch.tensor([0, 1]),
                torch.zeros(2, 1, 8, 8, 8, dtype=torch.long),
                optimizer=optimizer,
                class_weights=torch.ones(3),
                schedule=RescueSchedule(epochs=1, iterations_per_epoch=1),
            )
    finally:
        hook.remove()

    assert component_hashes(model) == before
    assert not optimizer.state


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
        "started_at_utc": "2026-08-06T00:00:00+00:00",
        "elapsed_seconds": 12.5,
        "resume_count": 0,
        "successful_optimizer_updates": 3,
        "training_only_history": [{"successful_optimizer_updates": 3}],
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


def test_complete_resume_state_can_repair_a_missing_public_audit() -> None:
    schedule = RescueSchedule(epochs=2, iterations_per_epoch=3)
    state = {
        "schema_version": RESCUE_SCHEMA_VERSION,
        "source_checkpoint_sha256": "b" * 64,
        "schedule": schedule.to_dict(),
        "status": "complete",
        "completed_epochs": 2,
        "optimizer_state": {},
        "rng_state": {},
        "started_at_utc": "2026-08-06T00:00:00+00:00",
        "elapsed_seconds": 25.0,
        "resume_count": 1,
        "successful_optimizer_updates": 6,
        "training_only_history": [
            {"successful_optimizer_updates": 3},
            {"successful_optimizer_updates": 3},
        ],
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
        == 2
    )
    state["status"] = "in_progress"
    with pytest.raises(ValueError, match="status='complete'"):
        validate_resume_state(
            state,
            source_checkpoint_sha256="b" * 64,
            schedule=schedule,
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


def test_live_split_is_hash_bound_to_frozen_pretraining_manifest() -> None:
    manifest = {
        "schema_version": 1,
        "source_splits_preserved": True,
        "planning_case_ids": ["train_b", "train_a"],
        "train_case_ids": ["train_a", "train_b"],
        "validation_case_ids": ["validation_a"],
    }

    binding = validate_frozen_split_manifest(
        manifest,
        ["train_b", "train_a"],
        ["validation_a"],
    )

    assert binding["matches_frozen_split_manifest"] is True
    assert binding["frozen_manifest_training_case_count"] == 2
    assert binding["frozen_manifest_validation_case_count"] == 1
    assert len(binding["frozen_manifest_training_case_ids_sha256"]) == 64
    assert len(binding["frozen_manifest_validation_case_ids_sha256"]) == 64

    with pytest.raises(ValueError, match="Live training split differs"):
        validate_frozen_split_manifest(
            manifest,
            ["train_a", "validation_a"],
            ["train_b"],
        )


def test_resume_rejects_invalid_cumulative_timing_provenance() -> None:
    schedule = RescueSchedule(epochs=2, iterations_per_epoch=3)
    state = {
        "schema_version": RESCUE_SCHEMA_VERSION,
        "source_checkpoint_sha256": "b" * 64,
        "schedule": schedule.to_dict(),
        "status": "in_progress",
        "completed_epochs": 1,
        "optimizer_state": {},
        "rng_state": {},
        "started_at_utc": "2026-08-06T00:00:00+00:00",
        "elapsed_seconds": -1.0,
        "resume_count": 0,
        "training_only_history": [{}],
        "current_component_sha256": {
            "encoder": "e",
            "decoder": "d",
            "classification": "c",
        },
    }

    with pytest.raises(ValueError, match="cumulative elapsed_seconds"):
        validate_resume_state(
            state,
            source_checkpoint_sha256="b" * 64,
            schedule=schedule,
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
    assert args.recovery_audit is None
    recovered_args = module.build_parser().parse_args(
        [
            "--source-checkpoint",
            "source.pth",
            "--output-checkpoint",
            "rescue.pth",
            "--activation-audit",
            "activation.json",
            "--recovery-audit",
            "classification_rescue_zero_update_recovery.json",
        ]
    )
    assert recovered_args.recovery_audit == Path("classification_rescue_zero_update_recovery.json")
    precision = module._precision_policy(torch.device("cuda"))
    assert precision["frozen_encoder_forward"] == "cuda_autocast_float16"
    assert precision["trainable_classification_forward"] == "float32"
    assert precision["classification_backward"] == "float32"
    assert precision["grad_scaler_enabled"] is False


def test_zero_update_recovery_artifact_is_strictly_hash_bound(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "train_classification_rescue.py"
    spec = importlib.util.spec_from_file_location("train_classification_rescue", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "checkpoint_final.pth"
    source.write_bytes(b"fixed source")
    source_hash = module.file_sha256(source)
    activation = tmp_path / "classification_rescue_activation.json"
    activation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "activation_approved": True,
                "source_checkpoint_name": "checkpoint_final.pth",
                "source_checkpoint_sha256": source_hash,
                "validation_metrics_read": False,
                "validation_used_for_activation": False,
            }
        ),
        encoding="utf-8",
    )
    activation_hash = module.file_sha256(activation)
    evidence = tmp_path / "classification_rescue_recovery_evidence"
    evidence.mkdir()
    stdout_log = evidence / "failed_launch.stdout.log"
    stderr_log = evidence / "failed_launch.stderr.log"
    stdout_log.write_text(
        "ACTIVATION_APPROVED=true\n"
        "CLASSIFICATION_RESCUE_START\n"
        "Using splits from existing split file\n"
        "This split has 252 training and 36 validation cases.\n",
        encoding="utf-8",
    )
    stderr_log.write_text(
        "classification_rescue_train_step\n"
        "clip_grad_norm_\n"
        "gradients from `parameters` is non-finite\n"
        "Classification rescue exited with code 1\n"
        "OVERNIGHT_PIPELINE_FAILED\n",
        encoding="utf-8",
    )
    stdout_artifact = {
        "name": "classification_rescue_recovery_evidence/failed_launch.stdout.log",
        "bytes": stdout_log.stat().st_size,
        "sha256": module.file_sha256(stdout_log),
    }
    stderr_artifact = {
        "name": "classification_rescue_recovery_evidence/failed_launch.stderr.log",
        "bytes": stderr_log.stat().st_size,
        "sha256": module.file_sha256(stderr_log),
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def git_blob(relative_path: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", f"{commit}:{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    payload = {
        "schema_version": 1,
        "event": "classification_rescue_zero_update_execution_recovery",
        "status": "authorized_before_custom_joint_fixed_validation",
        "source_checkpoint_name": "checkpoint_final.pth",
        "source_checkpoint_sha256": source_hash,
        "activation_audit_sha256": activation_hash,
        "activation_approved": True,
        "git_commit_at_failed_launch": commit,
        "rescue_protocol_commit": commit,
        "pre_failure_implementation": {
            "train_script_git_blob": git_blob("scripts/train_classification_rescue.py"),
            "rescue_module_git_blob": git_blob("src/pancreas_multitask/classification_rescue.py"),
        },
        "failed_launch": {
            "process_launch_index": 1,
            "failed_step_index": 0,
            "training_batches_consumed": 1,
            "training_samples_consumed": 2,
            "finite_loss_guard_passed": True,
            "failure_stage": "after_grad_scaler_unscale_before_gradient_clip_completion",
            "exception_type": "RuntimeError",
            "optimizer_step_reached": False,
            "optimizer_updates": 0,
            "completed_epochs": 0,
            "checkpoint_written": False,
            "rescue_audit_written": False,
            "first_step_zero_update_operator_attested": True,
            "stdout_artifact": stdout_artifact,
            "stderr_artifact": stderr_artifact,
        },
        "validation": {
            "stock_nnunet_segmentation_only_validation_completed": True,
            "stock_nnunet_validation_metrics_observed_before_recovery": True,
            "stock_nnunet_mean_foreground_dice_observed_before_recovery": 0.753518646,
            "stock_nnunet_validation_used_for_recovery": False,
            "custom_joint_fixed_validation_started": False,
            "custom_joint_fixed_validation_output_existed_at_authorization": False,
            "rescue_process_validation_images_opened": False,
            "rescue_process_validation_batches_consumed": 0,
            "rescue_process_validation_used_for_recovery": False,
        },
        "recovery_policy": {
            "schedule_changed": False,
            "source_checkpoint_changed": False,
            "reset_seed_changed": False,
            "maximum_update_bearing_trajectories": 1,
            "maximum_zero_update_runtime_recoveries": 1,
            "process_launch_count_after_relaunch": 2,
            "no_further_recovery_allowed": True,
        },
    }
    recovery_path = tmp_path / "classification_rescue_zero_update_recovery.json"
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, digest = module._load_execution_recovery(
        recovery_path,
        source_checkpoint=source,
        activation_audit=activation,
    )

    assert loaded == payload
    assert len(digest) == 64
    payload["failed_launch"]["optimizer_updates"] = 1
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="optimizer_updates"):
        module._load_execution_recovery(
            recovery_path,
            source_checkpoint=source,
            activation_audit=activation,
        )


def test_rescue_cli_rejects_resume_and_clean_recovery_laundering(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "train_classification_rescue.py"
    spec = importlib.util.spec_from_file_location("train_classification_rescue", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    resume_args = module.build_parser().parse_args(
        [
            "--source-checkpoint",
            "source.pth",
            "--output-checkpoint",
            "rescue.pth",
            "--activation-audit",
            "activation.json",
            "--resume",
        ]
    )
    with pytest.raises(ValueError, match="exactly one uninterrupted"):
        module._resolved_paths(resume_args)

    source = tmp_path / "checkpoint_final.pth"
    source.write_bytes(b"source")
    activation = tmp_path / "classification_rescue_activation.json"
    activation.write_text("{}\n", encoding="utf-8")
    canonical_recovery = tmp_path / "classification_rescue_zero_update_recovery.json"
    canonical_recovery.write_text("{}\n", encoding="utf-8")
    clean_args = module.build_parser().parse_args(
        [
            "--source-checkpoint",
            str(source),
            "--output-checkpoint",
            str(tmp_path / "checkpoint_classification_rescue.pth"),
            "--activation-audit",
            str(activation),
        ]
    )
    with pytest.raises(ValueError, match="Canonical execution-recovery evidence exists"):
        module._resolved_paths(clean_args)


def test_rescue_script_preserves_cumulative_resume_timing_and_manifest_binding() -> None:
    source = (ROOT / "scripts" / "train_classification_rescue.py").read_text(encoding="utf-8")

    assert '"started_at_utc": original_started_at' in source
    assert '"elapsed_seconds": prior_elapsed_seconds' in source
    assert "+ (time.perf_counter() - wall_start)" in source
    assert 'Path(trainer.preprocessed_dataset_folder_base) / "split_manifest.json"' in source
    assert "validate_frozen_split_manifest(" in source
    assert '"frozen_split_manifest_sha256": hash_before' in source


def test_activation_audit_refuses_to_replace_existing_artifact(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "audit_classification_rescue_activation.py"
    spec = importlib.util.spec_from_file_location("classification_rescue_activation", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    checkpoint = tmp_path / "checkpoint_final.pth"
    checkpoint.write_bytes(b"not-read-because-output-exists")
    output = tmp_path / "classification_rescue_activation.json"
    output.write_text('{"immutable": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable"):
        module.run(checkpoint, output)


def test_rescue_wrapper_never_deletes_the_python_ownership_lock() -> None:
    source = (ROOT / "scripts" / "Run-ClassificationRescue.ps1").read_text(encoding="utf-8")

    assert "Remove-Item" not in source
    assert '"train_classification_rescue.py"' in source
    assert '"Local\\PancreasMultitaskPostTraining501Fold0"' in source
    assert "$postTrainingMutex.WaitOne(0)" in source
    assert "$postTrainingMutex.ReleaseMutex()" in source
    assert "$recoveryAuditExplicit" in source
    assert "Explicit zero-update execution-recovery audit is missing" in source
    assert '$arguments += @("--recovery-audit", $resolvedRecoveryAudit)' in source
    assert "Resuming this fixed rescue is prohibited" in source
    assert '$arguments += "--resume"' not in source
