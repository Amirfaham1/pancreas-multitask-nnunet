"""Optional fixed-schedule classification-head rescue after joint training.

This command is intentionally separate from nnU-Net's normal training loop.
The stock ``get_dataloaders`` method constructs and primes a validation loader;
this script instead constructs one loader from the declared training keys only.
No validation image, loss, metric, gradient, stopping decision, or checkpoint
ranking enters this optimization path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pancreas_multitask.classification_rescue import (
    RESCUE_SCHEMA_VERSION,
    RescueSchedule,
    atomic_torch_save,
    atomic_write_json,
    build_inference_checkpoint,
    capture_rng_state,
    classification_activation_audit,
    classification_rescue_train_step,
    component_hashes,
    file_sha256,
    freeze_for_classification_rescue,
    reset_classification_parameters,
    restore_rng_state,
    seed_all,
    strict_load_network_weights,
    validate_activation_audit,
    validate_disjoint_split,
    validate_frozen_split_manifest,
    validate_resume_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reinitialize and train only the classification pool/head from an "
            "existing joint checkpoint, using training cases only."
        )
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--activation-audit", type=Path, required=True)
    parser.add_argument(
        "--recovery-audit",
        type=Path,
        help=(
            "Optional immutable evidence for a realized zero-update numerical recovery. "
            "Omit on a genuinely clean first process launch."
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        help="Default: <output-checkpoint>.audit.json",
    )
    parser.add_argument("--dataset", default="501")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--trainer", default="nnUNetTrainerPancreasMultiTask")
    parser.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--iterations-per-epoch", type=int, default=125)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--nonlesion-patch-weight", type=float, default=0.25)
    parser.add_argument("--reset-seed", type=int, default=20260806)
    parser.add_argument("--expected-training-cases", type=int, default=252)
    parser.add_argument("--expected-validation-cases", type=int, default=36)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Write an auditable interruption snapshot every N completed epochs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Rejected by this fixed protocol, which permits one uninterrupted "
            "update-bearing trajectory."
        ),
    )
    return parser


def _schedule_from_args(args: argparse.Namespace) -> RescueSchedule:
    return RescueSchedule(
        epochs=args.epochs,
        iterations_per_epoch=args.iterations_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        label_smoothing=args.label_smoothing,
        nonlesion_patch_weight=args.nonlesion_patch_weight,
        reset_seed=args.reset_seed,
    )


def _precision_policy(device: torch.device) -> dict[str, str | bool]:
    """Describe the fixed numerical boundary used by the rescue update."""

    return {
        "autocast_scope": "frozen_encoder_forward_only",
        "frozen_encoder_forward": ("cuda_autocast_float16" if device.type == "cuda" else "float32"),
        "trainable_classification_forward": "float32",
        "classification_loss": "float32",
        "classification_backward": "float32",
        "gradient_clipping": "float32",
        "optimizer_update": "float32",
        "grad_scaler_enabled": False,
    }


def _resolved_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path | None]:
    if args.resume:
        raise ValueError(
            "Resuming this fixed rescue is prohibited: the provenance contract allows "
            "exactly one uninterrupted update-bearing trajectory"
        )
    source = args.source_checkpoint.expanduser().resolve()
    output = args.output_checkpoint.expanduser().resolve()
    activation_audit = args.activation_audit.expanduser().resolve()
    recovery_audit = (
        args.recovery_audit.expanduser().resolve() if args.recovery_audit is not None else None
    )
    audit = (
        args.audit_json.expanduser().resolve()
        if args.audit_json is not None
        else output.with_name(f"{output.name}.audit.json")
    )
    if source == output:
        raise ValueError("source-checkpoint and output-checkpoint must be different files")
    if audit in (source, output):
        raise ValueError("audit-json must be different from both checkpoint paths")
    if activation_audit in (source, output, audit) or activation_audit == recovery_audit:
        raise ValueError("activation-audit must be a separate JSON file")
    if recovery_audit is not None and recovery_audit in (source, output, audit, activation_audit):
        raise ValueError("recovery-audit must be a separate JSON file")
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")
    if source.name != "checkpoint_final.pth":
        raise ValueError(
            f"The predeclared rescue source must be named checkpoint_final.pth; got {source.name!r}"
        )
    canonical_recovery_audit = source.with_name("classification_rescue_zero_update_recovery.json")
    if recovery_audit is None and canonical_recovery_audit.is_file():
        raise ValueError(
            "Canonical execution-recovery evidence exists beside the source checkpoint; "
            "pass it explicitly with --recovery-audit"
        )
    if not activation_audit.is_file():
        raise FileNotFoundError(f"Activation audit does not exist: {activation_audit}")
    if recovery_audit is not None:
        if not recovery_audit.is_file():
            raise FileNotFoundError(f"Execution recovery audit does not exist: {recovery_audit}")
        if recovery_audit.name != "classification_rescue_zero_update_recovery.json":
            raise ValueError(
                "Execution recovery audit must use the declared filename "
                "classification_rescue_zero_update_recovery.json"
            )
        if recovery_audit.parent != source.parent:
            raise ValueError(
                "Execution recovery audit must be a direct child of the source fold directory"
            )
    if args.save_every < 1:
        raise ValueError("save-every must be positive")
    if output.exists():
        raise FileExistsError(
            f"Output already exists; this fixed rescue cannot be restarted or resumed: {output}"
        )
    return source, output, audit, activation_audit, recovery_audit


def _load_execution_recovery(
    path: Path,
    *,
    source_checkpoint: Path,
    activation_audit: Path,
) -> tuple[dict[str, Any], str]:
    """Validate and hash-bind the authorized zero-update recovery artifact."""

    from classification_rescue_recovery import validate_recovery_payload

    hash_before = file_sha256(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    hash_after = file_sha256(path)
    if hash_before != hash_after:
        raise RuntimeError("Execution recovery audit changed while it was being read")
    if not isinstance(payload, dict):
        raise TypeError("Execution recovery audit must contain a JSON object")
    validate_recovery_payload(
        payload,
        artifact_path=path,
        source_checkpoint=source_checkpoint,
        activation_audit=activation_audit,
    )
    if file_sha256(path) != hash_before:
        raise RuntimeError("Execution recovery audit changed during validation")
    return payload, hash_before


@contextmanager
def _exclusive_output_lock(output: Path):
    lock_path = output.with_name(f"{output.name}.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another rescue may already own lock: {lock_path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _build_training_only_dataloader(trainer: Any) -> tuple[Any, list[str], list[str]]:
    """Mirror nnU-Net's training loader construction without a val loader."""

    from batchgenerators.dataloading.single_threaded_augmenter import (
        SingleThreadedAugmenter,
    )
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    if trainer.dataset_class is None:
        trainer.dataset_class = infer_dataset_class(trainer.preprocessed_dataset_folder)

    patch_size = trainer.configuration_manager.patch_size
    deep_supervision_scales = trainer._get_deep_supervision_scales()
    (
        rotation_for_augmentation,
        use_dummy_2d,
        initial_patch_size,
        mirror_axes,
    ) = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    transforms = trainer.get_training_transforms(
        patch_size,
        rotation_for_augmentation,
        deep_supervision_scales,
        mirror_axes,
        use_dummy_2d,
        use_mask_for_norm=trainer.configuration_manager.use_mask_for_norm,
        is_cascaded=trainer.is_cascaded,
        foreground_labels=trainer.label_manager.foreground_labels,
        regions=(
            trainer.label_manager.foreground_regions if trainer.label_manager.has_regions else None
        ),
        ignore_label=trainer.label_manager.ignore_label,
    )

    training_keys, validation_keys = trainer.do_split()
    training_keys = [str(case_id) for case_id in training_keys]
    validation_keys = [str(case_id) for case_id in validation_keys]
    dataset = trainer.dataset_class(
        trainer.preprocessed_dataset_folder,
        training_keys,
        folder_with_segs_from_previous_stage=trainer.folder_with_segs_from_previous_stage,
    )
    loader = nnUNetDataLoader(
        dataset,
        trainer.batch_size,
        initial_patch_size,
        patch_size,
        trainer.label_manager,
        oversample_foreground_percent=trainer.oversample_foreground_percent,
        sampling_probabilities=None,
        pad_sides=None,
        transforms=transforms,
        probabilistic_oversampling=trainer.probabilistic_oversampling,
    )
    return SingleThreadedAugmenter(loader, None), training_keys, validation_keys


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _frozen_split_manifest_binding(
    trainer: Any,
    training_keys: list[str],
    validation_keys: list[str],
) -> dict[str, Any]:
    """Load one stable manifest snapshot and bind it to the live split."""

    manifest_path = (
        Path(trainer.preprocessed_dataset_folder_base) / "split_manifest.json"
    ).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen pretraining split manifest does not exist: {manifest_path}"
        )
    hash_before = file_sha256(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    hash_after = file_sha256(manifest_path)
    if hash_before != hash_after:
        raise RuntimeError("Frozen split manifest changed while it was being read")
    if not isinstance(manifest, dict):
        raise TypeError("Frozen split manifest must contain a JSON object")
    binding = validate_frozen_split_manifest(
        manifest,
        training_keys,
        validation_keys,
    )
    return {
        **binding,
        "frozen_split_manifest": str(manifest_path),
        "frozen_split_manifest_sha256": hash_before,
    }


def _checkpoint_metadata(source_checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: source_checkpoint[key]
        for key in ("trainer_name", "init_args")
        if key in source_checkpoint
    }
    if "inference_allowed_mirroring_axes" in source_checkpoint:
        metadata["inference_allowed_mirroring_axes"] = source_checkpoint[
            "inference_allowed_mirroring_axes"
        ]
    return metadata


def _epoch_summary(results: list[Any], epoch: int, elapsed_seconds: float) -> dict[str, Any]:
    samples = sum(result.sample_count for result in results)
    target_counts = np.sum([result.target_counts for result in results], axis=0)
    prediction_counts = np.sum([result.prediction_counts for result in results], axis=0)
    gradient_norms = [result.gradient_norm_before_clip for result in results]
    return {
        "epoch": epoch,
        "training_loss_mean": float(np.mean([result.loss for result in results])),
        "training_patch_accuracy": float(
            sum(result.correct for result in results) / max(samples, 1)
        ),
        "training_lesion_patch_fraction": float(
            sum(result.lesion_patch_count for result in results) / max(samples, 1)
        ),
        "training_target_counts": [int(value) for value in target_counts],
        "training_prediction_counts": [int(value) for value in prediction_counts],
        "gradient_norm_before_clip_mean": float(np.mean(gradient_norms)),
        "gradient_norm_before_clip_max": float(np.max(gradient_norms)),
        "successful_optimizer_updates": int(
            sum(result.optimizer_update_count for result in results)
        ),
        "elapsed_seconds": elapsed_seconds,
        "generalization_metric": False,
    }


def _public_audit(rescue_state: dict[str, Any]) -> dict[str, Any]:
    excluded = {"optimizer_state", "grad_scaler_state", "rng_state"}
    return {key: value for key, value in rescue_state.items() if key not in excluded}


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    schedule = _schedule_from_args(args)
    (
        source_path,
        output_path,
        audit_path,
        activation_audit_path,
        recovery_audit_path,
    ) = _resolved_paths(args)

    # This fallback must never append to or create a W&B run, and compilation
    # would complicate strict parameter names/checkpoint portability.
    os.environ["nnUNet_wandb_enabled"] = "0"
    os.environ.pop("nnUNet_wandb_mode", None)
    os.environ["nnUNet_compile"] = "false"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    from nnunetv2.run.run_training import get_trainer_from_args

    from pancreas_multitask.trainer import (
        NUM_CLASSIFICATION_CLASSES,
        inverse_frequency_class_weights,
        resolve_case_labels,
    )

    source_file_hash = file_sha256(source_path)
    activation_audit_file_hash = file_sha256(activation_audit_path)
    execution_recovery: dict[str, Any] | None = None
    recovery_audit_file_hash: str | None = None
    if recovery_audit_path is not None:
        execution_recovery, recovery_audit_file_hash = _load_execution_recovery(
            recovery_audit_path,
            source_checkpoint=source_path,
            activation_audit=activation_audit_path,
        )
    execution_provenance: dict[str, Any] = {
        "process_launch_count": 2 if execution_recovery is not None else 1,
        "zero_update_recovery_count": 1 if execution_recovery is not None else 0,
        "update_bearing_trajectory_count": 1,
    }
    if execution_recovery is not None:
        execution_provenance.update(
            {
                "execution_recovery": execution_recovery,
                "execution_recovery_audit": str(recovery_audit_path),
                "execution_recovery_audit_sha256": recovery_audit_file_hash,
            }
        )
    with activation_audit_path.open("r", encoding="utf-8") as handle:
        activation_audit = json.load(handle)
    if file_sha256(activation_audit_path) != activation_audit_file_hash:
        raise RuntimeError("Activation audit changed while it was being read")
    if not isinstance(activation_audit, dict):
        raise TypeError("Activation audit must contain a JSON object")
    validate_activation_audit(
        activation_audit,
        source_checkpoint_sha256=source_file_hash,
    )
    started_at = datetime.now(UTC).isoformat()
    wall_start = time.perf_counter()

    with _exclusive_output_lock(output_path):
        trainer = get_trainer_from_args(
            str(args.dataset),
            args.configuration,
            args.fold,
            args.trainer,
            args.plans,
            False,
            device,
        )
        trainer.initialize()
        model = trainer.network

        source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        if file_sha256(source_path) != source_file_hash:
            raise RuntimeError("Source checkpoint changed while it was being loaded")
        if source_checkpoint.get("trainer_name") != args.trainer:
            raise ValueError(
                "Source checkpoint trainer mismatch: "
                f"{source_checkpoint.get('trainer_name')!r} != {args.trainer!r}"
            )
        source_init_args = source_checkpoint.get("init_args")
        if not isinstance(source_init_args, dict):
            raise TypeError("Source checkpoint init_args must be a mapping")
        if source_init_args.get("configuration") != args.configuration:
            raise ValueError("Source checkpoint configuration does not match rescue request")
        if int(source_init_args.get("fold", -1)) != args.fold:
            raise ValueError("Source checkpoint fold does not match rescue request")
        source_logging = source_checkpoint.get("logging")
        if not isinstance(source_logging, dict):
            raise TypeError("Source checkpoint logging must be a mapping")
        if int(source_checkpoint.get("current_epoch", -1)) != 200:
            raise ValueError("Rescue requires a completed checkpoint_final at epoch 200")
        if (
            len(source_logging.get("train_cls_losses", ())) != 200
            or len(source_logging.get("train_cls_accuracy", ())) != 200
        ):
            raise ValueError("Rescue source must contain exactly 200 training epochs")
        recomputed_activation = classification_activation_audit(
            source_logging.get("train_cls_losses", ()),
            source_logging.get("train_cls_accuracy", ()),
        )
        for key, expected_value in recomputed_activation.items():
            if activation_audit.get(key) != expected_value:
                raise ValueError(
                    "Activation audit does not reproduce from source checkpoint "
                    f"training logging at key {key!r}"
                )
        strict_load_network_weights(model, source_checkpoint)
        source_metadata = _checkpoint_metadata(source_checkpoint)
        source_component_hashes = component_hashes(model)
        del source_checkpoint

        completed_epochs = 0
        successful_optimizer_updates = 0
        history: list[dict[str, Any]] = []
        resume_state: dict[str, Any] | None = None
        repair_completed_audit = False
        original_started_at = started_at
        prior_elapsed_seconds = 0.0
        resume_count = 0
        last_resume_started_at: str | None = None
        if args.resume:
            resume_checkpoint = torch.load(
                output_path,
                map_location="cpu",
                weights_only=False,
            )
            raw_resume_state = resume_checkpoint.get("classification_rescue")
            if not isinstance(raw_resume_state, dict):
                raise ValueError("Output checkpoint does not contain rescue state")
            completed_epochs = validate_resume_state(
                raw_resume_state,
                source_checkpoint_sha256=source_file_hash,
                schedule=schedule,
            )
            repair_completed_audit = completed_epochs == schedule.epochs
            strict_load_network_weights(model, resume_checkpoint)
            resumed_hashes = component_hashes(model)
            if resumed_hashes != raw_resume_state["current_component_sha256"]:
                raise ValueError("Resume network state does not match embedded hashes")
            if resumed_hashes["encoder"] != source_component_hashes["encoder"]:
                raise ValueError("Resume encoder does not match the frozen source encoder")
            if resumed_hashes["decoder"] != source_component_hashes["decoder"]:
                raise ValueError("Resume decoder does not match the frozen source decoder")
            resume_state = raw_resume_state
            history = list(raw_resume_state.get("training_only_history", []))
            successful_optimizer_updates = int(raw_resume_state["successful_optimizer_updates"])
            original_started_at = str(raw_resume_state["started_at_utc"])
            prior_elapsed_seconds = float(raw_resume_state["elapsed_seconds"])
            resume_count = int(raw_resume_state["resume_count"])
            if not repair_completed_audit:
                resume_count += 1
                last_resume_started_at = started_at
            del resume_checkpoint
        else:
            immutable_before_reset = component_hashes(model)
            reset_classification_parameters(model, schedule.reset_seed)
            reset_hashes = component_hashes(model)
            if reset_hashes["encoder"] != immutable_before_reset["encoder"]:
                raise RuntimeError("Classification reset changed encoder state")
            if reset_hashes["decoder"] != immutable_before_reset["decoder"]:
                raise RuntimeError("Classification reset changed decoder state")
            if reset_hashes["classification"] == immutable_before_reset["classification"]:
                raise RuntimeError("Classification reset did not change classification state")

        latest_rescue_state = resume_state

        trainable_items = freeze_for_classification_rescue(model)
        trainable_parameters = [parameter for _, parameter in trainable_items]
        if resume_state is not None and resume_state.get("device_type") != device.type:
            raise ValueError("Resume device type differs from the original rescue")
        precision_policy = _precision_policy(device)
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=schedule.learning_rate,
            weight_decay=schedule.weight_decay,
        )
        scaler = torch.GradScaler(device.type, enabled=False)
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer_state"])
            _optimizer_state_to_device(optimizer, device)
            grad_scaler_state = resume_state.get("grad_scaler_state")
            if grad_scaler_state is not None:
                raise ValueError(
                    "Resume checkpoint unexpectedly contains GradScaler state for the "
                    "trainable-float32 precision policy"
                )

        training_loader, training_keys, validation_keys = _build_training_only_dataloader(trainer)
        split_audit = validate_disjoint_split(
            training_keys,
            validation_keys,
            expected_training_cases=args.expected_training_cases,
            expected_validation_cases=args.expected_validation_cases,
        )
        split_audit.update(
            _frozen_split_manifest_binding(
                trainer,
                training_keys,
                validation_keys,
            )
        )
        training_labels = resolve_case_labels(
            training_keys,
            trainer.classification_label_mapping,
        )
        class_counts = np.bincount(
            np.asarray(training_labels, dtype=np.int64),
            minlength=NUM_CLASSIFICATION_CLASSES,
        )
        class_weights_array = inverse_frequency_class_weights(training_labels)
        class_weights = torch.as_tensor(
            class_weights_array,
            dtype=torch.float32,
            device=device,
        )

        audit_base: dict[str, Any] = {
            "schema_version": RESCUE_SCHEMA_VERSION,
            "method": "post_training_frozen_backbone_classification_head_rescue",
            "status": "in_progress",
            "started_at_utc": original_started_at,
            "resume_count": resume_count,
            "last_resume_started_at_utc": last_resume_started_at,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": source_file_hash,
            "activation_audit": str(activation_audit_path),
            "activation_audit_sha256": activation_audit_file_hash,
            "activation_decision_epoch": activation_audit["decision_epoch"],
            **execution_provenance,
            "source_component_sha256": source_component_hashes,
            "output_checkpoint": str(output_path),
            "schedule": schedule.to_dict(),
            "optimizer": "AdamW",
            "device_type": device.type,
            "precision_policy": precision_policy,
            "wandb_enabled": False,
            "early_stopping": False,
            "maximum_attempts": 1,
            "training_updates_expected": (schedule.epochs * schedule.iterations_per_epoch),
            "training_loader": "single_threaded_training_split_only",
            "training_batch_size": int(trainer.batch_size),
            "decoder_executed_during_rescue": False,
            "encoder_gradient_enabled": False,
            "decoder_gradient_enabled": False,
            "validation_labels_indexed_for_targets": False,
            "classification_parameter_names": [name for name, _ in trainable_items],
            "classification_trainable_parameter_count": int(
                sum(parameter.numel() for parameter in trainable_parameters)
            ),
            "training_class_counts": [int(value) for value in class_counts],
            "training_class_weights": [float(value) for value in class_weights_array],
            "split_audit": split_audit,
            "selection_or_stopping_metric": None,
            "training_only_history": history,
        }

        if resume_state is not None:
            context_fields = {
                "split_audit": split_audit,
                "training_class_counts": [int(value) for value in class_counts],
                "training_class_weights": [float(value) for value in class_weights_array],
                "device_type": device.type,
                "precision_policy": precision_policy,
                **execution_provenance,
                "training_loader": "single_threaded_training_split_only",
                "training_batch_size": int(trainer.batch_size),
                "classification_parameter_names": [name for name, _ in trainable_items],
            }
            for key, expected_value in context_fields.items():
                if resume_state.get(key) != expected_value:
                    raise ValueError(f"Resume data/runtime context changed at {key!r}")

        if resume_state is None:
            seed_all(schedule.reset_seed, include_cuda=device.type == "cuda")
        else:
            restore_rng_state(
                resume_state["rng_state"],
                include_cuda=device.type == "cuda",
            )

        for epoch in range(completed_epochs, schedule.epochs):
            epoch_start = time.perf_counter()
            step_results = []
            for _ in range(schedule.iterations_per_epoch):
                batch = next(training_loader)
                batch_keys = [str(case_id) for case_id in batch["keys"]]
                unexpected_keys = sorted(set(batch_keys) - set(training_keys))
                if unexpected_keys:
                    raise RuntimeError(
                        f"Training loader emitted non-training cases: {unexpected_keys}"
                    )
                data = batch["data"].to(device, non_blocking=True)
                raw_target = batch["target"]
                highest_resolution_target = (
                    raw_target[0] if isinstance(raw_target, list) else raw_target
                ).to(device, non_blocking=True)
                targets = torch.as_tensor(
                    resolve_case_labels(
                        batch_keys,
                        trainer.classification_label_mapping,
                    ),
                    dtype=torch.long,
                    device=device,
                )
                step_result = classification_rescue_train_step(
                    model,
                    data,
                    targets,
                    highest_resolution_target,
                    optimizer=optimizer,
                    class_weights=class_weights,
                    schedule=schedule,
                    scaler=scaler,
                    use_encoder_amp=device.type == "cuda",
                    num_classes=NUM_CLASSIFICATION_CLASSES,
                )
                if step_result.optimizer_update_count != 1:
                    raise RuntimeError(
                        "A clean rescue step must execute exactly one optimizer update"
                    )
                successful_optimizer_updates += step_result.optimizer_update_count
                step_results.append(step_result)

            epoch_record = _epoch_summary(
                step_results,
                epoch,
                time.perf_counter() - epoch_start,
            )
            if epoch_record["successful_optimizer_updates"] != schedule.iterations_per_epoch:
                raise RuntimeError(
                    "A completed rescue epoch must contain exactly "
                    f"{schedule.iterations_per_epoch} successful optimizer updates"
                )
            history.append(epoch_record)
            completed_epochs = epoch + 1
            expected_completed_updates = completed_epochs * schedule.iterations_per_epoch
            if successful_optimizer_updates != expected_completed_updates:
                raise RuntimeError(
                    "Cumulative successful optimizer updates do not match completed epochs: "
                    f"{successful_optimizer_updates} != {expected_completed_updates}"
                )
            should_save = (
                completed_epochs % args.save_every == 0 or completed_epochs == schedule.epochs
            )
            if not should_save:
                continue

            current_hashes = component_hashes(model)
            if current_hashes["encoder"] != source_component_hashes["encoder"]:
                raise RuntimeError("Frozen encoder state changed during rescue")
            if current_hashes["decoder"] != source_component_hashes["decoder"]:
                raise RuntimeError("Frozen decoder state changed during rescue")
            status = "complete" if completed_epochs == schedule.epochs else "in_progress"
            canonical_recovery_audit = source_path.with_name(
                "classification_rescue_zero_update_recovery.json"
            )
            if execution_recovery is None:
                if canonical_recovery_audit.is_file():
                    raise RuntimeError(
                        "Canonical execution-recovery evidence appeared during a clean "
                        "rescue trajectory"
                    )
            else:
                if recovery_audit_path is None:
                    raise RuntimeError(
                        "Recovered rescue trajectory lost its recovery-audit binding"
                    )
                final_recovery, final_recovery_hash = _load_execution_recovery(
                    recovery_audit_path,
                    source_checkpoint=source_path,
                    activation_audit=activation_audit_path,
                )
                if (
                    final_recovery != execution_recovery
                    or final_recovery_hash != recovery_audit_file_hash
                ):
                    raise RuntimeError(
                        "Execution recovery evidence changed during rescue execution"
                    )
            rescue_state = {
                **audit_base,
                "status": status,
                "completed_epochs": completed_epochs,
                "successful_optimizer_updates": successful_optimizer_updates,
                "training_only_history": history,
                "current_component_sha256": current_hashes,
                "optimizer_state": optimizer.state_dict(),
                "grad_scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "rng_state": capture_rng_state(include_cuda=device.type == "cuda"),
                "elapsed_seconds": prior_elapsed_seconds + (time.perf_counter() - wall_start),
            }
            latest_rescue_state = rescue_state
            output_checkpoint = build_inference_checkpoint(
                source_metadata,
                model,
                rescue_state,
            )
            atomic_torch_save(output_checkpoint, output_path)
            public_audit = _public_audit(rescue_state)
            public_audit["output_checkpoint_sha256"] = file_sha256(output_path)
            public_audit["updated_at_utc"] = datetime.now(UTC).isoformat()
            atomic_write_json(public_audit, audit_path)
            print(
                f"RESCUE_EPOCH={completed_epochs}/{schedule.epochs} "
                f"TRAIN_LOSS={history[-1]['training_loss_mean']:.6f} "
                f"TRAIN_PATCH_ACCURACY={history[-1]['training_patch_accuracy']:.6f}",
                flush=True,
            )

        final_hashes = component_hashes(model)
        if final_hashes["encoder"] != source_component_hashes["encoder"]:
            raise RuntimeError("Final encoder hash differs from the source checkpoint")
        if final_hashes["decoder"] != source_component_hashes["decoder"]:
            raise RuntimeError("Final decoder hash differs from the source checkpoint")
        expected_total_updates = schedule.epochs * schedule.iterations_per_epoch
        if successful_optimizer_updates != expected_total_updates:
            raise RuntimeError(
                "Completed rescue did not execute the declared number of optimizer updates: "
                f"{successful_optimizer_updates} != {expected_total_updates}"
            )

        if repair_completed_audit:
            if latest_rescue_state is None or latest_rescue_state.get("status") != "complete":
                raise RuntimeError("Completed rescue checkpoint lacks a complete embedded audit")
            public_audit = _public_audit(latest_rescue_state)
            public_audit["output_checkpoint_sha256"] = file_sha256(output_path)
            public_audit["audit_repaired_from_complete_checkpoint"] = True
            public_audit["updated_at_utc"] = datetime.now(UTC).isoformat()
            atomic_write_json(public_audit, audit_path)
            print("RESCUE_AUDIT_REPAIRED=true", flush=True)

    print(f"RESCUE_CHECKPOINT={output_path}")
    print(f"RESCUE_AUDIT={audit_path}")
    return output_path, audit_path


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
