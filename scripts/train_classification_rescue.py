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
        help="Write a resumable inference checkpoint every N completed epochs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the same frozen schedule from output-checkpoint.",
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


def _resolved_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    source = args.source_checkpoint.expanduser().resolve()
    output = args.output_checkpoint.expanduser().resolve()
    activation_audit = args.activation_audit.expanduser().resolve()
    audit = (
        args.audit_json.expanduser().resolve()
        if args.audit_json is not None
        else output.with_name(f"{output.name}.audit.json")
    )
    if source == output:
        raise ValueError("source-checkpoint and output-checkpoint must be different files")
    if audit in (source, output):
        raise ValueError("audit-json must be different from both checkpoint paths")
    if activation_audit in (source, output, audit):
        raise ValueError("activation-audit must be a separate JSON file")
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")
    if source.name != "checkpoint_final.pth":
        raise ValueError(
            f"The predeclared rescue source must be named checkpoint_final.pth; got {source.name!r}"
        )
    if not activation_audit.is_file():
        raise FileNotFoundError(f"Activation audit does not exist: {activation_audit}")
    if args.save_every < 1:
        raise ValueError("save-every must be positive")
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists; use --resume only for a matching rescue: {output}"
        )
    if args.resume and not output.is_file():
        raise FileNotFoundError(f"Cannot resume because output does not exist: {output}")
    return source, output, audit, activation_audit


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
        "elapsed_seconds": elapsed_seconds,
        "generalization_metric": False,
    }


def _public_audit(rescue_state: dict[str, Any]) -> dict[str, Any]:
    excluded = {"optimizer_state", "grad_scaler_state", "rng_state"}
    return {key: value for key, value in rescue_state.items() if key not in excluded}


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    schedule = _schedule_from_args(args)
    source_path, output_path, audit_path, activation_audit_path = _resolved_paths(args)

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
    with activation_audit_path.open("r", encoding="utf-8") as handle:
        activation_audit = json.load(handle)
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
        history: list[dict[str, Any]] = []
        resume_state: dict[str, Any] | None = None
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
            if completed_epochs == schedule.epochs:
                raise RuntimeError("The classification rescue schedule is already complete")
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

        trainable_items = freeze_for_classification_rescue(model)
        trainable_parameters = [parameter for _, parameter in trainable_items]
        if resume_state is not None and resume_state.get("device_type") != device.type:
            raise ValueError("Resume device type differs from the original rescue")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=schedule.learning_rate,
            weight_decay=schedule.weight_decay,
        )
        scaler = torch.GradScaler(
            device.type,
            enabled=device.type == "cuda",
        )
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer_state"])
            _optimizer_state_to_device(optimizer, device)
            grad_scaler_state = resume_state.get("grad_scaler_state")
            if grad_scaler_state is not None:
                scaler.load_state_dict(grad_scaler_state)

        training_loader, training_keys, validation_keys = _build_training_only_dataloader(trainer)
        split_audit = validate_disjoint_split(
            training_keys,
            validation_keys,
            expected_training_cases=args.expected_training_cases,
            expected_validation_cases=args.expected_validation_cases,
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
            "started_at_utc": started_at,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": source_file_hash,
            "activation_audit": str(activation_audit_path),
            "activation_audit_sha256": activation_audit_file_hash,
            "activation_decision_epoch": activation_audit["decision_epoch"],
            "source_component_sha256": source_component_hashes,
            "output_checkpoint": str(output_path),
            "schedule": schedule.to_dict(),
            "optimizer": "AdamW",
            "device_type": device.type,
            "training_loader": "single_threaded_training_split_only",
            "training_batch_size": int(trainer.batch_size),
            "decoder_executed_during_rescue": False,
            "encoder_gradient_enabled": False,
            "decoder_gradient_enabled": False,
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
                step_results.append(
                    classification_rescue_train_step(
                        model,
                        data,
                        targets,
                        highest_resolution_target,
                        optimizer=optimizer,
                        class_weights=class_weights,
                        schedule=schedule,
                        scaler=scaler,
                        use_amp=device.type == "cuda",
                        num_classes=NUM_CLASSIFICATION_CLASSES,
                    )
                )

            history.append(
                _epoch_summary(
                    step_results,
                    epoch,
                    time.perf_counter() - epoch_start,
                )
            )
            completed_epochs = epoch + 1
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
            rescue_state = {
                **audit_base,
                "status": status,
                "completed_epochs": completed_epochs,
                "training_only_history": history,
                "current_component_sha256": current_hashes,
                "optimizer_state": optimizer.state_dict(),
                "grad_scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "rng_state": capture_rng_state(include_cuda=device.type == "cuda"),
                "elapsed_seconds": time.perf_counter() - wall_start,
            }
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

    print(f"RESCUE_CHECKPOINT={output_path}")
    print(f"RESCUE_AUDIT={audit_path}")
    return output_path, audit_path


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
