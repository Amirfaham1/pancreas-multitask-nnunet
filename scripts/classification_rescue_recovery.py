"""Create and validate evidence for one zero-update rescue relaunch.

The original classification-rescue process failed on its first training step
before ``AdamW.step``.  This utility preserves the two process logs, binds the
failure to the source checkpoint and activation audit, and records the narrow
recovery policy before the custom joint fixed-validation pass starts.

The resulting artifact is deliberately separate from the training command so
that the failed execution remains immutable even if the numerical code is
repaired.  A completed rescue audit can later embed the artifact verbatim and
bind it by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EVENT = "classification_rescue_zero_update_execution_recovery"
STATUS = "authorized_before_custom_joint_fixed_validation"
RECOVERY_FILENAME = "classification_rescue_zero_update_recovery.json"
EVIDENCE_DIRECTORY_NAME = "classification_rescue_recovery_evidence"
FAILED_STDOUT_NAME = "failed_launch.stdout.log"
FAILED_STDERR_NAME = "failed_launch.stderr.log"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class RecoveryEvidenceError(RuntimeError):
    """Raised when recovery evidence is absent, ambiguous, or inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryEvidenceError(f"{field} must be a JSON object")
    return value


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise RecoveryEvidenceError(f"{field} must equal {expected!r}; observed {actual!r}")


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value).lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise RecoveryEvidenceError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _read_json_object(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise RecoveryEvidenceError(f"{role} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RecoveryEvidenceError(f"{role} must contain a JSON object")
    return payload


def _artifact_descriptor(path: Path, *, artifact_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative_name = resolved.relative_to(artifact_root.resolve()).as_posix()
    except ValueError as exc:
        raise RecoveryEvidenceError(
            f"Preserved log is outside the recovery-artifact directory: {resolved}"
        ) from exc
    return {
        "name": relative_name,
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _resolve_preserved_artifact(
    descriptor: Mapping[str, Any],
    *,
    artifact_root: Path,
    field: str,
) -> Path:
    name = descriptor.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RecoveryEvidenceError(f"{field}.name must be a non-empty relative path")
    candidate = (artifact_root / name).resolve()
    try:
        candidate.relative_to(artifact_root.resolve())
    except ValueError as exc:
        raise RecoveryEvidenceError(f"{field}.name escapes the artifact directory") from exc
    if not candidate.is_file():
        raise RecoveryEvidenceError(f"{field} is missing: {candidate}")
    expected_bytes = descriptor.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise RecoveryEvidenceError(f"{field}.bytes must be an integer")
    if candidate.stat().st_size != expected_bytes:
        raise RecoveryEvidenceError(f"{field} byte count differs from the preserved file")
    expected_hash = _require_sha256(descriptor.get("sha256"), field=f"{field}.sha256")
    if file_sha256(candidate) != expected_hash:
        raise RecoveryEvidenceError(f"{field} SHA-256 differs from the preserved file")
    return candidate


def _validate_activation(
    activation: Mapping[str, Any],
    *,
    source_checkpoint: Path,
    activation_path: Path,
) -> tuple[str, str]:
    _require_equal(activation.get("schema_version"), 1, field="activation.schema_version")
    _require_equal(
        activation.get("activation_approved"), True, field="activation.activation_approved"
    )
    _require_equal(
        activation.get("source_checkpoint_name"),
        "checkpoint_final.pth",
        field="activation.source_checkpoint_name",
    )
    _require_equal(
        activation.get("validation_metrics_read"),
        False,
        field="activation.validation_metrics_read",
    )
    _require_equal(
        activation.get("validation_used_for_activation"),
        False,
        field="activation.validation_used_for_activation",
    )
    source_hash = file_sha256(source_checkpoint)
    recorded_source_hash = _require_sha256(
        activation.get("source_checkpoint_sha256"),
        field="activation.source_checkpoint_sha256",
    )
    if source_hash != recorded_source_hash:
        raise RecoveryEvidenceError("Activation audit is bound to a different checkpoint_final.pth")
    return source_hash, file_sha256(activation_path)


def _git_commit(repo_root: Path, commit: Any, *, field: str) -> str:
    if not isinstance(commit, str):
        raise RecoveryEvidenceError(f"{field} must be a full lowercase Git commit")
    normalized = commit.lower()
    if GIT_COMMIT_PATTERN.fullmatch(normalized) is None:
        raise RecoveryEvidenceError(f"{field} must be a full lowercase Git commit")
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{normalized}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RecoveryEvidenceError(f"{field} is not available in the repository: {normalized}")
    return normalized


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", f"{commit}:{relative_path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip().lower()
    if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RecoveryEvidenceError(
            f"Could not bind {relative_path!r} at pre-failure commit {commit}"
        )
    return value


def _atomic_write_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _preserve_logs(
    stdout_source: Path,
    stderr_source: Path,
    *,
    evidence_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if evidence_directory.exists():
        raise RecoveryEvidenceError(
            f"Recovery evidence directory already exists: {evidence_directory}"
        )
    for role, path in (("failed stdout", stdout_source), ("failed stderr", stderr_source)):
        if not path.is_file():
            raise RecoveryEvidenceError(f"{role} log does not exist: {path}")
        if path.stat().st_size <= 0:
            raise RecoveryEvidenceError(f"{role} log is empty: {path}")

    stdout_hash_before = file_sha256(stdout_source)
    stderr_hash_before = file_sha256(stderr_source)
    temporary_directory = Path(
        tempfile.mkdtemp(
            dir=evidence_directory.parent,
            prefix=f".{evidence_directory.name}.",
        )
    )
    try:
        preserved_stdout = temporary_directory / FAILED_STDOUT_NAME
        preserved_stderr = temporary_directory / FAILED_STDERR_NAME
        shutil.copy2(stdout_source, preserved_stdout)
        shutil.copy2(stderr_source, preserved_stderr)
        if file_sha256(stdout_source) != stdout_hash_before:
            raise RecoveryEvidenceError("Failed stdout log changed while it was preserved")
        if file_sha256(stderr_source) != stderr_hash_before:
            raise RecoveryEvidenceError("Failed stderr log changed while it was preserved")
        if file_sha256(preserved_stdout) != stdout_hash_before:
            raise RecoveryEvidenceError("Preserved stdout log differs from its source")
        if file_sha256(preserved_stderr) != stderr_hash_before:
            raise RecoveryEvidenceError("Preserved stderr log differs from its source")
        os.replace(temporary_directory, evidence_directory)
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)

    stdout_artifact = _artifact_descriptor(
        evidence_directory / FAILED_STDOUT_NAME,
        artifact_root=evidence_directory.parent,
    )
    stderr_artifact = _artifact_descriptor(
        evidence_directory / FAILED_STDERR_NAME,
        artifact_root=evidence_directory.parent,
    )
    return stdout_artifact, stderr_artifact


def _validate_failed_logs(stdout_text: str, stderr_text: str) -> None:
    required_stdout = (
        "CLASSIFICATION_RESCUE_START",
        "ACTIVATION_APPROVED=true",
        "Using splits from existing split file",
        "This split has 252 training and 36 validation cases.",
    )
    for marker in required_stdout:
        if marker not in stdout_text:
            raise RecoveryEvidenceError(f"Failed stdout log lacks marker: {marker!r}")
    if "RESCUE_EPOCH=" in stdout_text:
        raise RecoveryEvidenceError("Failed stdout unexpectedly records a completed rescue epoch")
    required_stderr = (
        "classification_rescue_train_step",
        "clip_grad_norm_",
        "gradients from `parameters` is non-finite",
        "Classification rescue exited with code 1",
        "OVERNIGHT_PIPELINE_FAILED",
    )
    for marker in required_stderr:
        if marker not in stderr_text:
            raise RecoveryEvidenceError(f"Failed stderr log lacks marker: {marker!r}")


def validate_recovery_payload(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path,
    source_checkpoint: Path,
    activation_audit: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Fail closed on the recovery artifact and return a compact summary."""

    _require_equal(payload.get("schema_version"), SCHEMA_VERSION, field="schema_version")
    _require_equal(payload.get("event"), EVENT, field="event")
    _require_equal(payload.get("status"), STATUS, field="status")
    _require_equal(
        payload.get("source_checkpoint_name"),
        "checkpoint_final.pth",
        field="source_checkpoint_name",
    )
    _require_equal(payload.get("activation_approved"), True, field="activation_approved")

    activation_payload = _read_json_object(activation_audit, role="activation audit")
    source_hash, activation_hash = _validate_activation(
        activation_payload,
        source_checkpoint=source_checkpoint,
        activation_path=activation_audit,
    )
    recorded_source_hash = _require_sha256(
        payload.get("source_checkpoint_sha256"), field="source_checkpoint_sha256"
    )
    recorded_activation_hash = _require_sha256(
        payload.get("activation_audit_sha256"), field="activation_audit_sha256"
    )
    if recorded_source_hash != source_hash:
        raise RecoveryEvidenceError("Recovery artifact source-checkpoint SHA-256 mismatch")
    if recorded_activation_hash != activation_hash:
        raise RecoveryEvidenceError("Recovery artifact activation-audit SHA-256 mismatch")

    resolved_repo_root = (
        repo_root.expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    failed_launch_commit = _git_commit(
        resolved_repo_root,
        payload.get("git_commit_at_failed_launch"),
        field="git_commit_at_failed_launch",
    )
    _git_commit(
        resolved_repo_root,
        payload.get("rescue_protocol_commit"),
        field="rescue_protocol_commit",
    )
    implementation = _mapping(
        payload.get("pre_failure_implementation"), field="pre_failure_implementation"
    )
    expected_blobs = {
        "train_script_git_blob": _git_blob(
            resolved_repo_root,
            failed_launch_commit,
            "scripts/train_classification_rescue.py",
        ),
        "rescue_module_git_blob": _git_blob(
            resolved_repo_root,
            failed_launch_commit,
            "src/pancreas_multitask/classification_rescue.py",
        ),
    }
    for name, expected_blob in expected_blobs.items():
        _require_equal(
            implementation.get(name),
            expected_blob,
            field=f"pre_failure_implementation.{name}",
        )

    failed = _mapping(payload.get("failed_launch"), field="failed_launch")
    exact_failed_values = {
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
    }
    for name, expected in exact_failed_values.items():
        _require_equal(failed.get(name), expected, field=f"failed_launch.{name}")
    stdout_artifact = _mapping(failed.get("stdout_artifact"), field="failed_launch.stdout_artifact")
    stderr_artifact = _mapping(failed.get("stderr_artifact"), field="failed_launch.stderr_artifact")
    preserved_stdout = _resolve_preserved_artifact(
        stdout_artifact,
        artifact_root=artifact_path.parent,
        field="failed_launch.stdout_artifact",
    )
    preserved_stderr = _resolve_preserved_artifact(
        stderr_artifact,
        artifact_root=artifact_path.parent,
        field="failed_launch.stderr_artifact",
    )
    _validate_failed_logs(
        preserved_stdout.read_text(encoding="utf-8"),
        preserved_stderr.read_text(encoding="utf-8"),
    )

    validation = _mapping(payload.get("validation"), field="validation")
    exact_validation_values = {
        "stock_nnunet_segmentation_only_validation_completed": True,
        "stock_nnunet_validation_metrics_observed_before_recovery": True,
        "stock_nnunet_mean_foreground_dice_observed_before_recovery": 0.753518646,
        "stock_nnunet_validation_used_for_recovery": False,
        "custom_joint_fixed_validation_started": False,
        "custom_joint_fixed_validation_output_existed_at_authorization": False,
        "rescue_process_validation_images_opened": False,
        "rescue_process_validation_batches_consumed": 0,
        "rescue_process_validation_used_for_recovery": False,
    }
    for name, expected in exact_validation_values.items():
        _require_equal(validation.get(name), expected, field=f"validation.{name}")

    policy = _mapping(payload.get("recovery_policy"), field="recovery_policy")
    exact_policy_values = {
        "schedule_changed": False,
        "source_checkpoint_changed": False,
        "reset_seed_changed": False,
        "maximum_update_bearing_trajectories": 1,
        "maximum_zero_update_runtime_recoveries": 1,
        "process_launch_count_after_relaunch": 2,
        "no_further_recovery_allowed": True,
    }
    for name, expected in exact_policy_values.items():
        _require_equal(policy.get(name), expected, field=f"recovery_policy.{name}")

    return {
        "artifact_sha256": file_sha256(artifact_path),
        "source_checkpoint_sha256": source_hash,
        "activation_audit_sha256": activation_hash,
        "process_launch_count_after_relaunch": 2,
        "zero_update_recovery_count": 1,
        "update_bearing_trajectory_count": 1,
        "failed_launch_optimizer_updates": 0,
        "rescue_process_validation_batches_consumed": 0,
    }


def _validate_rescue_binding(
    rescue: Mapping[str, Any],
    *,
    recovery_payload: Mapping[str, Any],
    recovery_path: Path,
) -> None:
    _require_equal(rescue.get("process_launch_count"), 2, field="rescue.process_launch_count")
    _require_equal(
        rescue.get("zero_update_recovery_count"),
        1,
        field="rescue.zero_update_recovery_count",
    )
    _require_equal(
        rescue.get("update_bearing_trajectory_count"),
        1,
        field="rescue.update_bearing_trajectory_count",
    )
    embedded = _mapping(rescue.get("execution_recovery"), field="rescue.execution_recovery")
    if dict(embedded) != dict(recovery_payload):
        raise RecoveryEvidenceError(
            "Rescue audit execution_recovery does not exactly match the recovery artifact"
        )
    recorded_hash = _require_sha256(
        rescue.get("execution_recovery_audit_sha256"),
        field="rescue.execution_recovery_audit_sha256",
    )
    if recorded_hash != file_sha256(recovery_path):
        raise RecoveryEvidenceError("Rescue audit recovery-artifact SHA-256 mismatch")
    recorded_path = rescue.get("execution_recovery_audit")
    if (
        not isinstance(recorded_path, str)
        or Path(recorded_path).resolve() != recovery_path.resolve()
    ):
        raise RecoveryEvidenceError("Rescue audit recovery-artifact path mismatch")


def create(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.name != RECOVERY_FILENAME:
        raise RecoveryEvidenceError(f"Recovery output must be named {RECOVERY_FILENAME}")
    if output.exists():
        raise RecoveryEvidenceError(f"Recovery artifact already exists: {output}")

    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    activation_audit = args.activation_audit.expanduser().resolve()
    rescue_checkpoint = args.rescue_checkpoint.expanduser().resolve()
    rescue_audit = args.rescue_audit.expanduser().resolve()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    for role, path in (
        ("rescue checkpoint", rescue_checkpoint),
        ("rescue audit", rescue_audit),
    ):
        if path.exists():
            raise RecoveryEvidenceError(
                f"Cannot authorize zero-update recovery because {role} exists: {path}"
            )
    if evaluation_root.exists() and any(evaluation_root.iterdir()):
        raise RecoveryEvidenceError(
            "Cannot authorize recovery after fixed-validation output has been created"
        )
    if not args.confirm_first_step_zero_update:
        raise RecoveryEvidenceError(
            "Creation requires --confirm-first-step-zero-update operator attestation"
        )

    activation_payload = _read_json_object(activation_audit, role="activation audit")
    source_hash, activation_hash = _validate_activation(
        activation_payload,
        source_checkpoint=source_checkpoint,
        activation_path=activation_audit,
    )
    repo_root = args.repo_root.expanduser().resolve()
    failed_launch_commit = _git_commit(
        repo_root,
        args.git_commit_at_failed_launch,
        field="git_commit_at_failed_launch",
    )
    rescue_protocol_commit = _git_commit(
        repo_root,
        args.rescue_protocol_commit,
        field="rescue_protocol_commit",
    )

    failed_stdout = args.failed_stdout_log.expanduser().resolve()
    failed_stderr = args.failed_stderr_log.expanduser().resolve()
    stdout_text = failed_stdout.read_text(encoding="utf-8")
    stderr_text = failed_stderr.read_text(encoding="utf-8")
    _validate_failed_logs(stdout_text, stderr_text)
    evidence_directory = output.parent / EVIDENCE_DIRECTORY_NAME
    stdout_artifact, stderr_artifact = _preserve_logs(
        failed_stdout,
        failed_stderr,
        evidence_directory=evidence_directory,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event": EVENT,
        "status": STATUS,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_checkpoint_name": "checkpoint_final.pth",
        "source_checkpoint_sha256": source_hash,
        "activation_audit_sha256": activation_hash,
        "activation_approved": True,
        "git_commit_at_failed_launch": failed_launch_commit,
        "rescue_protocol_commit": rescue_protocol_commit,
        "pre_failure_implementation": {
            "train_script_git_blob": _git_blob(
                repo_root,
                failed_launch_commit,
                "scripts/train_classification_rescue.py",
            ),
            "rescue_module_git_blob": _git_blob(
                repo_root,
                failed_launch_commit,
                "src/pancreas_multitask/classification_rescue.py",
            ),
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
            "reason": "mixed_precision_gradient_scaling_failed_before_first_optimizer_step",
            "scope": "numerical_execution_only",
            "schedule_changed": False,
            "source_checkpoint_changed": False,
            "reset_seed_changed": False,
            "maximum_update_bearing_trajectories": 1,
            "maximum_zero_update_runtime_recoveries": 1,
            "process_launch_count_after_relaunch": 2,
            "no_further_recovery_allowed": True,
        },
    }
    _atomic_write_json(payload, output)
    validate_recovery_payload(
        payload,
        artifact_path=output,
        source_checkpoint=source_checkpoint,
        activation_audit=activation_audit,
    )
    return payload


def validate(args: argparse.Namespace) -> dict[str, Any]:
    artifact_path = args.recovery_audit.expanduser().resolve()
    payload = _read_json_object(artifact_path, role="recovery audit")
    summary = validate_recovery_payload(
        payload,
        artifact_path=artifact_path,
        source_checkpoint=args.source_checkpoint.expanduser().resolve(),
        activation_audit=args.activation_audit.expanduser().resolve(),
    )
    if args.rescue_audit is not None:
        rescue = _read_json_object(
            args.rescue_audit.expanduser().resolve(), role="completed rescue audit"
        )
        _validate_rescue_binding(
            rescue,
            recovery_payload=payload,
            recovery_path=artifact_path,
        )
        summary["completed_rescue_binding_valid"] = True
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Preserve a failed launch once")
    create_parser.add_argument("--source-checkpoint", required=True, type=Path)
    create_parser.add_argument("--activation-audit", required=True, type=Path)
    create_parser.add_argument("--failed-stdout-log", required=True, type=Path)
    create_parser.add_argument("--failed-stderr-log", required=True, type=Path)
    create_parser.add_argument("--rescue-checkpoint", required=True, type=Path)
    create_parser.add_argument("--rescue-audit", required=True, type=Path)
    create_parser.add_argument("--evaluation-root", required=True, type=Path)
    create_parser.add_argument("--repo-root", required=True, type=Path)
    create_parser.add_argument("--git-commit-at-failed-launch", required=True)
    create_parser.add_argument("--rescue-protocol-commit", required=True)
    create_parser.add_argument("--output", required=True, type=Path)
    create_parser.add_argument("--confirm-first-step-zero-update", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Fail closed on an artifact")
    validate_parser.add_argument("--recovery-audit", required=True, type=Path)
    validate_parser.add_argument("--source-checkpoint", required=True, type=Path)
    validate_parser.add_argument("--activation-audit", required=True, type=Path)
    validate_parser.add_argument("--rescue-audit", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = create(args) if args.command == "create" else validate(args)
    except (OSError, ValueError, RecoveryEvidenceError) as exc:
        raise SystemExit(f"RECOVERY_EVIDENCE_ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
