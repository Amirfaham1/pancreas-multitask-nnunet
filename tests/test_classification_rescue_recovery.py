from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "classification_rescue_recovery.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_and_activation(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "checkpoint_final.pth"
    source.write_bytes(b"fixed-source-checkpoint")
    activation = tmp_path / "classification_rescue_activation.json"
    activation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "activation_approved": True,
                "source_checkpoint_name": "checkpoint_final.pth",
                "source_checkpoint_sha256": _sha256(source),
                "validation_metrics_read": False,
                "validation_used_for_activation": False,
            }
        ),
        encoding="utf-8",
    )
    return source, activation


def _write_failure_logs(tmp_path: Path) -> tuple[Path, Path]:
    stdout = tmp_path / "watcher.stdout.log"
    stdout.write_text(
        "2026-08-06T02:15:07-04:00 ACTIVATION_APPROVED=true\n"
        "2026-08-06T02:15:08-04:00 CLASSIFICATION_RESCUE_START\n"
        "Using splits from existing split file: splits_final.json\n"
        "This split has 252 training and 36 validation cases.",
        encoding="utf-8",
    )
    stderr = tmp_path / "watcher.stderr.log"
    stderr.write_text(
        "classification_rescue_train_step(\n"
        "gradient_norm = torch.nn.utils.clip_grad_norm_(\n"
        "RuntimeError: gradients from `parameters` is non-finite\n"
        "Classification rescue exited with code 1\n"
        "OVERNIGHT_PIPELINE_FAILED",
        encoding="utf-8",
    )
    return stdout, stderr


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_create_and_validate_zero_update_recovery(tmp_path: Path) -> None:
    source, activation = _write_source_and_activation(tmp_path)
    stdout, stderr = _write_failure_logs(tmp_path)
    output = tmp_path / "classification_rescue_zero_update_recovery.json"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    completed = _run(
        "create",
        "--source-checkpoint",
        str(source),
        "--activation-audit",
        str(activation),
        "--failed-stdout-log",
        str(stdout),
        "--failed-stderr-log",
        str(stderr),
        "--rescue-checkpoint",
        str(tmp_path / "checkpoint_classification_rescue.pth"),
        "--rescue-audit",
        str(tmp_path / "checkpoint_classification_rescue.pth.audit.json"),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--repo-root",
        str(ROOT),
        "--git-commit-at-failed-launch",
        commit,
        "--rescue-protocol-commit",
        commit,
        "--output",
        str(output),
        "--confirm-first-step-zero-update",
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["failed_launch"]["optimizer_updates"] == 0
    assert payload["failed_launch"]["process_launch_index"] == 1
    assert payload["recovery_policy"]["process_launch_count_after_relaunch"] == 2

    validated = _run(
        "validate",
        "--recovery-audit",
        str(output),
        "--source-checkpoint",
        str(source),
        "--activation-audit",
        str(activation),
    )
    assert validated.returncode == 0, validated.stderr
    summary = json.loads(validated.stdout)
    assert summary["failed_launch_optimizer_updates"] == 0
    assert summary["rescue_process_validation_batches_consumed"] == 0

    payload["failed_launch"]["first_step_zero_update_operator_attested"] = False
    output.write_text(json.dumps(payload), encoding="utf-8")
    rejected_attestation = _run(
        "validate",
        "--recovery-audit",
        str(output),
        "--source-checkpoint",
        str(source),
        "--activation-audit",
        str(activation),
    )
    assert rejected_attestation.returncode != 0
    assert "first_step_zero_update_operator_attested" in rejected_attestation.stderr

    payload["failed_launch"]["first_step_zero_update_operator_attested"] = True
    payload["pre_failure_implementation"]["train_script_git_blob"] = "0" * 40
    output.write_text(json.dumps(payload), encoding="utf-8")
    rejected_blob = _run(
        "validate",
        "--recovery-audit",
        str(output),
        "--source-checkpoint",
        str(source),
        "--activation-audit",
        str(activation),
    )
    assert rejected_blob.returncode != 0
    assert "train_script_git_blob" in rejected_blob.stderr


def test_validation_rejects_tampered_preserved_log(tmp_path: Path) -> None:
    source, activation = _write_source_and_activation(tmp_path)
    stdout, stderr = _write_failure_logs(tmp_path)
    output = tmp_path / "classification_rescue_zero_update_recovery.json"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    created = _run(
        "create",
        "--source-checkpoint",
        str(source),
        "--activation-audit",
        str(activation),
        "--failed-stdout-log",
        str(stdout),
        "--failed-stderr-log",
        str(stderr),
        "--rescue-checkpoint",
        str(tmp_path / "checkpoint_classification_rescue.pth"),
        "--rescue-audit",
        str(tmp_path / "checkpoint_classification_rescue.pth.audit.json"),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--repo-root",
        str(ROOT),
        "--git-commit-at-failed-launch",
        commit,
        "--rescue-protocol-commit",
        commit,
        "--output",
        str(output),
        "--confirm-first-step-zero-update",
    )
    assert created.returncode == 0, created.stderr

    preserved = tmp_path / "classification_rescue_recovery_evidence" / "failed_launch.stderr.log"
    preserved.write_text("tampered", encoding="utf-8")
    validated = _run(
        "validate",
        "--recovery-audit",
        str(output),
        "--source-checkpoint",
        str(source),
        "--activation-audit",
        str(activation),
    )
    assert validated.returncode != 0
    assert "byte count differs" in validated.stderr or "SHA-256 differs" in validated.stderr
