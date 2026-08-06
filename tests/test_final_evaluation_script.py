"""Static contract checks for the final-evaluation PowerShell orchestrator."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "Run-FinalEvaluation.ps1"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def test_final_evaluation_script_parses_as_powershell_without_running_it() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable; static contract tests still run")
    parser_command = r"""
& {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $env:FINAL_EVALUATION_SCRIPT_TO_PARSE, [ref] $tokens, [ref] $errors
    ) | Out-Null
    if ($errors.Count -gt 0) {
        $errors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }
        exit 1
    }
}
"""
    environment = os.environ.copy()
    environment["FINAL_EVALUATION_SCRIPT_TO_PARSE"] = str(SCRIPT)
    completed = subprocess.run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            parser_command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_final_evaluation_script_declares_fixed_candidates_and_validation_inputs() -> None:
    source = SOURCE

    assert '[string] $WorkRoot = "D:\\MLQuizWork"' in source
    assert "$workRoot = [IO.Path]::GetFullPath($WorkRoot)" in source
    for checkpoint in (
        "checkpoint_best.pth",
        "checkpoint_best_multitask.pth",
        "checkpoint_final.pth",
    ):
        assert checkpoint in source
    assert '"imagesVal"' in source
    assert '"labelsVal"' in source
    assert '"classification_manifest.json"' in source
    assert '"--classification-reference-split", "validation"' in source
    assert '"--folds", "0"' in source


def test_final_evaluation_script_preserves_resume_and_stops_before_submission() -> None:
    source = SOURCE

    assert 'if ($Force)' in source
    assert '$predictionArguments += "--overwrite"' in source
    assert '$predictionArguments += "--no-overwrite"' in source
    assert '"--probability-csv", $probabilityCsv' in source
    assert '"--runtime-json", $runtimeJson' in source
    assert "select_checkpoint.py" in source
    assert "Get-CimInstance Win32_Process" in source
    assert 'WandbMode = "disabled"' in source
    assert '$env:WANDB_MODE = "disabled"' in source
    assert "imagesTs" not in source
    assert "Compress-Archive" not in source
    assert "validate_submission.py" not in source


def test_final_evaluation_uses_process_lifetime_single_instance_mutex() -> None:
    source = SOURCE

    assert '"Local\\PancreasMultitaskPostTraining501Fold0"' in source
    assert "$postTrainingMutex.WaitOne(0)" in source
    assert "catch [Threading.AbandonedMutexException]" in source
    assert "$postTrainingMutex.ReleaseMutex()" in source
    assert "$postTrainingMutex.Dispose()" in source


def test_resume_preserves_completed_first_pass_runtime_artifacts() -> None:
    source = SOURCE

    assert "Preserving completed first-pass runtime artifact" in source
    assert 'Get-RequiredJsonProperty $existingRuntime "case_count"' in source
    assert 'Get-RequiredJsonProperty $existingRuntime "checkpoint"' in source
    assert 'Get-RequiredJsonProperty $existingRuntime "total_seconds"' in source
    assert '$predictionArguments += @("--runtime-json", $runtimeJson)' in source


def test_activation_audit_deterministically_controls_three_or_four_candidates() -> None:
    source = SOURCE

    assert "[switch] $IncludeClassificationRescue" in source
    assert '"classification_rescue_activation.json"' in source
    assert 'if (-not $activationApproved)' in source
    assert 'if ($IncludeClassificationRescue)' in source
    assert 'if (-not $IncludeClassificationRescue)' in source
    assert 'Name = "checkpoint_classification_rescue"' in source
    assert 'FileName = "checkpoint_classification_rescue.pth"' in source
    assert "evaluating exactly 3 candidates" in source
    assert "evaluating exactly 4 candidates" in source

    final_check = source.index("$finalCheckpointSha256 = Get-FileSha256")
    activation_read = source.index("$activationAudit = Read-JsonObject")
    output_creation = source.index(
        "New-Item -ItemType Directory -Path $evaluationRoot -Force"
    )
    assert final_check < activation_read < output_creation


def test_rescue_provenance_and_no_validation_use_are_checked_before_inference() -> None:
    source = SOURCE

    for field in (
        "source_checkpoint_sha256",
        "activation_audit_sha256",
        "output_checkpoint_sha256",
        "status",
        "completed_epochs",
        "validation_images_opened",
        "validation_batches_consumed",
        "validation_used_for_gradients",
        "validation_used_for_stopping",
    ):
        assert field in source
    assert "Get-FileHash -LiteralPath $Path -Algorithm SHA256" in source
    assert '"train_classification_rescue.py"' in source
    assert "classification rescue active" in source

    rescue_checks = source.index("$rescueCheckpointSha256 = Get-FileSha256")
    inference_loop = source.index('Write-Host "[$($candidate.Name)] Running fixed-validation')
    assert rescue_checks < inference_loop


def test_selection_is_one_equal_score_pass_over_the_final_candidate_array() -> None:
    source = SOURCE

    assert source.count('-Stage "checkpoint selection"') == 1
    assert source.count("$selectionArguments = @()") == 1
    assert "foreach ($candidate in $candidates)" in source
