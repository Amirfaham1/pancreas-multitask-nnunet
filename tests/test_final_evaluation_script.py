"""Static contract checks for the final-evaluation PowerShell orchestrator."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "Run-FinalEvaluation.ps1"


def test_final_evaluation_script_declares_fixed_candidates_and_validation_inputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

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
    source = SCRIPT.read_text(encoding="utf-8")

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
