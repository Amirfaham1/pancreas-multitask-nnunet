from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts" / "Run-InferenceSpeedBenchmark.ps1").read_text(
    encoding="utf-8"
)


def test_wrapper_locks_abba_fresh_process_arms_and_fixed_inference_settings() -> None:
    assert SOURCE.index("reference_1") < SOURCE.index("candidate_1")
    assert SOURCE.index("candidate_1") < SOURCE.index("candidate_2")
    assert SOURCE.index("candidate_2") < SOURCE.index("reference_2")
    for setting in (
        "'--folds', '0'",
        "'--device', 'cuda'",
        "'--tile-step-size', '0.5'",
        "'--overwrite'",
    ):
        assert setting in SOURCE
    assert "--disable-tta" not in SOURCE
    assert "--disable-gaussian" not in SOURCE
    assert "& $Python @predictionArguments" in SOURCE


def test_wrapper_guards_checkpoint_and_never_removes_or_reuses_work_root() -> None:
    assert SOURCE.count("Get-FileHash -LiteralPath $checkpointPath") == 2
    assert "Checkpoint changed during the paired benchmark" in SOURCE
    assert "WorkRoot must not already exist" in SOURCE
    assert "Remove-Item" not in SOURCE
    assert "--reference-runtime" in SOURCE
    assert "--candidate-runtime" in SOURCE
