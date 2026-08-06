from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts" / "Run-InferenceSpeedBenchmark.ps1").read_text(
    encoding="utf-8"
)


def test_wrapper_locks_v3_abba_full_vs_neural_only_arms() -> None:
    assert SOURCE.index("reference_1") < SOURCE.index("candidate_1")
    assert SOURCE.index("candidate_1") < SOURCE.index("candidate_2")
    assert SOURCE.index("candidate_2") < SOURCE.index("reference_2")
    assert SOURCE.count("ExtractionMode = 'full'") == 2
    assert SOURCE.count("ExtractionMode = 'neural_only'") == 2
    assert "'--v5-extraction-mode', $entry.Value.ExtractionMode" in SOURCE


def test_wrapper_propagates_final_v5_bundle_and_dataset_bindings() -> None:
    for parameter in (
        "$NeuralCaseHeadBundle",
        "$ExpectedNeuralCaseHeadBundleSha256",
        "$ExpectedNumericTrainDatasetSha256",
    ):
        assert parameter in SOURCE
    for argument in (
        "'--classification-mode', 'neural-v5'",
        "'--neural-case-head-bundle', $resolvedBundle",
        "'--expected-neural-case-head-bundle-sha256', $ExpectedNeuralCaseHeadBundleSha256.ToLowerInvariant()",
        "'--expected-numeric-train-dataset-sha256', $ExpectedNumericTrainDatasetSha256.ToLowerInvariant()",
    ):
        assert argument in SOURCE
    assert "$resolvedBundle = (Resolve-Path -LiteralPath $NeuralCaseHeadBundle).Path" in SOURCE
    assert SOURCE.count("Get-FileHash -LiteralPath $resolvedBundle") == 2
    assert "Neural case-head bundle changed during the paired benchmark" in SOURCE


def test_wrapper_locks_tile1_tta1_and_fixed_inference_settings_in_both_arms() -> None:
    for setting in (
        "'--folds', '0'",
        "'--device', 'cuda'",
        "'--tile-step-size', '0.5'",
        "'--tile-batch-size', '1'",
        "'--tta-batch-size', '1'",
        "'--overwrite'",
    ):
        assert setting in SOURCE
    assert "--disable-tta" not in SOURCE
    assert "--disable-gaussian" not in SOURCE
    assert "& $Python @predictionArguments" in SOURCE
    assert "$env:nnUNet_extTrainer = Join-Path $repositoryRoot 'src'" in SOURCE
    assert "$env:nnUNet_compile = 'false'" in SOURCE


def test_wrapper_verifies_and_archives_exact_frozen_v3_protocol_lock() -> None:
    assert "configs\\inference_speed_benchmark_v3.json" in SOURCE
    assert (
        "$expectedProtocolLockSha256 = "
        "'3a57ab79147a6dd9ab4ee3fa99fdb2be978e9c60f290cead7a52298673e926aa'"
    ) in SOURCE
    assert "Get-FileHash -LiteralPath $lockArtifact -Algorithm SHA256" in SOURCE
    assert "Inference-speed v3 protocol lock differs from its prospective SHA-256" in SOURCE
    assert "inference_speed_benchmark_v3.lock.json" in SOURCE


def test_wrapper_guards_checkpoint_and_never_removes_or_reuses_work_root() -> None:
    assert SOURCE.count("Get-FileHash -LiteralPath $checkpointPath") == 2
    assert "Checkpoint changed during the paired benchmark" in SOURCE
    assert "WorkRoot must not already exist" in SOURCE
    assert "Remove-Item" not in SOURCE
    assert "--reference-runtime" in SOURCE
    assert "--candidate-runtime" in SOURCE


def test_wrapper_locks_final_checkpoint_plans_dataset_and_raw_input_content() -> None:
    for digest in (
        "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116",
        "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f",
        "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff",
        "324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1",
        "b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2",
        "1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8",
    ):
        assert digest in SOURCE
    assert "$Checkpoint -ne 'checkpoint_classification_rescue.pth'" in SOURCE
    assert SOURCE.count("Get-FileHash -LiteralPath $plansPath") == 2
    assert SOURCE.count("Get-FileHash -LiteralPath $datasetJsonPath") == 2
    assert "function Get-RawInputManifestJson" in SOURCE
    assert "$inputManifestBefore = Get-RawInputManifestJson" in SOURCE
    assert "$inputManifestAfter = Get-RawInputManifestJson" in SOURCE
    assert "Raw input inventory or content changed" in SOURCE
    assert "ConvertFrom-Json" in SOURCE
    assert "component_hashes_before.$component" in SOURCE
    assert "component_hashes_after.$component" in SOURCE
    assert "did not preserve the frozen network" in SOURCE
