from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "scripts" / "V5-LockedDeliveryCommon.ps1"
FINAL = ROOT / "scripts" / "Run-V5LockedFinalEvaluation.ps1"
TEST_PACKAGE = ROOT / "scripts" / "Run-V5LockedSelectedTestAndPackage.ps1"
COMMON_SOURCE = COMMON.read_text(encoding="utf-8")
FINAL_SOURCE = FINAL.read_text(encoding="utf-8")
TEST_SOURCE = TEST_PACKAGE.read_text(encoding="utf-8")


FIXED_HASHES = {
    "checkpoint": "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116",
    "plans": "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f",
    "dataset": "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff",
    "encoder": "324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1",
    "decoder": "b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2",
    "classification": "1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8",
}
LOCAL_MODEL = Path(
    r"D:\MLQuizWork\nnUNet_results\Dataset501_PancreasMultitask"
    r"\nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres"
)
LOCAL_BUNDLE = Path(
    r"D:\MLQuizWork\phd_upgrade_v5\neural_training\neural_case_head_v5.pth"
)
LOCAL_BUNDLE_SHA256 = "6e4ed210bc23cd7c7bfe02c46816dd8461c0be84108d4f9d2a36f1409b6df09d"
LOCAL_NUMERIC_DATASET_SHA256 = (
    "ecbac559e3fa4d3353618c1b7a0e85e672a3e73daafe5a7d3bf67beaf1f1e140"
)


def _powershell_executable() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


@pytest.mark.parametrize("script", [COMMON, FINAL, TEST_PACKAGE])
def test_powershell_source_parses(script: Path) -> None:
    executable = _powershell_executable()
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    environment = os.environ.copy()
    environment["V5_SCRIPT_TO_PARSE"] = str(script)
    completed = subprocess.run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            r"""
$tokens = $null
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    $env:V5_SCRIPT_TO_PARSE, [ref] $tokens, [ref] $errors
) | Out-Null
if ($errors.Count -ne 0) {
    $errors | ForEach-Object { Write-Error $_.Message }
    exit 1
}
""",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_common_helper_hard_binds_every_known_final_artifact() -> None:
    for digest in FIXED_HASHES.values():
        assert digest in COMMON_SOURCE
    for digest in (
        "a8c2147493718acc96e4aa5dc471bf3f3277f0b99e8a8f7620bf966ab7b70d11",
        "e28a303c7d3da5dc7857ecc72787b6746d1e689e83167c500d4d2823c5ea540f",
    "3a57ab79147a6dd9ab4ee3fa99fdb2be978e9c60f290cead7a52298673e926aa",
    "563d9d5e4fbe0f92653c6b7295c476d0ddf5d239c47beb1948410bbb80a7c2e2",
    "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd",
        "bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503",
    ):
        assert digest in COMMON_SOURCE
    for field in (
        "ExpectedFinalCandidateLockSha256",
        "ExpectedCheckpointSha256",
        "ExpectedNeuralCaseHeadBundleSha256",
        "ExpectedNumericTrainDatasetSha256",
        "ExpectedPlansSha256",
        "ExpectedDatasetJsonSha256",
        "ExpectedEncoderComponentSha256",
        "ExpectedDecoderComponentSha256",
        "ExpectedClassificationComponentSha256",
    ):
        assert field in COMMON_SOURCE
        assert field in FINAL_SOURCE
        assert field in TEST_SOURCE


def test_final_lock_schema_and_code_manifest_are_fail_closed() -> None:
    for field in (
        "locked_before_single_v5_official_reevaluation_and_selected_v5_test_inference",
        "assignment_conforming_v5_neural_case_head",
        "best_of_two_locked_neural_heads",
        "v5_head_architecture_training_selection_and_offsets_used_official_validation_or_test_data",
        "v5_speed_development_used_official_validation_or_test_data",
        "baseline_official_validation_observed_before_v5_extension",
        "frozen_checkpoint_was_validation_selected",
        "baseline_test_inference_and_packaging_occurred_before_v5_extension",
        "baseline_test_inputs_were_a_v5_tuning_signal",
        "artifacts",
        "frozen_components",
        "inference_contract",
        "protocol_locks",
        "implementation_files",
        "train_only_audits",
        "run_ledger_files",
    ):
        assert field in COMMON_SOURCE
    for relative_path in (
        "scripts/predict_joint.py",
        "scripts/evaluate_predictions.py",
        "scripts/Package-Submission.ps1",
        "scripts/validate_submission.py",
        "scripts/V5-LockedDeliveryCommon.ps1",
        "scripts/Run-V5LockedFinalEvaluation.ps1",
        "scripts/Run-V5LockedSelectedTestAndPackage.ps1",
        "src/pancreas_multitask/classification_rescue.py",
        "src/pancreas_multitask/inference_determinism.py",
        "src/pancreas_multitask/network.py",
        "src/pancreas_multitask/predictor.py",
        "src/pancreas_multitask/case_features.py",
        "src/pancreas_multitask/case_feature_extractor.py",
        "src/pancreas_multitask/neural_case_head.py",
        "src/pancreas_multitask/neural_case_bundle.py",
        "src/pancreas_multitask/neural_case_training.py",
        "src/pancreas_multitask/neural_case_predictor.py",
    ):
        assert relative_path in COMMON_SOURCE
    assert "V5 output root must not exist" in COMMON_SOURCE
    assert "CreateNew" in COMMON_SOURCE
    assert "one-use run ledger" in COMMON_SOURCE


def test_cpu_preflight_strict_loads_bundle_and_recomputes_model_components() -> None:
    preflight = COMMON_SOURCE[
        COMMON_SOURCE.index("function Invoke-V5StrictCpuPreflight") :
        COMMON_SOURCE.index("function Invoke-V5CheckedPython")
    ]
    assert "load_neural_case_head_bundle" in preflight
    assert "NeuralCaseNNUNetPredictor" in preflight
    assert "torch.device(\"cpu\")" in preflight
    assert "use_folds=(0,)" in preflight
    assert "checkpoint_name=os.environ" in preflight
    assert "component_hashes_before" in preflight
    assert "component_hashes_after" in preflight
    assert "frozen_components_unchanged" in preflight
    assert "any_network_parameter_requires_grad" in preflight
    assert "selection_audit_sha256" in preflight
    assert "calibration_audit_sha256" in preflight
    assert "refit_audit_sha256" in preflight


def test_train_only_audit_chain_and_online_wandb_evidence_are_required() -> None:
    for filename in (
        "neural_case_head_fit_audit.json",
        "neural_case_head_selection.json",
        "neural_decision_calibration.json",
        "neural_case_head_refit.json",
    ):
        assert filename in COMMON_SOURCE
    for field in (
        "isolated_supplied_train_only",
        "official_validation_images_masks_labels_or_metrics_used",
        "official_validation_or_test_used",
        "test_data_used",
        "ground_truth_masks_used_as_features",
        "encoder_decoder_and_rescue_head_frozen",
        "effective_mode",
        "run_id",
        "run_url",
    ):
        assert field in COMMON_SOURCE


def test_validation_prediction_is_frozen_before_this_wrappers_reference_access() -> None:
    prediction = FINAL_SOURCE.index(
        '-Stage "One locked label-blind official-validation inference"'
    )
    output_audit = FINAL_SOURCE.index("Get-V5InferenceArtifactSet", prediction)
    manifest_write = FINAL_SOURCE.index(
        "Write-V5JsonAtomic -Path $preReferenceManifest", output_audit
    )
    manifest_hash = FINAL_SOURCE.index(
        "$preReferenceManifestSha256 = Get-V5FileSha256", manifest_write
    )
    reference_access = FINAL_SOURCE.index(
        'Assert-V5Directory $ReferenceMasks "Official reference-mask directory"',
        manifest_hash,
    )
    evaluation = FINAL_SOURCE.index(
        '-Stage "The single saved-output official evaluation"', reference_access
    )
    assert prediction < output_audit < manifest_write < manifest_hash < reference_access < evaluation

    before_reference = FINAL_SOURCE[:reference_access]
    assert "Get-ChildItem -LiteralPath $ReferenceMasks" not in before_reference
    assert "Get-Content -LiteralPath $ReferenceSubtypes" not in before_reference
    assert "Test-Path -LiteralPath $ReferenceMasks" not in before_reference
    assert "Test-Path -LiteralPath $ReferenceSubtypes" not in before_reference
    assert FINAL_SOURCE.count('-Stage "One locked label-blind official-validation inference"') == 1
    assert FINAL_SOURCE.count('-Stage "The single saved-output official evaluation"') == 1
    assert "select_checkpoint.py" not in FINAL_SOURCE
    assert "train_neural_case_heads.py" not in FINAL_SOURCE


def test_validation_contract_is_exactly_fold0_neural_only_tta_on_batch1() -> None:
    for fragment in (
        '"--folds", "0"',
        '"--checkpoint", $script:V5CheckpointName',
        '"--classification-mode", "neural-v5"',
        '"--v5-extraction-mode", "neural_only"',
        '"--tile-step-size", "0.5"',
        '"--tile-batch-size", "1"',
        '"--tta-batch-size", "1"',
        '"--overwrite"',
    ):
        assert fragment in FINAL_SOURCE
        assert fragment in TEST_SOURCE
    assert "--disable-tta" not in FINAL_SOURCE
    assert "--disable-gaussian" not in FINAL_SOURCE
    assert "--disable-tta" not in TEST_SOURCE
    assert "--disable-gaussian" not in TEST_SOURCE
    assert "ExpectedCaseCount 36" in FINAL_SOURCE
    assert "ExpectedCaseCount 72" in TEST_SOURCE


def test_official_gate_records_all_required_thresholds_without_iteration() -> None:
    for value in (
        "0.46399340516987575",
        "0.70",
        "0.91",
        "0.31",
    ):
        assert value in FINAL_SOURCE or value in COMMON_SOURCE
    for field in (
        "strict_macro_f1_improvement_over_baseline",
        "phd_macro_f1_gate",
        "whole_pancreas_gate",
        "lesion_gate",
        "phd_joint_metric_gate",
        "classifier_replacement_validation_gate",
        "complete_no_second_classifier_iteration_permitted",
        "further_classifier_training_selection_or_official_evaluation_permitted",
    ):
        assert field in FINAL_SOURCE
    assert "$macroF1 -gt $script:V5BaselineMacroF1" in FINAL_SOURCE
    assert "$macroF1 -ge 0.70" in FINAL_SOURCE
    assert "$wholeDice -ge 0.91" in FINAL_SOURCE
    assert "$lesionDice -ge 0.31" in FINAL_SOURCE


def test_test_pipeline_requires_passing_gate_and_packages_exact_flat_contract() -> None:
    gate_check = TEST_SOURCE.index("$verdicts = Get-V5RequiredProperty")
    artifact_rehash = TEST_SOURCE.index("Assert-V5RecordedInferenceArtifactSet")
    bound_metrics = TEST_SOURCE.index("$boundMetrics = Read-V5JsonObject")
    consumed_ledger = TEST_SOURCE.index(
        '$officialLedger = Read-V5JsonObject $officialLedgerPath'
    )
    inference = TEST_SOURCE.index('-Stage "The single locked selected-test inference"')
    package = TEST_SOURCE.index("& $packageScript", inference)
    completion = TEST_SOURCE.index(
        'status = "complete_single_selected_candidate_validated"', package
    )
    assert (
        gate_check
        < artifact_rehash
        < bound_metrics
        < consumed_ledger
        < inference
        < package
        < completion
    )
    assert TEST_SOURCE.count('-Stage "The single locked selected-test inference"') == 1
    assert TEST_SOURCE.count("& $packageScript") == 1
    assert "classifier_replacement_validation_gate" in TEST_SOURCE
    assert '"complete_and_consumed"' in TEST_SOURCE
    assert '"gate_artifact_sha256"' in TEST_SOURCE
    assert "hash-bound evaluator output" in TEST_SOURCE
    assert "-ProtectedPaths @($validationOutputRoot)" in TEST_SOURCE
    assert '"Amirfaham_Fallahpour_results.zip"' in TEST_SOURCE
    assert "Flat submission ZIP must contain exactly 73 files" in TEST_SOURCE
    assert '"expected_cases", "masks", "subtype_rows"' in TEST_SOURCE
    assert "-Force" not in TEST_SOURCE
    assert "input_or_baseline_replaced = $false" in TEST_SOURCE


def test_runtime_audit_requires_immutable_fresh_v5_execution() -> None:
    for field in (
        "checkpoint_unchanged_during_run",
        "model_configuration_unchanged_during_run",
        "input_files_unchanged_during_run",
        "disabled_online_fresh_extraction",
        "v5_offset_adjusted_three_class",
        "expected_bundle_sha256_verified",
        "head_in_eval_mode",
        "head_state_unchanged",
        "network_in_eval_mode",
        "frozen_components_unchanged",
        "v5_case_extractions_completed",
        "v5_neural_head_forward_calls",
        "v5_class_offset_applications",
        "v5_feature_cache_reads",
        "tile_batch_oom_fallback_count",
        "tta_batch_oom_fallback_count",
        "v5_neural_bag_sha256_sequence",
        "v5_implementation_files",
        "strict_cuda_inference_v1",
        "configured_before_cuda_initialization",
        "after_initial_configuration",
        "after_predictor_construction",
        "after_inference",
        "settings_unchanged",
        "installed_nnunet_source",
        "stock_export_conformance",
        "export_logit_dtype",
        "segmentation_export_logit_dtype",
        "segmentation_export_logit_dtype_sequence",
    ):
        assert field in COMMON_SOURCE
    assert "8 * $tileCount" in COMMON_SOURCE


def test_bad_lock_hash_fails_before_any_official_path_check(tmp_path: Path) -> None:
    executable = _powershell_executable()
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    bogus_lock = tmp_path / "candidate.json"
    bogus_lock.write_text('{"schema_version":1}\n', encoding="utf-8")
    output_root = tmp_path / "must_not_be_created"
    nonexistent = tmp_path / "official_tripwire_must_not_be_checked"
    nonexistent_model = tmp_path / "model_must_not_be_checked"
    nonexistent_bundle = tmp_path / "bundle_must_not_be_checked.pth"
    actual_lock_hash = hashlib.sha256(bogus_lock.read_bytes()).hexdigest()
    assert actual_lock_hash != "0" * 64

    command = [
        executable,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(FINAL),
        "-FinalCandidateLock",
        str(bogus_lock),
        "-ExpectedFinalCandidateLockSha256",
        "0" * 64,
        "-ModelDirectory",
        str(nonexistent_model),
        "-NeuralCaseHeadBundle",
        str(nonexistent_bundle),
        "-ExpectedCheckpointSha256",
        FIXED_HASHES["checkpoint"],
        "-ExpectedNeuralCaseHeadBundleSha256",
        "1" * 64,
        "-ExpectedNumericTrainDatasetSha256",
        "2" * 64,
        "-ExpectedPlansSha256",
        FIXED_HASHES["plans"],
        "-ExpectedDatasetJsonSha256",
        FIXED_HASHES["dataset"],
        "-ExpectedEncoderComponentSha256",
        FIXED_HASHES["encoder"],
        "-ExpectedDecoderComponentSha256",
        FIXED_HASHES["decoder"],
        "-ExpectedClassificationComponentSha256",
        FIXED_HASHES["classification"],
        "-ValidationImages",
        str(nonexistent),
        "-ReferenceMasks",
        str(nonexistent / "references"),
        "-ReferenceSubtypes",
        str(nonexistent / "labels.json"),
        "-OutputRoot",
        str(output_root),
        "-PythonExecutable",
        sys.executable,
        "-Device",
        "cuda",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = completed.stderr + completed.stdout
    assert completed.returncode != 0
    assert "Final-candidate lock file SHA-256 mismatch" in output
    assert "Official validation image directory was not found" not in output
    assert "Official reference-mask directory was not found" not in output
    assert not output_root.exists()
    assert not list(tmp_path.glob("*official*ledger*.json"))


def test_common_runtime_and_output_audits_execute_under_powershell(tmp_path: Path) -> None:
    executable = _powershell_executable()
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    model = tmp_path / "model"
    input_directory = tmp_path / "inputs"
    output_root = tmp_path / "output"
    predictions = output_root / "predictions"
    evidence = output_root / "evidence"
    predictions.mkdir(parents=True)
    evidence.mkdir()
    input_directory.mkdir()
    for case_id in ("case_a", "case_b"):
        (predictions / f"{case_id}.nii.gz").write_bytes(case_id.encode("ascii"))
    classification = predictions / "subtype_results.csv"
    probability = evidence / "subtype_probabilities.csv"
    classification.write_text(
        "Names,Subtype\ncase_a.nii.gz,0\ncase_b.nii.gz,2\n",
        encoding="utf-8",
    )
    probability.write_text(
        "Names,Subtype,Probability_0,Probability_1,Probability_2\n"
        "case_a.nii.gz,0,0.8,0.1,0.1\n"
        "case_b.nii.gz,2,0.1,0.2,0.7\n",
        encoding="utf-8",
    )

    implementation_paths = [
        "scripts/predict_joint.py",
        "src/pancreas_multitask/classification_rescue.py",
        "src/pancreas_multitask/inference_determinism.py",
        "src/pancreas_multitask/network.py",
        "src/pancreas_multitask/predictor.py",
        "src/pancreas_multitask/case_features.py",
        "src/pancreas_multitask/case_feature_extractor.py",
        "src/pancreas_multitask/neural_case_head.py",
        "src/pancreas_multitask/neural_case_bundle.py",
        "src/pancreas_multitask/neural_case_training.py",
        "src/pancreas_multitask/neural_case_predictor.py",
    ]
    implementation_hash = "d" * 64
    runtime = {
        "case_count": 2,
        "case_ids": ["case_a", "case_b"],
        "checkpoint": "checkpoint_classification_rescue.pth",
        "checkpoint_files": [{"fold": "0", "sha256": FIXED_HASHES["checkpoint"]}],
        "classifier_pipeline": "assignment_conforming_v5_neural_case_head",
        "device": "cpu",
        "folds": [0],
        "gaussian_enabled": True,
        "tta_enabled": True,
        "overwrite": True,
        "tile_step_size": 0.5,
        "input_directory": str(input_directory.resolve()),
        "model_directory": str(model.resolve()),
        "checkpoint_unchanged_during_run": True,
        "model_configuration_unchanged_during_run": True,
        "input_files_unchanged_during_run": True,
        "feature_cache_policy": "disabled_online_fresh_extraction",
        "class_probabilities": "v5_offset_adjusted_three_class",
        "case_identifiers_or_paths_used_as_model_inputs": False,
        "deterministic_execution": {
            "policy": "strict_cuda_inference_v1",
            "configured_before_cuda_initialization": True,
            "after_initial_configuration": {
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_tf32": False,
                "cudnn_tf32": False,
                "cublas_workspace_config": ":4096:8",
                "nnunet_compile": "false",
            },
            "after_predictor_construction": {
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_tf32": False,
                "cudnn_tf32": False,
                "cublas_workspace_config": ":4096:8",
                "nnunet_compile": "false",
            },
            "after_inference": {
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_tf32": False,
                "cudnn_tf32": False,
                "cublas_workspace_config": ":4096:8",
                "nnunet_compile": "false",
            },
            "settings_unchanged": True,
            "autocast_cuda_float16": False,
            "conformance_lock": {
                "path": str(
                    (
                        tmp_path
                        / "configs"
                        / "inference_determinism_conformance_v1.json"
                    ).resolve()
                ),
                "sha256": (
                    "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd"
                ),
                "size_bytes": 100,
                "unchanged_during_run": True,
            },
            "installed_nnunet_source": {
                "before": {
                    "path": r"C:\fake\nnunetv2\inference\predict_from_raw_data.py",
                    "sha256": (
                        "c350e3202a7a67c3aef12e9206a744add442110ff8a4377c1f9640104b20a31f"
                    ),
                    "size_bytes": 1000,
                },
                "after": {
                    "path": r"C:\fake\nnunetv2\inference\predict_from_raw_data.py",
                    "sha256": (
                        "c350e3202a7a67c3aef12e9206a744add442110ff8a4377c1f9640104b20a31f"
                    ),
                    "size_bytes": 1000,
                },
                "unchanged_during_run": True,
            },
        },
        "stock_export_conformance": {
            "export_logit_dtype": "torch.float16",
            "case_count_verified": 2,
            "all_case_exports_verified": True,
            "conformance_lock": {
                "path": str(
                    (
                        tmp_path
                        / "configs"
                        / "inference_stock_export_conformance_v1.json"
                    ).resolve()
                ),
                "sha256": (
                    "bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503"
                ),
                "size_bytes": 100,
                "unchanged_during_run": True,
            },
        },
        "input_file_manifest": {
            "file_count": 2,
            "manifest_sha256": "e" * 64,
            "files": [
                {"name": "case_a_0000.nii.gz", "sha256": "a" * 64, "size_bytes": 10},
                {"name": "case_b_0000.nii.gz", "sha256": "b" * 64, "size_bytes": 11},
            ],
        },
        "model_configuration_files": [
            {"name": "dataset.json", "sha256": FIXED_HASHES["dataset"]},
            {"name": "plans.json", "sha256": FIXED_HASHES["plans"]},
        ],
        "neural_case_head_bundle": {
            "bundle_sha256": "1" * 64,
            "numeric_train_dataset_sha256": "2" * 64,
            "eligible_for_official": True,
            "bundle_loaded_strictly": True,
            "expected_bundle_sha256_verified": True,
            "head_in_eval_mode": True,
            "head_state_unchanged": True,
            "any_head_parameter_requires_grad": False,
            "neural_lock_sha256": (
                "a8c2147493718acc96e4aa5dc471bf3f3277f0b99e8a8f7620bf966ab7b70d11"
            ),
            "decision_lock_sha256": (
                "e28a303c7d3da5dc7857ecc72787b6746d1e689e83167c500d4d2823c5ea540f"
            ),
        },
        "frozen_network": {
            "fold": 0,
            "frozen_components_unchanged": True,
            "network_in_eval_mode": True,
            "any_network_parameter_requires_grad": False,
            "component_hashes_before": {
                "encoder": FIXED_HASHES["encoder"],
                "decoder": FIXED_HASHES["decoder"],
                "classification": FIXED_HASHES["classification"],
            },
            "component_hashes_after": {
                "encoder": FIXED_HASHES["encoder"],
                "decoder": FIXED_HASHES["decoder"],
                "classification": FIXED_HASHES["classification"],
            },
        },
        "inference_execution": {
            "classifier_pipeline": "assignment_conforming_v5_neural_case_head",
            "v5_extraction_mode": "neural_only",
            "v5_feature_extraction_executed": True,
            "case_identifiers_or_paths_used_as_model_inputs": False,
            "segmentation_export_logit_dtype": "torch.float16",
            "segmentation_export_logit_dtype_sequence": [
                "torch.float16",
                "torch.float16",
            ],
            "v5_case_extractions_completed": 2,
            "v5_neural_head_forward_calls": 2,
            "v5_class_offset_applications": 2,
            "v5_feature_cache_reads": 0,
            "tile_batch_size_requested": 1,
            "tta_batch_size_requested": 1,
            "maximum_network_batch_size_observed": 1,
            "network_batch_size_limit": 1,
            "tile_batch_oom_fallback_count": 0,
            "tta_batch_oom_fallback_count": 0,
            "logical_tiles_completed": 2,
            "tta_views_completed": 16,
        },
        "v5_extraction_mode": "neural_only",
        "v5_neural_bag_sha256_sequence": ["3" * 64, "4" * 64],
        "v5_implementation_files": {
            path: implementation_hash for path in implementation_paths
        },
    }
    runtime_path = evidence / "runtime.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    candidate = {
        "ModelDirectory": str(model.resolve()),
        "ProjectRoot": str(tmp_path.resolve()),
        "BundleSha256": "1" * 64,
        "NumericTrainDatasetSha256": "2" * 64,
        "Lock": {
            "implementation_files": [
                {"path": path, "sha256": implementation_hash}
                for path in implementation_paths
            ]
        },
    }
    candidate_path = tmp_path / "candidate-harness.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    environment = os.environ.copy()
    environment.update(
        {
            "V5_COMMON": str(COMMON),
            "V5_RUNTIME": str(runtime_path),
            "V5_CANDIDATE": str(candidate_path),
            "V5_INPUT": str(input_directory),
            "V5_OUTPUT": str(output_root),
            "V5_PREDICTIONS": str(predictions),
            "V5_CLASSIFICATION": str(classification),
            "V5_PROBABILITY": str(probability),
        }
    )
    harness = rf"""
. $env:V5_COMMON
$candidate = Get-Content -LiteralPath $env:V5_CANDIDATE -Raw | ConvertFrom-Json
$runtime = Assert-V5RuntimeArtifact `
    -RuntimeJson $env:V5_RUNTIME `
    -ExpectedCaseCount 2 `
    -ExpectedInputDirectory $env:V5_INPUT `
    -ExpectedDevice cpu `
    -Candidate $candidate `
    -ExpectedEncoderComponentSha256 '{FIXED_HASHES['encoder']}' `
    -ExpectedDecoderComponentSha256 '{FIXED_HASHES['decoder']}' `
    -ExpectedClassificationComponentSha256 '{FIXED_HASHES['classification']}'
$set = Get-V5InferenceArtifactSet `
    -PredictionDirectory $env:V5_PREDICTIONS `
    -ClassificationCsv $env:V5_CLASSIFICATION `
    -ProbabilityCsv $env:V5_PROBABILITY `
    -RuntimeJson $env:V5_RUNTIME `
    -Runtime $runtime `
    -ExpectedCaseCount 2 `
    -OutputRoot $env:V5_OUTPUT
$manifest = [pscustomobject]@{{
    prediction_artifact_set_sha256 = $set.ArtifactSetSha256
    prediction_artifacts = $set.Artifacts
}}
$recorded = Assert-V5RecordedInferenceArtifactSet `
    -Manifest $manifest `
    -OutputRoot $env:V5_OUTPUT `
    -ExpectedArtifactSetSha256 $set.ArtifactSetSha256 `
    -ExpectedCaseCount 2
[IO.File]::AppendAllText(
    (Join-Path $env:V5_PREDICTIONS 'case_a.nii.gz'),
    'changed'
)
$mutationRejected = $false
try {{
    $null = Assert-V5RecordedInferenceArtifactSet `
        -Manifest $manifest `
        -OutputRoot $env:V5_OUTPUT `
        -ExpectedArtifactSetSha256 $set.ArtifactSetSha256 `
        -ExpectedCaseCount 2
}}
catch {{
    $mutationRejected = $true
}}
[pscustomobject]@{{
    MaskCount = $set.MaskCount
    ClassificationRowCount = $set.ClassificationRowCount
    ProbabilityRowCount = $set.ProbabilityRowCount
    Artifacts = $set.Artifacts
    ArtifactSetSha256 = $set.ArtifactSetSha256
    RecordedArtifactCount = $recorded.ArtifactCount
    MutationRejected = $mutationRejected
}} | ConvertTo-Json -Depth 8
"""
    completed = subprocess.run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            harness,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result["MaskCount"] == 2
    assert result["ClassificationRowCount"] == 2
    assert result["ProbabilityRowCount"] == 2
    assert len(result["Artifacts"]) == 5
    assert len(result["ArtifactSetSha256"]) == 64
    assert result["RecordedArtifactCount"] == 5
    assert result["MutationRejected"] is True


@pytest.mark.skipif(
    not LOCAL_BUNDLE.is_file() or not LOCAL_MODEL.is_dir(),
    reason="Local immutable v5 artifacts are unavailable",
)
def test_complete_final_candidate_lock_accepts_real_train_only_chain(
    tmp_path: Path,
) -> None:
    executable = _powershell_executable()
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    implementation_paths = [
        "scripts/predict_joint.py",
        "scripts/evaluate_predictions.py",
        "scripts/Package-Submission.ps1",
        "scripts/validate_submission.py",
        "scripts/V5-LockedDeliveryCommon.ps1",
        "scripts/Run-V5LockedFinalEvaluation.ps1",
        "scripts/Run-V5LockedSelectedTestAndPackage.ps1",
        "src/pancreas_multitask/classification_rescue.py",
        "src/pancreas_multitask/inference_determinism.py",
        "src/pancreas_multitask/network.py",
        "src/pancreas_multitask/predictor.py",
        "src/pancreas_multitask/case_features.py",
        "src/pancreas_multitask/case_feature_extractor.py",
        "src/pancreas_multitask/neural_case_head.py",
        "src/pancreas_multitask/neural_case_bundle.py",
        "src/pancreas_multitask/neural_case_training.py",
        "src/pancreas_multitask/neural_case_predictor.py",
    ]
    lock = {
        "schema_version": 1,
        "status": (
            "locked_before_single_v5_official_reevaluation_and_"
            "selected_v5_test_inference"
        ),
        "candidate_family": "assignment_conforming_v5_neural_case_head",
        "eligibility_scope": "best_of_two_locked_neural_heads",
        "development_boundary": {
            "v5_head_architecture_training_selection_and_offsets_used_"
            "official_validation_or_test_data": False,
            "v5_speed_development_used_official_validation_or_test_data": False,
            "baseline_official_validation_observed_before_v5_extension": True,
            "frozen_checkpoint_was_validation_selected": True,
            "baseline_test_inference_and_packaging_occurred_before_v5_extension": True,
            "baseline_test_inputs_were_a_v5_tuning_signal": False,
        },
        "artifacts": {
            "checkpoint": {
                "name": "checkpoint_classification_rescue.pth",
                "sha256": FIXED_HASHES["checkpoint"],
            },
            "neural_case_head_bundle": {"sha256": LOCAL_BUNDLE_SHA256},
            "numeric_train_dataset": {"sha256": LOCAL_NUMERIC_DATASET_SHA256},
            "plans_json": {"sha256": FIXED_HASHES["plans"]},
            "dataset_json": {"sha256": FIXED_HASHES["dataset"]},
        },
        "frozen_components": {
            "encoder": FIXED_HASHES["encoder"],
            "decoder": FIXED_HASHES["decoder"],
            "classification": FIXED_HASHES["classification"],
        },
        "inference_contract": {
            "fold": 0,
            "classification_mode": "neural-v5",
            "v5_extraction_mode": "neural_only",
            "device": "cuda",
            "tile_step_size": 0.5,
            "tile_batch_size": 1,
            "tta_batch_size": 1,
            "tta_enabled": True,
            "gaussian_enabled": True,
            "overwrite": True,
            "results_on_cpu": False,
            "deterministic_execution": True,
            "autocast_cuda_float16": True,
        },
        "protocol_locks": {
            "neural_case_head": {
                "path": "configs/phd_neural_case_head_lock_v5.json",
                "sha256": (
                    "a8c2147493718acc96e4aa5dc471bf3f3277f0b99e8a8f7620bf966ab7b70d11"
                ),
            },
            "neural_decision": {
                "path": "configs/phd_neural_decision_lock_v5.json",
                "sha256": (
                    "e28a303c7d3da5dc7857ecc72787b6746d1e689e83167c500d4d2823c5ea540f"
                ),
            },
            "inference_speed": {
                "path": "configs/inference_speed_benchmark_v3.json",
                "sha256": (
                    "3a57ab79147a6dd9ab4ee3fa99fdb2be978e9c60f290cead7a52298673e926aa"
                ),
            },
            "inference_speed_stock_gate": {
                "path": "configs/inference_speed_stock_gate_v1.json",
                "sha256": (
                    "563d9d5e4fbe0f92653c6b7295c476d0ddf5d239c47beb1948410bbb80a7c2e2"
                ),
            },
            "inference_determinism": {
                "path": "configs/inference_determinism_conformance_v1.json",
                "sha256": (
                    "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd"
                ),
            },
            "stock_export_conformance": {
                "path": "configs/inference_stock_export_conformance_v1.json",
                "sha256": (
                    "bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503"
                ),
            },
        },
        "implementation_files": [
            {
                "path": relative,
                "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
            }
            for relative in implementation_paths
        ],
        "train_only_audits": {
            "fit": {
                "filename": "neural_case_head_fit_audit.json",
                "sha256": (
                    "16831b1249de84e6f89391903cd9510f7a21e1fa85389d3ca33b76fbb70c7274"
                ),
            },
            "selection": {
                "filename": "neural_case_head_selection.json",
                "sha256": (
                    "a7f397ea0bf86c551b13692e5e4c329999573e6b398486046e9392639304444c"
                ),
            },
            "decision": {
                "filename": "neural_decision_calibration.json",
                "sha256": (
                    "87233541474dcd3226866912c7968c05b1ba4b855ed6075239763bfac9c6bcf0"
                ),
            },
            "refit": {
                "filename": "neural_case_head_refit.json",
                "sha256": (
                    "b2bb4adfebe4bc01edfb8e676c2c8894a0b7b328a39bf29c24f57b5ffc33e3f0"
                ),
            },
        },
        "run_ledger_files": {
            "official_validation": "official_validation_run_consumed.json",
            "selected_test": "selected_test_run_consumed.json",
        },
    }
    lock_path = tmp_path / "valid-final-candidate-lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    environment = os.environ.copy()
    environment.update(
        {
            "V5_COMMON": str(COMMON),
            "V5_LOCK": str(lock_path),
            "V5_LOCK_SHA": lock_hash,
            "V5_ROOT": str(ROOT),
            "V5_MODEL": str(LOCAL_MODEL),
            "V5_BUNDLE": str(LOCAL_BUNDLE),
        }
    )
    harness = rf"""
. $env:V5_COMMON
$candidate = Assert-V5FinalCandidateLock `
    -FinalCandidateLock $env:V5_LOCK `
    -ExpectedFinalCandidateLockSha256 $env:V5_LOCK_SHA `
    -ModelDirectory $env:V5_MODEL `
    -NeuralCaseHeadBundle $env:V5_BUNDLE `
    -ExpectedCheckpointSha256 '{FIXED_HASHES['checkpoint']}' `
    -ExpectedNeuralCaseHeadBundleSha256 '{LOCAL_BUNDLE_SHA256}' `
    -ExpectedNumericTrainDatasetSha256 '{LOCAL_NUMERIC_DATASET_SHA256}' `
    -ExpectedPlansSha256 '{FIXED_HASHES['plans']}' `
    -ExpectedDatasetJsonSha256 '{FIXED_HASHES['dataset']}' `
    -ExpectedEncoderComponentSha256 '{FIXED_HASHES['encoder']}' `
    -ExpectedDecoderComponentSha256 '{FIXED_HASHES['decoder']}' `
    -ExpectedClassificationComponentSha256 '{FIXED_HASHES['classification']}' `
    -ProjectRoot $env:V5_ROOT
[pscustomobject]@{{
    lock_sha256 = $candidate.LockSha256
    selected_candidate_id = $candidate.TrainOnlyAudits.SelectedCandidateId
    refit_state_sha256 = $candidate.TrainOnlyAudits.RefitStateSha256
}} | ConvertTo-Json
"""
    completed = subprocess.run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            harness,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result["lock_sha256"] == lock_hash
    assert result["selected_candidate_id"] == "neural_two_query_cross_attention_mil"
    assert result["refit_state_sha256"] == (
        "7954be6f9620f77dc80365df97bf374b84e976ad5df12f3dc4ea4acc34892e3f"
    )
