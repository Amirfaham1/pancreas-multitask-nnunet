<#
.SYNOPSIS
Run the one locked post-hoc v5 official-validation reevaluation.

.DESCRIPTION
The entry point fails closed unless a caller-supplied final-candidate lock and
every bound artifact match exact hashes. It then consumes a persistent,
lock-bound one-use ledger and runs one fresh, label-blind 36-case inference.
All masks, subtype decisions, probability details, and runtime evidence are
hashed into a pre-reference manifest before this script checks or opens either
reference path. The saved predictions are evaluated exactly once. There is no
candidate loop, selection, refit, threshold change, or second inference path.

This is a locked post-hoc reevaluation because the earlier baseline validation
result was already known.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $FinalCandidateLock,
    [Parameter(Mandatory)]
    [string] $ExpectedFinalCandidateLockSha256,
    [Parameter(Mandatory)]
    [string] $ModelDirectory,
    [Parameter(Mandatory)]
    [string] $NeuralCaseHeadBundle,
    [Parameter(Mandatory)]
    [string] $ExpectedCheckpointSha256,
    [Parameter(Mandatory)]
    [string] $ExpectedNeuralCaseHeadBundleSha256,
    [Parameter(Mandatory)]
    [string] $ExpectedNumericTrainDatasetSha256,
    [Parameter(Mandatory)]
    [string] $ExpectedPlansSha256,
    [Parameter(Mandatory)]
    [string] $ExpectedDatasetJsonSha256,
    [Parameter(Mandatory)]
    [string] $ExpectedEncoderComponentSha256,
    [Parameter(Mandatory)]
    [string] $ExpectedDecoderComponentSha256,
    [Parameter(Mandatory)]
    [string] $ExpectedClassificationComponentSha256,
    [Parameter(Mandatory)]
    [string] $ValidationImages,
    [Parameter(Mandatory)]
    [string] $ReferenceMasks,
    [Parameter(Mandatory)]
    [string] $ReferenceSubtypes,
    [Parameter(Mandatory)]
    [string] $OutputRoot,
    [string] $WorkRoot = "D:\MLQuizWork",
    [string] $PythonExecutable = "D:\MLQuizWork\.venv\Scripts\python.exe",
    [ValidateSet("cuda")]
    [string] $Device = "cuda",
    [string] $ImmutableBaselineRoot = "D:\MLQuizWork\baseline_20260806_509cbe2"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$commonScript = Join-Path $PSScriptRoot "V5-LockedDeliveryCommon.ps1"
if (-not (Test-Path -LiteralPath $commonScript -PathType Leaf)) {
    throw "Required v5 delivery helper was not found: '$commonScript'."
}
. $commonScript

$predictionScript = Join-Path $PSScriptRoot "predict_joint.py"
$evaluationScript = Join-Path $PSScriptRoot "evaluate_predictions.py"
$setupScript = Join-Path $PSScriptRoot "Set-QuizEnvironment.ps1"
foreach ($required in @($predictionScript, $evaluationScript, $setupScript, $PythonExecutable)) {
    Assert-V5LeafFile $required "Required executable or script"
}

$resolvedOutputRoot = Assert-V5NewSeparatedOutputRoot `
    -OutputRoot $OutputRoot `
    -ProtectedPaths @(
        $projectRoot,
        $ModelDirectory,
        $NeuralCaseHeadBundle,
        $FinalCandidateLock,
        $ValidationImages,
        $ReferenceMasks,
        $ReferenceSubtypes,
        $ImmutableBaselineRoot
    )

$candidateLockArguments = @{
    FinalCandidateLock = $FinalCandidateLock
    ExpectedFinalCandidateLockSha256 = $ExpectedFinalCandidateLockSha256
    ModelDirectory = $ModelDirectory
    NeuralCaseHeadBundle = $NeuralCaseHeadBundle
    ExpectedCheckpointSha256 = $ExpectedCheckpointSha256
    ExpectedNeuralCaseHeadBundleSha256 = $ExpectedNeuralCaseHeadBundleSha256
    ExpectedNumericTrainDatasetSha256 = $ExpectedNumericTrainDatasetSha256
    ExpectedPlansSha256 = $ExpectedPlansSha256
    ExpectedDatasetJsonSha256 = $ExpectedDatasetJsonSha256
    ExpectedEncoderComponentSha256 = $ExpectedEncoderComponentSha256
    ExpectedDecoderComponentSha256 = $ExpectedDecoderComponentSha256
    ExpectedClassificationComponentSha256 = $ExpectedClassificationComponentSha256
    ProjectRoot = $projectRoot
}
$candidate = Assert-V5FinalCandidateLock @candidateLockArguments

# Environment setup and the CPU strict-load preflight only touch code, model,
# bundle, and train-derived provenance. This v5 wrapper has not opened an
# official path at this point; historical baseline access is disclosed above.
$null = . $setupScript -WorkRoot $WorkRoot -WandbMode disabled -DataAugmentationProcesses 1
$cpuPreflight = Invoke-V5StrictCpuPreflight `
    -PythonExecutable $PythonExecutable `
    -ProjectRoot $projectRoot `
    -Candidate $candidate `
    -ExpectedEncoderComponentSha256 $ExpectedEncoderComponentSha256 `
    -ExpectedDecoderComponentSha256 $ExpectedDecoderComponentSha256 `
    -ExpectedClassificationComponentSha256 $ExpectedClassificationComponentSha256

$ledgerPath = Get-V5BareLedgerPath `
    -Lock $candidate.Lock `
    -Stage official_validation `
    -FinalCandidateLock $candidate.LockPath
$mutex = $null
$ledgerCreated = $false
$ledgerStartedAt = [DateTime]::UtcNow.ToString("o")
$currentStage = "preflight_complete"
try {
    $mutex = Enter-V5NamedMutex (
        "Local\PancreasMultitaskV5Official_" + $candidate.LockSha256.Substring(0, 16)
    )
    # Race-safe recheck immediately before consuming the immutable run.
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "V5 output root appeared after preflight; refusing to continue."
    }
    $ledgerPayload = [ordered]@{
        schema_version = 1
        status = "started_and_consumed"
        stage = "single_locked_post_hoc_official_validation_reevaluation"
        started_at_utc = $ledgerStartedAt
        final_candidate_lock = [ordered]@{
            path = $candidate.LockPath
            sha256 = $candidate.LockSha256
        }
        output_root = $resolvedOutputRoot
        label_blind_prediction_complete = $false
        pre_reference_manifest_written = $false
        reference_access_started = $false
        evaluation_invocation_count = 0
    }
    $ledgerPath = New-V5ExclusiveLedger -Path $ledgerPath -Payload $ledgerPayload
    $ledgerCreated = $true

    $null = New-Item -ItemType Directory -Path $resolvedOutputRoot
    $predictionDirectory = Join-Path $resolvedOutputRoot "predictions"
    $evidenceDirectory = Join-Path $resolvedOutputRoot "evidence"
    $null = New-Item -ItemType Directory -Path $evidenceDirectory
    $classificationCsv = Join-Path $predictionDirectory "subtype_results.csv"
    $probabilityCsv = Join-Path $evidenceDirectory "subtype_probabilities.csv"
    $runtimeJson = Join-Path $evidenceDirectory "runtime.json"
    $preflightJson = Join-Path $evidenceDirectory "candidate_preflight.json"
    $preReferenceManifest = Join-Path $evidenceDirectory "pre_reference_prediction_manifest.json"
    $metricsJson = Join-Path $evidenceDirectory "official_evaluation_metrics.json"
    $caseMetricsCsv = Join-Path $evidenceDirectory "official_evaluation_cases.csv"
    $gateJson = Join-Path $evidenceDirectory "official_evaluation_gates.json"

    Write-V5JsonAtomic -Path $preflightJson -Payload ([ordered]@{
        schema_version = 1
        status = "complete_before_this_v5_wrapper_opened_official_validation_images"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock_sha256 = $candidate.LockSha256
        checkpoint_sha256 = $script:V5CheckpointSha256
        neural_case_head_bundle_sha256 = $candidate.BundleSha256
        numeric_train_dataset_sha256 = $candidate.NumericTrainDatasetSha256
        plans_sha256 = $script:V5PlansSha256
        dataset_json_sha256 = $script:V5DatasetJsonSha256
        stock_export_conformance_lock_sha256 = $script:V5StockExportLockSha256
        cpu_strict_load = $cpuPreflight
        official_validation_targets_opened_by_this_v5_wrapper_before_preflight = $false
    })

    # This is the first official-validation image-path access within this
    # locked v5 wrapper run. Reference masks and subtype labels are intentionally
    # neither tested nor opened here.
    $currentStage = "label_blind_inference"
    Assert-V5Directory $ValidationImages "Official validation image directory"
    $predictionArguments = @(
        "--input", (Get-V5NormalizedFullPath $ValidationImages),
        "--output", $predictionDirectory,
        "--model", $candidate.ModelDirectory,
        "--folds", "0",
        "--checkpoint", $script:V5CheckpointName,
        "--classification-mode", "neural-v5",
        "--neural-case-head-bundle", $candidate.BundlePath,
        "--v5-extraction-mode", "neural_only",
        "--expected-neural-case-head-bundle-sha256", $candidate.BundleSha256,
        "--expected-numeric-train-dataset-sha256", $candidate.NumericTrainDatasetSha256,
        "--classification-csv", $classificationCsv,
        "--probability-csv", $probabilityCsv,
        "--runtime-json", $runtimeJson,
        "--device", $Device,
        "--tile-step-size", "0.5",
        "--tile-batch-size", "1",
        "--tta-batch-size", "1",
        "--overwrite"
    )
    Invoke-V5CheckedPython `
        -PythonExecutable $PythonExecutable `
        -ScriptPath $predictionScript `
        -Arguments $predictionArguments `
        -Stage "One locked label-blind official-validation inference"

    # Re-hash the lock, checkpoint, bundle, configuration, protocol locks, and
    # every implementation file before accepting or hashing model outputs.
    $null = Assert-V5FinalCandidateLock @candidateLockArguments

    # Audit and hash every model output before this wrapper's first
    # reference-path access.
    $currentStage = "hashing_predictions_before_references"
    $runtime = Assert-V5RuntimeArtifact `
        -RuntimeJson $runtimeJson `
        -ExpectedCaseCount 36 `
        -ExpectedInputDirectory $ValidationImages `
        -ExpectedDevice $Device `
        -Candidate $candidate `
        -ExpectedEncoderComponentSha256 $ExpectedEncoderComponentSha256 `
        -ExpectedDecoderComponentSha256 $ExpectedDecoderComponentSha256 `
        -ExpectedClassificationComponentSha256 $ExpectedClassificationComponentSha256
    $artifactSet = Get-V5InferenceArtifactSet `
        -PredictionDirectory $predictionDirectory `
        -ClassificationCsv $classificationCsv `
        -ProbabilityCsv $probabilityCsv `
        -RuntimeJson $runtimeJson `
        -Runtime $runtime `
        -ExpectedCaseCount 36 `
        -OutputRoot $resolvedOutputRoot
    $preflightHash = Get-V5FileSha256 $preflightJson
    $manifestPayload = [ordered]@{
        schema_version = 1
        status = "all_v5_label_blind_predictions_hashed_before_this_wrapper_reference_access"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        method = "single_locked_post_hoc_official_validation_reevaluation"
        final_candidate_lock = [ordered]@{
            path = $candidate.LockPath
            sha256 = $candidate.LockSha256
        }
        inference_contract = [ordered]@{
            fold = 0
            classification_mode = "neural-v5"
            v5_extraction_mode = "neural_only"
            tta_enabled = $true
            gaussian_enabled = $true
            tile_step_size = 0.5
            tile_batch_size = 1
            tta_batch_size = 1
            overwrite = $true
            device = "cuda"
            results_on_cpu = $false
            deterministic_execution = $true
            autocast_cuda_float16 = $true
            segmentation_export_logit_dtype = "torch.float16"
            stock_export_conformance_lock_sha256 = $script:V5StockExportLockSha256
        }
        counts = [ordered]@{
            cases = 36
            masks = $artifactSet.MaskCount
            classification_rows = $artifactSet.ClassificationRowCount
            probability_rows = $artifactSet.ProbabilityRowCount
        }
        candidate_preflight = [ordered]@{
            relative_path = "evidence/candidate_preflight.json"
            sha256 = $preflightHash
        }
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        prediction_artifacts = $artifactSet.Artifacts
        reference_masks_path_tested_or_opened_by_this_wrapper_before_this_manifest = $false
        reference_subtypes_path_tested_or_opened_by_this_wrapper_before_this_manifest = $false
        official_evaluation_invocation_count_before_this_manifest = 0
    }
    Write-V5JsonAtomic -Path $preReferenceManifest -Payload $manifestPayload
    $preReferenceManifestSha256 = Get-V5FileSha256 $preReferenceManifest
    Write-V5JsonAtomic -Path $ledgerPath -Payload ([ordered]@{
        schema_version = 1
        status = "predictions_frozen_before_this_wrapper_reference_access"
        stage = "single_locked_post_hoc_official_validation_reevaluation"
        started_at_utc = $ledgerStartedAt
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock = [ordered]@{
            path = $candidate.LockPath
            sha256 = $candidate.LockSha256
        }
        output_root = $resolvedOutputRoot
        label_blind_prediction_complete = $true
        pre_reference_manifest_written = $true
        pre_reference_manifest_sha256 = $preReferenceManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        reference_access_started = $false
        evaluation_invocation_count = 0
    })

    # The pre-reference manifest is durable and hashed. Only now does this
    # wrapper test/open either target path. evaluate_predictions.py is invoked once.
    $currentStage = "single_target_opening_evaluation"
    Assert-V5Directory $ReferenceMasks "Official reference-mask directory"
    Assert-V5LeafFile $ReferenceSubtypes "Official reference-subtype table"
    Write-V5JsonAtomic -Path $ledgerPath -Payload ([ordered]@{
        schema_version = 1
        status = "single_evaluation_started"
        stage = "single_locked_post_hoc_official_validation_reevaluation"
        started_at_utc = $ledgerStartedAt
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
        output_root = $resolvedOutputRoot
        label_blind_prediction_complete = $true
        pre_reference_manifest_written = $true
        pre_reference_manifest_sha256 = $preReferenceManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        reference_access_started = $true
        evaluation_invocation_count = 1
    })
    Invoke-V5CheckedPython `
        -PythonExecutable $PythonExecutable `
        -ScriptPath $evaluationScript `
        -Arguments @(
            "--predictions", $predictionDirectory,
            "--references", (Get-V5NormalizedFullPath $ReferenceMasks),
            "--classification-predictions", $classificationCsv,
            "--classification-references", (Get-V5NormalizedFullPath $ReferenceSubtypes),
            "--classification-reference-split", "validation",
            "--output-json", $metricsJson,
            "--output-csv", $caseMetricsCsv,
            "--empty-empty-dice", "1.0",
            "--label-tolerance", "0.0",
            "--affine-tolerance", "0.00001",
            "--bootstrap-samples", "2000",
            "--confidence", "0.95",
            "--seed", "12345"
        ) `
        -Stage "The single saved-output official evaluation"

    $currentStage = "gate_recording"
    $null = Assert-V5FinalCandidateLock @candidateLockArguments
    Assert-V5HashEquals `
        (Get-V5FileSha256 $preReferenceManifest) `
        $preReferenceManifestSha256 `
        "Pre-reference manifest after evaluation"
    $postEvaluationArtifactSet = Get-V5InferenceArtifactSet `
        -PredictionDirectory $predictionDirectory `
        -ClassificationCsv $classificationCsv `
        -ProbabilityCsv $probabilityCsv `
        -RuntimeJson $runtimeJson `
        -Runtime $runtime `
        -ExpectedCaseCount 36 `
        -OutputRoot $resolvedOutputRoot
    Assert-V5HashEquals `
        $postEvaluationArtifactSet.ArtifactSetSha256 `
        $artifactSet.ArtifactSetSha256 `
        "Prediction artifact set after evaluation"
    $metrics = Read-V5JsonObject $metricsJson "Official evaluation metrics"
    if ([int] (Get-V5RequiredProperty $metrics "case_count" "Evaluation metrics") -ne 36) {
        throw "Official evaluation must contain exactly 36 merged cases."
    }
    $segmentation = Get-V5RequiredProperty $metrics "segmentation" "Evaluation metrics"
    $classification = Get-V5RequiredProperty $metrics "classification" "Evaluation metrics"
    if ([int] (Get-V5RequiredProperty $segmentation "case_count" "Segmentation metrics") -ne 36 -or
        [int] (Get-V5RequiredProperty $classification "case_count" "Classification metrics") -ne 36) {
        throw "Both official tasks must contain exactly 36 evaluated cases."
    }
    $wholeSummary = Get-V5RequiredProperty $segmentation "whole_pancreas_dice" "Segmentation metrics"
    $lesionSummary = Get-V5RequiredProperty $segmentation "lesion_dice" "Segmentation metrics"
    $wholeDice = [double] (Get-V5RequiredProperty $wholeSummary "mean" "Whole-pancreas metrics")
    $lesionDice = [double] (Get-V5RequiredProperty $lesionSummary "mean" "Lesion metrics")
    $macroF1 = [double] (Get-V5RequiredProperty $classification "macro_f1" "Classification metrics")
    foreach ($metric in @($wholeDice, $lesionDice, $macroF1)) {
        if ([double]::IsNaN($metric) -or [double]::IsInfinity($metric) -or $metric -lt 0.0 -or $metric -gt 1.0) {
            throw "Official evaluation emitted a non-finite or out-of-range metric."
        }
    }
    $strictImprovement = $macroF1 -gt $script:V5BaselineMacroF1
    $wholeGate = $wholeDice -ge 0.91
    $lesionGate = $lesionDice -ge 0.31
    $phdClassificationGate = $macroF1 -ge 0.70
    $gatePayload = [ordered]@{
        schema_version = 1
        status = "complete_no_second_classifier_iteration_permitted"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        evaluation_scope = "single_locked_post_hoc_official_validation_reevaluation"
        validation_output_root = $resolvedOutputRoot
        final_candidate_lock_sha256 = $candidate.LockSha256
        pre_reference_prediction_manifest_sha256 = $preReferenceManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        evaluation_metrics = [ordered]@{
            path = "evidence/official_evaluation_metrics.json"
            sha256 = Get-V5FileSha256 $metricsJson
        }
        evaluation_cases = [ordered]@{
            path = "evidence/official_evaluation_cases.csv"
            sha256 = Get-V5FileSha256 $caseMetricsCsv
        }
        metrics = [ordered]@{
            whole_pancreas_dice = $wholeDice
            lesion_dice = $lesionDice
            macro_f1 = $macroF1
        }
        thresholds = [ordered]@{
            baseline_macro_f1_strictly_greater_than = $script:V5BaselineMacroF1
            phd_macro_f1_at_least = 0.70
            whole_pancreas_dice_at_least = 0.91
            lesion_dice_at_least = 0.31
        }
        verdicts = [ordered]@{
            strict_macro_f1_improvement_over_baseline = $strictImprovement
            phd_macro_f1_gate = $phdClassificationGate
            whole_pancreas_gate = $wholeGate
            lesion_gate = $lesionGate
            phd_joint_metric_gate = ($wholeGate -and $lesionGate -and $phdClassificationGate)
            classifier_replacement_validation_gate = ($strictImprovement -and $wholeGate -and $lesionGate)
        }
        official_inference_invocation_count = 1
        official_evaluation_invocation_count = 1
        further_classifier_training_selection_or_official_evaluation_permitted = $false
    }
    Write-V5JsonAtomic -Path $gateJson -Payload $gatePayload
    $gateSha256 = Get-V5FileSha256 $gateJson
    Write-V5JsonAtomic -Path $ledgerPath -Payload ([ordered]@{
        schema_version = 1
        status = "complete_and_consumed"
        stage = "single_locked_post_hoc_official_validation_reevaluation"
        started_at_utc = $ledgerStartedAt
        completed_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
        output_root = $resolvedOutputRoot
        label_blind_prediction_complete = $true
        pre_reference_manifest_written = $true
        pre_reference_manifest_sha256 = $preReferenceManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        reference_access_started = $true
        evaluation_invocation_count = 1
        gate_artifact_sha256 = $gateSha256
        no_second_classifier_iteration_permitted = $true
    })
    $currentStage = "complete"

    Write-Host "Locked post-hoc official-validation reevaluation complete."
    Write-Host "Pre-reference manifest: $preReferenceManifest"
    Write-Host "Gate artifact: $gateJson"
    Write-Host ("Macro-F1: {0:R}; strict baseline improvement: {1}" -f $macroF1, $strictImprovement)
    Write-Host ("Whole Dice: {0:R}; lesion Dice: {1:R}; PhD joint metric gate: {2}" -f $wholeDice, $lesionDice, ($wholeGate -and $lesionGate -and $phdClassificationGate))
}
catch {
    $failure = $_
    if ($ledgerCreated) {
        try {
            Write-V5JsonAtomic -Path $ledgerPath -Payload ([ordered]@{
                schema_version = 1
                status = "failed_and_consumed_no_rerun"
                stage = "single_locked_post_hoc_official_validation_reevaluation"
                failed_stage = $currentStage
                started_at_utc = $ledgerStartedAt
                failed_at_utc = [DateTime]::UtcNow.ToString("o")
                final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
                output_root = $resolvedOutputRoot
                error_type = $failure.Exception.GetType().FullName
                error_message = $failure.Exception.Message
                one_use_run_remains_consumed = $true
            })
        }
        catch {
            Write-Warning "Could not update the already-consumed failure ledger: $($_.Exception.Message)"
        }
    }
    throw $failure
}
finally {
    Exit-V5NamedMutex $mutex
}
