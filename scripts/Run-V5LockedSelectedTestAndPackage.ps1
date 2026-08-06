<#
.SYNOPSIS
Run one locked v5 test inference and build the validated 72-case ZIP.

.DESCRIPTION
This entry point accepts only the exact final candidate that passed the
strict-improvement validation gate. It verifies the lock/artifact/code chain,
strict-loads the model and neural head before touching test inputs, consumes a
persistent one-use ledger, runs one fresh 72-case neural_only inference, hashes
all outputs, and delegates flat ZIP creation and validation to the existing
Package-Submission.ps1 and validate_submission.py pipeline.

No input, baseline, existing output, or existing delivery is replaced.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $FinalCandidateLock,
    [Parameter(Mandatory)]
    [string] $ExpectedFinalCandidateLockSha256,
    [Parameter(Mandatory)]
    [string] $OfficialEvaluationGate,
    [Parameter(Mandatory)]
    [string] $ExpectedOfficialEvaluationGateSha256,
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
    [string] $TestImages,
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
$packageScript = Join-Path $PSScriptRoot "Package-Submission.ps1"
$validatorScript = Join-Path $PSScriptRoot "validate_submission.py"
$setupScript = Join-Path $PSScriptRoot "Set-QuizEnvironment.ps1"
foreach ($required in @(
    $predictionScript,
    $packageScript,
    $validatorScript,
    $setupScript,
    $PythonExecutable,
    $OfficialEvaluationGate
)) {
    Assert-V5LeafFile $required "Required executable, script, or gate artifact"
}

$resolvedOutputRoot = Assert-V5NewSeparatedOutputRoot `
    -OutputRoot $OutputRoot `
    -ProtectedPaths @(
        $projectRoot,
        $ModelDirectory,
        $NeuralCaseHeadBundle,
        $FinalCandidateLock,
        $OfficialEvaluationGate,
        $TestImages,
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

$expectedGateHash = ConvertTo-V5Sha256 $ExpectedOfficialEvaluationGateSha256 "Official gate hash"
$resolvedGate = Get-V5NormalizedFullPath $OfficialEvaluationGate
Assert-V5HashEquals (Get-V5FileSha256 $resolvedGate) $expectedGateHash "Official evaluation gate file"
$officialGate = Read-V5JsonObject $resolvedGate "Official evaluation gate"
if ([int] (Get-V5RequiredProperty $officialGate "schema_version" "Official gate") -ne 1) {
    throw "Official evaluation gate schema_version must be 1."
}
Assert-V5ExactValue `
    (Get-V5RequiredProperty $officialGate "status" "Official gate") `
    "complete_no_second_classifier_iteration_permitted" `
    "Official evaluation gate status"
Assert-V5ExactValue `
    (Get-V5RequiredProperty $officialGate "evaluation_scope" "Official gate") `
    "single_locked_post_hoc_official_validation_reevaluation" `
    "Official evaluation scope"
Assert-V5HashEquals `
    (Get-V5RequiredProperty $officialGate "final_candidate_lock_sha256" "Official gate") `
    $candidate.LockSha256 `
    "Official gate candidate binding"
if ([int] (Get-V5RequiredProperty $officialGate "official_inference_invocation_count" "Official gate") -ne 1 -or
    [int] (Get-V5RequiredProperty $officialGate "official_evaluation_invocation_count" "Official gate") -ne 1) {
    throw "Official gate must record exactly one inference and one evaluation."
}
Assert-V5Boolean `
    (Get-V5RequiredProperty $officialGate "further_classifier_training_selection_or_official_evaluation_permitted" "Official gate") `
    $false `
    "Official gate further-iteration policy"
$verdicts = Get-V5RequiredProperty $officialGate "verdicts" "Official gate"
$gateMetrics = Get-V5RequiredProperty $officialGate "metrics" "Official gate"
$gateThresholds = Get-V5RequiredProperty $officialGate "thresholds" "Official gate"
$gateMacroF1 = [double] (Get-V5RequiredProperty $gateMetrics "macro_f1" "Official gate metrics")
$gateWholeDice = [double] (Get-V5RequiredProperty $gateMetrics "whole_pancreas_dice" "Official gate metrics")
$gateLesionDice = [double] (Get-V5RequiredProperty $gateMetrics "lesion_dice" "Official gate metrics")
foreach ($metric in @($gateMacroF1, $gateWholeDice, $gateLesionDice)) {
    if ([double]::IsNaN($metric) -or [double]::IsInfinity($metric) -or $metric -lt 0.0 -or $metric -gt 1.0) {
        throw "Official gate contains a non-finite or out-of-range metric."
    }
}
if ([double] (Get-V5RequiredProperty $gateThresholds "baseline_macro_f1_strictly_greater_than" "Official gate thresholds") -ne $script:V5BaselineMacroF1 -or
    [double] (Get-V5RequiredProperty $gateThresholds "phd_macro_f1_at_least" "Official gate thresholds") -ne 0.70 -or
    [double] (Get-V5RequiredProperty $gateThresholds "whole_pancreas_dice_at_least" "Official gate thresholds") -ne 0.91 -or
    [double] (Get-V5RequiredProperty $gateThresholds "lesion_dice_at_least" "Official gate thresholds") -ne 0.31) {
    throw "Official gate thresholds differ from the frozen protocol."
}
$expectedVerdicts = [ordered]@{
    strict_macro_f1_improvement_over_baseline = ($gateMacroF1 -gt $script:V5BaselineMacroF1)
    phd_macro_f1_gate = ($gateMacroF1 -ge 0.70)
    whole_pancreas_gate = ($gateWholeDice -ge 0.91)
    lesion_gate = ($gateLesionDice -ge 0.31)
    phd_joint_metric_gate = (
        $gateMacroF1 -ge 0.70 -and $gateWholeDice -ge 0.91 -and $gateLesionDice -ge 0.31
    )
    classifier_replacement_validation_gate = (
        $gateMacroF1 -gt $script:V5BaselineMacroF1 -and
        $gateWholeDice -ge 0.91 -and
        $gateLesionDice -ge 0.31
    )
}
foreach ($field in $expectedVerdicts.Keys) {
    Assert-V5Boolean `
        (Get-V5RequiredProperty $verdicts $field "Official gate verdicts") `
        ([bool] $expectedVerdicts[$field]) `
        "Official gate recomputed '$field'"
}
foreach ($requiredPassingField in @(
    "strict_macro_f1_improvement_over_baseline",
    "whole_pancreas_gate",
    "lesion_gate",
    "classifier_replacement_validation_gate"
)) {
    Assert-V5Boolean (Get-V5RequiredProperty $verdicts $requiredPassingField "Official gate verdicts") $true "Official gate '$requiredPassingField'"
}

$validationOutputRoot = Get-V5NormalizedFullPath ([string] (
    Get-V5RequiredProperty $officialGate "validation_output_root" "Official gate"
))
$expectedGatePath = Join-Path $validationOutputRoot "evidence\official_evaluation_gates.json"
if (-not (Get-V5NormalizedFullPath $expectedGatePath).Equals(
    $resolvedGate,
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw "Official gate must be the exact gate artifact in its recorded validation output root."
}
$resolvedOutputRoot = Assert-V5NewSeparatedOutputRoot `
    -OutputRoot $resolvedOutputRoot `
    -ProtectedPaths @($validationOutputRoot)
$lockedPreReferenceHash = ConvertTo-V5Sha256 `
    (Get-V5RequiredProperty $officialGate "pre_reference_prediction_manifest_sha256" "Official gate") `
    "Official pre-reference manifest hash"
$lockedPredictionSetHash = ConvertTo-V5Sha256 `
    (Get-V5RequiredProperty $officialGate "prediction_artifact_set_sha256" "Official gate") `
    "Official prediction-artifact-set hash"
$officialPreReferenceManifest = Join-Path $validationOutputRoot "evidence\pre_reference_prediction_manifest.json"
Assert-V5HashEquals (Get-V5FileSha256 $officialPreReferenceManifest) $lockedPreReferenceHash "Official pre-reference manifest file"
$officialPreReference = Read-V5JsonObject $officialPreReferenceManifest "Official pre-reference manifest"
Assert-V5ExactValue `
    (Get-V5RequiredProperty $officialPreReference "status" "Official pre-reference manifest") `
    "all_v5_label_blind_predictions_hashed_before_this_wrapper_reference_access" `
    "Official pre-reference manifest status"
Assert-V5HashEquals `
    (Get-V5RequiredProperty (Get-V5RequiredProperty $officialPreReference "final_candidate_lock" "Official pre-reference manifest") "sha256" "Official pre-reference candidate") `
    $candidate.LockSha256 `
    "Official pre-reference candidate binding"
Assert-V5HashEquals `
    (Get-V5RequiredProperty $officialPreReference "prediction_artifact_set_sha256" "Official pre-reference manifest") `
    $lockedPredictionSetHash `
    "Official pre-reference prediction-set binding"
$preflightBinding = Get-V5RequiredProperty $officialPreReference "candidate_preflight" "Official pre-reference manifest"
Assert-V5ExactValue `
    (Get-V5RequiredProperty $preflightBinding "relative_path" "Official candidate-preflight binding") `
    "evidence/candidate_preflight.json" `
    "Official candidate-preflight path"
$officialPreflightPath = Join-Path $validationOutputRoot "evidence\candidate_preflight.json"
Assert-V5HashEquals `
    (Get-V5FileSha256 $officialPreflightPath) `
    (Get-V5RequiredProperty $preflightBinding "sha256" "Official candidate-preflight binding") `
    "Official candidate-preflight file"
$officialArtifactAudit = Assert-V5RecordedInferenceArtifactSet `
    -Manifest $officialPreReference `
    -OutputRoot $validationOutputRoot `
    -ExpectedArtifactSetSha256 $lockedPredictionSetHash `
    -ExpectedCaseCount 36

$boundEvidencePaths = @{}
$expectedEvidenceRelativePaths = [ordered]@{
    evaluation_metrics = "evidence/official_evaluation_metrics.json"
    evaluation_cases = "evidence/official_evaluation_cases.csv"
}
foreach ($entryName in $expectedEvidenceRelativePaths.Keys) {
    $entry = Get-V5RequiredProperty $officialGate $entryName "Official gate"
    $relative = [string] (Get-V5RequiredProperty $entry "path" "Official gate '$entryName'")
    Assert-V5ExactValue $relative $expectedEvidenceRelativePaths[$entryName] "Official gate '$entryName' path"
    $evidencePath = Join-Path $validationOutputRoot ($relative.Replace("/", [IO.Path]::DirectorySeparatorChar))
    if (-not (Test-V5PathAtOrBelow -Candidate $evidencePath -Parent $validationOutputRoot)) {
        throw "Official gate evidence path escapes its validation output root."
    }
    Assert-V5HashEquals `
        (Get-V5FileSha256 $evidencePath) `
        (Get-V5RequiredProperty $entry "sha256" "Official gate '$entryName'") `
        "Official gate '$entryName' file"
    $boundEvidencePaths[$entryName] = $evidencePath
}

$boundMetrics = Read-V5JsonObject $boundEvidencePaths["evaluation_metrics"] "Bound official evaluation metrics"
if ([int] (Get-V5RequiredProperty $boundMetrics "schema_version" "Bound evaluation metrics") -ne 1 -or
    [int] (Get-V5RequiredProperty $boundMetrics "case_count" "Bound evaluation metrics") -ne 36) {
    throw "Bound official evaluation metrics must use schema 1 and contain exactly 36 cases."
}
$boundSegmentation = Get-V5RequiredProperty $boundMetrics "segmentation" "Bound evaluation metrics"
$boundClassification = Get-V5RequiredProperty $boundMetrics "classification" "Bound evaluation metrics"
if ([int] (Get-V5RequiredProperty $boundSegmentation "case_count" "Bound segmentation metrics") -ne 36 -or
    [int] (Get-V5RequiredProperty $boundClassification "case_count" "Bound classification metrics") -ne 36) {
    throw "Bound official segmentation and classification metrics must each contain 36 cases."
}
$boundWholeDice = [double] (Get-V5RequiredProperty (
    Get-V5RequiredProperty $boundSegmentation "whole_pancreas_dice" "Bound segmentation metrics"
) "mean" "Bound whole-pancreas metrics")
$boundLesionDice = [double] (Get-V5RequiredProperty (
    Get-V5RequiredProperty $boundSegmentation "lesion_dice" "Bound segmentation metrics"
) "mean" "Bound lesion metrics")
$boundMacroF1 = [double] (Get-V5RequiredProperty $boundClassification "macro_f1" "Bound classification metrics")
foreach ($metric in @($boundWholeDice, $boundLesionDice, $boundMacroF1)) {
    if ([double]::IsNaN($metric) -or [double]::IsInfinity($metric) -or $metric -lt 0.0 -or $metric -gt 1.0) {
        throw "Bound official evaluation contains a non-finite or out-of-range metric."
    }
}
if ($boundWholeDice -ne $gateWholeDice -or
    $boundLesionDice -ne $gateLesionDice -or
    $boundMacroF1 -ne $gateMacroF1) {
    throw "Official gate metrics differ from the independently read, hash-bound evaluator output."
}
$boundMetricCases = @(Get-V5RequiredProperty $boundMetrics "cases" "Bound evaluation metrics")
$boundCaseRows = @(Import-Csv -LiteralPath $boundEvidencePaths["evaluation_cases"])
if ($boundMetricCases.Count -ne 36 -or $boundCaseRows.Count -ne 36) {
    throw "Bound evaluator JSON and CSV must each contain exactly 36 case records."
}
$boundJsonCaseIds = @($boundMetricCases | ForEach-Object {
    [string] (Get-V5RequiredProperty $_ "case_id" "Bound evaluator JSON case")
})
$boundCsvCaseIds = @($boundCaseRows | ForEach-Object {
    [string] (Get-V5RequiredProperty $_ "case_id" "Bound evaluator CSV case")
})
if ($boundJsonCaseIds -contains "" -or $boundCsvCaseIds -contains "" -or
    @($boundJsonCaseIds | Sort-Object -Unique).Count -ne 36 -or
    @($boundCsvCaseIds | Sort-Object -Unique).Count -ne 36 -or
    @(Compare-Object ($boundJsonCaseIds | Sort-Object) ($boundCsvCaseIds | Sort-Object)).Count -ne 0) {
    throw "Bound evaluator JSON and CSV must contain the same 36 non-empty unique case_id values."
}

$officialLedgerPath = Get-V5BareLedgerPath `
    -Lock $candidate.Lock `
    -Stage official_validation `
    -FinalCandidateLock $candidate.LockPath
$officialLedgerSha256 = Get-V5FileSha256 $officialLedgerPath
$officialLedger = Read-V5JsonObject $officialLedgerPath "Consumed official-validation ledger"
if ([int] (Get-V5RequiredProperty $officialLedger "schema_version" "Official-validation ledger") -ne 1) {
    throw "Official-validation ledger schema_version must be 1."
}
Assert-V5ExactValue `
    (Get-V5RequiredProperty $officialLedger "status" "Official-validation ledger") `
    "complete_and_consumed" `
    "Official-validation ledger status"
Assert-V5ExactValue `
    (Get-V5RequiredProperty $officialLedger "stage" "Official-validation ledger") `
    "single_locked_post_hoc_official_validation_reevaluation" `
    "Official-validation ledger stage"
Assert-V5HashEquals `
    (Get-V5RequiredProperty (Get-V5RequiredProperty $officialLedger "final_candidate_lock" "Official-validation ledger") "sha256" "Official-validation ledger candidate") `
    $candidate.LockSha256 `
    "Official-validation ledger candidate binding"
if (-not (Get-V5NormalizedFullPath ([string] (
    Get-V5RequiredProperty $officialLedger "output_root" "Official-validation ledger"
))).Equals($validationOutputRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Official-validation ledger records a different validation output root."
}
Assert-V5HashEquals `
    (Get-V5RequiredProperty $officialLedger "pre_reference_manifest_sha256" "Official-validation ledger") `
    $lockedPreReferenceHash `
    "Official-validation ledger pre-reference binding"
Assert-V5HashEquals `
    (Get-V5RequiredProperty $officialLedger "prediction_artifact_set_sha256" "Official-validation ledger") `
    $lockedPredictionSetHash `
    "Official-validation ledger prediction-set binding"
Assert-V5HashEquals `
    (Get-V5RequiredProperty $officialLedger "gate_artifact_sha256" "Official-validation ledger") `
    $expectedGateHash `
    "Official-validation ledger gate binding"
Assert-V5Boolean (Get-V5RequiredProperty $officialLedger "label_blind_prediction_complete" "Official-validation ledger") $true "Official-validation inference completion"
Assert-V5Boolean (Get-V5RequiredProperty $officialLedger "pre_reference_manifest_written" "Official-validation ledger") $true "Official-validation pre-reference completion"
Assert-V5Boolean (Get-V5RequiredProperty $officialLedger "reference_access_started" "Official-validation ledger") $true "Official-validation reference-access record"
Assert-V5Boolean (Get-V5RequiredProperty $officialLedger "no_second_classifier_iteration_permitted" "Official-validation ledger") $true "Official-validation no-second-iteration policy"
if ([int] (Get-V5RequiredProperty $officialLedger "evaluation_invocation_count" "Official-validation ledger") -ne 1) {
    throw "Official-validation ledger must record exactly one evaluator invocation."
}
Assert-V5HashEquals (Get-V5FileSha256 $resolvedGate) $expectedGateHash "Official gate after evidence-chain verification"
Assert-V5HashEquals (Get-V5FileSha256 $officialLedgerPath) $officialLedgerSha256 "Official-validation ledger after verification"
Assert-V5HashEquals (Get-V5FileSha256 $officialPreReferenceManifest) $lockedPreReferenceHash "Official pre-reference manifest after verification"

# This strict-load preflight uses no test path. It repeats the immutable model
# and bundle checks so test execution cannot rely only on an earlier process.
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
    -Stage selected_test `
    -FinalCandidateLock $candidate.LockPath
$mutex = $null
$ledgerCreated = $false
$ledgerStartedAt = [DateTime]::UtcNow.ToString("o")
$currentStage = "preflight_complete"
try {
    $mutex = Enter-V5NamedMutex (
        "Local\PancreasMultitaskV5SelectedTest_" + $candidate.LockSha256.Substring(0, 16)
    )
    # Race-safe final evidence check before this wrapper consumes the one-use
    # selected-test run. This still opens no selected-v5 test input.
    $null = Assert-V5FinalCandidateLock @candidateLockArguments
    Assert-V5HashEquals (Get-V5FileSha256 $resolvedGate) $expectedGateHash "Official gate immediately before selected-test run"
    Assert-V5HashEquals (Get-V5FileSha256 $officialLedgerPath) $officialLedgerSha256 "Official-validation ledger immediately before selected-test run"
    Assert-V5HashEquals (Get-V5FileSha256 $officialPreReferenceManifest) $lockedPreReferenceHash "Official pre-reference manifest immediately before selected-test run"
    foreach ($entryName in $expectedEvidenceRelativePaths.Keys) {
        $entry = Get-V5RequiredProperty $officialGate $entryName "Official gate"
        Assert-V5HashEquals `
            (Get-V5FileSha256 $boundEvidencePaths[$entryName]) `
            (Get-V5RequiredProperty $entry "sha256" "Official gate '$entryName'") `
            "Official gate '$entryName' immediately before selected-test run"
    }
    $null = Assert-V5RecordedInferenceArtifactSet `
        -Manifest $officialPreReference `
        -OutputRoot $validationOutputRoot `
        -ExpectedArtifactSetSha256 $lockedPredictionSetHash `
        -ExpectedCaseCount 36
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "V5 test output root appeared after preflight; refusing to continue."
    }
    $ledgerPath = New-V5ExclusiveLedger -Path $ledgerPath -Payload ([ordered]@{
        schema_version = 1
        status = "started_and_consumed"
        stage = "single_locked_selected_test_and_package"
        started_at_utc = $ledgerStartedAt
        final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
        official_evaluation_gate = [ordered]@{ path = $resolvedGate; sha256 = $expectedGateHash }
        official_validation_ledger_sha256 = $officialLedgerSha256
        output_root = $resolvedOutputRoot
        selected_test_inference_invocation_count = 0
        package_invocation_count = 0
    })
    $ledgerCreated = $true

    $null = New-Item -ItemType Directory -Path $resolvedOutputRoot
    $predictionDirectory = Join-Path $resolvedOutputRoot "predictions"
    $evidenceDirectory = Join-Path $resolvedOutputRoot "evidence"
    $deliveryDirectory = Join-Path $resolvedOutputRoot "delivery"
    $null = New-Item -ItemType Directory -Path $evidenceDirectory
    $classificationCsv = Join-Path $predictionDirectory "subtype_results.csv"
    $probabilityCsv = Join-Path $evidenceDirectory "subtype_probabilities.csv"
    $runtimeJson = Join-Path $evidenceDirectory "runtime.json"
    $preflightJson = Join-Path $evidenceDirectory "candidate_preflight.json"
    $prePackageManifest = Join-Path $evidenceDirectory "pre_package_prediction_manifest.json"
    $completionManifest = Join-Path $evidenceDirectory "selected_test_package_completion.json"

    Write-V5JsonAtomic -Path $preflightJson -Payload ([ordered]@{
        schema_version = 1
        status = "complete_before_this_wrapper_opened_selected_v5_test_images"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock_sha256 = $candidate.LockSha256
        official_evaluation_gate_sha256 = $expectedGateHash
        checkpoint_sha256 = $script:V5CheckpointSha256
        neural_case_head_bundle_sha256 = $candidate.BundleSha256
        numeric_train_dataset_sha256 = $candidate.NumericTrainDatasetSha256
        stock_export_conformance_lock_sha256 = $script:V5StockExportLockSha256
        cpu_strict_load = $cpuPreflight
        official_validation_evidence = [ordered]@{
            consumed_ledger_sha256 = $officialLedgerSha256
            pre_reference_manifest_sha256 = $lockedPreReferenceHash
            prediction_artifact_set_sha256 = $officialArtifactAudit.ArtifactSetSha256
            prediction_artifacts_rehashed = $officialArtifactAudit.ArtifactCount
        }
        selected_v5_test_images_opened_by_this_wrapper_before_preflight = $false
    })

    # This wrapper's first selected-v5 test-input path access occurs only after
    # every immutable binding and the one-use ledger have passed. Earlier
    # baseline test inference is disclosed in the final-candidate lock.
    $currentStage = "single_selected_test_inference"
    Assert-V5Directory $TestImages "Official test image directory"
    $predictionArguments = @(
        "--input", (Get-V5NormalizedFullPath $TestImages),
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
        -Stage "The single locked selected-test inference"

    $currentStage = "selected_test_output_hashing"
    $null = Assert-V5FinalCandidateLock @candidateLockArguments
    Assert-V5HashEquals (Get-V5FileSha256 $resolvedGate) $expectedGateHash "Official gate after selected-test inference"
    Assert-V5HashEquals (Get-V5FileSha256 $officialLedgerPath) $officialLedgerSha256 "Official-validation ledger after selected-test inference"
    $runtime = Assert-V5RuntimeArtifact `
        -RuntimeJson $runtimeJson `
        -ExpectedCaseCount 72 `
        -ExpectedInputDirectory $TestImages `
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
        -ExpectedCaseCount 72 `
        -OutputRoot $resolvedOutputRoot
    $preflightHash = Get-V5FileSha256 $preflightJson
    Write-V5JsonAtomic -Path $prePackageManifest -Payload ([ordered]@{
        schema_version = 1
        status = "all_selected_test_predictions_hashed_before_packaging"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock_sha256 = $candidate.LockSha256
        official_evaluation_gate_sha256 = $expectedGateHash
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
            cases = 72
            masks = $artifactSet.MaskCount
            classification_rows = $artifactSet.ClassificationRowCount
            probability_rows = $artifactSet.ProbabilityRowCount
        }
        candidate_preflight_sha256 = $preflightHash
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        prediction_artifacts = $artifactSet.Artifacts
        selected_test_inference_invocation_count = 1
        package_invocation_count_before_this_manifest = 0
    })
    $prePackageManifestSha256 = Get-V5FileSha256 $prePackageManifest
    Write-V5JsonAtomic -Path $ledgerPath -Payload ([ordered]@{
        schema_version = 1
        status = "predictions_frozen_before_packaging"
        stage = "single_locked_selected_test_and_package"
        started_at_utc = $ledgerStartedAt
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
        official_evaluation_gate = [ordered]@{ path = $resolvedGate; sha256 = $expectedGateHash }
        output_root = $resolvedOutputRoot
        selected_test_inference_invocation_count = 1
        pre_package_prediction_manifest_sha256 = $prePackageManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        package_invocation_count = 0
    })

    # Package-Submission explicitly validates the source directory, staged ZIP,
    # and committed ZIP using validate_submission.py. Delivery is fresh, so no
    # replacement switch is available or passed.
    $currentStage = "validated_flat_zip_packaging"
    & $packageScript `
        -PredictionDirectory $predictionDirectory `
        -TestImages (Get-V5NormalizedFullPath $TestImages) `
        -DeliveryRoot $deliveryDirectory `
        -PythonExecutable $PythonExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "Existing submission package/validation pipeline failed with exit code $LASTEXITCODE."
    }

    $null = Assert-V5FinalCandidateLock @candidateLockArguments
    Assert-V5HashEquals (Get-V5FileSha256 $resolvedGate) $expectedGateHash "Official gate after packaging"
    Assert-V5HashEquals (Get-V5FileSha256 $officialLedgerPath) $officialLedgerSha256 "Official-validation ledger after packaging"
    Assert-V5HashEquals `
        (Get-V5FileSha256 $prePackageManifest) `
        $prePackageManifestSha256 `
        "Pre-package prediction manifest after packaging"
    $postPackageArtifactSet = Get-V5InferenceArtifactSet `
        -PredictionDirectory $predictionDirectory `
        -ClassificationCsv $classificationCsv `
        -ProbabilityCsv $probabilityCsv `
        -RuntimeJson $runtimeJson `
        -Runtime $runtime `
        -ExpectedCaseCount 72 `
        -OutputRoot $resolvedOutputRoot
    Assert-V5HashEquals `
        $postPackageArtifactSet.ArtifactSetSha256 `
        $artifactSet.ArtifactSetSha256 `
        "Prediction artifact set after packaging"

    $archivePath = Join-Path $deliveryDirectory "Amirfaham_Fallahpour_results.zip"
    $packageManifestPath = Join-Path $deliveryDirectory "package_manifest.json"
    $directoryValidationPath = Join-Path $deliveryDirectory "submission_directory_validation.json"
    $directoryValidationCsv = Join-Path $deliveryDirectory "submission_directory_case_audit.csv"
    $archiveValidationPath = Join-Path $deliveryDirectory "submission_archive_validation.json"
    $archiveValidationCsv = Join-Path $deliveryDirectory "submission_archive_case_audit.csv"
    foreach ($artifact in @(
        $archivePath,
        $packageManifestPath,
        $directoryValidationPath,
        $directoryValidationCsv,
        $archiveValidationPath,
        $archiveValidationCsv
    )) {
        Assert-V5LeafFile $artifact "Required package artifact"
    }
    $deliveryFiles = @(Get-ChildItem -LiteralPath $deliveryDirectory -File)
    $expectedDeliveryNames = @(
        "Amirfaham_Fallahpour_results.zip",
        "package_manifest.json",
        "submission_directory_validation.json",
        "submission_directory_case_audit.csv",
        "submission_archive_validation.json",
        "submission_archive_case_audit.csv"
    ) | Sort-Object
    $actualDeliveryNames = @($deliveryFiles.Name | Sort-Object)
    if (@(Compare-Object $expectedDeliveryNames $actualDeliveryNames).Count -ne 0) {
        throw "Fresh delivery directory contains an unexpected package artifact set."
    }
    $packageManifest = Read-V5JsonObject $packageManifestPath "Package manifest"
    $archiveValidation = Read-V5JsonObject $archiveValidationPath "Archive validation"
    $directoryValidation = Read-V5JsonObject $directoryValidationPath "Prediction-directory validation"
    Assert-V5Boolean (Get-V5RequiredProperty $archiveValidation "valid" "Archive validation") $true "Committed ZIP validation"
    Assert-V5Boolean (Get-V5RequiredProperty $directoryValidation "valid" "Directory validation") $true "Prediction-directory validation"
    foreach ($validation in @($archiveValidation, $directoryValidation)) {
        if ([int] (Get-V5RequiredProperty $validation "expected_case_count" "Submission validation") -ne 72 -or
            [int] (Get-V5RequiredProperty $validation "validated_mask_count" "Submission validation") -ne 72 -or
            [int] (Get-V5RequiredProperty $validation "validated_csv_row_count" "Submission validation") -ne 72) {
            throw "Submission validator did not certify exactly 72 masks and 72 subtype rows."
        }
    }
    $manifestArchive = Get-V5RequiredProperty $packageManifest "archive" "Package manifest"
    Assert-V5ExactValue (Get-V5RequiredProperty $manifestArchive "path" "Package archive") $archivePath "Package archive path"
    Assert-V5Boolean (Get-V5RequiredProperty $manifestArchive "flat_root" "Package archive") $true "Flat ZIP-root contract"
    $archiveSha256 = Get-V5FileSha256 $archivePath
    Assert-V5HashEquals (Get-V5RequiredProperty $manifestArchive "sha256" "Package archive") $archiveSha256 "Package-manifest ZIP binding"
    Assert-V5HashEquals (Get-V5RequiredProperty $archiveValidation "archive_sha256" "Archive validation") $archiveSha256 "Validator ZIP binding"
    $packageCounts = Get-V5RequiredProperty $packageManifest "counts" "Package manifest"
    foreach ($field in @("expected_cases", "masks", "subtype_rows")) {
        if ([int] (Get-V5RequiredProperty $packageCounts $field "Package counts") -ne 72) {
            throw "Package count '$field' must equal 72."
        }
    }
    if ([int] (Get-V5RequiredProperty $packageCounts "archive_files" "Package counts") -ne 73) {
        throw "Flat submission ZIP must contain exactly 73 files."
    }
    $validatorBindings = Get-V5RequiredProperty $packageManifest "validator_artifacts" "Package manifest"
    foreach ($binding in @(
        [pscustomobject]@{ Name = "prediction_directory_json"; Path = $directoryValidationPath },
        [pscustomobject]@{ Name = "prediction_directory_csv"; Path = $directoryValidationCsv },
        [pscustomobject]@{ Name = "archive_json"; Path = $archiveValidationPath },
        [pscustomobject]@{ Name = "archive_csv"; Path = $archiveValidationCsv }
    )) {
        if (-not (Get-V5NormalizedFullPath ([string] (Get-V5RequiredProperty $validatorBindings $binding.Name "Package validator bindings"))).Equals(
            (Get-V5NormalizedFullPath $binding.Path),
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Package manifest validator binding '$($binding.Name)' is incorrect."
        }
    }

    Write-V5JsonAtomic -Path $completionManifest -Payload ([ordered]@{
        schema_version = 1
        status = "complete_single_selected_candidate_validated"
        completed_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock_sha256 = $candidate.LockSha256
        official_evaluation_gate_sha256 = $expectedGateHash
        official_validation_ledger_sha256 = $officialLedgerSha256
        pre_package_prediction_manifest_sha256 = $prePackageManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        archive = [ordered]@{
            name = "Amirfaham_Fallahpour_results.zip"
            path = $archivePath
            sha256 = $archiveSha256
            size_bytes = [int64] (Get-Item -LiteralPath $archivePath).Length
            flat_root = $true
            file_count = 73
        }
        package_manifest_sha256 = Get-V5FileSha256 $packageManifestPath
        directory_validation_sha256 = Get-V5FileSha256 $directoryValidationPath
        directory_case_audit_sha256 = Get-V5FileSha256 $directoryValidationCsv
        archive_validation_sha256 = Get-V5FileSha256 $archiveValidationPath
        archive_case_audit_sha256 = Get-V5FileSha256 $archiveValidationCsv
        selected_test_inference_invocation_count = 1
        package_invocation_count = 1
        input_or_baseline_replaced = $false
    })
    $completionManifestSha256 = Get-V5FileSha256 $completionManifest
    Write-V5JsonAtomic -Path $ledgerPath -Payload ([ordered]@{
        schema_version = 1
        status = "complete_and_consumed"
        stage = "single_locked_selected_test_and_package"
        started_at_utc = $ledgerStartedAt
        completed_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
        official_evaluation_gate = [ordered]@{ path = $resolvedGate; sha256 = $expectedGateHash }
        official_validation_ledger_sha256 = $officialLedgerSha256
        output_root = $resolvedOutputRoot
        selected_test_inference_invocation_count = 1
        pre_package_prediction_manifest_sha256 = $prePackageManifestSha256
        package_invocation_count = 1
        completion_manifest_sha256 = $completionManifestSha256
        archive_sha256 = $archiveSha256
    })
    $currentStage = "complete"

    Write-Host "Locked selected-test inference and package validation complete."
    Write-Host "ZIP: $archivePath"
    Write-Host "ZIP SHA-256: $archiveSha256"
    Write-Host "Completion manifest: $completionManifest"
}
catch {
    $failure = $_
    if ($ledgerCreated) {
        try {
            Write-V5JsonAtomic -Path $ledgerPath -Payload ([ordered]@{
                schema_version = 1
                status = "failed_and_consumed_no_rerun"
                stage = "single_locked_selected_test_and_package"
                failed_stage = $currentStage
                started_at_utc = $ledgerStartedAt
                failed_at_utc = [DateTime]::UtcNow.ToString("o")
                final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
                official_evaluation_gate = [ordered]@{ path = $resolvedGate; sha256 = $expectedGateHash }
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
