<#
.SYNOPSIS
Run selected-checkpoint test inference and create the validated submission ZIP.

.DESCRIPTION
This is the single foreground entry point after fixed-validation checkpoint
selection. It verifies the selection and train-only rescue branch, binds the
selected checkpoint by SHA-256, requires fresh prediction/evidence/delivery
paths, disables W&B, runs exactly one complete 72-case test inference, and then
invokes the validate-first submission packager without replacement authority.

The process-lifetime mutex is shared with classification rescue and final
validation evaluation. It remains owned through test inference, packaging, and
an additional ZIP validation against the untouched supplied test directory.
Probability and runtime evidence are deliberately outside the flat prediction
directory so only 72 masks plus subtype_results.csv can enter the archive.

This script does not build the report or stage final upload files.

.NOTES
Failure recovery is deliberately non-destructive. If inference fails before the
message "Packaging only after the complete runtime artifact passed validation",
preserve the partial directories and rerun this wrapper with new, empty
-PredictionDirectory and -EvidenceDirectory paths (and a new -DeliveryRoot if
the original was created). If that message was printed, inference is complete:
do not run it again. Run Package-Submission.ps1 directly against the preserved
prediction directory and prepared images, using a new delivery root, then run
validate_submission.py on the resulting ZIP with -SourceTestImages as
--test-images. The recovery commands are:

  .\scripts\Package-Submission.ps1 `
    -PredictionDirectory <preserved-predictions> `
    -TestImages <WorkRoot>\nnUNet_raw\Dataset501_PancreasMultitask\imagesTs `
    -DeliveryRoot <new-empty-delivery> `
    -PythonExecutable <WorkRoot>\.venv\Scripts\python.exe

  <WorkRoot>\.venv\Scripts\python.exe .\scripts\validate_submission.py `
    <new-empty-delivery>\Amirfaham_Fallahpour_results.zip `
    --test-images <SourceTestImages> --expected-count 72 `
    --output-json <new-evidence>\source_test_archive_validation.json `
    --output-csv <new-evidence>\source_test_archive_case_audit.csv

Preserve the original runtime.json and subtype_probabilities.csv as inference
evidence. Never pass -Force or delete the failed-run directories during recovery.

.EXAMPLE
.\scripts\Run-SelectedTestAndPackage.ps1

.EXAMPLE
.\scripts\Run-SelectedTestAndPackage.ps1 `
  -WorkRoot D:\MLQuizWork `
  -SourceTestImages .\ML-Quiz-3DMedImg\ML-Quiz-3DMedImg\test
#>

[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string] $WorkRoot = "D:\MLQuizWork",
    [string] $SelectionPath,
    [string] $PredictionDirectory,
    [string] $EvidenceDirectory,
    [string] $DeliveryRoot,
    [string] $SourceTestImages,
    [string] $PythonExecutable,
    [ValidateSet("cuda", "cpu")]
    [string] $Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# `$PSScriptRoot is not reliably populated while default parameter expressions
# are being bound under Windows PowerShell 5.1. Resolve script-relative
# defaults only after the script body has started.
if ([string]::IsNullOrWhiteSpace($DeliveryRoot)) {
    $DeliveryRoot = Join-Path $PSScriptRoot "..\delivery"
}
if ([string]::IsNullOrWhiteSpace($SourceTestImages)) {
    $SourceTestImages = Join-Path $PSScriptRoot (
        "..\ML-Quiz-3DMedImg\ML-Quiz-3DMedImg\test"
    )
}

$expectedCount = 72
$archiveName = "Amirfaham_Fallahpour_results.zip"
$classificationCsvName = "subtype_results.csv"
$mutexName = "Local\PancreasMultitaskPostTraining501Fold0"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$resolvedWorkRoot = [IO.Path]::GetFullPath($WorkRoot)
$datasetRoot = Join-Path $resolvedWorkRoot "nnUNet_raw\Dataset501_PancreasMultitask"
$preprocessedDatasetRoot = Join-Path $resolvedWorkRoot (
    "nnUNet_preprocessed\Dataset501_PancreasMultitask"
)
$preparedTestImages = Join-Path $datasetRoot "imagesTs"
$rawSplitManifest = Join-Path $datasetRoot "split_manifest.json"
$preprocessedSplitManifest = Join-Path $preprocessedDatasetRoot "split_manifest.json"
$modelRoot = Join-Path $resolvedWorkRoot (
    "nnUNet_results\Dataset501_PancreasMultitask\" +
    "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres"
)
$foldDirectory = Join-Path $modelRoot "fold_0"
$activationAuditPath = Join-Path $foldDirectory "classification_rescue_activation.json"
$recoveryAuditPath = Join-Path $foldDirectory (
    "classification_rescue_zero_update_recovery.json"
)
$rescueCheckpointPath = Join-Path $foldDirectory "checkpoint_classification_rescue.pth"
$rescueAuditPath = "$rescueCheckpointPath.audit.json"

if ([string]::IsNullOrWhiteSpace($SelectionPath)) {
    $SelectionPath = Join-Path (
        (Join-Path $resolvedWorkRoot "evaluation\fixed_validation")
    ) "checkpoint_selection.json"
}
if ([string]::IsNullOrWhiteSpace($PredictionDirectory)) {
    $PredictionDirectory = Join-Path (
        (Join-Path $resolvedWorkRoot "submission")
    ) "Amirfaham_Fallahpour_results"
}
if ([string]::IsNullOrWhiteSpace($EvidenceDirectory)) {
    $EvidenceDirectory = Join-Path (
        (Join-Path $resolvedWorkRoot "evaluation")
    ) "selected_test"
}
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $resolvedWorkRoot ".venv\Scripts\python.exe"
}

$resolvedSelectionPath = [IO.Path]::GetFullPath($SelectionPath)
$resolvedPredictionDirectory = [IO.Path]::GetFullPath($PredictionDirectory)
$resolvedEvidenceDirectory = [IO.Path]::GetFullPath($EvidenceDirectory)
$resolvedDeliveryRoot = [IO.Path]::GetFullPath($DeliveryRoot)
$resolvedSourceTestImages = [IO.Path]::GetFullPath($SourceTestImages)
$resolvedPython = [IO.Path]::GetFullPath($PythonExecutable)

$setupScript = Join-Path $PSScriptRoot "Set-QuizEnvironment.ps1"
$predictionScript = Join-Path $PSScriptRoot "predict_joint.py"
$packageScript = Join-Path $PSScriptRoot "Package-Submission.ps1"
$validatorScript = Join-Path $PSScriptRoot "validate_submission.py"
$recoveryValidationScript = Join-Path $PSScriptRoot (
    "classification_rescue_recovery.py"
)

$probabilityCsv = Join-Path $resolvedEvidenceDirectory "subtype_probabilities.csv"
$runtimeJson = Join-Path $resolvedEvidenceDirectory "runtime.json"
$sourceValidationJson = Join-Path (
    $resolvedEvidenceDirectory
) "source_test_archive_validation.json"
$sourceValidationCsv = Join-Path (
    $resolvedEvidenceDirectory
) "source_test_archive_case_audit.csv"
$classificationCsv = Join-Path $resolvedPredictionDirectory $classificationCsvName

$archivePath = Join-Path $resolvedDeliveryRoot $archiveName
$packageManifestPath = Join-Path $resolvedDeliveryRoot "package_manifest.json"
$directoryValidationJson = Join-Path (
    $resolvedDeliveryRoot
) "submission_directory_validation.json"
$directoryValidationCsv = Join-Path (
    $resolvedDeliveryRoot
) "submission_directory_case_audit.csv"
$archiveValidationJson = Join-Path (
    $resolvedDeliveryRoot
) "submission_archive_validation.json"
$archiveValidationCsv = Join-Path (
    $resolvedDeliveryRoot
) "submission_archive_case_audit.csv"

$candidateCheckpointNames = [ordered] @{
    checkpoint_best = "checkpoint_best.pth"
    checkpoint_best_multitask = "checkpoint_best_multitask.pth"
    checkpoint_final = "checkpoint_final.pth"
    checkpoint_classification_rescue = "checkpoint_classification_rescue.pth"
}
$originalCandidateNames = @(
    "checkpoint_best",
    "checkpoint_best_multitask",
    "checkpoint_final"
)

function Get-NormalizedFullPath {
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

    return [IO.Path]::GetFullPath($Path).TrimEnd([char[]] "\/")
}

function Test-PathAtOrBelow {
    param(
        [Parameter(Mandatory)]
        [string] $Candidate,
        [Parameter(Mandatory)]
        [string] $Parent
    )

    $normalizedCandidate = Get-NormalizedFullPath -Path $Candidate
    $normalizedParent = Get-NormalizedFullPath -Path $Parent
    if ($normalizedCandidate.Equals(
        $normalizedParent,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        return $true
    }
    $prefix = $normalizedParent + [IO.Path]::DirectorySeparatorChar
    return $normalizedCandidate.StartsWith(
        $prefix,
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Assert-DisjointPaths {
    param(
        [Parameter(Mandatory)]
        [string] $First,
        [Parameter(Mandatory)]
        [string] $Second,
        [Parameter(Mandatory)]
        [string] $Description
    )

    if ((Test-PathAtOrBelow -Candidate $First -Parent $Second) -or
        (Test-PathAtOrBelow -Candidate $Second -Parent $First)) {
        throw "$Description must be disjoint: '$First' and '$Second'."
    }
}

function Assert-NotVolumeRoot {
    param(
        [Parameter(Mandatory)]
        [string] $Path,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $resolved = Get-NormalizedFullPath -Path $Path
    $volumeRoot = Get-NormalizedFullPath -Path ([IO.Path]::GetPathRoot($resolved))
    if ($resolved.Equals($volumeRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description cannot be a filesystem volume root: '$Path'."
    }
}

function Assert-FreshPath {
    param(
        [Parameter(Mandatory)]
        [string] $Path,
        [Parameter(Mandatory)]
        [string] $Description
    )

    if (Test-Path -LiteralPath $Path) {
        throw (
            "$Description must not exist before the first selected test run: " +
            "'$Path'. Preserve and inspect existing work instead of overwriting it."
        )
    }
}

function Assert-LeafFile {
    param(
        [Parameter(Mandatory)]
        [string] $Path,
        [Parameter(Mandatory)]
        [string] $Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: '$Path'."
    }
    if ((Get-Item -LiteralPath $Path).Length -le 0) {
        throw "$Description is empty: '$Path'."
    }
}

function Assert-Directory {
    param(
        [Parameter(Mandatory)]
        [string] $Path,
        [Parameter(Mandatory)]
        [string] $Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Description was not found: '$Path'."
    }
}

function Read-JsonObject {
    param(
        [Parameter(Mandatory)]
        [string] $Path,
        [Parameter(Mandatory)]
        [string] $Description
    )

    Assert-LeafFile -Path $Path -Description $Description
    try {
        $payload = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        throw "Could not parse $Description '$Path': $($_.Exception.Message)"
    }
    if ($null -eq $payload -or $payload -is [System.Array]) {
        throw "$Description must contain exactly one JSON object: '$Path'."
    }
    return $payload
}

function Get-RequiredJsonProperty {
    param(
        [Parameter(Mandatory)]
        [object] $InputObject,
        [Parameter(Mandatory)]
        [string] $Name,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $property = $InputObject.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Description is missing required property '$Name'."
    }
    return $property.Value
}

function Get-FileSha256 {
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

    Assert-LeafFile -Path $Path -Description "SHA-256 input"
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Sha256Matches {
    param(
        [Parameter(Mandatory)]
        [object] $RecordedValue,
        [Parameter(Mandatory)]
        [string] $ActualValue,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $recorded = ([string] $RecordedValue).Trim().ToLowerInvariant()
    if ($recorded -notmatch "^[0-9a-f]{64}$") {
        throw "$Description is not a valid SHA-256 digest."
    }
    if (-not $recorded.Equals($ActualValue, [StringComparison]::Ordinal)) {
        throw "$Description does not match the current file."
    }
}

function Assert-PathEquals {
    param(
        [Parameter(Mandatory)]
        [object] $RecordedValue,
        [Parameter(Mandatory)]
        [string] $ActualPath,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $recorded = Get-NormalizedFullPath -Path ([string] $RecordedValue)
    $actual = Get-NormalizedFullPath -Path $ActualPath
    if (-not $recorded.Equals($actual, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description path mismatch: recorded='$recorded', actual='$actual'."
    }
}

function Assert-TrueJsonProperty {
    param(
        [Parameter(Mandatory)]
        [object] $InputObject,
        [Parameter(Mandatory)]
        [string] $Name,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $value = Get-RequiredJsonProperty $InputObject $Name $Description
    if ($value -isnot [bool] -or -not $value) {
        throw "$Description property '$Name' must be JSON true."
    }
}

function Assert-FalseJsonProperty {
    param(
        [Parameter(Mandatory)]
        [object] $InputObject,
        [Parameter(Mandatory)]
        [string] $Name,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $value = Get-RequiredJsonProperty $InputObject $Name $Description
    if ($value -isnot [bool] -or $value) {
        throw "$Description property '$Name' must be JSON false."
    }
}

function Assert-JsonNumberEquals {
    param(
        [Parameter(Mandatory)]
        [object] $InputObject,
        [Parameter(Mandatory)]
        [string] $Name,
        [Parameter(Mandatory)]
        [double] $Expected,
        [Parameter(Mandatory)]
        [double] $Tolerance,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $value = [double] (Get-RequiredJsonProperty $InputObject $Name $Description)
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or
        [Math]::Abs($value - $Expected) -gt $Tolerance) {
        throw (
            "$Description property '$Name' must equal $Expected " +
            "within tolerance $Tolerance; got $value."
        )
    }
}

function Assert-SubmissionAudit {
    param(
        [Parameter(Mandatory)]
        [object] $Audit,
        [Parameter(Mandatory)]
        [ValidateSet("directory", "zip")]
        [string] $SubmissionType,
        [Parameter(Mandatory)]
        [string] $SubmissionPath,
        [string] $ArchiveSha256,
        [Parameter(Mandatory)]
        [string] $Description
    )

    if ([int] (Get-RequiredJsonProperty $Audit "schema_version" $Description) -ne 1) {
        throw "$Description has an unsupported schema_version."
    }
    Assert-TrueJsonProperty $Audit "valid" $Description
    if ([int] (Get-RequiredJsonProperty $Audit "issue_count" $Description) -ne 0 -or
        [int] (Get-RequiredJsonProperty $Audit "expected_case_count" $Description) -ne
            $expectedCount -or
        [int] (Get-RequiredJsonProperty $Audit "validated_mask_count" $Description) -ne
            $expectedCount -or
        [int] (Get-RequiredJsonProperty $Audit "validated_csv_row_count" $Description) -ne
            $expectedCount) {
        throw "$Description does not record a clean 72-case validation."
    }
    if ([string] (Get-RequiredJsonProperty $Audit "csv_name" $Description) -cne
        $classificationCsvName) {
        throw "$Description records the wrong classification CSV name."
    }
    if ([string] (Get-RequiredJsonProperty $Audit "submission_type" $Description) -cne
        $SubmissionType) {
        throw "$Description records the wrong submission type."
    }
    Assert-PathEquals `
        -RecordedValue (Get-RequiredJsonProperty $Audit "submission" $Description) `
        -ActualPath $SubmissionPath `
        -Description "$Description submission"
    $cases = @((Get-RequiredJsonProperty $Audit "cases" $Description))
    if ($cases.Count -ne $expectedCount) {
        throw "$Description must contain exactly $expectedCount case audits."
    }
    if ($SubmissionType -eq "zip") {
        Assert-Sha256Matches `
            -RecordedValue (Get-RequiredJsonProperty $Audit "archive_sha256" $Description) `
            -ActualValue $ArchiveSha256 `
            -Description "$Description archive SHA-256"
    }
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory)]
        [string] $ScriptPath,
        [Parameter(Mandatory)]
        [object[]] $ScriptArguments,
        [Parameter(Mandatory)]
        [string] $Stage
    )

    $messages = @()
    $exitCode = $null
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 turns redirected native stderr into ErrorRecords.
        # Continue lets the native exit code remain the authoritative result.
        $ErrorActionPreference = "Continue"
        $messages = @(& $resolvedPython $ScriptPath @ScriptArguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    foreach ($message in $messages) {
        Write-Host ([string] $message)
    }
    if ($exitCode -ne 0) {
        throw "$Stage failed with exit code $exitCode."
    }
}

function Get-DirectTestImageNames {
    param(
        [Parameter(Mandatory)]
        [string] $Directory,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $files = @(
        Get-ChildItem -LiteralPath $Directory -File -Filter "*_0000.nii.gz" |
            Where-Object {
                -not $_.Name.StartsWith(".") -and -not $_.Name.StartsWith("._")
            }
    )
    if ($files.Count -ne $expectedCount) {
        throw "$Description must contain exactly $expectedCount test images."
    }
    return @($files.Name | Sort-Object)
}

$postTrainingMutex = [Threading.Mutex]::new($false, $mutexName)
$postTrainingMutexOwned = $false
try {
    try {
        $postTrainingMutexOwned = $postTrainingMutex.WaitOne(0)
    }
    catch [Threading.AbandonedMutexException] {
        $postTrainingMutexOwned = $true
    }
    if (-not $postTrainingMutexOwned) {
        throw "Another rescue/evaluation/test/package process owns '$mutexName'."
    }

    foreach ($requiredFile in @(
        $resolvedPython,
        $setupScript,
        $predictionScript,
        $packageScript,
        $validatorScript,
        $recoveryValidationScript,
        $resolvedSelectionPath,
        $activationAuditPath
    )) {
        Assert-LeafFile -Path $requiredFile -Description "Required pipeline file"
    }
    foreach ($requiredDirectory in @(
        $preparedTestImages,
        $resolvedSourceTestImages,
        $modelRoot,
        $foldDirectory
    )) {
        Assert-Directory -Path $requiredDirectory -Description "Required pipeline directory"
    }

    $preparedNames = Get-DirectTestImageNames `
        -Directory $preparedTestImages `
        -Description "Prepared test directory"
    $sourceNames = Get-DirectTestImageNames `
        -Directory $resolvedSourceTestImages `
        -Description "Untouched source test directory"
    $nameDifferences = @(Compare-Object $preparedNames $sourceNames)
    if ($nameDifferences.Count -ne 0) {
        throw "Prepared and untouched source test case names differ."
    }

    $activationAudit = Read-JsonObject `
        -Path $activationAuditPath `
        -Description "Classification-rescue activation audit"
    $activationAuditSha256 = Get-FileSha256 -Path $activationAuditPath
    if ([int] (Get-RequiredJsonProperty (
        $activationAudit
    ) "schema_version" "Activation audit") -ne 1) {
        throw "Activation audit has an unsupported schema_version."
    }
    if ([string] (Get-RequiredJsonProperty (
        $activationAudit
    ) "source_checkpoint_name" "Activation audit") -cne "checkpoint_final.pth") {
        throw "Activation audit source_checkpoint_name must be checkpoint_final.pth."
    }
    $activationApproved = Get-RequiredJsonProperty `
        $activationAudit `
        "activation_approved" `
        "Activation audit"
    if ($activationApproved -isnot [bool]) {
        throw "Activation audit activation_approved must be a JSON boolean."
    }
    $finalCheckpointPath = Join-Path $foldDirectory "checkpoint_final.pth"
    $finalCheckpointSha256 = Get-FileSha256 -Path $finalCheckpointPath
    $activationSourceCheckpointSha256 = [string] (Get-RequiredJsonProperty (
        $activationAudit
    ) "source_checkpoint_sha256" "Activation audit")
    Assert-Sha256Matches `
        -RecordedValue $activationSourceCheckpointSha256 `
        -ActualValue $finalCheckpointSha256 `
        -Description "Activation audit source checkpoint SHA-256"
    if ([int] (Get-RequiredJsonProperty (
        $activationAudit
    ) "checkpoint_current_epoch" "Activation audit") -ne 200) {
        throw "Activation audit must describe checkpoint_final at current_epoch 200."
    }
    if ([int] (Get-RequiredJsonProperty (
        $activationAudit
    ) "training_logging_epoch_count" "Activation audit") -ne 200) {
        throw "Activation audit must contain exactly 200 training logging epochs."
    }
    if ([string] (Get-RequiredJsonProperty (
        $activationAudit
    ) "metric_scope" "Activation audit") -cne
        "checkpoint_training_logging_only") {
        throw "Activation audit metric_scope is not checkpoint training logging only."
    }
    Assert-FalseJsonProperty `
        $activationAudit `
        "validation_metrics_read" `
        "Activation audit"
    Assert-FalseJsonProperty `
        $activationAudit `
        "validation_used_for_activation" `
        "Activation audit"

    $activationDecisionEpoch = Get-RequiredJsonProperty `
        $activationAudit `
        "decision_epoch" `
        "Activation audit"
    if ($activationApproved) {
        if ([int] $activationDecisionEpoch -notin @(40, 50)) {
            throw "An approved activation audit must record decision_epoch 40 or 50."
        }
    }
    elseif ($null -ne $activationDecisionEpoch) {
        throw "A negative activation audit must record a null decision_epoch."
    }

    $rescueCheckpointSha256 = $null
    if ($activationApproved) {
        Assert-LeafFile `
            -Path $rescueCheckpointPath `
            -Description "Completed classification-rescue checkpoint"
        Assert-LeafFile `
            -Path $rescueAuditPath `
            -Description "Completed classification-rescue audit"
        $rescueCheckpointSha256 = Get-FileSha256 -Path $rescueCheckpointPath
        $rescueAudit = Read-JsonObject `
            -Path $rescueAuditPath `
            -Description "Classification-rescue audit"

        if ([int] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "schema_version" "Rescue audit") -ne 1) {
            throw "Rescue audit has an unsupported schema_version."
        }
        if ([string] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "status" "Rescue audit") -cne "complete") {
            throw "Rescue audit status must be complete."
        }
        if ([int] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "completed_epochs" "Rescue audit") -ne 30) {
            throw "Rescue audit must record exactly 30 completed epochs."
        }
        if ([string] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "method" "Rescue audit") -cne
            "post_training_frozen_backbone_classification_head_rescue") {
            throw "Rescue audit method does not identify the frozen-head protocol."
        }
        $rescueSchedule = Get-RequiredJsonProperty `
            $rescueAudit `
            "schedule" `
            "Rescue audit"
        if ([int] (Get-RequiredJsonProperty (
            $rescueSchedule
        ) "epochs" "Rescue schedule") -ne 30) {
            throw "Rescue schedule must declare exactly 30 epochs."
        }
        if ([int] (Get-RequiredJsonProperty (
            $rescueSchedule
        ) "iterations_per_epoch" "Rescue schedule") -ne 125) {
            throw "Rescue schedule must declare exactly 125 iterations per epoch."
        }
        Assert-JsonNumberEquals $rescueSchedule "learning_rate" 0.0003 1e-12 `
            "Rescue schedule"
        Assert-JsonNumberEquals $rescueSchedule "weight_decay" 0.0001 1e-12 `
            "Rescue schedule"
        Assert-JsonNumberEquals $rescueSchedule "gradient_clip_norm" 1.0 1e-12 `
            "Rescue schedule"
        Assert-JsonNumberEquals $rescueSchedule "label_smoothing" 0.05 1e-12 `
            "Rescue schedule"
        Assert-JsonNumberEquals $rescueSchedule "nonlesion_patch_weight" 0.25 1e-12 `
            "Rescue schedule"
        if ([int] (Get-RequiredJsonProperty (
            $rescueSchedule
        ) "reset_seed" "Rescue schedule") -ne 20260806) {
            throw "Rescue schedule reset_seed must be 20260806."
        }
        if ([string] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "optimizer" "Rescue audit") -cne "AdamW") {
            throw "Rescue optimizer must be AdamW."
        }
        if ([string] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "device_type" "Rescue audit") -cne "cuda") {
            throw "Canonical rescue audit must record CUDA execution."
        }
        if ([string] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "training_loader" "Rescue audit") -cne
            "single_threaded_training_split_only") {
            throw "Rescue audit has an unexpected training-loader contract."
        }
        if ([int] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "training_batch_size" "Rescue audit") -ne 2) {
            throw "Rescue training batch size must be 2."
        }
        $precisionPolicy = Get-RequiredJsonProperty `
            $rescueAudit "precision_policy" "Rescue audit"
        $expectedPrecisionPolicy = [ordered]@{
            autocast_scope = "frozen_encoder_forward_only"
            frozen_encoder_forward = "cuda_autocast_float16"
            trainable_classification_forward = "float32"
            classification_loss = "float32"
            classification_backward = "float32"
            gradient_clipping = "float32"
            optimizer_update = "float32"
        }
        foreach ($precisionName in $expectedPrecisionPolicy.Keys) {
            if ([string] (Get-RequiredJsonProperty (
                $precisionPolicy
            ) $precisionName "Rescue precision policy") -cne
                [string] $expectedPrecisionPolicy[$precisionName]) {
                throw "Rescue precision policy differs at '$precisionName'."
            }
        }
        Assert-FalseJsonProperty `
            $precisionPolicy "grad_scaler_enabled" "Rescue precision policy"
        if ([int] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "successful_optimizer_updates" "Rescue audit") -ne 3750) {
            throw "Rescue audit must record exactly 3,750 successful optimizer updates."
        }
        $trainingOnlyHistory = @(
            Get-RequiredJsonProperty `
                $rescueAudit "training_only_history" "Rescue audit"
        )
        if ($trainingOnlyHistory.Count -ne 30) {
            throw "Rescue audit training_only_history must contain exactly 30 epochs."
        }
        for ($epochIndex = 0; $epochIndex -lt 30; $epochIndex++) {
            $epochDescription = "Rescue training-only history epoch $epochIndex"
            $epochRecord = $trainingOnlyHistory[$epochIndex]
            if ([int] (Get-RequiredJsonProperty (
                $epochRecord
            ) "successful_optimizer_updates" $epochDescription) -ne 125) {
                throw "$epochDescription must record exactly 125 successful optimizer updates."
            }
        }
        if ([int] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "maximum_attempts" "Rescue audit") -ne 1) {
            throw "Rescue audit must declare one maximum attempt."
        }
        if ([int] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "training_updates_expected" "Rescue audit") -ne 3750) {
            throw "Rescue audit must declare exactly 3,750 expected updates."
        }
        Assert-FalseJsonProperty $rescueAudit "wandb_enabled" "Rescue audit"
        Assert-FalseJsonProperty $rescueAudit "early_stopping" "Rescue audit"
        Assert-FalseJsonProperty `
            $rescueAudit `
            "decoder_executed_during_rescue" `
            "Rescue audit"
        Assert-FalseJsonProperty $rescueAudit "encoder_gradient_enabled" "Rescue audit"
        Assert-FalseJsonProperty $rescueAudit "decoder_gradient_enabled" "Rescue audit"
        Assert-FalseJsonProperty `
            $rescueAudit `
            "validation_labels_indexed_for_targets" `
            "Rescue audit"

        $rescueSourceCheckpointSha256 = [string] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "source_checkpoint_sha256" "Rescue audit")
        $rescueOutputCheckpointSha256 = [string] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "output_checkpoint_sha256" "Rescue audit")
        Assert-Sha256Matches `
            -RecordedValue $rescueSourceCheckpointSha256 `
            -ActualValue $finalCheckpointSha256 `
            -Description "Rescue audit source checkpoint SHA-256"
        Assert-Sha256Matches `
            -RecordedValue (Get-RequiredJsonProperty (
                $rescueAudit
            ) "activation_audit_sha256" "Rescue audit") `
            -ActualValue $activationAuditSha256 `
            -Description "Rescue audit activation-audit SHA-256"
        Assert-Sha256Matches `
            -RecordedValue $rescueOutputCheckpointSha256 `
            -ActualValue $rescueCheckpointSha256 `
            -Description "Rescue audit output checkpoint SHA-256"
        $processLaunchCount = [int] (Get-RequiredJsonProperty `
            $rescueAudit "process_launch_count" "Rescue audit")
        $zeroUpdateRecoveryCount = [int] (Get-RequiredJsonProperty `
            $rescueAudit "zero_update_recovery_count" "Rescue audit")
        $updateBearingTrajectoryCount = [int] (Get-RequiredJsonProperty `
            $rescueAudit "update_bearing_trajectory_count" "Rescue audit")
        if ($updateBearingTrajectoryCount -ne 1) {
            throw "Rescue audit must record exactly one update-bearing trajectory."
        }
        $recoveryBindingFields = @(
            "execution_recovery",
            "execution_recovery_audit",
            "execution_recovery_audit_sha256"
        )
        if ($processLaunchCount -eq 1 -and $zeroUpdateRecoveryCount -eq 0) {
            foreach ($field in $recoveryBindingFields) {
                if ($null -ne $rescueAudit.PSObject.Properties[$field]) {
                    throw "Clean rescue audit must not fabricate recovery field '$field'."
                }
            }
            if (Test-Path -LiteralPath $recoveryAuditPath) {
                throw "Clean rescue branch conflicts with a canonical recovery artifact."
            }
        }
        elseif ($processLaunchCount -eq 2 -and $zeroUpdateRecoveryCount -eq 1) {
            Assert-LeafFile `
                -Path $recoveryAuditPath `
                -Description "Zero-update execution-recovery audit"
            $recoveryValidationMessages = @(
                & $resolvedPython $recoveryValidationScript validate `
                    --recovery-audit $recoveryAuditPath `
                    --source-checkpoint (Join-Path $foldDirectory "checkpoint_final.pth") `
                    --activation-audit $activationAuditPath `
                    --rescue-audit $rescueAuditPath 2>&1
            )
            if ($LASTEXITCODE -ne 0) {
                throw (
                    "Classification-rescue execution-recovery provenance failed validation: " +
                    ($recoveryValidationMessages -join [Environment]::NewLine)
                )
            }
        }
        else {
            throw (
                "Rescue process/recovery counts must be the clean 1/0 branch or " +
                "the canonical recovered 2/1 branch."
            )
        }
        if ([int] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "activation_decision_epoch" "Rescue audit") -ne
            [int] $activationDecisionEpoch) {
            throw "Rescue audit activation_decision_epoch differs from activation."
        }

        $sourceComponentHashes = Get-RequiredJsonProperty `
            $rescueAudit `
            "source_component_sha256" `
            "Rescue audit"
        $currentComponentHashes = Get-RequiredJsonProperty `
            $rescueAudit `
            "current_component_sha256" `
            "Rescue audit"
        foreach ($componentName in @("encoder", "decoder")) {
            Assert-Sha256Matches `
                -RecordedValue (Get-RequiredJsonProperty `
                    $sourceComponentHashes `
                    $componentName `
                    "Source component hashes") `
                -ActualValue ([string] (Get-RequiredJsonProperty `
                    $currentComponentHashes `
                    $componentName `
                    "Current component hashes")) `
                -Description "Frozen $componentName component SHA-256"
        }

        $sourceClassificationHash = [string] (Get-RequiredJsonProperty `
            $sourceComponentHashes `
            "classification" `
            "Source component hashes")
        $currentClassificationHash = [string] (Get-RequiredJsonProperty `
            $currentComponentHashes `
            "classification" `
            "Current component hashes")
        Assert-Sha256Matches `
            -RecordedValue $sourceClassificationHash `
            -ActualValue $sourceClassificationHash `
            -Description "Source classification component SHA-256"
        Assert-Sha256Matches `
            -RecordedValue $currentClassificationHash `
            -ActualValue $currentClassificationHash `
            -Description "Current classification component SHA-256"
        if ($sourceClassificationHash.Equals(
            $currentClassificationHash,
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Rescue classification component did not change from checkpoint_final."
        }

        $classificationParameterNames = @(
            Get-RequiredJsonProperty `
                $rescueAudit `
                "classification_parameter_names" `
                "Rescue audit"
        )
        $expectedClassificationParameterNames = @(
            "classification_pool.query",
            "classification_pool.token_norm.weight",
            "classification_pool.token_norm.bias",
            "classification_pool.attention.in_proj_weight",
            "classification_pool.attention.in_proj_bias",
            "classification_pool.attention.out_proj.weight",
            "classification_pool.attention.out_proj.bias",
            "classification_pool.output_norm.weight",
            "classification_pool.output_norm.bias",
            "classification_head.0.weight",
            "classification_head.0.bias",
            "classification_head.1.weight",
            "classification_head.1.bias",
            "classification_head.4.weight",
            "classification_head.4.bias"
        )
        if ($classificationParameterNames.Count -ne
                $expectedClassificationParameterNames.Count -or
            @($classificationParameterNames | Sort-Object -Unique).Count -ne
                $expectedClassificationParameterNames.Count -or
            @(Compare-Object (
                @($classificationParameterNames | Sort-Object)
            ) (
                @($expectedClassificationParameterNames | Sort-Object)
            )).Count -ne 0) {
            throw "Rescue audit does not list exactly the frozen classification scope."
        }
        if ([int64] (Get-RequiredJsonProperty (
            $rescueAudit
        ) "classification_trainable_parameter_count" "Rescue audit") -ne 496195) {
            throw "Rescue classification trainable-parameter count must be 496,195."
        }

        $splitAudit = Get-RequiredJsonProperty `
            $rescueAudit `
            "split_audit" `
            "Rescue audit"
        Assert-TrueJsonProperty $splitAudit "split_disjoint" "Rescue split audit"
        Assert-FalseJsonProperty `
            $splitAudit `
            "validation_images_opened" `
            "Rescue split audit"
        Assert-FalseJsonProperty `
            $splitAudit `
            "validation_used_for_gradients" `
            "Rescue split audit"
        Assert-FalseJsonProperty `
            $splitAudit `
            "validation_used_for_stopping" `
            "Rescue split audit"
        if ([int] (Get-RequiredJsonProperty (
            $splitAudit
        ) "validation_batches_consumed" "Rescue split audit") -ne 0) {
            throw "Rescue split audit must record zero validation batches consumed."
        }
        if ([int] (Get-RequiredJsonProperty (
            $splitAudit
        ) "training_case_count" "Rescue split audit") -ne 252) {
            throw "Rescue split audit must record exactly 252 training cases."
        }
        if ([int] (Get-RequiredJsonProperty (
            $splitAudit
        ) "validation_case_count" "Rescue split audit") -ne 36) {
            throw "Rescue split audit must record exactly 36 validation cases."
        }
        if ([int] (Get-RequiredJsonProperty (
            $splitAudit
        ) "frozen_split_manifest_schema_version" "Rescue split audit") -ne 1) {
            throw "Frozen split-manifest schema version must be 1."
        }
        Assert-TrueJsonProperty `
            $splitAudit `
            "frozen_source_splits_preserved" `
            "Rescue split audit"
        Assert-TrueJsonProperty `
            $splitAudit `
            "matches_frozen_split_manifest" `
            "Rescue split audit"
        if ([int] (Get-RequiredJsonProperty (
            $splitAudit
        ) "frozen_manifest_training_case_count" "Rescue split audit") -ne 252) {
            throw "Frozen split manifest must record 252 training cases."
        }
        if ([int] (Get-RequiredJsonProperty (
            $splitAudit
        ) "frozen_manifest_validation_case_count" "Rescue split audit") -ne 36) {
            throw "Frozen split manifest must record 36 validation cases."
        }

        Assert-LeafFile `
            -Path $rawSplitManifest `
            -Description "Raw frozen split manifest"
        Assert-LeafFile `
            -Path $preprocessedSplitManifest `
            -Description "Preprocessed frozen split manifest"
        $rawSplitManifestSha256 = Get-FileSha256 -Path $rawSplitManifest
        $preprocessedSplitManifestSha256 = Get-FileSha256 `
            -Path $preprocessedSplitManifest
        Assert-Sha256Matches `
            -RecordedValue $rawSplitManifestSha256 `
            -ActualValue $preprocessedSplitManifestSha256 `
            -Description "Raw/preprocessed frozen split-manifest SHA-256"
        Assert-Sha256Matches `
            -RecordedValue (Get-RequiredJsonProperty `
                $splitAudit `
                "frozen_split_manifest_sha256" `
                "Rescue split audit") `
            -ActualValue $preprocessedSplitManifestSha256 `
            -Description "Rescue frozen split-manifest SHA-256"

        Assert-PathEquals `
            -RecordedValue (Get-RequiredJsonProperty `
                $splitAudit `
                "frozen_split_manifest" `
                "Rescue split audit") `
            -ActualPath $preprocessedSplitManifest `
            -Description "Rescue frozen split manifest"
        Assert-Sha256Matches `
            -RecordedValue (Get-RequiredJsonProperty `
                $splitAudit `
                "training_case_ids_sha256" `
                "Rescue split audit") `
            -ActualValue ([string] (Get-RequiredJsonProperty `
                $splitAudit `
                "frozen_manifest_training_case_ids_sha256" `
                "Rescue split audit")) `
            -Description "Training case-ID SHA-256 binding"
        Assert-Sha256Matches `
            -RecordedValue (Get-RequiredJsonProperty `
                $splitAudit `
                "validation_case_ids_sha256" `
                "Rescue split audit") `
            -ActualValue ([string] (Get-RequiredJsonProperty `
                $splitAudit `
                "frozen_manifest_validation_case_ids_sha256" `
                "Rescue split audit")) `
            -Description "Validation case-ID SHA-256 binding"
        }

    $selection = Read-JsonObject `
        -Path $resolvedSelectionPath `
        -Description "Checkpoint selection artifact"
    if ([int] (Get-RequiredJsonProperty (
        $selection
    ) "schema_version" "Selection artifact") -ne 1) {
        throw "Selection artifact has an unsupported schema_version."
    }

    $selectionPolicy = Get-RequiredJsonProperty `
        $selection `
        "selection_policy" `
        "Selection artifact"
    if ([string] (Get-RequiredJsonProperty (
        $selectionPolicy
    ) "direction" "Selection policy") -cne "maximize" -or
        [string] (Get-RequiredJsonProperty (
            $selectionPolicy
        ) "score" "Selection policy") -cne "equal-weight arithmetic mean" -or
        [string] (Get-RequiredJsonProperty (
            $selectionPolicy
        ) "tie_breaker" "Selection policy") -cne
            "candidate name ascending; no secondary metric") {
        throw "Selection artifact does not record the frozen selection policy."
    }
    $expectedMetricPaths = @(
        "segmentation.whole_pancreas_dice.mean",
        "segmentation.lesion_dice.mean",
        "classification.macro_f1"
    )
    $recordedMetricPaths = @((Get-RequiredJsonProperty (
        $selectionPolicy
    ) "metric_paths" "Selection policy"))
    if (@(Compare-Object $expectedMetricPaths $recordedMetricPaths).Count -ne 0) {
        throw "Selection policy records the wrong metric paths."
    }
    $selectionWeights = Get-RequiredJsonProperty `
        $selectionPolicy `
        "metric_weights" `
        "Selection policy"
    foreach ($metricName in @(
        "whole_pancreas_dice",
        "lesion_dice",
        "macro_f1"
    )) {
        $weight = [double] (Get-RequiredJsonProperty (
            $selectionWeights
        ) $metricName "Selection metric weights")
        if ([double]::IsNaN($weight) -or [double]::IsInfinity($weight) -or
            [Math]::Abs($weight - (1.0 / 3.0)) -gt 1e-12) {
            throw "Selection policy metric '$metricName' must have weight one third."
        }
    }

    $expectedCandidateNames = @($originalCandidateNames)
    if ($activationApproved) {
        $expectedCandidateNames += "checkpoint_classification_rescue"
    }
    $candidateCount = [int] (Get-RequiredJsonProperty (
        $selection
    ) "candidate_count" "Selection artifact")
    if ($candidateCount -ne $expectedCandidateNames.Count) {
        throw (
            "Selection candidate_count does not match the train-only activation branch; " +
            "expected $($expectedCandidateNames.Count), got $candidateCount."
        )
    }
    $ranking = @((Get-RequiredJsonProperty $selection "ranking" "Selection artifact"))
    if ($ranking.Count -ne $candidateCount) {
        throw "Selection ranking length differs from candidate_count."
    }
    $rankedNames = @(
        $ranking | ForEach-Object {
            [string] (Get-RequiredJsonProperty $_ "candidate" "Selection ranking entry")
        }
    )
    if (@($rankedNames | Sort-Object -Unique).Count -ne $candidateCount -or
        @(Compare-Object (
            @($expectedCandidateNames | Sort-Object)
        ) (@($rankedNames | Sort-Object))).Count -ne 0) {
        throw "Selection ranking contains the wrong or duplicate candidate names."
    }

    $previousSelectionScore = $null
    $previousCandidateName = $null
    $rankingCheckpointSha256ByCandidate = @{}
    for ($rankingIndex = 0; $rankingIndex -lt $ranking.Count; $rankingIndex++) {
        $rankingEntry = $ranking[$rankingIndex]
        $rankingCandidate = [string] (Get-RequiredJsonProperty (
            $rankingEntry
        ) "candidate" "Selection ranking entry")
        if ([int] (Get-RequiredJsonProperty (
            $rankingEntry
        ) "rank" "Selection ranking entry") -ne ($rankingIndex + 1)) {
            throw "Selection ranking ranks must be consecutive and match array order."
        }

        $rankingMetrics = Get-RequiredJsonProperty `
            $rankingEntry `
            "metrics" `
            "Selection ranking entry"
        $metricTotal = 0.0
        foreach ($metricName in @(
            "whole_pancreas_dice",
            "lesion_dice",
            "macro_f1"
        )) {
            $metricValue = [double] (Get-RequiredJsonProperty (
                $rankingMetrics
            ) $metricName "Selection ranking metrics")
            if ([double]::IsNaN($metricValue) -or
                [double]::IsInfinity($metricValue) -or
                $metricValue -lt 0.0 -or $metricValue -gt 1.0) {
                throw "Selection metric '$metricName' must be finite and in [0, 1]."
            }
            $metricTotal += $metricValue
        }
        $rankingScore = [double] (Get-RequiredJsonProperty (
            $rankingEntry
        ) "selection_score" "Selection ranking entry")
        if ([double]::IsNaN($rankingScore) -or
            [double]::IsInfinity($rankingScore) -or
            [Math]::Abs($rankingScore - ($metricTotal / 3.0)) -gt 1e-12) {
            throw "Selection ranking score is inconsistent with its three metrics."
        }
        if ($null -ne $previousSelectionScore) {
            if ($rankingScore -gt [double] $previousSelectionScore -or
                ($rankingScore -eq [double] $previousSelectionScore -and
                    [string]::CompareOrdinal(
                        [string] $previousCandidateName,
                        $rankingCandidate
                    ) -gt 0)) {
                throw "Selection ranking violates the frozen score/tie ordering."
            }
        }
        $previousSelectionScore = $rankingScore
        $previousCandidateName = $rankingCandidate

        $rankingCheckpointPath = [IO.Path]::GetFullPath([string] (
            Get-RequiredJsonProperty `
                $rankingEntry `
                "checkpoint_path" `
                "Selection ranking entry"
        ))
        if (-not (Get-NormalizedFullPath -Path (
            Split-Path -Parent $rankingCheckpointPath
        )).Equals(
            (Get-NormalizedFullPath -Path $foldDirectory),
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Every ranked checkpoint must be a direct child of fold_0."
        }
        if ((Split-Path -Leaf $rankingCheckpointPath) -cne
            $candidateCheckpointNames[$rankingCandidate]) {
            throw "A ranked checkpoint filename does not match its candidate."
        }
        $rankingCheckpointSha256 = Get-FileSha256 -Path $rankingCheckpointPath
        Assert-Sha256Matches `
            -RecordedValue (Get-RequiredJsonProperty (
                $rankingEntry
            ) "checkpoint_sha256" "Selection ranking entry") `
            -ActualValue $rankingCheckpointSha256 `
            -Description "Selection ranking checkpoint SHA-256"
        $rankingCheckpointSha256ByCandidate[$rankingCandidate] = `
            $rankingCheckpointSha256
    }

    # Complete the provenance chain explicitly: activation -> checkpoint_final,
    # rescue -> checkpoint_final/rescue output, and selection -> both exact files.
    # The individual file checks above make these direct digest comparisons
    # transitive over current bytes rather than trusting coincidentally named files.
    Assert-Sha256Matches `
        -RecordedValue $activationSourceCheckpointSha256 `
        -ActualValue $rankingCheckpointSha256ByCandidate["checkpoint_final"] `
        -Description "Activation-to-selection checkpoint_final SHA-256"
    if ($activationApproved) {
        Assert-Sha256Matches `
            -RecordedValue $rescueSourceCheckpointSha256 `
            -ActualValue $rankingCheckpointSha256ByCandidate["checkpoint_final"] `
            -Description "Rescue-source-to-selection checkpoint_final SHA-256"
        Assert-Sha256Matches `
            -RecordedValue $rescueOutputCheckpointSha256 `
            -ActualValue $rankingCheckpointSha256ByCandidate[
                "checkpoint_classification_rescue"
            ] `
            -Description "Rescue-output-to-selection checkpoint SHA-256"
    }

    $selectedCandidate = [string] (Get-RequiredJsonProperty (
        $selection
    ) "selected_candidate" "Selection artifact")
    if ($selectedCandidate -notin $expectedCandidateNames) {
        throw "Selection artifact names an inadmissible selected candidate."
    }
    $selectedEntries = @(
        $ranking | Where-Object {
            [string] $_.candidate -ceq $selectedCandidate -and [int] $_.rank -eq 1
        }
    )
    if ($selectedEntries.Count -ne 1) {
        throw "Selected candidate must appear exactly once at rank 1."
    }
    if ([string] $ranking[0].candidate -cne $selectedCandidate) {
        throw "Selected candidate must be the first ordered ranking entry."
    }
    $selectedScore = [double] (Get-RequiredJsonProperty (
        $selection
    ) "selected_score" "Selection artifact")
    if ([Math]::Abs(
        $selectedScore - [double] (Get-RequiredJsonProperty (
            $ranking[0]
        ) "selection_score" "Rank-1 selection entry")
    ) -gt 1e-12) {
        throw "Selection artifact selected_score differs from rank 1."
    }

    $selectedCheckpointPath = [IO.Path]::GetFullPath([string] (
        Get-RequiredJsonProperty $selection "selected_checkpoint_path" "Selection artifact"
    ))
    Assert-LeafFile -Path $selectedCheckpointPath -Description "Selected checkpoint"
    $selectedCheckpointParent = Get-NormalizedFullPath -Path (
        Split-Path -Parent $selectedCheckpointPath
    )
    if (-not $selectedCheckpointParent.Equals(
        (Get-NormalizedFullPath -Path $foldDirectory),
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Selected checkpoint must be a direct child of the configured fold_0."
    }
    $selectedCheckpointName = Split-Path -Leaf $selectedCheckpointPath
    if ($selectedCheckpointName -cne $candidateCheckpointNames[$selectedCandidate]) {
        throw (
            "Selected checkpoint filename '$selectedCheckpointName' does not match " +
            "candidate '$selectedCandidate'."
        )
    }
    $selectedCheckpointSha256 = Get-FileSha256 -Path $selectedCheckpointPath
    Assert-Sha256Matches `
        -RecordedValue (Get-RequiredJsonProperty (
            $selection
        ) "selected_checkpoint_sha256" "Selection artifact") `
        -ActualValue $selectedCheckpointSha256 `
        -Description "Selection artifact checkpoint SHA-256"

    $selectedEntry = $selectedEntries[0]
    Assert-PathEquals `
        -RecordedValue (Get-RequiredJsonProperty (
            $selectedEntry
        ) "checkpoint_path" "Selected ranking entry") `
        -ActualPath $selectedCheckpointPath `
        -Description "Selected ranking checkpoint"
    Assert-Sha256Matches `
        -RecordedValue (Get-RequiredJsonProperty (
            $selectedEntry
        ) "checkpoint_sha256" "Selected ranking entry") `
        -ActualValue $selectedCheckpointSha256 `
        -Description "Selected ranking checkpoint SHA-256"

    foreach ($outputPath in @(
        $resolvedPredictionDirectory,
        $resolvedEvidenceDirectory,
        $resolvedDeliveryRoot
    )) {
        Assert-NotVolumeRoot -Path $outputPath -Description "Output path"
    }
    if (-not (Test-PathAtOrBelow `
        -Candidate $resolvedPredictionDirectory `
        -Parent $resolvedWorkRoot) -or
        (Get-NormalizedFullPath -Path $resolvedPredictionDirectory).Equals(
            (Get-NormalizedFullPath -Path $resolvedWorkRoot),
            [StringComparison]::OrdinalIgnoreCase
        )) {
        throw "PredictionDirectory must be a proper descendant of WorkRoot."
    }
    if (-not (Test-PathAtOrBelow `
        -Candidate $resolvedEvidenceDirectory `
        -Parent $resolvedWorkRoot) -or
        (Get-NormalizedFullPath -Path $resolvedEvidenceDirectory).Equals(
            (Get-NormalizedFullPath -Path $resolvedWorkRoot),
            [StringComparison]::OrdinalIgnoreCase
        )) {
        throw "EvidenceDirectory must be a proper descendant of WorkRoot."
    }

    Assert-DisjointPaths `
        -First $resolvedPredictionDirectory `
        -Second $resolvedEvidenceDirectory `
        -Description "Prediction and evidence directories"
    Assert-DisjointPaths `
        -First $resolvedPredictionDirectory `
        -Second $resolvedDeliveryRoot `
        -Description "Prediction and delivery directories"
    Assert-DisjointPaths `
        -First $resolvedEvidenceDirectory `
        -Second $resolvedDeliveryRoot `
        -Description "Evidence and delivery directories"
    foreach ($outputPath in @(
        $resolvedPredictionDirectory,
        $resolvedEvidenceDirectory,
        $resolvedDeliveryRoot
    )) {
        foreach ($protectedPath in @(
            $preparedTestImages,
            $resolvedSourceTestImages,
            $modelRoot,
            $resolvedSelectionPath
        )) {
            Assert-DisjointPaths `
                -First $outputPath `
                -Second $protectedPath `
                -Description "Output and protected input paths"
        }
    }

    # All three guards run before the first output directory is created.
    Assert-FreshPath `
        -Path $resolvedPredictionDirectory `
        -Description "PredictionDirectory"
    Assert-FreshPath `
        -Path $resolvedEvidenceDirectory `
        -Description "EvidenceDirectory"
    Assert-FreshPath `
        -Path $resolvedDeliveryRoot `
        -Description "DeliveryRoot"

    New-Item -ItemType Directory -Path $resolvedEvidenceDirectory | Out-Null

    & $setupScript -WorkRoot $resolvedWorkRoot -WandbMode disabled | Out-Null
    $env:WANDB_MODE = "disabled"
    $env:WANDB_DISABLED = "true"
    $env:WANDB_SILENT = "true"
    if ($env:nnUNet_wandb_enabled -ne "0") {
        throw "Environment setup did not disable nnU-Net W&B logging."
    }

    Write-Host "Running one fresh selected-checkpoint test inference..."
    $predictionArguments = @(
        "--input", $preparedTestImages,
        "--output", $resolvedPredictionDirectory,
        "--model", $modelRoot,
        "--folds", "0",
        "--checkpoint", $selectedCheckpointName,
        "--classification-csv", $classificationCsv,
        "--probability-csv", $probabilityCsv,
        "--runtime-json", $runtimeJson,
        "--tile-step-size", "0.5",
        "--device", $Device,
        "--overwrite"
    )
    Invoke-CheckedPython `
        -ScriptPath $predictionScript `
        -ScriptArguments $predictionArguments `
        -Stage "Selected-checkpoint test inference"

    # Bind the completed predictions to the same checkpoint bytes verified
    # before inference. Any legitimate rescue/evaluation writer is excluded by
    # the mutex; this second hash also detects an external or stale trainer write.
    $postInferenceCheckpointSha256 = Get-FileSha256 -Path $selectedCheckpointPath
    Assert-Sha256Matches `
        -RecordedValue $selectedCheckpointSha256 `
        -ActualValue $postInferenceCheckpointSha256 `
        -Description "Post-inference selected checkpoint SHA-256"

    $runtime = Read-JsonObject `
        -Path $runtimeJson `
        -Description "Selected test runtime artifact"
    Assert-LeafFile -Path $probabilityCsv -Description "Selected test probability CSV"
    Assert-LeafFile -Path $classificationCsv -Description "Submission classification CSV"
    if ([int] (Get-RequiredJsonProperty $runtime "case_count" "Runtime artifact") -ne
        $expectedCount) {
        throw "Runtime artifact must record exactly $expectedCount cases."
    }
    if ([string] (Get-RequiredJsonProperty $runtime "checkpoint" "Runtime artifact") -cne
        $selectedCheckpointName) {
        throw "Runtime artifact records the wrong checkpoint."
    }
    if ([string] (Get-RequiredJsonProperty $runtime "device" "Runtime artifact") -cne
        $Device) {
        throw "Runtime artifact records the wrong device."
    }
    $runtimeFolds = @((Get-RequiredJsonProperty $runtime "folds" "Runtime artifact"))
    if ($runtimeFolds.Count -ne 1 -or [string] $runtimeFolds[0] -cne "0") {
        throw "Runtime artifact must record only fold 0."
    }
    Assert-TrueJsonProperty $runtime "gaussian_enabled" "Runtime artifact"
    Assert-TrueJsonProperty $runtime "tta_enabled" "Runtime artifact"
    if ([Math]::Abs(
        [double] (Get-RequiredJsonProperty $runtime "tile_step_size" "Runtime artifact") -
            0.5
    ) -gt 1e-12) {
        throw "Runtime artifact records the wrong tile step size."
    }
    $totalSeconds = [double] (Get-RequiredJsonProperty (
        $runtime
    ) "total_seconds" "Runtime artifact")
    $meanSeconds = [double] (Get-RequiredJsonProperty (
        $runtime
    ) "mean_seconds_per_case" "Runtime artifact")
    if ([double]::IsNaN($totalSeconds) -or [double]::IsInfinity($totalSeconds) -or
        $totalSeconds -le 0 -or [double]::IsNaN($meanSeconds) -or
        [double]::IsInfinity($meanSeconds) -or $meanSeconds -le 0) {
        throw "Runtime artifact must record finite positive timing values."
    }

    Write-Host "Packaging only after the complete runtime artifact passed validation..."
    $packageArguments = @{
        PredictionDirectory = $resolvedPredictionDirectory
        TestImages = $preparedTestImages
        DeliveryRoot = $resolvedDeliveryRoot
        PythonExecutable = $resolvedPython
    }
    & $packageScript @packageArguments | Out-Host

    Assert-LeafFile -Path $archivePath -Description "Committed submission ZIP"
    $archiveSha256 = Get-FileSha256 -Path $archivePath

    Write-Host "Independently validating the ZIP against untouched source test images..."
    $sourceValidationArguments = @(
        $archivePath,
        "--test-images", $resolvedSourceTestImages,
        "--expected-count", [string] $expectedCount,
        "--output-json", $sourceValidationJson,
        "--output-csv", $sourceValidationCsv
    )
    Invoke-CheckedPython `
        -ScriptPath $validatorScript `
        -ScriptArguments $sourceValidationArguments `
        -Stage "Untouched-source submission validation"

    foreach ($packageEvidencePath in @(
        $packageManifestPath,
        $directoryValidationJson,
        $directoryValidationCsv,
        $archiveValidationJson,
        $archiveValidationCsv,
        $sourceValidationJson,
        $sourceValidationCsv
    )) {
        Assert-LeafFile -Path $packageEvidencePath -Description "Package evidence artifact"
    }

    $packageManifest = Read-JsonObject `
        -Path $packageManifestPath `
        -Description "Package manifest"
    if ([int] (Get-RequiredJsonProperty (
        $packageManifest
    ) "schema_version" "Package manifest") -ne 1) {
        throw "Package manifest has an unsupported schema_version."
    }
    $manifestArchive = Get-RequiredJsonProperty `
        $packageManifest `
        "archive" `
        "Package manifest"
    Assert-PathEquals `
        -RecordedValue (Get-RequiredJsonProperty (
            $manifestArchive
        ) "path" "Package manifest archive") `
        -ActualPath $archivePath `
        -Description "Package manifest archive"
    Assert-Sha256Matches `
        -RecordedValue (Get-RequiredJsonProperty (
            $manifestArchive
        ) "sha256" "Package manifest archive") `
        -ActualValue $archiveSha256 `
        -Description "Package manifest archive SHA-256"
    Assert-TrueJsonProperty $manifestArchive "flat_root" "Package manifest archive"
    if ([long] (Get-RequiredJsonProperty (
        $manifestArchive
    ) "size_bytes" "Package manifest archive") -ne
        [long] (Get-Item -LiteralPath $archivePath).Length) {
        throw "Package manifest records the wrong archive size."
    }
    $manifestCounts = Get-RequiredJsonProperty `
        $packageManifest `
        "counts" `
        "Package manifest"
    if ([int] (Get-RequiredJsonProperty (
        $manifestCounts
    ) "expected_cases" "Package manifest counts") -ne $expectedCount -or
        [int] (Get-RequiredJsonProperty (
            $manifestCounts
        ) "masks" "Package manifest counts") -ne $expectedCount -or
        [int] (Get-RequiredJsonProperty (
            $manifestCounts
        ) "subtype_rows" "Package manifest counts") -ne $expectedCount -or
        [int] (Get-RequiredJsonProperty (
            $manifestCounts
        ) "archive_files" "Package manifest counts") -ne ($expectedCount + 1)) {
        throw "Package manifest records inconsistent submission counts."
    }
    $manifestValidation = Get-RequiredJsonProperty `
        $packageManifest `
        "validation" `
        "Package manifest"
    Assert-TrueJsonProperty `
        $manifestValidation `
        "prediction_directory_valid" `
        "Package manifest validation"
    Assert-TrueJsonProperty `
        $manifestValidation `
        "archive_valid" `
        "Package manifest validation"

    $manifestValidatorArtifacts = Get-RequiredJsonProperty `
        $packageManifest `
        "validator_artifacts" `
        "Package manifest"
    foreach ($artifactPair in @(
        [pscustomobject] @{
            Name = "prediction_directory_json"
            Path = $directoryValidationJson
        },
        [pscustomobject] @{
            Name = "prediction_directory_csv"
            Path = $directoryValidationCsv
        },
        [pscustomobject] @{
            Name = "archive_json"
            Path = $archiveValidationJson
        },
        [pscustomobject] @{
            Name = "archive_csv"
            Path = $archiveValidationCsv
        }
    )) {
        Assert-PathEquals `
            -RecordedValue (Get-RequiredJsonProperty (
                $manifestValidatorArtifacts
            ) $artifactPair.Name "Package manifest validator artifacts") `
            -ActualPath $artifactPair.Path `
            -Description "Package manifest validator artifact"
    }

    $directoryAudit = Read-JsonObject `
        -Path $directoryValidationJson `
        -Description "Prediction-directory validation"
    Assert-SubmissionAudit `
        -Audit $directoryAudit `
        -SubmissionType "directory" `
        -SubmissionPath $resolvedPredictionDirectory `
        -Description "Prediction-directory validation"

    $archiveAudit = Read-JsonObject `
        -Path $archiveValidationJson `
        -Description "Committed-archive validation"
    Assert-SubmissionAudit `
        -Audit $archiveAudit `
        -SubmissionType "zip" `
        -SubmissionPath $archivePath `
        -ArchiveSha256 $archiveSha256 `
        -Description "Committed-archive validation"

    $sourceAudit = Read-JsonObject `
        -Path $sourceValidationJson `
        -Description "Untouched-source archive validation"
    Assert-SubmissionAudit `
        -Audit $sourceAudit `
        -SubmissionType "zip" `
        -SubmissionPath $archivePath `
        -ArchiveSha256 $archiveSha256 `
        -Description "Untouched-source archive validation"

    $expectedDeliveryNames = @(
        $archiveName,
        "package_manifest.json",
        "submission_directory_validation.json",
        "submission_directory_case_audit.csv",
        "submission_archive_validation.json",
        "submission_archive_case_audit.csv"
    ) | Sort-Object
    $actualDeliveryEntries = @(Get-ChildItem -LiteralPath $resolvedDeliveryRoot -Force)
    $actualDeliveryNames = @($actualDeliveryEntries.Name | Sort-Object)
    if ($actualDeliveryEntries.Count -ne $expectedDeliveryNames.Count -or
        @((Compare-Object $expectedDeliveryNames $actualDeliveryNames)).Count -ne 0) {
        throw "Fresh delivery root contains unexpected or missing package artifacts."
    }

    $expectedEvidenceNames = @(
        "subtype_probabilities.csv",
        "runtime.json",
        "source_test_archive_validation.json",
        "source_test_archive_case_audit.csv"
    ) | Sort-Object
    $actualEvidenceEntries = @(Get-ChildItem -LiteralPath $resolvedEvidenceDirectory -Force)
    $actualEvidenceNames = @($actualEvidenceEntries.Name | Sort-Object)
    if ($actualEvidenceEntries.Count -ne $expectedEvidenceNames.Count -or
        @((Compare-Object $expectedEvidenceNames $actualEvidenceNames)).Count -ne 0) {
        throw "Fresh evidence directory contains unexpected or missing artifacts."
    }

    Write-Host "Selected test inference and validated packaging complete."
    Write-Host "Selected candidate: $selectedCandidate"
    Write-Host "Checkpoint SHA-256: $selectedCheckpointSha256"
    Write-Host "Archive: $archivePath"
    Write-Host "Archive SHA-256: $archiveSha256"

    [pscustomobject] @{
        SelectedCandidate = $selectedCandidate
        SelectionArtifact = $resolvedSelectionPath
        CheckpointPath = $selectedCheckpointPath
        CheckpointSha256 = $selectedCheckpointSha256
        PredictionDirectory = $resolvedPredictionDirectory
        EvidenceDirectory = $resolvedEvidenceDirectory
        DeliveryRoot = $resolvedDeliveryRoot
        RuntimeJson = $runtimeJson
        ProbabilityCsv = $probabilityCsv
        Archive = $archivePath
        ArchiveSha256 = $archiveSha256
        PackageManifest = $packageManifestPath
        DirectoryValidation = $directoryValidationJson
        ArchiveValidation = $archiveValidationJson
        SourceTestValidation = $sourceValidationJson
    }
}
finally {
    if ($postTrainingMutexOwned) {
        $postTrainingMutex.ReleaseMutex()
    }
    $postTrainingMutex.Dispose()
}
