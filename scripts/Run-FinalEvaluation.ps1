<#
.SYNOPSIS
Run fixed-validation inference, evaluation, and checkpoint selection.

.DESCRIPTION
This script evaluates the three original checkpoint candidates, plus the
predeclared classification rescue only when the train-only activation audit
approved and the completed rescue is explicitly included. It uses the held-out
36-case validation split and deliberately stops before test-set inference or ZIP
creation. Existing complete cases are resumed by default; pass -Force to recompute
them. Weights & Biases is disabled for this process because this is deterministic
post-training evaluation, not a training run.

.EXAMPLE
.\scripts\Run-FinalEvaluation.ps1

.EXAMPLE
.\scripts\Run-FinalEvaluation.ps1 -Device cpu

.EXAMPLE
.\scripts\Run-FinalEvaluation.ps1 -Force -ResultsOnCpu

.EXAMPLE
.\scripts\Run-FinalEvaluation.ps1 -IncludeClassificationRescue
#>

[CmdletBinding()]
param(
    [string] $WorkRoot = "D:\MLQuizWork",
    [ValidateSet("cuda", "cpu")]
    [string] $Device = "cuda",
    [switch] $ResultsOnCpu,
    [switch] $IncludeClassificationRescue,
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# A process-lifetime named mutex prevents two launchers from writing the same
# prediction/evaluation tree. It is released automatically if PowerShell dies,
# so interrupted runs can resume without a stale on-disk lock.
$postTrainingMutex = [Threading.Mutex]::new(
    $false,
    "Local\PancreasMultitaskPostTraining501Fold0"
)
$postTrainingMutexOwned = $false
try {
    try {
        $postTrainingMutexOwned = $postTrainingMutex.WaitOne(0)
    }
    catch [Threading.AbandonedMutexException] {
        $postTrainingMutexOwned = $true
    }
    if (-not $postTrainingMutexOwned) {
        throw "Another rescue/evaluation process already owns the post-training mutex."
    }

$workRoot = [IO.Path]::GetFullPath($WorkRoot)
$datasetRoot = Join-Path $workRoot "nnUNet_raw\Dataset501_PancreasMultitask"
$validationImages = Join-Path $datasetRoot "imagesVal"
$validationLabels = Join-Path $datasetRoot "labelsVal"
$classificationManifest = Join-Path $datasetRoot "classification_manifest.json"
$trainedModelRoot = Join-Path $workRoot (
    "nnUNet_results\Dataset501_PancreasMultitask\" +
    "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres"
)
$foldDirectory = Join-Path $trainedModelRoot "fold_0"
$activationAuditPath = Join-Path $foldDirectory "classification_rescue_activation.json"
$rescueCheckpointPath = Join-Path $foldDirectory "checkpoint_classification_rescue.pth"
$rescueAuditPath = "$rescueCheckpointPath.audit.json"
$evaluationRoot = Join-Path $workRoot "evaluation\fixed_validation"
$selectionOutput = Join-Path $evaluationRoot "checkpoint_selection.json"
$pythonExecutable = Join-Path $workRoot ".venv\Scripts\python.exe"
$setupScript = Join-Path $PSScriptRoot "Set-QuizEnvironment.ps1"
$predictionScript = Join-Path $PSScriptRoot "predict_joint.py"
$evaluationScript = Join-Path $PSScriptRoot "evaluate_predictions.py"
$selectionScript = Join-Path $PSScriptRoot "select_checkpoint.py"

# These candidates and the equal-weight selection policy in select_checkpoint.py
# were fixed before inspecting the full-volume validation results.
$candidates = @(
    [pscustomobject]@{
        Name = "checkpoint_best"
        FileName = "checkpoint_best.pth"
    },
    [pscustomobject]@{
        Name = "checkpoint_best_multitask"
        FileName = "checkpoint_best_multitask.pth"
    },
    [pscustomobject]@{
        Name = "checkpoint_final"
        FileName = "checkpoint_final.pth"
    }
)

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
        throw "$Description must contain one JSON object: '$Path'."
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

function Assert-FalseJsonProperty {
    param(
        [Parameter(Mandatory)]
        [object] $InputObject,
        [Parameter(Mandatory)]
        [string] $Name,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $value = Get-RequiredJsonProperty `
        -InputObject $InputObject `
        -Name $Name `
        -Description $Description
    if ($value -isnot [bool] -or $value -ne $false) {
        throw "$Description property '$Name' must be the JSON boolean false."
    }
}

function Get-FileSha256 {
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

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

    $recorded = [string] $RecordedValue
    if ($recorded -notmatch "^[0-9a-fA-F]{64}$") {
        throw "$Description must be a 64-digit hexadecimal SHA-256."
    }
    if (-not [String]::Equals($recorded, $ActualValue, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description does not match the current file."
    }
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory)]
        [string] $ScriptPath,
        [Parameter(Mandatory)]
        [string[]] $ScriptArguments,
        [Parameter(Mandatory)]
        [string] $Stage
    )

    $invocationArguments = @($ScriptPath) + $ScriptArguments
    & $pythonExecutable @invocationArguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Stage failed with exit code $exitCode."
    }
}

# Refuse to contend with the exact production trainer for GPU, RAM, or files.
$matchingTrainerProcesses = @(
    Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $processName = [string] $_.Name
        $commandLine = [string] $_.CommandLine
        $isTrainingEntryPoint =
            $processName -match "(?i)^nnUNetv2_train(?:\.exe)?$" -or
            $commandLine.IndexOf(
                "nnUNetv2_train",
                [StringComparison]::OrdinalIgnoreCase
            ) -ge 0
        $isMatchingRun =
            $commandLine -match "(?i)nnUNetTrainerPancreasMultiTask" -and
            $commandLine -match "(?:^|\s)501(?:\s|$)" -and
            $commandLine -match "(?:^|\s)0(?:\s|$)"
        $isTrainingEntryPoint -and $isMatchingRun
    }
)
if ($matchingTrainerProcesses.Count -gt 0) {
    $processIds = ($matchingTrainerProcesses.ProcessId | Sort-Object -Unique) -join ", "
    throw "Fixed validation refused: matching trainer process active (PID(s): $processIds)."
}


# A rescue writes its checkpoint and audit incrementally. Never evaluate while
# that writer is active, even if a previous epoch left both files present.
$matchingRescueProcesses = @(
    Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $commandLine = [string] $_.CommandLine
        $commandLine.IndexOf(
            "train_classification_rescue.py",
            [StringComparison]::OrdinalIgnoreCase
        ) -ge 0
    }
)
if ($matchingRescueProcesses.Count -gt 0) {
    $processIds = ($matchingRescueProcesses.ProcessId | Sort-Object -Unique) -join ", "
    throw "Fixed validation refused: classification rescue active (PID(s): $processIds)."
}

foreach ($requiredScript in @(
    $setupScript,
    $predictionScript,
    $evaluationScript,
    $selectionScript,
    $pythonExecutable
)) {
    Assert-LeafFile -Path $requiredScript -Description "Required executable or script"
}
foreach ($requiredDirectory in @(
    $validationImages,
    $validationLabels,
    $trainedModelRoot,
    $foldDirectory
)) {
    Assert-Directory -Path $requiredDirectory -Description "Required evaluation directory"
}
foreach ($modelMetadata in @(
    (Join-Path $trainedModelRoot "plans.json"),
    (Join-Path $trainedModelRoot "dataset.json"),
    $classificationManifest
)) {
    Assert-LeafFile -Path $modelMetadata -Description "Required model or label metadata"
}

$imageFiles = @(
    Get-ChildItem -LiteralPath $validationImages -File -Filter "*_0000.nii.gz"
)
$labelFiles = @(
    Get-ChildItem -LiteralPath $validationLabels -File -Filter "*.nii.gz"
)
if ($imageFiles.Count -ne 36) {
    throw "Expected 36 prepared validation images, found $($imageFiles.Count): '$validationImages'."
}
if ($labelFiles.Count -ne 36) {
    throw "Expected 36 validation labels, found $($labelFiles.Count): '$validationLabels'."
}
$imageCaseIds = @(
    $imageFiles | ForEach-Object {
        $_.Name.Substring(0, $_.Name.Length - "_0000.nii.gz".Length)
    }
)
$labelCaseIds = @(
    $labelFiles | ForEach-Object {
        $_.Name.Substring(0, $_.Name.Length - ".nii.gz".Length)
    }
)
$imageLabelDifferences = @(Compare-Object $imageCaseIds $labelCaseIds)
if ($imageLabelDifferences.Count -gt 0) {
    throw "Prepared validation image and label case identifiers do not match."
}

try {
    $manifestPayload = Get-Content -LiteralPath $classificationManifest -Raw | ConvertFrom-Json
}
catch {
    throw "Could not parse classification manifest '$classificationManifest': $($_.Exception.Message)"
}
$manifestValidationCases = @(
    $manifestPayload.cases | Where-Object { $_.split -eq "validation" }
)
if ($manifestValidationCases.Count -ne 36) {
    throw (
        "Expected 36 validation entries in the classification manifest, found " +
        "$($manifestValidationCases.Count): '$classificationManifest'."
    )
}
$manifestCaseIds = @($manifestValidationCases | ForEach-Object { [string] $_.case_id })
$imageManifestDifferences = @(Compare-Object $imageCaseIds $manifestCaseIds)
if ($imageManifestDifferences.Count -gt 0) {
    throw "Prepared validation image and classification-manifest identifiers do not match."
}

# The activation decision is mandatory and is checked only after proving that
# the completed checkpoint_final exists. This prevents validation from silently
# bypassing the predeclared train-only gate.
$finalCheckpointPath = Join-Path $foldDirectory "checkpoint_final.pth"
Assert-LeafFile -Path $finalCheckpointPath -Description "Completed checkpoint_final"
if ((Get-Item -LiteralPath $finalCheckpointPath).Length -le 0) {
    throw "Completed checkpoint_final is empty: '$finalCheckpointPath'."
}
$finalCheckpointSha256 = Get-FileSha256 -Path $finalCheckpointPath
$activationAudit = Read-JsonObject `
    -Path $activationAuditPath `
    -Description "Classification-rescue activation audit"
$activationAuditSha256 = Get-FileSha256 -Path $activationAuditPath

if ([int] (Get-RequiredJsonProperty $activationAudit "schema_version" "Activation audit") -ne 1) {
    throw "Activation audit has an unsupported schema_version."
}
if ([string] (Get-RequiredJsonProperty $activationAudit "source_checkpoint_name" "Activation audit") -ne "checkpoint_final.pth") {
    throw "Activation audit source_checkpoint_name must be checkpoint_final.pth."
}
Assert-Sha256Matches `
    -RecordedValue (Get-RequiredJsonProperty $activationAudit "source_checkpoint_sha256" "Activation audit") `
    -ActualValue $finalCheckpointSha256 `
    -Description "Activation audit source checkpoint SHA-256"
if ([int] (Get-RequiredJsonProperty $activationAudit "checkpoint_current_epoch" "Activation audit") -ne 200) {
    throw "Activation audit must describe checkpoint_final at current_epoch 200."
}
if ([int] (Get-RequiredJsonProperty $activationAudit "training_logging_epoch_count" "Activation audit") -ne 200) {
    throw "Activation audit must contain exactly 200 training logging epochs."
}
if ([string] (Get-RequiredJsonProperty $activationAudit "metric_scope" "Activation audit") -ne "checkpoint_training_logging_only") {
    throw "Activation audit metric_scope is not restricted to checkpoint training logging."
}
Assert-FalseJsonProperty $activationAudit "validation_metrics_read" "Activation audit"
Assert-FalseJsonProperty $activationAudit "validation_used_for_activation" "Activation audit"

$activationApproved = Get-RequiredJsonProperty `
    $activationAudit `
    "activation_approved" `
    "Activation audit"
if ($activationApproved -isnot [bool]) {
    throw "Activation audit activation_approved must be a JSON boolean."
}

if (-not $activationApproved) {
    if ($IncludeClassificationRescue) {
        throw (
            "-IncludeClassificationRescue was rejected because the train-only " +
            "activation audit did not approve rescue."
        )
    }
    Write-Host "Train-only activation audit is negative; evaluating exactly 3 candidates."
}
else {
    if (-not $IncludeClassificationRescue) {
        throw (
            "The train-only activation audit approved rescue. Complete it, then rerun " +
            "with -IncludeClassificationRescue; no validation candidate was evaluated."
        )
    }

    $activationDecisionEpoch = [int] (Get-RequiredJsonProperty `
        $activationAudit `
        "decision_epoch" `
        "Activation audit")
    if ($activationDecisionEpoch -notin @(40, 50)) {
        throw "An affirmative activation audit must record decision_epoch 40 or 50."
    }

    Assert-LeafFile `
        -Path $rescueCheckpointPath `
        -Description "Completed classification-rescue checkpoint"
    if ((Get-Item -LiteralPath $rescueCheckpointPath).Length -le 0) {
        throw "Classification-rescue checkpoint is empty: '$rescueCheckpointPath'."
    }
    $rescueAudit = Read-JsonObject `
        -Path $rescueAuditPath `
        -Description "Classification-rescue audit"
    $rescueCheckpointSha256 = Get-FileSha256 -Path $rescueCheckpointPath

    if ([int] (Get-RequiredJsonProperty $rescueAudit "schema_version" "Rescue audit") -ne 1) {
        throw "Rescue audit has an unsupported schema_version."
    }
    if ([string] (Get-RequiredJsonProperty $rescueAudit "status" "Rescue audit") -ne "complete") {
        throw "Rescue audit status must be complete before fixed validation."
    }
    if ([int] (Get-RequiredJsonProperty $rescueAudit "completed_epochs" "Rescue audit") -ne 30) {
        throw "Rescue audit must record exactly 30 completed epochs."
    }
    $rescueSchedule = Get-RequiredJsonProperty $rescueAudit "schedule" "Rescue audit"
    if ([int] (Get-RequiredJsonProperty $rescueSchedule "epochs" "Rescue schedule") -ne 30) {
        throw "Rescue schedule must declare exactly 30 epochs."
    }
    Assert-Sha256Matches `
        -RecordedValue (Get-RequiredJsonProperty $rescueAudit "source_checkpoint_sha256" "Rescue audit") `
        -ActualValue $finalCheckpointSha256 `
        -Description "Rescue audit source checkpoint SHA-256"
    Assert-Sha256Matches `
        -RecordedValue (Get-RequiredJsonProperty $rescueAudit "activation_audit_sha256" "Rescue audit") `
        -ActualValue $activationAuditSha256 `
        -Description "Rescue audit activation-audit SHA-256"
    Assert-Sha256Matches `
        -RecordedValue (Get-RequiredJsonProperty $rescueAudit "output_checkpoint_sha256" "Rescue audit") `
        -ActualValue $rescueCheckpointSha256 `
        -Description "Rescue audit output checkpoint SHA-256"
    if ([int] (Get-RequiredJsonProperty $rescueAudit "activation_decision_epoch" "Rescue audit") -ne $activationDecisionEpoch) {
        throw "Rescue audit activation_decision_epoch differs from the activation audit."
    }

    $splitAudit = Get-RequiredJsonProperty $rescueAudit "split_audit" "Rescue audit"
    if ((Get-RequiredJsonProperty $splitAudit "split_disjoint" "Rescue split audit") -ne $true) {
        throw "Rescue split audit must confirm a disjoint training/validation split."
    }
    Assert-FalseJsonProperty $splitAudit "validation_images_opened" "Rescue split audit"
    Assert-FalseJsonProperty $splitAudit "validation_used_for_gradients" "Rescue split audit"
    Assert-FalseJsonProperty $splitAudit "validation_used_for_stopping" "Rescue split audit"
    if ([int] (Get-RequiredJsonProperty $splitAudit "validation_batches_consumed" "Rescue split audit") -ne 0) {
        throw "Rescue split audit must record zero validation batches consumed."
    }

    $candidates += [pscustomobject]@{
        Name = "checkpoint_classification_rescue"
        FileName = "checkpoint_classification_rescue.pth"
    }
    Write-Host "Affirmative audit and completed rescue verified; evaluating exactly 4 candidates."
}

foreach ($candidate in $candidates) {
    $checkpointPath = Join-Path $foldDirectory $candidate.FileName
    Assert-LeafFile -Path $checkpointPath -Description "Candidate checkpoint $($candidate.Name)"
    if ((Get-Item -LiteralPath $checkpointPath).Length -le 0) {
        throw "Candidate checkpoint is empty: '$checkpointPath'."
    }
    $candidate | Add-Member -NotePropertyName CheckpointPath -NotePropertyValue $checkpointPath
}

# Dot-sourced setup affects this PowerShell process only. The standard W&B
# environment variable additionally prevents any library-level network logging.
$setupParameters = @{
    WorkRoot = $workRoot
    WandbMode = "disabled"
    DataAugmentationProcesses = 1
}
. $setupScript @setupParameters | Out-Null
$env:WANDB_MODE = "disabled"
$env:WANDB_SILENT = "true"
if ($env:nnUNet_wandb_enabled -ne "0") {
    throw "Environment setup did not disable nnU-Net W&B logging."
}

New-Item -ItemType Directory -Path $evaluationRoot -Force | Out-Null

foreach ($candidate in $candidates) {
    $candidateRoot = Join-Path $evaluationRoot $candidate.Name
    $predictionDirectory = Join-Path $candidateRoot "predictions"
    $classificationCsv = Join-Path $predictionDirectory "subtype_results.csv"
    $probabilityCsv = Join-Path $candidateRoot "subtype_probabilities.csv"
    $runtimeJson = Join-Path $candidateRoot "runtime.json"
    $metricsJson = Join-Path $candidateRoot "metrics.json"
    $caseMetricsCsv = Join-Path $candidateRoot "case_metrics.csv"
    New-Item -ItemType Directory -Path $candidateRoot -Force | Out-Null

    Write-Host ""
    Write-Host "[$($candidate.Name)] Running fixed-validation joint inference..."
    $predictionArguments = @(
        "--input", $validationImages,
        "--output", $predictionDirectory,
        "--model", $trainedModelRoot,
        "--folds", "0",
        "--checkpoint", $candidate.FileName,
        "--classification-csv", $classificationCsv,
        "--probability-csv", $probabilityCsv,
        "--device", $Device
    )
    if (-not $Force -and (Test-Path -LiteralPath $runtimeJson -PathType Leaf)) {
        $existingRuntime = Read-JsonObject `
            -Path $runtimeJson `
            -Description "$($candidate.Name) existing runtime artifact"
        if ([int] (Get-RequiredJsonProperty $existingRuntime "case_count" "Runtime artifact") -ne 36 -or
            [string] (Get-RequiredJsonProperty $existingRuntime "checkpoint" "Runtime artifact") -ne $candidate.FileName -or
            [double] (Get-RequiredJsonProperty $existingRuntime "total_seconds" "Runtime artifact") -le 0) {
            throw (
                "$($candidate.Name) has an invalid existing runtime artifact; " +
                "use -Force for a complete measured rerun."
            )
        }
        Write-Host "[$($candidate.Name)] Preserving completed first-pass runtime artifact."
    }
    else {
        $predictionArguments += @("--runtime-json", $runtimeJson)
    }
    if ($ResultsOnCpu) {
        $predictionArguments += "--results-on-cpu"
    }
    if ($Force) {
        $predictionArguments += "--overwrite"
    }
    else {
        $predictionArguments += "--no-overwrite"
    }
    Invoke-CheckedPython `
        -ScriptPath $predictionScript `
        -ScriptArguments $predictionArguments `
        -Stage "$($candidate.Name) inference"

    Write-Host "[$($candidate.Name)] Evaluating saved full-volume predictions..."
    $evaluationArguments = @(
        "--predictions", $predictionDirectory,
        "--references", $validationLabels,
        "--classification-predictions", $classificationCsv,
        "--classification-references", $classificationManifest,
        "--classification-reference-split", "validation",
        "--output-json", $metricsJson,
        "--output-csv", $caseMetricsCsv,
        "--empty-empty-dice", "1.0",
        "--bootstrap-samples", "2000",
        "--confidence", "0.95",
        "--seed", "12345"
    )
    Invoke-CheckedPython `
        -ScriptPath $evaluationScript `
        -ScriptArguments $evaluationArguments `
        -Stage "$($candidate.Name) evaluation"

    $candidate | Add-Member -NotePropertyName MetricsPath -NotePropertyValue $metricsJson
}

Write-Host ""
Write-Host "Selecting the checkpoint with the predeclared equal-weight validation score..."
$selectionArguments = @()
foreach ($candidate in $candidates) {
    $selectionArguments += @(
        "--candidate", "$($candidate.Name)=$($candidate.MetricsPath)",
        "--checkpoint", "$($candidate.Name)=$($candidate.CheckpointPath)"
    )
}
$selectionArguments += @("--output", $selectionOutput)
Invoke-CheckedPython `
    -ScriptPath $selectionScript `
    -ScriptArguments $selectionArguments `
    -Stage "checkpoint selection"

Assert-LeafFile -Path $selectionOutput -Description "Checkpoint selection artifact"
$selection = Get-Content -LiteralPath $selectionOutput -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "Fixed-validation checkpoint selection complete."
Write-Host "Selected candidate: $($selection.selected_candidate)"
Write-Host ("Selection score: {0:F6}" -f [double] $selection.selected_score)
Write-Host "Checkpoint: $($selection.selected_checkpoint_path)"
Write-Host "SHA-256: $($selection.selected_checkpoint_sha256)"
Write-Host "Selection artifact: $selectionOutput"
Write-Host "No test inference or submission ZIP was created."

[pscustomobject]@{
    SelectedCandidate = $selection.selected_candidate
    SelectionScore = [double] $selection.selected_score
    CheckpointPath = $selection.selected_checkpoint_path
    CheckpointSha256 = $selection.selected_checkpoint_sha256
    SelectionArtifact = $selectionOutput
}
}
finally {
    if ($postTrainingMutexOwned) {
        $postTrainingMutex.ReleaseMutex()
    }
    $postTrainingMutex.Dispose()
}
