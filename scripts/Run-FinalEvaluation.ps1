<#
.SYNOPSIS
Run fixed-validation inference, evaluation, and checkpoint selection.

.DESCRIPTION
This script evaluates the three predeclared checkpoint candidates on the held-out
36-case validation split. It deliberately stops before test-set inference or ZIP
creation. Existing complete cases are resumed by default; pass -Force to recompute
them. Weights & Biases is disabled for this process because this is deterministic
post-training evaluation, not a training run.

.EXAMPLE
.\scripts\Run-FinalEvaluation.ps1

.EXAMPLE
.\scripts\Run-FinalEvaluation.ps1 -Device cpu

.EXAMPLE
.\scripts\Run-FinalEvaluation.ps1 -Force -ResultsOnCpu
#>

[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string] $Device = "cuda",
    [switch] $ResultsOnCpu,
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$workRoot = [IO.Path]::GetFullPath("D:\MLQuizWork")
$datasetRoot = Join-Path $workRoot "nnUNet_raw\Dataset501_PancreasMultitask"
$validationImages = Join-Path $datasetRoot "imagesVal"
$validationLabels = Join-Path $datasetRoot "labelsVal"
$classificationManifest = Join-Path $datasetRoot "classification_manifest.json"
$trainedModelRoot = Join-Path $workRoot (
    "nnUNet_results\Dataset501_PancreasMultitask\" +
    "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres"
)
$foldDirectory = Join-Path $trainedModelRoot "fold_0"
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
        "--runtime-json", $runtimeJson,
        "--device", $Device
    )
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
