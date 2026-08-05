param(
    [string]$WorkRoot = "D:\MLQuizWork",
    [ValidateSet("online", "offline", "disabled")]
    [string]$WandbMode = "offline",
    [int]$DataAugmentationProcesses = 1,
    [switch]$Continue
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
. (Join-Path $PSScriptRoot "Set-QuizEnvironment.ps1") `
    -WorkRoot $WorkRoot `
    -WandbMode $WandbMode `
    -DataAugmentationProcesses $DataAugmentationProcesses | Out-Host

$trainer = Join-Path $WorkRoot ".venv\Scripts\nnUNetv2_train.exe"
if (-not (Test-Path -LiteralPath $trainer)) {
    throw "nnUNet training executable not found: $trainer"
}

$trainingArguments = @(
    "501",
    "3d_fullres",
    "0",
    "-tr", "nnUNetTrainerPancreasMultiTask",
    "-p", "nnUNetResEncUNetMPlans",
    "-device", "cuda"
)
if ($Continue) {
    $trainingArguments += "--c"
}

Set-Location -LiteralPath $projectRoot
& $trainer @trainingArguments
if ($LASTEXITCODE -ne 0) {
    throw "nnUNet training exited with code $LASTEXITCODE"
}
