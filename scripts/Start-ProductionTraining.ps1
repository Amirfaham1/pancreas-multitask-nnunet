[CmdletBinding()]
param(
    [string]$WorkRoot = "D:\MLQuizWork",
    [ValidateSet("offline", "online")]
    [string]$WandbMode = "offline",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$setupScript = Join-Path $PSScriptRoot "Set-QuizEnvironment.ps1"
$trainerExe = Join-Path $WorkRoot ".venv\Scripts\nnUNetv2_train.exe"
$logsDirectory = Join-Path $WorkRoot "logs"
$foldDirectory = Join-Path $WorkRoot (
    "nnUNet_results\Dataset501_PancreasMultitask\" +
    "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres\fold_0"
)

if (-not (Test-Path -LiteralPath $trainerExe -PathType Leaf)) {
    throw "Training executable not found: $trainerExe"
}

$matchingProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine.Contains("nnUNetv2_train") -and
    $_.CommandLine.Contains("nnUNetTrainerPancreasMultiTask") -and
    $_.CommandLine -match "(?:^|\s)501(?:\s|$)"
}
if ($matchingProcesses) {
    $processIds = ($matchingProcesses.ProcessId | Sort-Object -Unique) -join ", "
    throw "A matching production training run already exists (PID(s): $processIds)."
}

if ($Resume) {
    $latestCheckpoint = Join-Path $foldDirectory "checkpoint_latest.pth"
    if (-not (Test-Path -LiteralPath $latestCheckpoint -PathType Leaf)) {
        throw "Resume requested but checkpoint_latest.pth is missing: $latestCheckpoint"
    }
}

New-Item -ItemType Directory -Path $logsDirectory -Force | Out-Null
Set-ExecutionPolicy -Scope Process Bypass -Force
. $setupScript `
    -WorkRoot $WorkRoot `
    -WandbMode $WandbMode `
    -DataAugmentationProcesses 1

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $logsDirectory "training_$timestamp.stdout.log"
$stderrPath = Join-Path $logsDirectory "training_$timestamp.stderr.log"
$arguments = @(
    "501",
    "3d_fullres",
    "0",
    "-tr",
    "nnUNetTrainerPancreasMultiTask",
    "-p",
    "nnUNetResEncUNetMPlans",
    "-device",
    "cuda"
)
if ($Resume) {
    $arguments += "--c"
}

$process = Start-Process `
    -FilePath $trainerExe `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    ProcessId = $process.Id
    StartedAt = $process.StartTime
    Resume = [bool]$Resume
    StandardOutput = $stdoutPath
    StandardError = $stderrPath
}
