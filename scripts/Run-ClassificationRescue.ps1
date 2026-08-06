[CmdletBinding()]
param(
    [string]$WorkRoot = "D:\MLQuizWork",
    [string]$OutputCheckpoint = "",
    [string]$RecoveryAudit = "",
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Share a process-lifetime mutex with fixed validation so rescue, duplicate
# rescue launchers, and evaluation cannot contend for the model or GPU. An
# abandoned mutex is acquired safely after an unexpected process exit.
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
    if ($Resume) {
        throw (
            "Resuming this fixed rescue is prohibited: provenance permits exactly " +
            "one uninterrupted update-bearing trajectory."
        )
    }

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$setupScript = Join-Path $PSScriptRoot "Set-QuizEnvironment.ps1"
$python = Join-Path $WorkRoot ".venv\Scripts\python.exe"
$foldDirectory = Join-Path $WorkRoot (
    "nnUNet_results\Dataset501_PancreasMultitask\" +
    "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres\fold_0"
)
$resolvedFoldDirectory = [System.IO.Path]::GetFullPath($foldDirectory)
$resolvedSource = [System.IO.Path]::GetFullPath(
    (Join-Path $resolvedFoldDirectory "checkpoint_final.pth")
)
$activationAudit = Join-Path $resolvedFoldDirectory (
    "classification_rescue_activation.json"
)
$recoveryAuditExplicit = -not [string]::IsNullOrWhiteSpace($RecoveryAudit)
if ([string]::IsNullOrWhiteSpace($RecoveryAudit)) {
    $RecoveryAudit = Join-Path $resolvedFoldDirectory (
        "classification_rescue_zero_update_recovery.json"
    )
}
$resolvedRecoveryAudit = ""
if (Test-Path -LiteralPath $RecoveryAudit -PathType Leaf) {
    $resolvedRecoveryAudit = [System.IO.Path]::GetFullPath($RecoveryAudit)
}
elseif ($recoveryAuditExplicit) {
    throw "Explicit zero-update execution-recovery audit is missing: $RecoveryAudit"
}
if ([string]::IsNullOrWhiteSpace($OutputCheckpoint)) {
    $OutputCheckpoint = Join-Path $resolvedFoldDirectory (
        "checkpoint_classification_rescue.pth"
    )
}
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputCheckpoint)
$auditPath = "$resolvedOutput.audit.json"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable not found: $python"
}
if (-not (Test-Path -LiteralPath $resolvedSource -PathType Leaf)) {
    throw "Source checkpoint not found: $resolvedSource"
}
if (-not (Test-Path -LiteralPath $activationAudit -PathType Leaf)) {
    throw "Affirmative train-only activation audit is missing: $activationAudit"
}
if ([System.IO.Path]::GetDirectoryName($resolvedSource) -ne $resolvedFoldDirectory) {
    throw "Source checkpoint must be a direct child of the production fold directory."
}
if ([System.IO.Path]::GetDirectoryName($resolvedOutput) -ne $resolvedFoldDirectory) {
    throw "Output checkpoint must be a direct child of the production fold directory."
}
if (-not [string]::IsNullOrWhiteSpace($resolvedRecoveryAudit)) {
    if ([System.IO.Path]::GetDirectoryName($resolvedRecoveryAudit) -ne
        $resolvedFoldDirectory) {
        throw "Recovery audit must be a direct child of the production fold directory."
    }
    if ([System.IO.Path]::GetFileName($resolvedRecoveryAudit) -cne
        "classification_rescue_zero_update_recovery.json") {
        throw "Recovery audit has an unexpected filename: $resolvedRecoveryAudit"
    }
}
if ($resolvedSource -eq $resolvedOutput) {
    throw "Source and output checkpoints must differ."
}

$activeTraining = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine.Contains("nnUNetv2_train") -and
    $_.CommandLine.Contains("nnUNetTrainerPancreasMultiTask") -and
    $_.CommandLine -match "(?:^|\s)501(?:\s|$)"
}
if ($activeTraining) {
    $processIds = ($activeTraining.ProcessId | Sort-Object -Unique) -join ", "
    throw "Production training is still active (PID(s): $processIds)."
}

$activeRescue = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains("train_classification_rescue.py")
}
if ($activeRescue) {
    $processIds = ($activeRescue.ProcessId | Sort-Object -Unique) -join ", "
    throw "A classification rescue is already active (PID(s): $processIds)."
}

if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Output checkpoint already exists; this fixed rescue cannot restart or resume."
}

if (-not [string]::IsNullOrWhiteSpace($resolvedRecoveryAudit)) {
    $recoveryTool = Join-Path $PSScriptRoot "classification_rescue_recovery.py"
    & $python $recoveryTool validate `
        --recovery-audit $resolvedRecoveryAudit `
        --source-checkpoint $resolvedSource `
        --activation-audit $activationAudit
    if ($LASTEXITCODE -ne 0) {
        throw "Zero-update execution-recovery provenance failed validation."
    }
}

Set-ExecutionPolicy -Scope Process Bypass -Force
. $setupScript `
    -WorkRoot $WorkRoot `
    -WandbMode disabled `
    -DataAugmentationProcesses 0 | Out-Null

$arguments = @(
    (Join-Path $PSScriptRoot "train_classification_rescue.py"),
    "--source-checkpoint", $resolvedSource,
    "--output-checkpoint", $resolvedOutput,
    "--activation-audit", $activationAudit,
    "--audit-json", $auditPath,
    "--dataset", "501",
    "--configuration", "3d_fullres",
    "--fold", "0",
    "--trainer", "nnUNetTrainerPancreasMultiTask",
    "--plans", "nnUNetResEncUNetMPlans",
    "--device", "cuda",
    "--epochs", "30",
    "--iterations-per-epoch", "125",
    "--learning-rate", "0.0003",
    "--weight-decay", "0.0001",
    "--gradient-clip-norm", "1.0",
    "--label-smoothing", "0.05",
    "--nonlesion-patch-weight", "0.25",
    "--reset-seed", "20260806",
    "--expected-training-cases", "252",
    "--expected-validation-cases", "36",
    "--save-every", "1"
)
if (-not [string]::IsNullOrWhiteSpace($resolvedRecoveryAudit)) {
    $arguments += @("--recovery-audit", $resolvedRecoveryAudit)
}
Set-Location -LiteralPath $repoRoot
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Classification rescue exited with code $LASTEXITCODE"
}

[pscustomobject]@{
    SourceCheckpoint = $resolvedSource
    RescueCheckpoint = $resolvedOutput
    Audit = $auditPath
    RecoveryAudit = if ([string]::IsNullOrWhiteSpace($resolvedRecoveryAudit)) {
        $null
    } else {
        $resolvedRecoveryAudit
    }
    Resumed = $false
}
}
finally {
    if ($postTrainingMutexOwned) {
        $postTrainingMutex.ReleaseMutex()
    }
    $postTrainingMutex.Dispose()
}
