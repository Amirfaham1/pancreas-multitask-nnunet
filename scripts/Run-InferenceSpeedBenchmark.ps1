[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputDirectory,

    [Parameter(Mandatory = $true)]
    [string]$ModelDirectory,

    [Parameter(Mandatory = $true)]
    [string]$WorkRoot,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ExpectedCheckpointSha256,

    [string]$Checkpoint = 'checkpoint_classification_rescue.pth',
    [string]$Python = 'D:\MLQuizWork\.venv\Scripts\python.exe',
    [ValidateRange(1, 100000)]
    [int]$ExpectedCaseCount = 72
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$predictorScript = Join-Path $PSScriptRoot 'predict_joint.py'
$auditScript = Join-Path $PSScriptRoot 'benchmark_inference_speed.py'
$lockArtifact = Join-Path $repositoryRoot 'configs\inference_speed_benchmark.json'
$env:nnUNet_extTrainer = Join-Path $repositoryRoot 'src'
$env:nnUNet_compile = 'false'

foreach ($requiredFile in @($Python, $predictorScript, $auditScript, $lockArtifact)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file does not exist: $requiredFile"
    }
}
if (-not (Test-Path -LiteralPath $env:nnUNet_extTrainer -PathType Container)) {
    throw "Custom trainer search path does not exist: $env:nnUNet_extTrainer"
}
$resolvedInput = (Resolve-Path -LiteralPath $InputDirectory).Path
$resolvedModel = (Resolve-Path -LiteralPath $ModelDirectory).Path
$resolvedWorkRoot = [System.IO.Path]::GetFullPath($WorkRoot)
if (Test-Path -LiteralPath $resolvedWorkRoot) {
    throw "Benchmark WorkRoot must not already exist: $resolvedWorkRoot"
}

$checkpointPath = Join-Path $resolvedModel "fold_0\$Checkpoint"
if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
    throw "Checkpoint does not exist: $checkpointPath"
}
$checkpointHashBefore = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointHashBefore -ne $ExpectedCheckpointSha256.ToLowerInvariant()) {
    throw "Checkpoint SHA-256 does not match the predeclared value"
}

$null = New-Item -ItemType Directory -Path $resolvedWorkRoot
Copy-Item -LiteralPath $lockArtifact -Destination (Join-Path $resolvedWorkRoot 'inference_speed_benchmark.lock.json')

$runs = [ordered]@{
    reference_1 = @{ TileBatch = 1; TtaBatch = 1 }
    candidate_1 = @{ TileBatch = 2; TtaBatch = 2 }
    candidate_2 = @{ TileBatch = 2; TtaBatch = 2 }
    reference_2 = @{ TileBatch = 1; TtaBatch = 1 }
}

foreach ($entry in $runs.GetEnumerator()) {
    $runDirectory = Join-Path $resolvedWorkRoot $entry.Key
    $outputDirectory = Join-Path $runDirectory 'output'
    $runtimePath = Join-Path $runDirectory 'runtime.json'
    $probabilityPath = Join-Path $outputDirectory 'subtype_probabilities.csv'
    $null = New-Item -ItemType Directory -Path $runDirectory

    $predictionArguments = @(
        $predictorScript,
        '--input', $resolvedInput,
        '--output', $outputDirectory,
        '--model', $resolvedModel,
        '--folds', '0',
        '--checkpoint', $Checkpoint,
        '--runtime-json', $runtimePath,
        '--probability-csv', $probabilityPath,
        '--device', 'cuda',
        '--tile-step-size', '0.5',
        '--tile-batch-size', [string]$entry.Value.TileBatch,
        '--tta-batch-size', [string]$entry.Value.TtaBatch,
        '--overwrite'
    )
    & $Python @predictionArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Inference run $($entry.Key) failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $runtimePath -PathType Leaf)) {
        throw "Inference run $($entry.Key) did not produce runtime evidence"
    }
}

$checkpointHashAfter = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointHashAfter -ne $checkpointHashBefore) {
    throw 'Checkpoint changed during the paired benchmark'
}

$auditOutput = Join-Path $resolvedWorkRoot 'inference_speed_audit.json'
$auditArguments = @(
    $auditScript,
    '--reference-runtime',
    (Join-Path $resolvedWorkRoot 'reference_1\runtime.json'),
    (Join-Path $resolvedWorkRoot 'reference_2\runtime.json'),
    '--candidate-runtime',
    (Join-Path $resolvedWorkRoot 'candidate_1\runtime.json'),
    (Join-Path $resolvedWorkRoot 'candidate_2\runtime.json'),
    '--reference-output',
    (Join-Path $resolvedWorkRoot 'reference_1\output'),
    (Join-Path $resolvedWorkRoot 'reference_2\output'),
    '--candidate-output',
    (Join-Path $resolvedWorkRoot 'candidate_1\output'),
    (Join-Path $resolvedWorkRoot 'candidate_2\output'),
    '--expected-case-count', [string]$ExpectedCaseCount,
    '--output', $auditOutput
)
& $Python @auditArguments
$auditExitCode = $LASTEXITCODE
if ($auditExitCode -notin @(0, 2)) {
    throw "Speed audit failed with exit code $auditExitCode"
}
if (-not (Test-Path -LiteralPath $auditOutput -PathType Leaf)) {
    throw 'Speed audit did not produce its evidence artifact'
}
if ($auditExitCode -eq 2) {
    throw "The candidate did not meet the locked 10% speed gate; see $auditOutput"
}

Write-Host "Inference-speed benchmark accepted: $auditOutput"
