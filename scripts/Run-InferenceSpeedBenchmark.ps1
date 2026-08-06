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

    [Parameter(Mandatory = $true)]
    [string]$NeuralCaseHeadBundle,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ExpectedNeuralCaseHeadBundleSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ExpectedNumericTrainDatasetSha256,

    [string]$Checkpoint = 'checkpoint_classification_rescue.pth',
    [string]$Python = 'D:\MLQuizWork\.venv\Scripts\python.exe',
    [ValidateRange(1, 100000)]
    [int]$ExpectedCaseCount = 72
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$predictorScript = Join-Path $PSScriptRoot 'predict_joint.py'
$auditScript = Join-Path $PSScriptRoot 'benchmark_inference_speed.py'
$lockArtifact = Join-Path $repositoryRoot 'configs\inference_speed_benchmark_v3.json'
$expectedProtocolLockSha256 = '3a57ab79147a6dd9ab4ee3fa99fdb2be978e9c60f290cead7a52298673e926aa'
$lockedCheckpointSha256 = 'd7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116'
$lockedPlansSha256 = '8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f'
$lockedDatasetJsonSha256 = '4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff'
$lockedComponentHashes = [ordered]@{
    encoder = '324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1'
    decoder = 'b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2'
    classification = '1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8'
}
$env:nnUNet_extTrainer = Join-Path $repositoryRoot 'src'
$env:nnUNet_compile = 'false'

foreach ($requiredFile in @($Python, $predictorScript, $auditScript, $lockArtifact)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file does not exist: $requiredFile"
    }
}
$protocolLockHash = (Get-FileHash -LiteralPath $lockArtifact -Algorithm SHA256).Hash.ToLowerInvariant()
if ($protocolLockHash -ne $expectedProtocolLockSha256) {
    throw 'Inference-speed v3 protocol lock differs from its prospective SHA-256'
}
if (-not (Test-Path -LiteralPath $env:nnUNet_extTrainer -PathType Container)) {
    throw "Custom trainer search path does not exist: $env:nnUNet_extTrainer"
}
$resolvedInput = (Resolve-Path -LiteralPath $InputDirectory).Path
$resolvedModel = (Resolve-Path -LiteralPath $ModelDirectory).Path
$resolvedBundle = (Resolve-Path -LiteralPath $NeuralCaseHeadBundle).Path
$resolvedWorkRoot = [System.IO.Path]::GetFullPath($WorkRoot)
if (Test-Path -LiteralPath $resolvedWorkRoot) {
    throw "Benchmark WorkRoot must not already exist: $resolvedWorkRoot"
}
if ($ExpectedCheckpointSha256.ToLowerInvariant() -ne $lockedCheckpointSha256) {
    throw 'Expected checkpoint SHA-256 differs from the frozen final v5 artifact'
}
if ($Checkpoint -ne 'checkpoint_classification_rescue.pth') {
    throw 'V5 speed inference requires checkpoint_classification_rescue.pth'
}

$checkpointPath = Join-Path $resolvedModel "fold_0\$Checkpoint"
if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
    throw "Checkpoint does not exist: $checkpointPath"
}
$checkpointHashBefore = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointHashBefore -ne $ExpectedCheckpointSha256.ToLowerInvariant()) {
    throw "Checkpoint SHA-256 does not match the predeclared value"
}
$plansPath = Join-Path $resolvedModel 'plans.json'
$datasetJsonPath = Join-Path $resolvedModel 'dataset.json'
$plansHashBefore = (Get-FileHash -LiteralPath $plansPath -Algorithm SHA256).Hash.ToLowerInvariant()
$datasetJsonHashBefore = (Get-FileHash -LiteralPath $datasetJsonPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($plansHashBefore -ne $lockedPlansSha256) {
    throw 'plans.json differs from the frozen final v5 artifact'
}
if ($datasetJsonHashBefore -ne $lockedDatasetJsonSha256) {
    throw 'dataset.json differs from the frozen final v5 artifact'
}
$bundleHashBefore = (Get-FileHash -LiteralPath $resolvedBundle -Algorithm SHA256).Hash.ToLowerInvariant()
if ($bundleHashBefore -ne $ExpectedNeuralCaseHeadBundleSha256.ToLowerInvariant()) {
    throw "Neural case-head bundle SHA-256 does not match the final lock"
}

function Get-RawInputManifestJson {
    param([string]$Directory)
    $records = @(
        Get-ChildItem -LiteralPath $Directory -File |
            Where-Object { $_.Name.EndsWith('.nii.gz', [System.StringComparison]::Ordinal) } |
            Sort-Object -Property Name |
            ForEach-Object {
                [ordered]@{
                    Name = $_.Name
                    SizeBytes = $_.Length
                    Sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                }
            }
    )
    if ($records.Count -lt 1) {
        throw "No raw .nii.gz inputs found in $Directory"
    }
    return ($records | ConvertTo-Json -Compress -Depth 3)
}

$inputManifestBefore = Get-RawInputManifestJson -Directory $resolvedInput

$null = New-Item -ItemType Directory -Path $resolvedWorkRoot
Copy-Item -LiteralPath $lockArtifact -Destination (Join-Path $resolvedWorkRoot 'inference_speed_benchmark_v3.lock.json')

$runs = [ordered]@{
    reference_1 = @{ ExtractionMode = 'full' }
    candidate_1 = @{ ExtractionMode = 'neural_only' }
    candidate_2 = @{ ExtractionMode = 'neural_only' }
    reference_2 = @{ ExtractionMode = 'full' }
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
        '--classification-mode', 'neural-v5',
        '--v5-extraction-mode', $entry.Value.ExtractionMode,
        '--neural-case-head-bundle', $resolvedBundle,
        '--expected-neural-case-head-bundle-sha256', $ExpectedNeuralCaseHeadBundleSha256.ToLowerInvariant(),
        '--expected-numeric-train-dataset-sha256', $ExpectedNumericTrainDatasetSha256.ToLowerInvariant(),
        '--runtime-json', $runtimePath,
        '--probability-csv', $probabilityPath,
        '--device', 'cuda',
        '--tile-step-size', '0.5',
        '--tile-batch-size', '1',
        '--tta-batch-size', '1',
        '--overwrite'
    )
    & $Python @predictionArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Inference run $($entry.Key) failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $runtimePath -PathType Leaf)) {
        throw "Inference run $($entry.Key) did not produce runtime evidence"
    }
    $runtimeEvidence = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
    foreach ($component in $lockedComponentHashes.Keys) {
        $expectedHash = $lockedComponentHashes[$component]
        $beforeHash = $runtimeEvidence.frozen_network.component_hashes_before.$component
        $afterHash = $runtimeEvidence.frozen_network.component_hashes_after.$component
        if ($beforeHash -ne $expectedHash -or $afterHash -ne $expectedHash) {
            throw "Inference run $($entry.Key) used a non-locked $component component"
        }
    }
    if (
        $runtimeEvidence.frozen_network.frozen_components_unchanged -ne $true -or
        $runtimeEvidence.frozen_network.network_in_eval_mode -ne $true -or
        $runtimeEvidence.frozen_network.any_network_parameter_requires_grad -ne $false
    ) {
        throw "Inference run $($entry.Key) did not preserve the frozen network"
    }
}

$checkpointHashAfter = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointHashAfter -ne $checkpointHashBefore) {
    throw 'Checkpoint changed during the paired benchmark'
}
$bundleHashAfter = (Get-FileHash -LiteralPath $resolvedBundle -Algorithm SHA256).Hash.ToLowerInvariant()
if ($bundleHashAfter -ne $bundleHashBefore) {
    throw 'Neural case-head bundle changed during the paired benchmark'
}
$plansHashAfter = (Get-FileHash -LiteralPath $plansPath -Algorithm SHA256).Hash.ToLowerInvariant()
$datasetJsonHashAfter = (Get-FileHash -LiteralPath $datasetJsonPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($plansHashAfter -ne $plansHashBefore -or $datasetJsonHashAfter -ne $datasetJsonHashBefore) {
    throw 'nnU-Net plans or dataset configuration changed during the paired benchmark'
}
$inputManifestAfter = Get-RawInputManifestJson -Directory $resolvedInput
if ($inputManifestAfter -cne $inputManifestBefore) {
    throw 'Raw input inventory or content changed during the paired benchmark'
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
