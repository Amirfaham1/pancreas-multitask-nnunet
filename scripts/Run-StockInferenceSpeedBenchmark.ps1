[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $InputDirectory,

    [Parameter(Mandatory = $true)]
    [string] $ModelDirectory,

    [Parameter(Mandatory = $true)]
    [string] $NeuralCaseHeadBundle,

    [Parameter(Mandatory = $true)]
    [string] $WorkRoot,

    [Parameter(Mandatory = $true)]
    [string] $FinalCandidateLock,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string] $ExpectedFinalCandidateLockSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string] $ExpectedCheckpointSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string] $ExpectedNeuralCaseHeadBundleSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string] $ExpectedNumericTrainDatasetSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string] $ExpectedPlansSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string] $ExpectedDatasetJsonSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string] $ExpectedEncoderComponentSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string] $ExpectedDecoderComponentSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string] $ExpectedClassificationComponentSha256,

    [string] $PythonExecutable = 'D:\MLQuizWork\.venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$commonScript = Join-Path $PSScriptRoot 'V5-LockedDeliveryCommon.ps1'
$timedChildScript = Join-Path $PSScriptRoot 'run_timed_inference_child.py'
$auditScript = Join-Path $PSScriptRoot 'benchmark_stock_inference_speed.py'
$stockGatePath = Join-Path $projectRoot 'configs\inference_speed_stock_gate_v1.json'
$determinismLockPath = Join-Path $projectRoot 'configs\inference_determinism_conformance_v1.json'
$stockExportLockPath = Join-Path $projectRoot 'configs\inference_stock_export_conformance_v1.json'
$stockGateSha256 = '563d9d5e4fbe0f92653c6b7295c476d0ddf5d239c47beb1948410bbb80a7c2e2'
$determinismLockSha256 = '33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd'
$stockExportLockSha256 = 'bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503'
$expectedCaseCount = 72

if (-not (Test-Path -LiteralPath $commonScript -PathType Leaf)) {
    throw "Required common delivery script does not exist: $commonScript"
}
. $commonScript

foreach ($requiredFile in @(
    $PythonExecutable,
    $timedChildScript,
    $auditScript,
    $stockGatePath,
    $determinismLockPath,
    $stockExportLockPath
)) {
    Assert-V5LeafFile $requiredFile 'Required stock-speed executable or lock'
}
Assert-V5HashEquals (Get-V5FileSha256 $stockGatePath) $stockGateSha256 'Stock speed gate lock'
Assert-V5HashEquals (Get-V5FileSha256 $determinismLockPath) $determinismLockSha256 'Determinism lock'
Assert-V5HashEquals (Get-V5FileSha256 $stockExportLockPath) $stockExportLockSha256 'Stock export lock'

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

$resolvedWorkRoot = Assert-V5NewSeparatedOutputRoot `
    -OutputRoot $WorkRoot `
    -ProtectedPaths @(
        $projectRoot,
        $InputDirectory,
        $candidate.ModelDirectory,
        $candidate.BundlePath,
        $candidate.LockPath
    )
$ledgerPath = Get-V5BareLedgerPath `
    -Lock $candidate.Lock `
    -Stage stock_speed `
    -FinalCandidateLock $candidate.LockPath
$officialLedgerPath = Get-V5BareLedgerPath `
    -Lock $candidate.Lock `
    -Stage official_validation `
    -FinalCandidateLock $candidate.LockPath
$selectedTestLedgerPath = Get-V5BareLedgerPath `
    -Lock $candidate.Lock `
    -Stage selected_test `
    -FinalCandidateLock $candidate.LockPath
if (
    $ledgerPath.Equals($officialLedgerPath, [StringComparison]::OrdinalIgnoreCase) -or
    $ledgerPath.Equals($selectedTestLedgerPath, [StringComparison]::OrdinalIgnoreCase)
) {
    throw 'Stock-speed one-use ledger must be distinct from both official-data ledgers.'
}

$runOrder = @(
    [pscustomobject]@{ Label = 'stock_reference_1'; Arm = 'stock' },
    [pscustomobject]@{ Label = 'candidate_1'; Arm = 'candidate' },
    [pscustomobject]@{ Label = 'candidate_2'; Arm = 'candidate' },
    [pscustomobject]@{ Label = 'stock_reference_2'; Arm = 'stock' }
)
$benchmarkExecutionId = [Guid]::NewGuid().ToString('D')
$auditOutput = Join-Path $resolvedWorkRoot 'stock_inference_speed_audit.json'
$ledgerPayload = [ordered]@{
    schema_version = 1
    status = 'started_and_consumed'
    stage = 'single_locked_stock_inference_speed_benchmark'
    benchmark_execution_id = $benchmarkExecutionId
    claimed_at_utc = [DateTime]::UtcNow.ToString('o')
    orchestrator_process_id = $PID
    work_root = $resolvedWorkRoot
    final_candidate_lock = [ordered]@{
        path = $candidate.LockPath
        sha256 = $candidate.LockSha256
    }
    stock_gate_lock = [ordered]@{ path = $stockGatePath; sha256 = $stockGateSha256 }
    determinism_lock = [ordered]@{
        path = $determinismLockPath
        sha256 = $determinismLockSha256
    }
    stock_export_lock = [ordered]@{
        path = $stockExportLockPath
        sha256 = $stockExportLockSha256
    }
    run_order = @($runOrder | ForEach-Object { $_.Label })
    intended_audit_path = $auditOutput
    test_targets_or_submission_feedback_used = $false
}

$mutex = $null
try {
    $mutex = Enter-V5NamedMutex (
        'Local\PancreasMultitaskV5StockSpeed_' + $candidate.LockSha256.Substring(0, 16)
    )
    if (Test-Path -LiteralPath $resolvedWorkRoot) {
        throw "Stock benchmark WorkRoot must not already exist: $resolvedWorkRoot"
    }
    # This immutable CreateNew is deliberately before any input inventory or
    # prediction. A failed attempt still consumes the single final benchmark.
    $null = New-V5ExclusiveLedger -Path $ledgerPath -Payload $ledgerPayload
    $null = New-Item -ItemType Directory -Path $resolvedWorkRoot

    $externalRuntimePaths = [Collections.Generic.List[string]]::new()
    $outputPaths = [Collections.Generic.List[string]]::new()
    $candidateInternalRuntimePaths = [Collections.Generic.List[string]]::new()

    foreach ($run in $runOrder) {
        $runDirectory = Join-Path $resolvedWorkRoot $run.Label
        $null = New-Item -ItemType Directory -Path $runDirectory
        $outputDirectory = Join-Path $runDirectory 'output'
        $externalRuntime = Join-Path $runDirectory 'external_runtime.json'
        $determinismAudit = Join-Path $runDirectory 'determinism_bootstrap.json'
        $processLog = Join-Path $runDirectory 'process.log'
        $childArguments = @(
            $timedChildScript,
            '--execution-purpose', 'final_benchmark',
            '--run-label', $run.Label,
            '--arm', $run.Arm,
            '--input-directory', $InputDirectory,
            '--output-directory', $outputDirectory,
            '--model-directory', $candidate.ModelDirectory,
            '--external-runtime-json', $externalRuntime,
            '--determinism-audit-json', $determinismAudit,
            '--process-log', $processLog,
            '--final-candidate-lock', $candidate.LockPath,
            '--expected-final-candidate-lock-sha256', $candidate.LockSha256,
            '--one-use-ledger', $ledgerPath,
            '--benchmark-execution-id', $benchmarkExecutionId,
            '--expected-case-count', [string]$expectedCaseCount,
            '--python-executable', $PythonExecutable
        )
        if ($run.Arm -eq 'candidate') {
            $candidateRuntime = Join-Path $runDirectory 'candidate_internal_runtime.json'
            $childArguments += @(
                '--candidate-runtime-json', $candidateRuntime,
                '--neural-case-head-bundle', $candidate.BundlePath,
                '--expected-neural-case-head-bundle-sha256', $candidate.BundleSha256,
                '--expected-numeric-train-dataset-sha256', $candidate.NumericTrainDatasetSha256
            )
            $candidateInternalRuntimePaths.Add($candidateRuntime)
        }
        & $PythonExecutable @childArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Locked timed child $($run.Label) failed with exit code $LASTEXITCODE."
        }
        Assert-V5LeafFile $externalRuntime "External runtime for $($run.Label)"
        Assert-V5Directory $outputDirectory "Retained output for $($run.Label)"
        $externalRuntimePaths.Add($externalRuntime)
        $outputPaths.Add($outputDirectory)
    }

    $auditArguments = @(
        $auditScript,
        '--external-runtime'
    ) + $externalRuntimePaths.ToArray() + @(
        '--output-directory'
    ) + $outputPaths.ToArray() + @(
        '--candidate-internal-runtime'
    ) + $candidateInternalRuntimePaths.ToArray() + @(
        '--expected-case-count', [string]$expectedCaseCount,
        '--output', $auditOutput
    )
    & $PythonExecutable @auditArguments
    $auditExitCode = $LASTEXITCODE
    if ($auditExitCode -notin @(0, 2)) {
        throw "Stock speed auditor failed closed with exit code $auditExitCode."
    }
    Assert-V5LeafFile $auditOutput 'Stock speed audit artifact'
    if ($auditExitCode -eq 2) {
        throw "The final candidate did not meet the locked stock 10% speed gate: $auditOutput"
    }

    Write-Host "Stock inference-speed benchmark accepted: $auditOutput"
    Write-Host "Immutable one-use ledger: $ledgerPath"
}
finally {
    Exit-V5NamedMutex $mutex
}
