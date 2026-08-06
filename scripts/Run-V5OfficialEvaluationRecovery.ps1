<#
.SYNOPSIS
Continue the consumed v5 official-validation run from its saved predictions.

.DESCRIPTION
This is a narrowly locked recovery for the completed official inference whose
PowerShell 5.1 audit failed before any reference access. It never invokes an
inference entry point. Before either reference path is tested, it verifies the
frozen 40-file output snapshot, replays the original runtime validator with
exactly two in-memory collection-count substitutions, hashes the 39 inference
artifacts, snapshots the original ledger byte-for-byte, and creates the normal
pre-reference manifest plus a durable evaluation-start record. It then invokes
the unchanged evaluator exactly once and preserves the original gate/ledger
schemas expected by the selected-test wrapper.

The separate execution lock must use this schema:
  schema_version: 1
  status: frozen_after_recovery_implementation_before_any_official_reference_access
  recovery_implementation_commit: 40 lowercase hexadecimal digits
  recovery_protocol_sha256, recovery_wrapper_sha256, recovery_tests_sha256
  final_candidate_lock_sha256, original_common_sha256,
  original_official_wrapper_sha256, evaluator_sha256
  python_executable_path, python_executable_sha256, metrics_source_path,
  metrics_source_sha256, validation_images_path, reference_masks_path,
  reference_subtypes_path, output_root
  immutable_pre_recovery_evidence: the exact hash/count fields checked below
  execution_contract: recovery_inference_invocation_count=0,
    total_official_inference_invocation_count=1,
    official_evaluator_invocation_count=1, and test_data_access_permitted=false
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $RecoveryProtocol,
    [Parameter(Mandatory)]
    [string] $ExpectedRecoveryProtocolSha256,
    [Parameter(Mandatory)]
    [string] $RecoveryExecutionLock,
    [Parameter(Mandatory)]
    [string] $ExpectedRecoveryExecutionLockSha256,
    [Parameter(Mandatory)]
    [string] $FinalCandidateLock,
    [Parameter(Mandatory)]
    [string] $ExpectedFinalCandidateLockSha256,
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
    [string] $ValidationImages,
    [Parameter(Mandatory)]
    [string] $ReferenceMasks,
    [Parameter(Mandatory)]
    [string] $ReferenceSubtypes,
    [Parameter(Mandatory)]
    [string] $OutputRoot,
    [string] $WorkRoot = "D:\MLQuizWork",
    [string] $PythonExecutable = "D:\MLQuizWork\.venv\Scripts\python.exe",
    [ValidateSet("cuda")]
    [string] $Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:V5RecoveryProtocolSha256 =
    "a2e5ff8bffcf5a07ba797cabe1d245b00e4a352c06f9ffd4b8246c43d5646fdd"
$script:V5RecoveryMetricsSourceSha256 =
    "b357df75a8502972139dc45d78ea8c05683ff7d46cb3bb00f2fdc2b73118d5a9"

function ConvertTo-RecoverySha256 {
    param(
        [Parameter(Mandatory)]
        [object] $Value,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $text = ([string] $Value).Trim().ToLowerInvariant()
    if ($text -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Description must be one lowercase 64-digit SHA-256 digest."
    }
    return $text
}

function Get-RecoveryFileSha256 {
    param([Parameter(Mandatory)][string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required recovery file does not exist: '$Path'."
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RecoveryStringSha256 {
    param([Parameter(Mandatory)][AllowEmptyString()][string] $Value)

    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
        return ([BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Get-RecoveryNormalizedPath {
    param([Parameter(Mandatory)][string] $Path)

    return [IO.Path]::GetFullPath($Path).TrimEnd([char[]] "\/")
}

function Assert-RecoveryPathEqual {
    param(
        [Parameter(Mandatory)][string] $Actual,
        [Parameter(Mandatory)][string] $Expected,
        [Parameter(Mandatory)][string] $Description
    )

    if (-not (Get-RecoveryNormalizedPath $Actual).Equals(
        (Get-RecoveryNormalizedPath $Expected),
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Description differs from the prospectively bound path."
    }
}

function Test-RecoveryPathAtOrBelow {
    param(
        [Parameter(Mandatory)][string] $Candidate,
        [Parameter(Mandatory)][string] $Parent
    )

    $candidatePath = Get-RecoveryNormalizedPath $Candidate
    $parentPath = Get-RecoveryNormalizedPath $Parent
    return (
        $candidatePath.Equals($parentPath, [StringComparison]::OrdinalIgnoreCase) -or
        $candidatePath.StartsWith(
            $parentPath + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        )
    )
}

function Assert-RecoveryEqual {
    param(
        [Parameter(Mandatory)][object] $Actual,
        [Parameter(Mandatory)][object] $Expected,
        [Parameter(Mandatory)][string] $Description
    )

    if ([string] $Actual -cne [string] $Expected) {
        throw "$Description differs from the frozen recovery contract."
    }
}

function Assert-RecoveryHash {
    param(
        [Parameter(Mandatory)][object] $Actual,
        [Parameter(Mandatory)][object] $Expected,
        [Parameter(Mandatory)][string] $Description
    )

    $actualHash = ConvertTo-RecoverySha256 $Actual "$Description actual hash"
    $expectedHash = ConvertTo-RecoverySha256 $Expected "$Description expected hash"
    if ($actualHash -cne $expectedHash) {
        throw "$Description SHA-256 mismatch."
    }
}

function Get-RecoveryRequiredProperty {
    param(
        [Parameter(Mandatory)][object] $Object,
        [Parameter(Mandatory)][string] $Name,
        [Parameter(Mandatory)][string] $Description
    )

    if ($null -eq $Object) {
        throw "$Description is null."
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Description is missing required property '$Name'."
    }
    return $property.Value
}

function Read-RecoveryJson {
    param(
        [Parameter(Mandatory)][string] $Path,
        [Parameter(Mandatory)][string] $Description
    )

    try {
        $payload = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        throw "$Description is not valid JSON: $($_.Exception.Message)"
    }
    if ($null -eq $payload -or $payload -is [System.Array]) {
        throw "$Description must be one JSON object."
    }
    return $payload
}

function Write-RecoveryBytesCreateNew {
    param(
        [Parameter(Mandatory)][string] $Path,
        [Parameter(Mandatory)][byte[]] $Bytes
    )

    $destination = [IO.Path]::GetFullPath($Path)
    $parent = [IO.Path]::GetDirectoryName($destination)
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        throw "CreateNew parent directory does not exist: '$parent'."
    }
    $stream = [IO.File]::Open(
        $destination,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None
    )
    try {
        $stream.Write($Bytes, 0, $Bytes.Length)
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
    }
    return $destination
}

function Write-RecoveryJsonCreateNew {
    param(
        [Parameter(Mandatory)][string] $Path,
        [Parameter(Mandatory)][object] $Payload
    )

    $json = $Payload | ConvertTo-Json -Depth 32
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json + "`n")
    return Write-RecoveryBytesCreateNew -Path $Path -Bytes $bytes
}

function Copy-RecoveryFileBytesCreateNew {
    param(
        [Parameter(Mandatory)][string] $Source,
        [Parameter(Mandatory)][string] $Destination,
        [Parameter(Mandatory)][string] $ExpectedSha256
    )

    Assert-RecoveryHash (Get-RecoveryFileSha256 $Source) $ExpectedSha256 "Snapshot source"
    $bytes = [IO.File]::ReadAllBytes([IO.Path]::GetFullPath($Source))
    $resolved = Write-RecoveryBytesCreateNew -Path $Destination -Bytes $bytes
    Assert-RecoveryHash (Get-RecoveryFileSha256 $resolved) $ExpectedSha256 "Byte-identical ledger snapshot"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $Source) $ExpectedSha256 "Snapshot source after copy"
    return $resolved
}

function Write-RecoveryJsonAtomicWithBackup {
    param(
        [Parameter(Mandatory)][string] $Path,
        [Parameter(Mandatory)][object] $Payload,
        [Parameter(Mandatory)][string] $BackupPath,
        [Parameter(Mandatory)][string] $ExpectedPreviousSha256
    )

    $destination = [IO.Path]::GetFullPath($Path)
    $backup = [IO.Path]::GetFullPath($BackupPath)
    if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
        throw "Atomic-replacement destination does not exist: '$destination'."
    }
    if (Test-Path -LiteralPath $backup) {
        throw "Atomic-replacement backup already exists: '$backup'."
    }
    Assert-RecoveryHash (Get-RecoveryFileSha256 $destination) $ExpectedPreviousSha256 "Atomic-replacement previous bytes"
    $parent = [IO.Path]::GetDirectoryName($destination)
    if ([IO.Path]::GetPathRoot($backup) -cne [IO.Path]::GetPathRoot($destination)) {
        throw "Atomic-replacement backup must be on the destination volume."
    }
    $temporary = Join-Path $parent (".{0}.{1}.recovery.tmp" -f [IO.Path]::GetFileName($destination), [Guid]::NewGuid().ToString("N"))
    try {
        $json = $Payload | ConvertTo-Json -Depth 32
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json + "`n")
        $null = Write-RecoveryBytesCreateNew -Path $temporary -Bytes $bytes
        [IO.File]::Replace($temporary, $destination, $backup)
        Assert-RecoveryHash (Get-RecoveryFileSha256 $backup) $ExpectedPreviousSha256 "Atomic-replacement preserved backup"
    }
    finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
    return Get-RecoveryFileSha256 $destination
}

function Get-RecoveryRecursiveSnapshot {
    param([Parameter(Mandatory)][string] $Root)

    $resolvedRoot = [IO.Path]::GetFullPath($Root).TrimEnd([char[]] "\/")
    if (-not (Test-Path -LiteralPath $resolvedRoot -PathType Container)) {
        throw "Frozen recovery output root does not exist: '$resolvedRoot'."
    }
    $entries = @()
    foreach ($item in @(Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File)) {
        $relative = $item.FullName.Substring($resolvedRoot.Length).TrimStart(
            [IO.Path]::DirectorySeparatorChar,
            [IO.Path]::AltDirectorySeparatorChar
        ).Replace("\", "/")
        $entries += [pscustomobject]@{
            relative_path = $relative
            size_bytes = [int64] $item.Length
            sha256 = Get-RecoveryFileSha256 $item.FullName
        }
    }
    $ordered = @($entries | Sort-Object relative_path)
    $canonical = ($ordered | ForEach-Object {
        $_.relative_path + [char] 0 + [string] $_.size_bytes + [char] 0 + $_.sha256 + "`n"
    }) -join ""
    return [pscustomobject]@{
        FileCount = $ordered.Count
        SnapshotSha256 = Get-RecoveryStringSha256 $canonical
        Entries = $ordered
    }
}

function Assert-RecoveryBoolean {
    param(
        [Parameter(Mandatory)][object] $Value,
        [Parameter(Mandatory)][bool] $Expected,
        [Parameter(Mandatory)][string] $Description
    )

    if ($Value -isnot [bool] -or [bool] $Value -ne $Expected) {
        throw "$Description must be $Expected."
    }
}

function Assert-RecoveryImplementationCommitPublished {
    param(
        [Parameter(Mandatory)][string] $ProjectRoot,
        [Parameter(Mandatory)][string] $Commit,
        [Parameter(Mandatory)][string[]] $RelativePaths
    )

    if ($null -eq (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is required to verify the committed and pushed recovery implementation."
    }
    & git -C $ProjectRoot cat-file -e "$Commit^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Recovery implementation commit is not a local commit object."
    }
    & git -C $ProjectRoot merge-base --is-ancestor $Commit origin/main 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Recovery implementation commit is not published on origin/main."
    }
    foreach ($relativePath in $RelativePaths) {
        & git -C $ProjectRoot diff --quiet $Commit -- $relativePath
        if ($LASTEXITCODE -ne 0) {
            throw "Recovery file '$relativePath' differs from the published implementation commit."
        }
        $status = @(& git -C $ProjectRoot status --porcelain --untracked-files=all -- $relativePath)
        if ($LASTEXITCODE -ne 0 -or $status.Count -ne 0) {
            throw "Recovery file '$relativePath' has an uncommitted or untracked change."
        }
    }
}

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$commonScript = Join-Path $PSScriptRoot "V5-LockedDeliveryCommon.ps1"
$originalWrapper = Join-Path $PSScriptRoot "Run-V5LockedFinalEvaluation.ps1"
$evaluationScript = Join-Path $PSScriptRoot "evaluate_predictions.py"
$metricsScript = Join-Path $projectRoot "src\pancreas_multitask\metrics.py"
$setupScript = Join-Path $PSScriptRoot "Set-QuizEnvironment.ps1"
$recoveryTests = Join-Path $projectRoot "tests\test_v5_official_evaluation_recovery.py"
$thisWrapper = [IO.Path]::GetFullPath($PSCommandPath)

foreach ($required in @(
    $RecoveryProtocol,
    $RecoveryExecutionLock,
    $FinalCandidateLock,
    $commonScript,
    $originalWrapper,
    $evaluationScript,
    $metricsScript,
    $setupScript,
    $recoveryTests,
    $PythonExecutable
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required recovery code, lock, or executable is missing: '$required'."
    }
}

$protocolPath = [IO.Path]::GetFullPath($RecoveryProtocol)
$executionLockPath = [IO.Path]::GetFullPath($RecoveryExecutionLock)
$protocolSha256 = ConvertTo-RecoverySha256 $ExpectedRecoveryProtocolSha256 "Expected recovery-protocol hash"
$executionLockSha256 = ConvertTo-RecoverySha256 $ExpectedRecoveryExecutionLockSha256 "Expected recovery execution-lock hash"
Assert-RecoveryHash $protocolSha256 $script:V5RecoveryProtocolSha256 "Hard-bound recovery protocol"
Assert-RecoveryHash (Get-RecoveryFileSha256 $protocolPath) $protocolSha256 "Recovery protocol"
Assert-RecoveryHash (Get-RecoveryFileSha256 $executionLockPath) $executionLockSha256 "Recovery execution lock"
$protocol = Read-RecoveryJson $protocolPath "Recovery protocol"
$executionLock = Read-RecoveryJson $executionLockPath "Recovery execution lock"

if ([int] (Get-RecoveryRequiredProperty $protocol "schema_version" "Recovery protocol") -ne 1) {
    throw "Recovery protocol schema_version must be 1."
}
Assert-RecoveryEqual `
    (Get-RecoveryRequiredProperty $protocol "status" "Recovery protocol") `
    "frozen_before_recovery_implementation_or_any_official_reference_access" `
    "Recovery protocol status"
if ([int] (Get-RecoveryRequiredProperty $executionLock "schema_version" "Recovery execution lock") -ne 1) {
    throw "Recovery execution-lock schema_version must be 1."
}
Assert-RecoveryEqual `
    (Get-RecoveryRequiredProperty $executionLock "status" "Recovery execution lock") `
    "frozen_after_recovery_implementation_before_any_official_reference_access" `
    "Recovery execution-lock status"

$boundOriginal = Get-RecoveryRequiredProperty $protocol "bound_original_implementation" "Recovery protocol"
$protocolEvidence = Get-RecoveryRequiredProperty $protocol "immutable_pre_recovery_evidence" "Recovery protocol"
$protocolContract = Get-RecoveryRequiredProperty $protocol "recovery_execution_contract" "Recovery protocol"
$protocolPolicy = Get-RecoveryRequiredProperty $protocol "recovery_implementation_policy" "Recovery protocol"
$protocolFiles = Get-RecoveryRequiredProperty $protocol "recovery_files" "Recovery protocol"
$protocolFinalCandidate = Get-RecoveryRequiredProperty $protocol "final_candidate" "Recovery protocol"

Assert-RecoveryHash (Get-RecoveryFileSha256 $commonScript) (Get-RecoveryRequiredProperty $boundOriginal "common_sha256" "Bound implementation") "Original Common"
Assert-RecoveryHash (Get-RecoveryFileSha256 $originalWrapper) (Get-RecoveryRequiredProperty $boundOriginal "official_wrapper_sha256" "Bound implementation") "Original official wrapper"
Assert-RecoveryHash (Get-RecoveryFileSha256 $evaluationScript) (Get-RecoveryRequiredProperty $boundOriginal "evaluator_sha256" "Bound implementation") "Original evaluator"
Assert-RecoveryHash (Get-RecoveryFileSha256 $FinalCandidateLock) (Get-RecoveryRequiredProperty $protocolFinalCandidate "lock_sha256" "Recovery final candidate") "Final-candidate lock"
Assert-RecoveryHash (Get-RecoveryFileSha256 $FinalCandidateLock) $ExpectedFinalCandidateLockSha256 "Caller final-candidate lock"

$expectedImplementationCommit = [string] (Get-RecoveryRequiredProperty $executionLock "recovery_implementation_commit" "Recovery execution lock")
if ($expectedImplementationCommit -cnotmatch '^[0-9a-f]{40}$') {
    throw "Recovery implementation commit must be one lowercase 40-digit Git object name."
}
Assert-RecoveryImplementationCommitPublished `
    -ProjectRoot $projectRoot `
    -Commit $expectedImplementationCommit `
    -RelativePaths @(
        "configs/official_evaluation_recovery_protocol_v1.json",
        "scripts/Run-V5OfficialEvaluationRecovery.ps1",
        "tests/test_v5_official_evaluation_recovery.py",
        "src/pancreas_multitask/metrics.py"
    )
Assert-RecoveryHash (Get-RecoveryRequiredProperty $executionLock "recovery_protocol_sha256" "Recovery execution lock") $protocolSha256 "Execution-lock protocol binding"
Assert-RecoveryHash (Get-RecoveryRequiredProperty $executionLock "recovery_wrapper_sha256" "Recovery execution lock") (Get-RecoveryFileSha256 $thisWrapper) "Execution-lock wrapper binding"
Assert-RecoveryHash (Get-RecoveryRequiredProperty $executionLock "recovery_tests_sha256" "Recovery execution lock") (Get-RecoveryFileSha256 $recoveryTests) "Execution-lock tests binding"
Assert-RecoveryHash (Get-RecoveryRequiredProperty $executionLock "final_candidate_lock_sha256" "Recovery execution lock") (Get-RecoveryFileSha256 $FinalCandidateLock) "Execution-lock candidate binding"
Assert-RecoveryHash (Get-RecoveryRequiredProperty $executionLock "original_common_sha256" "Recovery execution lock") (Get-RecoveryFileSha256 $commonScript) "Execution-lock Common binding"
Assert-RecoveryHash (Get-RecoveryRequiredProperty $executionLock "original_official_wrapper_sha256" "Recovery execution lock") (Get-RecoveryFileSha256 $originalWrapper) "Execution-lock original-wrapper binding"
Assert-RecoveryHash (Get-RecoveryRequiredProperty $executionLock "evaluator_sha256" "Recovery execution lock") (Get-RecoveryFileSha256 $evaluationScript) "Execution-lock evaluator binding"
Assert-RecoveryPathEqual (Get-RecoveryRequiredProperty $executionLock "metrics_source_path" "Recovery execution lock") $metricsScript "Execution-lock metrics-source path"
Assert-RecoveryHash (Get-RecoveryRequiredProperty $executionLock "metrics_source_sha256" "Recovery execution lock") $script:V5RecoveryMetricsSourceSha256 "Execution-lock frozen metrics source"
Assert-RecoveryHash (Get-RecoveryFileSha256 $metricsScript) $script:V5RecoveryMetricsSourceSha256 "Current frozen metrics source"
$evaluatedImplementationCommit = [string] (Get-RecoveryRequiredProperty $protocolFinalCandidate "evaluated_implementation_commit" "Recovery final candidate")
& git -C $projectRoot diff --quiet $evaluatedImplementationCommit -- "src/pancreas_multitask/metrics.py"
if ($LASTEXITCODE -ne 0) {
    throw "Metrics source differs from the final candidate's evaluated implementation commit."
}
Assert-RecoveryEqual (Get-RecoveryRequiredProperty $executionLock "python_executable_path" "Recovery execution lock") ([IO.Path]::GetFullPath($PythonExecutable)) "Execution-lock Python path"
Assert-RecoveryHash (Get-RecoveryRequiredProperty $executionLock "python_executable_sha256" "Recovery execution lock") (Get-RecoveryFileSha256 $PythonExecutable) "Execution-lock Python binding"

$resolvedOutputRoot = Get-RecoveryNormalizedPath $OutputRoot
$resolvedValidationImages = Get-RecoveryNormalizedPath $ValidationImages
$resolvedReferenceMasks = Get-RecoveryNormalizedPath $ReferenceMasks
$resolvedReferenceSubtypes = Get-RecoveryNormalizedPath $ReferenceSubtypes
Assert-RecoveryPathEqual `
    (Get-RecoveryRequiredProperty $executionLock "validation_images_path" "Recovery execution lock") `
    $resolvedValidationImages `
    "Execution-lock validation-images path"
Assert-RecoveryPathEqual `
    (Get-RecoveryRequiredProperty $executionLock "reference_masks_path" "Recovery execution lock") `
    $resolvedReferenceMasks `
    "Execution-lock reference-masks path"
Assert-RecoveryPathEqual `
    (Get-RecoveryRequiredProperty $executionLock "reference_subtypes_path" "Recovery execution lock") `
    $resolvedReferenceSubtypes `
    "Execution-lock reference-subtypes path"

# These are lexical path checks only: neither reference path is tested or opened.
$datasetRoot = [IO.Path]::GetDirectoryName($resolvedValidationImages)
Assert-RecoveryPathEqual $resolvedValidationImages (Join-Path $datasetRoot "imagesVal") "Official imagesVal path"
Assert-RecoveryPathEqual $resolvedReferenceMasks (Join-Path $datasetRoot "labelsVal") "Official labelsVal path"
Assert-RecoveryPathEqual $resolvedReferenceSubtypes (Join-Path $datasetRoot "classification_manifest.json") "Official classification-manifest path"
foreach ($protectedPath in @(
    $resolvedValidationImages,
    $resolvedReferenceMasks,
    $resolvedReferenceSubtypes
)) {
    if ((Test-RecoveryPathAtOrBelow -Candidate $protectedPath -Parent $resolvedOutputRoot) -or
        (Test-RecoveryPathAtOrBelow -Candidate $resolvedOutputRoot -Parent $protectedPath)) {
        throw "Recovery output root and official input/reference paths must be bidirectionally disjoint."
    }
}
Assert-RecoveryEqual (Get-RecoveryRequiredProperty $protocolEvidence "output_root" "Recovery evidence") $resolvedOutputRoot "Protocol output root"
Assert-RecoveryEqual (Get-RecoveryRequiredProperty $executionLock "output_root" "Recovery execution lock") $resolvedOutputRoot "Execution-lock output root"

$executionEvidence = Get-RecoveryRequiredProperty $executionLock "immutable_pre_recovery_evidence" "Recovery execution lock"
foreach ($field in @(
    "original_ledger_sha256",
    "candidate_preflight_sha256",
    "runtime_sha256",
    "classification_csv_sha256",
    "probability_csv_sha256",
    "validated_inference_artifact_set_sha256",
    "full_existing_output_snapshot_sha256"
)) {
    Assert-RecoveryHash `
        (Get-RecoveryRequiredProperty $executionEvidence $field "Execution-lock evidence") `
        (Get-RecoveryRequiredProperty $protocolEvidence $field "Protocol evidence") `
        "Execution-lock evidence '$field'"
}
foreach ($field in @("validated_inference_artifact_count", "full_existing_output_file_count", "mask_count")) {
    if ([int] (Get-RecoveryRequiredProperty $executionEvidence $field "Execution-lock evidence") -ne
        [int] (Get-RecoveryRequiredProperty $protocolEvidence $field "Protocol evidence")) {
        throw "Execution-lock evidence '$field' differs from the protocol."
    }
}
$executionContract = Get-RecoveryRequiredProperty $executionLock "execution_contract" "Recovery execution lock"
foreach ($contractObject in @($protocolContract, $executionContract)) {
    if ([int] (Get-RecoveryRequiredProperty $contractObject "recovery_inference_invocation_count" "Recovery execution contract") -ne 0 -or
        [int] (Get-RecoveryRequiredProperty $contractObject "total_official_inference_invocation_count" "Recovery execution contract") -ne 1 -or
        [int] (Get-RecoveryRequiredProperty $contractObject "official_evaluator_invocation_count" "Recovery execution contract") -ne 1) {
        throw "Recovery execution counts differ from the frozen no-rerun contract."
    }
    Assert-RecoveryBoolean (Get-RecoveryRequiredProperty $contractObject "test_data_access_permitted" "Recovery execution contract") $false "Recovery test-data permission"
}

# The original Common is hash-verified before it is executed.
. $commonScript

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

# Set up the already-frozen environment without opening an official input or target.
$null = . $setupScript -WorkRoot $WorkRoot -WandbMode disabled -DataAugmentationProcesses 1

$recoveryLedgerRelative = [string] (Get-RecoveryRequiredProperty $protocolFiles "one_use_ledger" "Recovery files")
Assert-RecoveryEqual $recoveryLedgerRelative "configs/official_validation_recovery_run_consumed.json" "Recovery ledger path"
$recoveryLedgerPath = [IO.Path]::GetFullPath((Join-Path $projectRoot ($recoveryLedgerRelative.Replace("/", "\"))))
$originalLedgerRelative = [string] (Get-RecoveryRequiredProperty $protocolEvidence "original_ledger_path" "Recovery evidence")
$originalLedgerPath = [IO.Path]::GetFullPath((Join-Path $projectRoot ($originalLedgerRelative.Replace("/", "\"))))
$expectedOriginalLedgerSha256 = ConvertTo-RecoverySha256 (Get-RecoveryRequiredProperty $protocolEvidence "original_ledger_sha256" "Recovery evidence") "Frozen original-ledger hash"
Assert-RecoveryHash (Get-RecoveryFileSha256 $originalLedgerPath) $expectedOriginalLedgerSha256 "Original consumed ledger"

$mutex = $null
$recoveryLedgerCreated = $false
$preReferenceManifestWritten = $false
$originalLedgerUpdated = $false
$originalLedgerStage = "frozen_original"
$currentOriginalLedgerSha256 = $expectedOriginalLedgerSha256
$originalLedgerOriginalBackup = $originalLedgerPath + ".recovery_original_backup.json"
$originalLedgerPredictionsFrozenBackup = $originalLedgerPath + ".recovery_predictions_frozen_backup.json"
$originalLedgerEvaluationStartedBackup = $originalLedgerPath + ".recovery_evaluation_started_backup.json"
$evaluationInvocationCommitted = $false
$evaluationInvocationCount = 0
$referenceAccessStarted = $false
$startedAtUtc = [DateTime]::UtcNow.ToString("o")
$currentStage = "verified_recovery_locks"
try {
    $mutex = Enter-V5NamedMutex (
        "Local\PancreasMultitaskV5OfficialRecovery_" + $candidate.LockSha256.Substring(0, 16)
    )

    # Recheck every lock-bound implementation inside the mutex before consuming recovery.
    Assert-RecoveryHash (Get-RecoveryFileSha256 $protocolPath) $protocolSha256 "Recovery protocol inside mutex"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $executionLockPath) $executionLockSha256 "Recovery execution lock inside mutex"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $thisWrapper) (Get-RecoveryRequiredProperty $executionLock "recovery_wrapper_sha256" "Recovery execution lock") "Recovery wrapper inside mutex"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $originalLedgerPath) $expectedOriginalLedgerSha256 "Original ledger before recovery consumption"

    $currentStage = "recovery_ledger_consumption"
    $recoveryLedgerPayload = [ordered]@{
        schema_version = 1
        status = "started_and_consumed_no_recovery_rerun"
        stage = "saved_output_official_evaluation_recovery"
        started_at_utc = $startedAtUtc
        recovery_protocol_sha256 = $protocolSha256
        recovery_execution_lock_sha256 = $executionLockSha256
        final_candidate_lock_sha256 = $candidate.LockSha256
        output_root = $resolvedOutputRoot
        validation_images_path = $resolvedValidationImages
        reference_masks_path = $resolvedReferenceMasks
        reference_subtypes_path = $resolvedReferenceSubtypes
        recovery_inference_invocation_count = 0
        total_official_inference_invocation_count = 1
        official_evaluation_invocation_count = 0
        reference_access_started = $false
    }
    $recoveryLedgerPath = New-V5ExclusiveLedger -Path $recoveryLedgerPath -Payload $recoveryLedgerPayload
    $recoveryLedgerCreated = $true

    # This check must precede creation of any recovery evidence under OutputRoot.
    $currentStage = "frozen_40_file_snapshot_verification"
    $fullSnapshot = Get-RecoveryRecursiveSnapshot -Root $resolvedOutputRoot
    if ($fullSnapshot.FileCount -ne [int] (Get-RecoveryRequiredProperty $protocolEvidence "full_existing_output_file_count" "Recovery evidence")) {
        throw "Existing saved-output file count differs from the frozen 40-file snapshot."
    }
    Assert-RecoveryHash $fullSnapshot.SnapshotSha256 (Get-RecoveryRequiredProperty $protocolEvidence "full_existing_output_snapshot_sha256" "Recovery evidence") "Frozen saved-output snapshot"

    $predictionDirectory = Join-Path $resolvedOutputRoot "predictions"
    $evidenceDirectory = Join-Path $resolvedOutputRoot "evidence"
    $classificationCsv = Join-Path $predictionDirectory "subtype_results.csv"
    $probabilityCsv = Join-Path $evidenceDirectory "subtype_probabilities.csv"
    $runtimeJson = Join-Path $evidenceDirectory "runtime.json"
    $preflightJson = Join-Path $evidenceDirectory "candidate_preflight.json"
    $preReferenceManifest = Join-Path $resolvedOutputRoot ([string] (Get-RecoveryRequiredProperty $protocolFiles "pre_reference_manifest" "Recovery files"))
    $evaluationStartedRecord = Join-Path $resolvedOutputRoot ([string] (Get-RecoveryRequiredProperty $protocolFiles "evaluation_started_record" "Recovery files"))
    $metricsJson = Join-Path $resolvedOutputRoot ([string] (Get-RecoveryRequiredProperty $protocolFiles "metrics_json" "Recovery files"))
    $caseMetricsCsv = Join-Path $resolvedOutputRoot ([string] (Get-RecoveryRequiredProperty $protocolFiles "case_metrics_csv" "Recovery files"))
    $gateJson = Join-Path $resolvedOutputRoot ([string] (Get-RecoveryRequiredProperty $protocolFiles "gate_json" "Recovery files"))
    $completionRecord = Join-Path $resolvedOutputRoot ([string] (Get-RecoveryRequiredProperty $protocolFiles "completion_record" "Recovery files"))
    $originalLedgerSnapshot = Join-Path $resolvedOutputRoot ([string] (Get-RecoveryRequiredProperty $protocolFiles "original_ledger_snapshot" "Recovery files"))
    foreach ($mustBeAbsent in @(
        $preReferenceManifest,
        $evaluationStartedRecord,
        $metricsJson,
        $caseMetricsCsv,
        $gateJson,
        $completionRecord,
        $originalLedgerSnapshot
    )) {
        if (Test-Path -LiteralPath $mustBeAbsent) {
            throw "Recovery refuses to overwrite pre-existing continuation evidence: '$mustBeAbsent'."
        }
    }
    Assert-RecoveryHash (Get-RecoveryFileSha256 $preflightJson) (Get-RecoveryRequiredProperty $protocolEvidence "candidate_preflight_sha256" "Recovery evidence") "Candidate preflight"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $runtimeJson) (Get-RecoveryRequiredProperty $protocolEvidence "runtime_sha256" "Recovery evidence") "Runtime artifact"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $classificationCsv) (Get-RecoveryRequiredProperty $protocolEvidence "classification_csv_sha256" "Recovery evidence") "Classification CSV"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $probabilityCsv) (Get-RecoveryRequiredProperty $protocolEvidence "probability_csv_sha256" "Recovery evidence") "Probability CSV"

    # Patch only the two prospectively permitted PowerShell 5.1 count expressions,
    # in memory. The on-disk Common remains byte-identical to the final lock.
    $currentStage = "two_substitution_runtime_compatibility_audit"
    $commonSource = Get-Content -LiteralPath $commonScript -Raw
    if ($commonSource.Length -ne [int] (Get-RecoveryRequiredProperty $protocolPolicy "original_common_text_length" "Recovery policy")) {
        throw "Original Common text length differs from the recovery protocol."
    }
    $substitutions = @(Get-RecoveryRequiredProperty $protocol "only_permitted_compatibility_substitutions" "Recovery protocol")
    if ($substitutions.Count -ne 2) {
        throw "Recovery protocol must contain exactly two compatibility substitutions."
    }
    $patchedSource = $commonSource
    $substitutionCount = 0
    foreach ($substitution in $substitutions) {
        $original = [string] (Get-RecoveryRequiredProperty $substitution "original" "Compatibility substitution")
        $replacement = [string] (Get-RecoveryRequiredProperty $substitution "replacement" "Compatibility substitution")
        $requiredOccurrences = [int] (Get-RecoveryRequiredProperty $substitution "occurrences_required" "Compatibility substitution")
        $observedOccurrences = ([regex]::Matches($patchedSource, [regex]::Escape($original))).Count
        if ($observedOccurrences -ne $requiredOccurrences -or $requiredOccurrences -ne 1) {
            throw "A permitted compatibility expression does not occur exactly once."
        }
        $patchedSource = $patchedSource.Replace($original, $replacement)
        $substitutionCount += $observedOccurrences
    }
    if ($substitutionCount -ne 2 -or $patchedSource.Length -ne [int] (Get-RecoveryRequiredProperty $protocolPolicy "patched_common_text_length" "Recovery policy")) {
        throw "Recovery made an unexpected number or size of in-memory substitutions."
    }
    Assert-RecoveryHash (Get-RecoveryStringSha256 $patchedSource) (Get-RecoveryRequiredProperty $protocolPolicy "patched_common_text_sha256" "Recovery policy") "In-memory patched Common"
    . ([ScriptBlock]::Create($patchedSource))
    $candidate = Assert-V5FinalCandidateLock @candidateLockArguments
    $runtime = Assert-V5RuntimeArtifact `
        -RuntimeJson $runtimeJson `
        -ExpectedCaseCount 36 `
        -ExpectedInputDirectory $ValidationImages `
        -ExpectedDevice $Device `
        -Candidate $candidate `
        -ExpectedEncoderComponentSha256 $ExpectedEncoderComponentSha256 `
        -ExpectedDecoderComponentSha256 $ExpectedDecoderComponentSha256 `
        -ExpectedClassificationComponentSha256 $ExpectedClassificationComponentSha256
    Assert-RecoveryHash (Get-RecoveryFileSha256 $commonScript) (Get-RecoveryRequiredProperty $boundOriginal "common_sha256" "Bound implementation") "On-disk Common after in-memory audit"

    $currentStage = "frozen_39_artifact_verification"
    $artifactSet = Get-V5InferenceArtifactSet `
        -PredictionDirectory $predictionDirectory `
        -ClassificationCsv $classificationCsv `
        -ProbabilityCsv $probabilityCsv `
        -RuntimeJson $runtimeJson `
        -Runtime $runtime `
        -ExpectedCaseCount 36 `
        -OutputRoot $resolvedOutputRoot
    if (@($artifactSet.Artifacts).Count -ne [int] (Get-RecoveryRequiredProperty $protocolEvidence "validated_inference_artifact_count" "Recovery evidence") -or
        $artifactSet.MaskCount -ne [int] (Get-RecoveryRequiredProperty $protocolEvidence "mask_count" "Recovery evidence")) {
        throw "Inference artifact counts differ from the recovery protocol."
    }
    Assert-RecoveryHash $artifactSet.ArtifactSetSha256 (Get-RecoveryRequiredProperty $protocolEvidence "validated_inference_artifact_set_sha256" "Recovery evidence") "Frozen inference artifact set"

    $currentStage = "original_ledger_byte_snapshot"
    $originalLedgerSnapshot = Copy-RecoveryFileBytesCreateNew `
        -Source $originalLedgerPath `
        -Destination $originalLedgerSnapshot `
        -ExpectedSha256 $expectedOriginalLedgerSha256
    $originalLedgerSnapshotSha256 = Get-RecoveryFileSha256 $originalLedgerSnapshot
    $originalLedgerSnapshotPayload = Read-RecoveryJson $originalLedgerSnapshot "Original ledger snapshot"
    $originalStartedAtUtc = [string] (Get-RecoveryRequiredProperty $originalLedgerSnapshotPayload "started_at_utc" "Original ledger snapshot")
    $recoveryProvenance = [ordered]@{
        status = "separately_locked_saved_output_continuation_after_wrapper_failure"
        mandatory_disclosure = [string] (Get-RecoveryRequiredProperty $protocol "mandatory_disclosure" "Recovery protocol")
        recovery_protocol_sha256 = $protocolSha256
        recovery_execution_lock_sha256 = $executionLockSha256
        original_consumed_ledger_sha256 = $expectedOriginalLedgerSha256
        original_consumed_ledger_snapshot_sha256 = $originalLedgerSnapshotSha256
        frozen_40_file_output_snapshot_sha256 = $fullSnapshot.SnapshotSha256
        frozen_39_artifact_inference_set_sha256 = $artifactSet.ArtifactSetSha256
        runtime_sha256 = Get-RecoveryFileSha256 $runtimeJson
        runtime_validator_passed_after_exactly_two_in_memory_count_substitutions = $true
        recovery_inference_invocation_count = 0
        total_official_inference_invocation_count = 1
        validation_images_path = $resolvedValidationImages
        reference_masks_path = $resolvedReferenceMasks
        reference_subtypes_path = $resolvedReferenceSubtypes
        further_training_selection_threshold_model_or_prediction_changes_allowed = $false
        failure_forensics = Get-RecoveryRequiredProperty $protocol "failure_forensics" "Recovery protocol"
    }

    $currentStage = "pre_reference_manifest_creation"
    $preflightHash = Get-RecoveryFileSha256 $preflightJson
    $manifestPayload = [ordered]@{
        schema_version = 1
        status = "all_v5_label_blind_predictions_hashed_before_this_wrapper_reference_access"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        method = "single_locked_post_hoc_official_validation_reevaluation"
        final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
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
            cases = 36
            masks = $artifactSet.MaskCount
            classification_rows = $artifactSet.ClassificationRowCount
            probability_rows = $artifactSet.ProbabilityRowCount
        }
        candidate_preflight = [ordered]@{ relative_path = "evidence/candidate_preflight.json"; sha256 = $preflightHash }
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        prediction_artifacts = $artifactSet.Artifacts
        recovery = $recoveryProvenance
        reference_masks_path_tested_or_opened_by_this_wrapper_before_this_manifest = $false
        reference_subtypes_path_tested_or_opened_by_this_wrapper_before_this_manifest = $false
        official_evaluation_invocation_count_before_this_manifest = 0
    }
    $preReferenceManifest = Write-RecoveryJsonCreateNew -Path $preReferenceManifest -Payload $manifestPayload
    $preReferenceManifestSha256 = Get-RecoveryFileSha256 $preReferenceManifest
    $preReferenceManifestWritten = $true

    # Rebind everything after the durable prediction commitment and before any
    # reference-path operation. No per-case predictions are printed.
    $currentStage = "post_manifest_rebinding"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $protocolPath) $protocolSha256 "Recovery protocol after manifest"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $executionLockPath) $executionLockSha256 "Recovery execution lock after manifest"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $thisWrapper) (Get-RecoveryRequiredProperty $executionLock "recovery_wrapper_sha256" "Recovery execution lock") "Recovery wrapper after manifest"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $evaluationScript) (Get-RecoveryRequiredProperty $executionLock "evaluator_sha256" "Recovery execution lock") "Evaluator after manifest"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $metricsScript) (Get-RecoveryRequiredProperty $executionLock "metrics_source_sha256" "Recovery execution lock") "Metrics source after manifest"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $PythonExecutable) (Get-RecoveryRequiredProperty $executionLock "python_executable_sha256" "Recovery execution lock") "Python after manifest"
    $null = Assert-V5FinalCandidateLock @candidateLockArguments
    Assert-RecoveryHash (Get-RecoveryFileSha256 $preReferenceManifest) $preReferenceManifestSha256 "Pre-reference manifest after creation"
    $postManifestArtifactSet = Get-V5InferenceArtifactSet `
        -PredictionDirectory $predictionDirectory `
        -ClassificationCsv $classificationCsv `
        -ProbabilityCsv $probabilityCsv `
        -RuntimeJson $runtimeJson `
        -Runtime $runtime `
        -ExpectedCaseCount 36 `
        -OutputRoot $resolvedOutputRoot
    Assert-RecoveryHash $postManifestArtifactSet.ArtifactSetSha256 $artifactSet.ArtifactSetSha256 "Prediction set after manifest"

    # Resume the original ledger only after the durable manifest exists. Every
    # PowerShell 5.1 replacement has a unique, preserved, hash-chained backup.
    $currentStage = "original_ledger_predictions_frozen"
    $predictionsFrozenLedgerPayload = [ordered]@{
        schema_version = 1
        status = "predictions_frozen_before_this_wrapper_reference_access"
        stage = "single_locked_post_hoc_official_validation_reevaluation"
        started_at_utc = $originalStartedAtUtc
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
        output_root = $resolvedOutputRoot
        label_blind_prediction_complete = $true
        pre_reference_manifest_written = $true
        pre_reference_manifest_sha256 = $preReferenceManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        reference_access_started = $false
        evaluation_invocation_count = 0
        recovery = $recoveryProvenance
    }
    $currentOriginalLedgerSha256 = Write-RecoveryJsonAtomicWithBackup `
        -Path $originalLedgerPath `
        -Payload $predictionsFrozenLedgerPayload `
        -BackupPath $originalLedgerOriginalBackup `
        -ExpectedPreviousSha256 $currentOriginalLedgerSha256
    $originalLedgerUpdated = $true
    $originalLedgerStage = "predictions_frozen"

    # This CreateNew record durably consumes the sole evaluator invocation
    # before the first Test-Path or open of either target.
    $currentStage = "evaluation_invocation_commitment"
    $evaluationStartedPayload = [ordered]@{
        schema_version = 1
        status = "single_saved_output_evaluation_invocation_committed"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        recovery_protocol_sha256 = $protocolSha256
        recovery_execution_lock_sha256 = $executionLockSha256
        final_candidate_lock_sha256 = $candidate.LockSha256
        validation_images_path = $resolvedValidationImages
        reference_masks_path = $resolvedReferenceMasks
        reference_subtypes_path = $resolvedReferenceSubtypes
        pre_reference_manifest_sha256 = $preReferenceManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        recovery_inference_invocation_count = 0
        total_official_inference_invocation_count = 1
        official_evaluation_invocation_commitment_count = 1
        official_evaluation_process_invocation_count_before_this_record = 0
        reference_access_started_before_this_record = $false
        reference_access_permitted_after_this_record = $true
        further_evaluation_permitted = $false
        recovery = $recoveryProvenance
    }
    $evaluationStartedRecord = Write-RecoveryJsonCreateNew -Path $evaluationStartedRecord -Payload $evaluationStartedPayload
    $evaluationStartedSha256 = Get-RecoveryFileSha256 $evaluationStartedRecord
    $evaluationInvocationCommitted = $true

    # First reference-path access in the recovery continuation.
    $currentStage = "single_reference_opening_evaluation"
    $referenceAccessStarted = $true
    Assert-V5Directory $ReferenceMasks "Official reference-mask directory"
    Assert-V5LeafFile $ReferenceSubtypes "Official reference-subtype table"

    # Match the original wrapper's intended single_evaluation_started schema
    # after successful target assertions and immediately before process launch.
    $currentStage = "original_ledger_single_evaluation_started"
    $evaluationStartedLedgerPayload = [ordered]@{
        schema_version = 1
        status = "single_evaluation_started"
        stage = "single_locked_post_hoc_official_validation_reevaluation"
        started_at_utc = $originalStartedAtUtc
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
        output_root = $resolvedOutputRoot
        label_blind_prediction_complete = $true
        pre_reference_manifest_written = $true
        pre_reference_manifest_sha256 = $preReferenceManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        reference_access_started = $true
        evaluation_invocation_count = 1
        recovery = $recoveryProvenance
    }
    $currentOriginalLedgerSha256 = Write-RecoveryJsonAtomicWithBackup `
        -Path $originalLedgerPath `
        -Payload $evaluationStartedLedgerPayload `
        -BackupPath $originalLedgerPredictionsFrozenBackup `
        -ExpectedPreviousSha256 $currentOriginalLedgerSha256
    $originalLedgerStage = "evaluation_started"
    $evaluationInvocationCount = 1
    Invoke-V5CheckedPython `
        -PythonExecutable $PythonExecutable `
        -ScriptPath $evaluationScript `
        -Arguments @(
            "--predictions", $predictionDirectory,
            "--references", (Get-V5NormalizedFullPath $ReferenceMasks),
            "--classification-predictions", $classificationCsv,
            "--classification-references", (Get-V5NormalizedFullPath $ReferenceSubtypes),
            "--classification-reference-split", "validation",
            "--output-json", $metricsJson,
            "--output-csv", $caseMetricsCsv,
            "--empty-empty-dice", "1.0",
            "--label-tolerance", "0.0",
            "--affine-tolerance", "0.00001",
            "--bootstrap-samples", "2000",
            "--confidence", "0.95",
            "--seed", "12345"
        ) `
        -Stage "The single saved-output official evaluation recovery"

    $currentStage = "post_evaluation_verification"
    $null = Assert-V5FinalCandidateLock @candidateLockArguments
    Assert-RecoveryHash (Get-RecoveryFileSha256 $protocolPath) $protocolSha256 "Recovery protocol after evaluation"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $executionLockPath) $executionLockSha256 "Recovery execution lock after evaluation"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $thisWrapper) (Get-RecoveryRequiredProperty $executionLock "recovery_wrapper_sha256" "Recovery execution lock") "Recovery wrapper after evaluation"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $evaluationScript) (Get-RecoveryRequiredProperty $executionLock "evaluator_sha256" "Recovery execution lock") "Evaluator after evaluation"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $metricsScript) (Get-RecoveryRequiredProperty $executionLock "metrics_source_sha256" "Recovery execution lock") "Metrics source after evaluation"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $PythonExecutable) (Get-RecoveryRequiredProperty $executionLock "python_executable_sha256" "Recovery execution lock") "Python after evaluation"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $preReferenceManifest) $preReferenceManifestSha256 "Pre-reference manifest after evaluation"
    Assert-RecoveryHash (Get-RecoveryFileSha256 $evaluationStartedRecord) $evaluationStartedSha256 "Evaluation-started record after evaluation"
    $postEvaluationArtifactSet = Get-V5InferenceArtifactSet `
        -PredictionDirectory $predictionDirectory `
        -ClassificationCsv $classificationCsv `
        -ProbabilityCsv $probabilityCsv `
        -RuntimeJson $runtimeJson `
        -Runtime $runtime `
        -ExpectedCaseCount 36 `
        -OutputRoot $resolvedOutputRoot
    Assert-RecoveryHash $postEvaluationArtifactSet.ArtifactSetSha256 $artifactSet.ArtifactSetSha256 "Prediction artifact set after evaluation"

    $metrics = Read-V5JsonObject $metricsJson "Official evaluation metrics"
    if ([int] (Get-V5RequiredProperty $metrics "case_count" "Evaluation metrics") -ne 36) {
        throw "Official evaluation must contain exactly 36 merged cases."
    }
    $segmentation = Get-V5RequiredProperty $metrics "segmentation" "Evaluation metrics"
    $classification = Get-V5RequiredProperty $metrics "classification" "Evaluation metrics"
    if ([int] (Get-V5RequiredProperty $segmentation "case_count" "Segmentation metrics") -ne 36 -or
        [int] (Get-V5RequiredProperty $classification "case_count" "Classification metrics") -ne 36) {
        throw "Both official tasks must contain exactly 36 evaluated cases."
    }
    $wholeDice = [double] (Get-V5RequiredProperty (Get-V5RequiredProperty $segmentation "whole_pancreas_dice" "Segmentation metrics") "mean" "Whole-pancreas metrics")
    $lesionDice = [double] (Get-V5RequiredProperty (Get-V5RequiredProperty $segmentation "lesion_dice" "Segmentation metrics") "mean" "Lesion metrics")
    $macroF1 = [double] (Get-V5RequiredProperty $classification "macro_f1" "Classification metrics")
    foreach ($metric in @($wholeDice, $lesionDice, $macroF1)) {
        if ([double]::IsNaN($metric) -or [double]::IsInfinity($metric) -or $metric -lt 0.0 -or $metric -gt 1.0) {
            throw "Official evaluation emitted a non-finite or out-of-range metric."
        }
    }
    $strictImprovement = $macroF1 -gt $script:V5BaselineMacroF1
    $wholeGate = $wholeDice -ge 0.91
    $lesionGate = $lesionDice -ge 0.31
    $phdClassificationGate = $macroF1 -ge 0.70
    $gatePayload = [ordered]@{
        schema_version = 1
        status = "complete_no_second_classifier_iteration_permitted"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        evaluation_scope = "single_locked_post_hoc_official_validation_reevaluation"
        validation_output_root = $resolvedOutputRoot
        final_candidate_lock_sha256 = $candidate.LockSha256
        pre_reference_prediction_manifest_sha256 = $preReferenceManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        evaluation_metrics = [ordered]@{ path = "evidence/official_evaluation_metrics.json"; sha256 = Get-RecoveryFileSha256 $metricsJson }
        evaluation_cases = [ordered]@{ path = "evidence/official_evaluation_cases.csv"; sha256 = Get-RecoveryFileSha256 $caseMetricsCsv }
        metrics = [ordered]@{ whole_pancreas_dice = $wholeDice; lesion_dice = $lesionDice; macro_f1 = $macroF1 }
        thresholds = [ordered]@{
            baseline_macro_f1_strictly_greater_than = $script:V5BaselineMacroF1
            phd_macro_f1_at_least = 0.70
            whole_pancreas_dice_at_least = 0.91
            lesion_dice_at_least = 0.31
        }
        verdicts = [ordered]@{
            strict_macro_f1_improvement_over_baseline = $strictImprovement
            phd_macro_f1_gate = $phdClassificationGate
            whole_pancreas_gate = $wholeGate
            lesion_gate = $lesionGate
            phd_joint_metric_gate = ($wholeGate -and $lesionGate -and $phdClassificationGate)
            classifier_replacement_validation_gate = ($strictImprovement -and $wholeGate -and $lesionGate)
        }
        official_inference_invocation_count = 1
        official_evaluation_invocation_count = 1
        further_classifier_training_selection_or_official_evaluation_permitted = $false
        recovery = $recoveryProvenance
    }
    $gateJson = Write-RecoveryJsonCreateNew -Path $gateJson -Payload $gatePayload
    $gateSha256 = Get-RecoveryFileSha256 $gateJson

    # Complete the original intended ledger state. The preserved backup is the
    # exact evaluation-started state and closes the three-link recovery chain.
    $currentStage = "original_ledger_completion"
    $completedLedgerPayload = [ordered]@{
        schema_version = 1
        status = "complete_and_consumed"
        stage = "single_locked_post_hoc_official_validation_reevaluation"
        started_at_utc = $originalStartedAtUtc
        completed_at_utc = [DateTime]::UtcNow.ToString("o")
        final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
        output_root = $resolvedOutputRoot
        label_blind_prediction_complete = $true
        pre_reference_manifest_written = $true
        pre_reference_manifest_sha256 = $preReferenceManifestSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        reference_access_started = $true
        evaluation_invocation_count = 1
        gate_artifact_sha256 = $gateSha256
        no_second_classifier_iteration_permitted = $true
        recovery = $recoveryProvenance
    }
    $currentOriginalLedgerSha256 = Write-RecoveryJsonAtomicWithBackup `
        -Path $originalLedgerPath `
        -Payload $completedLedgerPayload `
        -BackupPath $originalLedgerEvaluationStartedBackup `
        -ExpectedPreviousSha256 $currentOriginalLedgerSha256
    $originalLedgerUpdated = $true
    $originalLedgerStage = "complete"

    $currentStage = "recovery_completion_record"
    $completionPayload = [ordered]@{
        schema_version = 1
        status = "saved_output_recovery_complete_no_rerun_permitted"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        mandatory_disclosure = [string] (Get-RecoveryRequiredProperty $protocol "mandatory_disclosure" "Recovery protocol")
        recovery_protocol_sha256 = $protocolSha256
        recovery_execution_lock_sha256 = $executionLockSha256
        final_candidate_lock_sha256 = $candidate.LockSha256
        original_ledger_snapshot_sha256 = $originalLedgerSnapshotSha256
        original_ledger_original_backup_sha256 = Get-RecoveryFileSha256 $originalLedgerOriginalBackup
        original_ledger_predictions_frozen_backup_sha256 = Get-RecoveryFileSha256 $originalLedgerPredictionsFrozenBackup
        original_ledger_evaluation_started_backup_sha256 = Get-RecoveryFileSha256 $originalLedgerEvaluationStartedBackup
        completed_original_ledger_sha256 = Get-RecoveryFileSha256 $originalLedgerPath
        pre_reference_manifest_sha256 = $preReferenceManifestSha256
        evaluation_started_record_sha256 = $evaluationStartedSha256
        prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
        gate_artifact_sha256 = $gateSha256
        recovery_inference_invocation_count = 0
        total_official_inference_invocation_count = 1
        official_evaluation_invocation_count = 1
        further_evaluation_permitted = $false
        validation_images_path = $resolvedValidationImages
        reference_masks_path = $resolvedReferenceMasks
        reference_subtypes_path = $resolvedReferenceSubtypes
        recovery = $recoveryProvenance
    }
    $completionRecord = Write-RecoveryJsonCreateNew -Path $completionRecord -Payload $completionPayload
    $completionSha256 = Get-RecoveryFileSha256 $completionRecord

    $currentStage = "recovery_ledger_completion"
    $previousRecoveryLedgerSha256 = Get-RecoveryFileSha256 $recoveryLedgerPath
    $recoveryLedgerBackup = $recoveryLedgerPath + ".started_backup.json"
    $null = Write-RecoveryJsonAtomicWithBackup `
        -Path $recoveryLedgerPath `
        -Payload ([ordered]@{
            schema_version = 1
            status = "complete_and_consumed_no_recovery_rerun"
            stage = "saved_output_official_evaluation_recovery"
            started_at_utc = $startedAtUtc
            completed_at_utc = [DateTime]::UtcNow.ToString("o")
            recovery_protocol_sha256 = $protocolSha256
            recovery_execution_lock_sha256 = $executionLockSha256
            final_candidate_lock_sha256 = $candidate.LockSha256
            output_root = $resolvedOutputRoot
            recovery_inference_invocation_count = 0
            total_official_inference_invocation_count = 1
            official_evaluation_invocation_count = 1
            reference_access_started = $true
            pre_reference_manifest_sha256 = $preReferenceManifestSha256
            prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
            gate_artifact_sha256 = $gateSha256
            completion_record_sha256 = $completionSha256
            no_second_recovery_or_evaluation_permitted = $true
            recovery = $recoveryProvenance
        }) `
        -BackupPath $recoveryLedgerBackup `
        -ExpectedPreviousSha256 $previousRecoveryLedgerSha256

    $currentStage = "complete"
    Write-Host "Saved-output official-evaluation recovery complete."
    Write-Host "Pre-reference manifest: $preReferenceManifest"
    Write-Host "Gate artifact: $gateJson"
    Write-Host ("Macro-F1: {0:R}; strict baseline improvement: {1}" -f $macroF1, $strictImprovement)
    Write-Host ("Whole Dice: {0:R}; lesion Dice: {1:R}; PhD joint metric gate: {2}" -f $wholeDice, $lesionDice, ($wholeGate -and $lesionGate -and $phdClassificationGate))
}
catch {
    $failure = $_
    if ($preReferenceManifestWritten -and $originalLedgerStage -notin @("complete", "failed")) {
        try {
            $failureBackup = switch ($originalLedgerStage) {
                "frozen_original" { $originalLedgerOriginalBackup }
                "predictions_frozen" { $originalLedgerPath + ".recovery_failure_from_predictions_frozen_backup.json" }
                "evaluation_started" { $originalLedgerPath + ".recovery_failure_from_evaluation_started_backup.json" }
                default { $originalLedgerPath + ".recovery_failure_from_unknown_state_backup.json" }
            }
            $failedLedgerPayload = [ordered]@{
                schema_version = 1
                status = "failed_and_consumed_no_rerun"
                stage = "single_locked_post_hoc_official_validation_reevaluation"
                failed_stage = $currentStage
                started_at_utc = $originalStartedAtUtc
                failed_at_utc = [DateTime]::UtcNow.ToString("o")
                final_candidate_lock = [ordered]@{ path = $candidate.LockPath; sha256 = $candidate.LockSha256 }
                output_root = $resolvedOutputRoot
                label_blind_prediction_complete = $true
                pre_reference_manifest_written = $true
                pre_reference_manifest_sha256 = $preReferenceManifestSha256
                prediction_artifact_set_sha256 = $artifactSet.ArtifactSetSha256
                reference_access_started = $referenceAccessStarted
                evaluation_invocation_count = $evaluationInvocationCount
                error_type = $failure.Exception.GetType().FullName
                error_message = $failure.Exception.Message
                one_use_run_remains_consumed = $true
                recovery = $recoveryProvenance
            }
            $currentOriginalLedgerSha256 = Write-RecoveryJsonAtomicWithBackup `
                -Path $originalLedgerPath `
                -Payload $failedLedgerPayload `
                -BackupPath $failureBackup `
                -ExpectedPreviousSha256 $currentOriginalLedgerSha256
            $originalLedgerUpdated = $true
            $originalLedgerStage = "failed"
        }
        catch {
            Write-Warning "Could not record the intended original-ledger failure schema: $($_.Exception.Message)"
        }
    }
    if ($recoveryLedgerCreated) {
        try {
            $previousRecoveryLedgerSha256 = Get-RecoveryFileSha256 $recoveryLedgerPath
            $failureBackup = $recoveryLedgerPath + ".failure_backup.json"
            $null = Write-RecoveryJsonAtomicWithBackup `
                -Path $recoveryLedgerPath `
                -Payload ([ordered]@{
                    schema_version = 1
                    status = "failed_and_consumed_no_recovery_rerun"
                    stage = "saved_output_official_evaluation_recovery"
                    failed_stage = $currentStage
                    started_at_utc = $startedAtUtc
                    failed_at_utc = [DateTime]::UtcNow.ToString("o")
                    recovery_protocol_sha256 = $protocolSha256
                    recovery_execution_lock_sha256 = $executionLockSha256
                    final_candidate_lock_sha256 = $candidate.LockSha256
                    output_root = $resolvedOutputRoot
                    validation_images_path = $resolvedValidationImages
                    reference_masks_path = $resolvedReferenceMasks
                    reference_subtypes_path = $resolvedReferenceSubtypes
                    recovery_inference_invocation_count = 0
                    total_official_inference_invocation_count = 1
                    official_evaluation_invocation_count = $evaluationInvocationCount
                    official_evaluation_invocation_committed = $evaluationInvocationCommitted
                    reference_access_started = $referenceAccessStarted
                    pre_reference_manifest_written = $preReferenceManifestWritten
                    original_ledger_updated = $originalLedgerUpdated
                    original_ledger_stage = $originalLedgerStage
                    error_type = $failure.Exception.GetType().FullName
                    error_message = $failure.Exception.Message
                    one_use_recovery_remains_consumed = $true
                    recovery = if ($preReferenceManifestWritten) { $recoveryProvenance } else { $null }
                }) `
                -BackupPath $failureBackup `
                -ExpectedPreviousSha256 $previousRecoveryLedgerSha256
        }
        catch {
            Write-Warning "Could not update the already-consumed recovery ledger: $($_.Exception.Message)"
        }
    }
    throw $failure
}
finally {
    Exit-V5NamedMutex $mutex
}
