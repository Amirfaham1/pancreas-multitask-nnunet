<#
.SYNOPSIS
Run the single locked v5 selected-test/package path under Windows PowerShell 5.1.

.DESCRIPTION
This prospective compatibility entry point preserves the frozen selected-test
wrapper body exactly. Before that body can inspect an official gate, selected
test path, output path, or one-use ledger, it verifies the prospective protocol,
the published compatibility implementation, the final-candidate lock, the
original wrapper/Common chain, every directly executed script, and Python.

It then applies exactly three protocol-locked substitutions to Common in memory:
two PSObject property-count expressions are made Windows PowerShell 5.1-safe,
and the null File.Replace backup is routed through a same-directory unique
backup helper. The original selected-test body is hash-extracted without any
substitution and executed once. Its CreateNew ledger ordering, inference,
packaging, validation, completion, and failed-and-consumed behavior remain the
original semantics. This wrapper itself performs no inference or packaging
outside that frozen body.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $CompatibilityProtocol,
    [Parameter(Mandatory)]
    [string] $ExpectedCompatibilityProtocolSha256,
    [Parameter(Mandatory)]
    [string] $ExpectedCompatibilityWrapperSha256,
    [Parameter(Mandatory)]
    [string] $ExpectedCompatibilityTestsSha256,
    [Parameter(Mandatory)]
    [string] $CompatibilityImplementationCommit,
    [Parameter(Mandatory)]
    [string] $FinalCandidateLock,
    [Parameter(Mandatory)]
    [string] $ExpectedFinalCandidateLockSha256,
    [Parameter(Mandatory)]
    [string] $OfficialEvaluationGate,
    [Parameter(Mandatory)]
    [string] $ExpectedOfficialEvaluationGateSha256,
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
    [string] $TestImages,
    [Parameter(Mandatory)]
    [string] $OutputRoot,
    [string] $WorkRoot = "D:\MLQuizWork",
    [string] $PythonExecutable = "D:\MLQuizWork\.venv\Scripts\python.exe",
    [ValidateSet("cuda")]
    [string] $Device = "cuda",
    [string] $ImmutableBaselineRoot = "D:\MLQuizWork\baseline_20260806_509cbe2"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:V5PS51ProtocolSha256 =
    "9c92e8c2d17107937ea913f2ea9bbe0f7f901b24a2228d31bee53e34a5923681"
$script:V5PS51PowerShellPath =
    "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
$script:V5PS51PowerShellVersion = "5.1.26100.8875"
$script:V5PS51PowerShellSha256 =
    "7600ffe12da441fe89d035b13801e8e91d064bc544a27b19a5cf49f6ab8b18f5"
$script:V5PS51PackageInitSha256 =
    "430cedcca46cd2c0ba5d1f88762e0c2831f2c11484baf16baea306c373e120d4"
$script:V5PS51TestImagesPath =
    "D:\MLQuizWork\nnUNet_raw\Dataset501_PancreasMultitask\imagesTs"
$script:V5PS51OutputRootPath =
    "D:\MLQuizWork\phd_upgrade_v5\selected_test_locked_v5"
$script:V5PS51ScriptRoot = [IO.Path]::GetFullPath($PSScriptRoot)

function ConvertTo-V5PS51Sha256 {
    param(
        [Parameter(Mandatory)][object] $Value,
        [Parameter(Mandatory)][string] $Description
    )

    $text = ([string] $Value).Trim().ToLowerInvariant()
    if ($text -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Description must be one lowercase 64-digit SHA-256 digest."
    }
    return $text
}

function Get-V5PS51FileSha256 {
    param([Parameter(Mandatory)][string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required compatibility file does not exist: '$Path'."
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-V5PS51StringSha256 {
    param([Parameter(Mandatory)][AllowEmptyString()][string] $Value)

    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
        return ([BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace(
            "-",
            ""
        ).ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Assert-V5PS51Equal {
    param(
        [Parameter(Mandatory)][object] $Actual,
        [Parameter(Mandatory)][object] $Expected,
        [Parameter(Mandatory)][string] $Description
    )

    if ($Actual -is [string] -or $Expected -is [string]) {
        if (-not ([string] $Actual).Equals(
            [string] $Expected,
            [StringComparison]::Ordinal
        )) {
            throw "$Description differs from the prospective compatibility protocol."
        }
        return
    }
    if ($Actual -ne $Expected) {
        throw "$Description differs from the prospective compatibility protocol."
    }
}

function Assert-V5PS51Hash {
    param(
        [Parameter(Mandatory)][object] $Actual,
        [Parameter(Mandatory)][object] $Expected,
        [Parameter(Mandatory)][string] $Description
    )

    $actualHash = ConvertTo-V5PS51Sha256 $Actual "$Description actual hash"
    $expectedHash = ConvertTo-V5PS51Sha256 $Expected "$Description expected hash"
    if ($actualHash -cne $expectedHash) {
        throw "$Description SHA-256 mismatch: expected $expectedHash, got $actualHash."
    }
}

function Assert-V5PS51PathEqual {
    param(
        [Parameter(Mandatory)][string] $Actual,
        [Parameter(Mandatory)][string] $Expected,
        [Parameter(Mandatory)][string] $Description
    )

    $actualPath = [IO.Path]::GetFullPath($Actual).TrimEnd([char[]] "\/")
    $expectedPath = [IO.Path]::GetFullPath($Expected).TrimEnd([char[]] "\/")
    if (-not $actualPath.Equals($expectedPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description path mismatch: expected '$expectedPath', got '$actualPath'."
    }
}

function Get-V5PS51RequiredProperty {
    param(
        [Parameter(Mandatory)][object] $Object,
        [Parameter(Mandatory)][string] $Name,
        [Parameter(Mandatory)][string] $Description
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Description is missing required property '$Name'."
    }
    return $property.Value
}

function Read-V5PS51Json {
    param(
        [Parameter(Mandatory)][string] $Path,
        [Parameter(Mandatory)][string] $Description
    )

    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        throw "Could not parse $Description JSON '$Path': $($_.Exception.Message)"
    }
}

function Assert-V5PS51PublishedImplementation {
    param(
        [Parameter(Mandatory)][string] $ProjectRoot,
        [Parameter(Mandatory)][string] $Commit,
        [Parameter(Mandatory)][string[]] $RelativePaths
    )

    if ($Commit -cnotmatch '^[0-9a-f]{40}$') {
        throw "Compatibility implementation commit must be 40 lowercase hexadecimal digits."
    }
    if ($null -eq (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is required to verify the published compatibility implementation."
    }
    & git -C $ProjectRoot cat-file -e "$Commit^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Compatibility implementation commit is not a local commit object."
    }
    & git -C $ProjectRoot merge-base --is-ancestor $Commit origin/main 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Compatibility implementation commit is not published on origin/main."
    }
    foreach ($relativePath in $RelativePaths) {
        & git -C $ProjectRoot diff --quiet $Commit -- $relativePath
        if ($LASTEXITCODE -ne 0) {
            throw "Compatibility file '$relativePath' differs from the published commit."
        }
        $status = @(& git -C $ProjectRoot status --porcelain --untracked-files=all -- $relativePath)
        if ($LASTEXITCODE -ne 0 -or $status.Count -ne 0) {
            throw "Compatibility file '$relativePath' has an uncommitted or untracked change."
        }
    }
}

function Invoke-V5PS51AtomicJsonReplace {
    param(
        [Parameter(Mandatory)][string] $Temporary,
        [Parameter(Mandatory)][string] $Destination
    )

    $resolvedTemporary = [IO.Path]::GetFullPath($Temporary)
    $resolvedDestination = [IO.Path]::GetFullPath($Destination)
    if (-not (Test-Path -LiteralPath $resolvedTemporary -PathType Leaf)) {
        throw "Atomic JSON replacement temporary file is missing: '$resolvedTemporary'."
    }
    if (-not (Test-Path -LiteralPath $resolvedDestination -PathType Leaf)) {
        throw "Atomic JSON replacement destination file is missing: '$resolvedDestination'."
    }
    $temporaryParent = [IO.Path]::GetDirectoryName($resolvedTemporary)
    $destinationParent = [IO.Path]::GetDirectoryName($resolvedDestination)
    if (-not $temporaryParent.Equals(
        $destinationParent,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Atomic JSON replacement endpoints must share one directory."
    }
    if ($resolvedTemporary.Equals(
        $resolvedDestination,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Atomic JSON replacement endpoints must be different files."
    }
    $backup = Join-Path $destinationParent (
        ".{0}.{1}.ps51-replace-backup" -f
            [IO.Path]::GetFileName($resolvedDestination),
            [Guid]::NewGuid().ToString("N")
    )
    if (Test-Path -LiteralPath $backup) {
        throw "Atomic JSON replacement backup unexpectedly exists: '$backup'."
    }

    $replacementSucceeded = $false
    try {
        # Windows PowerShell 5.1 requires a non-null backup path here.
        [IO.File]::Replace($resolvedTemporary, $resolvedDestination, $backup)
        $replacementSucceeded = $true
    }
    finally {
        # Match the original writer's artifact semantics after success. If the
        # replacement itself fails, preserve any surviving backup for forensics.
        if ($replacementSucceeded -and (Test-Path -LiteralPath $backup -PathType Leaf)) {
            Remove-Item -LiteralPath $backup -Force
        }
    }
}

function Assert-V5PS51FrozenCompatibilityBindings {
    param(
        [Parameter(Mandatory)][string] $Stage,
        [Parameter(Mandatory)][string] $ProjectRoot,
        [Parameter(Mandatory)][string] $ImplementationCommit,
        [Parameter(Mandatory)][string] $ProtocolPath,
        [Parameter(Mandatory)][string] $ProtocolSha256,
        [Parameter(Mandatory)][string] $WrapperPath,
        [Parameter(Mandatory)][string] $WrapperSha256,
        [Parameter(Mandatory)][string] $TestsPath,
        [Parameter(Mandatory)][string] $TestsSha256,
        [Parameter(Mandatory)][string] $FinalCandidatePath,
        [Parameter(Mandatory)][string] $FinalCandidateSha256,
        [Parameter(Mandatory)][object[]] $ImplementationBindings,
        [Parameter(Mandatory)][string] $PythonPath,
        [Parameter(Mandatory)][string] $PythonSha256,
        [Parameter(Mandatory)][string] $PackageInitPath,
        [Parameter(Mandatory)][string] $PackageInitSha256,
        [Parameter(Mandatory)][string] $PowerShellPath,
        [Parameter(Mandatory)][string] $PowerShellSha256
    )

    Assert-V5PS51Hash (Get-V5PS51FileSha256 $ProtocolPath) $ProtocolSha256 "$Stage protocol"
    Assert-V5PS51Hash (Get-V5PS51FileSha256 $WrapperPath) $WrapperSha256 "$Stage wrapper"
    Assert-V5PS51Hash (Get-V5PS51FileSha256 $TestsPath) $TestsSha256 "$Stage tests"
    Assert-V5PS51Hash `
        (Get-V5PS51FileSha256 $FinalCandidatePath) `
        $FinalCandidateSha256 `
        "$Stage final-candidate lock"
    foreach ($binding in $ImplementationBindings) {
        Assert-V5PS51Hash `
            (Get-V5PS51FileSha256 $binding.Path) `
            (Get-V5PS51RequiredProperty $binding.Protocol $binding.HashField "Bound $($binding.Name)") `
            "$Stage $($binding.Name)"
    }
    Assert-V5PS51Hash (Get-V5PS51FileSha256 $PythonPath) $PythonSha256 "$Stage Python"
    Assert-V5PS51Hash `
        (Get-V5PS51FileSha256 $PackageInitPath) `
        $PackageInitSha256 `
        "$Stage package initializer"
    Assert-V5PS51Hash `
        (Get-V5PS51FileSha256 $PowerShellPath) `
        $PowerShellSha256 `
        "$Stage PowerShell"
    Assert-V5PS51PublishedImplementation `
        -ProjectRoot $ProjectRoot `
        -Commit $ImplementationCommit `
        -RelativePaths @(
            "configs/selected_test_ps51_compatibility_protocol_v1.json",
            "scripts/Run-V5LockedSelectedTestAndPackagePS51.ps1",
            "tests/test_v5_selected_test_ps51_compatibility.py"
        )
}

$projectRoot = [IO.Path]::GetFullPath((Join-Path $script:V5PS51ScriptRoot ".."))
$protocolPath = [IO.Path]::GetFullPath($CompatibilityProtocol)
$expectedProtocolPath = Join-Path $projectRoot "configs\selected_test_ps51_compatibility_protocol_v1.json"
$thisWrapper = [IO.Path]::GetFullPath($PSCommandPath)
$compatibilityTests = Join-Path $projectRoot "tests\test_v5_selected_test_ps51_compatibility.py"
$commonScript = Join-Path $script:V5PS51ScriptRoot "V5-LockedDeliveryCommon.ps1"
$originalSelectedWrapper = Join-Path $script:V5PS51ScriptRoot "Run-V5LockedSelectedTestAndPackage.ps1"
$predictionScriptPath = Join-Path $script:V5PS51ScriptRoot "predict_joint.py"
$packageScriptPath = Join-Path $script:V5PS51ScriptRoot "Package-Submission.ps1"
$validatorScriptPath = Join-Path $script:V5PS51ScriptRoot "validate_submission.py"
$setupScriptPath = Join-Path $script:V5PS51ScriptRoot "Set-QuizEnvironment.ps1"
$packageInitPath = Join-Path $projectRoot "src\pancreas_multitask\__init__.py"

# This bootstrap intentionally does not test, resolve, enumerate, or open the
# official gate, TestImages, OutputRoot, ImmutableBaselineRoot, or any ledger.
# The following two checks normalize and compare caller strings only. They do
# not query either filesystem path, and prevent a typo from consuming the one
# selected-test ledger on a wrong input or output root.
Assert-V5PS51PathEqual $TestImages $script:V5PS51TestImagesPath "Selected-test images"
Assert-V5PS51PathEqual $OutputRoot $script:V5PS51OutputRootPath "Selected-test output root"
$hostExecutable = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
Assert-V5PS51PathEqual $hostExecutable $script:V5PS51PowerShellPath "PowerShell host"
Assert-V5PS51Equal $PSVersionTable.PSEdition "Desktop" "PowerShell edition"
Assert-V5PS51Equal `
    $PSVersionTable.PSVersion.ToString() `
    $script:V5PS51PowerShellVersion `
    "PowerShell version"
Assert-V5PS51Hash `
    (Get-V5PS51FileSha256 $hostExecutable) `
    $script:V5PS51PowerShellSha256 `
    "PowerShell executable"
Assert-V5PS51PathEqual $protocolPath $expectedProtocolPath "Compatibility protocol"
$protocolHash = ConvertTo-V5PS51Sha256 $ExpectedCompatibilityProtocolSha256 "Expected compatibility protocol hash"
Assert-V5PS51Hash $protocolHash $script:V5PS51ProtocolSha256 "Hard-bound compatibility protocol"
Assert-V5PS51Hash (Get-V5PS51FileSha256 $protocolPath) $protocolHash "Compatibility protocol file"
$protocol = Read-V5PS51Json $protocolPath "Compatibility protocol"
if ([int] (Get-V5PS51RequiredProperty $protocol "schema_version" "Compatibility protocol") -ne 1) {
    throw "Compatibility protocol schema_version must be 1."
}
Assert-V5PS51Equal `
    (Get-V5PS51RequiredProperty $protocol "status" "Compatibility protocol") `
    "frozen_before_selected_test_ps51_implementation_or_any_selected_test_access" `
    "Compatibility protocol status"

$accessAtLock = Get-V5PS51RequiredProperty $protocol "access_at_lock" "Compatibility protocol"
foreach ($field in @(
    "official_reference_masks_tested_or_opened",
    "official_reference_subtypes_tested_or_opened",
    "selected_test_inputs_tested_or_opened",
    "selected_test_outputs_tested_or_opened",
    "selected_test_ledger_tested_or_opened"
)) {
    $value = Get-V5PS51RequiredProperty $accessAtLock $field "Compatibility access-at-lock"
    if ($value -isnot [bool] -or [bool] $value) {
        throw "Compatibility access-at-lock '$field' must be the JSON boolean false."
    }
}
foreach ($field in @("inference_invocation_count", "package_invocation_count")) {
    if ([int] (Get-V5PS51RequiredProperty $accessAtLock $field "Compatibility access-at-lock") -ne 0) {
        throw "Compatibility access-at-lock '$field' must equal zero."
    }
}

$boundCandidate = Get-V5PS51RequiredProperty $protocol "bound_final_candidate" "Compatibility protocol"
$boundOriginal = Get-V5PS51RequiredProperty $protocol "bound_original_implementation" "Compatibility protocol"
$implementationPolicy = Get-V5PS51RequiredProperty $protocol "compatibility_implementation_policy" "Compatibility protocol"
$executionContract = Get-V5PS51RequiredProperty $protocol "execution_contract" "Compatibility protocol"

Assert-V5PS51Equal `
    (Get-V5PS51RequiredProperty $implementationPolicy "new_wrapper_path" "Compatibility implementation policy") `
    "scripts/Run-V5LockedSelectedTestAndPackagePS51.ps1" `
    "Compatibility wrapper path"
Assert-V5PS51Equal `
    (Get-V5PS51RequiredProperty $implementationPolicy "new_tests_path" "Compatibility implementation policy") `
    "tests/test_v5_selected_test_ps51_compatibility.py" `
    "Compatibility tests path"
if ([int] (Get-V5PS51RequiredProperty $executionContract "selected_test_inference_invocation_count" "Compatibility execution contract") -ne 1 -or
    [int] (Get-V5PS51RequiredProperty $executionContract "package_invocation_count" "Compatibility execution contract") -ne 1) {
    throw "Compatibility execution contract must permit exactly one inference and package invocation."
}
foreach ($field in @(
    "test_or_official_data_access_during_compatibility_tests",
    "inference_or_package_invocation_during_compatibility_tests",
    "package_force_or_replacement_is_permitted",
    "second_selected_test_run_or_package_attempt_is_permitted",
    "test_targets_submission_feedback_or_post_validation_model_changes_are_permitted"
)) {
    $value = Get-V5PS51RequiredProperty $executionContract $field "Compatibility execution contract"
    if ($value -isnot [bool] -or [bool] $value) {
        throw "Compatibility execution contract '$field' must be the JSON boolean false."
    }
}
foreach ($field in @(
    "strict_model_load_and_all_immutable_bindings_precede_selected_test_path_access",
    "one_use_selected_test_ledger_create_new_precedes_selected_test_path_access",
    "original_selected_test_ledger_stage_order_and_payload_schemas_are_preserved",
    "fresh_output_root_and_fresh_delivery_are_required",
    "original_prediction_package_and_validator_entry_points_are_unchanged",
    "official_gate_must_pass_the_original_unchanged_replacement_gate",
    "failure_after_ledger_creation_consumes_the_run"
)) {
    $value = Get-V5PS51RequiredProperty $executionContract $field "Compatibility execution contract"
    if ($value -isnot [bool] -or -not [bool] $value) {
        throw "Compatibility execution contract '$field' must be the JSON boolean true."
    }
}

$expectedWrapperHash = ConvertTo-V5PS51Sha256 $ExpectedCompatibilityWrapperSha256 "Expected compatibility wrapper hash"
$expectedTestsHash = ConvertTo-V5PS51Sha256 $ExpectedCompatibilityTestsSha256 "Expected compatibility tests hash"
Assert-V5PS51Hash (Get-V5PS51FileSha256 $thisWrapper) $expectedWrapperHash "Compatibility wrapper self-binding"
Assert-V5PS51Hash (Get-V5PS51FileSha256 $compatibilityTests) $expectedTestsHash "Compatibility tests binding"
Assert-V5PS51PublishedImplementation `
    -ProjectRoot $projectRoot `
    -Commit $CompatibilityImplementationCommit `
    -RelativePaths @(
        "configs/selected_test_ps51_compatibility_protocol_v1.json",
        "scripts/Run-V5LockedSelectedTestAndPackagePS51.ps1",
        "tests/test_v5_selected_test_ps51_compatibility.py"
    )

$expectedFinalCandidatePath = Join-Path $projectRoot (
    ([string] (Get-V5PS51RequiredProperty $boundCandidate "path" "Bound final candidate")).Replace("/", "\")
)
Assert-V5PS51PathEqual $FinalCandidateLock $expectedFinalCandidatePath "Final-candidate lock"
$protocolFinalCandidateHash = Get-V5PS51RequiredProperty $boundCandidate "sha256" "Bound final candidate"
Assert-V5PS51Hash $ExpectedFinalCandidateLockSha256 $protocolFinalCandidateHash "Caller final-candidate lock"
Assert-V5PS51Hash (Get-V5PS51FileSha256 $FinalCandidateLock) $protocolFinalCandidateHash "Final-candidate lock file"

$bindingSpecifications = @(
    [pscustomobject]@{ Name = "common"; Path = $commonScript; Protocol = $boundOriginal; PathField = "common_path"; HashField = "common_sha256" },
    [pscustomobject]@{ Name = "selected wrapper"; Path = $originalSelectedWrapper; Protocol = $boundOriginal; PathField = "selected_wrapper_path"; HashField = "selected_wrapper_sha256" },
    [pscustomobject]@{ Name = "prediction script"; Path = $predictionScriptPath; Protocol = (Get-V5PS51RequiredProperty $boundOriginal "prediction_script" "Bound original implementation"); PathField = "path"; HashField = "sha256" },
    [pscustomobject]@{ Name = "package script"; Path = $packageScriptPath; Protocol = (Get-V5PS51RequiredProperty $boundOriginal "package_script" "Bound original implementation"); PathField = "path"; HashField = "sha256" },
    [pscustomobject]@{ Name = "validator script"; Path = $validatorScriptPath; Protocol = (Get-V5PS51RequiredProperty $boundOriginal "validator_script" "Bound original implementation"); PathField = "path"; HashField = "sha256" },
    [pscustomobject]@{ Name = "environment script"; Path = $setupScriptPath; Protocol = (Get-V5PS51RequiredProperty $boundOriginal "environment_script" "Bound original implementation"); PathField = "path"; HashField = "sha256" }
)
foreach ($binding in $bindingSpecifications) {
    $relativePath = [string] (Get-V5PS51RequiredProperty $binding.Protocol $binding.PathField "Bound $($binding.Name)")
    $expectedPath = Join-Path $projectRoot ($relativePath.Replace("/", "\"))
    Assert-V5PS51PathEqual $binding.Path $expectedPath "Bound $($binding.Name)"
    Assert-V5PS51Hash `
        (Get-V5PS51FileSha256 $binding.Path) `
        (Get-V5PS51RequiredProperty $binding.Protocol $binding.HashField "Bound $($binding.Name)") `
        "Bound $($binding.Name)"
}
$pythonBinding = Get-V5PS51RequiredProperty $boundOriginal "python_executable" "Bound original implementation"
Assert-V5PS51PathEqual `
    $PythonExecutable `
    (Get-V5PS51RequiredProperty $pythonBinding "path" "Bound Python executable") `
    "Bound Python executable"
Assert-V5PS51Hash `
    (Get-V5PS51FileSha256 $PythonExecutable) `
    (Get-V5PS51RequiredProperty $pythonBinding "sha256" "Bound Python executable") `
    "Bound Python executable"
Assert-V5PS51Hash `
    (Get-V5PS51FileSha256 $packageInitPath) `
    $script:V5PS51PackageInitSha256 `
    "Imported pancreas_multitask package initializer"

$commonText = [IO.File]::ReadAllText([IO.Path]::GetFullPath($commonScript))
if ($commonText.Length -ne [int] (Get-V5PS51RequiredProperty $boundOriginal "common_text_length" "Bound Common")) {
    throw "Original Common text length differs from the prospective protocol."
}
$substitutions = @(Get-V5PS51RequiredProperty $protocol "only_permitted_common_compatibility_substitutions" "Compatibility protocol")
if ($substitutions.Count -ne 3) {
    throw "Compatibility protocol must contain exactly three Common substitutions."
}
$expectedSubstitutions = @(
    [pscustomobject]@{ Original = '$snapshot.PSObject.Properties.Count'; Replacement = '@($snapshot.PSObject.Properties).Count' },
    [pscustomobject]@{ Original = '$runtimeImplementation.PSObject.Properties.Count'; Replacement = '@($runtimeImplementation.PSObject.Properties).Count' },
    [pscustomobject]@{ Original = '[IO.File]::Replace($temporary, $destination, $null)'; Replacement = 'Invoke-V5PS51AtomicJsonReplace -Temporary $temporary -Destination $destination' }
)
$patchedCommonText = $commonText
for ($index = 0; $index -lt $expectedSubstitutions.Count; $index++) {
    $substitution = $substitutions[$index]
    $expectedSubstitution = $expectedSubstitutions[$index]
    $original = [string] (Get-V5PS51RequiredProperty $substitution "original" "Common substitution $index")
    $replacement = [string] (Get-V5PS51RequiredProperty $substitution "replacement" "Common substitution $index")
    Assert-V5PS51Equal $original $expectedSubstitution.Original "Common substitution $index original"
    Assert-V5PS51Equal $replacement $expectedSubstitution.Replacement "Common substitution $index replacement"
    if ([int] (Get-V5PS51RequiredProperty $substitution "occurrences_required" "Common substitution $index") -ne 1) {
        throw "Common substitution $index must require exactly one occurrence."
    }
    $occurrences = [regex]::Matches($patchedCommonText, [regex]::Escape($original)).Count
    if ($occurrences -ne 1) {
        throw "Common substitution $index matched $occurrences occurrences instead of one."
    }
    $patchedCommonText = $patchedCommonText.Replace($original, $replacement)
}
$transformedCommon = Get-V5PS51RequiredProperty $protocol "transformed_common" "Compatibility protocol"
if ($patchedCommonText.Length -ne [int] (Get-V5PS51RequiredProperty $transformedCommon "text_length" "Transformed Common")) {
    throw "Transformed Common text length differs from the prospective protocol."
}
Assert-V5PS51Hash `
    (Get-V5PS51StringSha256 $patchedCommonText) `
    (Get-V5PS51RequiredProperty $transformedCommon "utf8_text_sha256" "Transformed Common") `
    "Transformed Common text"

# Dot-source only the hash-bound, exactly three-substitution Common text.
. ([ScriptBlock]::Create($patchedCommonText))

$selectedWrapperText = [IO.File]::ReadAllText([IO.Path]::GetFullPath($originalSelectedWrapper))
$bodyAnchor = [string] (Get-V5PS51RequiredProperty $boundOriginal "selected_wrapper_semantic_body_anchor" "Bound selected wrapper")
if ([regex]::Matches($selectedWrapperText, [regex]::Escape($bodyAnchor)).Count -ne 1) {
    throw "Frozen selected-wrapper semantic-body anchor must occur exactly once."
}
$bodyIndex = $selectedWrapperText.IndexOf($bodyAnchor, [StringComparison]::Ordinal)
if ($bodyIndex -lt 0) {
    throw "Frozen selected-wrapper semantic-body anchor was not found."
}
$selectedWrapperBody = $selectedWrapperText.Substring($bodyIndex)
if ($selectedWrapperBody.Length -ne [int] (Get-V5PS51RequiredProperty $boundOriginal "selected_wrapper_semantic_body_length" "Bound selected wrapper")) {
    throw "Frozen selected-wrapper semantic-body length differs from the protocol."
}
Assert-V5PS51Hash `
    (Get-V5PS51StringSha256 $selectedWrapperBody) `
    (Get-V5PS51RequiredProperty $boundOriginal "selected_wrapper_semantic_body_sha256" "Bound selected wrapper") `
    "Frozen selected-wrapper semantic body"

$frozenBindingArguments = @{
    ProjectRoot = $projectRoot
    ImplementationCommit = $CompatibilityImplementationCommit
    ProtocolPath = $protocolPath
    ProtocolSha256 = $protocolHash
    WrapperPath = $thisWrapper
    WrapperSha256 = $expectedWrapperHash
    TestsPath = $compatibilityTests
    TestsSha256 = $expectedTestsHash
    FinalCandidatePath = $FinalCandidateLock
    FinalCandidateSha256 = $protocolFinalCandidateHash
    ImplementationBindings = $bindingSpecifications
    PythonPath = $PythonExecutable
    PythonSha256 = Get-V5PS51RequiredProperty $pythonBinding "sha256" "Bound Python executable"
    PackageInitPath = $packageInitPath
    PackageInitSha256 = $script:V5PS51PackageInitSha256
    PowerShellPath = $hostExecutable
    PowerShellSha256 = $script:V5PS51PowerShellSha256
}
Assert-V5PS51FrozenCompatibilityBindings `
    @frozenBindingArguments `
    -Stage "Immediately before frozen selected-test body"

# A dynamically created ScriptBlock has an empty automatic PSScriptRoot in
# Windows PowerShell 5.1. Set it to the already hash-bound original script
# directory, then dot-source the exact, unmodified frozen body.
$selectedBodyBootstrap = '$PSScriptRoot = $script:V5PS51ScriptRoot' + [Environment]::NewLine
$selectedBodyScript = [ScriptBlock]::Create($selectedBodyBootstrap + $selectedWrapperBody)
. $selectedBodyScript
