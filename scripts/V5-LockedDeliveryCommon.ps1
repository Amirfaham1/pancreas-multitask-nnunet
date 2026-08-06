<#
.SYNOPSIS
Shared fail-closed helpers for the locked v5 delivery entry points.

.DESCRIPTION
This file deliberately contains no executable pipeline. The two v5 wrappers
dot-source it, validate the final-candidate lock and immutable artifacts before
that wrapper run opens an official input, and then use the helpers below to audit the one
fresh inference run. Keep this file in the final candidate's implementation
manifest so changes after candidate lock are rejected.
#>

Set-StrictMode -Version Latest

$script:V5CheckpointName = "checkpoint_classification_rescue.pth"
$script:V5CheckpointSha256 =
    "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116"
$script:V5PlansSha256 =
    "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f"
$script:V5DatasetJsonSha256 =
    "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff"
$script:V5EncoderSha256 =
    "324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1"
$script:V5DecoderSha256 =
    "b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2"
$script:V5ClassificationSha256 =
    "1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8"
$script:V5NeuralLockSha256 =
    "a8c2147493718acc96e4aa5dc471bf3f3277f0b99e8a8f7620bf966ab7b70d11"
$script:V5DecisionLockSha256 =
    "e28a303c7d3da5dc7857ecc72787b6746d1e689e83167c500d4d2823c5ea540f"
$script:V5SpeedLockSha256 =
    "3a57ab79147a6dd9ab4ee3fa99fdb2be978e9c60f290cead7a52298673e926aa"
$script:V5StockSpeedLockSha256 =
    "563d9d5e4fbe0f92653c6b7295c476d0ddf5d239c47beb1948410bbb80a7c2e2"
$script:V5DeterminismLockSha256 =
    "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd"
$script:V5StockExportLockSha256 =
    "bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503"
$script:V5InstalledNnunetPredictSourceSha256 =
    "c350e3202a7a67c3aef12e9206a744add442110ff8a4377c1f9640104b20a31f"
$script:V5ClassifierPipeline = "assignment_conforming_v5_neural_case_head"
$script:V5BaselineMacroF1 = 0.46399340516987575

$script:V5MandatoryImplementationFiles = @(
    "scripts/predict_joint.py",
    "scripts/evaluate_predictions.py",
    "scripts/Package-Submission.ps1",
    "scripts/validate_submission.py",
    "scripts/V5-LockedDeliveryCommon.ps1",
    "scripts/Run-V5LockedFinalEvaluation.ps1",
    "scripts/Run-V5LockedSelectedTestAndPackage.ps1",
    "src/pancreas_multitask/classification_rescue.py",
    "src/pancreas_multitask/inference_determinism.py",
    "src/pancreas_multitask/network.py",
    "src/pancreas_multitask/predictor.py",
    "src/pancreas_multitask/case_features.py",
    "src/pancreas_multitask/case_feature_extractor.py",
    "src/pancreas_multitask/neural_case_head.py",
    "src/pancreas_multitask/neural_case_bundle.py",
    "src/pancreas_multitask/neural_case_training.py",
    "src/pancreas_multitask/neural_case_predictor.py"
)
$script:V5RuntimeImplementationFiles = @(
    "scripts/predict_joint.py",
    "src/pancreas_multitask/classification_rescue.py",
    "src/pancreas_multitask/inference_determinism.py",
    "src/pancreas_multitask/network.py",
    "src/pancreas_multitask/predictor.py",
    "src/pancreas_multitask/case_features.py",
    "src/pancreas_multitask/case_feature_extractor.py",
    "src/pancreas_multitask/neural_case_head.py",
    "src/pancreas_multitask/neural_case_bundle.py",
    "src/pancreas_multitask/neural_case_training.py",
    "src/pancreas_multitask/neural_case_predictor.py"
)

function Get-V5NormalizedFullPath {
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

    return [IO.Path]::GetFullPath($Path).TrimEnd([char[]] "\/")
}

function Test-V5PathAtOrBelow {
    param(
        [Parameter(Mandatory)]
        [string] $Candidate,
        [Parameter(Mandatory)]
        [string] $Parent
    )

    $candidatePath = Get-V5NormalizedFullPath -Path $Candidate
    $parentPath = Get-V5NormalizedFullPath -Path $Parent
    if ($candidatePath.Equals($parentPath, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    return $candidatePath.StartsWith(
        $parentPath + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Get-V5RelativePath {
    param(
        [Parameter(Mandatory)]
        [string] $Parent,
        [Parameter(Mandatory)]
        [string] $Child
    )

    $parentPath = Get-V5NormalizedFullPath $Parent
    $childPath = Get-V5NormalizedFullPath $Child
    if (-not (Test-V5PathAtOrBelow -Candidate $childPath -Parent $parentPath)) {
        throw "Cannot make a relative path for a child outside its parent."
    }
    $parentUri = [Uri] ($parentPath + [IO.Path]::DirectorySeparatorChar)
    $childUri = [Uri] $childPath
    return [Uri]::UnescapeDataString($parentUri.MakeRelativeUri($childUri).ToString())
}

function Assert-V5NewSeparatedOutputRoot {
    param(
        [Parameter(Mandatory)]
        [string] $OutputRoot,
        [Parameter(Mandatory)]
        [string[]] $ProtectedPaths
    )

    $resolvedOutput = Get-V5NormalizedFullPath -Path $OutputRoot
    if (Test-Path -LiteralPath $resolvedOutput) {
        throw "V5 output root must not exist; refusing resume or replacement: '$resolvedOutput'."
    }
    foreach ($protected in $ProtectedPaths) {
        if ([string]::IsNullOrWhiteSpace($protected)) {
            continue
        }
        $resolvedProtected = Get-V5NormalizedFullPath -Path $protected
        if (
            (Test-V5PathAtOrBelow -Candidate $resolvedOutput -Parent $resolvedProtected) -or
            (Test-V5PathAtOrBelow -Candidate $resolvedProtected -Parent $resolvedOutput)
        ) {
            throw (
                "Output root must be disjoint from every input/model/lock path: " +
                "output='$resolvedOutput', protected='$resolvedProtected'."
            )
        }
    }
    return $resolvedOutput
}

function Assert-V5LeafFile {
    param(
        [Parameter(Mandatory)]
        [string] $Path,
        [Parameter(Mandatory)]
        [string] $Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: '$Path'."
    }
    if ((Get-Item -LiteralPath $Path).Length -le 0) {
        throw "$Description is empty: '$Path'."
    }
}

function Assert-V5Directory {
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

function ConvertTo-V5Sha256 {
    param(
        [Parameter(Mandatory)]
        [object] $Value,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $normalized = ([string] $Value).Trim().ToLowerInvariant()
    if ($normalized -notmatch "^[0-9a-f]{64}$") {
        throw "$Description must be one lowercase 64-digit SHA-256 digest."
    }
    return $normalized
}

function Get-V5FileSha256 {
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

    Assert-V5LeafFile -Path $Path -Description "File to hash"
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-V5StringSha256 {
    param(
        [Parameter(Mandatory)]
        [string] $Value
    )

    $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Assert-V5HashEquals {
    param(
        [Parameter(Mandatory)]
        [object] $Actual,
        [Parameter(Mandatory)]
        [object] $Expected,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $actualHash = ConvertTo-V5Sha256 -Value $Actual -Description "$Description actual hash"
    $expectedHash = ConvertTo-V5Sha256 -Value $Expected -Description "$Description expected hash"
    if (-not $actualHash.Equals($expectedHash, [StringComparison]::Ordinal)) {
        throw "$Description SHA-256 mismatch: expected=$expectedHash actual=$actualHash."
    }
}

function Read-V5JsonObject {
    param(
        [Parameter(Mandatory)]
        [string] $Path,
        [Parameter(Mandatory)]
        [string] $Description
    )

    Assert-V5LeafFile -Path $Path -Description $Description
    try {
        $payload = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        throw "Could not parse $Description '$Path': $($_.Exception.Message)"
    }
    if ($null -eq $payload -or $payload -is [Array]) {
        throw "$Description must contain exactly one JSON object: '$Path'."
    }
    return $payload
}

function Get-V5RequiredProperty {
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

function Assert-V5ExactValue {
    param(
        [Parameter(Mandatory)]
        [object] $Actual,
        [Parameter(Mandatory)]
        [object] $Expected,
        [Parameter(Mandatory)]
        [string] $Description
    )

    if ([string] $Actual -cne [string] $Expected) {
        throw "$Description must equal '$Expected'; got '$Actual'."
    }
}

function Assert-V5Boolean {
    param(
        [Parameter(Mandatory)]
        [object] $Actual,
        [Parameter(Mandatory)]
        [bool] $Expected,
        [Parameter(Mandatory)]
        [string] $Description
    )

    if ($Actual -isnot [bool] -or [bool] $Actual -ne $Expected) {
        throw "$Description must be the JSON boolean $($Expected.ToString().ToLowerInvariant())."
    }
}

function Write-V5JsonAtomic {
    param(
        [Parameter(Mandatory)]
        [object] $Payload,
        [Parameter(Mandatory)]
        [string] $Path
    )

    $destination = [IO.Path]::GetFullPath($Path)
    $parent = [IO.Path]::GetDirectoryName($destination)
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        $null = New-Item -ItemType Directory -Path $parent
    }
    $temporary = Join-Path $parent (".{0}.{1}.tmp" -f [IO.Path]::GetFileName($destination), [Guid]::NewGuid().ToString("N"))
    try {
        $json = $Payload | ConvertTo-Json -Depth 32
        $encoding = [Text.UTF8Encoding]::new($false)
        $stream = [IO.File]::Open(
            $temporary,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None
        )
        try {
            $bytes = $encoding.GetBytes($json + "`n")
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush($true)
        }
        finally {
            $stream.Dispose()
        }
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            [IO.File]::Replace($temporary, $destination, $null)
        }
        else {
            [IO.File]::Move($temporary, $destination)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

function New-V5ExclusiveLedger {
    param(
        [Parameter(Mandatory)]
        [string] $Path,
        [Parameter(Mandatory)]
        [object] $Payload
    )

    $destination = [IO.Path]::GetFullPath($Path)
    $parent = [IO.Path]::GetDirectoryName($destination)
    Assert-V5Directory -Path $parent -Description "Run-ledger parent"
    $json = ($Payload | ConvertTo-Json -Depth 32) + "`n"
    $encoding = [Text.UTF8Encoding]::new($false)
    try {
        $stream = [IO.File]::Open(
            $destination,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None
        )
    }
    catch [IO.IOException] {
        throw (
            "The one-use run ledger already exists or could not be created; " +
            "this candidate/stage cannot be run again: '$destination'."
        )
    }
    try {
        $bytes = $encoding.GetBytes($json)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
    }
    return $destination
}

function Get-V5BareLedgerPath {
    param(
        [Parameter(Mandatory)]
        [object] $Lock,
        [Parameter(Mandatory)]
        [ValidateSet("official_validation", "selected_test")]
        [string] $Stage,
        [Parameter(Mandatory)]
        [string] $FinalCandidateLock
    )

    $ledgers = Get-V5RequiredProperty $Lock "run_ledger_files" "Final-candidate lock"
    $filename = [string] (Get-V5RequiredProperty $ledgers $Stage "Run-ledger contract")
    if (
        [string]::IsNullOrWhiteSpace($filename) -or
        [IO.Path]::GetFileName($filename) -cne $filename -or
        -not $filename.EndsWith(".json", [StringComparison]::Ordinal)
    ) {
        throw "Run-ledger filename for '$Stage' must be one bare, lowercase-suffix .json filename."
    }
    return Join-Path ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($FinalCandidateLock))) $filename
}

function Assert-V5FixedCallerHashes {
    param(
        [Parameter(Mandatory)]
        [string] $ExpectedCheckpointSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedPlansSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedDatasetJsonSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedEncoderComponentSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedDecoderComponentSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedClassificationComponentSha256
    )

    Assert-V5HashEquals $ExpectedCheckpointSha256 $script:V5CheckpointSha256 "Caller checkpoint contract"
    Assert-V5HashEquals $ExpectedPlansSha256 $script:V5PlansSha256 "Caller plans contract"
    Assert-V5HashEquals $ExpectedDatasetJsonSha256 $script:V5DatasetJsonSha256 "Caller dataset.json contract"
    Assert-V5HashEquals $ExpectedEncoderComponentSha256 $script:V5EncoderSha256 "Caller encoder-component contract"
    Assert-V5HashEquals $ExpectedDecoderComponentSha256 $script:V5DecoderSha256 "Caller decoder-component contract"
    Assert-V5HashEquals $ExpectedClassificationComponentSha256 $script:V5ClassificationSha256 "Caller classification-component contract"
}

function Assert-V5ProtocolLocks {
    param(
        [Parameter(Mandatory)]
        [object] $Lock,
        [Parameter(Mandatory)]
        [string] $ProjectRoot
    )

    $protocolLocks = Get-V5RequiredProperty $Lock "protocol_locks" "Final-candidate lock"
    $contracts = @(
        [pscustomobject]@{
            Name = "neural_case_head"
            Path = "configs/phd_neural_case_head_lock_v5.json"
            Sha256 = $script:V5NeuralLockSha256
        },
        [pscustomobject]@{
            Name = "neural_decision"
            Path = "configs/phd_neural_decision_lock_v5.json"
            Sha256 = $script:V5DecisionLockSha256
        },
        [pscustomobject]@{
            Name = "inference_speed"
            Path = "configs/inference_speed_benchmark_v3.json"
            Sha256 = $script:V5SpeedLockSha256
        },
        [pscustomobject]@{
            Name = "inference_speed_stock_gate"
            Path = "configs/inference_speed_stock_gate_v1.json"
            Sha256 = $script:V5StockSpeedLockSha256
        },
        [pscustomobject]@{
            Name = "inference_determinism"
            Path = "configs/inference_determinism_conformance_v1.json"
            Sha256 = $script:V5DeterminismLockSha256
        },
        [pscustomobject]@{
            Name = "stock_export_conformance"
            Path = "configs/inference_stock_export_conformance_v1.json"
            Sha256 = $script:V5StockExportLockSha256
        }
    )
    foreach ($contract in $contracts) {
        $entry = Get-V5RequiredProperty $protocolLocks $contract.Name "Protocol-lock bindings"
        Assert-V5ExactValue `
            (Get-V5RequiredProperty $entry "path" "Protocol lock '$($contract.Name)'") `
            $contract.Path `
            "Protocol lock '$($contract.Name)' path"
        Assert-V5HashEquals `
            (Get-V5RequiredProperty $entry "sha256" "Protocol lock '$($contract.Name)'") `
            $contract.Sha256 `
            "Protocol lock '$($contract.Name)' lock binding"
        $actualPath = Join-Path $ProjectRoot ($contract.Path.Replace("/", [IO.Path]::DirectorySeparatorChar))
        Assert-V5HashEquals `
            (Get-V5FileSha256 -Path $actualPath) `
            $contract.Sha256 `
            "Protocol lock '$($contract.Name)' file"
    }
}

function Assert-V5ImplementationManifest {
    param(
        [Parameter(Mandatory)]
        [object] $Lock,
        [Parameter(Mandatory)]
        [string] $ProjectRoot
    )

    $entries = @(Get-V5RequiredProperty $Lock "implementation_files" "Final-candidate lock")
    if ($entries.Count -lt $script:V5MandatoryImplementationFiles.Count) {
        throw "Final-candidate implementation manifest is incomplete."
    }
    $seen = @{}
    foreach ($entry in $entries) {
        $relative = [string] (Get-V5RequiredProperty $entry "path" "Implementation entry")
        if (
            [string]::IsNullOrWhiteSpace($relative) -or
            [IO.Path]::IsPathRooted($relative) -or
            $relative.Contains("\") -or
            $relative.StartsWith("../", [StringComparison]::Ordinal) -or
            $relative.Contains("/../")
        ) {
            throw "Implementation paths must be normalized project-relative forward-slash paths: '$relative'."
        }
        if ($seen.ContainsKey($relative)) {
            throw "Duplicate implementation path in final-candidate lock: '$relative'."
        }
        $seen[$relative] = $true
        $candidate = Join-Path $ProjectRoot ($relative.Replace("/", [IO.Path]::DirectorySeparatorChar))
        if (-not (Test-V5PathAtOrBelow -Candidate $candidate -Parent $ProjectRoot)) {
            throw "Implementation path escapes the project root: '$relative'."
        }
        Assert-V5HashEquals `
            (Get-V5FileSha256 -Path $candidate) `
            (Get-V5RequiredProperty $entry "sha256" "Implementation entry '$relative'") `
            "Implementation file '$relative'"
    }
    foreach ($required in $script:V5MandatoryImplementationFiles) {
        if (-not $seen.ContainsKey($required)) {
            throw "Final-candidate lock does not bind mandatory implementation file '$required'."
        }
    }
}

function Assert-V5TrainOnlyAudits {
    param(
        [Parameter(Mandatory)]
        [object] $Lock,
        [Parameter(Mandatory)]
        [string] $BundlePath,
        [Parameter(Mandatory)]
        [string] $ExpectedBundleSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedNumericTrainDatasetSha256
    )

    $auditBindings = Get-V5RequiredProperty $Lock "train_only_audits" "Final-candidate lock"
    $contracts = @(
        [pscustomobject]@{ Role = "fit"; Filename = "neural_case_head_fit_audit.json" },
        [pscustomobject]@{ Role = "selection"; Filename = "neural_case_head_selection.json" },
        [pscustomobject]@{ Role = "decision"; Filename = "neural_decision_calibration.json" },
        [pscustomobject]@{ Role = "refit"; Filename = "neural_case_head_refit.json" }
    )
    $bundleDirectory = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($BundlePath))
    $audits = @{}
    $hashes = @{}
    $paths = @{}
    foreach ($contract in $contracts) {
        $entry = Get-V5RequiredProperty $auditBindings $contract.Role "Train-only audit bindings"
        Assert-V5ExactValue `
            (Get-V5RequiredProperty $entry "filename" "Train-only '$($contract.Role)' audit") `
            $contract.Filename `
            "Train-only '$($contract.Role)' filename"
        $expectedHash = ConvertTo-V5Sha256 `
            (Get-V5RequiredProperty $entry "sha256" "Train-only '$($contract.Role)' audit") `
            "Train-only '$($contract.Role)' audit hash"
        $path = Join-Path $bundleDirectory $contract.Filename
        Assert-V5HashEquals (Get-V5FileSha256 $path) $expectedHash "Train-only '$($contract.Role)' audit file"
        $audits[$contract.Role] = Read-V5JsonObject $path "Train-only '$($contract.Role)' audit"
        $hashes[$contract.Role] = $expectedHash
        $paths[$contract.Role] = $path
    }

    $fit = $audits["fit"]
    if ([int] (Get-V5RequiredProperty $fit "schema_version" "Train-only fit audit") -ne 1) {
        throw "Train-only fit audit schema_version must be 1."
    }
    Assert-V5ExactValue (Get-V5RequiredProperty $fit "status" "Train-only fit audit") "complete" "Train-only fit status"
    Assert-V5ExactValue (Get-V5RequiredProperty $fit "scope" "Train-only fit audit") "isolated_supplied_train_only" "Train-only fit scope"
    Assert-V5ExactValue (Get-V5RequiredProperty $fit "eligible_comparison" "Train-only fit audit") "best_of_two_locked_neural_heads" "Train-only fit comparison"
    Assert-V5HashEquals (Get-V5RequiredProperty $fit "bundle_sha256" "Train-only fit audit") $ExpectedBundleSha256 "Train-only fit bundle binding"
    Assert-V5HashEquals (Get-V5RequiredProperty $fit "numeric_content_dataset_sha256" "Train-only fit audit") $ExpectedNumericTrainDatasetSha256 "Train-only fit numeric-dataset binding"
    Assert-V5HashEquals (Get-V5RequiredProperty $fit "selection_audit_sha256" "Train-only fit audit") $hashes["selection"] "Train-only selection binding"
    Assert-V5HashEquals (Get-V5RequiredProperty $fit "calibration_audit_sha256" "Train-only fit audit") $hashes["decision"] "Train-only decision binding"
    Assert-V5HashEquals (Get-V5RequiredProperty $fit "refit_audit_sha256" "Train-only fit audit") $hashes["refit"] "Train-only refit binding"
    Assert-V5HashEquals (Get-V5RequiredProperty $fit "neural_lock_sha256" "Train-only fit audit") $script:V5NeuralLockSha256 "Train-only fit neural lock"
    Assert-V5HashEquals (Get-V5RequiredProperty $fit "decision_lock_sha256" "Train-only fit audit") $script:V5DecisionLockSha256 "Train-only fit decision lock"
    Assert-V5Boolean (Get-V5RequiredProperty $fit "encoder_decoder_and_rescue_head_frozen" "Train-only fit audit") $true "Train-only frozen-network flag"
    Assert-V5Boolean (Get-V5RequiredProperty $fit "ground_truth_masks_used_as_features" "Train-only fit audit") $false "Train-only ground-truth feature flag"
    Assert-V5Boolean (Get-V5RequiredProperty $fit "case_ids_paths_filenames_or_order_used" "Train-only fit audit") $false "Train-only identifier feature flag"
    Assert-V5Boolean (Get-V5RequiredProperty $fit "official_validation_or_test_used" "Train-only fit audit") $false "Train-only official-data flag"
    $selectedCandidate = [string] (Get-V5RequiredProperty $fit "selected_candidate_id" "Train-only fit audit")
    if ($selectedCandidate -notin @("neural_lesion_mean_mil", "neural_two_query_cross_attention_mil")) {
        throw "Train-only fit audit selected an ineligible candidate."
    }
    $wandb = Get-V5RequiredProperty $fit "wandb_run" "Train-only fit audit"
    Assert-V5ExactValue (Get-V5RequiredProperty $wandb "effective_mode" "Train-only W&B evidence") "online" "Train-only W&B mode"
    foreach ($field in @("entity", "project", "run_id", "run_name", "run_url")) {
        if ([string]::IsNullOrWhiteSpace([string] (Get-V5RequiredProperty $wandb $field "Train-only W&B evidence"))) {
            throw "Train-only W&B evidence '$field' cannot be blank."
        }
    }

    $selection = $audits["selection"]
    Assert-V5ExactValue (Get-V5RequiredProperty $selection "status" "Train-only selection audit") "complete" "Train-only selection status"
    Assert-V5ExactValue (Get-V5RequiredProperty $selection "scope" "Train-only selection audit") "isolated_supplied_train_only" "Train-only selection scope"
    Assert-V5ExactValue (Get-V5RequiredProperty $selection "selected_candidate_id" "Train-only selection audit") $selectedCandidate "Train-only selected candidate"
    Assert-V5HashEquals (Get-V5RequiredProperty $selection "numeric_content_dataset_sha256" "Train-only selection audit") $ExpectedNumericTrainDatasetSha256 "Train-only selection numeric dataset"
    Assert-V5Boolean (Get-V5RequiredProperty $selection "official_validation_images_masks_labels_or_metrics_used" "Train-only selection audit") $false "Train-only selection official-validation flag"
    Assert-V5Boolean (Get-V5RequiredProperty $selection "test_data_used" "Train-only selection audit") $false "Train-only selection test flag"
    Assert-V5Boolean (Get-V5RequiredProperty $selection "ground_truth_masks_used_as_features" "Train-only selection audit") $false "Train-only selection ground-truth feature flag"

    $decision = $audits["decision"]
    Assert-V5ExactValue (Get-V5RequiredProperty $decision "status" "Train-only decision audit") "complete" "Train-only decision status"
    Assert-V5Boolean (Get-V5RequiredProperty $decision "eligible_for_official" "Train-only decision audit") $true "Train-only decision eligibility"
    Assert-V5Boolean (Get-V5RequiredProperty $decision "official_validation_or_test_used" "Train-only decision audit") $false "Train-only decision official-data flag"
    Assert-V5HashEquals (Get-V5RequiredProperty $decision "numeric_content_dataset_sha256" "Train-only decision audit") $ExpectedNumericTrainDatasetSha256 "Train-only decision numeric dataset"
    Assert-V5HashEquals (Get-V5RequiredProperty $decision "selection_audit_sha256" "Train-only decision audit") $hashes["selection"] "Train-only decision selection binding"

    $refit = $audits["refit"]
    Assert-V5ExactValue (Get-V5RequiredProperty $refit "status" "Train-only refit audit") "complete" "Train-only refit status"
    Assert-V5ExactValue (Get-V5RequiredProperty $refit "candidate_id" "Train-only refit audit") $selectedCandidate "Train-only refit candidate"
    Assert-V5HashEquals (Get-V5RequiredProperty $refit "numeric_content_dataset_sha256" "Train-only refit audit") $ExpectedNumericTrainDatasetSha256 "Train-only refit numeric dataset"
    Assert-V5HashEquals (Get-V5RequiredProperty $refit "selection_audit_sha256" "Train-only refit audit") $hashes["selection"] "Train-only refit selection binding"
    Assert-V5HashEquals (Get-V5RequiredProperty $refit "calibration_audit_sha256" "Train-only refit audit") $hashes["decision"] "Train-only refit decision binding"
    Assert-V5Boolean (Get-V5RequiredProperty $refit "official_validation_or_test_used" "Train-only refit audit") $false "Train-only refit official-data flag"
    $refitStateHash = ConvertTo-V5Sha256 (Get-V5RequiredProperty $refit "final_state_sha256" "Train-only refit audit") "Train-only refit state hash"

    return [pscustomobject]@{
        Fit = $fit
        Selection = $selection
        Decision = $decision
        Refit = $refit
        Hashes = $hashes
        Paths = $paths
        SelectedCandidateId = $selectedCandidate
        RefitStateSha256 = $refitStateHash
    }
}

function Assert-V5FinalCandidateLock {
    param(
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
        [string] $ProjectRoot
    )

    Assert-V5FixedCallerHashes `
        -ExpectedCheckpointSha256 $ExpectedCheckpointSha256 `
        -ExpectedPlansSha256 $ExpectedPlansSha256 `
        -ExpectedDatasetJsonSha256 $ExpectedDatasetJsonSha256 `
        -ExpectedEncoderComponentSha256 $ExpectedEncoderComponentSha256 `
        -ExpectedDecoderComponentSha256 $ExpectedDecoderComponentSha256 `
        -ExpectedClassificationComponentSha256 $ExpectedClassificationComponentSha256
    $expectedLockHash = ConvertTo-V5Sha256 $ExpectedFinalCandidateLockSha256 "Final-candidate lock hash"
    $expectedBundleHash = ConvertTo-V5Sha256 $ExpectedNeuralCaseHeadBundleSha256 "Neural bundle hash"
    $expectedNumericHash = ConvertTo-V5Sha256 $ExpectedNumericTrainDatasetSha256 "Numeric training-dataset hash"

    $resolvedLock = Get-V5NormalizedFullPath $FinalCandidateLock
    Assert-V5HashEquals (Get-V5FileSha256 $resolvedLock) $expectedLockHash "Final-candidate lock file"
    $lock = Read-V5JsonObject $resolvedLock "Final-candidate lock"
    if ([int] (Get-V5RequiredProperty $lock "schema_version" "Final-candidate lock") -ne 1) {
        throw "Final-candidate lock schema_version must be 1."
    }
    Assert-V5ExactValue `
        (Get-V5RequiredProperty $lock "status" "Final-candidate lock") `
        "locked_before_single_v5_official_reevaluation_and_selected_v5_test_inference" `
        "Final-candidate lock status"
    Assert-V5ExactValue `
        (Get-V5RequiredProperty $lock "candidate_family" "Final-candidate lock") `
        $script:V5ClassifierPipeline `
        "Final-candidate family"
    Assert-V5ExactValue `
        (Get-V5RequiredProperty $lock "eligibility_scope" "Final-candidate lock") `
        "best_of_two_locked_neural_heads" `
        "Final-candidate eligibility scope"
    $boundary = Get-V5RequiredProperty $lock "development_boundary" "Final-candidate lock"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $boundary "v5_head_architecture_training_selection_and_offsets_used_official_validation_or_test_data" "Development boundary") `
        $false `
        "V5 neural-head official-data flag"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $boundary "v5_speed_development_used_official_validation_or_test_data" "Development boundary") `
        $false `
        "V5 speed-development official-data flag"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $boundary "baseline_official_validation_observed_before_v5_extension" "Development boundary") `
        $true `
        "Pre-v5 baseline-validation disclosure"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $boundary "frozen_checkpoint_was_validation_selected" "Development boundary") `
        $true `
        "Validation-selected checkpoint disclosure"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $boundary "baseline_test_inference_and_packaging_occurred_before_v5_extension" "Development boundary") `
        $true `
        "Pre-v5 baseline-test disclosure"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $boundary "baseline_test_inputs_were_a_v5_tuning_signal" "Development boundary") `
        $false `
        "V5 test-tuning-signal disclosure"

    $artifacts = Get-V5RequiredProperty $lock "artifacts" "Final-candidate lock"
    $checkpoint = Get-V5RequiredProperty $artifacts "checkpoint" "Final-candidate artifacts"
    Assert-V5ExactValue `
        (Get-V5RequiredProperty $checkpoint "name" "Checkpoint lock entry") `
        $script:V5CheckpointName `
        "Locked checkpoint name"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $checkpoint "sha256" "Checkpoint lock entry") `
        $ExpectedCheckpointSha256 `
        "Checkpoint lock/caller binding"
    $bundle = Get-V5RequiredProperty $artifacts "neural_case_head_bundle" "Final-candidate artifacts"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $bundle "sha256" "Neural-bundle lock entry") `
        $expectedBundleHash `
        "Neural-bundle lock/caller binding"
    $numeric = Get-V5RequiredProperty $artifacts "numeric_train_dataset" "Final-candidate artifacts"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $numeric "sha256" "Numeric-dataset lock entry") `
        $expectedNumericHash `
        "Numeric-dataset lock/caller binding"
    $plans = Get-V5RequiredProperty $artifacts "plans_json" "Final-candidate artifacts"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $plans "sha256" "Plans lock entry") `
        $ExpectedPlansSha256 `
        "Plans lock/caller binding"
    $datasetJson = Get-V5RequiredProperty $artifacts "dataset_json" "Final-candidate artifacts"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $datasetJson "sha256" "Dataset lock entry") `
        $ExpectedDatasetJsonSha256 `
        "Dataset lock/caller binding"

    $components = Get-V5RequiredProperty $lock "frozen_components" "Final-candidate lock"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $components "encoder" "Frozen-component lock") `
        $ExpectedEncoderComponentSha256 `
        "Encoder-component lock/caller binding"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $components "decoder" "Frozen-component lock") `
        $ExpectedDecoderComponentSha256 `
        "Decoder-component lock/caller binding"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $components "classification" "Frozen-component lock") `
        $ExpectedClassificationComponentSha256 `
        "Classification-component lock/caller binding"

    $inference = Get-V5RequiredProperty $lock "inference_contract" "Final-candidate lock"
    if ([int] (Get-V5RequiredProperty $inference "fold" "Inference contract") -ne 0) {
        throw "V5 inference contract must select exactly fold 0."
    }
    Assert-V5ExactValue (Get-V5RequiredProperty $inference "classification_mode" "Inference contract") "neural-v5" "Classification mode"
    Assert-V5ExactValue (Get-V5RequiredProperty $inference "v5_extraction_mode" "Inference contract") "neural_only" "V5 extraction mode"
    Assert-V5ExactValue (Get-V5RequiredProperty $inference "device" "Inference contract") "cuda" "V5 inference device"
    if ([double] (Get-V5RequiredProperty $inference "tile_step_size" "Inference contract") -ne 0.5) {
        throw "V5 inference contract tile_step_size must equal 0.5."
    }
    if ([int] (Get-V5RequiredProperty $inference "tile_batch_size" "Inference contract") -ne 1) {
        throw "V5 inference contract tile_batch_size must equal 1."
    }
    if ([int] (Get-V5RequiredProperty $inference "tta_batch_size" "Inference contract") -ne 1) {
        throw "V5 inference contract tta_batch_size must equal 1."
    }
    Assert-V5Boolean (Get-V5RequiredProperty $inference "tta_enabled" "Inference contract") $true "TTA contract"
    Assert-V5Boolean (Get-V5RequiredProperty $inference "gaussian_enabled" "Inference contract") $true "Gaussian contract"
    Assert-V5Boolean (Get-V5RequiredProperty $inference "overwrite" "Inference contract") $true "Fresh-inference contract"
    Assert-V5Boolean (Get-V5RequiredProperty $inference "results_on_cpu" "Inference contract") $false "V5 result-accumulation contract"
    Assert-V5Boolean (Get-V5RequiredProperty $inference "deterministic_execution" "Inference contract") $true "V5 deterministic-execution contract"
    Assert-V5Boolean (Get-V5RequiredProperty $inference "autocast_cuda_float16" "Inference contract") $true "V5 CUDA autocast contract"

    $officialLedgerPath = Get-V5BareLedgerPath `
        -Lock $lock `
        -Stage official_validation `
        -FinalCandidateLock $resolvedLock
    $selectedTestLedgerPath = Get-V5BareLedgerPath `
        -Lock $lock `
        -Stage selected_test `
        -FinalCandidateLock $resolvedLock
    if ($officialLedgerPath.Equals($selectedTestLedgerPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Official-validation and selected-test one-use ledgers must be different files."
    }
    foreach ($runLedgerPath in @($officialLedgerPath, $selectedTestLedgerPath)) {
        if ($runLedgerPath.Equals($resolvedLock, [StringComparison]::OrdinalIgnoreCase)) {
            throw "A one-use run ledger cannot replace the final-candidate lock."
        }
    }

    $resolvedModel = Get-V5NormalizedFullPath $ModelDirectory
    $resolvedBundle = Get-V5NormalizedFullPath $NeuralCaseHeadBundle
    $checkpointPath = Join-Path (Join-Path $resolvedModel "fold_0") $script:V5CheckpointName
    $plansPath = Join-Path $resolvedModel "plans.json"
    $datasetPath = Join-Path $resolvedModel "dataset.json"
    Assert-V5HashEquals (Get-V5FileSha256 $checkpointPath) $ExpectedCheckpointSha256 "Frozen checkpoint file"
    Assert-V5HashEquals (Get-V5FileSha256 $resolvedBundle) $expectedBundleHash "Neural case-head bundle file"
    Assert-V5HashEquals (Get-V5FileSha256 $plansPath) $ExpectedPlansSha256 "Model plans.json file"
    Assert-V5HashEquals (Get-V5FileSha256 $datasetPath) $ExpectedDatasetJsonSha256 "Model dataset.json file"

    Assert-V5ProtocolLocks -Lock $lock -ProjectRoot $ProjectRoot
    Assert-V5ImplementationManifest -Lock $lock -ProjectRoot $ProjectRoot
    $trainOnlyAudits = Assert-V5TrainOnlyAudits `
        -Lock $lock `
        -BundlePath $resolvedBundle `
        -ExpectedBundleSha256 $expectedBundleHash `
        -ExpectedNumericTrainDatasetSha256 $expectedNumericHash
    return [pscustomobject]@{
        Lock = $lock
        LockPath = $resolvedLock
        LockSha256 = $expectedLockHash
        ModelDirectory = $resolvedModel
        CheckpointPath = $checkpointPath
        BundlePath = $resolvedBundle
        BundleSha256 = $expectedBundleHash
        NumericTrainDatasetSha256 = $expectedNumericHash
        PlansPath = $plansPath
        DatasetJsonPath = $datasetPath
        TrainOnlyAudits = $trainOnlyAudits
        ProjectRoot = (Get-V5NormalizedFullPath $ProjectRoot)
    }
}

function Invoke-V5StrictCpuPreflight {
    param(
        [Parameter(Mandatory)]
        [string] $PythonExecutable,
        [Parameter(Mandatory)]
        [string] $ProjectRoot,
        [Parameter(Mandatory)]
        [object] $Candidate,
        [Parameter(Mandatory)]
        [string] $ExpectedEncoderComponentSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedDecoderComponentSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedClassificationComponentSha256
    )

    Assert-V5LeafFile -Path $PythonExecutable -Description "Python executable"
    $variables = @{
        V5_PREFLIGHT_SRC = (Join-Path $ProjectRoot "src")
        V5_PREFLIGHT_MODEL = $Candidate.ModelDirectory
        V5_PREFLIGHT_CHECKPOINT = $script:V5CheckpointName
        V5_PREFLIGHT_BUNDLE = $Candidate.BundlePath
        V5_PREFLIGHT_BUNDLE_SHA = $Candidate.BundleSha256
        V5_PREFLIGHT_DATASET_SHA = $Candidate.NumericTrainDatasetSha256
    }
    foreach ($entry in $variables.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, [string] $entry.Value, "Process")
    }
    $python = @'
import json
import os
import sys

sys.path.insert(0, os.environ["V5_PREFLIGHT_SRC"])
import torch
from pancreas_multitask.neural_case_bundle import load_neural_case_head_bundle
from pancreas_multitask.neural_case_training import neural_state_sha256
from pancreas_multitask.neural_case_predictor import NeuralCaseNNUNetPredictor

head, offsets, metadata = load_neural_case_head_bundle(
    os.environ["V5_PREFLIGHT_BUNDLE"],
    torch.device("cpu"),
    expected_bundle_sha256=os.environ["V5_PREFLIGHT_BUNDLE_SHA"],
    expected_numeric_dataset_sha256=os.environ["V5_PREFLIGHT_DATASET_SHA"],
)
head_state = neural_state_sha256(head)
predictor = NeuralCaseNNUNetPredictor(
    tile_step_size=0.5,
    tile_batch_size=1,
    tta_batch_size=1,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=False,
    device=torch.device("cpu"),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=False,
    neural_case_head_bundle=os.environ["V5_PREFLIGHT_BUNDLE"],
    expected_neural_case_head_bundle_sha256=os.environ["V5_PREFLIGHT_BUNDLE_SHA"],
    expected_numeric_train_dataset_sha256=os.environ["V5_PREFLIGHT_DATASET_SHA"],
    v5_extraction_mode="neural_only",
)
predictor.initialize_from_trained_model_folder(
    os.environ["V5_PREFLIGHT_MODEL"],
    use_folds=(0,),
    checkpoint_name=os.environ["V5_PREFLIGHT_CHECKPOINT"],
)
network = predictor.frozen_network_provenance()
payload = {
    "bundle": {
        "selected_candidate_id": metadata["selected_candidate_id"],
        "numeric_train_dataset_sha256": metadata["numeric_content_dataset_sha256"],
        "head_state_sha256": head_state,
        "eligible_for_official": metadata["eligible_for_official"],
        "offsets": [float(value) for value in offsets],
        "selection_audit_sha256": metadata["selection_audit_sha256"],
        "calibration_audit_sha256": metadata["calibration_audit_sha256"],
        "refit_audit_sha256": metadata["refit_audit_sha256"],
        "neural_lock_sha256": metadata["neural_lock_sha256"],
        "decision_lock_sha256": metadata["decision_lock_sha256"],
    },
    "network": network,
}
print("__V5_STRICT_PREFLIGHT__=" + json.dumps(payload, sort_keys=True))
'@
    $messages = @()
    $exitCode = $null
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        # Feed source on stdin. Windows native argument parsing can otherwise
        # remove quotes inside a multiline ``python -c`` payload.
        $messages = @($python | & $PythonExecutable - 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldPreference
        foreach ($name in $variables.Keys) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
    }
    if ($exitCode -ne 0) {
        $rendered = ($messages | ForEach-Object { [string] $_ }) -join "`n"
        throw "Strict CPU model/bundle preflight failed with exit code $exitCode.`n$rendered"
    }
    $markers = @(
        $messages | ForEach-Object { [string] $_ } | Where-Object {
            $_.StartsWith("__V5_STRICT_PREFLIGHT__=", [StringComparison]::Ordinal)
        }
    )
    if ($markers.Count -ne 1) {
        throw "Strict CPU model/bundle preflight emitted $($markers.Count) result markers; expected one."
    }
    try {
        $payload = $markers[0].Substring("__V5_STRICT_PREFLIGHT__=".Length) | ConvertFrom-Json
    }
    catch {
        throw "Strict CPU preflight marker was not valid JSON: $($_.Exception.Message)"
    }
    $network = Get-V5RequiredProperty $payload "network" "CPU preflight"
    $before = Get-V5RequiredProperty $network "component_hashes_before" "CPU network preflight"
    $after = Get-V5RequiredProperty $network "component_hashes_after" "CPU network preflight"
    foreach ($contract in @(
        [pscustomobject]@{ Name = "encoder"; Expected = $ExpectedEncoderComponentSha256 },
        [pscustomobject]@{ Name = "decoder"; Expected = $ExpectedDecoderComponentSha256 },
        [pscustomobject]@{ Name = "classification"; Expected = $ExpectedClassificationComponentSha256 }
    )) {
        Assert-V5HashEquals (Get-V5RequiredProperty $before $contract.Name "CPU components before") $contract.Expected "CPU $($contract.Name) component before"
        Assert-V5HashEquals (Get-V5RequiredProperty $after $contract.Name "CPU components after") $contract.Expected "CPU $($contract.Name) component after"
    }
    Assert-V5Boolean (Get-V5RequiredProperty $network "frozen_components_unchanged" "CPU network preflight") $true "CPU frozen-component state"
    Assert-V5Boolean (Get-V5RequiredProperty $network "any_network_parameter_requires_grad" "CPU network preflight") $false "CPU network gradient state"
    $bundle = Get-V5RequiredProperty $payload "bundle" "CPU preflight"
    Assert-V5Boolean (Get-V5RequiredProperty $bundle "eligible_for_official" "CPU bundle preflight") $true "CPU bundle eligibility"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "numeric_train_dataset_sha256" "CPU bundle preflight") $Candidate.NumericTrainDatasetSha256 "CPU bundle numeric-dataset binding"
    Assert-V5ExactValue (Get-V5RequiredProperty $bundle "selected_candidate_id" "CPU bundle preflight") $Candidate.TrainOnlyAudits.SelectedCandidateId "CPU bundle selected candidate"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "head_state_sha256" "CPU bundle preflight") $Candidate.TrainOnlyAudits.RefitStateSha256 "CPU bundle refit state"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "selection_audit_sha256" "CPU bundle preflight") $Candidate.TrainOnlyAudits.Hashes["selection"] "CPU bundle selection audit"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "calibration_audit_sha256" "CPU bundle preflight") $Candidate.TrainOnlyAudits.Hashes["decision"] "CPU bundle decision audit"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "refit_audit_sha256" "CPU bundle preflight") $Candidate.TrainOnlyAudits.Hashes["refit"] "CPU bundle refit audit"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "neural_lock_sha256" "CPU bundle preflight") $script:V5NeuralLockSha256 "CPU bundle neural lock"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "decision_lock_sha256" "CPU bundle preflight") $script:V5DecisionLockSha256 "CPU bundle decision lock"
    return $payload
}

function Invoke-V5CheckedPython {
    param(
        [Parameter(Mandatory)]
        [string] $PythonExecutable,
        [Parameter(Mandatory)]
        [string] $ScriptPath,
        [Parameter(Mandatory)]
        [string[]] $Arguments,
        [Parameter(Mandatory)]
        [string] $Stage
    )

    $allArguments = @($ScriptPath) + $Arguments
    & $PythonExecutable @allArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE. The one-use run remains consumed."
    }
}

function Assert-V5DeterministicRuntime {
    param(
        [Parameter(Mandatory)]
        [object] $Runtime,
        [Parameter(Mandatory)]
        [object] $Candidate,
        [Parameter(Mandatory)]
        [string] $ExpectedDevice
    )

    $execution = Get-V5RequiredProperty $Runtime "deterministic_execution" "Runtime artifact"
    Assert-V5ExactValue `
        (Get-V5RequiredProperty $execution "policy" "Runtime deterministic execution") `
        "strict_cuda_inference_v1" `
        "Runtime deterministic policy"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $execution "configured_before_cuda_initialization" "Runtime deterministic execution") `
        $true `
        "Pre-CUDA deterministic configuration"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $execution "settings_unchanged" "Runtime deterministic execution") `
        $true `
        "Runtime deterministic setting immutability"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $execution "autocast_cuda_float16" "Runtime deterministic execution") `
        ($ExpectedDevice -ceq "cuda") `
        "Runtime CUDA float16 autocast provenance"

    $expectedSnapshot = [ordered]@{
        torch_deterministic_algorithms = $true
        cudnn_benchmark = $false
        cudnn_deterministic = $true
        cuda_matmul_tf32 = $false
        cudnn_tf32 = $false
        cublas_workspace_config = ":4096:8"
        nnunet_compile = "false"
    }
    foreach ($stage in @(
        "after_initial_configuration",
        "after_predictor_construction",
        "after_inference"
    )) {
        $snapshot = Get-V5RequiredProperty $execution $stage "Runtime deterministic execution"
        if ($snapshot.PSObject.Properties.Count -ne $expectedSnapshot.Count) {
            throw "Runtime deterministic snapshot '$stage' has an unexpected field count."
        }
        foreach ($field in $expectedSnapshot.Keys) {
            $actual = Get-V5RequiredProperty $snapshot $field "Runtime deterministic snapshot '$stage'"
            $expected = $expectedSnapshot[$field]
            if ($expected -is [bool]) {
                Assert-V5Boolean $actual ([bool] $expected) "Runtime deterministic '$stage.$field'"
            }
            else {
                Assert-V5ExactValue $actual $expected "Runtime deterministic '$stage.$field'"
            }
        }
    }

    $lock = Get-V5RequiredProperty $execution "conformance_lock" "Runtime deterministic execution"
    Assert-V5HashEquals (Get-V5RequiredProperty $lock "sha256" "Runtime determinism lock") $script:V5DeterminismLockSha256 "Runtime determinism-conformance lock"
    Assert-V5Boolean (Get-V5RequiredProperty $lock "unchanged_during_run" "Runtime determinism lock") $true "Runtime determinism-lock immutability"
    if ([int64] (Get-V5RequiredProperty $lock "size_bytes" "Runtime determinism lock") -le 0) {
        throw "Runtime determinism lock size must be positive."
    }
    $expectedLockPath = Join-Path $Candidate.ProjectRoot "configs\inference_determinism_conformance_v1.json"
    if (-not (Get-V5NormalizedFullPath ([string] (Get-V5RequiredProperty $lock "path" "Runtime determinism lock"))).Equals(
        (Get-V5NormalizedFullPath $expectedLockPath),
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Runtime determinism lock path differs from the final-candidate repository file."
    }

    $installedSource = Get-V5RequiredProperty $execution "installed_nnunet_source" "Runtime deterministic execution"
    Assert-V5Boolean (Get-V5RequiredProperty $installedSource "unchanged_during_run" "Installed nnU-Net source") $true "Installed nnU-Net source immutability"
    $before = Get-V5RequiredProperty $installedSource "before" "Installed nnU-Net source"
    $after = Get-V5RequiredProperty $installedSource "after" "Installed nnU-Net source"
    foreach ($snapshot in @($before, $after)) {
        Assert-V5HashEquals (Get-V5RequiredProperty $snapshot "sha256" "Installed nnU-Net source snapshot") $script:V5InstalledNnunetPredictSourceSha256 "Installed nnU-Net prediction source"
        if ([int64] (Get-V5RequiredProperty $snapshot "size_bytes" "Installed nnU-Net source snapshot") -le 0) {
            throw "Installed nnU-Net prediction source size must be positive."
        }
        $sourcePath = [string] (Get-V5RequiredProperty $snapshot "path" "Installed nnU-Net source snapshot")
        if ([IO.Path]::GetFileName($sourcePath) -cne "predict_from_raw_data.py") {
            throw "Runtime bound an unexpected installed nnU-Net prediction source."
        }
    }
    foreach ($field in @("path", "sha256", "size_bytes")) {
        Assert-V5ExactValue `
            (Get-V5RequiredProperty $after $field "Installed nnU-Net source after") `
            (Get-V5RequiredProperty $before $field "Installed nnU-Net source before") `
            "Installed nnU-Net source before/after '$field'"
    }
}

function Assert-V5RuntimeArtifact {
    param(
        [Parameter(Mandatory)]
        [string] $RuntimeJson,
        [Parameter(Mandatory)]
        [int] $ExpectedCaseCount,
        [Parameter(Mandatory)]
        [string] $ExpectedInputDirectory,
        [Parameter(Mandatory)]
        [string] $ExpectedDevice,
        [Parameter(Mandatory)]
        [object] $Candidate,
        [Parameter(Mandatory)]
        [string] $ExpectedEncoderComponentSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedDecoderComponentSha256,
        [Parameter(Mandatory)]
        [string] $ExpectedClassificationComponentSha256
    )

    $runtime = Read-V5JsonObject $RuntimeJson "V5 runtime artifact"
    Assert-V5DeterministicRuntime `
        -Runtime $runtime `
        -Candidate $Candidate `
        -ExpectedDevice $ExpectedDevice
    $stockExport = Get-V5RequiredProperty $runtime "stock_export_conformance" "Runtime artifact"
    Assert-V5ExactValue `
        (Get-V5RequiredProperty $stockExport "export_logit_dtype" "Runtime stock-export conformance") `
        "torch.float16" `
        "Runtime stock-export logit dtype"
    if ([int] (Get-V5RequiredProperty $stockExport "case_count_verified" "Runtime stock-export conformance") -ne $ExpectedCaseCount) {
        throw "Runtime stock-export conformance must verify all $ExpectedCaseCount cases."
    }
    Assert-V5Boolean `
        (Get-V5RequiredProperty $stockExport "all_case_exports_verified" "Runtime stock-export conformance") `
        $true `
        "Runtime stock-export all-case verification"
    $stockExportLock = Get-V5RequiredProperty $stockExport "conformance_lock" "Runtime stock-export conformance"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $stockExportLock "sha256" "Runtime stock-export lock") `
        $script:V5StockExportLockSha256 `
        "Runtime stock-export conformance lock"
    Assert-V5Boolean `
        (Get-V5RequiredProperty $stockExportLock "unchanged_during_run" "Runtime stock-export lock") `
        $true `
        "Runtime stock-export lock immutability"
    if ([int64] (Get-V5RequiredProperty $stockExportLock "size_bytes" "Runtime stock-export lock") -le 0) {
        throw "Runtime stock-export lock size must be positive."
    }
    $expectedStockExportLockPath = Join-Path $Candidate.ProjectRoot "configs\inference_stock_export_conformance_v1.json"
    if (-not (Get-V5NormalizedFullPath ([string] (
        Get-V5RequiredProperty $stockExportLock "path" "Runtime stock-export lock"
    ))).Equals(
        (Get-V5NormalizedFullPath $expectedStockExportLockPath),
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Runtime stock-export lock path differs from the final-candidate repository file."
    }
    if ([int] (Get-V5RequiredProperty $runtime "case_count" "Runtime artifact") -ne $ExpectedCaseCount) {
        throw "Runtime artifact must record exactly $ExpectedCaseCount cases."
    }
    Assert-V5ExactValue (Get-V5RequiredProperty $runtime "checkpoint" "Runtime artifact") $script:V5CheckpointName "Runtime checkpoint"
    Assert-V5ExactValue (Get-V5RequiredProperty $runtime "classifier_pipeline" "Runtime artifact") $script:V5ClassifierPipeline "Runtime classifier pipeline"
    Assert-V5ExactValue (Get-V5RequiredProperty $runtime "v5_extraction_mode" "Runtime artifact") "neural_only" "Runtime extraction mode"
    Assert-V5ExactValue (Get-V5RequiredProperty $runtime "device" "Runtime artifact") $ExpectedDevice "Runtime device"
    Assert-V5Boolean (Get-V5RequiredProperty $runtime "gaussian_enabled" "Runtime artifact") $true "Runtime Gaussian state"
    Assert-V5Boolean (Get-V5RequiredProperty $runtime "tta_enabled" "Runtime artifact") $true "Runtime TTA state"
    Assert-V5Boolean (Get-V5RequiredProperty $runtime "overwrite" "Runtime artifact") $true "Runtime overwrite state"
    Assert-V5Boolean (Get-V5RequiredProperty $runtime "checkpoint_unchanged_during_run" "Runtime artifact") $true "Runtime checkpoint immutability"
    Assert-V5Boolean (Get-V5RequiredProperty $runtime "model_configuration_unchanged_during_run" "Runtime artifact") $true "Runtime model-configuration immutability"
    Assert-V5Boolean (Get-V5RequiredProperty $runtime "input_files_unchanged_during_run" "Runtime artifact") $true "Runtime input-file immutability"
    Assert-V5ExactValue (Get-V5RequiredProperty $runtime "feature_cache_policy" "Runtime artifact") "disabled_online_fresh_extraction" "Runtime feature-cache policy"
    Assert-V5ExactValue (Get-V5RequiredProperty $runtime "class_probabilities" "Runtime artifact") "v5_offset_adjusted_three_class" "Runtime class-probability contract"
    Assert-V5Boolean (Get-V5RequiredProperty $runtime "case_identifiers_or_paths_used_as_model_inputs" "Runtime artifact") $false "Runtime top-level identifier-feature flag"
    if ([double] (Get-V5RequiredProperty $runtime "tile_step_size" "Runtime artifact") -ne 0.5) {
        throw "Runtime artifact tile_step_size must equal 0.5."
    }
    if (-not (Get-V5NormalizedFullPath ([string] (Get-V5RequiredProperty $runtime "input_directory" "Runtime artifact"))).Equals(
        (Get-V5NormalizedFullPath $ExpectedInputDirectory),
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Runtime artifact records a different input directory."
    }
    if (-not (Get-V5NormalizedFullPath ([string] (Get-V5RequiredProperty $runtime "model_directory" "Runtime artifact"))).Equals(
        $Candidate.ModelDirectory,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Runtime artifact records a different model directory."
    }
    $folds = @(Get-V5RequiredProperty $runtime "folds" "Runtime artifact")
    if ($folds.Count -ne 1 -or [int] $folds[0] -ne 0) {
        throw "Runtime artifact must record exactly numeric fold 0."
    }
    $checkpointFiles = @(Get-V5RequiredProperty $runtime "checkpoint_files" "Runtime artifact")
    if ($checkpointFiles.Count -ne 1 -or [string] $checkpointFiles[0].fold -cne "0") {
        throw "Runtime checkpoint provenance must contain only fold 0."
    }
    Assert-V5HashEquals $checkpointFiles[0].sha256 $script:V5CheckpointSha256 "Runtime checkpoint provenance"

    $inputManifest = Get-V5RequiredProperty $runtime "input_file_manifest" "Runtime artifact"
    if ([int] (Get-V5RequiredProperty $inputManifest "file_count" "Runtime input manifest") -ne $ExpectedCaseCount) {
        throw "Runtime input manifest must contain exactly $ExpectedCaseCount single-channel NIfTI files."
    }
    $inputFiles = @(Get-V5RequiredProperty $inputManifest "files" "Runtime input manifest")
    if ($inputFiles.Count -ne $ExpectedCaseCount) {
        throw "Runtime input-file inventory count is inconsistent."
    }
    $inputNames = @()
    foreach ($inputFile in $inputFiles) {
        $name = [string] (Get-V5RequiredProperty $inputFile "name" "Runtime input-file entry")
        if (-not $name.EndsWith("_0000.nii.gz", [StringComparison]::Ordinal)) {
            throw "Runtime input manifest contains a file outside the single-channel *_0000.nii.gz contract."
        }
        $null = ConvertTo-V5Sha256 (Get-V5RequiredProperty $inputFile "sha256" "Runtime input-file entry") "Runtime input-file hash"
        if ([int64] (Get-V5RequiredProperty $inputFile "size_bytes" "Runtime input-file entry") -le 0) {
            throw "Runtime input manifest contains an empty file."
        }
        $inputNames += $name
    }
    if (@($inputNames | Sort-Object -Unique).Count -ne $ExpectedCaseCount) {
        throw "Runtime input manifest contains duplicate filenames."
    }
    $null = ConvertTo-V5Sha256 (Get-V5RequiredProperty $inputManifest "manifest_sha256" "Runtime input manifest") "Runtime input-manifest hash"

    $configurations = @(Get-V5RequiredProperty $runtime "model_configuration_files" "Runtime artifact")
    if ($configurations.Count -ne 2) {
        throw "Runtime must bind exactly plans.json and dataset.json."
    }
    $configurationByName = @{}
    foreach ($entry in $configurations) {
        $configurationByName[[string] $entry.name] = [string] $entry.sha256
    }
    Assert-V5HashEquals $configurationByName["plans.json"] $script:V5PlansSha256 "Runtime plans.json provenance"
    Assert-V5HashEquals $configurationByName["dataset.json"] $script:V5DatasetJsonSha256 "Runtime dataset.json provenance"

    $bundle = Get-V5RequiredProperty $runtime "neural_case_head_bundle" "Runtime artifact"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "bundle_sha256" "Runtime neural bundle") $Candidate.BundleSha256 "Runtime neural-bundle provenance"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "numeric_train_dataset_sha256" "Runtime neural bundle") $Candidate.NumericTrainDatasetSha256 "Runtime numeric-dataset provenance"
    Assert-V5Boolean (Get-V5RequiredProperty $bundle "eligible_for_official" "Runtime neural bundle") $true "Runtime bundle eligibility"
    Assert-V5Boolean (Get-V5RequiredProperty $bundle "bundle_loaded_strictly" "Runtime neural bundle") $true "Runtime strict bundle load"
    Assert-V5Boolean (Get-V5RequiredProperty $bundle "expected_bundle_sha256_verified" "Runtime neural bundle") $true "Runtime expected-bundle verification"
    Assert-V5Boolean (Get-V5RequiredProperty $bundle "head_in_eval_mode" "Runtime neural bundle") $true "Runtime neural-head evaluation mode"
    Assert-V5Boolean (Get-V5RequiredProperty $bundle "head_state_unchanged" "Runtime neural bundle") $true "Runtime neural-head immutability"
    Assert-V5Boolean (Get-V5RequiredProperty $bundle "any_head_parameter_requires_grad" "Runtime neural bundle") $false "Runtime neural-head gradient state"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "neural_lock_sha256" "Runtime neural bundle") $script:V5NeuralLockSha256 "Runtime neural-head lock"
    Assert-V5HashEquals (Get-V5RequiredProperty $bundle "decision_lock_sha256" "Runtime neural bundle") $script:V5DecisionLockSha256 "Runtime neural-decision lock"

    $network = Get-V5RequiredProperty $runtime "frozen_network" "Runtime artifact"
    if ([int] (Get-V5RequiredProperty $network "fold" "Runtime frozen network") -ne 0) {
        throw "Runtime frozen network must identify fold 0."
    }
    Assert-V5Boolean (Get-V5RequiredProperty $network "frozen_components_unchanged" "Runtime frozen network") $true "Runtime frozen-component state"
    Assert-V5Boolean (Get-V5RequiredProperty $network "network_in_eval_mode" "Runtime frozen network") $true "Runtime network evaluation mode"
    Assert-V5Boolean (Get-V5RequiredProperty $network "any_network_parameter_requires_grad" "Runtime frozen network") $false "Runtime network gradient state"
    $componentBefore = Get-V5RequiredProperty $network "component_hashes_before" "Runtime frozen network"
    $componentAfter = Get-V5RequiredProperty $network "component_hashes_after" "Runtime frozen network"
    foreach ($contract in @(
        [pscustomobject]@{ Name = "encoder"; Expected = $ExpectedEncoderComponentSha256 },
        [pscustomobject]@{ Name = "decoder"; Expected = $ExpectedDecoderComponentSha256 },
        [pscustomobject]@{ Name = "classification"; Expected = $ExpectedClassificationComponentSha256 }
    )) {
        Assert-V5HashEquals (Get-V5RequiredProperty $componentBefore $contract.Name "Runtime components before") $contract.Expected "Runtime $($contract.Name) before"
        Assert-V5HashEquals (Get-V5RequiredProperty $componentAfter $contract.Name "Runtime components after") $contract.Expected "Runtime $($contract.Name) after"
    }

    $execution = Get-V5RequiredProperty $runtime "inference_execution" "Runtime artifact"
    Assert-V5ExactValue (Get-V5RequiredProperty $execution "classifier_pipeline" "Runtime execution") $script:V5ClassifierPipeline "Runtime execution classifier"
    Assert-V5ExactValue (Get-V5RequiredProperty $execution "v5_extraction_mode" "Runtime execution") "neural_only" "Runtime execution extraction mode"
    Assert-V5Boolean (Get-V5RequiredProperty $execution "v5_feature_extraction_executed" "Runtime execution") $true "Runtime feature extraction"
    Assert-V5Boolean (Get-V5RequiredProperty $execution "case_identifiers_or_paths_used_as_model_inputs" "Runtime execution") $false "Runtime identifier-feature flag"
    Assert-V5ExactValue `
        (Get-V5RequiredProperty $execution "segmentation_export_logit_dtype" "Runtime execution") `
        "torch.float16" `
        "Runtime execution segmentation-export dtype"
    $exportDtypeSequence = @(Get-V5RequiredProperty $execution "segmentation_export_logit_dtype_sequence" "Runtime execution")
    if ($exportDtypeSequence.Count -ne $ExpectedCaseCount -or
        @($exportDtypeSequence | Where-Object { [string] $_ -cne "torch.float16" }).Count -ne 0) {
        throw "Runtime execution must record a torch.float16 segmentation export for every case."
    }
    foreach ($field in @(
        "v5_case_extractions_completed",
        "v5_neural_head_forward_calls",
        "v5_class_offset_applications"
    )) {
        if ([int] (Get-V5RequiredProperty $execution $field "Runtime execution") -ne $ExpectedCaseCount) {
            throw "Runtime execution '$field' must equal $ExpectedCaseCount."
        }
    }
    if ([int] (Get-V5RequiredProperty $execution "v5_feature_cache_reads" "Runtime execution") -ne 0) {
        throw "Official/test v5 inference must not read a feature cache."
    }
    foreach ($field in @("tile_batch_size_requested", "tta_batch_size_requested", "maximum_network_batch_size_observed", "network_batch_size_limit")) {
        if ([int] (Get-V5RequiredProperty $execution $field "Runtime execution") -ne 1) {
            throw "Runtime execution '$field' must equal 1."
        }
    }
    foreach ($field in @("tile_batch_oom_fallback_count", "tta_batch_oom_fallback_count")) {
        if ([int] (Get-V5RequiredProperty $execution $field "Runtime execution") -ne 0) {
            throw "Runtime execution '$field' must equal 0."
        }
    }
    $tileCount = [int] (Get-V5RequiredProperty $execution "logical_tiles_completed" "Runtime execution")
    $ttaViewCount = [int] (Get-V5RequiredProperty $execution "tta_views_completed" "Runtime execution")
    if ($tileCount -le 0 -or $ttaViewCount -ne (8 * $tileCount)) {
        throw "Runtime must record exactly eight mirror-TTA views for every logical tile."
    }
    $bagHashes = @(Get-V5RequiredProperty $runtime "v5_neural_bag_sha256_sequence" "Runtime artifact")
    if ($bagHashes.Count -ne $ExpectedCaseCount) {
        throw "Runtime must retain one neural-bag content hash per case."
    }
    foreach ($bagHash in $bagHashes) {
        $null = ConvertTo-V5Sha256 $bagHash "Runtime neural-bag hash"
    }
    $runtimeImplementation = Get-V5RequiredProperty $runtime "v5_implementation_files" "Runtime artifact"
    if ($runtimeImplementation.PSObject.Properties.Count -ne $script:V5RuntimeImplementationFiles.Count) {
        throw "Runtime implementation manifest has an unexpected file count."
    }
    $lockedImplementationEntries = @(Get-V5RequiredProperty $Candidate.Lock "implementation_files" "Final-candidate lock")
    $lockedImplementation = @{}
    foreach ($entry in $lockedImplementationEntries) {
        $lockedImplementation[[string] $entry.path] = [string] $entry.sha256
    }
    foreach ($relative in $script:V5RuntimeImplementationFiles) {
        $runtimeProperty = $runtimeImplementation.PSObject.Properties[$relative]
        if ($null -eq $runtimeProperty) {
            throw "Runtime implementation manifest is missing '$relative'."
        }
        Assert-V5HashEquals $runtimeProperty.Value $lockedImplementation[$relative] "Runtime/lock implementation '$relative'"
    }
    $caseIds = @((Get-V5RequiredProperty $runtime "case_ids" "Runtime artifact") | ForEach-Object { [string] $_ })
    if ($caseIds.Count -ne $ExpectedCaseCount -or @($caseIds | Sort-Object -Unique).Count -ne $ExpectedCaseCount) {
        throw "Runtime case identifiers must be unique and total $ExpectedCaseCount."
    }
    return $runtime
}

function Get-V5InferenceArtifactSet {
    param(
        [Parameter(Mandatory)]
        [string] $PredictionDirectory,
        [Parameter(Mandatory)]
        [string] $ClassificationCsv,
        [Parameter(Mandatory)]
        [string] $ProbabilityCsv,
        [Parameter(Mandatory)]
        [string] $RuntimeJson,
        [Parameter(Mandatory)]
        [object] $Runtime,
        [Parameter(Mandatory)]
        [int] $ExpectedCaseCount,
        [Parameter(Mandatory)]
        [string] $OutputRoot
    )

    Assert-V5Directory $PredictionDirectory "Prediction directory"
    Assert-V5LeafFile $ClassificationCsv "Classification CSV"
    Assert-V5LeafFile $ProbabilityCsv "Probability CSV"
    Assert-V5LeafFile $RuntimeJson "Runtime JSON"
    $files = @(Get-ChildItem -LiteralPath $PredictionDirectory -File)
    $masks = @($files | Where-Object { $_.Name.EndsWith(".nii.gz", [StringComparison]::Ordinal) })
    $other = @($files | Where-Object { -not $_.Name.EndsWith(".nii.gz", [StringComparison]::Ordinal) })
    if ($masks.Count -ne $ExpectedCaseCount) {
        throw "Expected exactly $ExpectedCaseCount direct-child NIfTI masks; found $($masks.Count)."
    }
    if ($other.Count -ne 1 -or $other[0].Name -cne "subtype_results.csv") {
        throw "Prediction directory must contain only masks plus subtype_results.csv."
    }
    if (-not (Get-V5NormalizedFullPath $other[0].FullName).Equals(
        (Get-V5NormalizedFullPath $ClassificationCsv),
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Classification CSV must be the exact subtype_results.csv in the prediction directory."
    }

    $classificationHeader = Get-Content -LiteralPath $ClassificationCsv -TotalCount 1
    if ([string] $classificationHeader -cne "Names,Subtype") {
        throw "Classification CSV header must be exactly Names,Subtype."
    }
    $probabilityHeader = Get-Content -LiteralPath $ProbabilityCsv -TotalCount 1
    if ([string] $probabilityHeader -cne "Names,Subtype,Probability_0,Probability_1,Probability_2") {
        throw "Probability CSV header is not the locked five-column schema."
    }
    $classificationRows = @(Import-Csv -LiteralPath $ClassificationCsv)
    $probabilityRows = @(Import-Csv -LiteralPath $ProbabilityCsv)
    if ($classificationRows.Count -ne $ExpectedCaseCount -or $probabilityRows.Count -ne $ExpectedCaseCount) {
        throw "Classification and probability CSVs must each contain $ExpectedCaseCount rows."
    }
    $maskNames = @($masks.Name | Sort-Object)
    $classificationNames = @($classificationRows.Names | ForEach-Object { [string] $_ } | Sort-Object)
    $probabilityNames = @($probabilityRows.Names | ForEach-Object { [string] $_ } | Sort-Object)
    if (@(Compare-Object $maskNames $classificationNames).Count -ne 0 -or
        @(Compare-Object $maskNames $probabilityNames).Count -ne 0) {
        throw "Mask, classification, and probability case-name inventories differ."
    }
    if (@($classificationNames | Sort-Object -Unique).Count -ne $ExpectedCaseCount -or
        @($probabilityNames | Sort-Object -Unique).Count -ne $ExpectedCaseCount) {
        throw "Classification or probability CSV contains duplicate Names."
    }
    $labelByName = @{}
    foreach ($row in $classificationRows) {
        $label = [string] $row.Subtype
        if ($label -notin @("0", "1", "2")) {
            throw "Classification CSV contains a subtype outside {0,1,2}."
        }
        $labelByName[[string] $row.Names] = $label
    }
    foreach ($row in $probabilityRows) {
        $name = [string] $row.Names
        if ([string] $row.Subtype -cne [string] $labelByName[$name]) {
            throw "Probability/classification subtype disagreement for '$name'."
        }
        $values = @([double] $row.Probability_0, [double] $row.Probability_1, [double] $row.Probability_2)
        if ($values | Where-Object { [double]::IsNaN($_) -or [double]::IsInfinity($_) -or $_ -lt 0.0 -or $_ -gt 1.0 }) {
            throw "Probability CSV contains an invalid probability for '$name'."
        }
        if ([Math]::Abs(($values | Measure-Object -Sum).Sum - 1.0) -gt 0.0001) {
            throw "Probability CSV row does not sum to one for '$name'."
        }
    }
    $runtimeIds = @((Get-V5RequiredProperty $Runtime "case_ids" "Runtime artifact") | ForEach-Object { [string] $_ } | Sort-Object)
    $maskIds = @($maskNames | ForEach-Object { $_.Substring(0, $_.Length - ".nii.gz".Length) } | Sort-Object)
    if (@(Compare-Object $runtimeIds $maskIds).Count -ne 0) {
        throw "Runtime and mask case identifiers differ."
    }

    $artifacts = @()
    foreach ($mask in ($masks | Sort-Object Name)) {
        $artifacts += [pscustomobject]@{
            role = "segmentation_mask"
            relative_path = Get-V5RelativePath -Parent $OutputRoot -Child $mask.FullName
            sha256 = Get-V5FileSha256 $mask.FullName
            size_bytes = [int64] $mask.Length
        }
    }
    foreach ($entry in @(
        [pscustomobject]@{ Role = "classification_csv"; Path = $ClassificationCsv },
        [pscustomobject]@{ Role = "probability_csv"; Path = $ProbabilityCsv },
        [pscustomobject]@{ Role = "runtime_json"; Path = $RuntimeJson }
    )) {
        $item = Get-Item -LiteralPath $entry.Path
        $artifacts += [pscustomobject]@{
            role = $entry.Role
            relative_path = Get-V5RelativePath -Parent $OutputRoot -Child $item.FullName
            sha256 = Get-V5FileSha256 $item.FullName
            size_bytes = [int64] $item.Length
        }
    }
    $canonical = ($artifacts | ForEach-Object {
        "$($_.role)|$($_.relative_path)|$($_.sha256)|$($_.size_bytes)`n"
    }) -join ""
    return [pscustomobject]@{
        Artifacts = $artifacts
        ArtifactSetSha256 = Get-V5StringSha256 $canonical
        MaskCount = $masks.Count
        ClassificationRowCount = $classificationRows.Count
        ProbabilityRowCount = $probabilityRows.Count
    }
}

function Assert-V5RecordedInferenceArtifactSet {
    param(
        [Parameter(Mandatory)]
        [object] $Manifest,
        [Parameter(Mandatory)]
        [string] $OutputRoot,
        [Parameter(Mandatory)]
        [string] $ExpectedArtifactSetSha256,
        [Parameter(Mandatory)]
        [int] $ExpectedCaseCount
    )

    $resolvedRoot = Get-V5NormalizedFullPath $OutputRoot
    Assert-V5Directory $resolvedRoot "Recorded inference output root"
    $expectedSetHash = ConvertTo-V5Sha256 `
        $ExpectedArtifactSetSha256 `
        "Recorded inference artifact-set hash"
    Assert-V5HashEquals `
        (Get-V5RequiredProperty $Manifest "prediction_artifact_set_sha256" "Prediction manifest") `
        $expectedSetHash `
        "Prediction manifest artifact-set binding"

    $entries = @(Get-V5RequiredProperty $Manifest "prediction_artifacts" "Prediction manifest")
    if ($entries.Count -ne ($ExpectedCaseCount + 3)) {
        throw (
            "Prediction manifest must record exactly $ExpectedCaseCount masks plus " +
            "classification, probability, and runtime artifacts."
        )
    }
    $seenPaths = @{}
    $roleCounts = @{
        segmentation_mask = 0
        classification_csv = 0
        probability_csv = 0
        runtime_json = 0
    }
    $canonical = ""
    foreach ($entry in $entries) {
        $role = [string] (Get-V5RequiredProperty $entry "role" "Prediction-artifact entry")
        if (-not $roleCounts.ContainsKey($role)) {
            throw "Prediction manifest contains an unsupported artifact role '$role'."
        }
        $relative = [string] (Get-V5RequiredProperty $entry "relative_path" "Prediction-artifact entry")
        $segments = @($relative.Split([char] "/"))
        $unsafeSegments = @(
            $segments | Where-Object {
                [string]::IsNullOrWhiteSpace($_) -or $_ -in @(".", "..")
            }
        )
        if (
            [string]::IsNullOrWhiteSpace($relative) -or
            [IO.Path]::IsPathRooted($relative) -or
            $relative.Contains("\") -or
            $segments.Count -ne 2 -or
            $unsafeSegments.Count -ne 0
        ) {
            throw (
                "Prediction-artifact paths must be direct children of predictions/ " +
                "or evidence/: '$relative'."
            )
        }
        switch ($role) {
            "segmentation_mask" {
                if ($segments[0] -cne "predictions" -or
                    -not $segments[1].EndsWith(".nii.gz", [StringComparison]::Ordinal)) {
                    throw "Recorded segmentation mask has an invalid path '$relative'."
                }
            }
            "classification_csv" {
                if ($relative -cne "predictions/subtype_results.csv") {
                    throw "Recorded classification CSV has an invalid path '$relative'."
                }
            }
            "probability_csv" {
                if ($relative -cne "evidence/subtype_probabilities.csv") {
                    throw "Recorded probability CSV has an invalid path '$relative'."
                }
            }
            "runtime_json" {
                if ($relative -cne "evidence/runtime.json") {
                    throw "Recorded runtime JSON has an invalid path '$relative'."
                }
            }
        }
        if ($seenPaths.ContainsKey($relative)) {
            throw "Prediction manifest contains duplicate artifact path '$relative'."
        }
        $seenPaths[$relative] = $true
        $roleCounts[$role] = [int] $roleCounts[$role] + 1

        $path = Join-Path $resolvedRoot ($relative.Replace("/", [IO.Path]::DirectorySeparatorChar))
        if (-not (Test-V5PathAtOrBelow -Candidate $path -Parent $resolvedRoot)) {
            throw "Recorded prediction artifact escapes its output root: '$relative'."
        }
        Assert-V5LeafFile $path "Recorded prediction artifact"
        $recordedSize = [int64] (Get-V5RequiredProperty $entry "size_bytes" "Prediction-artifact entry")
        $actualSize = [int64] (Get-Item -LiteralPath $path).Length
        if ($recordedSize -le 0 -or $actualSize -ne $recordedSize) {
            throw "Recorded prediction artifact size changed for '$relative'."
        }
        $recordedHash = ConvertTo-V5Sha256 `
            (Get-V5RequiredProperty $entry "sha256" "Prediction-artifact entry") `
            "Recorded prediction artifact '$relative' hash"
        Assert-V5HashEquals `
            (Get-V5FileSha256 $path) `
            $recordedHash `
            "Recorded prediction artifact '$relative'"
        $canonical += "$role|$relative|$recordedHash|$recordedSize`n"
    }
    if (
        [int] $roleCounts["segmentation_mask"] -ne $ExpectedCaseCount -or
        [int] $roleCounts["classification_csv"] -ne 1 -or
        [int] $roleCounts["probability_csv"] -ne 1 -or
        [int] $roleCounts["runtime_json"] -ne 1
    ) {
        throw "Prediction manifest artifact-role counts differ from the frozen contract."
    }
    Assert-V5HashEquals `
        (Get-V5StringSha256 $canonical) `
        $expectedSetHash `
        "Recomputed prediction artifact set"
    return [pscustomobject]@{
        ArtifactSetSha256 = $expectedSetHash
        ArtifactCount = $entries.Count
        MaskCount = [int] $roleCounts["segmentation_mask"]
    }
}

function Enter-V5NamedMutex {
    param(
        [Parameter(Mandatory)]
        [string] $Name
    )

    $mutex = [Threading.Mutex]::new($false, $Name)
    try {
        try {
            $owned = $mutex.WaitOne(0)
        }
        catch [Threading.AbandonedMutexException] {
            $owned = $true
        }
        if (-not $owned) {
            throw "Another process owns the v5 stage mutex '$Name'."
        }
        return $mutex
    }
    catch {
        $mutex.Dispose()
        throw
    }
}

function Exit-V5NamedMutex {
    param(
        [object] $Mutex
    )

    if ($null -ne $Mutex) {
        try {
            $Mutex.ReleaseMutex()
        }
        finally {
            $Mutex.Dispose()
        }
    }
}
