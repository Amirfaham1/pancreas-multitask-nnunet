<#
.SYNOPSIS
Validate and package the final 72-case test submission.

.DESCRIPTION
The source directory is validated before packaging, then its 72 masks and
subtype_results.csv are written explicitly to a flat ZIP root. A staged ZIP is
validated before it can replace an existing delivery, and the committed ZIP is
validated independently before a SHA-256 manifest is written atomically.
Prepared test images are read only by validate_submission.py and are never
copied into the delivery directory or archive.

An existing Amirfaham_Fallahpour_results.zip is refused unless -Force is given.
Even with -Force, only that exact file can be atomically replaced, after its
resolved path is verified as a direct child of the selected delivery root.

.EXAMPLE
.\scripts\Package-Submission.ps1

.EXAMPLE
.\scripts\Package-Submission.ps1 `
  -PredictionDirectory D:\MLQuizWork\submission\Amirfaham_Fallahpour_results `
  -TestImages D:\MLQuizWork\nnUNet_raw\Dataset501_PancreasMultitask\imagesTs `
  -DeliveryRoot .\delivery `
  -Force
#>

[CmdletBinding()]
param(
    [string] $PredictionDirectory =
        "D:\MLQuizWork\submission\Amirfaham_Fallahpour_results",
    [string] $TestImages =
        "D:\MLQuizWork\nnUNet_raw\Dataset501_PancreasMultitask\imagesTs",
    [string] $DeliveryRoot = (Join-Path $PSScriptRoot "..\delivery"),
    [string] $PythonExecutable = "D:\MLQuizWork\.venv\Scripts\python.exe",
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedCount = 72
$csvName = "subtype_results.csv"
$archiveName = "Amirfaham_Fallahpour_results.zip"
$manifestName = "package_manifest.json"
$validatorScript = Join-Path $PSScriptRoot "validate_submission.py"

function Get-NormalizedFullPath {
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

    return [IO.Path]::GetFullPath($Path).TrimEnd([char[]] "\/")
}

function Test-PathAtOrBelow {
    param(
        [Parameter(Mandatory)]
        [string] $Candidate,
        [Parameter(Mandatory)]
        [string] $Parent
    )

    $normalizedCandidate = Get-NormalizedFullPath -Path $Candidate
    $normalizedParent = Get-NormalizedFullPath -Path $Parent
    if ($normalizedCandidate.Equals(
        $normalizedParent,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        return $true
    }
    $prefix = $normalizedParent + [IO.Path]::DirectorySeparatorChar
    return $normalizedCandidate.StartsWith(
        $prefix,
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Assert-DirectDeliveryChild {
    param(
        [Parameter(Mandatory)]
        [string] $Target,
        [Parameter(Mandatory)]
        [string] $ResolvedDeliveryRoot
    )

    $normalizedTarget = Get-NormalizedFullPath -Path $Target
    $normalizedDelivery = Get-NormalizedFullPath -Path $ResolvedDeliveryRoot
    $targetParent = Get-NormalizedFullPath -Path ([IO.Path]::GetDirectoryName(
        $normalizedTarget
    ))
    if (-not $targetParent.Equals(
        $normalizedDelivery,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw (
            "Refusing a file operation outside the delivery root: " +
            "target='$normalizedTarget', delivery='$normalizedDelivery'."
        )
    }
}

function Get-ReparsePointTag {
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

    $fsutil = Join-Path ([Environment]::SystemDirectory) "fsutil.exe"
    if (-not (Test-Path -LiteralPath $fsutil -PathType Leaf)) {
        throw "Cannot inspect reparse-point safety because fsutil.exe is unavailable."
    }
    $messages = @()
    $exitCode = $null
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $messages = @(& $fsutil reparsepoint query $Path 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Could not inspect reparse point '$Path' (fsutil exit $exitCode)."
    }
    $rendered = ($messages | ForEach-Object { [string] $_ }) -join "`n"
    $match = [regex]::Match($rendered, "0x(?<tag>[0-9a-fA-F]{8})")
    if (-not $match.Success) {
        throw "Could not parse the reparse tag for '$Path'."
    }
    return [Convert]::ToUInt32($match.Groups["tag"].Value, 16)
}

function Assert-NoRedirectingReparsePointInPath {
    param(
        [Parameter(Mandatory)]
        [string] $Path,
        [Parameter(Mandatory)]
        [string] $Description
    )

    $current = [IO.Path]::GetFullPath($Path)
    while ($null -ne $current) {
        $item = Get-Item -LiteralPath $current -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            $tag = Get-ReparsePointTag -Path $current
            # The Windows name-surrogate bit marks reparse points that redirect
            # path resolution (including symbolic links, junctions, and mount
            # points). OneDrive cloud placeholders are reparse points but do not
            # set this bit, so the normal OneDrive delivery path remains valid.
            if (($tag -band 0x20000000) -ne 0) {
                $renderedTag = "0x{0:x8}" -f $tag
                throw (
                    "$Description cannot use a symbolic link, junction, or mount " +
                    "point: '$current' (reparse tag $renderedTag)."
                )
            }
        }

        $parent = [IO.Directory]::GetParent($current)
        if ($null -eq $parent) {
            break
        }
        $parentPath = $parent.FullName
        if ($parentPath.Equals($current, [StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $current = $parentPath
    }
}

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

function Commit-AtomicFile {
    param(
        [Parameter(Mandatory)]
        [string] $Source,
        [Parameter(Mandatory)]
        [string] $Destination,
        [Parameter(Mandatory)]
        [string] $ResolvedDeliveryRoot,
        [switch] $AllowReplace
    )

    $resolvedSource = [IO.Path]::GetFullPath($Source)
    $resolvedDestination = [IO.Path]::GetFullPath($Destination)
    Assert-DirectDeliveryChild `
        -Target $resolvedSource `
        -ResolvedDeliveryRoot $ResolvedDeliveryRoot
    Assert-DirectDeliveryChild `
        -Target $resolvedDestination `
        -ResolvedDeliveryRoot $ResolvedDeliveryRoot
    Assert-LeafFile -Path $resolvedSource -Description "Atomic source file"

    if (-not $AllowReplace) {
        # File.Move is the commit-time create-if-absent primitive. In contrast to
        # a separate Test-Path check followed by File.Replace, it cannot silently
        # replace a destination created by another process after preflight.
        try {
            [IO.File]::Move($resolvedSource, $resolvedDestination)
        }
        catch [IO.IOException] {
            if (Test-Path -LiteralPath $resolvedDestination) {
                throw (
                    "Atomic destination appeared during packaging; refusing to " +
                    "replace it without -Force: '$resolvedDestination'."
                )
            }
            throw
        }
        return
    }

    if (-not (Test-Path -LiteralPath $resolvedDestination)) {
        [IO.File]::Move($resolvedSource, $resolvedDestination)
        return
    }
    if (-not (Test-Path -LiteralPath $resolvedDestination -PathType Leaf)) {
        throw "Atomic destination exists but is not a file: '$resolvedDestination'."
    }

    $resolvedExisting = (Resolve-Path -LiteralPath $resolvedDestination).ProviderPath
    Assert-DirectDeliveryChild `
        -Target $resolvedExisting `
        -ResolvedDeliveryRoot $ResolvedDeliveryRoot
    Assert-NoRedirectingReparsePointInPath `
        -Path $resolvedExisting `
        -Description "Atomic destination path"
    $backup = Join-Path $ResolvedDeliveryRoot (
        ".{0}.{1}.replace-backup" -f [IO.Path]::GetFileName($resolvedDestination),
        [guid]::NewGuid().ToString("N")
    )
    Assert-DirectDeliveryChild `
        -Target $backup `
        -ResolvedDeliveryRoot $ResolvedDeliveryRoot
    if (Test-Path -LiteralPath $backup) {
        throw "Atomic replacement backup unexpectedly exists: '$backup'."
    }

    $replacementSucceeded = $false
    $replaceAttempt = 0
    $maximumReplaceAttempts = 12
    try {
        while (-not $replacementSucceeded) {
            $replaceAttempt++
            try {
                # Windows PowerShell 5.1 requires a non-null backup path for this API.
                [IO.File]::Replace($resolvedSource, $resolvedExisting, $backup)
                $replacementSucceeded = $true
            }
            catch {
                # Virus scanners and cloud-sync clients can retain a ZIP handle for
                # a short interval after the validator process exits. PowerShell
                # wraps the IOException raised by File.Replace, so walk to the
                # underlying exception before checking its Win32 error code.
                $ioException = $_.Exception
                while ($ioException -isnot [IO.IOException] -and
                    $null -ne $ioException.InnerException) {
                    $ioException = $ioException.InnerException
                }
                if ($ioException -isnot [IO.IOException]) {
                    throw
                }
                $nativeErrorCode = [int] ($ioException.HResult -band 0xFFFF)
                $isTransientLock = $nativeErrorCode -in @(32, 33)
                if (-not $isTransientLock -or
                    $replaceAttempt -ge $maximumReplaceAttempts) {
                    throw
                }

                # Retry only while File.Replace demonstrably made no state change.
                # If a backup appeared or either endpoint moved, stop and preserve
                # every surviving file for recovery instead of guessing.
                if ((Test-Path -LiteralPath $backup) -or
                    -not (Test-Path -LiteralPath $resolvedSource -PathType Leaf) -or
                    -not (Test-Path -LiteralPath $resolvedExisting -PathType Leaf)) {
                    throw (
                        "Atomic replacement encountered a transient lock, but file " +
                        "state changed before retry; recovery files were preserved."
                    )
                }

                $delayMilliseconds = [Math]::Min(
                    50 * [Math]::Pow(2, $replaceAttempt - 1),
                    500
                )
                Write-Verbose (
                    "Atomic replacement temporarily blocked by a file lock " +
                    "(attempt $replaceAttempt of $maximumReplaceAttempts); " +
                    "retrying in $([int] $delayMilliseconds) ms."
                )
                Start-Sleep -Milliseconds ([int] $delayMilliseconds)
            }
        }
    }
    finally {
        # On failure, retain any backup for manual recovery rather than deleting it.
        if ($replacementSucceeded -and (Test-Path -LiteralPath $backup)) {
            Assert-DirectDeliveryChild `
                -Target $backup `
                -ResolvedDeliveryRoot $ResolvedDeliveryRoot
            Remove-Item -LiteralPath $backup -Force
        }
    }
}

function Invoke-SubmissionValidator {
    param(
        [Parameter(Mandatory)]
        [string] $SubmissionPath,
        [Parameter(Mandatory)]
        [string] $OutputJson,
        [Parameter(Mandatory)]
        [string] $OutputCsv,
        [Parameter(Mandatory)]
        [string] $Stage
    )

    $arguments = @(
        $validatorScript,
        $SubmissionPath,
        "--test-images", $resolvedTestImages,
        "--expected-count", [string] $expectedCount,
        "--output-json", $OutputJson,
        "--output-csv", $OutputCsv
    )
    $validatorMessages = @()
    $validatorExitCode = $null
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 converts redirected native stderr to non-terminating
        # ErrorRecords. Continue is required here so validator diagnostics can be
        # captured and interpreted through the native exit code below.
        $ErrorActionPreference = "Continue"
        $validatorMessages = @(& $resolvedPython @arguments 2>&1)
        $validatorExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    foreach ($message in $validatorMessages) {
        Write-Host ([string] $message)
    }
    if ($validatorExitCode -ne 0) {
        throw "$Stage failed validation with exit code $validatorExitCode."
    }
    Assert-LeafFile -Path $OutputJson -Description "$Stage JSON validation artifact"
    Assert-LeafFile -Path $OutputCsv -Description "$Stage CSV validation artifact"

    try {
        $validation = Get-Content -LiteralPath $OutputJson -Raw | ConvertFrom-Json
    }
    catch {
        throw "Could not parse $Stage validation JSON '$OutputJson': $($_.Exception.Message)"
    }
    if ($validation.valid -ne $true -or
        [int] $validation.expected_case_count -ne $expectedCount -or
        [int] $validation.validated_mask_count -ne $expectedCount -or
        [int] $validation.validated_csv_row_count -ne $expectedCount) {
        throw "$Stage validator returned inconsistent success counts in '$OutputJson'."
    }
    return $validation
}

function New-FlatZipArchive {
    param(
        [Parameter(Mandatory)]
        [System.IO.FileInfo[]] $Files,
        [Parameter(Mandatory)]
        [string] $Destination,
        [Parameter(Mandatory)]
        [string] $ResolvedDeliveryRoot
    )

    Assert-DirectDeliveryChild `
        -Target $Destination `
        -ResolvedDeliveryRoot $ResolvedDeliveryRoot
    if (Test-Path -LiteralPath $Destination) {
        throw "Refusing to overwrite ZIP staging path '$Destination'."
    }

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $outputStream = $null
    $archive = $null
    try {
        $outputStream = [IO.File]::Open(
            $Destination,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None
        )
        $archive = [IO.Compression.ZipArchive]::new(
            $outputStream,
            [IO.Compression.ZipArchiveMode]::Create,
            $false
        )
        foreach ($file in ($Files | Sort-Object Name)) {
            # Passing only FileInfo.Name is what guarantees a flat archive root.
            $entry = $archive.CreateEntry(
                $file.Name,
                [IO.Compression.CompressionLevel]::Optimal
            )
            $entryStream = $null
            $inputStream = $null
            try {
                $entryStream = $entry.Open()
                $inputStream = [IO.File]::Open(
                    $file.FullName,
                    [IO.FileMode]::Open,
                    [IO.FileAccess]::Read,
                    [IO.FileShare]::Read
                )
                $inputStream.CopyTo($entryStream)
            }
            finally {
                if ($null -ne $inputStream) {
                    $inputStream.Dispose()
                }
                if ($null -ne $entryStream) {
                    $entryStream.Dispose()
                }
            }
        }
    }
    finally {
        if ($null -ne $archive) {
            $archive.Dispose()
        }
        if ($null -ne $outputStream) {
            $outputStream.Dispose()
        }
    }
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory)]
        [object] $Value,
        [Parameter(Mandatory)]
        [string] $Destination,
        [Parameter(Mandatory)]
        [string] $ResolvedDeliveryRoot,
        [switch] $AllowReplace
    )

    $resolvedDestination = [IO.Path]::GetFullPath($Destination)
    Assert-DirectDeliveryChild `
        -Target $resolvedDestination `
        -ResolvedDeliveryRoot $ResolvedDeliveryRoot
    $temporary = Join-Path $ResolvedDeliveryRoot (
        ".{0}.{1}.tmp" -f [IO.Path]::GetFileName($resolvedDestination),
        [guid]::NewGuid().ToString("N")
    )
    Assert-DirectDeliveryChild `
        -Target $temporary `
        -ResolvedDeliveryRoot $ResolvedDeliveryRoot

    $stream = $null
    try {
        $json = ($Value | ConvertTo-Json -Depth 8) + [Environment]::NewLine
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json)
        $stream = [IO.File]::Open(
            $temporary,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None
        )
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
        $stream.Dispose()
        $stream = $null

        Commit-AtomicFile `
            -Source $temporary `
            -Destination $resolvedDestination `
            -ResolvedDeliveryRoot $ResolvedDeliveryRoot `
            -AllowReplace:$AllowReplace
    }
    finally {
        if ($null -ne $stream) {
            $stream.Dispose()
        }
        if (Test-Path -LiteralPath $temporary) {
            Assert-DirectDeliveryChild `
                -Target $temporary `
                -ResolvedDeliveryRoot $ResolvedDeliveryRoot
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

Assert-Directory -Path $PredictionDirectory -Description "Completed prediction directory"
Assert-Directory -Path $TestImages -Description "Prepared test-image directory"
Assert-LeafFile -Path $PythonExecutable -Description "Python executable"
Assert-LeafFile -Path $validatorScript -Description "Submission validator"
Assert-NoRedirectingReparsePointInPath `
    -Path $PredictionDirectory `
    -Description "Prediction directory path"
Assert-NoRedirectingReparsePointInPath `
    -Path $TestImages `
    -Description "Prepared test-image directory path"

$resolvedPredictionDirectory = (Resolve-Path -LiteralPath $PredictionDirectory).ProviderPath
$resolvedTestImages = (Resolve-Path -LiteralPath $TestImages).ProviderPath
$resolvedPython = (Resolve-Path -LiteralPath $PythonExecutable).ProviderPath
$requestedDeliveryRoot = [IO.Path]::GetFullPath($DeliveryRoot)
Assert-NoRedirectingReparsePointInPath `
    -Path $requestedDeliveryRoot `
    -Description "Delivery path"
$volumeRoot = [IO.Path]::GetPathRoot($requestedDeliveryRoot)
if ((Get-NormalizedFullPath -Path $requestedDeliveryRoot).Equals(
    (Get-NormalizedFullPath -Path $volumeRoot),
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw "The delivery root cannot be a filesystem volume root: '$requestedDeliveryRoot'."
}
if (Test-Path -LiteralPath $requestedDeliveryRoot -PathType Leaf) {
    throw "Delivery root is a file, not a directory: '$requestedDeliveryRoot'."
}
$pathsOverlapPrediction =
    (Test-PathAtOrBelow -Candidate $requestedDeliveryRoot -Parent $resolvedPredictionDirectory) -or
    (Test-PathAtOrBelow -Candidate $resolvedPredictionDirectory -Parent $requestedDeliveryRoot)
$pathsOverlapTestImages =
    (Test-PathAtOrBelow -Candidate $requestedDeliveryRoot -Parent $resolvedTestImages) -or
    (Test-PathAtOrBelow -Candidate $resolvedTestImages -Parent $requestedDeliveryRoot)
if ($pathsOverlapPrediction -or $pathsOverlapTestImages) {
    throw (
        "Delivery, prediction, and prepared test-image directories must be disjoint. " +
        "No validation or package artifact was written."
    )
}
New-Item -ItemType Directory -Path $requestedDeliveryRoot -Force | Out-Null
$resolvedDeliveryRoot = (Resolve-Path -LiteralPath $requestedDeliveryRoot).ProviderPath
Assert-NoRedirectingReparsePointInPath `
    -Path $resolvedDeliveryRoot `
    -Description "Resolved delivery path"

# Resolve and repeat the isolation check in case an existing path is a junction.
$resolvedPathsOverlapPrediction =
    (Test-PathAtOrBelow -Candidate $resolvedDeliveryRoot -Parent $resolvedPredictionDirectory) -or
    (Test-PathAtOrBelow -Candidate $resolvedPredictionDirectory -Parent $resolvedDeliveryRoot)
$resolvedPathsOverlapTestImages =
    (Test-PathAtOrBelow -Candidate $resolvedDeliveryRoot -Parent $resolvedTestImages) -or
    (Test-PathAtOrBelow -Candidate $resolvedTestImages -Parent $resolvedDeliveryRoot)
if ($resolvedPathsOverlapPrediction -or $resolvedPathsOverlapTestImages) {
    throw "Resolved delivery root overlaps a protected input directory."
}

$archivePath = Join-Path $resolvedDeliveryRoot $archiveName
$manifestPath = Join-Path $resolvedDeliveryRoot $manifestName
$directoryValidationJson = Join-Path $resolvedDeliveryRoot "submission_directory_validation.json"
$directoryValidationCsv = Join-Path $resolvedDeliveryRoot "submission_directory_case_audit.csv"
$archiveValidationJson = Join-Path $resolvedDeliveryRoot "submission_archive_validation.json"
$archiveValidationCsv = Join-Path $resolvedDeliveryRoot "submission_archive_case_audit.csv"
foreach ($deliveryFile in @(
    $archivePath,
    $manifestPath,
    $directoryValidationJson,
    $directoryValidationCsv,
    $archiveValidationJson,
    $archiveValidationCsv
)) {
    Assert-DirectDeliveryChild `
        -Target $deliveryFile `
        -ResolvedDeliveryRoot $resolvedDeliveryRoot
    Assert-NoRedirectingReparsePointInPath `
        -Path $deliveryFile `
        -Description "Delivery artifact path"
}

if (Test-Path -LiteralPath $archivePath) {
    if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        throw "Archive target exists but is not a file: '$archivePath'."
    }
    if (-not $Force) {
        throw "Archive already exists; pass -Force to replace only '$archivePath'."
    }
}

$sourceEntries = @(Get-ChildItem -LiteralPath $resolvedPredictionDirectory -Force)
$sourceDirectories = @($sourceEntries | Where-Object { $_.PSIsContainer })
$sourceReparsePoints = @(
    $sourceEntries | Where-Object {
        ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    }
)
$sourceFiles = @($sourceEntries | Where-Object { -not $_.PSIsContainer })
$maskFiles = @($sourceFiles | Where-Object { $_.Name.EndsWith(".nii.gz") })
$csvFiles = @($sourceFiles | Where-Object { $_.Name -ceq $csvName })
if ($sourceDirectories.Count -ne 0 -or
    $sourceReparsePoints.Count -ne 0 -or
    $sourceFiles.Count -ne ($expectedCount + 1) -or
    $maskFiles.Count -ne $expectedCount -or
    $csvFiles.Count -ne 1) {
    throw (
        "Prediction directory must be flat and contain exactly $expectedCount masks plus " +
        "$csvName, with no directories, links, or extra files: '$resolvedPredictionDirectory'."
    )
}

Write-Host "Preflighting the completed prediction directory..."
$directoryValidation = Invoke-SubmissionValidator `
    -SubmissionPath $resolvedPredictionDirectory `
    -OutputJson $directoryValidationJson `
    -OutputCsv $directoryValidationCsv `
    -Stage "Prediction directory"

# Only the exact, already validated masks and classification CSV enter the ZIP.
$packageFiles = @($maskFiles + $csvFiles)
$temporaryArchive = Join-Path $resolvedDeliveryRoot (
    ".{0}.{1}.tmp.zip" -f [IO.Path]::GetFileNameWithoutExtension($archiveName),
    [guid]::NewGuid().ToString("N")
)
$stagedValidationJson = Join-Path $resolvedDeliveryRoot (
    ".submission_archive_validation.{0}.tmp.json" -f [guid]::NewGuid().ToString("N")
)
$stagedValidationCsv = Join-Path $resolvedDeliveryRoot (
    ".submission_archive_case_audit.{0}.tmp.csv" -f [guid]::NewGuid().ToString("N")
)
foreach ($stagedFile in @(
    $temporaryArchive,
    $stagedValidationJson,
    $stagedValidationCsv
)) {
    Assert-DirectDeliveryChild `
        -Target $stagedFile `
        -ResolvedDeliveryRoot $resolvedDeliveryRoot
}
try {
    Write-Host "Creating a flat staged archive..."
    New-FlatZipArchive `
        -Files $packageFiles `
        -Destination $temporaryArchive `
        -ResolvedDeliveryRoot $resolvedDeliveryRoot
    Assert-LeafFile -Path $temporaryArchive -Description "Staged submission archive"
    if ((Get-Item -LiteralPath $temporaryArchive).Length -le 0) {
        throw "Staged submission archive is empty: '$temporaryArchive'."
    }

    Write-Host "Validating the staged archive before committing it..."
    $null = Invoke-SubmissionValidator `
        -SubmissionPath $temporaryArchive `
        -OutputJson $stagedValidationJson `
        -OutputCsv $stagedValidationCsv `
        -Stage "Staged submission archive"

    Commit-AtomicFile `
        -Source $temporaryArchive `
        -Destination $archivePath `
        -ResolvedDeliveryRoot $resolvedDeliveryRoot `
        -AllowReplace:$Force
}
finally {
    if (Test-Path -LiteralPath $temporaryArchive) {
        Assert-DirectDeliveryChild `
            -Target $temporaryArchive `
            -ResolvedDeliveryRoot $resolvedDeliveryRoot
        Remove-Item -LiteralPath $temporaryArchive -Force
    }
    if (Test-Path -LiteralPath $stagedValidationJson) {
        Assert-DirectDeliveryChild `
            -Target $stagedValidationJson `
            -ResolvedDeliveryRoot $resolvedDeliveryRoot
        Remove-Item -LiteralPath $stagedValidationJson -Force
    }
    if (Test-Path -LiteralPath $stagedValidationCsv) {
        Assert-DirectDeliveryChild `
            -Target $stagedValidationCsv `
            -ResolvedDeliveryRoot $resolvedDeliveryRoot
        Remove-Item -LiteralPath $stagedValidationCsv -Force
    }
}

Write-Host "Independently validating the finished ZIP..."
$archiveValidation = Invoke-SubmissionValidator `
    -SubmissionPath $archivePath `
    -OutputJson $archiveValidationJson `
    -OutputCsv $archiveValidationCsv `
    -Stage "Submission archive"

$archiveItem = Get-Item -LiteralPath $archivePath
$archiveSha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$createdUtc = [DateTimeOffset]::UtcNow.ToString(
    "o",
    [Globalization.CultureInfo]::InvariantCulture
)
$manifest = [ordered] @{
    schema_version = 1
    created_utc = $createdUtc
    archive = [ordered] @{
        path = $archiveItem.FullName
        sha256 = $archiveSha256
        size_bytes = [long] $archiveItem.Length
        flat_root = $true
    }
    counts = [ordered] @{
        expected_cases = $expectedCount
        masks = [int] $archiveValidation.validated_mask_count
        subtype_rows = [int] $archiveValidation.validated_csv_row_count
        archive_files = $expectedCount + 1
    }
    validator_artifacts = [ordered] @{
        prediction_directory_json = $directoryValidationJson
        prediction_directory_csv = $directoryValidationCsv
        archive_json = $archiveValidationJson
        archive_csv = $archiveValidationCsv
    }
    validation = [ordered] @{
        prediction_directory_valid = [bool] $directoryValidation.valid
        archive_valid = [bool] $archiveValidation.valid
    }
}
Write-AtomicJson `
    -Value $manifest `
    -Destination $manifestPath `
    -ResolvedDeliveryRoot $resolvedDeliveryRoot `
    -AllowReplace:$Force

Write-Host "Submission package complete."
Write-Host "Archive: $archivePath"
Write-Host "SHA-256: $archiveSha256"
Write-Host "Manifest: $manifestPath"

[pscustomobject] @{
    Archive = $archivePath
    Sha256 = $archiveSha256
    SizeBytes = [long] $archiveItem.Length
    Manifest = $manifestPath
    MaskCount = [int] $archiveValidation.validated_mask_count
    SubtypeRowCount = [int] $archiveValidation.validated_csv_row_count
}
