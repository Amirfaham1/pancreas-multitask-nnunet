<#
.SYNOPSIS
Build the submission report with Pandoc and Tectonic using Springer LNCS.

.EXAMPLE
.\scripts\build_report.ps1 `
  -TectonicPath D:\MLQuizWork\tools\tectonic-0.16.9\tectonic.exe

.EXAMPLE
.\scripts\build_report.ps1 `
  -TectonicPath D:\MLQuizWork\tools\tectonic-0.16.9\tectonic.exe `
  -Final

The -Final switch rejects unresolved result tokens and PENDING/DRAFT/TODO/TBD/
PLACEHOLDER markers, then requires at least eight content pages before the
references. The report Markdown is never modified: a temporary Pandoc filter
supports the LNCS abstract environment and starts References on a new page so
the content-page check is unambiguous.
#>

[CmdletBinding()]
param(
    [string] $InputPath,
    [string] $OutputPath,
    [string] $PandocPath,
    [string] $TectonicPath,
    [switch] $Final,
    [switch] $Offline,
    [switch] $KeepBuildDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$reportDirectory = Join-Path $repoRoot "report"
$templatePath = Join-Path $reportDirectory "llncs-pandoc.tex"

if ([string]::IsNullOrWhiteSpace($InputPath)) {
    $InputPath = Join-Path $reportDirectory "report.md"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $reportDirectory "Amirfaham_Fallahpour_results.pdf"
}

$InputPath = [IO.Path]::GetFullPath($InputPath)
$OutputPath = [IO.Path]::GetFullPath($OutputPath)

function Resolve-Executable {
    param(
        [string] $RequestedPath,
        [string] $CommandName,
        [string[]] $FallbackPaths
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        $resolvedRequestedPath = [IO.Path]::GetFullPath($RequestedPath)
        if (-not (Test-Path -LiteralPath $resolvedRequestedPath -PathType Leaf)) {
            throw "$CommandName was not found at '$resolvedRequestedPath'."
        }
        return $resolvedRequestedPath
    }

    $command = Get-Command $CommandName -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    foreach ($candidate in $FallbackPaths) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and
            (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return [IO.Path]::GetFullPath($candidate)
        }
    }

    throw "Could not find $CommandName. Pass -${CommandName}Path explicitly."
}

$pandocFallbacks = @()
if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    $pandocFallbacks += Join-Path $env:LOCALAPPDATA "Pandoc\pandoc.exe"
}
if (-not [string]::IsNullOrWhiteSpace($env:ProgramFiles)) {
    $pandocFallbacks += Join-Path $env:ProgramFiles "Pandoc\pandoc.exe"
}

$pandocExecutable = Resolve-Executable `
    -RequestedPath $PandocPath `
    -CommandName "pandoc" `
    -FallbackPaths $pandocFallbacks
$tectonicExecutable = Resolve-Executable `
    -RequestedPath $TectonicPath `
    -CommandName "tectonic" `
    -FallbackPaths @()

foreach ($requiredFile in @($InputPath, $templatePath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required report file not found: '$requiredFile'."
    }
}

$source = [IO.File]::ReadAllText($InputPath)
if ($Final) {
    $unresolvedV5Tokens = @(
        [regex]::Matches($source, "(?<![A-Za-z0-9_])V5_[A-Z0-9_]+(?![A-Za-z0-9_])") |
            ForEach-Object { $_.Value } |
            Sort-Object -Unique
    )
    if ($unresolvedV5Tokens.Count -gt 0) {
        $preview = ($unresolvedV5Tokens | Select-Object -First 10) -join ", "
        $suffix = if ($unresolvedV5Tokens.Count -gt 10) {
            " (and $($unresolvedV5Tokens.Count - 10) more)"
        }
        else {
            ""
        }
        throw "Final build refused: unresolved v5 result token(s): $preview$suffix."
    }
    if ($source -match "(?m)\bPENDING(?:_[A-Z0-9_]+)?\b") {
        throw "Final build refused: report.md still contains a PENDING marker."
    }
    if ($source -match "(?m)\b(?:TODO|TBD|PLACEHOLDER)(?:_[A-Z0-9_]+)?\b") {
        throw "Final build refused: report.md still contains an unresolved TODO/TBD/PLACEHOLDER marker."
    }
    if ($source -match "DRAFT[^\r\n]*NOT READY FOR SUBMISSION") {
        throw "Final build refused: report.md still contains the DRAFT warning."
    }
}

$temporaryRoot = Join-Path $repoRoot ".tmp"
New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
$buildDirectory = Join-Path $temporaryRoot ("report-build-{0}" -f $PID)
if (Test-Path -LiteralPath $buildDirectory) {
    throw "Refusing to reuse an existing build directory: '$buildDirectory'."
}
New-Item -ItemType Directory -Path $buildDirectory | Out-Null

$filterPath = Join-Path $buildDirectory "lncs_filter.lua"
$texPath = Join-Path $buildDirectory "report.tex"
$builtPdfPath = Join-Path $buildDirectory "report.pdf"
$auxPath = Join-Path $buildDirectory "report.aux"
$logPath = Join-Path $buildDirectory "report.log"

$luaFilter = @'
local stringify = pandoc.utils.stringify

function Pandoc(document)
  local output = pandoc.List()
  local abstract_blocks = pandoc.List()
  local collecting_abstract = false

  for _, block in ipairs(document.blocks) do
    local is_level_one = block.t == "Header" and block.level == 1
    local heading = is_level_one and stringify(block.content) or ""

    if is_level_one and heading == "Abstract" then
      collecting_abstract = true
    elseif collecting_abstract and is_level_one then
      document.meta.abstract = pandoc.MetaBlocks(abstract_blocks)
      collecting_abstract = false
      output:insert(block)
    elseif collecting_abstract then
      abstract_blocks:insert(block)
    else
      if is_level_one and heading == "References" then
        output:insert(pandoc.RawBlock(
          "latex",
          "\\clearpage\\phantomsection\\label{references-start}"
        ))
      end
      output:insert(block)
    end
  end

  if collecting_abstract then
    document.meta.abstract = pandoc.MetaBlocks(abstract_blocks)
  end
  document.blocks = output
  return document
end
'@
[IO.File]::WriteAllText(
    $filterPath,
    $luaFilter,
    [Text.UTF8Encoding]::new($false)
)

try {
    $resourcePath = "$reportDirectory;$repoRoot"
    $pandocArguments = @(
        $InputPath,
        "--from=markdown+raw_tex+tex_math_dollars+tex_math_single_backslash",
        "--to=latex",
        "--standalone",
        "--template=$templatePath",
        "--lua-filter=$filterPath",
        "--resource-path=$resourcePath",
        "--metadata=author-meta:Amirfaham Fallahpour",
        "--output=$texPath"
    )
    & $pandocExecutable @pandocArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pandoc failed with exit code $LASTEXITCODE."
    }

    $tectonicArguments = @(
        "-X", "compile", $texPath,
        "--outdir", $buildDirectory,
        "--reruns", "2",
        "--keep-logs",
        "--keep-intermediates",
        "-Z", "search-path=$reportDirectory",
        "-Z", "search-path=$repoRoot"
    )
    if ($Offline) {
        $tectonicArguments += "--only-cached"
    }

    Push-Location $reportDirectory
    try {
        & $tectonicExecutable @tectonicArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Tectonic failed with exit code $LASTEXITCODE. See '$logPath'."
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $builtPdfPath -PathType Leaf)) {
        throw "Tectonic exited successfully but did not create '$builtPdfPath'."
    }

    $referenceStartPage = $null
    if (Test-Path -LiteralPath $auxPath -PathType Leaf) {
        $auxText = [IO.File]::ReadAllText($auxPath)
        $labelMatch = [regex]::Match(
            $auxText,
            "\\newlabel\{references-start\}\{\{[^}]*\}\{(\d+)\}"
        )
        if ($labelMatch.Success) {
            $referenceStartPage = [int] $labelMatch.Groups[1].Value
        }
    }

    $totalPages = $null
    if (Test-Path -LiteralPath $logPath -PathType Leaf) {
        $logText = [IO.File]::ReadAllText($logPath)
        $pageMatch = [regex]::Match(
            $logText,
            "Output written on .*?\((\d+) pages?(?:,[^)]*)?\)",
            [Text.RegularExpressions.RegexOptions]::Singleline
        )
        if ($pageMatch.Success) {
            $totalPages = [int] $pageMatch.Groups[1].Value
        }
    }

    if ($null -eq $referenceStartPage) {
        if ($Final) {
            throw "Final build could not determine the first references page."
        }
        Write-Warning "Could not determine the first references page from report.aux."
    }
    else {
        $contentPages = $referenceStartPage - 1
        if ($Final -and $contentPages -lt 8) {
            throw "Final build has $contentPages content pages; at least 8 are required."
        }
        Write-Host "Content pages before References: $contentPages"
    }

    if ($null -ne $totalPages) {
        Write-Host "Total PDF pages: $totalPages"
    }

    $outputDirectory = Split-Path -Parent $OutputPath
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    $stagedOutput = "$OutputPath.tmp"
    Copy-Item -LiteralPath $builtPdfPath -Destination $stagedOutput -Force
    Move-Item -LiteralPath $stagedOutput -Destination $OutputPath -Force
    Write-Host "Report written to: $OutputPath"
}
finally {
    if (-not $KeepBuildDirectory -and
        (Test-Path -LiteralPath $buildDirectory)) {
        $safeTemporaryRoot = [IO.Path]::GetFullPath($temporaryRoot).TrimEnd('\') + '\'
        $resolvedBuildDirectory = [IO.Path]::GetFullPath($buildDirectory)
        if (-not $resolvedBuildDirectory.StartsWith(
            $safeTemporaryRoot,
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Refusing to clean a build directory outside '$temporaryRoot'."
        }
        Remove-Item -LiteralPath $resolvedBuildDirectory -Recurse -Force
    }
    elseif ($KeepBuildDirectory) {
        Write-Host "Build intermediates retained at: $buildDirectory"
    }
}
