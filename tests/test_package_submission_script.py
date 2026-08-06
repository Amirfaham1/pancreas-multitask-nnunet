from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "Package-Submission.ps1"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def _powershell_executable() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def test_package_script_parses_as_powershell_without_running_it() -> None:
    executable = _powershell_executable()
    if executable is None:
        pytest.skip(
            "PowerShell is unavailable; remaining tests still inspect the script statically"
        )
    parser_command = r"""
& {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $env:PACKAGE_SCRIPT_TO_PARSE, [ref] $tokens, [ref] $errors
    ) | Out-Null
    if ($errors.Count -gt 0) {
        $errors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }
        exit 1
    }
}
"""
    environment = os.environ.copy()
    environment["PACKAGE_SCRIPT_TO_PARSE"] = str(SCRIPT)
    completed = subprocess.run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            parser_command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_defaults_and_exact_submission_contract_are_fixed() -> None:
    assert r"D:\MLQuizWork\submission\Amirfaham_Fallahpour_results" in SOURCE
    assert r"D:\MLQuizWork\nnUNet_raw\Dataset501_PancreasMultitask\imagesTs" in SOURCE
    assert 'Join-Path $PSScriptRoot "..\\delivery"' in SOURCE
    assert "$expectedCount = 72" in SOURCE
    assert '$csvName = "subtype_results.csv"' in SOURCE
    assert '$archiveName = "Amirfaham_Fallahpour_results.zip"' in SOURCE


def test_validator_runs_before_packaging_and_before_and_after_commit() -> None:
    source_validation = SOURCE.index("$directoryValidation = Invoke-SubmissionValidator")
    archive_creation = SOURCE.index("New-FlatZipArchive", source_validation)
    staged_validation = SOURCE.index("$null = Invoke-SubmissionValidator", archive_creation)
    archive_commit = SOURCE.index("Commit-AtomicFile `", staged_validation)
    committed_validation = SOURCE.index(
        "$archiveValidation = Invoke-SubmissionValidator", archive_commit
    )
    assert (
        source_validation
        < archive_creation
        < staged_validation
        < archive_commit
        < committed_validation
    )
    assert '"--output-json", $OutputJson' in SOURCE
    assert '"--output-csv", $OutputCsv' in SOURCE
    assert "submission_directory_validation.json" in SOURCE
    assert "submission_archive_validation.json" in SOURCE


def test_archive_is_flat_and_cannot_copy_test_images_or_extra_source_files() -> None:
    assert "$file.Name" in SOURCE
    assert "$sourceFiles.Count -ne ($expectedCount + 1)" in SOURCE
    assert "$maskFiles.Count -ne $expectedCount" in SOURCE
    assert "$csvFiles.Count -ne 1" in SOURCE
    assert "$packageFiles = @($maskFiles + $csvFiles)" in SOURCE
    assert "Copy-Item" not in SOURCE
    assert "Compress-Archive" not in SOURCE


def test_force_atomic_replacement_and_every_removal_have_delivery_path_guards() -> None:
    refusal = SOURCE.index("if (-not $Force)")
    replacement = SOURCE.index("Commit-AtomicFile `", refusal)
    guard = SOURCE.rindex("Assert-DirectDeliveryChild", refusal, replacement)
    assert refusal < guard < replacement
    assert "Remove-Item -LiteralPath $resolvedExistingArchive" not in SOURCE
    assert "[IO.File]::Replace($resolvedSource, $resolvedExisting, $backup)" in SOURCE
    assert "if (-not $AllowReplace)" in SOURCE
    assert "[IO.File]::Move($resolvedSource, $resolvedDestination)" in SOURCE
    assert "-AllowReplace:$Force" in SOURCE
    assert "-Recurse" not in SOURCE

    for line_number, line in enumerate(SOURCE.splitlines()):
        if "Remove-Item -LiteralPath" not in line:
            continue
        nearby = SOURCE.splitlines()[max(0, line_number - 5) : line_number]
        assert any("Assert-DirectDeliveryChild" in item for item in nearby)


def test_commit_time_no_force_conflict_preserves_existing_file_and_cleans_stage(
    tmp_path: Path,
) -> None:
    """Exercise the atomic no-replace branch without running the package pipeline."""

    executable = _powershell_executable()
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    staged = delivery / ".staged.tmp.zip"
    archive = delivery / "Amirfaham_Fallahpour_results.zip"
    staged.write_bytes(b"new staged archive")
    archive.write_bytes(b"concurrent existing archive")

    harness = r"""
& {
    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile(
        $env:PACKAGE_SCRIPT_UNDER_TEST, [ref] $tokens, [ref] $errors
    )
    if ($errors.Count -gt 0) { throw "Package script did not parse." }
    foreach ($name in @(
        "Get-NormalizedFullPath", "Assert-DirectDeliveryChild", "Assert-LeafFile",
        "Commit-AtomicFile"
    )) {
        $definition = $ast.FindAll(
            {
                param($node)
                $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -ceq $name
            },
            $true
        )
        if ($definition.Count -ne 1) { throw "Missing function $name." }
        Invoke-Expression $definition[0].Extent.Text
    }
    try {
        Commit-AtomicFile `
            -Source $env:PACKAGE_STAGED_FILE `
            -Destination $env:PACKAGE_EXISTING_ARCHIVE `
            -ResolvedDeliveryRoot $env:PACKAGE_DELIVERY_ROOT
    }
    finally {
        if (Test-Path -LiteralPath $env:PACKAGE_STAGED_FILE) {
            Remove-Item -LiteralPath $env:PACKAGE_STAGED_FILE -Force
        }
    }
}
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PACKAGE_SCRIPT_UNDER_TEST": str(SCRIPT),
            "PACKAGE_STAGED_FILE": str(staged),
            "PACKAGE_EXISTING_ARCHIVE": str(archive),
            "PACKAGE_DELIVERY_ROOT": str(delivery),
        }
    )
    completed = subprocess.run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            harness,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )

    assert completed.returncode != 0
    assert "refusing to replace it without -Force" in completed.stderr + completed.stdout
    assert archive.read_bytes() == b"concurrent existing archive"
    assert not staged.exists()


def test_force_atomic_replacement_retries_a_transient_sharing_violation(
    tmp_path: Path,
) -> None:
    executable = _powershell_executable()
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    staged = delivery / ".staged.tmp.zip"
    archive = delivery / "Amirfaham_Fallahpour_results.zip"
    ready = tmp_path / "lock-ready"
    release = tmp_path / "release-lock"
    staged.write_bytes(b"new staged archive")
    archive.write_bytes(b"old committed archive")

    locker_script = r"""
& {
    $stream = [IO.File]::Open(
        $env:PACKAGE_EXISTING_ARCHIVE,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::None
    )
    try {
        [IO.File]::WriteAllText($env:PACKAGE_LOCK_READY, "ready")
        $deadline = [DateTime]::UtcNow.AddSeconds(15)
        while (-not (Test-Path -LiteralPath $env:PACKAGE_RELEASE_LOCK)) {
            if ([DateTime]::UtcNow -ge $deadline) { throw "Timed out waiting to release lock." }
            Start-Sleep -Milliseconds 10
        }
        # Hold beyond the former ~4.25-second retry ceiling so this test covers
        # the long transient lock observed under a busy OneDrive test run.
        Start-Sleep -Milliseconds 6000
    }
    finally {
        $stream.Dispose()
    }
}
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PACKAGE_EXISTING_ARCHIVE": str(archive),
            "PACKAGE_LOCK_READY": str(ready),
            "PACKAGE_RELEASE_LOCK": str(release),
        }
    )
    locker = subprocess.Popen(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            locker_script,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.is_file() and locker.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file(), locker.communicate(timeout=2)

        harness = r"""
& {
    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile(
        $env:PACKAGE_SCRIPT_UNDER_TEST, [ref] $tokens, [ref] $errors
    )
    if ($errors.Count -gt 0) { throw "Package script did not parse." }
    foreach ($name in @(
        "Get-NormalizedFullPath", "Assert-DirectDeliveryChild",
        "Get-ReparsePointTag", "Assert-NoRedirectingReparsePointInPath",
        "Assert-LeafFile", "Commit-AtomicFile"
    )) {
        $definition = $ast.FindAll(
            {
                param($node)
                $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -ceq $name
            },
            $true
        )
        if ($definition.Count -ne 1) { throw "Missing function $name." }
        Invoke-Expression $definition[0].Extent.Text
    }
    [IO.File]::WriteAllText($env:PACKAGE_RELEASE_LOCK, "release")
    Commit-AtomicFile `
        -Source $env:PACKAGE_STAGED_FILE `
        -Destination $env:PACKAGE_EXISTING_ARCHIVE `
        -ResolvedDeliveryRoot $env:PACKAGE_DELIVERY_ROOT `
        -AllowReplace -Verbose
}
"""
        commit_environment = environment.copy()
        commit_environment.update(
            {
                "PACKAGE_SCRIPT_UNDER_TEST": str(SCRIPT),
                "PACKAGE_STAGED_FILE": str(staged),
                "PACKAGE_DELIVERY_ROOT": str(delivery),
            }
        )
        completed = subprocess.run(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                harness,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=commit_environment,
        )
    finally:
        release.touch(exist_ok=True)
        locker_stdout, locker_stderr = locker.communicate(timeout=5)

    assert locker.returncode == 0, locker_stderr + locker_stdout
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "temporarily blocked by a file lock" in completed.stderr + completed.stdout
    assert archive.read_bytes() == b"new staged archive"
    assert not staged.exists()
    assert not list(delivery.glob("*.replace-backup"))


def test_native_stderr_and_reparse_points_are_handled_explicitly() -> None:
    native_call = SOURCE.index("$validatorMessages = @(& $resolvedPython")
    continue_setting = SOURCE.rindex('$ErrorActionPreference = "Continue"', 0, native_call)
    exit_capture = SOURCE.index("$validatorExitCode = $LASTEXITCODE", native_call)
    preference_restore = SOURCE.index(
        "$ErrorActionPreference = $previousErrorActionPreference", exit_capture
    )
    assert continue_setting < native_call < exit_capture < preference_restore

    assert "function Get-ReparsePointTag" in SOURCE
    assert "function Assert-NoRedirectingReparsePointInPath" in SOURCE
    assert SOURCE.count("Assert-NoRedirectingReparsePointInPath `") >= 5
    assert "[IO.FileAttributes]::ReparsePoint" in SOURCE
    assert '"fsutil.exe"' in SOURCE
    assert "($tag -band 0x20000000)" in SOURCE
    assert "OneDrive cloud placeholders" in SOURCE


def test_reparse_guard_accepts_default_onedrive_path_and_rejects_junction() -> None:
    executable = _powershell_executable()
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    parser_command = r"""
& {
    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile(
        $env:PACKAGE_SCRIPT_TO_PARSE, [ref] $tokens, [ref] $errors
    )
    if ($errors.Count -gt 0) { exit 1 }
    foreach ($functionName in @(
        "Get-ReparsePointTag",
        "Assert-NoRedirectingReparsePointInPath"
    )) {
        $functionAst = $ast.FindAll(
            {
                param($node)
                $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -eq $functionName
            },
            $true
        ) | Select-Object -First 1
        if ($null -eq $functionAst) { exit 2 }
        Invoke-Expression $functionAst.Extent.Text
    }

    Assert-NoRedirectingReparsePointInPath `
        -Path $env:PACKAGE_DEFAULT_DELIVERY `
        -Description "Default delivery path"

    $knownJunction = "C:\Documents and Settings"
    $junctionItem = Get-Item -LiteralPath $knownJunction -Force -ErrorAction SilentlyContinue
    if ($null -ne $junctionItem -and
        ($junctionItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        $rejected = $false
        try {
            Assert-NoRedirectingReparsePointInPath `
                -Path (Join-Path $knownJunction "delivery") `
                -Description "Known junction"
        }
        catch {
            $rejected = $true
        }
        if (-not $rejected) { exit 3 }
    }
}
"""
    environment = os.environ.copy()
    environment["PACKAGE_SCRIPT_TO_PARSE"] = str(SCRIPT)
    environment["PACKAGE_DEFAULT_DELIVERY"] = str(SCRIPT.parents[1] / "delivery")
    completed = subprocess.run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            parser_command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_manifest_is_atomic_and_records_required_provenance() -> None:
    assert "[IO.File]::Replace" in SOURCE
    assert "[IO.File]::Move($resolvedSource, $resolvedDestination)" in SOURCE
    assert "$stream.Flush($true)" in SOURCE
    for field in (
        "created_utc",
        "sha256",
        "size_bytes",
        "validator_artifacts",
        "expected_cases",
        "masks",
        "subtype_rows",
    ):
        assert field in SOURCE


def test_package_script_builds_and_revalidates_exact_flat_archive(tmp_path: Path) -> None:
    executable = _powershell_executable()
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    predictions = tmp_path / "predictions"
    test_images = tmp_path / "test_images"
    delivery = tmp_path / "delivery"
    predictions.mkdir()
    test_images.mkdir()

    affine = np.diag([0.75, 0.75, 2.0, 1.0])
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask = np.zeros((2, 3, 4), dtype=np.uint8)
    mask[0, :, :] = 1
    mask[0, 0, 0] = 2
    expected_members: list[str] = []
    csv_rows = ["Names,Subtype"]
    for index in range(72):
        case_id = f"quiz_{index:03d}"
        mask_name = f"{case_id}.nii.gz"
        nib.save(
            nib.Nifti1Image(image, affine),
            str(test_images / f"{case_id}_0000.nii.gz"),
        )
        nib.save(nib.Nifti1Image(mask, affine), str(predictions / mask_name))
        expected_members.append(mask_name)
        csv_rows.append(f"{mask_name},{index % 3}")
    (predictions / "subtype_results.csv").write_text(
        "\n".join(csv_rows) + "\n",
        encoding="utf-8",
    )
    expected_members.append("subtype_results.csv")

    command = [
        executable,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-PredictionDirectory",
        str(predictions),
        "-TestImages",
        str(test_images),
        "-DeliveryRoot",
        str(delivery),
        "-PythonExecutable",
        sys.executable,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout

    archive = delivery / "Amirfaham_Fallahpour_results.zip"
    manifest_path = delivery / "package_manifest.json"
    archive_validation_path = delivery / "submission_archive_validation.json"
    assert archive.is_file()
    assert manifest_path.is_file()
    assert archive_validation_path.is_file()

    with zipfile.ZipFile(archive) as handle:
        members = handle.namelist()
    assert sorted(members) == sorted(expected_members)
    assert len(members) == 73
    assert all("/" not in name and "\\" not in name for name in members)

    archive_bytes = archive.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(archive_validation_path.read_text(encoding="utf-8"))
    assert manifest["archive"]["sha256"] == archive_sha256
    assert manifest["counts"] == {
        "expected_cases": 72,
        "masks": 72,
        "subtype_rows": 72,
        "archive_files": 73,
    }
    assert manifest["validation"] == {
        "prediction_directory_valid": True,
        "archive_valid": True,
    }
    assert validation["valid"] is True
    assert validation["submission"] == str(archive.resolve())

    refused = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert refused.returncode != 0
    assert "pass -Force" in refused.stderr + refused.stdout
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_sha256

    invalid_mask = mask.copy()
    invalid_mask[1, 0, 0] = 3
    first_mask = predictions / "quiz_000.nii.gz"
    nib.save(nib.Nifti1Image(invalid_mask, affine), str(first_mask))
    force_command = [*command, "-Force"]
    failed_replacement = subprocess.run(
        force_command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    failed_output = failed_replacement.stderr + failed_replacement.stdout
    assert failed_replacement.returncode != 0
    assert "Prediction directory failed validation with exit code 1" in failed_output
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_sha256

    nib.save(nib.Nifti1Image(mask, affine), str(first_mask))
    replacement_rows = csv_rows.copy()
    replacement_rows[1] = "quiz_000.nii.gz,2"
    (predictions / "subtype_results.csv").write_text(
        "\n".join(replacement_rows) + "\n",
        encoding="utf-8",
    )
    forced = subprocess.run(
        force_command,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert forced.returncode == 0, forced.stderr + forced.stdout
    replacement_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert replacement_sha256 != archive_sha256
    with zipfile.ZipFile(archive) as handle:
        replacement_csv = handle.read("subtype_results.csv").decode("utf-8")
    assert "quiz_000.nii.gz,2" in replacement_csv.splitlines()
    replacement_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert replacement_manifest["archive"]["sha256"] == replacement_sha256
    assert replacement_manifest["validation"] == {
        "prediction_directory_valid": True,
        "archive_valid": True,
    }
    assert not list(delivery.glob("*.replace-backup"))
