from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "selected_test_ps51_compatibility_protocol_v1.json"
WRAPPER = ROOT / "scripts" / "Run-V5LockedSelectedTestAndPackagePS51.ps1"
COMMON = ROOT / "scripts" / "V5-LockedDeliveryCommon.ps1"
ORIGINAL_WRAPPER = ROOT / "scripts" / "Run-V5LockedSelectedTestAndPackage.ps1"
FINAL_LOCK = ROOT / "configs" / "phd_final_candidate_lock_v5.json"
PREDICT = ROOT / "scripts" / "predict_joint.py"
PACKAGE = ROOT / "scripts" / "Package-Submission.ps1"
VALIDATOR = ROOT / "scripts" / "validate_submission.py"
SETUP = ROOT / "scripts" / "Set-QuizEnvironment.ps1"
PACKAGE_INIT = ROOT / "src" / "pancreas_multitask" / "__init__.py"
PYTHON = Path(r"D:\MLQuizWork\.venv\Scripts\python.exe")
POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")

PROTOCOL_SHA256 = "9c92e8c2d17107937ea913f2ea9bbe0f7f901b24a2228d31bee53e34a5923681"
TRANSFORMED_COMMON_SHA256 = "b3d56bc5e9a7270770ffcb454bef9b3f0eef603b931419736ee2443dd08374bb"
SELECTED_BODY_SHA256 = "c8e0fe1678ec5334887c83b9ef3c9d645a7daee2ab9e1370773ebf1abbd042af"
POWERSHELL_SHA256 = "7600ffe12da441fe89d035b13801e8e91d064bc544a27b19a5cf49f6ab8b18f5"
PACKAGE_INIT_SHA256 = "430cedcca46cd2c0ba5d1f88762e0c2831f2c11484baf16baea306c373e120d4"

SOURCE = WRAPPER.read_bytes().decode("utf-8")
ORIGINAL_SOURCE = ORIGINAL_WRAPPER.read_bytes().decode("utf-8")
PROTOCOL_PAYLOAD = json.loads(PROTOCOL.read_bytes().decode("utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        pytest.skip("Native Windows PowerShell is unavailable")
    return executable


def _run_powershell(
    command: str,
    environment: dict[str, str],
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _native_powershell(),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )


def _function_loader(names: tuple[str, ...]) -> str:
    quoted = ",".join(f'"{name}"' for name in names)
    return rf"""
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $env:PS51_WRAPPER, [ref] $tokens, [ref] $errors
)
if ($errors.Count -ne 0) {{ throw ($errors | Out-String) }}
$wanted = @({quoted})
$nodes = @($ast.FindAll({{
    param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
        $wanted -contains $node.Name
}}, $true))
if ($nodes.Count -ne $wanted.Count) {{
    throw "Could not isolate every requested compatibility helper."
}}
$ordered = foreach ($name in $wanted) {{
    $nodes | Where-Object Name -ceq $name | Select-Object -First 1
}}
. ([ScriptBlock]::Create((($ordered | ForEach-Object {{
    $_.Extent.Text
}}) -join "`n")))
"""


def _patched_common_loader() -> str:
    return r"""
$protocol = Get-Content -LiteralPath $env:PS51_PROTOCOL -Raw | ConvertFrom-Json
$source = [IO.File]::ReadAllText($env:PS51_COMMON)
foreach ($substitution in @(
    $protocol.only_permitted_common_compatibility_substitutions
)) {
    $source = $source.Replace(
        [string] $substitution.original,
        [string] $substitution.replacement
    )
}
. ([ScriptBlock]::Create($source))
"""


def _transformed_common() -> str:
    source = COMMON.read_bytes().decode("utf-8")
    for substitution in PROTOCOL_PAYLOAD["only_permitted_common_compatibility_substitutions"]:
        assert source.count(substitution["original"]) == 1
        source = source.replace(
            substitution["original"],
            substitution["replacement"],
        )
    return source


def test_protocol_and_every_frozen_dependency_are_exact() -> None:
    assert _sha256(PROTOCOL) == PROTOCOL_SHA256
    assert PROTOCOL_SHA256 in SOURCE
    assert PROTOCOL_PAYLOAD["status"] == (
        "frozen_before_selected_test_ps51_implementation_or_any_selected_test_access"
    )

    bound_candidate = PROTOCOL_PAYLOAD["bound_final_candidate"]
    assert _sha256(FINAL_LOCK) == bound_candidate["sha256"]
    bound = PROTOCOL_PAYLOAD["bound_original_implementation"]
    expected = (
        (COMMON, bound["common_sha256"]),
        (ORIGINAL_WRAPPER, bound["selected_wrapper_sha256"]),
        (PREDICT, bound["prediction_script"]["sha256"]),
        (PACKAGE, bound["package_script"]["sha256"]),
        (VALIDATOR, bound["validator_script"]["sha256"]),
        (SETUP, bound["environment_script"]["sha256"]),
        (PYTHON, bound["python_executable"]["sha256"]),
        (PACKAGE_INIT, PACKAGE_INIT_SHA256),
        (POWERSHELL, POWERSHELL_SHA256),
    )
    for path, digest in expected:
        assert path.is_file()
        assert _sha256(path) == digest


def test_protocol_was_frozen_before_access_and_allows_no_test_iteration() -> None:
    access = PROTOCOL_PAYLOAD["access_at_lock"]
    assert access["inference_invocation_count"] == 0
    assert access["package_invocation_count"] == 0
    for field, value in access.items():
        if field.endswith("tested_or_opened"):
            assert value is False

    contract = PROTOCOL_PAYLOAD["execution_contract"]
    assert contract["selected_test_inference_invocation_count"] == 1
    assert contract["package_invocation_count"] == 1
    for field in (
        "test_or_official_data_access_during_compatibility_tests",
        "inference_or_package_invocation_during_compatibility_tests",
        "package_force_or_replacement_is_permitted",
        "second_selected_test_run_or_package_attempt_is_permitted",
        "test_targets_submission_feedback_or_post_validation_model_changes_are_permitted",
    ):
        assert contract[field] is False


def test_native_windows_powershell_51_parses_wrapper_and_transformed_common() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "PS51_WRAPPER": str(WRAPPER),
            "PS51_COMMON": str(COMMON),
            "PS51_PROTOCOL": str(PROTOCOL),
        }
    )
    command = r"""
if ($PSVersionTable.PSEdition -cne "Desktop" -or
    $PSVersionTable.PSVersion.ToString() -cne "5.1.26100.8875") {
    throw "The compatibility parser test did not use the frozen PS5.1 host."
}
$tokens = $null
$errors = $null
[Management.Automation.Language.Parser]::ParseFile(
    $env:PS51_WRAPPER, [ref] $tokens, [ref] $errors
) | Out-Null
if ($errors.Count -ne 0) { throw ($errors | Out-String) }
$protocol = Get-Content -LiteralPath $env:PS51_PROTOCOL -Raw | ConvertFrom-Json
$source = [IO.File]::ReadAllText($env:PS51_COMMON)
foreach ($substitution in @(
    $protocol.only_permitted_common_compatibility_substitutions
)) {
    $source = $source.Replace(
        [string] $substitution.original,
        [string] $substitution.replacement
    )
}
$tokens = $null
$errors = $null
[Management.Automation.Language.Parser]::ParseInput(
    $source, [ref] $tokens, [ref] $errors
) | Out-Null
if ($errors.Count -ne 0) { throw ($errors | Out-String) }
"parse_ok"
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "parse_ok" in completed.stdout


def test_wrapper_uses_no_powershell_7_only_construct() -> None:
    for forbidden in (
        "ConvertFrom-Json -AsHashtable",
        "ConvertFrom-Json -Depth",
        "$IsWindows",
        ".ArgumentList.Add",
        "ForEach-Object -Parallel",
    ):
        assert forbidden not in SOURCE


def test_exact_three_common_substitutions_are_the_only_patch() -> None:
    substitutions = PROTOCOL_PAYLOAD["only_permitted_common_compatibility_substitutions"]
    assert len(substitutions) == 3
    assert (
        COMMON.read_bytes()
        .decode("utf-8")
        .count("[IO.File]::Replace($temporary, $destination, $null)")
        == 1
    )
    transformed = _transformed_common()
    assert len(transformed) == 81725
    assert hashlib.sha256(transformed.encode()).hexdigest() == (TRANSFORMED_COMMON_SHA256)
    assert "$snapshot.PSObject.Properties.Count" not in transformed
    assert "$runtimeImplementation.PSObject.Properties.Count" not in transformed
    assert "[IO.File]::Replace($temporary, $destination, $null)" not in transformed
    assert "if ($substitutions.Count -ne 3)" in SOURCE
    assert "$patchedCommonText = $patchedCommonText.Replace($original, $replacement)" in (SOURCE)
    assert ". ([ScriptBlock]::Create($patchedCommonText))" in SOURCE


def test_frozen_original_selected_body_is_extracted_without_modification() -> None:
    bound = PROTOCOL_PAYLOAD["bound_original_implementation"]
    anchor = bound["selected_wrapper_semantic_body_anchor"]
    assert ORIGINAL_SOURCE.count(anchor) == 1
    body = ORIGINAL_SOURCE[ORIGINAL_SOURCE.index(anchor) :]
    assert len(body) == 36125
    assert hashlib.sha256(body.encode()).hexdigest() == SELECTED_BODY_SHA256

    extraction = SOURCE[SOURCE.index("$selectedWrapperBody =") :]
    assert ".Replace(" not in extraction
    assert " -replace " not in extraction.lower()
    assert "Get-V5PS51StringSha256 $selectedWrapperBody" in extraction
    assert ("$selectedBodyBootstrap = '$PSScriptRoot = $script:V5PS51ScriptRoot'") in extraction
    assert "$selectedBodyBootstrap + $selectedWrapperBody" in extraction
    assert ". $selectedBodyScript" in extraction


def test_original_parameter_surface_is_preserved_and_only_bootstrap_is_added() -> None:
    original_parameters = (
        "FinalCandidateLock",
        "ExpectedFinalCandidateLockSha256",
        "OfficialEvaluationGate",
        "ExpectedOfficialEvaluationGateSha256",
        "ModelDirectory",
        "NeuralCaseHeadBundle",
        "ExpectedCheckpointSha256",
        "ExpectedNeuralCaseHeadBundleSha256",
        "ExpectedNumericTrainDatasetSha256",
        "ExpectedPlansSha256",
        "ExpectedDatasetJsonSha256",
        "ExpectedEncoderComponentSha256",
        "ExpectedDecoderComponentSha256",
        "ExpectedClassificationComponentSha256",
        "TestImages",
        "OutputRoot",
        "WorkRoot",
        "PythonExecutable",
        "Device",
        "ImmutableBaselineRoot",
    )
    compatibility_parameters = (
        "CompatibilityProtocol",
        "ExpectedCompatibilityProtocolSha256",
        "ExpectedCompatibilityWrapperSha256",
        "ExpectedCompatibilityTestsSha256",
        "CompatibilityImplementationCommit",
    )
    for name in original_parameters + compatibility_parameters:
        assert f"${name}" in SOURCE
    assert '[ValidateSet("cuda")]' in SOURCE
    assert '$Device = "cuda"' in SOURCE


def test_every_binding_precedes_common_and_original_body_execution() -> None:
    common_execution = SOURCE.index(". ([ScriptBlock]::Create($patchedCommonText))")
    body_execution = SOURCE.index(". $selectedBodyScript")
    for token in (
        "PowerShell executable",
        "Hard-bound compatibility protocol",
        "Compatibility wrapper self-binding",
        "Compatibility tests binding",
        "merge-base --is-ancestor $Commit origin/main",
        "Final-candidate lock file",
        "Bound selected wrapper",
        "Bound Python executable",
        "Imported pancreas_multitask package initializer",
        "Transformed Common text",
        "Frozen selected-wrapper semantic body",
    ):
        assert SOURCE.index(token) < body_execution
    for name in (
        "common",
        "prediction script",
        "package script",
        "validator script",
        "environment script",
    ):
        assert SOURCE.index(f'Name = "{name}"') < body_execution
    assert common_execution < SOURCE.index("$selectedWrapperText =") < body_execution
    assert SOURCE.count("Assert-V5PS51FrozenCompatibilityBindings") == 2
    pre_body_recheck = SOURCE.index('Stage "Immediately before frozen selected-test body"')
    assert pre_body_recheck < body_execution


def test_bootstrap_only_lexically_binds_test_and_output_before_frozen_body() -> None:
    bootstrap = SOURCE[
        SOURCE.index("Assert-V5PS51PathEqual $TestImages") : SOURCE.index("$selectedWrapperText =")
    ]
    assert '"D:\\MLQuizWork\\nnUNet_raw\\Dataset501_PancreasMultitask\\imagesTs"' in SOURCE
    assert '"D:\\MLQuizWork\\phd_upgrade_v5\\selected_test_locked_v5"' in SOURCE
    assert bootstrap.count("$TestImages") == 1
    assert bootstrap.count("$OutputRoot") == 1
    for forbidden in (
        "$OfficialEvaluationGate",
        "$ExpectedOfficialEvaluationGateSha256",
        "$ImmutableBaselineRoot",
        "Get-V5BareLedgerPath",
        "New-V5ExclusiveLedger",
        "selected_test_run_consumed.json",
        "Test-Path -LiteralPath $TestImages",
        "Test-Path -LiteralPath $OutputRoot",
        "Resolve-Path -LiteralPath $TestImages",
        "Resolve-Path -LiteralPath $OutputRoot",
        "Get-Item -LiteralPath $TestImages",
        "Get-Item -LiteralPath $OutputRoot",
        "Get-ChildItem -LiteralPath $TestImages",
        "Get-ChildItem -LiteralPath $OutputRoot",
    ):
        assert forbidden not in bootstrap
    assert "Invoke-V5CheckedPython" not in bootstrap
    assert "Package-Submission.ps1 `" not in bootstrap


def test_original_ledger_test_inference_and_package_order_is_unchanged() -> None:
    body = ORIGINAL_SOURCE[
        ORIGINAL_SOURCE.index(
            PROTOCOL_PAYLOAD["bound_original_implementation"][
                "selected_wrapper_semantic_body_anchor"
            ]
        ) :
    ]
    cpu_preflight = body.index("Invoke-V5StrictCpuPreflight")
    mutex = body.index("Enter-V5NamedMutex", cpu_preflight)
    consume = body.index("New-V5ExclusiveLedger -Path $ledgerPath", mutex)
    output = body.index("New-Item -ItemType Directory -Path $resolvedOutputRoot", consume)
    preflight = body.index("Write-V5JsonAtomic -Path $preflightJson", output)
    test_access = body.index(
        'Assert-V5Directory $TestImages "Official test image directory"', preflight
    )
    inference = body.index("Invoke-V5CheckedPython", test_access)
    runtime = body.index("Assert-V5RuntimeArtifact", inference)
    artifacts = body.index("Get-V5InferenceArtifactSet", runtime)
    manifest = body.index("Write-V5JsonAtomic -Path $prePackageManifest", artifacts)
    frozen_ledger = body.index('status = "predictions_frozen_before_packaging"', manifest)
    package = body.index("& $packageScript", frozen_ledger)
    post_package = body.index("$postPackageArtifactSet =", package)
    completion = body.index("Write-V5JsonAtomic -Path $completionManifest", post_package)
    complete_ledger = body.index('status = "complete_and_consumed"', completion)
    assert (
        cpu_preflight
        < mutex
        < consume
        < output
        < preflight
        < test_access
        < inference
        < runtime
        < artifacts
        < manifest
        < frozen_ledger
        < package
        < post_package
        < completion
        < complete_ledger
    )
    assert body.count("Invoke-V5CheckedPython") == 1
    assert body.count("& $packageScript") == 1


def test_failure_path_and_second_run_remain_explicitly_consumed() -> None:
    body = ORIGINAL_SOURCE[ORIGINAL_SOURCE.index("$predictionScript = Join-Path") :]
    consume = body.index("New-V5ExclusiveLedger -Path $ledgerPath")
    ledger_created = body.index("$ledgerCreated = $true", consume)
    test_access = body.index("Assert-V5Directory $TestImages", ledger_created)
    failure = body.index('status = "failed_and_consumed_no_rerun"', test_access)
    assert consume < ledger_created < test_access < failure
    for token in (
        "one_use_run_remains_consumed = $true",
        "failed_stage = $currentStage",
        "error_type = $failure.Exception.GetType().FullName",
        "error_message = $failure.Exception.Message",
    ):
        assert token in body[failure:]
    failure_tail = body[failure:]
    assert "Remove-Item -LiteralPath $ledgerPath" not in failure_tail
    assert "Invoke-V5CheckedPython" not in failure_tail
    assert "& $packageScript" not in failure_tail


def test_package_validator_safety_and_no_post_validation_degree_of_freedom() -> None:
    body = ORIGINAL_SOURCE[ORIGINAL_SOURCE.index("$predictionScript = Join-Path") :]
    package_call = body[
        body.index("& $packageScript") : body.index(
            "if ($LASTEXITCODE -ne 0)", body.index("& $packageScript")
        )
    ]
    assert "-Force" not in package_call
    for token in (
        '"complete_no_second_classifier_iteration_permitted"',
        "further_classifier_training_selection_or_official_evaluation_permitted",
        "classifier_replacement_validation_gate",
        "$postPackageArtifactSet.ArtifactSetSha256",
        "Fresh delivery directory contains an unexpected package artifact set.",
        '"Amirfaham_Fallahpour_results.zip"',
        '"submission_archive_validation.json"',
        "validated_mask_count",
        "validated_csv_row_count",
        "archive_sha256",
        "file_count = 73",
    ):
        assert token in body
    for forbidden in (
        "train_neural_case_heads.py",
        "select_checkpoint.py",
        "evaluate_predictions.py",
        "--class-offset",
        "--threshold",
    ):
        assert forbidden not in body


def test_injected_psscriptroot_works_in_native_powershell_51(tmp_path: Path) -> None:
    script_root = tmp_path / "script root with spaces"
    script_root.mkdir()
    environment = os.environ.copy()
    environment["PS51_SYNTHETIC_ROOT"] = str(script_root)
    command = r"""
Set-StrictMode -Version Latest
$script:V5PS51ScriptRoot = [IO.Path]::GetFullPath($env:PS51_SYNTHETIC_ROOT)
$body = '$PSScriptRoot = $script:V5PS51ScriptRoot' + [Environment]::NewLine + @'
$resolved = Join-Path $PSScriptRoot "predict_joint.py"
if (-not $resolved.Equals(
    (Join-Path $script:V5PS51ScriptRoot "predict_joint.py"),
    [StringComparison]::OrdinalIgnoreCase
)) { throw "Injected PSScriptRoot did not flow into the exact body." }
'@
. ([ScriptBlock]::Create($body))
"root_ok"
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "root_ok" in completed.stdout


def test_lexical_test_and_output_path_typos_are_rejected_without_access(
    tmp_path: Path,
) -> None:
    expected_test = tmp_path / "never opened test" / "imagesTs"
    expected_output = tmp_path / "never opened output" / "selected"
    environment = os.environ.copy()
    environment.update(
        {
            "PS51_WRAPPER": str(WRAPPER),
            "PS51_EXPECTED_TEST": str(expected_test),
            "PS51_EXPECTED_OUTPUT": str(expected_output),
            "PS51_TYPO_TEST": str(expected_test) + "_typo",
            "PS51_TYPO_OUTPUT": str(expected_output) + "_typo",
        }
    )
    command = _function_loader(("Assert-V5PS51PathEqual",))
    command += r"""
Assert-V5PS51PathEqual `
    $env:PS51_EXPECTED_TEST `
    $env:PS51_EXPECTED_TEST `
    "Synthetic selected-test path"
Assert-V5PS51PathEqual `
    $env:PS51_EXPECTED_OUTPUT `
    $env:PS51_EXPECTED_OUTPUT `
    "Synthetic output path"
foreach ($pair in @(
    @($env:PS51_TYPO_TEST, $env:PS51_EXPECTED_TEST),
    @($env:PS51_TYPO_OUTPUT, $env:PS51_EXPECTED_OUTPUT)
)) {
    $rejected = $false
    try {
        Assert-V5PS51PathEqual $pair[0] $pair[1] "Synthetic typo"
    }
    catch {
        if ($_.Exception.Message -notmatch "path mismatch") { throw }
        $rejected = $true
    }
    if (-not $rejected) { throw "Synthetic typo was accepted." }
}
"typos_rejected"
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "typos_rejected" in completed.stdout
    assert not expected_test.exists()
    assert not expected_output.exists()


def test_native_ps51_reproduces_all_three_original_incident_primitives(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "existing-ledger.json"
    temporary = tmp_path / "replacement-ledger.json"
    environment = os.environ.copy()
    environment.update(
        {
            "PS51_DESTINATION": str(destination),
            "PS51_TEMPORARY": str(temporary),
        }
    )
    command = r"""
Set-StrictMode -Version Latest
$snapshot = [pscustomobject]@{ value = 1 }
$runtimeImplementation = [pscustomobject]@{ value = 1 }
$rawCountFailures = 0
foreach ($expression in @("snapshot", "runtimeImplementation")) {
    try {
        if ($expression -ceq "snapshot") {
            $null = $snapshot.PSObject.Properties.Count
        }
        else {
            $null = $runtimeImplementation.PSObject.Properties.Count
        }
    }
    catch [Management.Automation.PropertyNotFoundException] {
        $rawCountFailures++
    }
}
if ($rawCountFailures -ne 2) {
    throw "Native PS5.1 did not reproduce both property-count failures."
}
if (@($snapshot.PSObject.Properties).Count -ne 1 -or
    @($runtimeImplementation.PSObject.Properties).Count -ne 1) {
    throw "PS5.1-safe property counts are wrong."
}
[IO.File]::WriteAllText($env:PS51_DESTINATION, '{"status":"started"}')
[IO.File]::WriteAllText($env:PS51_TEMPORARY, '{"status":"complete"}')
$nullBackupFailed = $false
try {
    [IO.File]::Replace($env:PS51_TEMPORARY, $env:PS51_DESTINATION, $null)
}
catch [Management.Automation.MethodInvocationException] {
    $nullBackupFailed = $true
}
if (-not $nullBackupFailed) {
    throw "Native PS5.1 unexpectedly accepted a null File.Replace backup."
}
"three_incidents_reproduced"
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "three_incidents_reproduced" in completed.stdout


def test_ps51_atomic_replace_updates_existing_synthetic_ledger(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "selected-ledger.json"
    environment = os.environ.copy()
    environment.update(
        {
            "PS51_WRAPPER": str(WRAPPER),
            "PS51_COMMON": str(COMMON),
            "PS51_PROTOCOL": str(PROTOCOL),
            "PS51_LEDGER": str(ledger),
        }
    )
    command = _function_loader(("Invoke-V5PS51AtomicJsonReplace",))
    command += _patched_common_loader()
    command += r"""
$null = New-V5ExclusiveLedger -Path $env:PS51_LEDGER -Payload ([ordered]@{
    status = "started_and_consumed"
    selected_test_inference_invocation_count = 0
    package_invocation_count = 0
})
Write-V5JsonAtomic -Path $env:PS51_LEDGER -Payload ([ordered]@{
    status = "predictions_frozen_before_packaging"
    selected_test_inference_invocation_count = 1
    package_invocation_count = 0
})
Write-V5JsonAtomic -Path $env:PS51_LEDGER -Payload ([ordered]@{
    status = "complete_and_consumed"
    selected_test_inference_invocation_count = 1
    package_invocation_count = 1
})
"ledger_ok"
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "ledger_ok" in completed.stdout
    assert json.loads(ledger.read_text(encoding="utf-8")) == {
        "status": "complete_and_consumed",
        "selected_test_inference_invocation_count": 1,
        "package_invocation_count": 1,
    }
    assert ledger.read_bytes().startswith(b"{")
    assert not ledger.read_bytes().startswith(b"\xef\xbb\xbf")
    assert ledger.read_bytes().endswith(b"\n")
    assert not list(tmp_path.glob("*.ps51-replace-backup"))
    assert not list(tmp_path.glob("*.tmp"))


def test_synthetic_selected_ledger_cannot_be_created_twice(tmp_path: Path) -> None:
    ledger = tmp_path / "selected-ledger.json"
    environment = os.environ.copy()
    environment.update(
        {
            "PS51_WRAPPER": str(WRAPPER),
            "PS51_COMMON": str(COMMON),
            "PS51_PROTOCOL": str(PROTOCOL),
            "PS51_LEDGER": str(ledger),
        }
    )
    command = _function_loader(("Invoke-V5PS51AtomicJsonReplace",))
    command += _patched_common_loader()
    command += r"""
$null = New-V5ExclusiveLedger -Path $env:PS51_LEDGER -Payload ([ordered]@{
    status = "started_and_consumed"
})
Write-V5JsonAtomic -Path $env:PS51_LEDGER -Payload ([ordered]@{
    status = "complete_and_consumed"
})
$before = (Get-FileHash -LiteralPath $env:PS51_LEDGER -Algorithm SHA256).Hash
$rejected = $false
try {
    $null = New-V5ExclusiveLedger -Path $env:PS51_LEDGER -Payload ([ordered]@{
        status = "started_again"
    })
}
catch {
    if ($_.Exception.Message -notmatch "cannot be run again") { throw }
    $rejected = $true
}
if (-not $rejected) { throw "Second synthetic ledger creation was accepted." }
$after = (Get-FileHash -LiteralPath $env:PS51_LEDGER -Algorithm SHA256).Hash
if ($before -cne $after) { throw "Second creation mutated the consumed ledger." }
"second_rejected"
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "second_rejected" in completed.stdout
    assert json.loads(ledger.read_text(encoding="utf-8")) == {"status": "complete_and_consumed"}


def test_failed_ps51_replace_still_leaves_one_use_ledger_consumed(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "selected-ledger.json"
    environment = os.environ.copy()
    environment.update(
        {
            "PS51_WRAPPER": str(WRAPPER),
            "PS51_COMMON": str(COMMON),
            "PS51_PROTOCOL": str(PROTOCOL),
            "PS51_LEDGER": str(ledger),
        }
    )
    command = _function_loader(("Invoke-V5PS51AtomicJsonReplace",))
    command += _patched_common_loader()
    command += r"""
$null = New-V5ExclusiveLedger -Path $env:PS51_LEDGER -Payload ([ordered]@{
    status = "started_and_consumed"
})
$before = [IO.File]::ReadAllBytes($env:PS51_LEDGER)
$handle = [IO.File]::Open(
    $env:PS51_LEDGER,
    [IO.FileMode]::Open,
    [IO.FileAccess]::Read,
    [IO.FileShare]::None
)
$failed = $false
try {
    Write-V5JsonAtomic -Path $env:PS51_LEDGER -Payload ([ordered]@{
        status = "complete_and_consumed"
    })
}
catch {
    $failed = $true
}
finally {
    $handle.Dispose()
}
if (-not $failed) { throw "Locked synthetic replacement unexpectedly succeeded." }
$after = [IO.File]::ReadAllBytes($env:PS51_LEDGER)
if ([Convert]::ToBase64String($before) -cne [Convert]::ToBase64String($after)) {
    throw "Failed replacement changed the consumed ledger bytes."
}
try {
    $null = New-V5ExclusiveLedger -Path $env:PS51_LEDGER -Payload ([ordered]@{
        status = "started_again"
    })
    throw "Second synthetic ledger creation unexpectedly succeeded."
}
catch {
    if ($_.Exception.Message -notmatch "cannot be run again") { throw }
}
"failed_replace_consumed"
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "failed_replace_consumed" in completed.stdout
    assert json.loads(ledger.read_text(encoding="utf-8")) == {"status": "started_and_consumed"}


def test_mutated_bound_code_is_rejected_by_hash_helper(tmp_path: Path) -> None:
    mutated = tmp_path / "predict_joint.py"
    mutated.write_bytes(PREDICT.read_bytes() + b"\n# synthetic mutation\n")
    environment = os.environ.copy()
    environment.update(
        {
            "PS51_WRAPPER": str(WRAPPER),
            "PS51_MUTATED": str(mutated),
            "PS51_EXPECTED": _sha256(PREDICT),
        }
    )
    command = _function_loader(
        (
            "ConvertTo-V5PS51Sha256",
            "Get-V5PS51FileSha256",
            "Assert-V5PS51Hash",
        )
    )
    command += r"""
try {
    Assert-V5PS51Hash `
        (Get-V5PS51FileSha256 $env:PS51_MUTATED) `
        $env:PS51_EXPECTED `
        "Synthetic mutated prediction code"
    throw "Mutated code unexpectedly passed."
}
catch {
    if ($_.Exception.Message -notmatch "SHA-256 mismatch") { throw }
}
"mutation_rejected"
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "mutation_rejected" in completed.stdout


def test_compatibility_tests_never_launch_full_wrapper_or_inference() -> None:
    own_source = Path(__file__).read_bytes().decode("utf-8")
    forbidden = (
        '"-' + 'File"',
        "scripts\\" + "predict_joint.py",
        "nnUNetv2" + "_predict",
        "official_validation" + "_locked_v5",
    )
    for token in forbidden:
        assert token not in own_source
