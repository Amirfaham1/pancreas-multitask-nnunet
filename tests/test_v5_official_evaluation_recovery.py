from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "Run-V5OfficialEvaluationRecovery.ps1"
COMMON = ROOT / "scripts" / "V5-LockedDeliveryCommon.ps1"
ORIGINAL_WRAPPER = ROOT / "scripts" / "Run-V5LockedFinalEvaluation.ps1"
EVALUATOR = ROOT / "scripts" / "evaluate_predictions.py"
METRICS = ROOT / "src" / "pancreas_multitask" / "metrics.py"
PROTOCOL = ROOT / "configs" / "official_evaluation_recovery_protocol_v1.json"
SOURCE = WRAPPER.read_text(encoding="utf-8")
PROTOCOL_PAYLOAD = json.loads(PROTOCOL.read_text(encoding="utf-8"))

PROTOCOL_SHA256 = "a2e5ff8bffcf5a07ba797cabe1d245b00e4a352c06f9ffd4b8246c43d5646fdd"
FROZEN_OUTPUT_SHA256 = "e39e6abcc774a62adaa5ea1501416bba8cdb24ce5c569864b5cc20fc30ce3087"
FROZEN_ARTIFACT_SET_SHA256 = (
    "fec59a6b546d9158e6a32eb6be1d4f889b296a184b877dfc0e5baa323e180b28"
)
PATCHED_COMMON_SHA256 = "0e6ff47c0590f37f546e4937486ec16264dbe4212b70bb8202d7fc297bf6be98"
METRICS_SHA256 = "b357df75a8502972139dc45d78ea8c05683ff7d46cb3bb00f2fdc2b73118d5a9"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _powershell_executable() -> str | None:
    # The production incident is specific to Windows PowerShell 5.1, so prefer it.
    return shutil.which("powershell") or shutil.which("pwsh")


def _run_powershell(command: str, environment: dict[str, str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    executable = _powershell_executable()
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    return subprocess.run(
        [
            executable,
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
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:RECOVERY_WRAPPER, [ref] $tokens, [ref] $errors
)
if ($errors.Count -ne 0) {{ throw ($errors | Out-String) }}
$wanted = @({quoted})
$nodes = @($ast.FindAll({{
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $wanted -contains $node.Name
}}, $true))
if ($nodes.Count -ne $wanted.Count) {{
    throw "Could not isolate every requested recovery helper."
}}
$ordered = foreach ($name in $wanted) {{
    $nodes | Where-Object Name -ceq $name | Select-Object -First 1
}}
. ([ScriptBlock]::Create((($ordered | ForEach-Object Extent | ForEach-Object Text) -join "`n")))
"""


def test_recovery_source_parses_in_powershell() -> None:
    environment = os.environ.copy()
    environment["RECOVERY_WRAPPER"] = str(WRAPPER)
    completed = _run_powershell(
        r"""
$tokens = $null
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    $env:RECOVERY_WRAPPER, [ref] $tokens, [ref] $errors
) | Out-Null
if ($errors.Count -ne 0) {
    $errors | ForEach-Object { Write-Error $_.Message }
    exit 1
}
""",
        environment,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_protocol_and_original_evidence_hashes_are_exact() -> None:
    assert _sha256(PROTOCOL) == PROTOCOL_SHA256
    assert PROTOCOL_SHA256 in SOURCE
    assert "Hard-bound recovery protocol" in SOURCE
    assert PROTOCOL_PAYLOAD["status"] == (
        "frozen_before_recovery_implementation_or_any_official_reference_access"
    )
    bound = PROTOCOL_PAYLOAD["bound_original_implementation"]
    assert _sha256(COMMON) == bound["common_sha256"]
    assert _sha256(ORIGINAL_WRAPPER) == bound["official_wrapper_sha256"]
    assert _sha256(EVALUATOR) == bound["evaluator_sha256"]
    assert _sha256(METRICS) == METRICS_SHA256
    evidence = PROTOCOL_PAYLOAD["immutable_pre_recovery_evidence"]
    assert evidence["full_existing_output_file_count"] == 40
    assert evidence["full_existing_output_snapshot_sha256"] == FROZEN_OUTPUT_SHA256
    assert evidence["validated_inference_artifact_count"] == 39
    assert evidence["validated_inference_artifact_set_sha256"] == FROZEN_ARTIFACT_SET_SHA256


def test_execution_lock_must_bind_every_recovery_and_original_artifact() -> None:
    for field in (
        "ExpectedRecoveryProtocolSha256",
        "ExpectedRecoveryExecutionLockSha256",
        "ExpectedFinalCandidateLockSha256",
        "recovery_implementation_commit",
        "recovery_protocol_sha256",
        "recovery_wrapper_sha256",
        "recovery_tests_sha256",
        "final_candidate_lock_sha256",
        "original_common_sha256",
        "original_official_wrapper_sha256",
        "evaluator_sha256",
        "python_executable_path",
        "python_executable_sha256",
        "validation_images_path",
        "reference_masks_path",
        "reference_subtypes_path",
        "metrics_source_path",
        "metrics_source_sha256",
        "immutable_pre_recovery_evidence",
        "execution_contract",
    ):
        assert field in SOURCE
    assert "frozen_after_recovery_implementation_before_any_official_reference_access" in SOURCE
    assert "^[0-9a-f]{40}$" in SOURCE
    assert "Assert-RecoveryImplementationCommitPublished" in SOURCE
    assert "merge-base --is-ancestor $Commit origin/main" in SOURCE
    for relative_path in (
        "configs/official_evaluation_recovery_protocol_v1.json",
        "scripts/Run-V5OfficialEvaluationRecovery.ps1",
        "tests/test_v5_official_evaluation_recovery.py",
    ):
        assert relative_path in SOURCE
    for description in (
        "Recovery wrapper after manifest",
        "Evaluator after manifest",
        "Python after manifest",
        "Recovery wrapper after evaluation",
        "Evaluator after evaluation",
        "Python after evaluation",
        "Metrics source after manifest",
        "Metrics source after evaluation",
    ):
        assert description in SOURCE
    assert "Official imagesVal path" in SOURCE
    assert "Official labelsVal path" in SOURCE
    assert "Official classification-manifest path" in SOURCE
    assert "bidirectionally disjoint" in SOURCE
    assert METRICS_SHA256 in SOURCE
    assert 'diff --quiet $evaluatedImplementationCommit -- "src/pancreas_multitask/metrics.py"' in SOURCE


def test_exact_two_in_memory_substitutions_are_the_only_common_patch() -> None:
    substitutions = PROTOCOL_PAYLOAD["only_permitted_compatibility_substitutions"]
    assert len(substitutions) == 2
    source = COMMON.read_text(encoding="utf-8")
    assert len(source) == 81692
    for substitution in substitutions:
        assert substitution["occurrences_required"] == 1
        assert source.count(substitution["original"]) == 1
        source = source.replace(substitution["original"], substitution["replacement"])
    assert len(source) == 81698
    assert hashlib.sha256(source.encode()).hexdigest() == PATCHED_COMMON_SHA256
    assert "$substitutionCount -ne 2" in SOURCE
    assert ". ([ScriptBlock]::Create($patchedSource))" in SOURCE
    assert "On-disk Common after in-memory audit" in SOURCE


def test_no_inference_or_model_mutation_path_exists() -> None:
    forbidden = (
        "predict_joint.py",
        "nnUNetv2_predict",
        "Invoke-V5StrictCpuPreflight",
        "train_neural_case_heads.py",
        "select_checkpoint.py",
        "checkpoint_classification_rescue.pth --",
    )
    for token in forbidden:
        assert token not in SOURCE
    assert SOURCE.count("Invoke-V5CheckedPython") == 1
    assert SOURCE.count('-ScriptPath $evaluationScript') == 1
    assert "recovery_inference_invocation_count = 0" in SOURCE
    assert "total_official_inference_invocation_count = 1" in SOURCE


def test_hash_freeze_and_reference_access_order_is_fail_closed() -> None:
    consume = SOURCE.index("New-V5ExclusiveLedger -Path $recoveryLedgerPath")
    snapshot = SOURCE.index("Get-RecoveryRecursiveSnapshot -Root $resolvedOutputRoot", consume)
    artifact = SOURCE.index("$artifactSet = Get-V5InferenceArtifactSet", snapshot)
    ledger_snapshot = SOURCE.index("Copy-RecoveryFileBytesCreateNew", artifact)
    manifest = SOURCE.index(
        "Write-RecoveryJsonCreateNew -Path $preReferenceManifest", ledger_snapshot
    )
    manifest_hash = SOURCE.index(
        "$preReferenceManifestSha256 = Get-RecoveryFileSha256", manifest
    )
    rebind = SOURCE.index("$postManifestArtifactSet = Get-V5InferenceArtifactSet", manifest_hash)
    predictions_frozen_ledger = SOURCE.index(
        '$currentStage = "original_ledger_predictions_frozen"', rebind
    )
    start_record = SOURCE.index(
        "Write-RecoveryJsonCreateNew -Path $evaluationStartedRecord",
        predictions_frozen_ledger,
    )
    invocation_commit = SOURCE.index("$evaluationInvocationCommitted = $true", start_record)
    reference_masks = SOURCE.index(
        'Assert-V5Directory $ReferenceMasks "Official reference-mask directory"',
        start_record,
    )
    reference_subtypes = SOURCE.index(
        'Assert-V5LeafFile $ReferenceSubtypes "Official reference-subtype table"',
        reference_masks,
    )
    evaluation_started_ledger = SOURCE.index(
        '$currentStage = "original_ledger_single_evaluation_started"',
        reference_subtypes,
    )
    evaluator = SOURCE.index("Invoke-V5CheckedPython", reference_subtypes)
    post_evaluation = SOURCE.index("$postEvaluationArtifactSet = Get-V5InferenceArtifactSet", evaluator)
    gate = SOURCE.index("Write-RecoveryJsonCreateNew -Path $gateJson", post_evaluation)
    completed_ledger = SOURCE.index('$currentStage = "original_ledger_completion"', gate)
    assert (
        consume
        < snapshot
        < artifact
        < ledger_snapshot
        < manifest
        < manifest_hash
        < rebind
        < predictions_frozen_ledger
        < start_record
        < invocation_commit
        < reference_masks
        < reference_subtypes
        < evaluation_started_ledger
        < evaluator
        < post_evaluation
        < gate
        < completed_ledger
    )

    before_references = SOURCE[:reference_masks]
    for access in (
        "Test-Path -LiteralPath $ReferenceMasks",
        "Test-Path -LiteralPath $ReferenceSubtypes",
        "Get-Item -LiteralPath $ReferenceMasks",
        "Get-Item -LiteralPath $ReferenceSubtypes",
        "Get-Content -LiteralPath $ReferenceMasks",
        "Get-Content -LiteralPath $ReferenceSubtypes",
        "Get-ChildItem -LiteralPath $ReferenceMasks",
        "Get-ChildItem -LiteralPath $ReferenceSubtypes",
        "Resolve-Path $ReferenceMasks",
        "Resolve-Path $ReferenceSubtypes",
    ):
        assert access not in before_references


def test_standard_manifest_gate_and_ledger_schemas_are_preserved() -> None:
    for token in (
        'status = "all_v5_label_blind_predictions_hashed_before_this_wrapper_reference_access"',
        'method = "single_locked_post_hoc_official_validation_reevaluation"',
        'evaluation_scope = "single_locked_post_hoc_official_validation_reevaluation"',
        'status = "complete_no_second_classifier_iteration_permitted"',
        'status = "predictions_frozen_before_this_wrapper_reference_access"',
        'status = "single_evaluation_started"',
        'status = "complete_and_consumed"',
        'stage = "single_locked_post_hoc_official_validation_reevaluation"',
        "official_inference_invocation_count = 1",
        "official_evaluation_invocation_count = 1",
        "further_classifier_training_selection_or_official_evaluation_permitted = $false",
        "no_second_classifier_iteration_permitted = $true",
    ):
        assert token in SOURCE
    assert SOURCE.count("recovery = $recoveryProvenance") >= 6
    for threshold in (
        "baseline_macro_f1_strictly_greater_than = $script:V5BaselineMacroF1",
        "phd_macro_f1_at_least = 0.70",
        "whole_pancreas_dice_at_least = 0.91",
        "lesion_dice_at_least = 0.31",
    ):
        assert threshold in SOURCE
    evaluation_stage_start = SOURCE.index("$evaluationStartedLedgerPayload = [ordered]@{")
    evaluation_stage_end = SOURCE.index(
        "$currentOriginalLedgerSha256 = Write-RecoveryJsonAtomicWithBackup",
        evaluation_stage_start,
    )
    evaluation_stage = SOURCE[evaluation_stage_start:evaluation_stage_end]
    assert "reference_access_started = $true" in evaluation_stage
    assert "evaluation_invocation_count = 1" in evaluation_stage


def test_create_new_json_never_overwrites_synthetic_fixture(tmp_path: Path) -> None:
    destination = tmp_path / "evidence.json"
    environment = os.environ.copy()
    environment.update(
        {
            "RECOVERY_WRAPPER": str(WRAPPER),
            "RECOVERY_DESTINATION": str(destination),
        }
    )
    command = _function_loader(
        (
            "Write-RecoveryBytesCreateNew",
            "Write-RecoveryJsonCreateNew",
        )
    ) + r"""
$null = Write-RecoveryJsonCreateNew -Path $env:RECOVERY_DESTINATION -Payload ([ordered]@{ value = 1 })
try {
    $null = Write-RecoveryJsonCreateNew -Path $env:RECOVERY_DESTINATION -Payload ([ordered]@{ value = 2 })
    throw "Second CreateNew unexpectedly succeeded."
}
catch {
    if ($_.Exception.ToString() -notmatch "already exists") {
        throw
    }
    # Expected: PowerShell wraps the CreateNew IOException.
}
exit 0
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert json.loads(destination.read_text(encoding="utf-8")) == {"value": 1}


def test_synthetic_wrong_and_overlapping_reference_paths_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "recovery-output"
    dataset = tmp_path / "Dataset501"
    images = dataset / "imagesVal"
    labels = dataset / "labelsVal"
    manifest = dataset / "classification_manifest.json"
    environment = os.environ.copy()
    environment.update(
        {
            "RECOVERY_WRAPPER": str(WRAPPER),
            "RECOVERY_OUTPUT": str(output),
            "RECOVERY_IMAGES": str(images),
            "RECOVERY_LABELS": str(labels),
            "RECOVERY_MANIFEST": str(manifest),
        }
    )
    command = _function_loader(
        (
            "Get-RecoveryNormalizedPath",
            "Assert-RecoveryPathEqual",
            "Test-RecoveryPathAtOrBelow",
        )
    ) + r"""
Assert-RecoveryPathEqual $env:RECOVERY_IMAGES $env:RECOVERY_IMAGES "bound images"
try {
    Assert-RecoveryPathEqual $env:RECOVERY_LABELS $env:RECOVERY_MANIFEST "wrong ref type"
    throw "Wrong reference path unexpectedly passed."
}
catch {
    if ($_.Exception.Message -notmatch "prospectively bound path") { throw }
}
$selfReference = Join-Path $env:RECOVERY_OUTPUT "predictions"
if (-not (Test-RecoveryPathAtOrBelow -Candidate $selfReference -Parent $env:RECOVERY_OUTPUT)) {
    throw "Synthetic self-reference overlap was not detected."
}
exit 0
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_metrics_dependency_mutation_fails_hash_binding(tmp_path: Path) -> None:
    copied_metrics = tmp_path / "metrics.py"
    copied_metrics.write_bytes(METRICS.read_bytes())
    copied_metrics.write_bytes(copied_metrics.read_bytes() + b"\n# mutation\n")
    environment = os.environ.copy()
    environment.update(
        {
            "RECOVERY_WRAPPER": str(WRAPPER),
            "RECOVERY_MUTATED_METRICS": str(copied_metrics),
            "RECOVERY_EXPECTED_METRICS_SHA256": METRICS_SHA256,
        }
    )
    command = _function_loader(
        (
            "ConvertTo-RecoverySha256",
            "Get-RecoveryFileSha256",
            "Assert-RecoveryHash",
        )
    ) + r"""
try {
    Assert-RecoveryHash `
        (Get-RecoveryFileSha256 $env:RECOVERY_MUTATED_METRICS) `
        $env:RECOVERY_EXPECTED_METRICS_SHA256 `
        "mutated metrics"
    throw "Mutated metrics unexpectedly passed."
}
catch {
    if ($_.Exception.Message -notmatch "SHA-256 mismatch") { throw }
}
exit 0
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_ps5_atomic_replace_preserves_byte_identical_backup(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    backup = tmp_path / "ledger.original.backup.json"
    original_bytes = b'{"status":"started_and_consumed","spacing":"preserve me"}\n'
    ledger.write_bytes(original_bytes)
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()
    environment = os.environ.copy()
    environment.update(
        {
            "RECOVERY_WRAPPER": str(WRAPPER),
            "RECOVERY_LEDGER": str(ledger),
            "RECOVERY_BACKUP": str(backup),
            "RECOVERY_ORIGINAL_SHA256": original_sha256,
        }
    )
    command = _function_loader(
        (
            "ConvertTo-RecoverySha256",
            "Get-RecoveryFileSha256",
            "Assert-RecoveryHash",
            "Write-RecoveryBytesCreateNew",
            "Write-RecoveryJsonAtomicWithBackup",
        )
    ) + r"""
$null = Write-RecoveryJsonAtomicWithBackup `
    -Path $env:RECOVERY_LEDGER `
    -Payload ([ordered]@{ status = "complete_and_consumed" }) `
    -BackupPath $env:RECOVERY_BACKUP `
    -ExpectedPreviousSha256 $env:RECOVERY_ORIGINAL_SHA256
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert backup.read_bytes() == original_bytes
    assert json.loads(ledger.read_text(encoding="utf-8")) == {
        "status": "complete_and_consumed"
    }
    assert "[IO.File]::Replace($temporary, $destination, $backup)" in SOURCE
    assert "[IO.File]::Replace($temporary, $destination, $null)" not in SOURCE


def test_exact_byte_snapshot_rehashes_source_after_copy() -> None:
    helper_start = SOURCE.index("function Copy-RecoveryFileBytesCreateNew")
    helper_end = SOURCE.index("function Write-RecoveryJsonAtomicWithBackup", helper_start)
    helper = SOURCE[helper_start:helper_end]
    assert "[IO.File]::ReadAllBytes" in helper
    assert "Byte-identical ledger snapshot" in helper
    assert "Snapshot source after copy" in helper


def test_recursive_snapshot_matches_independent_synthetic_digest(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    (root / "nested").mkdir(parents=True)
    (root / "alpha.bin").write_bytes(b"alpha")
    (root / "nested" / "beta.bin").write_bytes(b"beta\x00gamma")

    rows: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            rows.append((relative, path.stat().st_size, _sha256(path)))
    canonical = "".join(f"{name}\0{size}\0{digest}\n" for name, size, digest in rows)
    expected = hashlib.sha256(canonical.encode()).hexdigest()

    environment = os.environ.copy()
    environment.update(
        {
            "RECOVERY_WRAPPER": str(WRAPPER),
            "RECOVERY_SNAPSHOT_ROOT": str(root),
            "RECOVERY_EXPECTED_SNAPSHOT": expected,
        }
    )
    command = _function_loader(
        (
            "Get-RecoveryFileSha256",
            "Get-RecoveryStringSha256",
            "Get-RecoveryRecursiveSnapshot",
        )
    ) + r"""
$snapshot = Get-RecoveryRecursiveSnapshot -Root $env:RECOVERY_SNAPSHOT_ROOT
if ($snapshot.FileCount -ne 2) { throw "Unexpected synthetic snapshot count." }
if ($snapshot.SnapshotSha256 -cne $env:RECOVERY_EXPECTED_SNAPSHOT) {
    throw "Synthetic snapshot digest mismatch."
}
"""
    completed = _run_powershell(command, environment)
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_recovery_outputs_are_create_new_and_failure_stays_consumed() -> None:
    for variable in (
        "$preReferenceManifest",
        "$evaluationStartedRecord",
        "$gateJson",
        "$completionRecord",
        "$originalLedgerSnapshot",
    ):
        assert f"-Path {variable}" in SOURCE or f"-Destination {variable}" in SOURCE
    assert 'status = "started_and_consumed_no_recovery_rerun"' in SOURCE
    assert 'status = "failed_and_consumed_no_recovery_rerun"' in SOURCE
    assert 'status = "saved_output_recovery_complete_no_rerun_permitted"' in SOURCE
    assert "one_use_recovery_remains_consumed = $true" in SOURCE
    assert "official_evaluation_invocation_commitment_count = 1" in SOURCE
    assert "official_evaluation_process_invocation_count_before_this_record = 0" in SOURCE
    reference_access = SOURCE.index(
        'Assert-V5Directory $ReferenceMasks "Official reference-mask directory"'
    )
    process_count = SOURCE.index("$evaluationInvocationCount = 1", reference_access)
    evaluator = SOURCE.index("Invoke-V5CheckedPython", process_count)
    assert reference_access < process_count < evaluator


def test_case_level_predictions_are_not_printed_before_evaluation() -> None:
    evaluator = SOURCE.index("Invoke-V5CheckedPython")
    prefix = SOURCE[:evaluator]
    for forbidden in (
        "Write-Host $classificationRows",
        "Write-Host $probabilityRows",
        "Write-Output $classificationRows",
        "Write-Output $probabilityRows",
        "Format-Table $classificationRows",
        "Format-Table $probabilityRows",
    ):
        assert forbidden not in prefix
