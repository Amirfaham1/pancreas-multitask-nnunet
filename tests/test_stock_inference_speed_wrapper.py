from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Run-StockInferenceSpeedBenchmark.ps1"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def test_wrapper_uses_exact_fresh_process_abba_order() -> None:
    labels = (
        "stock_reference_1",
        "candidate_1",
        "candidate_2",
        "stock_reference_2",
    )
    positions = [SOURCE.index(f"Label = '{label}'") for label in labels]
    assert positions == sorted(positions)
    assert SOURCE.count("Arm = 'stock'") == 2
    assert SOURCE.count("Arm = 'candidate'") == 2
    assert "foreach ($run in $runOrder)" in SOURCE
    assert "'--execution-purpose', 'final_benchmark'" in SOURCE


def test_wrapper_claims_immutable_stock_ledger_before_any_child() -> None:
    assert "-Stage stock_speed" in SOURCE
    assert "status = 'started_and_consumed'" in SOURCE
    assert "stage = 'single_locked_stock_inference_speed_benchmark'" in SOURCE
    assert "New-V5ExclusiveLedger -Path $ledgerPath" in SOURCE
    assert SOURCE.index("New-V5ExclusiveLedger -Path $ledgerPath") < SOURCE.index(
        "& $PythonExecutable @childArguments"
    )
    assert "run_order = @($runOrder | ForEach-Object { $_.Label })" in SOURCE
    assert "benchmark_execution_id = $benchmarkExecutionId" in SOURCE
    assert "Remove-Item" not in SOURCE
    assert "Write-V5JsonAtomic" not in SOURCE


def test_wrapper_binds_final_candidate_and_distinct_delivery_ledgers() -> None:
    assert "$candidate = Assert-V5FinalCandidateLock @candidateLockArguments" in SOURCE
    for stage in ("stock_speed", "official_validation", "selected_test"):
        assert f"-Stage {stage}" in SOURCE
    assert "Stock-speed one-use ledger must be distinct" in SOURCE
    for argument in (
        "'--final-candidate-lock', $candidate.LockPath",
        "'--expected-final-candidate-lock-sha256', $candidate.LockSha256",
        "'--one-use-ledger', $ledgerPath",
        "'--benchmark-execution-id', $benchmarkExecutionId",
    ):
        assert argument in SOURCE


def test_wrapper_locks_all_stock_protocol_hashes_and_exact_72_cases() -> None:
    for digest in (
        "563d9d5e4fbe0f92653c6b7295c476d0ddf5d239c47beb1948410bbb80a7c2e2",
        "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd",
        "bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503",
    ):
        assert digest in SOURCE
    assert "$expectedCaseCount = 72" in SOURCE
    assert "[int] $ExpectedCaseCount" not in SOURCE
    assert "'--expected-case-count', [string]$expectedCaseCount" in SOURCE


def test_wrapper_invokes_auditor_with_retained_ordered_artifacts() -> None:
    assert "run_timed_inference_child.py" in SOURCE
    assert "benchmark_stock_inference_speed.py" in SOURCE
    for argument in (
        "'--external-runtime'",
        "'--output-directory'",
        "'--candidate-internal-runtime'",
        "'--output', $auditOutput",
    ):
        assert argument in SOURCE
    assert "Assert-V5Directory $outputDirectory" in SOURCE
    assert "$auditExitCode -notin @(0, 2)" in SOURCE
    assert "$auditExitCode -eq 2" in SOURCE


def test_wrapper_uses_mutex_and_never_reuses_work_root() -> None:
    assert "Enter-V5NamedMutex" in SOURCE
    assert "Exit-V5NamedMutex $mutex" in SOURCE
    assert "WorkRoot must not already exist" in SOURCE
    assert SOURCE.index("WorkRoot must not already exist") < SOURCE.index(
        "New-V5ExclusiveLedger -Path $ledgerPath"
    )
