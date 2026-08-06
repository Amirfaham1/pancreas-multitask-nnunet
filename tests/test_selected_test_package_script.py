"""Static safety contracts for selected test inference and packaging."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "Run-SelectedTestAndPackage.ps1"
SOURCE = SCRIPT.read_text(encoding="utf-8")
CLASSIFICATION_PARAMETER_NAMES = [
    "classification_pool.query",
    "classification_pool.token_norm.weight",
    "classification_pool.token_norm.bias",
    "classification_pool.attention.in_proj_weight",
    "classification_pool.attention.in_proj_bias",
    "classification_pool.attention.out_proj.weight",
    "classification_pool.attention.out_proj.bias",
    "classification_pool.output_norm.weight",
    "classification_pool.output_norm.bias",
    "classification_head.0.weight",
    "classification_head.0.bias",
    "classification_head.1.weight",
    "classification_head.1.bias",
    "classification_head.4.weight",
    "classification_head.4.bias",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_positive_provenance_fixture(tmp_path: Path) -> dict[str, object]:
    work_root = tmp_path / "work"
    dataset = work_root / "nnUNet_raw" / "Dataset501_PancreasMultitask"
    preprocessed_dataset = work_root / "nnUNet_preprocessed" / "Dataset501_PancreasMultitask"
    prepared = dataset / "imagesTs"
    source = tmp_path / "source_test"
    model = (
        work_root
        / "nnUNet_results"
        / "Dataset501_PancreasMultitask"
        / "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres"
    )
    fold = model / "fold_0"
    selection_path = work_root / "evaluation" / "fixed_validation" / "checkpoint_selection.json"
    for directory in (
        prepared,
        preprocessed_dataset,
        source,
        fold,
        selection_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(72):
        name = f"quiz_{index:03d}_0000.nii.gz"
        (prepared / name).write_bytes(b"prepared")
        (source / name).write_bytes(b"source")

    candidates = {
        "checkpoint_best": "checkpoint_best.pth",
        "checkpoint_best_multitask": "checkpoint_best_multitask.pth",
        "checkpoint_final": "checkpoint_final.pth",
        "checkpoint_classification_rescue": "checkpoint_classification_rescue.pth",
    }
    checkpoints: dict[str, Path] = {}
    for index, (candidate, filename) in enumerate(candidates.items()):
        checkpoint = fold / filename
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        checkpoints[candidate] = checkpoint

    final_sha = _sha256(checkpoints["checkpoint_final"])
    activation_path = fold / "classification_rescue_activation.json"
    activation = {
        "schema_version": 1,
        "source_checkpoint_name": "checkpoint_final.pth",
        "source_checkpoint_sha256": final_sha,
        "checkpoint_current_epoch": 200,
        "training_logging_epoch_count": 200,
        "metric_scope": "checkpoint_training_logging_only",
        "validation_metrics_read": False,
        "validation_used_for_activation": False,
        "activation_approved": True,
        "decision_epoch": 40,
    }
    activation_path.write_text(json.dumps(activation), encoding="utf-8")

    raw_split_manifest = dataset / "split_manifest.json"
    preprocessed_split_manifest = preprocessed_dataset / "split_manifest.json"
    split_manifest_bytes = b'{"schema_version":1,"source_splits_preserved":true}\n'
    raw_split_manifest.write_bytes(split_manifest_bytes)
    preprocessed_split_manifest.write_bytes(split_manifest_bytes)
    split_manifest_sha256 = _sha256(raw_split_manifest)

    rescue_path = checkpoints["checkpoint_classification_rescue"]
    rescue_audit_path = Path(f"{rescue_path}.audit.json")
    rescue_audit = {
        "schema_version": 1,
        "method": "post_training_frozen_backbone_classification_head_rescue",
        "status": "complete",
        "completed_epochs": 30,
        "schedule": {
            "epochs": 30,
            "iterations_per_epoch": 125,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "label_smoothing": 0.05,
            "nonlesion_patch_weight": 0.25,
            "reset_seed": 20260806,
        },
        "optimizer": "AdamW",
        "device_type": "cuda",
        "training_loader": "single_threaded_training_split_only",
        "training_batch_size": 2,
        "maximum_attempts": 1,
        "training_updates_expected": 3750,
        "wandb_enabled": False,
        "early_stopping": False,
        "decoder_executed_during_rescue": False,
        "encoder_gradient_enabled": False,
        "decoder_gradient_enabled": False,
        "validation_labels_indexed_for_targets": False,
        "classification_parameter_names": CLASSIFICATION_PARAMETER_NAMES,
        "classification_trainable_parameter_count": 496_195,
        "source_checkpoint_sha256": final_sha,
        "activation_audit_sha256": _sha256(activation_path),
        "output_checkpoint_sha256": _sha256(rescue_path),
        "activation_decision_epoch": 40,
        "source_component_sha256": {
            "encoder": "1" * 64,
            "decoder": "2" * 64,
            "classification": "3" * 64,
        },
        "current_component_sha256": {
            "encoder": "1" * 64,
            "decoder": "2" * 64,
            "classification": "4" * 64,
        },
        "split_audit": {
            "split_disjoint": True,
            "training_case_count": 252,
            "validation_case_count": 36,
            "validation_images_opened": False,
            "validation_used_for_gradients": False,
            "validation_used_for_stopping": False,
            "validation_batches_consumed": 0,
            "frozen_split_manifest_schema_version": 1,
            "frozen_source_splits_preserved": True,
            "matches_frozen_split_manifest": True,
            "frozen_manifest_training_case_count": 252,
            "frozen_manifest_validation_case_count": 36,
            "frozen_split_manifest": str(preprocessed_split_manifest.resolve()),
            "frozen_split_manifest_sha256": split_manifest_sha256,
            "training_case_ids_sha256": "5" * 64,
            "frozen_manifest_training_case_ids_sha256": "5" * 64,
            "validation_case_ids_sha256": "6" * 64,
            "frozen_manifest_validation_case_ids_sha256": "6" * 64,
        },
    }
    rescue_audit_path.write_text(json.dumps(rescue_audit), encoding="utf-8")

    ordered_candidates = [
        "checkpoint_best",
        "checkpoint_best_multitask",
        "checkpoint_classification_rescue",
        "checkpoint_final",
    ]
    ranking: list[dict[str, object]] = []
    for rank, candidate in enumerate(ordered_candidates, start=1):
        metric = 1.0 - rank / 10.0
        ranking.append(
            {
                "candidate": candidate,
                "rank": rank,
                "metrics": {
                    "whole_pancreas_dice": metric,
                    "lesion_dice": metric,
                    "macro_f1": metric,
                },
                "selection_score": metric,
                "metrics_source": str(tmp_path / f"{candidate}.metrics.json"),
                "checkpoint_path": str(checkpoints[candidate]),
                "checkpoint_sha256": _sha256(checkpoints[candidate]),
            }
        )
    selection = {
        "schema_version": 1,
        "selection_policy": {
            "direction": "maximize",
            "metric_paths": [
                "segmentation.whole_pancreas_dice.mean",
                "segmentation.lesion_dice.mean",
                "classification.macro_f1",
            ],
            "metric_weights": {
                "whole_pancreas_dice": 1 / 3,
                "lesion_dice": 1 / 3,
                "macro_f1": 1 / 3,
            },
            "score": "equal-weight arithmetic mean",
            "tie_breaker": "candidate name ascending; no secondary metric",
        },
        "candidate_count": 4,
        "selected_candidate": "checkpoint_best",
        "selected_score": ranking[0]["selection_score"],
        "selected_checkpoint_path": ranking[0]["checkpoint_path"],
        "selected_checkpoint_sha256": ranking[0]["checkpoint_sha256"],
        "ranking": ranking,
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        '@echo off\r\n>>"%SELECTED_TEST_CHECKPOINT_TO_MUTATE%" echo changed\r\nexit /b 0\r\n',
        encoding="utf-8",
    )
    return {
        "work_root": work_root,
        "source": source,
        "selection_path": selection_path,
        "activation_path": activation_path,
        "rescue_audit_path": rescue_audit_path,
        "preprocessed_split_manifest": preprocessed_split_manifest,
        "selected_checkpoint": checkpoints["checkpoint_best"],
        "fake_python": fake_python,
        "prediction": work_root / "submission" / "results",
        "evidence": work_root / "evaluation" / "selected_test",
        "delivery": tmp_path / "delivery",
    }


def _wrapper_command(executable: str, fixture: dict[str, object]) -> list[str]:
    return [
        executable,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-WorkRoot",
        str(fixture["work_root"]),
        "-SourceTestImages",
        str(fixture["source"]),
        "-PythonExecutable",
        str(fixture["fake_python"]),
        "-PredictionDirectory",
        str(fixture["prediction"]),
        "-EvidenceDirectory",
        str(fixture["evidence"]),
        "-DeliveryRoot",
        str(fixture["delivery"]),
        "-Device",
        "cpu",
    ]


def test_selected_test_wrapper_parses_without_running_pipeline() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable; remaining static checks still run")
    parser_command = r"""
& {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $env:SELECTED_TEST_SCRIPT_TO_PARSE, [ref] $tokens, [ref] $errors
    ) | Out-Null
    if ($errors.Count -gt 0) {
        $errors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }
        exit 1
    }
}
"""
    environment = os.environ.copy()
    environment["SELECTED_TEST_SCRIPT_TO_PARSE"] = str(SCRIPT)
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


def test_wrapper_holds_shared_process_lifetime_mutex_across_pipeline() -> None:
    assert '"Local\\PancreasMultitaskPostTraining501Fold0"' in SOURCE
    assert "$postTrainingMutex.WaitOne(0)" in SOURCE
    assert "catch [Threading.AbandonedMutexException]" in SOURCE
    assert "$postTrainingMutex.ReleaseMutex()" in SOURCE
    assert "$postTrainingMutex.Dispose()" in SOURCE
    assert "Start-Process" not in SOURCE

    acquire = SOURCE.index("$postTrainingMutex.WaitOne(0)")
    inference = SOURCE.index('-Stage "Selected-checkpoint test inference"')
    package = SOURCE.index("& $packageScript @packageArguments", inference)
    release = SOURCE.index("$postTrainingMutex.ReleaseMutex()", package)
    assert acquire < inference < package < release


def test_selection_branch_location_and_sha256_are_verified_before_outputs() -> None:
    for required_contract in (
        '"classification_rescue_activation.json"',
        '"activation_approved"',
        '"candidate_count"',
        '"ranking"',
        '"selected_candidate"',
        '"selected_checkpoint_path"',
        '"selected_checkpoint_sha256"',
        "Get-FileHash -LiteralPath $Path -Algorithm SHA256",
        "Selected checkpoint must be a direct child",
        "checkpoint_classification_rescue.pth",
    ):
        assert required_contract in SOURCE

    selection_hash = SOURCE.index('-Description "Selection artifact checkpoint SHA-256"')
    first_output = SOURCE.index("New-Item -ItemType Directory -Path $resolvedEvidenceDirectory")
    assert selection_hash < first_output


def test_full_rescue_protocol_and_selection_hash_chain_are_fail_closed() -> None:
    for required_contract in (
        "post_training_frozen_backbone_classification_head_rescue",
        "iterations_per_epoch",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "label_smoothing",
        "nonlesion_patch_weight",
        "reset_seed",
        "training_updates_expected",
        "source_component_sha256",
        "current_component_sha256",
        "classification_parameter_names",
        "classification_trainable_parameter_count",
        "frozen_split_manifest_sha256",
        "frozen_manifest_training_case_ids_sha256",
        "frozen_manifest_validation_case_ids_sha256",
        "matches_frozen_split_manifest",
        "Activation-to-selection checkpoint_final SHA-256",
        "Rescue-source-to-selection checkpoint_final SHA-256",
        "Rescue-output-to-selection checkpoint SHA-256",
    ):
        assert required_contract in SOURCE

    protocol_gate = SOURCE.index("$rescueSchedule = Get-RequiredJsonProperty")
    component_gate = SOURCE.index("$sourceComponentHashes = Get-RequiredJsonProperty")
    split_gate = SOURCE.index("$rawSplitManifestSha256 = Get-FileSha256")
    ranking_gate = SOURCE.index("$rankingCheckpointSha256ByCandidate = @{}")
    transitive_gate = SOURCE.index('-Description "Rescue-output-to-selection checkpoint SHA-256"')
    first_output = SOURCE.index("New-Item -ItemType Directory -Path $resolvedEvidenceDirectory")
    assert (
        protocol_gate < component_gate < split_gate < ranking_gate < transitive_gate < first_output
    )


@pytest.mark.skipif(os.name != "nt", reason="The production wrapper targets Windows")
@pytest.mark.parametrize(
    ("field_path", "bad_value", "expected_message"),
    [
        (
            ("method",),
            "not_the_frozen_rescue",
            "method does not identify the frozen-head protocol",
        ),
        (
            ("schedule", "iterations_per_epoch"),
            124,
            "exactly 125 iterations per epoch",
        ),
        (
            ("classification_trainable_parameter_count",),
            1,
            "count must be 496,195",
        ),
        (
            ("current_component_sha256", "decoder"),
            "7" * 64,
            "Frozen decoder component SHA-256",
        ),
    ],
)
def test_wrapper_executes_frozen_rescue_protocol_gates(
    tmp_path: Path,
    field_path: tuple[str, ...],
    bad_value: object,
    expected_message: str,
) -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    fixture = _write_positive_provenance_fixture(tmp_path)
    rescue_path = Path(fixture["rescue_audit_path"])
    rescue = json.loads(rescue_path.read_text(encoding="utf-8"))
    target = rescue
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = bad_value
    rescue_path.write_text(json.dumps(rescue), encoding="utf-8")

    completed = subprocess.run(
        _wrapper_command(executable, fixture),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert expected_message in completed.stderr + completed.stdout
    assert not Path(fixture["evidence"]).exists()


@pytest.mark.skipif(os.name != "nt", reason="The production wrapper targets Windows")
def test_wrapper_executes_exact_split_manifest_hash_gate(tmp_path: Path) -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    fixture = _write_positive_provenance_fixture(tmp_path)
    Path(fixture["preprocessed_split_manifest"]).write_bytes(b"changed")
    completed = subprocess.run(
        _wrapper_command(executable, fixture),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "Raw/preprocessed frozen split-manifest SHA-256" in (completed.stderr + completed.stdout)
    assert not Path(fixture["evidence"]).exists()


@pytest.mark.skipif(os.name != "nt", reason="The production wrapper targets Windows")
def test_wrapper_executes_activation_and_rescue_zero_validation_gates(tmp_path: Path) -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    fixture = _write_positive_provenance_fixture(tmp_path)
    rescue_path = Path(fixture["rescue_audit_path"])
    rescue = json.loads(rescue_path.read_text(encoding="utf-8"))
    rescue["split_audit"]["validation_batches_consumed"] = 1
    rescue_path.write_text(json.dumps(rescue), encoding="utf-8")
    completed = subprocess.run(
        _wrapper_command(executable, fixture),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "zero validation batches consumed" in completed.stderr + completed.stdout
    assert not Path(fixture["prediction"]).exists()
    assert not Path(fixture["evidence"]).exists()
    assert not Path(fixture["delivery"]).exists()


@pytest.mark.skipif(os.name != "nt", reason="The production wrapper targets Windows")
def test_wrapper_executes_all_candidate_checkpoint_hash_gate(tmp_path: Path) -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    fixture = _write_positive_provenance_fixture(tmp_path)
    selection_path = Path(fixture["selection_path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["ranking"][1]["checkpoint_sha256"] = "0" * 64
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    completed = subprocess.run(
        _wrapper_command(executable, fixture),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "Selection ranking checkpoint SHA-256" in completed.stderr + completed.stdout
    assert not Path(fixture["evidence"]).exists()


@pytest.mark.skipif(os.name != "nt", reason="The production wrapper targets Windows")
def test_wrapper_rehashes_checkpoint_after_predictor_returns(tmp_path: Path) -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable")

    fixture = _write_positive_provenance_fixture(tmp_path)
    environment = os.environ.copy()
    environment["SELECTED_TEST_CHECKPOINT_TO_MUTATE"] = str(fixture["selected_checkpoint"])
    completed = subprocess.run(
        _wrapper_command(executable, fixture),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode != 0
    assert "Post-inference selected checkpoint SHA-256" in (completed.stderr + completed.stdout)
    assert Path(fixture["evidence"]).is_dir()
    assert not Path(fixture["delivery"]).exists()


def test_evidence_is_disjoint_and_never_enters_prediction_directory() -> None:
    assert (
        '$probabilityCsv = Join-Path $resolvedEvidenceDirectory "subtype_probabilities.csv"'
    ) in SOURCE
    assert '$runtimeJson = Join-Path $resolvedEvidenceDirectory "runtime.json"' in SOURCE
    assert (
        "$classificationCsv = Join-Path $resolvedPredictionDirectory $classificationCsvName"
    ) in SOURCE
    assert ('-Description "Prediction and evidence directories"') in SOURCE
    assert '"--probability-csv", $probabilityCsv' in SOURCE
    assert '"--runtime-json", $runtimeJson' in SOURCE
    assert '"--classification-csv", $classificationCsv' in SOURCE

    package = SOURCE.index("& $packageScript @packageArguments")
    independent_validation = SOURCE.index(
        '-Stage "Untouched-source submission validation"', package
    )
    assert package < independent_validation
    assert '"--test-images", $resolvedSourceTestImages' in SOURCE


def test_first_run_is_fresh_and_inference_is_exactly_one_overwrite_pass() -> None:
    guard_prediction = SOURCE.index(
        '-Path $resolvedPredictionDirectory `\n        -Description "PredictionDirectory"'
    )
    guard_evidence = SOURCE.index(
        '-Path $resolvedEvidenceDirectory `\n        -Description "EvidenceDirectory"'
    )
    guard_delivery = SOURCE.index(
        '-Path $resolvedDeliveryRoot `\n        -Description "DeliveryRoot"'
    )
    first_output = SOURCE.index("New-Item -ItemType Directory -Path $resolvedEvidenceDirectory")
    assert guard_prediction < guard_evidence < guard_delivery < first_output

    assert SOURCE.count('"--overwrite"') == 1
    assert '"--no-overwrite"' not in SOURCE
    assert SOURCE.count('-Stage "Selected-checkpoint test inference"') == 1
    assert "Preserve and inspect existing work instead of overwriting it" in SOURCE
    assert "If that message was printed, inference is complete" in SOURCE
    assert "Never pass -Force or delete the failed-run directories" in SOURCE


def test_wandb_runtime_packaging_and_hash_checks_are_strictly_serial() -> None:
    assert "-WandbMode disabled" in SOURCE
    assert '$env:WANDB_MODE = "disabled"' in SOURCE
    assert '$env:WANDB_DISABLED = "true"' in SOURCE
    assert '$env:nnUNet_wandb_enabled -ne "0"' in SOURCE

    inference = SOURCE.index('-Stage "Selected-checkpoint test inference"')
    runtime = SOURCE.index("$runtime = Read-JsonObject", inference)
    runtime_gate = SOURCE.index(
        'throw "Runtime artifact must record finite positive timing values."', runtime
    )
    package = SOURCE.index("& $packageScript @packageArguments", runtime_gate)
    independent_validation = SOURCE.index(
        '-Stage "Untouched-source submission validation"', package
    )
    package_hash = SOURCE.index(
        '-Description "Package manifest archive SHA-256"', independent_validation
    )
    assert inference < runtime < runtime_gate < package < independent_validation < package_hash

    package_block = SOURCE[SOURCE.index("$packageArguments = @{", runtime_gate) : package]
    assert "Force" not in package_block
    assert '"--expected-count", [string] $expectedCount' in SOURCE
    assert "Assert-SubmissionAudit" in SOURCE


def test_wrapper_stops_before_report_build_or_upload_staging() -> None:
    assert "build_report.ps1" not in SOURCE
    assert "final_upload" not in SOURCE
    assert "Copy-Item" not in SOURCE
    for returned_path in (
        "SelectionArtifact",
        "PredictionDirectory",
        "EvidenceDirectory",
        "DeliveryRoot",
        "RuntimeJson",
        "ProbabilityCsv",
        "Archive",
        "PackageManifest",
        "ArchiveValidation",
        "SourceTestValidation",
    ):
        assert f"{returned_path} =" in SOURCE
