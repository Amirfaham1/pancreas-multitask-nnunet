from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_final_evidence.py"
SPEC = importlib.util.spec_from_file_location("summarize_final_evidence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
EVALUATOR_METRICS = MODULE.EVALUATOR_METRICS


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float], *, seed: int) -> dict[str, Any]:
    return EVALUATOR_METRICS.summarize_values(
        values,
        bootstrap_samples=2000,
        confidence=0.95,
        seed=seed,
    )


def _classification(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result = EVALUATOR_METRICS.classification_metrics(
        [row["reference_subtype"] for row in cases],
        [row["predicted_subtype"] for row in cases],
        bootstrap_samples=2000,
        confidence=0.95,
        seed=12345,
    )
    result["unused_reference_case_count"] = 0
    return result


def _activation_window(audit_epoch: int, *, hard: bool) -> dict[str, Any]:
    losses = [1.1] * 10
    accuracies = [0.3] * 10
    mean_loss = float(np.mean(losses))
    mean_accuracy = float(np.mean(accuracies))
    slope = float(np.polyfit(np.arange(10, dtype=np.float64), losses, 1)[0])
    if hard:
        conditions = {
            "ce_above_1_03": mean_loss > 1.03,
            "accuracy_below_0_45": mean_accuracy < 0.45,
        }
        triggered = any(conditions.values())
    else:
        conditions = {
            "ce_at_least_1_05": mean_loss >= 1.05,
            "accuracy_at_most_0_42": mean_accuracy <= 0.42,
            "ce_slope_at_least_negative_0_001": slope >= -0.001,
        }
        triggered = all(conditions.values())
    return {
        "audit_epoch": audit_epoch,
        "window_epochs": list(range(audit_epoch - 9, audit_epoch + 1)),
        "training_classification_ce_values": losses,
        "training_patch_accuracy_values": accuracies,
        "mean_training_classification_ce": mean_loss,
        "mean_training_patch_accuracy": mean_accuracy,
        "classification_ce_ols_slope_per_epoch": slope,
        "conditions": conditions,
        "triggered": triggered,
    }


def _inactive_activation_window(audit_epoch: int, *, hard: bool) -> dict[str, Any]:
    window = _activation_window(audit_epoch, hard=hard)
    losses = [0.8] * 10
    accuracies = [0.8] * 10
    slope = float(np.polyfit(np.arange(10, dtype=np.float64), losses, 1)[0])
    window["training_classification_ce_values"] = losses
    window["training_patch_accuracy_values"] = accuracies
    window["mean_training_classification_ce"] = float(np.mean(losses))
    window["mean_training_patch_accuracy"] = float(np.mean(accuracies))
    window["classification_ce_ols_slope_per_epoch"] = slope
    if hard:
        window["conditions"] = {
            "ce_above_1_03": False,
            "accuracy_below_0_45": False,
        }
    else:
        window["conditions"] = {
            "ce_at_least_1_05": False,
            "accuracy_at_most_0_42": False,
            "ce_slope_at_least_negative_0_001": slope >= -0.001,
        }
    window["triggered"] = False
    return window


def _write_case_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MODULE.EXPECTED_CASE_COLUMNS))
        writer.writeheader()
        writer.writerows(cases)


def _build_bundle(tmp_path: Path) -> dict[str, Path]:
    cases: list[dict[str, Any]] = []
    for index in range(36):
        subtype = 0 if index < 9 else (1 if index < 24 else 2)
        lesion_dice = 0.0 if index < 2 else (index - 1) / 35
        lesion_predicted = 0 if index == 0 else (50 if index == 1 else 100 * (index + 1))
        cases.append(
            {
                "case_id": f"quiz_{index:03d}",
                "whole_pancreas_dice": 0.7 + index / 200,
                "lesion_dice": lesion_dice,
                "reference_subtype": subtype,
                "predicted_subtype": subtype,
                "classification_correct": True,
                "whole_pancreas_predicted_voxels": 9000 + index,
                "whole_pancreas_reference_voxels": 10000 + index,
                "lesion_predicted_voxels": lesion_predicted,
                "lesion_reference_voxels": 100 * (index + 1),
                "whole_pancreas_empty_empty": False,
                "lesion_empty_empty": False,
            }
        )

    classification = _classification(cases)
    metrics = {
        "schema_version": 1,
        "case_count": 36,
        "evaluation_policy": {
            "whole_pancreas": "label > 0",
            "lesion": "label == 2",
            "empty_empty_dice": 1.0,
            "one_sided_empty_dice": 0.0,
            "classification_labels": [0, 1, 2],
            "classification_zero_division": 0.0,
            "confusion_matrix_rows": "reference",
            "confusion_matrix_columns": "prediction",
            "aggregation": "unweighted case mean",
            "bootstrap_seed": 12345,
        },
        "segmentation": {
            "case_count": 36,
            "whole_pancreas_dice": _summary(
                [row["whole_pancreas_dice"] for row in cases], seed=12345
            ),
            "lesion_dice": _summary([row["lesion_dice"] for row in cases], seed=12346),
            "empty_cases": {
                "whole_pancreas_prediction_empty": 0,
                "whole_pancreas_reference_empty": 0,
                "whole_pancreas_both_empty": 0,
                "lesion_prediction_empty": 1,
                "lesion_reference_empty": 0,
                "lesion_both_empty": 0,
            },
        },
        "classification": classification,
        "cases": cases,
    }
    metrics_path = tmp_path / "checkpoint_classification_rescue" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    _write_json(metrics_path, metrics)
    case_metrics_path = metrics_path.with_name("case_metrics.csv")
    _write_case_csv(case_metrics_path, cases)

    checkpoints = {}
    for candidate in (
        "checkpoint_best",
        "checkpoint_best_multitask",
        "checkpoint_final",
        "checkpoint_classification_rescue",
    ):
        checkpoint = tmp_path / "fold_0" / f"{candidate}.pth"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(f"synthetic-{candidate}".encode())
        checkpoints[candidate] = checkpoint

    activation = {
        "schema_version": 1,
        "source_checkpoint": str(checkpoints["checkpoint_final"].resolve()),
        "source_checkpoint_name": "checkpoint_final.pth",
        "source_checkpoint_sha256": _sha256(checkpoints["checkpoint_final"]),
        "checkpoint_current_epoch": 200,
        "training_logging_epoch_count": 200,
        "metric_scope": "checkpoint_training_logging_only",
        "validation_metrics_read": False,
        "validation_used_for_activation": False,
        "activation_approved": True,
        "decision_epoch": 40,
        "epoch_40": _activation_window(40, hard=False),
        "epoch_50_hard_audit": _activation_window(50, hard=True),
    }
    activation_path = tmp_path / "fold_0" / "classification_rescue_activation.json"
    _write_json(activation_path, activation)

    recovery_evidence = tmp_path / "fold_0" / "classification_rescue_recovery_evidence"
    recovery_evidence.mkdir()
    failed_stdout = recovery_evidence / "failed_launch.stdout.log"
    failed_stderr = recovery_evidence / "failed_launch.stderr.log"
    failed_stdout.write_text("synthetic failed rescue stdout\n", encoding="utf-8")
    failed_stderr.write_text("synthetic failed rescue stderr\n", encoding="utf-8")
    recovery = {
        "schema_version": 1,
        "event": "classification_rescue_zero_update_execution_recovery",
        "status": "authorized_before_custom_joint_fixed_validation",
        "source_checkpoint_name": "checkpoint_final.pth",
        "source_checkpoint_sha256": _sha256(checkpoints["checkpoint_final"]),
        "activation_audit_sha256": _sha256(activation_path),
        "activation_approved": True,
        "failed_launch": {
            "process_launch_index": 1,
            "failed_step_index": 0,
            "training_batches_consumed": 1,
            "training_samples_consumed": 2,
            "finite_loss_guard_passed": True,
            "failure_stage": "after_grad_scaler_unscale_before_gradient_clip_completion",
            "exception_type": "RuntimeError",
            "optimizer_step_reached": False,
            "optimizer_updates": 0,
            "completed_epochs": 0,
            "checkpoint_written": False,
            "rescue_audit_written": False,
            "first_step_zero_update_operator_attested": True,
            "stdout_artifact": {
                "name": "classification_rescue_recovery_evidence/failed_launch.stdout.log",
                "bytes": failed_stdout.stat().st_size,
                "sha256": _sha256(failed_stdout),
            },
            "stderr_artifact": {
                "name": "classification_rescue_recovery_evidence/failed_launch.stderr.log",
                "bytes": failed_stderr.stat().st_size,
                "sha256": _sha256(failed_stderr),
            },
        },
        "validation": {
            "stock_nnunet_segmentation_only_validation_completed": True,
            "stock_nnunet_validation_metrics_observed_before_recovery": True,
            "stock_nnunet_mean_foreground_dice_observed_before_recovery": 0.753518646,
            "stock_nnunet_validation_used_for_recovery": False,
            "custom_joint_fixed_validation_started": False,
            "custom_joint_fixed_validation_output_existed_at_authorization": False,
            "rescue_process_validation_images_opened": False,
            "rescue_process_validation_batches_consumed": 0,
            "rescue_process_validation_used_for_recovery": False,
        },
        "recovery_policy": {
            "schedule_changed": False,
            "source_checkpoint_changed": False,
            "reset_seed_changed": False,
            "maximum_update_bearing_trajectories": 1,
            "maximum_zero_update_runtime_recoveries": 1,
            "process_launch_count_after_relaunch": 2,
            "no_further_recovery_allowed": True,
        },
    }
    recovery_path = tmp_path / "fold_0" / "classification_rescue_zero_update_recovery.json"
    _write_json(recovery_path, recovery)

    source_components = {"encoder": "b" * 64, "decoder": "c" * 64, "classification": "d" * 64}
    current_components = {"encoder": "b" * 64, "decoder": "c" * 64, "classification": "e" * 64}
    rescue = {
        "schema_version": 1,
        "method": "post_training_frozen_backbone_classification_head_rescue",
        "status": "complete",
        "source_checkpoint": str(checkpoints["checkpoint_final"].resolve()),
        "source_checkpoint_sha256": _sha256(checkpoints["checkpoint_final"]),
        "activation_audit": str(activation_path.resolve()),
        "activation_audit_sha256": _sha256(activation_path),
        "activation_decision_epoch": 40,
        "process_launch_count": 2,
        "zero_update_recovery_count": 1,
        "update_bearing_trajectory_count": 1,
        "execution_recovery": recovery,
        "execution_recovery_audit": str(recovery_path.resolve()),
        "execution_recovery_audit_sha256": _sha256(recovery_path),
        "source_component_sha256": source_components,
        "current_component_sha256": current_components,
        "output_checkpoint": str(checkpoints["checkpoint_classification_rescue"].resolve()),
        "output_checkpoint_sha256": _sha256(checkpoints["checkpoint_classification_rescue"]),
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
        "precision_policy": {
            "autocast_scope": "frozen_encoder_forward_only",
            "frozen_encoder_forward": "cuda_autocast_float16",
            "trainable_classification_forward": "float32",
            "classification_loss": "float32",
            "classification_backward": "float32",
            "gradient_clipping": "float32",
            "optimizer_update": "float32",
            "grad_scaler_enabled": False,
        },
        "successful_optimizer_updates": 3750,
        "training_loader": "single_threaded_training_split_only",
        "training_batch_size": 2,
        "decoder_executed_during_rescue": False,
        "encoder_gradient_enabled": False,
        "decoder_gradient_enabled": False,
        "classification_parameter_names": [
            "classification_pool.query",
            "classification_head.4.weight",
        ],
        "classification_trainable_parameter_count": 496_195,
        "training_class_counts": [62, 106, 84],
        "training_class_weights": [252 / (3 * count) for count in (62, 106, 84)],
        "split_audit": {
            "training_case_count": 252,
            "training_case_ids_sha256": "f" * 64,
            "validation_case_count": 36,
            "validation_case_ids_sha256": MODULE._case_ids_sha256(
                [row["case_id"] for row in cases]
            ),
            "split_disjoint": True,
            "validation_images_opened": False,
            "validation_batches_consumed": 0,
            "validation_used_for_gradients": False,
            "validation_used_for_stopping": False,
        },
        "selection_or_stopping_metric": None,
        "completed_epochs": 30,
        "training_only_history": [
            {
                "epoch": epoch,
                "training_loss_mean": 1.0 - epoch / 100,
                "training_patch_accuracy": 0.4 + epoch / 100,
                "elapsed_seconds": 1.5,
                "generalization_metric": False,
                "successful_optimizer_updates": 125,
            }
            for epoch in range(30)
        ],
    }
    rescue_path = tmp_path / "fold_0" / "checkpoint_classification_rescue.pth.audit.json"
    _write_json(rescue_path, rescue)

    candidate_payloads = {"checkpoint_classification_rescue": metrics}
    for candidate, factor in {
        "checkpoint_best": 0.70,
        "checkpoint_best_multitask": 0.80,
        "checkpoint_final": 0.90,
    }.items():
        payload = copy.deepcopy(metrics)
        payload_cases = payload["cases"]
        for row in payload_cases:
            row["whole_pancreas_dice"] *= factor
            row["lesion_dice"] *= factor
        payload["segmentation"]["whole_pancreas_dice"] = _summary(
            [row["whole_pancreas_dice"] for row in payload_cases], seed=12345
        )
        payload["segmentation"]["lesion_dice"] = _summary(
            [row["lesion_dice"] for row in payload_cases], seed=12346
        )
        payload["classification"] = _classification(payload_cases)
        candidate_payloads[candidate] = payload

    candidate_metrics = {
        candidate: {
            "whole_pancreas_dice": payload["segmentation"]["whole_pancreas_dice"]["mean"],
            "lesion_dice": payload["segmentation"]["lesion_dice"]["mean"],
            "macro_f1": payload["classification"]["macro_f1"],
        }
        for candidate, payload in candidate_payloads.items()
    }
    ranking = []
    for candidate, candidate_values in candidate_metrics.items():
        candidate_metrics_path = (
            metrics_path
            if candidate == "checkpoint_classification_rescue"
            else tmp_path / candidate / "metrics.json"
        )
        if candidate_metrics_path != metrics_path:
            _write_json(candidate_metrics_path, candidate_payloads[candidate])
        ranking.append(
            {
                "candidate": candidate,
                "metrics_source": str(candidate_metrics_path.resolve()),
                "metrics": candidate_values,
                "selection_score": sum(candidate_values.values()) / 3,
                "checkpoint_path": str(checkpoints[candidate].resolve()),
                "checkpoint_sha256": _sha256(checkpoints[candidate]),
            }
        )
    ranking.sort(key=lambda item: (-item["selection_score"], item["candidate"]))
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank
    selected = ranking[0]
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
        "selected_candidate": selected["candidate"],
        "selected_score": selected["selection_score"],
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "ranking": ranking,
    }
    selection_path = tmp_path / "checkpoint_selection.json"
    _write_json(selection_path, selection)

    runtime = {
        "case_count": 36,
        "checkpoint": "checkpoint_classification_rescue.pth",
        "device": "cuda",
        "folds": [0],
        "gaussian_enabled": True,
        "mean_seconds_per_case": 10.0,
        "peak_allocated_mib": 4000.0,
        "peak_reserved_mib": 5000.0,
        "tile_step_size": 0.5,
        "total_seconds": 360.0,
        "tta_enabled": True,
    }
    runtime_path = metrics_path.with_name("runtime.json")
    _write_json(runtime_path, runtime)
    return {
        "selection": selection_path,
        "metrics": metrics_path,
        "case_metrics": case_metrics_path,
        "runtime": runtime_path,
        "activation": activation_path,
        "rescue": rescue_path,
        "selected_checkpoint": checkpoints["checkpoint_classification_rescue"],
    }


def _summarize(bundle: dict[str, Path]) -> dict[str, Any]:
    return MODULE.summarize_final_evidence(
        selection_path=bundle["selection"],
        metrics_path=bundle["metrics"],
        case_metrics_path=bundle["case_metrics"],
        runtime_path=bundle["runtime"],
        activation_audit_path=bundle["activation"],
        rescue_audit_path=bundle["rescue"],
        selected_checkpoint_path=bundle["selected_checkpoint"],
    )


def _cli_arguments(bundle: dict[str, Path], output: Path) -> list[str]:
    return [
        "--selection",
        str(bundle["selection"]),
        "--metrics",
        str(bundle["metrics"]),
        "--case-metrics",
        str(bundle["case_metrics"]),
        "--runtime",
        str(bundle["runtime"]),
        "--activation-audit",
        str(bundle["activation"]),
        "--rescue-audit",
        str(bundle["rescue"]),
        "--selected-checkpoint",
        str(bundle["selected_checkpoint"]),
        "--output",
        str(output),
    ]


def test_summarizer_binds_artifacts_and_derives_report_evidence(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)

    summary = _summarize(bundle)

    assert summary["schema_version"] == 1
    assert summary["selected_checkpoint"]["candidate"] == "checkpoint_classification_rescue"
    assert summary["selected_checkpoint"]["file_hash_verified"] is True
    assert len(summary["checkpoint_comparison"]) == 4
    assert summary["rescue"]["audit_recorded_encoder_unchanged"] is True
    assert summary["rescue"]["audit_recorded_decoder_unchanged"] is True
    assert summary["rescue"]["declared_trainable_scope_validated"] is True
    assert summary["lesion_size_association"]["n"] == 36
    assert summary["lesion_size_association"]["rho"] > 0.9
    assert summary["lesion_failure_counts"]["lesion_zero_overlap_case_count"] == 2
    assert summary["lesion_failure_counts"]["lesion_empty_prediction_case_count"] == 1
    assert (
        summary["lesion_failure_counts"]["lesion_nonempty_prediction_zero_overlap_case_count"] == 1
    )
    assert [row["case_id"] for row in summary["qualitative_selection"]["weak_cases"]] == [
        "quiz_000",
        "quiz_001",
    ]
    assert [row["case_id"] for row in summary["qualitative_selection"]["strong_cases"]] == [
        "quiz_035",
        "quiz_034",
    ]


def test_clean_rescue_branch_has_no_fabricated_recovery(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    rescue = json.loads(bundle["rescue"].read_text(encoding="utf-8"))
    recovery_path = Path(rescue["execution_recovery_audit"])
    rescue["process_launch_count"] = 1
    rescue["zero_update_recovery_count"] = 0
    rescue["update_bearing_trajectory_count"] = 1
    for field in (
        "execution_recovery",
        "execution_recovery_audit",
        "execution_recovery_audit_sha256",
    ):
        rescue.pop(field)
    recovery_path.unlink()
    _write_json(bundle["rescue"], rescue)

    summary = _summarize(bundle)

    recovery = summary["rescue"]["execution_recovery"]
    assert recovery["process_launch_count"] == 1
    assert recovery["zero_update_recovery_count"] == 0
    assert recovery["update_bearing_trajectory_count"] == 1
    assert recovery["artifact"] is None
    assert recovery["failed_launch_logs"] == {}


def test_clean_rescue_branch_rejects_existing_canonical_recovery_artifact(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    rescue = json.loads(bundle["rescue"].read_text(encoding="utf-8"))
    rescue["process_launch_count"] = 1
    rescue["zero_update_recovery_count"] = 0
    rescue["update_bearing_trajectory_count"] = 1
    for field in (
        "execution_recovery",
        "execution_recovery_audit",
        "execution_recovery_audit_sha256",
    ):
        rescue.pop(field)
    _write_json(bundle["rescue"], rescue)

    with pytest.raises(
        MODULE.EvidenceError,
        match="Clean rescue branch conflicts with a canonical recovery artifact",
    ):
        _summarize(bundle)


def test_cli_writes_atomic_json(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    output = tmp_path / "evidence" / "summary.json"

    result = MODULE.main(_cli_arguments(bundle, output))

    assert result == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["artifacts"]["metrics"]["sha256"] == _sha256(bundle["metrics"])
    assert not list(output.parent.glob("*.tmp"))


def test_cli_refuses_every_selection_referenced_metric_and_checkpoint_output(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    selection = json.loads(bundle["selection"].read_text(encoding="utf-8"))
    protected = [
        Path(row[field])
        for row in selection["ranking"]
        for field in ("metrics_source", "checkpoint_path")
    ]

    for target in protected:
        before = _sha256(target)
        assert MODULE.main(_cli_arguments(bundle, target)) == 2
        assert _sha256(target) == before


def test_rejects_nonselected_candidate_case_set_or_policy_mutation(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    selection = json.loads(bundle["selection"].read_text(encoding="utf-8"))
    candidate = next(row for row in selection["ranking"] if row["candidate"] == "checkpoint_best")
    source = Path(candidate["metrics_source"])
    metrics = json.loads(source.read_text(encoding="utf-8"))
    metrics["cases"][0]["case_id"] = "quiz_000x"
    _write_json(source, metrics)

    with pytest.raises(MODULE.EvidenceError, match="exact selected validation case IDs"):
        _summarize(bundle)

    metrics["cases"][0]["case_id"] = "quiz_000"
    metrics["evaluation_policy"]["aggregation"] = "voxel weighted"
    _write_json(source, metrics)
    with pytest.raises(MODULE.EvidenceError, match="evaluation_policy.aggregation"):
        _summarize(bundle)


def test_rejects_tampered_fixed_seed_bootstrap_interval(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    metrics = json.loads(bundle["metrics"].read_text(encoding="utf-8"))
    metrics["segmentation"]["lesion_dice"]["bootstrap_ci"]["lower"] += 0.01
    _write_json(bundle["metrics"], metrics)

    with pytest.raises(MODULE.EvidenceError, match="fixed-seed case bootstrap"):
        _summarize(bundle)


def test_cpu_runtime_with_null_cuda_memory_is_supported(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    runtime = json.loads(bundle["runtime"].read_text(encoding="utf-8"))
    runtime["device"] = "cpu"
    runtime["peak_allocated_mib"] = None
    runtime["peak_reserved_mib"] = None
    _write_json(bundle["runtime"], runtime)

    summary = _summarize(bundle)

    assert summary["validation_runtime"]["device"] == "cpu"
    assert summary["validation_runtime"]["peak_allocated_mib"] is None


def test_rejects_runtime_outside_selected_candidate_directory(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    misplaced_runtime = tmp_path / "runtime.json"
    misplaced_runtime.write_bytes(bundle["runtime"].read_bytes())

    with pytest.raises(MODULE.EvidenceError, match="selected candidate directory"):
        MODULE.summarize_final_evidence(
            selection_path=bundle["selection"],
            metrics_path=bundle["metrics"],
            case_metrics_path=bundle["case_metrics"],
            runtime_path=misplaced_runtime,
            activation_audit_path=bundle["activation"],
            rescue_audit_path=bundle["rescue"],
            selected_checkpoint_path=bundle["selected_checkpoint"],
        )


def test_rejects_out_of_scope_rescue_trainable_parameter(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    rescue = json.loads(bundle["rescue"].read_text(encoding="utf-8"))
    rescue["classification_parameter_names"].append("encoder.stages.0.weight")
    _write_json(bundle["rescue"], rescue)

    with pytest.raises(MODULE.EvidenceError, match="outside the pool/head scope"):
        _summarize(bundle)


def test_rejects_case_csv_disagreement(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    with bundle["case_metrics"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[2]["lesion_dice"] = "0.25"
    with bundle["case_metrics"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MODULE.EXPECTED_CASE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(MODULE.EvidenceError, match="JSON/CSV mismatch"):
        _summarize(bundle)


def test_rejects_changed_frozen_encoder_hash(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    rescue = json.loads(bundle["rescue"].read_text(encoding="utf-8"))
    rescue["current_component_sha256"]["encoder"] = "9" * 64
    _write_json(bundle["rescue"], rescue)

    with pytest.raises(MODULE.EvidenceError, match="Frozen rescue component changed: encoder"):
        _summarize(bundle)


def test_rejects_selected_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    bundle["selected_checkpoint"].write_bytes(b"modified after selection")

    with pytest.raises(MODULE.EvidenceError, match="checkpoint SHA-256 is stale"):
        _summarize(bundle)


def test_rejects_nonselected_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    selection = json.loads(bundle["selection"].read_text(encoding="utf-8"))
    candidate = next(row for row in selection["ranking"] if row["candidate"] == "checkpoint_best")
    Path(candidate["checkpoint_path"]).write_bytes(b"modified nonselected checkpoint")

    with pytest.raises(MODULE.EvidenceError, match="stale for checkpoint_best"):
        _summarize(bundle)


def test_rejects_tampered_nonselected_candidate_metrics(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    selection = json.loads(bundle["selection"].read_text(encoding="utf-8"))
    candidate = next(row for row in selection["ranking"] if row["candidate"] == "checkpoint_best")
    source = Path(candidate["metrics_source"])
    metrics = json.loads(source.read_text(encoding="utf-8"))
    metrics["segmentation"]["lesion_dice"]["mean"] = 0.99
    _write_json(source, metrics)

    with pytest.raises(MODULE.EvidenceError, match="does not match the case-level values"):
        _summarize(bundle)


def test_positive_activation_requires_rescue_audit(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)

    with pytest.raises(MODULE.EvidenceError, match="requires --rescue-audit"):
        MODULE.summarize_final_evidence(
            selection_path=bundle["selection"],
            metrics_path=bundle["metrics"],
            case_metrics_path=bundle["case_metrics"],
            runtime_path=bundle["runtime"],
            activation_audit_path=bundle["activation"],
            rescue_audit_path=None,
            selected_checkpoint_path=bundle["selected_checkpoint"],
        )


def test_negative_activation_accepts_exact_three_candidate_branch(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    activation = json.loads(bundle["activation"].read_text(encoding="utf-8"))
    activation["activation_approved"] = False
    activation["decision_epoch"] = None
    activation["epoch_40"] = _inactive_activation_window(40, hard=False)
    activation["epoch_50_hard_audit"] = _inactive_activation_window(50, hard=True)
    _write_json(bundle["activation"], activation)

    selection = json.loads(bundle["selection"].read_text(encoding="utf-8"))
    ranking = [
        row
        for row in selection["ranking"]
        if row["candidate"] != "checkpoint_classification_rescue"
    ]
    final = next(row for row in ranking if row["candidate"] == "checkpoint_final")
    ranking.sort(key=lambda item: (-item["selection_score"], item["candidate"]))
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    selection["candidate_count"] = 3
    selection["ranking"] = ranking
    selection["selected_candidate"] = "checkpoint_final"
    selection["selected_score"] = final["selection_score"]
    selection["selected_checkpoint_path"] = final["checkpoint_path"]
    selection["selected_checkpoint_sha256"] = final["checkpoint_sha256"]
    _write_json(bundle["selection"], selection)

    final_metrics = Path(final["metrics_source"])
    final_payload = json.loads(final_metrics.read_text(encoding="utf-8"))
    final_case_metrics = final_metrics.with_name("case_metrics.csv")
    _write_case_csv(final_case_metrics, final_payload["cases"])
    runtime = json.loads(bundle["runtime"].read_text(encoding="utf-8"))
    runtime["checkpoint"] = "checkpoint_final.pth"
    final_runtime = final_metrics.with_name("runtime.json")
    _write_json(final_runtime, runtime)
    final_checkpoint = Path(final["checkpoint_path"])

    summary = MODULE.summarize_final_evidence(
        selection_path=bundle["selection"],
        metrics_path=final_metrics,
        case_metrics_path=final_case_metrics,
        runtime_path=final_runtime,
        activation_audit_path=bundle["activation"],
        rescue_audit_path=None,
        selected_checkpoint_path=final_checkpoint,
    )

    assert summary["activation"]["activation_approved"] is False
    assert summary["rescue"] is None
    assert summary["selected_checkpoint"]["candidate"] == "checkpoint_final"
    assert len(summary["checkpoint_comparison"]) == 3
