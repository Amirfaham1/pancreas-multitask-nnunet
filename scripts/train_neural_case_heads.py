#!/usr/bin/env python3
"""Select, calibrate, and refit the two prospectively locked v5 neural heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.classification_rescue import file_sha256
from pancreas_multitask.decision_calibration import evaluate_cross_fitted_offsets
from pancreas_multitask.neural_case_bundle import save_neural_case_head_bundle
from pancreas_multitask.neural_case_head import (
    V5_DECISION_LOCK_SHA256,
    V5_NEURAL_LOCK_SHA256,
    load_neural_bag_dataset,
    materialize_neural_bags,
)
from pancreas_multitask.neural_case_training import (
    configure_deterministic_execution,
    evaluate_neural_candidates,
    fit_selected_neural_head,
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _implementation_manifest() -> tuple[list[dict[str, str]], str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "pancreas_multitask" / "neural_case_head.py",
        ROOT / "src" / "pancreas_multitask" / "neural_case_training.py",
        ROOT / "src" / "pancreas_multitask" / "neural_case_bundle.py",
        ROOT / "src" / "pancreas_multitask" / "decision_calibration.py",
    )
    manifest = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path)} for path in paths
    ]
    return manifest, _canonical_json_sha256(manifest)


class _TrainingLogger:
    """Print sparse progress and optionally send identifier-free scalars to W&B."""

    def __init__(
        self,
        *,
        mode: str,
        output: Path,
        project: str,
        run_name: str,
        config: dict[str, Any],
    ) -> None:
        self._step = 0
        self._run: Any | None = None
        self._mode = mode
        if mode != "disabled":
            import wandb

            self._run = wandb.init(
                project=project,
                name=run_name,
                mode=mode,
                dir=str(output),
                config=config,
                tags=("v5-neural-head", "train-only", "assignment-conforming"),
            )

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_mode": self._mode,
            "effective_mode": getattr(getattr(self._run, "settings", None), "mode", self._mode),
            "entity": getattr(self._run, "entity", None),
            "project": getattr(self._run, "project", None),
            "run_id": getattr(self._run, "id", None),
            "run_name": getattr(self._run, "name", None),
            "run_url": getattr(self._run, "url", None),
        }

    def __call__(self, event: dict[str, Any]) -> None:
        epoch = int(event["epoch"])
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"{event['candidate_id']} {event['trajectory']} "
                f"repeat={event['repeat_index']} fold={event['fold_index']} "
                f"epoch={epoch + 1} loss={float(event['train/loss']):.6f}",
                flush=True,
            )
        if self._run is not None:
            self._run.log(dict(event), step=self._step)
        self._step += 1

    def summarize(self, values: MappingLike) -> None:
        if self._run is not None:
            for key, value in values.items():
                self._run.summary[key] = value

    def finish(self, exit_code: int = 0) -> None:
        if self._run is not None:
            self._run.finish(exit_code=exit_code)


MappingLike = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked two-head neural train-only comparison and refit"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--neural-lock",
        type=Path,
        default=ROOT / "configs" / "phd_neural_case_head_lock_v5.json",
    )
    parser.add_argument(
        "--decision-lock",
        type=Path,
        default=ROOT / "configs" / "phd_neural_decision_lock_v5.json",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    parser.add_argument("--wandb-project", default="pancreas-multitask-amirfaham-fallahpour")
    parser.add_argument("--wandb-run-name", default="v5-neural-case-head-train-only")
    return parser


def run(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Neural training output must be a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    neural_lock_path = args.neural_lock.expanduser().resolve()
    decision_lock_path = args.decision_lock.expanduser().resolve()
    if file_sha256(neural_lock_path) != V5_NEURAL_LOCK_SHA256:
        raise ValueError("Caller-supplied v5 neural-head lock differs from its hash")
    if file_sha256(decision_lock_path) != V5_DECISION_LOCK_SHA256:
        raise ValueError("Caller-supplied v5 decision lock differs from its hash")
    neural_lock = json.loads(neural_lock_path.read_text(encoding="utf-8"))
    decision_lock = json.loads(decision_lock_path.read_text(encoding="utf-8"))
    if decision_lock["neural_head_lock"]["sha256"] != V5_NEURAL_LOCK_SHA256:
        raise ValueError("V5 decision lock is not bound to the neural-head lock")
    expected_status = (
        "frozen_before_any_eligible_case_feature_extraction_or_neural_head_oof_training"
    )
    if (
        neural_lock.get("lock_status") != expected_status
        or decision_lock.get("lock_status") != expected_status
    ):
        raise ValueError("V5 neural and decision locks do not have exact frozen status")

    device = torch.device(args.device)
    determinism = configure_deterministic_execution(device)
    torch.set_num_threads(1)
    dataset = load_neural_bag_dataset(args.features, neural_lock_path)
    tensors = materialize_neural_bags(dataset, device)
    implementation_manifest, implementation_hash = _implementation_manifest()
    logger = _TrainingLogger(
        mode=args.wandb_mode,
        output=output,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config={
            "schema_version": 1,
            "scope": "isolated_supplied_train_only",
            "neural_lock_sha256": V5_NEURAL_LOCK_SHA256,
            "decision_lock_sha256": V5_DECISION_LOCK_SHA256,
            "numeric_content_dataset_sha256": dataset.content_sha256(),
            "candidate_count": 2,
            "candidate_ids": neural_lock["final_candidate_policy"]["eligible_candidates"],
            "official_validation_used": False,
            "case_ids_or_paths_logged": False,
            "frozen_segmentation_joint_run_id": "hrs05iyx",
            "frozen_segmentation_joint_run_url": (
                "https://wandb.ai/amirfahamfallahpour1379-university-of-toronto/"
                "pancreas-multitask-amirfaham-fallahpour/runs/hrs05iyx"
            ),
            "official_baseline_metrics_ingested": False,
        },
    )
    logger.summarize(
        {
            "frozen_baseline/joint_run_id": "hrs05iyx",
            "frozen_baseline/official_metrics_ingested": False,
        }
    )
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    exit_code = 1
    try:
        selection = evaluate_neural_candidates(
            dataset,
            tensors,
            neural_lock,
            event_sink=logger,
        )
        selection.update(
            {
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "neural_lock_sha256": V5_NEURAL_LOCK_SHA256,
                "decision_lock_sha256": V5_DECISION_LOCK_SHA256,
                "extraction_audit_sha256": dataset.provenance["extraction_audit_sha256"],
                "training_implementation_sha256": implementation_hash,
                "eligible_comparison": "best_of_two_locked_neural_heads",
            }
        )
        selection_path = output / "neural_case_head_selection.json"
        _atomic_write_json(selection_path, selection)
        logger.summarize(
            {
                f"oof/{row['candidate_id']}/mean_macro_f1": row["mean_repeat_oof_macro_f1"]
                for row in selection["candidate_results"]
            }
        )
        logger.summarize(
            {
                f"oof/{row['candidate_id']}/minimum_recall": row["minimum_repeat_per_class_recall"]
                for row in selection["candidate_results"]
            }
        )
        logger.summarize(
            {
                f"oof/{row['candidate_id']}/repeat_macro_f1": row["repeat_oof_macro_f1"]
                for row in selection["candidate_results"]
            }
        )
        logger.summarize(
            {
                f"oof/{row['candidate_id']}/repeat_per_class_recall": row[
                    "repeat_oof_per_class_recall"
                ]
                for row in selection["candidate_results"]
            }
        )

        calibration = evaluate_cross_fitted_offsets(selection, decision_lock)
        calibration.update(
            {
                "status": "complete",
                "eligible_for_official": True,
                "decision_lock_sha256": V5_DECISION_LOCK_SHA256,
                "neural_lock_sha256": V5_NEURAL_LOCK_SHA256,
                "selection_audit_sha256": file_sha256(selection_path),
                "numeric_content_dataset_sha256": dataset.content_sha256(),
                "official_validation_or_test_used": False,
                "non_nested_calibration_disclosure": (
                    "Class-offset cross-fitting was applied to cached OOF logits, not "
                    "nested around neural-head training. Every case's own base logit "
                    "came from a head that excluded that case, and its offset was fit "
                    "without that case's label/logit; however, some offset-fitting "
                    "logits were produced by heads trained on held calibration cases. "
                    "The measured calibration gain is therefore train-only stability "
                    "evidence, not an unbiased end-to-end calibrated generalization "
                    "estimate."
                ),
            }
        )
        calibration_path = output / "neural_decision_calibration.json"
        _atomic_write_json(calibration_path, calibration)

        model, refit = fit_selected_neural_head(
            dataset,
            tensors,
            neural_lock,
            selection,
            event_sink=logger,
        )
        refit.update(
            {
                "neural_lock_sha256": V5_NEURAL_LOCK_SHA256,
                "decision_lock_sha256": V5_DECISION_LOCK_SHA256,
                "selection_audit_sha256": file_sha256(selection_path),
                "calibration_audit_sha256": file_sha256(calibration_path),
                "training_implementation_sha256": implementation_hash,
                "finished_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        refit_path = output / "neural_case_head_refit.json"
        _atomic_write_json(refit_path, refit)

        implementation_manifest_after, implementation_hash_after = _implementation_manifest()
        if (
            implementation_manifest_after != implementation_manifest
            or implementation_hash_after != implementation_hash
        ):
            raise RuntimeError("Neural training implementation changed during execution")

        bundle_metadata = {
            "eligible_for_official": True,
            "eligibility_scope": "best_of_two_locked_neural_heads",
            "selected_candidate_id": selection["selected_candidate_id"],
            "neural_lock_sha256": V5_NEURAL_LOCK_SHA256,
            "decision_lock_sha256": V5_DECISION_LOCK_SHA256,
            "numeric_content_dataset_sha256": dataset.content_sha256(),
            "selection_audit_sha256": file_sha256(selection_path),
            "calibration_audit_sha256": file_sha256(calibration_path),
            "refit_audit_sha256": file_sha256(refit_path),
            "refit_final_state_sha256": refit["final_state_sha256"],
            "extraction_audit_sha256": dataset.provenance["extraction_audit_sha256"],
            "cache_manifest_sha256": dataset.provenance["cache_manifest_sha256"],
            "feature_schema_sha256": dataset.provenance["feature_schema_sha256"],
            "training_implementation_sha256": implementation_hash,
            "ground_truth_masks_used_as_features": False,
            "official_validation_or_test_used": False,
            "post_baseline_train_only_extension": True,
            "head_oof_unbiased_end_to_end": False,
            "wandb_run": logger.provenance(),
        }
        bundle_path = output / "neural_case_head_v5.pth"
        save_neural_case_head_bundle(
            bundle_path,
            model,
            candidate_id=selection["selected_candidate_id"],
            class_offsets=calibration["final_offsets"],
            metadata=bundle_metadata,
        )

        final_audit = {
            "schema_version": 1,
            "status": "complete",
            "scope": "isolated_supplied_train_only",
            "eligible_comparison": "best_of_two_locked_neural_heads",
            "selected_candidate_id": selection["selected_candidate_id"],
            "selected_mean_repeat_oof_macro_f1": selection["selected_mean_repeat_oof_macro_f1"],
            "selected_minimum_repeat_per_class_recall": selection[
                "selected_minimum_repeat_per_class_recall"
            ],
            "calibration_activated": calibration["calibration_activated"],
            "calibration_mean_macro_f1_gain": calibration["mean_macro_f1_gain"],
            "final_offsets": calibration["final_offsets"],
            "bundle_sha256": file_sha256(bundle_path),
            "selection_audit_sha256": file_sha256(selection_path),
            "calibration_audit_sha256": file_sha256(calibration_path),
            "refit_audit_sha256": file_sha256(refit_path),
            "neural_lock_sha256": V5_NEURAL_LOCK_SHA256,
            "decision_lock_sha256": V5_DECISION_LOCK_SHA256,
            "numeric_content_dataset_sha256": dataset.content_sha256(),
            "extraction_provenance": dict(dataset.provenance),
            "implementation_manifest": implementation_manifest,
            "implementation_manifest_after": implementation_manifest_after,
            "training_implementation_sha256": implementation_hash,
            "training_implementation_sha256_after": implementation_hash_after,
            "deterministic_execution": determinism,
            "encoder_decoder_and_rescue_head_frozen": True,
            "ground_truth_masks_used_as_features": False,
            "case_ids_paths_filenames_or_order_used": False,
            "official_validation_or_test_used": False,
            "head_oof_unbiased_end_to_end": False,
            "non_nested_calibration_disclosure": calibration["non_nested_calibration_disclosure"],
            "mandatory_disclosures": neural_lock["mandatory_disclosures"],
            "wandb_run": logger.provenance(),
            "frozen_segmentation_provenance": {
                "joint_run_id": "hrs05iyx",
                "joint_run_url": (
                    "https://wandb.ai/amirfahamfallahpour1379-university-of-toronto/"
                    "pancreas-multitask-amirfaham-fallahpour/runs/hrs05iyx"
                ),
                "official_metrics_ingested": False,
            },
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "sklearn_version": sklearn.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        final_audit_path = output / "neural_case_head_fit_audit.json"
        _atomic_write_json(final_audit_path, final_audit)
        logger.summarize(
            {
                "selected_candidate_id": selection["selected_candidate_id"],
                "selected_mean_repeat_oof_macro_f1": selection["selected_mean_repeat_oof_macro_f1"],
                "selected_minimum_repeat_per_class_recall": selection[
                    "selected_minimum_repeat_per_class_recall"
                ],
                "calibration_activated": calibration["calibration_activated"],
                "calibration_mean_macro_f1_gain": calibration["mean_macro_f1_gain"],
                "calibration_plain_repeat_macro_f1": calibration["plain_repeat_macro_f1"],
                "calibration_calibrated_repeat_macro_f1": calibration["calibrated_repeat_macro_f1"],
                "calibration_final_offsets": calibration["final_offsets"],
                "official_validation_used": False,
            }
        )
        print(f"Selected locked neural head: {selection['selected_candidate_id']}")
        print(
            "Train-only repeated-OOF macro-F1: "
            f"{selection['selected_mean_repeat_oof_macro_f1']:.6f}"
        )
        print(f"Final offsets: {calibration['final_offsets']}")
        print(f"Bundle: {bundle_path}")
        print(f"Fit audit: {final_audit_path}")
        exit_code = 0
        return bundle_path
    finally:
        logger.finish(exit_code=exit_code)


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
