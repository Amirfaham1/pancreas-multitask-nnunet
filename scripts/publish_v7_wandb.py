#!/usr/bin/env python3
"""Create provenance-explicit W&B records for the recovered V7 work.

This command never presents replayed history as a live training run.  It creates
three new runs: a replay of saved training events, an independently recomputed
validation record, and an inference audit.  Offline mode is the default so a
missing API key cannot lose the local records; ``wandb sync`` can publish them
unchanged after authentication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="pancreas-multitask-v7")
    parser.add_argument("--entity")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wandb_v7")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "docs" / "evidence" / "v7" / "wandb_runs.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit_id() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def finish_record(run: Any, *, kind: str, mode: str, output: Path) -> dict[str, Any]:
    run_id = run.id
    run_name = run.name
    run_url = run.url if mode == "online" else None
    run_dir = Path(run.dir).resolve()
    run.finish()
    try:
        relative_dir = run_dir.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        relative_dir = str(run_dir)
    offline_root = run_dir.parent if run_dir.name == "files" else run_dir
    try:
        relative_offline_root = offline_root.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        relative_offline_root = str(offline_root)
    return {
        "kind": kind,
        "run_id": run_id,
        "run_name": run_name,
        "mode": mode,
        "url": run_url,
        "run_directory": relative_dir,
        "sync_command": (
            f"wandb sync {relative_offline_root}" if mode == "offline" else None
        ),
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "docs" / "evidence" / "v7"
    history_path = evidence / "v7_run3_history.json"
    validation_path = evidence / "optimized_validation_metrics.json"
    speed_path = evidence / "inference_speed_audit.json"
    for path in (history_path, validation_path, speed_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    history = load_json(history_path)
    validation = load_json(validation_path)
    speed = load_json(speed_path)
    revision = commit_id()

    import wandb

    common = {
        "git_commit_at_logging": revision,
        "dataset": "supplied_pancreas_quiz_data",
        "external_data_used": False,
        "provenance_policy": "recovered_events_are_explicitly_labelled",
    }
    records: list[dict[str, Any]] = []

    training = wandb.init(
        project=args.project,
        entity=args.entity,
        name="v7-recovered-finetune-history",
        job_type="evidence-replay",
        group="v7-final-evidence",
        mode=args.mode,
        dir=str(output),
        config={
            **common,
            "source": "recovered_saved_history_json",
            "history_sha256": sha256(history_path),
            "replayed_from_saved_events": True,
            "live_training_run": False,
            "epochs_in_recovered_history": len(history["history"]),
        },
        notes=(
            "New provenance record that replays measurements from the recovered V7 "
            "history file. It is not represented as the original live training run."
        ),
        reinit="finish_previous",
    )
    assert training is not None
    for row in history["history"]:
        epoch = int(row["epoch"])
        training.log(
            {f"recovered_training/{key}": value for key, value in row.items() if key != "epoch"}
            | {"recovered_training/epoch": epoch},
            step=epoch,
        )
    training.summary.update(
        {
            "provenance/replayed_history": True,
            "provenance/live_training": False,
            "recovered_training/best_macro_f1": float(history["best"]["macro_f1"]),
            "recovered_training/best_epoch": int(history["best"]["epoch"]),
        }
    )
    artifact = wandb.Artifact("v7-recovered-training-history", type="evidence")
    artifact.add_file(str(history_path), name=history_path.name)
    training.log_artifact(artifact)
    records.append(finish_record(training, kind="recovered_training_history", mode=args.mode, output=output))

    evaluated = wandb.init(
        project=args.project,
        entity=args.entity,
        name="v7-independent-validation",
        job_type="evaluation",
        group="v7-final-evidence",
        mode=args.mode,
        dir=str(output),
        config={
            **common,
            "source": "independent_recomputation_from_saved_predictions_and_classifier",
            "metrics_sha256": sha256(validation_path),
            "classifier_sha256": validation["classifier_sha256"],
            "classifier_fit_rows": 252,
            "validation_rows_used_to_fit_classifier": validation[
                "validation_rows_used_to_fit_classifier"
            ],
            "validation_cases": validation["validation_cases"],
            "segmentation_tta_enabled": validation["segmentation_tta_enabled"],
            "sliding_window_step": validation["sliding_window_step"],
        },
        notes="Metrics independently recomputed during final-package verification.",
        reinit="finish_previous",
    )
    assert evaluated is not None
    metric_row = {
        "validation/whole_pancreas_dice": validation["whole_pancreas_dice_mean"],
        "validation/lesion_dice": validation["lesion_dice_mean"],
        "validation/macro_f1": validation["macro_f1"],
        "validation/whole_pancreas_dice_sd": validation["whole_pancreas_dice_std"],
        "validation/lesion_dice_sd": validation["lesion_dice_std"],
        "gate/whole_dice_ge_0.91": int(validation["whole_pancreas_dice_mean"] >= 0.91),
        "gate/lesion_dice_ge_0.31": int(validation["lesion_dice_mean"] >= 0.31),
        "gate/macro_f1_ge_0.70": int(validation["macro_f1"] >= 0.70),
    }
    evaluated.log(metric_row, step=0)
    evaluated.summary.update(metric_row)
    evaluated.summary["validation/confusion_matrix"] = validation["confusion_matrix"]
    artifact = wandb.Artifact("v7-independent-validation", type="evaluation")
    artifact.add_file(str(validation_path), name=validation_path.name)
    evaluated.log_artifact(artifact)
    records.append(finish_record(evaluated, kind="independent_validation", mode=args.mode, output=output))

    audited = wandb.init(
        project=args.project,
        entity=args.entity,
        name="v7-inference-equivalence-and-speed-audit",
        job_type="benchmark-audit",
        group="v7-final-evidence",
        mode=args.mode,
        dir=str(output),
        config={
            **common,
            "source": "complete_local_benchmark_and_recovered_archive_audit",
            "audit_sha256": sha256(speed_path),
            "local_gpu": speed["complete_local_benchmark"]["hardware"],
            "archived_h100_result_eligible": speed["recovered_h100_benchmark"]["eligible"],
        },
        notes=(
            "The speed requirement remains failed/not verified. The archived +11.17% "
            "number omitted required classifier work and is retained only as rejected evidence."
        ),
        reinit="finish_previous",
    )
    assert audited is not None
    local = speed["complete_local_benchmark"]
    archived = speed["recovered_h100_benchmark"]
    speed_row = {
        "speed/local_stock_seconds": local["stock_mean_seconds"],
        "speed/local_candidate_seconds": local["candidate_mean_seconds"],
        "speed/local_runtime_reduction_percent": local["runtime_reduction_percent"],
        "speed/local_gate_passed": int(local["meets_10_percent_gate"]),
        "speed/archived_h100_reported_reduction_percent": archived[
            "reported_runtime_reduction_percent"
        ],
        "speed/archived_h100_result_eligible": int(archived["eligible"]),
        "equivalence/cross_arm_disagreement_fraction": local["cross_arm"][
            "disagreement_fraction"
        ],
        "equivalence/geometry_matches": int(local["cross_arm"]["geometry_matches"]),
        "equivalence/dtype_matches": int(local["cross_arm"]["dtype_matches"]),
    }
    audited.log(speed_row, step=0)
    audited.summary.update(speed_row)
    artifact = wandb.Artifact("v7-inference-audit", type="benchmark")
    artifact.add_file(str(speed_path), name=speed_path.name)
    audited.log_artifact(artifact)
    records.append(finish_record(audited, kind="inference_audit", mode=args.mode, output=output))

    manifest = {
        "schema_version": 1,
        "project": args.project,
        "entity": args.entity,
        "mode": args.mode,
        "git_commit_at_logging": revision,
        "truthfulness_note": (
            "These are newly created evidence records. Recovered training events are "
            "explicitly marked as replayed and are not claimed to be a live historical run."
        ),
        "runs": records,
    }
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
