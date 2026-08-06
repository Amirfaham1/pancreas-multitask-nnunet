"""Create the train-only, checkpoint-bound rescue activation audit."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.classification_rescue import (
    ACTIVATION_AUDIT_SCHEMA_VERSION,
    atomic_write_json,
    classification_activation_audit,
    file_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(checkpoint_path: Path, output_path: Path) -> dict:
    checkpoint = checkpoint_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if checkpoint.name != "checkpoint_final.pth" or not checkpoint.is_file():
        raise ValueError("Activation audit requires an existing checkpoint_final.pth")
    if checkpoint == output:
        raise ValueError("Audit output must differ from the source checkpoint")
    if output.exists():
        raise FileExistsError(
            "Activation audit already exists and is immutable; refusing to replace it: "
            f"{output}"
        )

    source_hash = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    logging = payload.get("logging")
    if not isinstance(logging, dict):
        raise TypeError("Checkpoint logging must be a mapping")
    losses = logging.get("train_cls_losses")
    accuracies = logging.get("train_cls_accuracy")
    if not isinstance(losses, list) or not isinstance(accuracies, list):
        raise TypeError("Checkpoint lacks list-valued train_cls_losses/train_cls_accuracy")
    if int(payload.get("current_epoch", -1)) != 200:
        raise ValueError("checkpoint_final.pth must record current_epoch=200")
    if len(losses) != 200 or len(accuracies) != 200:
        raise ValueError("Final checkpoint must contain exactly 200 training metric entries")

    audit = {
        "schema_version": ACTIVATION_AUDIT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_name": checkpoint.name,
        "source_checkpoint_sha256": source_hash,
        "checkpoint_current_epoch": int(payload.get("current_epoch", -1)),
        "training_logging_epoch_count": len(losses),
        **classification_activation_audit(losses, accuracies),
    }
    atomic_write_json(audit, output)
    print(f"ACTIVATION_APPROVED={str(audit['activation_approved']).lower()}")
    print(f"ACTIVATION_AUDIT={output}")
    return audit


def main() -> None:
    args = build_parser().parse_args()
    run(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
