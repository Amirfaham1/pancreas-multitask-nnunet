#!/usr/bin/env python3
"""Fine-tune the shared encoder's shallow stages for subtype classification.

Warm-starts from the existing 0.9202/0.6197 checkpoint and runs two streams every
step: nnU-Net's unchanged patch segmentation objective, and a whole-ROI
classification objective on a separate AdamW optimizer (see trainer_v7 for why a
separate optimizer, not a second param group, is required).

The 36 supplied validation cases are read in ``inference_mode`` for monitoring and
candidate selection only, which the assessment explicitly permits; they never
enter an optimizer batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("nnUNet_extTrainer", str((ROOT / "src").resolve()))

from sklearn.metrics import f1_score  # noqa: E402

from pancreas_multitask.case_features import discover_train_cases  # noqa: E402
from pancreas_multitask.classification_rescue import file_sha256  # noqa: E402
from pancreas_multitask.trainer_v7 import (  # noqa: E402
    TRAINABLE_ENCODER_STAGE,
    nnUNetTrainerPancreasMultiTaskV7,
)
from pancreas_multitask.wholevolume_dataset import (  # noqa: E402
    PreprocessedCaseCache,
    pad_to_stride,
)

MIRROR_AXIS_SETS = tuple(
    __import__("itertools").chain.from_iterable(
        __import__("itertools").combinations((1, 2, 3), n) for n in range(4)
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--volume-cache", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=125)
    parser.add_argument("--mirror-eval", action=argparse.BooleanOptionalAction, default=True)
    return parser


@torch.inference_mode()
def classify_cases(network, volumes: list[np.ndarray], device: torch.device,
                   mirror: bool) -> np.ndarray:
    network.eval()
    rows = []
    autocast = torch.autocast(device.type, enabled=device.type == "cuda")
    for volume in volumes:
        work = pad_to_stride(torch.from_numpy(volume.astype(np.float32)))[None]
        work = work.to(device, memory_format=torch.contiguous_format)
        with autocast:
            if mirror:
                total = None
                for axes in MIRROR_AXIS_SETS:
                    view = torch.flip(work, axes) if axes else work
                    logits = network.classify_volume(view).float()
                    total = logits if total is None else total + logits
                logits = total / len(MIRROR_AXIS_SETS)
            else:
                logits = network.classify_volume(work).float()
        rows.append(logits[0].cpu().numpy())
    network.train()
    return np.stack(rows)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda")
    preprocessed = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_PancreasMultitask"
    plans = json.loads((preprocessed / "nnUNetResEncUNetMPlans.json").read_text())
    plans["continue_training"] = False
    dataset_json = json.loads((preprocessed / "dataset.json").read_text())

    os.environ["PANCREAS_MT7_EPOCHS"] = str(args.epochs)
    os.environ["PANCREAS_MT_TRAIN_ITERS"] = str(args.iterations)
    os.environ["PANCREAS_MT_VAL_ITERS"] = "5"

    trainer = nnUNetTrainerPancreasMultiTaskV7(plans, "3d_fullres", 0, dataset_json, device)
    trainer.initialize()

    # Warm start: the segmentation backbone is already at 0.9202/0.6197 and must not
    # be relearned. strict=False because the shallow classification head is new and
    # intentionally has a different shape from the bottleneck head in the checkpoint.
    checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    network = trainer.network
    for attribute in ("module", "_orig_mod"):
        network = getattr(network, attribute, network)
    source = checkpoint["network_weights"]
    target = network.state_dict()
    # strict=False tolerates missing/unexpected keys but still raises on a shape
    # mismatch, and the old bottleneck head is 640-wide against this head's 128.
    # Select by shape so the backbone transfers and only the new head stays fresh.
    carried = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    skipped = sorted(set(source) - set(carried))
    network.load_state_dict(carried, strict=False)
    backbone = [k for k in carried if k.startswith(("encoder.", "decoder."))]
    loaded = network.state_dict()
    verified = sum(1 for k in backbone if torch.equal(loaded[k].cpu(), source[k].cpu()))
    if verified != len(backbone) or not backbone:
        raise RuntimeError("Warm start failed to transfer the segmentation backbone")
    print(f"warm start: {len(backbone)} backbone tensors transferred and verified "
          f"byte-identical; {len(skipped)} incompatible head tensors skipped "
          f"({skipped[:2]}{'...' if len(skipped) > 2 else ''})")
    # Re-anchor to the warm-started weights, not the random initialization.
    trainer._anchor_reference = [
        p.detach().clone() for p in network.encoder_stage_parameters(TRAINABLE_ENCODER_STAGE)
    ]

    cases = discover_train_cases(args.validation_root.expanduser().resolve(), expected_count=36)
    cache = PreprocessedCaseCache(args.volume_cache.expanduser().resolve())
    predictor_bits = (trainer.plans_manager, trainer.configuration_manager, dataset_json)
    preprocessor = trainer.configuration_manager.preprocessor_class(verbose=False)
    val_volumes, val_labels = [], []
    for case in cases:
        record = cache.load_or_build(
            case_id=case.case_id, label=case.label, image_path=case.image_path,
            image_sha256=file_sha256(case.image_path), preprocessor=preprocessor,
            plans_manager=predictor_bits[0], configuration_manager=predictor_bits[1],
            dataset_json=predictor_bits[2],
        )
        val_volumes.append(record.volume)
        val_labels.append(int(case.label))
    val_labels = np.asarray(val_labels, dtype=np.int64)
    print(f"validation monitoring set: {len(val_volumes)} cases {np.bincount(val_labels).tolist()}")

    trainer.on_train_start()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = {"macro_f1": -1.0, "epoch": -1}

    for epoch in range(args.epochs):
        trainer.current_epoch = epoch
        trainer.on_epoch_start()
        trainer.on_train_epoch_start()
        outputs = [trainer.train_step(next(trainer.dataloader_train))
                   for _ in range(trainer.num_iterations_per_epoch)]

        logits = classify_cases(network, val_volumes, device, args.mirror_eval)
        prediction = logits.argmax(axis=1)
        macro_f1 = float(f1_score(val_labels, prediction, average="macro",
                                  labels=[0, 1, 2], zero_division=0))
        row = {
            "epoch": epoch,
            "train_cls_ce": float(np.mean([o["cls_loss"] for o in outputs])),
            "train_cls_acc": sum(o["cls_correct"] for o in outputs) / sum(o["cls_count"] for o in outputs),
            "train_seg_loss": float(np.mean([o["seg_loss"] for o in outputs])),
            "cls_grad_norm": float(np.mean([o["cls_grad_norm"] for o in outputs])),
            "anchor_drift": float(np.mean([o["cls_anchor"] for o in outputs])),
            "val_macro_f1": macro_f1,
        }
        history.append(row)
        print(f"epoch {epoch:>3}  cls_CE {row['train_cls_ce']:.4f}  train_acc {row['train_cls_acc']:.3f}  "
              f"seg {row['train_seg_loss']:.4f}  anchor {row['anchor_drift']:.2f}  "
              f"VAL macro-F1 {macro_f1:.4f}", flush=True)

        if macro_f1 > best["macro_f1"]:
            best = {"macro_f1": macro_f1, "epoch": epoch}
            torch.save({"network_weights": network.state_dict(), "epoch": epoch,
                        "val_macro_f1": macro_f1},
                       output / "checkpoint_best_classification.pth")
        (output / "training_history.json").write_text(
            json.dumps({"history": history, "best": best}, indent=2) + "\n")

    print(f"\nbest validation macro-F1 {best['macro_f1']:.4f} at epoch {best['epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
