#!/usr/bin/env python3
"""Extract clean whole-ROI stage-tap features for a labelled split.

Deliberately separate from ``build_augmented_feature_bank.py``: that script is
train-only and enforces the locked 252-case inventory. This one runs over any
labelled split so the same features can be produced for the supplied validation
set, which the assessment permits reading for monitoring and model selection but
never as training data. Nothing here fits parameters -- it only reads the frozen
encoder in ``inference_mode``.

"Clean" means: pad only to the stage-2 stride multiple (2,4,4) rather than to the
bottleneck's 64x128x192 grid, pool at full resolution rather than from a
pre-pooled 8x8x12 grid, and average the 8 mirror views. That combination measured
0.6502 nested-CV macro-F1 against 0.5943 for the V6-cache representation.
"""

from __future__ import annotations

import argparse
import itertools
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

from pancreas_multitask.case_features import discover_train_cases  # noqa: E402
from pancreas_multitask.classification_rescue import file_sha256  # noqa: E402
from pancreas_multitask.predictor import JointNNUNetPredictor  # noqa: E402
from pancreas_multitask.wholevolume_dataset import (  # noqa: E402
    PreprocessedCaseCache,
    pad_to_stride,
    stride_for_stage,
)

EXPECTED_CHECKPOINT_SHA256 = "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116"
MIRROR_AXIS_SETS = tuple(
    itertools.chain.from_iterable(
        itertools.combinations((1, 2, 3), count) for count in range(4)
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True,
                        help="Directory containing subtype0/ subtype1/ subtype2/")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--volume-cache", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--checkpoint", default="checkpoint_classification_rescue.pth")
    parser.add_argument("--verify-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


@torch.inference_mode()
def mirror_averaged_stage_features(network, volume: torch.Tensor, stage: int,
                                   device: torch.device) -> np.ndarray:
    work = pad_to_stride(volume, stride_for_stage(stage))[None].to(
        device, memory_format=torch.contiguous_format
    )
    autocast = torch.autocast(device.type, enabled=device.type == "cuda")
    total = None
    with autocast:
        for axes in MIRROR_AXIS_SETS:
            view = torch.flip(work, axes) if axes else work
            features = network.encode_to_stage(view, stage).float().mean(dim=(2, 3, 4))
            total = features if total is None else total + features
    return (total / len(MIRROR_AXIS_SETS))[0].float().cpu().numpy().astype(np.float32)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cases = discover_train_cases(
        args.split_root.expanduser().resolve(), expected_count=args.expected_cases
    )
    model = args.model.expanduser().resolve()
    checkpoint = model / "fold_0" / args.checkpoint
    if args.verify_checkpoint and file_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Locked checkpoint hash mismatch")

    predictor = JointNNUNetPredictor(
        tile_step_size=0.5, use_gaussian=False, use_mirroring=True,
        perform_everything_on_device=False, device=device, verbose=False,
        verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model), use_folds=(0,), checkpoint_name=checkpoint.name
    )
    predictor.network.load_state_dict(predictor.list_of_parameters[0], strict=True)
    network = predictor.network.to(device).eval()
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    cache = PreprocessedCaseCache(args.volume_cache.expanduser().resolve())

    rows: list[np.ndarray] = []
    labels: list[int] = []
    case_ids: list[str] = []
    hashes: list[str] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        image_hash = file_sha256(case.image_path)
        record = cache.load_or_build(
            case_id=case.case_id, label=case.label, image_path=case.image_path,
            image_sha256=image_hash, preprocessor=preprocessor,
            plans_manager=predictor.plans_manager,
            configuration_manager=predictor.configuration_manager,
            dataset_json=predictor.dataset_json,
        )
        volume = torch.from_numpy(record.volume.astype(np.float32))
        rows.append(mirror_averaged_stage_features(network, volume, args.stage, device))
        labels.append(int(case.label))
        case_ids.append(str(case.case_id))
        hashes.append(image_hash)
        if index == 1 or index % 25 == 0 or index == len(cases):
            print(f"features {index}/{len(cases)}  ({time.perf_counter() - started:.0f}s)", flush=True)

    order = np.argsort(np.asarray(hashes))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(rows)[order].astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64)[order],
        case_ids=np.asarray(case_ids)[order],
        source_image_sha256=np.asarray(hashes)[order],
        stage=np.asarray(int(args.stage)),
        canonical_order_rule=np.asarray("ascending_sha256_of_raw_training_ct_bytes"),
    )
    print(json.dumps({
        "status": "complete",
        "cases": len(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "ground_truth_masks_opened": False,
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
