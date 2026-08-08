#!/usr/bin/env python3
"""Build a live-augmented stage-tap feature bank over the 252 training cases.

Train-only: enforces the locked 252-case inventory and never opens a ground-truth
mask, a validation case, or a test case.

Each case contributes ``--replicas`` feature vectors.  Replica 0 is the identity
view (optionally mirror-averaged) and is the only one used to score a held-out
case; replicas >= 1 are nnU-Net augmentation draws and are used only for fitting.
That asymmetry is the point: the previous case-level heads trained on one fixed
vector per case and memorized them.
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

from pancreas_multitask.case_features import (  # noqa: E402
    discover_train_cases,
    train_case_inventory_audit,
)
from pancreas_multitask.classification_rescue import file_sha256  # noqa: E402
from pancreas_multitask.predictor import JointNNUNetPredictor  # noqa: E402
from pancreas_multitask.wholevolume_dataset import (  # noqa: E402
    PreprocessedCaseCache,
    augment_volume,
    build_training_transform,
    pad_to_stride,
    replica_seed,
)

EXPECTED_CHECKPOINT_SHA256 = "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116"
EXPECTED_TRAIN_CASE_DIGEST = "bc9eee511612fce42d700b256d26793e6d2c8aabe06f4bf8699bb2b1abbf17bb"
EXPECTED_CLASS_COUNTS = {"0": 62, "1": 106, "2": 84}
MIRROR_AXIS_SETS = tuple(
    itertools.chain.from_iterable(
        itertools.combinations((1, 2, 3), count) for count in range(4)
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--volume-cache", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--replicas", type=int, default=32)
    parser.add_argument("--mirror-tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augmentation", choices=("full", "geometry"), default="full")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


@torch.inference_mode()
def _stage_features(network, volume: torch.Tensor, stage: int, device: torch.device,
                    mirror: bool) -> np.ndarray:
    """Global-average-pooled stage features for one padded whole volume."""

    work = pad_to_stride(volume)[None].to(device, memory_format=torch.contiguous_format)
    autocast = torch.autocast(device.type, enabled=device.type == "cuda")
    with autocast:
        if not mirror:
            pooled = network.encode_to_stage(work, stage).float().mean(dim=(2, 3, 4))
        else:
            total = None
            for axes in MIRROR_AXIS_SETS:
                view = torch.flip(work, axes) if axes else work
                features = network.encode_to_stage(view, stage).float().mean(dim=(2, 3, 4))
                total = features if total is None else total + features
            pooled = total / len(MIRROR_AXIS_SETS)
    return pooled[0].float().cpu().numpy().astype(np.float32)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_root = args.train_root.expanduser().resolve()
    cases = discover_train_cases(train_root, expected_count=252)
    inventory = train_case_inventory_audit(cases)
    if (
        inventory["case_count"] != 252
        or inventory["class_counts"] != EXPECTED_CLASS_COUNTS
        or inventory["case_ids_sha256_length_prefixed_sorted"] != EXPECTED_TRAIN_CASE_DIGEST
    ):
        raise ValueError("Training inventory differs from the locked 252 cases")

    model = args.model.expanduser().resolve()
    checkpoint = model / "fold_0" / "checkpoint_classification_rescue.pth"
    if file_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
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
    transform = build_training_transform(
        predictor.configuration_manager.patch_size, mode=args.augmentation
    )

    replicas = int(args.replicas)
    if replicas < 1:
        raise ValueError("--replicas must be at least 1")

    bank: list[np.ndarray] = []
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
        identity = torch.from_numpy(record.volume.astype(np.float32))
        rows = [_stage_features(network, identity, args.stage, device, args.mirror_tta)]
        for replica in range(1, replicas):
            augmented = augment_volume(
                record.volume, transform, replica_seed(image_hash, replica)
            )
            rows.append(_stage_features(network, augmented, args.stage, device, mirror=False))
        bank.append(np.stack(rows))
        labels.append(int(case.label))
        case_ids.append(str(case.case_id))
        hashes.append(image_hash)
        if index == 1 or index % 25 == 0 or index == len(cases):
            print(f"bank {index}/{len(cases)}  ({time.perf_counter() - started:.0f}s)", flush=True)

    order = np.argsort(np.asarray(hashes))  # canonical content-hash order
    features = np.stack(bank)[order].astype(np.float32)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        labels=np.asarray(labels, dtype=np.int64)[order],
        case_ids=np.asarray(case_ids)[order],
        source_image_sha256=np.asarray(hashes)[order],
        stage=np.asarray(int(args.stage)),
        replicas=np.asarray(replicas),
        mirror_tta_on_identity=np.asarray(bool(args.mirror_tta)),
        augmentation_mode=np.asarray(str(args.augmentation)),
        canonical_order_rule=np.asarray("ascending_sha256_of_raw_training_ct_bytes"),
    )
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "status": "complete",
        "shape": list(features.shape),
        "elapsed_seconds": round(elapsed, 1),
        "ground_truth_masks_opened": False,
        "validation_or_test_files_opened": False,
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
