#!/usr/bin/env python3
"""Fit train-only shallow classifiers for faster spatial feature scales.

The frozen encoder is evaluated at several reduced spatial scales using the
validation-selected mirror view. One shrinkage-LDA pipeline is fitted per scale
on the 252 training cases only. Validation chooses the smallest scale that meets
the declared macro-F1 threshold; validation rows are never passed to ``fit``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("nnUNet_extTrainer", str((ROOT / "src").resolve()))

from pancreas_multitask.case_features import discover_train_cases  # noqa: E402
from pancreas_multitask.predictor import JointNNUNetPredictor  # noqa: E402


@dataclass(frozen=True, slots=True)
class ValidationCase:
    case_id: str
    label: int
    image_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scales", type=float, nargs="+", default=(0.25, 0.375, 0.5, 0.625)
    )
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--view-index", type=int, default=6)
    parser.add_argument("--stages", type=int, nargs="+", default=(1, 2))
    parser.add_argument(
        "--checkpoint", default="checkpoint_classification_rescue.pth"
    )
    return parser


def _arm_module():
    path = ROOT / "scripts" / "run_inference_arm.py"
    spec = importlib.util.spec_from_file_location("run_inference_arm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validation_cases(root: Path) -> tuple[ValidationCase, ...]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.name.casefold() != "validation":
        raise ValueError("validation_root must be an existing directory named 'validation'")
    cases = []
    for label in range(3):
        folder = root / f"subtype{label}"
        if not folder.is_dir():
            raise ValueError(f"Missing validation class directory: {folder}")
        for image in sorted(folder.glob("*_0000.nii.gz")):
            cases.append(
                ValidationCase(image.name[: -len("_0000.nii.gz")], label, image)
            )
    if len(cases) != 36 or len({case.case_id for case in cases}) != 36:
        raise ValueError(f"Expected 36 unique validation cases, found {len(cases)}")
    return tuple(cases)


def _feature_inventory_hash(case_ids: list[str], labels: list[int]) -> str:
    digest = hashlib.sha256()
    for case_id, label in zip(case_ids, labels, strict=True):
        encoded = f"{case_id}\0{label}".encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    scales = tuple(sorted({float(scale) for scale in args.scales}))
    if not scales or any(not 0.0 < scale <= 1.0 for scale in scales):
        raise ValueError("scales must be unique values in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    train_cases = discover_train_cases(args.train_root, expected_count=252)
    validation_cases = _validation_cases(args.validation_root)
    arm = _arm_module()
    if args.view_index < 0 or args.view_index >= len(arm.MIRROR_AXIS_SETS):
        raise ValueError("view-index must be between 0 and 7")

    predictor = JointNNUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=False,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model.expanduser().resolve()),
        use_folds=(0,),
        checkpoint_name=args.checkpoint,
    )
    predictor.network.load_state_dict(predictor.list_of_parameters[0], strict=True)
    network = predictor.network.to("cuda").eval()
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)

    split_features: dict[str, dict[float, list[np.ndarray]]] = {
        "train": {scale: [] for scale in scales},
        "validation": {scale: [] for scale in scales},
    }
    split_labels: dict[str, list[int]] = {"train": [], "validation": []}
    split_ids: dict[str, list[str]] = {"train": [], "validation": []}
    started = time.perf_counter()
    for split, cases in (("train", train_cases), ("validation", validation_cases)):
        split_started = time.perf_counter()
        for index, case in enumerate(cases, start=1):
            data, _segmentation, _properties = preprocessor.run_case(
                [str(case.image_path)],
                None,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.dataset_json,
            )
            tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
            for scale in scales:
                split_features[split][scale].append(
                    arm._classification_features(
                        network,
                        tensor,
                        torch.device("cuda"),
                        view_indices=(args.view_index,),
                        spatial_scale=scale,
                        stages=args.stages,
                    )
                )
            split_labels[split].append(int(case.label))
            split_ids[split].append(str(case.case_id))
            if index == 1 or index % 25 == 0 or index == len(cases):
                print(
                    f"{split} {index}/{len(cases)} "
                    f"({time.perf_counter() - split_started:.1f}s)",
                    flush=True,
                )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_y = np.asarray(split_labels["train"], dtype=np.int64)
    validation_y = np.asarray(split_labels["validation"], dtype=np.int64)
    results = {}
    models = {}
    for scale in scales:
        classifier = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ).fit(np.stack(split_features["train"][scale]), train_y)
        prediction = np.asarray(
            classifier.predict(np.stack(split_features["validation"][scale])),
            dtype=np.int64,
        )
        macro_f1 = float(
            f1_score(
                validation_y,
                prediction,
                average="macro",
                labels=[0, 1, 2],
                zero_division=0,
            )
        )
        stage_tag = "-".join(str(stage) for stage in args.stages)
        model_path = output_dir / (
            f"classifier_stage{stage_tag}_view{args.view_index}_scale{scale:g}.joblib"
        )
        joblib.dump(classifier, model_path)
        models[scale] = model_path
        results[str(scale)] = {
            "macro_f1": macro_f1,
            "confusion_matrix": confusion_matrix(
                validation_y, prediction, labels=[0, 1, 2]
            ).tolist(),
            "model_file": model_path.name,
        }

    passing = [scale for scale in scales if results[str(scale)]["macro_f1"] >= args.threshold]
    selected_scale = min(passing) if passing else None
    payload = {
        "study": "train-only refit for reduced-scale shallow inference",
        "encoder_updated": False,
        "classifier_fit_rows": len(train_cases),
        "validation_rows_used_in_fit": 0,
        "selection_data": "validation only",
        "selection_rule": f"smallest spatial scale with macro-F1 >= {args.threshold:g}",
        "view_index": args.view_index,
        "mirror_axes": list(arm.MIRROR_AXIS_SETS[args.view_index]),
        "stages": list(args.stages),
        "train_inventory_sha256": _feature_inventory_hash(
            split_ids["train"], split_labels["train"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "selected_scale": selected_scale,
        "selected_model_file": None if selected_scale is None else models[selected_scale].name,
    }
    (output_dir / "scaled_classifier_study.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if selected_scale is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
