#!/usr/bin/env python3
"""Measure shallow-classifier accuracy as mirror views are removed.

This is a validation-only deployment study: the fitted classifier is never
updated.  Each smaller view budget is compared with the locked eight-view
reference so a faster setting is accepted only when it preserves the reported
macro-F1 and per-case labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from joblib import load
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("nnUNet_extTrainer", str((ROOT / "src").resolve()))

from pancreas_multitask.predictor import JointNNUNetPredictor  # noqa: E402
from pancreas_multitask.wholevolume_dataset import (  # noqa: E402
    pad_to_stride,
    stride_for_stage,
)


@dataclass(frozen=True, slots=True)
class ValidationCase:
    case_id: str
    label: int
    image_path: Path


def _discover_validation_cases(root: Path, expected_count: int) -> tuple[ValidationCase, ...]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.name.casefold() != "validation":
        raise ValueError("validation_root must be an existing directory named 'validation'")
    expected_directories = {f"subtype{label}" for label in range(3)}
    observed_directories = {
        item.name for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")
    }
    if observed_directories != expected_directories:
        raise ValueError("validation must contain exactly subtype0/subtype1/subtype2")
    cases: list[ValidationCase] = []
    for label in range(3):
        for image in sorted((root / f"subtype{label}").glob("*_0000.nii.gz")):
            cases.append(ValidationCase(image.name[: -len("_0000.nii.gz")], label, image))
    if len(cases) != expected_count or len({case.case_id for case in cases}) != len(cases):
        raise ValueError(
            f"Expected {expected_count} unique validation cases, found {len(cases)}"
        )
    return tuple(cases)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=36)
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


@torch.inference_mode()
def _per_view_features(network, volume: torch.Tensor, arm) -> np.ndarray:
    """Return eight independent view features using four batched forwards."""

    work = pad_to_stride(volume, stride_for_stage(2))[None].to("cuda")
    per_view: list[torch.Tensor] = []
    with torch.autocast("cuda", enabled=True):
        for start in range(0, len(arm.MIRROR_AXIS_SETS), 2):
            axis_sets = arm.MIRROR_AXIS_SETS[start : start + 2]
            views = torch.cat(
                [torch.flip(work, axes) if axes else work for axes in axis_sets], dim=0
            )
            taps = network.encode_to_stages(views, (1, 2))
            pooled = torch.cat(
                [taps[stage].float().mean(dim=(2, 3, 4)) for stage in (1, 2)],
                dim=1,
            )
            per_view.extend(pooled[index] for index in range(len(axis_sets)))
    return torch.stack(per_view).cpu().numpy().astype(np.float32)


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the deployment view-budget study")

    cases = _discover_validation_cases(
        args.validation_root.expanduser().resolve(),
        args.expected_cases,
    )
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
    classifier = load(args.classifier.expanduser().resolve())
    arm = _arm_module()

    budgets = (1, 2, 4, 8)
    spatial_scales = (1.0, 0.875, 0.75, 0.625, 0.5)
    features = {budget: [] for budget in budgets}
    scale_features = {scale: [] for scale in spatial_scales}
    all_view_features: list[np.ndarray] = []
    labels: list[int] = []
    case_ids: list[str] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        data, _segmentation, _properties = preprocessor.run_case(
            [str(case.image_path)],
            None,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
        case_views = _per_view_features(network, tensor, arm)
        all_view_features.append(case_views)
        for budget in budgets:
            features[budget].append(case_views[:budget].mean(axis=0))
        scale_features[1.0].append(case_views[6])
        for scale in spatial_scales[1:]:
            scale_features[scale].append(
                arm._classification_features(
                    network,
                    tensor,
                    torch.device("cuda"),
                    view_indices=(6,),
                    spatial_scale=scale,
                )
            )
        labels.append(int(case.label))
        case_ids.append(str(case.case_id))
        if index == 1 or index % 12 == 0 or index == len(cases):
            print(
                f"view-budget {index}/{len(cases)} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )

    truth = np.asarray(labels, dtype=np.int64)
    predictions = {
        budget: np.asarray(classifier.predict(np.stack(features[budget])), dtype=np.int64)
        for budget in budgets
    }
    reference = predictions[8]
    results = {}
    for budget in budgets:
        predicted = predictions[budget]
        results[str(budget)] = {
            "macro_f1": float(
                f1_score(
                    truth,
                    predicted,
                    average="macro",
                    labels=[0, 1, 2],
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                truth, predicted, labels=[0, 1, 2]
            ).tolist(),
            "agreement_with_8_view": float(np.mean(predicted == reference)),
            "different_case_ids": [
                case_id
                for case_id, value, reference_value in zip(
                    case_ids, predicted, reference, strict=True
                )
                if value != reference_value
            ],
        }

    view_matrix = np.stack(all_view_features)
    subset_results = {}
    for subset_size in range(1, 9):
        candidates = []
        for subset in itertools.combinations(range(8), subset_size):
            subset_prediction = np.asarray(
                classifier.predict(view_matrix[:, list(subset), :].mean(axis=1)),
                dtype=np.int64,
            )
            candidates.append(
                {
                    "view_indices": list(subset),
                    "mirror_axes": [list(arm.MIRROR_AXIS_SETS[index]) for index in subset],
                    "macro_f1": float(
                        f1_score(
                            truth,
                            subset_prediction,
                            average="macro",
                            labels=[0, 1, 2],
                            zero_division=0,
                        )
                    ),
                    "agreement_with_8_view": float(
                        np.mean(subset_prediction == reference)
                    ),
                }
            )
        subset_results[str(subset_size)] = max(
            candidates,
            key=lambda item: (item["macro_f1"], item["agreement_with_8_view"]),
        )
    scale_results = {}
    full_scale_prediction = np.asarray(
        classifier.predict(np.stack(scale_features[1.0])), dtype=np.int64
    )
    for scale in spatial_scales:
        scale_prediction = np.asarray(
            classifier.predict(np.stack(scale_features[scale])), dtype=np.int64
        )
        scale_results[str(scale)] = {
            "macro_f1": float(
                f1_score(
                    truth,
                    scale_prediction,
                    average="macro",
                    labels=[0, 1, 2],
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                truth, scale_prediction, labels=[0, 1, 2]
            ).tolist(),
            "agreement_with_full_scale_view_6": float(
                np.mean(scale_prediction == full_scale_prediction)
            ),
        }

    payload = {
        "study": "frozen-classifier deployment view budget",
        "selection_data": "validation only",
        "classifier_refit": False,
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "budgets": results,
        "best_exhaustive_subset_by_size": subset_results,
        "selected_view_6_spatial_scale_budget": scale_results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
