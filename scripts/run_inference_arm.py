#!/usr/bin/env python3
"""One timed inference arm, run as a fresh child process.

``--arm stock`` is nnU-Net's own ``nnUNetPredictor`` at shipped defaults: 3
preprocessing worker processes, 3 export worker processes, a tile producer thread,
and one ``load_state_dict`` per case.

``--arm candidate`` is the complete V7 inference path: it skips the redundant
per-case weight reload, overlaps preprocessing/export with GPU work, extracts the
eight-view stage-1/stage-2 features, applies the fitted shrinkage-LDA classifier,
and writes the subtype CSV. Mirroring stays on and the sliding-window step stays at
0.5 in both arms. The parent benchmark measures mask agreement rather than assuming
these execution changes are bit-exact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("stock", "candidate"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--classifier-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--overlap", action=argparse.BooleanOptionalAction, default=True,
        help="P1: overlap preprocessing/export with GPU work. Measured to shift 10 of "
             "7.84M voxels versus the serial path, so it is disabled for any run that "
             "must be bit-identical to stock.",
    )
    parser.add_argument(
        "--optimizations", action=argparse.BooleanOptionalAction, default=True,
        help="Candidate arm only. --no-optimizations runs the original serial path "
             "with a per-case weight reload, so P0/P1 can be isolated.",
    )
    return parser


MIRROR_AXIS_SETS = tuple(
    itertools.chain.from_iterable(itertools.combinations((1, 2, 3), n) for n in range(4))
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def _classification_features(network, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Return the exact 192-D feature vector used to fit the final classifier."""

    from pancreas_multitask.wholevolume_dataset import pad_to_stride, stride_for_stage

    work = pad_to_stride(tensor, stride_for_stage(2))[None].to(device)
    totals: dict[int, torch.Tensor] = {}
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        for axes in MIRROR_AXIS_SETS:
            view = torch.flip(work, axes) if axes else work
            for stage, features in network.encode_to_stages(view, (1, 2)).items():
                pooled = features.float().mean(dim=(2, 3, 4))
                totals[stage] = pooled if stage not in totals else totals[stage] + pooled
    return torch.cat(
        [totals[stage] / len(MIRROR_AXIS_SETS) for stage in (1, 2)], dim=1
    )[0].cpu().numpy().astype(np.float32)


def main() -> int:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    if args.arm == "candidate":
        if args.classifier is None or not args.classifier.is_file():
            raise ValueError("--classifier is required for the candidate arm")
        if args.classifier_sha256:
            observed = _sha256(args.classifier)
            if observed.lower() != args.classifier_sha256.lower():
                raise ValueError(
                    f"Classifier SHA-256 mismatch: expected {args.classifier_sha256}, "
                    f"observed {observed}"
                )

    # Apply the identical determinism policy to BOTH arms before either predictor is
    # constructed. nnUNetPredictor.__init__ sets `cudnn.benchmark = True`
    # (predict_from_raw_data.py:61), which picks convolution algorithms by timing; the
    # joint predictor does not. Left alone, that asymmetry alone produced 19 disagreeing
    # voxels out of 7.8M between the arms -- boundary noise from different kernels, not
    # from anything being optimized here. Pinning both costs ~0% under fp16 autocast
    # (measured 37.9 ms/forward locked against 36.3-39.2 ms unlocked), so it does not
    # bias the timing, and it is what makes "zero disagreeing voxels" a meaningful claim.
    from pancreas_multitask.inference_determinism import reassert_deterministic_inference

    reassert_deterministic_inference()

    if args.arm == "stock":
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True, device=device,
            verbose=False, verbose_preprocessing=False, allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(args.model), use_folds=(0,),
            checkpoint_name="checkpoint_classification_rescue.pth",
        )
        predictor.predict_from_files(
            str(args.input), str(args.output),
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=args.workers,
            num_processes_segmentation_export=args.workers,
        )
        return 0

    from nnunetv2.inference.export_prediction import export_prediction_from_logits

    from pancreas_multitask.inference_pipeline import ExportPool, preprocess_in_background
    from pancreas_multitask.predictor import JointNNUNetPredictor

    predictor = JointNNUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=device,
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model), use_folds=(0,),
        checkpoint_name="checkpoint_classification_rescue.pth",
    )
    network = predictor.network.to(device).eval()
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    images = sorted(args.input.glob("*_0000.nii.gz"))
    cases = [(image.name[: -len("_0000.nii.gz")], [str(image)]) for image in images]

    def run_case(files):
        return preprocessor.run_case(
            files, None, predictor.plans_manager,
            predictor.configuration_manager, predictor.dataset_json,
        )

    def serial_source():
        for case_id, files in cases:
            data, _seg, properties = run_case(files)
            yield type("C", (), {"case_id": case_id, "data": data, "properties": properties})()

    use_overlap = args.optimizations and args.overlap
    source = (preprocess_in_background(cases, run_case, prefetch=args.workers)
              if use_overlap else serial_source())
    case_ids: list[str] = []
    feature_rows: list[np.ndarray] = []
    with ExportPool(processes=args.workers if use_overlap else 1) as pool:
        for case in source:
            if not args.optimizations:
                predictor._invalidate_resident_fold()   # force stock-style per-case reload
            tensor = torch.from_numpy(np.asarray(case.data, dtype=np.float32))
            prediction = predictor.predict_joint_from_preprocessed_data(tensor)
            # Classification uses the same mirror-averaged feature definition as
            # fitting and final validation. Omitting seven views makes the timing
            # look better but does not benchmark the submitted model.
            feature_rows.append(_classification_features(network, tensor, device))
            case_ids.append(case.case_id)
            pool.submit(
                export_prediction_from_logits,
                (prediction.segmentation_logits.numpy(), case.properties,
                 predictor.configuration_manager, predictor.plans_manager,
                 predictor.dataset_json, str(args.output / case.case_id), False),
            )

    from joblib import load

    classifier = load(args.classifier)
    subtype_predictions = classifier.predict(np.stack(feature_rows))
    with (args.output / "subtype_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Names", "Subtype"])
        for case_id, subtype in zip(case_ids, subtype_predictions, strict=True):
            writer.writerow([f"{case_id}.nii.gz", int(subtype)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
