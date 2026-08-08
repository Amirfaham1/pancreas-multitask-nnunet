#!/usr/bin/env python3
"""One timed inference arm, run as a fresh child process.

``--arm stock`` is nnU-Net's own ``nnUNetPredictor`` at shipped defaults: 3
preprocessing worker processes, 3 export worker processes, a tile producer thread,
and one ``load_state_dict`` per case.

``--arm candidate`` is the complete V7 inference path: it skips the redundant
per-case weight reload, overlaps preprocessing/export with GPU work, extracts the
validation-selected stage-1/stage-2 feature view, applies the fitted shrinkage-LDA
classifier, and writes the subtype CSV. Segmentation mirroring stays on and the
sliding-window step stays at 0.5 in both arms. The parent benchmark measures mask
agreement rather than assuming these execution changes are bit-exact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("nnUNet_extTrainer", str((ROOT / "src").resolve()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("stock", "candidate"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--classifier-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--tta-batch-size", type=int, default=2)
    parser.add_argument(
        "--batched-segmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use custom tile/TTA microbatching; the deployment default retains "
             "nnU-Net's faster producer-thread sliding window on this hardware.",
    )
    parser.add_argument("--classification-view-batch-size", type=int, default=2)
    parser.add_argument("--classification-spatial-scale", type=float, default=1.0)
    parser.add_argument(
        "--classification-stages", type=int, nargs="+", default=(1,)
    )
    parser.add_argument(
        "--classification-device",
        choices=("cpu-process", "cpu-async", "cuda"),
        default="cpu-async",
        help="Run the frozen shallow classifier in a CPU process/thread or serially on CUDA.",
    )
    parser.add_argument("--classification-cpu-threads", type=int, default=4)
    parser.add_argument(
        "--deterministic-backend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt into the repository's strict deterministic lock. The speed "
             "benchmark default uses nnU-Net/PyTorch's shipped CUDA backend.",
    )
    parser.add_argument(
        "--half-weights", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--classification-view-indices",
        type=int,
        nargs="+",
        default=(6,),
        help="Ordered mirror-view indices used by the shallow classifier. The "
             "validation-selected deployment default is view 6, axes (2, 3).",
    )
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
def _classification_features(
    network,
    tensor: torch.Tensor,
    device: torch.device,
    *,
    view_batch_size: int = 2,
    view_count: int = 8,
    view_indices: tuple[int, ...] | list[int] | None = None,
    spatial_scale: float = 1.0,
    stages: tuple[int, ...] | list[int] = (1, 2),
) -> np.ndarray:
    """Return the exact 192-D feature vector used to fit the final classifier."""

    from pancreas_multitask.wholevolume_dataset import pad_to_stride, stride_for_stage

    scale = float(spatial_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("spatial_scale must be in (0, 1]")
    selected_stages = tuple(int(stage) for stage in stages)
    if (
        not selected_stages
        or len(selected_stages) != len(set(selected_stages))
        or any(stage < 0 for stage in selected_stages)
    ):
        raise ValueError("stages must be unique non-negative indices")
    volume = tensor[None]
    if scale < 1.0:
        target_shape = tuple(max(2, round(length * scale)) for length in tensor.shape[1:])
        volume = torch_functional.interpolate(
            volume,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )
    work = pad_to_stride(volume[0], stride_for_stage(max(selected_stages)))[None].to(device)
    batch_size = int(view_batch_size)
    if batch_size < 1:
        raise ValueError("view_batch_size must be positive")
    if view_indices is None:
        count = int(view_count)
        if count not in (1, 2, 4, 8):
            raise ValueError("view_count must be one of 1, 2, 4, or 8")
        selected_indices = tuple(range(count))
    else:
        selected_indices = tuple(int(index) for index in view_indices)
        if (
            not selected_indices
            or len(selected_indices) != len(set(selected_indices))
            or any(index < 0 or index >= len(MIRROR_AXIS_SETS) for index in selected_indices)
        ):
            raise ValueError("view_indices must be unique values between 0 and 7")
    selected_views = tuple(MIRROR_AXIS_SETS[index] for index in selected_indices)
    totals: dict[int, torch.Tensor] = {}
    try:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            for start in range(0, len(selected_views), batch_size):
                axis_sets = selected_views[start : start + batch_size]
                views = torch.cat(
                    [torch.flip(work, axes) if axes else work for axes in axis_sets], dim=0
                )
                for stage, features in network.encode_to_stages(views, selected_stages).items():
                    pooled = features.float().mean(dim=(2, 3, 4)).sum(dim=0, keepdim=True)
                    totals[stage] = pooled if stage not in totals else totals[stage] + pooled
    except RuntimeError as error:
        if device.type != "cuda" or batch_size == 1 or "out of memory" not in str(error).lower():
            raise
        torch.cuda.empty_cache()
        return _classification_features(
            network, tensor, device, view_batch_size=1
        )
    return torch.cat(
        [totals[stage] / len(selected_views) for stage in selected_stages], dim=1
    )[0].cpu().numpy().astype(np.float32)


def main() -> int:
    process_started = time.perf_counter()
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

    # Strict determinism remains available for conformance checks, but it is not a
    # shipped nnU-Net inference default. The performance comparison leaves both arms
    # on the same fresh-process PyTorch defaults and audits their realized outputs.
    if args.deterministic_backend:
        from pancreas_multitask.inference_determinism import reassert_deterministic_inference

        reassert_deterministic_inference()
    else:
        os.environ.setdefault("nnUNet_compile", "false")

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

    from pancreas_multitask.inference_pipeline import (
        ExportPool,
        FrozenShallowEncoder,
        cpu_shallow_model_worker,
    )
    from pancreas_multitask.predictor import JointNNUNetPredictor

    predictor = JointNNUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=device,
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
        tile_batch_size=args.tile_batch_size,
        tta_batch_size=args.tta_batch_size,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model), use_folds=(0,),
        checkpoint_name="checkpoint_classification_rescue.pth",
    )
    if args.optimizations:
        predictor.retain_initialized_single_fold()
    cpu_classification_network = (
        FrozenShallowEncoder(
            predictor.network, through_stage=max(args.classification_stages)
        )
        if args.classification_device == "cpu-async"
        else None
    )
    if args.classification_device == "cpu-async":
        if args.classification_cpu_threads < 1:
            raise ValueError("classification-cpu-threads must be positive")
        torch.set_num_threads(args.classification_cpu_threads)
    elif args.classification_device == "cpu-process":
        if args.classification_cpu_threads < 1:
            raise ValueError("classification-cpu-threads must be positive")
        # Keep the CUDA launch/accumulation process light. The independent worker
        # owns its own intra-op thread pool for the shallow CPU convolutions.
        torch.set_num_threads(1)
    network = predictor.network.to(device).eval()
    if args.half_weights:
        network.half()
    initialization_seconds = time.perf_counter() - process_started
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

    def multiprocessing_source():
        iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
            [files for _case_id, files in cases],
            None,
            [str(args.output / case_id) for case_id, _files in cases],
            args.workers,
        )
        for item in iterator:
            data = item["data"]
            if isinstance(data, str):
                temporary = Path(data)
                data = torch.from_numpy(np.load(temporary))
                temporary.unlink()
            yield type(
                "C",
                (),
                {
                    "case_id": Path(item["ofile"]).name,
                    "data": data,
                    "properties": item["data_properties"],
                },
            )()

    source = multiprocessing_source() if use_overlap else serial_source()
    classification_process = None
    classification_task_queue = None
    classification_result_queue = None
    if args.classification_device == "cpu-process":
        process_context = multiprocessing.get_context("spawn")
        classification_task_queue = process_context.Queue(maxsize=2)
        classification_result_queue = process_context.Queue()
        selected_axis_sets = tuple(
            MIRROR_AXIS_SETS[index] for index in args.classification_view_indices
        )
        classification_process = process_context.Process(
            target=cpu_shallow_model_worker,
            args=(
                str(args.model.resolve()),
                "checkpoint_classification_rescue.pth",
                max(args.classification_stages),
                classification_task_queue,
                classification_result_queue,
            ),
            kwargs={
                "mirror_axis_sets": selected_axis_sets,
                "stages": tuple(args.classification_stages),
                "spatial_scale": args.classification_spatial_scale,
                "torch_threads": args.classification_cpu_threads,
            },
            name="shallow-classifier-process",
            daemon=True,
        )
        classification_process.start()
    case_ids: list[str] = []
    feature_rows: list[np.ndarray] = []
    feature_futures: list[Future[np.ndarray]] = []
    preprocessing_wait_seconds = 0.0
    segmentation_seconds = 0.0
    classification_seconds = 0.0
    export_submit_seconds = 0.0
    classification_collection_wait_seconds = 0.0
    previous_case_finished = time.perf_counter()
    loop_started = previous_case_finished
    classification_context = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="shallow-classifier")
        if args.classification_device == "cpu-async"
        else nullcontext(None)
    )
    with classification_context as classification_pool:
        with ExportPool(processes=args.workers if use_overlap else 1) as pool:
            for case in source:
                preprocessing_wait_seconds += time.perf_counter() - previous_case_finished
                if not args.optimizations:
                    predictor._invalidate_resident_fold()   # force stock-style per-case reload
                tensor = torch.from_numpy(np.asarray(case.data, dtype=np.float32))
                # Queue the shallow CPU prefix before GPU segmentation so both can
                # execute concurrently. The input tensor is read-only in each path.
                if classification_task_queue is not None:
                    stage_started = time.perf_counter()
                    classification_task_queue.put(
                        (len(case_ids), np.asarray(case.data, dtype=np.float32))
                    )
                    classification_seconds += time.perf_counter() - stage_started
                elif classification_pool is not None:
                    stage_started = time.perf_counter()
                    feature_futures.append(
                        classification_pool.submit(
                            _classification_features,
                            cpu_classification_network,
                            tensor,
                            torch.device("cpu"),
                            view_batch_size=args.classification_view_batch_size,
                            view_indices=args.classification_view_indices,
                            spatial_scale=args.classification_spatial_scale,
                            stages=args.classification_stages,
                        )
                    )
                    classification_seconds += time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                segmentation_logits = (
                    predictor.predict_batched_segmentation_only_from_preprocessed_data(tensor)
                    if args.optimizations and args.batched_segmentation
                    else predictor.predict_segmentation_only_from_preprocessed_data(tensor)
                )
                segmentation_seconds += time.perf_counter() - stage_started

                if args.classification_device == "cuda":
                    stage_started = time.perf_counter()
                    feature_rows.append(
                        _classification_features(
                            network,
                            tensor,
                            device,
                            view_batch_size=args.classification_view_batch_size,
                            view_indices=args.classification_view_indices,
                            spatial_scale=args.classification_spatial_scale,
                            stages=args.classification_stages,
                        )
                    )
                    classification_seconds += time.perf_counter() - stage_started
                case_ids.append(case.case_id)
                stage_started = time.perf_counter()
                pool.submit(
                    export_prediction_from_logits,
                    (segmentation_logits.numpy(), case.properties,
                     predictor.configuration_manager, predictor.plans_manager,
                     predictor.dataset_json, str(args.output / case.case_id), False),
                )
                export_submit_seconds += time.perf_counter() - stage_started
                previous_case_finished = time.perf_counter()

        if classification_task_queue is not None:
            classification_task_queue.put(None)

        if feature_futures:
            stage_started = time.perf_counter()
            feature_rows = [future.result() for future in feature_futures]
            classification_collection_wait_seconds = time.perf_counter() - stage_started
        elif classification_result_queue is not None and classification_process is not None:
            stage_started = time.perf_counter()
            indexed_features: dict[int, np.ndarray] = {}
            while len(indexed_features) < len(case_ids):
                kind, index, payload = classification_result_queue.get(timeout=300)
                if kind == "error":
                    raise RuntimeError(f"CPU classification worker failed: {payload}")
                indexed_features[int(index)] = np.asarray(payload, dtype=np.float32)
            feature_rows = [indexed_features[index] for index in range(len(case_ids))]
            classification_process.join(timeout=30)
            if classification_process.is_alive():
                classification_process.terminate()
                raise RuntimeError("CPU classification worker did not exit")
            if classification_process.exitcode != 0:
                raise RuntimeError(
                    f"CPU classification worker exited with {classification_process.exitcode}"
                )
            classification_collection_wait_seconds = time.perf_counter() - stage_started
    loop_with_export_drain_seconds = time.perf_counter() - loop_started

    from joblib import load

    classifier = load(args.classifier)
    if len(case_ids) != len(cases):
        raise RuntimeError(f"Candidate processed {len(case_ids)} cases; expected {len(cases)}")
    subtype_predictions = classifier.predict(np.stack(feature_rows))
    with (args.output / "subtype_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Names", "Subtype"])
        for case_id, subtype in zip(case_ids, subtype_predictions, strict=True):
            writer.writerow([f"{case_id}.nii.gz", int(subtype)])
    profile = {
        "arm": args.arm,
        "cases": len(case_ids),
        "classification_view_indices": list(args.classification_view_indices),
        "classification_spatial_scale": args.classification_spatial_scale,
        "classification_stages": list(args.classification_stages),
        "classification_device": args.classification_device,
        "batched_segmentation": args.batched_segmentation,
        "half_weights": args.half_weights,
        "initialization_seconds": initialization_seconds,
        "preprocessing_wait_seconds": preprocessing_wait_seconds,
        "segmentation_seconds": segmentation_seconds,
        "classification_seconds": classification_seconds,
        "classification_collection_wait_seconds": classification_collection_wait_seconds,
        "export_submit_wait_seconds": export_submit_seconds,
        "loop_with_export_drain_seconds": loop_with_export_drain_seconds,
        "total_inside_main_seconds": time.perf_counter() - process_started,
        "inference_runtime": predictor.inference_runtime_provenance(),
    }
    (args.output / "runtime_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(profile, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
