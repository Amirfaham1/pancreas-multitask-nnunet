#!/usr/bin/env python3
"""Run raw-file multi-task nnU-Net inference and produce submission outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.predictor import JointNNUNetPredictor


def _fold(value: str) -> int | str:
    if value == "all":
        return value
    try:
        fold = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fold must be an integer or 'all'") from exc
    if fold < 0:
        raise argparse.ArgumentTypeError("fold must be non-negative")
    return fold


def _selected_folds(values: list[int | str] | None) -> tuple[int | str, ...] | None:
    """Preserve nnU-Net's literal ``all`` fold while rejecting mixed selection."""

    if values is None:
        return None
    if "all" in values and values != ["all"]:
        raise ValueError("--folds all cannot be combined with numeric folds")
    return tuple(values)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON artifact through a same-directory temporary file."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_provenance(
    model_directory: Path,
    selected_folds: tuple[int | str, ...] | None,
    checkpoint_name: str,
) -> list[dict[str, object]]:
    """Hash the exact fold checkpoint files outside the timed region."""

    model_directory = model_directory.expanduser().resolve()
    if selected_folds is None:
        fold_directories = sorted(
            path
            for path in model_directory.glob("fold_*")
            if path.is_dir() and (path / checkpoint_name).is_file()
        )
    else:
        fold_directories = [model_directory / f"fold_{fold}" for fold in selected_folds]
    if not fold_directories:
        raise FileNotFoundError(
            f"No fold checkpoint named {checkpoint_name!r} found in {model_directory}"
        )

    provenance: list[dict[str, object]] = []
    for fold_directory in fold_directories:
        checkpoint_path = fold_directory / checkpoint_name
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        provenance.append(
            {
                "fold": fold_directory.name.removeprefix("fold_"),
                "sha256": _sha256(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            }
        )
    return provenance


def _case_ids_sha256(case_ids: list[str]) -> str:
    canonical = "".join(f"{case_id}\n" for case_id in sorted(case_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run joint segmentation/subtype inference on nnU-Net-formatted raw "
            "NIfTI files such as case_0000.nii.gz."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw input image directory")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for restored masks and subtype_results.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="nnU-Net trained-model directory containing plans.json and fold_*",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=_fold,
        help=(
            "Fold numbers to ensemble, or literal 'all' for nnU-Net's fold_all. "
            "Omit to auto-detect numeric folds."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best_multitask.pth",
        help="Checkpoint filename within each fold directory",
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        help="Strict Names,Subtype output (default: OUTPUT/subtype_results.csv)",
    )
    parser.add_argument(
        "--probability-csv",
        type=Path,
        help="Optional detailed per-class probability CSV",
    )
    parser.add_argument(
        "--runtime-json",
        type=Path,
        help="Optional atomic JSON artifact with end-to-end runtime and peak memory",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "mps"),
        default="cuda",
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--tile-step-size",
        type=float,
        default=0.5,
        help="Sliding-window step as a patch-size fraction (default: 0.5)",
    )
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=1,
        help=(
            "Number of consecutive spatial tiles per network forward (default: 1). "
            "An OOM is retried at a smaller batch and recorded in runtime evidence."
        ),
    )
    parser.add_argument(
        "--tta-batch-size",
        type=int,
        default=1,
        help=(
            "Number of mirror orientations packed per network forward (default: 1). "
            "All requested TTA views are still evaluated and averaged."
        ),
    )
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        help="Disable nnU-Net mirror test-time augmentation",
    )
    parser.add_argument(
        "--disable-gaussian",
        action="store_true",
        help="Disable Gaussian segmentation tile weighting",
    )
    parser.add_argument(
        "--results-on-cpu",
        action="store_true",
        help="Accumulate segmentation result arrays on CPU to reduce VRAM use",
    )
    parser.add_argument(
        "--save-segmentation-probabilities",
        action="store_true",
        help="Also save nnU-Net voxel probability .npz and properties .pkl files",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute complete cases (default: --no-overwrite)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if not 0 < args.tile_step_size <= 1:
        raise ValueError("--tile-step-size must be in (0, 1]")
    if args.tile_batch_size < 1:
        raise ValueError("--tile-batch-size must be at least 1")
    if args.tta_batch_size < 1:
        raise ValueError("--tta-batch-size must be at least 1")
    selected_folds = _selected_folds(args.folds)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is unavailable")

    if device.type == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
    else:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting this only once per process.
            pass

    model_directory = args.model.expanduser().resolve()
    checkpoint_provenance = _checkpoint_provenance(
        model_directory,
        selected_folds,
        args.checkpoint,
    )
    predictor = JointNNUNetPredictor(
        tile_step_size=args.tile_step_size,
        tile_batch_size=args.tile_batch_size,
        tta_batch_size=args.tta_batch_size,
        use_gaussian=not args.disable_gaussian,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=not args.results_on_cpu,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started_at_utc = datetime.now(UTC).isoformat()
    started_at = time.perf_counter()
    predictor.initialize_from_trained_model_folder(
        str(model_directory),
        use_folds=selected_folds,
        checkpoint_name=args.checkpoint,
    )
    predictor.reset_inference_runtime_counters()
    results = predictor.predict_from_files_joint(
        args.input,
        args.output,
        classification_csv=args.classification_csv,
        probability_csv=args.probability_csv,
        save_segmentation_probabilities=args.save_segmentation_probabilities,
        overwrite=args.overwrite,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - started_at

    case_count = len(results)
    case_ids = sorted(str(result.case_id) for result in results)
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Predictor returned duplicate case IDs")
    mean_seconds_per_case = total_seconds / case_count if case_count else None
    if device.type == "cuda":
        bytes_per_mib = 1024**2
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / bytes_per_mib
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / bytes_per_mib
    else:
        peak_allocated_mib = None
        peak_reserved_mib = None

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        device_capability: list[int] | None = list(torch.cuda.get_device_capability(device))
    else:
        device_name = platform.processor() or platform.machine()
        device_capability = None

    runtime = {
        "case_count": case_count,
        "case_ids": case_ids,
        "case_ids_sha256": _case_ids_sha256(case_ids),
        "checkpoint": args.checkpoint,
        "checkpoint_files": checkpoint_provenance,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_capability": device_capability,
        "device_name": device_name,
        "folds": list(selected_folds) if selected_folds is not None else "auto",
        "gaussian_enabled": not args.disable_gaussian,
        "inference_execution": predictor.inference_runtime_provenance(),
        "mean_seconds_per_case": mean_seconds_per_case,
        "overwrite": bool(args.overwrite),
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "process_id": os.getpid(),
        "python_version": platform.python_version(),
        "started_at_utc": started_at_utc,
        "tile_step_size": args.tile_step_size,
        "timing_scope": "fresh_process_model_initialization_preprocessing_inference_export",
        "total_seconds": total_seconds,
        "torch_version": torch.__version__,
        "tta_enabled": not args.disable_tta,
        "warmup_policy": "none_fresh_process_end_to_end",
    }
    if args.runtime_json is not None:
        _write_json_atomic(args.runtime_json, runtime)

    classification_path = args.classification_csv or args.output / "subtype_results.csv"
    print(f"Completed {case_count} cases")
    if mean_seconds_per_case is None:
        print(f"Runtime: {total_seconds:.2f} s total (no cases)")
    else:
        print(
            f"Runtime: {total_seconds:.2f} s total, "
            f"{mean_seconds_per_case:.2f} s/case"
        )
    print(f"Masks: {args.output.resolve()}")
    print(f"Subtype CSV: {classification_path.resolve()}")
    if args.probability_csv is not None:
        print(f"Probability details: {args.probability_csv.resolve()}")
    if args.runtime_json is not None:
        print(f"Runtime JSON: {args.runtime_json.resolve()}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    # Required for Windows multiprocessing used internally by nnU-Net utilities.
    multiprocessing.freeze_support()
    raise SystemExit(main())
