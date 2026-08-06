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

from pancreas_multitask.inference_determinism import (
    assert_deterministic_inference,
    configure_deterministic_inference,
    deterministic_inference_snapshot,
    reassert_deterministic_inference,
)
from pancreas_multitask.neural_case_head import (
    CHECKPOINT_SHA256,
    DATASET_JSON_SHA256,
    PLANS_SHA256,
)
from pancreas_multitask.neural_case_predictor import (
    LOCKED_BATCH_CONFIGURATIONS,
    V5_CLASSIFIER_PIPELINE,
    NeuralCaseNNUNetPredictor,
)
from pancreas_multitask.predictor import JointNNUNetPredictor

RESCUE_CLASSIFIER_PIPELINE = "baseline_rescue_tile_probability_mean"
V5_CHECKPOINT_NAME = "checkpoint_classification_rescue.pth"
V5_MODEL_CONFIGURATION_FILENAMES = ("dataset.json", "plans.json")
V5_MODEL_CONFIGURATION_SHA256 = {
    "dataset.json": DATASET_JSON_SHA256,
    "plans.json": PLANS_SHA256,
}
DETERMINISTIC_INFERENCE_POLICY = "strict_cuda_inference_v1"
DETERMINISM_CONFORMANCE_LOCK_PATH = (
    ROOT / "configs" / "inference_determinism_conformance_v1.json"
)
DETERMINISM_CONFORMANCE_LOCK_SHA256 = (
    "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd"
)
STOCK_EXPORT_CONFORMANCE_LOCK_PATH = (
    ROOT / "configs" / "inference_stock_export_conformance_v1.json"
)
STOCK_EXPORT_CONFORMANCE_LOCK_SHA256 = (
    "bf309ae1ff8475b0985089ac1db2ef6b35383be34d7eeda0e9c6e63478f19503"
)
NNUNET_PREDICT_SOURCE_SHA256 = (
    "c350e3202a7a67c3aef12e9206a744add442110ff8a4377c1f9640104b20a31f"
)
V5_IMPLEMENTATION_RELATIVE_PATHS = (
    "scripts/predict_joint.py",
    "src/pancreas_multitask/classification_rescue.py",
    "src/pancreas_multitask/inference_determinism.py",
    "src/pancreas_multitask/network.py",
    "src/pancreas_multitask/predictor.py",
    "src/pancreas_multitask/case_features.py",
    "src/pancreas_multitask/case_feature_extractor.py",
    "src/pancreas_multitask/neural_case_head.py",
    "src/pancreas_multitask/neural_case_bundle.py",
    "src/pancreas_multitask/neural_case_training.py",
    "src/pancreas_multitask/neural_case_predictor.py",
)


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


def _locked_file_provenance(
    path: Path, *, expected_sha256: str, description: str
) -> dict[str, object]:
    """Bind an execution-policy dependency to an exact immutable file."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    digest = _sha256(resolved)
    if digest != expected_sha256:
        raise ValueError(f"{description} differs from its locked SHA-256")
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": resolved.stat().st_size,
    }


def _nnunet_predict_source_provenance() -> dict[str, object]:
    """Hash the installed stock predictor source without modifying it."""

    from nnunetv2.inference import predict_from_raw_data

    source_path = Path(predict_from_raw_data.__file__).resolve()
    return _locked_file_provenance(
        source_path,
        expected_sha256=NNUNET_PREDICT_SOURCE_SHA256,
        description="Installed nnUNetv2 prediction source",
    )


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


def _v5_model_configuration_provenance(
    model_directory: Path,
) -> list[dict[str, object]]:
    """Bind preprocessing to the exact nnU-Net dataset and plans artifacts."""

    provenance: list[dict[str, object]] = []
    for filename in V5_MODEL_CONFIGURATION_FILENAMES:
        path = model_directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"V5 model configuration does not exist: {path}")
        digest = _sha256(path)
        if digest != V5_MODEL_CONFIGURATION_SHA256[filename]:
            raise ValueError(f"V5 {filename} differs from the locked final artifact")
        provenance.append(
            {
                "name": filename,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return provenance


def _input_file_manifest(input_directory: Path) -> dict[str, object]:
    """Hash every raw NIfTI input outside the timed v3 inference region."""

    directory = input_directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Raw input directory does not exist: {directory}")
    paths = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.name.endswith(".nii.gz")),
        key=lambda path: path.name,
    )
    if not paths:
        raise FileNotFoundError(f"Raw input directory contains no .nii.gz files: {directory}")
    files = [
        {
            "name": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _case_ids_sha256(case_ids: list[str]) -> str:
    canonical = "".join(f"{case_id}\n" for case_id in sorted(case_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_argument(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("value must be one lowercase SHA-256 digest")
    return normalized


def _validate_v5_arguments(
    args: argparse.Namespace,
    selected_folds: tuple[int | str, ...] | None,
) -> None:
    bundle_values = (
        args.neural_case_head_bundle,
        args.expected_neural_case_head_bundle_sha256,
        args.expected_numeric_train_dataset_sha256,
    )
    if args.classification_mode != "neural-v5":
        if any(value is not None for value in bundle_values) or (
            args.v5_extraction_mode is not None
        ):
            raise ValueError(
                "Neural bundle arguments require --classification-mode neural-v5"
            )
        return
    if any(value is None for value in bundle_values):
        raise ValueError(
            "V5 inference requires bundle path, expected bundle SHA-256, and "
            "numeric training-dataset SHA-256"
        )
    if args.v5_extraction_mode is None:
        raise ValueError("V5 inference requires an explicit --v5-extraction-mode")
    if (
        selected_folds is None
        or selected_folds != (0,)
    ):
        raise ValueError("V5 inference requires exactly the explicit numeric fold 0")
    if args.checkpoint != V5_CHECKPOINT_NAME:
        raise ValueError(f"V5 inference requires --checkpoint {V5_CHECKPOINT_NAME}")
    if float(args.tile_step_size) != 0.5:
        raise ValueError("V5 inference requires --tile-step-size 0.5")
    if (args.tile_batch_size, args.tta_batch_size) not in LOCKED_BATCH_CONFIGURATIONS:
        raise ValueError("V5 inference permits only the locked tile1/TTA1 schedule")
    if args.disable_tta or args.disable_gaussian:
        raise ValueError("V5 inference requires mirror TTA and Gaussian weighting")
    if not args.overwrite:
        raise ValueError("V5 inference requires --overwrite so every case executes the head")


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
        "--classification-mode",
        choices=("rescue", "neural-v5"),
        default="rescue",
        help="Case classifier path (default: rescue; final pipeline: neural-v5)",
    )
    parser.add_argument(
        "--neural-case-head-bundle",
        type=Path,
        help="Final locked v5 neural case-head .pth bundle",
    )
    parser.add_argument(
        "--v5-extraction-mode",
        choices=("full", "neural_only"),
        help=(
            "V5 feature path: full locked reference or dependency-pruned "
            "neural_only candidate"
        ),
    )
    parser.add_argument(
        "--expected-neural-case-head-bundle-sha256",
        type=_sha256_argument,
        help="Required exact SHA-256 for the final v5 bundle",
    )
    parser.add_argument(
        "--expected-numeric-train-dataset-sha256",
        type=_sha256_argument,
        help="Required numeric-content hash of the 252-case training feature dataset",
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
    _validate_v5_arguments(args, selected_folds)
    device = torch.device(args.device)
    deterministic_initial: dict[str, object] | None = None
    deterministic_after_constructor: dict[str, object] | None = None
    determinism_lock_before: dict[str, object] | None = None
    stock_export_lock_before: dict[str, object] | None = None
    nnunet_source_before: dict[str, object] | None = None
    if args.classification_mode == "neural-v5":
        determinism_lock_before = _locked_file_provenance(
            DETERMINISM_CONFORMANCE_LOCK_PATH,
            expected_sha256=DETERMINISM_CONFORMANCE_LOCK_SHA256,
            description="Determinism conformance lock",
        )
        stock_export_lock_before = _locked_file_provenance(
            STOCK_EXPORT_CONFORMANCE_LOCK_PATH,
            expected_sha256=STOCK_EXPORT_CONFORMANCE_LOCK_SHA256,
            description="Stock export-dtype conformance lock",
        )
        nnunet_source_before = _nnunet_predict_source_provenance()
        deterministic_initial = configure_deterministic_inference(device)
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
    if args.classification_mode == "neural-v5" and (
        len(checkpoint_provenance) != 1
        or checkpoint_provenance[0]["fold"] != "0"
        or checkpoint_provenance[0]["sha256"] != CHECKPOINT_SHA256
    ):
        raise ValueError("V5 checkpoint differs from the locked fold-0 final artifact")
    model_configuration_provenance = (
        _v5_model_configuration_provenance(model_directory)
        if args.classification_mode == "neural-v5"
        else None
    )
    input_file_manifest = (
        _input_file_manifest(args.input)
        if args.classification_mode == "neural-v5"
        else None
    )
    predictor_type = (
        NeuralCaseNNUNetPredictor
        if args.classification_mode == "neural-v5"
        else JointNNUNetPredictor
    )
    predictor_arguments: dict[str, object] = {}
    if args.classification_mode == "neural-v5":
        predictor_arguments = {
            "neural_case_head_bundle": args.neural_case_head_bundle,
            "expected_neural_case_head_bundle_sha256": (
                args.expected_neural_case_head_bundle_sha256
            ),
            "expected_numeric_train_dataset_sha256": (
                args.expected_numeric_train_dataset_sha256
            ),
            "v5_extraction_mode": args.v5_extraction_mode,
        }
    predictor = predictor_type(
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
        **predictor_arguments,
    )
    if args.classification_mode == "neural-v5":
        # nnUNetPredictor.__init__ enables cuDNN benchmarking for CUDA. Undo
        # that upstream mutation before any model initialization or inference.
        deterministic_after_constructor = reassert_deterministic_inference()

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
    if isinstance(predictor, NeuralCaseNNUNetPredictor):
        predictor.load_final_neural_case_head()
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

    deterministic_after_inference: dict[str, object] | None = None
    determinism_lock_after: dict[str, object] | None = None
    stock_export_lock_after: dict[str, object] | None = None
    nnunet_source_after: dict[str, object] | None = None
    if isinstance(predictor, NeuralCaseNNUNetPredictor):
        deterministic_after_inference = deterministic_inference_snapshot(device)
        assert_deterministic_inference(deterministic_after_inference)
        determinism_lock_after = _locked_file_provenance(
            DETERMINISM_CONFORMANCE_LOCK_PATH,
            expected_sha256=DETERMINISM_CONFORMANCE_LOCK_SHA256,
            description="Determinism conformance lock",
        )
        stock_export_lock_after = _locked_file_provenance(
            STOCK_EXPORT_CONFORMANCE_LOCK_PATH,
            expected_sha256=STOCK_EXPORT_CONFORMANCE_LOCK_SHA256,
            description="Stock export-dtype conformance lock",
        )
        nnunet_source_after = _nnunet_predict_source_provenance()
        if determinism_lock_after != determinism_lock_before:
            raise RuntimeError("Determinism conformance lock changed during inference")
        if stock_export_lock_after != stock_export_lock_before:
            raise RuntimeError("Stock export-dtype conformance lock changed during inference")
        if nnunet_source_after != nnunet_source_before:
            raise RuntimeError("Installed nnUNetv2 prediction source changed during inference")
        checkpoint_provenance_after = _checkpoint_provenance(
            model_directory,
            selected_folds,
            args.checkpoint,
        )
        if checkpoint_provenance_after != checkpoint_provenance:
            raise RuntimeError("V5 checkpoint changed during inference")
        model_configuration_after = _v5_model_configuration_provenance(
            model_directory
        )
        if model_configuration_after != model_configuration_provenance:
            raise RuntimeError("V5 model configuration changed during inference")
        input_file_manifest_after = _input_file_manifest(args.input)
        if input_file_manifest_after != input_file_manifest:
            raise RuntimeError("Raw input files changed during v5 inference")

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

    inference_execution = predictor.inference_runtime_provenance()
    classifier_pipeline = (
        V5_CLASSIFIER_PIPELINE
        if isinstance(predictor, NeuralCaseNNUNetPredictor)
        else RESCUE_CLASSIFIER_PIPELINE
    )
    runtime = {
        "case_count": case_count,
        "case_ids": case_ids,
        "case_ids_sha256": _case_ids_sha256(case_ids),
        "checkpoint": args.checkpoint,
        "checkpoint_files": checkpoint_provenance,
        "classifier_pipeline": classifier_pipeline,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_capability": device_capability,
        "device_name": device_name,
        "folds": list(selected_folds) if selected_folds is not None else "auto",
        "gaussian_enabled": not args.disable_gaussian,
        "inference_execution": inference_execution,
        "mean_seconds_per_case": mean_seconds_per_case,
        "input_directory": str(args.input.expanduser().resolve()),
        "model_directory": str(model_directory),
        "overwrite": bool(args.overwrite),
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "process_id": os.getpid(),
        "python_version": platform.python_version(),
        "started_at_utc": started_at_utc,
        "tile_step_size": args.tile_step_size,
        "timing_scope": (
            "fresh_process_model_and_v5_head_initialization_preprocessing_"
            "feature_extraction_neural_head_offsets_export"
            if isinstance(predictor, NeuralCaseNNUNetPredictor)
            else "fresh_process_model_initialization_preprocessing_inference_export"
        ),
        "total_seconds": total_seconds,
        "torch_version": torch.__version__,
        "tta_enabled": not args.disable_tta,
        "warmup_policy": "none_fresh_process_end_to_end",
    }
    if isinstance(predictor, NeuralCaseNNUNetPredictor):
        if (
            deterministic_initial is None
            or deterministic_after_constructor is None
            or deterministic_after_inference is None
            or determinism_lock_before is None
            or stock_export_lock_before is None
            or nnunet_source_before is None
        ):
            raise RuntimeError("V5 deterministic inference provenance is incomplete")
        if (
            inference_execution["v5_case_extractions_completed"] != case_count
            or inference_execution["v5_neural_head_forward_calls"] != case_count
            or inference_execution["v5_class_offset_applications"] != case_count
            or inference_execution["v5_feature_cache_reads"] != 0
            or inference_execution["segmentation_export_logit_dtype"]
            != "torch.float16"
            or inference_execution["segmentation_export_logit_dtype_sequence"]
            != ["torch.float16"] * case_count
        ):
            raise RuntimeError("Not every output case executed the complete fresh v5 path")
        runtime.update(
            {
                "deterministic_execution": {
                    "policy": DETERMINISTIC_INFERENCE_POLICY,
                    "configured_before_cuda_initialization": True,
                    "autocast_cuda_float16": device.type == "cuda",
                    "after_initial_configuration": deterministic_initial,
                    "after_predictor_construction": deterministic_after_constructor,
                    "after_inference": deterministic_after_inference,
                    "settings_unchanged": (
                        deterministic_initial
                        == deterministic_after_constructor
                        == deterministic_after_inference
                    ),
                    "conformance_lock": {
                        **determinism_lock_before,
                        "unchanged_during_run": True,
                    },
                    "installed_nnunet_source": {
                        "before": nnunet_source_before,
                        "after": nnunet_source_after,
                        "unchanged_during_run": True,
                    },
                },
                "stock_export_conformance": {
                    "export_logit_dtype": "torch.float16",
                    "case_count_verified": case_count,
                    "all_case_exports_verified": True,
                    "conformance_lock": {
                        **stock_export_lock_before,
                        "unchanged_during_run": True,
                    },
                },
                "neural_case_head_bundle": predictor.neural_case_head_provenance(),
                "frozen_network": predictor.frozen_network_provenance(),
                "model_configuration_files": model_configuration_provenance,
                "model_configuration_unchanged_during_run": True,
                "checkpoint_unchanged_during_run": True,
                "input_file_manifest": input_file_manifest,
                "input_files_unchanged_during_run": True,
                "feature_cache_policy": "disabled_online_fresh_extraction",
                "v5_extraction_mode": args.v5_extraction_mode,
                "v5_neural_bag_sha256_sequence": inference_execution[
                    "v5_neural_bag_sha256_sequence"
                ],
                "class_probabilities": "v5_offset_adjusted_three_class",
                "case_identifiers_or_paths_used_as_model_inputs": False,
                "v5_implementation_files": {
                    relative_path: _sha256(ROOT / relative_path)
                    for relative_path in V5_IMPLEMENTATION_RELATIVE_PATHS
                },
            }
        )
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
