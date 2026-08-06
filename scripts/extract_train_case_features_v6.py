#!/usr/bin/env python3
"""Extract the locked V6 whole-volume features from the isolated train tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.case_feature_extractor import FeatureExtractionRuntimeCounters
from pancreas_multitask.case_features import discover_train_cases, train_case_inventory_audit
from pancreas_multitask.classification_rescue import component_hashes, file_sha256
from pancreas_multitask.predictor import JointNNUNetPredictor
from pancreas_multitask.v6_case_extractor import (
    V6_H100_ORCHESTRATION_LOCK_SHA256,
    extract_v6_case_from_preprocessed,
)

EXPECTED_CHECKPOINT_SHA256 = "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116"
EXPECTED_PLANS_SHA256 = "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f"
EXPECTED_DATASET_JSON_SHA256 = (
    "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff"
)
EXPECTED_TRAIN_CASE_DIGEST = "bc9eee511612fce42d700b256d26793e6d2c8aabe06f4bf8699bb2b1abbf17bb"
EXPECTED_CLASS_COUNTS = {"0": 62, "1": 106, "2": 84}
EXPECTED_COMPONENT_HASHES = {
    "encoder": "324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1",
    "decoder": "b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2",
    "classification": "1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8",
}
CACHE_ARRAY_NAMES = (
    "spatial",
    "coordinates",
    "case_lesion_mass",
    "stage5_global_mean",
    "stage5_lesion_weighted_mean",
    "rescue_mean_logits",
    "rescue_mean_probabilities",
    "morphology",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary output exists: {temporary}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _numeric_dataset_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_array_sha256(arrays[name])))
    return digest.hexdigest()


def _cache_name(image_sha256: str) -> str:
    if len(image_sha256) != 64:
        raise ValueError("Training image SHA-256 is malformed")
    return f"case_{image_sha256}.npz"


def _load_cache(
    path: Path,
    *,
    expected_case_id: str,
    expected_label: int,
    expected_image_sha256: str,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], tuple[int, int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["case_id"].item()) != expected_case_id:
            raise ValueError(f"Cache case ID mismatch: {path}")
        if int(payload["label"].item()) != expected_label:
            raise ValueError(f"Cache label mismatch: {path}")
        if str(payload["image_sha256"].item()) != expected_image_sha256:
            raise ValueError(f"Cache image hash mismatch: {path}")
        arrays = {name: np.asarray(payload[name]) for name in CACHE_ARRAY_NAMES}
        names = tuple(str(value) for value in payload["morphology_names"].tolist())
        target_shape = tuple(int(value) for value in payload["target_shape"].tolist())
        stored_hashes = json.loads(str(payload["numeric_hashes_json"].item()))
    for name, value in arrays.items():
        if value.dtype != np.float16 or not np.isfinite(value).all():
            raise ValueError(f"Invalid float16 cache array {name}: {path}")
        if _array_sha256(value) != stored_hashes.get(name):
            raise ValueError(f"Cache numeric hash mismatch for {name}: {path}")
    return arrays, names, target_shape


def _environment() -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="checkpoint_classification_rescue.pth")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--expected-cases", type=int, default=252)
    parser.add_argument("--smoke-case-id")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.fold != 0 or args.checkpoint != "checkpoint_classification_rescue.pth":
        raise ValueError("V6 extraction is locked to fold 0 rescue checkpoint")
    lock_path = ROOT / "configs" / "v6_h100_orchestration_lock.json"
    if file_sha256(lock_path) != V6_H100_ORCHESTRATION_LOCK_SHA256:
        raise ValueError("V6 H100 orchestration lock hash differs from the code binding")
    train_root = args.train_root.expanduser().resolve()
    all_cases = discover_train_cases(train_root, expected_count=args.expected_cases)
    inventory = train_case_inventory_audit(all_cases)
    if (
        inventory["case_count"] != 252
        or inventory["class_counts"] != EXPECTED_CLASS_COUNTS
        or inventory["case_ids_sha256_length_prefixed_sorted"]
        != EXPECTED_TRAIN_CASE_DIGEST
    ):
        raise ValueError("Isolated training inventory differs from the locked 252 cases")
    cases = all_cases
    if args.smoke_case_id is not None:
        cases = tuple(case for case in all_cases if case.case_id == args.smoke_case_id)
        if len(cases) != 1:
            raise ValueError("--smoke-case-id must select one locked training case")
        if args.resume:
            raise ValueError("A one-case smoke must use --no-resume")

    model = args.model.expanduser().resolve()
    plans_path = model / "plans.json"
    dataset_path = model / "dataset.json"
    checkpoint_path = model / "fold_0" / args.checkpoint
    expected_files = {
        plans_path: EXPECTED_PLANS_SHA256,
        dataset_path: EXPECTED_DATASET_JSON_SHA256,
        checkpoint_path: EXPECTED_CHECKPOINT_SHA256,
    }
    for path, expected_hash in expected_files.items():
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"Missing or hash-mismatched locked model file: {path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction requested but CUDA is unavailable")
    torch.set_num_threads(1 if device.type == "cuda" else os.cpu_count() or 1)
    external_trainer_root = str((ROOT / "src").resolve())
    configured_external_root = os.environ.get("nnUNet_extTrainer")
    if configured_external_root not in (None, "", external_trainer_root):
        raise ValueError("nnUNet_extTrainer points outside the locked repository source")
    os.environ["nnUNet_extTrainer"] = external_trainer_root
    predictor = JointNNUNetPredictor(
        tile_step_size=0.5,
        tile_batch_size=1,
        tta_batch_size=1,
        use_gaussian=False,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model), use_folds=(0,), checkpoint_name=args.checkpoint
    )
    if tuple(predictor.allowed_mirroring_axes or ()) != (0, 1, 2):
        raise ValueError("The checkpoint plan does not permit all three mirror axes")
    parameters = predictor.list_of_parameters
    if parameters is None or len(parameters) != 1:
        raise RuntimeError("V6 extraction requires one explicit fold state")
    predictor.network.load_state_dict(parameters[0], strict=True)
    predictor.network.to(device).eval()
    components_before = component_hashes(predictor.network)
    if components_before != EXPECTED_COMPONENT_HASHES:
        raise ValueError("Loaded network components differ from the locked checkpoint")
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=args.verbose)

    output = args.output.expanduser().resolve()
    cache_root = output / "case_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    morphology_names: tuple[str, ...] | None = None
    started_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    for index, case in enumerate(cases, start=1):
        case_started = time.perf_counter()
        image_hash = file_sha256(case.image_path)
        cache_path = cache_root / _cache_name(image_hash)
        cache_hit = cache_path.is_file() and args.resume
        if cache_hit:
            arrays, names, target_shape = _load_cache(
                cache_path,
                expected_case_id=case.case_id,
                expected_label=case.label,
                expected_image_sha256=image_hash,
            )
        else:
            if cache_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing V6 cache: {cache_path}")
            data, _ignored_segmentation, _properties = preprocessor.run_case(
                [str(case.image_path)],
                None,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.dataset_json,
            )
            input_tensor = torch.from_numpy(np.asarray(data, dtype=np.float32)).contiguous()
            counters = FeatureExtractionRuntimeCounters(
                tile_batch_size_requested=1, tta_batch_size_requested=1
            )
            extracted = extract_v6_case_from_preprocessed(
                predictor,
                input_tensor,
                spacing=predictor.configuration_manager.spacing,
                runtime_counters=counters,
            )
            arrays = extracted.cache_arrays()
            names = extracted.morphology_names
            target_shape = extracted.target_shape
            _atomic_savez(
                cache_path,
                case_id=np.asarray(case.case_id),
                label=np.asarray(case.label, dtype=np.int64),
                image_sha256=np.asarray(image_hash),
                morphology_names=np.asarray(names),
                target_shape=np.asarray(target_shape, dtype=np.int64),
                numeric_hashes_json=np.asarray(
                    json.dumps(extracted.numeric_hashes(), sort_keys=True)
                ),
                **arrays,
            )
        if morphology_names is None:
            morphology_names = names
        elif morphology_names != names:
            raise RuntimeError("V6 morphology schema changed between cases")
        records.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "image_sha256": image_hash,
                "cache_path": cache_path,
                "cache_sha256": file_sha256(cache_path),
                "cache_hit": cache_hit,
                "target_shape": list(target_shape),
                "seconds": time.perf_counter() - case_started,
            }
        )
        if args.verbose or index == 1 or index % 10 == 0 or index == len(cases):
            print(f"V6 extraction {index}/{len(cases)}", flush=True)

    image_hashes = [row["image_sha256"] for row in records]
    if len(image_hashes) != len(set(image_hashes)):
        raise RuntimeError("Raw training CT hash collision; canonical order is undefined")
    records.sort(key=lambda row: row["image_sha256"])
    combined_rows: dict[str, list[np.ndarray]] = {name: [] for name in CACHE_ARRAY_NAMES}
    labels: list[int] = []
    case_ids: list[str] = []
    source_hashes: list[str] = []
    for row in records:
        arrays, names, _target = _load_cache(
            row["cache_path"],
            expected_case_id=row["case_id"],
            expected_label=row["label"],
            expected_image_sha256=row["image_sha256"],
        )
        if names != morphology_names:
            raise RuntimeError("Cached V6 morphology schemas differ")
        for name in CACHE_ARRAY_NAMES:
            combined_rows[name].append(arrays[name])
        labels.append(int(row["label"]))
        case_ids.append(str(row["case_id"]))
        source_hashes.append(str(row["image_sha256"]))
    combined = {
        name: np.stack(values).astype(np.float16, copy=False)
        for name, values in combined_rows.items()
    }
    combined["labels"] = np.asarray(labels, dtype=np.int64)
    numeric_hash = _numeric_dataset_sha256(combined)
    dataset_output = output / (
        "smoke_v6_features.npz" if args.smoke_case_id else "train_v6_features.npz"
    )
    if dataset_output.exists():
        raise FileExistsError(f"Refusing to overwrite combined feature dataset: {dataset_output}")
    _atomic_savez(
        dataset_output,
        **combined,
        case_ids=np.asarray(case_ids),
        source_image_sha256=np.asarray(source_hashes),
        morphology_names=np.asarray(morphology_names),
        numeric_dataset_sha256=np.asarray(numeric_hash),
        canonical_order_rule=np.asarray("ascending_sha256_of_raw_training_ct_bytes"),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    components_after = component_hashes(predictor.network)
    if components_after != components_before:
        raise RuntimeError("Frozen network parameters changed during V6 extraction")
    audit = {
        "schema_version": 1,
        "status": "complete",
        "started_at_utc": started_utc,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.0,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "environment": _environment(),
        "orchestration_lock_sha256": V6_H100_ORCHESTRATION_LOCK_SHA256,
        "train_inventory": inventory,
        "processed_case_count": len(records),
        "ground_truth_masks_opened": False,
        "validation_or_test_files_opened": False,
        "canonical_order_rule": "ascending_sha256_of_raw_training_ct_bytes",
        "case_manifest": [
            {key: value for key, value in row.items() if key != "cache_path"}
            for row in records
        ],
        "numeric_dataset_sha256": numeric_hash,
        "dataset_path": dataset_output.name,
        "dataset_sha256": file_sha256(dataset_output),
        "model_file_hashes": {
            "plans.json": EXPECTED_PLANS_SHA256,
            "dataset.json": EXPECTED_DATASET_JSON_SHA256,
            args.checkpoint: EXPECTED_CHECKPOINT_SHA256,
        },
        "network_component_hashes_before": components_before,
        "network_component_hashes_after": components_after,
    }
    _atomic_json(output / "extraction_audit.json", audit)
    return dataset_output


def main() -> int:
    path = run(build_parser().parse_args())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
