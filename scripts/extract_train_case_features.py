#!/usr/bin/env python3
"""Extract the prospectively locked case features from supplied training only."""

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

from pancreas_multitask.case_classifier_selection import (
    CaseFeatureDataset,
    identifier_independent_dataset_sha256,
    load_locked_search,
)
from pancreas_multitask.case_feature_extractor import (
    extract_case_from_preprocessed,
)
from pancreas_multitask.case_features import (
    build_case_feature_views,
    discover_train_cases,
    train_case_inventory_audit,
)
from pancreas_multitask.classification_rescue import (
    component_hashes,
    file_sha256,
)
from pancreas_multitask.predictor import JointNNUNetPredictor

EXPECTED_CHECKPOINT_SHA256 = "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116"
EXPECTED_V3_LOCK_SHA256 = "855e2be5a2dffa19902e4a81675bc8890801a023bd5e531dbb6fe0886c3c86d0"
EXPECTED_V5_NEURAL_LOCK_SHA256 = (
    "a8c2147493718acc96e4aa5dc471bf3f3277f0b99e8a8f7620bf966ab7b70d11"
)
EXPECTED_V5_DECISION_LOCK_SHA256 = (
    "e28a303c7d3da5dc7857ecc72787b6746d1e689e83167c500d4d2823c5ea540f"
)
EXPECTED_PLANS_SHA256 = "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f"
EXPECTED_DATASET_JSON_SHA256 = (
    "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff"
)
EXPECTED_COMPONENT_HASHES = {
    "encoder": "324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1",
    "decoder": "b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2",
    "classification": "1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8",
}


def _validate_static_lock_hashes(
    feature_lock: Path,
    neural_lock: Path,
    decision_lock: Path,
) -> None:
    expected = {
        feature_lock: EXPECTED_V3_LOCK_SHA256,
        neural_lock: EXPECTED_V5_NEURAL_LOCK_SHA256,
        decision_lock: EXPECTED_V5_DECISION_LOCK_SHA256,
    }
    for path, expected_hash in expected.items():
        if file_sha256(path) != expected_hash:
            raise ValueError(
                f"Caller-supplied prospective lock differs from its hash: {path.name}"
            )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
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
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary extraction file already exists: {temporary}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _schema_sha256(
    names_by_view: dict[str, tuple[str, ...]],
    tile_vector_names: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    for view_name in sorted(names_by_view):
        for value in (view_name, *names_by_view[view_name]):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    for value in tile_vector_names:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    """Bind caches to every source file that defines their numeric content."""

    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "pancreas_multitask" / "case_feature_extractor.py",
        ROOT / "src" / "pancreas_multitask" / "case_features.py",
        ROOT / "src" / "pancreas_multitask" / "network.py",
        ROOT / "src" / "pancreas_multitask" / "predictor.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _case_cache_name(case_id: str) -> str:
    # The readable ID is provenance only. A digest prevents path interpretation
    # and makes cache naming independent of label-bearing prefixes.
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return f"case_{digest}.npz"


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _cache_manifest_entry(case_id: str, path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "case_id": case_id,
            "cache_name": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "tile_vectors_sha256": _array_sha256(payload["tile_vectors"]),
            "tile_evidence_sha256": _array_sha256(payload["tile_evidence"]),
            "mil_stage3_maps_sha256": _array_sha256(payload["mil_stage3_maps"]),
            "mil_prediction_maps_sha256": _array_sha256(
                payload["mil_prediction_maps"]
            ),
            "mil_lesion_mass_sha256": _array_sha256(payload["mil_lesion_mass"]),
        }


def _load_case_cache(
    path: Path,
    *,
    case_id: str,
    label: int,
    view_names: tuple[str, str],
    expected_dimensions: dict[str, int],
    expected_tile_feature_count: int,
    expected_binding: dict[str, str | int | float | bool],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["case_id"].item()) != case_id:
            raise ValueError(f"Cache case-ID mismatch: {path}")
        if int(payload["label"].item()) != label:
            raise ValueError(f"Cache label mismatch: {path}")
        for key, expected in expected_binding.items():
            binding_key = f"binding_{key}"
            if binding_key not in payload.files:
                raise ValueError(f"Cache is missing provenance binding {key!r}: {path}")
            actual = payload[binding_key].item()
            if isinstance(expected, bool):
                matches = bool(actual) is expected
            elif isinstance(expected, int):
                matches = int(actual) == expected
            elif isinstance(expected, float):
                matches = float(actual) == expected
            else:
                matches = str(actual) == expected
            if not matches:
                raise ValueError(f"Cache provenance mismatch for {key!r}: {path}")
        views = {
            view_names[0]: np.asarray(payload["feature_view_0"], dtype=np.float32),
            view_names[1]: np.asarray(payload["feature_view_1"], dtype=np.float32),
        }
        for name, values in views.items():
            if values.shape != (expected_dimensions[name],) or not np.isfinite(values).all():
                raise ValueError(f"Invalid cached feature view {name!r}: {path}")
        mil = {
            "mil_stage3_maps": np.asarray(payload["mil_stage3_maps"], dtype=np.float16),
            "mil_prediction_maps": np.asarray(payload["mil_prediction_maps"], dtype=np.float16),
            "mil_lesion_mass": np.asarray(payload["mil_lesion_mass"], dtype=np.float32),
        }
        tile_vectors = np.asarray(payload["tile_vectors"], dtype=np.float32)
        tile_evidence = np.asarray(payload["tile_evidence"], dtype=np.float32)
        if (
            tile_vectors.ndim != 2
            or tile_vectors.shape[0] < 1
            or tile_vectors.shape[1] != expected_tile_feature_count
            or tile_evidence.shape != (tile_vectors.shape[0], 7)
            or not np.isfinite(tile_vectors).all()
            or not np.isfinite(tile_evidence).all()
            or mil["mil_stage3_maps"].ndim != 5
            or mil["mil_stage3_maps"].shape[0] not in (1, 2, 3)
            or mil["mil_stage3_maps"].shape[1:] != (256, 4, 4, 6)
            or mil["mil_prediction_maps"].shape
            != (mil["mil_stage3_maps"].shape[0], 2, 4, 4, 6)
            or mil["mil_lesion_mass"].shape != (mil["mil_stage3_maps"].shape[0],)
            or not np.isfinite(mil["mil_stage3_maps"]).all()
            or not np.isfinite(mil["mil_prediction_maps"]).all()
            or not np.isfinite(mil["mil_lesion_mass"]).all()
        ):
            raise ValueError(f"Invalid cached MIL bag: {path}")
    return views, mil


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frozen shared-encoder case features from supplied train only"
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        required=True,
        help="Isolated raw directory named train with subtype0/1/2 children",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", default="checkpoint_classification_rescue.pth")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "configs" / "phd_classification_upgrade_lock_v3.json",
    )
    parser.add_argument(
        "--neural-lock",
        type=Path,
        default=ROOT / "configs" / "phd_neural_case_head_lock_v5.json",
    )
    parser.add_argument(
        "--neural-decision-lock",
        type=Path,
        default=ROOT / "configs" / "phd_neural_decision_lock_v5.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--tile-step-size", type=float, default=0.5)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--expected-cases", type=int, default=252)
    parser.add_argument(
        "--smoke-case-id",
        help=(
            "Extract exactly one named training case after verifying the complete "
            "locked inventory; write smoke artifacts rather than a fit dataset"
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.fold < 0:
        raise ValueError("--fold must be non-negative")
    if float(args.tile_step_size) != 0.5:
        raise ValueError("The locked neural extraction requires --tile-step-size 0.5")
    if args.tile_batch_size < 1:
        raise ValueError("--tile-batch-size must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_directory = output / "case_cache"
    cache_directory.mkdir(exist_ok=True)
    lock_path = args.lock.expanduser().resolve()
    neural_lock_path = args.neural_lock.expanduser().resolve()
    neural_decision_lock_path = args.neural_decision_lock.expanduser().resolve()
    _validate_static_lock_hashes(
        lock_path,
        neural_lock_path,
        neural_decision_lock_path,
    )
    lock = load_locked_search(lock_path)
    neural_lock = json.loads(neural_lock_path.read_text(encoding="utf-8"))
    if neural_lock.get("lock_status") != (
        "frozen_before_any_eligible_case_feature_extraction_or_neural_head_oof_training"
    ):
        raise ValueError("Neural case-head lock is not frozen")
    neural_decision_lock = json.loads(
        neural_decision_lock_path.read_text(encoding="utf-8")
    )
    if neural_decision_lock["neural_head_lock"]["sha256"] != EXPECTED_V5_NEURAL_LOCK_SHA256:
        raise ValueError("V5 decision lock is not bound to the neural-head lock")
    view_names = tuple(lock["feature_extraction"]["feature_views"])
    if len(view_names) != 2:
        raise ValueError("The locked extractor requires exactly two feature views")

    all_cases = discover_train_cases(args.train_root, expected_count=args.expected_cases)
    inventory_audit = train_case_inventory_audit(all_cases)
    expected_case_hash = lock["development_boundary"]["case_ids_sha256_length_prefixed_sorted"]
    if inventory_audit["case_ids_sha256_length_prefixed_sorted"] != expected_case_hash:
        raise ValueError("Live isolated training inventory differs from the locked 252 cases")
    if inventory_audit["class_counts"] != lock["development_boundary"]["class_counts"]:
        raise ValueError("Live training class membership differs from the v3 lock")
    if (
        inventory_audit["case_ids_sha256_length_prefixed_sorted"]
        != neural_lock["development_boundary"][
            "case_ids_sha256_length_prefixed_sorted"
        ]
    ):
        raise ValueError("Live isolated training inventory differs from the neural lock")
    if inventory_audit["class_counts"] != neural_lock["development_boundary"][
        "class_counts"
    ]:
        raise ValueError("Live training class membership differs from the v5 neural lock")
    if args.smoke_case_id is None:
        cases = all_cases
    else:
        cases = tuple(case for case in all_cases if case.case_id == args.smoke_case_id)
        if len(cases) != 1:
            raise ValueError("--smoke-case-id must identify exactly one locked training case")

    model_directory = args.model.expanduser().resolve()
    plans_path = model_directory / "plans.json"
    dataset_json_path = model_directory / "dataset.json"
    if not plans_path.is_file() or not dataset_json_path.is_file():
        raise FileNotFoundError("Trained-model plans.json or dataset.json is missing")
    plans_hash = file_sha256(plans_path)
    dataset_json_hash = file_sha256(dataset_json_path)
    if plans_hash != EXPECTED_PLANS_SHA256 or dataset_json_hash != EXPECTED_DATASET_JSON_SHA256:
        raise ValueError("Live trained-model plans or dataset JSON differs from the lock")
    checkpoint_path = model_directory / f"fold_{args.fold}" / args.checkpoint
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Feature extraction checkpoint differs from the locked rescue file")
    if checkpoint_hash != lock["frozen_network"]["source_checkpoint_sha256"]:
        raise ValueError("Feature extraction checkpoint differs from the effective lock")
    if checkpoint_hash != neural_lock["frozen_multi_task_network"]["checkpoint_sha256"]:
        raise ValueError("Feature extraction checkpoint differs from the neural lock")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction requested, but CUDA is unavailable")
    torch.set_num_threads(1 if device.type == "cuda" else os.cpu_count() or 1)
    predictor = JointNNUNetPredictor(
        tile_step_size=args.tile_step_size,
        tile_batch_size=1,
        tta_batch_size=1,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_directory),
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint,
    )
    if tuple(predictor.allowed_mirroring_axes or ()) != (0, 1, 2):
        raise ValueError("The locked neural extraction requires mirror axes (0, 1, 2)")
    parameters = predictor.list_of_parameters
    if parameters is None or len(parameters) != 1:
        raise RuntimeError("Feature extraction requires exactly one explicit fold state")
    predictor.network.load_state_dict(parameters[0], strict=True)
    predictor.network.to(device)
    predictor.network.eval()
    component_hashes_before = component_hashes(predictor.network)
    if component_hashes_before != EXPECTED_COMPONENT_HASHES:
        raise ValueError("Loaded network component hashes differ from the locked model")

    feature_lock_hash = file_sha256(lock_path)
    neural_lock_hash = file_sha256(neural_lock_path)
    implementation_hash = _implementation_sha256()
    base_cache_binding: dict[str, str | int | float | bool] = {
        "checkpoint_sha256": checkpoint_hash,
        "feature_lock_sha256": feature_lock_hash,
        "neural_lock_sha256": neural_lock_hash,
        "implementation_sha256": implementation_hash,
        "neural_decision_lock_sha256": EXPECTED_V5_DECISION_LOCK_SHA256,
        "plans_sha256": plans_hash,
        "dataset_json_sha256": dataset_json_hash,
        "tile_step_size": float(args.tile_step_size),
        "tile_batch_size": int(args.tile_batch_size),
        "tta_enabled": True,
        "gaussian_enabled": True,
    }

    schema_path = output / "feature_schema.json"
    existing_schema: dict[str, Any] | None = None
    if schema_path.is_file():
        existing_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if existing_schema.get("cache_binding") != base_cache_binding:
            raise ValueError("Existing feature schema has incompatible cache provenance")
    elif any(cache_directory.glob("case_*.npz")):
        raise FileNotFoundError("Case cache exists without its feature-schema binding")

    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=args.verbose)
    feature_rows: dict[str, list[np.ndarray]] = {name: [] for name in view_names}
    labels: list[int] = []
    case_ids: list[str] = []
    source_manifest: list[dict[str, Any]] = []
    case_runtime_rows: list[dict[str, Any]] = []
    feature_names: dict[str, tuple[str, ...]] | None = None
    bound_tile_vector_names: tuple[str, ...] | None = None
    if existing_schema is not None:
        feature_names = {name: tuple(existing_schema["feature_names"][name]) for name in view_names}
        bound_tile_vector_names = tuple(existing_schema["tile_vector_names"])
        if _schema_sha256(
            feature_names, bound_tile_vector_names
        ) != existing_schema.get("feature_schema_sha256"):
            raise ValueError("Existing feature schema hash is invalid")
    if args.smoke_case_id is not None and args.resume:
        raise ValueError("One-case real-network smoke must use --no-resume")
    smoke_equivalence: dict[str, Any] | None = None
    started_at_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    for index, case in enumerate(cases, start=1):
        cache_path = cache_directory / _case_cache_name(case.case_id)
        case_started = time.perf_counter()
        image_hash = file_sha256(case.image_path)
        cache_hit = False
        if cache_path.is_file() and args.resume:
            if feature_names is None:
                raise RuntimeError("Cannot resume a cache before binding its schema")
            dimensions = {name: len(feature_names[name]) for name in view_names}
            if bound_tile_vector_names is None:
                raise RuntimeError("Cached feature schema has no tile-vector schema")
            cache_binding = {
                **base_cache_binding,
                "image_sha256": image_hash,
                "feature_schema_sha256": _schema_sha256(
                    feature_names, bound_tile_vector_names
                ),
            }
            views, _mil = _load_case_cache(
                cache_path,
                case_id=case.case_id,
                label=case.label,
                view_names=(view_names[0], view_names[1]),
                expected_dimensions=dimensions,
                expected_tile_feature_count=len(bound_tile_vector_names),
                expected_binding=cache_binding,
            )
            cache_hit = True
            tile_count = None
        else:
            data, _ignored_segmentation, _properties = preprocessor.run_case(
                [str(case.image_path)],
                None,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.dataset_json,
            )
            input_tensor = torch.from_numpy(np.asarray(data, dtype=np.float32)).contiguous()
            extraction = extract_case_from_preprocessed(
                predictor,
                input_tensor,
                tile_batch_size=args.tile_batch_size,
            )
            predicted_segmentation = (
                extraction.segmentation_logits.argmax(dim=0).numpy().astype(np.uint8)
            )
            built_views = build_case_feature_views(
                extraction.tile_vectors,
                extraction.tile_evidence,
                extraction.tile_vector_names,
                np.asarray(data[0], dtype=np.float32),
                predicted_segmentation,
                predictor.configuration_manager.spacing,
            )
            if tuple(built_views) != view_names:
                raise RuntimeError("Extractor feature views differ from the effective lock")
            if args.smoke_case_id is not None:
                production_reference = predictor.predict_sliding_window_return_joint(
                    input_tensor
                )
                reference_segmentation = (
                    production_reference.segmentation_logits.detach().float().cpu()
                )
                segmentation_equal = torch.equal(
                    extraction.segmentation_logits,
                    reference_segmentation,
                )
                maximum_logit_difference = float(
                    torch.max(
                        torch.abs(
                            extraction.segmentation_logits - reference_segmentation
                        )
                    ).item()
                )
                extracted_rescue_sum = np.zeros(3, dtype=np.float32)
                for probability_row in extraction.tile_vectors[:, -3:]:
                    extracted_rescue_sum += probability_row
                extracted_rescue_probabilities = (
                    extracted_rescue_sum / extraction.tile_count
                )
                reference_rescue_probabilities = (
                    production_reference.classification_probabilities.detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                probability_maximum_difference = float(
                    np.max(
                        np.abs(
                            extracted_rescue_probabilities
                            - reference_rescue_probabilities
                        )
                    )
                )
                if not segmentation_equal or probability_maximum_difference > 1e-6:
                    raise RuntimeError(
                        "Feature extractor does not match production batch-one inference"
                    )
                smoke_equivalence = {
                    "production_batch_one_segmentation_logits_exactly_equal": True,
                    "maximum_segmentation_logit_absolute_difference": maximum_logit_difference,
                    "predicted_segmentation_labels_exactly_equal": bool(
                        torch.equal(
                            extraction.segmentation_logits.argmax(dim=0),
                            reference_segmentation.argmax(dim=0),
                        )
                    ),
                    "rescue_probability_maximum_absolute_difference": (
                        probability_maximum_difference
                    ),
                    "rescue_probability_tolerance": 1e-6,
                }
            current_names = {name: built_views[name].names for name in view_names}
            if feature_names is None:
                feature_names = current_names
                bound_tile_vector_names = extraction.tile_vector_names
                schema_payload = {
                    "schema_version": 1,
                    "lock_path": lock_path.name,
                    "lock_sha256": file_sha256(lock_path),
                    "feature_names": {name: list(current_names[name]) for name in view_names},
                    "feature_dimensions": {name: len(current_names[name]) for name in view_names},
                    "tile_vector_names": list(bound_tile_vector_names),
                    "tile_vector_dimension": len(bound_tile_vector_names),
                    "feature_schema_sha256": _schema_sha256(
                        current_names, bound_tile_vector_names
                    ),
                    "cache_binding": base_cache_binding,
                    "identifiers_in_feature_schema": False,
                    "frozen_neural_head_included": True,
                    "ground_truth_masks_loaded": False,
                }
                _atomic_write_json(schema_path, schema_payload)
            elif (
                current_names != feature_names
                or extraction.tile_vector_names != bound_tile_vector_names
            ):
                raise RuntimeError("Feature schema changed between training cases")
            if bound_tile_vector_names is None:
                raise RuntimeError("Tile-vector schema was not bound")
            views = {name: built_views[name].values for name in view_names}
            cache_binding = {
                **base_cache_binding,
                "image_sha256": image_hash,
                "feature_schema_sha256": _schema_sha256(
                    feature_names, bound_tile_vector_names
                ),
            }
            _atomic_savez(
                cache_path,
                case_id=np.asarray(case.case_id),
                label=np.asarray(case.label, dtype=np.int64),
                feature_view_0=views[view_names[0]],
                feature_view_1=views[view_names[1]],
                tile_vectors=extraction.tile_vectors,
                tile_evidence=extraction.tile_evidence,
                mil_stage3_maps=extraction.mil_stage3_maps,
                mil_prediction_maps=extraction.mil_prediction_maps,
                mil_lesion_mass=extraction.mil_lesion_mass,
                **{
                    f"binding_{key}": np.asarray(value)
                    for key, value in cache_binding.items()
                },
            )
            tile_count = extraction.tile_count

        for name in view_names:
            feature_rows[name].append(np.asarray(views[name], dtype=np.float32))
        labels.append(case.label)
        case_ids.append(case.case_id)
        source_manifest.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "image_sha256": image_hash,
            }
        )
        case_runtime_rows.append(
            {
                "case_index": index,
                "case_id": case.case_id,
                "cache_hit": cache_hit,
                "tile_count": tile_count,
                "seconds": time.perf_counter() - case_started,
            }
        )
        if args.verbose or index == 1 or index % 10 == 0 or index == len(cases):
            print(
                f"Extracted training case {index}/{len(cases)} "
                f"(cache={'hit' if cache_hit else 'miss'})",
                flush=True,
            )

    if feature_names is None or bound_tile_vector_names is None:
        raise RuntimeError("No feature schema was produced")
    matrices = {
        name: np.stack(feature_rows[name]).astype(np.float32, copy=False) for name in view_names
    }
    dataset: CaseFeatureDataset | None
    if args.smoke_case_id is None:
        dataset = CaseFeatureDataset(
            tuple(case_ids),
            np.asarray(labels, dtype=np.int64),
            matrices,
            feature_names,
        )
        dataset_path = output / "train_case_features.npz"
    else:
        dataset = None
        dataset_path = output / "smoke_case_features.npz"
    _atomic_savez(
        dataset_path,
        case_ids=np.asarray(case_ids),
        labels=np.asarray(labels, dtype=np.int64),
        feature_view_0=matrices[view_names[0]],
        feature_view_1=matrices[view_names[1]],
    )
    expected_cache_paths = {
        _case_cache_name(case.case_id): case.case_id for case in cases
    }
    actual_cache_names = {path.name for path in cache_directory.glob("case_*.npz")}
    if actual_cache_names != set(expected_cache_paths):
        raise RuntimeError(
            "Cache set differs from the exact processed-case set: "
            f"missing={sorted(set(expected_cache_paths) - actual_cache_names)}, "
            f"extra={sorted(actual_cache_names - set(expected_cache_paths))}"
        )
    cache_manifest = [
        _cache_manifest_entry(case_id, cache_directory / cache_name)
        for cache_name, case_id in sorted(expected_cache_paths.items())
    ]
    cache_manifest_sha256 = hashlib.sha256(
        json.dumps(cache_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_manifest_sha256 = hashlib.sha256(
        json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    component_hashes_after = component_hashes(predictor.network)
    if component_hashes_after != component_hashes_before:
        raise RuntimeError("Frozen network changed during feature extraction")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 1024**2
    else:
        peak_allocated_mib = None
        peak_reserved_mib = None
    elapsed = time.perf_counter() - started
    audit = {
        "schema_version": 1,
        "status": "smoke_complete" if args.smoke_case_id is not None else "complete",
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "one_case_smoke_after_full_locked_train_inventory_verification"
            if args.smoke_case_id is not None
            else "isolated_supplied_train_only"
        ),
        "inventory": inventory_audit,
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_contains_local_paths": False,
        "checkpoint_name": args.checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "component_hashes_before": component_hashes_before,
        "component_hashes_after": component_hashes_after,
        "frozen_components_unchanged": True,
        "feature_lock_name": lock_path.name,
        "feature_lock_sha256": feature_lock_hash,
        "neural_lock_name": neural_lock_path.name,
        "neural_lock_sha256": neural_lock_hash,
        "implementation_sha256": implementation_hash,
        "cache_binding": base_cache_binding,
        "feature_schema_sha256": _schema_sha256(
            feature_names, bound_tile_vector_names
        ),
        "identifier_independent_dataset_sha256": (
            identifier_independent_dataset_sha256(dataset) if dataset is not None else None
        ),
        "dataset_npz_sha256": file_sha256(dataset_path),
        "case_cache_count": len(list(cache_directory.glob("case_*.npz"))),
        "cache_manifest": cache_manifest,
        "cache_manifest_sha256": cache_manifest_sha256,
        "cache_set_exact": True,
        "tile_batch_size": args.tile_batch_size,
        "tile_step_size": args.tile_step_size,
        "tta_enabled": True,
        "gaussian_enabled": True,
        "ground_truth_masks_loaded": False,
        "combined_train_validation_metadata_read": False,
        "official_validation_images_read": False,
        "official_validation_masks_read": False,
        "official_validation_labels_read": False,
        "test_data_read": False,
        "case_ids_paths_or_filenames_in_model_matrix": False,
        "elapsed_seconds": elapsed,
        "mean_seconds_per_case": elapsed / len(cases),
        "processed_case_count": len(cases),
        "smoke_case_id_recorded_for_audit_only": args.smoke_case_id,
        "smoke_production_equivalence": smoke_equivalence,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "case_runtime_rows": case_runtime_rows,
    }
    audit_path = output / (
        "smoke_case_feature_extraction_audit.json"
        if args.smoke_case_id is not None
        else "train_case_feature_extraction_audit.json"
    )
    _atomic_write_json(audit_path, audit)
    print(f"Feature dataset: {dataset_path}")
    print(f"Extraction audit: {audit_path}")
    return dataset_path


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
