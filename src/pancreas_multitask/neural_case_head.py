"""Assignment-conforming neural case heads over frozen shared-encoder bags."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from pancreas_multitask.classification_rescue import file_sha256

NEURAL_MEAN_CANDIDATE = "neural_lesion_mean_mil"
NEURAL_ATTENTION_CANDIDATE = "neural_two_query_cross_attention_mil"
NEURAL_CANDIDATES = (NEURAL_MEAN_CANDIDATE, NEURAL_ATTENTION_CANDIDATE)
V5_NEURAL_LOCK_SHA256 = "a8c2147493718acc96e4aa5dc471bf3f3277f0b99e8a8f7620bf966ab7b70d11"
V5_DECISION_LOCK_SHA256 = "e28a303c7d3da5dc7857ecc72787b6746d1e689e83167c500d4d2823c5ea540f"
V3_FEATURE_LOCK_SHA256 = "855e2be5a2dffa19902e4a81675bc8890801a023bd5e531dbb6fe0886c3c86d0"
CHECKPOINT_SHA256 = "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116"
PLANS_SHA256 = "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f"
DATASET_JSON_SHA256 = "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff"
SPEED_LOCK_SHA256 = "d8bfa51b40de6676db76227442540a505bac2e5965cbe7f8c1fa1940669271dc"
EXTRACTOR_IMPLEMENTATION_SHA256 = "68956b493c8004b86558841d830e633827f12b0c9a099d3d42f8ddab8de2c46f"
TRAIN_CASE_IDS_SHA256 = "bc9eee511612fce42d700b256d26793e6d2c8aabe06f4bf8699bb2b1abbf17bb"
TRAIN_CLASS_COUNTS = {"0": 62, "1": 106, "2": 84}
FROZEN_COMPONENT_HASHES = {
    "encoder": "324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1",
    "decoder": "b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2",
    "classification": "1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8",
}
MAX_TILES = 3
STAGE_CHANNELS = 256
SPATIAL_GRID = (4, 4, 6)
SUMMARY_DIMENSION = 646


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _case_cache_name(case_id: str) -> str:
    return f"case_{hashlib.sha256(case_id.encode('utf-8')).hexdigest()}.npz"


def _length_prefixed_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NeuralCaseBag:
    stage3_maps: np.ndarray
    prediction_maps: np.ndarray
    lesion_mass: np.ndarray
    all_tile_summary: np.ndarray

    def __post_init__(self) -> None:
        stage3 = np.asarray(self.stage3_maps)
        predictions = np.asarray(self.prediction_maps)
        masses = np.asarray(self.lesion_mass)
        summary = np.asarray(self.all_tile_summary)
        if stage3.ndim != 5 or not 1 <= stage3.shape[0] <= MAX_TILES:
            raise ValueError("A neural case bag must contain one to three tiles")
        if stage3.shape[1:] != (STAGE_CHANNELS, *SPATIAL_GRID):
            raise ValueError("Stage-3 bag has an invalid shape")
        if predictions.shape != (stage3.shape[0], 2, *SPATIAL_GRID):
            raise ValueError("Prediction-map bag has an invalid shape")
        if masses.shape != (stage3.shape[0],):
            raise ValueError("Lesion masses are not aligned to bag tiles")
        if summary.shape != (SUMMARY_DIMENSION,):
            raise ValueError("All-tile summary must contain 646 values")
        if not all(np.isfinite(value).all() for value in (stage3, predictions, masses, summary)):
            raise ValueError("Neural case bag contains non-finite values")
        if (
            np.any(predictions < 0)
            or np.any(predictions > 1)
            or np.any(predictions[:, 1] < predictions[:, 0])
        ):
            raise ValueError("Prediction maps violate lesion/whole probability semantics")
        if np.any(masses < 0) or np.any(masses > 1):
            raise ValueError("Predicted tile lesion mass must be in [0, 1]")
        if masses.sum() <= 0 or np.any(np.diff(masses) > 0):
            raise ValueError("Ranked neural tiles require positive non-increasing lesion mass")


@dataclass(frozen=True, slots=True)
class NeuralBagDataset:
    case_ids: tuple[str, ...]
    labels: np.ndarray
    bags: tuple[NeuralCaseBag, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if labels.shape != (len(self.case_ids),) or len(self.bags) != len(self.case_ids):
            raise ValueError("Neural bags, labels, and IDs are not aligned")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("Neural case IDs must be unique")
        if set(labels.tolist()) != {0, 1, 2}:
            raise ValueError("Neural bag labels must contain exactly classes 0, 1, and 2")

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    def content_canonical_order(self) -> np.ndarray:
        """Sort by numeric bag content, never label-bearing identifiers."""

        keys: list[bytes] = []
        for label, bag in zip(self.labels, self.bags, strict=True):
            digest = hashlib.sha256()
            digest.update(np.asarray(label, dtype="<i8").tobytes())
            for value, dtype in (
                (bag.stage3_maps, "<f2"),
                (bag.prediction_maps, "<f2"),
                (bag.lesion_mass, "<f4"),
                (bag.all_tile_summary, "<f4"),
            ):
                digest.update(np.asarray(value, dtype=dtype).tobytes(order="C"))
            keys.append(digest.digest())
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Duplicate numeric case-bag digests would make input order a split key"
            )
        return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)

    def content_sha256(self) -> str:
        digests: list[bytes] = []
        for label, bag in zip(self.labels, self.bags, strict=True):
            digest = hashlib.sha256()
            digest.update(np.asarray(label, dtype="<i8").tobytes())
            for value in (
                bag.stage3_maps,
                bag.prediction_maps,
                bag.lesion_mass,
                bag.all_tile_summary,
            ):
                digest.update(bytes.fromhex(_array_sha256(np.asarray(value))))
            digests.append(digest.digest())
        result = hashlib.sha256()
        for value in sorted(digests):
            result.update(value)
        return result.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _feature_schema_sha256(
    names_by_view: Mapping[str, Sequence[str]],
    tile_vector_names: Sequence[str],
) -> str:
    """Recompute the extractor schema hash without importing its CLI module."""

    digest = hashlib.sha256()
    for view_name in sorted(names_by_view):
        for value in (view_name, *names_by_view[view_name]):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    for value in tile_vector_names:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_cache_binding(
    binding: Mapping[str, Any],
    neural_lock_hash: str,
) -> None:
    expected = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "feature_lock_sha256": V3_FEATURE_LOCK_SHA256,
        "neural_lock_sha256": neural_lock_hash,
        "implementation_sha256": EXTRACTOR_IMPLEMENTATION_SHA256,
        "neural_decision_lock_sha256": V5_DECISION_LOCK_SHA256,
        "speed_lock_sha256": SPEED_LOCK_SHA256,
        "plans_sha256": PLANS_SHA256,
        "dataset_json_sha256": DATASET_JSON_SHA256,
        "tile_step_size": 0.5,
        "tile_batch_size": 1,
        "tta_batch_size": 1,
        "network_batch_size_limit": 1,
        "locked_network_microbatch_ceiling": 2,
        "tta_enabled": True,
        "gaussian_enabled": True,
    }
    if dict(binding) != expected:
        raise ValueError("Extraction cache binding differs from exact v5 reference semantics")


def _validate_reference_execution(execution: Mapping[str, Any]) -> None:
    required = {
        "network_batch_size_limit": 1,
        "locked_network_microbatch_ceiling": 2,
        "maximum_network_batch_size_observed": 1,
        "tile_batch_size_requested": 1,
        "tile_batch_size_adaptive_limit": 1,
        "tta_batch_size_requested": 1,
        "tta_batch_size_adaptive_limit": 1,
        "tile_batch_oom_fallback_count": 0,
        "tta_batch_oom_fallback_count": 0,
    }
    if any(execution.get(key) != value for key, value in required.items()):
        raise ValueError("Extraction execution differs from reference batch-one semantics")
    forward_calls = int(execution.get("shared_network_forward_calls", 0))
    tiles = int(execution.get("logical_tiles_completed", 0))
    tile_batches = int(execution.get("logical_tile_batches_completed", 0))
    views = int(execution.get("tta_views_completed", 0))
    view_batches = int(execution.get("tta_view_batches_completed", 0))
    if (
        forward_calls < 1
        or execution.get("joint_network_forward_calls") != forward_calls
        or tile_batches != tiles
        or views != tiles * 8
        or view_batches != views
        or forward_calls != views
        or execution.get("network_batch_size_histogram") != {"1": forward_calls}
        or execution.get("tile_batch_size_histogram") != {"1": tile_batches}
        or execution.get("tta_batch_size_histogram") != {"1": view_batches}
    ):
        raise ValueError("Extraction execution counters do not prove eight-view batch-one TTA")


def _validate_extraction_audit(
    feature_directory: Path,
    audit: Mapping[str, Any],
    neural_lock_hash: str,
) -> None:
    if audit.get("status") != "complete" or audit.get("processed_case_count") != 252:
        raise ValueError("Neural training requires one complete 252-case extraction")
    if audit.get("scope") != "isolated_supplied_train_only":
        raise ValueError("Extraction audit is not restricted to supplied train only")
    if audit.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Extraction audit checkpoint differs from the neural lock")
    if audit.get("component_hashes_before") != FROZEN_COMPONENT_HASHES:
        raise ValueError("Extraction audit has unexpected frozen component hashes")
    if audit.get("component_hashes_after") != FROZEN_COMPONENT_HASHES:
        raise ValueError("Frozen component hashes changed during extraction")
    if audit.get("feature_lock_sha256") != V3_FEATURE_LOCK_SHA256:
        raise ValueError("Extraction audit is not bound to the v3 feature lock")
    if audit.get("neural_lock_sha256") != neural_lock_hash:
        raise ValueError("Extraction audit is not bound to this v5 neural lock")
    if audit.get("speed_lock_sha256") != SPEED_LOCK_SHA256:
        raise ValueError("Extraction audit is not bound to the speed lock")
    if audit.get("implementation_sha256") != EXTRACTOR_IMPLEMENTATION_SHA256:
        raise ValueError("Extraction audit differs from the final extractor implementation")
    _validate_cache_binding(audit.get("cache_binding", {}), neural_lock_hash)
    if (
        audit.get("tile_step_size") != 0.5
        or audit.get("tile_batch_size") != 1
        or audit.get("tta_batch_size") != 1
        or audit.get("tta_enabled") is not True
        or audit.get("gaussian_enabled") is not True
    ):
        raise ValueError("Extraction audit does not use exact reference inference semantics")
    execution = audit.get("inference_execution")
    if not isinstance(execution, Mapping):
        raise TypeError("Extraction audit lacks reference execution counters")
    _validate_reference_execution(execution)
    inventory = audit.get("inventory")
    if not isinstance(inventory, Mapping):
        raise TypeError("Extraction audit lacks a locked training inventory")
    if (
        inventory.get("case_count") != 252
        or inventory.get("class_counts") != TRAIN_CLASS_COUNTS
        or inventory.get("case_ids_sha256_length_prefixed_sorted") != TRAIN_CASE_IDS_SHA256
    ):
        raise ValueError("Extraction audit inventory differs from the v5 lock")
    if audit.get("cache_set_exact") is not True:
        raise ValueError("Extraction audit did not prove an exact cache set")
    if audit.get("source_manifest_contains_local_paths") is not False:
        raise ValueError("Extraction source manifest may not record local paths")
    forbidden_true = (
        "ground_truth_masks_loaded",
        "combined_train_validation_metadata_read",
        "official_validation_images_read",
        "official_validation_masks_read",
        "official_validation_labels_read",
        "test_data_read",
        "case_ids_paths_or_filenames_in_model_matrix",
    )
    if any(audit.get(field) is not False for field in forbidden_true):
        raise ValueError("Extraction audit violates the train-only feature boundary")
    source_manifest = audit.get("source_manifest")
    cache_manifest = audit.get("cache_manifest")
    if not isinstance(source_manifest, list) or len(source_manifest) != 252:
        raise ValueError("Extraction source manifest must contain exactly 252 cases")
    if not isinstance(cache_manifest, list) or len(cache_manifest) != 252:
        raise ValueError("Extraction cache manifest must contain exactly 252 cases")
    if _canonical_json_sha256(source_manifest) != audit.get("source_manifest_sha256"):
        raise ValueError("Extraction source-manifest hash is invalid")
    if _canonical_json_sha256(cache_manifest) != audit.get("cache_manifest_sha256"):
        raise ValueError("Extraction cache-manifest hash is invalid")
    dataset_path = feature_directory / "train_case_features.npz"
    if file_sha256(dataset_path) != audit.get("dataset_npz_sha256"):
        raise ValueError("Aggregated feature dataset differs from its extraction audit")


def _summary_indices(tile_names: Sequence[str]) -> tuple[list[int], list[int], list[int]]:
    lookup = {name: index for index, name in enumerate(tile_names)}
    rescue_names = [
        *(f"rescue_logit_class_{index}" for index in range(3)),
        *(f"rescue_probability_class_{index}" for index in range(3)),
    ]
    stage5_names = [f"encoder_stage_5_global_mean_channel_{channel:03d}" for channel in range(320)]
    missing = [name for name in (*rescue_names, *stage5_names) if name not in lookup]
    if missing:
        raise ValueError(f"Tile-vector schema lacks locked neural summary fields: {missing[:3]}")
    return (
        [lookup[name] for name in rescue_names],
        [lookup[name] for name in stage5_names],
        list(range(len(tile_names))),
    )


def build_neural_case_bag(
    *,
    tile_vectors: np.ndarray,
    tile_evidence: np.ndarray,
    tile_vector_names: Sequence[str],
    mil_stage3_maps: np.ndarray,
    mil_prediction_maps: np.ndarray,
    mil_lesion_mass: np.ndarray,
) -> NeuralCaseBag:
    """Build the identical v5 bag for cached training or online inference."""

    vectors = np.asarray(tile_vectors, dtype=np.float32)
    evidence = np.asarray(tile_evidence, dtype=np.float32)
    names = tuple(str(name) for name in tile_vector_names)
    if len(names) != len(set(names)):
        raise ValueError("Tile-vector schema contains duplicate fields")
    if (
        vectors.ndim != 2
        or vectors.shape[0] < 1
        or vectors.shape[1] != len(names)
        or evidence.shape != (vectors.shape[0], 7)
        or not np.isfinite(vectors).all()
        or not np.isfinite(evidence).all()
    ):
        raise ValueError("All-tile vectors or evidence violate the v5 bag contract")
    rescue_indices, stage5_indices, _ = _summary_indices(names)
    lesion_weight = np.maximum(evidence[:, 0], 0)
    if lesion_weight.sum() <= 0:
        raise ValueError("All-tile predicted lesion mass must be positive")
    lesion_weight /= lesion_weight.sum()
    rescue_summary = vectors[:, rescue_indices].mean(axis=0)
    stage5 = vectors[:, stage5_indices]
    stage5_uniform = stage5.mean(axis=0)
    stage5_lesion_weighted = np.sum(stage5 * lesion_weight[:, None], axis=0)
    summary = np.concatenate((rescue_summary, stage5_uniform, stage5_lesion_weighted)).astype(
        np.float32, copy=False
    )
    return NeuralCaseBag(
        np.asarray(mil_stage3_maps, dtype=np.float16),
        np.asarray(mil_prediction_maps, dtype=np.float16),
        np.asarray(mil_lesion_mass, dtype=np.float32),
        summary,
    )


def load_neural_bag_dataset(
    feature_directory: str | Path,
    neural_lock_path: str | Path,
) -> NeuralBagDataset:
    """Load and independently verify every cached spatial bag and binding."""

    directory = Path(feature_directory).expanduser().resolve()
    lock_path = Path(neural_lock_path).expanduser().resolve()
    neural_lock_hash = file_sha256(lock_path)
    if neural_lock_hash != V5_NEURAL_LOCK_SHA256:
        raise ValueError("Caller-supplied neural lock differs from its prospective hash")
    neural_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if neural_lock.get("lock_status") != (
        "frozen_before_any_eligible_case_feature_extraction_or_neural_head_oof_training"
    ):
        raise ValueError("Neural-head lock is not frozen")
    audit_path = directory / "train_case_feature_extraction_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _validate_extraction_audit(directory, audit, neural_lock_hash)
    schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("feature_schema_sha256") != audit.get("feature_schema_sha256"):
        raise ValueError("Feature schema is not bound to the extraction audit")
    if schema.get("cache_binding") != audit.get("cache_binding"):
        raise ValueError("Feature schema and audit cache bindings differ")
    if (
        schema.get("identifiers_in_feature_schema") is not False
        or schema.get("ground_truth_masks_loaded") is not False
    ):
        raise ValueError("Feature schema violates identifier-free model inputs")
    tile_names = tuple(schema["tile_vector_names"])
    if len(tile_names) != int(schema["tile_vector_dimension"]) or len(tile_names) != len(
        set(tile_names)
    ):
        raise ValueError("Tile-vector schema dimension is invalid")
    names_by_view = {
        str(view): tuple(str(name) for name in names)
        for view, names in schema["feature_names"].items()
    }
    if any(len(names) != len(set(names)) for names in names_by_view.values()):
        raise ValueError("Feature schema contains duplicate numeric field names")
    if _feature_schema_sha256(names_by_view, tile_names) != schema.get("feature_schema_sha256"):
        raise ValueError("Feature schema content hash is invalid")
    _summary_indices(tile_names)

    with np.load(directory / "train_case_features.npz", allow_pickle=False) as payload:
        case_ids = tuple(str(value) for value in payload["case_ids"].tolist())
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if len(case_ids) != 252 or labels.shape != (252,):
        raise ValueError("Aggregated feature dataset does not contain 252 cases")
    if _length_prefixed_sha256(case_ids) != TRAIN_CASE_IDS_SHA256:
        raise ValueError("Aggregated feature case-ID digest differs from the v5 lock")
    if {str(label): int(np.sum(labels == label)) for label in range(3)} != neural_lock[
        "development_boundary"
    ]["class_counts"]:
        raise ValueError("Cached labels differ from the locked class counts")
    source_by_id = {row["case_id"]: row for row in audit["source_manifest"]}
    cache_by_id = {row["case_id"]: row for row in audit["cache_manifest"]}
    if set(source_by_id) != set(case_ids) or set(cache_by_id) != set(case_ids):
        raise ValueError("Source/cache manifests do not exactly match dataset case IDs")
    if any(
        int(source_by_id[case_id]["label"]) != int(label)
        for case_id, label in zip(case_ids, labels, strict=True)
    ):
        raise ValueError("Source-manifest labels differ from the aggregated dataset")
    cache_directory = directory / "case_cache"
    expected_names = {_case_cache_name(case_id) for case_id in case_ids}
    actual_names = {path.name for path in cache_directory.glob("case_*.npz")}
    if actual_names != expected_names:
        raise ValueError("Live cache directory has missing or extra case artifacts")

    bags: list[NeuralCaseBag] = []
    for case_id, label in zip(case_ids, labels, strict=True):
        cache_path = cache_directory / _case_cache_name(case_id)
        manifest = cache_by_id[case_id]
        if (
            manifest["cache_name"] != cache_path.name
            or manifest["bytes"] != cache_path.stat().st_size
            or manifest["sha256"] != file_sha256(cache_path)
        ):
            raise ValueError(f"Cache artifact differs from manifest for {case_id}")
        with np.load(cache_path, allow_pickle=False) as payload:
            if str(payload["case_id"].item()) != case_id:
                raise ValueError("Cache ID join failed")
            if int(payload["label"].item()) != int(label):
                raise ValueError("Cache label join failed")
            expected_binding = {
                **audit["cache_binding"],
                "image_sha256": source_by_id[case_id]["image_sha256"],
                "feature_schema_sha256": audit["feature_schema_sha256"],
            }
            for key, expected in expected_binding.items():
                binding_key = f"binding_{key}"
                if binding_key not in payload.files or payload[binding_key].item() != expected:
                    raise ValueError(f"Cache binding {key!r} failed for {case_id}")
            arrays = {
                "tile_vectors": np.asarray(payload["tile_vectors"], dtype=np.float32),
                "tile_evidence": np.asarray(payload["tile_evidence"], dtype=np.float32),
                "mil_stage3_maps": np.asarray(payload["mil_stage3_maps"], dtype=np.float16),
                "mil_prediction_maps": np.asarray(payload["mil_prediction_maps"], dtype=np.float16),
                "mil_lesion_mass": np.asarray(payload["mil_lesion_mass"], dtype=np.float32),
            }
        for name, values in arrays.items():
            if _array_sha256(values) != manifest[f"{name}_sha256"]:
                raise ValueError(f"Cached spatial input hash failed for {case_id}: {name}")
        tile_vectors = arrays["tile_vectors"]
        tile_evidence = arrays["tile_evidence"]
        if tile_vectors.ndim != 2 or tile_vectors.shape[1] != len(tile_names):
            raise ValueError("Cached tile-vector matrix differs from its schema")
        if tile_evidence.shape != (tile_vectors.shape[0], 7):
            raise ValueError("Cached tile evidence is not aligned")
        bags.append(
            build_neural_case_bag(
                tile_vectors=tile_vectors,
                tile_evidence=tile_evidence,
                tile_vector_names=tile_names,
                mil_stage3_maps=arrays["mil_stage3_maps"],
                mil_prediction_maps=arrays["mil_prediction_maps"],
                mil_lesion_mass=arrays["mil_lesion_mass"],
            )
        )

    dataset = NeuralBagDataset(
        case_ids,
        labels,
        tuple(bags),
        {
            "feature_directory_recorded": False,
            "extraction_audit_sha256": file_sha256(audit_path),
            "cache_manifest_sha256": audit["cache_manifest_sha256"],
            "source_manifest_sha256": audit["source_manifest_sha256"],
            "feature_schema_sha256": audit["feature_schema_sha256"],
            "neural_lock_sha256": neural_lock_hash,
            "decision_lock_sha256": V5_DECISION_LOCK_SHA256,
            "speed_lock_sha256": SPEED_LOCK_SHA256,
            "extractor_implementation_sha256": EXTRACTOR_IMPLEMENTATION_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "plans_sha256": PLANS_SHA256,
            "dataset_json_sha256": DATASET_JSON_SHA256,
            "frozen_component_hashes": dict(FROZEN_COMPONENT_HASHES),
            "reference_tile_batch_size": 1,
            "reference_tta_batch_size": 1,
            "reference_network_batch_size_limit": 1,
            "locked_network_microbatch_ceiling": 2,
            "tta_enabled": True,
            "gaussian_enabled": True,
            "rescue_checkpoint_previously_selected_on_official_validation": True,
            "head_oof_not_unbiased_end_to_end": True,
        },
    )
    if dataset.case_count != int(neural_lock["development_boundary"]["case_count"]):
        raise ValueError("Verified neural dataset count differs from the lock")
    return dataset


def collate_neural_bags(
    dataset: NeuralBagDataset,
    indices: Sequence[int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pad up to three tiles and move one numeric batch to the target device."""

    if not indices:
        raise ValueError("Cannot collate an empty neural bag batch")
    batch_size = len(indices)
    stage3 = torch.zeros(
        (batch_size, MAX_TILES, STAGE_CHANNELS, *SPATIAL_GRID), dtype=torch.float32
    )
    predictions = torch.zeros((batch_size, MAX_TILES, 2, *SPATIAL_GRID), dtype=torch.float32)
    masses = torch.zeros((batch_size, MAX_TILES), dtype=torch.float32)
    valid = torch.zeros((batch_size, MAX_TILES), dtype=torch.bool)
    summary = torch.zeros((batch_size, SUMMARY_DIMENSION), dtype=torch.float32)
    labels = torch.empty(batch_size, dtype=torch.long)
    for row, index in enumerate(indices):
        bag = dataset.bags[int(index)]
        count = bag.stage3_maps.shape[0]
        stage3[row, :count] = torch.from_numpy(bag.stage3_maps.astype(np.float32))
        predictions[row, :count] = torch.from_numpy(bag.prediction_maps.astype(np.float32))
        masses[row, :count] = torch.from_numpy(bag.lesion_mass.astype(np.float32))
        valid[row, :count] = True
        summary[row] = torch.from_numpy(bag.all_tile_summary.astype(np.float32))
        labels[row] = int(dataset.labels[int(index)])
    return tuple(
        value.to(device, non_blocking=device.type == "cuda")
        for value in (stage3, predictions, masses, valid, summary, labels)
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class NeuralBagTensors:
    """One float32 materialization reused by all deterministic trajectories."""

    stage3_maps: Tensor
    prediction_maps: Tensor
    lesion_mass: Tensor
    valid_tiles: Tensor
    all_tile_summary: Tensor
    labels: Tensor

    def __post_init__(self) -> None:
        case_count = int(self.labels.numel())
        if self.stage3_maps.shape != (
            case_count,
            MAX_TILES,
            STAGE_CHANNELS,
            *SPATIAL_GRID,
        ):
            raise ValueError("Materialized stage-3 tensor has an invalid shape")
        if self.prediction_maps.shape != (
            case_count,
            MAX_TILES,
            2,
            *SPATIAL_GRID,
        ):
            raise ValueError("Materialized prediction tensor has an invalid shape")
        if self.lesion_mass.shape != (case_count, MAX_TILES):
            raise ValueError("Materialized lesion masses have an invalid shape")
        if self.valid_tiles.shape != (case_count, MAX_TILES):
            raise ValueError("Materialized tile mask has an invalid shape")
        if self.all_tile_summary.shape != (case_count, SUMMARY_DIMENSION):
            raise ValueError("Materialized summary tensor has an invalid shape")
        devices = {
            value.device
            for value in (
                self.stage3_maps,
                self.prediction_maps,
                self.lesion_mass,
                self.valid_tiles,
                self.all_tile_summary,
                self.labels,
            )
        }
        if len(devices) != 1:
            raise ValueError("Materialized neural bags must share one device")

    @property
    def device(self) -> torch.device:
        return self.labels.device

    @property
    def case_count(self) -> int:
        return int(self.labels.numel())

    def batch(self, indices: Sequence[int] | Tensor) -> tuple[Tensor, ...]:
        index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if index.ndim != 1 or index.numel() < 1:
            raise ValueError("A materialized neural batch requires at least one index")
        return (
            self.stage3_maps.index_select(0, index),
            self.prediction_maps.index_select(0, index),
            self.lesion_mass.index_select(0, index),
            self.valid_tiles.index_select(0, index),
            self.all_tile_summary.index_select(0, index),
            self.labels.index_select(0, index),
        )


def materialize_neural_bags(
    dataset: NeuralBagDataset,
    device: torch.device,
) -> NeuralBagTensors:
    """Convert immutable float16 caches to the locked float32 compute dtype once."""

    tensors = collate_neural_bags(dataset, range(dataset.case_count), device)
    result = NeuralBagTensors(*tensors)
    if result.stage3_maps.dtype != torch.float32:
        raise RuntimeError("Neural training bags must use float32 compute tensors")
    return result


def neural_bag_inference_tensors(
    bag: NeuralCaseBag,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Materialize one label-free online bag with the training tensor contract."""

    count = int(bag.stage3_maps.shape[0])
    stage3 = torch.zeros((1, MAX_TILES, STAGE_CHANNELS, *SPATIAL_GRID))
    predictions = torch.zeros((1, MAX_TILES, 2, *SPATIAL_GRID))
    masses = torch.zeros((1, MAX_TILES))
    valid = torch.zeros((1, MAX_TILES), dtype=torch.bool)
    summary = torch.from_numpy(bag.all_tile_summary.astype(np.float32))[None]
    stage3[0, :count] = torch.from_numpy(bag.stage3_maps.astype(np.float32))
    predictions[0, :count] = torch.from_numpy(bag.prediction_maps.astype(np.float32))
    masses[0, :count] = torch.from_numpy(bag.lesion_mass.astype(np.float32))
    valid[0, :count] = True
    return tuple(
        value.to(device, non_blocking=device.type == "cuda")
        for value in (stage3, predictions, masses, valid, summary)
    )  # type: ignore[return-value]


class _SharedNeuralCaseFrontend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_projection = nn.Sequential(
            nn.Conv3d(258, 64, kernel_size=1, bias=True),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.summary_projection = nn.Sequential(
            nn.LayerNorm(SUMMARY_DIMENSION),
            nn.Linear(SUMMARY_DIMENSION, 64),
            nn.GELU(),
        )

    def _project(
        self,
        stage3: Tensor,
        prediction_maps: Tensor,
        summary: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if stage3.ndim != 6 or prediction_maps.shape[:2] != stage3.shape[:2]:
            raise ValueError("Neural head received an invalid padded bag")
        batch_size, tile_count = stage3.shape[:2]
        spatial = torch.cat((stage3, prediction_maps), dim=2).reshape(
            batch_size * tile_count, 258, *SPATIAL_GRID
        )
        projected = self.spatial_projection(spatial).reshape(
            batch_size, tile_count, 64, *SPATIAL_GRID
        )
        return projected, self.summary_projection(summary)


class NeuralLesionMeanMIL(_SharedNeuralCaseFrontend):
    """Small neural mean-pooling control from the immutable v5 lock."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(448, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        stage3: Tensor,
        prediction_maps: Tensor,
        lesion_mass: Tensor,
        valid_tiles: Tensor,
        summary: Tensor,
    ) -> Tensor:
        projected, summary_vector = self._project(stage3, prediction_maps, summary)
        global_mean = projected.mean(dim=(3, 4, 5))
        lesion = prediction_maps[:, :, 0:1]
        whole = prediction_maps[:, :, 1:2]

        def spatial_weighted(weight: Tensor) -> Tensor:
            denominator = weight.sum(dim=(3, 4, 5)).clamp_min(1e-6)
            return (projected * weight).sum(dim=(3, 4, 5)) / denominator

        descriptors = torch.cat(
            (global_mean, spatial_weighted(lesion), spatial_weighted(whole)), dim=2
        )
        valid_float = valid_tiles.float()
        uniform = (descriptors * valid_float[:, :, None]).sum(dim=1) / valid_float.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        tile_weight = lesion_mass.clamp_min(0) * valid_float
        tile_weight = tile_weight / tile_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        lesion_weighted = (descriptors * tile_weight[:, :, None]).sum(dim=1)
        return self.classifier(torch.cat((uniform, lesion_weighted, summary_vector), dim=1))


class NeuralTwoQueryCrossAttentionMIL(_SharedNeuralCaseFrontend):
    """Two-query, four-head cross-attention case classifier from v5."""

    def __init__(self) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, 2, 64))
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(64)
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 3),
        )
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(
        self,
        stage3: Tensor,
        prediction_maps: Tensor,
        lesion_mass: Tensor,
        valid_tiles: Tensor,
        summary: Tensor,
    ) -> Tensor:
        projected, summary_vector = self._project(stage3, prediction_maps, summary)
        batch_size, tile_count = projected.shape[:2]
        tokens_per_tile = int(np.prod(SPATIAL_GRID))
        tokens = (
            projected.flatten(start_dim=3)
            .transpose(2, 3)
            .reshape(batch_size, tile_count * tokens_per_tile, 64)
        )
        token_valid = (
            valid_tiles[:, :, None]
            .expand(batch_size, tile_count, tokens_per_tile)
            .reshape(batch_size, tile_count * tokens_per_tile)
        )
        tile_prior = 0.5 * torch.log(lesion_mass.clamp_min(0) + 1e-6)
        token_prior = (
            tile_prior[:, :, None]
            .expand(batch_size, tile_count, tokens_per_tile)
            .reshape(batch_size, tile_count * tokens_per_tile)
        )
        token_prior = token_prior.masked_fill(~token_valid, -torch.inf)
        attention_mask = (
            token_prior[:, None, None, :]
            .expand(batch_size, 4, 2, tile_count * tokens_per_tile)
            .reshape(batch_size * 4, 2, tile_count * tokens_per_tile)
        )
        query = self.queries.expand(batch_size, -1, -1)
        attended, _ = self.attention(
            query,
            tokens,
            tokens,
            attn_mask=attention_mask,
            need_weights=False,
        )
        attended = self.attention_norm(attended + query).reshape(batch_size, 128)
        return self.classifier(torch.cat((attended, summary_vector), dim=1))


def build_neural_case_head(candidate_id: str) -> nn.Module:
    if candidate_id == NEURAL_MEAN_CANDIDATE:
        model: nn.Module = NeuralLesionMeanMIL()
    elif candidate_id == NEURAL_ATTENTION_CANDIDATE:
        model = NeuralTwoQueryCrossAttentionMIL()
    else:
        raise ValueError(f"Unknown locked neural candidate: {candidate_id}")
    trainable = sum(parameter.numel() for parameter in model.parameters())
    exact_count = {
        NEURAL_MEAN_CANDIDATE: 117_263,
        NEURAL_ATTENTION_CANDIDATE: 101_391,
    }[candidate_id]
    if trainable != exact_count or trainable > 150_000:
        raise RuntimeError(
            "Neural head parameter count differs from the audited lock: "
            f"expected {exact_count}, got {trainable}"
        )
    return model


def neural_head_parameter_count(candidate_id: str) -> int:
    return sum(parameter.numel() for parameter in build_neural_case_head(candidate_id).parameters())


__all__ = [
    "CHECKPOINT_SHA256",
    "FROZEN_COMPONENT_HASHES",
    "MAX_TILES",
    "NEURAL_ATTENTION_CANDIDATE",
    "NEURAL_CANDIDATES",
    "NEURAL_MEAN_CANDIDATE",
    "V3_FEATURE_LOCK_SHA256",
    "V5_DECISION_LOCK_SHA256",
    "V5_NEURAL_LOCK_SHA256",
    "NeuralBagDataset",
    "NeuralBagTensors",
    "NeuralCaseBag",
    "NeuralLesionMeanMIL",
    "NeuralTwoQueryCrossAttentionMIL",
    "build_neural_case_bag",
    "build_neural_case_head",
    "collate_neural_bags",
    "load_neural_bag_dataset",
    "materialize_neural_bags",
    "neural_bag_inference_tensors",
    "neural_head_parameter_count",
]
