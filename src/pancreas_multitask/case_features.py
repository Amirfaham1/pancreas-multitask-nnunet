"""Train-only, inference-matched case features for subtype classification.

The feature path is deliberately separate from subtype identifiers.  Case IDs,
filenames, directory names, and enumeration order are retained only for audit
and output joins; they are never appended to a model matrix.  Dense features
come from frozen shared-encoder skips and the network's own segmentation
probabilities.  Ground-truth segmentation arrays are neither accepted nor
loaded by this module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import kurtosis, skew
from torch import Tensor

TRAIN_CLASS_DIRECTORIES: Mapping[str, int] = {
    "subtype0": 0,
    "subtype1": 1,
    "subtype2": 2,
}
NNUNET_CHANNEL_SUFFIX = "_0000.nii.gz"
NIFTI_SUFFIX = ".nii.gz"
LOCKED_ENCODER_STAGES = (2, 3, 4, 5)
LOCKED_ENCODER_CHANNELS = (128, 256, 320, 320)
MODEL_EVIDENCE_NAMES = (
    "lesion_probability_mean",
    "lesion_probability_max",
    "lesion_argmax_fraction",
    "whole_pancreas_probability_mean",
    "tile_center_x_normalized",
    "tile_center_y_normalized",
    "tile_center_z_normalized",
)
FROZEN_NEURAL_HEAD_NAMES = (
    "rescue_logit_class_0",
    "rescue_logit_class_1",
    "rescue_logit_class_2",
    "rescue_probability_class_0",
    "rescue_probability_class_1",
    "rescue_probability_class_2",
)


@dataclass(frozen=True, slots=True)
class TrainCase:
    """One case discovered exclusively below the supplied ``train`` root."""

    case_id: str
    label: int
    image_path: Path


@dataclass(frozen=True, slots=True)
class FeatureView:
    """A numeric model vector and its identifier-free schema."""

    values: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 1:
            raise ValueError(f"Feature values must be one-dimensional, got {values.shape}")
        if len(self.names) != values.size:
            raise ValueError("Feature-name count does not match the numeric vector")
        if len(self.names) != len(set(self.names)):
            raise ValueError("Feature names must be unique")
        if not np.isfinite(values).all():
            raise ValueError("Feature values must all be finite")


def _length_prefixed_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def discover_train_cases(
    train_root: str | Path,
    *,
    expected_count: int | None = 252,
) -> tuple[TrainCase, ...]:
    """Discover labels from an isolated raw ``train`` tree only.

    The returned ID/path fields are provenance, not features.  Requiring the
    exact three expected child directories prevents a caller from accidentally
    pointing this function at a combined train/validation dataset.
    """

    root = Path(train_root).expanduser().resolve()
    if not root.is_dir() or root.name.casefold() != "train":
        raise ValueError("train_root must be an existing directory named 'train'")
    visible_directories = {
        item.name for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")
    }
    if visible_directories != set(TRAIN_CLASS_DIRECTORIES):
        raise ValueError("The isolated train root must contain exactly subtype0/subtype1/subtype2")

    cases: list[TrainCase] = []
    seen: set[str] = set()
    for directory_name, label in TRAIN_CLASS_DIRECTORIES.items():
        class_root = root / directory_name
        for image_path in sorted(class_root.glob(f"*{NNUNET_CHANNEL_SUFFIX}")):
            case_id = image_path.name[: -len(NNUNET_CHANNEL_SUFFIX)]
            if not case_id or case_id in seen:
                raise ValueError(f"Missing or duplicate training case ID: {case_id!r}")
            if any(character in case_id for character in ("/", "\\", "\0")):
                raise ValueError(f"Unsafe training case ID: {case_id!r}")
            seen.add(case_id)
            cases.append(TrainCase(case_id, label, image_path.resolve()))

    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} train cases, found {len(cases)}")
    labels = np.asarray([case.label for case in cases], dtype=np.int64)
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("Training inventory must contain all three subtype labels")
    return tuple(sorted(cases, key=lambda item: item.case_id))


def train_case_inventory_audit(cases: Sequence[TrainCase]) -> dict[str, object]:
    """Return identifier provenance without constructing numeric features."""

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Training case IDs are not unique")
    counts = {str(label): sum(case.label == label for case in cases) for label in range(3)}
    return {
        "case_count": len(cases),
        "class_counts": counts,
        "case_ids_sha256_length_prefixed_sorted": _length_prefixed_sha256(ids),
        "case_ids_excluded_from_model_matrix": True,
        "paths_excluded_from_model_matrix": True,
        "enumeration_order_excluded_from_model_matrix": True,
        "combined_train_validation_metadata_read": False,
        "ground_truth_masks_opened": False,
    }


def _validate_encoder_inputs(
    skips: Sequence[Tensor],
    segmentation_logits: Tensor,
    stage_indices: Sequence[int],
) -> None:
    if segmentation_logits.ndim != 5 or segmentation_logits.shape[1] != 3:
        raise ValueError("segmentation_logits must have shape (batch, 3, depth, height, width)")
    if not stage_indices:
        raise ValueError("At least one encoder stage is required")
    if min(stage_indices) < 0 or max(stage_indices) >= len(skips):
        raise ValueError("Requested encoder stage is absent")
    batch_size = segmentation_logits.shape[0]
    for stage in stage_indices:
        feature = skips[stage]
        if feature.ndim != 5 or feature.shape[0] != batch_size:
            raise ValueError(f"Encoder stage {stage} has an invalid shape {feature.shape}")


def _weighted_channel_moments(
    feature: Tensor,
    weight: Tensor,
    *,
    denominator_floor: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    if feature.ndim != 5 or weight.ndim != 5 or weight.shape[1] != 1:
        raise ValueError("Feature and weight tensors must be NCDHW and N1DHW")
    if feature.shape[0] != weight.shape[0] or feature.shape[2:] != weight.shape[2:]:
        raise ValueError("Feature and weight tensors are not spatially aligned")
    if denominator_floor <= 0:
        raise ValueError("denominator_floor must be positive")
    feature32 = feature.float()
    weight32 = weight.float().clamp_min(0)
    denominator = weight32.sum(dim=(2, 3, 4)).clamp_min(denominator_floor)
    mean = (feature32 * weight32).sum(dim=(2, 3, 4)) / denominator
    centered = feature32 - mean[:, :, None, None, None]
    variance = (centered.square() * weight32).sum(dim=(2, 3, 4)) / denominator
    return mean, torch.sqrt(variance.clamp_min(0))


def _finite_masked_channel_max(
    feature: Tensor,
    probability: Tensor,
    *,
    threshold: float = 0.5,
) -> Tensor:
    selected = probability > threshold
    masked = feature.float().masked_fill(~selected, -torch.inf)
    maximum = masked.flatten(start_dim=2).amax(dim=2)
    return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))


def pool_multiscale_encoder_features(
    skips: Sequence[Tensor],
    segmentation_logits: Tensor,
    *,
    stage_indices: Sequence[int] = LOCKED_ENCODER_STAGES,
) -> tuple[Tensor, tuple[str, ...]]:
    """Pool frozen skip features with model-predicted spatial evidence.

    Seven statistics are emitted per channel and stage: global mean/std,
    lesion-probability-weighted mean/std, lesion-mask max, and whole-pancreas-
    probability-weighted mean/std.  Only model predictions provide weights.
    """

    _validate_encoder_inputs(skips, segmentation_logits, stage_indices)
    probabilities = torch.softmax(segmentation_logits.float(), dim=1)
    lesion_probability = probabilities[:, 2:3]
    whole_probability = probabilities[:, 1:2] + lesion_probability
    pooled: list[Tensor] = []
    names: list[str] = []
    statistic_names = (
        "global_mean",
        "global_std",
        "lesion_weighted_mean",
        "lesion_weighted_std",
        "lesion_masked_max",
        "whole_weighted_mean",
        "whole_weighted_std",
    )

    for stage in stage_indices:
        feature = skips[stage].float()
        lesion = F.interpolate(
            lesion_probability,
            size=feature.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        whole = F.interpolate(
            whole_probability,
            size=feature.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        flattened = feature.flatten(start_dim=2)
        global_mean = flattened.mean(dim=2)
        global_std = flattened.std(dim=2, correction=0)
        lesion_mean, lesion_std = _weighted_channel_moments(feature, lesion)
        lesion_max = _finite_masked_channel_max(feature, lesion)
        whole_mean, whole_std = _weighted_channel_moments(feature, whole)
        stage_statistics = (
            global_mean,
            global_std,
            lesion_mean,
            lesion_std,
            lesion_max,
            whole_mean,
            whole_std,
        )
        pooled.extend(stage_statistics)
        for statistic in statistic_names:
            names.extend(
                f"encoder_stage_{stage}_{statistic}_channel_{channel:03d}"
                for channel in range(feature.shape[1])
            )

    result = torch.cat(pooled, dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite multiscale encoder feature")
    return result, tuple(names)


def tile_model_evidence(
    segmentation_logits: Tensor,
    normalized_centers: Tensor,
) -> tuple[Tensor, tuple[str, ...]]:
    """Summarize predicted lesion/organ evidence for each spatial tile."""

    if segmentation_logits.ndim != 5 or segmentation_logits.shape[1] != 3:
        raise ValueError("segmentation_logits must have shape (batch, 3, D, H, W)")
    if normalized_centers.shape != (segmentation_logits.shape[0], 3):
        raise ValueError("normalized_centers must have shape (batch, 3)")
    probabilities = torch.softmax(segmentation_logits.float(), dim=1)
    lesion = probabilities[:, 2]
    whole = probabilities[:, 1] + lesion
    values = torch.cat(
        (
            lesion.mean(dim=(1, 2, 3), keepdim=False)[:, None],
            lesion.amax(dim=(1, 2, 3), keepdim=False)[:, None],
            (probabilities.argmax(dim=1) == 2).float().mean(dim=(1, 2, 3), keepdim=False)[:, None],
            whole.mean(dim=(1, 2, 3), keepdim=False)[:, None],
            normalized_centers.float(),
        ),
        dim=1,
    )
    if not torch.isfinite(values).all():
        raise FloatingPointError("Non-finite tile evidence")
    return values, MODEL_EVIDENCE_NAMES


def append_frozen_neural_head_features(
    encoder_features: Tensor,
    encoder_feature_names: Sequence[str],
    mirror_mean_logits: Tensor,
    mirror_mean_probabilities: Tensor,
) -> tuple[Tensor, tuple[str, ...]]:
    """Append the frozen rescue head's inference-matched tile evidence."""

    if encoder_features.ndim != 2:
        raise ValueError("encoder_features must have shape (batch, features)")
    expected_shape = (encoder_features.shape[0], 3)
    if mirror_mean_logits.shape != expected_shape:
        raise ValueError(f"mirror_mean_logits must have shape {expected_shape}")
    if mirror_mean_probabilities.shape != expected_shape:
        raise ValueError(f"mirror_mean_probabilities must have shape {expected_shape}")
    logits = mirror_mean_logits.float()
    probabilities = mirror_mean_probabilities.float()
    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError("Frozen neural-head evidence must be finite")
    if torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("Frozen neural-head probabilities must be in [0, 1]")
    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(probabilities.shape[0], device=probabilities.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("Frozen neural-head probabilities must sum to one")
    values = torch.cat((encoder_features.float(), logits, probabilities), dim=1)
    names = tuple(encoder_feature_names) + FROZEN_NEURAL_HEAD_NAMES
    if len(names) != values.shape[1]:
        raise ValueError("Frozen neural-head feature schema mismatch")
    return values, names


def normalized_tile_centers(
    slicers: Sequence[tuple[slice, ...]],
    spatial_shape: Sequence[int],
) -> Tensor:
    """Return geometry-only tile centers in a stable input order."""

    if len(spatial_shape) != 3 or any(int(size) < 1 for size in spatial_shape):
        raise ValueError("spatial_shape must contain three positive sizes")
    centers: list[list[float]] = []
    for item in slicers:
        spatial_slices = item[-3:]
        if len(spatial_slices) != 3:
            raise ValueError("Each sliding-window item must contain three spatial slices")
        row: list[float] = []
        for axis_slice, size in zip(spatial_slices, spatial_shape, strict=True):
            start = 0 if axis_slice.start is None else int(axis_slice.start)
            stop = int(size) if axis_slice.stop is None else int(axis_slice.stop)
            row.append(float((start + stop - 1) / (2 * max(int(size) - 1, 1))))
        centers.append(row)
    return torch.as_tensor(centers, dtype=torch.float32)


def aggregate_case_tiles(
    tile_vectors: np.ndarray,
    tile_evidence_values: np.ndarray,
    tile_vector_names: Sequence[str],
    *,
    top_k: int = 3,
) -> FeatureView:
    """Aggregate an ordered tile bag without using case identity or order."""

    vectors = np.asarray(tile_vectors, dtype=np.float64)
    evidence = np.asarray(tile_evidence_values, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] < 1:
        raise ValueError("tile_vectors must be a non-empty 2D matrix")
    if evidence.shape != (vectors.shape[0], len(MODEL_EVIDENCE_NAMES)):
        raise ValueError("tile_evidence_values has an invalid shape")
    if vectors.shape[1] != len(tile_vector_names):
        raise ValueError("tile vector schema does not match its matrix")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not np.isfinite(vectors).all() or not np.isfinite(evidence).all():
        raise ValueError("Tile arrays must be finite")

    lesion_mass = np.maximum(evidence[:, 0], 0.0)
    weights = lesion_mass + 1e-6
    weights /= weights.sum()
    # Resolve equal lesion masses by numeric row content rather than arrival
    # order, so a batching implementation cannot change the top-k aggregate.
    row_keys = [hashlib.sha256(row.astype("<f4").tobytes()).digest() for row in vectors]
    stable_top = np.asarray(
        sorted(
            range(vectors.shape[0]),
            key=lambda index: (-lesion_mass[index], row_keys[index]),
        )[: min(top_k, vectors.shape[0])],
        dtype=np.int64,
    )
    aggregates = (
        ("tile_uniform_mean", vectors.mean(axis=0)),
        ("tile_lesion_weighted_mean", np.sum(vectors * weights[:, None], axis=0)),
        ("tile_top3_lesion_mean", vectors[stable_top].mean(axis=0)),
        ("tile_population_std", vectors.std(axis=0, ddof=0)),
    )
    feature_parts = [item[1] for item in aggregates]
    names = [
        f"{aggregate_name}__{feature_name}"
        for aggregate_name, _ in aggregates
        for feature_name in tile_vector_names
    ]

    evidence_statistics = (
        ("mean", evidence.mean(axis=0)),
        ("std", evidence.std(axis=0, ddof=0)),
        ("minimum", evidence.min(axis=0)),
        ("maximum", evidence.max(axis=0)),
    )
    feature_parts.extend(item[1] for item in evidence_statistics)
    names.extend(
        f"tile_evidence_{statistic}__{evidence_name}"
        for statistic, _ in evidence_statistics
        for evidence_name in MODEL_EVIDENCE_NAMES
    )
    vector = np.concatenate(feature_parts).astype(np.float32, copy=False)
    return FeatureView(vector, tuple(names))


def _region_bbox_extent(mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    coordinates = np.nonzero(mask)
    if not coordinates[0].size:
        return np.zeros(3, dtype=np.float64)
    return (
        np.asarray(
            [values.max() - values.min() + 1 for values in coordinates],
            dtype=np.float64,
        )
        * spacing
    )


def _intensity_statistics(values: np.ndarray, *, include_higher_moments: bool) -> list[float]:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.size == 0:
        count = 9 if include_higher_moments else 7
        return [0.0] * count
    quantiles = np.quantile(values64, (0.1, 0.25, 0.5, 0.75, 0.9)).tolist()
    result = quantiles + [float(values64.mean()), float(values64.std(ddof=0))]
    if include_higher_moments:
        if values64.size < 3 or np.isclose(values64.std(ddof=0), 0):
            result.extend((0.0, 0.0))
        else:
            result.extend(
                (
                    float(skew(values64, bias=False)),
                    float(kurtosis(values64, fisher=True, bias=False)),
                )
            )
    return [value if math.isfinite(value) else 0.0 for value in result]


def predicted_mask_ct_features(
    normalized_image: np.ndarray,
    predicted_segmentation: np.ndarray,
    spacing: Sequence[float],
) -> FeatureView:
    """Compute fixed morphometric/intensity features from model predictions.

    ``predicted_segmentation`` must be the model argmax in the same preprocessed
    grid as ``normalized_image``.  There is intentionally no ground-truth mask
    parameter.
    """

    image = np.asarray(normalized_image, dtype=np.float64)
    segmentation = np.asarray(predicted_segmentation)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if image.ndim != 3 or segmentation.shape != image.shape:
        raise ValueError("Image and predicted segmentation must share one 3D grid")
    if (
        spacing_array.shape != (3,)
        or not np.isfinite(spacing_array).all()
        or np.any(spacing_array <= 0)
    ):
        raise ValueError("spacing must contain three finite positive values")
    labels = set(np.unique(segmentation).tolist())
    if not labels.issubset({0, 1, 2}):
        raise ValueError(f"Predicted segmentation has unexpected labels: {sorted(labels)}")
    if not np.isfinite(image).all():
        raise ValueError("Normalized CT contains non-finite values")

    whole = segmentation > 0
    lesion = segmentation == 2
    nonlesion_pancreas = segmentation == 1
    voxel_volume = float(np.prod(spacing_array))
    whole_voxels = int(whole.sum())
    lesion_voxels = int(lesion.sum())
    whole_volume = whole_voxels * voxel_volume
    lesion_volume = lesion_voxels * voxel_volume
    lesion_ratio = lesion_voxels / max(whole_voxels, 1)

    whole_extent = _region_bbox_extent(whole, spacing_array)
    lesion_extent = _region_bbox_extent(lesion, spacing_array)
    relative_centroid = np.zeros(3, dtype=np.float64)
    if lesion_voxels and whole_voxels:
        lesion_coordinates = np.column_stack(np.nonzero(lesion)).astype(np.float64)
        whole_coordinates = np.column_stack(np.nonzero(whole)).astype(np.float64)
        whole_minimum = whole_coordinates.min(axis=0)
        whole_range = np.maximum(whole_coordinates.max(axis=0) - whole_minimum, 1.0)
        relative_centroid = (lesion_coordinates.mean(axis=0) - whole_minimum) / whole_range
    else:
        lesion_coordinates = np.empty((0, 3), dtype=np.float64)

    component_labels, component_count = ndimage.label(
        lesion, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    if component_count:
        component_sizes = np.bincount(component_labels.ravel())[1:]
        largest_component_fraction = float(component_sizes.max() / max(lesion_voxels, 1))
    else:
        largest_component_fraction = 0.0

    eigenvalue_ratios = np.zeros(3, dtype=np.float64)
    if lesion_coordinates.shape[0] >= 3:
        physical_coordinates = lesion_coordinates * spacing_array[None, :]
        covariance = np.cov(physical_coordinates, rowvar=False, bias=True)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        largest = max(float(eigenvalues[0]), 1e-12)
        eigenvalue_ratios = eigenvalues / largest

    lesion_statistics = _intensity_statistics(image[lesion], include_higher_moments=True)
    pancreas_statistics = _intensity_statistics(
        image[nonlesion_pancreas], include_higher_moments=False
    )
    contrasts = [
        lesion_statistics[5] - pancreas_statistics[5],
        lesion_statistics[6] - pancreas_statistics[6],
        lesion_statistics[2] - pancreas_statistics[2],
    ]
    if lesion_voxels:
        ring = ndimage.binary_dilation(lesion, iterations=3) & ~lesion
        ring_values = image[ring]
    else:
        ring_values = np.asarray([], dtype=np.float64)
    ring_mean = float(ring_values.mean()) if ring_values.size else 0.0
    ring_std = float(ring_values.std(ddof=0)) if ring_values.size else 0.0

    values = np.asarray(
        [
            *spacing_array.tolist(),
            *[float(size) for size in image.shape],
            float(whole_voxels == 0),
            float(lesion_voxels == 0),
            whole_volume,
            lesion_volume,
            lesion_ratio,
            *whole_extent.tolist(),
            *lesion_extent.tolist(),
            *relative_centroid.tolist(),
            float(component_count),
            largest_component_fraction,
            *eigenvalue_ratios.tolist(),
            *lesion_statistics,
            *pancreas_statistics,
            *contrasts,
            ring_mean,
            ring_std,
        ],
        dtype=np.float32,
    )
    names = (
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "image_shape_x",
        "image_shape_y",
        "image_shape_z",
        "predicted_whole_empty",
        "predicted_lesion_empty",
        "predicted_whole_volume_mm3",
        "predicted_lesion_volume_mm3",
        "predicted_lesion_to_whole_voxel_ratio",
        "predicted_whole_bbox_extent_x_mm",
        "predicted_whole_bbox_extent_y_mm",
        "predicted_whole_bbox_extent_z_mm",
        "predicted_lesion_bbox_extent_x_mm",
        "predicted_lesion_bbox_extent_y_mm",
        "predicted_lesion_bbox_extent_z_mm",
        "predicted_lesion_centroid_relative_x",
        "predicted_lesion_centroid_relative_y",
        "predicted_lesion_centroid_relative_z",
        "predicted_lesion_component_count",
        "predicted_lesion_largest_component_fraction",
        "predicted_lesion_covariance_eigen_ratio_1",
        "predicted_lesion_covariance_eigen_ratio_2",
        "predicted_lesion_covariance_eigen_ratio_3",
        "predicted_lesion_ct_q10",
        "predicted_lesion_ct_q25",
        "predicted_lesion_ct_q50",
        "predicted_lesion_ct_q75",
        "predicted_lesion_ct_q90",
        "predicted_lesion_ct_mean",
        "predicted_lesion_ct_std",
        "predicted_lesion_ct_skew",
        "predicted_lesion_ct_kurtosis",
        "predicted_nonlesion_pancreas_ct_q10",
        "predicted_nonlesion_pancreas_ct_q25",
        "predicted_nonlesion_pancreas_ct_q50",
        "predicted_nonlesion_pancreas_ct_q75",
        "predicted_nonlesion_pancreas_ct_q90",
        "predicted_nonlesion_pancreas_ct_mean",
        "predicted_nonlesion_pancreas_ct_std",
        "lesion_minus_nonlesion_pancreas_ct_mean",
        "lesion_minus_nonlesion_pancreas_ct_std",
        "lesion_minus_nonlesion_pancreas_ct_median",
        "predicted_perilesional_ring_ct_mean",
        "predicted_perilesional_ring_ct_std",
    )
    return FeatureView(values, names)


def build_case_feature_views(
    tile_vectors: np.ndarray,
    tile_evidence_values: np.ndarray,
    tile_vector_names: Sequence[str],
    normalized_image: np.ndarray,
    predicted_segmentation: np.ndarray,
    spacing: Sequence[float],
) -> dict[str, FeatureView]:
    """Build the two feature views declared in the prospective v2 lock."""

    if not set(FROZEN_NEURAL_HEAD_NAMES).issubset(tile_vector_names):
        raise ValueError("Each tile vector must include the frozen rescue neural-head evidence")

    encoder = aggregate_case_tiles(
        tile_vectors,
        tile_evidence_values,
        tile_vector_names,
    )
    morphology = predicted_mask_ct_features(
        normalized_image,
        predicted_segmentation,
        spacing,
    )
    combined = FeatureView(
        np.concatenate((encoder.values, morphology.values)).astype(np.float32, copy=False),
        encoder.names + morphology.names,
    )
    return {
        ("multiscale_encoder_aggregates_plus_frozen_neural_head_plus_tile_evidence"): encoder,
        (
            "multiscale_encoder_aggregates_plus_frozen_neural_head_plus_"
            "tile_evidence_plus_predicted_mask_ct_features"
        ): combined,
    }


__all__ = [
    "FROZEN_NEURAL_HEAD_NAMES",
    "LOCKED_ENCODER_CHANNELS",
    "LOCKED_ENCODER_STAGES",
    "MODEL_EVIDENCE_NAMES",
    "FeatureView",
    "TrainCase",
    "aggregate_case_tiles",
    "append_frozen_neural_head_features",
    "build_case_feature_views",
    "discover_train_cases",
    "normalized_tile_centers",
    "pool_multiscale_encoder_features",
    "predicted_mask_ct_features",
    "tile_model_evidence",
    "train_case_inventory_audit",
]
