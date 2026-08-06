"""Frozen-network sliding-window extraction for case-level classification.

This module mirrors the production predictor's preprocessing, spatial tiles,
Gaussian stitching, and mirror TTA.  It additionally returns identifier-free
multiscale encoder features, the frozen rescue head output, and a compact
stage-3 cache for the prospectively locked MIL contingency.  Tile batches are
unpacked in original slicer order before any CPU aggregation.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from torch import Tensor, nn

from pancreas_multitask.case_features import (
    append_frozen_neural_head_features,
    normalized_tile_centers,
    pool_multiscale_encoder_features,
    tile_model_evidence,
)

MIL_GRID = (4, 4, 6)
MIL_STAGE_INDEX = 3
MIL_TOP_K = 3
LOCKED_NETWORK_MICROBATCH_CEILING = 2


def _validate_batch_size(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    if value > LOCKED_NETWORK_MICROBATCH_CEILING:
        raise ValueError(
            f"{name} cannot exceed the locked network microbatch ceiling "
            f"of {LOCKED_NETWORK_MICROBATCH_CEILING}"
        )
    return value


@dataclass(slots=True)
class FeatureExtractionRuntimeCounters:
    """Auditable scheduling state shared across extracted cases.

    Tile and mirror-view batching consume the same network microbatch budget.
    The prospective speed arms request ``1/1`` or ``2/2`` respectively; a
    two-tile group therefore runs one view per network call, while a one-tile
    group may run two views per call. Adaptive limits persist across cases so
    an OOM fallback can never be hidden by the next case.
    """

    tile_batch_size_requested: int = 1
    tta_batch_size_requested: int = 1
    tile_batch_size_adaptive_limit: int = field(init=False)
    tta_batch_size_adaptive_limit: int = field(init=False)
    tile_batch_oom_fallback_count: int = 0
    tta_batch_oom_fallback_count: int = 0
    logical_tile_batches_completed: int = 0
    logical_tiles_completed: int = 0
    tta_view_batches_completed: int = 0
    tta_views_completed: int = 0
    shared_network_forward_calls: int = 0
    maximum_network_batch_size_observed: int = 0
    network_batch_size_histogram: dict[int, int] = field(default_factory=dict)
    tile_batch_size_histogram: dict[int, int] = field(default_factory=dict)
    tta_batch_size_histogram: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tile_batch_size_requested = _validate_batch_size(
            self.tile_batch_size_requested,
            name="tile_batch_size_requested",
        )
        self.tta_batch_size_requested = _validate_batch_size(
            self.tta_batch_size_requested,
            name="tta_batch_size_requested",
        )
        self.tile_batch_size_adaptive_limit = self.tile_batch_size_requested
        self.tta_batch_size_adaptive_limit = self.tta_batch_size_requested

    @property
    def network_batch_size_limit(self) -> int:
        return max(
            self.tile_batch_size_requested,
            self.tta_batch_size_requested,
        )

    @staticmethod
    def is_out_of_memory_error(error: RuntimeError) -> bool:
        return isinstance(error, torch.OutOfMemoryError) or (
            "out of memory" in str(error).lower()
        )

    def record_network_forward(self, batch_size: int) -> None:
        if batch_size < 1 or batch_size > self.network_batch_size_limit:
            raise RuntimeError(
                "Shared encoder network batch exceeds the locked scheduling limit: "
                f"observed={batch_size}, limit={self.network_batch_size_limit}"
            )
        self.shared_network_forward_calls += 1
        self.maximum_network_batch_size_observed = max(
            self.maximum_network_batch_size_observed,
            batch_size,
        )
        self.network_batch_size_histogram[batch_size] = (
            self.network_batch_size_histogram.get(batch_size, 0) + 1
        )

    def record_tile_batch(self, batch_size: int) -> None:
        self.logical_tile_batches_completed += 1
        self.logical_tiles_completed += batch_size
        self.tile_batch_size_histogram[batch_size] = (
            self.tile_batch_size_histogram.get(batch_size, 0) + 1
        )

    def record_tta_batch(self, view_count: int) -> None:
        self.tta_view_batches_completed += 1
        self.tta_views_completed += view_count
        self.tta_batch_size_histogram[view_count] = (
            self.tta_batch_size_histogram.get(view_count, 0) + 1
        )

    def provenance(self) -> dict[str, object]:
        return {
            "joint_network_forward_calls": self.shared_network_forward_calls,
            "shared_network_forward_calls": self.shared_network_forward_calls,
            "maximum_network_batch_size_observed": (
                self.maximum_network_batch_size_observed
            ),
            "network_batch_size_histogram": {
                str(size): count
                for size, count in sorted(self.network_batch_size_histogram.items())
            },
            "network_batch_size_limit": self.network_batch_size_limit,
            "locked_network_microbatch_ceiling": (
                LOCKED_NETWORK_MICROBATCH_CEILING
            ),
            "logical_tile_batches_completed": self.logical_tile_batches_completed,
            "logical_tiles_completed": self.logical_tiles_completed,
            "tile_batch_oom_fallback_count": self.tile_batch_oom_fallback_count,
            "tile_batch_size_adaptive_limit": (
                self.tile_batch_size_adaptive_limit
            ),
            "tile_batch_size_histogram": {
                str(size): count
                for size, count in sorted(self.tile_batch_size_histogram.items())
            },
            "tile_batch_size_requested": self.tile_batch_size_requested,
            "tta_batch_oom_fallback_count": self.tta_batch_oom_fallback_count,
            "tta_batch_size_adaptive_limit": self.tta_batch_size_adaptive_limit,
            "tta_batch_size_histogram": {
                str(size): count
                for size, count in sorted(self.tta_batch_size_histogram.items())
            },
            "tta_batch_size_requested": self.tta_batch_size_requested,
            "tta_view_batches_completed": self.tta_view_batches_completed,
            "tta_views_completed": self.tta_views_completed,
        }


@dataclass(frozen=True, slots=True)
class TileBatchFeatures:
    segmentation_logits: Tensor
    tile_vectors: Tensor
    tile_vector_names: tuple[str, ...]
    mirror_mean_logits: Tensor
    mirror_mean_probabilities: Tensor
    mil_stage3_maps: Tensor
    mil_prediction_maps: Tensor


@dataclass(frozen=True, slots=True)
class CaseExtraction:
    segmentation_logits: Tensor
    tile_vectors: np.ndarray
    tile_evidence: np.ndarray
    tile_vector_names: tuple[str, ...]
    mil_stage3_maps: np.ndarray
    mil_prediction_maps: np.ndarray
    mil_lesion_mass: np.ndarray
    tile_count: int

    def __post_init__(self) -> None:
        if self.segmentation_logits.ndim != 4 or self.segmentation_logits.shape[0] != 3:
            raise ValueError("Case segmentation logits must have shape (3, D, H, W)")
        if self.tile_vectors.ndim != 2 or self.tile_vectors.shape[0] != self.tile_count:
            raise ValueError("Case tile-vector cache has an invalid shape")
        if self.tile_evidence.shape[0] != self.tile_count:
            raise ValueError("Case tile evidence is not aligned")
        if self.tile_vectors.shape[1] != len(self.tile_vector_names):
            raise ValueError("Case tile-vector schema is not aligned")
        if self.mil_stage3_maps.ndim != 5 or self.mil_stage3_maps.shape[1:] != (
            256,
            *MIL_GRID,
        ):
            raise ValueError("MIL stage-3 cache has an invalid shape")
        if self.mil_prediction_maps.shape != (
            self.mil_stage3_maps.shape[0],
            2,
            *MIL_GRID,
        ):
            raise ValueError("MIL prediction cache has an invalid shape")
        if self.mil_lesion_mass.shape != (self.mil_stage3_maps.shape[0],):
            raise ValueError("MIL lesion masses are not aligned")


def _mirror_combinations(
    allowed_mirroring_axes: Sequence[int] | None,
) -> tuple[tuple[int, ...], ...]:
    if not allowed_mirroring_axes:
        return ((),)
    axes = tuple(int(axis) for axis in allowed_mirroring_axes)
    if min(axes) < 0 or max(axes) > 2 or len(axes) != len(set(axes)):
        raise ValueError(f"Invalid 3D mirror axes: {axes}")
    tensor_axes = tuple(axis + 2 for axis in axes)
    return ((),) + tuple(
        combination
        for count in range(1, len(tensor_axes) + 1)
        for combination in itertools.combinations(tensor_axes, count)
    )


def _forward_shared_network(
    network: nn.Module,
    data: Tensor,
) -> tuple[Tensor, Sequence[Tensor], Tensor]:
    encoder = getattr(network, "encoder", None)
    decoder = getattr(network, "decoder", None)
    classify = getattr(network, "classify_bottleneck", None)
    if not isinstance(encoder, nn.Module) or not isinstance(decoder, nn.Module):
        raise TypeError("Network must expose the shared encoder and segmentation decoder")
    if not callable(classify):
        raise TypeError("Network must expose classify_bottleneck")
    skips = encoder(data)
    if not isinstance(skips, (list, tuple)) or len(skips) < 6:
        raise TypeError("Shared encoder must return all six ResEnc-M stages")
    segmentation_logits = decoder(skips)
    if not isinstance(segmentation_logits, Tensor):
        raise TypeError("Segmentation decoder must return one tensor during extraction")
    classification_logits = classify(skips[-1])
    if classification_logits.shape != (data.shape[0], 3):
        raise ValueError("Frozen rescue head must return three logits per tile")
    return segmentation_logits, skips, classification_logits


@torch.inference_mode()
def mirror_mean_tile_batch_features(
    network: nn.Module,
    data: Tensor,
    *,
    allowed_mirroring_axes: Sequence[int] | None,
    tta_batch_size: int = 1,
    runtime_counters: FeatureExtractionRuntimeCounters | None = None,
) -> TileBatchFeatures:
    """Average predictions and invariant features over declared mirror views.

    The batch-one TTA arm deliberately remains on the original one-view loop.
    For the candidate arm, consecutive views may share a forward only when the
    tile batch leaves capacity under the common network microbatch limit.
    """

    if data.ndim != 5 or data.shape[0] < 1:
        raise ValueError("data must have shape (batch, channels, D, H, W)")
    tta_batch_size = _validate_batch_size(
        tta_batch_size,
        name="tta_batch_size",
    )
    if runtime_counters is None:
        runtime_counters = FeatureExtractionRuntimeCounters(
            tile_batch_size_requested=min(
                int(data.shape[0]),
                LOCKED_NETWORK_MICROBATCH_CEILING,
            ),
            tta_batch_size_requested=tta_batch_size,
        )
    elif runtime_counters.tta_batch_size_requested != tta_batch_size:
        raise ValueError(
            "Runtime counters and mirror extraction request different TTA batch sizes"
        )
    if data.shape[0] > runtime_counters.network_batch_size_limit:
        raise ValueError(
            "Tile batch exceeds the shared network microbatch limit: "
            f"tiles={data.shape[0]}, limit={runtime_counters.network_batch_size_limit}"
        )
    combinations = _mirror_combinations(allowed_mirroring_axes)
    view_axes = ((), *combinations[1:])
    segmentation_sum: Tensor | None = None
    feature_sum: Tensor | None = None
    logit_sum: Tensor | None = None
    probability_sum: Tensor | None = None
    mil_stage3_sum: Tensor | None = None
    mil_prediction_sum: Tensor | None = None
    feature_names: tuple[str, ...] | None = None

    def accumulate_view(
        axes: tuple[int, ...],
        segmentation: Tensor,
        skips: Sequence[Tensor],
        logits: Tensor,
    ) -> None:
        nonlocal segmentation_sum
        nonlocal feature_sum
        nonlocal logit_sum
        nonlocal probability_sum
        nonlocal mil_stage3_sum
        nonlocal mil_prediction_sum
        nonlocal feature_names

        pooled, names = pool_multiscale_encoder_features(skips, segmentation)
        probabilities = torch.softmax(logits.float(), dim=1)
        segmentation_probabilities = torch.softmax(segmentation.float(), dim=1)
        stage3_map = F.adaptive_avg_pool3d(skips[MIL_STAGE_INDEX].float(), MIL_GRID)
        prediction_at_stage3 = F.interpolate(
            torch.cat(
                (
                    segmentation_probabilities[:, 2:3],
                    segmentation_probabilities[:, 1:2] + segmentation_probabilities[:, 2:3],
                ),
                dim=1,
            ),
            size=skips[MIL_STAGE_INDEX].shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        prediction_map = F.adaptive_avg_pool3d(prediction_at_stage3, MIL_GRID)

        if axes:
            segmentation = torch.flip(segmentation, axes)
            # Adaptive pooling preserves axis order; flip the compact maps back
            # to the unmirrored tile orientation for the spatial MIL cache.
            stage3_map = torch.flip(stage3_map, axes)
            prediction_map = torch.flip(prediction_map, axes)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Encoder feature schema changed between mirror views")

        # Preserve the production batch-one TTA accumulator dtype and operation
        # order exactly. Feature statistics use explicit float32 separately.
        segmentation_sum = (
            segmentation if segmentation_sum is None else segmentation_sum + segmentation
        )
        feature_sum = pooled if feature_sum is None else feature_sum + pooled
        logit_sum = logits.float() if logit_sum is None else logit_sum + logits.float()
        probability_sum = (
            probabilities if probability_sum is None else probability_sum + probabilities
        )
        mil_stage3_sum = stage3_map if mil_stage3_sum is None else mil_stage3_sum + stage3_map
        mil_prediction_sum = (
            prediction_map if mil_prediction_sum is None else mil_prediction_sum + prediction_map
        )

    # Keep the prospective reference byte-for-byte on its original sequence:
    # one network call and one accumulator update per mirror view.
    if tta_batch_size == 1:
        for axes in view_axes:
            view = torch.flip(data, axes) if axes else data
            runtime_counters.record_network_forward(int(view.shape[0]))
            segmentation, skips, logits = _forward_shared_network(network, view)
            accumulate_view(axes, segmentation, skips, logits)
            runtime_counters.record_tta_batch(1)
    else:
        base_batch_size = int(data.shape[0])
        active_tta_batch_size = min(
            runtime_counters.tta_batch_size_adaptive_limit,
            max(1, runtime_counters.network_batch_size_limit // base_batch_size),
        )
        view_index = 0
        while view_index < len(view_axes):
            batch_axes = view_axes[view_index : view_index + active_tta_batch_size]
            batched_input: Tensor | None = None
            try:
                if len(batch_axes) == 1:
                    axes = batch_axes[0]
                    batched_input = torch.flip(data, axes) if axes else data
                else:
                    batched_input = torch.cat(
                        [
                            torch.flip(data, axes) if axes else data
                            for axes in batch_axes
                        ],
                        dim=0,
                    )
                runtime_counters.record_network_forward(int(batched_input.shape[0]))
                batched_segmentation, batched_skips, batched_logits = (
                    _forward_shared_network(network, batched_input)
                )
            except RuntimeError as error:
                if (
                    active_tta_batch_size == 1
                    or not runtime_counters.is_out_of_memory_error(error)
                ):
                    raise
                del batched_input
                empty_cache(data.device)
                active_tta_batch_size = max(1, active_tta_batch_size // 2)
                runtime_counters.tta_batch_size_adaptive_limit = (
                    active_tta_batch_size
                )
                runtime_counters.tta_batch_oom_fallback_count += 1
                continue

            view_count = len(batch_axes)
            expected_network_batch = view_count * base_batch_size
            if batched_segmentation.shape[0] != expected_network_batch:
                raise ValueError("Segmentation batch cannot be restored to TTA views")
            if batched_logits.shape[0] != expected_network_batch:
                raise ValueError("Classification batch cannot be restored to TTA views")
            if any(skip.shape[0] != expected_network_batch for skip in batched_skips):
                raise ValueError("Encoder batch cannot be restored to TTA views")

            segmentation_views = batched_segmentation.reshape(
                view_count,
                base_batch_size,
                *batched_segmentation.shape[1:],
            )
            logit_views = batched_logits.reshape(
                view_count,
                base_batch_size,
                *batched_logits.shape[1:],
            )
            skip_views = tuple(
                skip.reshape(view_count, base_batch_size, *skip.shape[1:])
                for skip in batched_skips
            )
            for local_index, axes in enumerate(batch_axes):
                accumulate_view(
                    axes,
                    segmentation_views[local_index],
                    tuple(skip[local_index] for skip in skip_views),
                    logit_views[local_index],
                )
            runtime_counters.record_tta_batch(view_count)
            view_index += view_count

    if any(
        value is None
        for value in (
            segmentation_sum,
            feature_sum,
            logit_sum,
            probability_sum,
            mil_stage3_sum,
            mil_prediction_sum,
            feature_names,
        )
    ):
        raise RuntimeError("Mirror feature extraction produced no views")
    divisor = float(len(combinations))
    mean_logits = logit_sum / divisor
    mean_probabilities = probability_sum / divisor
    tile_vectors, tile_names = append_frozen_neural_head_features(
        feature_sum / divisor,
        feature_names,
        mean_logits,
        mean_probabilities,
    )
    return TileBatchFeatures(
        segmentation_sum / divisor,
        tile_vectors,
        tile_names,
        mean_logits,
        mean_probabilities,
        mil_stage3_sum / divisor,
        mil_prediction_sum / divisor,
    )


def _numeric_top_k(
    lesion_mass: np.ndarray,
    stage3_maps: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    keys = [hashlib.sha256(np.asarray(row, dtype="<f2").tobytes()).digest() for row in stage3_maps]
    return np.asarray(
        sorted(
            range(lesion_mass.size),
            key=lambda index: (-float(lesion_mass[index]), keys[index]),
        )[: min(top_k, lesion_mass.size)],
        dtype=np.int64,
    )


@torch.inference_mode()
def extract_case_from_preprocessed(
    predictor: object,
    input_image: Tensor,
    *,
    tile_batch_size: int = 1,
    tta_batch_size: int = 1,
    runtime_counters: FeatureExtractionRuntimeCounters | None = None,
) -> CaseExtraction:
    """Extract one full case while preserving predictor tile/TTA semantics."""

    if not isinstance(input_image, Tensor) or input_image.ndim != 4:
        raise ValueError("input_image must have shape (channels, D, H, W)")
    tile_batch_size = _validate_batch_size(
        tile_batch_size,
        name="tile_batch_size",
    )
    tta_batch_size = _validate_batch_size(
        tta_batch_size,
        name="tta_batch_size",
    )
    if runtime_counters is None:
        runtime_counters = FeatureExtractionRuntimeCounters(
            tile_batch_size_requested=tile_batch_size,
            tta_batch_size_requested=tta_batch_size,
        )
    elif (
        runtime_counters.tile_batch_size_requested != tile_batch_size
        or runtime_counters.tta_batch_size_requested != tta_batch_size
    ):
        raise ValueError(
            "Runtime counters must be configured for the requested tile/TTA batches"
        )
    network = getattr(predictor, "network", None)
    device = getattr(predictor, "device", None)
    configuration = getattr(predictor, "configuration_manager", None)
    label_manager = getattr(predictor, "label_manager", None)
    if not isinstance(network, nn.Module) or not isinstance(device, torch.device):
        raise TypeError("predictor must expose an initialized network and torch device")
    if configuration is None or label_manager is None:
        raise TypeError("predictor must expose configuration and label managers")

    network.to(device)
    network.eval()
    empty_cache(device)
    padded, slicer_revert_padding = pad_nd_image(
        input_image,
        configuration.patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )
    slicers = tuple(predictor._internal_get_sliding_window_slicers(padded.shape[1:]))
    if not slicers:
        raise RuntimeError("Sliding-window extraction produced no tiles")
    centers = normalized_tile_centers(slicers, padded.shape[1:])
    results_device = device if predictor.perform_everything_on_device else torch.device("cpu")
    padded_on_results = padded.to(results_device)
    predicted_logits = torch.zeros(
        (label_manager.num_segmentation_heads, *padded.shape[1:]),
        dtype=torch.float16,
        device=results_device,
    )
    n_predictions = torch.zeros(padded.shape[1:], dtype=torch.float16, device=results_device)
    gaussian: Tensor | float
    if predictor.use_gaussian:
        gaussian = compute_gaussian(
            tuple(configuration.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=results_device,
        )
    else:
        gaussian = 1.0

    vector_rows: list[np.ndarray] = []
    evidence_rows: list[np.ndarray] = []
    mil_stage3_rows: list[np.ndarray] = []
    mil_prediction_rows: list[np.ndarray] = []
    tile_vector_names: tuple[str, ...] | None = None
    mirror_axes = predictor.allowed_mirroring_axes if predictor.use_mirroring else None
    autocast = torch.autocast(device.type, enabled=device.type == "cuda")
    with autocast:
        active_tile_batch_size = runtime_counters.tile_batch_size_adaptive_limit
        batch_start = 0
        while batch_start < len(slicers):
            batch_slicers = slicers[
                batch_start : batch_start + active_tile_batch_size
            ]
            work: Tensor | None = None
            try:
                work = torch.cat(
                    [padded_on_results[item][None] for item in batch_slicers],
                    dim=0,
                ).to(device, memory_format=torch.contiguous_format)
                extracted = mirror_mean_tile_batch_features(
                    network,
                    work,
                    allowed_mirroring_axes=mirror_axes,
                    tta_batch_size=tta_batch_size,
                    runtime_counters=runtime_counters,
                )
            except RuntimeError as error:
                if (
                    active_tile_batch_size == 1
                    or not runtime_counters.is_out_of_memory_error(error)
                ):
                    raise
                del work
                empty_cache(device)
                active_tile_batch_size = max(1, active_tile_batch_size // 2)
                runtime_counters.tile_batch_size_adaptive_limit = (
                    active_tile_batch_size
                )
                runtime_counters.tile_batch_oom_fallback_count += 1
                continue

            batch_centers = centers[batch_start : batch_start + len(batch_slicers)].to(
                extracted.segmentation_logits.device
            )
            evidence, _ = tile_model_evidence(
                extracted.segmentation_logits,
                batch_centers,
            )
            if tile_vector_names is None:
                tile_vector_names = extracted.tile_vector_names
            elif tile_vector_names != extracted.tile_vector_names:
                raise RuntimeError("Tile feature schema changed between batches")

            for local_index, sliding_slice in enumerate(batch_slicers):
                tile_segmentation = extracted.segmentation_logits[local_index].to(
                    results_device, dtype=torch.float16
                )
                if predictor.use_gaussian:
                    tile_segmentation = tile_segmentation * gaussian
                predicted_logits[sliding_slice] += tile_segmentation
                n_predictions[sliding_slice[1:]] += gaussian

            vector_rows.extend(
                extracted.tile_vectors.detach().float().cpu().numpy().astype(np.float32)
            )
            evidence_rows.extend(evidence.detach().float().cpu().numpy().astype(np.float32))
            mil_stage3_rows.extend(
                extracted.mil_stage3_maps.detach().half().cpu().numpy().astype(np.float16)
            )
            mil_prediction_rows.extend(
                extracted.mil_prediction_maps.detach().half().cpu().numpy().astype(np.float16)
            )
            runtime_counters.record_tile_batch(len(batch_slicers))
            batch_start += len(batch_slicers)

    torch.div(predicted_logits, n_predictions, out=predicted_logits)
    if not torch.isfinite(predicted_logits).all():
        raise FloatingPointError("Non-finite stitched segmentation logits")
    cropped = predicted_logits[(slice(None), *slicer_revert_padding[1:])].cpu()
    tile_vectors = np.stack(vector_rows).astype(np.float32, copy=False)
    tile_evidence = np.stack(evidence_rows).astype(np.float32, copy=False)
    stage3_all = np.stack(mil_stage3_rows).astype(np.float16, copy=False)
    prediction_all = np.stack(mil_prediction_rows).astype(np.float16, copy=False)
    lesion_mass_all = tile_evidence[:, 0].astype(np.float32, copy=False)
    selected = _numeric_top_k(
        lesion_mass_all,
        stage3_all,
        top_k=MIL_TOP_K,
    )
    if tile_vector_names is None:
        raise RuntimeError("Tile extraction produced no feature schema")
    empty_cache(device)
    return CaseExtraction(
        cropped,
        tile_vectors,
        tile_evidence,
        tile_vector_names,
        stage3_all[selected],
        prediction_all[selected],
        lesion_mass_all[selected],
        len(slicers),
    )


__all__ = [
    "LOCKED_NETWORK_MICROBATCH_CEILING",
    "MIL_GRID",
    "MIL_STAGE_INDEX",
    "MIL_TOP_K",
    "CaseExtraction",
    "FeatureExtractionRuntimeCounters",
    "TileBatchFeatures",
    "extract_case_from_preprocessed",
    "mirror_mean_tile_batch_features",
]
