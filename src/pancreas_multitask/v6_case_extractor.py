"""Train-only extractor for the locked V6 adaptive whole-volume case head."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from torch import Tensor, nn

from pancreas_multitask.case_feature_extractor import FeatureExtractionRuntimeCounters
from pancreas_multitask.case_features import predicted_mask_ct_features
from pancreas_multitask.v6_case_head import (
    MORPHOLOGY_FEATURES,
    SPATIAL_CHANNELS,
    SPATIAL_GRID,
    STAGE5_CHANNELS,
)

# Re-frozen after the morphology float16 overflow fix. The pre-fix lock was
# 455c4f957e248506c5c4f51e4714aad46f5b1e6cd88db78a74f092cc8c92ca0b, which
# declared "cache_numeric_dtype": "float16" for every cached array.
V6_H100_ORCHESTRATION_LOCK_SHA256: Final = (
    "f81448b7143850713b8a95b3a8b820dde5b7a58962fc25f54bbf1e69cb36454c"
)
V6_STAGE2_INDEX: Final = 2
V6_STAGE3_INDEX: Final = 3
V6_STAGE5_INDEX: Final = 5
V6_MIRROR_AXES: Final = (0, 1, 2)
V6_MIRROR_VIEW_COUNT: Final = 8
V6_STAGE5_WEIGHT_FLOOR: Final = 1e-6
V6_ADAPTIVE_MINIMUM_SHAPE: Final = (64, 128, 192)
V6_ADAPTIVE_ENCODER_STRIDE: Final = (16, 32, 32)


def adaptive_whole_volume_target_shape(
    spatial_shape: Sequence[int],
) -> tuple[int, int, int]:
    """Apply the locked V6 minimum-and-stride padding rule to one ROI shape."""

    if len(spatial_shape) != 3:
        raise ValueError("Adaptive whole-volume shape must contain three dimensions")
    values: list[int] = []
    for raw_size in spatial_shape:
        if isinstance(raw_size, bool) or not isinstance(raw_size, (int, np.integer)):
            raise TypeError("Adaptive whole-volume dimensions must be integers")
        size = int(raw_size)
        if size < 1:
            raise ValueError("Adaptive whole-volume dimensions must be positive")
        values.append(size)
    return tuple(
        ((max(size, minimum) + stride - 1) // stride) * stride
        for size, minimum, stride in zip(
            values,
            V6_ADAPTIVE_MINIMUM_SHAPE,
            V6_ADAPTIVE_ENCODER_STRIDE,
            strict=True,
        )
    )


def _array_sha256(array: np.ndarray) -> str:
    import hashlib

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def normalized_v6_cell_coordinates(
    grid: Sequence[int] = SPATIAL_GRID,
) -> Tensor:
    """Return extractor-owned normalized cell centers with shape ``(3,D,H,W)``."""

    shape = tuple(int(value) for value in grid)
    if shape != SPATIAL_GRID:
        raise ValueError(f"The locked V6 grid is exactly {SPATIAL_GRID}")
    axes = [(torch.arange(size, dtype=torch.float32) + 0.5) / size for size in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0).contiguous()


@dataclass(frozen=True, slots=True)
class V6ExtractedCase:
    """One identifier-free numeric case representation before cache serialization."""

    spatial: np.ndarray
    coordinates: np.ndarray
    case_lesion_mass: np.ndarray
    stage5_global_mean: np.ndarray
    stage5_lesion_weighted_mean: np.ndarray
    rescue_mean_logits: np.ndarray
    rescue_mean_probabilities: np.ndarray
    morphology: np.ndarray
    morphology_names: tuple[str, ...]
    predicted_segmentation: np.ndarray
    target_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        expected = {
            "spatial": (SPATIAL_CHANNELS, *SPATIAL_GRID),
            "coordinates": (3, *SPATIAL_GRID),
            "case_lesion_mass": (),
            "stage5_global_mean": (STAGE5_CHANNELS,),
            "stage5_lesion_weighted_mean": (STAGE5_CHANNELS,),
            "rescue_mean_logits": (3,),
            "rescue_mean_probabilities": (3,),
            "morphology": (MORPHOLOGY_FEATURES,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid V6 extracted array {name}: {value.shape}")
        if len(self.morphology_names) != MORPHOLOGY_FEATURES:
            raise ValueError("V6 morphology schema must contain exactly 46 names")
        if self.predicted_segmentation.ndim != 3 or not set(
            np.unique(self.predicted_segmentation).tolist()
        ).issubset({0, 1, 2}):
            raise ValueError("V6 predicted segmentation must be one three-label volume")
        lesion = self.spatial[384]
        whole = self.spatial[385]
        if np.any(lesion < 0) or np.any(whole > 1) or np.any(lesion > whole):
            raise ValueError("V6 spatial probability channels are invalid")
        if not np.isclose(self.rescue_mean_probabilities.sum(), 1.0, atol=1e-5):
            raise ValueError("V6 rescue probabilities are not normalized")

    def cache_arrays(self) -> dict[str, np.ndarray]:
        """Return the locked cache representation: float16, morphology float32.

        Every array except ``morphology`` is bounded (probabilities, normalized
        coordinates, pooled activations) and stores exactly in float16.
        ``morphology`` carries two physical volumes in mm^3 --
        ``predicted_whole_volume_mm3`` and ``predicted_lesion_volume_mm3`` --
        whose natural magnitude for a pancreas (~3e4 to 1e5) exceeds the
        float16 maximum of 65504. Casting them to float16 silently produced
        ``+inf`` for 165 of the 252 training cases, which ``_load_cache`` then
        rejected on its finiteness check. Morphology is therefore stored in
        float32; it is only 46 values per case, so the size cost is negligible.
        """

        return {
            "spatial": self.spatial.astype(np.float16),
            "coordinates": self.coordinates.astype(np.float16),
            "case_lesion_mass": self.case_lesion_mass.astype(np.float16),
            "stage5_global_mean": self.stage5_global_mean.astype(np.float16),
            "stage5_lesion_weighted_mean": self.stage5_lesion_weighted_mean.astype(
                np.float16
            ),
            "rescue_mean_logits": self.rescue_mean_logits.astype(np.float16),
            "rescue_mean_probabilities": self.rescue_mean_probabilities.astype(
                np.float16
            ),
            "morphology": self.morphology.astype(np.float32),
        }

    def numeric_hashes(self) -> dict[str, str]:
        return {name: _array_sha256(value) for name, value in self.cache_arrays().items()}


def _mirror_views() -> tuple[tuple[int, ...], ...]:
    tensor_axes = tuple(axis + 2 for axis in V6_MIRROR_AXES)
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
        raise TypeError("V6 extraction requires the shared encoder and decoder")
    if not callable(classify):
        raise TypeError("V6 extraction requires the frozen rescue classification head")
    skips = encoder(data)
    if not isinstance(skips, (list, tuple)) or len(skips) != 6:
        raise TypeError("V6 extraction requires all six ResEnc-M encoder stages")
    logits = decoder(skips)
    if not isinstance(logits, Tensor) or logits.shape[:2] != (data.shape[0], 3):
        raise ValueError("V6 segmentation decoder must emit three logits per voxel")
    rescue_logits = classify(skips[V6_STAGE5_INDEX])
    if rescue_logits.shape != (data.shape[0], 3):
        raise ValueError("V6 rescue head must emit three logits per case")
    expected_channels = {V6_STAGE2_INDEX: 128, V6_STAGE3_INDEX: 256, V6_STAGE5_INDEX: 320}
    for index, channels in expected_channels.items():
        if skips[index].ndim != 5 or skips[index].shape[1] != channels:
            raise ValueError(f"Unexpected ResEnc-M stage {index} shape: {skips[index].shape}")
    return logits, skips, rescue_logits


@torch.inference_mode()
def extract_v6_case_from_preprocessed(
    predictor: object,
    input_image: Tensor,
    *,
    spacing: Sequence[float],
    runtime_counters: FeatureExtractionRuntimeCounters,
) -> V6ExtractedCase:
    """Extract exactly one locked V6 case without opening a ground-truth mask."""

    if not isinstance(input_image, Tensor) or input_image.dtype != torch.float32:
        raise TypeError("V6 input_image must be a float32 torch.Tensor")
    if input_image.ndim != 4 or input_image.shape[0] != 1:
        raise ValueError("V6 input_image must have shape (1,D,H,W)")
    if runtime_counters.network_batch_size_limit != 1:
        raise ValueError("V6 whole-volume extraction requires network batch limit one")
    network = getattr(predictor, "network", None)
    device = getattr(predictor, "device", None)
    configuration = getattr(predictor, "configuration_manager", None)
    if not isinstance(network, nn.Module) or not isinstance(device, torch.device):
        raise TypeError("predictor must expose an initialized network and torch device")
    if configuration is None or tuple(configuration.patch_size) != (64, 128, 192):
        raise ValueError("V6 extraction requires the locked 64x128x192 ResEnc-M plan")
    if getattr(predictor, "perform_everything_on_device", None) is not False:
        raise ValueError("V6 extraction requires CPU result ownership")
    if not getattr(predictor, "use_mirroring", False) or tuple(
        getattr(predictor, "allowed_mirroring_axes", ()) or ()
    ) != V6_MIRROR_AXES:
        raise ValueError("V6 extraction requires all eight mirror views")

    target_shape = adaptive_whole_volume_target_shape(input_image.shape[1:])
    padded, revert_slicer = pad_nd_image(
        input_image,
        target_shape,
        "constant",
        {"value": 0},
        True,
        None,
    )
    ct_grid = F.adaptive_avg_pool3d(padded[None].float(), SPATIAL_GRID)
    work = padded[None].to(device, memory_format=torch.contiguous_format)
    network.to(device)
    network.eval()

    sums: dict[str, Tensor] = {}
    autocast = torch.autocast(device.type, enabled=device.type == "cuda")
    with autocast:
        for axes in _mirror_views():
            view = torch.flip(work, axes) if axes else work
            runtime_counters.record_network_forward(1)
            segmentation_logits, skips, rescue_logits = _forward_shared_network(
                network, view
            )
            probabilities = torch.softmax(segmentation_logits.float(), dim=1)
            aligned_segmentation = (
                torch.flip(segmentation_logits.float(), axes)
                if axes
                else segmentation_logits.float()
            )
            aligned_probabilities = (
                torch.flip(probabilities, axes) if axes else probabilities
            )
            aligned_stage2 = torch.flip(skips[V6_STAGE2_INDEX].float(), axes) if axes else skips[
                V6_STAGE2_INDEX
            ].float()
            aligned_stage3 = torch.flip(skips[V6_STAGE3_INDEX].float(), axes) if axes else skips[
                V6_STAGE3_INDEX
            ].float()
            stage2_grid = F.adaptive_avg_pool3d(aligned_stage2, SPATIAL_GRID)
            stage3_grid = F.adaptive_avg_pool3d(aligned_stage3, SPATIAL_GRID)
            lesion_grid = F.adaptive_avg_pool3d(
                aligned_probabilities[:, 2:3], SPATIAL_GRID
            )
            whole_grid = F.adaptive_avg_pool3d(
                aligned_probabilities[:, 1:2] + aligned_probabilities[:, 2:3],
                SPATIAL_GRID,
            )
            stage5 = skips[V6_STAGE5_INDEX].float()
            lesion_at_stage5 = F.interpolate(
                probabilities[:, 2:3],
                size=stage5.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            stage5_global = stage5.mean(dim=(2, 3, 4))
            stage5_lesion = (stage5 * lesion_at_stage5).sum(dim=(2, 3, 4))
            stage5_lesion = stage5_lesion / lesion_at_stage5.sum(
                dim=(2, 3, 4)
            ).clamp_min(V6_STAGE5_WEIGHT_FLOOR)
            rescue_probabilities = torch.softmax(rescue_logits.float(), dim=1)
            current = {
                "segmentation": aligned_segmentation,
                "stage2": stage2_grid,
                "stage3": stage3_grid,
                "lesion": lesion_grid,
                "whole": whole_grid,
                "stage5_global": stage5_global,
                "stage5_lesion": stage5_lesion,
                "rescue_logits": rescue_logits.float(),
                "rescue_probabilities": rescue_probabilities,
            }
            for name, value in current.items():
                sums[name] = value if name not in sums else sums[name] + value
            runtime_counters.record_tta_batch(1)
    if runtime_counters.shared_network_forward_calls != V6_MIRROR_VIEW_COUNT:
        raise RuntimeError("V6 extraction did not execute exactly eight network forwards")

    means = {name: value / V6_MIRROR_VIEW_COUNT for name, value in sums.items()}
    cropped_logits = means["segmentation"][(0, slice(None), *revert_slicer[1:])]
    predicted_segmentation = cropped_logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
    morphology = predicted_mask_ct_features(
        input_image[0].cpu().numpy().astype(np.float32, copy=False),
        predicted_segmentation,
        spacing,
    )
    spatial = torch.cat(
        (
            means["stage2"],
            means["stage3"],
            means["lesion"],
            means["whole"],
            ct_grid.to(means["stage2"].device),
        ),
        dim=1,
    )[0]
    coordinates = normalized_v6_cell_coordinates().numpy()
    lesion_mass = means["lesion"].mean().reshape(())

    def cpu_array(value: Tensor) -> np.ndarray:
        return value.detach().float().cpu().numpy().astype(np.float32, copy=False)

    return V6ExtractedCase(
        spatial=cpu_array(spatial),
        coordinates=coordinates,
        case_lesion_mass=cpu_array(lesion_mass),
        stage5_global_mean=cpu_array(means["stage5_global"][0]),
        stage5_lesion_weighted_mean=cpu_array(means["stage5_lesion"][0]),
        rescue_mean_logits=cpu_array(means["rescue_logits"][0]),
        rescue_mean_probabilities=cpu_array(means["rescue_probabilities"][0]),
        morphology=morphology.values.astype(np.float32, copy=False),
        morphology_names=morphology.names,
        predicted_segmentation=predicted_segmentation,
        target_shape=target_shape,
    )


__all__ = [
    "V6_H100_ORCHESTRATION_LOCK_SHA256",
    "V6ExtractedCase",
    "adaptive_whole_volume_target_shape",
    "extract_v6_case_from_preprocessed",
    "normalized_v6_cell_coordinates",
]
