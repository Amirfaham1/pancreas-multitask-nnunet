"""Whole-ROI case dataset with live nnU-Net augmentation.

Why this exists
---------------
Every case-level classifier tried so far was trained on a *fixed cache* of 252
feature vectors, one per training case.  With nothing varying between epochs, a
head with more parameters than cases can simply memorize the table, and that is
exactly what was observed: resubstitution macro-F1 0.96-0.98 against 0.49-0.51
out-of-fold.  No amount of head redesign fixes a dataset of 252 constants.

Augmenting requires re-running the encoder, which the cached-feature design made
impossible.  It is affordable here because the supplied ROIs are small -- 94 of
the 252 cases fit inside a single 64x128x192 patch and the largest is
96x192x320 -- so a whole-volume forward is roughly 0.05 s/case, about 20 s per
epoch over the full training set.

The augmentation is nnU-Net's own training pipeline with the same parameters the
segmentation backbone was trained under.  Using a different distribution would
push the features off the manifold the frozen encoder understands.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import torch
from torch import Tensor

#: Cumulative encoder stride through stage 2 on the locked ResEnc-M plan
#: (per-stage strides [1,1,1], [1,2,2], [2,2,2]).  A stage-2 tap therefore only
#: needs the volume padded to a multiple of this, not to the full 64x128x192
#: patch grid the bottleneck would demand.
STAGE2_STRIDE: Final = (2, 4, 4)

#: nnU-Net's 3D defaults for this plan, from
#: ``configure_rotation_dummyDA_mirroring_and_inital_patch_size``.
ROTATION_FOR_DA: Final = (-30.0 / 360 * 2 * np.pi, 30.0 / 360 * 2 * np.pi)
MIRROR_AXES: Final = (0, 1, 2)


#: Per-stage strides of the locked ResEnc-M encoder.
ENCODER_STAGE_STRIDES: Final = ((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))


def stride_for_stage(stage: int) -> tuple[int, int, int]:
    """Cumulative encoder stride up to and including ``stage``.

    A volume must be padded to a multiple of this before a partial encoder pass,
    or the residual additions inside the stage blocks hit mismatched shapes
    (stage 3 on an odd-sized ROI fails with "size of tensor a (20) must match
    tensor b (19)"). Stage 2 needs (2,4,4); stage 3 needs (4,8,8).
    """

    index = int(stage)
    if index < 0:
        index += len(ENCODER_STAGE_STRIDES)
    if not 0 <= index < len(ENCODER_STAGE_STRIDES):
        raise ValueError(f"stage {stage} is outside the {len(ENCODER_STAGE_STRIDES)}-stage encoder")
    cumulative = [1, 1, 1]
    for position in range(index + 1):
        for axis in range(3):
            cumulative[axis] *= ENCODER_STAGE_STRIDES[position][axis]
    return tuple(cumulative)  # type: ignore[return-value]


def pad_to_stride(volume: Tensor, stride: Sequence[int] = STAGE2_STRIDE) -> Tensor:
    """Zero-pad the trailing three axes up to a multiple of ``stride``.

    At most ``stride - 1`` voxels per axis, so under ~1% of a typical ROI. The
    bottleneck tap by contrast requires padding to at least 64x128x192, which for
    these ROIs means a mean valid-voxel fraction of only 0.64.
    """

    if volume.ndim != 4:
        raise ValueError(f"Expected a (channels, D, H, W) volume, got {tuple(volume.shape)}")
    spatial = volume.shape[1:]
    padding: list[int] = []
    for size, step in zip(reversed(spatial), reversed(tuple(stride))):
        if step < 1:
            raise ValueError("Stride entries must be positive")
        padding.extend((0, (-size) % step))
    if not any(padding):
        return volume
    return torch.nn.functional.pad(volume, padding, mode="constant", value=0.0)


@dataclass(frozen=True, slots=True)
class PreprocessedCase:
    """One nnU-Net-preprocessed training case held in memory as float16."""

    case_id: str
    label: int
    image_sha256: str
    volume: np.ndarray

    def __post_init__(self) -> None:
        if self.volume.ndim != 4 or self.volume.shape[0] != 1:
            raise ValueError(f"Preprocessed volume must be (1, D, H, W), got {self.volume.shape}")
        if not 0 <= int(self.label) < 3:
            raise ValueError("Subtype label must be in {0, 1, 2}")
        if len(self.image_sha256) != 64:
            raise ValueError("Training image SHA-256 is malformed")


class PreprocessedCaseCache:
    """Run nnU-Net preprocessing once per case and reuse it across epochs.

    Preprocessing (read, crop, resample, normalize) costs roughly 0.6-1.0 s per
    case and is deterministic, so it is hoisted out of the augmentation loop.
    Augmentation is applied afterwards, to the cached array, which is what keeps
    live augmentation cheap.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, image_sha256: str) -> Path:
        if len(image_sha256) != 64:
            raise ValueError("Training image SHA-256 is malformed")
        return self._root / f"volume_{image_sha256}.npy"

    def load_or_build(
        self,
        *,
        case_id: str,
        label: int,
        image_path: Path,
        image_sha256: str,
        preprocessor: Any,
        plans_manager: Any,
        configuration_manager: Any,
        dataset_json: Any,
    ) -> PreprocessedCase:
        path = self._path(image_sha256)
        if path.is_file():
            volume = np.load(path, allow_pickle=False)
        else:
            data, _segmentation, _properties = preprocessor.run_case(
                [str(image_path)],
                None,
                plans_manager,
                configuration_manager,
                dataset_json,
            )
            volume = np.asarray(data, dtype=np.float32).astype(np.float16)
            temporary = path.with_name(f".{path.name}.tmp")
            # Write through a handle: np.save appends ".npy" to a path that lacks
            # it, which would silently place the file somewhere else.
            try:
                with temporary.open("wb") as handle:
                    np.save(handle, volume, allow_pickle=False)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if not np.isfinite(volume).all():
            raise ValueError(f"Preprocessed volume for {case_id} contains non-finite voxels")
        return PreprocessedCase(
            case_id=str(case_id),
            label=int(label),
            image_sha256=str(image_sha256),
            volume=volume,
        )


def replica_seed(image_sha256: str, replica: int) -> int:
    """Derive a stable per-(case, replica) seed from content, never from ordering.

    Keyed on the image hash rather than the case ID or enumeration index so the
    augmentation is reproducible and independent of directory order, matching the
    canonical row-ordering rule used elsewhere in this repository.
    """

    digest = hashlib.sha256(f"{image_sha256}:{int(replica)}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def build_training_transform(patch_size: Sequence[int], *, mode: str = "full") -> Any:
    """Augmentation for the classification stream.

    ``mode="full"`` is nnU-Net's own training pipeline, i.e. what the segmentation
    backbone was trained under.

    ``mode="geometry"`` keeps only the spatial and mirroring operations and drops
    every intensity operation (Gaussian noise, blur, multiplicative brightness,
    contrast, low-resolution simulation, both gamma variants).

    That split exists because the full pipeline measurably *destroys* the subtype
    signal.  Training a linear probe on K augmented replicas and scoring held-out
    cases on the clean view degrades monotonically -- 0.659 at K=1, 0.603 at K=2,
    0.554 at K=8, 0.547 at K=16 -- and a lone augmented draw scores 0.482 against
    0.650 clean.  Subtype here is a global parenchymal intensity/texture property,
    so the intensity augmentations that make segmentation robust are erasing the
    classifier's evidence.

    ``deep_supervision_scales=None`` because there is no decoder in this path.
    """

    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    if mode not in {"full", "geometry"}:
        raise ValueError(f"Unknown augmentation mode: {mode}")

    if mode == "full":
        return nnUNetTrainer.get_training_transforms(
            patch_size=np.asarray(patch_size),
            rotation_for_DA=ROTATION_FOR_DA,
            deep_supervision_scales=None,
            mirror_axes=MIRROR_AXES,
            do_dummy_2d_data_aug=False,
            use_mask_for_norm=None,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None,
        )

    from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
    from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
    from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

    return ComposeTransforms(
        [
            SpatialTransform(
                np.asarray(patch_size),
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=ROTATION_FOR_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                border_mode_seg="constant",
                padding_value_seg=-1,
            ),
            MirrorTransform(allowed_axes=set(MIRROR_AXES)),
        ]
    )


def augment_volume(volume: np.ndarray, transform: Any, seed: int) -> Tensor:
    """Apply one seeded augmentation draw to a cached whole ROI.

    Replica 0 is reserved by callers for the identity view, so the held-out
    evaluation feature is always the un-augmented one.
    """

    image = torch.from_numpy(np.asarray(volume, dtype=np.float32))
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected a (1, D, H, W) volume, got {tuple(image.shape)}")
    generator = torch.Generator().manual_seed(int(seed))
    state = torch.random.get_rng_state()
    try:
        torch.random.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        output = transform(**{"image": image, "segmentation": torch.zeros_like(image)})
    finally:
        torch.random.set_rng_state(state)
    augmented = output["image"] if isinstance(output, dict) else output
    augmented = torch.as_tensor(augmented, dtype=torch.float32)
    if augmented.ndim == 5:
        augmented = augmented[0]
    if not torch.isfinite(augmented).all():
        raise ValueError("Augmentation produced non-finite voxels")
    return augmented


__all__ = [
    "MIRROR_AXES",
    "ROTATION_FOR_DA",
    "STAGE2_STRIDE",
    "PreprocessedCase",
    "PreprocessedCaseCache",
    "augment_volume",
    "build_training_transform",
    "pad_to_stride",
    "replica_seed",
]
