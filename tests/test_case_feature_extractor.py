from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pancreas_multitask.case_feature_extractor import (
    extract_case_from_preprocessed,
    mirror_mean_tile_batch_features,
)


class _ToyEncoder(nn.Module):
    output_channels = (1, 2, 3, 256, 5, 6)

    def forward(self, data):
        base = data.mean(dim=1, keepdim=True)
        return [
            base.repeat(1, channels, 1, 1, 1) * (stage + 1)
            for stage, channels in enumerate(self.output_channels)
        ]


class _ToyDecoder(nn.Module):
    def forward(self, skips):
        base = skips[0][:, :1]
        return torch.cat((-base, base * 0.5, base), dim=1)


class _ToySharedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _ToyEncoder()
        self.decoder = _ToyDecoder()

    def classify_bottleneck(self, bottleneck):
        value = bottleneck.mean(dim=(1, 2, 3, 4))
        return torch.stack((value, -value, value * 0.25), dim=1)


class _ToyPredictor:
    def __init__(self):
        self.network = _ToySharedNetwork()
        self.device = torch.device("cpu")
        self.configuration_manager = SimpleNamespace(patch_size=(4, 4, 4))
        self.label_manager = SimpleNamespace(num_segmentation_heads=3)
        self.perform_everything_on_device = False
        self.use_gaussian = False
        self.use_mirroring = True
        self.allowed_mirroring_axes = (0, 1, 2)

    @staticmethod
    def _internal_get_sliding_window_slicers(shape):
        assert tuple(shape) == (4, 4, 8)
        return [
            (slice(None), slice(0, 4), slice(0, 4), slice(0, 4)),
            (slice(None), slice(0, 4), slice(0, 4), slice(4, 8)),
        ]


def test_mirror_feature_path_includes_rescue_head_and_spatial_cache() -> None:
    network = _ToySharedNetwork()
    data = torch.linspace(-1, 1, 2 * 4 * 4 * 4).reshape(2, 1, 4, 4, 4)

    result = mirror_mean_tile_batch_features(
        network,
        data,
        allowed_mirroring_axes=(0, 1, 2),
    )

    assert result.segmentation_logits.shape == (2, 3, 4, 4, 4)
    assert result.tile_vectors.shape[0] == 2
    assert result.tile_vector_names[-6:] == (
        "rescue_logit_class_0",
        "rescue_logit_class_1",
        "rescue_logit_class_2",
        "rescue_probability_class_0",
        "rescue_probability_class_1",
        "rescue_probability_class_2",
    )
    assert result.mil_stage3_maps.shape == (2, 256, 4, 4, 6)
    assert result.mil_prediction_maps.shape == (2, 2, 4, 4, 6)
    assert torch.allclose(result.mirror_mean_probabilities.sum(dim=1), torch.ones(2))


def test_case_extraction_is_tile_batch_invariant() -> None:
    predictor = _ToyPredictor()
    image = torch.arange(4 * 4 * 8, dtype=torch.float32).reshape(1, 4, 4, 8) / 100

    sequential = extract_case_from_preprocessed(predictor, image, tile_batch_size=1)
    batched = extract_case_from_preprocessed(predictor, image, tile_batch_size=2)

    assert torch.equal(sequential.segmentation_logits, batched.segmentation_logits)
    assert np.array_equal(sequential.tile_vectors, batched.tile_vectors)
    assert np.array_equal(sequential.tile_evidence, batched.tile_evidence)
    assert np.array_equal(sequential.mil_stage3_maps, batched.mil_stage3_maps)
    assert np.array_equal(sequential.mil_prediction_maps, batched.mil_prediction_maps)
    assert sequential.tile_vector_names == batched.tile_vector_names
    assert sequential.tile_count == batched.tile_count == 2
