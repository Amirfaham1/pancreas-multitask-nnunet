from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pancreas_multitask.case_feature_extractor import FeatureExtractionRuntimeCounters
from pancreas_multitask.v6_case_extractor import (
    extract_v6_case_from_preprocessed,
    normalized_v6_cell_coordinates,
)


class _MockEncoder(nn.Module):
    def forward(self, data: torch.Tensor) -> list[torch.Tensor]:
        return [
            data,
            F.adaptive_avg_pool3d(data, (8, 8, 12)),
            F.adaptive_avg_pool3d(data, (8, 8, 12)).repeat(1, 128, 1, 1, 1),
            F.adaptive_avg_pool3d(data, (8, 8, 12)).repeat(1, 256, 1, 1, 1),
            F.adaptive_avg_pool3d(data, (2, 2, 3)),
            F.adaptive_avg_pool3d(data, (1, 1, 1)).repeat(1, 320, 1, 1, 1),
        ]


class _MockDecoder(nn.Module):
    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        image = skips[0]
        return torch.cat((-image, torch.zeros_like(image), image), dim=1)


class _MockNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _MockEncoder()
        self.decoder = _MockDecoder()

    @staticmethod
    def classify_bottleneck(bottleneck: torch.Tensor) -> torch.Tensor:
        score = bottleneck.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return torch.cat((-score, torch.zeros_like(score), score), dim=1)


def test_normalized_coordinates_are_locked_cell_centers() -> None:
    coordinates = normalized_v6_cell_coordinates()
    assert coordinates.shape == (3, 8, 8, 12)
    assert torch.isclose(coordinates[0, 0, 0, 0], torch.tensor(0.5 / 8))
    assert torch.isclose(coordinates[2, -1, -1, -1], torch.tensor(11.5 / 12))


def test_v6_extractor_runs_exactly_eight_views_and_emits_float16_cache() -> None:
    predictor = SimpleNamespace(
        network=_MockNetwork(),
        device=torch.device("cpu"),
        configuration_manager=SimpleNamespace(patch_size=(64, 128, 192)),
        perform_everything_on_device=False,
        use_mirroring=True,
        allowed_mirroring_axes=(0, 1, 2),
    )
    counters = FeatureExtractionRuntimeCounters(1, 1)
    image = torch.zeros(1, 8, 9, 10, dtype=torch.float32)
    extracted = extract_v6_case_from_preprocessed(
        predictor,
        image,
        spacing=(1.0, 1.0, 1.0),
        runtime_counters=counters,
    )

    assert counters.shared_network_forward_calls == 8
    assert counters.tta_views_completed == 8
    assert extracted.spatial.shape == (387, 8, 8, 12)
    assert extracted.predicted_segmentation.shape == (8, 9, 10)
    assert extracted.target_shape == (64, 128, 192)
    assert np.isclose(extracted.case_lesion_mass, 1 / 3, atol=1e-6)
    assert all(value.dtype == np.float16 for value in extracted.cache_arrays().values())
