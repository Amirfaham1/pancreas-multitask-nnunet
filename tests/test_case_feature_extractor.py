from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pancreas_multitask.case_feature_extractor import (
    FeatureExtractionRuntimeCounters,
    extract_case_from_preprocessed,
    mirror_mean_tile_batch_features,
)


class _ToyEncoder(nn.Module):
    output_channels = (1, 2, 3, 256, 5, 6)

    def __init__(self):
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, data):
        self.batch_sizes.append(int(data.shape[0]))
        base = data.mean(dim=1, keepdim=True)
        return [
            base.repeat(1, channels, 1, 1, 1) * (stage + 1)
            for stage, channels in enumerate(self.output_channels)
        ]


class _OneShotBatchTwoOOMEncoder(_ToyEncoder):
    def __init__(self):
        super().__init__()
        self.failed = False

    def forward(self, data):
        if data.shape[0] == 2 and not self.failed:
            self.batch_sizes.append(2)
            self.failed = True
            raise RuntimeError("CUDA out of memory: synthetic unit-test failure")
        return super().forward(data)


class _ToyDecoder(nn.Module):
    def forward(self, skips):
        base = skips[0][:, :1]
        return torch.cat((-base, base * 0.5, base), dim=1)


class _ToySharedNetwork(nn.Module):
    def __init__(self, encoder: nn.Module | None = None):
        super().__init__()
        self.encoder = _ToyEncoder() if encoder is None else encoder
        self.decoder = _ToyDecoder()

    def classify_bottleneck(self, bottleneck):
        value = bottleneck.mean(dim=(1, 2, 3, 4))
        return torch.stack((value, -value, value * 0.25), dim=1)


class _ToyPredictor:
    def __init__(self, tile_count: int = 2, network: nn.Module | None = None):
        if tile_count < 1:
            raise ValueError("tile_count must be positive")
        self.tile_count = tile_count
        self.network = _ToySharedNetwork() if network is None else network
        self.device = torch.device("cpu")
        self.configuration_manager = SimpleNamespace(patch_size=(4, 4, 4))
        self.label_manager = SimpleNamespace(num_segmentation_heads=3)
        self.perform_everything_on_device = False
        self.use_gaussian = False
        self.use_mirroring = True
        self.allowed_mirroring_axes = (0, 1, 2)

    def _internal_get_sliding_window_slicers(self, shape):
        assert tuple(shape) == (4, 4, 4 * self.tile_count)
        return [
            (
                slice(None),
                slice(0, 4),
                slice(0, 4),
                slice(index * 4, (index + 1) * 4),
            )
            for index in range(self.tile_count)
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
    assert sequential.segmentation_logits.dtype == torch.float16
    assert batched.segmentation_logits.dtype == torch.float16
    assert np.array_equal(sequential.tile_vectors, batched.tile_vectors)
    assert np.array_equal(sequential.tile_evidence, batched.tile_evidence)
    assert np.array_equal(sequential.mil_stage3_maps, batched.mil_stage3_maps)
    assert np.array_equal(sequential.mil_prediction_maps, batched.mil_prediction_maps)
    assert sequential.tile_vector_names == batched.tile_vector_names
    assert sequential.tile_count == batched.tile_count == 2


def test_batch_one_tta_path_is_exact_and_records_original_schedule() -> None:
    predictor = _ToyPredictor(tile_count=3)
    image = torch.arange(4 * 4 * 12, dtype=torch.float32).reshape(1, 4, 4, 12) / 100
    first_counters = FeatureExtractionRuntimeCounters(1, 1)
    second_counters = FeatureExtractionRuntimeCounters(1, 1)

    first = extract_case_from_preprocessed(
        predictor,
        image,
        tile_batch_size=1,
        tta_batch_size=1,
        runtime_counters=first_counters,
    )
    second = extract_case_from_preprocessed(
        predictor,
        image,
        tile_batch_size=1,
        tta_batch_size=1,
        runtime_counters=second_counters,
    )

    assert torch.equal(first.segmentation_logits, second.segmentation_logits)
    assert np.array_equal(first.tile_vectors, second.tile_vectors)
    assert np.array_equal(first.tile_evidence, second.tile_evidence)
    assert np.array_equal(first.mil_stage3_maps, second.mil_stage3_maps)
    assert np.array_equal(first.mil_prediction_maps, second.mil_prediction_maps)
    assert first_counters.provenance() == second_counters.provenance() == {
        "joint_network_forward_calls": 24,
        "shared_network_forward_calls": 24,
        "maximum_network_batch_size_observed": 1,
        "network_batch_size_histogram": {"1": 24},
        "network_batch_size_limit": 1,
        "locked_network_microbatch_ceiling": 2,
        "logical_tile_batches_completed": 3,
        "logical_tiles_completed": 3,
        "tile_batch_oom_fallback_count": 0,
        "tile_batch_size_adaptive_limit": 1,
        "tile_batch_size_histogram": {"1": 3},
        "tile_batch_size_requested": 1,
        "tta_batch_oom_fallback_count": 0,
        "tta_batch_size_adaptive_limit": 1,
        "tta_batch_size_histogram": {"1": 24},
        "tta_batch_size_requested": 1,
        "tta_view_batches_completed": 24,
        "tta_views_completed": 24,
    }


def test_one_tile_uses_two_view_microbatches_under_shared_ceiling() -> None:
    reference_network = _ToySharedNetwork()
    candidate_network = _ToySharedNetwork()
    data = torch.linspace(-1, 1, 4 * 4 * 4).reshape(1, 1, 4, 4, 4)
    reference_counters = FeatureExtractionRuntimeCounters(1, 1)
    candidate_counters = FeatureExtractionRuntimeCounters(2, 2)

    reference = mirror_mean_tile_batch_features(
        reference_network,
        data,
        allowed_mirroring_axes=(0, 1, 2),
        tta_batch_size=1,
        runtime_counters=reference_counters,
    )
    candidate = mirror_mean_tile_batch_features(
        candidate_network,
        data,
        allowed_mirroring_axes=(0, 1, 2),
        tta_batch_size=2,
        runtime_counters=candidate_counters,
    )

    assert torch.equal(reference.segmentation_logits, candidate.segmentation_logits)
    assert torch.equal(reference.tile_vectors, candidate.tile_vectors)
    assert torch.equal(reference.mil_stage3_maps, candidate.mil_stage3_maps)
    assert torch.equal(reference.mil_prediction_maps, candidate.mil_prediction_maps)
    assert candidate_network.encoder.batch_sizes == [2, 2, 2, 2]
    provenance = candidate_counters.provenance()
    assert provenance["shared_network_forward_calls"] == 4
    assert provenance["maximum_network_batch_size_observed"] == 2
    assert provenance["network_batch_size_histogram"] == {"2": 4}
    assert provenance["tta_batch_size_histogram"] == {"2": 4}
    assert provenance["tta_views_completed"] == 8


def test_shared_ceiling_batches_tiles_or_views_and_preserves_features() -> None:
    reference_predictor = _ToyPredictor(tile_count=3)
    candidate_predictor = _ToyPredictor(tile_count=3)
    image = torch.arange(4 * 4 * 12, dtype=torch.float32).reshape(1, 4, 4, 12) / 100
    reference_counters = FeatureExtractionRuntimeCounters(1, 1)
    candidate_counters = FeatureExtractionRuntimeCounters(2, 2)

    reference = extract_case_from_preprocessed(
        reference_predictor,
        image,
        tile_batch_size=1,
        tta_batch_size=1,
        runtime_counters=reference_counters,
    )
    candidate = extract_case_from_preprocessed(
        candidate_predictor,
        image,
        tile_batch_size=2,
        tta_batch_size=2,
        runtime_counters=candidate_counters,
    )

    assert torch.equal(reference.segmentation_logits, candidate.segmentation_logits)
    assert np.array_equal(reference.tile_vectors, candidate.tile_vectors)
    assert np.array_equal(reference.tile_evidence, candidate.tile_evidence)
    assert np.array_equal(reference.mil_stage3_maps, candidate.mil_stage3_maps)
    assert np.array_equal(reference.mil_prediction_maps, candidate.mil_prediction_maps)
    assert np.array_equal(reference.mil_lesion_mass, candidate.mil_lesion_mass)
    assert candidate_predictor.network.encoder.batch_sizes == [2] * 12

    provenance = candidate_counters.provenance()
    assert provenance["maximum_network_batch_size_observed"] == 2
    assert provenance["network_batch_size_histogram"] == {"2": 12}
    assert provenance["network_batch_size_limit"] == 2
    assert provenance["tile_batch_size_histogram"] == {"1": 1, "2": 1}
    assert provenance["tta_batch_size_histogram"] == {"1": 8, "2": 4}
    assert provenance["logical_tiles_completed"] == 3
    assert provenance["tta_views_completed"] == 16
    assert provenance["tile_batch_oom_fallback_count"] == 0
    assert provenance["tta_batch_oom_fallback_count"] == 0


def test_tta_oom_retries_untouched_views_at_batch_one() -> None:
    reference_network = _ToySharedNetwork()
    fallback_network = _ToySharedNetwork(_OneShotBatchTwoOOMEncoder())
    data = torch.linspace(-1, 1, 4 * 4 * 4).reshape(1, 1, 4, 4, 4)

    reference = mirror_mean_tile_batch_features(
        reference_network,
        data,
        allowed_mirroring_axes=(0, 1, 2),
        tta_batch_size=1,
    )
    counters = FeatureExtractionRuntimeCounters(2, 2)
    fallback = mirror_mean_tile_batch_features(
        fallback_network,
        data,
        allowed_mirroring_axes=(0, 1, 2),
        tta_batch_size=2,
        runtime_counters=counters,
    )

    assert torch.equal(reference.segmentation_logits, fallback.segmentation_logits)
    assert torch.equal(reference.tile_vectors, fallback.tile_vectors)
    provenance = counters.provenance()
    assert provenance["tta_batch_oom_fallback_count"] == 1
    assert provenance["tta_batch_size_adaptive_limit"] == 1
    assert provenance["network_batch_size_histogram"] == {"1": 8, "2": 1}
    assert provenance["tta_batch_size_histogram"] == {"1": 8}


def test_tile_oom_retries_untouched_group_and_then_packs_views() -> None:
    reference_predictor = _ToyPredictor(tile_count=3)
    fallback_predictor = _ToyPredictor(
        tile_count=3,
        network=_ToySharedNetwork(_OneShotBatchTwoOOMEncoder()),
    )
    image = torch.arange(4 * 4 * 12, dtype=torch.float32).reshape(1, 4, 4, 12) / 100
    reference = extract_case_from_preprocessed(
        reference_predictor,
        image,
        tile_batch_size=1,
        tta_batch_size=1,
    )
    counters = FeatureExtractionRuntimeCounters(2, 2)
    fallback = extract_case_from_preprocessed(
        fallback_predictor,
        image,
        tile_batch_size=2,
        tta_batch_size=2,
        runtime_counters=counters,
    )

    assert torch.equal(reference.segmentation_logits, fallback.segmentation_logits)
    assert np.array_equal(reference.tile_vectors, fallback.tile_vectors)
    assert np.array_equal(reference.mil_stage3_maps, fallback.mil_stage3_maps)
    provenance = counters.provenance()
    assert provenance["tile_batch_oom_fallback_count"] == 1
    assert provenance["tile_batch_size_adaptive_limit"] == 1
    assert provenance["tta_batch_oom_fallback_count"] == 0
    assert provenance["tile_batch_size_histogram"] == {"1": 3}
    assert provenance["tta_batch_size_histogram"] == {"2": 12}
    assert provenance["network_batch_size_histogram"] == {"2": 13}
