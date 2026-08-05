from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.network import (
    HybridCrossAttentionPool3D,
    MultiTaskResEncUNet,
    _compatible_attention_heads,
)


class _ToyEncoder(nn.Module):
    output_channels = (4, 8)

    def __init__(self) -> None:
        super().__init__()
        self.stage_one = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.stage_two = nn.Conv3d(4, 8, kernel_size=3, padding=1)

    def forward(self, x):
        high_resolution = self.stage_one(x)
        bottleneck = self.stage_two(F.avg_pool3d(high_resolution, kernel_size=2))
        return [high_resolution, bottleneck]

    @staticmethod
    def compute_conv_feature_map_size(_input_size):
        return 100


class _ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deep_supervision = False
        self.segmentation = nn.Conv3d(4, 3, kernel_size=1)

    def forward(self, skips):
        output = self.segmentation(skips[0])
        if self.deep_supervision:
            return [output, F.avg_pool3d(output, kernel_size=2)]
        return output

    @staticmethod
    def compute_conv_feature_map_size(_input_size):
        return 50


class _ToySegmentationNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ToyEncoder()
        self.decoder = _ToyDecoder()


def test_default_forward_remains_segmentation_only() -> None:
    model = MultiTaskResEncUNet(
        _ToySegmentationNetwork(),
        classification_dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 1, 8, 8, 8)

    segmentation_only = model(inputs)
    segmentation_joint, classification = model(
        inputs,
        return_classification=True,
    )

    assert isinstance(segmentation_only, torch.Tensor)
    assert torch.equal(segmentation_only, segmentation_joint)
    assert segmentation_only.shape == (2, 3, 8, 8, 8)
    assert classification.shape == (2, 3)


def test_deep_supervision_and_decoder_attribute_are_preserved() -> None:
    model = MultiTaskResEncUNet(_ToySegmentationNetwork())
    model.decoder.deep_supervision = True

    segmentation, classification = model(
        torch.randn(2, 1, 8, 8, 8),
        return_classification=True,
    )

    assert isinstance(segmentation, list)
    assert [item.shape for item in segmentation] == [
        (2, 3, 8, 8, 8),
        (2, 3, 4, 4, 4),
    ]
    assert classification.shape == (2, 3)


def test_joint_loss_backpropagates_into_all_branches() -> None:
    model = MultiTaskResEncUNet(
        _ToySegmentationNetwork(),
        classification_dropout=0.0,
    )
    segmentation, classification = model(
        torch.randn(2, 1, 8, 8, 8),
        return_classification=True,
    )
    loss = segmentation.square().mean() + classification.square().mean()
    loss.backward()

    assert model.encoder.stage_one.weight.grad is not None
    assert model.decoder.segmentation.weight.grad is not None
    final_linear = model.classification_head[-1]
    assert isinstance(final_linear, nn.Linear)
    assert final_linear.weight.grad is not None


def test_attention_head_count_is_made_channel_compatible() -> None:
    assert _compatible_attention_heads(10, 8) == 5
    assert _compatible_attention_heads(7, 8) == 7
    pool = HybridCrossAttentionPool3D(10, requested_heads=8)
    assert pool(torch.randn(2, 10, 2, 3, 4)).shape == (2, 20)


def test_attention_pool_rejects_non_3d_feature_maps() -> None:
    pool = HybridCrossAttentionPool3D(8)
    with pytest.raises(ValueError, match="5D tensor"):
        pool(torch.randn(2, 8, 16, 16))


def test_feature_map_size_delegates_to_segmentation_backbone() -> None:
    model = MultiTaskResEncUNet(_ToySegmentationNetwork())
    assert model.compute_conv_feature_map_size((8, 8, 8)) == 150
