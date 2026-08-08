from __future__ import annotations

import torch

from pancreas_multitask.wholevolume_dataset import (
    pad_to_stride,
    replica_seed,
    stride_for_stage,
)


def test_stage_stride_and_padding_are_depth_aware() -> None:
    assert stride_for_stage(1) == (1, 2, 2)
    assert stride_for_stage(2) == (2, 4, 4)

    volume = torch.zeros(1, 5, 9, 10)
    padded = pad_to_stride(volume, stride_for_stage(2))
    assert padded.shape == (1, 6, 12, 12)
    torch.testing.assert_close(padded[:, :5, :9, :10], volume)


def test_replica_seed_is_stable_and_replica_specific() -> None:
    digest = "a" * 64
    assert replica_seed(digest, 3) == replica_seed(digest, 3)
    assert replica_seed(digest, 3) != replica_seed(digest, 4)
