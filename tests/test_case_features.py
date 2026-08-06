from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pancreas_multitask.case_features import (
    FROZEN_NEURAL_HEAD_NAMES,
    MODEL_EVIDENCE_NAMES,
    aggregate_case_tiles,
    append_frozen_neural_head_features,
    build_case_feature_views,
    discover_train_cases,
    normalized_tile_centers,
    pool_multiscale_encoder_features,
    predicted_mask_ct_features,
    tile_model_evidence,
    train_case_inventory_audit,
)


def test_train_discovery_accepts_only_isolated_train_tree(tmp_path: Path) -> None:
    root = tmp_path / "train"
    for class_index in range(3):
        directory = root / f"subtype{class_index}"
        directory.mkdir(parents=True)
        (directory / f"opaque_{class_index}_0000.nii.gz").write_bytes(b"image")
        # A paired reference may exist, but discovery must not open it.
        (directory / f"opaque_{class_index}.nii.gz").write_bytes(b"do-not-open")

    cases = discover_train_cases(root, expected_count=3)
    audit = train_case_inventory_audit(cases)

    assert [case.label for case in cases] == [0, 1, 2]
    assert audit["case_count"] == 3
    assert audit["case_ids_excluded_from_model_matrix"] is True
    assert audit["combined_train_validation_metadata_read"] is False
    assert audit["ground_truth_masks_opened"] is False

    wrong_root = tmp_path / "validation"
    wrong_root.mkdir()
    with pytest.raises(ValueError, match="named 'train'"):
        discover_train_cases(wrong_root, expected_count=None)


def test_multiscale_pool_uses_model_probabilities_and_has_locked_statistics() -> None:
    torch.manual_seed(4)
    skips = [
        torch.randn(2, 1, 8, 8, 8),
        torch.randn(2, 2, 8, 4, 4),
        torch.randn(2, 3, 4, 4, 4),
        torch.randn(2, 4, 2, 4, 4),
        torch.randn(2, 5, 2, 2, 2),
        torch.randn(2, 6, 1, 2, 2),
    ]
    logits = torch.zeros(2, 3, 8, 8, 8)
    logits[:, 2, 2:6, 2:6, 2:6] = 5

    pooled, names = pool_multiscale_encoder_features(skips, logits)

    expected_channels = 3 + 4 + 5 + 6
    assert pooled.shape == (2, expected_channels * 7)
    assert len(names) == pooled.shape[1]
    assert len(names) == len(set(names))
    assert torch.isfinite(pooled).all()
    assert all("case" not in name and "filename" not in name for name in names)


def test_empty_predicted_lesion_masked_max_is_finite_zero() -> None:
    skips = [torch.zeros(1, 1, 2, 2, 2) for _ in range(6)]
    skips[2].fill_(8)
    logits = torch.zeros(1, 3, 2, 2, 2)
    logits[:, 0] = 20

    pooled, names = pool_multiscale_encoder_features(skips, logits, stage_indices=(2,))
    masked_max_indices = [index for index, name in enumerate(names) if "lesion_masked_max" in name]

    assert len(masked_max_indices) == 1
    assert pooled[0, masked_max_indices[0]].item() == 0


def test_tile_evidence_and_aggregation_are_permutation_invariant() -> None:
    logits = torch.zeros(4, 3, 2, 2, 2)
    logits[:, 2] = torch.tensor([0.0, 1.0, 2.0, 3.0])[:, None, None, None]
    centers = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
    evidence, evidence_names = tile_model_evidence(logits, centers)
    vectors = np.arange(24, dtype=np.float32).reshape(4, 6)
    vector_names = tuple(f"feature_{index}" for index in range(6))

    first = aggregate_case_tiles(vectors, evidence.numpy(), vector_names)
    permutation = np.asarray([2, 0, 3, 1])
    second = aggregate_case_tiles(vectors[permutation], evidence.numpy()[permutation], vector_names)

    assert evidence_names == MODEL_EVIDENCE_NAMES
    assert np.array_equal(first.values, second.values)
    assert first.names == second.names


def test_frozen_neural_head_is_part_of_each_tile_vector() -> None:
    encoder = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    logits = torch.tensor([[1.0, 0.0, -1.0]]).repeat(3, 1)
    probabilities = torch.softmax(logits, dim=1)

    values, names = append_frozen_neural_head_features(
        encoder,
        tuple(f"encoder_{index}" for index in range(4)),
        logits,
        probabilities,
    )

    assert values.shape == (3, 10)
    assert names[-6:] == FROZEN_NEURAL_HEAD_NAMES
    assert torch.allclose(values[:, -3:].sum(dim=1), torch.ones(3))


def test_normalized_tile_centers_have_geometry_only_values() -> None:
    slicers = [
        (slice(None), slice(0, 4), slice(0, 4), slice(0, 4)),
        (slice(None), slice(4, 8), slice(4, 8), slice(4, 8)),
    ]
    centers = normalized_tile_centers(slicers, (8, 8, 8))

    assert centers.shape == (2, 3)
    assert torch.all((centers >= 0) & (centers <= 1))
    assert torch.all(centers[0] < centers[1])


def test_predicted_mask_features_use_only_predicted_grid_and_handle_empty() -> None:
    image = np.linspace(-2, 2, 6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    predicted = np.zeros_like(image, dtype=np.uint8)

    empty = predicted_mask_ct_features(image, predicted, (2.0, 0.7, 0.7))
    assert np.isfinite(empty.values).all()
    assert empty.values[empty.names.index("predicted_lesion_empty")] == 1

    predicted[1:5, 1:6, 1:7] = 1
    predicted[2:4, 3:5, 3:6] = 2
    nonempty = predicted_mask_ct_features(image, predicted, (2.0, 0.7, 0.7))
    assert nonempty.values[nonempty.names.index("predicted_lesion_empty")] == 0
    assert nonempty.values[nonempty.names.index("predicted_lesion_volume_mm3")] > 0


def test_two_feature_views_share_encoder_prefix_and_never_take_ground_truth() -> None:
    encoder = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    logits = torch.tensor([[1.0, 0.0, -1.0]]).repeat(3, 1)
    vectors_tensor, vector_names = append_frozen_neural_head_features(
        encoder,
        tuple(f"encoder_{index}" for index in range(6)),
        logits,
        torch.softmax(logits, dim=1),
    )
    vectors = vectors_tensor.numpy()
    evidence = np.zeros((3, len(MODEL_EVIDENCE_NAMES)), dtype=np.float32)
    evidence[:, 0] = (0.1, 0.2, 0.3)
    image = np.ones((4, 4, 4), dtype=np.float32)
    prediction = np.zeros((4, 4, 4), dtype=np.uint8)
    prediction[1:3, 1:3, 1:3] = 2

    views = build_case_feature_views(
        vectors,
        evidence,
        vector_names,
        image,
        prediction,
        (1, 1, 1),
    )

    assert len(views) == 2
    encoder_view = views["multiscale_encoder_aggregates_plus_frozen_neural_head_plus_tile_evidence"]
    combined_view = views[
        "multiscale_encoder_aggregates_plus_frozen_neural_head_plus_tile_evidence_"
        "plus_predicted_mask_ct_features"
    ]
    assert np.array_equal(combined_view.values[: encoder_view.values.size], encoder_view.values)
    assert len(combined_view.values) > len(encoder_view.values)
