from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pancreas_multitask.classification_rescue import file_sha256
from pancreas_multitask.neural_case_bundle import (
    load_neural_case_head_bundle,
    predict_neural_case_bag,
    predict_neural_case_extraction,
    save_neural_case_head_bundle,
)
from pancreas_multitask.neural_case_head import (
    NEURAL_MEAN_CANDIDATE,
    V5_DECISION_LOCK_SHA256,
    V5_NEURAL_LOCK_SHA256,
    build_neural_case_bag,
    build_neural_case_head,
)
from pancreas_multitask.neural_case_training import neural_state_sha256


def _fake_extraction() -> SimpleNamespace:
    names = (
        *(f"rescue_logit_class_{index}" for index in range(3)),
        *(f"rescue_probability_class_{index}" for index in range(3)),
        *(f"encoder_stage_5_global_mean_channel_{channel:03d}" for channel in range(320)),
    )
    generator = np.random.default_rng(80)
    tile_vectors = generator.normal(size=(2, len(names))).astype(np.float32)
    tile_evidence = generator.uniform(size=(2, 7)).astype(np.float32)
    lesion = generator.uniform(0, 0.6, size=(2, 1, 4, 4, 6))
    whole = lesion + generator.uniform(size=(2, 1, 4, 4, 6)) * (1 - lesion)
    return SimpleNamespace(
        tile_vectors=tile_vectors,
        tile_evidence=tile_evidence,
        tile_vector_names=names,
        mil_stage3_maps=generator.normal(size=(2, 256, 4, 4, 6)).astype(np.float16),
        mil_prediction_maps=np.concatenate((lesion, whole), axis=1).astype(np.float16),
        mil_lesion_mass=np.sort(tile_evidence[:, 0])[::-1].copy(),
    )


def test_strict_bundle_round_trip_and_label_free_case_extraction(tmp_path) -> None:
    model = build_neural_case_head(NEURAL_MEAN_CANDIDATE)
    numeric_dataset_hash = "1" * 64
    metadata = {
        "eligible_for_official": True,
        "neural_lock_sha256": V5_NEURAL_LOCK_SHA256,
        "decision_lock_sha256": V5_DECISION_LOCK_SHA256,
        "numeric_content_dataset_sha256": numeric_dataset_hash,
        "selected_candidate_id": NEURAL_MEAN_CANDIDATE,
        "selection_audit_sha256": "2" * 64,
        "calibration_audit_sha256": "3" * 64,
        "refit_audit_sha256": "4" * 64,
        "refit_final_state_sha256": neural_state_sha256(model),
    }
    bundle_path = save_neural_case_head_bundle(
        tmp_path / "head.pth",
        model,
        candidate_id=NEURAL_MEAN_CANDIDATE,
        class_offsets=(0.25, 0.0, -0.25),
        metadata=metadata,
    )
    bundle_hash = file_sha256(bundle_path)
    extraction = _fake_extraction()
    shared_bag = build_neural_case_bag(
        tile_vectors=extraction.tile_vectors,
        tile_evidence=extraction.tile_evidence,
        tile_vector_names=extraction.tile_vector_names,
        mil_stage3_maps=extraction.mil_stage3_maps,
        mil_prediction_maps=extraction.mil_prediction_maps,
        mil_lesion_mass=extraction.mil_lesion_mass,
    )
    model.eval()
    before = predict_neural_case_bag(model, shared_bag, (0.25, 0.0, -0.25))

    loaded, offsets, loaded_metadata = load_neural_case_head_bundle(
        bundle_path,
        torch.device("cpu"),
        expected_bundle_sha256=bundle_hash,
        expected_numeric_dataset_sha256=numeric_dataset_hash,
    )
    prediction = predict_neural_case_extraction(
        loaded,
        extraction,
        offsets,
    )

    assert loaded_metadata == metadata
    assert prediction.logits.shape == (3,)
    assert prediction.raw_probabilities.dtype == torch.float64
    assert prediction.offset_probabilities.dtype == torch.float64
    one = torch.tensor(1.0, dtype=torch.float64)
    assert torch.isclose(prediction.raw_probabilities.sum(), one)
    assert torch.isclose(prediction.offset_probabilities.sum(), one)
    assert prediction.subtype in (0, 1, 2)
    assert torch.equal(before.logits, prediction.logits)
    assert torch.equal(before.raw_probabilities, prediction.raw_probabilities)
    assert torch.equal(before.offset_probabilities, prediction.offset_probabilities)
    assert before.subtype == prediction.subtype

    with pytest.raises(ValueError, match="final candidate lock"):
        load_neural_case_head_bundle(
            bundle_path,
            torch.device("cpu"),
            expected_bundle_sha256="0" * 64,
            expected_numeric_dataset_sha256=numeric_dataset_hash,
        )
