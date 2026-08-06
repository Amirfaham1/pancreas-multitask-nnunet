"""Serialization and label-free inference for the selected locked v5 neural head."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from pancreas_multitask.classification_rescue import file_sha256
from pancreas_multitask.neural_case_head import (
    NEURAL_CANDIDATES,
    V5_DECISION_LOCK_SHA256,
    V5_NEURAL_LOCK_SHA256,
    NeuralCaseBag,
    build_neural_case_bag,
    build_neural_case_head,
    neural_bag_inference_tensors,
    neural_head_parameter_count,
)
from pancreas_multitask.neural_case_training import neural_state_sha256


@dataclass(frozen=True, slots=True)
class NeuralCasePrediction:
    logits: Tensor
    raw_probabilities: Tensor
    offset_probabilities: Tensor
    subtype: int


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_offsets(values: Sequence[float]) -> np.ndarray:
    offsets = np.asarray(values, dtype=np.float64)
    grid = np.asarray(
        [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=np.float64,
    )
    if (
        offsets.shape != (3,)
        or not np.isfinite(offsets).all()
        or offsets[1] != 0
        or offsets[0] not in grid
        or offsets[2] not in grid
    ):
        raise ValueError("V5 class offsets differ from the locked grid")
    return offsets


def save_neural_case_head_bundle(
    destination: str | Path,
    model: nn.Module,
    *,
    candidate_id: str,
    class_offsets: Sequence[float],
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically save only the selected head state and audited numeric decision rule."""

    if candidate_id not in NEURAL_CANDIDATES:
        raise ValueError("A v5 bundle must identify one of the two locked neural heads")
    offsets = _validated_offsets(class_offsets)
    expected_state_hash = neural_state_sha256(model)
    if metadata.get("refit_final_state_sha256") != expected_state_hash:
        raise ValueError("Bundled model state differs from the final-refit audit binding")
    expected_model = build_neural_case_head(candidate_id)
    expected_model.load_state_dict(model.state_dict(), strict=True)
    payload = {
        "schema_version": 1,
        "model_family": "assignment_conforming_v5_neural_case_head",
        "candidate_id": candidate_id,
        "trainable_parameter_count": neural_head_parameter_count(candidate_id),
        "class_offsets": [float(value) for value in offsets],
        "state_sha256": expected_state_hash,
        "state_dict": {
            name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()
        },
        "metadata": dict(metadata),
    }
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_neural_case_head_bundle(
    source: str | Path,
    device: torch.device,
    *,
    expected_bundle_sha256: str,
    expected_numeric_dataset_sha256: str,
) -> tuple[nn.Module, np.ndarray, dict[str, Any]]:
    """Fail closed when a serialized v5 head differs from its internal audit."""

    path = Path(source).expanduser().resolve()
    if not _is_sha256(expected_bundle_sha256) or file_sha256(path) != (expected_bundle_sha256):
        raise ValueError("Neural case-head bundle differs from the final candidate lock")
    if not _is_sha256(expected_numeric_dataset_sha256):
        raise ValueError("Expected numeric training-dataset hash is invalid")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Neural case-head bundle schema is invalid")
    if payload.get("model_family") != "assignment_conforming_v5_neural_case_head":
        raise ValueError("Serialized classifier is not an eligible v5 neural head")
    candidate_id = str(payload.get("candidate_id"))
    if candidate_id not in NEURAL_CANDIDATES:
        raise ValueError("Neural bundle has an unknown candidate ID")
    if int(payload.get("trainable_parameter_count", -1)) != neural_head_parameter_count(
        candidate_id
    ):
        raise ValueError("Neural bundle parameter count differs from the lock")
    model = build_neural_case_head(candidate_id)
    model.load_state_dict(payload["state_dict"], strict=True)
    if neural_state_sha256(model) != payload.get("state_sha256"):
        raise ValueError("Neural bundle state hash is invalid")
    offsets = _validated_offsets(payload["class_offsets"])
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("eligible_for_official") is not True:
        raise ValueError("Neural bundle is not marked eligible for the one official run")
    required_metadata = {
        "neural_lock_sha256": V5_NEURAL_LOCK_SHA256,
        "decision_lock_sha256": V5_DECISION_LOCK_SHA256,
        "numeric_content_dataset_sha256": expected_numeric_dataset_sha256,
        "selected_candidate_id": candidate_id,
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise ValueError("Neural bundle lock, dataset, or candidate binding is invalid")
    if metadata.get("refit_final_state_sha256") != payload.get("state_sha256"):
        raise ValueError("Neural bundle state is not bound to the final-refit audit")
    for key in (
        "selection_audit_sha256",
        "calibration_audit_sha256",
        "refit_audit_sha256",
    ):
        if not _is_sha256(metadata.get(key)):
            raise ValueError(f"Neural bundle lacks a valid {key}")
    model.to(device)
    model.eval()
    return model, offsets, dict(metadata)


@torch.inference_mode()
def predict_neural_case_bag(
    model: nn.Module,
    bag: NeuralCaseBag,
    class_offsets: Sequence[float],
) -> NeuralCasePrediction:
    """Return label-free logits/probabilities and the offset-adjusted subtype."""

    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("Neural case head has no parameters") from error
    inputs = neural_bag_inference_tensors(bag, device)
    logits = model(*inputs)
    if logits.shape != (1, 3) or not torch.isfinite(logits).all():
        raise FloatingPointError("Neural case head emitted invalid online logits")
    log_scores = torch.log_softmax(logits.double(), dim=1)
    offsets = torch.as_tensor(_validated_offsets(class_offsets), dtype=torch.float64, device=device)
    offset_log_scores = torch.log_softmax(log_scores[0] + offsets, dim=0)
    prediction = int(torch.argmax(offset_log_scores).item())
    return NeuralCasePrediction(
        logits[0],
        torch.exp(log_scores[0]),
        torch.exp(offset_log_scores),
        prediction,
    )


def predict_neural_case_extraction(
    model: nn.Module,
    extraction: object,
    class_offsets: Sequence[float],
) -> NeuralCasePrediction:
    """Use the shared offline/online bag constructor on one CaseExtraction."""

    required = (
        "tile_vectors",
        "tile_evidence",
        "tile_vector_names",
        "mil_stage3_maps",
        "mil_prediction_maps",
        "mil_lesion_mass",
    )
    if any(not hasattr(extraction, name) for name in required):
        raise TypeError("Online neural inference requires one complete CaseExtraction")
    bag = build_neural_case_bag(
        tile_vectors=extraction.tile_vectors,
        tile_evidence=extraction.tile_evidence,
        tile_vector_names=extraction.tile_vector_names,
        mil_stage3_maps=extraction.mil_stage3_maps,
        mil_prediction_maps=extraction.mil_prediction_maps,
        mil_lesion_mass=extraction.mil_lesion_mass,
    )
    return predict_neural_case_bag(model, bag, class_offsets)


__all__ = [
    "NeuralCasePrediction",
    "load_neural_case_head_bundle",
    "predict_neural_case_bag",
    "predict_neural_case_extraction",
    "save_neural_case_head_bundle",
]
