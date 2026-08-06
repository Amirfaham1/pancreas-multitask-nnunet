from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from pancreas_multitask.v6_case_head import (
    LESION_PROBABILITY_CHANNEL,
    MORPHOLOGY_FEATURES,
    SPATIAL_CHANNELS,
    SPATIAL_GRID,
    SPATIAL_TOKEN_COUNT,
    V6_CLARIFICATIONS_SHA256,
    V6_HEAD_LOCK_SHA256,
    V6_SPATIAL_CANDIDATE,
    V6_SPATIAL_MORPH_CANDIDATE,
    WHOLE_PROBABILITY_CHANNEL,
    build_v6_case_head,
    v6_attention_log_priors,
    v6_case_head_parameter_count,
    validate_v6_case_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(2606)
    spatial = torch.randn(
        batch_size,
        SPATIAL_CHANNELS,
        *SPATIAL_GRID,
        generator=generator,
    )
    lesion = torch.linspace(0.01, 0.75, SPATIAL_TOKEN_COUNT).reshape(1, *SPATIAL_GRID)
    lesion = lesion.expand(batch_size, -1, -1, -1).clone()
    whole = (lesion + 0.20).clamp_max(1.0)
    spatial[:, LESION_PROBABILITY_CHANNEL] = lesion
    spatial[:, WHOLE_PROBABILITY_CHANNEL] = whole

    axes = [
        (torch.arange(size, dtype=torch.float32) + 0.5) / size for size in SPATIAL_GRID
    ]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
    coordinates = coordinates.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).clone()
    case_mass = lesion.mean(dim=(1, 2, 3))
    stage5_global = torch.randn(batch_size, 320, generator=generator)
    stage5_lesion = torch.randn(batch_size, 320, generator=generator)
    rescue_logits = torch.randn(batch_size, 3, generator=generator)
    rescue_probabilities = torch.softmax(
        torch.randn(batch_size, 3, generator=generator), dim=1
    )
    morphology = torch.randn(batch_size, MORPHOLOGY_FEATURES, generator=generator)
    return (
        spatial,
        coordinates,
        case_mass,
        stage5_global,
        stage5_lesion,
        rescue_logits,
        rescue_probabilities,
        morphology,
    )


def test_v6_head_is_bound_to_both_prospective_locks() -> None:
    assert _sha256(ROOT / "configs" / "v6_spatial_case_head_attempt_1.json") == (
        V6_HEAD_LOCK_SHA256
    )
    assert _sha256(
        ROOT / "configs" / "v6_spatial_case_head_attempt_1_clarifications.json"
    ) == V6_CLARIFICATIONS_SHA256


def test_locked_v6_parameter_counts_and_all_parameter_gradients() -> None:
    assert v6_case_head_parameter_count(V6_SPATIAL_CANDIDATE) == 246_691
    assert v6_case_head_parameter_count(V6_SPATIAL_MORPH_CANDIDATE) == 249_779
    inputs = _inputs()
    for candidate_id in (V6_SPATIAL_CANDIDATE, V6_SPATIAL_MORPH_CANDIDATE):
        torch.manual_seed(91)
        kwargs = {}
        arguments = inputs[:7]
        if candidate_id == V6_SPATIAL_MORPH_CANDIDATE:
            kwargs = {
                "morphology_median": torch.zeros(MORPHOLOGY_FEATURES),
                "morphology_iqr": torch.ones(MORPHOLOGY_FEATURES),
            }
            arguments = inputs
        model = build_v6_case_head(candidate_id, **kwargs)
        output = model.forward_with_details(*arguments)
        (output.logits.square().mean() + output.bounded_correction.square().mean()).backward()

        assert output.logits.shape == (2, 3)
        assert torch.isfinite(output.logits).all()
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_anchored_logits_are_exactly_rescue_plus_bounded_correction() -> None:
    inputs = _inputs()
    model = build_v6_case_head(V6_SPATIAL_CANDIDATE).eval()
    output = model.forward_with_details(*inputs[:7])

    assert torch.equal(output.logits, inputs[5] + output.bounded_correction)
    assert torch.all(output.bounded_correction <= 3.0)
    assert torch.all(output.bounded_correction >= -3.0)


def test_attention_priors_use_local_maps_and_case_mass_exactly() -> None:
    spatial, _, case_mass, *_ = _inputs(1)
    priors = v6_attention_log_priors(spatial, case_mass)
    lesion = spatial[:, LESION_PROBABILITY_CHANNEL].flatten(1)
    whole = spatial[:, WHOLE_PROBABILITY_CHANNEL].flatten(1)
    expected_lesion = 0.75 * torch.log(lesion.clamp_min(1e-4))
    expected_lesion += 0.25 * torch.log(case_mass.clamp_min(1e-4)).unsqueeze(1)
    expected_context = 0.25 * torch.log(whole.clamp_min(1e-4))

    assert priors.shape == (1, 2, SPATIAL_TOKEN_COUNT)
    assert torch.equal(priors[:, 0], expected_lesion)
    assert torch.equal(priors[:, 1], expected_context)
    assert priors[0, 0].max() - priors[0, 0].min() > 0
    assert priors[0, 1].max() - priors[0, 1].min() > 0


def test_cross_attention_has_four_heads_and_normalized_two_query_weights() -> None:
    inputs = _inputs(1)
    model = build_v6_case_head(V6_SPATIAL_CANDIDATE).eval()
    output = model.forward_with_details(*inputs[:7], need_attention_weights=True)

    assert output.attention_weights is not None
    assert output.attention_weights.shape == (1, 4, 2, SPATIAL_TOKEN_COUNT)
    assert torch.allclose(
        output.attention_weights.sum(dim=-1),
        torch.ones(1, 4, 2),
        atol=1e-6,
        rtol=1e-6,
    )


def test_position_coordinates_are_consumed_and_not_rederived() -> None:
    inputs = list(_inputs(1))
    torch.manual_seed(73)
    model = build_v6_case_head(V6_SPATIAL_CANDIDATE).eval()
    reference = model(*inputs[:7])
    changed_coordinates = 1.0 - inputs[1]
    changed = model(inputs[0], changed_coordinates, *inputs[2:7])

    assert not torch.equal(reference, changed)


def test_morphology_scaler_floor_and_clip_are_inside_morph_candidate() -> None:
    inputs = list(_inputs(1))
    inputs[7] = torch.full((1, MORPHOLOGY_FEATURES), 100.0)
    model = build_v6_case_head(
        V6_SPATIAL_MORPH_CANDIDATE,
        morphology_median=torch.zeros(MORPHOLOGY_FEATURES),
        morphology_iqr=torch.zeros(MORPHOLOGY_FEATURES),
    ).eval()
    captured: list[torch.Tensor] = []
    first_linear = model.morphology_projection[0]
    handle = first_linear.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )
    try:
        model(*inputs)
    finally:
        handle.remove()

    assert torch.equal(model.morphology_iqr, torch.full((MORPHOLOGY_FEATURES,), 1e-6))
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.full((1, MORPHOLOGY_FEATURES), 5.0))


def test_no_morphology_candidate_rejects_morphology_and_scaler_state() -> None:
    inputs = _inputs(1)
    with pytest.raises(ValueError, match="cannot own morphology scaler"):
        build_v6_case_head(
            V6_SPATIAL_CANDIDATE,
            morphology_median=torch.zeros(MORPHOLOGY_FEATURES),
        )
    model = build_v6_case_head(V6_SPATIAL_CANDIDATE)
    with pytest.raises(ValueError, match="cannot consume morphology"):
        model(*inputs)


def test_input_contract_rejects_wrong_dtype_shape_and_probability_semantics() -> None:
    inputs = list(_inputs(1))
    validate_v6_case_inputs(*inputs, require_morphology=True)

    changed = inputs.copy()
    changed[0] = changed[0].half()
    with pytest.raises(TypeError, match="float32"):
        validate_v6_case_inputs(*changed, require_morphology=True)

    changed = inputs.copy()
    changed[1] = changed[1][:, :, :-1]
    with pytest.raises(ValueError, match="expected"):
        validate_v6_case_inputs(*changed, require_morphology=True)

    changed = inputs.copy()
    changed[0] = changed[0].clone()
    changed[0][:, LESION_PROBABILITY_CHANNEL] = 0.9
    changed[0][:, WHOLE_PROBABILITY_CHANNEL] = 0.2
    with pytest.raises(ValueError, match="cover lesion"):
        validate_v6_case_inputs(*changed, require_morphology=True)


def test_zero_initialized_queries_are_explicit_and_priors_break_symmetry() -> None:
    inputs = _inputs(1)
    model = build_v6_case_head(V6_SPATIAL_CANDIDATE).eval()
    assert torch.count_nonzero(model.queries) == 0
    output = model.forward_with_details(*inputs[:7], need_attention_weights=True)
    assert output.attention_weights is not None
    assert not torch.equal(
        output.attention_weights[:, :, 0], output.attention_weights[:, :, 1]
    )
