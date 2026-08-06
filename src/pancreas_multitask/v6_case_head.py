"""Locked V6 coordinate-aware whole-volume classification heads.

This module is deliberately parallel to the immutable V5 implementation.  It
contains only the two candidates frozen in
``configs/v6_spatial_case_head_attempt_1.json`` and consumes, but never
reconstructs, the extractor-owned whole-volume geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn

V6_HEAD_LOCK_SHA256: Final = (
    "260dd55f9d3bd49e4d60cf472eaa6f7716e0119ba68bb4d56ba664bb2abf621a"
)
V6_CLARIFICATIONS_SHA256: Final = (
    "33dbd3fae0ec83340184c03b68441e0d46fe1c234119569e64807929eb9ae848"
)

V6_SPATIAL_CANDIDATE: Final = "v6_spatial_residual"
V6_SPATIAL_MORPH_CANDIDATE: Final = "v6_spatial_morph_residual"
V6_CASE_CANDIDATES: Final = (V6_SPATIAL_CANDIDATE, V6_SPATIAL_MORPH_CANDIDATE)

SPATIAL_CHANNELS: Final = 387
SPATIAL_GRID: Final = (8, 8, 12)
SPATIAL_TOKEN_COUNT: Final = math.prod(SPATIAL_GRID)
STAGE5_CHANNELS: Final = 320
MORPHOLOGY_FEATURES: Final = 46
CLASS_COUNT: Final = 3
LESION_PROBABILITY_CHANNEL: Final = 384
WHOLE_PROBABILITY_CHANNEL: Final = 385
CT_CHANNEL: Final = 386
ATTENTION_PRIOR_FLOOR: Final = 1e-4
MORPHOLOGY_IQR_FLOOR: Final = 1e-6


@dataclass(frozen=True, slots=True)
class V6CaseHeadOutput:
    """Training details in addition to the anchored three-class logits."""

    logits: Tensor
    bounded_correction: Tensor
    attention_weights: Tensor | None = None


def _require_float32(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use the locked float32 training dtype")


def validate_v6_case_inputs(
    spatial: Tensor,
    coordinates: Tensor,
    case_lesion_mass: Tensor,
    stage5_global_mean: Tensor,
    stage5_lesion_weighted_mean: Tensor,
    rescue_mean_logits: Tensor,
    rescue_mean_probabilities: Tensor,
    morphology: Tensor | None = None,
    *,
    require_morphology: bool = False,
) -> None:
    """Fail closed on the locked one-volume, 387-channel tensor contract.

    Coordinates and case lesion mass are extractor-owned.  The head validates
    and consumes those values without deriving replacement geometry or mass.
    """

    named = {
        "spatial": spatial,
        "coordinates": coordinates,
        "case_lesion_mass": case_lesion_mass,
        "stage5_global_mean": stage5_global_mean,
        "stage5_lesion_weighted_mean": stage5_lesion_weighted_mean,
        "rescue_mean_logits": rescue_mean_logits,
        "rescue_mean_probabilities": rescue_mean_probabilities,
    }
    if morphology is not None:
        named["morphology"] = morphology
    for name, value in named.items():
        _require_float32(name, value)

    batch_size = int(spatial.shape[0]) if spatial.ndim > 0 else -1
    expected_shapes = {
        "spatial": (batch_size, SPATIAL_CHANNELS, *SPATIAL_GRID),
        "coordinates": (batch_size, 3, *SPATIAL_GRID),
        "case_lesion_mass": (batch_size,),
        "stage5_global_mean": (batch_size, STAGE5_CHANNELS),
        "stage5_lesion_weighted_mean": (batch_size, STAGE5_CHANNELS),
        "rescue_mean_logits": (batch_size, CLASS_COUNT),
        "rescue_mean_probabilities": (batch_size, CLASS_COUNT),
    }
    if batch_size < 1:
        raise ValueError("A V6 head batch must contain at least one case")
    for name, expected in expected_shapes.items():
        if tuple(named[name].shape) != expected:
            raise ValueError(f"{name} has shape {tuple(named[name].shape)}, expected {expected}")
    if require_morphology:
        if morphology is None:
            raise ValueError("The locked morphology candidate requires 46 morphology features")
        if tuple(morphology.shape) != (batch_size, MORPHOLOGY_FEATURES):
            raise ValueError("morphology must have shape (batch, 46)")

    devices = {value.device for value in named.values()}
    if len(devices) != 1:
        raise ValueError("All V6 case-head inputs must be on the same device")
    with torch.no_grad():
        finite_flags = torch.stack(tuple(torch.isfinite(value).all() for value in named.values()))
        if not bool(finite_flags.all().item()):
            raise ValueError("V6 case-head inputs must all be finite")
        lesion = spatial[:, LESION_PROBABILITY_CHANNEL]
        whole = spatial[:, WHOLE_PROBABILITY_CHANNEL]
        lesion_valid = ((lesion >= 0) & (lesion <= 1)).all()
        whole_valid = ((whole >= 0) & (whole <= 1) & (lesion <= whole)).all()
        coordinates_valid = ((coordinates >= 0) & (coordinates <= 1)).all()
        mass_valid = ((case_lesion_mass >= 0) & (case_lesion_mass <= 1)).all()
        probabilities_valid = (
            (rescue_mean_probabilities >= 0).all()
            & (rescue_mean_probabilities <= 1).all()
            & torch.isclose(
                rescue_mean_probabilities.sum(dim=1),
                torch.ones(
                    batch_size,
                    device=rescue_mean_probabilities.device,
                    dtype=torch.float32,
                ),
                atol=1e-5,
                rtol=1e-5,
            ).all()
        )
        semantic_flags = torch.stack(
            (
                lesion_valid,
                whole_valid,
                coordinates_valid,
                mass_valid,
                probabilities_valid,
            )
        )
        if bool(semantic_flags.all().item()):
            return
        if not bool(lesion_valid.item()):
            raise ValueError("Local lesion probabilities must be in [0, 1]")
        if not bool(whole_valid.item()):
            raise ValueError("Local whole probabilities must be in [0, 1] and cover lesion")
        if not bool(coordinates_valid.item()):
            raise ValueError("Extractor-owned normalized cell coordinates must be in [0, 1]")
        if not bool(mass_valid.item()):
            raise ValueError("Extractor-owned case lesion mass must be in [0, 1]")
        if not bool(probabilities_valid.item()):
            raise ValueError("Rescue mean probabilities must be normalized three-class values")


def v6_attention_log_priors(spatial: Tensor, case_lesion_mass: Tensor) -> Tensor:
    """Return the locked lesion/context additive attention biases as ``(B,2,768)``."""

    if spatial.ndim != 5 or tuple(spatial.shape[1:]) != (SPATIAL_CHANNELS, *SPATIAL_GRID):
        raise ValueError("spatial violates the locked 387-channel 8x8x12 grid")
    if case_lesion_mass.shape != (spatial.shape[0],):
        raise ValueError("case_lesion_mass must have one scalar per case")
    local_lesion = spatial[:, LESION_PROBABILITY_CHANNEL].flatten(1)
    local_whole = spatial[:, WHOLE_PROBABILITY_CHANNEL].flatten(1)
    case_mass = case_lesion_mass.clamp_min(ATTENTION_PRIOR_FLOOR).unsqueeze(1)
    lesion_prior = 0.75 * torch.log(local_lesion.clamp_min(ATTENTION_PRIOR_FLOOR))
    lesion_prior = lesion_prior + 0.25 * torch.log(case_mass)
    context_prior = 0.25 * torch.log(local_whole.clamp_min(ATTENTION_PRIOR_FLOOR))
    return torch.stack((lesion_prior, context_prior), dim=1)


class V6CaseHead(nn.Module):
    """Coordinate-aware residual head for exactly one adaptive whole volume."""

    def __init__(
        self,
        candidate_id: str,
        *,
        morphology_median: Tensor | None = None,
        morphology_iqr: Tensor | None = None,
    ) -> None:
        super().__init__()
        if candidate_id not in V6_CASE_CANDIDATES:
            raise ValueError(f"Unknown locked V6 candidate: {candidate_id}")
        self.candidate_id = candidate_id
        self.uses_morphology = candidate_id == V6_SPATIAL_MORPH_CANDIDATE

        self.spatial_frontend = nn.Sequential(
            nn.Conv3d(SPATIAL_CHANNELS, 96, kernel_size=1, bias=False),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Conv3d(96, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout3d(0.10),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(9, 32),
            nn.GELU(),
            nn.Linear(32, 64),
        )
        # Zero is the deterministic neutral initialization for the otherwise
        # unconstrained learned query parameters.  The distinct locked priors
        # break lesion/context symmetry on the first forward pass.
        self.queries = nn.Parameter(torch.zeros(1, 2, 64))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.query_layer_norm = nn.LayerNorm(64)

        self.shared_stage5_projection = nn.Sequential(
            nn.LayerNorm(STAGE5_CHANNELS),
            nn.Linear(STAGE5_CHANNELS, 32),
            nn.GELU(),
        )
        self.summary_projection = nn.Sequential(
            nn.Linear(70, 32),
            nn.GELU(),
        )

        if self.uses_morphology:
            if morphology_median is None or morphology_iqr is None:
                raise ValueError(
                    "The morphology candidate requires fold-fitted median and IQR buffers"
                )
            median = torch.as_tensor(morphology_median, dtype=torch.float32).detach().clone()
            iqr = torch.as_tensor(morphology_iqr, dtype=torch.float32).detach().clone()
            if median.shape != (MORPHOLOGY_FEATURES,) or iqr.shape != (
                MORPHOLOGY_FEATURES,
            ):
                raise ValueError("Morphology median and IQR must each contain 46 values")
            if not torch.isfinite(median).all() or not torch.isfinite(iqr).all():
                raise ValueError("Morphology scaler buffers must be finite")
            if torch.any(iqr < 0):
                raise ValueError("Morphology IQR values cannot be negative")
            self.register_buffer("morphology_median", median)
            self.register_buffer("morphology_iqr", iqr.clamp_min(MORPHOLOGY_IQR_FLOOR))
            self.morphology_projection: nn.Module | None = nn.Sequential(
                nn.Linear(MORPHOLOGY_FEATURES, 32),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(32, 16),
                nn.GELU(),
            )
            classifier_input = 176
        else:
            if morphology_median is not None or morphology_iqr is not None:
                raise ValueError("The no-morphology candidate cannot own morphology scaler state")
            self.morphology_projection = None
            classifier_input = 160

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, CLASS_COUNT),
        )

    @staticmethod
    def _position_features(coordinates: Tensor) -> Tensor:
        grid = coordinates.permute(0, 2, 3, 4, 1)
        return torch.cat(
            (
                2.0 * grid - 1.0,
                torch.sin(math.pi * grid),
                torch.cos(math.pi * grid),
            ),
            dim=-1,
        )

    def _encode_spatial(
        self,
        spatial: Tensor,
        coordinates: Tensor,
        case_lesion_mass: Tensor,
        *,
        need_attention_weights: bool,
    ) -> tuple[Tensor, Tensor | None]:
        batch_size = spatial.shape[0]
        encoded = self.spatial_frontend(spatial)
        tokens = encoded.flatten(2).transpose(1, 2)
        position = self.position_projection(self._position_features(coordinates)).reshape(
            batch_size, SPATIAL_TOKEN_COUNT, 64
        )
        tokens = tokens + position

        priors = v6_attention_log_priors(spatial, case_lesion_mass)
        attention_mask = (
            priors[:, None]
            .expand(batch_size, self.cross_attention.num_heads, 2, SPATIAL_TOKEN_COUNT)
            .reshape(batch_size * self.cross_attention.num_heads, 2, SPATIAL_TOKEN_COUNT)
        )
        queries = self.queries.expand(batch_size, -1, -1)
        attended, weights = self.cross_attention(
            queries,
            tokens,
            tokens,
            attn_mask=attention_mask,
            need_weights=need_attention_weights,
            average_attn_weights=False,
        )
        return self.query_layer_norm(queries + attended), weights

    def _forward_details(
        self,
        spatial: Tensor,
        coordinates: Tensor,
        case_lesion_mass: Tensor,
        stage5_global_mean: Tensor,
        stage5_lesion_weighted_mean: Tensor,
        rescue_mean_logits: Tensor,
        rescue_mean_probabilities: Tensor,
        morphology: Tensor | None,
        *,
        need_attention_weights: bool,
    ) -> V6CaseHeadOutput:
        if not self.uses_morphology and morphology is not None:
            raise ValueError("The locked no-morphology candidate cannot consume morphology")
        validate_v6_case_inputs(
            spatial,
            coordinates,
            case_lesion_mass,
            stage5_global_mean,
            stage5_lesion_weighted_mean,
            rescue_mean_logits,
            rescue_mean_probabilities,
            morphology,
            require_morphology=self.uses_morphology,
        )
        queries, attention_weights = self._encode_spatial(
            spatial,
            coordinates,
            case_lesion_mass,
            need_attention_weights=need_attention_weights,
        )
        global_summary = self.shared_stage5_projection(stage5_global_mean)
        lesion_summary = self.shared_stage5_projection(stage5_lesion_weighted_mean)
        summary = self.summary_projection(
            torch.cat(
                (
                    global_summary,
                    lesion_summary,
                    rescue_mean_logits,
                    rescue_mean_probabilities,
                ),
                dim=1,
            )
        )
        classifier_parts = [queries.flatten(1), summary]
        if self.uses_morphology:
            if morphology is None or self.morphology_projection is None:
                raise RuntimeError("Validated morphology input unexpectedly disappeared")
            scaled = ((morphology - self.morphology_median) / self.morphology_iqr).clamp(
                -5.0, 5.0
            )
            classifier_parts.append(self.morphology_projection(scaled))
        raw_delta = self.classifier(torch.cat(classifier_parts, dim=1))
        bounded_correction = 3.0 * torch.tanh(raw_delta / 3.0)
        logits = rescue_mean_logits + bounded_correction
        return V6CaseHeadOutput(logits, bounded_correction, attention_weights)

    def forward(
        self,
        spatial: Tensor,
        coordinates: Tensor,
        case_lesion_mass: Tensor,
        stage5_global_mean: Tensor,
        stage5_lesion_weighted_mean: Tensor,
        rescue_mean_logits: Tensor,
        rescue_mean_probabilities: Tensor,
        morphology: Tensor | None = None,
    ) -> Tensor:
        """Return the locked anchored logits; dropout follows module train/eval mode."""

        return self._forward_details(
            spatial,
            coordinates,
            case_lesion_mass,
            stage5_global_mean,
            stage5_lesion_weighted_mean,
            rescue_mean_logits,
            rescue_mean_probabilities,
            morphology,
            need_attention_weights=False,
        ).logits

    def forward_with_details(
        self,
        spatial: Tensor,
        coordinates: Tensor,
        case_lesion_mass: Tensor,
        stage5_global_mean: Tensor,
        stage5_lesion_weighted_mean: Tensor,
        rescue_mean_logits: Tensor,
        rescue_mean_probabilities: Tensor,
        morphology: Tensor | None = None,
        *,
        need_attention_weights: bool = False,
    ) -> V6CaseHeadOutput:
        """Return logits and the correction needed by the locked training loss."""

        return self._forward_details(
            spatial,
            coordinates,
            case_lesion_mass,
            stage5_global_mean,
            stage5_lesion_weighted_mean,
            rescue_mean_logits,
            rescue_mean_probabilities,
            morphology,
            need_attention_weights=need_attention_weights,
        )


def build_v6_case_head(
    candidate_id: str,
    *,
    morphology_median: Tensor | None = None,
    morphology_iqr: Tensor | None = None,
) -> V6CaseHead:
    """Construct exactly one prospectively locked V6 candidate."""

    return V6CaseHead(
        candidate_id,
        morphology_median=morphology_median,
        morphology_iqr=morphology_iqr,
    )


def v6_case_head_parameter_count(candidate_id: str) -> int:
    """Recompute the audited trainable parameter count from a fresh model."""

    kwargs: dict[str, Tensor] = {}
    if candidate_id == V6_SPATIAL_MORPH_CANDIDATE:
        kwargs = {
            "morphology_median": torch.zeros(MORPHOLOGY_FEATURES),
            "morphology_iqr": torch.ones(MORPHOLOGY_FEATURES),
        }
    model = build_v6_case_head(candidate_id, **kwargs)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "ATTENTION_PRIOR_FLOOR",
    "CLASS_COUNT",
    "CT_CHANNEL",
    "LESION_PROBABILITY_CHANNEL",
    "MORPHOLOGY_FEATURES",
    "MORPHOLOGY_IQR_FLOOR",
    "SPATIAL_CHANNELS",
    "SPATIAL_GRID",
    "SPATIAL_TOKEN_COUNT",
    "STAGE5_CHANNELS",
    "V6_CASE_CANDIDATES",
    "V6_CLARIFICATIONS_SHA256",
    "V6_HEAD_LOCK_SHA256",
    "V6_SPATIAL_CANDIDATE",
    "V6_SPATIAL_MORPH_CANDIDATE",
    "WHOLE_PROBABILITY_CHANNEL",
    "V6CaseHead",
    "V6CaseHeadOutput",
    "build_v6_case_head",
    "v6_attention_log_priors",
    "v6_case_head_parameter_count",
    "validate_v6_case_inputs",
]
