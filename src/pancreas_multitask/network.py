"""Network components for joint 3D segmentation and case classification.

The wrapper deliberately preserves nnU-Net's normal inference contract:
``forward(x)`` returns only segmentation logits. Training code can request the
classification logits explicitly with ``forward(x, return_classification=True)``.
This matters because nnU-Net's stock sliding-window predictor performs tensor
arithmetic directly on the value returned by ``forward``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _compatible_attention_heads(channels: int, requested_heads: int) -> int:
    """Return the largest requested-or-smaller head count dividing channels."""

    if channels < 1:
        raise ValueError(f"channels must be positive, got {channels}")
    if requested_heads < 1:
        raise ValueError(
            f"requested_heads must be positive, got {requested_heads}"
        )
    upper = min(channels, requested_heads)
    return next(heads for heads in range(upper, 0, -1) if channels % heads == 0)


class HybridCrossAttentionPool3D(nn.Module):
    """Pool a 3D bottleneck with global mean and learned-query attention.

    Global average pooling gives the classifier a conservative summary of the
    whole patch. A single learned query can additionally focus on a small,
    discriminative subset of bottleneck tokens. Concatenating both makes the
    learned attention an addition rather than a fragile replacement.
    """

    def __init__(
        self,
        channels: int,
        *,
        requested_heads: int = 8,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        heads = _compatible_attention_heads(channels, requested_heads)
        self.channels = channels
        self.token_norm = nn.LayerNorm(channels)
        self.query = nn.Parameter(torch.empty(1, 1, channels))
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(channels)
        nn.init.trunc_normal_(self.query, std=0.02)

    def reset_parameters(self) -> None:
        """Reinitialize only the learned pooling path.

        This is intentionally explicit instead of walking all child modules:
        ``MultiheadAttention`` owns parameters both directly and through its
        output projection.  Resetting each owner makes a post-training
        classification-head rescue reproducible without touching the shared
        encoder or segmentation decoder.
        """

        self.token_norm.reset_parameters()
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attention.out_proj.reset_parameters()
        self.attention._reset_parameters()
        self.output_norm.reset_parameters()

    @property
    def output_channels(self) -> int:
        return self.channels * 2

    def forward(self, bottleneck: Tensor) -> Tensor:
        if bottleneck.ndim != 5:
            raise ValueError(
                "The classification bottleneck must be a 5D tensor "
                f"(batch, channels, depth, height, width), got {bottleneck.shape}"
            )
        if bottleneck.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} bottleneck channels, got "
                f"{bottleneck.shape[1]}"
            )

        global_average = bottleneck.mean(dim=(2, 3, 4))
        tokens = bottleneck.flatten(start_dim=2).transpose(1, 2)
        tokens = self.token_norm(tokens)
        query = self.query.expand(bottleneck.shape[0], -1, -1)
        attended, _ = self.attention(
            query,
            tokens,
            tokens,
            need_weights=False,
        )
        attended = self.output_norm((attended + query).squeeze(1))
        return torch.cat((global_average, attended), dim=1)


class GlobalAveragePool3D(nn.Module):
    """Plain global average pool, used by the shallow classification tap.

    Deliberately parameterless.  Richer pooling was measured against this on the
    252 training cases with nested cross-validation and every variant lost:
    GAP+STD 0.573, lesion-probability-weighted 0.476, +morphology 0.551, and an
    8x8x12 -> 2x2x3 spatial grid 0.606, versus 0.608 for plain GAP.  At 252 cases
    the extra parameters buy variance, not signal.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        self.output_channels = int(channels)

    def reset_parameters(self) -> None:
        """No state to reset; present so the rescue helpers can call it uniformly."""

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 5:
            raise ValueError(
                f"Expected a (batch, channels, D, H, W) feature map, got {tuple(features.shape)}"
            )
        if features.shape[1] != self.output_channels:
            raise ValueError(
                f"Expected {self.output_channels} channels, got {features.shape[1]}"
            )
        return features.mean(dim=(2, 3, 4))


class MultiTaskResEncUNet(nn.Module):
    """Wrap an nnU-Net ResidualEncoderUNet with a classification branch.

    ``classification_tap`` selects which encoder stage feeds the classification
    head.  It defaults to ``-1`` (the bottleneck), which reproduces the original
    architecture exactly so existing checkpoints keep loading.

    Tapping a shallower stage is not a stylistic choice.  Nested-CV linear probes
    over the 252 training cases put stage-2 at macro-F1 0.594, stage-3 at 0.525
    and the stage-5 bottleneck at 0.399 -- the last is barely above the 0.354
    permuted-label control.  Subtype signal is a shallow, largely parenchymal
    property that the segmentation objective progressively discards with depth,
    so a head reading only the bottleneck is reading almost pure noise.
    """

    def __init__(
        self,
        segmentation_network: nn.Module,
        *,
        num_classification_classes: int = 3,
        classification_hidden_channels: int = 128,
        classification_dropout: float = 0.3,
        requested_attention_heads: int = 8,
        classification_tap: int = -1,
    ) -> None:
        super().__init__()
        if not hasattr(segmentation_network, "encoder"):
            raise TypeError("segmentation_network must expose an encoder")
        if not hasattr(segmentation_network, "decoder"):
            raise TypeError("segmentation_network must expose a decoder")

        output_channels: Sequence[int] | None = getattr(
            segmentation_network.encoder, "output_channels", None
        )
        if not output_channels:
            raise TypeError(
                "segmentation_network.encoder must expose non-empty output_channels"
            )
        stage_count = len(output_channels)
        tap = int(classification_tap)
        if tap < 0:
            tap += stage_count
        if not 0 <= tap < stage_count:
            raise ValueError(
                f"classification_tap must index one of {stage_count} encoder stages, "
                f"got {classification_tap}"
            )
        self.classification_tap = tap
        self._taps_bottleneck = tap == stage_count - 1
        tap_channels = int(output_channels[tap])

        # Expose these at the wrapper's top level. nnUNetTrainer toggles deep
        # supervision through ``network.decoder.deep_supervision``.
        self.encoder = segmentation_network.encoder
        self.decoder = segmentation_network.decoder

        if self._taps_bottleneck:
            self.classification_pool = HybridCrossAttentionPool3D(
                tap_channels,
                requested_heads=requested_attention_heads,
            )
            self.classification_head = nn.Sequential(
                nn.LayerNorm(self.classification_pool.output_channels),
                nn.Linear(
                    self.classification_pool.output_channels,
                    classification_hidden_channels,
                ),
                nn.GELU(),
                nn.Dropout(classification_dropout),
                nn.Linear(
                    classification_hidden_channels,
                    num_classification_classes,
                ),
            )
        else:
            # A shallow tap gets a deliberately tiny head. Every >=100k-parameter
            # head tried on these 252 cases memorized them (resubstitution 0.96-0.98
            # against 0.49-0.51 out-of-fold); a 128-dim regularized linear map on the
            # same features scores 0.594 out-of-fold with a far smaller gap.
            self.classification_pool = GlobalAveragePool3D(tap_channels)
            self.classification_head = nn.Sequential(
                nn.LayerNorm(tap_channels),
                nn.Dropout(classification_dropout),
                nn.Linear(tap_channels, num_classification_classes),
            )
        self._initialize_classification_head()

    def _initialize_classification_head(self) -> None:
        # The supplied segmentation network has already been initialized by
        # get_network_from_plans. Restrict custom initialization to new layers.
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reset_classification_parameters(self) -> None:
        """Reset the pooling and classification head, but no backbone state."""

        self.classification_pool.reset_parameters()
        for module in self.classification_head.modules():
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
        # Keep the declared small truncated-normal initialization for both
        # Linear layers after their generic ``reset_parameters`` calls.
        self._initialize_classification_head()

    def _encode(self, x: Tensor) -> Sequence[Tensor]:
        skips = self.encoder(x)
        if not isinstance(skips, (list, tuple)) or len(skips) == 0:
            raise RuntimeError(
                "The residual encoder must return a non-empty sequence of skips"
            )
        return skips

    def classify_bottleneck(self, bottleneck: Tensor) -> Tensor:
        """Classify a previously computed encoder feature map from the configured tap."""

        return self.classification_head(self.classification_pool(bottleneck))

    def encode_bottleneck(self, x: Tensor) -> Tensor:
        """Encode an input and return only its lowest-resolution feature map."""

        return self._encode(x)[-1]

    def encode_to_stage(self, x: Tensor, stage: int | None = None) -> Tensor:
        """Run the encoder only as far as ``stage`` and return that feature map.

        Equivalent to ``self._encode(x)[stage]`` but stops early, so a shallow tap
        costs a fraction of a full encoder pass and retains activations only for
        the stages it actually ran.  Verified bit-identical to the full-encoder
        skip on the locked ResEnc-M plan.
        """

        index = self.classification_tap if stage is None else int(stage)
        stages = getattr(self.encoder, "stages", None)
        if stages is None:
            raise TypeError("The residual encoder must expose a `stages` sequence")
        if index < 0:
            index += len(stages)
        if not 0 <= index < len(stages):
            raise ValueError(f"stage {stage} is outside the encoder's {len(stages)} stages")

        features = x
        stem = getattr(self.encoder, "stem", None)
        if stem is not None:
            features = stem(features)
        for position, module in enumerate(stages):
            features = module(features)
            if position == index:
                return features
        raise RuntimeError("Encoder stage iteration ended without reaching the tap")

    def encode_to_stages(self, x: Tensor, stages: Sequence[int]) -> dict[int, Tensor]:
        """Return several encoder stage outputs from a **single** forward pass.

        Calling ``encode_to_stage`` once per stage recomputes every shallower stage:
        asking for stages 1 and 2 separately runs stem->0->1 twice. Since the deepest
        requested stage's pass already produces the shallower ones, capturing them in
        flight is exactly the same arithmetic for roughly half the work. Measured at
        0.167 s/case for two separate calls against ~0.09 s combined, which is 18% of
        total inference runtime on this hardware.
        """

        wanted = {int(s) for s in stages}
        encoder_stages = getattr(self.encoder, "stages", None)
        if encoder_stages is None:
            raise TypeError("The residual encoder must expose a `stages` sequence")
        resolved = {s + len(encoder_stages) if s < 0 else s for s in wanted}
        if not resolved or any(not 0 <= s < len(encoder_stages) for s in resolved):
            raise ValueError(f"stages {sorted(wanted)} are outside the encoder")

        deepest = max(resolved)
        features = x
        stem = getattr(self.encoder, "stem", None)
        if stem is not None:
            features = stem(features)
        collected: dict[int, Tensor] = {}
        for position, module in enumerate(encoder_stages):
            features = module(features)
            if position in resolved:
                collected[position] = features
            if position == deepest:
                break
        return collected

    def classify_volume(self, x: Tensor) -> Tensor:
        """Return subtype logits for a whole volume without running the decoder.

        This is the case-level classification entry point.  nnU-Net trains on
        patches, so a patch-level classification loss carrying a case-level label
        is only a proxy for the quantity being scored; subtype is a whole-organ
        property.  Because these ROIs are small (94 of 252 cases fit one
        64x128x192 patch), classifying the entire volume in one pass is affordable
        and removes that train/score mismatch.
        """

        return self.classify_bottleneck(self.encode_to_stage(x))

    def classification_parameters(self) -> list[nn.Parameter]:
        """Parameters of the classification branch only (pool + head)."""

        return [
            *self.classification_pool.parameters(),
            *self.classification_head.parameters(),
        ]

    def encoder_stage_parameters(self, through_stage: int) -> list[nn.Parameter]:
        """Parameters of the stem plus encoder stages ``0..through_stage``."""

        stages = getattr(self.encoder, "stages", None)
        if stages is None:
            raise TypeError("The residual encoder must expose a `stages` sequence")
        limit = int(through_stage)
        if limit < 0:
            limit += len(stages)
        if not 0 <= limit < len(stages):
            raise ValueError(f"through_stage {through_stage} is outside the encoder")
        collected: list[nn.Parameter] = []
        stem = getattr(self.encoder, "stem", None)
        if stem is not None:
            collected.extend(stem.parameters())
        for position, module in enumerate(stages):
            if position > limit:
                break
            collected.extend(module.parameters())
        return collected

    def forward_classification(self, x: Tensor) -> Tensor:
        """Return subtype logits without executing the segmentation decoder.

        The method is used only by the optional frozen-backbone rescue.  Joint
        training and inference retain the normal :meth:`forward` contract.
        """

        return self.classify_bottleneck(self.encode_bottleneck(x))

    def forward(
        self,
        x: Tensor,
        *,
        return_classification: bool = False,
    ) -> Tensor | list[Tensor] | tuple[Tensor | list[Tensor], Tensor]:
        skips = self._encode(x)
        segmentation = self.decoder(skips)
        if not return_classification:
            return segmentation

        classification = self.classify_bottleneck(skips[self.classification_tap])
        return segmentation, classification

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> int:
        """Delegate nnU-Net's approximate activation-size accounting."""

        encoder_size = self.encoder.compute_conv_feature_map_size(input_size)
        decoder_size = self.decoder.compute_conv_feature_map_size(input_size)
        return int(encoder_size + decoder_size)
