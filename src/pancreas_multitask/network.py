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


class MultiTaskResEncUNet(nn.Module):
    """Wrap an nnU-Net ResidualEncoderUNet with a classification branch."""

    def __init__(
        self,
        segmentation_network: nn.Module,
        *,
        num_classification_classes: int = 3,
        classification_hidden_channels: int = 128,
        classification_dropout: float = 0.3,
        requested_attention_heads: int = 8,
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
        bottleneck_channels = int(output_channels[-1])

        # Expose these at the wrapper's top level. nnUNetTrainer toggles deep
        # supervision through ``network.decoder.deep_supervision``.
        self.encoder = segmentation_network.encoder
        self.decoder = segmentation_network.decoder
        self.classification_pool = HybridCrossAttentionPool3D(
            bottleneck_channels,
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
        """Classify a previously computed encoder bottleneck."""

        return self.classification_head(self.classification_pool(bottleneck))

    def encode_bottleneck(self, x: Tensor) -> Tensor:
        """Encode an input and return only its lowest-resolution feature map."""

        return self._encode(x)[-1]

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

        classification = self.classify_bottleneck(skips[-1])
        return segmentation, classification

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> int:
        """Delegate nnU-Net's approximate activation-size accounting."""

        encoder_size = self.encoder.compute_conv_feature_map_size(input_size)
        decoder_size = self.decoder.compute_conv_feature_map_size(input_size)
        return int(encoder_size + decoder_size)
