#!/usr/bin/env python3
"""Draw the implemented multi-task ResEnc-M architecture as PDF and PNG."""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.switch_backend("Agg")


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 8.5,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.04",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.5,
        )
    )
    axis.text(x, y, label, ha="center", va="center", fontsize=fontsize)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            **kwargs,
        )
    )


def draw(output_stem: Path, *, dpi: int) -> list[Path]:
    figure, axis = plt.subplots(figsize=(13.5, 5.8))
    axis.set(xlim=(0, 15), ylim=(0, 7))
    axis.axis("off")

    encoder = [
        ("E0\n32", 1.9, 4.9),
        ("E1\n64", 3.25, 4.5),
        ("E2\n128", 4.6, 4.1),
        ("E3\n256", 5.95, 3.7),
        ("E4\n320", 7.3, 3.3),
        ("Bottleneck\n320", 8.65, 2.9),
    ]
    axis.text(0.67, 5.28, "CT patch\n1×64×128×192", ha="center", va="center", fontsize=9.0)
    _arrow(axis, (1.23, 5.02), (1.37, 4.96))
    for label, x, y in encoder:
        _box(axis, x, y, 1.02, 0.76, label, face="#DDEBF7", edge=BLUE)
    for (_, x1, y1), (_, x2, y2) in pairwise(encoder):
        _arrow(axis, (x1 + 0.52, y1), (x2 - 0.52, y2))

    decoder = [
        ("D4", 9.9, 4.0),
        ("D3", 10.85, 4.5),
        ("D2", 11.8, 5.0),
        ("D1", 12.75, 5.5),
        ("D0", 13.7, 6.0),
    ]
    for label, x, y in decoder:
        _box(axis, x, y, 0.68, 0.56, label, face="#FCE4D6", edge=ORANGE)
    _arrow(axis, (9.15, 3.15), (9.56, 3.8))
    for (_, x1, y1), (_, x2, y2) in pairwise(decoder):
        _arrow(axis, (x1 + 0.35, y1), (x2 - 0.35, y2))
    _arrow(axis, (14.06, 6.0), (14.48, 6.0))
    axis.text(14.62, 6.0, "3-class voxel logits", ha="left", va="center", fontsize=8.5)

    for (_, encoder_x, encoder_y), (_, decoder_x, decoder_y) in zip(
        encoder[:5], reversed(decoder), strict=True
    ):
        _arrow(
            axis,
            (encoder_x, encoder_y + 0.4),
            (decoder_x, decoder_y + 0.3),
            color="#6B7280",
            alpha=0.65,
            connectionstyle="arc3,rad=-0.10",
        )

    _box(
        axis,
        9.05,
        1.05,
        2.75,
        0.85,
        "Global average +\n8-head learned-query attention",
        face="#E2F0D9",
        edge=GREEN,
    )
    _box(
        axis,
        12.1,
        1.05,
        2.15,
        0.85,
        "LayerNorm → 128 → GELU\nDropout 0.30 → 3",
        face="#E4DFEC",
        edge=PURPLE,
    )
    _arrow(axis, (8.65, 2.48), (8.9, 1.5), color=GREEN)
    _arrow(axis, (10.44, 1.05), (11.0, 1.05))
    _arrow(axis, (13.2, 1.05), (14.45, 1.05))
    axis.text(14.62, 1.05, "3 subtype logits", ha="left", va="center", fontsize=8.5)

    axis.text(
        7.5,
        6.72,
        "Shared nnU-Net v2 3D ResEnc-M encoder with segmentation and classification outputs",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )
    axis.text(1.35, 3.85, "Residual encoder", color=BLUE, fontsize=9.5, weight="bold")
    axis.text(10.45, 3.28, "Segmentation decoder + deep supervision", color=ORANGE, fontsize=9.5, weight="bold")
    axis.text(7.9, 0.25, "Classification branch (same bottleneck features)", color=GREEN, fontsize=9.5, weight="bold")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    for output in outputs:
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("report/figures/architecture"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    for output in draw(args.output, dpi=args.dpi):
        print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
