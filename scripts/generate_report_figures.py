#!/usr/bin/env python3
"""Generate report figures only from saved training/evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.lines import Line2D

plt.switch_backend("Agg")


BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
RED = "#D55E00"
GREY = "#5B6573"


class FigureInputError(ValueError):
    """Raised when a requested figure cannot be grounded in its inputs."""


def _finite_series(logging: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in logging:
        raise FigureInputError(f"Checkpoint logging does not contain {key!r}")
    try:
        values = np.asarray(logging[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FigureInputError(f"Logging field {key!r} is not numeric") from exc
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise FigureInputError(f"Logging field {key!r} must be a finite non-empty vector")
    return values


def _aligned(*series: np.ndarray) -> tuple[np.ndarray, ...]:
    length = min(item.size for item in series)
    if length < 1:
        raise FigureInputError("No aligned training epochs are available")
    return tuple(item[:length] for item in series)


def _save(figure: plt.Figure, output_stem: Path, *, dpi: int) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    for path in paths:
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return paths


def load_checkpoint_logging(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logging = checkpoint.get("logging") if isinstance(checkpoint, dict) else None
    if not isinstance(logging, dict):
        raise FigureInputError(f"Checkpoint has no nnU-Net logging dictionary: {checkpoint_path}")
    return logging


def plot_training_curves(
    logging: Mapping[str, Any], output_directory: Path, *, dpi: int
) -> list[Path]:
    train_seg, val_seg, train_cls, val_cls = _aligned(
        _finite_series(logging, "train_seg_losses"),
        _finite_series(logging, "val_seg_losses"),
        _finite_series(logging, "train_cls_losses"),
        _finite_series(logging, "val_cls_losses"),
    )
    epochs = np.arange(train_seg.size)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for axis, train, validation, title in (
        (axes[0], train_seg, val_seg, "Segmentation objective"),
        (axes[1], train_cls, val_cls, "Classification objective"),
    ):
        axis.plot(epochs, train, color=BLUE, linewidth=1.7, label="training")
        axis.plot(epochs, validation, color=ORANGE, linewidth=1.7, label="validation patches")
        axis.set(title=title, xlabel="Epoch", ylabel="Loss")
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.legend(frameon=False)
    figure.suptitle("Task-specific optimization traces", fontsize=13, weight="bold")
    outputs = _save(figure, output_directory / "loss_curves", dpi=dpi)

    mean_dice = _finite_series(logging, "mean_fg_dice")
    whole_dice = _finite_series(logging, "val_whole_pancreas_dice")
    macro_f1 = _finite_series(logging, "val_cls_macro_f1")
    accuracy = _finite_series(logging, "val_cls_accuracy")
    per_label = np.asarray(logging.get("dice_per_class_or_region"), dtype=np.float64)
    if per_label.ndim != 2 or per_label.shape[1] < 2 or not np.isfinite(per_label).all():
        raise FigureInputError("dice_per_class_or_region must contain two finite label series")
    mean_dice, whole_dice, macro_f1, accuracy, label_one, label_two = _aligned(
        mean_dice,
        whole_dice,
        macro_f1,
        accuracy,
        per_label[:, 0],
        per_label[:, 1],
    )
    epochs = np.arange(mean_dice.size)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].plot(epochs, whole_dice, color=GREEN, linewidth=1.8, label="whole pancreas")
    axes[0].plot(epochs, label_one, color=BLUE, linewidth=1.4, label="label 1")
    axes[0].plot(epochs, label_two, color=RED, linewidth=1.4, label="lesion (label 2)")
    axes[0].plot(epochs, mean_dice, color=GREY, linewidth=1.2, linestyle="--", label="mean labels 1/2")
    axes[0].set(title="Patch-level segmentation monitoring", xlabel="Epoch", ylabel="Dice")
    axes[1].plot(epochs, macro_f1, color=PURPLE, linewidth=1.8, label="macro-F1")
    axes[1].plot(epochs, accuracy, color=ORANGE, linewidth=1.4, label="accuracy")
    axes[1].set(title="Patch-aggregated subtype monitoring", xlabel="Epoch", ylabel="Score")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.legend(frameon=False, fontsize=8.5)
    figure.suptitle(
        "Online monitoring metrics (model selection uses fixed full-volume validation)",
        fontsize=12.3,
        weight="bold",
    )
    outputs.extend(_save(figure, output_directory / "validation_curves", dpi=dpi))
    return outputs


def read_case_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "case_id" not in rows[0]:
        raise FigureInputError(f"Case CSV is empty or lacks case_id: {path}")
    return rows


def plot_confusion_matrix(metrics: Mapping[str, Any], output_directory: Path, *, dpi: int) -> list[Path]:
    classification = metrics.get("classification")
    if not isinstance(classification, dict):
        raise FigureInputError("Metrics JSON has no classification section")
    matrix = np.asarray(classification.get("confusion_matrix"), dtype=np.int64)
    labels = classification.get("labels")
    if matrix.shape != (3, 3) or labels != [0, 1, 2] or np.any(matrix < 0):
        raise FigureInputError("Expected a non-negative 3×3 confusion matrix for labels 0,1,2")
    figure, axis = plt.subplots(figsize=(5.0, 4.3), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(3):
        for column in range(3):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#1F2933",
                fontsize=12,
                weight="bold",
            )
    axis.set(
        xticks=np.arange(3),
        yticks=np.arange(3),
        xticklabels=["0", "1", "2"],
        yticklabels=["0", "1", "2"],
        xlabel="Predicted subtype",
        ylabel="Reference subtype",
        title=f"Validation confusion matrix (n={int(matrix.sum())})",
    )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04, label="Cases")
    return _save(figure, output_directory / "confusion_matrix", dpi=dpi)


def plot_dice_distribution(
    rows: Sequence[Mapping[str, str]], output_directory: Path, *, dpi: int
) -> list[Path]:
    try:
        whole = np.asarray([float(row["whole_pancreas_dice"]) for row in rows])
        lesion = np.asarray([float(row["lesion_dice"]) for row in rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise FigureInputError("Case CSV lacks numeric whole/lesion Dice") from exc
    if not np.isfinite(whole).all() or not np.isfinite(lesion).all():
        raise FigureInputError("Dice values must be finite")
    figure, axis = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    violin = axis.violinplot(
        [whole, lesion],
        positions=[1, 2],
        showmeans=True,
        showmedians=True,
        widths=0.72,
    )
    for body, color in zip(violin["bodies"], (GREEN, RED), strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.55)
    rng = np.random.default_rng(12345)
    axis.scatter(1 + rng.uniform(-0.09, 0.09, whole.size), whole, s=16, color=GREEN, alpha=0.72)
    axis.scatter(2 + rng.uniform(-0.09, 0.09, lesion.size), lesion, s=16, color=RED, alpha=0.72)
    axis.hlines(0.90, 0.64, 1.36, colors=GREEN, linestyles=":", linewidth=1.5)
    axis.hlines(0.27, 1.64, 2.36, colors=RED, linestyles=":", linewidth=1.5)
    axis.text(1.37, 0.90, "target 0.90", color=GREEN, fontsize=7.5, va="center")
    axis.text(2.37, 0.27, "target 0.27", color=RED, fontsize=7.5, va="center")
    axis.set(
        xticks=[1, 2],
        xticklabels=["Whole pancreas\nlabel > 0", "Lesion\nlabel = 2"],
        ylabel="Case-level Dice",
        ylim=(-0.03, 1.03),
        title=f"Fixed validation Dice distributions (n={len(rows)})",
    )
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    return _save(figure, output_directory / "dice_distributions", dpi=dpi)


def select_qualitative_cases(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if len(rows) < 4:
        raise FigureInputError("At least four cases are required for qualitative selection")
    scored: list[tuple[float, float, str]] = []
    for row in rows:
        try:
            scored.append(
                (
                    float(row["whole_pancreas_dice"]),
                    float(row["lesion_dice"]),
                    str(row["case_id"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FigureInputError("Case CSV lacks qualitative-selection fields") from exc
    # Lesion is the harder target, so it is the primary ordering criterion;
    # whole-pancreas Dice and case ID make ties deterministic.
    ordered = sorted(scored, key=lambda item: (item[1], item[0], item[2]))
    selected = ordered[:2] + ordered[-2:]
    return [case_id for _, _, case_id in selected]


def _load_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(path)
    return np.asanyarray(image.dataobj), np.asarray(image.affine)


def _best_index(mask: np.ndarray, axis: int) -> int:
    other_axes = tuple(index for index in range(3) if index != axis)
    counts = mask.sum(axis=other_axes)
    return int(np.argmax(counts)) if np.any(counts) else mask.shape[axis] // 2


def _slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(volume, index, axis=axis))


def _window_ct(image: np.ndarray, *, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    low, high = center - width / 2, center + width / 2
    return np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def _qualitative_row_label(
    descriptor: str, case_id: str, metrics: Mapping[str, str]
) -> str:
    lines = [
        f"{descriptor}: {case_id}",
        (
            f"whole={float(metrics['whole_pancreas_dice']):.3f} | "
            f"lesion={float(metrics['lesion_dice']):.3f}"
        ),
    ]
    if "reference_subtype" in metrics and "predicted_subtype" in metrics:
        lines.append(
            f"subtype ref/pred={metrics['reference_subtype']}/{metrics['predicted_subtype']}"
        )
    if "lesion_reference_voxels" in metrics:
        lines.append(f"ref lesion={int(float(metrics['lesion_reference_voxels'])):,} vox")
    return "\n".join(lines)


def plot_qualitative_cases(
    rows: Sequence[Mapping[str, str]],
    images: Path,
    predictions: Path,
    references: Path,
    output_directory: Path,
    *,
    dpi: int,
) -> list[Path]:
    selected = select_qualitative_cases(rows)
    row_lookup = {str(row["case_id"]): row for row in rows}
    figure, axes = plt.subplots(len(selected), 3, figsize=(12.0, 10.5), constrained_layout=True)
    orientation_names = ("Sagittal", "Coronal", "Axial")
    for row_index, case_id in enumerate(selected):
        image, image_affine = _load_volume(images / f"{case_id}_0000.nii.gz")
        prediction, prediction_affine = _load_volume(predictions / f"{case_id}.nii.gz")
        reference, reference_affine = _load_volume(references / f"{case_id}.nii.gz")
        if image.shape != prediction.shape or image.shape != reference.shape:
            raise FigureInputError(f"Geometry shape mismatch for {case_id}")
        if not np.allclose(image_affine, prediction_affine, atol=1e-5, rtol=1e-5) or not np.allclose(
            image_affine, reference_affine, atol=1e-5, rtol=1e-5
        ):
            raise FigureInputError(f"Geometry affine mismatch for {case_id}")
        focus = (reference == 2) | (prediction == 2)
        if not np.any(focus):
            focus = (reference > 0) | (prediction > 0)
        for axis_index in range(3):
            index = _best_index(focus, axis_index)
            panel = axes[row_index, axis_index]
            panel.imshow(_slice(_window_ct(image), axis_index, index), cmap="gray", vmin=0, vmax=1)
            reference_pancreas = _slice(reference == 1, axis_index, index)
            reference_lesion = _slice(reference == 2, axis_index, index)
            prediction_pancreas = _slice(prediction == 1, axis_index, index)
            prediction_lesion = _slice(prediction == 2, axis_index, index)
            if np.any(reference_pancreas):
                panel.contour(reference_pancreas, levels=[0.5], colors=[GREEN], linewidths=1.2)
            if np.any(reference_lesion):
                panel.contour(reference_lesion, levels=[0.5], colors=[ORANGE], linewidths=1.6)
            if np.any(prediction_pancreas):
                panel.contour(prediction_pancreas, levels=[0.5], colors=[SKY], linewidths=1.1, linestyles="--")
            if np.any(prediction_lesion):
                panel.contour(prediction_lesion, levels=[0.5], colors=[PURPLE], linewidths=1.5, linestyles="--")
            if row_index == 0:
                panel.set_title(orientation_names[axis_index], fontsize=10, weight="bold")
            panel.set_xticks([])
            panel.set_yticks([])
        metrics = row_lookup[case_id]
        descriptor = "weak" if row_index < 2 else "strong"
        axes[row_index, 0].set_ylabel(
            _qualitative_row_label(descriptor, case_id, metrics),
            fontsize=8.0,
            rotation=0,
            horizontalalignment="right",
            verticalalignment="center",
            labelpad=7,
        )
    legend = [
        Line2D([0], [0], color=GREEN, linewidth=1.7, label="Reference pancreas"),
        Line2D([0], [0], color=ORANGE, linewidth=1.7, label="Reference lesion"),
        Line2D([0], [0], color=SKY, linestyle="--", linewidth=1.7, label="Predicted pancreas"),
        Line2D([0], [0], color=PURPLE, linestyle="--", linewidth=1.7, label="Predicted lesion"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    figure.suptitle(
        "Deterministically selected weak and strong validation cases (CT window 40/400 HU)",
        fontsize=12.3,
        weight="bold",
    )
    return _save(figure, output_directory / "qualitative_cases", dpi=dpi)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Final nnU-Net checkpoint for curves")
    parser.add_argument("--metrics-json", type=Path, help="Frozen aggregate validation metrics")
    parser.add_argument("--case-csv", type=Path, help="Frozen case-level validation metrics")
    parser.add_argument("--images", type=Path, help="Validation image folder for overlays")
    parser.add_argument("--predictions", type=Path, help="Validation prediction mask folder")
    parser.add_argument("--references", type=Path, help="Validation reference mask folder")
    parser.add_argument("--output-dir", type=Path, default=Path("report/figures"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    if not any((args.checkpoint, args.metrics_json, args.case_csv)):
        parser.error("provide --checkpoint, --metrics-json, and/or --case-csv")
    qualitative_args = (args.images, args.predictions, args.references)
    if any(qualitative_args) and (not all(qualitative_args) or args.case_csv is None):
        parser.error("qualitative overlays require --case-csv, --images, --predictions, and --references")

    outputs: list[Path] = []
    try:
        if args.checkpoint is not None:
            outputs.extend(
                plot_training_curves(
                    load_checkpoint_logging(args.checkpoint), args.output_dir, dpi=args.dpi
                )
            )
        metrics = None
        if args.metrics_json is not None:
            metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
            if not isinstance(metrics, dict):
                raise FigureInputError("Metrics JSON root must be an object")
            outputs.extend(plot_confusion_matrix(metrics, args.output_dir, dpi=args.dpi))
        rows = None
        if args.case_csv is not None:
            rows = read_case_rows(args.case_csv)
            outputs.extend(plot_dice_distribution(rows, args.output_dir, dpi=args.dpi))
        if all(qualitative_args):
            assert rows is not None
            outputs.extend(
                plot_qualitative_cases(
                    rows,
                    args.images,
                    args.predictions,
                    args.references,
                    args.output_dir,
                    dpi=args.dpi,
                )
            )
    except (FigureInputError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    for output in outputs:
        print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
