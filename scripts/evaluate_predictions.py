#!/usr/bin/env python3
"""Evaluate saved validation predictions independently from model training.

The command can evaluate segmentation masks, subtype predictions, or both.  It
writes a machine-readable aggregate JSON report and a case-level CSV.  No
metric is inferred from training logs.

Example::

    python scripts/evaluate_predictions.py \
      --predictions artifacts/validation_predictions/masks \
      --references "$nnUNet_raw/Dataset501_PancreasMultitask/labelsVal" \
      --classification-predictions artifacts/validation_predictions/subtypes.csv \
      --classification-references \
        "$nnUNet_raw/Dataset501_PancreasMultitask/classification_manifest.json" \
      --classification-reference-split validation \
      --output-json artifacts/validation_metrics.json \
      --output-csv artifacts/validation_case_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METRICS_SOURCE = REPOSITORY_ROOT / "src" / "pancreas_multitask"
if str(METRICS_SOURCE) not in sys.path:
    # Import the standalone metric module without running package __init__,
    # which imports the PyTorch network and is unnecessary for evaluation.
    sys.path.insert(0, str(METRICS_SOURCE))

from metrics import (
    MetricInputError,
    aggregate_segmentation_metrics,
    classification_metrics,
    segmentation_case_metrics,
    validate_label_array,
)

NIFTI_SUFFIX = ".nii.gz"
CHANNEL_SUFFIX = "_0000.nii.gz"
NAME_COLUMNS = ("Names", "case_id", "name", "filename", "id")
LABEL_COLUMNS = ("Subtype", "label", "class_id", "prediction", "predicted_subtype")


class EvaluationError(RuntimeError):
    """Raised when saved predictions cannot be evaluated safely."""


def _import_nibabel() -> Any:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise EvaluationError(
            "nibabel is required for segmentation evaluation; install project dependencies"
        ) from exc
    return nib


def _canonical_case_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise EvaluationError("Encountered an empty case identifier")
    if "/" in text or "\\" in text:
        raise EvaluationError(f"Case identifier must not contain a path: {text!r}")
    if text.endswith(CHANNEL_SUFFIX):
        text = text[: -len(CHANNEL_SUFFIX)]
    elif text.endswith(NIFTI_SUFFIX):
        text = text[: -len(NIFTI_SUFFIX)]
    if not text:
        raise EvaluationError(f"Invalid case identifier: {value!r}")
    return text


def _discover_masks(directory: Path, *, role: str) -> dict[str, Path]:
    root = directory.resolve()
    if not root.is_dir():
        raise EvaluationError(f"{role} mask directory does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob(f"*{NIFTI_SUFFIX}")):
        if path.name.startswith(".") or path.name.startswith("._"):
            continue
        if path.name.endswith(CHANNEL_SUFFIX):
            continue
        case_id = _canonical_case_id(path.name)
        if case_id in result:
            raise EvaluationError(
                f"Duplicate {role} mask for {case_id}: {result[case_id]} and {path}"
            )
        result[case_id] = path
    if not result:
        raise EvaluationError(f"No {role} masks ending in {NIFTI_SUFFIX} found below {root}")
    return result


def _geometry(image: Any) -> tuple[tuple[int, ...], np.ndarray, tuple[float, ...]]:
    shape = tuple(int(value) for value in image.shape)
    affine = np.asarray(image.affine, dtype=np.float64)
    zooms = tuple(float(value) for value in image.header.get_zooms()[: len(shape)])
    return shape, affine, zooms


def _load_mask(path: Path, *, role: str, tolerance: float) -> tuple[np.ndarray, Any]:
    nib = _import_nibabel()
    try:
        image = nib.load(str(path))
        shape, affine, zooms = _geometry(image)
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise EvaluationError(f"{role} mask must be non-empty and 3D, got {shape}: {path}")
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise EvaluationError(f"{role} mask has an invalid affine: {path}")
        if len(zooms) != 3 or any(not math.isfinite(value) or value <= 0 for value in zooms):
            raise EvaluationError(f"{role} mask has invalid voxel spacing {zooms}: {path}")
        data = np.asanyarray(image.dataobj)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Could not read {role} mask {path}: {exc}") from exc
    try:
        labels = validate_label_array(
            data,
            name=f"{role} mask {path}",
            integer_tolerance=tolerance,
        )
    except MetricInputError as exc:
        raise EvaluationError(str(exc)) from exc
    return labels, image


def _assert_same_geometry(
    prediction_image: Any,
    reference_image: Any,
    *,
    prediction_path: Path,
    reference_path: Path,
    affine_tolerance: float,
) -> None:
    predicted_shape, predicted_affine, predicted_zooms = _geometry(prediction_image)
    reference_shape, reference_affine, reference_zooms = _geometry(reference_image)
    if predicted_shape != reference_shape:
        raise EvaluationError(
            f"Geometry mismatch for {prediction_path.name}: prediction shape "
            f"{predicted_shape}, reference shape {reference_shape} ({reference_path})"
        )
    if not np.allclose(
        predicted_affine,
        reference_affine,
        rtol=affine_tolerance,
        atol=affine_tolerance,
    ):
        raise EvaluationError(
            f"Geometry mismatch for {prediction_path.name}: affine differs from {reference_path}"
        )
    if not np.allclose(
        predicted_zooms,
        reference_zooms,
        rtol=affine_tolerance,
        atol=affine_tolerance,
    ):
        raise EvaluationError(
            f"Geometry mismatch for {prediction_path.name}: voxel spacing "
            f"{predicted_zooms} versus {reference_zooms}"
        )


def evaluate_segmentation(
    prediction_directory: Path,
    reference_directory: Path,
    *,
    empty_empty_dice: float,
    label_tolerance: float,
    affine_tolerance: float,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = _discover_masks(prediction_directory, role="prediction")
    references = _discover_masks(reference_directory, role="reference")
    missing = sorted(set(references) - set(predictions))
    unexpected = sorted(set(predictions) - set(references))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing predictions={missing}")
        if unexpected:
            details.append(f"predictions without references={unexpected}")
        raise EvaluationError("Segmentation case mismatch: " + "; ".join(details))

    case_rows: list[dict[str, Any]] = []
    for case_id in sorted(references):
        predicted, predicted_image = _load_mask(
            predictions[case_id], role="prediction", tolerance=label_tolerance
        )
        target, target_image = _load_mask(
            references[case_id], role="reference", tolerance=label_tolerance
        )
        _assert_same_geometry(
            predicted_image,
            target_image,
            prediction_path=predictions[case_id],
            reference_path=references[case_id],
            affine_tolerance=affine_tolerance,
        )
        try:
            row = segmentation_case_metrics(
                predicted,
                target,
                case_id=case_id,
                empty_empty=empty_empty_dice,
            )
        except MetricInputError as exc:
            raise EvaluationError(f"Could not score segmentation case {case_id}: {exc}") from exc
        case_rows.append(row)

    aggregate = aggregate_segmentation_metrics(
        case_rows,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    return aggregate, case_rows


def _choose_column(
    fieldnames: Sequence[str] | None,
    candidates: Sequence[str],
    *,
    purpose: str,
    path: Path,
) -> str:
    if not fieldnames:
        raise EvaluationError(f"Label CSV has no header: {path}")
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    casefold_lookup = {name.casefold(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.casefold() in casefold_lookup:
            return casefold_lookup[candidate.casefold()]
    raise EvaluationError(
        f"Could not find a {purpose} column in {path}; columns are {list(fieldnames)}"
    )


def _parse_class_label(value: Any, *, path: Path, case_id: str) -> int:
    if isinstance(value, bool):
        raise EvaluationError(f"Boolean subtype for {case_id} in {path}")
    text = str(value).strip()
    try:
        label = int(text)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"Invalid subtype {value!r} for {case_id} in {path}") from exc
    if text not in {str(label), f"+{label}"}:
        raise EvaluationError(f"Subtype must be an integer, got {value!r} for {case_id} in {path}")
    if label not in {0, 1, 2}:
        raise EvaluationError(f"Subtype must be 0, 1, or 2, got {label} for {case_id} in {path}")
    return label


def _rows_to_label_mapping(
    rows: Sequence[Mapping[str, Any]],
    *,
    path: Path,
    split: str | None,
) -> dict[str, int]:
    if not rows:
        raise EvaluationError(f"Label table is empty: {path}")
    available_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in available_fields:
                available_fields.append(key)
    name_column = _choose_column(available_fields, NAME_COLUMNS, purpose="case-name", path=path)
    label_column = _choose_column(available_fields, LABEL_COLUMNS, purpose="subtype", path=path)
    result: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=2):
        if split is not None and str(row.get("split", "")).strip() != split:
            continue
        try:
            case_id = _canonical_case_id(row.get(name_column, ""))
        except EvaluationError as exc:
            raise EvaluationError(f"{exc} (row {row_number} of {path})") from exc
        if case_id in result:
            raise EvaluationError(f"Duplicate classification case {case_id} in {path}")
        result[case_id] = _parse_class_label(row.get(label_column, ""), path=path, case_id=case_id)
    if not result:
        suffix = f" for split {split!r}" if split is not None else ""
        raise EvaluationError(f"No classification labels found in {path}{suffix}")
    return result


def read_label_table(path: Path, *, split: str | None = None) -> dict[str, int]:
    """Read classification labels from CSV or supported JSON manifests."""

    source = path.resolve()
    if not source.is_file():
        raise EvaluationError(f"Classification label file does not exist: {source}")
    if source.suffix.casefold() == ".csv":
        try:
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise EvaluationError(f"Could not read classification CSV {source}: {exc}") from exc
        return _rows_to_label_mapping(rows, path=source, split=split)
    if source.suffix.casefold() == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"Could not read classification JSON {source}: {exc}") from exc
        if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
            rows = payload["cases"]
            if not all(isinstance(row, dict) for row in rows):
                raise EvaluationError(f"JSON cases must be objects: {source}")
            return _rows_to_label_mapping(rows, path=source, split=split)
        if isinstance(payload, dict):
            if split is not None:
                raise EvaluationError(
                    f"Cannot filter split {split!r} from mapping-only JSON {source}"
                )
            result: dict[str, int] = {}
            for raw_name, raw_label in payload.items():
                case_id = _canonical_case_id(raw_name)
                if case_id in result:
                    raise EvaluationError(f"Duplicate classification case {case_id} in {source}")
                result[case_id] = _parse_class_label(raw_label, path=source, case_id=case_id)
            if not result:
                raise EvaluationError(f"Classification JSON mapping is empty: {source}")
            return result
        if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
            return _rows_to_label_mapping(payload, path=source, split=split)
        raise EvaluationError(
            f"Unsupported classification JSON structure in {source}; expected a mapping or cases list"
        )
    raise EvaluationError(f"Classification labels must be .csv or .json: {source}")


def evaluate_classification(
    prediction_path: Path,
    reference_path: Path,
    *,
    reference_split: str | None,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = read_label_table(prediction_path)
    references = read_label_table(reference_path, split=reference_split)
    missing_references = sorted(set(predictions) - set(references))
    if missing_references:
        raise EvaluationError(
            f"Classification predictions have no reference label: {missing_references}"
        )
    evaluated_ids = sorted(predictions)
    truth = [references[case_id] for case_id in evaluated_ids]
    predicted = [predictions[case_id] for case_id in evaluated_ids]
    try:
        aggregate = classification_metrics(
            truth,
            predicted,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )
    except MetricInputError as exc:
        raise EvaluationError(f"Could not score classification predictions: {exc}") from exc
    case_rows = [
        {
            "case_id": case_id,
            "reference_subtype": references[case_id],
            "predicted_subtype": predictions[case_id],
            "classification_correct": references[case_id] == predictions[case_id],
        }
        for case_id in evaluated_ids
    ]
    aggregate["unused_reference_case_count"] = len(set(references) - set(predictions))
    return aggregate, case_rows


def _merge_case_rows(
    segmentation_rows: Sequence[Mapping[str, Any]],
    classification_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    segmentation = {str(row["case_id"]): dict(row) for row in segmentation_rows}
    classification = {str(row["case_id"]): dict(row) for row in classification_rows}
    if segmentation and classification and set(segmentation) != set(classification):
        raise EvaluationError(
            "Segmentation and classification evaluated different cases: "
            f"segmentation-only={sorted(set(segmentation) - set(classification))}; "
            f"classification-only={sorted(set(classification) - set(segmentation))}"
        )
    case_ids = sorted(set(segmentation) | set(classification))
    merged: list[dict[str, Any]] = []
    for case_id in case_ids:
        row: dict[str, Any] = {"case_id": case_id}
        row.update(segmentation.get(case_id, {}))
        row.update(classification.get(case_id, {}))
        merged.append(row)
    return merged


def _atomic_write_text(path: Path, text: str) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_case_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvaluationError("Refusing to write an empty case-level CSV")
    preferred = [
        "case_id",
        "whole_pancreas_dice",
        "lesion_dice",
        "reference_subtype",
        "predicted_subtype",
        "classification_correct",
        "whole_pancreas_predicted_voxels",
        "whole_pancreas_reference_voxels",
        "lesion_predicted_voxels",
        "lesion_reference_voxels",
        "whole_pancreas_empty_empty",
        "lesion_empty_empty",
    ]
    present = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in present]
    extra = sorted(present - set(fieldnames))
    fieldnames.extend(extra)
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        "--prediction-masks",
        dest="predictions",
        type=Path,
        help="Directory containing predicted validation masks",
    )
    parser.add_argument(
        "--references",
        "--reference-masks",
        dest="references",
        type=Path,
        help="Directory containing reference validation masks",
    )
    parser.add_argument(
        "--classification-predictions",
        "--predicted-subtypes",
        dest="classification_predictions",
        type=Path,
        help="Predicted subtype CSV or JSON",
    )
    parser.add_argument(
        "--classification-references",
        "--reference-subtypes",
        dest="classification_references",
        type=Path,
        help="Reference subtype CSV or JSON",
    )
    parser.add_argument(
        "--classification-reference-split",
        help="Optional split value (for example 'validation') used with a manifest",
    )
    parser.add_argument("--output-json", type=Path, help="Aggregate JSON output path")
    parser.add_argument("--output-csv", type=Path, help="Case-level CSV output path")
    parser.add_argument(
        "--empty-empty-dice",
        type=float,
        default=1.0,
        help="Dice assigned when prediction and reference foreground are both empty (default: 1)",
    )
    parser.add_argument(
        "--label-tolerance",
        type=float,
        default=0.0,
        help="Maximum allowed distance from integer mask labels (default: exact)",
    )
    parser.add_argument(
        "--affine-tolerance",
        type=float,
        default=1e-5,
        help="Absolute/relative tolerance for affine and spacing comparison",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Case-resampling iterations for percentile confidence intervals (0 disables)",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    segmentation_requested = args.predictions is not None or args.references is not None
    classification_requested = (
        args.classification_predictions is not None or args.classification_references is not None
    )
    if not segmentation_requested and not classification_requested:
        raise EvaluationError(
            "Provide a segmentation prediction/reference pair, a classification pair, or both"
        )
    if (args.predictions is None) != (args.references is None):
        raise EvaluationError("--predictions and --references must be provided together")
    if (args.classification_predictions is None) != (args.classification_references is None):
        raise EvaluationError(
            "--classification-predictions and --classification-references must be provided together"
        )
    if not math.isfinite(args.empty_empty_dice) or not 0 <= args.empty_empty_dice <= 1:
        raise EvaluationError("--empty-empty-dice must be finite and in [0, 1]")
    if not math.isfinite(args.label_tolerance) or args.label_tolerance < 0:
        raise EvaluationError("--label-tolerance must be finite and non-negative")
    if not math.isfinite(args.affine_tolerance) or args.affine_tolerance < 0:
        raise EvaluationError("--affine-tolerance must be finite and non-negative")
    if args.bootstrap_samples < 0:
        raise EvaluationError("--bootstrap-samples must be non-negative")
    if not math.isfinite(args.confidence) or not 0 < args.confidence < 1:
        raise EvaluationError("--confidence must be finite and strictly between 0 and 1")

    result: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_policy": {
            "whole_pancreas": "label > 0",
            "lesion": "label == 2",
            "empty_empty_dice": float(args.empty_empty_dice),
            "one_sided_empty_dice": 0.0,
            "classification_labels": [0, 1, 2],
            "classification_zero_division": 0.0,
            "confusion_matrix_rows": "reference",
            "confusion_matrix_columns": "prediction",
            "aggregation": "unweighted case mean",
            "bootstrap_seed": int(args.seed),
        },
    }
    segmentation_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    if segmentation_requested:
        result["segmentation"], segmentation_rows = evaluate_segmentation(
            args.predictions,
            args.references,
            empty_empty_dice=args.empty_empty_dice,
            label_tolerance=args.label_tolerance,
            affine_tolerance=args.affine_tolerance,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
    if classification_requested:
        result["classification"], classification_rows = evaluate_classification(
            args.classification_predictions,
            args.classification_references,
            reference_split=args.classification_reference_split,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
    case_rows = _merge_case_rows(segmentation_rows, classification_rows)
    result["case_count"] = len(case_rows)
    result["cases"] = case_rows

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        _atomic_write_text(args.output_json, rendered)
    if args.output_csv is not None:
        _write_case_csv(args.output_csv, case_rows)
    if args.output_json is None:
        print(rendered, end="")
    else:
        print(f"Wrote aggregate metrics: {args.output_json.resolve()}")
        if args.output_csv is not None:
            print(f"Wrote case metrics: {args.output_csv.resolve()}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (EvaluationError, MetricInputError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
