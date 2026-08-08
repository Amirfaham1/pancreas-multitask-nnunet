#!/usr/bin/env python3
"""Recompute V7 evidence from saved features, masks, and the submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import SimpleITK as sitk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pancreas_multitask.metrics import segmentation_case_metrics  # noqa: E402
from validate_submission import validate_submission  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deliverable", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--speed-benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        if features.ndim == 3:
            features = features[:, 0, :]
        labels = np.asarray(payload["labels"], dtype=np.int64)
        case_ids = np.asarray(payload["case_ids"])
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError(f"Invalid feature bank: {path}")
    return features, labels, case_ids


def _classifier() -> Any:
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def verify_classifier(deliverable: Path) -> dict[str, Any]:
    features = deliverable / "features"
    train1, train_y, train_ids = _load_features(features / "train_stage1.npz")
    train2, train_y2, train_ids2 = _load_features(features / "bank_stage2_geom_k8.npz")
    val1, val_y, val_ids = _load_features(features / "val_stage1.npz")
    val2, val_y2, val_ids2 = _load_features(features / "val_stage2.npz")
    if not (
        np.array_equal(train_y, train_y2)
        and np.array_equal(train_ids, train_ids2)
        and np.array_equal(val_y, val_y2)
        and np.array_equal(val_ids, val_ids2)
    ):
        raise ValueError("Stage-1 and stage-2 feature banks are not aligned")

    train_x = np.concatenate((train1, train2), axis=1)
    val_x = np.concatenate((val1, val2), axis=1)
    refit = _classifier().fit(train_x, train_y)
    refit_prediction = refit.predict(val_x)

    classifier_path = deliverable / "checkpoints" / "classifier_final.joblib"
    metadata = json.loads(
        (deliverable / "checkpoints" / "classifier_final.json").read_text(encoding="utf-8")
    )
    saved_prediction = joblib.load(classifier_path).predict(val_x)
    classifier_digest = sha256(classifier_path)

    oof_scores: list[float] = []
    for seed in range(10):
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        prediction = cross_val_predict(_classifier(), train_x, train_y, cv=folds, method="predict")
        oof_scores.append(float(f1_score(train_y, prediction, average="macro")))

    macro_f1 = float(f1_score(val_y, refit_prediction, average="macro"))
    return {
        "training_cases": int(train_y.size),
        "validation_cases": int(val_y.size),
        "dimensions": int(train_x.shape[1]),
        "macro_f1": macro_f1,
        "accuracy": float(np.mean(val_y == refit_prediction)),
        "confusion_matrix": confusion_matrix(val_y, refit_prediction, labels=[0, 1, 2]).tolist(),
        "oof_macro_f1_10_seeds": oof_scores,
        "oof_macro_f1_mean": float(np.mean(oof_scores)),
        "oof_macro_f1_sd": float(np.std(oof_scores)),
        "refit_matches_saved_predictions": bool(np.array_equal(refit_prediction, saved_prediction)),
        "classifier_sha256": classifier_digest,
        "classifier_hash_matches_metadata": classifier_digest == metadata["sha256"],
        "case_order_sha256": hashlib.sha256("\n".join(val_ids.tolist()).encode()).hexdigest(),
    }


def _snap_reference(array: np.ndarray, *, tolerance: float = 2e-4) -> tuple[np.ndarray, bool]:
    rounded = np.rint(array)
    delta = float(np.max(np.abs(array - rounded)))
    if delta > tolerance:
        raise ValueError(f"Reference mask is not within {tolerance} of integer labels: {delta}")
    labels = set(np.unique(rounded).tolist())
    if not labels.issubset({0.0, 1.0, 2.0}):
        raise ValueError(f"Reference mask has invalid labels: {sorted(labels)}")
    return rounded.astype(np.uint8), bool(delta > 0)


def verify_segmentation(deliverable: Path, validation_root: Path) -> dict[str, Any]:
    prediction_root = deliverable / "results" / "validation_predictions"
    whole: list[float] = []
    lesion: list[float] = []
    repaired = 0
    case_ids: list[str] = []
    for reference_path in sorted(validation_root.glob("subtype*/*.nii.gz")):
        if reference_path.name.endswith("_0000.nii.gz"):
            continue
        prediction_path = prediction_root / reference_path.name
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing validation prediction: {prediction_path}")
        reference_image = sitk.ReadImage(str(reference_path))
        prediction_image = sitk.ReadImage(str(prediction_path))
        reference, was_repaired = _snap_reference(sitk.GetArrayFromImage(reference_image))
        prediction = sitk.GetArrayFromImage(prediction_image)
        if reference.shape != prediction.shape:
            raise ValueError(f"Shape mismatch for {reference_path.name}")
        if not np.allclose(reference_image.GetSpacing(), prediction_image.GetSpacing(), atol=1e-5):
            raise ValueError(f"Spacing mismatch for {reference_path.name}")
        if not np.issubdtype(prediction.dtype, np.integer):
            raise ValueError(f"Prediction is not integer-valued: {prediction_path}")
        if not set(np.unique(prediction).tolist()).issubset({0, 1, 2}):
            raise ValueError(f"Prediction has invalid labels: {prediction_path}")
        metrics = segmentation_case_metrics(prediction, reference)
        whole.append(float(metrics["whole_pancreas_dice"]))
        lesion.append(float(metrics["lesion_dice"]))
        repaired += int(was_repaired)
        case_ids.append(reference_path.name[:-7])
    if len(case_ids) != 36:
        raise ValueError(f"Expected 36 validation cases, found {len(case_ids)}")
    return {
        "cases": len(case_ids),
        "references_requiring_integer_snap": repaired,
        "whole_pancreas_dice_mean": float(np.mean(whole)),
        "whole_pancreas_dice_sd": float(np.std(whole, ddof=1)),
        "whole_pancreas_dice_min": float(np.min(whole)),
        "lesion_dice_mean": float(np.mean(lesion)),
        "lesion_dice_sd": float(np.std(lesion, ddof=1)),
        "lesion_dice_min": float(np.min(lesion)),
    }


def verify_speed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stock = [float(value) for value in payload["stock_seconds"]]
    candidate = [float(value) for value in payload["candidate_seconds"]]
    stock_mean = float(np.mean(stock))
    candidate_mean = float(np.mean(candidate))
    reduction = 100.0 * (stock_mean - candidate_mean) / stock_mean
    arithmetic_matches = bool(
        np.isclose(stock_mean, payload["stock_mean"], rtol=0, atol=1e-9)
        and np.isclose(candidate_mean, payload["candidate_mean"], rtol=0, atol=1e-9)
        and np.isclose(reduction, payload["runtime_reduction_percent"], rtol=0, atol=1e-9)
    )
    equivalence = payload.get("equivalence")
    complete_protocol = bool(isinstance(equivalence, dict) and equivalence.get("passed"))
    return {
        "stock_mean_seconds": stock_mean,
        "candidate_mean_seconds": candidate_mean,
        "runtime_reduction_percent": reduction,
        "arithmetic_matches": arithmetic_matches,
        "complete_output_audit_present": complete_protocol,
        "passed": bool(arithmetic_matches and complete_protocol and reduction >= 10.0),
        "sha256": sha256(path),
    }


def _close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0, atol=1e-12))


def main() -> int:
    args = build_parser().parse_args()
    deliverable = args.deliverable.expanduser().resolve()
    final_results = json.loads(
        (deliverable / "results" / "FINAL_RESULTS.json").read_text(encoding="utf-8")
    )
    classification = verify_classifier(deliverable)
    segmentation = verify_segmentation(deliverable, args.validation_root.expanduser().resolve())
    submission = validate_submission(
        deliverable / "results" / "Amirfaham_Fallahpour_results.zip",
        args.test_images.expanduser().resolve(),
        expected_count=72,
    )
    checks = {
        "classification_matches_final_results": _close(
            classification["macro_f1"], final_results["validation"]["macro_f1"]
        ),
        "whole_dice_matches_final_results": _close(
            segmentation["whole_pancreas_dice_mean"],
            final_results["validation"]["whole_pancreas_dice_mean"],
        ),
        "lesion_dice_matches_final_results": _close(
            segmentation["lesion_dice_mean"],
            final_results["validation"]["lesion_dice_mean"],
        ),
        "classifier_predictions_reproduced": classification["refit_matches_saved_predictions"],
        "classifier_hash_verified": classification["classifier_hash_matches_metadata"],
        "submission_contract_valid": submission["valid"],
    }
    speed = verify_speed(args.speed_benchmark.resolve()) if args.speed_benchmark else None
    if speed is not None:
        checks["speed_protocol_and_gate_valid"] = speed["passed"]

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "verified" if all(checks.values()) else "failed",
        "checks": checks,
        "classification": classification,
        "segmentation": segmentation,
        "submission": {
            "valid": submission["valid"],
            "mask_count": submission["validated_mask_count"],
            "csv_row_count": submission["validated_csv_row_count"],
            "archive_sha256": submission["archive_sha256"],
        },
        "speed": speed,
        "source": {
            "deliverable_directory": deliverable.name,
            "archive_sha256": sha256(args.source_archive.resolve())
            if args.source_archive
            else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
