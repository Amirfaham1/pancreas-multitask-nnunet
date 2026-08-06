#!/usr/bin/env python3
"""Select, calibrate, and refit the locked classifier using training only."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.case_classifier_selection import (
    CaseFeatureDataset,
    evaluate_locked_candidates,
    fit_selected_classifier,
    identifier_independent_dataset_sha256,
    load_locked_search,
    save_classifier_bundle,
)
from pancreas_multitask.classification_rescue import file_sha256
from pancreas_multitask.decision_calibration import (
    evaluate_cross_fitted_offsets,
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_dataset(feature_directory: Path, lock: dict[str, Any]) -> CaseFeatureDataset:
    schema = json.loads((feature_directory / "feature_schema.json").read_text(encoding="utf-8"))
    view_names = tuple(lock["feature_extraction"]["feature_views"])
    with np.load(feature_directory / "train_case_features.npz", allow_pickle=False) as data:
        dataset = CaseFeatureDataset(
            tuple(str(value) for value in data["case_ids"].tolist()),
            np.asarray(data["labels"], dtype=np.int64),
            {
                view_names[0]: np.asarray(data["feature_view_0"], dtype=np.float32),
                view_names[1]: np.asarray(data["feature_view_1"], dtype=np.float32),
            },
            {name: tuple(schema["feature_names"][name]) for name in view_names},
        )
    if dataset.case_count != int(lock["development_boundary"]["case_count"]):
        raise ValueError("Feature dataset count differs from the locked training split")
    return dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run locked repeated-CV selection and class-offset calibration"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Required acknowledgement that v3 classical results cannot be official",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "configs" / "phd_classification_upgrade_lock_v3.json",
    )
    parser.add_argument(
        "--decision-lock",
        type=Path,
        default=ROOT / "configs" / "phd_classification_decision_lock_v4.json",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    if not args.diagnostic_only:
        raise ValueError("V3 classical selection requires explicit --diagnostic-only")
    feature_directory = args.features.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = args.lock.expanduser().resolve()
    decision_lock_path = args.decision_lock.expanduser().resolve()
    lock = load_locked_search(lock_path)
    decision_lock = json.loads(decision_lock_path.read_text(encoding="utf-8"))
    if decision_lock.get("lock_status") != (
        "frozen_before_primary_v3_feature_extraction_or_candidate_cv"
    ):
        raise ValueError("Decision-calibration lock is not frozen")
    if file_sha256(lock_path) != decision_lock["base_classifier_lock"]["sha256"]:
        raise ValueError("Decision lock is not bound to the supplied base lock")
    dataset = _load_dataset(feature_directory, lock)
    dataset_hash = identifier_independent_dataset_sha256(dataset)
    extraction_audit_path = feature_directory / "train_case_feature_extraction_audit.json"
    extraction_audit = json.loads(extraction_audit_path.read_text(encoding="utf-8"))
    if extraction_audit["identifier_independent_dataset_sha256"] != dataset_hash:
        raise ValueError("Feature dataset differs from its extraction audit")
    if any(
        extraction_audit[field]
        for field in (
            "ground_truth_masks_loaded",
            "combined_train_validation_metadata_read",
            "official_validation_images_read",
            "official_validation_masks_read",
            "official_validation_labels_read",
            "test_data_read",
            "case_ids_paths_or_filenames_in_model_matrix",
        )
    ):
        raise ValueError("Feature extraction audit violates the train-only boundary")

    started_at_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    selection = evaluate_locked_candidates(dataset, lock)
    selection["diagnostic_only"] = True
    selection["eligible_for_official"] = False
    selection["started_at_utc"] = started_at_utc
    selection["finished_at_utc"] = datetime.now(UTC).isoformat()
    selection["elapsed_seconds"] = time.perf_counter() - started
    selection["feature_dataset_path_recorded"] = False
    selection["feature_dataset_sha256"] = file_sha256(feature_directory / "train_case_features.npz")
    selection["feature_extraction_audit_sha256"] = file_sha256(extraction_audit_path)
    selection["base_lock_sha256"] = file_sha256(lock_path)
    selection_path = output / "train_only_candidate_selection.json"
    _atomic_write_json(selection_path, selection)

    calibration = evaluate_cross_fitted_offsets(selection, decision_lock)
    calibration["diagnostic_only"] = True
    calibration["eligible_for_official"] = False
    calibration["decision_lock_sha256"] = file_sha256(decision_lock_path)
    calibration["selection_audit_sha256"] = file_sha256(selection_path)
    calibration_path = output / "train_only_decision_calibration.json"
    _atomic_write_json(calibration_path, calibration)

    estimator, metadata = fit_selected_classifier(dataset, lock, selection)
    metadata.update(
        {
            "class_offsets": calibration["final_offsets"],
            "calibration_activated": calibration["calibration_activated"],
            "selection_audit_sha256": file_sha256(selection_path),
            "calibration_audit_sha256": file_sha256(calibration_path),
            "feature_schema_sha256": extraction_audit["feature_schema_sha256"],
            "base_lock_sha256": file_sha256(lock_path),
            "decision_lock_sha256": file_sha256(decision_lock_path),
            "ground_truth_masks_used_as_features": False,
            "official_validation_used": False,
            "smote_used": False,
            "diagnostic_only": True,
            "eligible_for_official": False,
            "final_inference_must_reject": True,
        }
    )
    classifier_path = output / "diagnostic_classical_case_classifier.joblib"
    save_classifier_bundle(classifier_path, estimator, metadata)
    final_audit = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic_only": True,
        "eligible_for_official": False,
        "scope": "252_supplied_training_cases_only",
        "selected_candidate_id": selection["selected_candidate_id"],
        "selected_mean_repeat_oof_macro_f1": selection["selected_mean_repeat_oof_macro_f1"],
        "selected_std_repeat_oof_macro_f1_population": selection[
            "selected_std_repeat_oof_macro_f1_population"
        ],
        "selected_minimum_repeat_per_class_recall": selection[
            "selected_minimum_repeat_per_class_recall"
        ],
        "calibration_activated": calibration["calibration_activated"],
        "final_offsets": calibration["final_offsets"],
        "classifier_sha256": file_sha256(classifier_path),
        "selection_audit_sha256": file_sha256(selection_path),
        "calibration_audit_sha256": file_sha256(calibration_path),
        "feature_dataset_sha256": selection["feature_dataset_sha256"],
        "identifier_independent_dataset_sha256": dataset_hash,
        "official_validation_used": False,
        "test_data_used": False,
        "identifiers_used_as_features": False,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
    }
    final_audit_path = output / "case_classifier_fit_audit.json"
    _atomic_write_json(final_audit_path, final_audit)
    print(f"Selected candidate: {selection['selected_candidate_id']}")
    print(f"Train-only repeated-OOF macro-F1: {selection['selected_mean_repeat_oof_macro_f1']:.6f}")
    print(f"Classifier: {classifier_path}")
    print(f"Fit audit: {final_audit_path}")
    return classifier_path


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
