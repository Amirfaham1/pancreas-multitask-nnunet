"""Deterministic train-only selection for the locked case classifier.

All transformations are fitted inside each training fold.  Identifiers are
used only to join audit rows after prediction; split assignment and fitting use
numeric arrays in a content-canonical order, making the result invariant to
case renaming and input enumeration order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


@dataclass(frozen=True, slots=True)
class CaseFeatureDataset:
    """Identifier-separated numeric case features."""

    case_ids: tuple[str, ...]
    labels: np.ndarray
    views: Mapping[str, np.ndarray]
    feature_names: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if labels.ndim != 1 or labels.size != len(self.case_ids):
            raise ValueError("labels and case_ids must be aligned one-dimensional arrays")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("case_ids must be unique")
        if set(labels.tolist()) != {0, 1, 2}:
            raise ValueError("labels must contain classes 0, 1, and 2")
        if set(self.views) != set(self.feature_names) or not self.views:
            raise ValueError("Every feature view must have exactly one schema")
        for view_name, raw_matrix in self.views.items():
            matrix = np.asarray(raw_matrix)
            names = self.feature_names[view_name]
            if matrix.ndim != 2 or matrix.shape[0] != labels.size:
                raise ValueError(f"Feature view {view_name!r} has invalid shape {matrix.shape}")
            if matrix.shape[1] != len(names) or len(names) != len(set(names)):
                raise ValueError(f"Feature schema mismatch in view {view_name!r}")
            if not np.isfinite(matrix).all():
                raise ValueError(f"Feature view {view_name!r} contains non-finite values")
            forbidden = ("case_id", "filename", "file_path", "directory", "enumeration")
            if any(any(token in name.casefold() for token in forbidden) for name in names):
                raise ValueError(f"Feature view {view_name!r} contains identifier-like names")

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    def with_case_ids(self, case_ids: Sequence[str]) -> CaseFeatureDataset:
        """Replace provenance IDs without changing any numeric model input."""

        return CaseFeatureDataset(
            tuple(case_ids),
            np.asarray(self.labels).copy(),
            {name: np.asarray(values).copy() for name, values in self.views.items()},
            dict(self.feature_names),
        )


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    feature_view: str
    family: str
    parameters: Mapping[str, Any]


def _float_id(value: float) -> str:
    return format(float(value), ".12g").replace(".", "p").replace("-", "m")


def load_locked_search(path: str | Path) -> dict[str, Any]:
    lock_path = Path(path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == 3:
        inheritance = payload.get("inherits", {})
        base_path = lock_path.parent.parent / str(inheritance.get("path", ""))
        if not base_path.is_file():
            raise ValueError("Classification v3 lock has no readable v2 base")
        base_bytes = base_path.read_bytes()
        if hashlib.sha256(base_bytes).hexdigest() != inheritance.get("sha256"):
            raise ValueError("Classification v3 base-lock hash does not match")
        base = json.loads(base_bytes.decode("utf-8"))

        def merge(first: dict[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
            result = dict(first)
            for key, value in second.items():
                if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                    result[key] = merge(dict(result[key]), value)
                else:
                    result[key] = value
            return result

        payload = merge(base, payload.get("overrides", {}))
        payload["schema_version"] = 3
        payload["effective_lock_path"] = str(lock_path)
    elif payload.get("schema_version") != 2:
        raise ValueError("Classification search requires the prospective v2/v3 lock")
    if payload.get("lock_status") != (
        "frozen_before_inference_matched_encoder_feature_extraction_or_candidate_cv"
    ):
        raise ValueError("Classification search lock is not frozen")
    return payload


def enumerate_locked_candidates(lock: Mapping[str, Any]) -> tuple[CandidateSpec, ...]:
    """Expand exactly the finite grid declared in the v2 JSON lock."""

    feature_views = tuple(lock["feature_extraction"]["feature_views"])
    by_family = {item["family"]: item for item in lock["candidate_grid"]}
    expected_families = {
        "balanced_multinomial_logistic_regression",
        "balanced_linear_svm",
        "balanced_rbf_svm",
        "balanced_extra_trees",
    }
    if set(by_family) != expected_families:
        raise ValueError("Candidate families differ from the locked implementation")

    candidates: list[CandidateSpec] = []
    for view in feature_views:
        for c_value in by_family["balanced_multinomial_logistic_regression"]["C"]:
            candidates.append(
                CandidateSpec(
                    f"{view}__logreg_C_{_float_id(c_value)}",
                    view,
                    "balanced_multinomial_logistic_regression",
                    {"C": float(c_value), "max_iter": 5000},
                )
            )
        for c_value in by_family["balanced_linear_svm"]["C"]:
            candidates.append(
                CandidateSpec(
                    f"{view}__linear_svm_C_{_float_id(c_value)}",
                    view,
                    "balanced_linear_svm",
                    {"C": float(c_value), "max_iter": 20000},
                )
            )
        rbf = by_family["balanced_rbf_svm"]
        for pca_variance in rbf["pca_variance"]:
            for c_value in rbf["C"]:
                candidates.append(
                    CandidateSpec(
                        (f"{view}__rbf_svm_pca_{_float_id(pca_variance)}_C_{_float_id(c_value)}"),
                        view,
                        "balanced_rbf_svm",
                        {
                            "pca_variance": float(pca_variance),
                            "C": float(c_value),
                            "gamma": "scale",
                        },
                    )
                )
        extra_trees = by_family["balanced_extra_trees"]
        for minimum_leaf in extra_trees["min_samples_leaf"]:
            candidates.append(
                CandidateSpec(
                    f"{view}__extra_trees_leaf_{int(minimum_leaf)}",
                    view,
                    "balanced_extra_trees",
                    {
                        "n_estimators": 600,
                        "max_features": "sqrt",
                        "min_samples_leaf": int(minimum_leaf),
                        "class_weight": "balanced",
                    },
                )
            )
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Locked candidate IDs are not unique")
    return tuple(candidates)


def build_estimator(candidate: CandidateSpec, *, seed: int) -> Pipeline:
    """Instantiate one candidate with all learned preprocessing in-pipeline."""

    if candidate.family == "balanced_multinomial_logistic_regression":
        estimator: BaseEstimator = LogisticRegression(
            C=float(candidate.parameters["C"]),
            class_weight="balanced",
            solver="lbfgs",
            max_iter=int(candidate.parameters["max_iter"]),
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    elif candidate.family == "balanced_linear_svm":
        estimator = LinearSVC(
            C=float(candidate.parameters["C"]),
            class_weight="balanced",
            max_iter=int(candidate.parameters["max_iter"]),
            dual="auto",
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    elif candidate.family == "balanced_rbf_svm":
        estimator = SVC(
            C=float(candidate.parameters["C"]),
            gamma=str(candidate.parameters["gamma"]),
            class_weight="balanced",
            decision_function_shape="ovr",
            probability=False,
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=float(candidate.parameters["pca_variance"]),
                    svd_solver="full",
                ),
            ),
            ("classifier", estimator),
        ]
    elif candidate.family == "balanced_extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=int(candidate.parameters["n_estimators"]),
            max_features=str(candidate.parameters["max_features"]),
            min_samples_leaf=int(candidate.parameters["min_samples_leaf"]),
            class_weight=str(candidate.parameters["class_weight"]),
            random_state=seed,
            n_jobs=-1,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator),
        ]
    else:
        raise ValueError(f"Unknown candidate family: {candidate.family}")
    return Pipeline(steps)


def canonical_content_order(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Order samples by numeric content, never ID/path/enumeration position."""

    values = np.asarray(matrix, dtype="<f4")
    targets = np.asarray(labels, dtype="<i8")
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("matrix and labels are not aligned")
    keys: list[bytes] = []
    for row, target in zip(values, targets, strict=True):
        digest = hashlib.sha256()
        digest.update(row.tobytes(order="C"))
        digest.update(target.tobytes())
        keys.append(digest.digest())
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)


def identifier_independent_dataset_sha256(dataset: CaseFeatureDataset) -> str:
    """Hash numeric content independent of IDs and row enumeration order."""

    digest = hashlib.sha256()
    labels = np.asarray(dataset.labels, dtype="<i8")
    for view in sorted(dataset.views):
        encoded_view = view.encode("utf-8")
        digest.update(len(encoded_view).to_bytes(8, "little"))
        digest.update(encoded_view)
        for name in dataset.feature_names[view]:
            encoded_name = name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(8, "little"))
            digest.update(encoded_name)
        matrix = np.asarray(dataset.views[view], dtype="<f4", order="C")
        digest.update(np.asarray(matrix.shape[1:], dtype="<i8").tobytes())
    row_hashes: list[bytes] = []
    ordered_views = sorted(dataset.views)
    for row_index, target in enumerate(labels):
        row_digest = hashlib.sha256()
        row_digest.update(target.tobytes())
        for view in ordered_views:
            matrix = np.asarray(dataset.views[view], dtype="<f4", order="C")
            row_digest.update(matrix[row_index].tobytes(order="C"))
        row_hashes.append(row_digest.digest())
    for row_hash in sorted(row_hashes):
        digest.update(row_hash)
    return digest.hexdigest()


def _class_counts(values: np.ndarray) -> dict[str, int]:
    return {str(label): int(np.sum(values == label)) for label in range(3)}


def estimator_log_scores(estimator: Pipeline, matrix: np.ndarray) -> np.ndarray:
    """Return comparable three-class log scores under the v4 contract."""

    predict_probabilities = getattr(estimator, "predict_proba", None)
    if callable(predict_probabilities):
        probabilities = np.asarray(predict_probabilities(matrix), dtype=np.float64)
        if probabilities.shape != (matrix.shape[0], 3):
            raise ValueError("Estimator predict_proba did not return three classes")
        scores = np.log(np.clip(probabilities, 1e-7, 1.0))
    else:
        decision_function = getattr(estimator, "decision_function", None)
        if not callable(decision_function):
            raise TypeError("Estimator exposes neither predict_proba nor decision_function")
        margins = np.asarray(decision_function(matrix), dtype=np.float64)
        if margins.shape != (matrix.shape[0], 3):
            raise ValueError("Estimator decision_function did not return three classes")
        scores = margins - logsumexp(margins, axis=1, keepdims=True)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Estimator emitted non-finite log scores")
    predictions = np.asarray(estimator.predict(matrix), dtype=np.int64)
    if not np.array_equal(scores.argmax(axis=1), predictions):
        raise RuntimeError("Log-score argmax differs from the estimator prediction")
    return scores


def evaluate_locked_candidates(
    dataset: CaseFeatureDataset,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the complete prospective repeated stratified train-only search."""

    candidates = enumerate_locked_candidates(lock)
    seeds = tuple(int(seed) for seed in lock["selection"]["repeat_seeds"])
    fold_count = int(lock["selection"]["folds"])
    labels_original = np.asarray(dataset.labels, dtype=np.int64)
    candidate_results: list[dict[str, Any]] = []

    for candidate in candidates:
        matrix_original = np.asarray(dataset.views[candidate.feature_view], dtype=np.float32)
        order = canonical_content_order(matrix_original, labels_original)
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(order.size)
        matrix = matrix_original[order]
        labels = labels_original[order]
        repeat_scores: list[float] = []
        repeat_per_class_recalls: list[list[float]] = []
        fold_rows: list[dict[str, Any]] = []
        repeat_predictions: list[dict[str, Any]] = []

        for repeat_index, seed in enumerate(seeds):
            splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
            oof = np.full(labels.shape, -1, dtype=np.int64)
            oof_log_scores = np.full((labels.size, 3), np.nan, dtype=np.float64)
            for fold_index, (train_indices, held_indices) in enumerate(
                splitter.split(matrix, labels)
            ):
                estimator = build_estimator(candidate, seed=seed + fold_index)
                estimator.fit(matrix[train_indices], labels[train_indices])
                predictions = np.asarray(estimator.predict(matrix[held_indices]), dtype=np.int64)
                log_scores = estimator_log_scores(estimator, matrix[held_indices])
                if not set(predictions.tolist()).issubset({0, 1, 2}):
                    raise RuntimeError("Candidate emitted an invalid subtype")
                oof[held_indices] = predictions
                oof_log_scores[held_indices] = log_scores
                fold_rows.append(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "fold_index": fold_index,
                        "train_count": int(train_indices.size),
                        "held_out_count": int(held_indices.size),
                        "train_class_counts": _class_counts(labels[train_indices]),
                        "held_out_class_counts": _class_counts(labels[held_indices]),
                        "held_out_macro_f1": float(
                            f1_score(
                                labels[held_indices],
                                predictions,
                                labels=[0, 1, 2],
                                average="macro",
                                zero_division=0,
                            )
                        ),
                    }
                )
            if np.any(oof < 0) or not np.isfinite(oof_log_scores).all():
                raise RuntimeError("Repeated CV left at least one training case unpredicted")
            repeat_macro_f1 = float(
                f1_score(
                    labels,
                    oof,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            )
            repeat_scores.append(repeat_macro_f1)
            repeat_per_class_recalls.append(
                [
                    float(value)
                    for value in recall_score(
                        labels,
                        oof,
                        labels=[0, 1, 2],
                        average=None,
                        zero_division=0,
                    )
                ]
            )
            restored = oof[inverse_order]
            restored_log_scores = oof_log_scores[inverse_order]
            repeat_predictions.append(
                {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "macro_f1": repeat_macro_f1,
                    "predictions": [
                        {
                            "case_id": case_id,
                            "reference": int(reference),
                            "prediction": int(prediction),
                            "log_scores": [float(value) for value in log_scores],
                        }
                        for case_id, reference, prediction, log_scores in zip(
                            dataset.case_ids,
                            labels_original,
                            restored,
                            restored_log_scores,
                            strict=True,
                        )
                    ],
                }
            )
        candidate_results.append(
            {
                "candidate_id": candidate.candidate_id,
                "feature_view": candidate.feature_view,
                "family": candidate.family,
                "parameters": dict(candidate.parameters),
                "repeat_oof_macro_f1": repeat_scores,
                "repeat_oof_per_class_recall": repeat_per_class_recalls,
                "minimum_repeat_per_class_recall": float(
                    np.min(np.asarray(repeat_per_class_recalls, dtype=np.float64))
                ),
                "mean_repeat_oof_macro_f1": float(np.mean(repeat_scores)),
                "std_repeat_oof_macro_f1_population": float(np.std(repeat_scores, ddof=0)),
                "fold_scores": fold_rows,
                "oof_predictions": repeat_predictions,
                "canonical_order_excludes_case_ids": True,
            }
        )

    best_mean = max(row["mean_repeat_oof_macro_f1"] for row in candidate_results)
    tolerance = float(lock["selection"]["tie_tolerance"])
    eligible = [
        row for row in candidate_results if row["mean_repeat_oof_macro_f1"] >= best_mean - tolerance
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["std_repeat_oof_macro_f1_population"],
            row["candidate_id"],
        ),
    )
    return {
        "schema_version": 1,
        "scope": "supplied_training_cases_only",
        "case_count": dataset.case_count,
        "class_counts": _class_counts(labels_original),
        "identifier_independent_dataset_sha256": identifier_independent_dataset_sha256(dataset),
        "case_ids_used_as_model_features": False,
        "paths_or_filenames_used_as_model_features": False,
        "official_validation_images_read": False,
        "official_validation_masks_read": False,
        "official_validation_labels_read": False,
        "candidate_count": len(candidate_results),
        "candidate_results": candidate_results,
        "selection_rule": dict(lock["selection"]),
        "selected_candidate_id": selected["candidate_id"],
        "selected_feature_view": selected["feature_view"],
        "selected_mean_repeat_oof_macro_f1": selected["mean_repeat_oof_macro_f1"],
        "selected_std_repeat_oof_macro_f1_population": selected[
            "std_repeat_oof_macro_f1_population"
        ],
        "selected_minimum_repeat_per_class_recall": selected["minimum_repeat_per_class_recall"],
        "sklearn_version": sklearn.__version__,
    }


def selected_candidate_from_audit(
    lock: Mapping[str, Any],
    selection_audit: Mapping[str, Any],
) -> CandidateSpec:
    candidate_id = str(selection_audit["selected_candidate_id"])
    matches = [
        candidate
        for candidate in enumerate_locked_candidates(lock)
        if candidate.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("Selection audit does not identify one locked candidate")
    return matches[0]


def fit_selected_classifier(
    dataset: CaseFeatureDataset,
    lock: Mapping[str, Any],
    selection_audit: Mapping[str, Any],
) -> tuple[Pipeline, dict[str, Any]]:
    """Refit the selected pipeline on all 252 canonicalized training cases."""

    candidate = selected_candidate_from_audit(lock, selection_audit)
    matrix_original = np.asarray(dataset.views[candidate.feature_view], dtype=np.float32)
    labels_original = np.asarray(dataset.labels, dtype=np.int64)
    order = canonical_content_order(matrix_original, labels_original)
    seed = int(lock["final_fit"]["random_seed"])
    estimator = build_estimator(candidate, seed=seed)
    estimator.fit(matrix_original[order], labels_original[order])
    fitted_predictions = np.asarray(estimator.predict(matrix_original), dtype=np.int64)
    metadata = {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "feature_view": candidate.feature_view,
        "family": candidate.family,
        "parameters": dict(candidate.parameters),
        "training_case_count": dataset.case_count,
        "training_class_counts": _class_counts(labels_original),
        "training_macro_f1_resubstitution_not_for_selection": float(
            f1_score(
                labels_original,
                fitted_predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "identifier_independent_dataset_sha256": identifier_independent_dataset_sha256(dataset),
        "feature_count": int(matrix_original.shape[1]),
        "feature_names": list(dataset.feature_names[candidate.feature_view]),
        "seed": seed,
        "case_ids_used_as_model_features": False,
        "canonical_content_order_used": True,
        "official_validation_accessed": False,
    }
    return estimator, metadata


def save_classifier_bundle(
    destination: str | Path,
    estimator: Pipeline,
    metadata: Mapping[str, Any],
) -> Path:
    """Serialize the fitted classical head and its identifier-free schema."""

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary classifier artifact already exists: {temporary}")
    try:
        joblib.dump({"estimator": estimator, "metadata": dict(metadata)}, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "CandidateSpec",
    "CaseFeatureDataset",
    "build_estimator",
    "canonical_content_order",
    "enumerate_locked_candidates",
    "estimator_log_scores",
    "evaluate_locked_candidates",
    "fit_selected_classifier",
    "identifier_independent_dataset_sha256",
    "load_locked_search",
    "save_classifier_bundle",
    "selected_candidate_from_audit",
]
