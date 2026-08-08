"""Train-only shallow-tap probing and honest validation-size forecasting.

Two things live here, both deliberately train-only:

``stage_pool`` / ``nested_cv_macro_f1``
    Measure how much three-class subtype signal each encoder stage carries, using
    nested cross-validation so the regularization strength is chosen *inside* each
    training fold and never sees the scored rows.

``validation_subsample_distribution``
    Turn out-of-fold probabilities over the 252 training cases into the predictive
    distribution of the score a 36-case validation set would produce.  A single OOF
    point estimate hides the fact that 36 cases is a very small ruler: roughly four
    cases separate 0.60 from 0.70.  This function reports that spread directly so a
    threshold decision can be made on the distribution rather than on one number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASS_COUNT: Final = 3
CLASS_LABELS: Final = (0, 1, 2)

#: Channel layout of the V6 whole-volume ``spatial`` cache, as written by
#: :mod:`pancreas_multitask.v6_case_extractor`.
STAGE_CHANNEL_SLICES: Final[dict[str, tuple[int, int]]] = {
    "stage2": (0, 128),
    "stage3": (128, 384),
    "lesion_probability": (384, 385),
    "whole_probability": (385, 386),
    "preprocessed_ct": (386, 387),
}

#: Supplied validation split composition (subtype 0 / 1 / 2).
VALIDATION_CLASS_COUNTS: Final = (9, 15, 12)

DEFAULT_C_GRID: Final = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def stage_pool(spatial: np.ndarray, stage: str) -> np.ndarray:
    """Global-average-pool one named channel block of the whole-volume cache.

    Plain global mean is deliberate.  Richer alternatives were measured on the
    252 training cases and every one of them lost to this: GAP+STD 0.573,
    lesion-probability-weighted 0.476, +morphology 0.551, and an 8x8x12 -> 2x2x3
    spatial grid 0.606, against 0.608 for plain GAP.  The subtype signal behaves
    as a global parenchymal property, not a lesion-local one.
    """

    if stage not in STAGE_CHANNEL_SLICES:
        raise ValueError(f"Unknown stage block: {stage}")
    values = np.asarray(spatial)
    if values.ndim != 5:
        raise ValueError("Whole-volume spatial cache must have shape (cases, C, D, H, W)")
    start, stop = STAGE_CHANNEL_SLICES[stage]
    if values.shape[1] < stop:
        raise ValueError(f"Cache has {values.shape[1]} channels; {stage} needs at least {stop}")
    return values[:, start:stop].astype(np.float32).mean(axis=(2, 3, 4))


def _linear_probe(c_grid: Sequence[float], inner_seed: int) -> GridSearchCV:
    return GridSearchCV(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ),
        {"logisticregression__C": list(c_grid)},
        scoring="f1_macro",
        cv=StratifiedKFold(5, shuffle=True, random_state=inner_seed),
        n_jobs=-1,
    )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Nested-CV summary for one feature block."""

    mean_macro_f1: float
    std_macro_f1: float
    per_seed_macro_f1: tuple[float, ...]
    oof_probabilities: np.ndarray
    labels: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "per_seed_macro_f1": list(self.per_seed_macro_f1),
        }


def nested_cv_macro_f1(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    folds: int = 5,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> ProbeResult:
    """Nested-CV macro-F1 with the regularization strength chosen inside each fold.

    Selecting ``C`` on the same out-of-fold predictions that are then reported
    inflates the estimate.  Here the inner ``GridSearchCV`` only ever sees the
    outer training rows, so the returned number is honest.
    """

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Probe requires a 2-D feature matrix aligned with 1-D labels")
    if set(np.unique(y).tolist()) != set(CLASS_LABELS):
        raise ValueError("Probe requires all three subtype classes to be present")

    scores: list[float] = []
    probability_sum = np.zeros((y.size, CLASS_COUNT), dtype=np.float64)
    for seed in seeds:
        estimator = _linear_probe(c_grid, inner_seed=100 + int(seed))
        outer = StratifiedKFold(folds, shuffle=True, random_state=int(seed))
        probabilities = cross_val_predict(
            estimator, x, y, cv=outer, method="predict_proba", n_jobs=1
        )
        probability_sum += probabilities
        scores.append(
            float(
                f1_score(
                    y,
                    probabilities.argmax(axis=1),
                    average="macro",
                    labels=list(CLASS_LABELS),
                    zero_division=0,
                )
            )
        )
    return ProbeResult(
        mean_macro_f1=float(np.mean(scores)),
        std_macro_f1=float(np.std(scores)),
        per_seed_macro_f1=tuple(scores),
        oof_probabilities=(probability_sum / len(tuple(seeds))).astype(np.float64),
        labels=y,
    )


def validation_subsample_distribution(
    oof_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    class_counts: Sequence[int] = VALIDATION_CLASS_COUNTS,
    repeats: int = 20_000,
    seed: int = 12345,
    thresholds: Sequence[float] = (0.60, 0.65, 0.70),
) -> dict[str, Any]:
    """Forecast the score a validation set of this size and composition would give.

    Draws ``class_counts`` cases per class without replacement from the out-of-fold
    predictions and recomputes macro-F1, ``repeats`` times.  The resulting spread is
    the sampling variability contributed by evaluating on only ``sum(class_counts)``
    cases, holding model quality fixed.

    This is a *lower bound* on the true uncertainty: it captures evaluation-set size
    but not train/validation distribution shift, so treat ``probability_at_or_above``
    as optimistic rather than as a calibrated forecast.
    """

    probabilities = np.asarray(oof_probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != CLASS_COUNT:
        raise ValueError("Out-of-fold probabilities must have shape (cases, 3)")
    if probabilities.shape[0] != y.size:
        raise ValueError("Probabilities and labels are not aligned")
    if not np.isfinite(probabilities).all():
        raise ValueError("Out-of-fold probabilities must be finite")
    requested = [int(value) for value in class_counts]
    if len(requested) != CLASS_COUNT or any(value < 1 for value in requested):
        raise ValueError("Validation composition must request at least one case per class")

    predictions = probabilities.argmax(axis=1)
    index_by_class = [np.flatnonzero(y == label) for label in CLASS_LABELS]
    for label, (available, needed) in enumerate(zip(index_by_class, requested)):
        if available.size < needed:
            raise ValueError(
                f"Class {label} has {available.size} cases; cannot draw {needed} without replacement"
            )

    generator = np.random.default_rng(seed)
    scores = np.empty(int(repeats), dtype=np.float64)
    for draw in range(int(repeats)):
        chosen = np.concatenate(
            [
                generator.choice(available, size=needed, replace=False)
                for available, needed in zip(index_by_class, requested)
            ]
        )
        scores[draw] = f1_score(
            y[chosen],
            predictions[chosen],
            average="macro",
            labels=list(CLASS_LABELS),
            zero_division=0,
        )
    return {
        "full_set_macro_f1": float(
            f1_score(y, predictions, average="macro", labels=list(CLASS_LABELS), zero_division=0)
        ),
        "validation_case_count": int(sum(requested)),
        "validation_class_counts": requested,
        "repeats": int(repeats),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "percentiles": {
            str(p): float(np.percentile(scores, p)) for p in (2.5, 5, 25, 50, 75, 95, 97.5)
        },
        "probability_at_or_above": {
            f"{threshold:.2f}": float(np.mean(scores >= threshold)) for threshold in thresholds
        },
    }


def best_of_n_inflation(standard_error: float, looks: int, *, repeats: int = 200_000,
                        seed: int = 12345) -> float:
    """Expected optimism from reporting the best of ``looks`` noisy evaluations.

    If a candidate is chosen by taking the maximum over ``looks`` independent
    validation evaluations, the reported score exceeds the true score by roughly
    ``standard_error * E[max of `looks` standard normals]``.  Quantifying this is
    what keeps a selection artefact from being read as a real improvement.
    """

    if standard_error < 0 or looks < 1:
        raise ValueError("Standard error must be non-negative and looks must be positive")
    if looks == 1:
        return 0.0
    generator = np.random.default_rng(seed)
    draws = generator.standard_normal(size=(int(repeats), int(looks)))
    return float(standard_error * draws.max(axis=1).mean())


__all__ = [
    "CLASS_LABELS",
    "DEFAULT_C_GRID",
    "STAGE_CHANNEL_SLICES",
    "VALIDATION_CLASS_COUNTS",
    "ProbeResult",
    "best_of_n_inflation",
    "nested_cv_macro_f1",
    "stage_pool",
    "validation_subsample_distribution",
]
