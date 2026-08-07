#!/usr/bin/env python3
"""Measure which encoder stage carries subtype signal, and forecast the 36-case score.

Train-only: reads the 252-case whole-volume feature cache and never opens validation
or test data.  Emits a stage-wise nested-CV table plus, for the best block, the
predictive distribution of the score a 36-case validation set would return.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.shallow_probe import (  # noqa: E402
    VALIDATION_CLASS_COUNTS,
    best_of_n_inflation,
    nested_cv_macro_f1,
    stage_pool,
    validation_subsample_distribution,
)

EXPECTED_CASES = 252
EXPECTED_CLASS_COUNTS = [62, 106, 84]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        spatial = np.asarray(payload["spatial"])
        labels = np.asarray(payload["labels"]).astype(np.int64)
        extra = {
            name: np.asarray(payload[name])
            for name in ("stage5_global_mean", "rescue_mean_logits", "morphology")
            if name in payload
        }
    if labels.shape != (EXPECTED_CASES,):
        raise ValueError(f"Expected {EXPECTED_CASES} training cases, found {labels.shape}")
    if np.bincount(labels, minlength=3).tolist() != EXPECTED_CLASS_COUNTS:
        raise ValueError("Training class counts differ from the supplied 62/106/84 split")
    return spatial, labels, extra


def main() -> int:
    args = build_parser().parse_args()
    spatial, labels, extra = _load(args.features.expanduser().resolve())

    blocks: dict[str, np.ndarray] = {
        "stage2_gap": stage_pool(spatial, "stage2"),
        "stage3_gap": stage_pool(spatial, "stage3"),
        "stage2_plus_stage3_gap": np.concatenate(
            [stage_pool(spatial, "stage2"), stage_pool(spatial, "stage3")], axis=1
        ),
    }
    for name, key in (
        ("stage5_bottleneck", "stage5_global_mean"),
        ("rescue_head_logits", "rescue_mean_logits"),
    ):
        if key in extra:
            blocks[name] = extra[key].astype(np.float32)

    results: dict[str, Any] = {}
    probes = {}
    print(f"{'feature block':<28} {'dims':>6} {'macro-F1':>10} {'sd':>8}")
    print("-" * 56)
    for name, matrix in blocks.items():
        probe = nested_cv_macro_f1(matrix, labels, seeds=args.seeds)
        probes[name] = probe
        results[name] = {"dims": int(matrix.shape[1]), **probe.as_dict()}
        print(f"{name:<28} {matrix.shape[1]:>6} {probe.mean_macro_f1:>10.4f} {probe.std_macro_f1:>8.4f}")

    best_name = max(probes, key=lambda key: probes[key].mean_macro_f1)
    best = probes[best_name]
    print(f"\nbest block: {best_name} ({best.mean_macro_f1:.4f})")

    forecast = validation_subsample_distribution(
        best.oof_probabilities,
        best.labels,
        class_counts=VALIDATION_CLASS_COUNTS,
        repeats=args.repeats,
    )
    standard_error = forecast["std"]
    forecast["best_of_n_inflation"] = {
        str(n): best_of_n_inflation(standard_error, n) for n in (1, 2, 4, 10, 30)
    }
    results["validation_forecast"] = {"block": best_name, **forecast}

    print(f"\n36-case validation forecast for {best_name}:")
    print(f"  full-set OOF macro-F1 : {forecast['full_set_macro_f1']:.4f}")
    print(f"  mean over 36-case draws: {forecast['mean']:.4f}  (sd {standard_error:.4f})")
    print(f"  95% interval          : [{forecast['percentiles']['2.5']:.4f}, "
          f"{forecast['percentiles']['97.5']:.4f}]")
    for threshold, probability in forecast["probability_at_or_above"].items():
        print(f"  P(macro-F1 >= {threshold})   : {probability:.3f}")
    print("\n  optimism from reporting the best of N validation looks:")
    for looks, inflation in forecast["best_of_n_inflation"].items():
        print(f"    N={looks:<3} +{inflation:.4f}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
