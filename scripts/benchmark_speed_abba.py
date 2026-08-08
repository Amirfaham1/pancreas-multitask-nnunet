#!/usr/bin/env python3
"""ABBA wall-clock benchmark: our joint pipeline against stock nnU-Net.

Protocol, fixed before running:

* Each arm is a **fresh child process**, timed with an external monotonic clock, so
  interpreter start-up, CUDA context creation and model loading all count. That is
  the honest scope for a claim about "inference speed", and it is the scope the
  original gate used.
* **ABBA ordering** (stock, ours, ours, stock, ...) so any thermal or contention
  drift is shared between arms rather than favouring whichever ran first.
* The **mean** is the decision statistic, declared in advance; per-run times and
  spread are reported alongside so a near-threshold result cannot hide behind a
  lucky minimum.
* Stock runs at its shipped defaults (`-npp 3 -nps 3`, step 0.5, mirroring on).
  Our arm uses the same worker counts, so the comparison is not simply bought with
  extra CPU processes.
* TTA is **not** disabled and the sliding-window step size is **not** increased in
  either arm -- the assignment rules those out explicitly.

The harness does not assume that execution-path changes are numerically inert.
It compares every mask from every repeat, checks within-arm stability, geometry and
dtype, and applies the same declared voxel-disagreement bound to repeat and cross-arm
comparisons.
The candidate timing includes the fitted classifier and its subtype CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--classifier-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--order", default="SCCSCS",
                        help="S=stock, C=candidate; default is ABBA-balanced")
    parser.add_argument(
        "--max-disagreement-fraction",
        type=float,
        default=1e-5,
        help="Maximum cross-arm mask disagreement (default 0.001%% of voxels)",
    )
    parser.add_argument("--min-whole-agreement-dice", type=float, default=0.9999)
    parser.add_argument("--min-lesion-agreement-dice", type=float, default=0.999)
    parser.add_argument("--min-lesion-case-agreement-dice", type=float, default=0.99)
    return parser


def _dice(first, second) -> float:
    import numpy as np

    denominator = int(first.sum()) + int(second.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(first, second).sum() / denominator)


def compare_mask_directories(
    first: Path,
    second: Path,
    expected_case_ids: list[str],
) -> dict:
    """Compare two complete prediction directories case by case."""

    import numpy as np
    import SimpleITK as sitk

    expected = set(expected_case_ids)
    first_files = {path.name[:-7]: path for path in first.glob("*.nii.gz")}
    second_files = {path.name[:-7]: path for path in second.glob("*.nii.gz")}
    if set(first_files) != expected or set(second_files) != expected:
        raise RuntimeError(
            "Mask comparison requires exact input membership: "
            f"first={len(first_files)}, second={len(second_files)}, expected={len(expected)}"
        )

    disagreeing = 0
    total = 0
    whole_dice: list[float] = []
    lesion_dice: list[float] = []
    geometry_matches = True
    dtype_matches = True
    for case_id in sorted(expected):
        first_image = sitk.ReadImage(str(first_files[case_id]))
        second_image = sitk.ReadImage(str(second_files[case_id]))
        first_array = sitk.GetArrayFromImage(first_image)
        second_array = sitk.GetArrayFromImage(second_image)
        geometry_matches &= (
            first_array.shape == second_array.shape
            and np.allclose(first_image.GetSpacing(), second_image.GetSpacing(), atol=1e-5)
            and np.allclose(first_image.GetOrigin(), second_image.GetOrigin(), atol=1e-5)
            and np.allclose(first_image.GetDirection(), second_image.GetDirection(), atol=1e-5)
        )
        dtype_matches &= first_array.dtype == second_array.dtype
        if first_array.shape != second_array.shape:
            continue
        disagreeing += int(np.count_nonzero(first_array != second_array))
        total += int(first_array.size)
        whole_dice.append(_dice(first_array > 0, second_array > 0))
        lesion_dice.append(_dice(first_array == 2, second_array == 2))

    return {
        "cases": len(expected),
        "total_voxels": total,
        "disagreeing_voxels": disagreeing,
        "disagreement_fraction": float(disagreeing / total) if total else 1.0,
        "exact": bool(disagreeing == 0 and geometry_matches and dtype_matches),
        "geometry_matches": bool(geometry_matches),
        "dtype_matches": bool(dtype_matches),
        "whole_pancreas_agreement_dice_mean": float(statistics.fmean(whole_dice)),
        "whole_pancreas_agreement_dice_min": float(min(whole_dice)),
        "lesion_agreement_dice_mean": float(statistics.fmean(lesion_dice)),
        "lesion_agreement_dice_min": float(min(lesion_dice)),
    }


def validate_subtype_csv(path: Path, expected_case_ids: list[str]) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_names = {f"{case_id}.nii.gz" for case_id in expected_case_ids}
    observed_names = {row.get("Names", "") for row in rows}
    labels_valid = all(row.get("Subtype") in {"0", "1", "2"} for row in rows)
    return {
        "rows": len(rows),
        "membership_matches": observed_names == expected_names,
        "labels_valid": labels_valid,
        "valid": len(rows) == len(expected_names) and observed_names == expected_names and labels_valid,
    }


def read_subtype_csv(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["Names"]: int(row["Subtype"]) for row in csv.DictReader(stream)}


def run_child(command: list[str], log: Path) -> float:
    """Time one complete child process on an external monotonic clock."""

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        started = time.monotonic()
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"arm failed (exit {completed.returncode}); see {log}")
    return elapsed


def main() -> int:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Benchmark output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Deliberately NOT .resolve(): a venv's bin/python is a symlink to the base
    # interpreter, and resolving it launches the base interpreter, which has none of
    # the venv's site-packages ("No module named numpy").
    python = str(args.python.expanduser())
    model = str(args.model.expanduser().resolve())
    classifier = str(args.classifier.expanduser().resolve())
    source = str(args.input.expanduser().resolve())

    order = args.order[: 2 * args.repeats]
    if len(order) != 2 * args.repeats or set(order) - {"S", "C"}:
        raise ValueError("--order must provide exactly 2*repeats entries from {S,C}")
    if order.count("S") != args.repeats or order.count("C") != args.repeats:
        raise ValueError("--order must contain exactly one stock and candidate run per repeat")
    expected_case_ids = sorted(path.name[:-len("_0000.nii.gz")] for path in Path(source).glob("*_0000.nii.gz"))
    if not expected_case_ids:
        raise ValueError("No test images found")

    runs: list[dict] = []
    for index, arm in enumerate(order):
        label = "stock" if arm == "S" else "candidate"
        target = output / f"{label}_{index}"
        command = [
            python, str(ROOT / "scripts" / "run_inference_arm.py"),
            "--arm", label, "--input", source, "--model", model,
            "--output", str(target),
        ]
        if label == "candidate":
            command.extend(["--classifier", classifier])
            if args.classifier_sha256:
                command.extend(["--classifier-sha256", args.classifier_sha256])
        seconds = run_child(command, output / f"{label}_{index}.log")
        masks = list(target.glob("*.nii.gz"))
        if len(masks) != len(expected_case_ids):
            raise RuntimeError(
                f"{label} run {index} wrote {len(masks)} masks; expected {len(expected_case_ids)}"
            )
        if label == "candidate" and not (target / "subtype_results.csv").is_file():
            raise RuntimeError(f"candidate run {index} did not write subtype_results.csv")
        runs.append({"index": index, "arm": label, "seconds": seconds,
                     "output": str(target)})
        print(f"  [{index}] {label:<10} {seconds:8.2f} s", flush=True)

    stock = [r["seconds"] for r in runs if r["arm"] == "stock"]
    candidate = [r["seconds"] for r in runs if r["arm"] == "candidate"]
    stock_mean, candidate_mean = statistics.fmean(stock), statistics.fmean(candidate)
    reduction = 100.0 * (stock_mean - candidate_mean) / stock_mean
    stock_directories = [Path(r["output"]) for r in runs if r["arm"] == "stock"]
    candidate_directories = [Path(r["output"]) for r in runs if r["arm"] == "candidate"]
    stock_repeat = [
        compare_mask_directories(stock_directories[0], path, expected_case_ids)
        for path in stock_directories[1:]
    ]
    candidate_repeat = [
        compare_mask_directories(candidate_directories[0], path, expected_case_ids)
        for path in candidate_directories[1:]
    ]
    cross_arm = compare_mask_directories(
        stock_directories[0], candidate_directories[0], expected_case_ids
    )
    subtype_audits = [
        validate_subtype_csv(path / "subtype_results.csv", expected_case_ids)
        for path in candidate_directories
    ]
    reference_subtypes = read_subtype_csv(
        candidate_directories[0] / "subtype_results.csv"
    )
    subtype_repeat_exact = all(
        read_subtype_csv(path / "subtype_results.csv") == reference_subtypes
        for path in candidate_directories[1:]
    )
    def within_declared_tolerance(item: dict) -> bool:
        return bool(
            item["geometry_matches"]
            and item["dtype_matches"]
            and item["disagreement_fraction"] <= args.max_disagreement_fraction
            and item["whole_pancreas_agreement_dice_mean"]
            >= args.min_whole_agreement_dice
            and item["lesion_agreement_dice_mean"]
            >= args.min_lesion_agreement_dice
            and item["lesion_agreement_dice_min"]
            >= args.min_lesion_case_agreement_dice
        )

    repeat_stable = all(
        within_declared_tolerance(item) for item in stock_repeat + candidate_repeat
    )
    cross_arm_equivalent = within_declared_tolerance(cross_arm)
    equivalence_passed = bool(
        repeat_stable
        and cross_arm_equivalent
        and all(item["valid"] for item in subtype_audits)
        and subtype_repeat_exact
    )
    speed_passed = bool(reduction >= 10.0)
    summary = {
        "runs": runs,
        "order": order,
        "stock_seconds": stock, "candidate_seconds": candidate,
        "stock_mean": stock_mean, "candidate_mean": candidate_mean,
        "stock_spread_pct": 100 * (max(stock) - min(stock)) / stock_mean,
        "candidate_spread_pct": 100 * (max(candidate) - min(candidate)) / candidate_mean,
        "runtime_reduction_percent": reduction,
        "meets_10_percent_speed_gate": speed_passed,
        "equivalence": {
            "max_disagreement_fraction": args.max_disagreement_fraction,
            "min_whole_pancreas_agreement_dice": args.min_whole_agreement_dice,
            "min_lesion_agreement_dice": args.min_lesion_agreement_dice,
            "min_lesion_case_agreement_dice": args.min_lesion_case_agreement_dice,
            "stock_repeat_comparisons": stock_repeat,
            "candidate_repeat_comparisons": candidate_repeat,
            "cross_arm": cross_arm,
            "subtype_csv_audits": subtype_audits,
            "subtype_repeat_exact": subtype_repeat_exact,
            "repeat_bit_exact": all(item["exact"] for item in stock_repeat + candidate_repeat),
            "repeat_stable_within_declared_tolerance": repeat_stable,
            "cross_arm_equivalent": cross_arm_equivalent,
            "passed": equivalence_passed,
        },
        "meets_10_percent_gate": bool(speed_passed and equivalence_passed),
        "decision_statistic": "arithmetic mean of full-process wall clock, declared in advance",
    }
    (output / "speed_benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nstock     mean {stock_mean:8.2f} s  (spread {summary['stock_spread_pct']:.1f}%)")
    print(f"candidate mean {candidate_mean:8.2f} s  (spread {summary['candidate_spread_pct']:.1f}%)")
    print(f"runtime reduction {reduction:+.2f}%  -> "
          f"{'MEETS' if speed_passed else 'MISSES'} the >=10% speed requirement")
    print(
        f"output agreement: {cross_arm['disagreeing_voxels']} / "
        f"{cross_arm['total_voxels']} voxels differ; "
        f"{'PASS' if equivalence_passed else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
