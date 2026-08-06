#!/usr/bin/env python3
"""Select a checkpoint from fixed-validation evaluator reports.

The selection rule is deliberately narrow: maximize the equal-weight mean of
whole-pancreas Dice, lesion Dice, and subtype macro-F1.  Candidate-name order
is the only tie-breaker, so no unreported secondary metric affects selection.

Example::

    python scripts/select_checkpoint.py \
      --candidate checkpoint_best=artifacts/validation/best/metrics.json \
      --candidate checkpoint_best_multitask=artifacts/validation/multitask/metrics.json \
      --candidate checkpoint_final=artifacts/validation/final/metrics.json \
      --checkpoint checkpoint_best=D:/models/fold_0/checkpoint_best.pth \
      --checkpoint checkpoint_best_multitask=D:/models/fold_0/checkpoint_best_multitask.pth \
      --checkpoint checkpoint_final=D:/models/fold_0/checkpoint_final.pth \
      --output artifacts/checkpoint_selection.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

METRIC_PATHS = (
    ("whole_pancreas_dice", ("segmentation", "whole_pancreas_dice", "mean")),
    ("lesion_dice", ("segmentation", "lesion_dice", "mean")),
    ("macro_f1", ("classification", "macro_f1")),
)
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
EXPECTED_VALIDATION_CASES = 36


class SelectionError(RuntimeError):
    """Raised when checkpoint selection inputs are incomplete or invalid."""


def _parse_mapping_specs(specs: Sequence[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs:
        name, separator, value = spec.partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not name or not value:
            raise SelectionError(f"{option} values must have the form NAME=VALUE: {spec!r}")
        if name in parsed:
            raise SelectionError(f"Duplicate {option} mapping for candidate {name!r}")
        parsed[name] = value
    return parsed


def _load_evaluator_json(path: Path) -> Mapping[str, Any]:
    source = path.resolve()
    if not source.is_file():
        raise SelectionError(f"Evaluator JSON does not exist or is not a file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"Could not read evaluator JSON {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SelectionError(f"Evaluator JSON must contain an object at its root: {source}")
    return payload


def _required_metric(payload: Mapping[str, Any], path: Sequence[str], *, source: Path) -> float:
    dotted_path = ".".join(path)
    value: Any = payload
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise SelectionError(f"Missing required metric {dotted_path} in {source}")
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(
            f"Metric {dotted_path} in {source} must be a JSON number, got {value!r}"
        )
    try:
        metric = float(value)
    except (OverflowError, ValueError) as exc:
        raise SelectionError(f"Metric {dotted_path} in {source} is not finite") from exc
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise SelectionError(
            f"Metric {dotted_path} in {source} must be finite and in [0, 1], got {value!r}"
        )
    return metric


def _required_integer(
    payload: Mapping[str, Any],
    path: Sequence[str],
    *,
    source: Path,
    expected: int,
) -> int:
    dotted_path = ".".join(path)
    value: Any = payload
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise SelectionError(f"Missing required field {dotted_path} in {source}")
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionError(f"Field {dotted_path} in {source} must be an integer")
    if value != expected:
        raise SelectionError(
            f"Field {dotted_path} in {source} must equal {expected}, got {value}"
        )
    return value


def _validate_complete_evaluation(payload: Mapping[str, Any], *, source: Path) -> None:
    _required_integer(payload, ("schema_version",), source=source, expected=1)
    _required_integer(
        payload,
        ("case_count",),
        source=source,
        expected=EXPECTED_VALIDATION_CASES,
    )
    _required_integer(
        payload,
        ("segmentation", "case_count"),
        source=source,
        expected=EXPECTED_VALIDATION_CASES,
    )
    _required_integer(
        payload,
        ("classification", "case_count"),
        source=source,
        expected=EXPECTED_VALIDATION_CASES,
    )
    _required_integer(
        payload,
        ("classification", "unused_reference_case_count"),
        source=source,
        expected=0,
    )


def _sha256(path: Path) -> str:
    checkpoint = path.resolve()
    if not checkpoint.is_file():
        raise SelectionError(f"Checkpoint does not exist or is not a file: {checkpoint}")
    digest = hashlib.sha256()
    try:
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SelectionError(f"Could not hash checkpoint {checkpoint}: {exc}") from exc
    return digest.hexdigest()


def _normalise_sha256(value: str, *, candidate: str) -> str:
    digest = value.strip().lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise SelectionError(
            f"Checkpoint SHA-256 for candidate {candidate!r} must contain exactly 64 hex digits"
        )
    return digest


def build_selection_artifact(
    candidates: Mapping[str, Path],
    *,
    checkpoint_paths: Mapping[str, Path] | None = None,
    checkpoint_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate, score, and deterministically rank evaluator outputs."""

    if len(candidates) < 2:
        raise SelectionError("At least two checkpoint candidates are required")
    checkpoint_paths = checkpoint_paths or {}
    checkpoint_sha256 = checkpoint_sha256 or {}
    candidate_names = set(candidates)
    unknown_mappings = (set(checkpoint_paths) | set(checkpoint_sha256)) - candidate_names
    if unknown_mappings:
        raise SelectionError(
            "Checkpoint mapping supplied for unknown candidate(s): "
            + ", ".join(sorted(unknown_mappings))
        )

    resolved_sources: dict[str, Path] = {
        name: Path(path).resolve() for name, path in candidates.items()
    }
    if len(set(resolved_sources.values())) != len(resolved_sources):
        raise SelectionError("Each candidate must use a distinct evaluator JSON file")

    ranking: list[dict[str, Any]] = []
    for name in sorted(candidates):
        source = resolved_sources[name]
        payload = _load_evaluator_json(source)
        _validate_complete_evaluation(payload, source=source)
        metrics = {
            output_name: _required_metric(payload, metric_path, source=source)
            for output_name, metric_path in METRIC_PATHS
        }
        entry: dict[str, Any] = {
            "candidate": name,
            "metrics_source": str(source),
            "metrics": metrics,
            "selection_score": math.fsum(metrics.values()) / len(METRIC_PATHS),
        }

        supplied_digest: str | None = None
        if name in checkpoint_sha256:
            supplied_digest = _normalise_sha256(checkpoint_sha256[name], candidate=name)
        if name in checkpoint_paths:
            checkpoint_path = Path(checkpoint_paths[name]).resolve()
            discovered_digest = _sha256(checkpoint_path)
            if supplied_digest is not None and supplied_digest != discovered_digest:
                raise SelectionError(
                    f"Supplied SHA-256 does not match checkpoint for candidate {name!r}: "
                    f"{checkpoint_path}"
                )
            entry["checkpoint_path"] = str(checkpoint_path)
            entry["checkpoint_sha256"] = discovered_digest
        elif supplied_digest is not None:
            entry["checkpoint_sha256"] = supplied_digest
        ranking.append(entry)

    ranking.sort(key=lambda item: (-item["selection_score"], item["candidate"]))
    for index, entry in enumerate(ranking, start=1):
        entry["rank"] = index

    selected = ranking[0]
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "selection_policy": {
            "direction": "maximize",
            "metric_paths": [".".join(path) for _, path in METRIC_PATHS],
            "metric_weights": {name: 1.0 / len(METRIC_PATHS) for name, _ in METRIC_PATHS},
            "score": "equal-weight arithmetic mean",
            "tie_breaker": "candidate name ascending; no secondary metric",
        },
        "candidate_count": len(ranking),
        "selected_candidate": selected["candidate"],
        "selected_score": selected["selection_score"],
        "ranking": ranking,
    }
    if "checkpoint_path" in selected:
        artifact["selected_checkpoint_path"] = selected["checkpoint_path"]
    if "checkpoint_sha256" in selected:
        artifact["selected_checkpoint_sha256"] = selected["checkpoint_sha256"]
    return artifact


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=METRICS.json",
        help="Candidate name and evaluator JSON path; repeat at least twice",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT.pth",
        help="Optional candidate checkpoint path to hash; repeat as needed",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        action="append",
        default=[],
        metavar="NAME=HEX",
        help="Optional precomputed digest; checked against --checkpoint when both are given",
    )
    parser.add_argument("--output", required=True, type=Path, help="Selection artifact JSON path")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_specs = _parse_mapping_specs(args.candidate, option="--candidate")
    checkpoint_specs = _parse_mapping_specs(args.checkpoint, option="--checkpoint")
    digest_specs = _parse_mapping_specs(args.checkpoint_sha256, option="--checkpoint-sha256")
    candidates = {name: Path(path) for name, path in candidate_specs.items()}
    checkpoint_paths = {name: Path(path) for name, path in checkpoint_specs.items()}

    output = args.output.resolve()
    protected_paths = {path.resolve() for path in candidates.values()} | {
        path.resolve() for path in checkpoint_paths.values()
    }
    if output in protected_paths:
        raise SelectionError("--output must not overwrite an evaluator JSON or checkpoint")

    artifact = build_selection_artifact(
        candidates,
        checkpoint_paths=checkpoint_paths,
        checkpoint_sha256=digest_specs,
    )
    _atomic_write_json(output, artifact)
    print(
        f"Selected {artifact['selected_candidate']} "
        f"(score={artifact['selected_score']:.6f}); wrote {output}"
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (SelectionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
