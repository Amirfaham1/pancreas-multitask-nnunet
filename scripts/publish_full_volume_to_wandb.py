#!/usr/bin/env python3
"""Publish a sanitized full-volume summary to one completed W&B run.

The command accepts no credential argument and never resumes a run, appends
history, finishes a run, or uploads an artifact. ``--dry-run`` performs all
local validation and prints the exact credential-free summary mutation without
importing W&B or making a network request.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

EXPECTED_CASE_COUNT = 36
EXPECTED_HISTORY_STEP = 199
SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")

METRIC_PATHS = {
    "whole_pancreas_dice": ("segmentation", "whole_pancreas_dice", "mean"),
    "lesion_dice": ("segmentation", "lesion_dice", "mean"),
    "macro_f1": ("classification", "macro_f1"),
}
SELECTION_METRIC_PATHS = [
    "segmentation.whole_pancreas_dice.mean",
    "segmentation.lesion_dice.mean",
    "classification.macro_f1",
]
BASE_CANDIDATES = frozenset({"checkpoint_best", "checkpoint_best_multitask", "checkpoint_final"})
RESCUE_CANDIDATE = "checkpoint_classification_rescue"
CASE_FIELDS = (
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
)


class PublishError(RuntimeError):
    """Raised when local evidence or the remote W&B target is unsafe."""


def _resolve_input(path: Path, *, description: str, suffix: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise PublishError(f"{description} does not exist or is not a file: {resolved}")
    if resolved.suffix.casefold() != suffix:
        raise PublishError(f"{description} must end in {suffix}: {resolved}")
    return resolved


def _load_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublishError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PublishError(f"{description} must contain a JSON object: {path}")
    return payload


def _value_at(payload: Mapping[str, Any], path: Sequence[str], *, source: Path) -> Any:
    value: Any = payload
    dotted = ".".join(path)
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise PublishError(f"Missing required field {dotted} in {source}")
        value = value[component]
    return value


def _required_integer_value(
    value: Any,
    *,
    field: str,
    source: Path | str,
    expected: int | None = None,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublishError(f"Field {field} in {source} must be an integer")
    if expected is not None and value != expected:
        raise PublishError(f"Field {field} in {source} must equal {expected}, got {value}")
    if minimum is not None and value < minimum:
        raise PublishError(f"Field {field} in {source} must be >= {minimum}, got {value}")
    return value


def _required_integer(
    payload: Mapping[str, Any],
    path: Sequence[str],
    *,
    source: Path,
    expected: int | None = None,
    minimum: int | None = None,
) -> int:
    return _required_integer_value(
        _value_at(payload, path, source=source),
        field=".".join(path),
        source=source,
        expected=expected,
        minimum=minimum,
    )


def _unit_number(value: Any, *, field: str, source: Path | str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublishError(f"Field {field} in {source} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise PublishError(f"Field {field} in {source} must be finite and in [0, 1]")
    return number


def _required_metric(payload: Mapping[str, Any], path: Sequence[str], *, source: Path) -> float:
    return _unit_number(
        _value_at(payload, path, source=source),
        field=".".join(path),
        source=source,
    )


def _require_exact(value: Any, expected: Any, *, field: str, source: Path) -> None:
    if value != expected:
        raise PublishError(f"Field {field} in {source} must equal {expected!r}, got {value!r}")


def _validate_evaluation_policy(payload: Mapping[str, Any], *, source: Path) -> None:
    policy = _value_at(payload, ("evaluation_policy",), source=source)
    if not isinstance(policy, Mapping):
        raise PublishError(f"Field evaluation_policy in {source} must be an object")
    expected = {
        "whole_pancreas": "label > 0",
        "lesion": "label == 2",
        "empty_empty_dice": 1.0,
        "one_sided_empty_dice": 0.0,
        "classification_labels": [0, 1, 2],
        "classification_zero_division": 0.0,
        "confusion_matrix_rows": "reference",
        "confusion_matrix_columns": "prediction",
        "aggregation": "unweighted case mean",
    }
    for field, expected_value in expected.items():
        _require_exact(
            policy.get(field), expected_value, field=f"evaluation_policy.{field}", source=source
        )
    _required_integer_value(
        policy.get("bootstrap_seed"),
        field="evaluation_policy.bootstrap_seed",
        source=source,
    )


def _normalize_json_case(row: Mapping[str, Any], *, index: int, source: Path) -> dict[str, Any]:
    if set(row) != set(CASE_FIELDS):
        raise PublishError(f"cases[{index}] in {source} must contain exactly {list(CASE_FIELDS)}")
    case_id = row["case_id"]
    if (
        not isinstance(case_id, str)
        or not case_id
        or case_id != case_id.strip()
        or len(case_id) > 256
        or any(ord(character) < 32 for character in case_id)
    ):
        raise PublishError(f"cases[{index}].case_id in {source} is invalid")

    normalized: dict[str, Any] = {
        "case_id": case_id,
        "whole_pancreas_dice": _unit_number(
            row["whole_pancreas_dice"],
            field=f"cases[{index}].whole_pancreas_dice",
            source=source,
        ),
        "lesion_dice": _unit_number(
            row["lesion_dice"],
            field=f"cases[{index}].lesion_dice",
            source=source,
        ),
    }
    for field in ("reference_subtype", "predicted_subtype"):
        value = _required_integer_value(row[field], field=f"cases[{index}].{field}", source=source)
        if value not in (0, 1, 2):
            raise PublishError(f"cases[{index}].{field} in {source} must be 0, 1, or 2")
        normalized[field] = value
    for field in (
        "whole_pancreas_predicted_voxels",
        "whole_pancreas_reference_voxels",
        "lesion_predicted_voxels",
        "lesion_reference_voxels",
    ):
        normalized[field] = _required_integer_value(
            row[field], field=f"cases[{index}].{field}", source=source, minimum=0
        )
    for field in (
        "classification_correct",
        "whole_pancreas_empty_empty",
        "lesion_empty_empty",
    ):
        if not isinstance(row[field], bool):
            raise PublishError(f"cases[{index}].{field} in {source} must be boolean")
        normalized[field] = row[field]

    if normalized["classification_correct"] != (
        normalized["reference_subtype"] == normalized["predicted_subtype"]
    ):
        raise PublishError(f"cases[{index}].classification_correct in {source} is inconsistent")
    expected_whole_empty = (
        normalized["whole_pancreas_predicted_voxels"] == 0
        and normalized["whole_pancreas_reference_voxels"] == 0
    )
    expected_lesion_empty = (
        normalized["lesion_predicted_voxels"] == 0 and normalized["lesion_reference_voxels"] == 0
    )
    if normalized["whole_pancreas_empty_empty"] != expected_whole_empty:
        raise PublishError(f"cases[{index}].whole_pancreas_empty_empty in {source} is inconsistent")
    if normalized["lesion_empty_empty"] != expected_lesion_empty:
        raise PublishError(f"cases[{index}].lesion_empty_empty in {source} is inconsistent")
    if expected_whole_empty and normalized["whole_pancreas_dice"] != 1.0:
        raise PublishError(f"cases[{index}].whole_pancreas_dice in {source} violates empty policy")
    if expected_lesion_empty and normalized["lesion_dice"] != 1.0:
        raise PublishError(f"cases[{index}].lesion_dice in {source} violates empty policy")
    return normalized


def _macro_f1(cases: Sequence[Mapping[str, Any]]) -> float:
    scores: list[float] = []
    for label in (0, 1, 2):
        true_positive = sum(
            row["reference_subtype"] == label and row["predicted_subtype"] == label for row in cases
        )
        false_positive = sum(
            row["reference_subtype"] != label and row["predicted_subtype"] == label for row in cases
        )
        false_negative = sum(
            row["reference_subtype"] == label and row["predicted_subtype"] != label for row in cases
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return math.fsum(scores) / 3


def _validate_metrics(
    payload: Mapping[str, Any], *, source: Path
) -> tuple[dict[str, float], tuple[dict[str, Any], ...]]:
    _required_integer(payload, ("schema_version",), source=source, expected=SCHEMA_VERSION)
    _required_integer(payload, ("case_count",), source=source, expected=EXPECTED_CASE_COUNT)
    _required_integer(
        payload, ("segmentation", "case_count"), source=source, expected=EXPECTED_CASE_COUNT
    )
    _required_integer(
        payload, ("classification", "case_count"), source=source, expected=EXPECTED_CASE_COUNT
    )
    _required_integer(
        payload,
        ("classification", "unused_reference_case_count"),
        source=source,
        expected=0,
    )
    _validate_evaluation_policy(payload, source=source)

    metrics = {
        name: _required_metric(payload, path, source=source) for name, path in METRIC_PATHS.items()
    }
    raw_cases = _value_at(payload, ("cases",), source=source)
    if not isinstance(raw_cases, list) or len(raw_cases) != EXPECTED_CASE_COUNT:
        raise PublishError(
            f"Field cases in {source} must be a list of exactly {EXPECTED_CASE_COUNT} rows"
        )
    normalized_cases: list[dict[str, Any]] = []
    for index, row in enumerate(raw_cases):
        if not isinstance(row, Mapping):
            raise PublishError(f"cases[{index}] in {source} must be an object")
        normalized_cases.append(_normalize_json_case(row, index=index, source=source))
    case_ids = [row["case_id"] for row in normalized_cases]
    if len(set(case_ids)) != len(case_ids):
        raise PublishError(f"Field cases in {source} contains duplicate case IDs")
    if case_ids != sorted(case_ids):
        raise PublishError(f"Field cases in {source} must be sorted by case_id")

    recomputed = {
        "whole_pancreas_dice": math.fsum(row["whole_pancreas_dice"] for row in normalized_cases)
        / EXPECTED_CASE_COUNT,
        "lesion_dice": math.fsum(row["lesion_dice"] for row in normalized_cases)
        / EXPECTED_CASE_COUNT,
        "macro_f1": _macro_f1(normalized_cases),
    }
    for name, observed in metrics.items():
        if not math.isclose(observed, recomputed[name], rel_tol=0.0, abs_tol=1e-12):
            raise PublishError(
                f"Aggregate metric {name} in {source} is inconsistent with its case rows"
            )
    return metrics, tuple(normalized_cases)


def _parse_csv_bool(value: str, *, field: str, source: Path) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise PublishError(f"Field {field} in {source} must be True or False")


def _parse_csv_case(row: Mapping[str, str], *, index: int, source: Path) -> dict[str, Any]:
    def parse_float(field: str) -> float:
        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise PublishError(f"Field {field} in row {index + 2} of {source} is invalid") from exc
        return _unit_number(value, field=field, source=source)

    def parse_int(field: str) -> int:
        raw = row[field]
        if not isinstance(raw, str) or re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
            raise PublishError(f"Field {field} in row {index + 2} of {source} is invalid")
        return int(raw)

    case_id = row["case_id"]
    result: dict[str, Any] = {
        "case_id": case_id,
        "whole_pancreas_dice": parse_float("whole_pancreas_dice"),
        "lesion_dice": parse_float("lesion_dice"),
        "reference_subtype": parse_int("reference_subtype"),
        "predicted_subtype": parse_int("predicted_subtype"),
        "classification_correct": _parse_csv_bool(
            row["classification_correct"], field="classification_correct", source=source
        ),
        "whole_pancreas_predicted_voxels": parse_int("whole_pancreas_predicted_voxels"),
        "whole_pancreas_reference_voxels": parse_int("whole_pancreas_reference_voxels"),
        "lesion_predicted_voxels": parse_int("lesion_predicted_voxels"),
        "lesion_reference_voxels": parse_int("lesion_reference_voxels"),
        "whole_pancreas_empty_empty": _parse_csv_bool(
            row["whole_pancreas_empty_empty"], field="whole_pancreas_empty_empty", source=source
        ),
        "lesion_empty_empty": _parse_csv_bool(
            row["lesion_empty_empty"], field="lesion_empty_empty", source=source
        ),
    }
    return result


def _validate_case_csv(path: Path, *, expected_cases: Sequence[Mapping[str, Any]]) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CASE_FIELDS:
                raise PublishError(
                    f"Case CSV columns in {path} must be exactly {list(CASE_FIELDS)}"
                )
            rows = list(reader)
    except PublishError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PublishError(f"Could not read case CSV {path}: {exc}") from exc
    if len(rows) != EXPECTED_CASE_COUNT:
        raise PublishError(
            f"Case CSV must contain exactly {EXPECTED_CASE_COUNT} data rows, got {len(rows)}"
        )
    normalized = tuple(
        _parse_csv_case(row, index=index, source=path) for index, row in enumerate(rows)
    )
    if normalized != tuple(expected_cases):
        raise PublishError("Case CSV content does not exactly match aggregate JSON case rows")


def _validate_selection_policy(payload: Mapping[str, Any], *, source: Path) -> None:
    policy = payload.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise PublishError(f"Field selection_policy in {source} must be an object")
    expected = {
        "direction": "maximize",
        "metric_paths": SELECTION_METRIC_PATHS,
        "score": "equal-weight arithmetic mean",
        "tie_breaker": "candidate name ascending; no secondary metric",
    }
    for field, value in expected.items():
        _require_exact(policy.get(field), value, field=f"selection_policy.{field}", source=source)
    weights = policy.get("metric_weights")
    if not isinstance(weights, Mapping) or set(weights) != set(METRIC_PATHS):
        raise PublishError(f"selection_policy.metric_weights in {source} is invalid")
    for name in METRIC_PATHS:
        observed = _unit_number(
            weights[name], field=f"selection_policy.metric_weights.{name}", source=source
        )
        if not math.isclose(observed, 1 / 3, rel_tol=0.0, abs_tol=1e-12):
            raise PublishError(f"selection_policy.metric_weights.{name} in {source} must be 1/3")


def _validate_selection(
    payload: Mapping[str, Any],
    *,
    source: Path,
    metrics: Mapping[str, float],
) -> tuple[str, str, float, tuple[dict[str, Any], ...]]:
    _required_integer(payload, ("schema_version",), source=source, expected=SCHEMA_VERSION)
    _validate_selection_policy(payload, source=source)
    candidate_count = _required_integer(payload, ("candidate_count",), source=source)
    if candidate_count not in (3, 4):
        raise PublishError(f"Field candidate_count in {source} must be 3 or 4")

    selected_candidate = payload.get("selected_candidate")
    if not isinstance(selected_candidate, str) or not selected_candidate.strip():
        raise PublishError(f"Field selected_candidate in {source} must be a non-empty string")
    selected_candidate = selected_candidate.strip()

    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != candidate_count:
        raise PublishError(f"Field ranking in {source} must contain candidate_count entries")
    sanitized_ranking: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(ranking, start=1):
        if not isinstance(entry, Mapping):
            raise PublishError(f"ranking[{index - 1}] in {source} must be an object")
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or candidate != candidate.strip() or not candidate:
            raise PublishError(f"ranking[{index - 1}].candidate in {source} is invalid")
        if candidate in seen:
            raise PublishError(f"Duplicate candidate {candidate!r} in {source}")
        seen.add(candidate)
        _required_integer_value(
            entry.get("rank"), field=f"ranking[{index - 1}].rank", source=source, expected=index
        )
        source_value = entry.get("metrics_source")
        if not isinstance(source_value, str) or not source_value.strip():
            raise PublishError(f"ranking[{index - 1}].metrics_source in {source} is invalid")
        entry_metrics = entry.get("metrics")
        if not isinstance(entry_metrics, Mapping) or set(entry_metrics) != set(METRIC_PATHS):
            raise PublishError(f"ranking[{index - 1}].metrics in {source} is invalid")
        normalized_metrics = {
            name: _unit_number(
                entry_metrics[name],
                field=f"ranking[{index - 1}].metrics.{name}",
                source=source,
            )
            for name in METRIC_PATHS
        }
        expected_score = math.fsum(normalized_metrics.values()) / len(METRIC_PATHS)
        score = _unit_number(
            entry.get("selection_score"),
            field=f"ranking[{index - 1}].selection_score",
            source=source,
        )
        if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-12):
            raise PublishError(f"ranking[{index - 1}].selection_score in {source} is inconsistent")
        digest = entry.get("checkpoint_sha256")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise PublishError(f"ranking[{index - 1}].checkpoint_sha256 in {source} is invalid")
        sanitized_ranking.append(
            {
                "rank": index,
                "candidate": candidate,
                "metrics": normalized_metrics,
                "selection_score": score,
                "checkpoint_sha256": digest.casefold(),
            }
        )

    expected_candidates = BASE_CANDIDATES | ({RESCUE_CANDIDATE} if candidate_count == 4 else set())
    if seen != expected_candidates:
        raise PublishError(
            f"Candidate set in {source} is invalid; expected {sorted(expected_candidates)}, got {sorted(seen)}"
        )
    expected_order = sorted(
        sanitized_ranking, key=lambda entry: (-entry["selection_score"], entry["candidate"])
    )
    if sanitized_ranking != expected_order:
        raise PublishError(f"Ranking in {source} is not in deterministic score order")
    selected_entry = sanitized_ranking[0]
    if selected_entry["candidate"] != selected_candidate:
        raise PublishError(f"selected_candidate in {source} must be the rank-1 candidate")
    for name, expected_value in metrics.items():
        if selected_entry["metrics"][name] != expected_value:
            raise PublishError(f"Selected metric {name} in {source} does not match metrics JSON")

    selected_score = _unit_number(
        payload.get("selected_score"), field="selected_score", source=source
    )
    if not math.isclose(
        selected_score, selected_entry["selection_score"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise PublishError(f"selected_score in {source} does not match rank 1")
    digest = payload.get("selected_checkpoint_sha256")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise PublishError(f"selected_checkpoint_sha256 in {source} must be 64 hex digits")
    if digest.casefold() != selected_entry["checkpoint_sha256"]:
        raise PublishError(f"selected_checkpoint_sha256 in {source} does not match rank 1")
    return selected_candidate, digest.casefold(), selected_score, tuple(sanitized_ranking)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PublishError(f"Sanitized evaluation payload is not strict JSON: {exc}") from exc
    return rendered.encode("utf-8")


def validate_bundle(
    metrics_path: Path, case_csv_path: Path, selection_path: Path
) -> dict[str, Any]:
    """Validate the complete private bundle and derive a sanitized public digest."""

    metrics_path = _resolve_input(
        metrics_path, description="Aggregate metrics JSON", suffix=".json"
    )
    case_csv_path = _resolve_input(case_csv_path, description="Case metrics CSV", suffix=".csv")
    selection_path = _resolve_input(
        selection_path, description="Checkpoint-selection JSON", suffix=".json"
    )
    if len({metrics_path, case_csv_path, selection_path}) != 3:
        raise PublishError("Metrics, case CSV, and selection inputs must be distinct files")

    metrics_payload = _load_json(metrics_path, description="aggregate metrics JSON")
    selection_payload = _load_json(selection_path, description="checkpoint-selection JSON")
    metrics, cases = _validate_metrics(metrics_payload, source=metrics_path)
    _validate_case_csv(case_csv_path, expected_cases=cases)
    selected_candidate, checkpoint_sha256, selected_score, ranking = _validate_selection(
        selection_payload, source=selection_path, metrics=metrics
    )

    public_payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": {"case_count": EXPECTED_CASE_COUNT, "metrics": dict(metrics)},
        "selection": {
            "candidate_count": len(ranking),
            "selected_candidate": selected_candidate,
            "selected_checkpoint_sha256": checkpoint_sha256,
            "selection_score": selected_score,
            "ranking": list(ranking),
        },
    }
    payload_sha256 = hashlib.sha256(_canonical_json_bytes(public_payload)).hexdigest()
    summary = {
        "full_volume/schema_version": SCHEMA_VERSION,
        "full_volume/evaluation_payload_sha256": payload_sha256,
        "full_volume/whole_pancreas_dice": metrics["whole_pancreas_dice"],
        "full_volume/lesion_dice": metrics["lesion_dice"],
        "full_volume/macro_f1": metrics["macro_f1"],
        "full_volume/selected_checkpoint": selected_candidate,
        "full_volume/selected_checkpoint_sha256": checkpoint_sha256,
        "full_volume/selection_score": selected_score,
        "full_volume/case_count": EXPECTED_CASE_COUNT,
        "full_volume/candidate_count": len(ranking),
    }
    return {"public_payload": public_payload, "payload_sha256": payload_sha256, "summary": summary}


def _validate_identifier(value: str, *, option: str) -> str:
    normalized = value.strip()
    if not normalized or IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise PublishError(
            f"{option} must contain only letters, digits, underscores, periods, and hyphens"
        )
    return normalized


def build_publish_plan(
    bundle: Mapping[str, Any], *, entity: str, project: str, run_id: str
) -> dict[str, Any]:
    """Build the sanitized, credential-free plan shown by ``--dry-run``."""

    entity = _validate_identifier(entity, option="--entity")
    project = _validate_identifier(project, option="--project")
    run_id = _validate_identifier(run_id, option="--run-id")
    return {
        "target": {
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "run_path": f"{entity}/{project}/{run_id}",
        },
        "summary": dict(bundle["summary"]),
        "sanitized_evaluation": dict(bundle["public_payload"]),
        "network_operation": "one Public API summary update after remote preflight; no history or artifact",
    }


def _summary_epoch(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublishError(f"Remote summary field {field!r} must be an integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise PublishError(f"Remote summary field {field!r} must be an integer")
    return int(number)


def _summary_value_matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            not isinstance(observed, bool)
            and isinstance(observed, (int, float))
            and math.isfinite(float(observed))
            and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        )
    return observed == expected


def _validate_canonical_run_url(value: Any, *, entity: str, project: str, run_id: str) -> str:
    """Require the exact public W&B SaaS URL for the intended run."""

    if not isinstance(value, str):
        raise PublishError("W&B run URL must be a string")
    expected_path = "/" + "/".join(
        (
            quote(entity, safe=""),
            quote(project, safe=""),
            "runs",
            quote(run_id, safe=""),
        )
    )
    expected_url = f"https://wandb.ai{expected_path}"
    try:
        parsed = urlsplit(value)
        has_userinfo = parsed.username is not None or parsed.password is not None
        port = parsed.port
    except ValueError as exc:
        raise PublishError(f"W&B run URL is invalid: {value!r}") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc != "wandb.ai"
        or parsed.hostname != "wandb.ai"
        or port is not None
        or has_userinfo
        or parsed.query
        or parsed.fragment
        or parsed.path != expected_path
        or value != expected_url
    ):
        raise PublishError(
            "W&B run URL must be the exact canonical public SaaS URL "
            f"{expected_url!r}, got {value!r}"
        )
    return value


def validate_remote_run(
    run: Any, *, target: Mapping[str, str], expected_summary: Mapping[str, Any]
) -> str:
    """Validate the exact completed run and return ``update`` or ``unchanged``."""

    identity = (str(run.entity), str(run.project), str(run.id))
    expected_identity = (target["entity"], target["project"], target["run_id"])
    if identity != expected_identity:
        raise PublishError(f"W&B returned run identity {identity}, expected {expected_identity}")
    _validate_canonical_run_url(
        run.url,
        entity=target["entity"],
        project=target["project"],
        run_id=target["run_id"],
    )
    if run.state != "finished":
        raise PublishError(f"W&B run must be finished before publication, got {run.state!r}")
    last_step = run.lastHistoryStep
    if (
        isinstance(last_step, bool)
        or not isinstance(last_step, int)
        or last_step != EXPECTED_HISTORY_STEP
    ):
        raise PublishError(
            f"W&B last history step must be {EXPECTED_HISTORY_STEP}, got {last_step!r}"
        )
    summary = dict(run.summary_metrics)
    if _summary_epoch(summary.get("_step"), field="_step") != EXPECTED_HISTORY_STEP:
        raise PublishError("W&B summary _step must equal 199")
    if _summary_epoch(summary.get("current_epoch"), field="current_epoch") != EXPECTED_HISTORY_STEP:
        raise PublishError("W&B summary current_epoch must equal 199")

    existing = {key: value for key, value in summary.items() if key.startswith("full_volume/")}
    if not existing:
        return "update"
    if set(existing) != set(expected_summary):
        raise PublishError(
            "Remote run already contains a partial, extra, or conflicting full_volume summary"
        )
    mismatches = [
        key
        for key, expected in expected_summary.items()
        if not _summary_value_matches(existing.get(key), expected)
    ]
    if mismatches:
        raise PublishError(
            "Remote run already contains conflicting full_volume values: " + ", ".join(mismatches)
        )
    return "unchanged"


def _import_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise PublishError("wandb is required; install the pinned project dependencies") from exc
    return wandb


def publish(plan: Mapping[str, Any], *, wandb_module: Any | None = None) -> str:
    """Preflight and update the summary once, or perform an idempotent no-op."""

    wandb_module = _import_wandb() if wandb_module is None else wandb_module
    target = plan["target"]
    try:
        api = wandb_module.Api()
        run = api.run(target["run_path"])
        run.load(force=True)
        action = validate_remote_run(run, target=target, expected_summary=plan["summary"])
        if action == "unchanged":
            return action
        run.summary.update(dict(plan["summary"]))
    except PublishError:
        raise
    except Exception as exc:
        raise PublishError(f"W&B rejected the summary publication: {exc}") from exc
    return "updated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--case-csv", required=True, type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the sanitized plan without importing W&B or using the network",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = validate_bundle(args.metrics_json, args.case_csv, args.selection_json)
    plan = build_publish_plan(bundle, entity=args.entity, project=args.project, run_id=args.run_id)
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        status = "dry_run"
    else:
        status = publish(plan)
        if status == "unchanged":
            print(
                "W&B already contains the identical validated full-volume summary; no update made."
            )
        else:
            print(
                "Published sanitized full-volume summary to "
                f"{args.entity}/{args.project}/{args.run_id}; no history or artifact was added."
            )
    return {"status": status, "plan": plan}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (PublishError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
