from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import publish_full_volume_to_wandb as publisher


def _evaluation_policy() -> dict[str, Any]:
    return {
        "whole_pancreas": "label > 0",
        "lesion": "label == 2",
        "empty_empty_dice": 1.0,
        "one_sided_empty_dice": 0.0,
        "classification_labels": [0, 1, 2],
        "classification_zero_division": 0.0,
        "confusion_matrix_rows": "reference",
        "confusion_matrix_columns": "prediction",
        "aggregation": "unweighted case mean",
        "bootstrap_seed": 12345,
    }


def _write_bundle(root: Path) -> tuple[Path, Path, Path]:
    cases: list[dict[str, Any]] = []
    for index in range(36):
        subtype = index % 3
        cases.append(
            {
                "case_id": f"quiz_private_{index + 1:03d}",
                "whole_pancreas_dice": 0.91,
                "lesion_dice": 0.31,
                "reference_subtype": subtype,
                "predicted_subtype": subtype,
                "classification_correct": True,
                "whole_pancreas_predicted_voxels": 1000 + index,
                "whole_pancreas_reference_voxels": 1001 + index,
                "lesion_predicted_voxels": 100 + index,
                "lesion_reference_voxels": 101 + index,
                "whole_pancreas_empty_empty": False,
                "lesion_empty_empty": False,
            }
        )

    metrics_path = root / "metrics.json"
    metrics = {
        "schema_version": 1,
        "evaluation_policy": _evaluation_policy(),
        "case_count": 36,
        "segmentation": {
            "case_count": 36,
            "whole_pancreas_dice": {"mean": 0.91},
            "lesion_dice": {"mean": 0.31},
        },
        "classification": {
            "case_count": 36,
            "unused_reference_case_count": 0,
            "macro_f1": 1.0,
        },
        "cases": cases,
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    case_csv_path = root / "case_metrics.csv"
    with case_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=publisher.CASE_FIELDS)
        writer.writeheader()
        writer.writerows(cases)

    candidate_metrics = {
        "checkpoint_best_multitask": {
            "whole_pancreas_dice": 0.91,
            "lesion_dice": 0.31,
            "macro_f1": 1.0,
        },
        "checkpoint_final": {
            "whole_pancreas_dice": 0.80,
            "lesion_dice": 0.20,
            "macro_f1": 0.50,
        },
        "checkpoint_best": {
            "whole_pancreas_dice": 0.75,
            "lesion_dice": 0.15,
            "macro_f1": 0.45,
        },
    }
    ranking: list[dict[str, Any]] = []
    for rank, (candidate, values) in enumerate(candidate_metrics.items(), start=1):
        ranking.append(
            {
                "candidate": candidate,
                "rank": rank,
                "metrics_source": str(
                    metrics_path.resolve()
                    if rank == 1
                    else (root / f"private-{candidate}.json").resolve()
                ),
                "checkpoint_path": str((root / f"private-{candidate}.pth").resolve()),
                "metrics": values,
                "selection_score": math.fsum(values.values()) / 3,
                "checkpoint_sha256": f"{rank:02x}" * 32,
            }
        )
    selection_path = root / "checkpoint_selection.json"
    selection = {
        "schema_version": 1,
        "selection_policy": {
            "direction": "maximize",
            "metric_paths": publisher.SELECTION_METRIC_PATHS,
            "metric_weights": {
                "whole_pancreas_dice": 1 / 3,
                "lesion_dice": 1 / 3,
                "macro_f1": 1 / 3,
            },
            "score": "equal-weight arithmetic mean",
            "tie_breaker": "candidate name ascending; no secondary metric",
        },
        "candidate_count": 3,
        "selected_candidate": ranking[0]["candidate"],
        "selected_score": ranking[0]["selection_score"],
        "selected_checkpoint_path": ranking[0]["checkpoint_path"],
        "selected_checkpoint_sha256": ranking[0]["checkpoint_sha256"],
        "ranking": ranking,
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    return metrics_path, case_csv_path, selection_path


def _arguments(paths: tuple[Path, Path, Path], *, dry_run: bool) -> list[str]:
    metrics_path, case_csv_path, selection_path = paths
    arguments = [
        "--metrics-json",
        str(metrics_path),
        "--case-csv",
        str(case_csv_path),
        "--selection-json",
        str(selection_path),
        "--entity",
        "candidate-entity",
        "--project",
        "pancreas-project",
        "--run-id",
        "hrs05iyx",
    ]
    if dry_run:
        arguments.append("--dry-run")
    return arguments


def _plan(paths: tuple[Path, Path, Path]) -> dict[str, Any]:
    bundle = publisher.validate_bundle(*paths)
    return publisher.build_publish_plan(
        bundle,
        entity="candidate-entity",
        project="pancreas-project",
        run_id="hrs05iyx",
    )


class FakeSummary:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, values: dict[str, Any]) -> None:
        self.updates.append(values)


class FakeRun:
    entity = "candidate-entity"
    project = "pancreas-project"
    id = "hrs05iyx"
    url = "https://wandb.ai/candidate-entity/pancreas-project/runs/hrs05iyx"

    def __init__(
        self,
        *,
        state: str = "finished",
        last_step: int = 199,
        existing: dict[str, Any] | None = None,
    ) -> None:
        self.state = state
        self.lastHistoryStep = last_step
        self.summary_metrics = {"_step": 199, "current_epoch": 199, **(existing or {})}
        self.summary = FakeSummary()
        self.load_calls: list[bool] = []

    def load(self, *, force: bool) -> None:
        self.load_calls.append(force)


def _fake_wandb(fake_run: FakeRun, paths: list[str]) -> Any:
    class FakeApi:
        def run(self, path: str) -> FakeRun:
            paths.append(path)
            return fake_run

    return SimpleNamespace(Api=lambda: FakeApi())


def test_dry_run_is_network_free_and_contains_no_private_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_bundle(tmp_path)

    def fail_import() -> Any:
        pytest.fail("dry-run must not import W&B")

    monkeypatch.setattr(publisher, "_import_wandb", fail_import)
    assert publisher.main(_arguments(paths, dry_run=True)) == 0
    rendered = capsys.readouterr().out
    plan = json.loads(rendered)
    assert plan["dry_run"] is True
    assert "quiz_private" not in rendered
    assert "reference_subtype" not in rendered
    assert str(tmp_path) not in rendered
    assert ".pth" not in rendered
    assert plan["summary"]["full_volume/case_count"] == 36
    assert len(plan["summary"]["full_volume/evaluation_payload_sha256"]) == 64


def test_publish_preflights_and_updates_summary_exactly_once(tmp_path: Path) -> None:
    plan = _plan(_write_bundle(tmp_path))
    fake_run = FakeRun()
    api_paths: list[str] = []
    fake_wandb = _fake_wandb(fake_run, api_paths)

    assert publisher.publish(plan, wandb_module=fake_wandb) == "updated"
    assert api_paths == ["candidate-entity/pancreas-project/hrs05iyx"]
    assert fake_run.load_calls == [True]
    assert fake_run.summary.updates == [plan["summary"]]
    assert not hasattr(fake_wandb, "init")
    assert not hasattr(fake_wandb, "Artifact")


@pytest.mark.parametrize(
    ("state", "last_step", "summary_change", "message"),
    [
        ("running", 199, {}, "must be finished"),
        ("finished", 198, {}, "last history step"),
        ("finished", 199, {"_step": 198}, "summary _step"),
        ("finished", 199, {"current_epoch": 198}, "current_epoch"),
    ],
)
def test_publish_rejects_partial_or_active_remote_run(
    tmp_path: Path,
    state: str,
    last_step: int,
    summary_change: dict[str, Any],
    message: str,
) -> None:
    plan = _plan(_write_bundle(tmp_path))
    fake_run = FakeRun(state=state, last_step=last_step)
    fake_run.summary_metrics.update(summary_change)
    with pytest.raises(publisher.PublishError, match=message):
        publisher.publish(plan, wandb_module=_fake_wandb(fake_run, []))
    assert fake_run.summary.updates == []


@pytest.mark.parametrize(
    "url",
    [
        "http://wandb.ai/candidate-entity/pancreas-project/runs/hrs05iyx",
        "https://wandb.example/candidate-entity/pancreas-project/runs/hrs05iyx",
        "https://user@wandb.ai/candidate-entity/pancreas-project/runs/hrs05iyx",
        "https://wandb.ai/candidate-entity/pancreas-project/runs/hrs05iyx?view=1",
        "https://wandb.ai/candidate-entity/pancreas-project/runs/hrs05iyx#overview",
        "https://wandb.ai/candidate-entity/pancreas-project/hrs05iyx",
    ],
)
def test_publish_rejects_noncanonical_run_url(tmp_path: Path, url: str) -> None:
    plan = _plan(_write_bundle(tmp_path))
    fake_run = FakeRun()
    fake_run.url = url
    with pytest.raises(publisher.PublishError, match="canonical public SaaS URL"):
        publisher.publish(plan, wandb_module=_fake_wandb(fake_run, []))
    assert fake_run.summary.updates == []


def test_identical_remote_summary_is_noop_and_conflict_fails(tmp_path: Path) -> None:
    plan = _plan(_write_bundle(tmp_path))
    identical = FakeRun(existing=dict(plan["summary"]))
    assert publisher.publish(plan, wandb_module=_fake_wandb(identical, [])) == "unchanged"
    assert identical.summary.updates == []

    conflict_values = dict(plan["summary"])
    conflict_values["full_volume/macro_f1"] = 0.5
    conflict = FakeRun(existing=conflict_values)
    with pytest.raises(publisher.PublishError, match="conflicting"):
        publisher.publish(plan, wandb_module=_fake_wandb(conflict, []))
    assert conflict.summary.updates == []


def test_partial_existing_full_volume_summary_fails_closed(tmp_path: Path) -> None:
    plan = _plan(_write_bundle(tmp_path))
    fake_run = FakeRun(existing={"full_volume/case_count": 36})
    with pytest.raises(publisher.PublishError, match="partial"):
        publisher.publish(plan, wandb_module=_fake_wandb(fake_run, []))
    assert fake_run.summary.updates == []


def test_case_csv_values_must_exactly_match_json(tmp_path: Path) -> None:
    metrics_path, case_csv_path, selection_path = _write_bundle(tmp_path)
    rows = list(csv.DictReader(case_csv_path.read_text(encoding="utf-8").splitlines()))
    rows[0]["lesion_dice"] = "0.99"
    with case_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=publisher.CASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(publisher.PublishError, match="does not exactly match"):
        publisher.validate_bundle(metrics_path, case_csv_path, selection_path)


def test_aggregate_and_every_ranking_score_are_recomputed(tmp_path: Path) -> None:
    metrics_path, case_csv_path, selection_path = _write_bundle(tmp_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["segmentation"]["lesion_dice"]["mean"] = 0.30
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(publisher.PublishError, match="inconsistent with its case rows"):
        publisher.validate_bundle(metrics_path, case_csv_path, selection_path)

    metrics["segmentation"]["lesion_dice"]["mean"] = 0.31
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["ranking"][1]["selection_score"] = 0.1
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(publisher.PublishError, match=r"ranking\[1\].*inconsistent"):
        publisher.validate_bundle(metrics_path, case_csv_path, selection_path)


def test_cli_exposes_no_api_key_or_artifact_option() -> None:
    options = {
        option for action in publisher.build_parser()._actions for option in action.option_strings
    }
    assert "--api-key" not in options
    assert "--token" not in options
    assert "--password" not in options
    assert "--artifact" not in options
