from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_wandb_evidence as exporter


def _row(step: int, *, timestamp: float | None = None) -> dict[str, Any]:
    start = 1_800_000_000.0 + step * 10.0
    class_1 = 0.55 + step / 1_000
    class_2 = 0.25 + step / 2_000
    mean_dice = math.fsum((class_1, class_2)) / 2
    macro_f1 = 0.30 + step / 1_000
    row: dict[str, Any] = {
        "_step": step,
        "_timestamp": start + 6.0 if timestamp is None else timestamp,
        "_runtime": float(step * 10 + 6),
        "dice_per_class_or_region/class_1": class_1,
        "dice_per_class_or_region/class_2": class_2,
        "ema_fg_dice": mean_dice,
        "epoch_end_timestamps": start + 5.0,
        "epoch_start_timestamps": start,
        "lrs": 0.01 * (1.0 - step / 200.0) ** 0.9,
        "mean_fg_dice": mean_dice,
        "train_cls_accuracy": 0.32,
        "train_cls_losses": 1.10,
        "train_lesion_patch_fraction": 0.50,
        "train_losses": -0.10,
        "train_seg_losses": -0.65,
        "val_cls_accuracy": 0.34,
        "val_cls_case_coverage": 1.0,
        "val_cls_f1_per_class/class_1": 0.31,
        "val_cls_f1_per_class/class_2": 0.32,
        "val_cls_f1_per_class/class_3": 0.33,
        "val_cls_losses": 1.09,
        "val_cls_macro_f1": macro_f1,
        "val_lesion_patch_fraction": 0.50,
        "val_losses": -0.11,
        "val_multitask_score": math.fsum((mean_dice, macro_f1)) / 2,
        "val_seg_losses": -0.66,
        "val_whole_pancreas_dice": 0.70 + step / 1_000,
        "private_case_id": f"quiz_private_{step:03d}",
    }
    return row


def _complete_rows() -> list[dict[str, Any]]:
    return [_row(step) for step in exporter.EXPECTED_EPOCHS]


def _terminal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    terminal = rows[-1]
    return {
        "_step": 199,
        "current_epoch": 199,
        "_timestamp": terminal["_timestamp"],
        "_runtime": terminal["_runtime"],
        **{field: terminal[field] for field in exporter.REQUIRED_METRIC_FIELDS},
        "private/path": "C:/private/source",
    }


def test_canonicalization_keeps_later_duplicate_and_strips_extra_fields() -> None:
    rows = _complete_rows()
    older = _row(8, timestamp=float(rows[8]["_timestamp"]) - 1.0)
    older["train_losses"] = -0.123
    rows.insert(8, older)

    history = exporter.validate_and_canonicalize_history(rows)

    assert len(history.raw_rows) == 201
    assert len(history.canonical_rows) == 200
    assert history.duplicate_steps == {8: 2}
    assert history.canonical_rows[8]["_timestamp"] == _row(8)["_timestamp"]
    assert history.canonical_rows[8]["train_losses"] != -0.123
    assert "private_case_id" not in history.raw_rows[0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(17), r"coverage.*missing=\[17\]"),
        (lambda rows: rows.append(_row(9, timestamp=1_900_000_000.0)), r"Only step 8"),
        (
            lambda rows: rows[3].pop("val_cls_macro_f1"),
            r"missing required fields.*val_cls_macro_f1",
        ),
        (
            lambda rows: rows[4].__setitem__("train_losses", float("nan")),
            r"train_losses.*must be finite",
        ),
        (
            lambda rows: rows[5].__setitem__("val_cls_accuracy", 1.01),
            r"val_cls_accuracy.*\[0, 1\]",
        ),
        (lambda rows: rows[6].__setitem__("lrs", 0.0), r"learning rate.*positive"),
        (
            lambda rows: rows[7].__setitem__("val_multitask_score", 0.1),
            r"val_multitask_score.*inconsistent",
        ),
        (
            lambda rows: rows[9].__setitem__(
                "epoch_start_timestamps", rows[8]["epoch_end_timestamps"] - 1
            ),
            r"starts before",
        ),
    ],
)
def test_history_validation_rejects_invalid_or_inconsistent_rows(
    mutation: Any, message: str
) -> None:
    rows = _complete_rows()
    mutation(rows)
    with pytest.raises(exporter.EvidenceValidationError, match=message):
        exporter.validate_and_canonicalize_history(rows)


def test_metadata_requires_terminal_summary_to_equal_epoch_199() -> None:
    rows = _complete_rows()
    summary = _terminal_summary(rows)
    metadata = exporter.validate_run_metadata(
        requested_run_path="candidate/project/run123",
        entity="candidate",
        project="project",
        run_id="run123",
        run_url="https://wandb.ai/candidate/project/runs/run123",
        state="finished",
        last_history_step=199,
        summary=summary,
        terminal_row=rows[-1],
    )
    assert metadata["full_volume_summary"] == "absent"

    summary["val_cls_macro_f1"] = float(summary["val_cls_macro_f1"]) + 0.01
    with pytest.raises(exporter.EvidenceValidationError, match="does not match epoch 199"):
        exporter.validate_run_metadata(
            requested_run_path="candidate/project/run123",
            entity="candidate",
            project="project",
            run_id="run123",
            run_url="https://wandb.ai/candidate/project/runs/run123",
            state="finished",
            last_history_step=199,
            summary=summary,
            terminal_row=rows[-1],
        )


def test_metadata_rejects_partial_full_volume_summary() -> None:
    rows = _complete_rows()
    summary = _terminal_summary(rows)
    summary["full_volume/case_count"] = 36
    with pytest.raises(exporter.EvidenceValidationError, match="partial"):
        exporter.validate_run_metadata(
            requested_run_path="candidate/project/run123",
            entity="candidate",
            project="project",
            run_id="run123",
            run_url="https://wandb.ai/candidate/project/runs/run123",
            state="finished",
            last_history_step=199,
            summary=summary,
            terminal_row=rows[-1],
        )


def test_export_refreshes_remote_and_writes_fresh_sanitized_bundle(tmp_path: Path) -> None:
    rows = _complete_rows()
    rows.insert(8, _row(8, timestamp=float(rows[8]["_timestamp"]) - 1.0))
    summary = _terminal_summary(_complete_rows())
    scan_calls: list[dict[str, Any]] = []
    load_calls: list[bool] = []

    class FakeRun:
        entity = "candidate"
        project = "project"
        id = "run123"
        state = "finished"
        lastHistoryStep = 199
        url = "https://wandb.ai/candidate/project/runs/run123"
        summary_metrics = summary

        def load(self, *, force: bool) -> None:
            load_calls.append(force)

        def scan_history(self, **kwargs: Any) -> list[dict[str, Any]]:
            scan_calls.append(kwargs)
            return rows

    class FakeApi:
        def run(self, path: str) -> FakeRun:
            assert path == "candidate/project/run123"
            return FakeRun()

    output = tmp_path / "fresh-evidence"
    fake_wandb = SimpleNamespace(Api=lambda: FakeApi())
    audit = exporter.export_run_evidence(
        "candidate/project/run123",
        output,
        wandb_module=fake_wandb,
        created_at_utc="2026-08-06T00:00:00+00:00",
    )

    assert load_calls == [True, True]
    assert scan_calls == [{"page_size": 1_000, "use_cache": False}]
    assert audit["history"]["duplicate_steps"] == {"8": 2}
    expected_files = {
        exporter.RAW_HISTORY_FILENAME,
        exporter.CANONICAL_HISTORY_FILENAME,
        exporter.CANONICAL_CSV_FILENAME,
        exporter.SUMMARY_FILENAME,
        exporter.AUDIT_FILENAME,
        exporter.HASHES_FILENAME,
    }
    assert {path.name for path in output.iterdir()} == expected_files

    raw = json.loads((output / exporter.RAW_HISTORY_FILENAME).read_text(encoding="utf-8"))
    saved_summary = json.loads((output / exporter.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    manifest = json.loads((output / exporter.HASHES_FILENAME).read_text(encoding="utf-8"))
    assert "private_case_id" not in raw[0]
    assert "private/path" not in saved_summary
    assert set(manifest["files"]) == expected_files - {exporter.HASHES_FILENAME}
    for filename, evidence in manifest["files"].items():
        payload = (output / filename).read_bytes()
        assert evidence["bytes"] == len(payload)
        assert evidence["sha256"] == hashlib.sha256(payload).hexdigest()

    with pytest.raises(exporter.EvidenceValidationError, match="must be fresh"):
        exporter.export_run_evidence(
            "candidate/project/run123",
            output,
            wandb_module=fake_wandb,
            created_at_utc="2026-08-06T00:00:00+00:00",
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://wandb.ai/candidate/project/runs/run123",
        "https://wandb.example/candidate/project/runs/run123",
        "https://user@wandb.ai/candidate/project/runs/run123",
        "https://wandb.ai/candidate/project/runs/run123?view=1",
        "https://wandb.ai/candidate/project/runs/run123#overview",
        "https://wandb.ai/candidate/project/run123",
    ],
)
def test_metadata_rejects_noncanonical_run_url(url: str) -> None:
    rows = _complete_rows()
    with pytest.raises(exporter.EvidenceValidationError, match="canonical public SaaS URL"):
        exporter.validate_run_metadata(
            requested_run_path="candidate/project/run123",
            entity="candidate",
            project="project",
            run_id="run123",
            run_url=url,
            state="finished",
            last_history_step=199,
            summary=_terminal_summary(rows),
            terminal_row=rows[-1],
        )


def test_canonical_run_url_requires_expected_percent_quoted_path() -> None:
    assert (
        exporter._validate_canonical_run_url(
            "https://wandb.ai/candidate%20name/project/runs/run%20123",
            entity="candidate name",
            project="project",
            run_id="run 123",
        )
        == "https://wandb.ai/candidate%20name/project/runs/run%20123"
    )
    with pytest.raises(exporter.EvidenceValidationError, match="canonical public SaaS URL"):
        exporter._validate_canonical_run_url(
            "https://wandb.ai/candidate name/project/runs/run 123",
            entity="candidate name",
            project="project",
            run_id="run 123",
        )


def test_cli_has_no_credential_or_identity_defaults() -> None:
    actions = exporter.build_parser()._actions
    options = {option for action in actions for option in action.option_strings}
    assert options == {"-h", "--help", "--run-path", "--output-dir"}
    for action in actions:
        if action.dest in {"run_path", "output_dir"}:
            assert action.required is True
