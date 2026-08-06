from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import select_checkpoint as selector


def _write_metrics(
    path: Path,
    *,
    whole: object = 0.90,
    lesion: object = 0.30,
    macro_f1: object = 0.60,
) -> Path:
    payload = {
        "schema_version": 1,
        "case_count": 36,
        "segmentation": {
            "case_count": 36,
            "whole_pancreas_dice": {"mean": whole},
            "lesion_dice": {"mean": lesion},
        },
        "classification": {
            "case_count": 36,
            "unused_reference_case_count": 0,
            "macro_f1": macro_f1,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ranks_equal_weight_score_and_hashes_mapped_checkpoint(tmp_path: Path) -> None:
    best = _write_metrics(tmp_path / "best.json", whole=0.90, lesion=0.30, macro_f1=0.60)
    multitask = _write_metrics(
        tmp_path / "multitask.json", whole=0.91, lesion=0.31, macro_f1=0.70
    )
    final = _write_metrics(tmp_path / "final.json", whole=0.92, lesion=0.32, macro_f1=0.65)
    checkpoint = tmp_path / "checkpoint_best_multitask.pth"
    checkpoint.write_bytes(b"measured checkpoint bytes")

    artifact = selector.build_selection_artifact(
        {
            "checkpoint_final": final,
            "checkpoint_best": best,
            "checkpoint_best_multitask": multitask,
        },
        checkpoint_paths={"checkpoint_best_multitask": checkpoint},
    )

    assert artifact["selected_candidate"] == "checkpoint_best_multitask"
    assert artifact["selected_score"] == pytest.approx(0.64)
    assert [entry["candidate"] for entry in artifact["ranking"]] == [
        "checkpoint_best_multitask",
        "checkpoint_final",
        "checkpoint_best",
    ]
    assert [entry["rank"] for entry in artifact["ranking"]] == [1, 2, 3]
    selected = artifact["ranking"][0]
    assert selected["metrics_source"] == str(multitask.resolve())
    assert selected["metrics"] == {
        "whole_pancreas_dice": 0.91,
        "lesion_dice": 0.31,
        "macro_f1": 0.70,
    }
    expected_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert selected["checkpoint_sha256"] == expected_digest
    assert artifact["selected_checkpoint_sha256"] == expected_digest
    assert artifact["selection_policy"]["tie_breaker"].endswith("no secondary metric")


def test_exact_tie_uses_candidate_name_independent_of_input_order(tmp_path: Path) -> None:
    alpha = _write_metrics(tmp_path / "alpha.json")
    zeta = _write_metrics(tmp_path / "zeta.json")

    first = selector.build_selection_artifact({"zeta": zeta, "alpha": alpha})
    second = selector.build_selection_artifact({"alpha": alpha, "zeta": zeta})

    assert first == second
    assert first["selected_candidate"] == "alpha"
    assert [entry["candidate"] for entry in first["ranking"]] == ["alpha", "zeta"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("whole", float("nan"), r"finite and in \[0, 1\]"),
        ("lesion", 1.01, r"finite and in \[0, 1\]"),
        ("macro_f1", "0.60", "must be a JSON number"),
        ("macro_f1", True, "must be a JSON number"),
    ],
)
def test_rejects_invalid_required_metrics(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    invalid_kwargs = {field: value}
    invalid = _write_metrics(tmp_path / "invalid.json", **invalid_kwargs)
    valid = _write_metrics(tmp_path / "valid.json")

    with pytest.raises(selector.SelectionError, match=message):
        selector.build_selection_artifact({"invalid": invalid, "valid": valid})


def test_rejects_missing_metric_and_fewer_than_two_candidates(tmp_path: Path) -> None:
    one = _write_metrics(tmp_path / "one.json")
    missing = tmp_path / "missing.json"
    missing_payload = json.loads(one.read_text(encoding="utf-8"))
    del missing_payload["segmentation"]["whole_pancreas_dice"]
    missing.write_text(json.dumps(missing_payload), encoding="utf-8")

    with pytest.raises(selector.SelectionError, match="At least two"):
        selector.build_selection_artifact({"one": one})
    with pytest.raises(selector.SelectionError, match="whole_pancreas_dice.mean"):
        selector.build_selection_artifact({"one": one, "missing": missing})


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), 2, r"schema_version.*must equal 1"),
        (("case_count",), 35, r"case_count.*must equal 36"),
        (("segmentation", "case_count"), 35, r"segmentation.case_count.*must equal 36"),
        (("classification", "case_count"), 35, r"classification.case_count.*must equal 36"),
        (
            ("classification", "unused_reference_case_count"),
            1,
            r"unused_reference_case_count.*must equal 0",
        ),
    ],
)
def test_rejects_partial_or_wrong_schema_evaluations(
    tmp_path: Path,
    path: tuple[str, ...],
    value: int,
    message: str,
) -> None:
    valid = _write_metrics(tmp_path / "valid.json")
    invalid_payload = json.loads(valid.read_text(encoding="utf-8"))
    destination = invalid_payload
    for component in path[:-1]:
        destination = destination[component]
    destination[path[-1]] = value
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(invalid_payload), encoding="utf-8")

    with pytest.raises(selector.SelectionError, match=message):
        selector.build_selection_artifact({"invalid": invalid, "valid": valid})


def test_cli_atomically_writes_artifact_and_accepts_supplied_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    alpha = _write_metrics(tmp_path / "alpha.json", macro_f1=0.61)
    beta = _write_metrics(tmp_path / "beta.json", macro_f1=0.62)
    output = tmp_path / "nested" / "selection.json"
    digest = "aB" * 32

    exit_code = selector.main(
        [
            "--candidate",
            f"alpha={alpha}",
            "--candidate",
            f"beta={beta}",
            "--checkpoint-sha256",
            f"beta={digest}",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["selected_candidate"] == "beta"
    assert written["selected_checkpoint_sha256"] == digest.lower()
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    assert "Selected beta" in capsys.readouterr().out


def test_rejects_digest_mismatch_and_unknown_checkpoint_mapping(tmp_path: Path) -> None:
    alpha = _write_metrics(tmp_path / "alpha.json")
    beta = _write_metrics(tmp_path / "beta.json")
    checkpoint = tmp_path / "alpha.pth"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(selector.SelectionError, match="does not match"):
        selector.build_selection_artifact(
            {"alpha": alpha, "beta": beta},
            checkpoint_paths={"alpha": checkpoint},
            checkpoint_sha256={"alpha": "0" * 64},
        )
    with pytest.raises(selector.SelectionError, match="unknown candidate.*typo"):
        selector.build_selection_artifact(
            {"alpha": alpha, "beta": beta}, checkpoint_paths={"typo": checkpoint}
        )
