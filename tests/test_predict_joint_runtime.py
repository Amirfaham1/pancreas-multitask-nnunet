from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnunetv2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_cli_module():
    script_path = ROOT / "scripts" / "predict_joint.py"
    spec = importlib.util.spec_from_file_location("predict_joint_runtime_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_run_writes_end_to_end_runtime_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()
    events: list[str] = []
    initialized: dict[str, object] = {}
    predicted: dict[str, object] = {}

    class FakePredictor:
        def __init__(self, **kwargs: object) -> None:
            events.append("construct")
            initialized["constructor"] = kwargs

        def initialize_from_trained_model_folder(
            self,
            model: str,
            *,
            use_folds: tuple[int | str, ...] | None,
            checkpoint_name: str,
        ) -> None:
            events.append("initialize")
            initialized.update(
                model=model,
                use_folds=use_folds,
                checkpoint_name=checkpoint_name,
            )

        def reset_inference_runtime_counters(self) -> None:
            pass

        def inference_runtime_provenance(self) -> dict[str, object]:
            return {
                "joint_network_forward_calls": 8,
                "maximum_network_batch_size_observed": 2,
                "network_batch_size_histogram": {"2": 8},
                "network_batch_size_limit": 2,
                "logical_tile_batches_completed": 4,
                "logical_tiles_completed": 8,
                "tile_batch_oom_fallback_count": 0,
                "tile_batch_size_adaptive_limit": 2,
                "tile_batch_size_histogram": {"2": 4},
                "tile_batch_size_requested": 2,
                "tta_batch_oom_fallback_count": 0,
                "tta_batch_size_adaptive_limit": 2,
                "tta_batch_size_histogram": {"2": 4},
                "tta_batch_size_requested": 2,
                "tta_view_batches_completed": 4,
                "tta_views_completed": 8,
            }

        def predict_from_files_joint(self, *args: object, **kwargs: object) -> list[object]:
            events.append("predict_and_export")
            predicted["args"] = args
            predicted["kwargs"] = kwargs
            return [SimpleNamespace(case_id="case_b"), SimpleNamespace(case_id="case_a")]

    clock_values = iter((10.0, 14.0))

    def fake_perf_counter() -> float:
        events.append("clock")
        return next(clock_values)

    monkeypatch.setattr(module, "JointNNUNetPredictor", FakePredictor)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=fake_perf_counter))
    monkeypatch.setattr(module.torch, "set_num_threads", lambda _threads: None)

    runtime_path = tmp_path / "artifacts" / "runtime.json"
    for fold in (0, 1):
        fold_directory = tmp_path / "model" / f"fold_{fold}"
        fold_directory.mkdir(parents=True)
        (fold_directory / "checkpoint_final.pth").write_bytes(f"fold-{fold}".encode())
    args = module.build_parser().parse_args(
        [
            "--input",
            str(tmp_path / "input"),
            "--output",
            str(tmp_path / "output"),
            "--model",
            str(tmp_path / "model"),
            "--folds",
            "0",
            "1",
            "--checkpoint",
            "checkpoint_final.pth",
            "--runtime-json",
            str(runtime_path),
            "--device",
            "cpu",
            "--tile-step-size",
            "0.75",
            "--tile-batch-size",
            "2",
            "--tta-batch-size",
            "2",
            "--disable-tta",
            "--overwrite",
        ]
    )

    assert module.run(args) == 0

    assert events == ["construct", "clock", "initialize", "predict_and_export", "clock"]
    assert initialized["use_folds"] == (0, 1)
    assert initialized["checkpoint_name"] == "checkpoint_final.pth"
    assert initialized["constructor"]["tile_batch_size"] == 2
    assert initialized["constructor"]["tta_batch_size"] == 2
    assert predicted["args"] == (args.input, args.output)

    artifact = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert artifact["case_count"] == 2
    assert artifact["case_ids"] == ["case_a", "case_b"]
    assert artifact["case_ids_sha256"] == module._case_ids_sha256(
        ["case_a", "case_b"]
    )
    assert artifact["checkpoint"] == "checkpoint_final.pth"
    assert artifact["checkpoint_files"] == [
        {
            "fold": str(fold),
            "sha256": module._sha256(
                tmp_path / "model" / f"fold_{fold}" / "checkpoint_final.pth"
            ),
            "size_bytes": 6,
        }
        for fold in (0, 1)
    ]
    assert artifact["device"] == "cpu"
    assert artifact["folds"] == [0, 1]
    assert artifact["gaussian_enabled"] is True
    assert artifact["inference_execution"]["tile_batch_size_requested"] == 2
    assert artifact["mean_seconds_per_case"] == 2.0
    assert artifact["overwrite"] is True
    assert artifact["peak_allocated_mib"] is None
    assert artifact["peak_reserved_mib"] is None
    assert artifact["tile_step_size"] == 0.75
    assert artifact["timing_scope"] == (
        "fresh_process_model_initialization_preprocessing_inference_export"
    )
    assert artifact["total_seconds"] == 4.0
    assert artifact["tta_enabled"] is False
    assert artifact["warmup_policy"] == "none_fresh_process_end_to_end"
    assert not list(runtime_path.parent.glob(".runtime.json.*.tmp"))
    output = capsys.readouterr().out
    assert "Runtime: 4.00 s total, 2.00 s/case" in output
    assert f"Runtime JSON: {runtime_path.resolve()}" in output


def test_atomic_runtime_writer_replaces_existing_json(tmp_path: Path) -> None:
    module = _load_cli_module()
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text('{"stale": true}\n', encoding="utf-8")

    module._write_json_atomic(runtime_path, {"case_count": 0})

    assert json.loads(runtime_path.read_text(encoding="utf-8")) == {"case_count": 0}
    assert not list(tmp_path.glob(".runtime.json.*.tmp"))
