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

        def predict_from_files_joint(self, *args: object, **kwargs: object) -> list[object]:
            events.append("predict_and_export")
            predicted["args"] = args
            predicted["kwargs"] = kwargs
            return [object(), object()]

    clock_values = iter((10.0, 14.0))

    def fake_perf_counter() -> float:
        events.append("clock")
        return next(clock_values)

    monkeypatch.setattr(module, "JointNNUNetPredictor", FakePredictor)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=fake_perf_counter))
    monkeypatch.setattr(module.torch, "set_num_threads", lambda _threads: None)

    runtime_path = tmp_path / "artifacts" / "runtime.json"
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
            "--disable-tta",
        ]
    )

    assert module.run(args) == 0

    assert events == ["construct", "clock", "initialize", "predict_and_export", "clock"]
    assert initialized["use_folds"] == (0, 1)
    assert initialized["checkpoint_name"] == "checkpoint_final.pth"
    assert predicted["args"] == (args.input, args.output)

    artifact = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert artifact == {
        "case_count": 2,
        "checkpoint": "checkpoint_final.pth",
        "device": "cpu",
        "folds": [0, 1],
        "gaussian_enabled": True,
        "mean_seconds_per_case": 2.0,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "tile_step_size": 0.75,
        "total_seconds": 4.0,
        "tta_enabled": False,
    }
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
