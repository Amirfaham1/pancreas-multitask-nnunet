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


def test_cpu_v5_run_times_head_load_and_writes_strict_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module()
    events: list[str] = []
    initialized: dict[str, object] = {}
    bundle_path = tmp_path / "final_neural_case_head.pth"
    bundle_path.write_bytes(b"locked-v5-bundle")
    bundle_sha256 = "a" * 64
    numeric_dataset_sha256 = "b" * 64
    bag_sha256 = "c" * 64
    component_hashes = {
        "encoder": "d" * 64,
        "decoder": "e" * 64,
        "classification": "f" * 64,
    }

    class FakeNeuralPredictor:
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

        def load_final_neural_case_head(self) -> None:
            events.append("load_head")

        def reset_inference_runtime_counters(self) -> None:
            pass

        def predict_from_files_joint(self, *args: object, **kwargs: object) -> list[object]:
            events.append("predict_and_export")
            return [SimpleNamespace(case_id="case_a")]

        def inference_runtime_provenance(self) -> dict[str, object]:
            return {
                "joint_network_forward_calls": 8,
                "maximum_network_batch_size_observed": 1,
                "network_batch_size_histogram": {"1": 8},
                "network_batch_size_limit": 1,
                "logical_tile_batches_completed": 1,
                "logical_tiles_completed": 1,
                "tile_batch_oom_fallback_count": 0,
                "tile_batch_size_adaptive_limit": 1,
                "tile_batch_size_histogram": {"1": 1},
                "tile_batch_size_requested": 1,
                "tta_batch_oom_fallback_count": 0,
                "tta_batch_size_adaptive_limit": 1,
                "tta_batch_size_histogram": {"1": 8},
                "tta_batch_size_requested": 1,
                "tta_view_batches_completed": 8,
                "tta_views_completed": 8,
                "classifier_pipeline": module.V5_CLASSIFIER_PIPELINE,
                "v5_extraction_mode": "neural_only",
                "speed_v3_network_batch_ceiling": 1,
                "v5_feature_extraction_executed": True,
                "v5_case_extractions_completed": 1,
                "v5_neural_head_forward_calls": 1,
                "v5_class_offset_applications": 1,
                "v5_feature_cache_reads": 0,
                "case_identifiers_or_paths_used_as_model_inputs": False,
                "v5_neural_bag_sha256_sequence": [bag_sha256],
            }

        def neural_case_head_provenance(self) -> dict[str, object]:
            return {
                "classifier_pipeline": module.V5_CLASSIFIER_PIPELINE,
                "bundle_path": str(bundle_path.resolve()),
                "bundle_name": bundle_path.name,
                "bundle_sha256": bundle_sha256,
                "bundle_size_bytes": bundle_path.stat().st_size,
                "expected_bundle_sha256_verified": True,
                "numeric_train_dataset_sha256": numeric_dataset_sha256,
                "selected_candidate_id": "mean_v1",
                "head_parameter_count": 17,
                "head_in_eval_mode": True,
                "any_head_parameter_requires_grad": False,
                "head_state_sha256": "1" * 64,
                "head_state_sha256_before": "1" * 64,
                "head_state_sha256_after": "1" * 64,
                "head_state_unchanged": True,
                "class_offsets": [0.1, 0.0, -0.1],
                "neural_lock_sha256": "2" * 64,
                "decision_lock_sha256": "3" * 64,
                "selection_audit_sha256": "4" * 64,
                "calibration_audit_sha256": "5" * 64,
                "refit_audit_sha256": "6" * 64,
                "eligible_for_official": True,
                "bundle_loaded_strictly": True,
            }

        def frozen_network_provenance(self) -> dict[str, object]:
            return {
                "fold": 0,
                "component_hashes_before": component_hashes,
                "component_hashes_after": component_hashes,
                "frozen_components_unchanged": True,
                "network_in_eval_mode": True,
                "any_network_parameter_requires_grad": False,
            }

    clock_values = iter((10.0, 14.0))

    def fake_perf_counter() -> float:
        events.append("clock")
        return next(clock_values)

    monkeypatch.setattr(module, "NeuralCaseNNUNetPredictor", FakeNeuralPredictor)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=fake_perf_counter))
    monkeypatch.setattr(module.torch, "set_num_threads", lambda _threads: None)

    model_directory = tmp_path / "model"
    checkpoint_directory = model_directory / "fold_0"
    checkpoint_directory.mkdir(parents=True)
    checkpoint_path = checkpoint_directory / "checkpoint_classification_rescue.pth"
    checkpoint_path.write_bytes(b"fold-0")
    (model_directory / "dataset.json").write_text("{}\n", encoding="utf-8")
    (model_directory / "plans.json").write_text("{}\n", encoding="utf-8")
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "case_a_0000.nii.gz").write_bytes(b"raw-volume")
    monkeypatch.setattr(module, "CHECKPOINT_SHA256", module._sha256(checkpoint_path))
    monkeypatch.setattr(
        module,
        "V5_MODEL_CONFIGURATION_SHA256",
        {
            "dataset.json": module._sha256(model_directory / "dataset.json"),
            "plans.json": module._sha256(model_directory / "plans.json"),
        },
    )
    runtime_path = tmp_path / "runtime.json"
    args = module.build_parser().parse_args(
        [
            "--input",
            str(input_directory),
            "--output",
            str(tmp_path / "output"),
            "--model",
            str(model_directory),
            "--folds",
            "0",
            "--checkpoint",
            "checkpoint_classification_rescue.pth",
            "--classification-mode",
            "neural-v5",
            "--neural-case-head-bundle",
            str(bundle_path),
            "--expected-neural-case-head-bundle-sha256",
            bundle_sha256,
            "--expected-numeric-train-dataset-sha256",
            numeric_dataset_sha256,
            "--v5-extraction-mode",
            "neural_only",
            "--runtime-json",
            str(runtime_path),
            "--device",
            "cpu",
            "--tile-step-size",
            "0.5",
            "--tile-batch-size",
            "1",
            "--tta-batch-size",
            "1",
            "--overwrite",
        ]
    )

    assert module.run(args) == 0

    assert events == [
        "construct",
        "clock",
        "initialize",
        "load_head",
        "predict_and_export",
        "clock",
    ]
    assert initialized["use_folds"] == (0,)
    constructor = initialized["constructor"]
    assert constructor["neural_case_head_bundle"] == bundle_path
    assert constructor["expected_neural_case_head_bundle_sha256"] == bundle_sha256
    assert constructor["expected_numeric_train_dataset_sha256"] == numeric_dataset_sha256
    assert constructor["v5_extraction_mode"] == "neural_only"

    artifact = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert artifact["classifier_pipeline"] == module.V5_CLASSIFIER_PIPELINE
    assert artifact["class_probabilities"] == "v5_offset_adjusted_three_class"
    assert artifact["feature_cache_policy"] == "disabled_online_fresh_extraction"
    assert artifact["v5_extraction_mode"] == "neural_only"
    assert artifact["v5_neural_bag_sha256_sequence"] == [bag_sha256]
    assert artifact["case_identifiers_or_paths_used_as_model_inputs"] is False
    assert artifact["checkpoint_unchanged_during_run"] is True
    assert artifact["input_files_unchanged_during_run"] is True
    assert artifact["input_file_manifest"]["file_count"] == 1
    assert artifact["input_file_manifest"]["files"][0]["name"] == (
        "case_a_0000.nii.gz"
    )
    assert artifact["model_configuration_unchanged_during_run"] is True
    assert artifact["neural_case_head_bundle"]["bundle_sha256"] == bundle_sha256
    assert artifact["frozen_network"]["component_hashes_before"] == component_hashes
    assert artifact["timing_scope"] == (
        "fresh_process_model_and_v5_head_initialization_preprocessing_"
        "feature_extraction_neural_head_offsets_export"
    )
    assert set(artifact["v5_implementation_files"]) == {
        "scripts/predict_joint.py",
        "src/pancreas_multitask/classification_rescue.py",
        "src/pancreas_multitask/network.py",
        "src/pancreas_multitask/predictor.py",
        "src/pancreas_multitask/case_features.py",
        "src/pancreas_multitask/neural_case_predictor.py",
        "src/pancreas_multitask/case_feature_extractor.py",
        "src/pancreas_multitask/neural_case_bundle.py",
        "src/pancreas_multitask/neural_case_head.py",
        "src/pancreas_multitask/neural_case_training.py",
    }


@pytest.mark.parametrize(
    ("extra_arguments", "message"),
    [
        ([], "bundle path"),
        (["--folds", "0", "1"], "explicit numeric fold 0"),
        (["--tile-batch-size", "2"], "tile1/TTA1"),
        (["--disable-tta"], "mirror TTA and Gaussian weighting"),
        (["--no-overwrite"], "requires --overwrite"),
    ],
)
def test_v5_arguments_fail_closed(
    tmp_path: Path,
    extra_arguments: list[str],
    message: str,
) -> None:
    module = _load_cli_module()
    bundle = tmp_path / "bundle.pth"
    bundle.write_bytes(b"bundle")
    base = [
        "--input",
        str(tmp_path / "input"),
        "--output",
        str(tmp_path / "output"),
        "--model",
        str(tmp_path / "model"),
        "--folds",
        "0",
        "--classification-mode",
        "neural-v5",
        "--checkpoint",
        "checkpoint_classification_rescue.pth",
        "--neural-case-head-bundle",
        str(bundle),
        "--expected-neural-case-head-bundle-sha256",
        "a" * 64,
        "--expected-numeric-train-dataset-sha256",
        "b" * 64,
        "--v5-extraction-mode",
        "full",
        "--overwrite",
    ]
    if not extra_arguments:
        missing_index = base.index("--neural-case-head-bundle")
        del base[missing_index : missing_index + 2]
    args = module.build_parser().parse_args([*base, *extra_arguments])
    selected_folds = module._selected_folds(args.folds)

    with pytest.raises(ValueError, match=message):
        module._validate_v5_arguments(args, selected_folds)
