from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from pancreas_multitask import inference_determinism as determinism

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LOCK_SHA256 = (
    "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd"
)
EXPECTED_SNAPSHOT = {
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "cublas_workspace_config": ":4096:8",
    "nnunet_compile": "false",
}


def _load_bootstrap_module():
    path = ROOT / "scripts" / "run_deterministic_inference.py"
    spec = importlib.util.spec_from_file_location("run_deterministic_inference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_global_torch_policy():
    environment = {
        name: os.environ.get(name)
        for name in ("CUBLAS_WORKSPACE_CONFIG", "nnUNet_compile")
    }
    torch_policy = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    yield
    torch.use_deterministic_algorithms(
        torch_policy["deterministic"], warn_only=torch_policy["warn_only"]
    )
    torch.backends.cudnn.benchmark = torch_policy["benchmark"]
    torch.backends.cudnn.deterministic = torch_policy["cudnn_deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = torch_policy["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = torch_policy["cudnn_tf32"]
    for name, value in environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _set_wrong_policy() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["nnUNet_compile"] = "true"
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _fake_target_module(
    tmp_path: Path,
    module_name: str,
    entry_point,
    *,
    predictor_type=None,
) -> tuple[ModuleType, Path]:
    source = tmp_path / f"{module_name.rsplit('.', 1)[-1]}.py"
    source.write_text("# immutable fake installed source\n", encoding="utf-8")
    module = ModuleType(module_name)
    module.__file__ = str(source)
    module.predict_entry_point = entry_point
    module.main = entry_point
    if predictor_type is not None:
        module.nnUNetPredictor = predictor_type
    return module, source


def _patch_metadata_version(monkeypatch: pytest.MonkeyPatch, bootstrap) -> None:
    def version(name: str) -> str:
        if name == "nnunetv2":
            return "2.8.1"
        if name == "pancreas-multitask":
            return "0.1.0"
        raise bootstrap.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(bootstrap.importlib.metadata, "version", version)


def test_frozen_determinism_lock_is_exact_and_any_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = determinism.determinism_lock_provenance()
    assert provenance["path"] == str(
        (ROOT / "configs" / "inference_determinism_conformance_v1.json").resolve()
    )
    assert provenance["sha256"] == EXPECTED_LOCK_SHA256
    assert determinism.DETERMINISM_LOCK_SHA256 == EXPECTED_LOCK_SHA256

    changed_lock = tmp_path / "changed-lock.json"
    changed_lock.write_bytes(Path(provenance["path"]).read_bytes() + b"\n")
    monkeypatch.setattr(determinism, "DETERMINISM_LOCK_PATH", changed_lock)
    with pytest.raises(RuntimeError, match="lock SHA-256 mismatch"):
        determinism.determinism_lock_provenance()


def test_configure_rejects_conflicting_workspace_without_querying_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: pytest.fail("CUDA initialization must not be queried after conflict"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("CUDA availability must not be queried after conflict"),
    )

    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        determinism.configure_deterministic_inference(torch.device("cuda"))


def test_configure_requires_pre_cuda_call_before_mutating_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("nnUNet_compile", raising=False)
    torch.backends.cudnn.benchmark = True
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("availability cannot be queried after initialized CUDA"),
    )

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        determinism.configure_deterministic_inference(torch.device("cuda"))
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert "nnUNet_compile" not in os.environ
    assert torch.backends.cudnn.benchmark is True


def test_configure_sets_every_flag_before_cuda_availability_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("nnUNet_compile", "true")
    _set_wrong_policy()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    availability_calls: list[dict[str, object]] = []

    def is_available() -> bool:
        snapshot = determinism.deterministic_inference_snapshot(torch.device("cuda"))
        determinism.assert_deterministic_inference(snapshot)
        availability_calls.append(snapshot)
        return True

    monkeypatch.setattr(torch.cuda, "is_available", is_available)
    snapshot = determinism.configure_deterministic_inference(torch.device("cuda"))

    assert snapshot == EXPECTED_SNAPSHOT
    assert availability_calls == [EXPECTED_SNAPSHOT]


def test_reassert_is_safe_after_initialization_and_never_queries_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_wrong_policy()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("post-constructor reassertion cannot query availability"),
    )

    snapshot = determinism.reassert_deterministic_inference()

    assert snapshot == EXPECTED_SNAPSHOT


@pytest.mark.parametrize("field", tuple(EXPECTED_SNAPSHOT))
def test_assertion_rejects_every_policy_field(field: str) -> None:
    snapshot = dict(EXPECTED_SNAPSHOT)
    current = snapshot[field]
    snapshot[field] = (not current) if isinstance(current, bool) else "wrong"

    with pytest.raises(RuntimeError, match="Deterministic inference policy"):
        determinism.assert_deterministic_inference(snapshot)


def test_assertion_rejects_missing_policy_field() -> None:
    snapshot = dict(EXPECTED_SNAPSHOT)
    snapshot.pop("cudnn_benchmark")
    with pytest.raises(RuntimeError, match="missing keys"):
        determinism.assert_deterministic_inference(snapshot)


def test_assertion_rejects_integer_stand_in_for_boolean() -> None:
    snapshot = dict(EXPECTED_SNAPSHOT)
    snapshot["torch_deterministic_algorithms"] = 1
    with pytest.raises(RuntimeError, match="mismatched values"):
        determinism.assert_deterministic_inference(snapshot)


def test_stock_bootstrap_neutralizes_constructor_override_and_preserves_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _load_bootstrap_module()
    _patch_metadata_version(monkeypatch, bootstrap)
    observations: dict[str, object] = {}

    class FakeStockPredictor:
        def __init__(self) -> None:
            torch.backends.cudnn.benchmark = True

    original_constructor = FakeStockPredictor.__init__

    def predict_entry_point() -> None:
        observations["argv"] = list(sys.argv)
        FakeStockPredictor()
        observations["snapshot_after_constructor"] = (
            determinism.deterministic_inference_snapshot(torch.device("cpu"))
        )

    fake_module, source = _fake_target_module(
        tmp_path,
        bootstrap.STOCK_MODULE,
        predict_entry_point,
        predictor_type=FakeStockPredictor,
    )
    real_import_module = bootstrap.importlib.import_module

    def import_module(name: str):
        if name == bootstrap.STOCK_MODULE:
            return fake_module
        return real_import_module(name)

    monkeypatch.setattr(bootstrap.importlib, "import_module", import_module)
    monkeypatch.setattr(
        bootstrap,
        "STOCK_SOURCE_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    audit_path = tmp_path / "stock-audit.json"
    original_argv = list(sys.argv)
    target_argv = ["-i", "input with spaces", "-device", "cpu", "--verbose"]

    result = bootstrap.dispatch_deterministic_inference(
        "stock", target_argv, audit_path
    )

    assert result == 0
    assert observations["argv"] == [bootstrap.STOCK_ENTRY_POINT, *target_argv]
    assert observations["snapshot_after_constructor"] == EXPECTED_SNAPSHOT
    assert sys.argv == original_argv
    assert FakeStockPredictor.__init__ is original_constructor
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "succeeded"
    assert audit["target_argv"] == target_argv
    assert audit["stock_constructor_reassertion_count"] == 1
    assert audit["determinism_snapshots"]["after_initial_configuration"] == (
        EXPECTED_SNAPSHOT
    )
    assert audit["determinism_snapshots"]["after_predictor_construction"] == (
        EXPECTED_SNAPSHOT
    )
    assert audit["determinism_snapshots"]["after_inference"] == EXPECTED_SNAPSHOT
    assert audit["installed_sources_before"] == audit["installed_sources_after"]
    assert audit["installed_sources_unchanged"] is True
    assert audit["determinism_lock_unchanged"] is True
    assert audit["determinism_lock_before"]["sha256"] == EXPECTED_LOCK_SHA256


def test_candidate_bootstrap_dispatches_repository_main_with_exact_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _load_bootstrap_module()
    monkeypatch.setattr(
        bootstrap.importlib.metadata,
        "version",
        lambda _name: pytest.fail(
            "repository candidate cannot depend on installed package metadata"
        ),
    )
    observations: dict[str, object] = {}

    def candidate_main() -> int:
        observations["argv"] = list(sys.argv)
        observations["snapshot"] = determinism.deterministic_inference_snapshot(
            torch.device("cpu")
        )
        return 0

    fake_module, _ = _fake_target_module(
        tmp_path, bootstrap.CANDIDATE_MODULE, candidate_main
    )
    monkeypatch.setattr(bootstrap, "_candidate_module", lambda: fake_module)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    audit_path = tmp_path / "candidate-audit.json"
    target_argv = ["--input", "input", "--device=cpu", "--overwrite"]

    assert (
        bootstrap.dispatch_deterministic_inference(
            "candidate", target_argv, audit_path
        )
        == 0
    )
    assert observations["argv"] == [bootstrap.CANDIDATE_ENTRY_POINT, *target_argv]
    assert observations["snapshot"] == EXPECTED_SNAPSHOT
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["mode"] == "candidate"
    assert audit["target"]["package_version"] == "0.1.0"
    assert audit["stock_constructor_reassertion_count"] == 0
    assert audit["determinism_snapshots"]["after_predictor_construction"] is None
    assert audit["installed_sources_unchanged"] is True


def test_candidate_bootstrap_never_initializes_cuda_before_nested_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _load_bootstrap_module()
    observations = {"target_started": False, "availability_calls": 0}

    def candidate_main() -> int:
        observations["target_started"] = True
        nested = determinism.configure_deterministic_inference(torch.device("cuda"))
        assert nested == EXPECTED_SNAPSHOT
        return 0

    fake_module, _ = _fake_target_module(
        tmp_path, bootstrap.CANDIDATE_MODULE, candidate_main
    )
    monkeypatch.setattr(bootstrap, "_candidate_module", lambda: fake_module)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def is_available() -> bool:
        observations["availability_calls"] += 1
        return True

    monkeypatch.setattr(torch.cuda, "is_available", is_available)

    def synchronize(_device) -> None:
        assert observations["target_started"] is True

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda _device: pytest.fail("candidate bootstrap cannot reset before target"),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 10 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 20 * 1024**2)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 100 * 1024**2)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 120 * 1024**2)
    audit_path = tmp_path / "candidate-cuda-audit.json"

    assert (
        bootstrap.dispatch_deterministic_inference(
            "candidate", ["--device", "cuda"], audit_path
        )
        == 0
    )

    assert observations["availability_calls"] == 2
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["cuda_memory"] == {
        "unit": "MiB",
        "collector": "torch.cuda process-local memory counters",
        "bootstrap_reset_before_target": False,
        "before_target": None,
        "after_target": {
            "allocated_mib": 10.0,
            "reserved_mib": 20.0,
            "peak_allocated_mib": 100.0,
            "peak_reserved_mib": 120.0,
        },
    }


def test_bootstrap_writes_fail_closed_audit_when_target_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _load_bootstrap_module()
    _patch_metadata_version(monkeypatch, bootstrap)

    class TargetFailure(RuntimeError):
        pass

    def failing_main() -> int:
        raise TargetFailure("synthetic target failure")

    fake_module, _ = _fake_target_module(
        tmp_path, bootstrap.CANDIDATE_MODULE, failing_main
    )
    monkeypatch.setattr(bootstrap, "_candidate_module", lambda: fake_module)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    audit_path = tmp_path / "failed-audit.json"

    with pytest.raises(TargetFailure, match="synthetic target failure"):
        bootstrap.dispatch_deterministic_inference(
            "candidate", ["--device", "cpu"], audit_path
        )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["exit_code"] == 1
    assert audit["exception"] == {
        "type": "TargetFailure",
        "message": "synthetic target failure",
    }
    assert audit["determinism_snapshots"]["after_inference"] == EXPECTED_SNAPSHOT
    assert audit["installed_sources_unchanged"] is True


def test_bootstrap_detects_source_mutation_during_target_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _load_bootstrap_module()
    _patch_metadata_version(monkeypatch, bootstrap)
    source_holder: dict[str, Path] = {}

    def mutating_main() -> int:
        source_holder["path"].write_text("# source changed\n", encoding="utf-8")
        return 0

    fake_module, source = _fake_target_module(
        tmp_path, bootstrap.CANDIDATE_MODULE, mutating_main
    )
    source_holder["path"] = source
    monkeypatch.setattr(bootstrap, "_candidate_module", lambda: fake_module)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    audit_path = tmp_path / "changed-source-audit.json"

    with pytest.raises(RuntimeError, match="source changed during execution"):
        bootstrap.dispatch_deterministic_inference(
            "candidate", ["--device", "cpu"], audit_path
        )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["installed_sources_unchanged"] is False
    assert audit["installed_sources_before"] != audit["installed_sources_after"]


def test_bootstrap_writes_audit_when_initial_configuration_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _load_bootstrap_module()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    audit_path = tmp_path / "configuration-failure.json"

    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        bootstrap.dispatch_deterministic_inference(
            "candidate", ["--device", "cpu"], audit_path
        )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["target"] is None
    assert audit["determinism_snapshots"]["after_initial_configuration"] is None
    assert audit["determinism_lock_unchanged"] is True


def test_bootstrap_parser_requires_separator_and_preserves_every_target_token() -> None:
    bootstrap = _load_bootstrap_module()
    audit_path = Path("audit.json")
    arguments, target = bootstrap._parse_arguments(
        [
            "--mode",
            "stock",
            "--determinism-audit-json",
            str(audit_path),
            "--",
            "-i",
            "space path",
            "--",
            "literal-after-second-separator",
        ]
    )
    assert arguments.mode == "stock"
    assert arguments.determinism_audit_json == audit_path
    assert target == [
        "-i",
        "space path",
        "--",
        "literal-after-second-separator",
    ]

    with pytest.raises(ValueError, match="literal --"):
        bootstrap._parse_arguments(
            ["--mode", "stock", "--determinism-audit-json", str(audit_path)]
        )


def test_successful_target_system_exit_is_audited_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _load_bootstrap_module()
    _patch_metadata_version(monkeypatch, bootstrap)

    def help_exit() -> None:
        raise SystemExit(0)

    fake_module, _ = _fake_target_module(
        tmp_path, bootstrap.CANDIDATE_MODULE, help_exit
    )
    monkeypatch.setattr(bootstrap, "_candidate_module", lambda: fake_module)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    audit_path = tmp_path / "help-audit.json"

    with pytest.raises(SystemExit) as error:
        bootstrap.dispatch_deterministic_inference(
            "candidate", ["--device", "cpu", "--help"], audit_path
        )
    assert error.value.code == 0
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "succeeded"
    assert audit["exit_code"] == 0
    assert audit["exception"] is None
