#!/usr/bin/env python3
"""Configure deterministic execution, then transparently run an inference CLI.

Bootstrap options must precede ``--``.  Every token after that separator is
forwarded byte-for-byte as a Python argument string to either the installed
``nnUNetv2_predict`` entry point or this repository's ``predict_joint`` entry
point.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import tomllib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pancreas_multitask.inference_determinism import (
    DETERMINISM_LOCK_PATH,
    DETERMINISM_LOCK_SHA256,
    assert_deterministic_inference,
    configure_deterministic_inference,
    determinism_lock_provenance,
    deterministic_inference_snapshot,
    reassert_deterministic_inference,
)

STOCK_MODULE = "nnunetv2.inference.predict_from_raw_data"
STOCK_ENTRY_POINT = "predict_entry_point"
STOCK_PREDICTOR = "nnUNetPredictor"
STOCK_PACKAGE = "nnunetv2"
STOCK_PACKAGE_VERSION = "2.8.1"
STOCK_SOURCE_SHA256 = (
    "c350e3202a7a67c3aef12e9206a744add442110ff8a4377c1f9640104b20a31f"
)
CANDIDATE_MODULE = "predict_joint"
CANDIDATE_ENTRY_POINT = "main"
AUDIT_SCHEMA_VERSION = 1


def _file_provenance(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        content = resolved.read_bytes()
    except OSError as error:
        raise RuntimeError(f"Cannot hash inference source {resolved}: {error}") from error
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _module_source_paths(module: ModuleType, *symbols: object) -> tuple[Path, ...]:
    candidates: list[str] = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str) and module_file:
        candidates.append(module_file)
    for symbol in symbols:
        try:
            source_path = inspect.getsourcefile(symbol)
        except (TypeError, OSError):
            source_path = None
        if source_path:
            candidates.append(source_path)
    resolved = {Path(value).expanduser().resolve() for value in candidates}
    if not resolved:
        raise RuntimeError(f"Cannot resolve source file for target module {module.__name__}")
    return tuple(sorted(resolved, key=lambda path: str(path)))


def _source_manifest(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {str(path.resolve()): _file_provenance(path) for path in paths}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(resolved)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _candidate_module() -> ModuleType:
    try:
        return importlib.import_module(CANDIDATE_MODULE)
    except ModuleNotFoundError as error:
        if error.name != CANDIDATE_MODULE:
            raise
    source_path = Path(__file__).resolve().with_name("predict_joint.py")
    module_name = "_deterministic_inference_predict_joint"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load candidate entry point from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _repository_package_version() -> str:
    pyproject_path = ROOT / "pyproject.toml"
    try:
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8")).get(
            "project"
        )
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"Cannot read repository metadata {pyproject_path}: {error}") from error
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"Repository metadata lacks project.version: {pyproject_path}")
    return version


def _load_target(
    mode: str,
) -> tuple[ModuleType, Callable[[], object], type[object] | None, dict[str, Any]]:
    if mode == "stock":
        module = importlib.import_module(STOCK_MODULE)
        package_version = importlib.metadata.version(STOCK_PACKAGE)
        if package_version != STOCK_PACKAGE_VERSION:
            raise RuntimeError(
                f"Stock {STOCK_PACKAGE} version must be {STOCK_PACKAGE_VERSION}, "
                f"got {package_version}"
            )
        target = getattr(module, STOCK_ENTRY_POINT, None)
        predictor_type = getattr(module, STOCK_PREDICTOR, None)
        if not callable(target) or not isinstance(predictor_type, type):
            raise RuntimeError("Installed stock nnU-Net entry point is incomplete")
        metadata = {
            "module": STOCK_MODULE,
            "entry_point": STOCK_ENTRY_POINT,
            "package": STOCK_PACKAGE,
            "package_version": package_version,
        }
        return module, target, predictor_type, metadata

    if mode == "candidate":
        module = _candidate_module()
        target = getattr(module, CANDIDATE_ENTRY_POINT, None)
        if not callable(target):
            raise RuntimeError("Repository candidate predict_joint.main is unavailable")
        metadata = {
            "module": CANDIDATE_MODULE,
            "entry_point": CANDIDATE_ENTRY_POINT,
            "package": "pancreas-multitask",
            "package_version": _repository_package_version(),
        }
        return module, target, None, metadata
    raise ValueError(f"Unsupported deterministic inference mode: {mode!r}")


def _target_device(mode: str, target_argv: Sequence[str]) -> torch.device:
    option = "-device" if mode == "stock" else "--device"
    value = "cuda"
    for index, token in enumerate(target_argv):
        if token == option:
            if index + 1 >= len(target_argv):
                raise ValueError(f"{option} requires a value")
            value = target_argv[index + 1]
        elif token.startswith(f"{option}="):
            value = token.split("=", 1)[1]
    if value not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported target inference device: {value!r}")
    return torch.device(value)


@contextmanager
def _forwarded_argv(program_name: str, target_argv: Sequence[str]) -> Iterator[None]:
    original = sys.argv
    sys.argv = [program_name, *target_argv]
    try:
        yield
    finally:
        sys.argv = original


def _patch_stock_constructor(
    predictor_type: type[object],
    device: torch.device,
    snapshots: list[dict[str, Any]],
) -> Callable[..., object]:
    original = predictor_type.__init__

    @functools.wraps(original)
    def deterministic_constructor(self: object, *args: object, **kwargs: object) -> None:
        original(self, *args, **kwargs)
        reassert_deterministic_inference()
        snapshot = deterministic_inference_snapshot(device)
        assert_deterministic_inference(snapshot)
        snapshots.append(snapshot)

    predictor_type.__init__ = deterministic_constructor
    return original


def _exception_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _system_exit_code(error: BaseException) -> int:
    if isinstance(error, SystemExit):
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
    if isinstance(error, KeyboardInterrupt):
        return 130
    return 1


def dispatch_deterministic_inference(
    mode: str,
    target_argv: Sequence[str],
    audit_path: Path,
) -> int:
    """Run one target entry point and always emit its deterministic audit."""

    started_at = datetime.now(UTC).isoformat()
    target_arguments = list(target_argv)
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "mode": mode,
        "target_argv": target_arguments,
        "process_id": os.getpid(),
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "device": None,
        "autocast_cuda_float16": None,
        "status": "failed",
        "exit_code": None,
        "exception": None,
        "postflight_exception": None,
        "target": None,
        "determinism_lock_expected": {
            "path": str(DETERMINISM_LOCK_PATH.resolve()),
            "sha256": DETERMINISM_LOCK_SHA256,
        },
        "determinism_lock_before": None,
        "determinism_lock_after": None,
        "determinism_lock_unchanged": False,
        "determinism_snapshots": {
            "after_initial_configuration": None,
            "after_predictor_construction": None,
            "after_inference": None,
        },
        "stock_constructor_reassertion_count": 0,
        "cuda_memory": None,
        "installed_sources_before": None,
        "installed_sources_after": None,
        "installed_sources_unchanged": False,
    }
    pending_error: BaseException | None = None
    pending_traceback = None
    exit_code = 1
    source_paths: tuple[Path, ...] = ()
    predictor_type: type[object] | None = None
    original_constructor: Callable[..., object] | None = None
    constructor_snapshots: list[dict[str, Any]] = []
    configured = False

    try:
        audit["determinism_lock_before"] = determinism_lock_provenance()
        device = _target_device(mode, target_arguments)
        audit["device"] = str(device)
        audit["autocast_cuda_float16"] = device.type == "cuda"
        initial_snapshot = configure_deterministic_inference(device)
        audit["determinism_snapshots"]["after_initial_configuration"] = initial_snapshot
        configured = True

        module, target, predictor_type, target_metadata = _load_target(mode)
        audit["target"] = target_metadata
        source_paths = _module_source_paths(
            module,
            target,
            predictor_type if predictor_type is not None else target,
        )
        sources_before = _source_manifest(source_paths)
        audit["installed_sources_before"] = sources_before
        if mode == "stock":
            module_path = str(Path(module.__file__).resolve())
            if sources_before[module_path]["sha256"] != STOCK_SOURCE_SHA256:
                raise RuntimeError(
                    "Installed stock nnUNet predict_from_raw_data source SHA-256 mismatch"
                )
            assert predictor_type is not None
            original_constructor = _patch_stock_constructor(
                predictor_type,
                device,
                constructor_snapshots,
            )

        if device.type == "cuda":
            audit["cuda_memory"] = {
                "unit": "MiB",
                "collector": "torch.cuda process-local memory counters",
                "bootstrap_reset_before_target": mode == "stock",
                "before_target": None,
                "after_target": None,
            }
            if mode == "stock":
                # Stock has no nested pre-CUDA configuration call. Candidate
                # does, so CUDA must remain uninitialized until candidate.run.
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                bytes_per_mib = 1024**2
                audit["cuda_memory"]["before_target"] = {
                    "allocated_mib": torch.cuda.memory_allocated(device) / bytes_per_mib,
                    "reserved_mib": torch.cuda.memory_reserved(device) / bytes_per_mib,
                }

        with _forwarded_argv(target_metadata["entry_point"], target_arguments):
            result = target()
        if result is None:
            exit_code = 0
        elif isinstance(result, int) and not isinstance(result, bool):
            exit_code = result
        else:
            raise RuntimeError(
                "Inference entry point must return an integer exit code or None"
            )
    except BaseException as error:  # noqa: BLE001 - audit every target exit
        pending_error = error
        pending_traceback = error.__traceback__
        exit_code = _system_exit_code(error)
    finally:
        if predictor_type is not None and original_constructor is not None:
            predictor_type.__init__ = original_constructor
        audit["stock_constructor_reassertion_count"] = len(constructor_snapshots)
        if constructor_snapshots:
            audit["determinism_snapshots"]["after_predictor_construction"] = (
                constructor_snapshots[-1]
            )

        if configured:
            try:
                configured_device = torch.device(audit["device"])
                if configured_device.type == "cuda":
                    torch.cuda.synchronize(configured_device)
                    bytes_per_mib = 1024**2
                    audit["cuda_memory"]["after_target"] = {
                        "allocated_mib": (
                            torch.cuda.memory_allocated(configured_device) / bytes_per_mib
                        ),
                        "reserved_mib": (
                            torch.cuda.memory_reserved(configured_device) / bytes_per_mib
                        ),
                        "peak_allocated_mib": (
                            torch.cuda.max_memory_allocated(configured_device)
                            / bytes_per_mib
                        ),
                        "peak_reserved_mib": (
                            torch.cuda.max_memory_reserved(configured_device)
                            / bytes_per_mib
                        ),
                    }
                final_snapshot = deterministic_inference_snapshot(
                    configured_device
                )
                audit["determinism_snapshots"]["after_inference"] = final_snapshot
                assert_deterministic_inference(final_snapshot)
            except BaseException as error:  # noqa: BLE001 - retain postflight evidence
                audit["postflight_exception"] = _exception_record(error)
                if pending_error is None:
                    pending_error = error
                    pending_traceback = error.__traceback__
                    exit_code = _system_exit_code(error)

        if source_paths:
            try:
                sources_after = _source_manifest(source_paths)
                audit["installed_sources_after"] = sources_after
                audit["installed_sources_unchanged"] = (
                    audit["installed_sources_before"] == sources_after
                )
                if not audit["installed_sources_unchanged"]:
                    raise RuntimeError("Inference target source changed during execution")
            except BaseException as error:  # noqa: BLE001 - retain postflight evidence
                if audit["postflight_exception"] is None:
                    audit["postflight_exception"] = _exception_record(error)
                if pending_error is None:
                    pending_error = error
                    pending_traceback = error.__traceback__
                    exit_code = _system_exit_code(error)

        try:
            audit["determinism_lock_after"] = determinism_lock_provenance()
            audit["determinism_lock_unchanged"] = (
                audit["determinism_lock_before"] == audit["determinism_lock_after"]
            )
            if not audit["determinism_lock_unchanged"]:
                raise RuntimeError("Deterministic inference lock changed during execution")
        except BaseException as error:  # noqa: BLE001 - retain postflight evidence
            if audit["postflight_exception"] is None:
                audit["postflight_exception"] = _exception_record(error)
            if pending_error is None:
                pending_error = error
                pending_traceback = error.__traceback__
                exit_code = _system_exit_code(error)

        successful_system_exit = (
            isinstance(pending_error, SystemExit) and exit_code == 0
        )
        audit["exit_code"] = exit_code
        audit["exception"] = (
            None
            if pending_error is None or successful_system_exit
            else _exception_record(pending_error)
        )
        audit["status"] = (
            "succeeded"
            if exit_code == 0 and (pending_error is None or successful_system_exit)
            else "failed"
        )
        audit["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json_atomic(audit_path, audit)

    if pending_error is not None:
        raise pending_error.with_traceback(pending_traceback)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stock", "candidate"), required=True)
    parser.add_argument(
        "--determinism-audit-json",
        type=Path,
        required=True,
        help="Atomic audit JSON written on success or target failure",
    )
    return parser


def _parse_arguments(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    values = list(argv)
    if "--" not in values:
        raise ValueError(
            "Bootstrap and target arguments must be separated by a literal --"
        )
    separator = values.index("--")
    bootstrap = build_parser().parse_args(values[:separator])
    return bootstrap, values[separator + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    bootstrap, target_argv = _parse_arguments(
        sys.argv[1:] if argv is None else argv
    )
    return dispatch_deterministic_inference(
        bootstrap.mode,
        target_argv,
        bootstrap.determinism_audit_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
