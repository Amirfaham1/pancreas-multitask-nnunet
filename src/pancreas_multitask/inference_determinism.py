"""Fail-closed deterministic settings for nnU-Net inference.

The initial configuration must run before CUDA is initialized because cuBLAS
reads ``CUBLAS_WORKSPACE_CONFIG`` during CUDA startup.  nnU-Net's stock
predictor subsequently enables cuDNN benchmarking in its constructor, so the
lighter-weight reassertion helper is intentionally safe to call after CUDA
initialization as well.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_NNUNET_COMPILE = "false"
DETERMINISM_LOCK_SHA256 = (
    "33b5aed4027651f999875e2340a65173c620c5673f845a186115cd3a7adb1ddd"
)
DETERMINISM_LOCK_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "inference_determinism_conformance_v1.json"
)

_EXPECTED_SNAPSHOT: dict[str, object] = {
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "cublas_workspace_config": _CUBLAS_WORKSPACE_CONFIG,
    "nnunet_compile": _NNUNET_COMPILE,
}


def determinism_lock_provenance() -> dict[str, Any]:
    """Resolve and verify the frozen deterministic-conformance lock."""

    path = DETERMINISM_LOCK_PATH.resolve()
    try:
        content = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"Cannot read deterministic inference lock {path}: {error}") from error
    digest = hashlib.sha256(content).hexdigest()
    if digest != DETERMINISM_LOCK_SHA256:
        raise RuntimeError(
            "Deterministic inference lock SHA-256 mismatch: "
            f"expected {DETERMINISM_LOCK_SHA256}, got {digest}"
        )
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": len(content),
    }


def deterministic_inference_snapshot(device: torch.device) -> dict[str, Any]:
    """Return the complete locked inference-determinism state.

    ``device`` is accepted deliberately so callers bind the snapshot to the
    same device decision used for configuration.  Reading these flags does not
    query CUDA availability or initialize a CUDA context.
    """

    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    return {
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nnunet_compile": os.environ.get("nnUNet_compile"),
    }


def assert_deterministic_inference(snapshot: dict[str, Any]) -> None:
    """Raise if an audit snapshot differs from the locked policy."""

    if not isinstance(snapshot, dict):
        raise TypeError("deterministic inference snapshot must be a dictionary")
    missing = sorted(set(_EXPECTED_SNAPSHOT) - set(snapshot))
    mismatches = {
        key: {"expected": expected, "observed": snapshot.get(key)}
        for key, expected in _EXPECTED_SNAPSHOT.items()
        if key in snapshot
        and (
            not isinstance(snapshot[key], expected.__class__)
            or snapshot[key] != expected
        )
    }
    if missing or mismatches:
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {missing}")
        if mismatches:
            details.append(f"mismatched values: {mismatches}")
        raise RuntimeError(
            "Deterministic inference policy is not active (" + "; ".join(details) + ")"
        )


def reassert_deterministic_inference() -> dict[str, Any]:
    """Reapply the policy without requiring an uninitialized CUDA runtime.

    This is used immediately after constructors in third-party code that
    mutate backend flags.  The initial call must still be made through
    :func:`configure_deterministic_inference` before any CUDA initialization.
    """

    determinism_lock_provenance()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    os.environ["nnUNet_compile"] = _NNUNET_COMPILE
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = deterministic_inference_snapshot(torch.device("cpu"))
    assert_deterministic_inference(snapshot)
    return snapshot


def configure_deterministic_inference(device: torch.device) -> dict[str, Any]:
    """Configure deterministic inference before the first CUDA query.

    A pre-existing, conflicting cuBLAS workspace value is rejected instead of
    silently changing the intended execution contract.  CUDA availability is
    checked only after every deterministic setting has been applied.
    """

    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    determinism_lock_provenance()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, _CUBLAS_WORKSPACE_CONFIG):
        raise ValueError(
            "Deterministic inference requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "Deterministic inference must be configured before CUDA initialization"
        )

    reassert_deterministic_inference()
    snapshot = deterministic_inference_snapshot(device)
    assert_deterministic_inference(snapshot)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA deterministic inference was requested but is unavailable")
    return snapshot


__all__ = [
    "DETERMINISM_LOCK_PATH",
    "DETERMINISM_LOCK_SHA256",
    "assert_deterministic_inference",
    "configure_deterministic_inference",
    "determinism_lock_provenance",
    "deterministic_inference_snapshot",
    "reassert_deterministic_inference",
]
