from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_v7_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_v7_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_snap_repairs_only_small_integer_encoding_noise() -> None:
    module = _module()
    repaired, changed = module._snap_reference(
        np.asarray([0.0, 1.0000152587890625, 2.0], dtype=np.float64)
    )
    np.testing.assert_array_equal(repaired, np.asarray([0, 1, 2], dtype=np.uint8))
    assert changed is True

    with pytest.raises(ValueError, match="not within"):
        module._snap_reference(np.asarray([0.0, 1.01, 2.0]))


def test_legacy_speed_json_without_output_audit_cannot_pass(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "speed.json"
    path.write_text(
        '{"stock_seconds":[100],"candidate_seconds":[80],"stock_mean":100,'
        '"candidate_mean":80,"runtime_reduction_percent":20}',
        encoding="utf-8",
    )

    result = module.verify_speed(path)

    assert result["arithmetic_matches"] is True
    assert result["complete_output_audit_present"] is False
    assert result["passed"] is False
