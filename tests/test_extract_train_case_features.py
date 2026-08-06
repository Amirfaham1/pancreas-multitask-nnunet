from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_train_case_features.py"
SPEC = importlib.util.spec_from_file_location("extract_train_case_features", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_prospective_lock_hash_constants_match_committed_files() -> None:
    MODULE._validate_static_lock_hashes(
        ROOT / "configs" / "phd_classification_upgrade_lock_v3.json",
        ROOT / "configs" / "phd_neural_case_head_lock_v5.json",
        ROOT / "configs" / "phd_neural_decision_lock_v5.json",
    )


def _write_cache(path: Path, binding: dict[str, object]) -> None:
    MODULE._atomic_savez(
        path,
        case_id=np.asarray("opaque"),
        label=np.asarray(1, dtype=np.int64),
        feature_view_0=np.asarray([1.0], dtype=np.float32),
        feature_view_1=np.asarray([2.0], dtype=np.float32),
        tile_vectors=np.ones((1, 2), dtype=np.float32),
        tile_evidence=np.ones((1, 7), dtype=np.float32),
        mil_stage3_maps=np.ones((1, 256, 4, 4, 6), dtype=np.float16),
        mil_prediction_maps=np.ones((1, 2, 4, 4, 6), dtype=np.float16),
        mil_lesion_mass=np.ones(1, dtype=np.float32),
        **{f"binding_{key}": np.asarray(value) for key, value in binding.items()},
    )


def test_cache_resume_requires_every_exact_provenance_binding(tmp_path: Path) -> None:
    binding = {
        "checkpoint_sha256": "checkpoint",
        "implementation_sha256": "implementation",
        "tile_step_size": 0.5,
        "tile_batch_size": 1,
        "tta_enabled": True,
    }
    complete = tmp_path / "complete.npz"
    _write_cache(complete, binding)

    views, mil = MODULE._load_case_cache(
        complete,
        case_id="opaque",
        label=1,
        view_names=("first", "second"),
        expected_dimensions={"first": 1, "second": 1},
        expected_tile_feature_count=2,
        expected_binding=binding,
    )
    assert views["first"].tolist() == [1.0]
    assert mil["mil_stage3_maps"].shape == (1, 256, 4, 4, 6)

    missing = tmp_path / "missing.npz"
    _write_cache(missing, {key: value for key, value in binding.items() if key != "tta_enabled"})
    with pytest.raises(ValueError, match="missing provenance binding"):
        MODULE._load_case_cache(
            missing,
            case_id="opaque",
            label=1,
            view_names=("first", "second"),
            expected_dimensions={"first": 1, "second": 1},
            expected_tile_feature_count=2,
            expected_binding=binding,
        )


def test_cache_resume_rejects_wrong_bound_value(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    _write_cache(path, {"tile_batch_size": 1})

    with pytest.raises(ValueError, match="provenance mismatch"):
        MODULE._load_case_cache(
            path,
            case_id="opaque",
            label=1,
            view_names=("first", "second"),
            expected_dimensions={"first": 1, "second": 1},
            expected_tile_feature_count=2,
            expected_binding={"tile_batch_size": 2},
        )
