from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_speed_abba.py"
ARM_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_inference_arm.py"


def _module(path: Path = SCRIPT):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_mask(path: Path, array: np.ndarray) -> None:
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.SetSpacing((1.0, 1.1, 1.2))
    sitk.WriteImage(image, str(path))


def test_mask_comparison_counts_real_voxel_disagreement(tmp_path: Path) -> None:
    module = _module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask[2, 2, 2] = 2
    _write_mask(first / "case.nii.gz", mask)
    _write_mask(second / "case.nii.gz", mask)

    exact = module.compare_mask_directories(first, second, ["case"])
    assert exact["exact"] is True
    assert exact["disagreeing_voxels"] == 0

    changed = mask.copy()
    changed[0, 0, 0] = 1
    _write_mask(second / "case.nii.gz", changed)
    comparison = module.compare_mask_directories(first, second, ["case"])
    assert comparison["exact"] is False
    assert comparison["disagreeing_voxels"] == 1
    assert comparison["disagreement_fraction"] == 1 / mask.size


def test_subtype_csv_requires_exact_membership_and_label_domain(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "subtype_results.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Names", "Subtype"])
        writer.writerow(["case_a.nii.gz", 0])
        writer.writerow(["case_b.nii.gz", 2])

    audit = module.validate_subtype_csv(path, ["case_a", "case_b"])
    assert audit == {
        "rows": 2,
        "membership_matches": True,
        "labels_valid": True,
        "valid": True,
    }


def test_candidate_classification_batches_all_eight_mirror_views() -> None:
    module = _module(ARM_SCRIPT)

    class Network:
        def __init__(self) -> None:
            self.calls = 0

        def encode_to_stages(self, volume, stages):
            self.calls += 1
            return {
                1: volume.repeat(1, 64, 1, 1, 1),
                2: volume.repeat(1, 128, 1, 1, 1),
            }

    network = Network()
    features = module._classification_features(
        network,
        torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4),
        torch.device("cpu"),
    )

    assert network.calls == 4
    assert features.shape == (192,)
    assert np.isfinite(features).all()


def test_candidate_classification_can_run_locked_single_view() -> None:
    module = _module(ARM_SCRIPT)

    class Network:
        def __init__(self) -> None:
            self.calls = 0

        def encode_to_stages(self, volume, stages):
            self.calls += 1
            return {
                1: volume.repeat(1, 64, 1, 1, 1),
                2: volume.repeat(1, 128, 1, 1, 1),
            }

    network = Network()
    features = module._classification_features(
        network,
        torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4),
        torch.device("cpu"),
        view_indices=(6,),
    )

    assert network.calls == 1
    assert features.shape == (192,)
    assert np.isfinite(features).all()
