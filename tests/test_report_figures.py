from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("nibabel")
pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_report_figures as figures


def _rows() -> list[dict[str, str]]:
    return [
        {
            "case_id": f"case_{index}",
            "whole_pancreas_dice": str(0.50 + index * 0.05),
            "lesion_dice": str(0.10 + index * 0.10),
        }
        for index in range(6)
    ]


def test_qualitative_selection_returns_two_weak_and_two_strong_cases() -> None:
    assert figures.select_qualitative_cases(_rows()) == [
        "case_0",
        "case_1",
        "case_4",
        "case_5",
    ]


def test_ct_window_and_slice_selection_are_deterministic() -> None:
    image = np.array([-200.0, -160.0, 40.0, 240.0, 300.0])
    assert figures._window_ct(image).tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])

    mask = np.zeros((3, 4, 5), dtype=bool)
    mask[2, :, :] = True
    assert figures._best_index(mask, 0) == 2


def test_qualitative_row_label_is_compact_and_multiline() -> None:
    label = figures._qualitative_row_label(
        "weak",
        "quiz_2_191",
        {
            "whole_pancreas_dice": "0.793472",
            "lesion_dice": "0.0",
            "reference_subtype": "2",
            "predicted_subtype": "0",
            "lesion_reference_voxels": "4248",
        },
    )
    assert label.splitlines() == [
        "weak: quiz_2_191",
        "whole=0.793 | lesion=0.000",
        "subtype ref/pred=2/0",
        "ref lesion=4,248 vox",
    ]


def test_metric_figures_write_pdf_and_png(tmp_path: Path) -> None:
    metrics = {
        "classification": {
            "labels": [0, 1, 2],
            "confusion_matrix": [[2, 0, 0], [0, 1, 1], [0, 0, 2]],
        }
    }
    confusion = figures.plot_confusion_matrix(metrics, tmp_path, dpi=72)
    distribution = figures.plot_dice_distribution(_rows(), tmp_path, dpi=72)
    for output in confusion + distribution:
        assert output.is_file()
        assert output.stat().st_size > 0


def test_training_curves_reject_missing_measured_fields(tmp_path: Path) -> None:
    with pytest.raises(figures.FigureInputError, match="train_seg_losses"):
        figures.plot_training_curves({}, tmp_path, dpi=72)
