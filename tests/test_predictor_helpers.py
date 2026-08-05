from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnunetv2")
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pancreas_multitask.predictor as predictor_module
from pancreas_multitask.predictor import (
    JointNNUNetPredictor,
    JointPrediction,
    average_classification_probabilities,
    discover_nnunet_input_cases,
    normalize_submission_mapping,
    read_classification_csv,
    submission_filename,
    write_classification_csv,
)


class _OrientationAwareJointNetwork(nn.Module):
    """Toy network whose class logits distinguish a depth-axis mirror."""

    def forward(self, x, *, return_classification=False):
        segmentation = x[:, :1]
        if not return_classification:
            return segmentation
        first_plane = x[:, 0, 0].mean(dim=(1, 2))
        last_plane = x[:, 0, -1].mean(dim=(1, 2))
        classification = torch.stack((first_plane, last_plane), dim=1)
        return segmentation, classification


def _bare_predictor(
    network: nn.Module,
    predictor_class: type[JointNNUNetPredictor] = JointNNUNetPredictor,
) -> JointNNUNetPredictor:
    predictor = object.__new__(predictor_class)
    predictor.network = network.eval()
    predictor.device = torch.device("cpu")
    predictor.use_mirroring = False
    predictor.allowed_mirroring_axes = None
    predictor.use_gaussian = False
    predictor.allow_tqdm = False
    predictor.verbose = False
    predictor.perform_everything_on_device = False
    predictor.label_manager = SimpleNamespace(num_segmentation_heads=1)
    predictor.configuration_manager = SimpleNamespace(patch_size=(2, 2, 2))
    predictor.list_of_parameters = None
    return predictor


def test_probability_helper_averages_in_float32_without_mutating_inputs() -> None:
    first = torch.tensor([0.75, 0.25], dtype=torch.float16)
    second = torch.tensor([0.25, 0.75], dtype=torch.float16)

    result = average_classification_probabilities((first, second))

    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([0.5, 0.5]))
    assert torch.equal(first, torch.tensor([0.75, 0.25], dtype=torch.float16))


def test_probability_helper_rejects_empty_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="At least one"):
        average_classification_probabilities(())
    with pytest.raises(ValueError, match="same shape"):
        average_classification_probabilities(
            (torch.ones(2), torch.ones(3))
        )


def test_submission_names_include_nifti_ending_and_round_trip() -> None:
    assert submission_filename("quiz_037", ".nii.gz") == "quiz_037.nii.gz"
    assert submission_filename("quiz_037.nii.gz", ".nii.gz") == "quiz_037.nii.gz"
    assert normalize_submission_mapping(
        {"quiz_037.nii.gz": 2, "quiz_045.nii.gz": 0}, ".nii.gz"
    ) == {"quiz_037": 2, "quiz_045": 0}


def test_submission_mapping_rejects_bare_or_duplicate_names() -> None:
    with pytest.raises(ValueError, match="must end"):
        normalize_submission_mapping({"quiz_037": 1}, ".nii.gz")
    with pytest.raises(ValueError, match="duplicate"):
        normalize_submission_mapping(
            {"quiz_037.nii.gz": 1, " quiz_037.nii.gz ": 1}, ".nii.gz"
        )


def test_strict_classification_csv_writes_suffixed_names_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "subtype_results.csv"
    write_classification_csv(
        output,
        {"quiz_037.nii.gz": 2, "quiz_001.nii.gz": 0},
    )

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["Names", "Subtype"],
        ["quiz_001.nii.gz", "0"],
        ["quiz_037.nii.gz", "2"],
    ]
    assert normalize_submission_mapping(
        read_classification_csv(output), ".nii.gz"
    ) == {"quiz_001": 0, "quiz_037": 2}
    with pytest.raises(ValueError, match="bare .nii.gz"):
        write_classification_csv(output, {"quiz_037": 2})


def test_discovers_sorted_contiguous_nnunet_channels(tmp_path: Path) -> None:
    for name in ("quiz_002_0000.nii.gz", "quiz_001_0001.nii.gz", "quiz_001_0000.nii.gz"):
        (tmp_path / name).write_bytes(b"")

    cases = discover_nnunet_input_cases(tmp_path, ".nii.gz")

    assert [case_id for case_id, _files in cases] == ["quiz_001", "quiz_002"]
    assert [Path(path).name for path in cases[0][1]] == [
        "quiz_001_0000.nii.gz",
        "quiz_001_0001.nii.gz",
    ]
    with pytest.raises(ValueError, match="expected 1"):
        discover_nnunet_input_cases(tmp_path, ".nii.gz", expected_channels=1)


def test_mirror_tta_flips_only_segmentation_and_averages_probabilities() -> None:
    predictor = _bare_predictor(_OrientationAwareJointNetwork())
    predictor.use_mirroring = True
    predictor.allowed_mirroring_axes = (0,)
    image = torch.zeros((1, 1, 2, 2, 2))
    image[:, :, 0] = 2.0

    prediction = predictor._internal_maybe_mirror_and_predict_joint(image)

    expected_probabilities = torch.stack(
        (
            torch.softmax(torch.tensor([2.0, 0.0]), dim=0),
            torch.softmax(torch.tensor([0.0, 2.0]), dim=0),
        )
    ).mean(dim=0)
    assert torch.equal(prediction.segmentation_logits, image)
    assert torch.allclose(
        prediction.classification_probabilities[0], expected_probabilities
    )


def test_tiles_are_averaged_locally_and_segmentation_is_stitched() -> None:
    predictor = _bare_predictor(_OrientationAwareJointNetwork())
    data = torch.zeros((1, 4, 2, 2))
    data[:, :2] = 2.0
    slicers = (
        (slice(None), slice(0, 2), slice(0, 2), slice(0, 2)),
        (slice(None), slice(2, 4), slice(0, 2), slice(0, 2)),
    )

    prediction = predictor._internal_predict_sliding_window_return_joint(
        data, slicers, False
    )

    expected = torch.stack(
        (
            torch.softmax(torch.tensor([2.0, 2.0]), dim=0),
            torch.softmax(torch.tensor([0.0, 0.0]), dim=0),
        )
    ).mean(dim=0)
    assert torch.equal(prediction.segmentation_logits.float(), data)
    assert torch.allclose(prediction.classification_probabilities, expected)
    assert not hasattr(predictor, "classification_sum")


class _FoldStateNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, *, return_classification=False):
        segmentation = x[:, :1] + self.value
        classification = torch.stack((self.value, -self.value))[None]
        return (segmentation, classification) if return_classification else segmentation


class _FoldOnlyPredictor(JointNNUNetPredictor):
    def predict_sliding_window_return_joint(self, input_image):
        segmentation, logits = self.network(
            input_image[None], return_classification=True
        )
        return JointPrediction(
            segmentation[0],
            torch.softmax(logits.float(), dim=1)[0],
        )


class _RetryOnlyPredictor(JointNNUNetPredictor):
    def _internal_predict_sliding_window_return_joint(
        self, data, slicers, do_on_device=True
    ):
        self.attempt_devices.append(do_on_device)
        if do_on_device:
            # This object is deliberately unreachable outside the failed call.
            _partial_prediction = JointPrediction(
                torch.full((1, 2, 2, 2), 99.0),
                torch.tensor([0.99, 0.01]),
            )
            raise RuntimeError("simulated result-array OOM")
        return JointPrediction(
            torch.zeros((1, 2, 2, 2)),
            torch.tensor([0.25, 0.75]),
        )


class _RawOnlyPredictor(JointNNUNetPredictor):
    def predict_joint_from_preprocessed_data(self, data):
        self.joint_call_count += 1
        assert data.shape == (1, 2, 2, 2)
        return JointPrediction(
            torch.zeros((3, 2, 2, 2)),
            torch.tensor([0.1, 0.2, 0.7]),
        )


class _ToyPreprocessor:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def run_case(
        self,
        image_files,
        previous_stage,
        plans_manager,
        configuration_manager,
        dataset_json,
    ):
        assert previous_stage is None
        return (
            np.zeros((1, 2, 2, 2), dtype=np.float32),
            None,
            {"input_name": Path(image_files[0]).name},
        )


def test_fold_ensemble_loads_each_state_and_averages_explicit_results() -> None:
    network = _FoldStateNetwork()
    predictor = _bare_predictor(network, _FoldOnlyPredictor)
    predictor.list_of_parameters = (
        {"value": torch.tensor(1.0)},
        {"value": torch.tensor(-1.0)},
    )

    prediction = predictor.predict_joint_from_preprocessed_data(
        torch.zeros((1, 2, 2, 2))
    )

    assert torch.equal(
        prediction.segmentation_logits, torch.zeros((1, 2, 2, 2))
    )
    assert torch.allclose(
        prediction.classification_probabilities,
        torch.tensor([0.5, 0.5]),
    )


def test_failed_device_attempt_cannot_leak_votes_into_cpu_retry() -> None:
    predictor = object.__new__(_RetryOnlyPredictor)
    predictor.device = torch.device("cuda")
    predictor.perform_everything_on_device = True
    predictor.verbose = False
    predictor.attempt_devices = []

    prediction = predictor._predict_sliding_window_with_results_fallback(
        torch.zeros((1, 2, 2, 2)), ()
    )

    assert predictor.attempt_devices == [True, False]
    assert torch.equal(
        prediction.classification_probabilities,
        torch.tensor([0.25, 0.75]),
    )


def test_raw_pipeline_predicts_once_exports_and_skips_complete_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    for case_id in ("quiz_002", "quiz_001"):
        (input_directory / f"{case_id}_0000.nii.gz").write_bytes(b"input")

    predictor = object.__new__(_RawOnlyPredictor)
    predictor.configuration_manager = SimpleNamespace(
        previous_stage_name=None,
        preprocessor_class=_ToyPreprocessor,
    )
    predictor.plans_manager = SimpleNamespace()
    predictor.dataset_json = {
        "file_ending": ".nii.gz",
        "channel_names": {"0": "CT"},
    }
    predictor.verbose_preprocessing = False
    predictor.device = torch.device("cpu")
    predictor.joint_call_count = 0
    exported_properties = []

    def fake_export(
        segmentation,
        properties,
        configuration_manager,
        plans_manager,
        dataset_json,
        output_base,
        save_probabilities,
    ):
        assert segmentation.shape == (3, 2, 2, 2)
        assert not save_probabilities
        exported_properties.append(properties)
        Path(f"{output_base}.nii.gz").write_bytes(b"mask")

    monkeypatch.setattr(predictor_module, "export_prediction_from_logits", fake_export)
    probability_csv = output_directory / "probabilities.csv"

    first_results = predictor.predict_from_files_joint(
        input_directory,
        output_directory,
        probability_csv=probability_csv,
    )

    assert predictor.joint_call_count == 2
    assert [result.case_id for result in first_results] == ["quiz_001", "quiz_002"]
    assert [result.subtype for result in first_results] == [2, 2]
    assert {item["input_name"] for item in exported_properties} == {
        "quiz_001_0000.nii.gz",
        "quiz_002_0000.nii.gz",
    }
    assert read_classification_csv(output_directory / "subtype_results.csv") == {
        "quiz_001.nii.gz": 2,
        "quiz_002.nii.gz": 2,
    }
    assert probability_csv.read_text(encoding="utf-8").splitlines()[1].startswith(
        "quiz_001.nii.gz,2,"
    )

    second_results = predictor.predict_from_files_joint(
        input_directory,
        output_directory,
        probability_csv=probability_csv,
        overwrite=False,
    )

    assert predictor.joint_call_count == 2
    assert [result.case_id for result in second_results] == ["quiz_001", "quiz_002"]


def test_cli_parser_accepts_fold_all_and_no_overwrite(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "predict_joint.py"
    spec = importlib.util.spec_from_file_location("predict_joint_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "--input",
            str(tmp_path / "input"),
            "--output",
            str(tmp_path / "output"),
            "--model",
            str(tmp_path / "model"),
            "--folds",
            "all",
            "--no-overwrite",
        ]
    )

    assert args.folds == ["all"]
    assert module._selected_folds(args.folds) == ("all",)
    assert args.overwrite is False
    with pytest.raises(ValueError, match="cannot be combined"):
        module._selected_folds(["all", 0])


def test_joint_prediction_rejects_non_vector_classification_output() -> None:
    with pytest.raises(ValueError, match="classes"):
        JointPrediction(torch.zeros(1), torch.zeros(1, 2, 3))
