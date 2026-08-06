from __future__ import annotations

import csv
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pancreas_multitask.case_feature_extractor import (
    FeatureExtractionRuntimeCounters,
    extract_case_from_preprocessed,
)
from pancreas_multitask.neural_case_bundle import (
    predict_neural_case_extraction,
)
from pancreas_multitask.neural_case_head import (
    NEURAL_MEAN_CANDIDATE,
    build_neural_case_bag,
    build_neural_case_head,
)
from pancreas_multitask.neural_case_predictor import (
    NeuralCaseNNUNetPredictor,
    extract_neural_only_case_from_preprocessed,
)
from pancreas_multitask.predictor import JointPrediction


class _ToyEncoder(nn.Module):
    output_channels = (1, 2, 3, 256, 5, 320)

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, data: torch.Tensor) -> list[torch.Tensor]:
        self.batch_sizes.append(int(data.shape[0]))
        base = data.mean(dim=1, keepdim=True)
        return [
            base.repeat(1, channels, 1, 1, 1) * (stage + 1)
            for stage, channels in enumerate(self.output_channels)
        ]


class _ToyDecoder(nn.Module):
    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        base = skips[0][:, :1]
        return torch.cat((-base, base * 0.5, base), dim=1)


class _ToySharedNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ToyEncoder()
        self.decoder = _ToyDecoder()

    def classify_bottleneck(self, bottleneck: torch.Tensor) -> torch.Tensor:
        value = bottleneck.mean(dim=(1, 2, 3, 4))
        return torch.stack((value, -value, value * 0.25), dim=1)


def _bare_predictor(*, tile_count: int, extraction_mode: str) -> NeuralCaseNNUNetPredictor:
    predictor = object.__new__(NeuralCaseNNUNetPredictor)
    predictor.network = _ToySharedNetwork()
    predictor.device = torch.device("cpu")
    predictor.configuration_manager = SimpleNamespace(patch_size=(4, 4, 4))
    predictor.label_manager = SimpleNamespace(num_segmentation_heads=3)
    predictor.perform_everything_on_device = False
    predictor.use_gaussian = False
    predictor.use_mirroring = True
    predictor.allowed_mirroring_axes = (0, 1, 2)
    predictor.tile_batch_size = 1
    predictor.tta_batch_size = 1
    predictor.v5_extraction_mode = extraction_mode
    predictor._initialized_fold = 0
    predictor._frozen_component_hashes = {"test": "frozen"}
    predictor._neural_case_head = build_neural_case_head(NEURAL_MEAN_CANDIDATE)
    predictor._neural_case_head.eval()
    predictor._neural_case_head.requires_grad_(False)
    predictor._class_offsets = np.asarray((0.25, 0.0, -0.25), dtype=np.float64)
    predictor._neural_bundle_metadata = {}

    def slicers(_self: object, shape: tuple[int, ...]):
        assert tuple(shape) == (4, 4, 4 * tile_count)
        return [
            (
                slice(None),
                slice(0, 4),
                slice(0, 4),
                slice(index * 4, (index + 1) * 4),
            )
            for index in range(tile_count)
        ]

    predictor._internal_get_sliding_window_slicers = MethodType(slicers, predictor)
    predictor.reset_inference_runtime_counters()
    return predictor


def _bag(extraction: object):
    return build_neural_case_bag(
        tile_vectors=extraction.tile_vectors,
        tile_evidence=extraction.tile_evidence,
        tile_vector_names=extraction.tile_vector_names,
        mil_stage3_maps=extraction.mil_stage3_maps,
        mil_prediction_maps=extraction.mil_prediction_maps,
        mil_lesion_mass=extraction.mil_lesion_mass,
    )


def test_dependency_pruned_extraction_reconstructs_exact_full_neural_bag() -> None:
    full_predictor = _bare_predictor(tile_count=3, extraction_mode="full")
    pruned_predictor = _bare_predictor(tile_count=3, extraction_mode="neural_only")
    pruned_predictor.network.load_state_dict(full_predictor.network.state_dict())
    image = torch.arange(4 * 4 * 12, dtype=torch.float32).reshape(1, 4, 4, 12) / 100
    full_counters = FeatureExtractionRuntimeCounters(1, 1)
    pruned_counters = FeatureExtractionRuntimeCounters(1, 1)

    full = extract_case_from_preprocessed(
        full_predictor,
        image,
        tile_batch_size=1,
        tta_batch_size=1,
        runtime_counters=full_counters,
    )
    pruned = extract_neural_only_case_from_preprocessed(
        pruned_predictor,
        image,
        runtime_counters=pruned_counters,
    )
    full_bag = _bag(full)
    pruned_bag = _bag(pruned)

    assert torch.equal(full.segmentation_logits, pruned.segmentation_logits)
    assert np.array_equal(full_bag.stage3_maps, pruned_bag.stage3_maps)
    assert np.array_equal(full_bag.prediction_maps, pruned_bag.prediction_maps)
    assert np.array_equal(full_bag.lesion_mass, pruned_bag.lesion_mass)
    assert np.array_equal(full_bag.all_tile_summary, pruned_bag.all_tile_summary)
    assert pruned.tile_vectors.shape == (3, 326)
    assert full.tile_vectors.shape[1] > pruned.tile_vectors.shape[1]
    assert full_counters.provenance() == pruned_counters.provenance()
    assert pruned_counters.provenance()["network_batch_size_histogram"] == {"1": 24}


def test_pruned_extraction_matches_with_padding_overlap_and_gaussian() -> None:
    full_predictor = _bare_predictor(tile_count=1, extraction_mode="full")
    pruned_predictor = _bare_predictor(tile_count=1, extraction_mode="neural_only")
    pruned_predictor.network.load_state_dict(full_predictor.network.state_dict())
    for predictor in (full_predictor, pruned_predictor):
        predictor.use_gaussian = True

        def overlapping_slicers(_self: object, shape: tuple[int, ...]):
            assert tuple(shape) == (4, 4, 6)
            return [
                (slice(None), slice(0, 4), slice(0, 4), slice(0, 4)),
                (slice(None), slice(0, 4), slice(0, 4), slice(2, 6)),
            ]

        predictor._internal_get_sliding_window_slicers = MethodType(
            overlapping_slicers,
            predictor,
        )

    image = torch.arange(1 * 3 * 3 * 6, dtype=torch.float32).reshape(1, 3, 3, 6) / 50
    full_counters = FeatureExtractionRuntimeCounters(1, 1)
    pruned_counters = FeatureExtractionRuntimeCounters(1, 1)
    full = extract_case_from_preprocessed(
        full_predictor,
        image,
        tile_batch_size=1,
        tta_batch_size=1,
        runtime_counters=full_counters,
    )
    pruned = extract_neural_only_case_from_preprocessed(
        pruned_predictor,
        image,
        runtime_counters=pruned_counters,
    )
    full_bag = _bag(full)
    pruned_bag = _bag(pruned)

    assert full.segmentation_logits.shape == (3, 3, 3, 6)
    assert full.segmentation_logits.dtype == torch.float16
    assert pruned.segmentation_logits.dtype == torch.float16
    assert torch.equal(full.segmentation_logits, pruned.segmentation_logits)
    for name in ("stage3_maps", "prediction_maps", "lesion_mass", "all_tile_summary"):
        assert np.array_equal(getattr(full_bag, name), getattr(pruned_bag, name))
    assert full_counters.provenance() == pruned_counters.provenance()


def test_online_full_and_neural_only_modes_match_offline_head_output() -> None:
    full_predictor = _bare_predictor(tile_count=2, extraction_mode="full")
    pruned_predictor = _bare_predictor(tile_count=2, extraction_mode="neural_only")
    pruned_predictor.network.load_state_dict(full_predictor.network.state_dict())
    pruned_predictor._neural_case_head.load_state_dict(
        full_predictor._neural_case_head.state_dict()
    )
    image = torch.arange(4 * 4 * 8, dtype=torch.float32).reshape(1, 4, 4, 8) / 100

    offline_extraction = extract_case_from_preprocessed(
        full_predictor,
        image,
        tile_batch_size=1,
        tta_batch_size=1,
    )
    offline_prediction = predict_neural_case_extraction(
        full_predictor._neural_case_head,
        offline_extraction,
        full_predictor._class_offsets,
    )
    full_predictor.reset_inference_runtime_counters()
    full_online = full_predictor.predict_joint_from_preprocessed_data(image)
    pruned_online = pruned_predictor.predict_joint_from_preprocessed_data(image)

    assert torch.equal(
        offline_extraction.segmentation_logits,
        full_online.segmentation_logits,
    )
    assert torch.equal(full_online.segmentation_logits, pruned_online.segmentation_logits)
    assert torch.equal(
        offline_prediction.offset_probabilities,
        full_online.classification_probabilities,
    )
    assert torch.equal(
        full_online.classification_probabilities,
        pruned_online.classification_probabilities,
    )
    assert (
        full_predictor.inference_runtime_provenance()[
            "v5_neural_bag_sha256_sequence"
        ]
        == pruned_predictor.inference_runtime_provenance()[
            "v5_neural_bag_sha256_sequence"
        ]
    )
    for predictor in (full_predictor, pruned_predictor):
        execution = predictor.inference_runtime_provenance()
        assert execution["maximum_network_batch_size_observed"] == 1
        assert execution["v5_case_extractions_completed"] == 1
        assert execution["v5_neural_head_forward_calls"] == 1
        assert execution["v5_class_offset_applications"] == 1
        assert execution["v5_feature_cache_reads"] == 0


def test_constructor_rejects_wrong_schedule_mode_hash_or_bundle_path(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle.pth"
    bundle.write_bytes(b"not loaded by this constructor test")
    common = {
        "neural_case_head_bundle": bundle,
        "expected_neural_case_head_bundle_sha256": "a" * 64,
        "expected_numeric_train_dataset_sha256": "b" * 64,
        "device": torch.device("cpu"),
    }
    with pytest.raises(ValueError, match="tile1/TTA1"):
        NeuralCaseNNUNetPredictor(
            **common,
            tile_batch_size=2,
            tta_batch_size=2,
        )
    with pytest.raises(ValueError, match="v5_extraction_mode"):
        NeuralCaseNNUNetPredictor(
            **common,
            v5_extraction_mode="unknown",
        )
    with pytest.raises(ValueError, match="bundle SHA-256"):
        NeuralCaseNNUNetPredictor(
            **{**common, "expected_neural_case_head_bundle_sha256": "wrong"},
        )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        NeuralCaseNNUNetPredictor(
            **{**common, "neural_case_head_bundle": tmp_path / "missing.pth"},
        )


@pytest.mark.parametrize(
    ("logit_dtype", "accepted"),
    ((torch.float16, True), (torch.float32, False)),
)
def test_raw_export_requires_stock_float16_logits_and_adjusted_probabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logit_dtype: torch.dtype,
    accepted: bool,
) -> None:
    import pancreas_multitask.neural_case_predictor as predictor_module

    predictor = _bare_predictor(tile_count=1, extraction_mode="neural_only")
    predictor.dataset_json = {"file_ending": ".nii.gz", "channel_names": {"0": "CT"}}
    predictor.plans_manager = object()

    class Preprocessor:
        def __init__(self, verbose: bool) -> None:
            assert verbose is False

        def run_case(self, *_args: object):
            return np.zeros((1, 4, 4, 4), dtype=np.float32), None, {"geometry": "kept"}

    predictor.configuration_manager = SimpleNamespace(
        patch_size=(4, 4, 4),
        spacing=(1.0, 1.0, 1.0),
        previous_stage_name=None,
        preprocessor_class=Preprocessor,
    )
    predictor.dataset_json = {"file_ending": ".nii.gz", "channel_names": {"0": "CT"}}
    predictor.verbose_preprocessing = False
    expected_logits = torch.arange(3 * 4 * 4 * 4, dtype=logit_dtype).reshape(
        3, 4, 4, 4
    )
    adjusted = torch.tensor((0.1, 0.2, 0.7), dtype=torch.float64)
    predictor.predict_joint_from_preprocessed_data = MethodType(
        lambda _self, _data: JointPrediction(expected_logits, adjusted),
        predictor,
    )
    exported: dict[str, object] = {}

    def fake_export(
        logits: torch.Tensor,
        properties: dict[str, object],
        _configuration: object,
        _plans: object,
        _dataset: object,
        output_base: str,
        save_probabilities: bool,
    ) -> None:
        exported.update(
            logits=logits.clone(),
            properties=properties,
            save_probabilities=save_probabilities,
        )
        Path(f"{output_base}.nii.gz").write_bytes(b"mask")

    monkeypatch.setattr(predictor_module, "export_prediction_from_logits", fake_export)
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "case_0000.nii.gz").write_bytes(b"raw")
    output_directory = tmp_path / "output"
    probability_csv = output_directory / "subtype_probabilities.csv"

    if not accepted:
        with pytest.raises(RuntimeError, match="stock float16 dtype"):
            predictor.predict_from_files_joint(
                input_directory,
                output_directory,
                probability_csv=probability_csv,
                overwrite=True,
            )
        assert predictor.inference_runtime_provenance()[
            "segmentation_export_logit_dtype_sequence"
        ] == []
        return

    results = predictor.predict_from_files_joint(
        input_directory,
        output_directory,
        probability_csv=probability_csv,
        overwrite=True,
    )

    assert torch.equal(exported["logits"], expected_logits)
    assert exported["properties"] == {"geometry": "kept"}
    assert results[0].subtype == 2
    assert results[0].classification_probabilities == pytest.approx((0.1, 0.2, 0.7))
    execution = predictor.inference_runtime_provenance()
    assert execution["segmentation_export_logit_dtype"] == "torch.float16"
    assert execution["segmentation_export_logit_dtype_sequence"] == ["torch.float16"]
    with probability_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Names"] == "case.nii.gz"
    assert int(rows[0]["Subtype"]) == 2
    assert float(rows[0]["Probability_2"]) == pytest.approx(0.7)


def test_raw_export_preserves_float64_offset_decision_for_near_tie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pancreas_multitask.neural_case_predictor as predictor_module

    predictor = _bare_predictor(tile_count=1, extraction_mode="neural_only")

    class Preprocessor:
        def __init__(self, verbose: bool) -> None:
            assert verbose is False

        def run_case(self, *_args: object):
            return np.zeros((1, 4, 4, 4), dtype=np.float32), None, {}

    predictor.configuration_manager = SimpleNamespace(
        patch_size=(4, 4, 4),
        spacing=(1.0, 1.0, 1.0),
        previous_stage_name=None,
        preprocessor_class=Preprocessor,
    )
    predictor.dataset_json = {"file_ending": ".nii.gz", "channel_names": {"0": "CT"}}
    predictor.plans_manager = object()
    predictor.verbose_preprocessing = False
    logits = torch.zeros((3, 4, 4, 4), dtype=torch.float16)
    adjusted = torch.tensor(
        (0.49999999995, 0.50000000005, 0.0),
        dtype=torch.float64,
    )
    assert int(torch.argmax(adjusted).item()) == 1
    assert int(torch.argmax(adjusted.float()).item()) == 0
    predictor.predict_joint_from_preprocessed_data = MethodType(
        lambda _self, _data: JointPrediction(logits, adjusted),
        predictor,
    )

    def fake_export(
        _logits: torch.Tensor,
        _properties: dict[str, object],
        _configuration: object,
        _plans: object,
        _dataset: object,
        output_base: str,
        _save_probabilities: bool,
    ) -> None:
        Path(f"{output_base}.nii.gz").write_bytes(b"mask")

    monkeypatch.setattr(predictor_module, "export_prediction_from_logits", fake_export)
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "case_0000.nii.gz").write_bytes(b"raw")
    output_directory = tmp_path / "output"
    probability_csv = output_directory / "subtype_probabilities.csv"

    results = predictor.predict_from_files_joint(
        input_directory,
        output_directory,
        probability_csv=probability_csv,
        overwrite=True,
    )

    assert results[0].subtype == 1
    assert results[0].classification_probabilities == tuple(adjusted.tolist())
    with (output_directory / "subtype_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        subtype_rows = list(csv.DictReader(handle))
    with probability_csv.open("r", encoding="utf-8", newline="") as handle:
        probability_rows = list(csv.DictReader(handle))
    assert int(subtype_rows[0]["Subtype"]) == 1
    assert int(probability_rows[0]["Subtype"]) == 1
    assert float(probability_rows[0]["Probability_1"]) == adjusted[1].item()
