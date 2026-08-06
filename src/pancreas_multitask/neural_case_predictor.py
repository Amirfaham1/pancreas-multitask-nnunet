"""Final v5 neural case-head inference on raw nnU-Net case tensors.

The class in this module deliberately reuses the existing raw-file discovery,
preprocessing, segmentation export, and CSV machinery from
``JointNNUNetPredictor``.  It changes only the preprocessed-case prediction
hook: the frozen ResEnc-M network produces a complete label-free
``CaseExtraction``, and the separately locked v5 neural head converts that bag
to offset-adjusted case probabilities.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.utilities.helpers import empty_cache
from torch import Tensor, nn
from torch._dynamo import OptimizedModule

from pancreas_multitask.case_feature_extractor import (
    LOCKED_NETWORK_MICROBATCH_CEILING,
    MIL_GRID,
    MIL_STAGE_INDEX,
    MIL_TOP_K,
    CaseExtraction,
    FeatureExtractionRuntimeCounters,
    TileBatchFeatures,
    _forward_shared_network,
    _mirror_combinations,
    _numeric_top_k,
    extract_case_from_preprocessed,
)
from pancreas_multitask.case_features import (
    FROZEN_NEURAL_HEAD_NAMES,
    normalized_tile_centers,
    tile_model_evidence,
)
from pancreas_multitask.classification_rescue import component_hashes, file_sha256
from pancreas_multitask.neural_case_bundle import (
    load_neural_case_head_bundle,
    predict_neural_case_extraction,
)
from pancreas_multitask.neural_case_head import (
    FROZEN_COMPONENT_HASHES,
    build_neural_case_bag,
)
from pancreas_multitask.neural_case_training import neural_state_sha256
from pancreas_multitask.predictor import (
    SUBMISSION_FILE_ENDING,
    JointCasePrediction,
    JointNNUNetPredictor,
    JointPrediction,
    _atomic_write_csv,
    discover_nnunet_input_cases,
    submission_filename,
    write_classification_csv,
)

V5_CLASSIFIER_PIPELINE = "assignment_conforming_v5_neural_case_head"
V5_EXTRACTION_MODES = ("full", "neural_only")
LOCKED_BATCH_CONFIGURATIONS = ((1, 1),)
STAGE5_GLOBAL_MEAN_NAMES = tuple(
    f"encoder_stage_5_global_mean_channel_{channel:03d}" for channel in range(320)
)


def _write_v5_probability_csv(
    path: Path,
    labels: dict[str, int],
    probabilities: dict[str, tuple[float, float, float]],
) -> None:
    """Atomically retain enough decimal digits to round-trip float64 values."""

    fieldnames = (
        "Names",
        "Subtype",
        "Probability_0",
        "Probability_1",
        "Probability_2",
    )
    rows = (
        {
            "Names": case_id,
            "Subtype": labels[case_id],
            **{
                f"Probability_{index}": f"{probabilities[case_id][index]:.17g}"
                for index in range(3)
            },
        }
        for case_id in sorted(labels)
    )
    _atomic_write_csv(path, fieldnames, rows)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _network_module(network: nn.Module) -> nn.Module:
    if isinstance(network, OptimizedModule):
        return network._orig_mod
    return network


def _neural_bag_sha256(bag: object) -> str:
    digest = hashlib.sha256()
    for name in (
        "stage3_maps",
        "prediction_maps",
        "lesion_mass",
        "all_tile_summary",
    ):
        value = np.ascontiguousarray(getattr(bag, name))
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@torch.inference_mode()
def mirror_mean_neural_only_tile_features(
    network: nn.Module,
    data: Tensor,
    *,
    allowed_mirroring_axes: Sequence[int] | None,
    runtime_counters: FeatureExtractionRuntimeCounters,
) -> TileBatchFeatures:
    """Compute only tensors consumed by either locked v5 neural case head."""

    if data.ndim != 5 or data.shape[0] != 1:
        raise ValueError("Locked neural-only inference requires one tile per forward")
    if (
        runtime_counters.tile_batch_size_requested != 1
        or runtime_counters.tta_batch_size_requested != 1
        or runtime_counters.network_batch_size_limit != 1
    ):
        raise ValueError("Locked neural-only inference requires tile1/TTA1 scheduling")

    segmentation_sum: Tensor | None = None
    stage5_global_mean_sum: Tensor | None = None
    logit_sum: Tensor | None = None
    probability_sum: Tensor | None = None
    mil_stage3_sum: Tensor | None = None
    mil_prediction_sum: Tensor | None = None
    combinations = _mirror_combinations(allowed_mirroring_axes)

    for axes in combinations:
        view = torch.flip(data, axes) if axes else data
        runtime_counters.record_network_forward(int(view.shape[0]))
        segmentation, skips, logits = _forward_shared_network(network, view)
        if skips[5].shape[1] != len(STAGE5_GLOBAL_MEAN_NAMES):
            raise ValueError("Frozen ResEnc-M stage 5 must expose exactly 320 channels")
        stage5_global_mean = skips[5].float().flatten(start_dim=2).mean(dim=2)
        probabilities = torch.softmax(logits.float(), dim=1)
        segmentation_probabilities = torch.softmax(segmentation.float(), dim=1)
        stage3_map = F.adaptive_avg_pool3d(
            skips[MIL_STAGE_INDEX].float(),
            MIL_GRID,
        )
        prediction_at_stage3 = F.interpolate(
            torch.cat(
                (
                    segmentation_probabilities[:, 2:3],
                    segmentation_probabilities[:, 1:2]
                    + segmentation_probabilities[:, 2:3],
                ),
                dim=1,
            ),
            size=skips[MIL_STAGE_INDEX].shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        prediction_map = F.adaptive_avg_pool3d(prediction_at_stage3, MIL_GRID)
        if axes:
            segmentation = torch.flip(segmentation, axes)
            stage3_map = torch.flip(stage3_map, axes)
            prediction_map = torch.flip(prediction_map, axes)

        # This is the same dtype and left-to-right view order as the full
        # committed extractor. Only unconsumed statistics are absent.
        segmentation_sum = (
            segmentation if segmentation_sum is None else segmentation_sum + segmentation
        )
        stage5_global_mean_sum = (
            stage5_global_mean
            if stage5_global_mean_sum is None
            else stage5_global_mean_sum + stage5_global_mean
        )
        logit_sum = logits.float() if logit_sum is None else logit_sum + logits.float()
        probability_sum = (
            probabilities if probability_sum is None else probability_sum + probabilities
        )
        mil_stage3_sum = (
            stage3_map if mil_stage3_sum is None else mil_stage3_sum + stage3_map
        )
        mil_prediction_sum = (
            prediction_map
            if mil_prediction_sum is None
            else mil_prediction_sum + prediction_map
        )
        runtime_counters.record_tta_batch(1)

    if any(
        value is None
        for value in (
            segmentation_sum,
            stage5_global_mean_sum,
            logit_sum,
            probability_sum,
            mil_stage3_sum,
            mil_prediction_sum,
        )
    ):
        raise RuntimeError("Neural-only mirror extraction produced no views")
    divisor = float(len(combinations))
    mean_logits = logit_sum / divisor
    mean_probabilities = probability_sum / divisor
    tile_vectors = torch.cat(
        (
            stage5_global_mean_sum / divisor,
            mean_logits,
            mean_probabilities,
        ),
        dim=1,
    )
    return TileBatchFeatures(
        segmentation_sum / divisor,
        tile_vectors,
        STAGE5_GLOBAL_MEAN_NAMES + FROZEN_NEURAL_HEAD_NAMES,
        mean_logits,
        mean_probabilities,
        mil_stage3_sum / divisor,
        mil_prediction_sum / divisor,
    )


@torch.inference_mode()
def extract_neural_only_case_from_preprocessed(
    predictor: object,
    input_image: Tensor,
    *,
    runtime_counters: FeatureExtractionRuntimeCounters,
) -> CaseExtraction:
    """Dependency-pruned twin of the full extractor for final v5 inference."""

    if not isinstance(input_image, Tensor) or input_image.ndim != 4:
        raise ValueError("input_image must have shape (channels, D, H, W)")
    if (
        runtime_counters.tile_batch_size_requested != 1
        or runtime_counters.tta_batch_size_requested != 1
    ):
        raise ValueError("Neural-only extraction requires the locked 1/1 schedule")
    network = getattr(predictor, "network", None)
    device = getattr(predictor, "device", None)
    configuration = getattr(predictor, "configuration_manager", None)
    label_manager = getattr(predictor, "label_manager", None)
    if not isinstance(network, nn.Module) or not isinstance(device, torch.device):
        raise TypeError("predictor must expose an initialized network and torch device")
    if configuration is None or label_manager is None:
        raise TypeError("predictor must expose configuration and label managers")

    network.to(device)
    network.eval()
    empty_cache(device)
    padded, slicer_revert_padding = pad_nd_image(
        input_image,
        configuration.patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )
    slicers = tuple(predictor._internal_get_sliding_window_slicers(padded.shape[1:]))
    if not slicers:
        raise RuntimeError("Sliding-window extraction produced no tiles")
    centers = normalized_tile_centers(slicers, padded.shape[1:])
    results_device = device if predictor.perform_everything_on_device else torch.device("cpu")
    padded_on_results = padded.to(results_device)
    predicted_logits = torch.zeros(
        (label_manager.num_segmentation_heads, *padded.shape[1:]),
        dtype=torch.float16,
        device=results_device,
    )
    n_predictions = torch.zeros(
        padded.shape[1:],
        dtype=torch.float16,
        device=results_device,
    )
    gaussian: Tensor | float
    if predictor.use_gaussian:
        gaussian = compute_gaussian(
            tuple(configuration.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=results_device,
        )
    else:
        gaussian = 1.0

    vector_rows: list[np.ndarray] = []
    evidence_rows: list[np.ndarray] = []
    mil_stage3_rows: list[np.ndarray] = []
    mil_prediction_rows: list[np.ndarray] = []
    mirror_axes = predictor.allowed_mirroring_axes if predictor.use_mirroring else None
    autocast = torch.autocast(device.type, enabled=device.type == "cuda")
    with autocast:
        for tile_index, sliding_slice in enumerate(slicers):
            work = torch.cat(
                [padded_on_results[sliding_slice][None]],
                dim=0,
            ).to(device, memory_format=torch.contiguous_format)
            extracted = mirror_mean_neural_only_tile_features(
                network,
                work,
                allowed_mirroring_axes=mirror_axes,
                runtime_counters=runtime_counters,
            )
            tile_center = centers[tile_index : tile_index + 1].to(
                extracted.segmentation_logits.device
            )
            evidence, _ = tile_model_evidence(
                extracted.segmentation_logits,
                tile_center,
            )
            tile_segmentation = extracted.segmentation_logits[0].to(
                results_device,
                dtype=torch.float16,
            )
            if predictor.use_gaussian:
                tile_segmentation = tile_segmentation * gaussian
            predicted_logits[sliding_slice] += tile_segmentation
            n_predictions[sliding_slice[1:]] += gaussian
            vector_rows.append(
                extracted.tile_vectors[0].detach().float().cpu().numpy().astype(np.float32)
            )
            evidence_rows.append(
                evidence[0].detach().float().cpu().numpy().astype(np.float32)
            )
            mil_stage3_rows.append(
                extracted.mil_stage3_maps[0]
                .detach()
                .half()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            mil_prediction_rows.append(
                extracted.mil_prediction_maps[0]
                .detach()
                .half()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            runtime_counters.record_tile_batch(1)

    torch.div(predicted_logits, n_predictions, out=predicted_logits)
    if not torch.isfinite(predicted_logits).all():
        raise FloatingPointError("Non-finite stitched segmentation logits")
    cropped = predicted_logits[(slice(None), *slicer_revert_padding[1:])].float().cpu()
    tile_vectors = np.stack(vector_rows).astype(np.float32, copy=False)
    tile_evidence = np.stack(evidence_rows).astype(np.float32, copy=False)
    stage3_all = np.stack(mil_stage3_rows).astype(np.float16, copy=False)
    prediction_all = np.stack(mil_prediction_rows).astype(np.float16, copy=False)
    lesion_mass_all = tile_evidence[:, 0].astype(np.float32, copy=False)
    selected = _numeric_top_k(
        lesion_mass_all,
        stage3_all,
        top_k=MIL_TOP_K,
    )
    empty_cache(device)
    return CaseExtraction(
        cropped,
        tile_vectors,
        tile_evidence,
        STAGE5_GLOBAL_MEAN_NAMES + FROZEN_NEURAL_HEAD_NAMES,
        stage3_all[selected],
        prediction_all[selected],
        lesion_mass_all[selected],
        len(slicers),
    )


class NeuralCaseNNUNetPredictor(JointNNUNetPredictor):
    """A single-fold frozen nnU-Net predictor with the locked v5 case head."""

    def __init__(
        self,
        *args: Any,
        neural_case_head_bundle: str | Path,
        expected_neural_case_head_bundle_sha256: str,
        expected_numeric_train_dataset_sha256: str,
        v5_extraction_mode: str = "full",
        **kwargs: Any,
    ) -> None:
        tile_batch_size = kwargs.get("tile_batch_size", 1)
        tta_batch_size = kwargs.get("tta_batch_size", 1)
        if (tile_batch_size, tta_batch_size) not in LOCKED_BATCH_CONFIGURATIONS:
            raise ValueError(
                "V5 speed-v3 inference requires tile1/TTA1 in both arms"
            )
        if v5_extraction_mode not in V5_EXTRACTION_MODES:
            raise ValueError(
                f"v5_extraction_mode must be one of {V5_EXTRACTION_MODES}"
            )
        bundle_path = Path(neural_case_head_bundle).expanduser().resolve()
        if not bundle_path.is_file():
            raise FileNotFoundError(f"Neural case-head bundle does not exist: {bundle_path}")
        if not _is_sha256(expected_neural_case_head_bundle_sha256):
            raise ValueError("Expected neural case-head bundle SHA-256 is invalid")
        if not _is_sha256(expected_numeric_train_dataset_sha256):
            raise ValueError("Expected numeric training-dataset SHA-256 is invalid")

        super().__init__(*args, **kwargs)
        self._neural_case_head_bundle_path = bundle_path
        self._expected_neural_case_head_bundle_sha256 = (
            expected_neural_case_head_bundle_sha256
        )
        self._expected_numeric_train_dataset_sha256 = (
            expected_numeric_train_dataset_sha256
        )
        self.v5_extraction_mode = v5_extraction_mode
        self._neural_case_head: nn.Module | None = None
        self._class_offsets: np.ndarray | None = None
        self._neural_bundle_metadata: dict[str, Any] | None = None
        self._frozen_component_hashes: dict[str, str] | None = None
        self._initialized_fold: int | None = None
        self._neural_head_state_sha256_before: str | None = None

    def reset_inference_runtime_counters(self) -> None:
        """Reset both inherited counters and the v5 extraction audit state."""

        super().reset_inference_runtime_counters()
        self._v5_extraction_runtime = FeatureExtractionRuntimeCounters(
            tile_batch_size_requested=int(getattr(self, "tile_batch_size", 1)),
            tta_batch_size_requested=int(getattr(self, "tta_batch_size", 1)),
        )
        self._v5_case_extractions_completed = 0
        self._v5_neural_head_forward_calls = 0
        self._v5_class_offset_applications = 0
        self._v5_feature_cache_reads = 0
        self._v5_neural_bag_sha256_sequence: list[str] = []

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: tuple[int | str, ...] | None,
        checkpoint_name: str = "checkpoint_final.pth",
    ) -> None:
        """Load exactly one explicit fold and freeze every shared-network weight."""

        if (
            use_folds is None
            or len(use_folds) != 1
            or isinstance(use_folds[0], bool)
            or not isinstance(use_folds[0], int)
            or use_folds[0] != 0
        ):
            raise ValueError("V5 neural inference requires exactly the locked numeric fold 0")
        if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError("V5 checkpoint must be one bare filename")
        super().initialize_from_trained_model_folder(
            model_training_output_dir,
            use_folds=use_folds,
            checkpoint_name=checkpoint_name,
        )
        parameters = self.list_of_parameters
        if parameters is None or len(parameters) != 1:
            raise RuntimeError("V5 inference did not load exactly one fold state")
        network = _network_module(self.network)
        network.load_state_dict(parameters[0], strict=True)
        network.to(self.device)
        network.eval()
        network.requires_grad_(False)
        actual_component_hashes = component_hashes(network)
        if actual_component_hashes != FROZEN_COMPONENT_HASHES:
            raise RuntimeError("Loaded nnU-Net components differ from the locked v5 state")
        self._initialized_fold = use_folds[0]
        self._frozen_component_hashes = actual_component_hashes

    def load_final_neural_case_head(self) -> None:
        """Load and freeze the separately hashed v5 head inside the timed region."""

        if self._frozen_component_hashes is None or self._initialized_fold is None:
            raise RuntimeError("The single-fold frozen network must be initialized first")
        model, offsets, metadata = load_neural_case_head_bundle(
            self._neural_case_head_bundle_path,
            self.device,
            expected_bundle_sha256=(
                self._expected_neural_case_head_bundle_sha256
            ),
            expected_numeric_dataset_sha256=(
                self._expected_numeric_train_dataset_sha256
            ),
        )
        model.eval()
        model.requires_grad_(False)
        state_hash = neural_state_sha256(model)
        if state_hash != metadata["refit_final_state_sha256"]:
            raise RuntimeError("Loaded neural case-head state differs from its final audit")
        self._neural_case_head = model
        self._class_offsets = np.asarray(offsets, dtype=np.float64)
        self._neural_bundle_metadata = dict(metadata)
        self._neural_head_state_sha256_before = state_hash

    def _require_ready(self) -> tuple[nn.Module, np.ndarray]:
        if self._frozen_component_hashes is None or self._initialized_fold is None:
            raise RuntimeError("The single-fold frozen network is not initialized")
        if self._neural_case_head is None or self._class_offsets is None:
            raise RuntimeError("The final v5 neural case-head bundle is not loaded")
        if self._neural_case_head.training or any(
            parameter.requires_grad for parameter in self._neural_case_head.parameters()
        ):
            raise RuntimeError("The final v5 neural case head is not frozen in eval mode")
        return self._neural_case_head, self._class_offsets

    @torch.inference_mode()
    def predict_joint_from_preprocessed_data(self, data: Tensor) -> JointPrediction:
        """Extract one label-free bag and return segmentation plus v5 probabilities."""

        neural_head, class_offsets = self._require_ready()
        if self.v5_extraction_mode == "full":
            extraction = extract_case_from_preprocessed(
                self,
                data,
                tile_batch_size=1,
                tta_batch_size=1,
                runtime_counters=self._v5_extraction_runtime,
            )
        elif self.v5_extraction_mode == "neural_only":
            extraction = extract_neural_only_case_from_preprocessed(
                self,
                data,
                runtime_counters=self._v5_extraction_runtime,
            )
        else:
            raise RuntimeError("V5 extraction mode changed after initialization")
        self._v5_case_extractions_completed += 1
        bag = build_neural_case_bag(
            tile_vectors=extraction.tile_vectors,
            tile_evidence=extraction.tile_evidence,
            tile_vector_names=extraction.tile_vector_names,
            mil_stage3_maps=extraction.mil_stage3_maps,
            mil_prediction_maps=extraction.mil_prediction_maps,
            mil_lesion_mass=extraction.mil_lesion_mass,
        )
        self._v5_neural_bag_sha256_sequence.append(_neural_bag_sha256(bag))
        prediction = predict_neural_case_extraction(
            neural_head,
            extraction,
            class_offsets,
        )
        self._v5_neural_head_forward_calls += 1
        self._v5_class_offset_applications += 1
        probabilities = prediction.offset_probabilities.detach()
        if (
            probabilities.shape != (3,)
            or probabilities.dtype != torch.float64
            or not torch.isfinite(probabilities).all()
            or not torch.isclose(
                probabilities.sum(),
                torch.ones((), dtype=torch.float64, device=probabilities.device),
                atol=1e-12,
                rtol=1e-12,
            )
            or int(torch.argmax(probabilities).item()) != prediction.subtype
        ):
            raise RuntimeError("V5 neural case head returned invalid adjusted probabilities")
        return JointPrediction(extraction.segmentation_logits, probabilities)

    def predict_from_files_joint(
        self,
        input_folder: str | os.PathLike[str],
        output_folder: str | os.PathLike[str],
        *,
        classification_csv: str | os.PathLike[str] | None = None,
        probability_csv: str | os.PathLike[str] | None = None,
        save_segmentation_probabilities: bool = False,
        overwrite: bool = False,
    ) -> list[JointCasePrediction]:
        """Export v5 outputs without reducing adjusted probabilities to float32."""

        if not overwrite:
            raise ValueError("V5 raw inference requires overwrite=True")
        if self.configuration_manager.previous_stage_name is not None:
            raise NotImplementedError(
                "V5 raw inference does not accept previous-stage segmentations"
            )
        output_directory = Path(output_folder)
        if Path(input_folder).resolve() == output_directory.resolve():
            raise ValueError("Input and output directories must be different")
        output_directory.mkdir(parents=True, exist_ok=True)
        classification_path = (
            Path(classification_csv)
            if classification_csv is not None
            else output_directory / "subtype_results.csv"
        )
        probability_path = Path(probability_csv) if probability_csv is not None else None
        if (
            probability_path is not None
            and probability_path.resolve() == classification_path.resolve()
        ):
            raise ValueError("classification_csv and probability_csv must be different files")

        file_ending = self.dataset_json["file_ending"]
        if file_ending != SUBMISSION_FILE_ENDING:
            raise ValueError(
                "The strict v5 submission pipeline requires file ending "
                f"{SUBMISSION_FILE_ENDING!r}"
            )
        channel_names = self.dataset_json.get("channel_names")
        expected_channels = len(channel_names) if isinstance(channel_names, dict) else None
        cases = discover_nnunet_input_cases(
            input_folder,
            file_ending,
            expected_channels=expected_channels,
        )
        preprocessor = self.configuration_manager.preprocessor_class(
            verbose=self.verbose_preprocessing
        )
        labels: dict[str, int] = {}
        probabilities: dict[str, tuple[float, float, float]] = {}
        results: list[JointCasePrediction] = []
        try:
            for case_id, image_files in cases:
                data, _segmentation, properties = preprocessor.run_case(
                    image_files,
                    None,
                    self.plans_manager,
                    self.configuration_manager,
                    self.dataset_json,
                )
                input_tensor = torch.from_numpy(
                    np.asarray(data, dtype=np.float32)
                ).contiguous()
                prediction = self.predict_joint_from_preprocessed_data(input_tensor)
                probability_tensor = prediction.classification_probabilities.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
                if probability_tensor.shape != (3,):
                    raise ValueError(
                        f"Expected three v5 subtype probabilities for {case_id!r}"
                    )
                if (
                    not torch.isfinite(probability_tensor).all()
                    or torch.any(probability_tensor < 0)
                    or torch.any(probability_tensor > 1)
                    or not torch.isclose(
                        probability_tensor.sum(),
                        torch.ones((), dtype=torch.float64),
                        atol=1e-12,
                        rtol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"Invalid adjusted v5 probabilities for {case_id!r}"
                    )
                subtype = int(torch.argmax(probability_tensor).item())
                case_probabilities = (
                    float(probability_tensor[0]),
                    float(probability_tensor[1]),
                    float(probability_tensor[2]),
                )

                output_base = output_directory / case_id
                segmentation_path = Path(f"{output_base}{file_ending}")
                export_prediction_from_logits(
                    prediction.segmentation_logits.detach().cpu(),
                    properties,
                    self.configuration_manager,
                    self.plans_manager,
                    self.dataset_json,
                    str(output_base),
                    save_segmentation_probabilities,
                )
                if not segmentation_path.is_file():
                    raise RuntimeError(
                        f"nnU-Net export did not create the expected mask: {segmentation_path}"
                    )
                labels[case_id] = subtype
                probabilities[case_id] = case_probabilities
                results.append(
                    JointCasePrediction(
                        case_id,
                        subtype,
                        segmentation_path,
                        case_probabilities,
                    )
                )

            write_classification_csv(
                classification_path,
                {
                    submission_filename(case_id, file_ending): labels[case_id]
                    for case_id, _ in cases
                },
            )
            if probability_path is not None:
                _write_v5_probability_csv(
                    probability_path,
                    {
                        submission_filename(case_id, file_ending): labels[case_id]
                        for case_id, _ in cases
                    },
                    {
                        submission_filename(case_id, file_ending): probabilities[case_id]
                        for case_id, _ in cases
                    },
                )
            return sorted(results, key=lambda result: result.case_id)
        finally:
            compute_gaussian.cache_clear()
            empty_cache(self.device)

    def inference_runtime_provenance(self) -> dict[str, object]:
        """Return fail-closed proof that every case executed the complete v5 path."""

        execution = self._v5_extraction_runtime.provenance()
        execution.update(
            {
                "classifier_pipeline": V5_CLASSIFIER_PIPELINE,
                "v5_extraction_mode": self.v5_extraction_mode,
                "speed_v3_network_batch_ceiling": 1,
                "v5_feature_extraction_executed": (
                    self._v5_case_extractions_completed > 0
                ),
                "v5_case_extractions_completed": (
                    self._v5_case_extractions_completed
                ),
                "v5_neural_head_forward_calls": (
                    self._v5_neural_head_forward_calls
                ),
                "v5_class_offset_applications": (
                    self._v5_class_offset_applications
                ),
                "v5_feature_cache_reads": self._v5_feature_cache_reads,
                "case_identifiers_or_paths_used_as_model_inputs": False,
                "v5_neural_bag_sha256_sequence": list(
                    self._v5_neural_bag_sha256_sequence
                ),
            }
        )
        return execution

    def neural_case_head_provenance(self) -> dict[str, object]:
        """Describe the exact verified head, offsets, locks, and training dataset."""

        neural_head, class_offsets = self._require_ready()
        metadata = self._neural_bundle_metadata
        if metadata is None:
            raise RuntimeError("Neural case-head metadata is unavailable")
        before_state_hash = self._neural_head_state_sha256_before
        if before_state_hash is None:
            raise RuntimeError("Neural case-head initial state hash is unavailable")
        after_state_hash = neural_state_sha256(neural_head)
        if (
            before_state_hash != metadata["refit_final_state_sha256"]
            or after_state_hash != before_state_hash
        ):
            raise RuntimeError("Neural case-head state changed during v5 inference")
        actual_bundle_hash = file_sha256(self._neural_case_head_bundle_path)
        if actual_bundle_hash != self._expected_neural_case_head_bundle_sha256:
            raise RuntimeError("Neural case-head bundle changed after it was loaded")
        return {
            "classifier_pipeline": V5_CLASSIFIER_PIPELINE,
            "bundle_path": str(self._neural_case_head_bundle_path),
            "bundle_name": self._neural_case_head_bundle_path.name,
            "bundle_sha256": actual_bundle_hash,
            "bundle_size_bytes": self._neural_case_head_bundle_path.stat().st_size,
            "expected_bundle_sha256_verified": True,
            "numeric_train_dataset_sha256": (
                self._expected_numeric_train_dataset_sha256
            ),
            "selected_candidate_id": metadata["selected_candidate_id"],
            "head_parameter_count": sum(
                parameter.numel() for parameter in neural_head.parameters()
            ),
            "head_in_eval_mode": not neural_head.training,
            "any_head_parameter_requires_grad": any(
                parameter.requires_grad for parameter in neural_head.parameters()
            ),
            "head_state_sha256": metadata["refit_final_state_sha256"],
            "head_state_sha256_before": before_state_hash,
            "head_state_sha256_after": after_state_hash,
            "head_state_unchanged": True,
            "class_offsets": [float(value) for value in class_offsets],
            "neural_lock_sha256": metadata["neural_lock_sha256"],
            "decision_lock_sha256": metadata["decision_lock_sha256"],
            "selection_audit_sha256": metadata["selection_audit_sha256"],
            "calibration_audit_sha256": metadata["calibration_audit_sha256"],
            "refit_audit_sha256": metadata["refit_audit_sha256"],
            "eligible_for_official": metadata["eligible_for_official"],
            "bundle_loaded_strictly": True,
        }

    def frozen_network_provenance(self) -> dict[str, object]:
        """Verify that feature extraction never changed the checkpoint state."""

        if self._frozen_component_hashes is None or self._initialized_fold is None:
            raise RuntimeError("Frozen network provenance is unavailable")
        after = component_hashes(_network_module(self.network))
        if after != self._frozen_component_hashes:
            raise RuntimeError("Frozen nnU-Net components changed during v5 inference")
        return {
            "fold": self._initialized_fold,
            "component_hashes_before": self._frozen_component_hashes,
            "component_hashes_after": after,
            "frozen_components_unchanged": True,
            "network_in_eval_mode": not _network_module(self.network).training,
            "any_network_parameter_requires_grad": any(
                parameter.requires_grad
                for parameter in _network_module(self.network).parameters()
            ),
        }


__all__ = [
    "LOCKED_BATCH_CONFIGURATIONS",
    "LOCKED_NETWORK_MICROBATCH_CEILING",
    "NeuralCaseNNUNetPredictor",
    "V5_CLASSIFIER_PIPELINE",
    "V5_EXTRACTION_MODES",
    "extract_neural_only_case_from_preprocessed",
    "mirror_mean_neural_only_tile_features",
]
