"""Explicit joint segmentation/classification inference for nnU-Net v2.8.1.

nnU-Net's stock sliding-window implementation assumes that every network call
returns one segmentation tensor.  The multi-task network deliberately keeps
that default, while this module provides a separate path that carries the
classification result through test-time augmentation, tiles, and folds.

Tiles may be forwarded in small consecutive batches while remaining sequential
at the accumulator.  This preserves the declared tile-order aggregation for
both outputs while reducing Python and kernel-launch overhead.  All auxiliary
aggregation remains local to one prediction attempt.  In particular, an
unsuccessful on-device accumulation cannot leak classification votes into the
CPU-results retry.
"""

from __future__ import annotations

import csv
import itertools
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from torch import Tensor
from torch._dynamo import OptimizedModule
from tqdm import tqdm

SUBMISSION_FILE_ENDING = ".nii.gz"
DEFAULT_PREDICTION_DEVICE = torch.device("cuda")


@dataclass(frozen=True, slots=True)
class JointPrediction:
    """A segmentation prediction and its case-classification probabilities.

    ``segmentation_logits`` retain nnU-Net's normal preprocessed-image shape.
    ``classification_probabilities`` are batched in the mirror helper and are
    a one-dimensional class vector after tile/fold aggregation.
    """

    segmentation_logits: Tensor
    classification_probabilities: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.segmentation_logits, Tensor):
            raise TypeError("segmentation_logits must be a torch.Tensor")
        if not isinstance(self.classification_probabilities, Tensor):
            raise TypeError("classification_probabilities must be a torch.Tensor")
        if self.classification_probabilities.ndim not in (1, 2):
            raise ValueError(
                "classification_probabilities must have shape (classes,) or "
                f"(batch, classes), got {self.classification_probabilities.shape}"
            )
        if self.classification_probabilities.shape[-1] < 2:
            raise ValueError("classification_probabilities must contain at least two classes")


@dataclass(frozen=True, slots=True)
class JointCasePrediction:
    """Serializable case-level outcome from raw-file joint inference."""

    case_id: str
    subtype: int
    segmentation_path: Path
    classification_probabilities: tuple[float, ...] | None


def discover_nnunet_input_cases(
    input_folder: str | os.PathLike[str],
    file_ending: str,
    *,
    expected_channels: int | None = None,
) -> list[tuple[str, list[str]]]:
    """Discover ``case_XXXX<ending>`` inputs without a multiprocessing pool."""

    folder = Path(input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {folder}")
    pattern = re.compile(rf"^(.+)_([0-9]{{4}}){re.escape(file_ending)}$")
    grouped: dict[str, dict[int, str]] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not path.name.endswith(file_ending):
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(
                f"Input image does not follow case_XXXX{file_ending}: {path.name}"
            )
        case_id, channel_text = match.groups()
        channel = int(channel_text)
        case_channels = grouped.setdefault(case_id, {})
        if channel in case_channels:
            raise ValueError(f"Duplicate channel {channel:04d} for case {case_id!r}")
        case_channels[channel] = str(path.resolve())

    if not grouped:
        raise ValueError(f"No *{file_ending} nnU-Net channel images found in {folder}")

    cases: list[tuple[str, list[str]]] = []
    for case_id, channel_mapping in sorted(grouped.items()):
        channels = sorted(channel_mapping)
        expected_indices = list(range(len(channels)))
        if channels != expected_indices:
            raise ValueError(
                f"Channels for case {case_id!r} must be contiguous from 0000; got {channels}"
            )
        if expected_channels is not None and len(channels) != expected_channels:
            raise ValueError(
                f"Case {case_id!r} has {len(channels)} channels; expected {expected_channels}"
            )
        cases.append((case_id, [channel_mapping[index] for index in channels]))
    return cases


def submission_filename(case_id: str, file_ending: str) -> str:
    """Return the exact mask filename required in ``subtype_results.csv``."""

    normalized_case_id = str(case_id).strip()
    if not normalized_case_id:
        raise ValueError("case_id must not be blank")
    if Path(normalized_case_id).name != normalized_case_id:
        raise ValueError(f"case_id must not contain a directory: {case_id!r}")
    if not file_ending or not file_ending.startswith("."):
        raise ValueError(f"file_ending must be a non-empty extension, got {file_ending!r}")
    if normalized_case_id.endswith(file_ending):
        return normalized_case_id
    return f"{normalized_case_id}{file_ending}"


def normalize_submission_mapping(
    values: dict[str, object], file_ending: str
) -> dict[str, object]:
    """Convert strict submission filenames back to bare internal case IDs."""

    normalized: dict[str, object] = {}
    for raw_name, value in values.items():
        name = str(raw_name).strip()
        if Path(name).name != name:
            raise ValueError(f"Submission name must not contain a directory: {name!r}")
        if not name.endswith(file_ending):
            raise ValueError(
                f"Submission name must end with {file_ending!r}, got {name!r}"
            )
        case_id = name[: -len(file_ending)]
        if not case_id or case_id in normalized:
            raise ValueError(
                f"Blank or duplicate case after removing {file_ending!r}: {name!r}"
            )
        normalized[case_id] = value
    return normalized


def read_classification_csv(path: str | os.PathLike[str]) -> dict[str, int]:
    """Read and strictly validate the required ``Names,Subtype`` CSV."""

    source = Path(path)
    if not source.is_file():
        return {}
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Names", "Subtype"]:
            raise ValueError(
                f"Classification CSV header must be exactly Names,Subtype: {source}"
            )
        labels: dict[str, int] = {}
        for row in reader:
            case_id = str(row["Names"]).strip()
            if (
                Path(case_id).name != case_id
                or not case_id.endswith(SUBMISSION_FILE_ENDING)
            ):
                raise ValueError(
                    "Submission Names must be bare .nii.gz filenames, got "
                    f"{case_id!r}"
                )
            if not case_id or case_id in labels:
                raise ValueError(f"Blank or duplicate case name in {source}: {case_id!r}")
            try:
                subtype = int(row["Subtype"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid subtype for {case_id!r} in {source}") from exc
            if subtype not in (0, 1, 2):
                raise ValueError(f"Subtype for {case_id!r} must be 0, 1, or 2")
            labels[case_id] = subtype
    return labels


def _read_probability_csv(path: Path) -> dict[str, tuple[float, ...]]:
    if not path.is_file():
        return {}
    expected_header = [
        "Names",
        "Subtype",
        "Probability_0",
        "Probability_1",
        "Probability_2",
    ]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise ValueError(
                "Probability CSV header must be exactly "
                f"{','.join(expected_header)}: {path}"
            )
        probabilities: dict[str, tuple[float, ...]] = {}
        for row in reader:
            case_id = str(row["Names"]).strip()
            if (
                Path(case_id).name != case_id
                or not case_id.endswith(SUBMISSION_FILE_ENDING)
            ):
                raise ValueError(
                    "Probability CSV Names must be bare .nii.gz filenames, got "
                    f"{case_id!r}"
                )
            if not case_id or case_id in probabilities:
                raise ValueError(f"Blank or duplicate case name in {path}: {case_id!r}")
            try:
                values = tuple(float(row[f"Probability_{index}"]) for index in range(3))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid probabilities for {case_id!r} in {path}") from exc
            if (
                not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
                or not np.isclose(sum(values), 1.0, atol=1e-4)
            ):
                raise ValueError(f"Invalid probability distribution for {case_id!r} in {path}")
            probabilities[case_id] = values
    return probabilities


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_classification_csv(
    path: str | os.PathLike[str], labels: dict[str, int]
) -> None:
    """Atomically write a deterministic strict submission CSV."""

    invalid = {case_id: value for case_id, value in labels.items() if value not in (0, 1, 2)}
    if invalid:
        raise ValueError(f"Subtype labels must be 0, 1, or 2: {invalid}")
    invalid_names = [
        case_id
        for case_id in labels
        if Path(case_id).name != case_id
        or not case_id.endswith(SUBMISSION_FILE_ENDING)
    ]
    if invalid_names:
        raise ValueError(
            "Submission Names must be bare .nii.gz filenames: "
            f"{sorted(invalid_names)}"
        )
    rows = ({"Names": case_id, "Subtype": labels[case_id]} for case_id in sorted(labels))
    _atomic_write_csv(Path(path), ("Names", "Subtype"), rows)


def _write_probability_csv(
    path: Path,
    labels: dict[str, int],
    probabilities: dict[str, tuple[float, ...]],
) -> None:
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
                f"Probability_{index}": f"{probabilities[case_id][index]:.9g}"
                for index in range(3)
            },
        }
        for case_id in sorted(labels)
    )
    _atomic_write_csv(path, fieldnames, rows)


def average_classification_probabilities(probabilities: Iterable[Tensor]) -> Tensor:
    """Average equally weighted class-probability vectors in float32.

    This helper intentionally does not accept logits: probability averaging is
    the declared ensemble rule for mirror views, spatial tiles, and folds.
    """

    items = tuple(probability.float() for probability in probabilities)
    if not items:
        raise ValueError("At least one classification probability tensor is required")
    reference_shape = items[0].shape
    if any(item.shape != reference_shape for item in items[1:]):
        raise ValueError("All classification probability tensors must have the same shape")
    if any(not torch.isfinite(item).all() for item in items):
        raise ValueError("Classification probabilities must be finite")
    return torch.stack(items, dim=0).mean(dim=0)


class JointNNUNetPredictor(nnUNetPredictor):
    """nnU-Net v2.8.1 predictor with a pure-return joint inference path.

    Supported joint entry points consume already preprocessed tensors.  Stock
    file preprocessing and segmentation export remain available through the
    inherited methods, but they are segmentation-only; case-level probabilities
    must not be passed through nnU-Net's spatial resampling/export functions.
    """

    def __init__(
        self,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_device: bool = True,
        device: torch.device = DEFAULT_PREDICTION_DEVICE,
        verbose: bool = False,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = True,
        *,
        tile_batch_size: int = 1,
        tta_batch_size: int = 1,
    ) -> None:
        if isinstance(tile_batch_size, bool) or not isinstance(tile_batch_size, int):
            raise TypeError("tile_batch_size must be an integer")
        if tile_batch_size < 1:
            raise ValueError("tile_batch_size must be at least 1")
        if isinstance(tta_batch_size, bool) or not isinstance(tta_batch_size, int):
            raise TypeError("tta_batch_size must be an integer")
        if tta_batch_size < 1:
            raise ValueError("tta_batch_size must be at least 1")
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm,
        )
        self.tile_batch_size = tile_batch_size
        self.tta_batch_size = tta_batch_size
        self.reset_inference_runtime_counters()

    def reset_inference_runtime_counters(self) -> None:
        """Reset counters used to audit batched sliding-window execution."""

        requested = getattr(self, "tile_batch_size", 1)
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError("tile_batch_size must be an integer of at least 1")
        self._adaptive_tile_batch_size = requested
        requested_tta = getattr(self, "tta_batch_size", 1)
        if (
            isinstance(requested_tta, bool)
            or not isinstance(requested_tta, int)
            or requested_tta < 1
        ):
            raise ValueError("tta_batch_size must be an integer of at least 1")
        self._adaptive_tta_batch_size = requested_tta
        self._tile_batch_oom_fallback_count = 0
        self._tta_batch_oom_fallback_count = 0
        self._logical_tile_batches_completed = 0
        self._logical_tiles_completed = 0
        self._tta_view_batches_completed = 0
        self._tta_views_completed = 0
        self._joint_network_forward_calls = 0
        self._maximum_network_batch_size_observed = 0
        self._network_batch_size_histogram: dict[int, int] = {}
        self._tile_batch_size_histogram: dict[int, int] = {}
        self._tta_batch_size_histogram: dict[int, int] = {}

    def inference_runtime_provenance(self) -> dict[str, object]:
        """Return JSON-safe execution counters without resetting them."""

        requested = int(getattr(self, "tile_batch_size", 1))
        adaptive = int(getattr(self, "_adaptive_tile_batch_size", requested))
        requested_tta = int(getattr(self, "tta_batch_size", 1))
        adaptive_tta = int(
            getattr(self, "_adaptive_tta_batch_size", requested_tta)
        )
        histogram = getattr(self, "_tile_batch_size_histogram", {})
        tta_histogram = getattr(self, "_tta_batch_size_histogram", {})
        return {
            "joint_network_forward_calls": int(
                getattr(self, "_joint_network_forward_calls", 0)
            ),
            "maximum_network_batch_size_observed": int(
                getattr(self, "_maximum_network_batch_size_observed", 0)
            ),
            "network_batch_size_histogram": {
                str(size): int(count)
                for size, count in sorted(
                    getattr(self, "_network_batch_size_histogram", {}).items()
                )
            },
            "network_batch_size_limit": max(requested, requested_tta),
            "logical_tile_batches_completed": int(
                getattr(self, "_logical_tile_batches_completed", 0)
            ),
            "logical_tiles_completed": int(
                getattr(self, "_logical_tiles_completed", 0)
            ),
            "tile_batch_oom_fallback_count": int(
                getattr(self, "_tile_batch_oom_fallback_count", 0)
            ),
            "tile_batch_size_adaptive_limit": adaptive,
            "tile_batch_size_histogram": {
                str(size): int(count) for size, count in sorted(histogram.items())
            },
            "tile_batch_size_requested": requested,
            "tta_batch_oom_fallback_count": int(
                getattr(self, "_tta_batch_oom_fallback_count", 0)
            ),
            "tta_batch_size_adaptive_limit": adaptive_tta,
            "tta_batch_size_histogram": {
                str(size): int(count)
                for size, count in sorted(tta_histogram.items())
            },
            "tta_batch_size_requested": requested_tta,
            "tta_view_batches_completed": int(
                getattr(self, "_tta_view_batches_completed", 0)
            ),
            "tta_views_completed": int(getattr(self, "_tta_views_completed", 0)),
        }

    @staticmethod
    def _is_out_of_memory_error(error: RuntimeError) -> bool:
        return isinstance(error, torch.OutOfMemoryError) or (
            "out of memory" in str(error).lower()
        )

    def _record_joint_forward(self, batch_size: int) -> None:
        self._joint_network_forward_calls = int(
            getattr(self, "_joint_network_forward_calls", 0)
        ) + 1
        self._maximum_network_batch_size_observed = max(
            int(getattr(self, "_maximum_network_batch_size_observed", 0)),
            batch_size,
        )
        histogram = getattr(self, "_network_batch_size_histogram", None)
        if histogram is None:
            histogram = {}
            self._network_batch_size_histogram = histogram
        histogram[batch_size] = histogram.get(batch_size, 0) + 1

    def _record_completed_tile_batch(self, batch_size: int) -> None:
        self._logical_tile_batches_completed = int(
            getattr(self, "_logical_tile_batches_completed", 0)
        ) + 1
        self._logical_tiles_completed = int(
            getattr(self, "_logical_tiles_completed", 0)
        ) + batch_size
        histogram = getattr(self, "_tile_batch_size_histogram", None)
        if histogram is None:
            histogram = {}
            self._tile_batch_size_histogram = histogram
        histogram[batch_size] = histogram.get(batch_size, 0) + 1

    def _record_completed_tta_batch(self, batch_size: int) -> None:
        self._tta_view_batches_completed = int(
            getattr(self, "_tta_view_batches_completed", 0)
        ) + 1
        self._tta_views_completed = int(
            getattr(self, "_tta_views_completed", 0)
        ) + batch_size
        histogram = getattr(self, "_tta_batch_size_histogram", None)
        if histogram is None:
            histogram = {}
            self._tta_batch_size_histogram = histogram
        histogram[batch_size] = histogram.get(batch_size, 0) + 1

    @staticmethod
    def _validate_joint_network_output(output: object) -> tuple[Tensor, Tensor]:
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(
                "The multi-task network must return (segmentation_logits, "
                "classification_logits) when return_classification=True"
            )
        segmentation_logits, classification_logits = output
        if not isinstance(segmentation_logits, Tensor):
            raise TypeError(
                "Inference requires a segmentation tensor. Ensure deep supervision is disabled."
            )
        if not isinstance(classification_logits, Tensor) or classification_logits.ndim != 2:
            shape = getattr(classification_logits, "shape", None)
            raise TypeError(
                "classification_logits must be a tensor with shape (batch, classes), "
                f"got {shape}"
            )
        if segmentation_logits.shape[0] != classification_logits.shape[0]:
            raise ValueError("Segmentation and classification batch sizes do not match")
        return segmentation_logits, classification_logits

    def _forward_joint(self, x: Tensor) -> tuple[Tensor, Tensor]:
        self._record_joint_forward(int(x.shape[0]))
        return self._validate_joint_network_output(
            self.network(x, return_classification=True)
        )

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict_joint(self, x: Tensor) -> JointPrediction:
        """Average segmentation logits and class probabilities over mirror TTA."""

        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        if mirror_axes and (
            max(mirror_axes) > x.ndim - 3 or min(mirror_axes) < 0
        ):
            raise ValueError(
                f"mirror axes {mirror_axes} do not match input shape {tuple(x.shape)}"
            )
        tensor_axes = (
            tuple(axis + 2 for axis in mirror_axes) if mirror_axes else ()
        )
        combinations = [
            combination
            for count in range(1, len(tensor_axes) + 1)
            for combination in itertools.combinations(tensor_axes, count)
        ]

        requested_tta_batch_size = int(getattr(self, "tta_batch_size", 1))
        if requested_tta_batch_size < 1:
            raise ValueError("tta_batch_size must be at least 1")

        # Keep the batch-one arm byte-for-byte on the original execution path.
        # It is the prospective reference for the causal speed comparison.
        if requested_tta_batch_size == 1:
            segmentation, classification_logits = self._forward_joint(x)
            classification_probabilities = torch.softmax(
                classification_logits.float(), dim=1
            )
            self._record_completed_tta_batch(1)
            for axes in combinations:
                mirrored_segmentation, mirrored_classification_logits = self._forward_joint(
                    torch.flip(x, axes)
                )
                segmentation = segmentation + torch.flip(mirrored_segmentation, axes)
                classification_probabilities = classification_probabilities + torch.softmax(
                    mirrored_classification_logits.float(), dim=1
                )
                self._record_completed_tta_batch(1)
            divisor = len(combinations) + 1
            return JointPrediction(
                segmentation / divisor,
                classification_probabilities / divisor,
            )

        view_axes = [(), *combinations]
        active_tta_batch_size = int(
            getattr(
                self,
                "_adaptive_tta_batch_size",
                requested_tta_batch_size,
            )
        )
        segmentation_sum = None
        classification_sum = None
        view_index = 0
        base_batch_size = x.shape[0]
        # Tile and view batching share one microbatch ceiling. Thus configured
        # size 2 never becomes a product batch of four: two tiles use one view
        # per forward, while a one-tile case can pack two mirror views.
        network_batch_size_limit = max(
            int(getattr(self, "tile_batch_size", 1)),
            requested_tta_batch_size,
        )
        active_tta_batch_size = min(
            active_tta_batch_size,
            max(1, network_batch_size_limit // base_batch_size),
        )
        while view_index < len(view_axes):
            batch_axes = view_axes[view_index : view_index + active_tta_batch_size]
            batched_input = None
            try:
                batched_input = torch.cat(
                    [x if not axes else torch.flip(x, axes) for axes in batch_axes],
                    dim=0,
                )
                batched_segmentation, batched_classification_logits = self._forward_joint(
                    batched_input
                )
            except RuntimeError as error:
                if (
                    active_tta_batch_size == 1
                    or not self._is_out_of_memory_error(error)
                ):
                    raise
                del batched_input
                batched_input = None
                empty_cache(self.device)
                active_tta_batch_size = max(1, active_tta_batch_size // 2)
                self._adaptive_tta_batch_size = active_tta_batch_size
                self._tta_batch_oom_fallback_count = int(
                    getattr(self, "_tta_batch_oom_fallback_count", 0)
                ) + 1
                if self.verbose:
                    print(
                        "TTA-view batch ran out of memory; retrying the untouched "
                        f"view group with batch size {active_tta_batch_size}"
                    )
                continue

            view_count = len(batch_axes)
            expected_network_batch = view_count * base_batch_size
            if batched_segmentation.shape[0] != expected_network_batch:
                raise ValueError("Segmentation batch cannot be restored to TTA views")
            if batched_classification_logits.shape[0] != expected_network_batch:
                raise ValueError("Classification batch cannot be restored to TTA views")
            segmentation_views = batched_segmentation.reshape(
                view_count, base_batch_size, *batched_segmentation.shape[1:]
            )
            probability_views = torch.softmax(
                batched_classification_logits.float(), dim=1
            ).reshape(
                view_count,
                base_batch_size,
                batched_classification_logits.shape[1],
            )
            for local_index, axes in enumerate(batch_axes):
                view_segmentation = segmentation_views[local_index]
                if axes:
                    view_segmentation = torch.flip(view_segmentation, axes)
                if segmentation_sum is None:
                    segmentation_sum = view_segmentation.clone()
                    classification_sum = probability_views[local_index].clone()
                else:
                    segmentation_sum = segmentation_sum + view_segmentation
                    classification_sum = classification_sum + probability_views[local_index]
            self._record_completed_tta_batch(view_count)
            view_index += view_count
            del (
                batched_classification_logits,
                batched_input,
                batched_segmentation,
                probability_views,
                segmentation_views,
            )

        if segmentation_sum is None or classification_sum is None:
            raise RuntimeError("TTA prediction produced no views")
        divisor = len(view_axes)
        return JointPrediction(
            segmentation_sum / divisor,
            classification_sum / divisor,
        )

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_joint(
        self,
        data: Tensor,
        slicers: Sequence[tuple[slice, ...]],
        do_on_device: bool = True,
    ) -> JointPrediction:
        """Aggregate one complete sliding-window attempt without shared state."""

        results_device = self.device if do_on_device else torch.device("cpu")
        predicted_logits = None
        n_predictions = None
        gaussian: Tensor | float | None = None
        classification_sum = None
        classification_count = 0
        workon = None
        progress = None

        try:
            empty_cache(self.device)
            data = data.to(results_device)
            predicted_logits = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            n_predictions = torch.zeros(
                data.shape[1:], dtype=torch.half, device=results_device
            )
            if self.use_gaussian:
                gaussian = compute_gaussian(
                    tuple(self.configuration_manager.patch_size),
                    sigma_scale=1.0 / 8,
                    value_scaling_factor=10,
                    device=results_device,
                )
            else:
                gaussian = 1.0

            requested_batch_size = int(getattr(self, "tile_batch_size", 1))
            if requested_batch_size < 1:
                raise ValueError("tile_batch_size must be at least 1")
            active_batch_size = int(
                getattr(self, "_adaptive_tile_batch_size", requested_batch_size)
            )
            progress = tqdm(total=len(slicers), disable=not self.allow_tqdm)
            tile_index = 0
            while tile_index < len(slicers):
                batch_slicers = tuple(
                    slicers[tile_index : tile_index + active_batch_size]
                )
                workon = None
                try:
                    workon = torch.stack(
                        [
                            torch.clone(
                                data[sliding_slice],
                                memory_format=torch.contiguous_format,
                            )
                            for sliding_slice in batch_slicers
                        ],
                        dim=0,
                    ).to(self.device)
                    tile = self._internal_maybe_mirror_and_predict_joint(workon)
                except RuntimeError as error:
                    if active_batch_size == 1 or not self._is_out_of_memory_error(error):
                        raise
                    del workon
                    workon = None
                    empty_cache(self.device)
                    active_batch_size = max(1, active_batch_size // 2)
                    self._adaptive_tile_batch_size = active_batch_size
                    self._tile_batch_oom_fallback_count = int(
                        getattr(self, "_tile_batch_oom_fallback_count", 0)
                    ) + 1
                    if self.verbose:
                        print(
                            "Tile batch ran out of memory; retrying the untouched "
                            f"tile group with batch size {active_batch_size}"
                        )
                    continue

                batch_count = len(batch_slicers)
                if tile.segmentation_logits.shape[0] != batch_count:
                    raise ValueError(
                        "Segmentation output batch size changed during sliding-window "
                        f"inference: expected {batch_count}, got "
                        f"{tile.segmentation_logits.shape[0]}"
                    )
                if tile.classification_probabilities.shape[0] != batch_count:
                    raise ValueError(
                        "Classification output batch size changed during sliding-window "
                        f"inference: expected {batch_count}, got "
                        f"{tile.classification_probabilities.shape[0]}"
                    )

                # Deliberately accumulate in the original slicer order. Batched
                # execution changes only network scheduling, not output weighting.
                for batch_index, sliding_slice in enumerate(batch_slicers):
                    tile_segmentation = tile.segmentation_logits[batch_index].to(
                        results_device
                    )
                    if self.use_gaussian:
                        tile_segmentation = tile_segmentation * gaussian
                    predicted_logits[sliding_slice] += tile_segmentation
                    n_predictions[sliding_slice[1:]] += gaussian

                    tile_probabilities = tile.classification_probabilities[
                        batch_index
                    ].to(device=results_device, dtype=torch.float32)
                    if classification_sum is None:
                        classification_sum = torch.zeros_like(tile_probabilities)
                    elif classification_sum.shape != tile_probabilities.shape:
                        raise ValueError(
                            "Classification output shape changed between tiles"
                        )
                    classification_sum += tile_probabilities
                    classification_count += 1

                self._record_completed_tile_batch(batch_count)
                tile_index += batch_count
                progress.update(batch_count)
                del tile, workon
                workon = None

            if classification_sum is None or classification_count == 0:
                raise RuntimeError("Sliding-window prediction produced no tiles")

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            if not torch.isfinite(predicted_logits).all():
                raise RuntimeError("Encountered a non-finite value in segmentation logits")
            classification_probabilities = classification_sum / classification_count
            if not torch.isfinite(classification_probabilities).all():
                raise RuntimeError("Encountered a non-finite classification probability")
            return JointPrediction(predicted_logits, classification_probabilities)
        except Exception:
            del predicted_logits, n_predictions, gaussian, classification_sum, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise
        finally:
            if progress is not None:
                progress.close()

    def _predict_sliding_window_with_results_fallback(
        self,
        data: Tensor,
        slicers: Sequence[tuple[slice, ...]],
    ) -> JointPrediction:
        """Run one atomic attempt and, like nnU-Net, retry GPU results on CPU.

        Both attempts return their classification result explicitly.  Nothing
        from a partially completed first attempt can be observed by the retry.
        """

        if self.perform_everything_on_device and self.device.type == "cuda":
            try:
                return self._internal_predict_sliding_window_return_joint(
                    data, slicers, True
                )
            except RuntimeError:
                if self.verbose:
                    print(
                        "Prediction on device was unsuccessful; retrying with "
                        "segmentation result arrays on CPU"
                    )
                empty_cache(self.device)
                return self._internal_predict_sliding_window_return_joint(
                    data, slicers, False
                )
        return self._internal_predict_sliding_window_return_joint(
            data, slicers, self.perform_everything_on_device
        )

    @torch.inference_mode()
    def predict_sliding_window_return_joint(self, input_image: Tensor) -> JointPrediction:
        """Predict one preprocessed image, retrying failed GPU accumulation on CPU."""

        if not isinstance(input_image, Tensor) or input_image.ndim != 4:
            raise ValueError("input_image must be a 4D tensor with shape (channels, x, y, z)")
        self.network = self.network.to(self.device)
        self.network.eval()
        empty_cache(self.device)

        context = (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            data, slicer_revert_padding = pad_nd_image(
                input_image,
                self.configuration_manager.patch_size,
                "constant",
                {"value": 0},
                True,
                None,
            )
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            prediction = self._predict_sliding_window_with_results_fallback(
                data, slicers
            )

            segmentation_logits = prediction.segmentation_logits[
                (slice(None), *slicer_revert_padding[1:])
            ]
        empty_cache(self.device)
        return JointPrediction(
            segmentation_logits,
            prediction.classification_probabilities,
        )

    @torch.inference_mode()
    def predict_joint_from_preprocessed_data(self, data: Tensor) -> JointPrediction:
        """Average explicit joint predictions over all configured folds.

        If ``manual_initialization`` supplied ``parameters=None``, the current
        network is treated as one fold.  This supports sequential validation as
        well as predictors initialized from trained-model folders.
        """

        old_num_threads = torch.get_num_threads()
        torch.set_num_threads(min(default_num_processes, old_num_threads))
        segmentation_sum = None
        classification_probabilities: list[Tensor] = []
        parameters = self.list_of_parameters
        fold_parameters: Sequence[dict[str, Tensor] | None]
        fold_parameters = (None,) if parameters is None else tuple(parameters)
        if not fold_parameters:
            torch.set_num_threads(old_num_threads)
            raise ValueError("No fold parameters are configured")

        try:
            for state_dict in fold_parameters:
                if state_dict is not None:
                    if isinstance(self.network, OptimizedModule):
                        self.network._orig_mod.load_state_dict(state_dict)
                    else:
                        self.network.load_state_dict(state_dict)

                fold_prediction = self.predict_sliding_window_return_joint(data)
                fold_segmentation = fold_prediction.segmentation_logits.to("cpu")
                if segmentation_sum is None:
                    segmentation_sum = fold_segmentation
                else:
                    if segmentation_sum.shape != fold_segmentation.shape:
                        raise ValueError("Segmentation output shape changed between folds")
                    segmentation_sum += fold_segmentation
                classification_probabilities.append(
                    fold_prediction.classification_probabilities.to(
                        device="cpu", dtype=torch.float32
                    )
                )

            if segmentation_sum is None:
                raise RuntimeError("Joint prediction did not produce a segmentation")
            if len(fold_parameters) > 1:
                segmentation_sum /= len(fold_parameters)
            return JointPrediction(
                segmentation_sum,
                average_classification_probabilities(classification_probabilities),
            )
        finally:
            torch.set_num_threads(old_num_threads)

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
        """Run sequential joint inference on raw nnU-Net-formatted image files.

        Raw files are processed with the configuration's official preprocessor,
        segmentation logits are restored with nnU-Net's stock exporter, and the
        case classification bypasses all spatial transforms. The final strict
        CSV is written atomically only after every requested case is complete.
        """

        if self.configuration_manager.previous_stage_name is not None:
            raise NotImplementedError(
                "Joint raw-file inference does not currently accept previous-stage "
                "segmentations required by cascaded configurations"
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
        if probability_path is not None and probability_path.resolve() == classification_path.resolve():
            raise ValueError("classification_csv and probability_csv must be different files")

        file_ending = self.dataset_json["file_ending"]
        if file_ending != SUBMISSION_FILE_ENDING:
            raise ValueError(
                "The strict submission pipeline requires dataset file_ending "
                f"{SUBMISSION_FILE_ENDING!r}, got {file_ending!r}"
            )
        channel_names = self.dataset_json.get("channel_names")
        expected_channels = len(channel_names) if isinstance(channel_names, dict) else None
        cases = discover_nnunet_input_cases(
            input_folder,
            file_ending,
            expected_channels=expected_channels,
        )
        requested_case_ids = {case_id for case_id, _ in cases}

        existing_labels = (
            {}
            if overwrite
            else normalize_submission_mapping(
                read_classification_csv(classification_path), file_ending
            )
        )
        existing_probabilities = {}
        if not overwrite and probability_path is not None:
            existing_probabilities = normalize_submission_mapping(
                _read_probability_csv(probability_path), file_ending
            )
        unexpected_labels = set(existing_labels) - requested_case_ids
        unexpected_probabilities = set(existing_probabilities) - requested_case_ids
        if unexpected_labels or unexpected_probabilities:
            raise ValueError(
                "Existing classification output contains cases absent from the input: "
                f"{sorted(unexpected_labels | unexpected_probabilities)}"
            )

        labels = dict(existing_labels)
        probabilities = dict(existing_probabilities)
        preprocessor = self.configuration_manager.preprocessor_class(
            verbose=self.verbose_preprocessing
        )
        results: list[JointCasePrediction] = []
        try:
            for case_id, image_files in cases:
                output_base = output_directory / case_id
                segmentation_path = Path(f"{output_base}{file_ending}")
                segmentation_complete = segmentation_path.is_file()
                if save_segmentation_probabilities:
                    segmentation_complete = (
                        segmentation_complete
                        and Path(f"{output_base}.npz").is_file()
                        and Path(f"{output_base}.pkl").is_file()
                    )
                classification_complete = case_id in labels
                probability_complete = probability_path is None or case_id in probabilities
                if (
                    not overwrite
                    and segmentation_complete
                    and classification_complete
                    and probability_complete
                ):
                    results.append(
                        JointCasePrediction(
                            case_id,
                            labels[case_id],
                            segmentation_path,
                            probabilities.get(case_id),
                        )
                    )
                    continue

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
                # This is the sole joint prediction call for this preprocessed case.
                prediction = self.predict_joint_from_preprocessed_data(input_tensor)
                case_probabilities = tuple(
                    float(value)
                    for value in prediction.classification_probabilities.detach()
                    .float()
                    .cpu()
                    .tolist()
                )
                if len(case_probabilities) != 3:
                    raise ValueError(
                        f"Expected three subtype probabilities for {case_id!r}, "
                        f"got {len(case_probabilities)}"
                    )
                if (
                    not all(
                        np.isfinite(value) and 0.0 <= value <= 1.0
                        for value in case_probabilities
                    )
                    or not np.isclose(sum(case_probabilities), 1.0, atol=1e-4)
                ):
                    raise ValueError(
                        f"Invalid classification probabilities for {case_id!r}: "
                        f"{case_probabilities}"
                    )
                subtype = int(np.argmax(case_probabilities))

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

            missing_labels = requested_case_ids - set(labels)
            if missing_labels:
                raise RuntimeError(
                    f"No classification prediction was produced for: {sorted(missing_labels)}"
                )
            write_classification_csv(
                classification_path,
                {
                    submission_filename(case_id, file_ending): labels[case_id]
                    for case_id in requested_case_ids
                },
            )
            if probability_path is not None:
                missing_probabilities = requested_case_ids - set(probabilities)
                if missing_probabilities:
                    raise RuntimeError(
                        "No probability details were produced for: "
                        f"{sorted(missing_probabilities)}"
                    )
                _write_probability_csv(
                    probability_path,
                    {
                        submission_filename(case_id, file_ending): labels[case_id]
                        for case_id in requested_case_ids
                    },
                    {
                        submission_filename(case_id, file_ending): probabilities[case_id]
                        for case_id in requested_case_ids
                    },
                )
            return sorted(results, key=lambda result: result.case_id)
        finally:
            compute_gaussian.cache_clear()
            empty_cache(self.device)


__all__ = [
    "JointCasePrediction",
    "JointNNUNetPredictor",
    "JointPrediction",
    "average_classification_probabilities",
    "discover_nnunet_input_cases",
    "normalize_submission_mapping",
    "read_classification_csv",
    "submission_filename",
    "write_classification_csv",
]
