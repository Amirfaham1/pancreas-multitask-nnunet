"""Overlap preprocessing and export with GPU work during joint inference.

Stock nnU-Net hides almost all of its CPU cost: it runs preprocessing in worker
processes (``-npp``), exports predictions in a second pool (``-nps``), and stages
sliding-window tiles from a producer thread. The joint predictor in this
repository did all three serially on the main thread, which is the single largest
reason the custom pipeline measured slower than stock despite doing byte-identical
network arithmetic.

Neither stage intentionally changes model arithmetic: preprocessing is
deterministic and export is pure post-processing. Nevertheless, the benchmark
compares fresh-process masks and reports any boundary-voxel differences instead
of inferring bit-exactness from the design.

One deliberate difference from nnU-Net's own ``preprocessing_iterator_fromfiles``:
that generator's drain loop is

    while (not done_events[w].is_set()) or (not target_queues[w].empty()):

which stops as soon as the *current* worker is finished and empty, silently
dropping items still queued behind later workers whenever
``len(cases) % num_processes != 0``. It happens to be safe at 72 cases over 3
workers. The implementation here tracks per-worker completion explicitly so an
uneven split cannot truncate the case list.
"""

from __future__ import annotations

import copy
import multiprocessing
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
import torch.nn.functional as torch_functional
from torch import Tensor, nn


class FrozenShallowEncoder(nn.Module):
    """CPU copy of only the encoder prefix needed for shallow classification."""

    def __init__(self, network: nn.Module, through_stage: int = 2) -> None:
        super().__init__()
        encoder = getattr(network, "encoder", None)
        stages = getattr(encoder, "stages", None)
        if encoder is None or stages is None:
            raise TypeError("network must expose encoder.stages")
        if not 0 <= through_stage < len(stages):
            raise ValueError("through_stage is outside the encoder")
        stem = getattr(encoder, "stem", None)
        self.stem = None if stem is None else copy.deepcopy(stem).cpu()
        self.stages = nn.ModuleList(
            copy.deepcopy(stage).cpu() for stage in stages[: through_stage + 1]
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def encode_to_stages(self, x: Tensor, stages: Sequence[int]) -> dict[int, Tensor]:
        wanted = tuple(int(stage) for stage in stages)
        if not wanted or min(wanted) < 0 or max(wanted) >= len(self.stages):
            raise ValueError("requested stage is outside the copied encoder prefix")
        features = x
        if self.stem is not None:
            features = self.stem(features)
        outputs = {}
        for position, stage in enumerate(self.stages):
            features = stage(features)
            if position in wanted:
                outputs[position] = features
        return outputs


@torch.inference_mode()
def cpu_shallow_features(
    network: nn.Module,
    volume_array: Any,
    *,
    mirror_axis_sets: Sequence[Sequence[int]],
    stages: Sequence[int],
    spatial_scale: float = 1.0,
) -> Any:
    """Extract mirror-averaged GAP features entirely on CPU."""

    import numpy as np

    from pancreas_multitask.wholevolume_dataset import pad_to_stride, stride_for_stage

    selected_stages = tuple(int(stage) for stage in stages)
    if not selected_stages:
        raise ValueError("stages must not be empty")
    scale = float(spatial_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("spatial_scale must be in (0, 1]")
    volume = torch.from_numpy(np.asarray(volume_array, dtype=np.float32))[None]
    if scale < 1.0:
        target_shape = tuple(max(2, round(length * scale)) for length in volume.shape[2:])
        volume = torch_functional.interpolate(
            volume, size=target_shape, mode="trilinear", align_corners=False
        )
    work = pad_to_stride(volume[0], stride_for_stage(max(selected_stages)))[None]
    totals: dict[int, Tensor] = {}
    axis_sets = tuple(tuple(int(axis) for axis in axes) for axes in mirror_axis_sets)
    if not axis_sets:
        raise ValueError("mirror_axis_sets must not be empty")
    for axes in axis_sets:
        view = torch.flip(work, axes) if axes else work
        for stage, features in network.encode_to_stages(view, selected_stages).items():
            pooled = features.float().mean(dim=(2, 3, 4))
            totals[stage] = pooled if stage not in totals else totals[stage] + pooled
    return torch.cat(
        [totals[stage] / len(axis_sets) for stage in selected_stages], dim=1
    )[0].numpy().astype(np.float32)


def cpu_shallow_feature_worker(
    network: nn.Module,
    task_queue: Any,
    result_queue: Any,
    *,
    mirror_axis_sets: Sequence[Sequence[int]],
    stages: Sequence[int],
    spatial_scale: float,
    torch_threads: int,
) -> None:
    """Multiprocessing target for independent CPU shallow-feature extraction."""

    torch.set_num_threads(int(torch_threads))
    try:
        while True:
            item = task_queue.get()
            if item is None:
                break
            index, volume_array = item
            result_queue.put(
                (
                    "result",
                    int(index),
                    cpu_shallow_features(
                        network,
                        volume_array,
                        mirror_axis_sets=mirror_axis_sets,
                        stages=stages,
                        spatial_scale=spatial_scale,
                    ),
                )
            )
    except BaseException as error:
        result_queue.put(("error", -1, f"{type(error).__name__}: {error}"))


def cpu_shallow_model_worker(
    model_folder: str,
    checkpoint_name: str,
    through_stage: int,
    task_queue: Any,
    result_queue: Any,
    *,
    mirror_axis_sets: Sequence[Sequence[int]],
    stages: Sequence[int],
    spatial_scale: float,
    torch_threads: int,
) -> None:
    """Load an independent CPU encoder prefix, then serve feature tasks.

    nnU-Net residual blocks contain local lambdas and cannot be pickled by the
    Windows ``spawn`` start method. Reconstructing from the checkpoint inside the
    worker avoids that unsafe serialization and also overlaps its one-time load
    with the main process's GPU inference.
    """

    torch.set_num_threads(int(torch_threads))
    try:
        from pancreas_multitask.predictor import JointNNUNetPredictor

        predictor = JointNNUNetPredictor(
            device=torch.device("cpu"),
            perform_everything_on_device=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(0,),
            checkpoint_name=checkpoint_name,
        )
        network = FrozenShallowEncoder(
            predictor.network, through_stage=int(through_stage)
        )
        cpu_shallow_feature_worker(
            network,
            task_queue,
            result_queue,
            mirror_axis_sets=mirror_axis_sets,
            stages=stages,
            spatial_scale=spatial_scale,
            torch_threads=torch_threads,
        )
    except BaseException as error:
        result_queue.put(("error", -1, f"{type(error).__name__}: {error}"))


@dataclass(frozen=True, slots=True)
class PreprocessedCase:
    """One preprocessed case handed from a producer to the GPU loop."""

    case_id: str
    data: Any
    properties: dict[str, Any]


def preprocess_in_background(
    cases: Sequence[tuple[str, Sequence[str]]],
    run_case: Callable[[Sequence[str]], tuple[Any, Any, dict[str, Any]]],
    *,
    prefetch: int = 2,
) -> Iterator[PreprocessedCase]:
    """Yield preprocessed cases in order while the next ones are prepared.

    A thread is the right tool here rather than a process: nnU-Net's preprocessing
    is dominated by SimpleITK reads and scipy resampling, both of which release the
    GIL, so a thread overlaps with GPU work without paying spawn cost or moving
    volumes through a pickle round-trip.

    ``prefetch`` bounds memory: at most that many preprocessed volumes are held
    beyond the one being consumed.
    """

    if prefetch < 1:
        raise ValueError("prefetch must be at least 1")
    pending: queue.Queue = queue.Queue(maxsize=prefetch)
    sentinel = object()

    def producer() -> None:
        try:
            for case_id, image_files in cases:
                data, _segmentation, properties = run_case(list(image_files))
                pending.put(PreprocessedCase(case_id, data, properties))
        except BaseException as error:  # surfaced on the consumer side
            pending.put(error)
        finally:
            pending.put(sentinel)

    thread = threading.Thread(target=producer, name="joint-preprocess", daemon=True)
    thread.start()
    try:
        while True:
            item = pending.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        thread.join(timeout=5.0)


class ExportPool:
    """Write predictions in worker processes instead of blocking the GPU loop.

    Export is expensive and entirely off the critical path: resampling logits back
    to native CT geometry, argmax, un-cropping and gzip-writing a NIfTI. Stock
    nnU-Net runs it in a spawn pool; this mirrors that so the comparison is
    like-for-like rather than a worker-count advantage in stock's favour.
    """

    def __init__(self, processes: int = 3, max_queued: int = 2) -> None:
        if processes < 1:
            raise ValueError("processes must be at least 1")
        self._processes = int(processes)
        self._max_queued = int(max_queued)
        self._pool: Any = None
        self._pending: list[Any] = []

    def __enter__(self) -> "ExportPool":
        self._pool = multiprocessing.get_context("spawn").Pool(self._processes)
        return self

    def submit(self, function: Callable[..., Any], arguments: tuple[Any, ...]) -> None:
        if self._pool is None:
            raise RuntimeError("ExportPool must be used as a context manager")
        # Bound the backlog so a slow disk cannot grow unbounded memory. Waiting
        # on the oldest export releases the main CPU instead of busy-spinning on
        # ``ready()``. The latter starved preprocessing/export workers on Windows
        # and made the supposedly overlapped pipeline slower than stock nnU-Net.
        while len(self._pending) >= self._max_queued:
            oldest = self._pending.pop(0)
            oldest.get()
        self._pending.append(self._pool.apply_async(function, arguments))

    def drain(self) -> None:
        for result in self._pending:
            result.get()
        self._pending.clear()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc is None:
                self.drain()
        finally:
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None


__all__ = [
    "ExportPool",
    "FrozenShallowEncoder",
    "PreprocessedCase",
    "cpu_shallow_feature_worker",
    "cpu_shallow_features",
    "cpu_shallow_model_worker",
    "preprocess_in_background",
]
