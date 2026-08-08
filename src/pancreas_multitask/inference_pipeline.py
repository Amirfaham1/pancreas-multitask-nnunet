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

import multiprocessing
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence


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
        # Bound the backlog so a slow disk cannot grow unbounded memory. Checking
        # readiness rather than sleeping blindly keeps the GPU loop moving.
        while sum(1 for r in self._pending if not r.ready()) >= self._max_queued:
            self._pending = [r for r in self._pending if not r.ready()]
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


__all__ = ["ExportPool", "PreprocessedCase", "preprocess_in_background"]
