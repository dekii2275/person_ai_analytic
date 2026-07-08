"""
src/benchmark.py — per-stage latency measurement for the pipeline.

Design rules:
  - No changes to pipeline behavior.
  - time.perf_counter() for wall-clock timing.
  - torch.cuda.synchronize() before/after inference timing to flush GPU queue.
  - warmup frames excluded from aggregate statistics.
  - No model-only FPS — all stages measured end-to-end.

Public API:
  StageTimer          context manager, returns ms elapsed
  FrameRecord         per-frame timing data (dataclass)
  BenchmarkResult     full benchmark result (dataclass)
  compute_stats()     compute mean/p50/p95/p99/min/max for a list of values
  aggregate_result()  build BenchmarkResult from list of FrameRecords
  save_json()         write benchmark.json
  print_summary()     print terminal table
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Dict, Generator, List, Optional

# ---------------------------------------------------------------------------
# CUDA synchronization helper
# ---------------------------------------------------------------------------

def _cuda_sync() -> None:
    """Call torch.cuda.synchronize() only when CUDA is available and in use.

    Importing torch here keeps the benchmark module independent of torch
    when CUDA is not available (CPU-only machines).
    """
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# StageTimer
# ---------------------------------------------------------------------------

@contextmanager
def StageTimer(cuda_sync: bool = False) -> Generator[List[float], None, None]:
    """Context manager that measures elapsed wall-clock time in milliseconds.

    Usage::

        with StageTimer(cuda_sync=True) as t:
            model(frame)
        elapsed_ms = t[0]

    Parameters
    ----------
    cuda_sync : bool
        If True, calls ``torch.cuda.synchronize()`` before *and* after
        the timed block to ensure GPU work is complete.  Required for
        accurate CUDA timing.

    Yields
    ------
    List[float]
        Single-element list.  After the ``with`` block, ``result[0]``
        contains elapsed milliseconds.
    """
    result: List[float] = [0.0]
    if cuda_sync:
        _cuda_sync()
    t0 = time.perf_counter()
    yield result
    if cuda_sync:
        _cuda_sync()
    result[0] = (time.perf_counter() - t0) * 1_000.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """Per-frame timing (all values in milliseconds)."""

    frame_index: int
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    draw_write_ms: float
    total_ms: float
    n_detections: int


@dataclass
class StageStats:
    """Aggregate statistics for a single stage."""

    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float


@dataclass
class BenchmarkResult:
    """Full benchmark result."""

    # Metadata
    model: str
    device: str
    video: str
    resolution: str
    fps: float

    # Aggregate per stage
    decode_ms: StageStats
    preprocess_ms: StageStats
    inference_ms: StageStats
    postprocess_ms: StageStats
    draw_write_ms: StageStats
    total_ms: StageStats

    # Top-level metrics
    processed_frames: int
    warmup_frames: int
    total_wall_time_s: float
    effective_fps: float

    # Per-frame records
    frames: List[FrameRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(values: List[float]) -> StageStats:
    """Compute mean/p50/p95/p99/min/max for a list of ms values.

    Parameters
    ----------
    values : List[float]
        Non-empty list of timing measurements in milliseconds.

    Returns
    -------
    StageStats

    Raises
    ------
    ValueError
        If ``values`` is empty.
    """
    if not values:
        raise ValueError("compute_stats requires at least one value")

    sorted_v = sorted(values)
    n = len(sorted_v)

    def _percentile(p: float) -> float:
        # Nearest-rank method
        idx = max(0, min(n - 1, int(p / 100.0 * n + 0.5) - 1))
        return sorted_v[idx]

    return StageStats(
        mean=sum(values) / n,
        p50=_percentile(50),
        p95=_percentile(95),
        p99=_percentile(99),
        min=sorted_v[0],
        max=sorted_v[-1],
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_result(
    records: List[FrameRecord],
    *,
    model: str,
    device: str,
    video: str,
    resolution: str,
    fps: float,
    warmup_frames: int,
    total_wall_time_s: float,
) -> BenchmarkResult:
    """Build a ``BenchmarkResult`` from per-frame records (warmup excluded).

    Parameters
    ----------
    records : List[FrameRecord]
        Only the *benchmark* frames (warmup already removed by caller).
    """
    if not records:
        raise ValueError("aggregate_result requires at least one record")

    def _extract(attr: str) -> List[float]:
        return [getattr(r, attr) for r in records]

    processed = len(records)
    effective_fps = processed / total_wall_time_s if total_wall_time_s > 0 else 0.0

    return BenchmarkResult(
        model=model,
        device=device,
        video=video,
        resolution=resolution,
        fps=fps,
        decode_ms=compute_stats(_extract("decode_ms")),
        preprocess_ms=compute_stats(_extract("preprocess_ms")),
        inference_ms=compute_stats(_extract("inference_ms")),
        postprocess_ms=compute_stats(_extract("postprocess_ms")),
        draw_write_ms=compute_stats(_extract("draw_write_ms")),
        total_ms=compute_stats(_extract("total_ms")),
        processed_frames=processed,
        warmup_frames=warmup_frames,
        total_wall_time_s=total_wall_time_s,
        effective_fps=effective_fps,
        frames=records,
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def _result_to_dict(result: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to a JSON-serialisable dict."""

    def _stats_dict(s: StageStats) -> dict:
        return {
            "mean_ms": round(s.mean, 4),
            "p50_ms": round(s.p50, 4),
            "p95_ms": round(s.p95, 4),
            "p99_ms": round(s.p99, 4),
            "min_ms": round(s.min, 4),
            "max_ms": round(s.max, 4),
        }

    frames_list = [
        {
            "frame_index": r.frame_index,
            "decode_ms": round(r.decode_ms, 4),
            "preprocess_ms": round(r.preprocess_ms, 4),
            "inference_ms": round(r.inference_ms, 4),
            "postprocess_ms": round(r.postprocess_ms, 4),
            "draw_write_ms": round(r.draw_write_ms, 4),
            "total_ms": round(r.total_ms, 4),
            "n_detections": r.n_detections,
        }
        for r in result.frames
    ]

    return {
        "metadata": {
            "model": result.model,
            "device": result.device,
            "video": result.video,
            "frames": result.processed_frames,
            "warmup_frames": result.warmup_frames,
            "resolution": result.resolution,
            "fps": result.fps,
        },
        "aggregate": {
            "decode_ms":      _stats_dict(result.decode_ms),
            "preprocess_ms":  _stats_dict(result.preprocess_ms),
            "inference_ms":   _stats_dict(result.inference_ms),
            "postprocess_ms": _stats_dict(result.postprocess_ms),
            "draw_write_ms":  _stats_dict(result.draw_write_ms),
            "total_ms":       _stats_dict(result.total_ms),
        },
        "processed_frames": result.processed_frames,
        "total_wall_time_s": round(result.total_wall_time_s, 4),
        "effective_fps": round(result.effective_fps, 4),
        "frames": frames_list,
    }


def save_json(result: BenchmarkResult, output_path: str) -> None:
    """Serialise ``BenchmarkResult`` to ``output_path`` as JSON.

    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = _result_to_dict(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

_STAGE_LABELS = [
    ("decode_ms",      "Decode"),
    ("preprocess_ms",  "Preprocess"),
    ("inference_ms",   "Inference"),
    ("postprocess_ms", "Postprocess"),
    ("draw_write_ms",  "Draw + Write"),
    ("total_ms",       "Total"),
]


def print_summary(result: BenchmarkResult) -> None:
    """Print a formatted benchmark summary table to stdout."""
    col_w = 12  # numeric column width

    header = (
        f"{'Stage':<16}"
        f"{'Mean':>{col_w}}"
        f"{'P50':>{col_w}}"
        f"{'P95':>{col_w}}"
        f"{'P99':>{col_w}}"
        f"{'Min':>{col_w}}"
        f"{'Max':>{col_w}}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("Benchmark Summary  (all values in ms)")
    print("=" * len(header))
    print(f"  Model   : {result.model}")
    print(f"  Device  : {result.device}")
    print(f"  Video   : {result.video}")
    print(f"  Frames  : {result.processed_frames}  (warmup excluded: {result.warmup_frames})")
    print(f"  Wall    : {result.total_wall_time_s:.3f} s")
    print("=" * len(header))
    print(header)
    print(sep)

    for attr, label in _STAGE_LABELS:
        s: StageStats = getattr(result, attr)
        print(
            f"{label:<16}"
            f"{s.mean:>{col_w}.3f}"
            f"{s.p50:>{col_w}.3f}"
            f"{s.p95:>{col_w}.3f}"
            f"{s.p99:>{col_w}.3f}"
            f"{s.min:>{col_w}.3f}"
            f"{s.max:>{col_w}.3f}"
        )

    print(sep)
    print(f"\nEffective FPS: {result.effective_fps:.2f}")
    print("=" * len(header))
    print()
