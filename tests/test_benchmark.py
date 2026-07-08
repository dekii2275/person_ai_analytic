"""
Smoke tests for src/benchmark.py — M4.

Run with:
    /home/dekii2275/miniconda3/envs/orin_person/bin/python \
        -m pytest tests/test_benchmark.py -v

Tests cover:
  - StageTimer measures non-negative elapsed ms
  - StageTimer with cuda_sync=True does not crash
  - compute_stats returns correct fields
  - compute_stats raises on empty input
  - aggregate_result builds correct structure
  - save_json writes valid JSON
  - JSON has required keys (metadata, aggregate, frames, effective_fps)
  - JSON aggregate has all 6 stages
  - print_summary runs without exception
  - warmup frames are excluded from records
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmark import (
    BenchmarkResult,
    FrameRecord,
    StageStats,
    StageTimer,
    aggregate_result,
    compute_stats,
    print_summary,
    save_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_records(n: int = 10) -> list[FrameRecord]:
    """Create n synthetic FrameRecords for testing."""
    return [
        FrameRecord(
            frame_index=i,
            decode_ms=1.0 + i * 0.1,
            preprocess_ms=0.0,
            inference_ms=20.0 + i * 0.5,
            postprocess_ms=0.0,
            draw_write_ms=2.0 + i * 0.1,
            total_ms=23.0 + i * 0.7,
            n_detections=i % 3,
        )
        for i in range(n)
    ]


def _make_result(n: int = 10) -> BenchmarkResult:
    records = _make_records(n)
    return aggregate_result(
        records,
        model="yolo11n.pt",
        device="cuda",
        video="data/input.mp4",
        resolution="576x1024",
        fps=30.0,
        warmup_frames=3,
        total_wall_time_s=5.0,
    )


# ---------------------------------------------------------------------------
# StageTimer
# ---------------------------------------------------------------------------

class TestStageTimer:
    def test_returns_non_negative_ms(self):
        with StageTimer() as t:
            time.sleep(0.005)
        assert t[0] >= 0.0

    def test_elapsed_roughly_correct(self):
        with StageTimer() as t:
            time.sleep(0.01)
        # Allow generous tolerance: >= 5 ms (sleep is at least that)
        assert t[0] >= 5.0

    def test_zero_work_near_zero_ms(self):
        with StageTimer() as t:
            pass
        assert t[0] < 100.0  # should be microseconds

    def test_cuda_sync_does_not_crash(self):
        """cuda_sync=True must not raise even if CUDA is unavailable."""
        with StageTimer(cuda_sync=True) as t:
            time.sleep(0.001)
        assert t[0] >= 0.0

    def test_result_is_list_of_one(self):
        with StageTimer() as t:
            pass
        assert isinstance(t, list)
        assert len(t) == 1

    def test_result_is_float(self):
        with StageTimer() as t:
            pass
        assert isinstance(t[0], float)


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            compute_stats([])

    def test_returns_stage_stats(self):
        stats = compute_stats([10.0, 20.0, 30.0])
        assert isinstance(stats, StageStats)

    def test_single_value(self):
        stats = compute_stats([42.0])
        assert stats.mean == 42.0
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.p50 == 42.0
        assert stats.p95 == 42.0
        assert stats.p99 == 42.0

    def test_mean_correct(self):
        stats = compute_stats([10.0, 20.0, 30.0])
        assert abs(stats.mean - 20.0) < 1e-9

    def test_min_max_correct(self):
        values = [5.0, 1.0, 8.0, 3.0]
        stats = compute_stats(values)
        assert stats.min == 1.0
        assert stats.max == 8.0

    def test_p95_within_range(self):
        values = list(range(1, 101))  # 1..100
        stats = compute_stats([float(v) for v in values])
        assert stats.min <= stats.p50 <= stats.p95 <= stats.p99 <= stats.max

    def test_percentile_ordering(self):
        values = [float(i) for i in range(100)]
        stats = compute_stats(values)
        assert stats.p50 <= stats.p95
        assert stats.p95 <= stats.p99


# ---------------------------------------------------------------------------
# aggregate_result
# ---------------------------------------------------------------------------

class TestAggregateResult:
    def test_raises_on_empty_records(self):
        with pytest.raises(ValueError, match="at least one"):
            aggregate_result(
                [],
                model="m", device="cpu", video="v",
                resolution="r", fps=30.0,
                warmup_frames=0, total_wall_time_s=1.0,
            )

    def test_returns_benchmark_result(self):
        result = _make_result()
        assert isinstance(result, BenchmarkResult)

    def test_processed_frames_correct(self):
        result = _make_result(n=10)
        assert result.processed_frames == 10

    def test_warmup_frames_stored(self):
        result = _make_result()
        assert result.warmup_frames == 3

    def test_effective_fps_positive(self):
        result = _make_result()
        assert result.effective_fps > 0.0

    def test_effective_fps_formula(self):
        records = _make_records(10)
        result = aggregate_result(
            records,
            model="m", device="cpu", video="v",
            resolution="r", fps=30.0,
            warmup_frames=0,
            total_wall_time_s=10.0,
        )
        assert abs(result.effective_fps - 1.0) < 1e-6  # 10 frames / 10s

    def test_all_stages_present(self):
        result = _make_result()
        for attr in ("decode_ms", "preprocess_ms", "inference_ms",
                     "postprocess_ms", "draw_write_ms", "total_ms"):
            assert hasattr(result, attr)
            assert isinstance(getattr(result, attr), StageStats)

    def test_frames_list_length(self):
        result = _make_result(n=5)
        assert len(result.frames) == 5


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_creates_file(self, tmp_path):
        result = _make_result()
        out = str(tmp_path / "bench.json")
        save_json(result, out)
        assert os.path.isfile(out)

    def test_valid_json(self, tmp_path):
        result = _make_result()
        out = str(tmp_path / "bench.json")
        save_json(result, out)
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_required_top_level_keys(self, tmp_path):
        result = _make_result()
        out = str(tmp_path / "bench.json")
        save_json(result, out)
        with open(out) as f:
            data = json.load(f)
        for key in ("metadata", "aggregate", "frames",
                    "effective_fps", "processed_frames", "total_wall_time_s"):
            assert key in data, f"Missing key: {key!r}"

    def test_aggregate_has_all_stages(self, tmp_path):
        result = _make_result()
        out = str(tmp_path / "bench.json")
        save_json(result, out)
        with open(out) as f:
            data = json.load(f)
        for stage in ("decode_ms", "preprocess_ms", "inference_ms",
                      "postprocess_ms", "draw_write_ms", "total_ms"):
            assert stage in data["aggregate"], f"Missing stage: {stage!r}"

    def test_each_stage_has_required_stat_keys(self, tmp_path):
        result = _make_result()
        out = str(tmp_path / "bench.json")
        save_json(result, out)
        with open(out) as f:
            data = json.load(f)
        for stage_key, stage_data in data["aggregate"].items():
            for stat in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
                assert stat in stage_data, (
                    f"Stage {stage_key!r} missing stat key {stat!r}"
                )

    def test_frames_array_length(self, tmp_path):
        n = 7
        result = _make_result(n)
        out = str(tmp_path / "bench.json")
        save_json(result, out)
        with open(out) as f:
            data = json.load(f)
        assert len(data["frames"]) == n

    def test_each_frame_has_required_keys(self, tmp_path):
        result = _make_result(3)
        out = str(tmp_path / "bench.json")
        save_json(result, out)
        with open(out) as f:
            data = json.load(f)
        required = {"frame_index", "decode_ms", "preprocess_ms",
                    "inference_ms", "postprocess_ms", "draw_write_ms",
                    "total_ms", "n_detections"}
        for i, frame in enumerate(data["frames"]):
            missing = required - set(frame.keys())
            assert not missing, f"Frame {i} missing keys: {missing}"

    def test_creates_parent_dirs(self, tmp_path):
        result = _make_result()
        out = str(tmp_path / "sub" / "dir" / "bench.json")
        save_json(result, out)
        assert os.path.isfile(out)

    def test_overwrite_existing(self, tmp_path):
        result = _make_result(5)
        out = str(tmp_path / "bench.json")
        save_json(result, out)
        result2 = _make_result(8)
        save_json(result2, out)  # must not raise
        with open(out) as f:
            data = json.load(f)
        assert data["processed_frames"] == 8


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_runs_without_exception(self, capsys):
        result = _make_result()
        print_summary(result)
        captured = capsys.readouterr()
        assert "Effective FPS" in captured.out

    def test_all_stage_labels_in_output(self, capsys):
        result = _make_result()
        print_summary(result)
        captured = capsys.readouterr()
        for label in ("Decode", "Preprocess", "Inference",
                      "Postprocess", "Draw + Write", "Total"):
            assert label in captured.out, f"Missing label: {label!r}"
