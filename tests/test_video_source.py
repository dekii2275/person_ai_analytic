"""
Smoke tests for VideoSource — M1.

Run with:
    /home/dekii2275/miniconda3/envs/orin_person/bin/python -m pytest tests/test_video_source.py -v

All tests use data/input.mp4 (must exist in working directory).
"""

import os
import sys

import numpy as np
import pytest

# Allow import from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.video_source import VideoSource, VideoSourceError

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "input.mp4")
MISSING_PATH = "/tmp/definitely_does_not_exist_xyz.mp4"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def source():
    """Open VideoSource and ensure it is released after each test."""
    vs = VideoSource(VIDEO_PATH)
    yield vs
    vs.release()


# ---------------------------------------------------------------------------
# Construction / validation checks
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_file_not_found_raises(self):
        with pytest.raises(VideoSourceError, match="not found"):
            VideoSource(MISSING_PATH)

    def test_opens_successfully(self, source):
        assert source is not None

    def test_fps_positive(self, source):
        assert source.fps > 0, f"FPS should be positive, got {source.fps}"

    def test_width_positive(self, source):
        assert source.width > 0, f"Width should be positive, got {source.width}"

    def test_height_positive(self, source):
        assert source.height > 0, f"Height should be positive, got {source.height}"

    def test_frame_count_positive(self, source):
        assert source.frame_count > 0


# ---------------------------------------------------------------------------
# Frame iterator
# ---------------------------------------------------------------------------

class TestFrames:
    def test_yields_tuples(self, source):
        """First frame must be a 3-tuple."""
        item = next(iter(source.frames()))
        assert len(item) == 3

    def test_frame_index_zero_first(self, source):
        frame_index, _, _ = next(iter(source.frames()))
        assert frame_index == 0

    def test_timestamp_ms_non_negative(self, source):
        _, ts_ms, _ = next(iter(source.frames()))
        assert ts_ms >= 0.0

    def test_frame_is_ndarray(self, source):
        _, _, frame = next(iter(source.frames()))
        assert isinstance(frame, np.ndarray)

    def test_frame_shape_matches_metadata(self, source):
        _, _, frame = next(iter(source.frames()))
        h, w, c = frame.shape
        assert w == source.width
        assert h == source.height
        assert c == 3

    def test_frame_dtype_uint8(self, source):
        _, _, frame = next(iter(source.frames()))
        assert frame.dtype == np.uint8

    def test_frame_index_monotonically_increasing(self, source):
        indices = [fi for fi, _, _ in source.frames()]
        assert indices == list(range(len(indices)))

    def test_timestamps_non_decreasing(self, source):
        timestamps = [ts for _, ts, _ in source.frames()]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamp decreased at frame {i}: "
                f"{timestamps[i-1]} -> {timestamps[i]}"
            )

    def test_total_frame_count(self, source):
        """Actual decoded frames should be close to container-reported count."""
        count = sum(1 for _ in source.frames())
        # Some containers report off-by-one; allow 1-frame tolerance
        assert abs(count - source.frame_count) <= 1, (
            f"Decoded {count} frames but container reports {source.frame_count}"
        )

    def test_rewind_resets_iterator(self, source):
        """Calling frames() twice should restart from frame 0."""
        first_a, _, _ = next(iter(source.frames()))
        first_b, _, _ = next(iter(source.frames()))
        assert first_a == first_b == 0


# ---------------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------------

class TestRelease:
    def test_context_manager_releases(self):
        with VideoSource(VIDEO_PATH) as vs:
            assert vs._cap is not None
        # After __exit__, _cap must be None
        assert vs._cap is None

    def test_double_release_safe(self, source):
        source.release()
        source.release()  # must not raise

    def test_frames_after_release_raises(self):
        vs = VideoSource(VIDEO_PATH)
        vs.release()
        with pytest.raises(VideoSourceError, match="released"):
            next(iter(vs.frames()))


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_path(self, source):
        assert "input.mp4" in repr(source)

    def test_repr_contains_fps(self, source):
        assert str(int(source.fps)) in repr(source)
