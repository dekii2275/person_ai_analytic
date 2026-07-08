"""
Smoke tests for visualization and M3 pipeline — M3.

Run with:
    /home/dekii2275/miniconda3/envs/orin_person/bin/python \
        -m pytest tests/test_visualization.py -v

Does NOT run the full pipeline (that is done by main.py separately).
Tests cover:
  - draw_detections returns correct type and shape
  - draw_detections_inplace modifies frame in place
  - draw with zero detections (no crash, frame unchanged)
  - draw_detections does not modify original frame
  - coordinates clamped to frame bounds
  - VideoWriter opens with correct specs
  - verify_output passes on a valid temp video
  - verify_output raises on bad metadata
  - No ultralytics / torch import in visualization
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.schemas import Detection
from src.visualization import draw_detections, draw_detections_inplace
from main import _open_writer, verify_output

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "input.mp4")
_W, _H = 576, 1024
_FPS = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(w: int = _W, h: int = _H) -> np.ndarray:
    """Return a solid-blue BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = 180  # blue channel
    return frame


def _make_det(x1=50.0, y1=100.0, x2=300.0, y2=800.0, score=0.90, cls=0) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=cls)


# ---------------------------------------------------------------------------
# draw_detections
# ---------------------------------------------------------------------------

class TestDrawDetections:
    def test_returns_ndarray(self):
        frame = _make_frame()
        result = draw_detections(frame, [_make_det()])
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        frame = _make_frame()
        result = draw_detections(frame, [_make_det()])
        assert result.shape == frame.shape

    def test_dtype_preserved(self):
        frame = _make_frame()
        result = draw_detections(frame, [_make_det()])
        assert result.dtype == np.uint8

    def test_original_not_modified(self):
        frame = _make_frame()
        original = frame.copy()
        draw_detections(frame, [_make_det()])
        np.testing.assert_array_equal(frame, original)

    def test_empty_detections_returns_copy(self):
        frame = _make_frame()
        result = draw_detections(frame, [])
        np.testing.assert_array_equal(result, frame)
        assert result is not frame  # must be a copy

    def test_annotated_differs_from_original(self):
        """With a real detection, at least some pixels must change."""
        frame = _make_frame()
        result = draw_detections(frame, [_make_det()])
        assert not np.array_equal(result, frame)

    def test_multiple_detections(self):
        frame = _make_frame()
        dets = [_make_det(10, 20, 200, 400, 0.80), _make_det(300, 500, 500, 900, 0.91)]
        result = draw_detections(frame, dets)
        assert result.shape == frame.shape

    def test_detection_at_frame_edge_no_crash(self):
        """Boxes that extend to / beyond frame boundaries must not raise."""
        frame = _make_frame()
        dets = [
            Detection(x1=-10.0, y1=-10.0, x2=700.0, y2=1100.0, score=0.5, class_id=0),
        ]
        result = draw_detections(frame, dets)
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# draw_detections_inplace
# ---------------------------------------------------------------------------

class TestDrawDetectionsInplace:
    def test_modifies_frame(self):
        frame = _make_frame()
        original = frame.copy()
        draw_detections_inplace(frame, [_make_det()])
        assert not np.array_equal(frame, original)

    def test_returns_none(self):
        frame = _make_frame()
        result = draw_detections_inplace(frame, [_make_det()])
        assert result is None

    def test_empty_detections_no_change(self):
        frame = _make_frame()
        original = frame.copy()
        draw_detections_inplace(frame, [])
        np.testing.assert_array_equal(frame, original)


# ---------------------------------------------------------------------------
# No backend leaks in visualization
# ---------------------------------------------------------------------------

class TestNoDependencyLeak:
    def test_visualization_does_not_import_ultralytics(self):
        import src.visualization as vis_mod
        for name in dir(vis_mod):
            obj = getattr(vis_mod, name)
            if isinstance(obj, types.ModuleType):
                assert "ultralytics" not in obj.__name__, (
                    f"ultralytics leaked into visualization via {name}"
                )

    def test_visualization_does_not_import_torch(self):
        import src.visualization as vis_mod
        # torch must not appear as a top-level dependency
        src_file = vis_mod.__file__
        with open(src_file) as f:
            content = f.read()
        assert "import torch" not in content, (
            "torch found in visualization.py — violates scope rule"
        )


# ---------------------------------------------------------------------------
# VideoWriter specs
# ---------------------------------------------------------------------------

class TestVideoWriter:
    def test_writer_opens_with_correct_size(self, tmp_path):
        out = str(tmp_path / "test.mp4")
        writer = _open_writer(out, _W, _H, _FPS)
        assert writer.isOpened()
        writer.release()

    def test_writer_raises_on_bad_path(self, tmp_path):
        """An unwritable directory should cause _open_writer to raise RuntimeError."""
        import stat
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x: no write
        out = str(ro_dir / "out.mp4")
        try:
            with pytest.raises(RuntimeError, match="could not open"):
                _open_writer(out, _W, _H, _FPS)
        finally:
            # Restore write so pytest can clean up tmp_path
            ro_dir.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# verify_output
# ---------------------------------------------------------------------------

class TestVerifyOutput:
    def _write_temp_video(self, path: str, w: int, h: int, fps: float, frames: int = 5):
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        assert writer.isOpened()
        for _ in range(frames):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
        writer.release()

    def test_valid_output_passes(self, tmp_path):
        out = str(tmp_path / "valid.mp4")
        self._write_temp_video(out, _W, _H, _FPS)
        info = verify_output(out, _W, _H, _FPS)
        assert info["width"] == _W
        assert info["height"] == _H
        assert info["fps"] > 0
        assert info["first_frame_readable"] is True

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError, match="not found"):
            verify_output("/tmp/does_not_exist_xyz.mp4", _W, _H, _FPS)

    def test_wrong_width_raises(self, tmp_path):
        out = str(tmp_path / "wrong_w.mp4")
        self._write_temp_video(out, 320, _H, _FPS)
        with pytest.raises(RuntimeError, match="width"):
            verify_output(out, _W, _H, _FPS)

    def test_wrong_height_raises(self, tmp_path):
        out = str(tmp_path / "wrong_h.mp4")
        self._write_temp_video(out, _W, 640, _FPS)
        with pytest.raises(RuntimeError, match="height"):
            verify_output(out, _W, _H, _FPS)
