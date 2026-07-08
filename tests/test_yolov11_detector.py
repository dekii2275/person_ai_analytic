"""
Smoke tests for YOLO11Detector — M2.

Run with:
    /home/dekii2275/miniconda3/envs/orin_person/bin/python \
        -m pytest tests/test_yolov11_detector.py -v

Uses real frames from data/input.mp4 and models/yolo11n.pt.
Does NOT process the full video — tests use only a few representative frames.
"""

from __future__ import annotations

import os
import sys
from typing import List

import cv2
import numpy as np
import pytest
import torch

# Allow import from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detectors.yolov11 import YOLO11Detector
from src.schemas import Detection
from src.video_source import VideoSource

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "input.mp4")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "yolo11n.pt")
MISSING_MODEL = "/tmp/does_not_exist_yolo11n.pt"
PERSON_CLASS_ID = 0

# Representative frame indices to test (spread across the video)
SAMPLE_FRAME_INDICES = [0, 30, 60, 120, 180, 258]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_frame(frame_index: int) -> np.ndarray:
    """Read a single frame from data/input.mp4 by index."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Could not read frame {frame_index}"
    return frame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    """Single YOLO11Detector instance reused across all tests in this module."""
    det = YOLO11Detector(model_path=MODEL_PATH)
    yield det
    det.release()


@pytest.fixture(scope="module")
def sample_frames() -> List[np.ndarray]:
    """Load representative frames once for the whole test module."""
    return [_read_frame(i) for i in SAMPLE_FRAME_INDICES]


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_missing_weights_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            YOLO11Detector(model_path=MISSING_MODEL)

    def test_loads_successfully(self, detector):
        assert detector is not None

    def test_device_is_string(self, detector):
        assert isinstance(detector.device, str)

    def test_device_is_cuda_when_available(self, detector):
        if torch.cuda.is_available():
            assert "cuda" in detector.device, (
                f"CUDA is available but detector.device={detector.device!r}"
            )
        else:
            assert detector.device == "cpu"

    def test_confidence_threshold_positive(self, detector):
        assert 0.0 < detector.confidence_threshold <= 1.0

    def test_repr_contains_model_path(self, detector):
        assert "yolo11n.pt" in repr(detector)


# ---------------------------------------------------------------------------
# Predict return-type tests
# ---------------------------------------------------------------------------

class TestPredictReturnType:
    def test_predict_returns_list(self, detector, sample_frames):
        result = detector.predict(sample_frames[0])
        assert isinstance(result, list)

    def test_all_items_are_detection(self, detector, sample_frames):
        for frame in sample_frames:
            result = detector.predict(frame)
            for item in result:
                assert isinstance(item, Detection), (
                    f"Expected Detection, got {type(item)}"
                )

    def test_no_ultralytics_objects_returned(self, detector, sample_frames):
        """Verify that no ultralytics types leak out of the detector."""
        for frame in sample_frames:
            result = detector.predict(frame)
            for item in result:
                assert type(item).__module__.startswith("src"), (
                    f"Unexpected type from external module: {type(item)}"
                )

    def test_predict_on_black_frame_returns_list(self, detector):
        """Empty frame should return a list (possibly empty), not raise."""
        black = np.zeros((1024, 576, 3), dtype=np.uint8)
        result = detector.predict(black)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Detection field validity tests
# ---------------------------------------------------------------------------

class TestDetectionFields:
    def _get_all_detections(
        self, detector: YOLO11Detector, frames: List[np.ndarray]
    ) -> List[Detection]:
        all_dets: List[Detection] = []
        for frame in frames:
            all_dets.extend(detector.predict(frame))
        return all_dets

    def test_all_class_ids_are_person(self, detector, sample_frames):
        for det in self._get_all_detections(detector, sample_frames):
            assert det.class_id == PERSON_CLASS_ID, (
                f"Non-person class_id {det.class_id} leaked through"
            )

    def test_scores_in_range(self, detector, sample_frames):
        for det in self._get_all_detections(detector, sample_frames):
            assert 0.0 <= det.score <= 1.0, (
                f"score={det.score} is outside [0, 1]"
            )

    def test_bbox_x2_gt_x1(self, detector, sample_frames):
        for det in self._get_all_detections(detector, sample_frames):
            assert det.x2 > det.x1, (
                f"x2 ({det.x2}) must be > x1 ({det.x1})"
            )

    def test_bbox_y2_gt_y1(self, detector, sample_frames):
        for det in self._get_all_detections(detector, sample_frames):
            assert det.y2 > det.y1, (
                f"y2 ({det.y2}) must be > y1 ({det.y1})"
            )

    def test_bbox_within_frame_bounds(self, detector, sample_frames):
        for frame, det in zip(
            sample_frames, detector.predict(sample_frames[0])
        ):
            h, w = frame.shape[:2]
            assert det.x1 >= 0, f"x1={det.x1} < 0"
            assert det.y1 >= 0, f"y1={det.y1} < 0"
            assert det.x2 <= w, f"x2={det.x2} > frame width {w}"
            assert det.y2 <= h, f"y2={det.y2} > frame height {h}"

    def test_bbox_coords_are_floats(self, detector, sample_frames):
        for det in self._get_all_detections(detector, sample_frames):
            assert isinstance(det.x1, float)
            assert isinstance(det.y1, float)
            assert isinstance(det.x2, float)
            assert isinstance(det.y2, float)

    def test_score_is_float(self, detector, sample_frames):
        for det in self._get_all_detections(detector, sample_frames):
            assert isinstance(det.score, float)

    def test_class_id_is_int(self, detector, sample_frames):
        for det in self._get_all_detections(detector, sample_frames):
            assert isinstance(det.class_id, int)


# ---------------------------------------------------------------------------
# Integration test: VideoSource → Detector
# ---------------------------------------------------------------------------

class TestVideoSourceIntegration:
    def test_first_five_frames_via_video_source(self, detector):
        """End-to-end: read frames from VideoSource → pass to Detector."""
        with VideoSource(VIDEO_PATH) as vs:
            for frame_index, ts_ms, frame in vs.frames():
                result = detector.predict(frame)
                assert isinstance(result, list)
                for item in result:
                    assert isinstance(item, Detection)
                    assert item.class_id == PERSON_CLASS_ID
                if frame_index >= 4:
                    break


# ---------------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------------

class TestRelease:
    def test_context_manager_releases(self):
        det = YOLO11Detector(model_path=MODEL_PATH)
        with det:
            _ = det.predict(_read_frame(0))
        assert det._model is None

    def test_double_release_safe(self, detector):
        # Create a separate instance to avoid breaking other tests
        det = YOLO11Detector(model_path=MODEL_PATH)
        det.release()
        det.release()  # must not raise
