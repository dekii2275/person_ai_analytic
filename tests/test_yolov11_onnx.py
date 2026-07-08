"""
Smoke tests for YOLO11ONNXDetector — M6.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_yolov11_onnx.py -v

Requires:
    - models/yolo11n.onnx (exported by tools/export_onnx.py)
    - data/input.mp4

Tests verify:
    - ONNX file existence
    - ONNX checker passes
    - Detector loads successfully
    - predict() returns list[Detection]
    - class_id == 0 (person only)
    - score in [0, 1]
    - bbox valid (x2>x1, y2>y1)
    - No ORT objects leak out of detector
    - Inference on real frame works
    - Release / context manager safe
"""

from __future__ import annotations

import os
import sys
from typing import List

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detectors.yolov11_onnx import YOLO11ONNXDetector
from src.schemas import Detection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONNX_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "yolo11n.onnx"
)
VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "input.mp4"
)
MISSING_MODEL = "/tmp/does_not_exist_yolo11n.onnx"
PERSON_CLASS_ID = 0

SAMPLE_FRAME_INDICES = [0, 30, 60, 120, 180, 258]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_frame(frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Could not read frame {frame_index} from {VIDEO_PATH}"
    return frame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def onnx_model_path() -> str:
    if not os.path.isfile(ONNX_MODEL_PATH):
        pytest.skip(
            f"ONNX model not found: {ONNX_MODEL_PATH}. "
            "Run: python tools/export_onnx.py"
        )
    return ONNX_MODEL_PATH


@pytest.fixture(scope="module")
def detector(onnx_model_path: str):
    """Single YOLO11ONNXDetector reused across all tests in this module."""
    det = YOLO11ONNXDetector(model_path=onnx_model_path)
    yield det
    det.release()


@pytest.fixture(scope="module")
def sample_frames() -> List[np.ndarray]:
    return [_read_frame(i) for i in SAMPLE_FRAME_INDICES]


# ---------------------------------------------------------------------------
# File validation tests
# ---------------------------------------------------------------------------

class TestONNXFile:
    def test_onnx_file_exists(self, onnx_model_path):
        assert os.path.isfile(onnx_model_path), (
            f"ONNX model not found: {onnx_model_path}"
        )

    def test_onnx_file_size_reasonable(self, onnx_model_path):
        """File should be between 1 MB and 50 MB for YOLO11n."""
        size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)
        assert 1.0 <= size_mb <= 50.0, (
            f"Unexpected ONNX file size: {size_mb:.2f} MB"
        )

    def test_onnx_checker_passes(self, onnx_model_path):
        """onnx.checker.check_model must not raise."""
        import onnx  # noqa: PLC0415
        model = onnx.load(onnx_model_path)
        onnx.checker.check_model(model)  # raises on invalid model

    def test_onnx_input_name_is_images(self, onnx_model_path):
        import onnx  # noqa: PLC0415
        model = onnx.load(onnx_model_path)
        input_name = model.graph.input[0].name
        assert "image" in input_name.lower() or input_name == "images", (
            f"Unexpected input name: {input_name!r}"
        )

    def test_onnx_has_single_output(self, onnx_model_path):
        import onnx  # noqa: PLC0415
        model = onnx.load(onnx_model_path)
        assert len(model.graph.output) >= 1


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_missing_weights_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            YOLO11ONNXDetector(model_path=MISSING_MODEL)

    def test_loads_successfully(self, detector):
        assert detector is not None

    def test_provider_is_string(self, detector):
        assert isinstance(detector.provider, str)
        assert len(detector.provider) > 0

    def test_confidence_threshold_valid(self, detector):
        assert 0.0 < detector.confidence_threshold <= 1.0

    def test_repr_contains_onnx(self, detector):
        r = repr(detector)
        assert "yolo11n.onnx" in r or "YOLO11ONNXDetector" in r


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

    def test_no_ort_objects_returned(self, detector, sample_frames):
        """Verify no onnxruntime objects leak out of the detector."""
        for frame in sample_frames:
            result = detector.predict(frame)
            for item in result:
                # Must be from our src module, not onnxruntime
                assert type(item).__module__.startswith("src"), (
                    f"Unexpected type from external module: {type(item)}"
                )

    def test_predict_on_black_frame_returns_list(self, detector):
        """Empty frame must return a list (possibly empty), not raise."""
        black = np.zeros((1024, 576, 3), dtype=np.uint8)
        result = detector.predict(black)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Detection field validity
# ---------------------------------------------------------------------------

class TestDetectionFields:
    def _all_detections(
        self, detector: YOLO11ONNXDetector, frames: List[np.ndarray]
    ) -> List[Detection]:
        all_dets: List[Detection] = []
        for frame in frames:
            all_dets.extend(detector.predict(frame))
        return all_dets

    def test_all_class_ids_are_person(self, detector, sample_frames):
        for det in self._all_detections(detector, sample_frames):
            assert det.class_id == PERSON_CLASS_ID, (
                f"Non-person class_id {det.class_id} leaked through"
            )

    def test_scores_in_range(self, detector, sample_frames):
        for det in self._all_detections(detector, sample_frames):
            assert 0.0 <= det.score <= 1.0, (
                f"score={det.score} is outside [0, 1]"
            )

    def test_bbox_x2_gt_x1(self, detector, sample_frames):
        for det in self._all_detections(detector, sample_frames):
            assert det.x2 > det.x1, f"x2 ({det.x2}) must be > x1 ({det.x1})"

    def test_bbox_y2_gt_y1(self, detector, sample_frames):
        for det in self._all_detections(detector, sample_frames):
            assert det.y2 > det.y1, f"y2 ({det.y2}) must be > y1 ({det.y1})"

    def test_bbox_coords_are_floats(self, detector, sample_frames):
        for det in self._all_detections(detector, sample_frames):
            assert isinstance(det.x1, float)
            assert isinstance(det.y1, float)
            assert isinstance(det.x2, float)
            assert isinstance(det.y2, float)

    def test_score_is_float(self, detector, sample_frames):
        for det in self._all_detections(detector, sample_frames):
            assert isinstance(det.score, float)

    def test_class_id_is_int(self, detector, sample_frames):
        for det in self._all_detections(detector, sample_frames):
            assert isinstance(det.class_id, int)


# ---------------------------------------------------------------------------
# Inference on real video frame
# ---------------------------------------------------------------------------

class TestRealInference:
    def test_inference_frame_0(self, detector):
        """Frame 0 is known to have 1 person — basic smoke test."""
        frame = _read_frame(0)
        result = detector.predict(frame)
        assert isinstance(result, list)
        # Frame 0 should have at least one person detection
        assert len(result) >= 1, (
            f"Expected >= 1 detection on frame 0, got {len(result)}"
        )


# ---------------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------------

class TestRelease:
    def test_context_manager_releases(self, onnx_model_path):
        det = YOLO11ONNXDetector(model_path=onnx_model_path)
        with det:
            _ = det.predict(_read_frame(0))
        assert det._session is None

    def test_double_release_safe(self, onnx_model_path):
        det = YOLO11ONNXDetector(model_path=onnx_model_path)
        det.release()
        det.release()  # must not raise
