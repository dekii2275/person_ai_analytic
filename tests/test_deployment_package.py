"""
tests/test_deployment_package.py
---------------------------------
Automated tests for M7-PREP deployment package.

Checks:
  - deployment_manifest.json exists and is valid JSON
  - Required fields are present in manifest
  - ONNX model file exists
  - input.mp4 exists
  - ONNX parity report exists
  - Benchmark report exists
  - All 10 parity frames exist
  - Frame naming convention is correct
  - ONNX SHA256 matches manifest
  - JSON reports parse without error
  - Manifest has correct confidence threshold
  - Manifest has correct person class id
  - Manifest has correct input shape
  - ONNX checker PASS (if onnx available)

Does NOT run inference. Does NOT require ONNX Runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = REPO_ROOT / "deployment"
MANIFEST_PATH = DEPLOYMENT_DIR / "deployment_manifest.json"

EXPECTED_FRAME_INDICES = [0, 30, 60, 79, 90, 120, 155, 180, 196, 258]
EXPECTED_PARITY_FRAME_COUNT = 10

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> Dict[str, Any]:
    """Load and return parsed deployment_manifest.json."""
    assert MANIFEST_PATH.exists(), f"deployment_manifest.json not found: {MANIFEST_PATH}"
    with open(MANIFEST_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def onnx_path(manifest: Dict[str, Any]) -> Path:
    rel = manifest["model"]["path"]
    return DEPLOYMENT_DIR / rel


@pytest.fixture(scope="module")
def video_path(manifest: Dict[str, Any]) -> Path:
    rel = manifest["test_data"]["video"]["path"]
    return DEPLOYMENT_DIR / rel


@pytest.fixture(scope="module")
def parity_frame_dir(manifest: Dict[str, Any]) -> Path:
    rel = manifest["test_data"]["parity_frames"]["dir"]
    return DEPLOYMENT_DIR / rel


# ---------------------------------------------------------------------------
# Check 1: manifest exists
# ---------------------------------------------------------------------------


def test_manifest_exists():
    """deployment_manifest.json exists."""
    assert MANIFEST_PATH.exists(), f"Not found: {MANIFEST_PATH}"


def test_manifest_parses():
    """deployment_manifest.json is valid JSON."""
    assert MANIFEST_PATH.exists()
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Check 2: Required manifest fields
# ---------------------------------------------------------------------------


def test_manifest_has_project_fields(manifest):
    """Manifest contains project metadata."""
    assert "project" in manifest
    assert manifest["project"]["milestone"] == "M7-PREP"
    assert manifest["project"]["name"] == "person_ai_analytic"


def test_manifest_has_model_fields(manifest):
    """Manifest contains model metadata."""
    model = manifest["model"]
    assert model["format"] == "ONNX"
    assert model["opset"] == 12
    assert "sha256" in model
    assert len(model["sha256"]) == 64, "sha256 should be 64 hex chars"
    assert model["file_size_bytes"] > 0


def test_manifest_has_input_spec(manifest):
    """Manifest contains correct ONNX input specification."""
    inp = manifest["input"]
    assert inp["tensor_name"] == "images"
    assert inp["shape"] == [1, 3, 640, 640]
    assert inp["dtype"] == "float32"
    assert inp["layout"] == "NCHW"
    assert inp["color_space"] == "RGB"


def test_manifest_has_output_spec(manifest):
    """Manifest contains correct ONNX output specification."""
    out = manifest["output"]
    assert out["tensor_name"] == "output0"
    assert out["shape"] == [1, 84, 8400]


def test_manifest_confidence_threshold(manifest):
    """Manifest records confidence threshold = 0.25 (baseline)."""
    pp = manifest["postprocessing"]
    assert pp["confidence_threshold"] == 0.25, (
        f"Expected 0.25, got {pp['confidence_threshold']}"
    )


def test_manifest_person_class_id(manifest):
    """Manifest records person class id = 0 (COCO)."""
    pp = manifest["postprocessing"]
    assert pp["person_class_id"] == 0


def test_manifest_preprocessing_fields(manifest):
    """Manifest contains preprocessing pipeline description."""
    pre = manifest["preprocessing"]
    assert pre["source_color_space"] == "BGR"
    assert pre["target_color_space"] == "RGB"
    assert pre["resize_strategy"] == "letterbox"
    assert pre["target_size"] == [640, 640]
    assert pre["normalization"] == "0-1"


def test_manifest_has_parity_frame_indices(manifest):
    """Manifest lists all 10 parity frame indices."""
    indices = manifest["test_data"]["parity_frames"]["frame_indices"]
    assert sorted(indices) == sorted(EXPECTED_FRAME_INDICES)
    assert manifest["test_data"]["parity_frames"]["count"] == EXPECTED_PARITY_FRAME_COUNT


# ---------------------------------------------------------------------------
# Check 3: ONNX model file exists
# ---------------------------------------------------------------------------


def test_onnx_model_exists(onnx_path):
    """deployment/models/yolo11n.onnx exists."""
    assert onnx_path.exists(), f"Not found: {onnx_path}"


def test_onnx_model_size(onnx_path, manifest):
    """ONNX model file size matches manifest."""
    actual_size = onnx_path.stat().st_size
    expected_size = manifest["model"]["file_size_bytes"]
    assert actual_size == expected_size, (
        f"Size mismatch: actual={actual_size}, expected={expected_size}"
    )


# ---------------------------------------------------------------------------
# Check 4: input.mp4 exists
# ---------------------------------------------------------------------------


def test_video_exists(video_path):
    """deployment/test_data/input.mp4 exists."""
    assert video_path.exists(), f"Not found: {video_path}"


def test_video_size(video_path, manifest):
    """input.mp4 file size matches manifest."""
    actual = video_path.stat().st_size
    expected = manifest["test_data"]["video"]["file_size_bytes"]
    assert actual == expected, f"Size mismatch: actual={actual}, expected={expected}"


# ---------------------------------------------------------------------------
# Check 5: ONNX parity report exists
# ---------------------------------------------------------------------------


def test_parity_report_exists(manifest):
    """deployment/reports/onnx_parity.json exists."""
    rel = manifest["reference"]["onnx_parity_report"]
    path = DEPLOYMENT_DIR / rel
    assert path.exists(), f"Not found: {path}"


# ---------------------------------------------------------------------------
# Check 6: benchmark report exists
# ---------------------------------------------------------------------------


def test_benchmark_report_exists(manifest):
    """deployment/reports/benchmark_local.json exists."""
    rel = manifest["reference"]["local_benchmark_report"]
    path = DEPLOYMENT_DIR / rel
    assert path.exists(), f"Not found: {path}"


# ---------------------------------------------------------------------------
# Check 7: parity frames count and naming
# ---------------------------------------------------------------------------


def test_parity_frame_count(parity_frame_dir):
    """Exactly 10 parity frame files exist."""
    jpg_files = list(parity_frame_dir.glob("frame_*.jpg"))
    assert len(jpg_files) == EXPECTED_PARITY_FRAME_COUNT, (
        f"Expected {EXPECTED_PARITY_FRAME_COUNT}, found {len(jpg_files)}"
    )


def test_parity_frame_names(parity_frame_dir):
    """Parity frames follow naming convention frame_{:06d}.jpg."""
    for idx in EXPECTED_FRAME_INDICES:
        expected = parity_frame_dir / f"frame_{idx:06d}.jpg"
        assert expected.exists(), f"Missing: {expected}"


def test_parity_frame_nonzero_size(parity_frame_dir):
    """All parity frames have non-zero file size."""
    for idx in EXPECTED_FRAME_INDICES:
        fpath = parity_frame_dir / f"frame_{idx:06d}.jpg"
        if fpath.exists():
            assert fpath.stat().st_size > 0, f"Zero-size file: {fpath}"


# ---------------------------------------------------------------------------
# Check 8: ONNX SHA256 checksum
# ---------------------------------------------------------------------------


def test_onnx_checksum(onnx_path, manifest):
    """ONNX file SHA256 matches manifest."""
    expected = manifest["model"]["sha256"]
    h = hashlib.sha256()
    with open(onnx_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    assert actual == expected, (
        f"Checksum mismatch:\n  expected={expected}\n  actual  ={actual}"
    )


# ---------------------------------------------------------------------------
# Check 9: JSON files parse without error
# ---------------------------------------------------------------------------


def test_parity_report_parses(manifest):
    """onnx_parity.json parses as valid JSON with expected structure."""
    rel = manifest["reference"]["onnx_parity_report"]
    path = DEPLOYMENT_DIR / rel
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert "summary" in data
    assert "frames" in data
    assert data["summary"]["frames_compared"] == 10


def test_benchmark_report_parses(manifest):
    """benchmark_local.json parses as valid JSON with expected structure."""
    rel = manifest["reference"]["local_benchmark_report"]
    path = DEPLOYMENT_DIR / rel
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert "aggregate" in data
    assert "effective_fps" in data


# ---------------------------------------------------------------------------
# Check 10: ONNX checker (if onnx package available)
# ---------------------------------------------------------------------------


def test_onnx_checker(onnx_path):
    """ONNX model passes onnx.checker.check_model."""
    pytest.importorskip("onnx", reason="onnx package not installed, skipping checker test")
    import onnx  # noqa: PLC0415

    model = onnx.load(str(onnx_path))
    # check_model raises an exception on failure
    onnx.checker.check_model(model)


# ---------------------------------------------------------------------------
# Check 11: parity frame resolution (if cv2 available)
# ---------------------------------------------------------------------------


def test_parity_frame_resolution(parity_frame_dir, manifest):
    """Parity frames have the expected resolution (576x1024)."""
    cv2 = pytest.importorskip("cv2", reason="cv2 not available, skipping resolution test")
    expected_res = manifest["test_data"]["parity_frames"]["resolution"]
    exp_w, exp_h = (int(x) for x in expected_res.split("x"))

    for idx in EXPECTED_FRAME_INDICES:
        fpath = parity_frame_dir / f"frame_{idx:06d}.jpg"
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        assert img is not None, f"cv2.imread failed for {fpath}"
        h, w = img.shape[:2]
        assert w == exp_w and h == exp_h, (
            f"frame_{idx:06d}.jpg: resolution {w}x{h} != {expected_res}"
        )


# ---------------------------------------------------------------------------
# Check 12: video openable (if cv2 available)
# ---------------------------------------------------------------------------


def test_video_openable(video_path):
    """input.mp4 can be opened with cv2.VideoCapture."""
    cv2 = pytest.importorskip("cv2", reason="cv2 not available, skipping video open test")
    cap = cv2.VideoCapture(str(video_path))
    try:
        assert cap.isOpened(), f"cv2.VideoCapture failed to open: {video_path}"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert frame_count == 259, f"Expected 259 frames, got {frame_count}"
        assert width == 576, f"Expected width=576, got {width}"
        assert height == 1024, f"Expected height=1024, got {height}"
        assert abs(fps - 30.0) < 0.1, f"Expected fps=30, got {fps}"
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Check 13: no TensorRT engine in package (scope compliance)
# ---------------------------------------------------------------------------


def test_no_tensorrt_engine_in_package():
    """Deployment package must NOT contain .engine files (scope compliance)."""
    engine_files = list(DEPLOYMENT_DIR.rglob("*.engine"))
    assert len(engine_files) == 0, (
        f"TensorRT engine files found in deployment package (out of scope for M7-PREP): {engine_files}"
    )


def test_no_deepstream_config_in_package():
    """Deployment package must NOT contain DeepStream config files (scope compliance)."""
    ds_files = list(DEPLOYMENT_DIR.rglob("*.txt")) 
    deepstream_files = [f for f in ds_files if "deepstream" in f.name.lower()]
    assert len(deepstream_files) == 0, (
        f"DeepStream config files found (out of scope): {deepstream_files}"
    )
