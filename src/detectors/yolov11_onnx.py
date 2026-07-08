"""
YOLO11n person-only detector — ONNX Runtime backend.

This module is the ONLY place where onnxruntime is imported.
No onnxruntime objects (InferenceSession, OrtValue, etc.) are allowed
to escape this module's boundary.

Preprocessing mirrors Ultralytics letterbox exactly so that ONNX output
is numerically equivalent to the PyTorch baseline.

Future swap path:
    YOLO11ONNXDetector (ONNX Runtime)  →  YOLO11TensorRTDetector (TensorRT)
    Both implement BaseDetector.  Nothing else changes.

ONNX tensor layout (YOLO11n, opset 12, static 640×640):
    Input  : "images"  shape=(1, 3, 640, 640)  dtype=float32  layout=BCHW RGB norm
    Output : "output0" shape=(1, 84, 8400)      dtype=float32
                       84 = 4 (cx,cy,w,h) + 80 class scores

Postprocessing steps (all inside this module):
    1. Transpose output → (8400, 84)
    2. Extract cx,cy,w,h and class scores
    3. Filter by person class (index 0) AND confidence threshold
    4. Convert cx,cy,w,h → x1,y1,x2,y2 in letterboxed space
    5. Scale coords back to original frame (reverse letterbox)
    6. Apply NMS
    7. Return list[Detection]
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.detectors.base import BaseDetector
from src.schemas import Detection

# COCO class index for "person"
_PERSON_CLASS_ID: int = 0

# YOLO model input resolution (must match export imgsz)
_INPUT_SIZE: int = 640

# Default NMS IoU threshold
_NMS_IOU_THRESHOLD: float = 0.45


def _letterbox(
    image: np.ndarray,
    new_size: int = _INPUT_SIZE,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image to square with letterbox padding.

    Mirrors Ultralytics letterbox behaviour: resize to fit within new_size×new_size
    while preserving aspect ratio, then pad to exact new_size×new_size.

    Parameters
    ----------
    image : np.ndarray
        BGR image, shape (H, W, 3).
    new_size : int
        Target square size (default 640).
    color : tuple
        Padding colour (default grey 114,114,114 — Ultralytics default).

    Returns
    -------
    padded : np.ndarray
        Letterboxed image, shape (new_size, new_size, 3), dtype uint8.
    scale : float
        Scale factor applied (same for both axes).
    (pad_w, pad_h) : tuple[int, int]
        Padding added on each side (left/right for w, top/bottom for h).
    """
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute symmetric padding
    pad_h = (new_size - new_h) // 2
    pad_w = (new_size - new_w) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_h, new_size - new_h - pad_h,
        pad_w, new_size - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, scale, (pad_w, pad_h)


def _preprocess(frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Prepare BGR frame for ONNX inference.

    Steps:
        1. Letterbox to 640×640
        2. BGR → RGB
        3. uint8 → float32 / 255.0
        4. HWC → CHW
        5. Add batch dim → (1, 3, 640, 640)

    Returns
    -------
    blob : np.ndarray
        Shape (1, 3, 640, 640), dtype float32, contiguous.
    scale : float
        Letterbox scale factor.
    padding : tuple[int, int]
        (pad_w, pad_h) padding on each side.
    """
    lb, scale, padding = _letterbox(frame, _INPUT_SIZE)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    norm = rgb.astype(np.float32) / 255.0
    chw = np.transpose(norm, (2, 0, 1))          # HWC → CHW
    blob = np.ascontiguousarray(chw[np.newaxis])  # add batch dim
    return blob, scale, padding


def _nms_numpy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Pure-numpy NMS.  Returns kept indices (sorted by score desc)."""
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int32)

    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        iw = np.maximum(0.0, ix2 - ix1)
        ih = np.maximum(0.0, iy2 - iy1)
        inter = iw * ih
        union = areas[i] + areas[rest] - inter
        iou = inter / np.where(union > 0, union, 1e-10)
        order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int32)


def _postprocess(
    raw_output: np.ndarray,
    scale: float,
    padding: Tuple[int, int],
    orig_h: int,
    orig_w: int,
    confidence: float,
    iou_threshold: float = _NMS_IOU_THRESHOLD,
) -> List[Detection]:
    """Decode ONNX output to list[Detection].

    Parameters
    ----------
    raw_output : np.ndarray
        Shape (1, 84, 8400) — direct ONNX session output.
    scale : float
        Letterbox scale factor from _preprocess.
    padding : tuple[int, int]
        (pad_w, pad_h) from _preprocess.
    orig_h, orig_w : int
        Original frame dimensions.
    confidence : float
        Minimum confidence threshold.
    iou_threshold : float
        NMS IoU threshold.

    Returns
    -------
    list[Detection]
        Person-only detections in original frame coordinates.
    """
    # raw_output: (1, 84, 8400) → drop batch dim → (84, 8400) → transpose → (8400, 84)
    preds = raw_output[0].T           # (8400, 84)

    cx  = preds[:, 0]                 # center x in letterboxed space
    cy  = preds[:, 1]                 # center y
    w   = preds[:, 2]
    h   = preds[:, 3]
    cls_scores = preds[:, 4:]         # (8400, 80)

    # Person class scores
    person_scores = cls_scores[:, _PERSON_CLASS_ID]

    # Filter by confidence
    mask = person_scores >= confidence
    if not np.any(mask):
        return []

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    scores = person_scores[mask]

    # cx,cy,w,h → x1,y1,x2,y2 in letterboxed pixel space
    x1_lb = cx - w / 2
    y1_lb = cy - h / 2
    x2_lb = cx + w / 2
    y2_lb = cy + h / 2

    # Reverse letterbox: subtract padding, divide by scale
    pad_w, pad_h = padding
    x1_orig = (x1_lb - pad_w) / scale
    y1_orig = (y1_lb - pad_h) / scale
    x2_orig = (x2_lb - pad_w) / scale
    y2_orig = (y2_lb - pad_h) / scale

    # Clip to frame bounds
    x1_orig = np.clip(x1_orig, 0, orig_w)
    y1_orig = np.clip(y1_orig, 0, orig_h)
    x2_orig = np.clip(x2_orig, 0, orig_w)
    y2_orig = np.clip(y2_orig, 0, orig_h)

    # Stack for NMS
    boxes_xyxy = np.stack([x1_orig, y1_orig, x2_orig, y2_orig], axis=1)

    # Apply NMS
    kept = _nms_numpy(boxes_xyxy, scores, iou_threshold)

    detections: List[Detection] = []
    for idx in kept:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        # Skip degenerate boxes that can arise from clipping
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(scores[idx]),
                class_id=_PERSON_CLASS_ID,
            )
        )

    return detections


class YOLO11ONNXDetector(BaseDetector):
    """Person-only detector backed by YOLO11n ONNX Runtime.

    Parameters
    ----------
    model_path : str
        Path to ``yolo11n.onnx`` weights file.
    confidence : float
        Minimum confidence threshold. Default: 0.25 (matches PyTorch baseline).
    providers : list[str] or None
        ONNX Runtime execution providers in priority order.
        Default: ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    RuntimeError
        If the ONNX session fails to load.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        providers: Optional[List[str]] = None,
    ) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YOLO11n ONNX weights not found: {model_path!r}"
            )

        self._model_path = model_path
        self._confidence = confidence
        self._requested_providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self._session = None
        self._actual_provider: str = "unknown"
        self._session = self._load_session()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_session(self):
        """Load ONNX Runtime session. onnxruntime stays inside this module."""
        import onnxruntime as ort  # noqa: PLC0415

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # suppress verbose ORT logs

        session = ort.InferenceSession(
            self._model_path,
            sess_options=session_options,
            providers=self._requested_providers,
        )
        # Record which provider is actually active (first in list)
        self._actual_provider = session.get_providers()[0]
        return session

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO11n ONNX inference and return person-only detections.

        Parameters
        ----------
        frame : np.ndarray
            BGR image, shape (H, W, 3), dtype uint8.

        Returns
        -------
        List[Detection]
            Only detections whose class_id == 0 (person).
            Coordinates are pixel positions on the original frame.
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess: letterbox → RGB → float32/255 → CHW → batch
        blob, scale, padding = _preprocess(frame)

        # Run inference — only numpy arrays cross the session boundary
        input_name = self._session.get_inputs()[0].name
        raw_output = self._session.run(None, {input_name: blob})
        # raw_output is a list; raw_output[0] is shape (1, 84, 8400)

        # Postprocess: decode, threshold, NMS, scale to original frame
        return _postprocess(
            raw_output[0],
            scale=scale,
            padding=padding,
            orig_h=orig_h,
            orig_w=orig_w,
            confidence=self._confidence,
        )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release ONNX Runtime session. Safe to call multiple times."""
        if hasattr(self, "_session") and self._session is not None:
            del self._session
            self._session = None

    def __del__(self) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        """Actual ONNX Runtime execution provider in use."""
        return self._actual_provider

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence threshold in use."""
        return self._confidence

    def __repr__(self) -> str:
        return (
            f"YOLO11ONNXDetector("
            f"model={self._model_path!r}, "
            f"provider={self._actual_provider!r}, "
            f"conf={self._confidence})"
        )
