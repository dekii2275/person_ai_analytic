"""
YOLO11n person-only detector — local PyTorch backend.

This module is the ONLY place where ultralytics is imported.
No ultralytics objects (Results, Boxes, etc.) are allowed
to escape this module's boundary.

Future swap path:
    YOLO11Detector (PyTorch)  →  YOLO11TensorRTDetector (TensorRT)
    Both implement BaseDetector.  Nothing else changes.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

from src.detectors.base import BaseDetector
from src.schemas import Detection

# COCO class index for "person"
_PERSON_CLASS_ID: int = 0


class YOLO11Detector(BaseDetector):
    """Person-only detector backed by YOLO11n (Ultralytics / PyTorch).

    Parameters
    ----------
    model_path : str
        Path to ``yolo11n.pt`` weights file.
    confidence : float
        Minimum confidence threshold. Detections below this are dropped.
        Default: 0.25 (Ultralytics default).
    device : str or None
        Torch device string, e.g. ``'cuda'``, ``'cpu'``, ``'cuda:0'``.
        If ``None`` (default), uses CUDA when available, else CPU.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    RuntimeError
        If the model fails to load.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YOLO11n weights not found: {model_path!r}"
            )

        # Resolve device
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._model_path = model_path
        self._confidence = confidence
        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load YOLO11n weights. ultralytics stays inside this module."""
        # Local import: ultralytics must NOT be imported at module level
        # so that future TensorRT backends can replace this file entirely.
        from ultralytics import YOLO  # noqa: PLC0415

        model = YOLO(self._model_path)
        # Warm up: move model to the correct device
        model.to(self._device)
        return model

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO11n inference and return person-only detections.

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
        # Run inference — classes=[0] filters at NMS level (faster)
        results = self._model.predict(
            frame,
            device=self._device,
            classes=[_PERSON_CLASS_ID],
            conf=self._confidence,
            verbose=False,
        )

        detections: List[Detection] = []

        # results is a list with one element per image (we always pass 1)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return detections

        # Convert to CPU numpy — no ultralytics object leaves this block
        xyxy = r.boxes.xyxy.cpu().numpy()   # shape (N, 4)
        confs = r.boxes.conf.cpu().numpy()  # shape (N,)
        clses = r.boxes.cls.cpu().numpy()   # shape (N,)

        # Belt-and-suspenders person filter (Ultralytics already filtered,
        # but we enforce the contract explicitly)
        for box, conf, cls in zip(xyxy, confs, clses):
            if int(cls) != _PERSON_CLASS_ID:
                continue
            detections.append(
                Detection(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    score=float(conf),
                    class_id=int(cls),
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release model resources.  Safe to call multiple times."""
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Device string this detector is running on."""
        return self._device

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence threshold in use."""
        return self._confidence

    def __repr__(self) -> str:
        return (
            f"YOLO11Detector("
            f"model={self._model_path!r}, "
            f"device={self._device!r}, "
            f"conf={self._confidence})"
        )
