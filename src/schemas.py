"""
Framework-independent detection schema.

All detectors — regardless of backend (PyTorch, TensorRT, ONNX) —
MUST return a list of Detection objects.

No ultralytics, torch, or backend-specific objects may appear here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """Single object detection result.

    All coordinates are in pixels on the original (un-scaled) frame.

    Parameters
    ----------
    x1, y1 : float
        Top-left corner of the bounding box.
    x2, y2 : float
        Bottom-right corner of the bounding box.
    score : float
        Confidence score in [0, 1].
    class_id : int
        COCO class index (0 = person).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"Detection.score must be in [0, 1], got {self.score}"
            )
        if self.class_id < 0:
            raise ValueError(
                f"Detection.class_id must be >= 0, got {self.class_id}"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def __repr__(self) -> str:
        return (
            f"Detection("
            f"bbox=[{self.x1:.1f},{self.y1:.1f},{self.x2:.1f},{self.y2:.1f}], "
            f"score={self.score:.3f}, "
            f"class_id={self.class_id})"
        )
