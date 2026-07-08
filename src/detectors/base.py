"""
Abstract base for all person detectors.

Contract:
  - Input : numpy BGR frame (uint8)
  - Output: list[Detection]

Implementing this interface allows swapping backends
(PyTorch → TensorRT) without touching VideoSource,
visualization, or benchmark code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.schemas import Detection


class BaseDetector(ABC):
    """Abstract detector interface.

    Subclasses must implement ``predict``.
    No backend-specific objects may cross this boundary.
    """

    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image, shape (H, W, 3), dtype uint8.

        Returns
        -------
        List[Detection]
            Zero or more detections on this frame.
            Coordinates are in pixels on the original frame.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Free any resources held by this detector.  Safe to call multiple times."""
        ...

    def __enter__(self) -> "BaseDetector":
        return self

    def __exit__(self, *_) -> None:
        self.release()
