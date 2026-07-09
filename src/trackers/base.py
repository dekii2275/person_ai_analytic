"""
Abstract base for all person trackers.

Contract:
  - Input : list[Detection] from any BaseDetector + optional BGR frame
  - Output: list[Track]

Implementing this interface allows swapping tracker backends
without touching VideoSource, detector, visualization, or benchmark code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from src.schemas import Detection, Track


class BaseTracker(ABC):
    """Abstract tracker interface.

    Subclasses must implement ``update`` and ``reset``.
    No backend-specific objects may cross this boundary.

    The ``frame`` parameter is optional so that lightweight trackers
    (e.g. pure IoU-based) can omit it, while appearance-based trackers
    (e.g. BotSORT) can use it for re-ID features.
    """

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Update tracker with new detections for the current frame.

        Parameters
        ----------
        detections : List[Detection]
            Person-only detections from the current frame.
            May be empty.
        frame : np.ndarray or None
            BGR image (H, W, 3) uint8.  Optional — pass if the
            tracker backend uses visual features.

        Returns
        -------
        List[Track]
            Active tracks after association.  May be empty.
            Coordinates are in pixels on the original frame.
            No backend-specific objects escape this boundary.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state.  Call before processing a new video.

        After reset, track IDs restart from 1 and all internal state
        (Kalman filters, lost tracks, etc.) is cleared.
        """
        ...

    def __enter__(self) -> "BaseTracker":
        return self

    def __exit__(self, *_) -> None:
        self.reset()
