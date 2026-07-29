"""Abstract interface for M04 person attribute recognizers.

Implementations may use different inference backends, but tensor, logits,
and backend-specific result objects must not cross this boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.body_attributes.schemas import BodyAttributePrediction


class BaseBodyAttributeRecognizer(ABC):
    """Backend-independent recognizer for one BGR person crop."""

    @abstractmethod
    def predict(
        self,
        person_crop: np.ndarray,
    ) -> List[BodyAttributePrediction]:
        """Return body attribute predictions for one person crop.

        Parameters
        ----------
        person_crop:
            BGR image with shape ``(height, width, 3)`` and dtype ``uint8``.

        Returns
        -------
        list[BodyAttributePrediction]
            Zero or more predictions from the fixed M04 taxonomy. Backend
            tensors, logits, and model result objects must be converted before
            returning.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release recognizer resources; implementations should be idempotent."""
        ...

    def __enter__(self) -> "BaseBodyAttributeRecognizer":
        return self

    def __exit__(self, *_) -> None:
        self.release()
