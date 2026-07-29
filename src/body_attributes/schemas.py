"""Framework-independent schemas and taxonomy for M04 body attributes.

The M04-01 baseline intentionally contains only boolean, full-body
attributes that can later be converted to M12 ``AttributeObservation``
objects under the ``body`` namespace. Clothing colour belongs to M05, while
``glasses`` remains outside this baseline until model and crop evaluation
show that it is reliable enough.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Union


BODY_ATTRIBUTE_NAMESPACE = "body"
BODY_ATTRIBUTE_SCHEMA_VERSION = "1.0"


class BodyAttributeKey(str, Enum):
    """Canonical boolean keys in the initial M04 taxonomy."""

    BACKPACK = "backpack"
    BAG = "bag"
    HAT = "hat"
    LONG_SLEEVE = "long_sleeve"


@dataclass(frozen=True)
class BodyAttributePrediction:
    """One backend-independent boolean body attribute prediction."""

    key: BodyAttributeKey
    value: bool
    score: float
    schema_version: str = field(
        default=BODY_ATTRIBUTE_SCHEMA_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.key, BodyAttributeKey):
            raise TypeError(
                "BodyAttributePrediction.key must be a BodyAttributeKey, "
                f"got {self.key!r}"
            )
        if not isinstance(self.value, bool):
            raise TypeError(
                "BodyAttributePrediction.value must be a bool, "
                f"got {self.value!r}"
            )
        if isinstance(self.score, bool) or not isinstance(
            self.score,
            (int, float),
        ):
            raise TypeError(
                "BodyAttributePrediction.score must be a number, "
                f"got {self.score!r}"
            )
        if not math.isfinite(self.score):
            raise ValueError(
                "BodyAttributePrediction.score must be finite, "
                f"got {self.score!r}"
            )
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "BodyAttributePrediction.score must be in [0, 1], "
                f"got {self.score}"
            )
        object.__setattr__(self, "score", float(self.score))

    def to_dict(self) -> Dict[str, Union[str, bool, float]]:
        """Return the stable, JSON-ready M04 prediction shape."""
        return {
            "schema_version": self.schema_version,
            "key": self.key.value,
            "value": self.value,
            "score": self.score,
        }
