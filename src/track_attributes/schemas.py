"""Framework-independent contracts for M12 track attribute state.

The classes in this module are immutable value objects.  A future state
manager may mutate its private internal state, but every public profile it
emits is a detached snapshot that callers cannot change accidentally.

Only JSON-scalar attribute values are accepted.  Model-specific tensors,
arrays, and result objects must be converted before crossing this boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Tuple, Union


TRACK_PROFILE_SCHEMA_VERSION = "1.0"
JSONScalar = Union[bool, int, float, str]


def _require_int(value: Any, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _require_finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")


def _require_name(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {value!r}")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def _require_json_scalar(value: Any, name: str) -> None:
    if isinstance(value, bool) or isinstance(value, str):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return
    raise TypeError(
        f"{name} must be a JSON scalar (bool, int, float, or str), "
        f"got {type(value).__name__}"
    )


class TrackLifecycle(str, Enum):
    """Lifecycle state owned by the M12 manager."""

    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


@dataclass(frozen=True)
class TrajectoryPoint:
    """Bottom-center position of a track at one video frame."""

    frame_index: int
    timestamp_ms: float
    x: float
    y: float

    def __post_init__(self) -> None:
        _require_int(self.frame_index, "TrajectoryPoint.frame_index", minimum=0)
        _require_finite_number(
            self.timestamp_ms,
            "TrajectoryPoint.timestamp_ms",
            minimum=0.0,
        )
        _require_finite_number(self.x, "TrajectoryPoint.x")
        _require_finite_number(self.y, "TrajectoryPoint.y")

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Return the stable JSON representation."""
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "x": self.x,
            "y": self.y,
        }


@dataclass(frozen=True)
class AttributeObservation:
    """One model-agnostic attribute observation for a track."""

    namespace: str
    key: str
    value: JSONScalar
    score: float
    quality_score: float
    frame_index: int
    timestamp_ms: float

    def __post_init__(self) -> None:
        _require_name(self.namespace, "AttributeObservation.namespace")
        _require_name(self.key, "AttributeObservation.key")
        _require_json_scalar(self.value, "AttributeObservation.value")
        _require_finite_number(
            self.score,
            "AttributeObservation.score",
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_number(
            self.quality_score,
            "AttributeObservation.quality_score",
            minimum=0.0,
            maximum=1.0,
        )
        _require_int(
            self.frame_index,
            "AttributeObservation.frame_index",
            minimum=0,
        )
        _require_finite_number(
            self.timestamp_ms,
            "AttributeObservation.timestamp_ms",
            minimum=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the stable JSON representation."""
        return {
            "namespace": self.namespace,
            "key": self.key,
            "value": self.value,
            "score": self.score,
            "quality_score": self.quality_score,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass(frozen=True)
class StableAttribute:
    """Temporally aggregated value for one attribute key."""

    value: JSONScalar
    score: float
    observation_count: int
    last_updated_frame_index: int
    last_updated_ms: float

    def __post_init__(self) -> None:
        _require_json_scalar(self.value, "StableAttribute.value")
        _require_finite_number(
            self.score,
            "StableAttribute.score",
            minimum=0.0,
            maximum=1.0,
        )
        _require_int(
            self.observation_count,
            "StableAttribute.observation_count",
            minimum=1,
        )
        _require_int(
            self.last_updated_frame_index,
            "StableAttribute.last_updated_frame_index",
            minimum=0,
        )
        _require_finite_number(
            self.last_updated_ms,
            "StableAttribute.last_updated_ms",
            minimum=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the stable JSON representation."""
        return {
            "value": self.value,
            "score": self.score,
            "observation_count": self.observation_count,
            "last_updated_frame_index": self.last_updated_frame_index,
            "last_updated_ms": self.last_updated_ms,
        }


@dataclass(frozen=True)
class TrackProfile:
    """Immutable public snapshot for one tracked person.

    ``trajectory`` is copied to a tuple.  ``attributes`` is deep-copied at
    both mapping levels and exposed through read-only mapping proxies.
    """

    track_id: int
    lifecycle: TrackLifecycle
    first_seen_frame_index: int
    last_seen_frame_index: int
    first_seen_ms: float
    last_seen_ms: float
    age_frames: int
    observed_frames: int
    missed_frames: int
    trajectory: Sequence[TrajectoryPoint] = ()
    attributes: Mapping[str, Mapping[str, StableAttribute]] = field(
        default_factory=dict
    )
    schema_version: str = field(
        default=TRACK_PROFILE_SCHEMA_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        _require_int(self.track_id, "TrackProfile.track_id", minimum=0)
        if not isinstance(self.lifecycle, TrackLifecycle):
            raise TypeError(
                "TrackProfile.lifecycle must be a TrackLifecycle, "
                f"got {self.lifecycle!r}"
            )

        _require_int(
            self.first_seen_frame_index,
            "TrackProfile.first_seen_frame_index",
            minimum=0,
        )
        _require_int(
            self.last_seen_frame_index,
            "TrackProfile.last_seen_frame_index",
            minimum=0,
        )
        if self.last_seen_frame_index < self.first_seen_frame_index:
            raise ValueError(
                "TrackProfile.last_seen_frame_index must be >= "
                "first_seen_frame_index"
            )

        _require_finite_number(
            self.first_seen_ms,
            "TrackProfile.first_seen_ms",
            minimum=0.0,
        )
        _require_finite_number(
            self.last_seen_ms,
            "TrackProfile.last_seen_ms",
            minimum=0.0,
        )
        if self.last_seen_ms < self.first_seen_ms:
            raise ValueError(
                "TrackProfile.last_seen_ms must be >= first_seen_ms"
            )

        _require_int(self.age_frames, "TrackProfile.age_frames", minimum=1)
        _require_int(
            self.observed_frames,
            "TrackProfile.observed_frames",
            minimum=1,
        )
        _require_int(
            self.missed_frames,
            "TrackProfile.missed_frames",
            minimum=0,
        )
        if self.age_frames < self.observed_frames + self.missed_frames:
            raise ValueError(
                "TrackProfile.age_frames must cover observed_frames and "
                "missed_frames"
            )

        trajectory = self._copy_trajectory(self.trajectory)
        attributes = self._copy_attributes(self.attributes)
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "attributes", attributes)

    def _copy_trajectory(
        self,
        points: Sequence[TrajectoryPoint],
    ) -> Tuple[TrajectoryPoint, ...]:
        if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
            raise TypeError("TrackProfile.trajectory must be a sequence")

        copied = tuple(points)
        previous: TrajectoryPoint | None = None
        for point in copied:
            if not isinstance(point, TrajectoryPoint):
                raise TypeError(
                    "TrackProfile.trajectory items must be TrajectoryPoint"
                )
            if not (
                self.first_seen_frame_index
                <= point.frame_index
                <= self.last_seen_frame_index
            ):
                raise ValueError(
                    "TrajectoryPoint.frame_index must be within the profile "
                    "first/last seen range"
                )
            if not self.first_seen_ms <= point.timestamp_ms <= self.last_seen_ms:
                raise ValueError(
                    "TrajectoryPoint.timestamp_ms must be within the profile "
                    "first/last seen range"
                )
            if previous is not None:
                if point.frame_index <= previous.frame_index:
                    raise ValueError(
                        "TrackProfile.trajectory frame indexes must be "
                        "strictly increasing"
                    )
                if point.timestamp_ms < previous.timestamp_ms:
                    raise ValueError(
                        "TrackProfile.trajectory timestamps must be "
                        "non-decreasing"
                    )
            previous = point
        return copied

    @staticmethod
    def _copy_attributes(
        attributes: Mapping[str, Mapping[str, StableAttribute]],
    ) -> Mapping[str, Mapping[str, StableAttribute]]:
        if not isinstance(attributes, Mapping):
            raise TypeError("TrackProfile.attributes must be a mapping")

        copied: Dict[str, Mapping[str, StableAttribute]] = {}
        for namespace, values in attributes.items():
            _require_name(namespace, "TrackProfile attribute namespace")
            if not isinstance(values, Mapping):
                raise TypeError(
                    "TrackProfile attribute namespaces must contain mappings"
                )

            namespace_copy: Dict[str, StableAttribute] = {}
            for key, stable in values.items():
                _require_name(key, "TrackProfile attribute key")
                if not isinstance(stable, StableAttribute):
                    raise TypeError(
                        "TrackProfile attribute values must be StableAttribute"
                    )
                namespace_copy[key] = stable
            copied[namespace] = MappingProxyType(namespace_copy)

        return MappingProxyType(copied)

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-ready dictionary."""
        return {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "lifecycle": self.lifecycle.value,
            "first_seen_frame_index": self.first_seen_frame_index,
            "last_seen_frame_index": self.last_seen_frame_index,
            "first_seen_ms": self.first_seen_ms,
            "last_seen_ms": self.last_seen_ms,
            "age_frames": self.age_frames,
            "observed_frames": self.observed_frames,
            "missed_frames": self.missed_frames,
            "trajectory": [point.to_dict() for point in self.trajectory],
            "attributes": {
                namespace: {
                    key: stable.to_dict()
                    for key, stable in values.items()
                }
                for namespace, values in self.attributes.items()
            },
        }


__all__ = [
    "TRACK_PROFILE_SCHEMA_VERSION",
    "AttributeObservation",
    "StableAttribute",
    "TrackLifecycle",
    "TrackProfile",
    "TrajectoryPoint",
]
