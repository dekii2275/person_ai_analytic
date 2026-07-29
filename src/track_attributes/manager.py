"""In-memory lifecycle state manager for M12.

This module owns state keyed by ``Track.track_id``.  It does not run models,
perform tracking, or persist data.  Public results are immutable
``TrackProfile`` snapshots from the M12 schema boundary.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from src.schemas import Track
from src.track_attributes.schemas import (
    TrackLifecycle,
    TrackProfile,
    TrajectoryPoint,
)


def _require_int(value: object, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _require_timestamp(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class TrackAttributeManagerConfig:
    """Lifecycle and retention configuration.

    A track becomes ``lost`` on its first missing frame.  It becomes
    ``removed`` once consecutive missing frames exceed
    ``max_missed_frames``.  Removed state is retained for
    ``removed_ttl_ms`` before cleanup.  Each track retains only its most
    recent ``max_trajectory_points`` bottom-center observations.
    """

    max_missed_frames: int = 30
    removed_ttl_ms: float = 5_000.0
    max_trajectory_points: int = 120

    def __post_init__(self) -> None:
        _require_int(
            self.max_missed_frames,
            "TrackAttributeManagerConfig.max_missed_frames",
            minimum=1,
        )
        _require_timestamp(
            self.removed_ttl_ms,
            "TrackAttributeManagerConfig.removed_ttl_ms",
        )
        _require_int(
            self.max_trajectory_points,
            "TrackAttributeManagerConfig.max_trajectory_points",
            minimum=1,
        )


@dataclass
class _TrackState:
    """Mutable internal state; never returned across the public boundary."""

    track_id: int
    lifecycle: TrackLifecycle
    first_seen_frame_index: int
    last_seen_frame_index: int
    first_seen_ms: float
    last_seen_ms: float
    age_frames: int
    observed_frames: int
    missed_frames: int
    trajectory: Deque[TrajectoryPoint]
    removed_at_ms: Optional[float] = None

    @classmethod
    def create(
        cls,
        track: Track,
        point: TrajectoryPoint,
        frame_index: int,
        timestamp_ms: float,
        max_trajectory_points: int,
    ) -> "_TrackState":
        return cls(
            track_id=track.track_id,
            lifecycle=TrackLifecycle.ACTIVE,
            first_seen_frame_index=frame_index,
            last_seen_frame_index=frame_index,
            first_seen_ms=timestamp_ms,
            last_seen_ms=timestamp_ms,
            age_frames=1,
            observed_frames=1,
            missed_frames=0,
            trajectory=deque(
                (point,),
                maxlen=max_trajectory_points,
            ),
        )

    def observe(
        self,
        point: TrajectoryPoint,
        frame_index: int,
        timestamp_ms: float,
    ) -> None:
        self.lifecycle = TrackLifecycle.ACTIVE
        self.last_seen_frame_index = frame_index
        self.last_seen_ms = timestamp_ms
        self.age_frames = frame_index - self.first_seen_frame_index + 1
        self.observed_frames += 1
        self.missed_frames = 0
        self.removed_at_ms = None
        self.trajectory.append(point)

    def mark_missing(
        self,
        frame_index: int,
        timestamp_ms: float,
        max_missed_frames: int,
    ) -> None:
        self.age_frames = frame_index - self.first_seen_frame_index + 1
        self.missed_frames = frame_index - self.last_seen_frame_index
        if self.missed_frames > max_missed_frames:
            self.lifecycle = TrackLifecycle.REMOVED
            self.removed_at_ms = timestamp_ms
        else:
            self.lifecycle = TrackLifecycle.LOST

    def to_profile(self) -> TrackProfile:
        return TrackProfile(
            track_id=self.track_id,
            lifecycle=self.lifecycle,
            first_seen_frame_index=self.first_seen_frame_index,
            last_seen_frame_index=self.last_seen_frame_index,
            first_seen_ms=self.first_seen_ms,
            last_seen_ms=self.last_seen_ms,
            age_frames=self.age_frames,
            observed_frames=self.observed_frames,
            missed_frames=self.missed_frames,
            trajectory=tuple(self.trajectory),
        )


class TrackAttributeManager:
    """Own deterministic per-track lifecycle state in memory."""

    def __init__(
        self,
        config: Optional[TrackAttributeManagerConfig] = None,
    ) -> None:
        self._config = config or TrackAttributeManagerConfig()
        if not isinstance(self._config, TrackAttributeManagerConfig):
            raise TypeError(
                "config must be a TrackAttributeManagerConfig or None"
            )
        self._states: Dict[int, _TrackState] = {}
        self._last_frame_index: Optional[int] = None
        self._last_timestamp_ms: Optional[float] = None

    @property
    def config(self) -> TrackAttributeManagerConfig:
        return self._config

    @property
    def last_frame_index(self) -> Optional[int]:
        return self._last_frame_index

    @property
    def last_timestamp_ms(self) -> Optional[float]:
        return self._last_timestamp_ms

    def update(
        self,
        tracks: Sequence[Track],
        frame_index: int,
        timestamp_ms: float,
    ) -> List[TrackProfile]:
        """Apply one video frame and return retained profile snapshots.

        Validation is completed before any state mutation.  Frame indexes
        must increase strictly; presentation timestamps may stay equal but
        must never decrease.
        """
        observations = self._validate_update_input(
            tracks,
            frame_index,
            timestamp_ms,
        )
        seen_ids = {track.track_id for track, _point in observations}
        expired_ids = set(self._expired_ids(timestamp_ms))

        for track_id in seen_ids:
            state = self._states.get(track_id)
            if (
                state is not None
                and state.lifecycle is TrackLifecycle.REMOVED
                and track_id not in expired_ids
            ):
                raise ValueError(
                    f"Track ID {track_id} is removed and retained; "
                    "it cannot reactivate before TTL cleanup"
                )

        for track_id in expired_ids:
            del self._states[track_id]

        for state in self._states.values():
            if (
                state.track_id not in seen_ids
                and state.lifecycle is not TrackLifecycle.REMOVED
            ):
                state.mark_missing(
                    frame_index,
                    timestamp_ms,
                    self._config.max_missed_frames,
                )

        for track, point in observations:
            state = self._states.get(track.track_id)
            if state is None:
                self._states[track.track_id] = _TrackState.create(
                    track,
                    point,
                    frame_index,
                    timestamp_ms,
                    self._config.max_trajectory_points,
                )
            else:
                state.observe(point, frame_index, timestamp_ms)

        self._last_frame_index = frame_index
        self._last_timestamp_ms = timestamp_ms
        self._remove_expired(timestamp_ms)
        return self.get_profiles(include_removed=True)

    def get_profile(self, track_id: int) -> Optional[TrackProfile]:
        """Return an immutable snapshot for one retained track."""
        _require_int(track_id, "track_id", minimum=0)
        state = self._states.get(track_id)
        return None if state is None else state.to_profile()

    def get_profiles(
        self,
        *,
        include_removed: bool = True,
    ) -> List[TrackProfile]:
        """Return retained snapshots sorted by ``track_id``."""
        if not isinstance(include_removed, bool):
            raise TypeError("include_removed must be a bool")
        return [
            self._states[track_id].to_profile()
            for track_id in sorted(self._states)
            if (
                include_removed
                or self._states[track_id].lifecycle
                is not TrackLifecycle.REMOVED
            )
        ]

    def cleanup(self, timestamp_ms: float) -> List[int]:
        """Delete removed states whose retention TTL has elapsed.

        Returns sorted IDs that were deleted.  The supplied timestamp also
        advances the manager clock, so later updates cannot move back in time.
        """
        _require_timestamp(timestamp_ms, "timestamp_ms")
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            raise ValueError(
                "timestamp_ms cannot move backward relative to manager clock"
            )

        removed_ids = self._remove_expired(timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        return removed_ids

    def reset(self) -> None:
        """Clear all retained state and reset the input timeline."""
        self._states.clear()
        self._last_frame_index = None
        self._last_timestamp_ms = None

    def _validate_update_input(
        self,
        tracks: Sequence[Track],
        frame_index: int,
        timestamp_ms: float,
    ) -> Tuple[Tuple[Track, TrajectoryPoint], ...]:
        _require_int(frame_index, "frame_index", minimum=0)
        _require_timestamp(timestamp_ms, "timestamp_ms")

        if (
            self._last_frame_index is not None
            and frame_index <= self._last_frame_index
        ):
            raise ValueError(
                "frame_index must be strictly increasing; "
                f"last={self._last_frame_index}, got={frame_index}"
            )
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            raise ValueError(
                "timestamp_ms cannot move backward; "
                f"last={self._last_timestamp_ms}, got={timestamp_ms}"
            )
        if isinstance(tracks, (str, bytes)) or not isinstance(tracks, Sequence):
            raise TypeError("tracks must be a sequence of Track objects")

        track_items = tuple(tracks)
        seen_ids = set()
        observations = []
        for track in track_items:
            if not isinstance(track, Track):
                raise TypeError(
                    "tracks must contain only Track objects, "
                    f"got {type(track).__name__}"
                )
            _require_int(track.track_id, "Track.track_id", minimum=0)
            if track.track_id in seen_ids:
                raise ValueError(
                    f"Duplicate track_id in frame: {track.track_id}"
                )
            seen_ids.add(track.track_id)
            observations.append(
                (
                    track,
                    TrajectoryPoint(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        x=(track.x1 + track.x2) / 2.0,
                        y=track.y2,
                    ),
                )
            )
        return tuple(observations)

    def _expired_ids(self, timestamp_ms: float) -> List[int]:
        expired = []
        for track_id, state in self._states.items():
            if (
                state.lifecycle is TrackLifecycle.REMOVED
                and state.removed_at_ms is not None
            ):
                elapsed = timestamp_ms - state.removed_at_ms
                if elapsed >= self._config.removed_ttl_ms or math.isclose(
                    elapsed,
                    self._config.removed_ttl_ms,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    expired.append(track_id)
        return sorted(expired)

    def _remove_expired(self, timestamp_ms: float) -> List[int]:
        expired_ids = self._expired_ids(timestamp_ms)
        for track_id in expired_ids:
            del self._states[track_id]
        return expired_ids

    def __len__(self) -> int:
        return len(self._states)

    def __repr__(self) -> str:
        return (
            "TrackAttributeManager("
            f"states={len(self)}, "
            f"max_missed_frames={self._config.max_missed_frames}, "
            f"removed_ttl_ms={self._config.removed_ttl_ms}, "
            f"max_trajectory_points="
            f"{self._config.max_trajectory_points})"
        )


__all__ = [
    "TrackAttributeManager",
    "TrackAttributeManagerConfig",
]
