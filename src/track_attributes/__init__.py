"""M12 track attribute state contracts and management."""

from src.track_attributes.schemas import (
    TRACK_PROFILE_SCHEMA_VERSION,
    AttributeObservation,
    StableAttribute,
    TrackLifecycle,
    TrackProfile,
    TrajectoryPoint,
)
from src.track_attributes.manager import (
    TrackAttributeManager,
    TrackAttributeManagerConfig,
)

__all__ = [
    "TRACK_PROFILE_SCHEMA_VERSION",
    "AttributeObservation",
    "StableAttribute",
    "TrackAttributeManager",
    "TrackAttributeManagerConfig",
    "TrackLifecycle",
    "TrackProfile",
    "TrajectoryPoint",
]
