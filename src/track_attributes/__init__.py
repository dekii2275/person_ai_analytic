"""M12 track attribute state contracts and management."""

from src.track_attributes.artifacts import (
    PROFILE_ARTIFACT_SCHEMA_VERSION,
    build_profile_artifact,
    save_profile_artifact,
    verify_profile_artifact,
)
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
    "PROFILE_ARTIFACT_SCHEMA_VERSION",
    "TRACK_PROFILE_SCHEMA_VERSION",
    "AttributeObservation",
    "StableAttribute",
    "TrackAttributeManager",
    "TrackAttributeManagerConfig",
    "TrackLifecycle",
    "TrackProfile",
    "TrajectoryPoint",
    "build_profile_artifact",
    "save_profile_artifact",
    "verify_profile_artifact",
]
