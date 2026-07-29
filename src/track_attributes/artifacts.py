"""JSON artifact helpers for M12 track profiles."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Dict, Optional

from src.track_attributes.schemas import (
    TRACK_PROFILE_SCHEMA_VERSION,
    TrackLifecycle,
    TrackProfile,
)


PROFILE_ARTIFACT_SCHEMA_VERSION = "m12.track_profiles.v1"


def build_profile_artifact(
    profiles: Sequence[TrackProfile],
    *,
    source: str,
    processed_frames: int,
) -> Dict[str, Any]:
    """Build a deterministic, JSON-ready profile artifact."""
    if isinstance(profiles, (str, bytes)) or not isinstance(
        profiles, Sequence
    ):
        raise TypeError("profiles must be a sequence of TrackProfile objects")
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if not source.strip():
        raise ValueError("source must not be blank")
    if isinstance(processed_frames, bool) or not isinstance(
        processed_frames, int
    ):
        raise TypeError("processed_frames must be an int")
    if processed_frames < 0:
        raise ValueError("processed_frames must be >= 0")

    by_id: Dict[int, TrackProfile] = {}
    for profile in profiles:
        if not isinstance(profile, TrackProfile):
            raise TypeError(
                "profiles must contain only TrackProfile objects"
            )
        if profile.track_id in by_id:
            raise ValueError(
                f"Duplicate track_id in profile artifact: {profile.track_id}"
            )
        by_id[profile.track_id] = profile

    ordered = [by_id[track_id] for track_id in sorted(by_id)]
    lifecycle_counts = {
        lifecycle.value: sum(
            profile.lifecycle is lifecycle for profile in ordered
        )
        for lifecycle in TrackLifecycle
    }

    return {
        "schema_version": PROFILE_ARTIFACT_SCHEMA_VERSION,
        "profile_schema_version": TRACK_PROFILE_SCHEMA_VERSION,
        "metadata": {
            "source": source,
            "timestamp_source": "VideoSource",
            "processed_frames": processed_frames,
        },
        "summary": {
            "total_profiles": len(ordered),
            "lifecycle_counts": lifecycle_counts,
        },
        "profiles": [profile.to_dict() for profile in ordered],
    }


def save_profile_artifact(
    artifact: Dict[str, Any],
    output_path: str,
) -> None:
    """Write a profile artifact, creating parent directories if needed."""
    if not isinstance(artifact, dict):
        raise TypeError("artifact must be a dict")
    if not isinstance(output_path, str):
        raise TypeError("output_path must be a string")
    if not output_path.strip():
        raise ValueError("output_path must not be blank")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)


def verify_profile_artifact(
    output_path: str,
    *,
    expected_processed_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Open and validate a generated M12 profile artifact."""
    if not os.path.isfile(output_path):
        raise RuntimeError(f"Profile artifact not found: {output_path!r}")

    try:
        with open(output_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not parse profile artifact: {output_path!r}"
        ) from exc

    errors = []
    if not isinstance(artifact, dict):
        raise RuntimeError("Profile artifact root must be an object")
    if artifact.get("schema_version") != PROFILE_ARTIFACT_SCHEMA_VERSION:
        errors.append("schema_version is invalid")
    if artifact.get("profile_schema_version") != (
        TRACK_PROFILE_SCHEMA_VERSION
    ):
        errors.append("profile_schema_version is invalid")

    metadata = artifact.get("metadata")
    summary = artifact.get("summary")
    profiles = artifact.get("profiles")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        metadata = {}
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    if not isinstance(profiles, list):
        errors.append("profiles must be an array")
        profiles = []

    processed_frames = metadata.get("processed_frames")
    if isinstance(processed_frames, bool) or not isinstance(
        processed_frames, int
    ):
        errors.append("metadata.processed_frames must be an int")
    elif processed_frames < 0:
        errors.append("metadata.processed_frames must be >= 0")
    if (
        expected_processed_frames is not None
        and processed_frames != expected_processed_frames
    ):
        errors.append(
            "processed_frames mismatch: "
            f"expected {expected_processed_frames}, got {processed_frames}"
        )

    total_profiles = summary.get("total_profiles")
    if total_profiles != len(profiles):
        errors.append(
            "summary.total_profiles does not match profiles length"
        )

    track_ids = []
    actual_lifecycle_counts = {
        lifecycle.value: 0 for lifecycle in TrackLifecycle
    }
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            errors.append(f"profiles[{index}] must be an object")
            continue
        if profile.get("schema_version") != TRACK_PROFILE_SCHEMA_VERSION:
            errors.append(f"profiles[{index}].schema_version is invalid")
        track_id = profile.get("track_id")
        if isinstance(track_id, bool) or not isinstance(track_id, int):
            errors.append(f"profiles[{index}].track_id must be an int")
        else:
            track_ids.append(track_id)
        lifecycle = profile.get("lifecycle")
        if lifecycle not in actual_lifecycle_counts:
            errors.append(f"profiles[{index}].lifecycle is invalid")
        else:
            actual_lifecycle_counts[lifecycle] += 1

    if len(track_ids) != len(set(track_ids)):
        errors.append("profiles contain duplicate track_id values")
    if summary.get("lifecycle_counts") != actual_lifecycle_counts:
        errors.append("summary.lifecycle_counts does not match profiles")

    if errors:
        raise RuntimeError(
            "Profile artifact verification failed:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )

    return {
        "path": output_path,
        "schema_version": artifact["schema_version"],
        "processed_frames": processed_frames,
        "total_profiles": total_profiles,
        "lifecycle_counts": actual_lifecycle_counts,
    }


__all__ = [
    "PROFILE_ARTIFACT_SCHEMA_VERSION",
    "build_profile_artifact",
    "save_profile_artifact",
    "verify_profile_artifact",
]
