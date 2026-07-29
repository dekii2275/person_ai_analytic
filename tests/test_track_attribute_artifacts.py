"""M12-06 tests for profile artifact contracts and CLI wiring."""

from __future__ import annotations

import ast
import inspect
import json

import pytest

from main import _parse_args
from src.track_attributes.artifacts import (
    PROFILE_ARTIFACT_SCHEMA_VERSION,
    build_profile_artifact,
    save_profile_artifact,
    verify_profile_artifact,
)
from src.track_attributes.schemas import TrackLifecycle, TrackProfile


def _profile(
    track_id: int,
    lifecycle: TrackLifecycle = TrackLifecycle.ACTIVE,
) -> TrackProfile:
    return TrackProfile(
        track_id=track_id,
        lifecycle=lifecycle,
        first_seen_frame_index=0,
        last_seen_frame_index=2,
        first_seen_ms=0.0,
        last_seen_ms=66.7,
        age_frames=3,
        observed_frames=3,
        missed_frames=0,
    )


class TestTrackingProfilesCli:
    def test_default_output_path(self):
        args = _parse_args(["--tracking"])
        assert args.tracking_profiles == "outputs/tracking_profiles.json"

    def test_custom_output_path(self):
        args = _parse_args(
            ["--tracking", "--tracking-profiles", "tmp/profiles.json"]
        )
        assert args.tracking_profiles == "tmp/profiles.json"


class TestBuildProfileArtifact:
    def test_has_versioned_json_shape(self):
        artifact = build_profile_artifact(
            [_profile(2), _profile(1, TrackLifecycle.LOST)],
            source="data/input.mp4",
            processed_frames=259,
        )

        assert artifact["schema_version"] == PROFILE_ARTIFACT_SCHEMA_VERSION
        assert artifact["profile_schema_version"] == "1.0"
        assert artifact["metadata"] == {
            "source": "data/input.mp4",
            "timestamp_source": "VideoSource",
            "processed_frames": 259,
        }
        assert artifact["summary"]["total_profiles"] == 2
        assert artifact["summary"]["lifecycle_counts"] == {
            "active": 1,
            "lost": 1,
            "removed": 0,
        }
        assert [item["track_id"] for item in artifact["profiles"]] == [1, 2]
        assert json.loads(json.dumps(artifact)) == artifact

    def test_rejects_duplicate_track_ids(self):
        with pytest.raises(ValueError, match="Duplicate"):
            build_profile_artifact(
                [_profile(1), _profile(1)],
                source="data/input.mp4",
                processed_frames=10,
            )

    @pytest.mark.parametrize("processed_frames", [-1, 1.5, True])
    def test_rejects_invalid_processed_frames(self, processed_frames):
        with pytest.raises((TypeError, ValueError)):
            build_profile_artifact(
                [],
                source="data/input.mp4",
                processed_frames=processed_frames,
            )

    def test_rejects_non_profile_item(self):
        with pytest.raises(TypeError):
            build_profile_artifact(
                ["not-a-profile"],
                source="data/input.mp4",
                processed_frames=1,
            )


class TestSaveAndVerifyProfileArtifact:
    def test_save_creates_parseable_artifact(self, tmp_path):
        output = tmp_path / "nested" / "profiles.json"
        artifact = build_profile_artifact(
            [_profile(1)],
            source="data/input.mp4",
            processed_frames=3,
        )

        save_profile_artifact(artifact, str(output))
        info = verify_profile_artifact(
            str(output),
            expected_processed_frames=3,
        )

        assert output.is_file()
        assert info["total_profiles"] == 1
        assert info["processed_frames"] == 3
        assert info["schema_version"] == PROFILE_ARTIFACT_SCHEMA_VERSION

    def test_verify_rejects_wrong_frame_count(self, tmp_path):
        output = tmp_path / "profiles.json"
        artifact = build_profile_artifact(
            [],
            source="data/input.mp4",
            processed_frames=3,
        )
        save_profile_artifact(artifact, str(output))

        with pytest.raises(RuntimeError, match="processed_frames"):
            verify_profile_artifact(
                str(output),
                expected_processed_frames=4,
            )

    def test_verify_rejects_mismatched_profile_total(self, tmp_path):
        output = tmp_path / "profiles.json"
        artifact = build_profile_artifact(
            [_profile(1)],
            source="data/input.mp4",
            processed_frames=3,
        )
        artifact["summary"]["total_profiles"] = 99
        save_profile_artifact(artifact, str(output))

        with pytest.raises(RuntimeError, match="total_profiles"):
            verify_profile_artifact(str(output))


def test_artifact_module_has_no_model_backend_imports():
    import src.track_attributes.artifacts as artifacts

    tree = ast.parse(inspect.getsource(artifacts))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden = ("torch", "ultralytics", "onnxruntime")
    assert not any(
        module == name or module.startswith(f"{name}.")
        for module in imported_modules
        for name in forbidden
    )
