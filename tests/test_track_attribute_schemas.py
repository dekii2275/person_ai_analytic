"""M12-01 tests for framework-independent track attribute contracts."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import FrozenInstanceError

import pytest

from src.track_attributes.schemas import (
    TRACK_PROFILE_SCHEMA_VERSION,
    AttributeObservation,
    StableAttribute,
    TrackLifecycle,
    TrackProfile,
    TrajectoryPoint,
)


def _point(frame_index: int = 0, timestamp_ms: float = 0.0) -> TrajectoryPoint:
    return TrajectoryPoint(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        x=100.0 + frame_index,
        y=200.0 + frame_index,
    )


def _stable(value: object = "blue") -> StableAttribute:
    return StableAttribute(
        value=value,
        score=0.9,
        observation_count=3,
        last_updated_frame_index=2,
        last_updated_ms=66.7,
    )


def _profile(**overrides) -> TrackProfile:
    values = {
        "track_id": 7,
        "lifecycle": TrackLifecycle.ACTIVE,
        "first_seen_frame_index": 0,
        "last_seen_frame_index": 2,
        "first_seen_ms": 0.0,
        "last_seen_ms": 66.7,
        "age_frames": 3,
        "observed_frames": 3,
        "missed_frames": 0,
        "trajectory": (_point(0, 0.0), _point(2, 66.7)),
        "attributes": {"clothing": {"upper_color": _stable()}},
    }
    values.update(overrides)
    return TrackProfile(**values)


class TestTrackLifecycle:
    def test_values_are_stable_strings(self):
        assert [state.value for state in TrackLifecycle] == [
            "active",
            "lost",
            "removed",
        ]

    def test_is_json_serializable(self):
        assert json.dumps({"state": TrackLifecycle.ACTIVE}) == '{"state": "active"}'


class TestTrajectoryPoint:
    def test_to_dict_shape(self):
        point = _point(2, 66.7)
        assert point.to_dict() == {
            "frame_index": 2,
            "timestamp_ms": 66.7,
            "x": 102.0,
            "y": 202.0,
        }

    def test_is_frozen(self):
        point = _point()
        with pytest.raises(FrozenInstanceError):
            point.x = 999.0

    @pytest.mark.parametrize("frame_index", [-1, 1.5, True])
    def test_rejects_invalid_frame_index(self, frame_index):
        with pytest.raises((TypeError, ValueError)):
            TrajectoryPoint(frame_index, 0.0, 1.0, 2.0)

    @pytest.mark.parametrize("timestamp_ms", [-0.1, float("nan"), float("inf")])
    def test_rejects_invalid_timestamp(self, timestamp_ms):
        with pytest.raises(ValueError):
            TrajectoryPoint(0, timestamp_ms, 1.0, 2.0)

    @pytest.mark.parametrize("coordinate", [float("nan"), float("inf")])
    def test_rejects_non_finite_coordinate(self, coordinate):
        with pytest.raises(ValueError):
            TrajectoryPoint(0, 0.0, coordinate, 2.0)


class TestAttributeObservation:
    def test_to_dict_shape(self):
        observation = AttributeObservation(
            namespace="clothing",
            key="upper_color",
            value="blue",
            score=0.87,
            quality_score=0.75,
            frame_index=10,
            timestamp_ms=333.3,
        )
        assert observation.to_dict() == {
            "namespace": "clothing",
            "key": "upper_color",
            "value": "blue",
            "score": 0.87,
            "quality_score": 0.75,
            "frame_index": 10,
            "timestamp_ms": 333.3,
        }

    def test_supports_common_json_scalar_values(self):
        for value in (True, 4, 1.5, "unknown"):
            observation = AttributeObservation(
                "body", "attribute", value, 0.8, 0.9, 0, 0.0
            )
            assert observation.value == value

    @pytest.mark.parametrize("value", [None, [], {}, float("nan"), float("inf")])
    def test_rejects_non_scalar_or_non_finite_values(self, value):
        with pytest.raises((TypeError, ValueError)):
            AttributeObservation("body", "attribute", value, 0.8, 0.9, 0, 0.0)

    @pytest.mark.parametrize("namespace,key", [("", "color"), ("body", "  ")])
    def test_rejects_blank_namespace_or_key(self, namespace, key):
        with pytest.raises(ValueError):
            AttributeObservation(namespace, key, True, 0.8, 0.9, 0, 0.0)

    @pytest.mark.parametrize("score", [-0.1, 1.1, float("nan")])
    def test_rejects_invalid_score(self, score):
        with pytest.raises(ValueError):
            AttributeObservation("body", "bag", True, score, 0.9, 0, 0.0)

    @pytest.mark.parametrize("quality_score", [-0.1, 1.1, float("nan")])
    def test_rejects_invalid_quality_score(self, quality_score):
        with pytest.raises(ValueError):
            AttributeObservation(
                "body", "bag", True, 0.8, quality_score, 0, 0.0
            )

    def test_is_frozen(self):
        observation = AttributeObservation(
            "body", "bag", True, 0.8, 0.9, 0, 0.0
        )
        with pytest.raises(FrozenInstanceError):
            observation.score = 0.1


class TestStableAttribute:
    def test_to_dict_shape(self):
        assert _stable().to_dict() == {
            "value": "blue",
            "score": 0.9,
            "observation_count": 3,
            "last_updated_frame_index": 2,
            "last_updated_ms": 66.7,
        }

    @pytest.mark.parametrize("observation_count", [0, -1, 1.5, True])
    def test_rejects_invalid_observation_count(self, observation_count):
        with pytest.raises((TypeError, ValueError)):
            StableAttribute("blue", 0.9, observation_count, 2, 66.7)

    def test_is_frozen(self):
        stable = _stable()
        with pytest.raises(FrozenInstanceError):
            stable.value = "red"


class TestTrackProfile:
    def test_schema_version_is_fixed(self):
        assert _profile().schema_version == TRACK_PROFILE_SCHEMA_VERSION
        assert TRACK_PROFILE_SCHEMA_VERSION == "1.0"

    def test_to_dict_is_json_ready(self):
        payload = _profile().to_dict()
        assert payload == {
            "schema_version": "1.0",
            "track_id": 7,
            "lifecycle": "active",
            "first_seen_frame_index": 0,
            "last_seen_frame_index": 2,
            "first_seen_ms": 0.0,
            "last_seen_ms": 66.7,
            "age_frames": 3,
            "observed_frames": 3,
            "missed_frames": 0,
            "trajectory": [
                {
                    "frame_index": 0,
                    "timestamp_ms": 0.0,
                    "x": 100.0,
                    "y": 200.0,
                },
                {
                    "frame_index": 2,
                    "timestamp_ms": 66.7,
                    "x": 102.0,
                    "y": 202.0,
                },
            ],
            "attributes": {
                "clothing": {
                    "upper_color": {
                        "value": "blue",
                        "score": 0.9,
                        "observation_count": 3,
                        "last_updated_frame_index": 2,
                        "last_updated_ms": 66.7,
                    }
                }
            },
        }
        assert json.loads(json.dumps(payload)) == payload

    def test_copies_and_freezes_nested_inputs(self):
        trajectory = [_point()]
        attributes = {"body": {"backpack": _stable(True)}}
        profile = _profile(trajectory=trajectory, attributes=attributes)

        trajectory.append(_point(1, 33.3))
        attributes["body"]["backpack"] = _stable(False)

        assert len(profile.trajectory) == 1
        assert profile.attributes["body"]["backpack"].value is True
        with pytest.raises(TypeError):
            profile.attributes["body"]["backpack"] = _stable(False)

    def test_is_frozen(self):
        profile = _profile()
        with pytest.raises(FrozenInstanceError):
            profile.track_id = 99

    @pytest.mark.parametrize("track_id", [-1, 1.5, True])
    def test_rejects_invalid_track_id(self, track_id):
        with pytest.raises((TypeError, ValueError)):
            _profile(track_id=track_id)

    def test_requires_lifecycle_enum(self):
        with pytest.raises(TypeError):
            _profile(lifecycle="active")

    def test_rejects_last_seen_before_first_seen(self):
        with pytest.raises(ValueError):
            _profile(last_seen_frame_index=-1)
        with pytest.raises(ValueError):
            _profile(last_seen_ms=-0.1)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("age_frames", 0),
            ("observed_frames", 0),
            ("missed_frames", -1),
        ],
    )
    def test_rejects_invalid_counters(self, field, value):
        with pytest.raises(ValueError):
            _profile(**{field: value})

    def test_age_must_cover_observed_and_missed_frames(self):
        with pytest.raises(ValueError):
            _profile(age_frames=3, observed_frames=3, missed_frames=1)

    def test_rejects_unordered_trajectory(self):
        with pytest.raises(ValueError):
            _profile(
                trajectory=(_point(2, 66.7), _point(1, 33.3)),
            )

    def test_rejects_wrong_nested_types(self):
        with pytest.raises(TypeError):
            _profile(trajectory=("not-a-point",))
        with pytest.raises(TypeError):
            _profile(attributes={"body": {"bag": "not-a-stable-attribute"}})


def test_m12_schema_module_has_no_model_backend_imports():
    import src.track_attributes.schemas as schemas

    tree = ast.parse(inspect.getsource(schemas))
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
