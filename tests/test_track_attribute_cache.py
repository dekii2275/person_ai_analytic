"""M12-04 tests for attribute cache and inference scheduling."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.schemas import Track
from src.track_attributes.manager import (
    TrackAttributeManager,
    TrackAttributeManagerConfig,
)
from src.track_attributes.schemas import AttributeObservation


def _track(track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        x1=10.0,
        y1=20.0,
        x2=70.0,
        y2=180.0,
        score=0.9,
        class_id=0,
    )


def _observation(
    *,
    frame_index: int,
    timestamp_ms: float,
    namespace: str = "clothing",
    key: str = "upper_color",
    value: object = "blue",
    quality_score: float = 0.9,
) -> AttributeObservation:
    return AttributeObservation(
        namespace=namespace,
        key=key,
        value=value,
        score=0.85,
        quality_score=quality_score,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
    )


class TestCacheConfig:
    def test_defaults_are_valid(self):
        config = TrackAttributeManagerConfig()
        assert config.inference_interval_frames >= 1
        assert 0.0 <= config.min_inference_quality_score <= 1.0
        assert config.max_observations_per_attribute >= 1

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_rejects_invalid_inference_interval(self, value):
        with pytest.raises((TypeError, ValueError)):
            TrackAttributeManagerConfig(inference_interval_frames=value)

    @pytest.mark.parametrize(
        "value",
        [-0.1, 1.1, float("nan"), float("inf"), True],
    )
    def test_rejects_invalid_quality_gate(self, value):
        with pytest.raises((TypeError, ValueError)):
            TrackAttributeManagerConfig(min_inference_quality_score=value)

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_rejects_invalid_cache_bound(self, value):
        with pytest.raises((TypeError, ValueError)):
            TrackAttributeManagerConfig(
                max_observations_per_attribute=value
            )

    def test_config_remains_frozen(self):
        config = TrackAttributeManagerConfig()
        with pytest.raises(FrozenInstanceError):
            config.inference_interval_frames = 99


class TestShouldInfer:
    def test_first_eligible_frame_returns_true(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)

        assert manager.should_infer(1, "body", "backpack", 0.9) is True

    def test_quality_below_gate_returns_false(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_inference_quality_score=0.6)
        )
        manager.update([_track()], 0, 0.0)

        assert manager.should_infer(1, "body", "backpack", 0.59) is False
        assert manager.should_infer(1, "body", "backpack", 0.6) is True

    def test_call_is_pure_until_inference_is_recorded(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)

        assert manager.should_infer(1, "body", "backpack", 0.9) is True
        assert manager.should_infer(1, "body", "backpack", 0.9) is True

    def test_interval_boundary_is_inclusive(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(inference_interval_frames=3)
        )
        manager.update([_track()], 0, 0.0)
        manager.record_inference(1, "body", "backpack")

        manager.update([_track()], 1, 33.3)
        assert manager.should_infer(1, "body", "backpack", 0.9) is False
        manager.update([_track()], 2, 66.7)
        assert manager.should_infer(1, "body", "backpack", 0.9) is False
        manager.update([_track()], 3, 100.0)
        assert manager.should_infer(1, "body", "backpack", 0.9) is True

    def test_schedule_is_independent_per_namespace_and_key(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.record_inference(1, "body", "backpack")

        assert manager.should_infer(1, "body", "backpack", 0.9) is False
        assert manager.should_infer(1, "body", "long_sleeve", 0.9) is True
        assert manager.should_infer(1, "head", "backpack", 0.9) is True

    def test_lost_track_is_not_eligible(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.update([], 1, 33.3)

        assert manager.should_infer(1, "body", "backpack", 0.9) is False

    @pytest.mark.parametrize(
        "quality_score",
        [-0.1, 1.1, float("nan"), float("inf"), True],
    )
    def test_rejects_invalid_candidate_quality(self, quality_score):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        with pytest.raises((TypeError, ValueError)):
            manager.should_infer(1, "body", "backpack", quality_score)

    def test_unknown_track_is_rejected(self):
        manager = TrackAttributeManager()
        with pytest.raises(KeyError):
            manager.should_infer(99, "body", "backpack", 0.9)


class TestRecordInference:
    def test_records_current_frame_and_timestamp_without_observation(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 4, 133.3)
        manager.record_inference(1, "body", "backpack")

        assert manager.get_last_inference(1, "body", "backpack") == (
            4,
            133.3,
        )
        assert manager.get_observations(1, "body", "backpack") == ()
        assert manager.should_infer(1, "body", "backpack", 0.9) is False

    def test_duplicate_inference_in_same_frame_is_rejected(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.record_inference(1, "body", "backpack")

        with pytest.raises(ValueError, match="already"):
            manager.record_inference(1, "body", "backpack")

    def test_lost_track_cannot_record_inference(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.update([], 1, 33.3)

        with pytest.raises(ValueError, match="active"):
            manager.record_inference(1, "body", "backpack")

    @pytest.mark.parametrize("namespace,key", [("", "bag"), ("body", "  ")])
    def test_rejects_blank_namespace_or_key(self, namespace, key):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        with pytest.raises(ValueError):
            manager.record_inference(1, namespace, key)


class TestObservationCache:
    def test_record_observation_caches_and_marks_inference(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        observation = _observation(frame_index=0, timestamp_ms=0.0)

        manager.record_observation(1, observation)

        assert manager.get_observations(
            1, "clothing", "upper_color"
        ) == (observation,)
        assert manager.get_last_inference(
            1, "clothing", "upper_color"
        ) == (0, 0.0)
        assert manager.should_infer(
            1, "clothing", "upper_color", 0.9
        ) is False

    def test_observation_can_follow_mark_inference_in_same_frame(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.record_inference(1, "clothing", "upper_color")
        observation = _observation(frame_index=0, timestamp_ms=0.0)

        manager.record_observation(1, observation)

        assert manager.get_observations(
            1, "clothing", "upper_color"
        ) == (observation,)

    def test_duplicate_observation_in_same_frame_is_rejected(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        observation = _observation(frame_index=0, timestamp_ms=0.0)
        manager.record_observation(1, observation)

        with pytest.raises(ValueError, match="already"):
            manager.record_observation(1, observation)

    def test_rejects_observation_from_different_frame(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 1, 33.3)
        stale = _observation(frame_index=0, timestamp_ms=0.0)

        with pytest.raises(ValueError, match="current manager frame"):
            manager.record_observation(1, stale)

        assert manager.get_observations(
            1, "clothing", "upper_color"
        ) == ()
        assert manager.get_last_inference(
            1, "clothing", "upper_color"
        ) is None

    def test_rejects_observation_with_mismatched_timestamp(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 1, 33.3)
        bad_timestamp = _observation(frame_index=1, timestamp_ms=99.0)

        with pytest.raises(ValueError, match="current manager timestamp"):
            manager.record_observation(1, bad_timestamp)

    def test_cache_keeps_only_latest_configured_observations(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                max_observations_per_attribute=3,
            )
        )
        for frame_index in range(100):
            timestamp_ms = frame_index * 10.0
            manager.update([_track()], frame_index, timestamp_ms)
            manager.record_observation(
                1,
                _observation(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    value=str(frame_index),
                ),
            )

        cached = manager.get_observations(
            1, "clothing", "upper_color"
        )
        assert len(cached) == 3
        assert [item.frame_index for item in cached] == [97, 98, 99]
        assert [item.value for item in cached] == ["97", "98", "99"]

    def test_cache_is_independent_per_attribute(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        upper = _observation(frame_index=0, timestamp_ms=0.0)
        backpack = _observation(
            frame_index=0,
            timestamp_ms=0.0,
            namespace="body",
            key="backpack",
            value=True,
        )
        manager.record_observation(1, upper)
        manager.record_observation(1, backpack)

        assert manager.get_observations(
            1, "clothing", "upper_color"
        ) == (upper,)
        assert manager.get_observations(1, "body", "backpack") == (
            backpack,
        )

    def test_returned_cache_snapshot_is_detached(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.record_observation(
            1, _observation(frame_index=0, timestamp_ms=0.0)
        )
        old_snapshot = manager.get_observations(
            1, "clothing", "upper_color"
        )

        manager.update([_track()], 1, 33.3)
        manager.record_observation(
            1, _observation(frame_index=1, timestamp_ms=33.3, value="red")
        )

        assert len(old_snapshot) == 1
        assert len(
            manager.get_observations(1, "clothing", "upper_color")
        ) == 2

    def test_profile_attributes_remain_empty_before_voting(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.record_observation(
            1, _observation(frame_index=0, timestamp_ms=0.0)
        )

        assert dict(manager.get_profile(1).attributes) == {}


class TestCacheLifecycle:
    def test_reset_clears_cache_and_inference_history(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.record_observation(
            1, _observation(frame_index=0, timestamp_ms=0.0)
        )
        manager.reset()
        manager.update([_track()], 0, 0.0)

        assert manager.get_observations(
            1, "clothing", "upper_color"
        ) == ()
        assert manager.get_last_inference(
            1, "clothing", "upper_color"
        ) is None

    def test_ttl_cleanup_discards_cache_with_track_state(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                max_missed_frames=1,
                removed_ttl_ms=10.0,
            )
        )
        manager.update([_track()], 0, 0.0)
        manager.record_observation(
            1, _observation(frame_index=0, timestamp_ms=0.0)
        )
        manager.update([], 1, 10.0)
        manager.update([], 2, 20.0)
        assert manager.cleanup(30.0) == [1]

        with pytest.raises(KeyError):
            manager.get_observations(1, "clothing", "upper_color")
