"""M12-05 tests for deterministic temporal voting."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from src.schemas import Track
from src.track_attributes.manager import (
    TrackAttributeManager,
    TrackAttributeManagerConfig,
)
from src.track_attributes.schemas import AttributeObservation, StableAttribute


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


def _record(
    manager: TrackAttributeManager,
    frame_index: int,
    value: object,
    *,
    score: float = 0.8,
    namespace: str = "clothing",
    key: str = "upper_color",
    track_id: int = 1,
) -> None:
    timestamp_ms = frame_index * 10.0
    manager.update([_track(track_id)], frame_index, timestamp_ms)
    manager.record_observation(
        track_id,
        AttributeObservation(
            namespace=namespace,
            key=key,
            value=value,
            score=score,
            quality_score=0.9,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        ),
    )


class TestVotingConfig:
    def test_defaults_are_valid(self):
        config = TrackAttributeManagerConfig()
        assert config.voting_window_size >= 1
        assert 1 <= config.min_voting_observations <= (
            config.voting_window_size
        )
        assert config.min_voting_observations <= (
            config.max_observations_per_attribute
        )

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_rejects_invalid_window_size(self, value):
        with pytest.raises((TypeError, ValueError)):
            TrackAttributeManagerConfig(voting_window_size=value)

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_rejects_invalid_minimum_observations(self, value):
        with pytest.raises((TypeError, ValueError)):
            TrackAttributeManagerConfig(min_voting_observations=value)

    def test_rejects_minimum_larger_than_window(self):
        with pytest.raises(ValueError):
            TrackAttributeManagerConfig(
                voting_window_size=3,
                min_voting_observations=4,
            )

    def test_rejects_minimum_larger_than_cache_bound(self):
        with pytest.raises(ValueError):
            TrackAttributeManagerConfig(
                max_observations_per_attribute=2,
                min_voting_observations=3,
            )

    def test_config_is_frozen(self):
        config = TrackAttributeManagerConfig()
        with pytest.raises(FrozenInstanceError):
            config.voting_window_size = 99


class TestWeightedVoting:
    def test_no_stable_value_before_minimum_observations(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_voting_observations=3)
        )
        _record(manager, 0, "blue")
        _record(manager, 1, "blue")

        assert manager.get_stable_attribute(
            1, "clothing", "upper_color"
        ) is None
        assert dict(manager.get_profile(1).attributes) == {}

    def test_minimum_observations_produces_stable_attribute(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_voting_observations=3)
        )
        _record(manager, 0, "blue", score=0.8)
        _record(manager, 1, "blue", score=0.9)
        _record(manager, 2, "red", score=0.6)

        stable = manager.get_stable_attribute(
            1, "clothing", "upper_color"
        )
        assert isinstance(stable, StableAttribute)
        assert stable.value == "blue"
        assert stable.score == pytest.approx(1.7 / 2.3)
        assert stable.observation_count == 2
        assert stable.last_updated_frame_index == 2
        assert stable.last_updated_ms == 20.0

    def test_confidence_weight_can_beat_raw_vote_count(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_voting_observations=3)
        )
        _record(manager, 0, "blue", score=0.2)
        _record(manager, 1, "blue", score=0.2)
        _record(manager, 2, "red", score=0.9)

        stable = manager.get_stable_attribute(
            1, "clothing", "upper_color"
        )
        assert stable.value == "red"
        assert stable.score == pytest.approx(0.9 / 1.3)
        assert stable.observation_count == 1

    def test_single_noisy_frame_does_not_flip_stable_value(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                voting_window_size=5,
                min_voting_observations=3,
            )
        )
        for frame_index in range(3):
            _record(manager, frame_index, "blue")
        before_noise = manager.get_stable_attribute(
            1, "clothing", "upper_color"
        )

        _record(manager, 3, "red")
        after_noise = manager.get_stable_attribute(
            1, "clothing", "upper_color"
        )

        assert before_noise.value == "blue"
        assert after_noise.value == "blue"
        assert after_noise.observation_count == 3

    def test_sliding_window_eventually_allows_real_change(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                voting_window_size=5,
                min_voting_observations=3,
            )
        )
        for frame_index in range(3):
            _record(manager, frame_index, "blue")
        _record(manager, 3, "red")
        _record(manager, 4, "red")
        assert manager.get_stable_attribute(
            1, "clothing", "upper_color"
        ).value == "blue"

        _record(manager, 5, "red")
        stable = manager.get_stable_attribute(
            1, "clothing", "upper_color"
        )
        assert stable.value == "red"
        assert stable.observation_count == 3

    def test_more_votes_break_equal_weight_tie(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                voting_window_size=3,
                min_voting_observations=3,
            )
        )
        _record(manager, 0, "blue", score=0.25)
        _record(manager, 1, "blue", score=0.25)
        _record(manager, 2, "red", score=0.5)

        assert manager.get_stable_attribute(
            1, "clothing", "upper_color"
        ).value == "blue"

    def test_zero_weight_window_has_zero_stable_score(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                voting_window_size=2,
                min_voting_observations=2,
            )
        )
        _record(manager, 0, "red", score=0.0)
        _record(manager, 1, "blue", score=0.0)

        stable = manager.get_stable_attribute(
            1, "clothing", "upper_color"
        )
        assert stable.value == "blue"
        assert stable.score == 0.0


class TestDeterministicTieBreak:
    def _vote(self, values) -> StableAttribute:
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                voting_window_size=4,
                min_voting_observations=4,
            )
        )
        for frame_index, value in enumerate(values):
            _record(manager, frame_index, value, score=0.5)
        return manager.get_stable_attribute(
            1, "clothing", "upper_color"
        )

    def test_equal_vote_tie_uses_canonical_value(self):
        stable = self._vote(["red", "blue", "red", "blue"])
        assert stable.value == "blue"

    def test_tie_result_does_not_depend_on_observation_order(self):
        forward = self._vote(["red", "blue", "red", "blue"])
        reverse = self._vote(["blue", "red", "blue", "red"])
        assert forward.value == reverse.value == "blue"
        assert forward.score == reverse.score

    def test_bool_and_int_values_do_not_share_a_vote_bucket(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                voting_window_size=2,
                min_voting_observations=2,
            )
        )
        _record(manager, 0, 1, score=0.5)
        _record(manager, 1, True, score=0.5)

        stable = manager.get_stable_attribute(
            1, "clothing", "upper_color"
        )
        assert stable.value is True
        assert stable.observation_count == 1


class TestStableProfile:
    def test_stable_attributes_are_nested_by_namespace_and_key(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_voting_observations=1)
        )
        _record(manager, 0, "blue")

        profile = manager.get_profile(1)
        stable = profile.attributes["clothing"]["upper_color"]
        assert stable.value == "blue"
        assert json.loads(json.dumps(profile.to_dict()))["attributes"][
            "clothing"
        ]["upper_color"]["value"] == "blue"

    def test_multiple_attributes_are_independent(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_voting_observations=1)
        )
        manager.update([_track()], 0, 0.0)
        manager.record_observation(
            1,
            AttributeObservation(
                "clothing", "upper_color", "blue", 0.8, 0.9, 0, 0.0
            ),
        )
        manager.record_observation(
            1,
            AttributeObservation(
                "body", "backpack", True, 0.9, 0.9, 0, 0.0
            ),
        )

        profile = manager.get_profile(1)
        assert profile.attributes["clothing"]["upper_color"].value == "blue"
        assert profile.attributes["body"]["backpack"].value is True

    def test_stable_attribute_is_immutable(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_voting_observations=1)
        )
        _record(manager, 0, "blue")
        stable = manager.get_stable_attribute(
            1, "clothing", "upper_color"
        )

        with pytest.raises(FrozenInstanceError):
            stable.value = "red"

    def test_record_inference_without_observation_does_not_vote(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_voting_observations=1)
        )
        manager.update([_track()], 0, 0.0)
        manager.record_inference(1, "clothing", "upper_color")

        assert manager.get_stable_attribute(
            1, "clothing", "upper_color"
        ) is None

    def test_stable_value_is_retained_while_track_is_lost(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_voting_observations=1)
        )
        _record(manager, 0, "blue")
        manager.update([], 1, 10.0)

        assert manager.get_profile(1).attributes[
            "clothing"
        ]["upper_color"].value == "blue"

    def test_missing_stable_key_returns_none(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)

        assert manager.get_stable_attribute(
            1, "body", "backpack"
        ) is None


class TestStableLifecycle:
    def test_reset_clears_stable_attributes(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(min_voting_observations=1)
        )
        _record(manager, 0, "blue")
        manager.reset()
        manager.update([_track()], 0, 0.0)

        assert manager.get_stable_attribute(
            1, "clothing", "upper_color"
        ) is None

    def test_ttl_cleanup_removes_stable_attributes_with_state(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                max_missed_frames=1,
                removed_ttl_ms=10.0,
                min_voting_observations=1,
            )
        )
        _record(manager, 0, "blue")
        manager.update([], 1, 10.0)
        manager.update([], 2, 20.0)
        assert manager.cleanup(30.0) == [1]

        with pytest.raises(KeyError):
            manager.get_stable_attribute(
                1, "clothing", "upper_color"
            )
