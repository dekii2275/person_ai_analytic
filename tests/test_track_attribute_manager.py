"""M12-02 tests for lifecycle state management."""

from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError

import pytest

from src.schemas import Track
from src.track_attributes.manager import (
    TrackAttributeManager,
    TrackAttributeManagerConfig,
)
from src.track_attributes.schemas import TrackLifecycle, TrackProfile


def _track(track_id: int = 1, x1: float = 10.0) -> Track:
    return Track(
        track_id=track_id,
        x1=x1,
        y1=20.0,
        x2=x1 + 50.0,
        y2=180.0,
        score=0.9,
        class_id=0,
    )


class TestManagerConfig:
    def test_defaults_are_valid(self):
        config = TrackAttributeManagerConfig()
        assert config.max_missed_frames >= 1
        assert config.removed_ttl_ms >= 0.0
        assert config.max_trajectory_points >= 1

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_rejects_invalid_max_missed_frames(self, value):
        with pytest.raises((TypeError, ValueError)):
            TrackAttributeManagerConfig(max_missed_frames=value)

    @pytest.mark.parametrize(
        "value",
        [-0.1, float("nan"), float("inf"), True],
    )
    def test_rejects_invalid_removed_ttl(self, value):
        with pytest.raises((TypeError, ValueError)):
            TrackAttributeManagerConfig(removed_ttl_ms=value)

    def test_is_frozen(self):
        config = TrackAttributeManagerConfig()
        with pytest.raises(FrozenInstanceError):
            config.max_missed_frames = 10

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_rejects_invalid_max_trajectory_points(self, value):
        with pytest.raises((TypeError, ValueError)):
            TrackAttributeManagerConfig(max_trajectory_points=value)


class TestCreateUpdateAndQuery:
    def test_new_track_creates_active_profile(self):
        manager = TrackAttributeManager()
        profiles = manager.update([_track(7)], frame_index=0, timestamp_ms=0.0)

        assert len(profiles) == 1
        assert isinstance(profiles[0], TrackProfile)
        assert profiles[0].track_id == 7
        assert profiles[0].lifecycle is TrackLifecycle.ACTIVE
        assert profiles[0].first_seen_frame_index == 0
        assert profiles[0].last_seen_frame_index == 0
        assert profiles[0].first_seen_ms == 0.0
        assert profiles[0].last_seen_ms == 0.0
        assert profiles[0].age_frames == 1
        assert profiles[0].observed_frames == 1
        assert profiles[0].missed_frames == 0

    def test_existing_track_updates_seen_metadata(self):
        manager = TrackAttributeManager()
        manager.update([_track()], frame_index=3, timestamp_ms=100.0)
        manager.update([_track()], frame_index=5, timestamp_ms=166.7)

        profile = manager.get_profile(1)
        assert profile is not None
        assert profile.first_seen_frame_index == 3
        assert profile.last_seen_frame_index == 5
        assert profile.first_seen_ms == 100.0
        assert profile.last_seen_ms == 166.7
        assert profile.age_frames == 3
        assert profile.observed_frames == 2
        assert profile.missed_frames == 0

    def test_profiles_are_sorted_by_track_id(self):
        manager = TrackAttributeManager()
        profiles = manager.update(
            [_track(9), _track(2), _track(5)],
            frame_index=0,
            timestamp_ms=0.0,
        )
        assert [profile.track_id for profile in profiles] == [2, 5, 9]

    def test_get_missing_profile_returns_none(self):
        manager = TrackAttributeManager()
        assert manager.get_profile(123) is None

    def test_get_profiles_can_exclude_removed(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                max_missed_frames=1,
                removed_ttl_ms=1_000.0,
            )
        )
        manager.update([_track()], 0, 0.0)
        manager.update([], 1, 33.3)
        manager.update([], 2, 66.7)

        assert manager.get_profiles(include_removed=False) == []
        assert manager.get_profiles(include_removed=True)[0].lifecycle is (
            TrackLifecycle.REMOVED
        )

    def test_len_reports_retained_state_count(self):
        manager = TrackAttributeManager()
        assert len(manager) == 0
        manager.update([_track(1), _track(2)], 0, 0.0)
        assert len(manager) == 2


class TestLifecycleTransitions:
    def test_missing_track_becomes_lost(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        profiles = manager.update([], 1, 33.3)

        assert profiles[0].lifecycle is TrackLifecycle.LOST
        assert profiles[0].missed_frames == 1
        assert profiles[0].age_frames == 2
        assert profiles[0].last_seen_frame_index == 0

    def test_lost_track_can_become_active_again(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.update([], 1, 33.3)
        profiles = manager.update([_track()], 2, 66.7)

        assert profiles[0].lifecycle is TrackLifecycle.ACTIVE
        assert profiles[0].missed_frames == 0
        assert profiles[0].observed_frames == 2
        assert profiles[0].age_frames == 3

    def test_lost_track_becomes_removed_after_threshold(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                max_missed_frames=2,
                removed_ttl_ms=1_000.0,
            )
        )
        manager.update([_track()], 0, 0.0)
        assert manager.update([], 1, 33.3)[0].lifecycle is TrackLifecycle.LOST
        assert manager.update([], 2, 66.7)[0].lifecycle is TrackLifecycle.LOST
        removed = manager.update([], 3, 100.0)[0]

        assert removed.lifecycle is TrackLifecycle.REMOVED
        assert removed.missed_frames == 3
        assert removed.age_frames == 4

    def test_skipped_frames_apply_deterministic_threshold(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(max_missed_frames=2)
        )
        manager.update([_track()], 10, 100.0)
        profile = manager.update([], 14, 233.3)[0]

        assert profile.lifecycle is TrackLifecycle.REMOVED
        assert profile.missed_frames == 4
        assert profile.age_frames == 5

    def test_retained_removed_track_cannot_reactivate(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                max_missed_frames=1,
                removed_ttl_ms=1_000.0,
            )
        )
        manager.update([_track()], 0, 0.0)
        manager.update([], 1, 33.3)
        manager.update([], 2, 66.7)

        with pytest.raises(ValueError, match="removed"):
            manager.update([_track()], 3, 100.0)


class TestBoundedTrajectory:
    def test_new_track_stores_bottom_center_point(self):
        manager = TrackAttributeManager()
        track = Track(
            track_id=4,
            x1=10.0,
            y1=20.0,
            x2=70.0,
            y2=180.0,
            score=0.9,
            class_id=0,
        )
        profile = manager.update([track], 5, 166.7)[0]

        assert len(profile.trajectory) == 1
        point = profile.trajectory[0]
        assert point.frame_index == 5
        assert point.timestamp_ms == 166.7
        assert point.x == pytest.approx(40.0)
        assert point.y == pytest.approx(180.0)

    def test_observed_frames_append_real_points(self):
        manager = TrackAttributeManager()
        manager.update([_track(x1=10.0)], 0, 0.0)
        manager.update([_track(x1=20.0)], 1, 33.3)
        profile = manager.update([_track(x1=30.0)], 2, 66.7)[0]

        assert [point.frame_index for point in profile.trajectory] == [0, 1, 2]
        assert [point.timestamp_ms for point in profile.trajectory] == [
            0.0,
            33.3,
            66.7,
        ]
        assert [point.x for point in profile.trajectory] == [
            35.0,
            45.0,
            55.0,
        ]

    def test_missing_frames_do_not_add_synthetic_points(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        profile = manager.update([], 1, 33.3)[0]

        assert profile.lifecycle is TrackLifecycle.LOST
        assert len(profile.trajectory) == 1
        assert profile.trajectory[0].frame_index == 0

    def test_reactivated_track_appends_new_observation(self):
        manager = TrackAttributeManager()
        manager.update([_track(x1=10.0)], 0, 0.0)
        manager.update([], 1, 33.3)
        profile = manager.update([_track(x1=30.0)], 2, 66.7)[0]

        assert profile.lifecycle is TrackLifecycle.ACTIVE
        assert [point.frame_index for point in profile.trajectory] == [0, 2]
        assert profile.observed_frames == 2
        assert profile.age_frames == 3
        assert profile.missed_frames == 0

    def test_trajectory_keeps_only_most_recent_configured_points(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(max_trajectory_points=3)
        )
        for frame_index in range(1_000):
            manager.update(
                [_track(x1=float(frame_index))],
                frame_index,
                frame_index * 10.0,
            )

        profile = manager.get_profile(1)
        assert len(profile.trajectory) == 3
        assert [point.frame_index for point in profile.trajectory] == [
            997,
            998,
            999,
        ]
        assert profile.first_seen_frame_index == 0
        assert profile.last_seen_frame_index == 999
        assert profile.age_frames == 1_000
        assert profile.observed_frames == 1_000

    def test_trajectory_limit_is_applied_per_track(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(max_trajectory_points=2)
        )
        for frame_index in range(4):
            manager.update(
                [_track(1, x1=float(frame_index)), _track(2, x1=100.0)],
                frame_index,
                frame_index * 10.0,
            )

        assert len(manager.get_profile(1).trajectory) == 2
        assert len(manager.get_profile(2).trajectory) == 2

    def test_profile_snapshot_is_detached_from_future_updates(self):
        manager = TrackAttributeManager()
        old_profile = manager.update([_track()], 0, 0.0)[0]
        manager.update([_track()], 1, 33.3)

        assert len(old_profile.trajectory) == 1
        assert len(manager.get_profile(1).trajectory) == 2

    def test_reset_discards_trajectory(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.reset()
        profile = manager.update([_track()], 0, 0.0)[0]

        assert len(profile.trajectory) == 1
        assert profile.observed_frames == 1

    def test_invalid_coordinate_does_not_partially_mutate_manager(self):
        manager = TrackAttributeManager()
        manager.update([_track(1)], 0, 0.0)
        bad_track = _track(2, x1=float("nan"))

        with pytest.raises(ValueError):
            manager.update([bad_track], 1, 33.3)

        assert manager.get_profile(2) is None
        assert manager.get_profile(1).lifecycle is TrackLifecycle.ACTIVE
        assert manager.last_frame_index == 0
        assert manager.last_timestamp_ms == 0.0


class TestTtlCleanup:
    def test_cleanup_removes_state_at_ttl_boundary(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                max_missed_frames=1,
                removed_ttl_ms=100.0,
            )
        )
        manager.update([_track()], 0, 0.0)
        manager.update([], 1, 33.3)
        manager.update([], 2, 66.7)

        assert manager.cleanup(166.6) == []
        assert manager.cleanup(166.7) == [1]
        assert manager.get_profile(1) is None

    def test_update_automatically_cleans_expired_removed_state(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                max_missed_frames=1,
                removed_ttl_ms=50.0,
            )
        )
        manager.update([_track(1)], 0, 0.0)
        manager.update([], 1, 10.0)
        manager.update([], 2, 20.0)
        profiles = manager.update([_track(2)], 3, 70.0)

        assert [profile.track_id for profile in profiles] == [2]

    def test_expired_track_id_can_be_used_as_new_state(self):
        manager = TrackAttributeManager(
            TrackAttributeManagerConfig(
                max_missed_frames=1,
                removed_ttl_ms=50.0,
            )
        )
        manager.update([_track()], 0, 0.0)
        manager.update([], 1, 10.0)
        manager.update([], 2, 20.0)
        profile = manager.update([_track()], 3, 70.0)[0]

        assert profile.lifecycle is TrackLifecycle.ACTIVE
        assert profile.first_seen_frame_index == 3
        assert profile.observed_frames == 1

    def test_cleanup_rejects_timestamp_behind_manager_clock(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 100.0)
        with pytest.raises(ValueError, match="backward"):
            manager.cleanup(99.9)


class TestInputValidationAndReset:
    def test_rejects_duplicate_or_backward_frame_index(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 5, 100.0)

        with pytest.raises(ValueError, match="strictly increasing"):
            manager.update([_track()], 5, 100.0)
        with pytest.raises(ValueError, match="strictly increasing"):
            manager.update([_track()], 4, 101.0)

    def test_rejects_backward_timestamp(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 100.0)
        with pytest.raises(ValueError, match="backward"):
            manager.update([_track()], 1, 99.9)

    def test_equal_timestamp_on_new_frame_is_allowed(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 0, 0.0)
        manager.update([_track()], 1, 0.0)
        assert manager.get_profile(1).observed_frames == 2

    @pytest.mark.parametrize("frame_index", [-1, 1.5, True])
    def test_rejects_invalid_frame_index(self, frame_index):
        manager = TrackAttributeManager()
        with pytest.raises((TypeError, ValueError)):
            manager.update([], frame_index, 0.0)

    @pytest.mark.parametrize(
        "timestamp_ms",
        [-0.1, float("nan"), float("inf"), True],
    )
    def test_rejects_invalid_timestamp(self, timestamp_ms):
        manager = TrackAttributeManager()
        with pytest.raises((TypeError, ValueError)):
            manager.update([], 0, timestamp_ms)

    def test_rejects_duplicate_track_ids_without_mutating_state(self):
        manager = TrackAttributeManager()
        manager.update([_track(1)], 0, 0.0)

        with pytest.raises(ValueError, match="Duplicate"):
            manager.update([_track(2), _track(2)], 1, 33.3)

        assert manager.get_profile(2) is None
        assert manager.last_frame_index == 0
        assert manager.last_timestamp_ms == 0.0

    def test_rejects_non_track_input(self):
        manager = TrackAttributeManager()
        with pytest.raises(TypeError):
            manager.update(["not-a-track"], 0, 0.0)

    def test_reset_clears_state_and_timeline(self):
        manager = TrackAttributeManager()
        manager.update([_track()], 10, 500.0)
        manager.reset()

        assert len(manager) == 0
        assert manager.last_frame_index is None
        assert manager.last_timestamp_ms is None
        profile = manager.update([_track()], 0, 0.0)[0]
        assert profile.first_seen_frame_index == 0


def test_manager_module_has_no_model_or_database_imports():
    import src.track_attributes.manager as manager_module

    tree = ast.parse(inspect.getsource(manager_module))
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
    forbidden = (
        "torch",
        "ultralytics",
        "onnxruntime",
        "sqlite3",
        "sqlalchemy",
    )
    assert not any(
        module == name or module.startswith(f"{name}.")
        for module in imported_modules
        for name in forbidden
    )
