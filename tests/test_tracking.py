"""
M8 tracking tests.

Tests cover:
  - Track schema
  - BaseTracker interface
  - ByteTrackTracker behaviour
  - draw_tracks visualization
  - Pipeline integration (smoke: a few real frames)
  - Output video existence + metadata
  - Tracking statistics JSON structure
"""

from __future__ import annotations

import json
import os
import time
from typing import List

import cv2
import numpy as np
import pytest

from src.schemas import Detection, Track
from src.trackers.base import BaseTracker
from src.trackers.bytetrack import ByteTrackConfig, ByteTrackTracker
from src.visualization import draw_tracks, draw_tracks_inplace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(
    x1=50.0, y1=100.0, x2=200.0, y2=400.0,
    score=0.85, class_id=0
) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)


def _make_track(
    track_id=1, x1=50.0, y1=100.0, x2=200.0, y2=400.0,
    score=0.85, class_id=0
) -> Track:
    return Track(
        track_id=track_id, x1=x1, y1=y1, x2=x2, y2=y2,
        score=score, class_id=class_id
    )


def _blank_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _warm_tracker(tracker: ByteTrackTracker, det: Detection, frames: int = 3) -> List[Track]:
    """Run the tracker for `frames` frames to confirm a track."""
    last = []
    for _ in range(frames):
        last = tracker.update([det])
    return last


# ---------------------------------------------------------------------------
# Track schema tests
# ---------------------------------------------------------------------------

class TestTrackSchema:
    def test_creates_track(self):
        t = _make_track()
        assert t.track_id == 1

    def test_bbox_accessible(self):
        t = _make_track(x1=10, y1=20, x2=100, y2=300)
        assert t.x1 == 10
        assert t.y1 == 20
        assert t.x2 == 100
        assert t.y2 == 300

    def test_score_in_range(self):
        t = _make_track(score=0.5)
        assert 0.0 <= t.score <= 1.0

    def test_track_id_is_int(self):
        t = _make_track()
        assert isinstance(t.track_id, int)

    def test_track_id_zero_valid(self):
        t = _make_track(track_id=0)
        assert t.track_id == 0

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError):
            Track(track_id=1, x1=0, y1=0, x2=10, y2=10, score=1.5, class_id=0)

    def test_invalid_class_id_raises(self):
        with pytest.raises(ValueError):
            Track(track_id=1, x1=0, y1=0, x2=10, y2=10, score=0.5, class_id=-1)

    def test_invalid_track_id_raises(self):
        with pytest.raises(ValueError):
            Track(track_id=-1, x1=0, y1=0, x2=10, y2=10, score=0.5, class_id=0)

    def test_width_height_area(self):
        t = _make_track(x1=10, y1=20, x2=110, y2=220)
        assert t.width == pytest.approx(100.0)
        assert t.height == pytest.approx(200.0)
        assert t.area == pytest.approx(20000.0)

    def test_repr_contains_id(self):
        t = _make_track(track_id=42)
        assert "42" in repr(t)

    def test_class_id_is_person(self):
        t = _make_track(class_id=0)
        assert t.class_id == 0


# ---------------------------------------------------------------------------
# BaseTracker interface tests
# ---------------------------------------------------------------------------

class TestBaseTrackerInterface:
    """Verify the ABC contract via ByteTrackTracker as a concrete subclass."""

    def test_is_subclass_of_base(self):
        tracker = ByteTrackTracker()
        assert isinstance(tracker, BaseTracker)

    def test_has_update_method(self):
        tracker = ByteTrackTracker()
        assert callable(tracker.update)

    def test_has_reset_method(self):
        tracker = ByteTrackTracker()
        assert callable(tracker.reset)

    def test_context_manager_calls_reset(self):
        """__exit__ should reset without error."""
        with ByteTrackTracker() as tracker:
            _warm_tracker(tracker, _make_detection())
        # After context exits, tracker is reset (no crash expected)

    def test_update_returns_list(self):
        tracker = ByteTrackTracker()
        result = tracker.update([])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ByteTrackTracker tests
# ---------------------------------------------------------------------------

class TestByteTrackTracker:
    def test_empty_detections_no_crash(self):
        tracker = ByteTrackTracker()
        tracks = tracker.update([])
        assert tracks == []

    def test_empty_detections_return_list(self):
        tracker = ByteTrackTracker()
        result = tracker.update([])
        assert isinstance(result, list)

    def test_single_detection_creates_track_after_warmup(self):
        tracker = ByteTrackTracker()
        det = _make_detection()
        tracks = _warm_tracker(tracker, det, frames=3)
        assert len(tracks) >= 1

    def test_track_id_is_positive_int(self):
        tracker = ByteTrackTracker()
        det = _make_detection()
        tracks = _warm_tracker(tracker, det, frames=3)
        if tracks:
            assert isinstance(tracks[0].track_id, int)
            assert tracks[0].track_id > 0

    def test_output_is_list_of_track(self):
        tracker = ByteTrackTracker()
        det = _make_detection()
        tracks = _warm_tracker(tracker, det, frames=3)
        for t in tracks:
            assert isinstance(t, Track)

    def test_no_backend_object_leaked(self):
        """Ensure no ultralytics STrack or BYTETracker in output."""
        tracker = ByteTrackTracker()
        det = _make_detection()
        tracks = _warm_tracker(tracker, det, frames=3)
        for t in tracks:
            assert type(t).__module__.startswith("src"), (
                f"Backend object leaked: {type(t)}"
            )

    def test_only_person_class_in_output(self):
        """Track.class_id must be 0 (person) for all tracks."""
        tracker = ByteTrackTracker()
        dets = [_make_detection(class_id=0) for _ in range(3)]
        for _ in range(3):
            tracks = tracker.update(dets)
        for t in tracks:
            assert t.class_id == 0, f"Non-person class leaked: {t}"

    def test_multiple_frames_return_tracks(self):
        tracker = ByteTrackTracker()
        det1 = _make_detection(x1=10, y1=10, x2=100, y2=200)
        det2 = _make_detection(x1=300, y1=100, x2=450, y2=400)
        all_ids = set()
        for _ in range(5):
            tracks = tracker.update([det1, det2])
            for t in tracks:
                all_ids.add(t.track_id)
        assert len(all_ids) >= 1

    def test_stable_track_id_across_frames(self):
        """Same detection region should keep the same ID across frames."""
        tracker = ByteTrackTracker()
        det = _make_detection()
        seen_ids = []
        for _ in range(5):
            tracks = tracker.update([det])
            for t in tracks:
                seen_ids.append(t.track_id)
        if seen_ids:
            assert len(set(seen_ids)) == 1, "Track ID changed across frames"

    def test_reset_clears_state(self):
        tracker = ByteTrackTracker()
        det = _make_detection()
        _warm_tracker(tracker, det, frames=3)
        tracker.reset()
        # After reset, frame 1 should produce unconfirmed track (empty or len 1)
        tracks_after = tracker.update([det])
        # ByteTrack needs 2 frames to confirm — frame 1 after reset may return empty
        assert isinstance(tracks_after, list)

    def test_score_clamped_to_0_1(self):
        tracker = ByteTrackTracker()
        det = _make_detection(score=0.99)
        tracks = _warm_tracker(tracker, det, frames=3)
        for t in tracks:
            assert 0.0 <= t.score <= 1.0

    def test_bbox_in_frame_coordinates(self):
        tracker = ByteTrackTracker()
        det = _make_detection(x1=100, y1=200, x2=300, y2=600)
        tracks = _warm_tracker(tracker, det, frames=3)
        for t in tracks:
            assert t.x1 >= 0
            assert t.y1 >= 0
            assert t.x2 > t.x1
            assert t.y2 > t.y1

    def test_config_respected(self):
        cfg = ByteTrackConfig(track_high_thresh=0.3, track_buffer=15)
        tracker = ByteTrackTracker(config=cfg)
        assert tracker.config.track_high_thresh == 0.3
        assert tracker.config.track_buffer == 15

    def test_repr_contains_config(self):
        tracker = ByteTrackTracker()
        r = repr(tracker)
        assert "ByteTrackTracker" in r


# ---------------------------------------------------------------------------
# draw_tracks visualization tests
# ---------------------------------------------------------------------------

class TestDrawTracks:
    def test_returns_ndarray(self):
        frame = _blank_frame()
        tracks = [_make_track()]
        out = draw_tracks(frame, tracks)
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self):
        frame = _blank_frame(480, 640)
        out = draw_tracks(frame, [_make_track()])
        assert out.shape == frame.shape

    def test_dtype_preserved(self):
        frame = _blank_frame()
        out = draw_tracks(frame, [_make_track()])
        assert out.dtype == np.uint8

    def test_original_not_modified(self):
        frame = _blank_frame()
        original = frame.copy()
        draw_tracks(frame, [_make_track()])
        np.testing.assert_array_equal(frame, original)

    def test_empty_tracks_returns_copy_unchanged(self):
        frame = _blank_frame()
        original = frame.copy()
        out = draw_tracks(frame, [])
        np.testing.assert_array_equal(out, original)

    def test_annotated_differs_from_original_when_track_present(self):
        frame = _blank_frame()
        out = draw_tracks(frame, [_make_track(x1=50, y1=100, x2=200, y2=400)])
        assert not np.array_equal(frame, out), "Annotated frame should differ"

    def test_multiple_tracks_no_crash(self):
        frame = _blank_frame()
        tracks = [_make_track(track_id=i, x1=i*50, y1=50, x2=i*50+40, y2=200)
                  for i in range(1, 5)]
        out = draw_tracks(frame, tracks)
        assert out.shape == frame.shape

    def test_track_at_frame_edge_no_crash(self):
        frame = _blank_frame(100, 100)
        track = _make_track(x1=-10, y1=-10, x2=110, y2=110)
        out = draw_tracks(frame, [track])
        assert out.shape == (100, 100, 3)

    def test_draw_tracks_inplace_modifies_frame(self):
        frame = _blank_frame()
        track = _make_track(x1=50, y1=100, x2=200, y2=400)
        original = frame.copy()
        draw_tracks_inplace(frame, [track])
        assert not np.array_equal(frame, original)

    def test_draw_tracks_inplace_returns_none(self):
        frame = _blank_frame()
        result = draw_tracks_inplace(frame, [_make_track()])
        assert result is None

    def test_draw_tracks_inplace_empty_no_change(self):
        frame = _blank_frame()
        original = frame.copy()
        draw_tracks_inplace(frame, [])
        np.testing.assert_array_equal(frame, original)

    def test_visualization_does_not_import_ultralytics(self):
        import src.visualization as vis
        import sys
        assert "ultralytics" not in sys.modules or True  # noqa: just import check
        # We specifically check the module doesn't directly import ultralytics
        import importlib
        import inspect
        src_text = inspect.getsource(vis)
        assert "ultralytics" not in src_text


# ---------------------------------------------------------------------------
# Pipeline integration test (smoke — real frames from input.mp4)
# ---------------------------------------------------------------------------

class TestTrackingPipelineSmoke:
    """Run the tracking pipeline on a small set of real frames.

    Requires data/input.mp4 and models/yolo11n.pt.
    """

    VIDEO_PATH = os.path.join("data", "input.mp4")
    MODEL_PATH = os.path.join("models", "yolo11n.pt")
    SMOKE_FRAMES = 10

    def _available(self) -> bool:
        return (
            os.path.isfile(self.VIDEO_PATH)
            and os.path.isfile(self.MODEL_PATH)
        )

    def test_pipeline_runs_without_crash(self):
        if not self._available():
            pytest.skip("data/input.mp4 or models/yolo11n.pt not found")

        from src.detectors.yolov11 import YOLO11Detector
        from src.video_source import VideoSource

        source = VideoSource(self.VIDEO_PATH)
        detector = YOLO11Detector(model_path=self.MODEL_PATH, confidence=0.25)
        tracker = ByteTrackTracker()

        n_frames = 0
        try:
            for frame_index, _ts, frame in source.frames():
                dets = detector.predict(frame)
                tracks = tracker.update(dets)
                assert isinstance(tracks, list)
                for t in tracks:
                    assert isinstance(t, Track)
                n_frames += 1
                if n_frames >= self.SMOKE_FRAMES:
                    break
        finally:
            source.release()
            detector.release()
            tracker.reset()

        assert n_frames == self.SMOKE_FRAMES

    def test_draw_tracks_on_real_frame(self):
        if not self._available():
            pytest.skip("data/input.mp4 not found")

        from src.video_source import VideoSource

        source = VideoSource(self.VIDEO_PATH)
        try:
            for _idx, _ts, frame in source.frames():
                track = _make_track(x1=10, y1=10, x2=100, y2=200)
                out = draw_tracks(frame, [track])
                assert out.shape == frame.shape
                break
        finally:
            source.release()


# ---------------------------------------------------------------------------
# Output artifacts tests
# ---------------------------------------------------------------------------

class TestTrackingOutputArtifacts:
    """Tests that run ONLY if outputs/tracking_output.mp4 exists."""

    VIDEO_PATH = os.path.join("outputs", "tracking_output.mp4")
    SUMMARY_PATH = os.path.join("outputs", "tracking_summary.json")
    TRACKS_PATH = os.path.join("outputs", "tracking_tracks.json")

    def _skip_if_missing(self, path: str) -> None:
        if not os.path.isfile(path):
            pytest.skip(f"{path} not yet generated — run main.py --tracking first")

    def test_tracking_video_exists(self):
        self._skip_if_missing(self.VIDEO_PATH)
        assert os.path.isfile(self.VIDEO_PATH)

    def test_tracking_video_nonzero_size(self):
        self._skip_if_missing(self.VIDEO_PATH)
        assert os.path.getsize(self.VIDEO_PATH) > 1000

    def test_tracking_video_openable(self):
        self._skip_if_missing(self.VIDEO_PATH)
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        assert cap.isOpened()
        cap.release()

    def test_tracking_video_first_frame_readable(self):
        self._skip_if_missing(self.VIDEO_PATH)
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        ret, _ = cap.read()
        cap.release()
        assert ret, "First frame not readable"

    def test_tracking_video_frame_count(self):
        self._skip_if_missing(self.VIDEO_PATH)
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert fc >= 250, f"Expected ~259 frames, got {fc}"

    def test_tracking_video_resolution(self):
        self._skip_if_missing(self.VIDEO_PATH)
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert w == 576, f"Expected width=576, got {w}"
        assert h == 1024, f"Expected height=1024, got {h}"

    def test_tracking_video_fps(self):
        self._skip_if_missing(self.VIDEO_PATH)
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert fps > 0, f"Invalid FPS: {fps}"

    def test_summary_json_exists(self):
        self._skip_if_missing(self.SUMMARY_PATH)
        assert os.path.isfile(self.SUMMARY_PATH)

    def test_summary_json_parseable(self):
        self._skip_if_missing(self.SUMMARY_PATH)
        with open(self.SUMMARY_PATH) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_summary_has_metadata(self):
        self._skip_if_missing(self.SUMMARY_PATH)
        with open(self.SUMMARY_PATH) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "detector" in data["metadata"]
        assert "tracker" in data["metadata"]

    def test_summary_has_summary_section(self):
        self._skip_if_missing(self.SUMMARY_PATH)
        with open(self.SUMMARY_PATH) as f:
            data = json.load(f)
        s = data["summary"]
        assert "processed_frames" in s
        assert "total_unique_track_ids" in s
        assert "max_concurrent_tracks" in s
        assert "mean_active_tracks" in s
        assert "effective_fps" in s

    def test_summary_processed_frames_correct(self):
        self._skip_if_missing(self.SUMMARY_PATH)
        with open(self.SUMMARY_PATH) as f:
            data = json.load(f)
        assert data["summary"]["processed_frames"] == 259

    def test_summary_has_latency_section(self):
        self._skip_if_missing(self.SUMMARY_PATH)
        with open(self.SUMMARY_PATH) as f:
            data = json.load(f)
        assert "latency" in data
        assert "tracker_update" in data["latency"]

    def test_summary_unique_track_ids_positive(self):
        self._skip_if_missing(self.SUMMARY_PATH)
        with open(self.SUMMARY_PATH) as f:
            data = json.load(f)
        assert data["summary"]["total_unique_track_ids"] > 0

    def test_tracks_json_exists(self):
        self._skip_if_missing(self.TRACKS_PATH)
        assert os.path.isfile(self.TRACKS_PATH)

    def test_tracks_json_parseable(self):
        self._skip_if_missing(self.TRACKS_PATH)
        with open(self.TRACKS_PATH) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_tracks_json_correct_frame_count(self):
        self._skip_if_missing(self.TRACKS_PATH)
        with open(self.TRACKS_PATH) as f:
            data = json.load(f)
        assert len(data) == 259

    def test_tracks_json_frame_structure(self):
        self._skip_if_missing(self.TRACKS_PATH)
        with open(self.TRACKS_PATH) as f:
            data = json.load(f)
        for entry in data[:10]:  # check first 10 frames
            assert "frame_index" in entry
            assert "tracks" in entry
            assert isinstance(entry["tracks"], list)

    def test_tracks_json_track_entry_structure(self):
        self._skip_if_missing(self.TRACKS_PATH)
        with open(self.TRACKS_PATH) as f:
            data = json.load(f)
        # find a frame with tracks
        for entry in data:
            if entry["tracks"]:
                t = entry["tracks"][0]
                assert "track_id" in t
                assert "bbox" in t
                assert "score" in t
                assert len(t["bbox"]) == 4
                break
