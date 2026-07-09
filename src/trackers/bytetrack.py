"""
YOLO11n person tracker — ByteTrack backend (ultralytics BYTETracker).

This module is the ONLY place where ultralytics.trackers is imported.
No BYTETracker, STrack, or internal ultralytics tracking objects may
escape this module's boundary.

Data flow inside this module:
    list[Detection]
        → _DetectionResults (duck-typed wrapper for BYTETracker.update)
        → BYTETracker.update(results, img) → np.ndarray  shape (N, 8)
        → _parse_output(arr) → list[Track]

Output array column layout (from STrack.result):
    [x1, y1, x2, y2, track_id, score, cls, det_idx]
    col:  0   1   2   3     4      5    6       7

Future swap path:
    ByteTrackTracker (ultralytics BYTETracker)
    → YOLO11TensorRTTracker (custom TRT backend)
    Both implement BaseTracker.  Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.schemas import Detection, Track
from src.trackers.base import BaseTracker

# COCO person class id
_PERSON_CLASS_ID: int = 0

# ────────────────────────────────────────────────────────────
# Tracker configuration (single place, no rogue hard-codes)
# ────────────────────────────────────────────────────────────


@dataclass
class ByteTrackConfig:
    """Baseline ByteTrack configuration.

    Parameters mirror BYTETracker args as documented in ultralytics:
        https://docs.ultralytics.com/reference/trackers/byte_tracker/

    Do NOT tune during M8 unless the tracker is completely broken.
    """

    # Confidence thresholds
    track_high_thresh: float = 0.25   # match M2 detection baseline
    track_low_thresh: float = 0.10    # second-association low-score detections
    new_track_thresh: float = 0.25    # minimum score to initialise a new track

    # Association
    match_thresh: float = 0.80        # max IoU distance for first-stage match
    fuse_score: bool = False          # do not fuse score into IoU cost

    # Lifespan
    track_buffer: int = 30            # frames to hold a lost track (= 1 s at 30 fps)
    frame_rate: int = 30              # expected frame rate of input video


# ────────────────────────────────────────────────────────────
# Duck-typed Results wrapper (internal only)
# ────────────────────────────────────────────────────────────


class _DetectionResults:
    """Minimal duck-typed object that satisfies BYTETracker.update(results).

    BYTETracker.update calls:
    1. _split_detections(results) which uses results.conf and results[mask]
    2. init_track(results) which calls parse_bboxes(results) reading
       results.xywh, results.conf, results.cls

    This class converts list[Detection] → those arrays plus __getitem__.
    It does NOT expose any ultralytics internals.
    """

    def __init__(self, detections: List[Detection]) -> None:
        n = len(detections)
        if n == 0:
            self.xywh = np.empty((0, 4), dtype=np.float32)
            self.conf = np.empty((0,), dtype=np.float32)
            self.cls = np.empty((0,), dtype=np.float32)
        else:
            # Convert x1y1x2y2 → cx cy w h  (BYTETracker expects xywh)
            x1 = np.array([d.x1 for d in detections], dtype=np.float32)
            y1 = np.array([d.y1 for d in detections], dtype=np.float32)
            x2 = np.array([d.x2 for d in detections], dtype=np.float32)
            y2 = np.array([d.y2 for d in detections], dtype=np.float32)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = x2 - x1
            h  = y2 - y1
            self.xywh = np.stack([cx, cy, w, h], axis=1)   # (N, 4)
            self.conf = np.array([d.score for d in detections], dtype=np.float32)
            self.cls  = np.array([d.class_id for d in detections], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask):
        """Support boolean/integer array indexing used by BYTETracker._split_detections."""
        sub = _DetectionResults.__new__(_DetectionResults)
        sub.xywh = self.xywh[mask]
        sub.conf = self.conf[mask]
        sub.cls  = self.cls[mask]
        return sub


# ────────────────────────────────────────────────────────────
# Output parser (internal only)
# ────────────────────────────────────────────────────────────


def _parse_output(arr: np.ndarray) -> List[Track]:
    """Convert BYTETracker output array to list[Track].

    Column layout (STrack.result):
        0   x1
        1   y1
        2   x2
        3   y2
        4   track_id
        5   score
        6   cls        (float, e.g. 0.0 for person)
        7   det_idx    (original detection index — discarded)

    Only rows where cls == _PERSON_CLASS_ID are kept.
    (ByteTrack in our pipeline only receives person detections,
    but this guard is defensive.)
    """
    if arr is None or len(arr) == 0:
        return []

    tracks: List[Track] = []
    for row in arr:
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        track_id = int(row[4])
        score    = float(row[5])
        cls      = int(row[6])

        if cls != _PERSON_CLASS_ID:
            continue   # defensive filter — should never trigger in M8

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Clamp score to [0, 1] — ByteTrack can occasionally return tiny
        # values slightly above 1.0 due to Kalman smoothing
        score = min(max(score, 0.0), 1.0)

        tracks.append(
            Track(
                track_id=track_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=score,
                class_id=cls,
            )
        )
    return tracks


# ────────────────────────────────────────────────────────────
# ByteTrackTracker (public)
# ────────────────────────────────────────────────────────────


class ByteTrackTracker(BaseTracker):
    """Person tracker backed by ultralytics BYTETracker.

    Parameters
    ----------
    config : ByteTrackConfig or None
        Tracker hyper-parameters.  If None, defaults are used.

    Raises
    ------
    ImportError
        If ``ultralytics`` is not installed.

    Notes
    -----
    BYTETracker internal objects (STrack, KalmanFilter, etc.) are
    fully contained within this module.  Only list[Track] exits.
    """

    def __init__(self, config: Optional[ByteTrackConfig] = None) -> None:
        self._config = config or ByteTrackConfig()
        self._tracker = self._build_tracker()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tracker(self):
        """Instantiate BYTETracker with our config namespace."""
        from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: PLC0415

        # BYTETracker expects an `args` namespace with attribute access.
        # We create a simple object from our config dataclass.
        args = _ConfigNamespace(self._config)
        return BYTETracker(args=args)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Update tracker and return active tracks for this frame.

        Parameters
        ----------
        detections : List[Detection]
            Person-only detections from the current frame (may be empty).
        frame : np.ndarray or None
            BGR image.  BYTETracker doesn't use it; accepted for interface
            compatibility with appearance-based trackers.

        Returns
        -------
        List[Track]
            Active confirmed tracks.  Coordinates in original frame pixels.
        """
        results = _DetectionResults(detections)
        raw = self._tracker.update(results, img=frame)
        return _parse_output(raw)

    def reset(self) -> None:
        """Reset tracker state — call before processing a new video."""
        self._tracker.reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> ByteTrackConfig:
        """Return current tracker configuration."""
        return self._config

    def __repr__(self) -> str:
        cfg = self._config
        return (
            f"ByteTrackTracker("
            f"high={cfg.track_high_thresh}, "
            f"low={cfg.track_low_thresh}, "
            f"new={cfg.new_track_thresh}, "
            f"match={cfg.match_thresh}, "
            f"buffer={cfg.track_buffer}f, "
            f"fps={cfg.frame_rate})"
        )


# ────────────────────────────────────────────────────────────
# Config bridge (internal only)
# ────────────────────────────────────────────────────────────


class _ConfigNamespace:
    """Convert ByteTrackConfig dataclass → attribute-access namespace.

    BYTETracker reads its config via args.track_high_thresh etc.
    This class provides that interface without leaking our dataclass.
    """

    def __init__(self, cfg: ByteTrackConfig) -> None:
        self.track_high_thresh = cfg.track_high_thresh
        self.track_low_thresh  = cfg.track_low_thresh
        self.new_track_thresh  = cfg.new_track_thresh
        self.match_thresh      = cfg.match_thresh
        self.fuse_score        = cfg.fuse_score
        self.track_buffer      = cfg.track_buffer
        self.frame_rate        = cfg.frame_rate
