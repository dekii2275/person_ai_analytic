"""
Visualization — draw person bounding boxes on a frame.

Dependencies: opencv-python, numpy only.
No ultralytics, no torch, no model objects.

Public API:
    draw_detections(frame, detections) -> np.ndarray   (copy)
    draw_detections_inplace(frame, detections)          (in-place)
    draw_tracks(frame, tracks)         -> np.ndarray   (copy)
    draw_tracks_inplace(frame, tracks)                  (in-place)
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.schemas import Detection, Track


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

_BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)       # green (BGR)
_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)   # white (BGR)
_LABEL_BG_COLOR: Tuple[int, int, int] = (0, 180, 0)  # darker green (BGR)
_BOX_THICKNESS: int = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE: float = 0.55
_FONT_THICKNESS: int = 1
_LABEL_PADDING: int = 4   # pixels around text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    """Return a new frame with bounding boxes and confidence scores drawn.

    Parameters
    ----------
    frame : np.ndarray
        BGR image, shape (H, W, 3), dtype uint8.  Not modified.
    detections : List[Detection]
        Detections to draw.  Empty list is allowed (returns copy unchanged).

    Returns
    -------
    np.ndarray
        Copy of ``frame`` with annotations overlaid.
    """
    out = frame.copy()
    draw_detections_inplace(out, detections)
    return out


def draw_detections_inplace(
    frame: np.ndarray,
    detections: List[Detection],
) -> None:
    """Draw bounding boxes and confidence scores directly onto ``frame``.

    Parameters
    ----------
    frame : np.ndarray
        BGR image, shape (H, W, 3), dtype uint8.  Modified in place.
    detections : List[Detection]
        Detections to draw.
    """
    for det in detections:
        _draw_single(frame, det)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _draw_single(frame: np.ndarray, det: Detection) -> None:
    """Draw one Detection onto ``frame`` in place."""
    h, w = frame.shape[:2]

    # Clamp coordinates to frame bounds
    x1 = int(max(0, min(det.x1, w - 1)))
    y1 = int(max(0, min(det.y1, h - 1)))
    x2 = int(max(0, min(det.x2, w - 1)))
    y2 = int(max(0, min(det.y2, h - 1)))

    # Skip degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)

    # Label: "person 0.956"
    label = f"person {det.score:.3f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, _FONT, _FONT_SCALE, _FONT_THICKNESS
    )

    # Position label above the top-left corner; flip below if out of frame
    label_y1 = y1 - text_h - baseline - 2 * _LABEL_PADDING
    label_y2 = y1
    if label_y1 < 0:
        label_y1 = y2
        label_y2 = y2 + text_h + baseline + 2 * _LABEL_PADDING

    label_x2 = min(x1 + text_w + 2 * _LABEL_PADDING, w - 1)

    # Filled background rectangle
    cv2.rectangle(
        frame,
        (x1, label_y1),
        (label_x2, label_y2),
        _LABEL_BG_COLOR,
        cv2.FILLED,
    )

    # Text
    text_y = label_y2 - _LABEL_PADDING - baseline
    cv2.putText(
        frame,
        label,
        (x1 + _LABEL_PADDING, text_y),
        _FONT,
        _FONT_SCALE,
        _TEXT_COLOR,
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Track visualization (M8)
# ---------------------------------------------------------------------------

# Distinct color palette for up to 32 track IDs; cycles after that.
# Colors are BGR.  We use a fixed HSV rainbow so each ID maps to a unique hue.
_TRACK_PALETTE_SIZE: int = 32


def _track_color(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR color for the given track_id."""
    hue = int((track_id % _TRACK_PALETTE_SIZE) * (180 / _TRACK_PALETTE_SIZE))
    hsv = np.array([[[hue, 220, 230]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
) -> np.ndarray:
    """Return a new frame with track bounding boxes and IDs drawn.

    Parameters
    ----------
    frame : np.ndarray
        BGR image, shape (H, W, 3), dtype uint8.  Not modified.
    tracks : List[Track]
        Active tracks to draw.  Empty list is allowed (returns copy unchanged).

    Returns
    -------
    np.ndarray
        Copy of ``frame`` with track annotations overlaid.
    """
    out = frame.copy()
    draw_tracks_inplace(out, tracks)
    return out


def draw_tracks_inplace(
    frame: np.ndarray,
    tracks: List[Track],
) -> None:
    """Draw track bounding boxes and IDs directly onto ``frame``.

    Parameters
    ----------
    frame : np.ndarray
        BGR image, shape (H, W, 3), dtype uint8.  Modified in place.
    tracks : List[Track]
        Active tracks to draw.
    """
    for track in tracks:
        _draw_single_track(frame, track)


def _draw_single_track(frame: np.ndarray, track: Track) -> None:
    """Draw one Track onto ``frame`` in place."""
    h, w = frame.shape[:2]

    # Clamp coordinates to frame bounds
    x1 = int(max(0, min(track.x1, w - 1)))
    y1 = int(max(0, min(track.y1, h - 1)))
    x2 = int(max(0, min(track.x2, w - 1)))
    y2 = int(max(0, min(track.y2, h - 1)))

    # Skip degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return

    color = _track_color(track.track_id)

    # Bounding box (slightly thicker than detection boxes for visual distinction)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, _BOX_THICKNESS + 1)

    # Label: "ID 1 | 0.90"
    label = f"ID {track.track_id} | {track.score:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, _FONT, _FONT_SCALE, _FONT_THICKNESS
    )

    # Position label above top-left; flip below if out of frame
    label_y1 = y1 - text_h - baseline - 2 * _LABEL_PADDING
    label_y2 = y1
    if label_y1 < 0:
        label_y1 = y2
        label_y2 = y2 + text_h + baseline + 2 * _LABEL_PADDING

    label_x2 = min(x1 + text_w + 2 * _LABEL_PADDING, w - 1)

    # Filled background rectangle (same color, darkened)
    bg = tuple(max(0, c - 60) for c in color)
    cv2.rectangle(
        frame,
        (x1, label_y1),
        (label_x2, label_y2),
        bg,
        cv2.FILLED,
    )

    # Text
    text_y = label_y2 - _LABEL_PADDING - baseline
    cv2.putText(
        frame,
        label,
        (x1 + _LABEL_PADDING, text_y),
        _FONT,
        _FONT_SCALE,
        _TEXT_COLOR,
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )
