"""
VideoSource — local MP4 reader.

Yields (frame_index, timestamp_ms, frame) tuples.
No RTSP, no webcam, no async, no threading.

Compatible contract for future TensorRT backend swap:
  VideoSource stays unchanged when detector changes.
"""

from __future__ import annotations

import os
from typing import Generator, Tuple

import cv2
import numpy as np


class VideoSourceError(RuntimeError):
    """Raised when VideoSource cannot open or validate the video."""


class VideoSource:
    """Reads a local MP4 file frame-by-frame.

    Parameters
    ----------
    path : str
        Absolute or relative path to the MP4 file.

    Raises
    ------
    VideoSourceError
        If the file does not exist, cannot be opened by OpenCV,
        or has invalid FPS / resolution metadata.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap: cv2.VideoCapture | None = None
        self._validate_and_open()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        """Total frames reported by the container (may differ from actual)."""
        return self._frame_count

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_and_open(self) -> None:
        """Check file, open capture, validate metadata."""
        # 1. File existence
        if not os.path.isfile(self._path):
            raise VideoSourceError(f"File not found: {self._path!r}")

        # 2. Open with OpenCV
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            cap.release()
            raise VideoSourceError(f"OpenCV could not open: {self._path!r}")

        # 3. FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            raise VideoSourceError(
                f"Invalid FPS ({fps}) in {self._path!r}. "
                "Video metadata may be corrupt."
            )

        # 4. Width / Height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise VideoSourceError(
                f"Invalid resolution ({width}x{height}) in {self._path!r}."
            )

        # 5. First frame readability
        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise VideoSourceError(
                f"Could not read first frame from {self._path!r}."
            )

        # Reset to beginning after the probe read
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Store validated state
        self._fps = fps
        self._width = width
        self._height = height
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cap = cap

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield ``(frame_index, timestamp_ms, frame)`` for every frame.

        The generator resets the capture to the beginning on each call,
        so it can be iterated more than once (on the same open VideoSource).

        Yields
        ------
        frame_index : int
            Zero-based frame counter.
        timestamp_ms : float
            Presentation timestamp in milliseconds (from container PTS).
            Falls back to ``frame_index / fps * 1000`` if PTS is missing.
        frame : np.ndarray
            BGR image array with shape (height, width, 3), dtype uint8.
        """
        if self._cap is None:
            raise VideoSourceError("VideoSource has already been released.")

        # Rewind to the first frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_index = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            # CAP_PROP_POS_MSEC is the PTS of the *last read* frame
            pts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if pts_ms <= 0 and frame_index > 0:
                # Fallback: compute from index
                pts_ms = frame_index / self._fps * 1000.0

            yield frame_index, pts_ms, frame
            frame_index += 1

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release the underlying VideoCapture.  Safe to call multiple times."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoSource(path={self._path!r}, "
            f"fps={self._fps}, "
            f"resolution={self._width}x{self._height}, "
            f"frames={self._frame_count})"
        )
