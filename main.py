"""
main.py — M3 pipeline orchestrator.

Reads data/input.mp4, runs YOLO11n person detection on every frame,
draws bounding boxes, and writes outputs/output.mp4.

Usage:
    python main.py
    python main.py --source data/input.mp4 --output outputs/output.mp4
    python main.py --model models/yolo11n.pt --confidence 0.25 --device cuda

Pipeline:
    VideoSource
        ↓  (frame_index, timestamp_ms, frame)
    YOLO11Detector
        ↓  List[Detection]
    draw_detections
        ↓  annotated frame
    cv2.VideoWriter
        ↓
    outputs/output.mp4

No benchmark, no tracking, no ONNX/TRT — M3 scope only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from src.detectors.yolov11 import YOLO11Detector
from src.video_source import VideoSource
from src.visualization import draw_detections_inplace

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_SOURCE = os.path.join("data", "input.mp4")
_DEFAULT_OUTPUT = os.path.join("outputs", "output.mp4")
_DEFAULT_MODEL = os.path.join("models", "yolo11n.pt")
_DEFAULT_CONFIDENCE = 0.25
_DEFAULT_DEVICE = None  # auto: cuda if available, else cpu

# Codec for output MP4 — mp4v is broadly readable on Linux
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person detection pipeline — M3"
    )
    parser.add_argument(
        "--source",
        default=_DEFAULT_SOURCE,
        help=f"Input MP4 path (default: {_DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Output MP4 path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"YOLO11n weights path (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=_DEFAULT_CONFIDENCE,
        help=f"Confidence threshold (default: {_DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--device",
        default=_DEFAULT_DEVICE,
        help="Torch device, e.g. 'cuda' or 'cpu' (default: auto)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _open_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Open a VideoWriter and validate it opened successfully.

    Raises
    ------
    RuntimeError
        If the writer cannot be opened.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, _FOURCC, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter could not open {output_path!r}. "
            f"codec=mp4v, size={width}x{height}, fps={fps}"
        )
    return writer


# ---------------------------------------------------------------------------
# Post-write verification (SKILL requirement)
# ---------------------------------------------------------------------------

def verify_output(output_path: str, expected_width: int, expected_height: int, expected_fps: float) -> dict:
    """Open the written video and verify metadata.  Returns a result dict.

    Raises
    ------
    RuntimeError
        If any check fails.
    """
    if not os.path.isfile(output_path):
        raise RuntimeError(f"Output file not found: {output_path!r}")

    cap = cv2.VideoCapture(output_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open {output_path!r}")

    try:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, _ = cap.read()
    finally:
        cap.release()

    errors = []
    if actual_w != expected_width:
        errors.append(f"width: expected {expected_width}, got {actual_w}")
    if actual_h != expected_height:
        errors.append(f"height: expected {expected_height}, got {actual_h}")
    if actual_fps <= 0:
        errors.append(f"fps invalid: {actual_fps}")
    if not ret:
        errors.append("could not read first frame from output video")

    if errors:
        raise RuntimeError("Output video verification failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return {
        "path": output_path,
        "width": actual_w,
        "height": actual_h,
        "fps": actual_fps,
        "frame_count": actual_fc,
        "first_frame_readable": ret,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    """Execute the full pipeline.  Returns a summary dict."""
    t_start = time.perf_counter()

    print(f"[M3] Source  : {args.source}")
    print(f"[M3] Output  : {args.output}")
    print(f"[M3] Model   : {args.model}")
    print(f"[M3] Conf    : {args.confidence}")
    print(f"[M3] Device  : {args.device or 'auto'}")
    print()

    # --- VideoSource -------------------------------------------------------
    source = VideoSource(args.source)
    print(
        f"[M3] Input   : {source.width}x{source.height} @ {source.fps} fps "
        f"({source.frame_count} frames)"
    )

    # --- Detector ----------------------------------------------------------
    detector = YOLO11Detector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
    )
    print(f"[M3] Detector: {detector}")
    print()

    # --- VideoWriter -------------------------------------------------------
    writer = _open_writer(
        args.output,
        width=source.width,
        height=source.height,
        fps=source.fps,
    )

    # --- Process frames ----------------------------------------------------
    frames_read = 0
    frames_written = 0
    total_detections = 0

    try:
        for frame_index, timestamp_ms, frame in source.frames():
            frames_read += 1

            # Detect
            detections = detector.predict(frame)
            total_detections += len(detections)

            # Annotate in-place (avoids an extra copy per frame)
            draw_detections_inplace(frame, detections)

            # Write
            writer.write(frame)
            frames_written += 1

            # Progress every 30 frames
            if frame_index % 30 == 0:
                elapsed = time.perf_counter() - t_start
                print(
                    f"  frame {frame_index:4d}/{source.frame_count}  "
                    f"dets={len(detections)}  "
                    f"elapsed={elapsed:.1f}s"
                )
    finally:
        writer.release()
        detector.release()
        source.release()

    elapsed_total = time.perf_counter() - t_start
    print()
    print(f"[M3] Done — {frames_written}/{frames_read} frames written in {elapsed_total:.2f}s")
    print(f"[M3] Total detections across all frames: {total_detections}")

    # --- Verify output video -----------------------------------------------
    print(f"\n[M3] Verifying output: {args.output} ...")
    info = verify_output(
        args.output,
        expected_width=source.width,
        expected_height=source.height,
        expected_fps=source.fps,
    )
    print(f"[M3] Output OK — {info['width']}x{info['height']} "
          f"@ {info['fps']} fps  frames={info['frame_count']}  "
          f"first_frame_readable={info['first_frame_readable']}")

    return {
        "source": args.source,
        "output": args.output,
        "frames_read": frames_read,
        "frames_written": frames_written,
        "total_detections": total_detections,
        "elapsed_s": elapsed_total,
        "output_info": info,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    try:
        summary = run(args)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
