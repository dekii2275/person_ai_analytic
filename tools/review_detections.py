"""
tools/review_detections.py — Visual review tool for M5.

Generates annotated JPEG frames and a CSV report for systematic
review of YOLO11n detection quality on a given video.

Does NOT:
  - Make automatic pass/fail judgements (no ground truth).
  - Add tracking.
  - Export ONNX.

Usage:
    python tools/review_detections.py
    python tools/review_detections.py \\
        --source data/input.mp4 \\
        --output-dir outputs/review \\
        --model models/yolo11n.pt \\
        --step 10 \\
        --extra-frames 0 30 60 79 90 120 155 180 196 210 240 258

The script:
  1. Reads the video with VideoSource.
  2. Re-runs YOLO11Detector on each selected frame.
  3. Draws bounding boxes + confidence scores.
  4. Stamps frame index + detection count on the image.
  5. Saves as outputs/review/frame_NNNNNN.jpg
  6. Writes outputs/review/review_report.csv

CSV columns:
  frame_index, timestamp_ms, num_detections, max_confidence,
  review_category, issue_type, severity, note

review_category is auto-assigned from detection count only.
All other columns (issue_type, severity, note) default to values
that reviewers fill in manually — this script does NOT auto-classify.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

# Allow import from project root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.detectors.yolov11 import YOLO11Detector
from src.schemas import Detection
from src.video_source import VideoSource
from src.visualization import draw_detections

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_SOURCE = os.path.join("data", "input.mp4")
_DEFAULT_OUTPUT_DIR = os.path.join("outputs", "review")
_DEFAULT_MODEL = os.path.join("models", "yolo11n.pt")
_DEFAULT_CONFIDENCE = 0.25
_DEFAULT_STEP = 30          # save every N-th frame by default
_DEFAULT_DEVICE = None      # auto

# Mandatory frames from M2 smoke tests + interesting ones from benchmark
_MANDATORY_FRAMES = [0, 30, 60, 120, 180, 258]

# Extra frames of interest discovered from benchmark analysis
_BENCHMARK_EXTRA = [79, 90, 155, 196, 210, 240]

# JPEG quality
_JPEG_QUALITY = 92

# Overlay font
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# CSV column names
_CSV_COLUMNS = [
    "frame_index",
    "timestamp_ms",
    "num_detections",
    "max_confidence",
    "review_category",
    "issue_type",
    "severity",
    "note",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual review tool for YOLO11n person detection — M5"
    )
    parser.add_argument(
        "--source", default=_DEFAULT_SOURCE,
        help=f"Input video path (default: {_DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output-dir", default=_DEFAULT_OUTPUT_DIR, dest="output_dir",
        help=f"Directory to save review frames (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL,
        help=f"YOLO11n weights path (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--confidence", type=float, default=_DEFAULT_CONFIDENCE,
        help=f"Confidence threshold (default: {_DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--device", default=_DEFAULT_DEVICE,
        help="Torch device (default: auto)",
    )
    parser.add_argument(
        "--step", type=int, default=_DEFAULT_STEP,
        help=f"Save every N-th frame from the video (default: {_DEFAULT_STEP})",
    )
    parser.add_argument(
        "--extra-frames", nargs="*", type=int,
        default=_MANDATORY_FRAMES + _BENCHMARK_EXTRA,
        dest="extra_frames",
        help="Additional frame indices to always include (default: mandatory + benchmark extras)",
    )
    parser.add_argument(
        "--benchmark-json",
        default=os.path.join("outputs", "benchmark.json"),
        dest="benchmark_json",
        help="Path to benchmark.json (used to annotate timing on frames)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _stamp_overlay(
    frame: np.ndarray,
    frame_index: int,
    timestamp_ms: float,
    detections: list[Detection],
    inference_ms: float | None,
) -> np.ndarray:
    """Draw a text overlay with frame metadata at the top of the frame."""
    out = frame.copy()

    det_str = f"dets={len(detections)}"
    if detections:
        max_conf = max(d.score for d in detections)
        det_str += f"  max_conf={max_conf:.3f}"

    infer_str = f"  infer={inference_ms:.1f}ms" if inference_ms is not None else ""
    line1 = f"frame={frame_index:06d}  t={timestamp_ms:.0f}ms  {det_str}{infer_str}"

    # Semi-transparent black bar at the top
    bar_h = 32
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], bar_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    cv2.putText(
        out, line1,
        (6, 22),
        _FONT, 0.55, (255, 255, 0), 1, cv2.LINE_AA,
    )
    return out


def _auto_review_category(num_detections: int) -> str:
    """Assign a provisional category from detection count only.

    Reviewers must validate and override these in the CSV.
    """
    if num_detections == 0:
        return "empty_scene"
    if num_detections == 1:
        return "normal"
    return "normal"   # ≥2 — reviewer decides: near_person / partial_body / etc.


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_review(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load benchmark timing if available --------------------------------
    benchmark_timing: dict[int, float] = {}
    if os.path.isfile(args.benchmark_json):
        with open(args.benchmark_json) as f:
            bdata = json.load(f)
        for rec in bdata.get("frames", []):
            benchmark_timing[rec["frame_index"]] = rec["inference_ms"]
        print(f"[M5] Loaded benchmark timing for {len(benchmark_timing)} frames")
    else:
        print(f"[M5] benchmark.json not found — timing overlay disabled")

    # --- VideoSource -------------------------------------------------------
    source = VideoSource(args.source)
    print(
        f"[M5] Source  : {args.source} "
        f"({source.width}x{source.height} @ {source.fps} fps, "
        f"{source.frame_count} frames)"
    )

    # --- Detector ----------------------------------------------------------
    detector = YOLO11Detector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
    )
    print(f"[M5] Detector: {detector}")

    # --- Build the set of frames to process --------------------------------
    step_frames = set(range(0, source.frame_count, args.step))
    extra = set(args.extra_frames or [])
    # Clamp to valid range
    target_frames = sorted(
        fi for fi in (step_frames | extra) if 0 <= fi < source.frame_count
    )
    print(f"[M5] Frames to review: {len(target_frames)} "
          f"(step={args.step} + {len(extra)} extras)")
    print()

    # --- Process -----------------------------------------------------------
    csv_rows: list[dict] = []
    saved_count = 0

    try:
        for frame_index, timestamp_ms, frame in source.frames():
            if frame_index not in target_frames:
                # Fast path — just convert set to sorted list for membership test
                # We already have `target_frames` as sorted list, but set lookup is O(1)
                continue

            # Detect
            detections = detector.predict(frame)

            # Annotated frame (copy — visualization does not modify original)
            annotated = draw_detections(frame, detections)

            # Overlay metadata stamp
            infer_ms = benchmark_timing.get(frame_index)
            annotated = _stamp_overlay(
                annotated, frame_index, timestamp_ms, detections, infer_ms
            )

            # Save JPEG
            out_name = f"frame_{frame_index:06d}.jpg"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            saved_count += 1

            # CSV row — reviewer fills issue_type / severity / note manually
            max_conf = (
                round(max(d.score for d in detections), 4)
                if detections else 0.0
            )
            csv_rows.append({
                "frame_index": frame_index,
                "timestamp_ms": round(timestamp_ms, 1),
                "num_detections": len(detections),
                "max_confidence": max_conf,
                "review_category": _auto_review_category(len(detections)),
                "issue_type": "none",
                "severity": "none",
                "note": "",
            })

            if frame_index % 30 == 0 or frame_index in extra:
                print(
                    f"  frame {frame_index:4d}  dets={len(detections)}  "
                    f"max_conf={max_conf:.3f}  → {out_name}"
                )

    finally:
        detector.release()
        source.release()

    # --- Write CSV ---------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "review_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    print()
    print(f"[M5] Saved {saved_count} review frames → {args.output_dir}/")
    print(f"[M5] CSV report                          → {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    try:
        run_review(args)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
