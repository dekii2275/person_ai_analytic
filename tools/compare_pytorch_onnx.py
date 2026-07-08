"""
tools/compare_pytorch_onnx.py — Parity comparison: PyTorch vs ONNX detector.

Usage:
    python tools/compare_pytorch_onnx.py
    python tools/compare_pytorch_onnx.py \\
        --source data/input.mp4 \\
        --pytorch-model models/yolo11n.pt \\
        --onnx-model models/yolo11n.onnx \\
        --output outputs/onnx_parity.json

Compares detection outputs on fixed frame indices and writes a parity report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Allow import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detectors.yolov11 import YOLO11Detector
from src.detectors.yolov11_onnx import YOLO11ONNXDetector
from src.schemas import Detection

# Fixed parity frame set (as required by M6 spec)
_PARITY_FRAMES = [0, 30, 60, 79, 90, 120, 155, 180, 196, 258]

_PERSON_CLASS_ID = 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch vs ONNX YOLO11n detections"
    )
    parser.add_argument(
        "--source",
        default=os.path.join("data", "input.mp4"),
        help="Input video path (default: data/input.mp4)",
    )
    parser.add_argument(
        "--pytorch-model",
        default=os.path.join("models", "yolo11n.pt"),
        dest="pytorch_model",
        help="PyTorch model path (default: models/yolo11n.pt)",
    )
    parser.add_argument(
        "--onnx-model",
        default=os.path.join("models", "yolo11n.onnx"),
        dest="onnx_model",
        help="ONNX model path (default: models/yolo11n.onnx)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "onnx_parity.json"),
        help="Output JSON path (default: outputs/onnx_parity.json)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25 — same as baseline)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _iou(a: Detection, b: Detection) -> float:
    """Compute IoU between two Detection bounding boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    a_area = (a.x2 - a.x1) * (a.y2 - a.y1)
    b_area = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = a_area + b_area - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def _greedy_match(
    pt_dets: List[Detection],
    onnx_dets: List[Detection],
) -> List[Dict]:
    """Greedy one-to-one matching by class + highest IoU.

    Returns list of match dicts, one per matched pair.
    Unmatched detections are noted in summary but not in matches list.
    """
    if not pt_dets or not onnx_dets:
        return []

    # Build IoU matrix
    n_pt = len(pt_dets)
    n_onnx = len(onnx_dets)
    iou_matrix = np.zeros((n_pt, n_onnx), dtype=np.float64)

    for i, pd in enumerate(pt_dets):
        for j, od in enumerate(onnx_dets):
            if pd.class_id == od.class_id:
                iou_matrix[i, j] = _iou(pd, od)

    matched_pt = set()
    matched_onnx = set()
    matches = []

    # Greedy: pick the highest-IoU pair repeatedly
    flat = np.argsort(iou_matrix.ravel())[::-1]
    for idx in flat:
        i, j = divmod(int(idx), n_onnx)
        if i in matched_pt or j in matched_onnx:
            continue
        if iou_matrix[i, j] == 0.0:
            break
        pd = pt_dets[i]
        od = onnx_dets[j]
        matches.append({
            "pytorch_det": {
                "x1": pd.x1, "y1": pd.y1, "x2": pd.x2, "y2": pd.y2,
                "score": pd.score, "class_id": pd.class_id,
            },
            "onnx_det": {
                "x1": od.x1, "y1": od.y1, "x2": od.x2, "y2": od.y2,
                "score": od.score, "class_id": od.class_id,
            },
            "iou": round(float(iou_matrix[i, j]), 6),
            "confidence_diff": round(abs(pd.score - od.score), 6),
        })
        matched_pt.add(i)
        matched_onnx.add(j)

    return matches


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------

def _read_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """Read a single frame from a video file by index."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


# ---------------------------------------------------------------------------
# Main parity logic
# ---------------------------------------------------------------------------

def run_parity(args: argparse.Namespace) -> dict:
    """Run parity comparison and return the full report dict."""
    print(f"[M6] Source        : {args.source}")
    print(f"[M6] PyTorch model : {args.pytorch_model}")
    print(f"[M6] ONNX model    : {args.onnx_model}")
    print(f"[M6] Output        : {args.output}")
    print(f"[M6] Confidence    : {args.confidence}")
    print(f"[M6] Parity frames : {_PARITY_FRAMES}")
    print()

    # Validate inputs
    for path in (args.source, args.pytorch_model, args.onnx_model):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path!r}")

    # Load detectors
    print("[M6] Loading PyTorch detector ...")
    pt_detector = YOLO11Detector(
        model_path=args.pytorch_model,
        confidence=args.confidence,
    )
    print(f"[M6]   {pt_detector}")

    print("[M6] Loading ONNX detector ...")
    onnx_detector = YOLO11ONNXDetector(
        model_path=args.onnx_model,
        confidence=args.confidence,
    )
    print(f"[M6]   {onnx_detector}")
    print(f"[M6]   ONNX provider (actual): {onnx_detector.provider}")
    print()

    frame_results = []
    all_ious: List[float] = []
    all_conf_diffs: List[float] = []
    matching_count_frames = 0

    for frame_idx in _PARITY_FRAMES:
        frame = _read_frame(args.source, frame_idx)
        if frame is None:
            print(f"  frame {frame_idx:4d} — SKIP (could not read)")
            continue

        t0 = time.perf_counter()
        pt_dets = pt_detector.predict(frame)
        pt_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        onnx_dets = onnx_detector.predict(frame)
        onnx_ms = (time.perf_counter() - t0) * 1000

        matches = _greedy_match(pt_dets, onnx_dets)

        frame_ious = [m["iou"] for m in matches]
        frame_conf_diffs = [m["confidence_diff"] for m in matches]
        all_ious.extend(frame_ious)
        all_conf_diffs.extend(frame_conf_diffs)

        count_match = len(pt_dets) == len(onnx_dets)
        if count_match:
            matching_count_frames += 1

        mean_iou_str = f"{float(np.mean(frame_ious)):.4f}" if frame_ious else "—"
        print(
            f"  frame {frame_idx:4d}  "
            f"pt={len(pt_dets)}  onnx={len(onnx_dets)}  "
            f"matches={len(matches)}  "
            f"mean_iou={mean_iou_str}  "
            f"pt_ms={pt_ms:.1f}  onnx_ms={onnx_ms:.1f}"
        )

        frame_results.append({
            "frame_index": frame_idx,
            "pytorch_count": len(pt_dets),
            "onnx_count": len(onnx_dets),
            "detection_count_match": count_match,
            "pytorch_inference_ms": round(pt_ms, 3),
            "onnx_inference_ms": round(onnx_ms, 3),
            "matches": matches,
        })

    # Summary
    n_frames = len(frame_results)
    mean_iou = float(np.mean(all_ious)) if all_ious else None
    min_iou = float(np.min(all_ious)) if all_ious else None
    mean_conf_diff = float(np.mean(all_conf_diffs)) if all_conf_diffs else None
    max_conf_diff = float(np.max(all_conf_diffs)) if all_conf_diffs else None

    report = {
        "metadata": {
            "pytorch_model": args.pytorch_model,
            "onnx_model": args.onnx_model,
            "onnx_provider": onnx_detector.provider,
            "confidence_threshold": args.confidence,
            "frames": _PARITY_FRAMES,
            "source_video": args.source,
        },
        "summary": {
            "frames_compared": n_frames,
            "matching_detection_count_frames": matching_count_frames,
            "total_matched_pairs": len(all_ious),
            "mean_bbox_iou": round(mean_iou, 6) if mean_iou is not None else None,
            "min_bbox_iou": round(min_iou, 6) if min_iou is not None else None,
            "mean_confidence_diff": round(mean_conf_diff, 6) if mean_conf_diff is not None else None,
            "max_confidence_diff": round(max_conf_diff, 6) if max_conf_diff is not None else None,
        },
        "frames": frame_results,
    }

    pt_detector.release()
    onnx_detector.release()

    return report


def main(argv=None) -> None:
    args = _parse_args(argv)

    try:
        report = run_parity(args)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # Save JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[M6] Saved → {args.output}")

    # Verify JSON is re-parseable
    with open(args.output, "r", encoding="utf-8") as f:
        _verified = json.load(f)
    print("[M6] JSON re-parse: OK")

    # Print summary
    s = report["summary"]
    print("\n[M6] Parity Summary:")
    print(f"  frames compared              : {s['frames_compared']}")
    print(f"  matching detection count     : {s['matching_detection_count_frames']}/{s['frames_compared']}")
    print(f"  total matched pairs          : {s['total_matched_pairs']}")
    print(f"  mean bbox IoU                : {s['mean_bbox_iou']}")
    print(f"  min  bbox IoU                : {s['min_bbox_iou']}")
    print(f"  mean confidence diff         : {s['mean_confidence_diff']}")
    print(f"  max  confidence diff         : {s['max_confidence_diff']}")

    provider = report["metadata"]["onnx_provider"]
    print(f"\n[M6] ONNX provider used: {provider}")

    if provider == "CPUExecutionProvider":
        print("[M6] NOTE: ONNX is running on CPU (not GPU).")


if __name__ == "__main__":
    main()
