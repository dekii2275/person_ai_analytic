"""
main.py — pipeline orchestrator (M3 + M4).

Reads data/input.mp4, runs YOLO11n person detection on every frame,
draws bounding boxes, and writes outputs/output.mp4.

Usage (M3 — plain pipeline):
    python main.py
    python main.py --source data/input.mp4 --output outputs/output.mp4
    python main.py --model models/yolo11n.pt --confidence 0.25 --device cuda

Usage (M4 — benchmark):
    python main.py --benchmark
    python main.py --benchmark --benchmark-output outputs/benchmark.json

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

No tracking, no ONNX/TRT.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from src.benchmark import (
    FrameRecord,
    StageTimer,
    aggregate_result,
    print_summary,
    save_json,
)
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
_DEFAULT_BENCHMARK_OUTPUT = os.path.join("outputs", "benchmark.json")
_DEFAULT_WARMUP_FRAMES = 3

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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark pipeline (measures per-stage latency)",
    )
    parser.add_argument(
        "--benchmark-output",
        default=_DEFAULT_BENCHMARK_OUTPUT,
        dest="benchmark_output",
        help=f"Benchmark JSON output path (default: {_DEFAULT_BENCHMARK_OUTPUT})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=_DEFAULT_WARMUP_FRAMES,
        help=f"Number of warmup frames excluded from stats (default: {_DEFAULT_WARMUP_FRAMES})",
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
# Benchmark pipeline (M4)
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full pipeline with per-stage latency measurement.

    Stages timed per frame:
        decode        — cv2 frame read
        preprocess    — (currently 0 ms: YOLO handles internally)
        inference     — YOLO11Detector.predict() with CUDA sync
        postprocess   — (absorbed into inference; reserved for future use)
        draw_write    — draw_detections_inplace + writer.write

    Warmup frames are processed but excluded from aggregate stats.
    """
    import torch  # noqa: PLC0415  (benchmark only; not needed in M3 plain path)

    cuda_available = torch.cuda.is_available()

    print(f"[M4] Source    : {args.source}")
    print(f"[M4] Output    : {args.output}")
    print(f"[M4] Benchmark : {args.benchmark_output}")
    print(f"[M4] Model     : {args.model}")
    print(f"[M4] Conf      : {args.confidence}")
    print(f"[M4] Device    : {args.device or 'auto'}")
    print(f"[M4] Warmup    : {args.warmup} frames (excluded from stats)")
    print()

    # --- VideoSource -------------------------------------------------------
    source = VideoSource(args.source)
    print(
        f"[M4] Input     : {source.width}x{source.height} "
        f"@ {source.fps} fps ({source.frame_count} frames)"
    )

    # --- Detector ----------------------------------------------------------
    detector = YOLO11Detector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
    )
    actual_device = detector.device
    print(f"[M4] Detector  : {detector}")
    print()

    # --- VideoWriter -------------------------------------------------------
    writer = _open_writer(
        args.output,
        width=source.width,
        height=source.height,
        fps=source.fps,
    )

    records: list[FrameRecord] = []
    warmup_done = 0
    t_wall_start = time.perf_counter()

    try:
        for frame_index, _ts_ms, frame in source.frames():
            is_warmup = warmup_done < args.warmup

            # ---- Decode timing: frame is already decoded by VideoSource ----
            # We measure a no-op "decode" pass here because VideoSource's
            # generator has already read the frame; a proper decode timer
            # would require splitting VideoSource internals (out of scope).
            # Instead we time just the numpy copy to simulate the boundary.
            with StageTimer(cuda_sync=False) as t_decode:
                frame_copy = frame.copy()   # isolate from inplace draw

            # ---- Preprocess -----------------------------------------------
            # YOLO handles all preprocessing internally (resize, normalize).
            # We record 0.0 as a reserved stage for future explicit work.
            with StageTimer(cuda_sync=False) as t_pre:
                pass  # reserved — YOLO internal

            # ---- Inference (CUDA-synchronised) ----------------------------
            with StageTimer(cuda_sync=cuda_available) as t_infer:
                detections = detector.predict(frame_copy)

            # ---- Postprocess ----------------------------------------------
            # Filtering is already done inside predict(); nothing extra here.
            with StageTimer(cuda_sync=False) as t_post:
                pass  # reserved

            # ---- Draw + Write ---------------------------------------------
            with StageTimer(cuda_sync=False) as t_dw:
                draw_detections_inplace(frame_copy, detections)
                writer.write(frame_copy)

            total_ms = (
                t_decode[0] + t_pre[0] + t_infer[0] + t_post[0] + t_dw[0]
            )

            if is_warmup:
                warmup_done += 1
                if frame_index % 1 == 0:
                    print(f"  [warmup] frame {frame_index}")
                continue

            records.append(
                FrameRecord(
                    frame_index=frame_index,
                    decode_ms=t_decode[0],
                    preprocess_ms=t_pre[0],
                    inference_ms=t_infer[0],
                    postprocess_ms=t_post[0],
                    draw_write_ms=t_dw[0],
                    total_ms=total_ms,
                    n_detections=len(detections),
                )
            )

            if frame_index % 30 == 0:
                print(
                    f"  frame {frame_index:4d}/{source.frame_count}  "
                    f"infer={t_infer[0]:.1f}ms  total={total_ms:.1f}ms  "
                    f"dets={len(detections)}"
                )

    finally:
        writer.release()
        detector.release()
        source.release()

    t_wall_end = time.perf_counter()
    wall_time = t_wall_end - t_wall_start

    # --- Build result ------------------------------------------------------
    result = aggregate_result(
        records,
        model=os.path.basename(args.model),
        device=actual_device,
        video=args.source,
        resolution=f"{source.width}x{source.height}",
        fps=source.fps,
        warmup_frames=args.warmup,
        total_wall_time_s=wall_time,
    )

    # --- Print summary ------------------------------------------------------
    print_summary(result)

    # --- Save JSON ----------------------------------------------------------
    save_json(result, args.benchmark_output)
    print(f"[M4] Saved benchmark → {args.benchmark_output}")

    # --- Verify output video ------------------------------------------------
    print(f"[M4] Verifying output video: {args.output} ...")
    info = verify_output(
        args.output,
        expected_width=source.width,
        expected_height=source.height,
        expected_fps=source.fps,
    )
    print(
        f"[M4] Output OK — {info['width']}x{info['height']} "
        f"@ {info['fps']} fps  frames={info['frame_count']}  "
        f"first_frame_readable={info['first_frame_readable']}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    try:
        if args.benchmark:
            run_benchmark(args)
        else:
            run(args)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
