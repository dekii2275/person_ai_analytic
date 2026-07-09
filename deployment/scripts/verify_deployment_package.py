"""
verify_deployment_package.py
----------------------------
Verifies the deployment package is complete and consistent before
copying to NVIDIA Orin.

Checks (in order):
  1. deployment_manifest.json exists and parses
  2. ONNX model file exists
  3. input.mp4 exists
  4. ONNX parity report exists
  5. Local benchmark report exists
  6. All 10 parity frames exist
  7. ONNX SHA256 matches manifest
  8. input.mp4 can be opened as video
  9. Video metadata is valid (frame count, resolution, FPS)
 10. Parity frames match expected resolution
 11. JSON files parse without error
 12. ONNX checker PASS (if onnx package available)

Does NOT run inference. Does NOT require ONNX Runtime.

Usage:
    python deployment/scripts/verify_deployment_package.py

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Locate deployment root (this script lives at deployment/scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = SCRIPT_DIR.parent

MANIFEST_PATH = DEPLOYMENT_DIR / "deployment_manifest.json"

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
SKIP = "\033[33m[SKIP]\033[0m"
INFO = "\033[36m[INFO]\033[0m"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  {status} {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return condition


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_manifest() -> Tuple[bool, dict]:
    """Check 1: deployment_manifest.json exists and parses."""
    if not MANIFEST_PATH.exists():
        _check("manifest exists", False, str(MANIFEST_PATH))
        return False, {}
    try:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        _check("manifest exists and parses", True, str(MANIFEST_PATH))
        return True, manifest
    except json.JSONDecodeError as exc:
        _check("manifest JSON valid", False, str(exc))
        return False, {}


def check_onnx(manifest: dict) -> bool:
    """Check 2: ONNX model file exists."""
    rel = manifest.get("model", {}).get("path", "models/yolo11n.onnx")
    path = DEPLOYMENT_DIR / rel
    return _check("ONNX model exists", path.exists(), str(path))


def check_video(manifest: dict) -> bool:
    """Check 3: input.mp4 exists."""
    rel = manifest.get("test_data", {}).get("video", {}).get("path", "test_data/input.mp4")
    path = DEPLOYMENT_DIR / rel
    return _check("input.mp4 exists", path.exists(), str(path))


def check_parity_report(manifest: dict) -> bool:
    """Check 4: ONNX parity report exists."""
    rel = manifest.get("reference", {}).get("onnx_parity_report", "reports/onnx_parity.json")
    path = DEPLOYMENT_DIR / rel
    return _check("onnx_parity.json exists", path.exists(), str(path))


def check_benchmark_report(manifest: dict) -> bool:
    """Check 5: local benchmark report exists."""
    rel = manifest.get("reference", {}).get("local_benchmark_report", "reports/benchmark_local.json")
    path = DEPLOYMENT_DIR / rel
    return _check("benchmark_local.json exists", path.exists(), str(path))


def check_parity_frames(manifest: dict) -> bool:
    """Check 6: all 10 parity frames exist."""
    frame_dir_rel = manifest.get("test_data", {}).get("parity_frames", {}).get("dir", "test_data/parity_frames")
    frame_dir = DEPLOYMENT_DIR / frame_dir_rel
    expected_indices: List[int] = manifest.get("test_data", {}).get("parity_frames", {}).get(
        "frame_indices", [0, 30, 60, 79, 90, 120, 155, 180, 196, 258]
    )
    expected_count: int = manifest.get("test_data", {}).get("parity_frames", {}).get("count", 10)

    missing = []
    for idx in expected_indices:
        fname = frame_dir / f"frame_{idx:06d}.jpg"
        if not fname.exists():
            missing.append(str(fname))

    found_count = expected_count - len(missing)
    ok = len(missing) == 0
    detail = f"found {found_count}/{expected_count}"
    if missing:
        detail += f", missing: {missing[:3]}{'...' if len(missing) > 3 else ''}"
    return _check("parity frames (10/10)", ok, detail)


def check_onnx_checksum(manifest: dict) -> bool:
    """Check 7: ONNX SHA256 matches manifest."""
    rel = manifest.get("model", {}).get("path", "models/yolo11n.onnx")
    path = DEPLOYMENT_DIR / rel
    expected_sha = manifest.get("model", {}).get("sha256", "")

    if not path.exists():
        return _check("ONNX checksum", False, "file not found")
    if not expected_sha:
        return _check("ONNX checksum", False, "no sha256 in manifest")

    actual_sha = _sha256(path)
    match = actual_sha == expected_sha
    detail = (
        f"expected={expected_sha[:16]}...\n         actual  ={actual_sha[:16]}..."
    )
    return _check("ONNX SHA256 matches manifest", match, detail)


def check_video_openable(manifest: dict) -> bool:
    """Check 8: input.mp4 can be opened as a video."""
    try:
        import cv2
    except ImportError:
        print(f"  {SKIP} video openable (cv2 not available, skipping)")
        return True  # Non-blocking skip

    rel = manifest.get("test_data", {}).get("video", {}).get("path", "test_data/input.mp4")
    path = DEPLOYMENT_DIR / rel
    if not path.exists():
        return _check("video openable", False, "file not found")

    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    cap.release()
    return _check("video openable with cv2", ok, str(path))


def check_video_metadata(manifest: dict) -> bool:
    """Check 9: video metadata is valid."""
    try:
        import cv2
    except ImportError:
        print(f"  {SKIP} video metadata (cv2 not available, skipping)")
        return True

    rel = manifest.get("test_data", {}).get("video", {}).get("path", "test_data/input.mp4")
    path = DEPLOYMENT_DIR / rel
    if not path.exists():
        return _check("video metadata", False, "file not found")

    expected_frames: int = manifest.get("test_data", {}).get("video", {}).get("frames", 0)
    expected_res: str = manifest.get("test_data", {}).get("video", {}).get("resolution", "576x1024")
    expected_fps: float = manifest.get("test_data", {}).get("video", {}).get("fps", 30.0)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return _check("video metadata", False, "cannot open")

    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    actual_res = f"{actual_w}x{actual_h}"
    ok = (
        actual_frames == expected_frames
        and actual_res == expected_res
        and abs(actual_fps - expected_fps) < 0.1
    )
    detail = (
        f"frames={actual_frames} (expected {expected_frames}), "
        f"res={actual_res} (expected {expected_res}), "
        f"fps={actual_fps:.1f} (expected {expected_fps:.1f})"
    )
    return _check("video metadata valid", ok, detail)


def check_parity_frame_resolution(manifest: dict) -> bool:
    """Check 10: parity frames match expected resolution."""
    try:
        import cv2
    except ImportError:
        print(f"  {SKIP} frame resolution (cv2 not available, skipping)")
        return True

    frame_dir_rel = manifest.get("test_data", {}).get("parity_frames", {}).get("dir", "test_data/parity_frames")
    frame_dir = DEPLOYMENT_DIR / frame_dir_rel
    expected_res: str = manifest.get("test_data", {}).get("parity_frames", {}).get("resolution", "576x1024")
    expected_indices: List[int] = manifest.get("test_data", {}).get("parity_frames", {}).get(
        "frame_indices", [0, 30, 60, 79, 90, 120, 155, 180, 196, 258]
    )

    # Parse expected WxH
    try:
        exp_w, exp_h = (int(x) for x in expected_res.split("x"))
    except ValueError:
        return _check("frame resolution", False, f"cannot parse expected_res={expected_res}")

    mismatches = []
    for idx in expected_indices:
        fpath = frame_dir / f"frame_{idx:06d}.jpg"
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        if img is None:
            mismatches.append(f"frame_{idx:06d}.jpg: cannot read")
            continue
        h, w = img.shape[:2]
        if w != exp_w or h != exp_h:
            mismatches.append(f"frame_{idx:06d}.jpg: {w}x{h} != {expected_res}")

    ok = len(mismatches) == 0
    detail = f"expected {expected_res}" + (f", mismatches: {mismatches[:2]}" if mismatches else " — all match")
    return _check("parity frame resolution matches video", ok, detail)


def check_json_files(manifest: dict) -> bool:
    """Check 11: all JSON files parse without error."""
    json_paths = [
        DEPLOYMENT_DIR / manifest.get("reference", {}).get("onnx_parity_report", "reports/onnx_parity.json"),
        DEPLOYMENT_DIR / manifest.get("reference", {}).get("local_benchmark_report", "reports/benchmark_local.json"),
    ]
    all_ok = True
    for p in json_paths:
        if not p.exists():
            ok = _check(f"JSON parse: {p.name}", False, "file not found")
            all_ok = all_ok and ok
            continue
        try:
            with open(p) as f:
                json.load(f)
            ok = _check(f"JSON parse: {p.name}", True)
        except json.JSONDecodeError as exc:
            ok = _check(f"JSON parse: {p.name}", False, str(exc))
        all_ok = all_ok and ok
    return all_ok


def check_onnx_checker(manifest: dict) -> bool:
    """Check 12: ONNX model passes onnx.checker (if onnx package available)."""
    try:
        import onnx  # noqa: PLC0415
    except ImportError:
        print(f"  {SKIP} ONNX checker (onnx package not installed, skipping)")
        return True  # Non-blocking

    rel = manifest.get("model", {}).get("path", "models/yolo11n.onnx")
    path = DEPLOYMENT_DIR / rel
    if not path.exists():
        return _check("ONNX checker", False, "file not found")

    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        ok = True
        detail = "onnx.checker.check_model PASS"
    except Exception as exc:  # noqa: BLE001
        ok = False
        detail = str(exc)[:200]

    return _check("ONNX checker PASS", ok, detail)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    sep = "=" * 60
    print(sep)
    print("  DEPLOYMENT PACKAGE VERIFICATION")
    print(f"  Package: {DEPLOYMENT_DIR}")
    print(sep)

    results = []

    # Check 1: manifest
    manifest_ok, manifest = check_manifest()
    results.append(manifest_ok)
    if not manifest_ok:
        print()
        print(f"{FAIL} Cannot continue without manifest.")
        return 1

    # Checks 2–12
    results.append(check_onnx(manifest))
    results.append(check_video(manifest))
    results.append(check_parity_report(manifest))
    results.append(check_benchmark_report(manifest))
    results.append(check_parity_frames(manifest))
    results.append(check_onnx_checksum(manifest))
    results.append(check_video_openable(manifest))
    results.append(check_video_metadata(manifest))
    results.append(check_parity_frame_resolution(manifest))
    results.append(check_json_files(manifest))
    results.append(check_onnx_checker(manifest))

    # Summary
    passed = sum(results)
    total = len(results)
    all_pass = all(results)

    print()
    print(sep)
    if all_pass:
        print(f"\033[32m  DEPLOYMENT PACKAGE: PASS  ({passed}/{total} checks)\033[0m")
    else:
        print(f"\033[31m  DEPLOYMENT PACKAGE: FAIL  ({passed}/{total} checks passed)\033[0m")
    print(sep)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
