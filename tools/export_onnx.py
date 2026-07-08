"""
tools/export_onnx.py — Export YOLO11n PyTorch weights to ONNX.

Usage:
    python tools/export_onnx.py
    python tools/export_onnx.py --model models/yolo11n.pt --output models/yolo11n.onnx

Details:
    - opset      : 12  (TensorRT-compatible, widely supported)
    - imgsz      : 640 (static shape — TensorRT-friendly)
    - dynamic    : False (no dynamic axes)
    - quantize   : False (FP32 only)
    - format     : onnx
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO11n .pt → .onnx"
    )
    parser.add_argument(
        "--model",
        default=os.path.join("models", "yolo11n.pt"),
        help="Path to yolo11n.pt weights (default: models/yolo11n.pt)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("models", "yolo11n.onnx"),
        help="Output path for .onnx file (default: models/yolo11n.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for export (default: 640)",
    )
    return parser.parse_args(argv)


def _print_model_info(onnx_path: str) -> dict:
    """Load ONNX model, run checker, print metadata. Returns info dict."""
    import onnx  # noqa: PLC0415

    print("\n[M6] Loading ONNX model for validation ...")
    model = onnx.load(onnx_path)

    # Run onnx checker — raises onnx.checker.ValidationError on failure
    onnx.checker.check_model(model)
    print("[M6] ONNX checker: PASS")

    graph = model.graph
    inputs = []
    for inp in graph.input:
        shape = [
            (d.dim_value if d.HasField("dim_value") else d.dim_param)
            for d in inp.type.tensor_type.shape.dim
        ]
        dtype_map = {1: "float32", 2: "uint8", 7: "int64", 6: "int32"}
        dtype = dtype_map.get(inp.type.tensor_type.elem_type, "unknown")
        inputs.append({"name": inp.name, "shape": shape, "dtype": dtype})

    outputs = []
    for out in graph.output:
        shape = [
            (d.dim_value if d.HasField("dim_value") else d.dim_param)
            for d in out.type.tensor_type.shape.dim
        ]
        dtype_map = {1: "float32", 2: "uint8", 7: "int64", 6: "int32"}
        dtype = dtype_map.get(out.type.tensor_type.elem_type, "unknown")
        outputs.append({"name": out.name, "shape": shape, "dtype": dtype})

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    print("\n[M6] ONNX Model Metadata:")
    print(f"  File           : {onnx_path}")
    print(f"  File size      : {file_size_mb:.2f} MB")
    print(f"  ONNX IR version: {model.ir_version}")
    for inp in inputs:
        print(f"  Input  name    : {inp['name']}")
        print(f"  Input  shape   : {inp['shape']}")
        print(f"  Input  dtype   : {inp['dtype']}")
    for out in outputs:
        print(f"  Output name    : {out['name']}")
        print(f"  Output shape   : {out['shape']}")

    return {
        "path": onnx_path,
        "file_size_mb": round(file_size_mb, 2),
        "ir_version": model.ir_version,
        "inputs": inputs,
        "outputs": outputs,
    }


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Validate input
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model not found: {args.model!r}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"[M6] Exporting: {args.model}")
    print(f"[M6] Destination: {args.output}")
    print(f"[M6] Opset: {args.opset}  |  imgsz: {args.imgsz}  |  dynamic: False")

    # ultralytics export — confined to this function
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(args.model)
    export_result = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,          # static shape — TensorRT-friendly
        simplify=True,          # onnxsim simplification if available
        half=False,             # FP32 only — no quantization
        verbose=False,
    )
    # ultralytics places the .onnx next to the .pt unless we move it
    # export_result is the path it wrote to
    exported_path = str(export_result)
    print(f"[M6] ultralytics wrote: {exported_path}")

    # Move to desired output path if different
    if os.path.abspath(exported_path) != os.path.abspath(args.output):
        import shutil  # noqa: PLC0415
        shutil.move(exported_path, args.output)
        print(f"[M6] Moved → {args.output}")

    if not os.path.isfile(args.output):
        print(f"[ERROR] Expected output not found: {args.output!r}", file=sys.stderr)
        sys.exit(1)

    info = _print_model_info(args.output)

    print("\n[M6] Export complete.")
    print(f"  Input  : {info['inputs'][0]['name']}  shape={info['inputs'][0]['shape']}  dtype={info['inputs'][0]['dtype']}")
    print(f"  Output : {info['outputs'][0]['name']}  shape={info['outputs'][0]['shape']}")
    print(f"  Size   : {info['file_size_mb']} MB")
    print("\n[M6] READY — models/yolo11n.onnx validated successfully.")


if __name__ == "__main__":
    main()
