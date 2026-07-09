# Deployment Package — YOLO11n Person Detector (M7-PREP)

## 1. Purpose

This package contains everything needed to deploy YOLO11n person detection on **NVIDIA Jetson Orin** via TensorRT.

Current status: **READY\_FOR\_ORIN\_INSPECTION**  
The ONNX model has been verified against the PyTorch baseline (PyTorch ↔ ONNX parity confirmed).  
Next step: build a TensorRT FP16 engine on the Orin board.

---

## 2. Package Structure

```
deployment/
├── models/
│   └── yolo11n.onnx           # ONNX model (opset 12, static 640×640)
│
├── test_data/
│   ├── input.mp4              # 259 frames, 576×1024, 30 FPS
│   └── parity_frames/         # 10 reference frames for parity testing
│       ├── frame_000000.jpg
│       ├── frame_000030.jpg
│       ├── frame_000060.jpg
│       ├── frame_000079.jpg
│       ├── frame_000090.jpg
│       ├── frame_000120.jpg
│       ├── frame_000155.jpg
│       ├── frame_000180.jpg
│       ├── frame_000196.jpg
│       └── frame_000258.jpg
│
├── reports/
│   ├── onnx_parity.json       # PyTorch vs ONNX parity results
│   └── benchmark_local.json   # Local GPU (CUDA) benchmark results
│
├── scripts/
│   ├── inspect_orin.sh        # Orin environment inspection (read-only)
│   └── verify_deployment_package.py  # Package completeness verifier
│
├── deployment_manifest.json   # Machine-readable package metadata
└── README.md                  # This file
```

---

## 3. Model Metadata

| Field         | Value                  |
|---------------|------------------------|
| Model         | YOLO11n                |
| Format        | ONNX                   |
| ONNX Opset    | 12                     |
| Input tensor  | `images`               |
| Input shape   | `[1, 3, 640, 640]`     |
| Input dtype   | `float32`              |
| Input layout  | NCHW                   |
| Output tensor | `output0`              |
| Output shape  | `[1, 84, 8400]`        |
| Output dtype  | `float32`              |

---

## 4. Preprocessing

Applied to every input frame **before** feeding to the model:

| Step | Operation                                  |
|------|--------------------------------------------|
| 1    | Letterbox resize to 640×640 (INTER_LINEAR) |
| 2    | Pad colour: (114, 114, 114) grey           |
| 3    | BGR → RGB                                  |
| 4    | uint8 → float32 / 255.0 (normalize 0–1)   |
| 5    | HWC → CHW (channel-first)                  |
| 6    | Add batch dimension → (1, 3, 640, 640)     |

---

## 5. Postprocessing

Applied to model raw output `[1, 84, 8400]`:

| Step | Operation                                                     |
|------|---------------------------------------------------------------|
| 1    | Transpose → `(8400, 84)`                                      |
| 2    | Extract `cx, cy, w, h` and 80 class scores                    |
| 3    | Filter person class (index **0**) by confidence ≥ **0.25**   |
| 4    | Convert `cx,cy,w,h` → `x1,y1,x2,y2` in letterboxed space     |
| 5    | Reverse letterbox → original frame pixel coordinates          |
| 6    | Clip coordinates to frame bounds                              |
| 7    | Apply NMS (IoU threshold = **0.45**)                          |
| 8    | Return `List[Detection]`                                      |

Detection schema:

```python
Detection(x1, y1, x2, y2, score, class_id)
# All coordinates in original frame pixels
# class_id = 0 (person only)
```

---

## 6. Test Data

| Field       | Value                         |
|-------------|-------------------------------|
| File        | `test_data/input.mp4`         |
| Frames      | 259                           |
| Resolution  | 576 × 1024                    |
| FPS         | 30.0                          |

---

## 7. Parity Frames

10 reference frames extracted at indices:

```
0, 30, 60, 79, 90, 120, 155, 180, 196, 258
```

These frames are used to compare:
- PyTorch baseline output  
- ONNX Runtime output  
- TensorRT engine output (on Orin)

**Confirmed parity (PyTorch ↔ ONNX):**

| Metric                  | Value   |
|-------------------------|---------|
| Frames compared         | 10      |
| Detection count match   | 10/10   |
| Mean bbox IoU           | 0.9931  |
| Min bbox IoU            | 0.9671  |
| Mean confidence diff    | 0.0107  |
| Max confidence diff     | 0.0665  |

---

## 8. Verify Package

Before copying to Orin, verify the package is complete:

```bash
python deployment/scripts/verify_deployment_package.py
```

Expected output:

```
  DEPLOYMENT PACKAGE: PASS  (12/12 checks)
```

---

## 9. On Orin: Next Steps

### Step 1 — Inspect Orin environment

```bash
bash deployment/scripts/inspect_orin.sh 2>&1 | tee orin_env.txt
```

Send `orin_env.txt` back for review. This determines:
- JetPack version
- TensorRT version
- Available providers (FP16 support, INT8 calibration)

### Step 2 — Send environment output for TensorRT flow decision

After reviewing JetPack/TensorRT version from the inspect output, the appropriate `trtexec` command and FP16/INT8 options will be determined.

> **Do NOT run trtexec until the environment has been reviewed.**  
> TensorRT engine files are not portable between JetPack versions.

### Step 3 — Build TensorRT FP16 engine on-device

(Command to be determined after Step 2 environment review.)

Inputs:
- `deployment/models/yolo11n.onnx`

Outputs:
- `yolo11n_fp16.engine` (stays on Orin, not included in this package)

### Step 4 — TensorRT parity test

Run inference on the 10 parity frames using the `.engine` file.  
Compare against `reports/onnx_parity.json` reference values.  
Accept if: mean IoU ≥ 0.95, detection count matches ≥ 9/10.

### Step 5 — TensorRT benchmark

Run `tegrastats` + inference loop.  
Compare latency and FPS against `reports/benchmark_local.json`.

---

## 10. Checksums

Verify file integrity after copy:

```bash
sha256sum deployment/models/yolo11n.onnx
# expected: 2d1acca764239e491cb1b6ac1150e7367236e12355af0d2109215a3bab56d39e

sha256sum deployment/test_data/input.mp4
# expected: f3686138ca686d76d9c3b06c1d840b8dbede66a70c5e5293f2c8059d2e3db5ae
```

---

## 11. Local Benchmark (Reference)

Measured on local GPU (CUDA):

| Metric              | Value        |
|---------------------|--------------|
| Effective FPS       | 44.59        |
| Inference mean      | 10.78 ms     |
| Total pipeline mean | 13.83 ms     |
| Processed frames    | 256          |
| Device              | CUDA         |

---

*Package created: M7-PREP milestone*
