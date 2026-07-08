# AGENTS.md

## 1. Project Goal

Milestone hiện tại:

```text
data/input.mp4
→ YOLO11n local inference
→ keep person class only
→ Detection[]
→ draw bounding boxes
→ outputs/output.mp4
→ outputs/benchmark.json
```

Target dài hạn là NVIDIA Orin, nhưng task hiện tại chỉ chạy local.

## 2. Current Scope

Được phép:

- environment setup;
- repo skeleton;
- MP4 video reading;
- YOLO11n person detection;
- framework-independent Detection schema;
- visualization;
- output video;
- latency benchmark;
- tests/smoke tests;
- debugging;
- code review.

Không được phép:

- ByteTrack hoặc tracking;
- person attributes;
- face pipeline;
- crowd/behavior;
- Locate Anything;
- RTSP;
- ONNX export;
- TensorRT;
- DeepStream;
- Orin deployment;
- Docker;
- async/multiprocessing architecture.

## 3. Engineering Rules

### Small task only

Mỗi lần chỉ xử lý một task nhỏ có thể kiểm chứng.

### Inspect before edit

Trước khi sửa:

1. Đọc repo tree.
2. Đọc file liên quan.
3. Tóm tắt trạng thái hiện tại.
4. Chỉ sửa file cần thiết.

### No scope expansion

Không tự thêm tính năng ngoài task.

### Stable interfaces

Mọi detector phải trả về schema chung:

```python
Detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float,
    class_id: int,
)
```

Không leak `ultralytics.Results` ra ngoài detector module.

### Local first, Orin aware

Code hiện tại phải cho phép sau này thay:

```text
YOLO11 PyTorch backend
→ YOLO11 TensorRT backend
```

mà không viết lại:

- VideoSource;
- Detection schema;
- visualization;
- benchmark;
- application flow.

### Evidence before success

Không nói "đã hoàn thành" nếu chưa chạy command thật.

Không nói "video output hợp lệ" nếu chưa mở lại bằng code.

Không nói "benchmark xong" nếu chưa sinh file thật.

## 4. Required Workflow

```text
Understand
→ Inspect
→ Plan
→ Implement
→ Run
→ Verify artifacts
→ Review
→ Report
```

## 5. Final Report Format

```text
Changed:
- ...

Tested:
- ...

Commands executed:
- ...

Result:
- ...

Generated artifacts:
- ...

Remaining risks:
- ...

Code review:
- PASS / NEEDS_FIX

Next recommended task:
- ...
```
