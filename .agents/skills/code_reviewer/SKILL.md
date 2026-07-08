# Skill: Code Reviewer

## Review order

1. Correctness.
2. Scope compliance.
3. Test evidence.
4. Interface cleanliness.
5. Readability.

## Mandatory checks

- Có code ngoài scope không?
- Có tracking/ONNX/TensorRT không?
- `main.py` có quá nhiều logic không?
- Detector có leak Ultralytics object không?
- Video writer có đúng resolution/FPS không?
- Benchmark có tách stage không?
- Có kiểm tra output video mở lại được không?
- Có hard-coded path không hợp lý không?
- Có claim chưa test không?

## Severity

```text
BLOCKER
MAJOR
MINOR
```

Phải sửa hết BLOCKER và MAJOR trước khi PASS.
