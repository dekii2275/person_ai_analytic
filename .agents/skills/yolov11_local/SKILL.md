# Skill: YOLO11n Local Person Detection

## Vai trò

Implement baseline YOLO11n local.

## Exact pipeline

```text
frame
→ YOLO11n
→ keep class person only
→ convert to Detection[]
```

## Model

```text
YOLO11n
```

Không tự đổi model.

## Common schema

```python
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
```

## Rules

- Chỉ giữ `person`.
- Không trả raw Ultralytics objects.
- Coordinate phải map về frame gốc.
- Ghi rõ confidence threshold.
- Không thêm tracking.
- Không export ONNX trong task này.

## Cases to inspect

- người rất gần camera;
- partial body;
- người nhỏ;
- motion blur;
- quay lưng;
- empty frame false positive.
