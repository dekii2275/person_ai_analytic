# Skill: Environment Setup

## Vai trò

Chuẩn bị môi trường local sạch cho YOLO11n baseline.

## Required environment

```text
Conda env: orin_person
Python: 3.10
OS: Linux
```

## Workflow

1. Kiểm tra Conda.
2. Kiểm tra `nvidia-smi`.
3. Tạo env nếu chưa có.
4. Cài PyTorch phù hợp với GPU local.
5. Xác minh `torch.cuda.is_available()`.
6. Cài dependency tối thiểu.
7. Ghi lại version thực tế.

## Rules

- Không reuse env cũ có nhiều package không liên quan.
- Không cài TensorRT/DeepStream.
- Không upgrade hàng loạt package để "thử".
- Không đoán PyTorch CUDA wheel.
- Không thay system Python.

## Acceptance criteria

```text
python --version
→ 3.10.x

import torch
→ success

torch.cuda.is_available()
→ True nếu local GPU được sử dụng

import ultralytics
→ success

import cv2
→ success
```

## Required output

Agent phải báo cáo:

```text
Python version
Torch version
CUDA available
GPU name
Ultralytics version
OpenCV version
```
