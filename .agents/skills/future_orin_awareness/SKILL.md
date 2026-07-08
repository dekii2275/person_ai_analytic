# Skill: Future Orin Awareness

## Vai trò

Chỉ kiểm tra kiến trúc hiện tại có cản trở deployment sau này hay không.

Không triển khai Orin ở milestone hiện tại.

## Future path

```text
YOLO11n PyTorch
→ ONNX
→ TensorRT
→ NVIDIA Orin
→ DeepStream/GStreamer production pipeline
```

## Current design constraints

Giữ độc lập:

- VideoSource;
- Detection schema;
- visualization;
- benchmark;
- application flow.

Sau này chỉ thay detector backend.

## Forbidden now

- ONNX export;
- TensorRT code;
- DeepStream config;
- JetPack-specific install;
- cross compile.
