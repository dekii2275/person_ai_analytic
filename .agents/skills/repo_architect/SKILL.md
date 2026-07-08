# Skill: Repo Architect

## Vai trò

Tạo cấu trúc repo tối thiểu, không over-engineer.

## Target structure

```text
project/
├── data/
│   └── input.mp4
├── outputs/
├── src/
│   ├── schemas.py
│   ├── video_source.py
│   ├── visualization.py
│   ├── benchmark.py
│   └── detectors/
│       ├── __init__.py
│       ├── base.py
│       └── yolov11.py
├── tests/
├── main.py
├── requirements.txt
├── environment.yml
└── README.md
```

## Rules

- `main.py` chỉ điều phối.
- Detector không đọc video.
- VideoSource không biết model.
- Visualization không phụ thuộc Ultralytics.
- Benchmark không phụ thuộc Ultralytics Results.
- Không tạo placeholder cho module tương lai.

## Done

Repo đủ khi có thể thay detector backend mà application flow gần như không đổi.
