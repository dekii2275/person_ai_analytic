# Skill: Benchmark

## Vai trò

Đo end-to-end baseline có ý nghĩa.

## Required stages

```text
decode_ms
preprocess_ms
inference_ms
postprocess_ms
draw_write_ms
total_ms
```

## Required statistics

```text
processed_frames
mean
median
p95
effective_fps
```

## Warmup

Nếu warmup:

- ghi rõ số frame;
- không trộn warmup vào benchmark chính.

## Required artifact

```text
outputs/benchmark.json
```

## Rules

- Không chỉ báo FPS của model.
- Không dùng số benchmark từ nguồn khác thay cho máy hiện tại.
- Không claim GPU benchmark nếu thực tế chạy CPU.
