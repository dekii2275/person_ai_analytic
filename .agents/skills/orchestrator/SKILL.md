# Skill: Orchestrator

## Vai trò

Điều phối Coding Agent theo milestone nhỏ.

Không cho Agent làm toàn bộ pipeline trong một lần.

## Milestone order

```text
M0 — Environment + repo skeleton
M1 — VideoSource đọc input.mp4
M2 — YOLO11n person-only local inference
M3 — Draw + save output.mp4
M4 — benchmark.json + terminal summary
```

Không nhảy milestone.

## Quy tắc chia task

Một task tốt:

- mục tiêu duy nhất;
- sửa ít file;
- có command chạy;
- có acceptance criteria;
- có output kiểm chứng.

Task xấu:

```text
setup + code detector + benchmark + export ONNX + deploy Orin
```

## Handoff format

```text
Context:
Current milestone:
Exact task:
Files to inspect:
Files allowed to edit:
Acceptance criteria:
Commands to run:
Out of scope:
```

## Gate

Không chuyển bước nếu milestone trước chưa có bằng chứng chạy thật.
