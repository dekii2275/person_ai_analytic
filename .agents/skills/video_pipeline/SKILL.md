# Skill: Video Pipeline

## Vai trò

Đọc đúng `data/input.mp4`.

## Current scope

Chỉ local MP4.

Không RTSP, webcam, threading, async.

## Required iterator output

```text
frame_index
timestamp_ms
frame
```

## Checks

- file tồn tại;
- video mở được;
- FPS hợp lệ;
- width/height hợp lệ;
- frame đầu đọc được;
- resource được release.

## Output writer checks

- đúng resolution gốc;
- FPS hợp lệ;
- writer mở thành công;
- output video mở lại được sau khi ghi.

## Benchmark

Decode timing phải tách khỏi model inference.
