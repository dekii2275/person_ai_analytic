# AGENTS.md

## 1. Project Goal

Milestone vừa hoàn tất: **M12 — Track Attribute Manager foundation**.

```text
data/input.mp4
→ YOLO11n local inference
→ keep person class only
→ Detection[]
→ ByteTrack
→ Track[]
→ state keyed by Track ID
→ bounded trajectory + lifecycle
→ attribute cache + temporal voting
→ stable TrackProfile[]
```

M01, M02, M03 và M12 local foundation đã có implementation, test và
artifact kiểm chứng. Toàn bộ task M12-01 đến M12-06 đã hoàn tất.

Target dài hạn là NVIDIA Orin. M12 hiện hoàn tất ở mức local foundation và
chưa thêm model thuộc tính mới.

## 2. Current Scope

Được phép:

- environment setup;
- repo skeleton;
- MP4 video reading;
- YOLO11n person detection;
- framework-independent Detection schema;
- ByteTrack person tracking;
- framework-independent Track schema;
- per-frame track history;
- state management keyed by Track ID;
- track lifecycle: active, lost, removed;
- first/last seen timestamps, age và bounded trajectory;
- cache kết quả thuộc tính và thời điểm inference cuối;
- generic attribute observations cho unit test;
- temporal voting / weighted aggregation;
- cleanup theo TTL và giới hạn bộ nhớ;
- stable TrackProfile schema;
- visualization;
- output video;
- latency benchmark;
- tests/smoke tests;
- debugging;
- code review.

Không được phép:

- detector hoặc tracker backend mới;
- ReID;
- body attribute model / person attribute inference;
- clothing color;
- face pipeline;
- crowd/behavior;
- Event Manager;
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

Mọi tracker phải trả về schema chung:

```python
Track(
    track_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float,
    class_id: int,
)
```

Không leak `BYTETracker`, `STrack` hoặc object backend-specific ra ngoài
tracker module.

M12 chỉ nhận schema framework-independent (`Track`, timestamp và attribute
observation). M12 không được import `ultralytics`, `torch` hoặc model cụ thể.

### M12 state rules

- State phải được key bằng `track_id`.
- Timestamp phải lấy từ `VideoSource`, không tự dùng wall-clock time.
- Trajectory phải có giới hạn cấu hình; không được tăng vô hạn.
- Track không xuất hiện phải chuyển lifecycle theo quy tắc xác định và test được.
- Track quá TTL phải được cleanup.
- Temporal voting phải deterministic với cùng input/config.
- Attribute cache không được tự chạy inference; model bên ngoài cung cấp observation.
- Schema và lifecycle contract phải được chốt bằng unit test trước khi tích hợp pipeline.

### M12 task sequence — hoàn tất

Các task đã được hoàn tất theo thứ tự:

1. **M12-01 — Contracts and schemas**
   - Chốt `TrackLifecycle`, `TrajectoryPoint`, `AttributeObservation`,
     `StableAttribute` và `TrackProfile`.
   - Chốt validation, immutability/copy semantics và JSON shape.
   - Chưa viết manager hoặc tích hợp pipeline.

2. **M12-02 — Lifecycle state manager**
   - Tạo, cập nhật và truy vấn state theo `track_id`.
   - Chuyển trạng thái `active → lost → removed`.
   - Hỗ trợ `reset`, TTL và cleanup deterministic.
   - Reject frame index hoặc timestamp đi lùi.

3. **M12-03 — Bounded trajectory**
   - Lưu bottom-center point kèm frame index và timestamp.
   - Giới hạn số point bằng config.
   - Cập nhật `first_seen_ms`, `last_seen_ms`, `age_frames`,
     `observed_frames` và `missed_frames`.
   - Không thêm crowd/behavior analytics.

4. **M12-04 — Attribute cache and scheduling**
   - Cache observation theo namespace/key.
   - Lưu thời điểm inference cuối.
   - Cung cấp quyết định `should_infer` dựa trên interval và quality gate.
   - Không import hoặc gọi model.

5. **M12-05 — Temporal voting**
   - Voting theo cửa sổ cấu hình và confidence weight.
   - Có minimum observations và tie-break rule deterministic.
   - Một vài frame nhiễu không được làm kết quả ổn định flip ngay.

6. **M12-06 — Pipeline integration and artifacts**
   - Nối manager sau `ByteTrackTracker`.
   - Dùng timestamp thật từ `VideoSource`.
   - Xuất JSON track profile/state có schema version.
   - Đo latency M12 riêng.
   - Smoke test trên `data/input.mp4` và MOT17-09.

### M12 acceptance criteria

- Toàn bộ test M01–M03 hiện có vẫn pass.
- Unit test M12 cover schema validation, lifecycle, TTL, reset, trajectory bound,
  cache interval, quality gate, voting và deterministic tie-break.
- Integration test dùng track history thật, không chỉ mock.
- Không có import `ultralytics`, `torch`, detector hoặc attribute model trong M12.
- Trajectory và cache có bound cấu hình; test chứng minh không tăng vô hạn.
- JSON artifact parse được, có schema version và số frame/profile hợp lệ.
- Benchmark báo latency M12 riêng, không gộp mơ hồ vào detector/tracker.

### Local first, Orin aware

Code hiện tại phải cho phép sau này thay:

```text
YOLO11 PyTorch backend
→ YOLO11 TensorRT backend
```

mà không viết lại:

- VideoSource;
- Detection schema;
- Track schema;
- Track Attribute Manager;
- visualization;
- benchmark;
- application flow.

### Evidence before success

Không nói "đã hoàn thành" nếu chưa chạy command thật.

Không nói "video output hợp lệ" nếu chưa mở lại bằng code.

Không nói "benchmark xong" nếu chưa sinh file thật.

Không nói "M12 hoàn thành" nếu chưa chạy unit test lifecycle/cache/voting,
integration test với track history thật và kiểm tra cleanup/bounded memory.

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
