# AGENTS.md

## 1. Project Goal

Milestone hiện tại: **M04 — Person Attribute Recognition local baseline**.

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
→ person crop + lightweight body attribute inference
→ body AttributeObservation[]
→ stable TrackProfile[]
```

M01, M02, M03 và M12 local foundation đã có implementation, test và
artifact kiểm chứng. Toàn bộ task M12-01 đến M12-06 đã hoàn tất.

M04-00 và M04-01 đã hoàn tất: phạm vi milestone, contract và taxonomy đã
được chốt trước khi đánh giá hoặc tích hợp model. Target dài hạn vẫn là
NVIDIA Orin, nhưng M04 phải có local baseline chạy được khi chưa có thiết bị
Orin.

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
- framework-independent body attribute schema và recognizer interface;
- taxonomy thuộc tính toàn thân M04;
- person crop extraction và crop quality gate;
- đánh giá model PAR pretrained nhẹ;
- local body attribute inference;
- chuyển body attribute prediction thành `AttributeObservation`;
- tích hợp M04 sau M03 và trước temporal aggregation của M12;
- body attribute artifacts và latency benchmark riêng;
- visualization;
- output video;
- latency benchmark;
- tests/smoke tests;
- debugging;
- code review.

Không được phép:

- detector hoặc tracker backend mới;
- ReID;
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

Mọi body attribute recognizer phải trả về schema chung:

```python
BodyAttributePrediction(
    key: BodyAttributeKey,
    value: bool,
    score: float,
)
```

Không leak tensor, logits hoặc object backend-specific ra ngoài module
recognizer. M04 dùng person crop BGR `uint8` và không được thay đổi
`Detection`, `Track` hoặc schema M12 để phù hợp riêng với một model.

### M04 rules

- Namespace khi đưa observation vào M12 phải là `body`.
- Taxonomy baseline chỉ gồm `backpack`, `bag`, `hat` và `long_sleeve`.
- Giá trị của taxonomy baseline là boolean; confidence phải nằm trong `[0, 1]`.
- `glasses` chỉ được thêm sau khi model và chất lượng crop được đánh giá.
- Không đưa màu áo/quần vào M04; đó là phạm vi M05.
- Không inference mọi frame hoặc mọi track nếu crop không đạt quality gate.
- Timestamp và frame index của observation phải lấy từ `VideoSource`.
- Backend local phải nằm sau interface chung và không được leak object model.
- Model, weights, license và label mapping phải được kiểm chứng trước tích hợp.
- Contract và taxonomy phải có unit test trước khi thêm model backend.

### M04 task sequence

1. **M04-00 — Scope and roadmap**
   - Chuyển milestone hiện tại từ M12 sang M04.
   - Mở đúng phạm vi body attributes và giữ các module khác ngoài scope.
   - Đồng bộ trạng thái và bước tiếp theo trong tài liệu kế hoạch.

2. **M04-01 — Contracts and taxonomy**
   - Chốt namespace `body`, schema `BodyAttributePrediction` và interface
     `BaseBodyAttributeRecognizer`.
   - Chốt taxonomy boolean baseline: `backpack`, `bag`, `hat`,
     `long_sleeve`.
   - Chốt validation, immutability và JSON shape bằng unit test.

3. **M04-02 — Model evaluation**
   - So sánh model PAR pretrained nhẹ theo label coverage, license,
     preprocessing, latency CPU và khả năng chuyển backend sau này.
   - Dùng tập person crop có ground truth thủ công.
   - Chưa tích hợp pipeline trong task này.

4. **M04-03 — Crop and quality gate**
   - Clip person bbox, reject crop không hợp lệ hoặc quá nhỏ.
   - Tính quality score deterministic và test các trường hợp biên.

5. **M04-04 — Local model adapter**
   - Implement preprocessing, inference và label mapping sau interface chung.
   - Unit test postprocessing và smoke test bằng weights thật.

6. **M04-05 — M03/M04/M12 integration**
   - Dùng scheduling của M12 để giới hạn inference theo Track ID.
   - Chuyển prediction thành `AttributeObservation` với timestamp thật.
   - Temporal voting tạo stable body attributes.

7. **M04-06 — Artifacts and benchmark**
   - Xuất stable body attributes trong track profile artifact.
   - Báo latency crop, M04 inference và M12 aggregation riêng.
   - Smoke test, parse JSON và mở lại output video bằng code.

### M04 acceptance criteria

- Toàn bộ regression test M01–M03 và M12 vẫn pass.
- Unit test cover taxonomy, schema validation, immutability và JSON shape.
- M04 contract không import backend inference cụ thể.
- Crop/quality gate và inference interval deterministic, có test biên.
- Số lần model inference nhỏ hơn số track-frame đủ điều kiện trên video smoke.
- Stable body attributes chỉ xuất hiện sau minimum observations của M12.
- JSON artifact parse được và benchmark báo latency M04 riêng.
- Có đánh giá bằng ground truth thủ công; không tuyên bố accuracy chỉ từ
  kiểm tra trực quan.
- Không thêm clothing color, face attributes, ReID hoặc backend production.

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

Không nói "M04 hoàn thành" nếu chưa chạy unit test contract/crop/mapping,
integration test với M12, smoke test bằng weights thật và kiểm tra artifact,
latency cùng inference scheduling.

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
