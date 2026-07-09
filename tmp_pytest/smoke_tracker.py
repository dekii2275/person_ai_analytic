"""Quick smoke test for ByteTrackTracker before full integration."""
import sys
sys.path.insert(0, '.')

from src.schemas import Detection, Track
from src.trackers.bytetrack import ByteTrackTracker, ByteTrackConfig

print("=== Smoke test: ByteTrackTracker ===")

tracker = ByteTrackTracker()
print(f"Created: {tracker}")

# Test 1: empty detections
tracks = tracker.update([])
print(f"Test 1 (empty): {tracks}")
assert tracks == [], f"Expected [], got {tracks}"

# Test 2: single detection
det = Detection(x1=100, y1=200, x2=300, y2=600, score=0.9, class_id=0)
tracks = tracker.update([det])
print(f"Test 2 (single, frame 1): {tracks}")

# Test 3: same detection again (frame 2)
tracks2 = tracker.update([det])
print(f"Test 3 (same det, frame 2): {tracks2}")

if tracks2:
    t = tracks2[0]
    print(f"  track_id={t.track_id}, score={t.score:.3f}, class_id={t.class_id}")
    assert isinstance(t.track_id, int) and t.track_id >= 0
    assert 0.0 <= t.score <= 1.0
    assert t.class_id == 0

# Test 4: reset
tracker.reset()
tracks_after_reset = tracker.update([det])
print(f"Test 4 (after reset, frame 1 again): {tracks_after_reset}")

# Test 5: multiple detections
dets = [
    Detection(x1=10, y1=20, x2=100, y2=300, score=0.85, class_id=0),
    Detection(x1=200, y1=50, x2=350, y2=400, score=0.76, class_id=0),
]
tracker.reset()
# Run a few frames
for i in range(5):
    tracks = tracker.update(dets)
    print(f"Test 5 frame {i}: {[t.track_id for t in tracks]}")

print()
print("ALL SMOKE TESTS PASS")
