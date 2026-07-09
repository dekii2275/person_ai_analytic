"""Visual review script — reads tracking_tracks.json and reports track IDs at parity frames and range transitions."""
import json
import sys

with open("outputs/tracking_tracks.json") as f:
    data = json.load(f)

# Index by frame
by_frame = {entry["frame_index"]: entry["tracks"] for entry in data}

# Parity frames
parity_frames = [0, 30, 60, 79, 90, 120, 155, 180, 196, 258]
print("=== PARITY FRAMES ===")
for fi in parity_frames:
    tracks = by_frame.get(fi, [])
    ids = [t["track_id"] for t in tracks]
    bboxes = [t["bbox"] for t in tracks]
    scores = [round(t["score"], 3) for t in tracks]
    print(f"  frame {fi:4d}: IDs={ids}  scores={scores}")
print()

# Range transitions
ranges = [(0, 60), (70, 100), (110, 130), (145, 165), (180, 210), (240, 258)]
print("=== RANGE ANALYSIS ===")
for (start, end) in ranges:
    print(f"\n  Range {start}-{end}:")
    prev_ids = None
    for fi in range(start, end + 1):
        tracks = by_frame.get(fi, [])
        ids = frozenset(t["track_id"] for t in tracks)
        if prev_ids is not None and ids != prev_ids:
            added = ids - prev_ids
            removed = prev_ids - ids
            if added or removed:
                print(f"    frame {fi:4d}: CHANGE  +{sorted(added)} -{sorted(removed)}  (now: {sorted(ids)})")
        elif fi in [start, end]:
            print(f"    frame {fi:4d}: IDs={sorted(ids)}")
        prev_ids = ids

print()

# Overall ID switch analysis
print("=== TRACK CONTINUITY SUMMARY ===")
# Find where each track_id first and last appears
track_spans = {}
for entry in data:
    fi = entry["frame_index"]
    for t in entry["tracks"]:
        tid = t["track_id"]
        if tid not in track_spans:
            track_spans[tid] = [fi, fi]
        else:
            track_spans[tid][1] = fi

for tid in sorted(track_spans):
    start, end = track_spans[tid]
    duration = end - start + 1
    print(f"  Track {tid}: frames {start:4d}-{end:4d}  ({duration} frames)")

print()
print("=== FRAME 155 ANALYSIS (known false positive region) ===")
for fi in range(150, 162):
    tracks = by_frame.get(fi, [])
    ids = [t["track_id"] for t in tracks]
    scores = [round(t["score"], 3) for t in tracks]
    print(f"  frame {fi:4d}: IDs={ids}  scores={scores}")
