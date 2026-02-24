"""
Local debug script — run this directly to iterate on the analyzer
without needing the phone to submit a video each time.

Usage:
    python test_tracking.py                          # uses debug_last_video.mp4
    python test_tracking.py path/to/video.mp4        # uses a specific video

It runs the full analyze_video pipeline and prints detailed diagnostics.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

BACKEND = Path(__file__).parent

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = str(BACKEND / "debug_last_video.mp4")

    if not Path(video_path).exists():
        print(f"[error] Video not found: {video_path}")
        print("  Submit a video from the app first — the server will save it as debug_last_video.mp4")
        sys.exit(1)

    print(f"[test] Video: {video_path}")
    out_path = str(BACKEND / "debug_test_output.mp4")

    # ── Quick YOLO sanity check on frame 0 before running full pipeline ──────
    cap = cv2.VideoCapture(video_path)
    ok, frame0 = cap.read()
    cap.release()

    if not ok or frame0 is None:
        print("[error] Could not read even one frame — OpenCV can't decode this file")
        sys.exit(1)

    print(f"[test] Frame0: shape={frame0.shape}  dtype={frame0.dtype}  mean={frame0.mean():.1f}")
    cv2.imwrite(str(BACKEND / "debug_test_frame0.jpg"), frame0)
    print("[test] Saved debug_test_frame0.jpg")

    print("[test] Running YOLO probe on frame0...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    for conf_thresh in [0.50, 0.25, 0.10, 0.05]:
        res = model(frame0, classes=[0], verbose=False, conf=conf_thresh)
        n = len(res[0].boxes) if res else 0
        confs = [round(float(c), 3) for c in res[0].boxes.conf] if n > 0 else []
        print(f"  conf>={conf_thresh}: {n} detections  {confs}")

    # ── Full pipeline ─────────────────────────────────────────────────────────
    print("\n[test] Running full analyze_video pipeline...")
    from analyzer import analyze_video
    stats = analyze_video(video_path, out_path, runway_meters=40.0)
    print("\n[test] Stats:", stats)

    import json
    debug = json.loads((BACKEND / "debug_last_run.json").read_text())
    non_null_heights = [h for h in debug["bbox_heights_px"] if h is not None]
    non_null_cx = [x for x in debug["centroid_x"] if x is not None]
    print(f"\n[test] Non-null bbox heights: {len(non_null_heights)}/{len(debug['bbox_heights_px'])}")
    print(f"[test] Non-null centroids:    {len(non_null_cx)}/{len(debug['centroid_x'])}")
    if non_null_heights:
        print(f"[test] Median bbox height: {np.median(non_null_heights):.1f}px")
    print(f"[test] avg_kmh={debug['avg_kmh']}  peak_kmh={debug['peak_kmh']}")


if __name__ == "__main__":
    main()
