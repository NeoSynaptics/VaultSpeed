"""
VaultSpeed — Core CV pipeline
Analyzes pole vault approach video: tracks runner, computes speed, overlays bar.
"""

import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional
import tempfile
import os


# ─── Color map ───────────────────────────────────────────────────────────────
# BGR colors
RED    = (0,   50,  220)
YELLOW = (0,   200, 220)
GREEN  = (50,  200, 50)

def speed_color(ratio: float) -> tuple:
    """ratio = fraction of max speed in this run (0–1)"""
    if ratio < 0.70:
        return RED
    elif ratio < 0.85:
        return YELLOW
    return GREEN


# ─── Person tracker ──────────────────────────────────────────────────────────
class RunnerTracker:
    def __init__(self, smooth_window: int = 5):
        self.positions: list[tuple[float, float]] = []
        self.frame_indices: list[int] = []
        self._smooth_window = smooth_window
        self._model = None

    def _load_model(self):
        from ultralytics import YOLO
        self._model = YOLO("yolov8n.pt")

    def track(self, frames: list[np.ndarray]) -> list[Optional[tuple[float, float]]]:
        """Returns list of (cx, cy) centroids per frame, None if not detected."""
        if self._model is None:
            self._load_model()

        centroids: list[Optional[tuple[float, float]]] = []
        for frame in frames:
            results = self._model(frame, classes=[0], verbose=False)  # class 0 = person
            if results and len(results[0].boxes) > 0:
                # Pick the largest bounding box (closest runner)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best = boxes[np.argmax(areas)]
                cx = (best[0] + best[2]) / 2
                cy = (best[1] + best[3]) / 2
                centroids.append((float(cx), float(cy)))
            else:
                centroids.append(None)

        return self._fill_gaps(centroids)

    def _fill_gaps(self, centroids):
        """Linear interpolation for missing detections."""
        result = list(centroids)
        n = len(result)
        for i in range(n):
            if result[i] is None:
                # Find next valid
                j = i + 1
                while j < n and result[j] is None:
                    j += 1
                if i > 0 and j < n:
                    x0, y0 = result[i - 1]
                    x1, y1 = result[j]
                    for k in range(i, j):
                        t = (k - i + 1) / (j - i + 1)
                        result[k] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
                elif j < n:
                    result[i] = result[j]
                elif i > 0:
                    result[i] = result[i - 1]
        return result


# ─── Speed computation ───────────────────────────────────────────────────────
def compute_velocities(centroids: list[tuple], fps: float, step: int = 3) -> list[float]:
    """
    Velocity in pixels/second at each frame.
    Uses step-frame difference to reduce noise.
    """
    n = len(centroids)
    velocities = [0.0] * n
    for i in range(step, n - step):
        p0 = centroids[i - step]
        p1 = centroids[i + step]
        if p0 and p1:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dist = np.sqrt(dx**2 + dy**2)
            velocities[i] = dist / (2 * step / fps)
    return velocities


def smooth_velocities(velocities: list[float], window: int = 7) -> list[float]:
    arr = np.array(velocities)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same").tolist()


def detect_pole_plant(velocities: list[float], fps: float) -> int:
    """
    Find frame index where velocity drops > 50% within 5 frames.
    Returns end frame index (exclusive), or len(velocities) if not found.
    """
    smoothed = smooth_velocities(velocities, window=5)
    n = len(smoothed)
    window = max(3, int(fps * 0.15))  # ~0.15s window

    for i in range(n - window):
        peak = max(smoothed[i:i + window])
        valley = min(smoothed[i:i + window])
        if peak > 0 and valley / peak < 0.50:
            # Confirm: this isn't just start of run
            if i > n * 0.3:
                return i + int(window / 2)

    return n  # no plant detected


def calibrate(centroids: list[tuple], end_frame: int, runway_meters: float) -> float:
    """
    px_per_meter: total pixel displacement across run / runway_meters
    """
    valid = [c for c in centroids[:end_frame] if c is not None]
    if len(valid) < 2:
        return 1.0  # fallback: no calibration

    total_px = sum(
        np.sqrt((valid[i+1][0] - valid[i][0])**2 + (valid[i+1][1] - valid[i][1])**2)
        for i in range(len(valid) - 1)
    )
    if total_px < 1:
        return 1.0
    return total_px / runway_meters


# ─── Speed bar overlay ───────────────────────────────────────────────────────
NUM_BUCKETS = 10
BAR_HEIGHT = 40
SPEED_TEXT_Y_OFFSET = 20


def build_buckets(velocities_kmh: list[float], end_frame: int) -> list[float]:
    """Divide run into NUM_BUCKETS, return avg km/h per bucket."""
    run = velocities_kmh[:end_frame]
    if not run:
        return [0.0] * NUM_BUCKETS
    chunk = max(1, len(run) // NUM_BUCKETS)
    buckets = []
    for i in range(NUM_BUCKETS):
        sl = run[i * chunk: (i + 1) * chunk]
        buckets.append(float(np.mean(sl)) if sl else 0.0)
    return buckets


def draw_bar(frame: np.ndarray, buckets: list[float], current_frame: int,
             end_frame: int, total_frames: int,
             avg_kmh: float, delta_kmh: Optional[float]) -> np.ndarray:
    """Burn speed bar onto frame. Returns modified frame."""
    h, w = frame.shape[:2]
    bar_y = h - BAR_HEIGHT

    # Background strip
    cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)

    # Draw 10 colored segments
    seg_w = w // NUM_BUCKETS
    max_v = max(buckets) if max(buckets) > 0 else 1.0

    for i, bv in enumerate(buckets):
        ratio = bv / max_v
        color = speed_color(ratio)
        x0 = i * seg_w
        x1 = x0 + seg_w - 2
        cv2.rectangle(frame, (x0, bar_y + 4), (x1, h - 4), color, -1)

    # Progress cursor — which bucket we're in
    if end_frame > 0:
        progress = min(current_frame / end_frame, 1.0)
        cursor_x = int(progress * w)
        cv2.line(frame, (cursor_x, bar_y), (cursor_x, h), (255, 255, 255), 2)

    # Speed text
    font = cv2.FONT_HERSHEY_SIMPLEX
    speed_str = f"{avg_kmh:.1f} km/h"
    cv2.putText(frame, speed_str, (8, bar_y + SPEED_TEXT_Y_OFFSET),
                font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Delta text
    if delta_kmh is not None:
        sign = "+" if delta_kmh >= 0 else ""
        delta_str = f"{sign}{delta_kmh:.1f} km/h"
        color_d = GREEN if delta_kmh >= 0 else RED
        text_size = cv2.getTextSize(delta_str, font, 0.55, 1)[0]
        cv2.putText(frame, delta_str, (w - text_size[0] - 8, bar_y + SPEED_TEXT_Y_OFFSET),
                    font, 0.55, color_d, 1, cv2.LINE_AA)

    return frame


# ─── Main pipeline ────────────────────────────────────────────────────────────
def analyze_video(
    input_path: str,
    output_path: str,
    runway_meters: float = 40.0,
    prev_avg_kmh: Optional[float] = None,
) -> dict:
    """
    Full pipeline: track → speed → overlay → write output video.
    Returns stats dict: {avg_kmh, peak_kmh, pole_plant_frame, total_frames}
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[analyzer] {total_frames} frames @ {fps:.1f}fps  {width}x{height}")

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames read from video")

    # Track
    tracker = RunnerTracker()
    centroids = tracker.track(frames)

    # Velocities (px/s)
    velocities_px = compute_velocities(centroids, fps)
    velocities_px = smooth_velocities(velocities_px)

    # Pole plant detection
    end_frame = detect_pole_plant(velocities_px, fps)
    print(f"[analyzer] Pole plant detected at frame {end_frame}/{total_frames}")

    # Calibrate px → m
    px_per_meter = calibrate(centroids, end_frame, runway_meters)
    velocities_ms  = [v / px_per_meter for v in velocities_px]
    velocities_kmh = [v * 3.6 for v in velocities_ms]

    # Stats
    run_speeds = [v for v in velocities_kmh[:end_frame] if v > 0]
    avg_kmh  = float(np.mean(run_speeds)) if run_speeds else 0.0
    peak_kmh = float(np.max(run_speeds))  if run_speeds else 0.0
    delta_kmh = (avg_kmh - prev_avg_kmh) if prev_avg_kmh is not None else None

    # Buckets for bar
    buckets = build_buckets(velocities_kmh, end_frame)

    # Write output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        frame = draw_bar(frame, buckets, i, end_frame, total_frames, avg_kmh, delta_kmh)
        out.write(frame)

    out.release()
    print(f"[analyzer] Done → {output_path}  avg={avg_kmh:.1f} km/h  peak={peak_kmh:.1f} km/h")

    return {
        "avg_kmh": round(avg_kmh, 2),
        "peak_kmh": round(peak_kmh, 2),
        "delta_kmh": round(delta_kmh, 2) if delta_kmh is not None else None,
        "pole_plant_frame": end_frame,
        "total_frames": total_frames,
        "fps": fps,
    }


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python analyzer.py input.mp4 output.mp4 [runway_meters]")
        sys.exit(1)
    inp  = sys.argv[1]
    outp = sys.argv[2]
    rm   = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
    stats = analyze_video(inp, outp, runway_meters=rm)
    print("Stats:", stats)
