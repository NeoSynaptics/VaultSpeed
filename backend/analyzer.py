"""
VaultSpeed — Core CV pipeline
Analyzes pole vault approach video: tracks runner, computes speed, overlays bar.

Smart trimming:
  - Full video fed in (may include idle time before/after run)
  - Pole plant detected via abrupt velocity drop → END marker
  - Run onset detected by scanning backwards from plant → START marker
  - Only the approach window is output (+ small tail after plant)
"""

import cv2
import numpy as np
from typing import Optional


# ─── Color palette (BGR) ──────────────────────────────────────────────────────
RED    = (0,   50,  220)
YELLOW = (0,  200,  220)
GREEN  = (50, 200,   50)

NUM_BUCKETS   = 10
BAR_HEIGHT    = 44
ONSET_PCT     = 0.20   # velocity below this % of peak = "not yet running"
PLANT_DROP    = 0.40   # velocity drops to this % of peak = pole plant
START_BUFFER  = 2.0    # seconds of buffer before detected onset
TAIL_BUFFER   = 0.5    # seconds to keep after plant (shows the plant itself)


def speed_color(ratio: float) -> tuple:
    """ratio = v / peak_v in this run (0–1)"""
    if ratio < 0.70:
        return RED
    elif ratio < 0.85:
        return YELLOW
    return GREEN


# ─── YOLO person tracker ──────────────────────────────────────────────────────
class RunnerTracker:
    """Detects person per frame, returns smoothed centroid list."""

    def __init__(self):
        self._model = None

    def _load_model(self):
        from ultralytics import YOLO
        self._model = YOLO("yolov8n.pt")

    def track(self, frames: list[np.ndarray]) -> list[Optional[tuple[float, float]]]:
        if self._model is None:
            self._load_model()

        raw: list[Optional[tuple[float, float]]] = []
        for frame in frames:
            res = self._model(frame, classes=[0], verbose=False)
            if res and len(res[0].boxes) > 0:
                boxes = res[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                b = boxes[np.argmax(areas)]
                raw.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            else:
                raw.append(None)

        return self._fill_gaps(raw)

    @staticmethod
    def _fill_gaps(pts: list) -> list:
        """Linear interpolation for missing detections."""
        result = list(pts)
        n = len(result)
        for i in range(n):
            if result[i] is not None:
                continue
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


# ─── Velocity computation ─────────────────────────────────────────────────────
def compute_velocities(centroids: list, fps: float, step: int = 3) -> list[float]:
    """px/second at each frame using step-frame difference."""
    n = len(centroids)
    v = [0.0] * n
    for i in range(step, n - step):
        p0, p1 = centroids[i - step], centroids[i + step]
        if p0 and p1:
            v[i] = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (2 * step / fps)
    return v


def smooth(v: list[float], window: int = 9) -> list[float]:
    arr = np.array(v, dtype=float)
    k = np.ones(window) / window
    return np.convolve(arr, k, mode="same").tolist()


# ─── Smart run window detection ───────────────────────────────────────────────
def find_run_window(velocities: list[float], fps: float) -> tuple[int, int]:
    """
    Returns (start_frame, end_frame) that tightly wraps the approach run.

    Algorithm:
      END  — find global peak velocity (must be in last 80% of video to skip
             camera-setup movement). Then scan forward until velocity drops
             below PLANT_DROP * peak → pole plant.
      START — scan backward from end_frame; find the last frame where velocity
              was below ONSET_PCT * peak (athlete still standing). Add 2s buffer.
    """
    s = smooth(velocities, window=9)
    n = len(s)

    if n == 0:
        return 0, n

    # ── Find pole plant (END) ─────────────────────────────────────────────────
    skip = max(1, n // 5)               # ignore first 20% (camera setup, etc.)
    peak_idx = skip + int(np.argmax(s[skip:]))
    peak_v   = s[peak_idx]

    if peak_v < 1.0:                    # no real motion detected
        return 0, n

    drop_thresh = peak_v * PLANT_DROP
    end_frame = n
    for i in range(peak_idx, n):
        if s[i] < drop_thresh:
            end_frame = i
            break

    # ── Find run onset (START) ────────────────────────────────────────────────
    onset_thresh = peak_v * ONSET_PCT
    onset_frame  = 0                    # fallback: beginning of video
    for i in range(end_frame - 1, -1, -1):
        if s[i] < onset_thresh:
            onset_frame = i             # last "still" frame before run
            break

    buf = int(fps * START_BUFFER)
    start_frame = max(0, onset_frame - buf)

    tail = int(fps * TAIL_BUFFER)
    end_with_tail = min(n, end_frame + tail)

    print(
        f"[window] peak@{peak_idx} ({peak_v:.0f}px/s)  "
        f"onset@{onset_frame}  plant@{end_frame}  "
        f"window=[{start_frame}, {end_with_tail}]  "
        f"({(end_with_tail - start_frame) / fps:.1f}s)"
    )
    return start_frame, end_with_tail


# ─── px → km/h calibration ───────────────────────────────────────────────────
def calibrate_px_per_meter(centroids: list, start: int, end: int,
                            runway_meters: float) -> float:
    """Map total pixel displacement in [start, end] to runway_meters."""
    valid = [c for c in centroids[start:end] if c is not None]
    if len(valid) < 2:
        return 1.0
    total_px = sum(
        np.hypot(valid[i+1][0] - valid[i][0], valid[i+1][1] - valid[i][1])
        for i in range(len(valid) - 1)
    )
    return max(total_px / runway_meters, 0.001)


# ─── Speed bar overlay ────────────────────────────────────────────────────────
def _build_buckets(velocities_kmh: list[float], plant_local: int) -> list[float]:
    """10 buckets from run start to pole plant (local indices)."""
    run = velocities_kmh[:plant_local] if plant_local < len(velocities_kmh) else velocities_kmh
    if not run:
        return [0.0] * NUM_BUCKETS
    chunk = max(1, len(run) // NUM_BUCKETS)
    return [
        float(np.mean(run[i * chunk:(i + 1) * chunk]) or 0)
        for i in range(NUM_BUCKETS)
    ]


def draw_bar(
    frame: np.ndarray,
    buckets: list[float],
    frame_local: int,
    plant_local: int,
    avg_kmh: float,
    delta_kmh: Optional[float],
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_y = h - BAR_HEIGHT

    # Dark background strip
    cv2.rectangle(frame, (0, bar_y), (w, h), (18, 18, 18), -1)

    max_v = max(buckets) if max(buckets) > 0 else 1.0
    seg_w = w // NUM_BUCKETS

    for i, bv in enumerate(buckets):
        color = speed_color(bv / max_v)
        x0 = i * seg_w
        x1 = x0 + seg_w - 3
        cv2.rectangle(frame, (x0, bar_y + 5), (x1, h - 5), color, -1)

    # Progress cursor
    if plant_local > 0:
        pct = min(frame_local / plant_local, 1.0)
        cx  = int(pct * w)
        cv2.line(frame, (cx, bar_y + 2), (cx, h - 2), (255, 255, 255), 2)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55

    # km/h left
    cv2.putText(frame, f"{avg_kmh:.1f} km/h",
                (8, bar_y + 28), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Δ right
    if delta_kmh is not None:
        sign = "+" if delta_kmh >= 0 else ""
        txt  = f"{sign}{delta_kmh:.1f} km/h"
        col  = GREEN if delta_kmh >= 0 else RED
        tw   = cv2.getTextSize(txt, font, scale, 1)[0][0]
        cv2.putText(frame, txt, (w - tw - 8, bar_y + 28),
                    font, scale, col, 1, cv2.LINE_AA)

    return frame


# ─── Main pipeline ────────────────────────────────────────────────────────────
def analyze_video(
    input_path: str,
    output_path: str,
    runway_meters: float = 40.0,
    prev_avg_kmh: Optional[float] = None,
) -> dict:
    """
    Full pipeline on input video.
    1. Track runner across ALL frames (so window detection works even if video
       starts before the run).
    2. Detect run window [start, end] automatically.
    3. Output only trimmed frames with speed bar overlay.

    Returns stats dict.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[analyzer] {total} frames @ {fps:.1f}fps  {width}x{height}")

    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    if not frames:
        raise ValueError("No frames read")

    # ── Track across full video ───────────────────────────────────────────────
    tracker   = RunnerTracker()
    centroids = tracker.track(frames)

    # ── Velocity timeline ─────────────────────────────────────────────────────
    vel_px = smooth(compute_velocities(centroids, fps))

    # ── Auto-detect run window ────────────────────────────────────────────────
    win_start, win_end = find_run_window(vel_px, fps)

    # Pole plant position relative to window start (for bar cursor)
    plant_abs   = win_end - int(fps * TAIL_BUFFER)  # end before tail
    plant_local = plant_abs - win_start

    # ── Calibrate px → km/h using only the run window ────────────────────────
    px_per_m   = calibrate_px_per_meter(centroids, win_start, plant_abs, runway_meters)
    vel_kmh_all = [v / px_per_m * 3.6 for v in vel_px]

    # Stats: just the run portion (start → plant)
    run_kmh = [v for v in vel_kmh_all[win_start:plant_abs] if v > 0]
    avg_kmh  = float(np.mean(run_kmh))  if run_kmh else 0.0
    peak_kmh = float(np.max(run_kmh))   if run_kmh else 0.0
    delta    = (avg_kmh - prev_avg_kmh) if prev_avg_kmh is not None else None

    # ── Build buckets from the run window ─────────────────────────────────────
    vel_kmh_local = vel_kmh_all[win_start:win_end]
    buckets = _build_buckets(vel_kmh_local, plant_local)

    # ── Write trimmed + annotated output ─────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trimmed = frames[win_start:win_end]
    for local_i, frame in enumerate(trimmed):
        frame = draw_bar(frame, buckets, local_i, plant_local, avg_kmh, delta)
        out.write(frame)

    out.release()

    trimmed_s = len(trimmed) / fps
    print(
        f"[analyzer] output={trimmed_s:.1f}s  "
        f"avg={avg_kmh:.1f} km/h  peak={peak_kmh:.1f} km/h"
    )

    return {
        "avg_kmh":         round(avg_kmh, 2),
        "peak_kmh":        round(peak_kmh, 2),
        "delta_kmh":       round(delta, 2) if delta is not None else None,
        "pole_plant_frame": plant_local,
        "total_frames":    len(trimmed),
        "fps":             fps,
        "trimmed_seconds": round(trimmed_s, 1),
        "original_frames": total,
    }


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python analyzer.py input.mp4 output.mp4 [runway_meters]")
        sys.exit(1)
    stats = analyze_video(sys.argv[1], sys.argv[2],
                          runway_meters=float(sys.argv[3]) if len(sys.argv) > 3 else 40.0)
    print("Stats:", stats)
