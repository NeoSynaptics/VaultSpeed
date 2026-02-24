"""
VaultSpeed — Core CV pipeline
Analyzes pole vault approach video: tracks runner, computes speed, overlays bar.

Speed calibration uses the athlete's own bounding-box height as the px→m scale
factor (assumes ~1.80 m tall vaulter).  This is robust to camera distance,
zoom level, and how much of the runway is visible — unlike the old approach that
assumed the athlete covered a fixed runway length.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional

DEBUG_DIR = Path(__file__).parent   # saves debug_last_run.json + debug_peak_frame.jpg here


# ─── Color palette (BGR) ──────────────────────────────────────────────────────
RED    = (0,   50,  220)
YELLOW = (0,  200,  220)
GREEN  = (50, 200,   50)

NUM_BUCKETS      = 10
BAR_HEIGHT       = 44
ONSET_PCT        = 0.20   # velocity below this % of peak = "not yet running"
PLANT_DROP       = 0.40   # velocity drops to this % of peak = pole plant
START_BUFFER     = 2.0    # seconds of buffer before detected onset
TAIL_BUFFER      = 0.5    # seconds to keep after plant (shows the plant itself)
ATHLETE_HEIGHT_M = 1.81   # Duplantis height (m) — temporary test value


def speed_color(ratio: float) -> tuple:
    """ratio = v / peak_v in this run (0-1)
    Green = building (early run), yellow = mid, red = peak/plant (committed)"""
    if ratio < 0.70:
        return GREEN
    elif ratio < 0.85:
        return YELLOW
    return RED


# ─── YOLO person tracker ──────────────────────────────────────────────────────
class RunnerTracker:
    """
    Tracks the primary runner using YOLO with ID persistence.
    Returns centroids AND bounding-box heights for scale calibration.
    """

    def __init__(self):
        self._model = None

    def _load_model(self):
        from ultralytics import YOLO
        self._model = YOLO("yolov8n.pt")

    def track(self, frames: list[np.ndarray]) -> tuple[list, list, list]:
        """
        Returns (centroids, bbox_heights, bboxes).

        Two-phase tracking:
        Phase 1 (probe): Run YOLO on the first N_PROBE frames.  For each
          candidate detected in frame 0, simulate tracking them forward and
          compute their horizontal displacement.  The person who moves MOST
          is the runner — spectators in the foreground are stationary.
        Phase 2 (track): Process all frames with nearest-neighbor tracking
          locked on the runner identified in Phase 1.

        bboxes is a list of (x0,y0,x1,y1) tuples (or None) used for debug images.
        """
        if self._model is None:
            self._load_model()

        N_PROBE = min(10, len(frames))
        frame_h, frame_w = frames[0].shape[:2]

        # ── Phase 1: collect probe detections ─────────────────────────────
        probe: list[tuple] = []   # (boxes_np, centers, areas, heights) per frame
        for fi in range(N_PROBE):
            res = self._model(frames[fi], classes=[0], verbose=False, conf=0.15)
            if fi < 3:
                n_det = len(res[0].boxes) if res else 0
                confs_str = ""
                if res and n_det > 0:
                    confs = res[0].boxes.conf.cpu().numpy().tolist()
                    confs_str = "  confs=" + str([round(c, 3) for c in confs])
                print(f"[tracker] frame{fi}: shape={frame_w}x{frame_h}"
                      f"  dtype={frames[fi].dtype}  mean={frames[fi].mean():.1f}"
                      f"  detections={n_det}{confs_str}")
            if res and len(res[0].boxes) > 0:
                boxes = res[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
                heights = [float(b[3]-b[1]) for b in boxes]
                probe.append((boxes, centers, list(areas), heights))
            else:
                probe.append((np.zeros((0, 4)), [], [], []))

        # ── Phase 1: pick the runner ───────────────────────────────────────
        # For each candidate in frame 0, follow them through the probe frames
        # and measure horizontal displacement.  Most-moving = the runner.
        last_pos: Optional[tuple[float, float]] = None

        if probe and len(probe[0][1]) > 0:
            boxes0, centers0, areas0, _ = probe[0]
            n_cands = len(centers0)
            max_jump = 0.35 * frame_w   # max pixel jump between consecutive frames

            best_idx  = int(np.argmax(areas0))   # fallback: largest person
            best_disp = -1.0

            for ci in range(n_cands):
                pos = centers0[ci]
                x0  = pos[0]
                for fi in range(1, len(probe)):
                    _, centers, _, _ = probe[fi]
                    if not centers:
                        break
                    dists = [np.hypot(c[0]-pos[0], c[1]-pos[1]) for c in centers]
                    nearest = int(np.argmin(dists))
                    if dists[nearest] < max_jump:
                        pos = centers[nearest]
                x_disp = abs(pos[0] - x0)
                if x_disp > best_disp:
                    best_disp = x_disp
                    best_idx  = ci

            print(f"[tracker] runner: cand={best_idx}/{n_cands}"
                  f"  x_disp={best_disp:.1f}px"
                  f"  start_cx={centers0[best_idx][0]:.1f}")
            last_pos = centers0[best_idx]

        # ── Phase 2: build full track ──────────────────────────────────────
        # Uses velocity-extrapolated position prediction AND a maximum-jump
        # guard: any candidate further than max_jump from the predicted pos
        # is rejected, preventing the tracker from teleporting to a different
        # person when the runner moves quickly.
        raw_centroids: list[Optional[tuple[float, float]]] = []
        raw_heights:   list[Optional[float]]               = []
        raw_bboxes:    list[Optional[tuple]]               = []
        prev_pos: Optional[tuple[float, float]] = None   # one frame behind last_pos

        def _predict(last, prev):
            """First-order extrapolation: next ≈ last + (last - prev)."""
            if last is None or prev is None:
                return last
            return (2*last[0] - prev[0], 2*last[1] - prev[1])

        def _process(boxes, areas, centers, heights):
            nonlocal last_pos, prev_pos
            if not centers:
                raw_centroids.append(None)
                raw_heights.append(None)
                raw_bboxes.append(None)
                return

            pred = _predict(last_pos, prev_pos)
            anchor = pred if pred is not None else last_pos

            # Max jump: if we have velocity history use it (2× measured step),
            # otherwise fall back to 40% of frame width.
            if last_pos is not None and prev_pos is not None:
                step_px = np.hypot(last_pos[0]-prev_pos[0], last_pos[1]-prev_pos[1])
                max_jump = max(60, step_px * 2.5)
            else:
                max_jump = 0.40 * frame_w

            if anchor is not None:
                dists = [np.hypot(c[0]-anchor[0], c[1]-anchor[1]) for c in centers]
                # Only consider candidates within max_jump of predicted position
                valid = [(i, d) for i, d in enumerate(dists) if d <= max_jump]
                if valid:
                    scores = [areas[i] / (d + 50) for i, d in valid]
                    idx = valid[int(np.argmax(scores))][0]
                else:
                    # No candidate close enough — runner temporarily out of frame
                    raw_centroids.append(None)
                    raw_heights.append(None)
                    raw_bboxes.append(None)
                    return
            else:
                idx = int(np.argmax(areas))

            center = centers[idx]
            prev_pos = last_pos
            last_pos = center
            raw_centroids.append(center)
            raw_heights.append(heights[idx])
            b = boxes[idx]
            raw_bboxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))

        # Replay probe frames with the selected runner as anchor
        for fi in range(N_PROBE):
            boxes, centers, areas, heights = probe[fi]
            _process(boxes, areas, centers, heights)

        # Continue with remaining frames
        for frame in frames[N_PROBE:]:
            res = self._model(frame, classes=[0], verbose=False, conf=0.15)
            if res and len(res[0].boxes) > 0:
                boxes   = res[0].boxes.xyxy.cpu().numpy()
                areas   = list((boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]))
                centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
                heights = [float(b[3]-b[1]) for b in boxes]
                _process(boxes, areas, centers, heights)
            else:
                raw_centroids.append(None)
                raw_heights.append(None)
                raw_bboxes.append(None)

        return (
            self._fill_gaps(raw_centroids),
            self._fill_gaps_1d(raw_heights),
            raw_bboxes,
        )

    @staticmethod
    def _fill_gaps(pts: list) -> list:
        """Linear interpolation for missing centroid detections."""
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

    @staticmethod
    def _fill_gaps_1d(vals: list) -> list:
        """Linear interpolation for missing 1-D float values (e.g. heights)."""
        result = list(vals)
        n = len(result)
        for i in range(n):
            if result[i] is not None:
                continue
            j = i + 1
            while j < n and result[j] is None:
                j += 1
            if i > 0 and j < n:
                v0, v1 = result[i - 1], result[j]
                for k in range(i, j):
                    t = (k - i + 1) / (j - i + 1)
                    result[k] = v0 + t * (v1 - v0)
            elif j < n:
                result[i] = result[j]
            elif i > 0:
                result[i] = result[i - 1]
        return result


# ─── Velocity computation ─────────────────────────────────────────────────────
def compute_velocities_kmh(
    centroids:    list,
    bbox_heights: list,
    fps:          float,
    step:         int = 3,
) -> list[float]:
    """
    Compute per-frame speed in km/h directly.

    Calibration: median bbox height → px/m scale.

    Motion axis: auto-detect whether the runner moves mainly in x or y
    (handles both landscape and portrait source videos after rotation fix).
    Only the dominant axis is used — the other axis is YOLO jitter/perspective
    noise that would inflate the reading by 30-80 % if included via hypot.
    """
    n = len(centroids)
    valid_h = [h for h in bbox_heights if h is not None and h > 10]
    if not valid_h:
        return [0.0] * n

    median_h = float(np.median(valid_h))
    px_per_m = median_h / ATHLETE_HEIGHT_M
    print(f"[calibration] median bbox height={median_h:.1f}px  px/m={px_per_m:.2f}")

    # Determine dominant motion axis from median per-frame displacements
    all_dx = [abs(centroids[i][0] - centroids[i-1][0])
              for i in range(1, n) if centroids[i] and centroids[i-1]]
    all_dy = [abs(centroids[i][1] - centroids[i-1][1])
              for i in range(1, n) if centroids[i] and centroids[i-1]]
    med_dx = float(np.median(all_dx)) if all_dx else 0.0
    med_dy = float(np.median(all_dy)) if all_dy else 0.0
    use_x  = med_dx >= med_dy   # True → runner moves left-right; False → up-down
    print(f"[speed] axis={'x' if use_x else 'y'}  med_dx={med_dx:.1f}  med_dy={med_dy:.1f}")

    v = [0.0] * n
    for i in range(step, n - step):
        p0, p1 = centroids[i - step], centroids[i + step]
        if p0 and p1:
            dist = abs(p1[0] - p0[0]) if use_x else abs(p1[1] - p0[1])
            v[i] = dist / (2 * step / fps) / px_per_m * 3.6
    return v


def smooth(v: list[float], window: int = 9) -> list[float]:
    arr = np.array(v, dtype=float)
    k = np.ones(window) / window
    return np.convolve(arr, k, mode="same").tolist()


# ─── Smart run window detection ───────────────────────────────────────────────
def find_run_window(velocities: list[float], fps: float) -> tuple[int, int, int]:
    """
    Returns (start_frame, end_frame, onset_frame).
    start_frame includes a visual buffer before the run.
    onset_frame is where the athlete actually starts accelerating.
    """
    s = smooth(velocities, window=9)
    n = len(s)

    if n == 0:
        return 0, n, 0

    skip     = max(3, n // 15)   # smaller skip: don't miss an early peak
    peak_idx = skip + int(np.argmax(s[skip:]))
    peak_v   = s[peak_idx]

    if peak_v < 1.0:
        return 0, n, 0

    drop_thresh = peak_v * PLANT_DROP
    end_frame   = n
    for i in range(peak_idx, n):
        if s[i] < drop_thresh:
            end_frame = i
            break

    onset_thresh = peak_v * ONSET_PCT
    onset_frame  = 0
    for i in range(end_frame - 1, -1, -1):
        if s[i] < onset_thresh:
            onset_frame = i
            break

    buf           = int(fps * START_BUFFER)
    start_frame   = max(0, onset_frame - buf)
    tail          = int(fps * TAIL_BUFFER)
    end_with_tail = min(n, end_frame + tail)

    print(
        f"[window] peak@{peak_idx} ({peak_v:.1f} km/h)  "
        f"onset@{onset_frame}  plant@{end_frame}  "
        f"window=[{start_frame}, {end_with_tail}]  "
        f"({(end_with_tail - start_frame) / fps:.1f}s)"
    )
    return start_frame, end_with_tail, onset_frame


# ─── Speed bar overlay ────────────────────────────────────────────────────────
def _build_buckets(velocities_kmh: list[float], win_start: int, plant_local: int) -> list[float]:
    """10 equal-time buckets from run start to pole plant."""
    run = velocities_kmh[win_start:plant_local] if plant_local < len(velocities_kmh) else velocities_kmh[win_start:]
    if not run:
        return [0.0] * NUM_BUCKETS
    chunk = max(1, len(run) // NUM_BUCKETS)
    return [
        float(np.mean(run[i * chunk:(i + 1) * chunk]) or 0)
        for i in range(NUM_BUCKETS)
    ]


def draw_bar(
    frame:      np.ndarray,
    buckets:    list[float],
    frame_local: int,
    plant_local: int,
    avg_kmh:    float,
    delta_kmh:  Optional[float],
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_y = h - BAR_HEIGHT

    cv2.rectangle(frame, (0, bar_y), (w, h), (18, 18, 18), -1)

    max_v = max(buckets) if max(buckets) > 0 else 1.0
    seg_w = w // NUM_BUCKETS
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55

    for i, bv in enumerate(buckets):
        color = speed_color(bv / max_v)
        x0 = i * seg_w
        x1 = x0 + seg_w - 3
        cv2.rectangle(frame, (x0, bar_y + 5), (x1, h - 5), color, -1)

        # Per-bucket speed label (integer km/h) centred in the segment
        if bv > 0:
            label = str(int(round(bv)))
            (tw, th), _ = cv2.getTextSize(label, font, 0.36, 1)
            tx = x0 + max(0, (seg_w - 3 - tw) // 2)
            ty = bar_y + 5 + th + 3
            cv2.putText(frame, label, (tx, ty), font, 0.36, (20, 20, 20), 1, cv2.LINE_AA)

    if plant_local > 0:
        pct = min(frame_local / plant_local, 1.0)
        cx  = int(pct * w)
        cv2.line(frame, (cx, bar_y + 2), (cx, h - 2), (255, 255, 255), 2)

    cv2.putText(frame, f"{avg_kmh:.1f} km/h",
                (8, bar_y + 28), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

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
    input_path:   str,
    output_path:  str,
    runway_meters: float = 40.0,   # kept for API compat; no longer used for speed
    prev_avg_kmh: Optional[float] = None,
) -> dict:
    """
    Full pipeline:
    1. Track runner with YOLO (ID-persistent) — gets centroids + bbox heights.
    2. Compute per-frame km/h via bbox-height scale (no runway assumption).
    3. Auto-detect run window [onset → pole plant].
    4. Write full annotated video with speed bar overlay.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Handle mobile phone rotation metadata ────────────────────────────────
    # Phones embed a rotation angle in the MP4 container. OpenCV (FFMPEG backend)
    # reads raw pixel data without applying it, so portrait videos appear sideways.
    # This would make bbox heights wrong (person's width instead of height) and
    # flip the dominant motion axis.
    try:
        rotation_meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    except (AttributeError, TypeError):
        rotation_meta = 0
    rotate_code = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(rotation_meta if rotation_meta in (90, 180, 270) else 0)
    if rotate_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        width, height = height, width   # dimensions swap for 90/270

    print(f"[analyzer] {total} frames @ {fps:.1f}fps  {width}x{height}  rotation={rotation_meta}°")

    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if rotate_code is not None:
            f = cv2.rotate(f, rotate_code)
        frames.append(f)
    cap.release()

    if not frames:
        raise ValueError("No frames read")

    # Save frame 0 for debugging
    try:
        cv2.imwrite(str(DEBUG_DIR / "debug_frame0.jpg"), frames[0])
        brightness = float(np.mean(frames[0]))
        print(f"[debug] frame0: shape={frames[0].shape}  brightness={brightness:.1f}")
    except Exception as e:
        print(f"[debug] frame0 save failed: {e}")

    # ── Track across full video ───────────────────────────────────────────────
    tracker = RunnerTracker()
    centroids, bbox_heights, raw_bboxes = tracker.track(frames)

    # ── Post-tracking cleanup: reject ID switches ────────────────────────────
    # After the pole plant the tracker often locks onto a spectator, pole tip,
    # or the vaulter at bar height.  Two signals expose this:
    #   1. bbox height drops well below the athlete's typical height
    #   2. centroid teleports across the frame in a single step
    # Mark affected frames as None, then re-interpolate so velocities stay
    # smooth instead of spiking.
    first_q = bbox_heights[:max(1, len(bbox_heights) // 4)]
    ref_heights = [h for h in first_q if h is not None and h > 10]
    if ref_heights:
        ref_h = float(np.median(ref_heights))
        min_h = ref_h * 0.40          # reject anything <40% of reference
        invalidated = 0
        for i in range(len(centroids)):
            # Height check: tiny bbox = different/distant person
            if bbox_heights[i] is not None and bbox_heights[i] < min_h:
                centroids[i] = None
                bbox_heights[i] = None
                invalidated += 1
            # Teleport check: centroid jumps >15% of frame size in either axis
            if (i > 0 and centroids[i] is not None and centroids[i - 1] is not None):
                dx = abs(centroids[i][0] - centroids[i - 1][0])
                dy = abs(centroids[i][1] - centroids[i - 1][1])
                if dx > width * 0.15 or dy > height * 0.15:
                    centroids[i] = None
                    bbox_heights[i] = None
                    invalidated += 1
        if invalidated:
            print(f"[cleanup] invalidated {invalidated} frames  "
                  f"(ref_h={ref_h:.0f}px  min_h={min_h:.0f}px)")
            centroids    = RunnerTracker._fill_gaps(centroids)
            bbox_heights = RunnerTracker._fill_gaps_1d(bbox_heights)

    # ── Velocity in km/h (bbox-height calibrated, x-axis only) ───────────────
    vel_kmh = smooth(compute_velocities_kmh(centroids, bbox_heights, fps))

    # ── Auto-detect run window ────────────────────────────────────────────────
    win_start, win_end, onset_frame = find_run_window(vel_kmh, fps)
    plant_abs = win_end - int(fps * TAIL_BUFFER)

    # Stats: onset → plant (excludes the pre-run visual buffer)
    run_kmh  = [v for v in vel_kmh[onset_frame:plant_abs] if v > 0]
    avg_kmh  = float(np.mean(run_kmh))  if run_kmh else 0.0
    peak_kmh = float(np.max(run_kmh))   if run_kmh else 0.0
    delta    = (avg_kmh - prev_avg_kmh) if prev_avg_kmh is not None else None

    # ── Build buckets for bar chart ───────────────────────────────────────────
    buckets = _build_buckets(vel_kmh, onset_frame, plant_abs)

    # ── Write full annotated output ───────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        frame = draw_bar(frame, buckets, i, plant_abs, avg_kmh, delta)
        out.write(frame)

    out.release()

    full_s     = total / fps
    trimmed_s  = (win_end - win_start) / fps
    print(
        f"[analyzer] output={full_s:.1f}s  trimmed={trimmed_s:.1f}s  "
        f"avg={avg_kmh:.1f} km/h  peak={peak_kmh:.1f} km/h"
    )

    # ── Debug output ──────────────────────────────────────────────────────────
    try:
        # 1. JSON: full velocity timeline + tracking data
        # Force Python floats to avoid float32 JSON serialization errors
        def _f(x):
            return round(float(x), 2) if x is not None else None

        debug = {
            "fps": round(fps, 2),
            "total_frames": total,
            "actual_frames_read": len(frames),
            "win_start": win_start,
            "win_end": win_end,
            "plant_abs": plant_abs,
            "avg_kmh": round(avg_kmh, 2),
            "peak_kmh": round(peak_kmh, 2),
            "buckets_kmh": [_f(b) for b in buckets],
            "vel_kmh": [_f(v) for v in vel_kmh],
            "bbox_heights_px": [_f(h) for h in bbox_heights],
            "centroid_x": [_f(c[0]) if c is not None else None for c in centroids],
            "centroid_y": [_f(c[1]) if c is not None else None for c in centroids],
            "frame_brightness": [round(float(np.mean(f)), 1) for f in frames[:10]],
        }
        (DEBUG_DIR / "debug_last_run.json").write_text(json.dumps(debug, indent=2))

        # 2. JPEG: save 3 labelled debug frames (onset, peak, plant)
        key_frames = {
            "onset": win_start,
            "peak":  min(win_start + (int(np.argmax(vel_kmh[win_start:plant_abs])) if run_kmh else 0), len(frames)-1),
            "plant": min(plant_abs, len(frames)-1),
        }
        for label, fi in key_frames.items():
            dbg = frames[fi].copy()
            bbox = raw_bboxes[fi] if fi < len(raw_bboxes) else None
            if bbox:
                x0b, y0b, x1b, y1b = [int(v) for v in bbox]
                cv2.rectangle(dbg, (x0b, y0b), (x1b, y1b), (0, 255, 0), 2)
            c = centroids[fi] if fi < len(centroids) else None
            if c is not None:
                cv2.circle(dbg, (int(c[0]), int(c[1])), 8, (0, 0, 255), -1)
            cv2.putText(dbg, f"{label} f{fi}  {vel_kmh[fi]:.1f} km/h",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(str(DEBUG_DIR / f"debug_{label}_frame.jpg"), dbg)
        print(f"[debug] saved debug_last_run.json + 3 debug frame JPEGs")
    except Exception as e:
        print(f"[debug] save failed: {e}")

    return {
        "avg_kmh":          round(avg_kmh, 2),
        "peak_kmh":         round(peak_kmh, 2),
        "delta_kmh":        round(delta, 2) if delta is not None else None,
        "pole_plant_frame": plant_abs,
        "total_frames":     total,
        "fps":              fps,
        "trimmed_seconds":  round(trimmed_s, 1),
        "original_frames":  total,
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
