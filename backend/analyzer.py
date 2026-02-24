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
ATHLETE_HEIGHT_M = 1.70   # hardcoded average athlete height (m)

# ─── Pose skeleton (COCO 17-keypoint layout) ─────────────────────────────────
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6),                                      # shoulders
    (5, 7), (7, 9),                              # left arm
    (6, 8), (8, 10),                             # right arm
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15),                          # left leg
    (12, 14), (14, 16),                          # right leg
]
LIMB_COLORS = [
    (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0),   # head: cyan
    (230, 230, 230),                                                # shoulders: white
    (0, 255, 100), (0, 255, 100),                                   # left arm: green
    (0, 165, 255), (0, 165, 255),                                   # right arm: orange
    (230, 230, 230), (230, 230, 230), (230, 230, 230),             # torso: white
    (255, 0, 200), (255, 0, 200),                                   # left leg: magenta
    (0, 200, 255), (0, 200, 255),                                   # right leg: yellow
]
KP_CONF_THRESH = 0.3


def speed_color(ratio: float) -> tuple:
    """Smooth green → yellow → red gradient based on speed ratio (0-1)."""
    ratio = max(0.0, min(1.0, ratio))
    # Green (0,200,50) → Yellow (0,200,220) → Red (0,50,220)
    if ratio < 0.5:
        t = ratio / 0.5
        r = int(50 + t * 170)     # 50 → 220
        g = 200
        b = int(50 * (1 - t))     # 50 → 0
    else:
        t = (ratio - 0.5) / 0.5
        r = 220
        g = int(200 - t * 150)    # 200 → 50
        b = 0
    return (b, g, r)  # BGR


# ─── YOLO person tracker ──────────────────────────────────────────────────────
class RunnerTracker:
    """
    Tracks the primary runner using YOLO with ID persistence.
    Returns centroids AND bounding-box heights for scale calibration.
    """

    def __init__(self):
        self._det_model = None
        self._pose_model = None

    def _load_models(self):
        from ultralytics import YOLO
        self._det_model = YOLO("yolov8n.pt")       # detection: high detection rate
        self._pose_model = YOLO("yolov8n-pose.pt")  # pose: skeleton overlay only

    def track_all(self, frames: list[np.ndarray]) -> dict:
        """
        Track ALL people using ByteTrack with persistent IDs.

        Returns dict of tracks:
        {track_id: {"centroids": [...], "heights": [...], "bboxes": [...]}}
        Each list has len(frames) entries, with None for frames where that
        person wasn't detected.
        """
        if self._det_model is None:
            self._load_models()

        tracker_cfg = str(Path(__file__).parent / "tracker_config.yaml")
        tracks: dict[int, dict] = {}
        CONF = 0.15

        for fi, frame in enumerate(frames):
            results = self._det_model.track(
                frame, persist=True, tracker=tracker_cfg,
                conf=CONF, classes=[0], verbose=False,
            )

            if fi < 3:
                n_det = len(results[0].boxes) if results else 0
                has_ids = results[0].boxes.id is not None if results else False
                print(f"[tracker] frame{fi}: detections={n_det}  has_ids={has_ids}")

            if results[0].boxes.id is None:
                # No detections this frame — extend all active tracks with None
                for tid in tracks:
                    tracks[tid]["centroids"].append(None)
                    tracks[tid]["heights"].append(None)
                    tracks[tid]["bboxes"].append(None)
                continue

            track_ids = results[0].boxes.id.int().cpu().tolist()
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            seen_this_frame: set[int] = set()
            for j, tid in enumerate(track_ids):
                b = bboxes[j]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                h = float(b[3] - b[1])
                bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

                if tid not in tracks:
                    # New track — backfill with None for all previous frames
                    tracks[tid] = {
                        "centroids": [None] * fi,
                        "heights": [None] * fi,
                        "bboxes": [None] * fi,
                    }

                tracks[tid]["centroids"].append((cx, cy))
                tracks[tid]["heights"].append(h)
                tracks[tid]["bboxes"].append(bbox)
                seen_this_frame.add(tid)

            # Tracks not seen this frame get None
            for tid in tracks:
                if tid not in seen_this_frame:
                    tracks[tid]["centroids"].append(None)
                    tracks[tid]["heights"].append(None)
                    tracks[tid]["bboxes"].append(None)

        # Summary
        for tid, t in tracks.items():
            det_count = sum(1 for c in t["centroids"] if c is not None)
            print(f"[tracker] track {tid}: {det_count}/{len(frames)} frames detected")

        return tracks

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

    def extract_poses(self, frames: list[np.ndarray], bboxes: list) -> list:
        """Run pose model on each frame, match keypoints to tracked bboxes."""
        if self._pose_model is None:
            self._load_models()

        n = len(frames)
        keypoints = [None] * n
        detected = 0

        for i in range(n):
            bbox = bboxes[i] if i < len(bboxes) else None
            if bbox is None:
                continue

            res = self._pose_model(frames[i], verbose=False, conf=0.10)
            if not res or len(res[0].boxes) == 0 or res[0].keypoints is None:
                continue

            # Find pose detection closest to tracked bbox center
            tx = (bbox[0] + bbox[2]) / 2
            ty = (bbox[1] + bbox[3]) / 2
            boxes = res[0].boxes.xyxy.cpu().numpy()
            kps_data = res[0].keypoints.data.cpu().numpy()

            dists = [np.hypot((b[0]+b[2])/2 - tx, (b[1]+b[3])/2 - ty) for b in boxes]
            best = int(np.argmin(dists))

            if best < len(kps_data):
                keypoints[i] = kps_data[best]
                detected += 1

        print(f"[pose] extracted keypoints for {detected}/{n} frames")
        return keypoints


# ─── Module-level singletons — models load once, stay in memory ───────────────
# Creating RunnerTracker / VaulterIdentifier fresh on every request caused the
# YOLO weights and V-JEPA2 (300 MB) to be loaded from disk each time.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vaulter_identifier import VaulterIdentifier as _VIType

_tracker_singleton: "RunnerTracker | None" = None
_identifier_singleton: "_VIType | None" = None


def _get_tracker() -> "RunnerTracker":
    global _tracker_singleton
    if _tracker_singleton is None:
        _tracker_singleton = RunnerTracker()
    return _tracker_singleton


def _get_identifier():
    global _identifier_singleton
    if _identifier_singleton is None:
        from vaulter_identifier import VaulterIdentifier
        _identifier_singleton = VaulterIdentifier()
    return _identifier_singleton


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


# ─── Pose skeleton overlay ───────────────────────────────────────────────────
def draw_pose(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Draw skeleton lines and keypoint dots for one person."""
    h = frame.shape[0]
    thickness = max(2, h // 200)
    radius = max(3, h // 150)

    # Skeleton lines
    for li, (i, j) in enumerate(SKELETON):
        if keypoints[i][2] > KP_CONF_THRESH and keypoints[j][2] > KP_CONF_THRESH:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, pt1, pt2, LIMB_COLORS[li], thickness, cv2.LINE_AA)

    # Keypoint dots
    for k in range(17):
        if keypoints[k][2] > KP_CONF_THRESH:
            pt = (int(keypoints[k][0]), int(keypoints[k][1]))
            cv2.circle(frame, pt, radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, max(1, radius // 2), (0, 0, 0), 1, cv2.LINE_AA)

    return frame


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
    source:       str = "camera",  # "camera" or "library"
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
    # Library-picked videos are already properly oriented by the OS export;
    # only apply rotation for raw camera recordings.
    try:
        rotation_meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    except (AttributeError, TypeError):
        rotation_meta = 0

    if source == "library":
        rotate_code = None
        print(f"[analyzer] {total} frames @ {fps:.1f}fps  {width}x{height}  "
              f"rotation={rotation_meta} (skipped — library video)")
    else:
        rotate_code = {
            90:  cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(rotation_meta if rotation_meta in (90, 180, 270) else 0)
        if rotate_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            width, height = height, width
        print(f"[analyzer] {total} frames @ {fps:.1f}fps  {width}x{height}  rotation={rotation_meta}")

    # ── Resize to max 640px wide — YOLO resizes internally anyway, so this is
    # free speed and cuts memory usage 4-8x compared to raw 720p/1080p frames.
    MAX_WIDTH = 640
    if width > MAX_WIDTH:
        scale  = MAX_WIDTH / width
        width  = MAX_WIDTH
        height = int(height * scale)
    else:
        scale = 1.0

    # ── For high-fps video (>32fps) keep only every other frame — halves YOLO
    # calls with no meaningful loss for tracking or speed measurement.
    frame_step = 2 if fps > 32 else 1
    effective_fps = fps / frame_step

    frames = []
    fi_raw = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if fi_raw % frame_step == 0:
            if rotate_code is not None:
                f = cv2.rotate(f, rotate_code)
            if scale != 1.0:
                f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(f)
        fi_raw += 1
    cap.release()

    fps   = effective_fps   # downstream speed math uses this fps
    total = len(frames)     # actual processed frame count
    print(f"[analyzer] loaded {total} frames @ {fps:.1f}fps  {width}x{height}  "
          f"(step={frame_step} scale={scale:.2f})")

    if not frames:
        raise ValueError("No frames read")

    # Save frame 0 for debugging
    try:
        cv2.imwrite(str(DEBUG_DIR / "debug_frame0.jpg"), frames[0])
        brightness = float(np.mean(frames[0]))
        print(f"[debug] frame0: shape={frames[0].shape}  brightness={brightness:.1f}")
    except Exception as e:
        print(f"[debug] frame0 save failed: {e}")

    # ── Track all people with ByteTrack ──────────────────────────────────────
    tracker = _get_tracker()
    all_tracks = tracker.track_all(frames)

    if not all_tracks:
        print("[analyzer] warning: no people detected in video")
        # Write a blank output video so the caller still gets a file
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        return {
            "avg_kmh": 0.0, "peak_kmh": 0.0, "delta_kmh": None,
            "pole_plant_frame": 0, "total_frames": total,
            "fps": fps, "trimmed_seconds": 0.0, "original_frames": total,
        }

    # ── Identify the pole vaulter ──────────────────────────────────────────
    identifier = _get_identifier()
    best_id = identifier.identify(all_tracks, frames)
    track = all_tracks[best_id]

    # Extract single-track data
    raw_bboxes   = track["bboxes"]
    centroids    = RunnerTracker._fill_gaps(track["centroids"])
    bbox_heights = RunnerTracker._fill_gaps_1d(track["heights"])

    # ── Post-tracking cleanup: reject remaining anomalies ──────────────────
    invalidated = 0
    for i in range(1, len(centroids)):
        if (bbox_heights[i] is not None and bbox_heights[i - 1] is not None
                and bbox_heights[i - 1] > 10):
            ratio = bbox_heights[i] / bbox_heights[i - 1]
            if ratio < 0.4 or ratio > 2.5:
                centroids[i] = None
                bbox_heights[i] = None
                raw_bboxes[i] = None
                invalidated += 1
                continue
        if centroids[i] is not None and centroids[i - 1] is not None:
            dx = abs(centroids[i][0] - centroids[i - 1][0])
            dy = abs(centroids[i][1] - centroids[i - 1][1])
            if dx > width * 0.15 or dy > height * 0.15:
                centroids[i] = None
                bbox_heights[i] = None
                raw_bboxes[i] = None
                invalidated += 1
    if invalidated:
        print(f"[cleanup] invalidated {invalidated} frames")
        centroids    = RunnerTracker._fill_gaps(centroids)
        bbox_heights = RunnerTracker._fill_gaps_1d(bbox_heights)

    # ── Extract pose keypoints (separate pass with pose model) ─────────
    all_keypoints = tracker.extract_poses(frames, raw_bboxes)

    # ── Velocity in km/h (bbox-height calibrated, x-axis only) ───────────────
    vel_kmh = smooth(compute_velocities_kmh(centroids, bbox_heights, fps))

    # ── Speed spike filter: cap at physical maximum (no human > 45 km/h) ──
    MAX_SPEED_KMH = 45.0
    for i in range(len(vel_kmh)):
        if vel_kmh[i] > MAX_SPEED_KMH:
            vel_kmh[i] = MAX_SPEED_KMH

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
        # Draw pose skeleton
        if i < len(all_keypoints) and all_keypoints[i] is not None:
            frame = draw_pose(frame, all_keypoints[i])
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
