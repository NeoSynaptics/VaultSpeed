"""
Generates a synthetic pole vault approach video for pipeline testing.
A white rectangle accelerates L→R across the frame, then abruptly slows (pole plant).
"""
import cv2
import numpy as np
import math

FPS = 30
DURATION = 8        # seconds total
W, H = 1280, 720

# Timeline (frames)
IDLE_START   = 0              # athlete standing still
IDLE_END     = int(FPS * 1.5) # 1.5s idle
RUN_START    = IDLE_END
PLANT_FRAME  = int(FPS * 6.5) # pole plant at 6.5s
TOTAL        = int(FPS * DURATION)

def ease_in(t):
    """t in [0,1] → smooth acceleration curve"""
    return t * t * (3 - 2 * t)

def make_frame(frame_idx: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Background: dark grey with faint lane markings
    img[:] = (20, 20, 20)
    for y in [H // 3, 2 * H // 3]:
        cv2.line(img, (0, y), (W, y), (40, 40, 40), 1)
    for x in range(0, W, 60):
        cv2.line(img, (x, 0), (x, H), (35, 35, 35), 1)

    # Runner rectangle size
    rw, rh = 60, 140
    cy = H // 2  # vertical center (runner stays in middle height)

    if frame_idx < IDLE_END:
        # Standing still at left edge
        cx = 80
    elif frame_idx < PLANT_FRAME:
        # Accelerating run
        run_frames = PLANT_FRAME - RUN_START
        t = (frame_idx - RUN_START) / run_frames
        eased = ease_in(t)
        cx = int(80 + eased * (W - 200))
    else:
        # Post-plant: sharp deceleration
        post = frame_idx - PLANT_FRAME
        decay = math.exp(-post * 0.15)
        plant_x = int(80 + ease_in(1.0) * (W - 200))
        cx = int(plant_x + (1 - decay) * 80)

    x0 = max(0, cx - rw // 2)
    x1 = min(W, cx + rw // 2)
    y0 = cy - rh // 2
    y1 = cy + rh // 2

    # Draw runner body
    cv2.rectangle(img, (x0, y0), (x1, y1), (220, 220, 220), -1)
    # Head
    cv2.circle(img, (cx, y0 - 18), 18, (200, 180, 160), -1)

    if frame_idx >= PLANT_FRAME and (frame_idx - PLANT_FRAME) < int(FPS * 0.5):
        # Draw a rough pole line
        pole_end_x = cx - 40
        cv2.line(img, (cx, y0 + 20), (pole_end_x - 60, cy + 60), (160, 160, 80), 4)

    # Frame counter (debug)
    cv2.putText(img, f"frame {frame_idx}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    return img


def main():
    out_path = "test_run.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    for i in range(TOTAL):
        writer.write(make_frame(i))

    writer.release()
    print(f"Written {TOTAL} frames -> {out_path}")
    print(f"Duration: {TOTAL/FPS:.1f}s  |  FPS: {FPS}  |  Resolution: {W}x{H}")


if __name__ == "__main__":
    main()
