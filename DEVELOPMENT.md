# VaultSpeed — Development Guide

Pick up where you left off on any PC.

---

## Architecture

```
VaultSpeed/
├── backend/                  Python FastAPI server
│   ├── analyzer.py           CV pipeline: YOLO tracking → speed → annotated video
│   ├── local_server.py       FastAPI server (run this for local dev)
│   ├── test_tracking.py      Local debug script — iterate without the phone
│   ├── make_test_video.py    Generate a synthetic test video (no real footage needed)
│   └── requirements.txt      Python dependencies
├── app/                      React Native / Expo app
│   ├── screens/
│   │   ├── CameraScreen.tsx  Record video, send to backend
│   │   ├── ResultScreen.tsx  Show speed bar + annotated video playback
│   │   └── SetupScreen.tsx   Set backend URL + athlete height
│   └── services/api.ts       POST /analyze + response handling
└── start-dev.sh              One-command launcher (backend + Expo)
```

---

## Quick Start — New PC

### 1. Prerequisites

- **Python 3.10+** — `python --version`
- **Node.js 18+** — `node --version`
- **Expo Go** on your phone (iOS / Android)
- Git

### 2. Clone

```bash
git clone https://github.com/NeoSynaptics/VaultSpeed.git
cd VaultSpeed
```

### 3. Backend setup

```bash
cd backend
python -m venv .venv

# Windows (Git Bash / CMD):
.venv/Scripts/pip install -r requirements.txt

# macOS / Linux:
.venv/bin/pip install -r requirements.txt
```

> YOLO model weights (`yolov8n.pt`) download automatically on first run (~6 MB).

### 4. Run the backend server

```bash
# Windows Git Bash:
.venv/Scripts/python local_server.py

# macOS / Linux:
.venv/bin/python local_server.py
```

Server starts on `http://0.0.0.0:8000`.
Check it's alive: `curl http://localhost:8000/health`

### 5. Connect the phone app

**Same-WiFi (simplest):**
1. Find your laptop's local IP: `ipconfig` (Windows) or `ifconfig` / `ip a` (Linux/Mac)
2. Open the app → Setup screen → set Backend URL to `http://<YOUR_IP>:8000`
3. Make sure phone and laptop are on the same WiFi

**Over internet (ngrok):**
```bash
pip install pyngrok
# Then restart local_server.py — it will print the public URL automatically
```

### 6. Run the Expo app

```bash
cd app
npm install
npx expo start --tunnel   # tunnel = works anywhere, no same-WiFi needed
```

Scan the QR code with Expo Go on your phone.

---

## Iterating Without the Phone

Every video submitted from the app is saved to `backend/debug_last_video.mp4`.

Once you have that file, you can re-run the full pipeline locally:

```bash
cd backend
.venv/Scripts/python test_tracking.py                    # uses debug_last_video.mp4
.venv/Scripts/python test_tracking.py path/to/video.mp4  # specific file
```

This prints:
- YOLO detections at 4 confidence thresholds (sanity check)
- Full pipeline stats: avg/peak km/h, window, non-null frame counts
- Writes `debug_test_frame0.jpg` so you can see what YOLO sees

### Generate a synthetic test video (no real footage)

```bash
.venv/Scripts/python make_test_video.py
# → writes test_run.mp4 (8s, white rectangle accelerates L→R then plants)
.venv/Scripts/python test_tracking.py test_run.mp4
```

---

## Debug Artefacts (generated each run)

| File | Description |
|------|-------------|
| `backend/debug_last_video.mp4` | Raw video from last phone upload |
| `backend/debug_last_run.json` | Full velocity timeline, window, bbox heights per frame |
| `backend/debug_frame0.jpg` | First frame of last video (check YOLO can see the runner) |
| `backend/debug_onset_frame.jpg` | Frame where approach starts (should have green bbox on runner) |
| `backend/debug_peak_frame.jpg` | Frame of peak speed (green bbox on runner at full stride) |
| `backend/debug_plant_frame.jpg` | Frame of pole plant (green bbox on runner at pole) |

These files are in `.gitignore` — they are runtime artefacts, not source.

---

## Key Configuration (analyzer.py)

```python
ATHLETE_HEIGHT_M = 1.81   # athlete height used for px→m calibration
ONSET_PCT        = 0.20   # velocity below 20% of peak = "not yet running"
PLANT_DROP       = 0.40   # velocity drops to 40% of peak = pole plant detected
START_BUFFER     = 2.0    # seconds of pre-onset to include in output video
TAIL_BUFFER      = 0.5    # seconds to keep after pole plant
```

**TODO:** Athlete height should come from the app's Setup screen (currently hardcoded).

---

## How the Speed Calculation Works

1. **YOLO** detects all people in each frame (`yolov8n`, conf ≥ 0.15)
2. **Phase 1 (probe):** First 10 frames → pick the candidate with most horizontal displacement (runner moves, spectators don't)
3. **Phase 2 (track):** Velocity-prediction + max-jump constraint keeps the box on the same person even at full sprint
4. **Calibration:** `px_per_m = median(bbox_height_px) / ATHLETE_HEIGHT_M`
5. **Speed:** `km/h = hypot(dx, dy) / (2*step/fps) / px_per_m * 3.6`  (3-frame step, smoothed with 9-frame window)
6. **Window:** Auto-detected — peak velocity → scan back to onset (20% of peak), scan forward to plant drop (40% of peak)

---

## Known Limitations

- **Broadcast/distant-camera videos:** bbox calibration underestimates speed by ~30% (runner appears small). Works best with side-on phone recordings at ≤10m distance.
- **Athlete height hardcoded:** 1.81m (Duplantis). Needs per-athlete setting in the app.
- **Vertical motion counted:** After pole plant, upward vault motion adds to measured speed. The window detection cuts off at plant, so this doesn't affect reported peak/avg.

---

## One-Command Start (Windows Git Bash)

```bash
bash start-dev.sh
```

This starts the backend and Expo in one terminal. Requires the venv to exist at `backend/.venv/` or `backend/env2/`.
