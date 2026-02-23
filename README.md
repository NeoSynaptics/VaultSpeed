# VaultSpeed

Record a pole vault approach run → get a speed-annotated video back in seconds.

## What it does

1. **Hold** the record button → **release** to stop
2. Original video saved to camera roll immediately
3. Video uploaded to GPU cloud (Modal) for analysis
4. YOLOv8 tracks runner → computes km/h per frame
5. 10-segment color bar burned into video: red=slow, yellow=ok, green=fast
6. Annotated video returned → playable in app → save to camera roll

## Stack

- **App**: Expo (React Native) — iOS + Android
- **Backend**: Python + FastAPI + OpenCV + YOLOv8n
- **Cloud**: [Modal](https://modal.com) — T4 GPU on-demand (~$0.0003/run)

## Setup

### Backend (Modal)
```bash
cd backend
pip install -r requirements.txt
modal deploy modal_app.py
# Copy the /analyze URL into app/.env
```

### App
```bash
cd app
npm install
# Edit .env with your Modal URL
npx expo start
# Scan QR with Expo Go on iPhone
```

## Calibration

On first launch, enter your runway length in meters (default: 40m).
The backend maps total pixel displacement across the run to this distance → km/h.

## Speed bar

```
[Run start] ████████████████████████░░░░░░ [Pole plant]
             ^red=slow  ^yellow  ^green=fast
              Avg: 31.2 km/h    Δ +0.8 vs last run
```

## Notes

- Pole plant auto-detected via velocity drop (>50% in <5 frames)
- Delta vs previous run stored in Modal volume (persistent per athlete_id)
- Videos ~10s → ~3-5s processing on T4 GPU
