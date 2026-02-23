"""
VaultSpeed — Local dev server
Run on your PC, point the Expo app at your local IP.

Usage:
  pip install -r requirements.txt
  python local_server.py

Then set in app/.env:
  EXPO_PUBLIC_API_URL=http://<your-pc-ip>:8000

Find your IP:  ipconfig  (look for IPv4 under your WiFi adapter)
Phone must be on the same WiFi network.
"""

import base64
import json
import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from analyzer import analyze_video

app = FastAPI(title="VaultSpeed local")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple JSON file to persist per-athlete averages across runs
HISTORY_FILE = Path(__file__).parent / "run_history.json"


def _load_history() -> dict:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_history(data: dict) -> None:
    HISTORY_FILE.write_text(json.dumps(data, indent=2))


@app.get("/health")
def health():
    return {"status": "ok", "service": "vaultspeed-local"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    runway_meters: float = Form(40.0),
    athlete_id: str = Form("default"),
):
    history = _load_history()
    prev_avg = history.get(athlete_id)

    video_bytes = await video.read()

    # Write to temp files
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
        f_in.write(video_bytes)
        input_path = f_in.name

    output_path = input_path.replace(".mp4", "_annotated.mp4")

    try:
        stats = analyze_video(
            input_path=input_path,
            output_path=output_path,
            runway_meters=runway_meters,
            prev_avg_kmh=prev_avg,
        )

        # Persist this run's avg for next delta
        history[athlete_id] = stats["avg_kmh"]
        _save_history(history)

        # Read annotated video and base64 encode
        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        return JSONResponse({"stats": stats, "video_b64": video_b64})

    finally:
        for p in [input_path, output_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


if __name__ == "__main__":
    import socket

    # Print local IP so you know what to put in app/.env
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  VaultSpeed local server")
    print(f"  Local IP:  http://{local_ip}:8000")
    print(f"  Set in app/.env:")
    print(f"  EXPO_PUBLIC_API_URL=http://{local_ip}:8000")
    print(f"{'='*50}\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
