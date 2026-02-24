"""
VaultSpeed — Local dev server with auto-tunnel (ngrok)

Usage:
  pip install -r requirements.txt
  python local_server.py

On startup it will:
  1. Start the FastAPI server on port 8000
  2. Open a public ngrok tunnel automatically
  3. Print the URL — paste it into app/.env as EXPO_PUBLIC_API_URL

No same-WiFi required. Works from anywhere.
"""

import base64
import datetime
import json
import os
import sys
import tempfile
import threading
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

    # Save every uploaded video with a timestamp so examples are never overwritten
    saved_dir = Path(__file__).parent / "saved_videos"
    saved_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_video_path = saved_dir / f"{ts}_{athlete_id}_raw.mp4"
    try:
        saved_video_path.write_bytes(video_bytes)
        print(f"[debug] saved {saved_video_path.name} ({len(video_bytes)//1024} KB)")
    except Exception as e:
        print(f"[debug] could not save video: {e}")

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

        history[athlete_id] = stats["avg_kmh"]
        _save_history(history)

        with open(output_path, "rb") as f:
            video_bytes_out = f.read()
        video_b64 = base64.b64encode(video_bytes_out).decode()

        # Also persist the annotated video next to the raw input
        try:
            annotated_path = saved_dir / f"{ts}_{athlete_id}_annotated.mp4"
            annotated_path.write_bytes(video_bytes_out)
            print(f"[debug] saved {annotated_path.name}")
        except Exception as e:
            print(f"[debug] could not save annotated video: {e}")

        return JSONResponse({"stats": stats, "video_b64": video_b64})

    finally:
        for p in [input_path, output_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


# ─── ngrok tunnel ─────────────────────────────────────────────────────────────
def _start_tunnel(port: int = 8000) -> None:
    """Start ngrok tunnel and print the public URL."""
    try:
        from pyngrok import ngrok, conf

        # Use free tier — no auth token needed for basic HTTP tunnels
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url

        # ngrok gives http:// — upgrade to https://
        if public_url.startswith("http://"):
            public_url = "https://" + public_url[7:]

        print("\n" + "=" * 54)
        print("  ✓  ngrok tunnel active")
        print(f"  URL: {public_url}")
        print()
        print("  Paste into app/.env:")
        print(f"  EXPO_PUBLIC_API_URL={public_url}")
        print("=" * 54 + "\n")

    except ImportError:
        print("\n[tunnel] pyngrok not installed — running local only.")
        print("[tunnel] Install with:  pip install pyngrok")
        print("[tunnel] Or connect phone to the same WiFi and use local IP.\n")
    except Exception as e:
        print(f"\n[tunnel] Could not start ngrok: {e}")
        print("[tunnel] Falling back to local-only mode.\n")


if __name__ == "__main__":
    PORT = 8000

    # Start tunnel in background thread (server starts immediately in parallel)
    t = threading.Thread(target=_start_tunnel, args=(PORT,), daemon=True)
    t.start()
    t.join(timeout=6)   # wait up to 6s for URL to print before server logs flood output

    print(f"[server] Starting on http://0.0.0.0:{PORT}")
    uvicorn.run("local_server:app", host="0.0.0.0", port=PORT, reload=True)
