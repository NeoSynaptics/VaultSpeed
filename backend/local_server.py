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

import datetime
import json
import os
import tempfile
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from analyzer import analyze_video

app = FastAPI(title="VaultSpeed local")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE  = Path(__file__).parent / "run_history.json"
ANNOTATED_DIR = Path(__file__).parent / "annotated_videos"
ANNOTATED_DIR.mkdir(exist_ok=True)


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
    source: str = Form("camera"),
):
    history = _load_history()
    prev_avg = history.get(athlete_id)

    video_bytes = await video.read()

    # Save raw upload for debugging
    saved_dir = Path(__file__).parent / "saved_videos"
    saved_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    src_tag = "lib" if source == "library" else "cam"
    try:
        (saved_dir / f"{ts}_{athlete_id}_{src_tag}_raw.mp4").write_bytes(video_bytes)
        print(f"[debug] saved raw ({len(video_bytes)//1024} KB)")
    except Exception as e:
        print(f"[debug] could not save raw video: {e}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
        f_in.write(video_bytes)
        input_path = f_in.name

    # Annotated video stored in ANNOTATED_DIR, served via /video/{filename}
    video_filename = f"{ts}_{athlete_id}_{src_tag}_annotated.mp4"
    output_path = str(ANNOTATED_DIR / video_filename)

    try:
        stats = analyze_video(
            input_path=input_path,
            output_path=output_path,
            runway_meters=runway_meters,
            prev_avg_kmh=prev_avg,
            source=source,
        )

        history[athlete_id] = stats["avg_kmh"]
        _save_history(history)

        print(f"[debug] annotated video ready: {video_filename} "
              f"({Path(output_path).stat().st_size // 1024} KB)")

        # Return only stats + a download token — video is fetched separately
        # so the JSON response stays small (< 1 KB) and tunnels don't choke.
        return JSONResponse({"stats": stats, "video_filename": video_filename})

    finally:
        try:
            os.unlink(input_path)
        except Exception:
            pass


@app.get("/video/{filename}")
def get_video(filename: str):
    """Serve an annotated video file by name (binary MP4)."""
    # Sanitise — strip any path separators to prevent directory traversal
    safe_name = Path(filename).name
    path = ANNOTATED_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=safe_name,
    )


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
