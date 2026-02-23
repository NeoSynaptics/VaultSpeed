"""
VaultSpeed — Modal GPU cloud endpoint
Deploy: modal deploy backend/modal_app.py
"""

import modal
import io
import json
import tempfile
import os
from pathlib import Path

# ─── Modal app setup ──────────────────────────────────────────────────────────
app = modal.App("vaultspeed")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ultralytics==8.3.0",
        "opencv-python-headless==4.10.0.84",
        "numpy",
        "fastapi[standard]",
    )
    .run_commands("apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg")
)

# Persistent volume for storing previous run stats (delta comparison)
volume = modal.Volume.from_name("vaultspeed-runs", create_if_missing=True)
RUNS_DIR = "/runs"


# ─── Analysis function ────────────────────────────────────────────────────────
@app.function(
    gpu="T4",
    timeout=120,
    image=image,
    volumes={RUNS_DIR: volume},
)
@modal.web_endpoint(method="POST", label="analyze")
async def analyze(request: modal.Request) -> modal.Response:
    """
    Accepts multipart form:
      - video: video file bytes
      - runway_meters: float (default 40.0)
      - athlete_id: str (optional, for delta tracking)

    Returns multipart response:
      - annotated_video: mp4 bytes
      - stats: JSON {avg_kmh, peak_kmh, delta_kmh, pole_plant_frame, ...}
    """
    import sys
    sys.path.insert(0, "/root")

    from fastapi import UploadFile
    import cv2
    import numpy as np

    form = await request.form()

    video_file = form.get("video")
    runway_meters = float(form.get("runway_meters", 40.0))
    athlete_id = str(form.get("athlete_id", "default"))

    if video_file is None:
        return modal.Response(content=json.dumps({"error": "no video"}), status_code=400)

    video_bytes = await video_file.read()

    # Load previous avg for delta
    history_path = Path(RUNS_DIR) / f"{athlete_id}.json"
    prev_avg_kmh = None
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text())
            prev_avg_kmh = data.get("avg_kmh")
        except Exception:
            pass

    # Write input to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        input_path = tmp_in.name

    output_path = input_path.replace(".mp4", "_annotated.mp4")

    try:
        # Run analyzer (imported from same package)
        from analyzer import analyze_video
        stats = analyze_video(
            input_path=input_path,
            output_path=output_path,
            runway_meters=runway_meters,
            prev_avg_kmh=prev_avg_kmh,
        )

        # Save this run's avg for next delta
        history_path.write_text(json.dumps({"avg_kmh": stats["avg_kmh"]}))
        volume.commit()

        # Read annotated video
        with open(output_path, "rb") as f:
            annotated_bytes = f.read()

        # Return multipart: video + stats JSON
        from fastapi.responses import JSONResponse
        import base64

        return modal.Response(
            content=json.dumps({
                "stats": stats,
                "video_b64": base64.b64encode(annotated_bytes).decode(),
            }),
            media_type="application/json",
        )

    finally:
        for p in [input_path, output_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


# ─── Health check ─────────────────────────────────────────────────────────────
@app.function(image=image)
@modal.web_endpoint(method="GET", label="health")
def health() -> dict:
    return {"status": "ok", "service": "vaultspeed"}
