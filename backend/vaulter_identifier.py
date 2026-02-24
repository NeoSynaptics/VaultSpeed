"""
VaultSpeed — Vaulter Identification Module

Identifies which ByteTrack track corresponds to the pole vaulter.
Three-signal ensemble inside VaulterIdentifier:

  1. Frozen Substrate Channel B (primary, ~60%)
     Bio-inspired pre-layer: runs on full frames, produces a spatial
     motion-entropy map. Channel B = persistent structured motion.
     The pole vaulter is the dominant mid/high entropy object.

  2. Displacement + R² heuristic (fallback, ~40%)
     Net displacement weighted by directional consistency.
     Always available, no dependencies.

  3. V-JEPA2 ViT-L motion energy (optional boost, ~30%)
     Deep learning temporal embeddings from cropped person clips.
     Contributes only when model is loaded; degrades gracefully.
"""

import cv2
import numpy as np
from typing import Optional


# ─── Frozen Substrate pre-layer ───────────────────────────────────────────────

class FrozenSubstrateScorer:
    """
    Runs Frozen Substrate (V1 Pipeline, fast 32×32 preset) on all video frames
    to produce a spatial motion-entropy map.

    Channel B = residual activity in mid/deep substrate layers = what PERSISTS
    after depth degradation = sustained, structured motion.

    The pole vaulter — the only person accelerating in a straight line — creates
    the dominant high-entropy hot zone. Spectators standing still → Channel B ≈ 0.

    Returns normalized 0-1 scores per track: higher = more structured motion at
    that track's centroid positions = more likely to be the vaulter.
    """

    def score_tracks(self, tracks: dict, frames: list) -> dict:
        """
        Returns {track_id: float} normalized 0-1.
        Returns {} if frozen_substrate is unavailable.
        """
        try:
            from frozen_substrate.redesign.pipeline import Pipeline
            from frozen_substrate.redesign.config import (
                SubstrateConfig, ReadoutConfig, VideoIOConfig
            )
        except ImportError as e:
            print(f"[fs] frozen_substrate unavailable: {e}")
            return {}

        H, W = frames[0].shape[:2]
        scfg = SubstrateConfig.fast()           # 32×32 grid, 6 layers — fast
        rcfg = ReadoutConfig.for_substrate(scfg)
        vcfg = VideoIOConfig(grayscale=True, normalize=True, input_gain=0.35)
        pipeline = Pipeline(scfg, rcfg, vcfg)

        gh, gw = scfg.height, scfg.width        # 32, 32

        # Process all frames; assign each output cube's Channel B map back to
        # the window of frames that produced it (every integrate_steps frames).
        entropy_maps: dict[int, np.ndarray] = {}  # frame_idx → (gh, gw) b_map

        for fi, frame in enumerate(frames):
            result = pipeline.process_frame(frame)
            if result is None:
                continue

            cube, meta = result
            b_layers = list(meta["b_layers"])         # direct cube indices
            b_map = cube[b_layers].mean(axis=0)       # (gh, gw) Channel B mean
            steps = meta["integrate_steps"]

            # Back-assign to each frame in the just-completed window
            for k in range(steps):
                idx = fi - steps + 1 + k
                if 0 <= idx < len(frames):
                    entropy_maps[idx] = b_map

        if not entropy_maps:
            print("[fs] no entropy maps produced (too few frames?)")
            return {}

        print(f"[fs] entropy maps for {len(entropy_maps)}/{len(frames)} frames  "
              f"grid={gh}×{gw}  b_layers={meta['b_layers']}")

        # Score each track: mean Channel B at its centroid positions
        raw_scores: dict[int, float] = {}
        for tid, track in tracks.items():
            vals = []
            for i, centroid in enumerate(track["centroids"]):
                if centroid is None or i not in entropy_maps:
                    continue
                cx, cy = centroid
                # Map video pixel → substrate grid cell
                sx = int(cx / W * gw)
                sy = int(cy / H * gh)
                sx = max(0, min(gw - 1, sx))
                sy = max(0, min(gh - 1, sy))
                vals.append(float(entropy_maps[i][sy, sx]))
            raw_scores[tid] = float(np.mean(vals)) if vals else 0.0

        score_str = "  ".join(f"t{k}={v:.4f}" for k, v in sorted(raw_scores.items()))
        print(f"[fs] raw entropy scores: {score_str}")

        # Normalize to 0-1
        max_s = max(raw_scores.values()) if raw_scores else 1.0
        if max_s < 1e-9:
            return {tid: 0.0 for tid in raw_scores}
        return {tid: v / max_s for tid, v in raw_scores.items()}


# ─── Main identifier ───────────────────────────────────────────────────────────

class VaulterIdentifier:
    """
    Selects which ByteTrack track is the pole vaulter using a three-signal ensemble:
      1. Frozen Substrate Channel B (spatial entropy pre-layer) — primary
      2. Displacement + R² heuristic — always-available fallback
      3. V-JEPA2 ViT-L motion energy — optional deep learning boost
    """

    def __init__(self):
        self._vjepa_model = None
        self._vjepa_processor = None
        self._vjepa_available: Optional[bool] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def identify(self, tracks: dict, frames: list) -> int:
        """Return track_id of the most likely pole vaulter."""
        # Filter: need at least 20 detected frames to be a viable candidate
        viable = {
            tid: t for tid, t in tracks.items()
            if sum(1 for c in t["centroids"] if c is not None) >= 20
        }
        if not viable:
            best = max(tracks, key=lambda t: sum(
                1 for c in tracks[t]["centroids"] if c is not None))
            print(f"[vaulter] no viable tracks (≥20 det), fallback to longest: {best}")
            return best

        # ── Signal 1: Frozen Substrate spatial entropy (primary) ─────────────
        fs_scores = FrozenSubstrateScorer().score_tracks(viable, frames)

        # ── Signal 2: Displacement + R² heuristic (always available) ─────────
        h_raw = {tid: self._heuristic_score(t) for tid, t in viable.items()}
        h_max = max(h_raw.values()) or 1.0
        h_scores = {tid: v / h_max for tid, v in h_raw.items()}

        # ── Signal 3: V-JEPA2 motion energy (optional) ───────────────────────
        vj_scores: dict[int, float] = {}
        for tid, track in viable.items():
            s = self._motion_energy_score(track, frames)
            if s is not None:
                vj_scores[tid] = s
        if vj_scores:
            vj_max = max(vj_scores.values()) or 1.0
            vj_scores = {tid: v / vj_max for tid, v in vj_scores.items()}

        # ── Blend (all normalized 0-1) ────────────────────────────────────────
        scores: dict[int, float] = {}
        for tid in viable:
            fs = fs_scores.get(tid, 0.0)
            h  = h_scores.get(tid, 0.0)
            vj = vj_scores.get(tid)

            if fs_scores and vj is not None:
                scores[tid] = fs * 0.50 + h * 0.20 + vj * 0.30
            elif fs_scores:
                scores[tid] = fs * 0.60 + h * 0.40
            else:
                scores[tid] = h     # pure heuristic fallback

        best = max(scores, key=scores.get)
        score_str = "  ".join(f"t{k}={v:.3f}" for k, v in sorted(scores.items()))
        print(f"[vaulter] selected track {best}  ({score_str})")
        return best

    # ── Heuristic scoring ─────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_score(track: dict) -> float:
        """Net displacement * R² direction consistency."""
        pts = [c for c in track["centroids"] if c is not None]
        if len(pts) < 5:
            return 0.0

        net_disp = np.hypot(pts[-1][0] - pts[0][0],
                            pts[-1][1] - pts[0][1])

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        t  = np.arange(len(xs), dtype=float)

        dx_range = max(xs) - min(xs)
        dy_range = max(ys) - min(ys)
        vals = xs if dx_range >= dy_range else ys

        if len(set(vals)) > 1 and len(t) > 2:
            r2 = np.corrcoef(t, vals)[0, 1] ** 2
        else:
            r2 = 0.0

        return net_disp * (0.3 + 0.7 * r2)

    # ── V-JEPA2 motion energy scoring ─────────────────────────────────────────

    def _motion_energy_score(self, track: dict, frames: list) -> Optional[float]:
        """
        V-JEPA2 temporal embedding variance.
        Higher variance = more active person = more likely the vaulter.
        Returns None if V-JEPA2 is unavailable.
        """
        if not self._ensure_vjepa():
            return None

        import torch

        clips = self._extract_clips(track, frames, n_frames=16)
        if clips is None:
            return None

        try:
            inputs = self._vjepa_processor(clips, return_tensors="pt")
            device = next(self._vjepa_model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self._vjepa_model.get_vision_features(**inputs)

            if embeddings.dim() >= 2 and embeddings.shape[0] > 1:
                diffs = embeddings[1:] - embeddings[:-1]
                energy = float(diffs.norm(dim=-1).mean())
            else:
                energy = float(embeddings.norm())

            return energy

        except Exception as e:
            print(f"[vjepa2] inference failed: {e}")
            return None

    @staticmethod
    def _extract_clips(track: dict, frames: list, n_frames: int = 16) -> Optional[list]:
        """Crop person bbox from evenly-spaced detected frames, resize to 256×256."""
        detected = [
            (i, track["bboxes"][i])
            for i in range(len(frames))
            if i < len(track["bboxes"]) and track["bboxes"][i] is not None
        ]
        if len(detected) < n_frames:
            return None

        indices = np.linspace(0, len(detected) - 1, n_frames, dtype=int)
        clips = []
        for idx in indices:
            fi, bbox = detected[idx]
            x0, y0, x1, y1 = [int(v) for v in bbox]
            h, w = frames[fi].shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            crop = frames[fi][y0:y1, x0:x1]
            if crop.size == 0:
                return None
            crop = cv2.resize(crop, (256, 256))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            clips.append(crop)

        return clips

    # ── V-JEPA2 model loading ─────────────────────────────────────────────────

    def _ensure_vjepa(self) -> bool:
        """Lazy-load V-JEPA2 ViT-L. Returns False if unavailable."""
        if self._vjepa_available is False:
            return False
        if self._vjepa_model is not None:
            return True

        try:
            from transformers import AutoVideoProcessor, AutoModel
            repo = "facebook/vjepa2-vitl-fpc64-256"
            print(f"[vjepa2] loading {repo}...")
            self._vjepa_processor = AutoVideoProcessor.from_pretrained(repo)
            self._vjepa_model = AutoModel.from_pretrained(repo)
            self._vjepa_model.eval()
            self._vjepa_available = True
            print("[vjepa2] ready")
            return True
        except Exception as e:
            print(f"[vjepa2] unavailable, skipping: {e}")
            self._vjepa_available = False
            return False
