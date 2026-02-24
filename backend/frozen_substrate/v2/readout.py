"""ReadoutV2: Channel A + B + C with local flood clamping and temporal coherence."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import ReadoutV2Config
from ..redesign.ops import blur3x3


class ReadoutV2:
    """Extended readout with Channel C (temporal coherence) and local flood clamping.

    Channel A: existence (shallow layer states) -- same as v1
    Channel B: persistence (deviation magnitude) -- same as v1, with optional local flood
    Channel C: coherence (temporal consistency of deviation direction)
        - High when deviation is structured (oscillation, drift)
        - Low when deviation is random noise
        - Computed as |EMA(signed_dev)| / EMA(|dev|)
    """

    def __init__(self, cfg: ReadoutV2Config, n_layers: int, height: int, width: int):
        self.cfg = cfg
        self._n_layers = n_layers
        self._height = height
        self._width = width

        self._baseline = np.zeros((n_layers, height, width), dtype=np.float32)
        self._init = False

        # Channel C coherence state
        c_layers = cfg.resolve_c_layers()
        n_c = len(c_layers)
        self._c_layers = c_layers
        if n_c > 0:
            self._dev_signed_ema = np.zeros((n_c, height, width), dtype=np.float32)
            self._dev_mag_ema = np.zeros((n_c, height, width), dtype=np.float32)
            self._coherence_init = False

    def reset(self) -> None:
        self._baseline.fill(0.0)
        self._init = False
        if len(self._c_layers) > 0:
            self._dev_signed_ema.fill(0.0)
            self._dev_mag_ema.fill(0.0)
            self._coherence_init = False

    def _update_baseline(self, x: np.ndarray) -> None:
        a = self.cfg.baseline_alpha
        if not self._init:
            self._baseline[:] = x
            self._init = True
            return
        self._baseline[:] = (1.0 - a) * self._baseline + a * x

    def _residual_per_layer(self, x: np.ndarray) -> np.ndarray:
        """Compute signed residuals (same as v1)."""
        self._update_baseline(x)
        A = x
        B = self._baseline
        if not self.cfg.spatial_baseline:
            return (A - B).astype(np.float32)

        Ahat = np.empty_like(A)
        Bhat = np.empty_like(B)
        for l in range(A.shape[0]):
            Ahat[l] = A[l] - blur3x3(A[l])
            Bhat[l] = B[l] - blur3x3(B[l])
        return (Ahat - Bhat).astype(np.float32)

    def _update_coherence(self, residuals: np.ndarray) -> np.ndarray:
        """Update coherence tracking and return current coherence maps.

        Coherence = |EMA(signed_dev)| / EMA(|dev|)
        - Structured deviation: signed EMA tracks it -> ratio ~1
        - Random noise: signed EMA cancels -> ratio ~0
        """
        if not self._c_layers:
            return np.zeros((0, self._height, self._width), dtype=np.float32)

        c_res = np.stack([residuals[l] for l in self._c_layers], axis=0)
        a = self.cfg.coherence_alpha

        if not self._coherence_init:
            self._dev_signed_ema[:] = c_res
            self._dev_mag_ema[:] = np.abs(c_res)
            self._coherence_init = True
        else:
            self._dev_signed_ema[:] = (1.0 - a) * self._dev_signed_ema + a * c_res
            self._dev_mag_ema[:] = (1.0 - a) * self._dev_mag_ema + a * np.abs(c_res)

        coherence = np.abs(self._dev_signed_ema) / (self._dev_mag_ema + 1e-8)
        return np.clip(coherence, 0.0, 1.0).astype(np.float32)

    def _local_flood_scale(self, mag: np.ndarray) -> Tuple[np.ndarray, int]:
        """Compute per-patch flood scaling.

        Returns (scale_array, flood_count).
        """
        cfg = self.cfg
        ps = cfg.flood_patch_size
        shape = mag.shape  # (n_b, H, W)
        H, W = shape[-2], shape[-1]
        scale = np.ones(shape, dtype=np.float32)
        flood_count = 0

        for y0 in range(0, H, ps):
            for x0 in range(0, W, ps):
                y1 = min(y0 + ps, H)
                x1 = min(x0 + ps, W)
                patch = mag[..., y0:y1, x0:x1]
                frac = float((patch > cfg.b_threshold).mean())
                if frac > cfg.flood_fraction:
                    scale[..., y0:y1, x0:x1] = cfg.flood_scale
                    flood_count += 1

        return scale, flood_count

    def emit_cube(self, x_window: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        """Emit one output cube from a window of substrate states.

        Returns cube of shape (C_a + C_b + C_c, H, W) and metadata dict.
        """
        cfg = self.cfg
        T, L, H, W = x_window.shape

        # Channel A: last-frame snapshot of shallow layers
        x_last = x_window[-1]
        A_slices = np.stack([x_last[l] for l in cfg.a_layers], axis=0).astype(np.float32)

        # Channel B: accumulated deviation magnitude
        B_accum = None
        flood_events = 0
        coherence = None

        for ti in range(T):
            r = self._residual_per_layer(x_window[ti])

            # Update coherence tracking
            coherence = self._update_coherence(r)

            # B accumulation
            r_mid = np.stack([r[l] for l in cfg.b_layers], axis=0)
            mag = np.abs(r_mid)

            # Flood clamping (local or global)
            if cfg.flood_patch_size > 0:
                scale, fc = self._local_flood_scale(mag)
                if fc > 0:
                    r_mid = (r_mid * scale).astype(np.float32)
                    flood_events += fc
            else:
                frac = float((mag > cfg.b_threshold).mean())
                if frac > cfg.flood_fraction:
                    r_mid = (cfg.flood_scale * r_mid).astype(np.float32)
                    flood_events += 1

            if B_accum is None:
                B_accum = np.abs(r_mid) if cfg.b_policy == "max" else r_mid.copy()
            else:
                if cfg.b_policy == "max":
                    B_accum = np.maximum(B_accum, np.abs(r_mid))
                elif cfg.b_policy == "mean":
                    B_accum += r_mid
                else:
                    raise ValueError("b_policy must be 'max' or 'mean'")

        if cfg.b_policy == "mean":
            B_slices = (B_accum / float(T)).astype(np.float32)
        else:
            B_slices = B_accum.astype(np.float32)

        # Assemble cube: A + B + C
        parts = [A_slices, B_slices]
        if coherence is not None and coherence.shape[0] > 0:
            parts.append(coherence)

        cube = np.concatenate(parts, axis=0)

        c_layers = self._c_layers
        meta = {
            "a_layers": cfg.a_layers,
            "b_layers": cfg.b_layers,
            "c_layers": c_layers,
            "b_policy": cfg.b_policy,
            "integrate_steps": cfg.integrate_steps,
            "flood_events_in_window": flood_events,
            "cube_channels": cube.shape[0],
            "n_a": len(cfg.a_layers),
            "n_b": len(cfg.b_layers),
            "n_c": len(c_layers),
        }
        return cube, meta
