from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import ReadoutConfig
from .ops import blur3x3


class Readout:
    """Passive readout that emits Channel A (existence) and Channel B (persistence).

    Channel B is computed on residual activation R = A - baseline(A),
    where the baseline is an EMA over time (mechanical expectation).
    """

    def __init__(self, cfg: ReadoutConfig, n_layers: int, height: int, width: int):
        self.cfg = cfg
        self._baseline = np.zeros((n_layers, height, width), dtype=np.float32)
        self._init = False

    def reset(self) -> None:
        self._baseline.fill(0.0)
        self._init = False

    def _update_baseline(self, x: np.ndarray) -> None:
        a = self.cfg.baseline_alpha
        if not self._init:
            self._baseline[:] = x
            self._init = True
            return
        self._baseline[:] = (1.0 - a) * self._baseline + a * x

    def _residual_per_layer(self, x: np.ndarray) -> np.ndarray:
        """Compute residuals with optional spatial baseline removal."""
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

    def emit_cube(self, x_window: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        """Emit one output cube from a window of substrate states.

        Parameters
        ----------
        x_window : np.ndarray
            Array of shape (T, L, H, W). T = integrate_steps.

        Returns
        -------
        cube : np.ndarray
            Array of shape (C, H, W).
        meta : dict
            Channel layer indices and flood event counts.
        """
        cfg = self.cfg
        T, L, H, W = x_window.shape

        x_last = x_window[-1]
        A_slices = np.stack([x_last[l] for l in cfg.a_layers], axis=0).astype(np.float32)

        B_accum = None
        flood_events = 0
        for ti in range(T):
            r = self._residual_per_layer(x_window[ti])
            r_mid = np.stack([r[l] for l in cfg.b_layers], axis=0)
            mag = np.abs(r_mid)

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

        cube = np.concatenate([A_slices, B_slices], axis=0)

        meta = {
            "a_layers": cfg.a_layers,
            "b_layers": cfg.b_layers,
            "b_policy": cfg.b_policy,
            "integrate_steps": cfg.integrate_steps,
            "flood_events_in_window": flood_events,
            "cube_channels": cube.shape[0],
        }
        return cube, meta
