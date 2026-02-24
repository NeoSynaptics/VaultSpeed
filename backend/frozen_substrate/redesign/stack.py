from __future__ import annotations

from typing import List

import numpy as np

from .config import SubstrateConfig
from .ops import blur3x3, clamp, global_rms_normalize


class SubstrateStack:
    """Multi-layer lossy transformation stack.

    A deterministic spatiotemporal substrate: a stack of H x W fields.
    Each step applies leakage, feedforward coupling, lateral diffusion,
    optional global RMS normalization, and hard clipping.
    """

    def __init__(self, cfg: SubstrateConfig, seed: int = 0):
        if cfg.n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._x = np.zeros((cfg.n_layers, cfg.height, cfg.width), dtype=np.float32)

    @property
    def x(self) -> np.ndarray:
        """Direct view of substrate state: shape (L, H, W)."""
        return self._x

    def reset(self) -> None:
        self._x.fill(0.0)

    def inject_l0(self, stim: np.ndarray, gain: float) -> None:
        if stim.shape != (self.cfg.height, self.cfg.width):
            raise ValueError(f"stim must be shape {(self.cfg.height, self.cfg.width)}")
        self._x[0] += (gain * stim).astype(np.float32)

    def step(self) -> None:
        c = self.cfg
        x = self._x

        if c.noise_std > 0:
            x += (c.noise_std * self.rng.standard_normal(x.shape)).astype(np.float32)

        for l in range(1, c.n_layers):
            prev = x[l - 1]
            ff = blur3x3(prev)
            ff *= 1.0 / (1.0 + c.depth_gain * (l - 1))
            lat = blur3x3(x[l])
            x_new = (
                c.leak * x[l]
                + c.feedforward * ff
                + c.lateral * (lat - x[l])
            ).astype(np.float32)

            if c.global_rms_norm:
                x_new = global_rms_normalize(x_new, target=c.rms_target, eps=c.rms_eps)
            x[l] = clamp(x_new, -c.clip_value, c.clip_value)

        x0 = (c.leak * x[0]).astype(np.float32)
        if c.global_rms_norm:
            x0 = global_rms_normalize(x0, target=c.rms_target, eps=c.rms_eps)
        x[0] = clamp(x0, -c.clip_value, c.clip_value)

    def layers(self) -> List[np.ndarray]:
        return [self._x[i] for i in range(self.cfg.n_layers)]
