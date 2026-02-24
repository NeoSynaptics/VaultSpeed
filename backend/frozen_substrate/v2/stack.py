"""ResonantStack: feedforward substrate with top-down feedback resonance."""
from __future__ import annotations

from typing import List

import numpy as np

from .config import ResonantConfig
from ..redesign.ops import blur3x3, clamp, global_rms_normalize


class ResonantStack:
    """Multi-layer substrate with feedback resonance from deep to shallow layers.

    Extends SubstrateStack with:
    - Top-down feedback: deep layers that exceed a threshold feed back
      to L0, creating a self-reinforcing resonance loop for persistent stimuli
    - Adaptive input normalization: auto-adjusts injection gain based on
      running signal statistics
    """

    def __init__(self, cfg: ResonantConfig, seed: int = 0):
        if cfg.n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._x = np.zeros((cfg.n_layers, cfg.height, cfg.width), dtype=np.float32)

        # Feedback state
        self._fb_layers = cfg.resolve_feedback_layers()

        # Adaptive input state
        self._input_rms_ema = float(cfg.input_rms_target)

    @property
    def x(self) -> np.ndarray:
        return self._x

    def reset(self) -> None:
        self._x.fill(0.0)
        self._input_rms_ema = float(self.cfg.input_rms_target)

    def inject_l0(self, stim: np.ndarray, gain: float) -> None:
        if stim.shape != (self.cfg.height, self.cfg.width):
            raise ValueError(f"stim must be shape {(self.cfg.height, self.cfg.width)}")

        if self.cfg.adaptive_input:
            rms = float(np.sqrt(np.mean(stim * stim) + 1e-8))
            a = self.cfg.input_rms_alpha
            self._input_rms_ema = (1.0 - a) * self._input_rms_ema + a * rms
            if self._input_rms_ema > 1e-6:
                effective_gain = gain * (self.cfg.input_rms_target / self._input_rms_ema)
            else:
                effective_gain = gain
        else:
            effective_gain = gain

        self._x[0] += (effective_gain * stim).astype(np.float32)

    def _apply_feedback(self) -> None:
        """Top-down feedback: deep layer activity feeds back to L0."""
        if not self._fb_layers:
            return

        cfg = self.cfg

        # Mean absolute activation across feedback layers
        deep = np.stack([self._x[l] for l in self._fb_layers], axis=0)
        deep_mean = np.mean(np.abs(deep), axis=0)  # (H, W)

        # Gate: only where deep activity exceeds threshold
        excess = deep_mean - cfg.feedback_threshold
        gate = np.clip(excess / max(1e-6, cfg.feedback_threshold), 0.0, 1.0)

        # Build feedback signal
        fb = (cfg.feedback_gain * gate * deep_mean).astype(np.float32)

        if cfg.feedback_blur:
            fb = blur3x3(fb)

        self._x[0] += fb

    def step(self) -> None:
        c = self.cfg
        x = self._x

        if c.noise_std > 0:
            x += (c.noise_std * self.rng.standard_normal(x.shape)).astype(np.float32)

        # Feedforward propagation (same as v1)
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

        # L0 decay
        x0 = (c.leak * x[0]).astype(np.float32)
        if c.global_rms_norm:
            x0 = global_rms_normalize(x0, target=c.rms_target, eps=c.rms_eps)
        x[0] = clamp(x0, -c.clip_value, c.clip_value)

        # Feedback resonance: deep -> L0
        self._apply_feedback()
        x[0] = clamp(x[0], -c.clip_value, c.clip_value)

    def layers(self) -> List[np.ndarray]:
        return [self._x[i] for i in range(self.cfg.n_layers)]
