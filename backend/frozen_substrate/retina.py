"""Retina + Deviation Channel B depth stack.

This module implements:
- RetinaDepthStack: L0 retina buffer (existence) + L1+ deviation physics
  with z-score band-pass gating.
- IntegratedResidual: Channel B temporal integral of residual novelty
  across the depth stack.

Key insight: mid-entropy stimuli (micro-motion) penetrate deeper than
static objects or fast flicker, matching biological persistence behavior.
"""

import numpy as np


def blur3x3(x: np.ndarray) -> np.ndarray:
    xp = np.pad(x, ((1, 1), (1, 1)), mode="edge")
    y = (
        xp[0:-2, 0:-2] + 2 * xp[0:-2, 1:-1] + xp[0:-2, 2:]
        + 2 * xp[1:-1, 0:-2] + 4 * xp[1:-1, 1:-1] + 2 * xp[1:-1, 2:]
        + xp[2:, 0:-2] + 2 * xp[2:, 1:-1] + xp[2:, 2:]
    ) / 16.0
    return y.astype(np.float32)


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi).astype(np.float32)


class RetinaDepthStack:
    """L0 retina buffer + L1+ deviation-gated depth propagation.

    L0 acts as an EMA retina buffer that registers existence of stimuli.
    L1 receives z-score deviations from L0, band-pass gated so only
    mid-entropy changes propagate. L2+ are passive depth layers.
    """

    def __init__(self, H=50, W=50, L=8,
                 l0_leak=0.995,
                 clip=1.0,
                 clip_l0=3.0,
                 leak1=0.88, leak_depth_gain=0.012,
                 ff0=0.16, ff_depth_gain=0.30,
                 lateral=0.015,
                 mu_alpha=0.02,
                 var_alpha=0.02,
                 z_eps=1e-4,
                 z_low=0.55,
                 z_high=2.2,
                 z_hard=4.0):
        self.H, self.W, self.L = H, W, L
        self.x = np.zeros((L, H, W), dtype=np.float32)

        self.l0_leak = float(l0_leak)
        self.clip = float(clip)
        self.clip_l0 = float(clip_l0)

        self.leak1 = float(leak1)
        self.leak_depth_gain = float(leak_depth_gain)
        self.ff0 = float(ff0)
        self.ff_depth_gain = float(ff_depth_gain)
        self.lateral = float(lateral)

        self.mu = np.zeros((H, W), dtype=np.float32)
        self.var = np.ones((H, W), dtype=np.float32) * 1e-2
        self.mu_alpha = float(mu_alpha)
        self.var_alpha = float(var_alpha)
        self.z_eps = float(z_eps)
        self._init_stats = False

        self.z_low = float(z_low)
        self.z_high = float(z_high)
        self.z_hard = float(z_hard)

    def inject_frame_to_l0(self, frame: np.ndarray, gain: float = 1.0):
        """L0 retina buffer (EMA): keeps background without unbounded accumulation."""
        a0 = 1.0 - float(self.l0_leak)
        a0 = float(np.clip(a0, 0.0, 1.0))
        self.x[0] = (1.0 - a0) * self.x[0] + a0 * (gain * frame.astype(np.float32))
        self.x[0] = clamp(self.x[0], -self.clip_l0, self.clip_l0)

    def _compute_z(self):
        x0 = self.x[0]
        if not self._init_stats:
            self.mu[:] = x0
            self.var[:] = 1e-2
            self._init_stats = True
        else:
            a = self.mu_alpha
            self.mu[:] = (1.0 - a) * self.mu + a * x0
            d = (x0 - self.mu)
            b = self.var_alpha
            self.var[:] = (1.0 - b) * self.var + b * (d * d)
        return ((x0 - self.mu) / np.sqrt(self.var + self.z_eps)).astype(np.float32)

    def _bandpass(self, z: np.ndarray) -> np.ndarray:
        az = np.abs(z)
        gate = np.clip((az - self.z_low) / max(1e-6, (self.z_high - self.z_low)), 0.0, 1.0)
        if self.z_hard > self.z_high:
            decay = np.clip((self.z_hard - az) / max(1e-6, (self.z_hard - self.z_high)), 0.0, 1.0)
            gate = gate * decay
        return (gate * z).astype(np.float32)

    def step(self):
        z = self._compute_z()
        z_bp = self._bandpass(z)
        self.x[1] = clamp(self.x[1] + z_bp, -self.clip, self.clip)

        for l in range(2, self.L):
            leak = min(max(self.leak1 + self.leak_depth_gain * (l - 1), 0.0), 0.995)
            ff = blur3x3(self.x[l - 1])
            ff_gain = self.ff0 / (1.0 + self.ff_depth_gain * (l - 2))
            lat = blur3x3(self.x[l])
            x_new = leak * self.x[l] + ff_gain * ff + self.lateral * (lat - self.x[l])
            self.x[l] = clamp(x_new, -self.clip, self.clip)

        self.x[1] = clamp(0.96 * self.x[1], -self.clip, self.clip)


class IntegratedResidual:
    """Channel B: temporal integral of residual novelty across depth layers.

    Tracks how much each spatial location deviates from its running
    baseline at each depth layer. The integrated residual serves as
    a persistence metric -- stimuli that consistently produce novelty
    accumulate higher Channel B values.
    """

    def __init__(self, L, H, W, baseline_alpha=0.04, integral_beta=0.12):
        self.base = np.zeros((L, H, W), dtype=np.float32)
        self.I = np.zeros((L, H, W), dtype=np.float32)
        self.a = float(baseline_alpha)
        self.b = float(integral_beta)
        self._init = False

    def step(self, x: np.ndarray) -> np.ndarray:
        if not self._init:
            self.base[:] = x
            self.I[:] = 0.0
            self._init = True
        else:
            self.base[:] = (1.0 - self.a) * self.base + self.a * x

        Ahat = np.empty_like(x)
        Bhat = np.empty_like(self.base)
        for l in range(x.shape[0]):
            Ahat[l] = x[l] - blur3x3(x[l])
            Bhat[l] = self.base[l] - blur3x3(self.base[l])
        R = (Ahat - Bhat).astype(np.float32)
        self.I[:] = (1.0 - self.b) * self.I + self.b * R
        return np.abs(self.I).astype(np.float32)
