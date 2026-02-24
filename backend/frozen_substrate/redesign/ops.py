from __future__ import annotations

import numpy as np


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def blur3x3(x: np.ndarray) -> np.ndarray:
    """Cheap blur using a 3x3 binomial kernel (edge padded)."""
    xp = np.pad(x, ((1, 1), (1, 1)), mode="edge")
    y = (
        xp[0:-2, 0:-2] + 2 * xp[0:-2, 1:-1] + xp[0:-2, 2:]
        + 2 * xp[1:-1, 0:-2] + 4 * xp[1:-1, 1:-1] + 2 * xp[1:-1, 2:]
        + xp[2:, 0:-2] + 2 * xp[2:, 1:-1] + xp[2:, 2:]
    ) / 16.0
    return y.astype(np.float32)


def global_rms_normalize(x: np.ndarray, target: float, eps: float) -> np.ndarray:
    """Scale x so its RMS is approximately 'target'."""
    rms = float(np.sqrt(np.mean(x * x) + eps))
    if rms < eps:
        return x
    return (x * (target / rms)).astype(np.float32)
