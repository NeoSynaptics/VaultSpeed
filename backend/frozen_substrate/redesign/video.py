from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import VideoIOConfig
from .ops import blur3x3


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    raise ValueError("frame must be HxW or HxWx3(+)")


def _resize_nearest(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize (dependency-free)."""
    out_h, out_w = out_hw
    in_h, in_w = img.shape
    ys = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
    xs = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
    return img[ys[:, None], xs[None, :]].astype(np.float32)


def frame_to_stim(frame: np.ndarray, out_hw: Tuple[int, int], cfg: VideoIOConfig) -> np.ndarray:
    """Convert a video frame to an injection stimulus for Layer 0."""
    x = _to_grayscale(frame) if cfg.grayscale else frame.astype(np.float32)
    x = _resize_nearest(x, out_hw)

    if cfg.normalize:
        mn, mx = float(x.min()), float(x.max())
        if mx > mn:
            x = 2.0 * (x - mn) / (mx - mn) - 1.0
        else:
            x = x * 0.0

    if cfg.input_spatial_baseline:
        x = x - blur3x3(x)

    return x.astype(np.float32)
