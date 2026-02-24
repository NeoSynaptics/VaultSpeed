"""Utility functions for generating spatial drive patterns used to
stimulate a FrozenCoreV3 substrate.

The functions are stateless and return numpy arrays of shape (H, W)
suitable for injection via the substrate's add_drive() method.
"""

from __future__ import annotations

import numpy as np


def gaussian_bump(
    height: int,
    width: int,
    center_y: float,
    center_x: float,
    sigma: float = 2.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return a 2D Gaussian bump (a "pen tip")."""
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    g = np.exp(-(((y - center_y) ** 2 + (x - center_x) ** 2) / (2.0 * sigma ** 2))).astype(np.float32)
    if amplitude != 1.0:
        g *= np.float32(amplitude)
    return g


def orbit_center(
    t: int,
    center_y: float,
    center_x: float,
    radius: float,
    period: int,
    phase: float = 0.0,
) -> tuple[float, float]:
    """Return the (y, x) position of a point moving on a circle."""
    ang = (2.0 * np.pi * (t % period) / float(period)) + float(phase)
    return (
        float(center_y + radius * np.sin(ang)),
        float(center_x + radius * np.cos(ang)),
    )


def orbit_gaussian_pen(
    height: int,
    width: int,
    t: int,
    center_y: float,
    center_x: float,
    radius: float,
    period: int = 160,
    sigma: float = 2.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """A moving Gaussian "pen" that traces a circle over time."""
    cy, cx = orbit_center(t, center_y, center_x, radius, period, phase=phase)
    return gaussian_bump(height, width, cy, cx, sigma=sigma, amplitude=amplitude)


def gaussian_ring(
    height_or_shape,
    width_or_center=None,
    center_y: float | None = None,
    center_x: float | None = None,
    radius: float | None = None,
    sigma: float = 2.5,
) -> np.ndarray:
    """Return a 2D Gaussian ring.

    Supports two calling conventions:
    1) gaussian_ring(H, W, center_y, center_x, radius, sigma=2.5)
    2) gaussian_ring((H, W), (cy, cx), radius, sigma=2.5)
    """
    if isinstance(height_or_shape, (tuple, list)) and len(height_or_shape) == 2:
        height, width = int(height_or_shape[0]), int(height_or_shape[1])
        if not (isinstance(width_or_center, (tuple, list)) and len(width_or_center) == 2):
            raise TypeError("When first arg is a shape (H,W), second arg must be center (cy,cx).")
        cy, cx = float(width_or_center[0]), float(width_or_center[1])
        if center_y is None:
            raise TypeError("Missing radius argument: gaussian_ring((H,W),(cy,cx), radius, sigma=...)")
        r = float(center_y)
        sig = float(sigma)
    else:
        height, width = int(height_or_shape), int(width_or_center)
        if center_y is None or center_x is None or radius is None:
            raise TypeError("Expected gaussian_ring(H, W, cy, cx, radius, sigma=...).")
        cy, cx = float(center_y), float(center_x)
        r = float(radius)
        sig = float(sigma)

    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    ring = np.exp(-((dist - r) ** 2) / (2.0 * sig ** 2)).astype(np.float32)
    return ring
