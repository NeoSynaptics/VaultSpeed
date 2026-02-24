"""Coupling utilities for wiring multiple FrozenCoreV3 substrates together.

Defines helper functions to downsample one layer's fields to another
layer's resolution and a high-level driver for the two-layer demo.
FrozenCoreV3 remains a self-contained physics engine; this module
provides the plumbing to stack instances.
"""

import numpy as np

from frozen_substrate.core import FrozenCoreV3
from frozen_substrate.gaussian_pen import orbit_gaussian_pen


def downsample(field: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Average pool a field from (H,W) to (h1,w1).

    Requires that the original dimensions are divisible by the target
    dimensions. Pools over non-overlapping blocks and returns block averages.
    """
    H, W = field.shape
    h1, w1 = target_shape
    if H % h1 != 0 or W % w1 != 0:
        raise ValueError("Field dimensions must be divisible by target shape")
    fy, fx = H // h1, W // w1
    return field.reshape(h1, fy, w1, fx).mean(axis=(1, 3))


def run_two_layer_demo(
    *,
    steps: int = 4000,
    L0_shape: tuple[int, int] = (40, 40),
    L1_shape: tuple[int, int] = (20, 20),
    pen_radius: float = 8.0,
    pen_period: int = 160,
    pen_sigma: float = 2.0,
    pen_gain_L0: float = 0.20,
    emitter_ema_alpha: float = 0.10,
    l1_drive_gain: float = 2.0,
    l0_params: dict | None = None,
    l1_params: dict | None = None,
) -> dict:
    """Run a two-layer Gaussian ring experiment.

    Constructs two FrozenCoreV3 layers (L0 and L1) and injects a moving
    Gaussian pen into L0 at every step. An EMA of the pen is downsampled
    and fed as drive to L1. Returns time-series traces and snapshots.
    """
    l0 = FrozenCoreV3(L0_shape[0], L0_shape[1], **(l0_params or {}))
    l1 = FrozenCoreV3(L1_shape[0], L1_shape[1], **(l1_params or {}))
    l0.enable_pacemakers = False
    l1.enable_pacemakers = False

    cy0 = (L0_shape[0] - 1) / 2.0
    cx0 = (L0_shape[1] - 1) / 2.0

    ema_emitter = np.zeros(L0_shape, dtype=np.float32)

    trace_error_L0: list[float] = []
    trace_error_L1: list[float] = []
    trace_alive_L0: list[float] = []
    trace_alive_L1: list[float] = []

    sample_times = [0, steps // 3, 2 * steps // 3, steps - 1]
    snapshots: list[dict] = []

    for t in range(steps):
        pen = orbit_gaussian_pen(
            L0_shape[0], L0_shape[1], t,
            center_y=cy0, center_x=cx0,
            radius=pen_radius,
            period=pen_period,
            sigma=pen_sigma,
            amplitude=1.0,
        )
        l0.add_drive(pen, gain=pen_gain_L0)
        l0.step()

        ema_emitter = emitter_ema_alpha * pen + (1.0 - emitter_ema_alpha) * ema_emitter
        drive = downsample(ema_emitter, L1_shape)
        l1.add_drive(drive, gain=l1_drive_gain)
        l1.step()

        err0 = np.abs(l0.e[l0.alive.astype(bool)])
        err1 = np.abs(l1.e[l1.alive.astype(bool)])
        trace_error_L0.append(float(err0.mean() if err0.size else 0.0))
        trace_error_L1.append(float(err1.mean() if err1.size else 0.0))
        trace_alive_L0.append(float(np.mean(l0.alive)))
        trace_alive_L1.append(float(np.mean(l1.alive)))

        if t in sample_times:
            snapshots.append({
                "t": t,
                "x0": l0.x.copy(),
                "A_mid0": l0.A_mid.copy(),
                "C0": l0.C.copy(),
                "x1": l1.x.copy(),
                "A_mid1": l1.A_mid.copy(),
                "C1": l1.C.copy(),
            })

    return {
        "trace_error_L0": trace_error_L0,
        "trace_error_L1": trace_error_L1,
        "trace_alive_L0": trace_alive_L0,
        "trace_alive_L1": trace_alive_L1,
        "snapshots": snapshots,
    }
