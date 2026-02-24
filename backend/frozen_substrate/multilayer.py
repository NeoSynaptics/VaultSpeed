"""Multi-layer passive depth stack for the Frozen Core project.

L0 = FrozenCoreV3 (active substrate), deeper layers are passive filters
with no death, no plasticity. Propagation is causal, local, and stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from frozen_substrate.core import FrozenCoreV3


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _blur3x3(x: np.ndarray) -> np.ndarray:
    """Cheap 3x3 blur with edge-padding."""
    xp = np.pad(x, ((1, 1), (1, 1)), mode="edge")
    y = (
        xp[0:-2, 0:-2] + 2 * xp[0:-2, 1:-1] + xp[0:-2, 2:]
        + 2 * xp[1:-1, 0:-2] + 4 * xp[1:-1, 1:-1] + 2 * xp[1:-1, 2:]
        + xp[2:, 0:-2] + 2 * xp[2:, 1:-1] + xp[2:, 2:]
    ) / 16.0
    return y.astype(np.float32)


@dataclass
class PassiveLayerParams:
    leak: float = 0.93
    feedforward: float = 0.08
    lateral: float = 0.02
    noise: float = 0.001
    clamp: float = 1.0
    mean_center: bool = True
    center_strength: float = 0.05


class MultiLayerFrozenSubstrate:
    """L0 = FrozenCoreV3, deeper layers are passive fields.

    Fixes two common failure modes:
      (1) deeper layers saturating to a constant baseline,
      (2) deeper layers dying out due to pruning/death rules.

    Because layers 1..N are passive, their state remains predictable
    over long runs (10k+ steps) without resets.
    """

    def __init__(
        self,
        H: int = 50,
        W: int = 50,
        n_layers: int = 10,
        seed: int = 0,
        passive: Optional[PassiveLayerParams] = None,
        core_kwargs: Optional[dict] = None,
    ):
        assert n_layers >= 2, "Need at least 2 layers"
        self.H, self.W = int(H), int(W)
        self.n_layers = int(n_layers)
        self.rng = np.random.default_rng(seed)
        self.passive = passive or PassiveLayerParams()

        self.core = FrozenCoreV3(H=self.H, W=self.W, seed=seed, **(core_kwargs or {}))

        # Passive states (L1..L{n-1})
        self.x_passive = np.zeros((self.n_layers - 1, self.H, self.W), dtype=np.float32)

    @property
    def x_layers(self) -> List[np.ndarray]:
        """List of x fields for all layers (views)."""
        return [self.core.x] + [self.x_passive[i] for i in range(self.n_layers - 1)]

    def inject(self, stim: np.ndarray, amp: float = 1.0) -> None:
        """Inject external stimulus into L0 (alive-masked)."""
        self.core.x += (amp * stim).astype(np.float32) * self.core.alive

    def step(self) -> None:
        """Advance one time step."""
        self.core.step()

        p = self.passive
        prev = self.core.x
        for i in range(self.n_layers - 1):
            x = self.x_passive[i]

            ff = _blur3x3(prev)
            depth_gain = 1.0 / (1.0 + 0.15 * i)
            ff = ff * depth_gain
            ff = ff + (1e-4 * self.rng.standard_normal(ff.shape))

            lat = _blur3x3(x)

            x_new = (
                p.leak * x
                + p.feedforward * ff
                + p.lateral * (lat - x)
            )

            if p.noise > 0:
                x_new += (p.noise * self.rng.standard_normal(x.shape)).astype(np.float32)
            if p.mean_center:
                x_new = x_new - (p.center_strength * float(x_new.mean()))

            self.x_passive[i] = _clamp(x_new, -p.clamp, p.clamp)
            prev = self.x_passive[i]

    def persistence_depth_map(
        self,
        threshold: float = 0.15,
        layers: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Return a (H,W) map with the deepest layer index whose |x| exceeds threshold."""
        if layers is None:
            layers = range(self.n_layers)
        xs = self.x_layers
        depth = np.zeros((self.H, self.W), dtype=np.int16)
        for k in layers:
            m = (np.abs(xs[k]) >= threshold)
            depth[m] = np.maximum(depth[m], k)
        return depth

    def snapshot(self, layer_ids: Sequence[int]) -> Dict[int, np.ndarray]:
        """Copy-out snapshots of selected layers."""
        xs = self.x_layers
        return {int(k): xs[int(k)].copy() for k in layer_ids}
