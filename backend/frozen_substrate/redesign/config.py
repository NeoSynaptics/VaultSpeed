from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class SubstrateConfig:
    """Mechanical substrate configuration.

    All operations are local and deterministic (except optional tiny noise).
    """
    height: int = 50
    width: int = 50
    n_layers: int = 10

    # Dynamics
    leak: float = 0.93                 # memory
    feedforward: float = 0.08          # coupling from layer l-1 to l
    lateral: float = 0.02              # diffusion strength within a layer
    depth_gain: float = 0.15           # attenuate deeper layers: gain(l)=1/(1+depth_gain*l)

    # Safety / stability
    clip_value: float = 1.0            # hard bound on activation
    global_rms_norm: bool = True       # prevents saturation / flooding
    rms_target: float = 0.25           # typical activation scale if global_rms_norm enabled
    rms_eps: float = 1e-6

    # Optional noise (kept tiny; helps avoid exact-zero lock)
    noise_std: float = 0.0

    @classmethod
    def default(cls) -> SubstrateConfig:
        """50x50, 10 layers -- good balance for most inputs."""
        return cls()

    @classmethod
    def high_res(cls) -> SubstrateConfig:
        """128x128, 15 layers -- more spatial detail and deeper propagation."""
        return cls(height=128, width=128, n_layers=15)

    @classmethod
    def fast(cls) -> SubstrateConfig:
        """32x32, 6 layers -- quick experiments and real-time preview."""
        return cls(height=32, width=32, n_layers=6)


@dataclass(frozen=True)
class ReadoutConfig:
    """Readout configuration for Channel A/B."""
    # Channel A: boundary layers (existence)
    a_layers: Tuple[int, ...] = (0, 1, 2)

    # Channel B: mid layers (persistence)
    b_layers: Tuple[int, ...] = (3, 4, 5, 6)

    # Baseline removal
    baseline_alpha: float = 0.02   # EMA speed; smaller = slower expectation
    spatial_baseline: bool = True  # subtract blur(A) and blur(B) before residual

    # B gating / stability
    b_threshold: float = 0.06
    flood_fraction: float = 0.30   # if >30% pixels exceed threshold, scale down
    flood_scale: float = 0.35      # scaling factor applied under flood

    # Temporal integration for Channel B
    integrate_steps: int = 4       # number of substrate steps per output
    b_policy: str = "max"          # "max" or "mean" across integrate window

    @classmethod
    def for_substrate(cls, scfg: SubstrateConfig) -> ReadoutConfig:
        """Auto-configure readout layers based on substrate depth."""
        n = scfg.n_layers
        a_end = max(1, n // 3)
        b_start = a_end
        b_end = max(b_start + 1, n - 1)
        return cls(
            a_layers=tuple(range(a_end)),
            b_layers=tuple(range(b_start, b_end)),
        )


@dataclass(frozen=True)
class VideoIOConfig:
    """Video adaptation configuration for Layer 0 injection."""
    output_fps: int = 12

    # Frame preprocessing (mechanical)
    grayscale: bool = True
    normalize: bool = True          # map to [-1,1]
    input_gain: float = 0.35        # injection amplitude

    # Optional spatial high-pass at input
    input_spatial_baseline: bool = False
