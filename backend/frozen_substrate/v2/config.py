"""V2 configuration: Resonant substrate with feedback, coherence, and adaptive input."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ResonantConfig:
    """Substrate config with top-down feedback resonance.

    Extends the v1 SubstrateConfig with:
    - Feedback loop from deep layers back to L0
    - Adaptive input normalization
    """
    height: int = 50
    width: int = 50
    n_layers: int = 10

    # Feedforward dynamics (same as v1)
    leak: float = 0.93
    feedforward: float = 0.08
    lateral: float = 0.02
    depth_gain: float = 0.15

    # Stability
    clip_value: float = 1.0
    global_rms_norm: bool = True
    rms_target: float = 0.25
    rms_eps: float = 1e-6
    noise_std: float = 0.0

    # --- NEW: Feedback resonance ---
    # Which deep layers feed back to L0 (empty = auto: deepest 1/3)
    feedback_layers: Tuple[int, ...] = ()
    feedback_gain: float = 0.06        # strength of feedback (small vs 0.08 feedforward)
    feedback_threshold: float = 0.04   # deep activity must exceed this to trigger feedback
    feedback_blur: bool = True         # spatially smooth feedback signal

    # --- NEW: Adaptive input normalization ---
    adaptive_input: bool = True
    input_rms_alpha: float = 0.01      # EMA speed for tracking input signal level
    input_rms_target: float = 0.30     # target RMS for auto-gain

    def resolve_feedback_layers(self) -> Tuple[int, ...]:
        """Return feedback layers, auto-selecting deepest 1/3 if empty."""
        if self.feedback_layers:
            return self.feedback_layers
        start = max(1, self.n_layers - self.n_layers // 3)
        return tuple(range(start, self.n_layers))

    @classmethod
    def default(cls) -> ResonantConfig:
        return cls()

    @classmethod
    def fast(cls) -> ResonantConfig:
        return cls(height=32, width=32, n_layers=6)

    @classmethod
    def high_res(cls) -> ResonantConfig:
        return cls(height=128, width=128, n_layers=15)


@dataclass(frozen=True)
class ReadoutV2Config:
    """Readout config with Channel C (temporal coherence) and local flood clamping.

    Output cube layout: [Channel A | Channel B | Channel C]
    - A: existence (shallow layer states)
    - B: persistence (deviation magnitude, same as v1)
    - C: coherence (how temporally structured the deviation is)
    """
    # Channel A
    a_layers: Tuple[int, ...] = (0, 1, 2)

    # Channel B
    b_layers: Tuple[int, ...] = (3, 4, 5, 6)

    # --- NEW: Channel C ---
    # Which layers to compute coherence for (empty = same as b_layers)
    c_layers: Tuple[int, ...] = ()
    coherence_alpha: float = 0.015     # EMA speed for signed deviation tracking

    # Baseline
    baseline_alpha: float = 0.02
    spatial_baseline: bool = True

    # Flood clamping
    b_threshold: float = 0.06
    flood_fraction: float = 0.30
    flood_scale: float = 0.35
    flood_patch_size: int = 0          # 0 = global (v1 behavior), >0 = local patch size

    # Integration
    integrate_steps: int = 4
    b_policy: str = "max"

    def resolve_c_layers(self) -> Tuple[int, ...]:
        """Return C layers, defaulting to b_layers if empty."""
        return self.c_layers if self.c_layers else self.b_layers

    @classmethod
    def for_substrate(cls, scfg: ResonantConfig) -> ReadoutV2Config:
        """Auto-configure readout layers based on substrate depth."""
        n = scfg.n_layers
        a_end = max(1, n // 3)
        b_start = a_end
        b_end = max(b_start + 1, n - 1)
        return cls(
            a_layers=tuple(range(a_end)),
            b_layers=tuple(range(b_start, b_end)),
        )
