"""PipelineV2: end-to-end pipeline with resonant substrate and 3-channel readout."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import ResonantConfig, ReadoutV2Config
from .stack import ResonantStack
from .readout import ReadoutV2
from ..redesign.config import VideoIOConfig
from ..redesign.video import frame_to_stim


class PipelineV2:
    """End-to-end pipeline with feedback resonance and Channel C coherence.

    Improvements over v1 Pipeline:
    - ResonantStack: deep layers feed back to L0, amplifying persistent stimuli
    - Adaptive input: auto-normalizes gain based on running signal statistics
    - Channel C: temporal coherence of deviation (structured vs random)
    - Local flood clamping: per-patch saturation detection
    """

    def __init__(
        self,
        substrate_cfg: ResonantConfig,
        readout_cfg: ReadoutV2Config,
        video_cfg: VideoIOConfig,
        seed: int = 0,
    ):
        self.substrate = ResonantStack(substrate_cfg, seed=seed)
        self.readout = ReadoutV2(
            readout_cfg,
            n_layers=substrate_cfg.n_layers,
            height=substrate_cfg.height,
            width=substrate_cfg.width,
        )
        self.video_cfg = video_cfg
        self.readout_cfg = readout_cfg
        self._buffer: list = []
        self._frame_count = 0

    def reset(self) -> None:
        self.substrate.reset()
        self.readout.reset()
        self._buffer.clear()
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, dict]]:
        """Process one input frame.

        Returns a cube + metadata whenever enough substrate steps have
        accumulated to satisfy the readout integration window.
        """
        hw = (self.substrate.cfg.height, self.substrate.cfg.width)
        stim = frame_to_stim(frame, hw, self.video_cfg)
        self.substrate.inject_l0(stim, gain=self.video_cfg.input_gain)
        self.substrate.step()
        self._frame_count += 1

        self._buffer.append(self.substrate.x.copy())

        if len(self._buffer) >= self.readout_cfg.integrate_steps:
            window = np.stack(self._buffer, axis=0)
            self._buffer.clear()
            cube, meta = self.readout.emit_cube(window)
            return cube, meta

        return None
