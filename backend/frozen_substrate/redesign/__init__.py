"""Frozen Substrate Redesign: Production-minded, mechanically-defined pipeline.

Core commitment (the prior):
  Information = persistence of deviation from expectation under degradation.

- No learning.
- No semantics.
- Deterministic local dynamics.
- Passive readout (non-interference).
"""

from .config import SubstrateConfig, ReadoutConfig, VideoIOConfig
from .stack import SubstrateStack
from .readout import Readout
from .pipeline import Pipeline

__all__ = [
    "SubstrateConfig",
    "ReadoutConfig",
    "VideoIOConfig",
    "SubstrateStack",
    "Readout",
    "Pipeline",
]
