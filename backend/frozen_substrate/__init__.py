"""Frozen Substrate: A bio-inspired computational substrate with
local prediction, plasticity, survival dynamics, and multi-layer
depth propagation."""

from frozen_substrate.core import FrozenCoreV3
from frozen_substrate.multilayer import MultiLayerFrozenSubstrate, PassiveLayerParams
from frozen_substrate.retina import RetinaDepthStack, IntegratedResidual
from frozen_substrate.ghost import GhostNeurons
from frozen_substrate.viz import save_cubes_npz, load_cubes_npz
