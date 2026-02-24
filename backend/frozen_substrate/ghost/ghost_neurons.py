"""Ghost Neurons: novelty detection via EMA baseline deviation.

Ghost neurons read selected depth layers from the substrate and compute
the deviation from a slowly-adapting background (EMA). This highlights
recently-changed or novel activity patterns.
"""

import numpy as np


class GhostNeurons:
    """Track deviation from EMA baseline across specified depth layers.

    Parameters
    ----------
    ghost_layers : tuple of int
        Which substrate layer indices to read.
    alpha_bg : float
        EMA learning rate for the background model. Smaller = slower adaptation.
    """

    def __init__(self, ghost_layers=(2, 3, 4, 5), alpha_bg=0.01):
        self.ghost_layers = ghost_layers
        self.alpha = alpha_bg
        self.bg = None

    def read(self, substrate_layers):
        """Read substrate layers and return (activations, novelty).

        Parameters
        ----------
        substrate_layers : list of np.ndarray
            All substrate layer activations (list of H x W arrays).

        Returns
        -------
        G_t : np.ndarray
            Current activations at ghost layers, shape (len(ghost_layers), H, W).
        dG_t : np.ndarray
            Absolute deviation from background, shape (len(ghost_layers), H, W).
        """
        G_t = np.stack([
            np.clip(substrate_layers[z], 0.0, 1.0)
            for z in self.ghost_layers
        ], axis=0)

        if self.bg is None:
            self.bg = G_t.copy()

        self.bg = (1 - self.alpha) * self.bg + self.alpha * G_t
        dG_t = np.abs(G_t - self.bg)

        return G_t, dG_t
