"""Visualization utilities for Frozen Substrate output cubes."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np


def save_cubes_npz(path: str, cubes: np.ndarray, metadata: Optional[list] = None) -> None:
    """Save a cube stream as a compressed .npz file.

    Parameters
    ----------
    path : str
        Output file path (should end in .npz).
    cubes : np.ndarray
        Shape (N, C, H, W) tensor stream.
    metadata : list of dict, optional
        Per-cube metadata from Pipeline.process_frame().
    """
    save_dict = {"cubes": cubes}
    if metadata is not None and len(metadata) > 0:
        m = metadata[-1]
        save_dict["a_layers"] = np.array(m.get("a_layers", []))
        save_dict["b_layers"] = np.array(m.get("b_layers", []))
    np.savez_compressed(path, **save_dict)


def load_cubes_npz(path: str):
    """Load cubes saved with save_cubes_npz.

    Returns (cubes, a_layers, b_layers).
    """
    data = np.load(path)
    cubes = data["cubes"]
    a_layers = tuple(data["a_layers"]) if "a_layers" in data else ()
    b_layers = tuple(data["b_layers"]) if "b_layers" in data else ()
    return cubes, a_layers, b_layers


def render_channel_pngs(
    cubes: np.ndarray,
    out_dir: str,
    a_channels: int = 3,
    cmap: str = "viridis",
    dpi: int = 120,
) -> None:
    """Render Channel A and Channel B as PNG image sequences.

    Parameters
    ----------
    cubes : np.ndarray
        Shape (N, C, H, W).
    out_dir : str
        Directory to write PNG files into. Creates channel_a/ and channel_b/ subdirs.
    a_channels : int
        Number of leading channels that belong to Channel A.
    cmap : str
        Matplotlib colormap name.
    dpi : int
        Output DPI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dir_a = os.path.join(out_dir, "channel_a")
    dir_b = os.path.join(out_dir, "channel_b")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    N = cubes.shape[0]
    for i in range(N):
        # Channel A: mean across A channels
        a_img = cubes[i, :a_channels].mean(axis=0)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(a_img, cmap=cmap, interpolation="nearest")
        ax.set_title(f"Channel A  t={i}")
        ax.axis("off")
        fig.savefig(os.path.join(dir_a, f"a_{i:04d}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Channel B: mean across B channels
        b_img = cubes[i, a_channels:].mean(axis=0)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(b_img, cmap="inferno", interpolation="nearest")
        ax.set_title(f"Channel B  t={i}")
        ax.axis("off")
        fig.savefig(os.path.join(dir_b, f"b_{i:04d}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {N} Channel A frames to {dir_a}")
    print(f"Saved {N} Channel B frames to {dir_b}")


def render_composite(
    cubes: np.ndarray,
    out_dir: str,
    a_channels: int = 3,
    dpi: int = 140,
) -> None:
    """Render side-by-side Channel A | Channel B composite PNGs.

    Parameters
    ----------
    cubes : np.ndarray
        Shape (N, C, H, W).
    out_dir : str
        Directory to write composite PNGs.
    a_channels : int
        Number of leading channels belonging to Channel A.
    dpi : int
        Output DPI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp_dir = os.path.join(out_dir, "composite")
    os.makedirs(comp_dir, exist_ok=True)

    N = cubes.shape[0]
    for i in range(N):
        a_img = cubes[i, :a_channels].mean(axis=0)
        b_img = cubes[i, a_channels:].mean(axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(a_img, cmap="viridis", interpolation="nearest")
        ax1.set_title("Channel A (existence)")
        ax1.axis("off")
        ax2.imshow(b_img, cmap="inferno", interpolation="nearest")
        ax2.set_title("Channel B (persistence)")
        ax2.axis("off")
        fig.suptitle(f"t = {i}", fontsize=10)
        fig.savefig(os.path.join(comp_dir, f"comp_{i:04d}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {N} composite frames to {comp_dir}")


def print_cube_summary(cubes: np.ndarray, metadata: Optional[list] = None) -> None:
    """Print a human-readable summary of a cube stream."""
    N, C, H, W = cubes.shape
    print(f"Cube stream: {N} frames, {C} channels, {H}x{W} spatial")

    if metadata and len(metadata) > 0:
        m = metadata[-1]
        n_a = len(m.get("a_layers", []))
        n_b = len(m.get("b_layers", []))
        print(f"  Channel A: {n_a} layers {m.get('a_layers', '?')}")
        print(f"  Channel B: {n_b} layers {m.get('b_layers', '?')}")
        print(f"  B policy: {m.get('b_policy', '?')}")
        flood = sum(mm.get("flood_events_in_window", 0) for mm in metadata)
        print(f"  Total flood events: {flood}")

    a_mean = float(cubes[:, :3].mean()) if C >= 3 else float(cubes.mean())
    b_mean = float(cubes[:, 3:].mean()) if C > 3 else 0.0
    print(f"  Channel A mean: {a_mean:.5f}")
    print(f"  Channel B mean: {b_mean:.5f}")
    print(f"  Global range: [{float(cubes.min()):.4f}, {float(cubes.max()):.4f}]")
