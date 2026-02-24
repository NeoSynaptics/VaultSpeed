"""Plotting utilities for visualizing layered Frozen Substrate experiments."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_snapshots(snapshots: list[dict], prefix: str = "snap") -> None:
    """Plot field snapshots from a two-layer experiment.

    Each entry should have keys: t, x0, A_mid0, C0, x1, A_mid1, C1.
    """
    for idx, snap in enumerate(snapshots):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes = axes.ravel()
        titles = [
            f"L0 x (t={snap['t']})",
            f"L0 A_mid (t={snap['t']})",
            f"L0 C (t={snap['t']})",
            f"L1 x (t={snap['t']})",
            f"L1 A_mid (t={snap['t']})",
            f"L1 C (t={snap['t']})",
        ]
        fields = [
            snap["x0"], snap["A_mid0"], snap["C0"],
            snap["x1"], snap["A_mid1"], snap["C1"],
        ]
        for ax, title, field in zip(axes, titles, fields):
            ax.imshow(field, cmap="viridis", interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        fname = f"{prefix}_{idx}.png"
        plt.savefig(fname)
        plt.close(fig)


def plot_traces(results: dict, prefix: str = "trace") -> None:
    """Plot time-series traces from a two-layer experiment."""
    trace_error_L0 = results.get("trace_error_L0", [])
    trace_error_L1 = results.get("trace_error_L1", [])
    trace_alive_L0 = results.get("trace_alive_L0", [])
    trace_alive_L1 = results.get("trace_alive_L1", [])
    t = np.arange(len(trace_error_L0))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax0, ax1 = axes
    ax0.plot(t, trace_error_L0, label="L0", linewidth=1.0)
    ax0.plot(t, trace_error_L1, label="L1", linewidth=1.0)
    ax0.set_ylabel("Mean |error| (alive)")
    ax0.set_title("Prediction error over time")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1.plot(t, trace_alive_L0, label="L0", linewidth=1.0)
    ax1.plot(t, trace_alive_L1, label="L1", linewidth=1.0)
    ax1.set_ylabel("Alive fraction")
    ax1.set_xlabel("Time step")
    ax1.set_title("Alive fraction over time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{prefix}.png"
    plt.savefig(fname)
    plt.close(fig)
