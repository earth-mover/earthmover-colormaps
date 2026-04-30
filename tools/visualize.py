"""Diagnostic visualization for Earthmover colormaps.

Generates:
1. Swatch strips for all colormaps
2. J' lightness profiles
3. Step-wise ΔE uniformity charts
4. a'b' chromaticity paths
5. Colorblind simulation panels

Usage:
    cd tools/
    uv run python visualize.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from colorspace_utils import srgb_to_cam02ucs, compute_uniformity_metrics

# Import to register colormaps
import earthmover_colormaps  # noqa: F401
from earthmover_colormaps._data import COLORMAPS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")

# Viénot et al. (1999) CVD simulation matrices (applied in linear sRGB)
# These are standard full-dichromacy (severity=100%) simulation matrices.
CVD_MATRICES = {
    "deuteranopia": np.array([
        [0.625, 0.375, 0.0],
        [0.700, 0.300, 0.0],
        [0.000, 0.300, 0.700],
    ]),
    "protanopia": np.array([
        [0.567, 0.433, 0.0],
        [0.558, 0.442, 0.0],
        [0.000, 0.242, 0.758],
    ]),
    "tritanopia": np.array([
        [0.950, 0.050, 0.0],
        [0.000, 0.433, 0.567],
        [0.000, 0.475, 0.525],
    ]),
}


def _srgb_to_linear(c):
    """sRGB gamma to linear."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    """Linear to sRGB gamma."""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def simulate_cvd(rgb, cvd_type):
    """Simulate color vision deficiency using Viénot matrices."""
    linear = _srgb_to_linear(np.clip(rgb, 0, 1))
    mat = CVD_MATRICES[cvd_type]
    simulated = linear @ mat.T
    return np.clip(_linear_to_srgb(np.clip(simulated, 0, 1)), 0, 1)


def plot_swatches(output_path=None):
    """Plot color swatches for all colormaps."""
    names = list(COLORMAPS.keys())
    n = len(names)

    fig, axes = plt.subplots(n, 1, figsize=(10, 0.5 * n + 0.5))
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.18, right=0.99, hspace=0.4)

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    for ax, name in zip(axes, names):
        ax.imshow(gradient, aspect="auto", cmap=mpl.colormaps[name])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)

    fig.suptitle("Earthmover Colormaps", fontsize=14, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {output_path}")
    return fig


def plot_lightness_profiles(output_path=None):
    """Plot J' lightness profiles for all colormaps."""
    names = list(COLORMAPS.keys())

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for idx, name in enumerate(names):
        ax = axes[idx]
        rgb = np.asarray(COLORMAPS[name], dtype=np.float64)
        Jpapbp = srgb_to_cam02ucs(rgb)
        Jp = Jpapbp[:, 0]
        x = np.linspace(0, 1, len(Jp))

        # Color the line by the colormap itself
        for i in range(len(x) - 1):
            ax.plot(x[i:i + 2], Jp[i:i + 2], color=rgb[i], linewidth=2)

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Position")
        ax.set_ylabel("J' (lightness)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        m = compute_uniformity_metrics(COLORMAPS[name])
        ax.text(0.02, 0.02, f"CV={m['cv_percent']:.1f}%",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle("Lightness Profiles (J' in CAM02-UCS)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {output_path}")
    return fig


def plot_delta_e(output_path=None):
    """Plot step-wise ΔE for all colormaps."""
    names = list(COLORMAPS.keys())

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for idx, name in enumerate(names):
        ax = axes[idx]
        rgb = np.asarray(COLORMAPS[name], dtype=np.float64)
        Jpapbp = srgb_to_cam02ucs(rgb)

        diffs = np.diff(Jpapbp, axis=0)
        step_dE = np.sqrt(np.sum(diffs ** 2, axis=1))
        mean_dE = np.mean(step_dE)

        ax.bar(range(len(step_dE)), step_dE, width=1, color="steelblue", alpha=0.7)
        ax.axhline(mean_dE, color="red", linestyle="--", linewidth=1, label=f"mean={mean_dE:.3f}")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("ΔE (CAM02-UCS)")
        ax.legend(fontsize=8)

        m = compute_uniformity_metrics(COLORMAPS[name])
        ax.text(0.98, 0.98, f"CV={m['cv_percent']:.1f}%",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle("Perceptual Uniformity (Step-wise ΔE in CAM02-UCS)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {output_path}")
    return fig


def plot_ab_paths(output_path=None):
    """Plot a'b' chromaticity paths in CAM02-UCS."""
    names = list(COLORMAPS.keys())

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for idx, name in enumerate(names):
        ax = axes[idx]
        rgb = np.asarray(COLORMAPS[name], dtype=np.float64)
        Jpapbp = srgb_to_cam02ucs(rgb)

        ap = Jpapbp[:, 1]
        bp = Jpapbp[:, 2]

        # Color by position in the colormap
        for i in range(len(ap) - 1):
            ax.plot(ap[i:i + 2], bp[i:i + 2], color=rgb[i], linewidth=2)

        # Mark start and end
        ax.plot(ap[0], bp[0], "ko", markersize=8, label="start")
        ax.plot(ap[-1], bp[-1], "k^", markersize=8, label="end")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("a' (green-red)")
        ax.set_ylabel("b' (blue-yellow)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Chromaticity Paths in CAM02-UCS (a'b' plane)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {output_path}")
    return fig


def plot_colorblind_sim(output_path=None):
    """Plot colorblind simulation panels."""
    names = list(COLORMAPS.keys())
    cvd_types = ["deuteranopia", "protanopia", "tritanopia"]

    fig, axes = plt.subplots(len(names), 4, figsize=(14, 0.6 * len(names) + 1))
    fig.subplots_adjust(top=0.94, bottom=0.02, left=0.12, right=0.99, hspace=0.4, wspace=0.05)

    for row, name in enumerate(names):
        rgb = np.asarray(COLORMAPS[name], dtype=np.float64)

        # Original
        ax = axes[row, 0]
        ax.imshow(rgb[np.newaxis, :, :], aspect="auto")
        ax.set_yticks([])
        ax.set_xticks([])
        if row == 0:
            ax.set_title("Original", fontsize=9)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)

        # CVD simulations using Viénot matrices
        for col, cvd_type in enumerate(cvd_types):
            ax = axes[row, col + 1]
            rgb_cvd = simulate_cvd(rgb, cvd_type)
            ax.imshow(rgb_cvd[np.newaxis, :, :], aspect="auto")
            ax.set_yticks([])
            ax.set_xticks([])
            if row == 0:
                ax.set_title(cvd_type.replace("anopia", "."), fontsize=9)

    fig.suptitle("Colorblind Simulation (100% dichromacy, Viénot 1999)",
                 fontsize=14, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {output_path}")
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating diagnostic visualizations...")

    plot_swatches(os.path.join(OUTPUT_DIR, "swatches.png"))
    plot_lightness_profiles(os.path.join(OUTPUT_DIR, "lightness_profiles.png"))
    plot_delta_e(os.path.join(OUTPUT_DIR, "delta_e.png"))
    plot_ab_paths(os.path.join(OUTPUT_DIR, "ab_paths.png"))
    plot_colorblind_sim(os.path.join(OUTPUT_DIR, "colorblind_sim.png"))

    print("\nAll visualizations saved to images/")


if __name__ == "__main__":
    main()
