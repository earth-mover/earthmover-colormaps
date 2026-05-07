"""Interactive Qt GUI for comparing Earthmover colormaps to matplotlib builtins.

Tabs:
  1. Diagnostics — swatch, greyscale, J' lightness, per-step ΔE, a'b' chromaticity.
  2. Colorblind — deuteranopia / protanopia / tritanopia simulation swatches.
  3. Example data — real geospatial data (xarray tutorial) plotted with each map.

Usage:
    cd tools/
    uv run --group design python compare_gui.py
"""

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6 import QtWidgets, QtCore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import earthmover_colormaps  # noqa: F401  registers em.* colormaps
from colorspace_utils import srgb_to_cam02ucs


EM_NAMES = [
    "em.violet", "em.lime", "em.signal", "em.diverging", "em.ocean",
    "em.bloom", "em.cycle",
]

MPL_NAMES = [
    "viridis", "plasma", "magma", "inferno", "cividis",
    "twilight", "twilight_shifted",
    "coolwarm", "RdBu_r", "Spectral_r", "PiYG", "BrBG",
    "Blues", "Greens", "Reds", "Purples", "Oranges",
    "YlGn", "YlOrBr", "PuBuGn",
    "gray", "bone", "hot", "afmhot", "copper",
]

# Viénot, Brettel & Mollon (1999) full-severity dichromacy matrices in linear sRGB.
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


def sample_cmap(name, n=256):
    """(n, 3) sRGB array sampled from a registered matplotlib colormap."""
    cmap = plt.get_cmap(name)
    return np.asarray(cmap(np.linspace(0, 1, n)))[:, :3]


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0, None)
    return np.where(rgb <= 0.0031308, 12.92 * rgb,
                    1.055 * np.power(rgb, 1 / 2.4) - 0.055)


def simulate_cvd(rgb, kind):
    """Simulate dichromacy (kind ∈ deuteranopia/protanopia/tritanopia)."""
    M = CVD_MATRICES[kind]
    lin = srgb_to_linear(np.clip(rgb, 0, 1))
    sim = lin @ M.T
    return np.clip(linear_to_srgb(sim), 0, 1)


def srgb_gamut_slice(Jp, ap_lim=(-50, 50), bp_lim=(-50, 50), res=80):
    """sRGB gamut slice at fixed J' as an image; out-of-gamut pixels are light grey."""
    from colorspacious import cspace_convert
    aa = np.linspace(ap_lim[0], ap_lim[1], res)
    bb = np.linspace(bp_lim[0], bp_lim[1], res)
    A, B = np.meshgrid(aa, bb)
    grid = np.stack([np.full_like(A, Jp), A, B], axis=-1)
    rgb = cspace_convert(grid, "CAM02-UCS", "sRGB1")
    in_gamut = np.all((rgb >= 0) & (rgb <= 1), axis=-1)
    return np.where(in_gamut[..., None], np.clip(rgb, 0, 1), 0.85), ap_lim, bp_lim


# Cache the sRGB gamut wireframe quads — they don't depend on the colormap.
_GAMUT_CACHE: dict = {}


def srgb_gamut_quads_in_jpapbp(resolution=12):
    """Return Poly3DCollection-ready quads for the sRGB cube faces, in (a',b',J')
    coordinates (matching the 3D plot's xyz axes). Cached.
    """
    from colorspacious import cspace_convert
    key = resolution
    if key in _GAMUT_CACHE:
        return _GAMUT_CACHE[key]

    step = 1.0 / resolution
    quads = []
    for fixed in (0.0, 1.0):
        for i in range(resolution):
            for j in range(resolution):
                a, b = i * step, j * step
                A, B = (i + 1) * step, (j + 1) * step
                quads.append([[fixed, a, b], [fixed, A, b], [fixed, A, B], [fixed, a, B]])
                quads.append([[a, fixed, b], [A, fixed, b], [A, fixed, B], [a, fixed, B]])
                quads.append([[a, b, fixed], [A, b, fixed], [A, B, fixed], [a, B, fixed]])
    quads = np.asarray(quads)
    flat = quads.reshape((-1, 3))
    jpapbp_flat = cspace_convert(flat, "sRGB1", "CAM02-UCS")
    jpapbp_quads = jpapbp_flat.reshape((-1, 4, 3))
    # 3D axes use (a', b', J') as (x, y, z)
    xyz_quads = jpapbp_quads[:, :, [1, 2, 0]]
    _GAMUT_CACHE[key] = xyz_quads
    return xyz_quads


def _swatch(ax, rgb, title=None):
    ax.imshow(rgb[np.newaxis, :, :], aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# xarray example data — loaded once on first use of the Example-data tab.
# ---------------------------------------------------------------------------

class _ExampleData:
    """Lazy-load the xarray tutorial dataset and derive scenario fields."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            import xarray as xr
            ds = xr.tutorial.open_dataset("air_temperature")
            air = ds.air.isel(time=0)             # 2D field at first timestep
            mean = ds.air.mean("time")            # time-mean field
            anomaly = air - mean                  # anomaly (centered ~0)
            self.lon = ds.lon.values
            self.lat = ds.lat.values
            self.scenarios = {
                "Mean temperature (K)": mean.values,
                "Temperature anomaly (K)": anomaly.values,
                "Snapshot (K)": air.values,
            }
        except Exception as exc:                  # noqa: BLE001
            self.scenarios = {"(error loading xarray tutorial)": None}
            self._error = str(exc)
        self._loaded = True


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ComparisonWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Earthmover ↔ matplotlib colormap comparison")
        self.resize(1500, 950)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Top bar
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Earthmover:"))
        self.em_combo = QtWidgets.QComboBox()
        self.em_combo.addItems(EM_NAMES)
        self.em_combo.setCurrentText("em.signal")
        bar.addWidget(self.em_combo)

        bar.addSpacing(20)
        bar.addWidget(QtWidgets.QLabel("Matplotlib:"))
        self.mpl_combo = QtWidgets.QComboBox()
        self.mpl_combo.addItems(MPL_NAMES)
        self.mpl_combo.setCurrentText("viridis")
        bar.addWidget(self.mpl_combo)

        self.reverse_em = QtWidgets.QCheckBox("Reverse EM")
        self.reverse_mpl = QtWidgets.QCheckBox("Reverse MPL")
        bar.addSpacing(20)
        bar.addWidget(self.reverse_em)
        bar.addWidget(self.reverse_mpl)

        bar.addStretch(1)
        layout.addLayout(bar)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        # One Figure per tab
        self.fig_diag = Figure(figsize=(14, 9), tight_layout=True)
        self.canvas_diag = FigureCanvas(self.fig_diag)
        self.tabs.addTab(self.canvas_diag, "Diagnostics")

        self.fig_cvd = Figure(figsize=(14, 9), tight_layout=True)
        self.canvas_cvd = FigureCanvas(self.fig_cvd)
        self.tabs.addTab(self.canvas_cvd, "Colorblind")

        # Example-data tab gets its own scenario selector
        ex_widget = QtWidgets.QWidget()
        ex_layout = QtWidgets.QVBoxLayout(ex_widget)
        ex_bar = QtWidgets.QHBoxLayout()
        ex_bar.addWidget(QtWidgets.QLabel("Scenario:"))
        self.scenario_combo = QtWidgets.QComboBox()
        ex_bar.addWidget(self.scenario_combo)
        ex_bar.addStretch(1)
        ex_layout.addLayout(ex_bar)
        self.fig_data = Figure(figsize=(14, 7), tight_layout=True)
        self.canvas_data = FigureCanvas(self.fig_data)
        ex_layout.addWidget(self.canvas_data)
        self.tabs.addTab(ex_widget, "Example data")

        self.scenario_combo.currentTextChanged.connect(self.redraw_data)

        # Wire up
        for w in (self.em_combo, self.mpl_combo):
            w.currentTextChanged.connect(self.redraw_all)
        for w in (self.reverse_em, self.reverse_mpl):
            w.stateChanged.connect(self.redraw_all)
        self.tabs.currentChanged.connect(self._tab_changed)

        self.redraw_all()

    # -- helpers --------------------------------------------------------

    def _get_rgbs(self):
        em_name = self.em_combo.currentText()
        mpl_name = self.mpl_combo.currentText()
        em_rgb = sample_cmap(em_name)
        mpl_rgb = sample_cmap(mpl_name)
        if self.reverse_em.isChecked():
            em_rgb = em_rgb[::-1]
        if self.reverse_mpl.isChecked():
            mpl_rgb = mpl_rgb[::-1]
        return em_name, em_rgb, mpl_name, mpl_rgb

    def redraw_all(self):
        self.redraw_diagnostics()
        self.redraw_cvd()
        self.redraw_data()

    def _tab_changed(self, idx):
        if idx == 2:
            self._populate_scenarios()

    def _populate_scenarios(self):
        if self.scenario_combo.count() > 0:
            return
        ex = _ExampleData()
        ex._ensure_loaded()
        self.scenario_combo.blockSignals(True)
        self.scenario_combo.addItems(list(ex.scenarios.keys()))
        # Default: pick a scenario that suits the currently selected em map
        em_name = self.em_combo.currentText()
        if "diverging" in em_name and "Temperature anomaly (K)" in ex.scenarios:
            self.scenario_combo.setCurrentText("Temperature anomaly (K)")
        self.scenario_combo.blockSignals(False)
        self.redraw_data()

    # -- panels ---------------------------------------------------------

    def redraw_diagnostics(self):
        em_name, em_rgb, mpl_name, mpl_rgb = self._get_rgbs()
        fig = self.fig_diag
        fig.clear()

        gs = fig.add_gridspec(
            6, 2, height_ratios=[0.6, 0.6, 2, 2, 3, 4],
            hspace=0.5, wspace=0.18,
        )

        for col, (name, rgb) in enumerate([(em_name, em_rgb), (mpl_name, mpl_rgb)]):
            _swatch(fig.add_subplot(gs[0, col]), rgb, title=name)
            Jp = srgb_to_cam02ucs(rgb)[:, 0]
            grey = (Jp - Jp.min()) / max(Jp.max() - Jp.min(), 1e-9)
            grey_strip = np.repeat(grey[np.newaxis, :, np.newaxis], 3, axis=2)
            ax2 = fig.add_subplot(gs[1, col])
            ax2.imshow(grey_strip, aspect="auto")
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_ylabel("greyscale", fontsize=9)

        ax_j = fig.add_subplot(gs[2, :])
        for label, rgb, color in [(em_name, em_rgb, "tab:purple"),
                                  (mpl_name, mpl_rgb, "tab:olive")]:
            ax_j.plot(srgb_to_cam02ucs(rgb)[:, 0], label=label, color=color, lw=2)
        ax_j.set_xlabel("colormap index"); ax_j.set_ylabel("J' (lightness)")
        ax_j.set_title("J' lightness profile (straight line = perceptually linear ramp)")
        ax_j.legend(loc="best"); ax_j.grid(True, alpha=0.3)

        ax_de = fig.add_subplot(gs[3, :])
        for label, rgb, color in [(em_name, em_rgb, "tab:purple"),
                                  (mpl_name, mpl_rgb, "tab:olive")]:
            d = np.diff(srgb_to_cam02ucs(rgb), axis=0)
            step = np.sqrt(np.sum(d ** 2, axis=1))
            cv = np.std(step) / np.mean(step) * 100
            ax_de.plot(step, label=f"{label} (CV={cv:.1f}%)", color=color, lw=1.5)
        ax_de.set_xlabel("step index"); ax_de.set_ylabel("ΔE per step")
        ax_de.set_title("Perceptual step size (flat line = uniform)")
        ax_de.legend(loc="best"); ax_de.grid(True, alpha=0.3)

        Jp_em = srgb_to_cam02ucs(em_rgb)[:, 0]
        Jp_mid = float(np.mean([Jp_em.mean(), srgb_to_cam02ucs(mpl_rgb)[:, 0].mean()]))
        gamut_img, ap_lim, bp_lim = srgb_gamut_slice(Jp_mid, res=80)
        for col, (label, rgb) in enumerate([(em_name, em_rgb), (mpl_name, mpl_rgb)]):
            ax = fig.add_subplot(gs[4, col])
            ax.imshow(gamut_img,
                      extent=[ap_lim[0], ap_lim[1], bp_lim[0], bp_lim[1]],
                      origin="lower")
            Jpapbp = srgb_to_cam02ucs(rgb)
            ax.plot(Jpapbp[:, 1], Jpapbp[:, 2], "-", color="black", lw=1.0, alpha=0.5)
            ax.scatter(Jpapbp[:, 1], Jpapbp[:, 2], c=rgb, s=12, edgecolors="none")
            ax.set_xlabel("a'"); ax.set_ylabel("b'")
            ax.set_title(f"{label} chromaticity  (gamut at J'≈{Jp_mid:.0f})")
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

        # Row 5: 3D path through CAM02-UCS J'a'b' with sRGB gamut wireframe
        # (the same view viscm shows). Colormap is the trajectory inside the
        # cube; the cube is the achievable sRGB volume.
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
        gamut_xyz = srgb_gamut_quads_in_jpapbp(resolution=10)
        for col, (label, rgb) in enumerate([(em_name, em_rgb), (mpl_name, mpl_rgb)]):
            ax = fig.add_subplot(gs[5, col], projection="3d")
            patch = Poly3DCollection(gamut_xyz)
            patch.set_facecolor([0.5, 0.5, 0.5, 0.06])
            patch.set_edgecolor([0.2, 0.2, 0.2, 0.10])
            ax.add_collection3d(patch)

            Jpapbp = srgb_to_cam02ucs(rgb)
            ax.plot(Jpapbp[:, 1], Jpapbp[:, 2], Jpapbp[:, 0],
                    color="black", lw=1.0, alpha=0.6)
            # Subsample dots (matches viscm; 256 is too dense for clarity)
            idx = np.linspace(0, len(rgb) - 1, 32).astype(int)
            ax.scatter(Jpapbp[idx, 1], Jpapbp[idx, 2], Jpapbp[idx, 0],
                       c=rgb[idx], s=70, edgecolors="none", depthshade=False)
            ax.set_xlabel("a' (green→red)"); ax.set_ylabel("b' (blue→yellow)")
            ax.set_zlabel("J' (dark→light)")
            ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_zlim(0, 100)
            ax.view_init(elev=25, azim=-65)
            ax.set_title(f"{label} path through CAM02-UCS")

        self.canvas_diag.draw_idle()

    def redraw_cvd(self):
        em_name, em_rgb, mpl_name, mpl_rgb = self._get_rgbs()
        fig = self.fig_cvd
        fig.clear()

        # 4 rows (original + 3 CVD), 2 cols (em / mpl)
        rows = [
            ("Original", em_rgb, mpl_rgb),
            ("Deuteranopia", simulate_cvd(em_rgb, "deuteranopia"),
                            simulate_cvd(mpl_rgb, "deuteranopia")),
            ("Protanopia", simulate_cvd(em_rgb, "protanopia"),
                           simulate_cvd(mpl_rgb, "protanopia")),
            ("Tritanopia", simulate_cvd(em_rgb, "tritanopia"),
                           simulate_cvd(mpl_rgb, "tritanopia")),
        ]
        gs = fig.add_gridspec(len(rows) + 1, 2,
                              height_ratios=[0.7] * len(rows) + [3],
                              hspace=0.55, wspace=0.12)

        for r, (label, em_v, mpl_v) in enumerate(rows):
            ax_em = fig.add_subplot(gs[r, 0])
            _swatch(ax_em, em_v, title=f"{em_name} — {label}")
            ax_mpl = fig.add_subplot(gs[r, 1])
            _swatch(ax_mpl, mpl_v, title=f"{mpl_name} — {label}")

        # J' lightness curves under each simulation overlaid
        ax = fig.add_subplot(gs[-1, :])
        styles = {"Original": "-", "Deuteranopia": "--",
                  "Protanopia": ":", "Tritanopia": "-."}
        for label, em_v, mpl_v in rows:
            ax.plot(srgb_to_cam02ucs(em_v)[:, 0],
                    color="tab:purple", linestyle=styles[label],
                    label=f"{em_name} {label}", lw=1.6)
            ax.plot(srgb_to_cam02ucs(mpl_v)[:, 0],
                    color="tab:olive", linestyle=styles[label],
                    label=f"{mpl_name} {label}", lw=1.6)
        ax.set_xlabel("colormap index"); ax.set_ylabel("J' (lightness)")
        ax.set_title("Lightness profile under simulated CVD "
                     "(monotonic = readable in greyscale equivalent)")
        ax.legend(loc="best", ncols=2, fontsize=8); ax.grid(True, alpha=0.3)

        self.canvas_cvd.draw_idle()

    def redraw_data(self):
        em_name, em_rgb, mpl_name, mpl_rgb = self._get_rgbs()
        fig = self.fig_data
        fig.clear()

        # Always populate scenarios on first draw
        if self.scenario_combo.count() == 0:
            self._populate_scenarios()

        scenario = self.scenario_combo.currentText() if self.scenario_combo.count() else ""
        ex = _ExampleData()
        ex._ensure_loaded()
        field = ex.scenarios.get(scenario)

        if field is None:
            ax = fig.add_subplot(111)
            err = getattr(ex, "_error", "no scenario data available")
            ax.text(0.5, 0.5, f"Could not load xarray tutorial data:\n{err}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas_data.draw_idle()
            return

        # Same color limits for both panels (so visual comparison is fair)
        if "anomaly" in scenario.lower():
            v = float(np.nanmax(np.abs(field)))
            vmin, vmax = -v, v
        else:
            vmin, vmax = float(np.nanmin(field)), float(np.nanmax(field))

        em_reversed = self.reverse_em.isChecked()
        mpl_reversed = self.reverse_mpl.isChecked()
        em_cmap_name = em_name + ("_r" if em_reversed else "")
        mpl_cmap_name = mpl_name + ("_r" if mpl_reversed else "")

        ax1 = fig.add_subplot(1, 2, 1)
        m = ax1.pcolormesh(ex.lon, ex.lat, field,
                           cmap=em_cmap_name, vmin=vmin, vmax=vmax,
                           shading="auto")
        ax1.set_title(f"{em_cmap_name} — {scenario}")
        ax1.set_xlabel("lon"); ax1.set_ylabel("lat")
        fig.colorbar(m, ax=ax1, shrink=0.8)

        ax2 = fig.add_subplot(1, 2, 2)
        m = ax2.pcolormesh(ex.lon, ex.lat, field,
                           cmap=mpl_cmap_name, vmin=vmin, vmax=vmax,
                           shading="auto")
        ax2.set_title(f"{mpl_cmap_name} — {scenario}")
        ax2.set_xlabel("lon"); ax2.set_ylabel("lat")
        fig.colorbar(m, ax=ax2, shrink=0.8)

        self.canvas_data.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)
    w = ComparisonWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
