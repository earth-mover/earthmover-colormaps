"""Tests for earthmover-colormaps."""

import numpy as np
import pytest

ALL_NAMES = [
    "em.violet", "em.lime", "em.signal", "em.diverging", "em.ocean", "em.bloom",
    "em.heat", "em.earth", "em.twilight",
]
ALL_NAMES_WITH_REVERSED = ALL_NAMES + [f"{n}_r" for n in ALL_NAMES]
SHORT_NAMES = ["violet", "lime", "signal", "diverging", "ocean", "bloom",
               "heat", "earth", "twilight"]

# Sequential colormaps that must have monotonic lightness
SEQUENTIAL_MONOTONIC = ["em.violet", "em.lime", "em.signal", "em.ocean",
                        "em.heat", "em.earth"]


def test_import_registers_colormaps():
    import matplotlib.pyplot as plt
    import earthmover_colormaps  # noqa: F401

    for name in ALL_NAMES_WITH_REVERSED:
        cmap = plt.get_cmap(name)
        assert cmap is not None
        assert cmap.name == name


def test_attribute_access():
    import earthmover_colormaps

    for attr in SHORT_NAMES:
        cmap = getattr(earthmover_colormaps, attr)
        assert cmap is not None
        assert cmap.N == 256


def test_dict_access():
    import earthmover_colormaps

    assert len(earthmover_colormaps.cm) == 18  # 9 + 9 reversed
    cmap = earthmover_colormaps.cm["em.signal"]
    assert cmap.name == "em.signal"


def test_colormap_values_in_range():
    import earthmover_colormaps

    for name in ALL_NAMES:
        cmap = earthmover_colormaps.cm[name]
        values = cmap(np.linspace(0, 1, 256))
        assert values.min() >= 0.0
        assert values.max() <= 1.0


def test_unknown_attribute_raises():
    import earthmover_colormaps

    with pytest.raises(AttributeError):
        earthmover_colormaps.nonexistent


def test_unknown_key_raises():
    import earthmover_colormaps

    with pytest.raises(KeyError):
        earthmover_colormaps.cm["em.nonexistent"]


def test_perceptual_uniformity():
    """All colormaps should have CV < 2% in CAM02-UCS."""
    from colorspacious import cspace_convert
    from earthmover_colormaps._data import COLORMAPS

    for name, rgb_list in COLORMAPS.items():
        rgb = np.asarray(rgb_list, dtype=np.float64)
        Jpapbp = cspace_convert(rgb, "sRGB1", "CAM02-UCS")

        diffs = np.diff(Jpapbp, axis=0)
        step_dE = np.sqrt(np.sum(diffs ** 2, axis=1))

        mean_dE = np.mean(step_dE)
        std_dE = np.std(step_dE)
        cv = std_dE / mean_dE * 100

        assert cv < 2.0, f"{name}: CV={cv:.1f}% exceeds 2% threshold"


def test_monotonic_lightness():
    """Sequential colormaps must have monotonic J' (with tolerance for roundtrip noise)."""
    from colorspacious import cspace_convert
    from earthmover_colormaps._data import COLORMAPS

    tol = 1e-4  # tolerance for sRGB roundtrip noise

    for name in SEQUENTIAL_MONOTONIC:
        rgb = np.asarray(COLORMAPS[name], dtype=np.float64)
        Jpapbp = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
        Jp = Jpapbp[:, 0]
        diffs = np.diff(Jp)

        # Must be either all increasing or all decreasing (within tolerance)
        assert np.all(diffs > -tol) or np.all(diffs < tol), (
            f"{name}: J' is not monotonic. "
            f"Violations: {np.sum(diffs < -tol)} decreasing, {np.sum(diffs > tol)} increasing"
        )


def test_diverging_lightness_profile():
    """em.diverging must have a V-shaped (or inverted V) J' profile."""
    from colorspacious import cspace_convert
    from earthmover_colormaps._data import COLORMAPS

    rgb = np.asarray(COLORMAPS["em.diverging"], dtype=np.float64)
    Jpapbp = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
    Jp = Jpapbp[:, 0]

    # The midpoint should be a lightness extremum (bright center)
    mid_idx = np.argmax(Jp)
    assert 64 < mid_idx < 192, f"Midpoint at {mid_idx}, expected between 64 and 192"

    # Left arm should be increasing, right arm decreasing (with tolerance)
    tol = 1e-3
    left_diffs = np.diff(Jp[:mid_idx + 1])
    right_diffs = np.diff(Jp[mid_idx:])

    left_violations = np.sum(left_diffs < -tol)
    right_violations = np.sum(right_diffs > tol)
    assert left_violations < 3, f"Left arm: {left_violations} decreasing violations"
    assert right_violations < 3, f"Right arm: {right_violations} increasing violations"


def test_cyclic_endpoints():
    """em.twilight must start and end at the same color."""
    from earthmover_colormaps._data import COLORMAPS

    rgb = np.asarray(COLORMAPS["em.twilight"], dtype=np.float64)
    # Start and end should be very close (same color = midnight)
    diff = np.abs(rgb[0] - rgb[-1])
    assert np.all(diff < 0.01), (
        f"em.twilight endpoints differ by {diff}: start={rgb[0]}, end={rgb[-1]}"
    )


def test_raw_data_has_256_entries():
    """Each colormap must have exactly 256 entries."""
    from earthmover_colormaps._data import COLORMAPS

    for name, rgb_list in COLORMAPS.items():
        assert len(rgb_list) == 256, f"{name} has {len(rgb_list)} entries, expected 256"


def test_rgb_values_valid():
    """All raw RGB values must be in [0, 1]."""
    from earthmover_colormaps._data import COLORMAPS

    for name, rgb_list in COLORMAPS.items():
        rgb = np.asarray(rgb_list)
        assert rgb.shape == (256, 3), f"{name} shape is {rgb.shape}"
        assert np.all(rgb >= 0.0), f"{name} has values < 0"
        assert np.all(rgb <= 1.0), f"{name} has values > 1"
