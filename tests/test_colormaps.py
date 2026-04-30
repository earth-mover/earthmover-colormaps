"""Basic tests for earthmover-colors."""

import pytest


def test_import_registers_colormaps():
    import matplotlib.pyplot as plt
    import earthmover_colormaps  # noqa: F401

    for name in [
        "em.violet", "em.lime", "em.signal", "em.diverging", "em.ocean", "em.bloom",
        "em.violet_r", "em.lime_r", "em.signal_r", "em.diverging_r", "em.ocean_r", "em.bloom_r",
    ]:
        cmap = plt.get_cmap(name)
        assert cmap is not None
        assert cmap.name == name


def test_attribute_access():
    import earthmover_colormaps

    for attr in ["violet", "lime", "signal", "diverging", "ocean", "bloom"]:
        cmap = getattr(earthmover_colormaps, attr)
        assert cmap is not None
        assert cmap.N == 256


def test_dict_access():
    import earthmover_colormaps

    assert len(earthmover_colormaps.cm) == 12  # 6 + 6 reversed
    cmap = earthmover_colormaps.cm["em.signal"]
    assert cmap.name == "em.signal"


def test_colormap_values_in_range():
    import numpy as np
    import earthmover_colormaps

    for name in ["em.violet", "em.lime", "em.signal", "em.diverging", "em.ocean", "em.bloom"]:
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
