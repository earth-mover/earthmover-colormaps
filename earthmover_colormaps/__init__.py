"""
Earthmover Colors — perceptually uniform colormaps from the Earthmover brand palette.

Usage::

    import earthmover_colormaps  # registers all colormaps with matplotlib

    # Then use by name anywhere matplotlib accepts a colormap string:
    plt.imshow(data, cmap="em.signal")

    # Or access colormap objects directly:
    earthmover_colormaps.signal
    earthmover_colormaps.cm["em.signal"]

All colormaps are also available in reversed variants (e.g. ``"em.signal_r"``).
"""

from ._data import COLORMAPS as _COLORMAPS

__version__ = "0.1.0"

__all__ = [
    "cm",
    "register",
    # Short attribute names
    "violet",
    "lime",
    "signal",
    "diverging",
    "ocean",
    "bloom",
    "cycle",
]

# ---------------------------------------------------------------------------
# Build matplotlib colormaps from inlined RGB tables
# ---------------------------------------------------------------------------

_cmap_objects: dict = {}
_registered = False


def _build():
    """Build LinearSegmentedColormap objects from RGB tables (lazy, on first access)."""
    global _cmap_objects
    if _cmap_objects:
        return

    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        raise ImportError(
            "matplotlib is required to use earthmover_colormaps colormaps. "
            "Install it with: pip install matplotlib"
        )

    for name, rgb_list in _COLORMAPS.items():
        cmap = LinearSegmentedColormap.from_list(name, rgb_list, N=256)
        _cmap_objects[name] = cmap

        # Reversed variant
        rname = f"{name}_r"
        cmap_r = LinearSegmentedColormap.from_list(rname, list(reversed(rgb_list)), N=256)
        _cmap_objects[rname] = cmap_r


def register():
    """Register all Earthmover colormaps with matplotlib's global registry.

    Called automatically on import. Safe to call multiple times.
    """
    global _registered
    if _registered:
        return

    _build()

    try:
        import matplotlib as mpl
        for name, cmap in _cmap_objects.items():
            mpl.colormaps.register(cmap, name=name, force=True)
        _registered = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Dict-style access: earthmover_colormaps.cm["em.signal"]
# ---------------------------------------------------------------------------

class _CmapAccessor(dict):
    """Dict-like accessor that builds colormaps lazily."""

    def __missing__(self, key):
        _build()
        if key in _cmap_objects:
            self[key] = _cmap_objects[key]
            return self[key]
        raise KeyError(f"Unknown colormap: {key!r}. Available: {list(_COLORMAPS.keys())}")

    def __repr__(self):
        _build()
        return f"EarthmoverColormaps({list(_cmap_objects.keys())})"

    def keys(self):
        _build()
        return _cmap_objects.keys()

    def values(self):
        _build()
        return _cmap_objects.values()

    def items(self):
        _build()
        return _cmap_objects.items()

    def __len__(self):
        _build()
        return len(_cmap_objects)


cm = _CmapAccessor()


# ---------------------------------------------------------------------------
# Attribute access: earthmover_colormaps.signal
# ---------------------------------------------------------------------------

# Map short names (no "em." prefix) to full names
_SHORT_NAMES = {name.removeprefix("em."): name for name in _COLORMAPS}


def __getattr__(name):
    if name in _SHORT_NAMES:
        _build()
        return _cmap_objects[_SHORT_NAMES[name]]
    raise AttributeError(f"module 'earthmover_colormaps' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Auto-register on import
# ---------------------------------------------------------------------------
register()
