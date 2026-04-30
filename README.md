# earthmover-colormaps

Perceptually uniform colormaps built from the [Earthmover](https://earthmover.io) brand palette. Designed for scientific visualization — monotonic lightness, colorblind-safe, and zero required dependencies.

![earthmover colormaps](images/swatches.png)

## Install

```bash
uv add earthmover-colormaps
```

Or with pip:

```bash
pip install earthmover-colormaps
```

Or from source:

```bash
uv add git+https://github.com/earth-mover/earthmover-colormaps.git
```

## Quick start

```python
import earthmover_colormaps  # registers colormaps with matplotlib on import
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(100, 100)
plt.imshow(data, cmap="em.signal")
plt.colorbar()
plt.show()
```

That's it. `import earthmover_colormaps` registers all colormaps with matplotlib's global registry. Use them anywhere a colormap string is accepted — `plt.imshow`, `plt.pcolormesh`, `xarray.plot()`, `cartopy`, etc.

## Colormaps

| Name | Type | Description |
|------|------|-------------|
| `em.violet` | Sequential | Light grey → violet → midnight. Single-hue purple family. |
| `em.lime` | Sequential | Midnight → green → lime. Ideal for concentrations. |
| `em.signal` | Sequential | Midnight → violet → lime → light grey. The signature Earthmover axis. |
| `em.diverging` | Diverging | Violet ↔ light grey ↔ lime. Centered neutral for anomalies. |
| `em.ocean` | Sequential | Midnight → blue → green → lime. Cool multi-hue, oceanographic. |
| `em.bloom` | Sequential | Midnight → magenta → pink → lavender → light grey. Warm pink family. |
| `em.heat` | Sequential | Midnight → deep red → orange → amber → warm white. Fire/thermal. |
| `em.earth` | Sequential | Dark blue → teal → green → sand → cream. Terrain/bathymetry. |
| `em.twilight` | Cyclic | Midnight → violet → light → lime → midnight. For phase/angle data. |

Every colormap has a reversed variant (append `_r`): `"em.signal_r"`, `"em.violet_r"`, etc.

![example data](images/example_usage.png)

## Access patterns

```python
import earthmover_colormaps

# 1. String name (after import registers them with matplotlib)
plt.imshow(data, cmap="em.signal")

# 2. Attribute access (short name, no "em." prefix)
earthmover_colormaps.signal
earthmover_colormaps.diverging
earthmover_colormaps.heat

# 3. Dict access (full name)
earthmover_colormaps.cm["em.signal"]
earthmover_colormaps.cm["em.signal_r"]
```

## Integration with xarray

```python
import earthmover_colormaps
import xarray as xr

ds = xr.open_dataset("temperature.nc")
ds.temperature.plot(cmap="em.signal")
```

## Integration with cartopy

```python
import earthmover_colormaps
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
ax.coastlines()
cs = ax.pcolormesh(lon, lat, sst, cmap="em.ocean", transform=ccrs.PlateCarree())
fig.colorbar(cs, ax=ax)
```

## Setting as default

To use an Earthmover colormap as your default across all plots:

```python
import earthmover_colormaps
import matplotlib as mpl

mpl.rcParams["image.cmap"] = "em.signal"
```

Or in a matplotlibrc file:

```
image.cmap: em.signal
```

## Design principles

All colormaps are designed with:

- **Perceptual uniformity** — designed and optimized in CAM02-UCS color space with arc-length parameterization, so equal data steps produce equal visual steps. All colormaps achieve a ΔE coefficient of variation < 2% in CAM02-UCS, comparable to or better than matplotlib's `viridis`.
- **Monotonic lightness** — sequential maps go strictly light-to-dark or dark-to-light; diverging maps have a V-shaped lightness profile; cyclic maps have a symmetric arch. This means they print well in grayscale.
- **Colorblind safety** — tested under simulated deuteranopia, protanopia, and tritanopia at full severity. The violet–lime axis maps to blue–yellow under red-green CVD, preserving discriminability. See [docs/colorblind-accessibility.md](docs/colorblind-accessibility.md).
- **Brand coherence** — derived from the Earthmover brand palette (Midnight `#201F2C`, Violet `#A653FF`, Lime `#B7E400`).

![lightness profiles](images/lightness_profiles.png)

## Dependencies

**Zero required dependencies.** The RGB lookup tables are inlined as plain Python lists. matplotlib is only needed at registration time (when you `import earthmover_colormaps`). If you only need the raw RGB data:

```python
from earthmover_colormaps._data import COLORMAPS

# COLORMAPS["em.signal"] is a list of 256 [R, G, B] triples (0-1 floats)
```

## Development

```bash
git clone https://github.com/earth-mover/earthmover-colormaps.git
cd earthmover-colormaps
uv sync --group dev
uv run pytest
```

### Colormap design tooling

The `tools/` directory contains scripts for creating and optimizing colormaps using CAM02-UCS perceptual color space:

```bash
uv sync --group design     # install colorspacious, scipy, viscm
cd tools/
uv run python generate_data.py   # optimize all colormaps, regenerate _data.py
uv run python visualize.py       # generate diagnostic plots in images/
```

Each colormap is also exported as a `.jscm` file (in `earthmover_colormaps/_jscm/`) compatible with [viscm](https://github.com/matplotlib/viscm) for interactive inspection and refinement.

## License

Apache 2.0. See [LICENSE](LICENSE).
