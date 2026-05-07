# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`earthmover-colormaps` is a zero-dependency Python package that ships nine perceptually uniform matplotlib colormaps derived from the Earthmover brand palette (Midnight `#201F2C`, Violet `#A653FF`, Lime `#B7E400`). The runtime package has no required dependencies — RGB lookup tables are inlined as plain Python lists in `earthmover_colormaps/_data.py`. matplotlib is only touched at registration time.

## Commands

Uses [uv](https://docs.astral.sh/uv/) for environment management. Python 3.10+.

```bash
uv sync --group dev                       # install runtime deps
uv sync --group design                    # install design deps (adds viscm, pyqt6, xarray)
uv run python tools/generate_data.py      # regenerate _data.py + .jscm files
uv run python tools/visualize.py          # regenerate diagnostic plots in images/
uv run --group design python tools/compare_gui.py   # interactive comparison GUI
```

There is no test suite or lint/format config — correctness is verified by the validation report `tools/generate_data.py` prints (CV %, J' range, monotonicity per map) and by visual inspection in [`tools/compare_gui.py`](tools/compare_gui.py).

## Architecture

Two-layer split: a tiny runtime package, and a separate design pipeline that produces its data.

### Runtime package (`earthmover_colormaps/`)

- [`_data.py`](earthmover_colormaps/_data.py) — generated. A `COLORMAPS` dict mapping names like `"em.signal"` to a 256-entry list of `[R, G, B]` floats. Treat as a build artifact: do **not** hand-edit; regenerate with `tools/generate_data.py`.
- [`__init__.py`](earthmover_colormaps/__init__.py) — exposes three access patterns: matplotlib string lookup (`cmap="em.signal"` after import auto-registers), attribute access (`earthmover_colormaps.signal`), and dict access (`earthmover_colormaps.cm["em.signal"]`). Colormap objects are built lazily via `_build()` on first access; `register()` then pushes them into matplotlib's global registry. Reversed variants (`_r`) are generated automatically. Importing matplotlib is wrapped in `try/except ImportError` so the package can still be imported and `_data.COLORMAPS` consumed even without matplotlib installed.
- [`_jscm/`](earthmover_colormaps/_jscm/) — generated viscm-compatible `.jscm` files for each colormap (interactive inspection in [viscm](https://github.com/matplotlib/viscm)).

### Design pipeline (`tools/`)

The design system operates in **CAM02-UCS** perceptual color space (J', a', b'), not sRGB. Sequential maps target **linear J' as a function of index**, built via `bezier_colormap` which arc-length-parameterizes the (a',b') chroma curve and pairs it with linear J'. Diverging targets a V-shaped J' profile with equal-J' endpoints (the `find_most_vibrant_hue` helper pushes the dark endpoint to the gamut boundary so brand hues stay saturated when both arms are pulled to a shared dark J'). Cyclic (`em.cycle`) requires matching start/end. Step-wise ΔE uniformity is a soft constraint — `generate_data.py` prints CV per map; expect <2% for everything currently shipped.

- [`colorspace_utils.py`](tools/colorspace_utils.py) — primitives: sRGB↔CAM02-UCS conversion, arc-length parameterization, Bézier evaluation, gamut clipping, monotonic-lightness enforcement (isotonic regression), uniformity metrics.
- [`optimize_existing.py`](tools/optimize_existing.py) — re-fits the original six colormaps (`violet`, `lime`, `signal`, `diverging`, `ocean`, `bloom`): converts existing RGB → J'a'b', arc-length parameterizes, gamut-clips back to sRGB, fits Bézier control points for metadata. Diverging maps are split at the lightness extremum and each arm processed independently.
- [`design_new.py`](tools/design_new.py) — designs `cycle` from scratch as a Bézier curve directly in CAM02-UCS using the brand anchor colors.
- [`generate_data.py`](tools/generate_data.py) — orchestrator. Runs optimize → design → validate (CV%, ΔE, J' range, monotonicity) → write `_data.py` and the `_jscm/` files. This is the single entry point; running it overwrites the generated files.
- [`visualize.py`](tools/visualize.py) — produces `images/` (swatches, lightness profiles, ΔE uniformity, a'b' chromaticity, colorblind simulation panels using Viénot et al. dichromacy matrices).

When adding a new colormap, the contract is: add it to `design_new.py` (or wherever appropriate), wire it into `generate_data.py`'s `SEQUENTIAL_INPUT`/`DESIGNED` tuples, regenerate, then update `__init__.py`'s `__all__` and `tools/compare_gui.py`'s `EM_NAMES`.
