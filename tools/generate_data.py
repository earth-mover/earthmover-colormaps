"""Master script: optimize existing colormaps, generate new ones, write _data.py and .jscm files.

Usage:
    cd tools/
    uv run python generate_data.py
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from colorspace_utils import compute_uniformity_metrics, check_monotonic_lightness
from optimize_existing import optimize_all
from design_new import design_all
from earthmover_colormaps._data import COLORMAPS


def write_data_py(all_colormaps, output_path):
    """Write the _data.py file with all colormap RGB tables.

    Parameters
    ----------
    all_colormaps : dict
        Keys are colormap names, values are (rgb_array, metadata) tuples.
    output_path : str
        Path to write _data.py.
    """
    lines = ['"""Inlined 256-entry RGB lookup tables for Earthmover colormaps."""\n']
    lines.append("")
    lines.append("COLORMAPS = {")

    for name, (rgb, meta) in all_colormaps.items():
        lines.append(f'    "{name}": [')
        for i, (r, g, b) in enumerate(rgb):
            comma = "," if i < len(rgb) - 1 else ","
            lines.append(f"        [{r:.6f}, {g:.6f}, {b:.6f}]{comma}")
        lines.append("    ],")

    lines.append("}")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def write_jscm(name, rgb, meta, output_dir):
    """Write a viscm-compatible .jscm file.

    Parameters
    ----------
    name : str
        Colormap name (e.g., 'em.signal').
    rgb : ndarray, shape (256, 3)
        sRGB values.
    meta : dict
        Colormap metadata with control points.
    output_dir : str
        Directory to write .jscm files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert RGB to hex string
    hex_colors = []
    for r, g, b in rgb:
        ri, gi, bi = int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
        hex_colors.append(f"#{ri:02x}{gi:02x}{bi:02x}")

    cmap_type = meta.get("type", "sequential")
    usage_hints = []
    if cmap_type == "sequential":
        usage_hints = ["sequential", "greyscale-safe"]
    elif cmap_type == "diverging":
        usage_hints = ["diverging", "greyscale-safe"]
    elif cmap_type == "cyclic":
        usage_hints = ["cyclic"]

    jscm = {
        "content-type": "application/vnd.matplotlib.colormap-v1+json",
        "name": name,
        "license": "http://www.apache.org/licenses/LICENSE-2.0",
        "usage-hints": usage_hints,
        "colorspace": "sRGB",
        "domain": "continuous",
        "colors": "".join(hex_colors),
        "extensions": {
            "https://matplotlib.org/viscm": meta,
        },
    }

    filename = name.replace(".", "_") + ".jscm"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(jscm, f, indent=2)


def main():
    print("=" * 60)
    print("Earthmover Colormaps Generator")
    print("=" * 60)

    # Step 1: Optimize existing colormaps
    print("\n--- Optimizing existing colormaps ---")
    optimized = optimize_all(COLORMAPS)

    # Step 2: Design new colormaps
    print("\n--- Designing new colormaps ---")
    new_cmaps = design_all()

    # Step 3: Combine all colormaps (preserving order)
    all_colormaps = {}

    # Existing (optimized) in original order
    for name in COLORMAPS:
        all_colormaps[name] = optimized[name]

    # New colormaps
    for name in ("em.heat", "em.earth", "em.twilight"):
        all_colormaps[name] = new_cmaps[name]

    # Step 4: Validate all
    print("\n\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    header = f"{'Name':<16} {'CV%':>6} {'Mean dE':>8} {'Max dE':>8} {'Jp range':>12} {'Mono':>6}"
    print(header)
    print("-" * 60)

    all_pass = True
    for name, (rgb, meta) in all_colormaps.items():
        m = compute_uniformity_metrics(rgb)
        mono, direction, violations = check_monotonic_lightness(rgb)
        Jp_lo, Jp_hi = m["Jp_range"]
        mono_str = "Y" if mono else "N"

        cv = m["cv_percent"]
        status = "OK" if cv < 2.0 else "WARN"
        if status == "WARN":
            all_pass = False

        print(f"{name:<16} {cv:>5.1f}% {m['mean_dE']:>8.3f} {m['max_dE']:>8.3f}"
              f" {Jp_lo:>5.1f}-{Jp_hi:>5.1f} {mono_str:>6}  {status}")

    if all_pass:
        print("\nAll colormaps pass validation (CV < 2%).")
    else:
        print("\nWARNING: Some colormaps have CV >= 2%.")

    # Step 5: Write _data.py
    data_path = os.path.join(os.path.dirname(__file__), "..",
                             "earthmover_colormaps", "_data.py")
    data_path = os.path.normpath(data_path)
    write_data_py(all_colormaps, data_path)
    print(f"\nWrote {data_path}")

    # Step 6: Write .jscm files
    jscm_dir = os.path.join(os.path.dirname(__file__), "..",
                            "earthmover_colormaps", "_jscm")
    jscm_dir = os.path.normpath(jscm_dir)
    for name, (rgb, meta) in all_colormaps.items():
        write_jscm(name, rgb, meta, jscm_dir)
    print(f"Wrote {len(all_colormaps)} .jscm files to {jscm_dir}/")

    print(f"\nTotal colormaps: {len(all_colormaps)}")
    print("Done!")


if __name__ == "__main__":
    main()
