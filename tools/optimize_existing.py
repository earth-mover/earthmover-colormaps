"""Re-fit existing colormaps via arc-length parameterization in CAM02-UCS.

For each colormap:
1. Convert 256-point RGB → CAM02-UCS J'a'b'
2. Arc-length parameterize in full 3D J'a'b' space
3. Gamut-clip back to sRGB
4. Fit Bézier control points for metadata/viscm compatibility

Special handling for diverging colormaps (split at lightness extremum).
"""

import numpy as np
from colorspace_utils import (
    srgb_to_cam02ucs,
    cam02ucs_to_srgb,
    compute_arc_length,
    parameterize_by_arc_length,
    enforce_monotonic_lightness,
    fit_bezier_to_path,
    clip_to_gamut,
    compute_uniformity_metrics,
)


def optimize_sequential(rgb_list, n_control=6, n=256, force_monotonic=True):
    """Optimize a sequential colormap via arc-length parameterization in CAM02-UCS.

    Parameters
    ----------
    rgb_list : list of [R, G, B]
        Original 256-entry RGB table (0-1 float).
    n_control : int
        Number of Bézier control points to fit for metadata.
    n : int
        Number of output samples.
    force_monotonic : bool
        If True, enforce monotonic lightness via isotonic regression.

    Returns
    -------
    rgb_optimized : ndarray, shape (n, 3)
        Optimized sRGB values in [0, 1].
    metadata : dict
        Bézier control points and parameters for reproducibility.
    """
    rgb = np.asarray(rgb_list, dtype=np.float64)
    Jpapbp = srgb_to_cam02ucs(rgb)

    # Arc-length parameterize in full 3D J'a'b'
    Jpapbp_uniform = parameterize_by_arc_length(Jpapbp, n=n)

    if force_monotonic:
        # Determine lightness direction and enforce monotonicity
        direction = "increasing" if Jpapbp_uniform[-1, 0] >= Jpapbp_uniform[0, 0] else "decreasing"
        Jpapbp_uniform = enforce_monotonic_lightness(Jpapbp_uniform, direction)

    # Convert back to sRGB and gamut-clip
    rgb_opt = cam02ucs_to_srgb(Jpapbp_uniform)
    rgb_opt = clip_to_gamut(Jpapbp_uniform, rgb_opt)

    if force_monotonic:
        # Final pass: enforce monotonicity after sRGB roundtrip
        Jpapbp_final = srgb_to_cam02ucs(rgb_opt)
        Jpapbp_final = enforce_monotonic_lightness(Jpapbp_final, direction)
        rgb_opt = cam02ucs_to_srgb(Jpapbp_final)
        rgb_opt = np.clip(rgb_opt, 0.0, 1.0)

    # Fit Bézier control points for metadata
    cp_ab, min_Jp, max_Jp = fit_bezier_to_path(Jpapbp_uniform, n_control=n_control)

    metadata = {
        "type": "sequential",
        "min_Jp": min_Jp,
        "max_Jp": max_Jp,
        "control_points_ab": cp_ab.tolist(),
        "n_control": n_control,
        "spline_method": "Bezier",
        "uniform_colorspace": "CAM02-UCS",
    }

    return rgb_opt, metadata


def optimize_diverging(rgb_list, n_control_per_arm=5, n=256):
    """Optimize a diverging colormap.

    Splits at the lightness extremum (midpoint), arc-length parameterizes
    each arm in full 3D J'a'b' space with proportional sample allocation,
    then recombines.

    Parameters
    ----------
    rgb_list : list of [R, G, B]
    n_control_per_arm : int
    n : int

    Returns
    -------
    rgb_optimized : ndarray, shape (n, 3)
    metadata : dict
    """
    rgb = np.asarray(rgb_list, dtype=np.float64)
    Jpapbp = srgb_to_cam02ucs(rgb)
    Jp = Jpapbp[:, 0]

    # Find the midpoint (maximum lightness for this diverging map)
    mid_idx = np.argmax(Jp)

    # Split into two arms
    left = Jpapbp[:mid_idx + 1]
    right = Jpapbp[mid_idx:]

    # Allocate samples proportional to each arm's arc-length
    left_arc = compute_arc_length(left)[-1]
    right_arc = compute_arc_length(right)[-1]
    total_arc = left_arc + right_arc
    n_left = int(round(n * left_arc / total_arc)) + 1  # +1 for shared midpoint
    n_right = n - n_left + 1

    # Arc-length parameterize each arm directly in 3D J'a'b'
    left_uniform = parameterize_by_arc_length(left, n=n_left)
    right_uniform = parameterize_by_arc_length(right, n=n_right)

    # Convert back to sRGB and gamut-clip
    rgb_left = cam02ucs_to_srgb(left_uniform)
    rgb_left = clip_to_gamut(left_uniform, rgb_left)
    rgb_right = cam02ucs_to_srgb(right_uniform)
    rgb_right = clip_to_gamut(right_uniform, rgb_right)

    # Combine (drop duplicate midpoint from right arm)
    rgb_opt = np.vstack([rgb_left, rgb_right[1:]])

    # Ensure exactly n entries
    if len(rgb_opt) != n:
        t_old = np.linspace(0, 1, len(rgb_opt))
        t_new = np.linspace(0, 1, n)
        rgb_opt = np.column_stack([
            np.interp(t_new, t_old, rgb_opt[:, c])
            for c in range(3)
        ])

    # Fit Bézier control points for metadata
    cp_left_ab, min_Jp_left, max_Jp_left = fit_bezier_to_path(
        left_uniform, n_control=n_control_per_arm
    )
    cp_right_ab, min_Jp_right, max_Jp_right = fit_bezier_to_path(
        right_uniform, n_control=n_control_per_arm
    )

    metadata = {
        "type": "diverging",
        "left": {
            "min_Jp": min_Jp_left,
            "max_Jp": max_Jp_left,
            "control_points_ab": cp_left_ab.tolist(),
        },
        "right": {
            "min_Jp": min_Jp_right,
            "max_Jp": max_Jp_right,
            "control_points_ab": cp_right_ab.tolist(),
        },
        "n_control_per_arm": n_control_per_arm,
        "spline_method": "Bezier",
        "uniform_colorspace": "CAM02-UCS",
    }

    return rgb_opt, metadata


def optimize_all(colormaps_dict):
    """Optimize all colormaps.

    Parameters
    ----------
    colormaps_dict : dict
        Keys are colormap names, values are 256-entry RGB lists.

    Returns
    -------
    optimized : dict
        Keys are colormap names, values are (rgb_array, metadata) tuples.
    """
    results = {}

    diverging_names = {"em.diverging"}
    # Colormaps with intentional lightness non-monotonicity
    non_monotonic_names = {"em.bloom"}

    for name, rgb_list in colormaps_dict.items():
        print(f"\nOptimizing {name}...")

        if name in diverging_names:
            rgb_opt, meta = optimize_diverging(rgb_list, n_control_per_arm=5)
        else:
            if name in ("em.signal", "em.ocean"):
                n_ctrl = 7
            elif name in ("em.bloom",):
                n_ctrl = 6
            else:
                n_ctrl = 5
            force_mono = name not in non_monotonic_names
            rgb_opt, meta = optimize_sequential(
                rgb_list, n_control=n_ctrl, force_monotonic=force_mono
            )

        # Report metrics
        old_metrics = compute_uniformity_metrics(rgb_list)
        new_metrics = compute_uniformity_metrics(rgb_opt)

        print(f"  CV: {old_metrics['cv_percent']:.1f}% -> {new_metrics['cv_percent']:.1f}%")
        print(f"  Mean dE: {old_metrics['mean_dE']:.3f} -> {new_metrics['mean_dE']:.3f}")
        print(f"  J' range: {old_metrics['Jp_range'][0]:.1f}-{old_metrics['Jp_range'][1]:.1f}"
              f" -> {new_metrics['Jp_range'][0]:.1f}-{new_metrics['Jp_range'][1]:.1f}")
        print(f"  Monotonic: {old_metrics['Jp_monotonic']} -> {new_metrics['Jp_monotonic']}")

        results[name] = (rgb_opt, meta)

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from earthmover_colormaps._data import COLORMAPS

    results = optimize_all(COLORMAPS)

    print("\n\n=== Summary ===")
    print(f"{'Name':<16} {'Old CV%':>8} {'New CV%':>8} {'Improvement':>12}")
    print("-" * 50)
    for name, (rgb_opt, meta) in results.items():
        old = compute_uniformity_metrics(COLORMAPS[name])
        new = compute_uniformity_metrics(rgb_opt)
        improvement = old["cv_percent"] - new["cv_percent"]
        print(f"{name:<16} {old['cv_percent']:>7.1f}% {new['cv_percent']:>7.1f}% {improvement:>+10.1f}pp")
