"""Design new colormaps from Bézier control points in CAM02-UCS.

Creates em.heat, em.earth, and em.twilight colormaps defined directly
as Bézier curves in the perceptually uniform CAM02-UCS color space.
"""

import numpy as np
from colorspace_utils import (
    srgb_to_cam02ucs,
    cam02ucs_to_srgb,
    compute_arc_length,
    parameterize_by_arc_length,
    evaluate_bezier,
    enforce_monotonic_lightness,
    bezier_colormap,
    clip_to_gamut,
    compute_uniformity_metrics,
    check_monotonic_lightness,
)


# ---------------------------------------------------------------------------
# Brand anchor colors in CAM02-UCS
# ---------------------------------------------------------------------------
# Midnight: J'=15.42, a'= 1.27, b'=-7.96  (sRGB 0.126, 0.122, 0.173)
# Violet:   J'=57.03, a'=16.24, b'=-31.44 (sRGB 0.651, 0.325, 1.000)
# Lime:     J'=85.85, a'=-17.26, b'=32.71 (sRGB 0.718, 0.894, 0.000)
# Light:    J'=97.09, a'=-1.87, b'=-1.12  (sRGB 0.961, 0.961, 0.961)


def design_heat(n=256):
    """em.heat: midnight -> deep red -> orange -> amber -> warm white.

    A warm sequential colormap for temperature, energy, intensity.

    Control points trace a path through warm hues in CAM02-UCS:
    - Start at Midnight (brand dark anchor)
    - Through deep red-browns
    - Through orange
    - Through amber/gold
    - End at warm light
    """
    # Define path in CAM02-UCS as (J', a', b')
    # J' increases linearly, a'b' follows a Bézier curve through warm hues
    control_points_Jpapbp = np.array([
        [15.4,   1.3,   -8.0],   # Midnight (brand)
        [28.0,  20.0,    8.0],   # Dark red-brown
        [45.0,  30.0,   22.0],   # Deep red
        [62.0,  24.0,   28.0],   # Red-orange
        [76.0,  12.0,   30.0],   # Orange-amber
        [90.0,   0.0,   18.0],   # Warm gold
        [97.0,  -1.0,    4.0],   # Warm white (near brand light)
    ])

    # Evaluate Bézier in full 3D J'a'b'
    curve = evaluate_bezier(control_points_Jpapbp, n_eval=10000)

    # Arc-length parameterize
    resampled = parameterize_by_arc_length(curve, n=n)

    # Enforce monotonic lightness
    resampled = enforce_monotonic_lightness(resampled, "increasing")

    # Convert to sRGB and gamut-clip
    rgb = cam02ucs_to_srgb(resampled)
    rgb = clip_to_gamut(resampled, rgb)

    # Final monotonicity pass after roundtrip
    Jpapbp_final = srgb_to_cam02ucs(rgb)
    Jpapbp_final = enforce_monotonic_lightness(Jpapbp_final, "increasing")
    rgb = cam02ucs_to_srgb(Jpapbp_final)
    rgb = np.clip(rgb, 0.0, 1.0)

    metadata = {
        "type": "sequential",
        "control_points_Jpapbp": control_points_Jpapbp.tolist(),
        "spline_method": "Bezier",
        "uniform_colorspace": "CAM02-UCS",
    }

    return rgb, metadata


def design_earth(n=256):
    """em.earth: deep blue -> teal -> green -> sand -> brown -> cream.

    A terrain/bathymetry colormap for geographic/topographic data.
    Starts with deep ocean blues, transitions through vegetation greens,
    to earthy browns and sandy neutrals.
    """
    control_points_Jpapbp = np.array([
        [15.4,   0.0,  -12.0],   # Near midnight, slight blue
        [27.0,  -8.0,  -10.0],   # Dark teal
        [40.0, -14.0,   -1.0],   # Teal
        [54.0, -10.0,    7.0],   # Green transitioning to earthy
        [68.0,   1.0,   14.0],   # Sandy tan
        [82.0,   3.0,   11.0],   # Light sand
        [94.5,  -1.0,    4.0],   # Cream
    ])

    curve = evaluate_bezier(control_points_Jpapbp, n_eval=10000)
    resampled = parameterize_by_arc_length(curve, n=n)
    resampled = enforce_monotonic_lightness(resampled, "increasing")
    rgb = cam02ucs_to_srgb(resampled)
    rgb = clip_to_gamut(resampled, rgb)

    Jpapbp_final = srgb_to_cam02ucs(rgb)
    Jpapbp_final = enforce_monotonic_lightness(Jpapbp_final, "increasing")
    rgb = cam02ucs_to_srgb(Jpapbp_final)
    rgb = np.clip(rgb, 0.0, 1.0)

    metadata = {
        "type": "sequential",
        "control_points_Jpapbp": control_points_Jpapbp.tolist(),
        "spline_method": "Bezier",
        "uniform_colorspace": "CAM02-UCS",
    }

    return rgb, metadata


def design_twilight(n=256):
    """em.twilight: cyclic colormap midnight -> violet -> light -> lime -> midnight.

    A cyclic colormap for phase, angle, or time-of-day data.
    The path forms a closed loop through the brand palette:
    midnight (dark) -> violet (cool) -> light grey (bright) -> lime (warm) -> midnight (dark)

    The lightness profile is symmetric: dark at both ends, light in the middle.
    """
    # For a cyclic map, we trace through the brand colors in a loop.
    # J' follows an inverted-V: low -> high -> low
    # a'b' traces violet -> neutral -> lime in the brand axis

    # Split into two halves and combine
    n_half = n // 2 + 1  # include midpoint

    # First half: midnight -> violet -> light grey
    half1_Jpapbp = np.array([
        [15.4,   1.3,   -8.0],   # Midnight
        [35.0,  10.0,  -22.0],   # Dark violet
        [57.0,  16.0,  -31.0],   # Violet (brand)
        [78.0,   6.0,  -18.0],   # Light violet
        [97.0,  -1.9,   -1.1],   # Light grey (brand)
    ])

    # Second half: light grey -> lime -> midnight
    half2_Jpapbp = np.array([
        [97.0,  -1.9,   -1.1],   # Light grey (brand)
        [92.0, -10.0,   16.0],   # Light lime
        [85.8, -17.3,   32.7],   # Lime (brand)
        [55.0, -10.0,   16.0],   # Dark green
        [15.4,   1.3,   -8.0],   # Midnight
    ])

    # Evaluate each half
    curve1 = evaluate_bezier(half1_Jpapbp, n_eval=5000)
    curve2 = evaluate_bezier(half2_Jpapbp, n_eval=5000)

    # Arc-length parameterize each half
    s1 = compute_arc_length(curve1)[-1]
    s2 = compute_arc_length(curve2)[-1]
    total = s1 + s2

    # Allocate samples proportional to arc length
    n1 = int(round(n * s1 / total)) + 1
    n2 = n - n1 + 1

    r1 = parameterize_by_arc_length(curve1, n=n1)
    r2 = parameterize_by_arc_length(curve2, n=n2)

    # Convert to sRGB and gamut-clip
    rgb1 = cam02ucs_to_srgb(r1)
    rgb1 = clip_to_gamut(r1, rgb1)
    rgb2 = cam02ucs_to_srgb(r2)
    rgb2 = clip_to_gamut(r2, rgb2)

    # Combine (drop duplicate midpoint)
    rgb = np.vstack([rgb1, rgb2[1:]])

    # Ensure exactly n entries
    if len(rgb) != n:
        t_old = np.linspace(0, 1, len(rgb))
        t_new = np.linspace(0, 1, n)
        rgb = np.column_stack([
            np.interp(t_new, t_old, rgb[:, c])
            for c in range(3)
        ])

    rgb = np.clip(rgb, 0.0, 1.0)

    metadata = {
        "type": "cyclic",
        "half1_control_points": half1_Jpapbp.tolist(),
        "half2_control_points": half2_Jpapbp.tolist(),
        "spline_method": "Bezier",
        "uniform_colorspace": "CAM02-UCS",
    }

    return rgb, metadata


def design_all():
    """Design all new colormaps and report metrics."""
    results = {}

    designers = {
        "em.heat": design_heat,
        "em.earth": design_earth,
        "em.twilight": design_twilight,
    }

    for name, fn in designers.items():
        print(f"\nDesigning {name}...")
        rgb, meta = fn()

        metrics = compute_uniformity_metrics(rgb)
        mono, direction, violations = check_monotonic_lightness(rgb)

        print(f"  CV: {metrics['cv_percent']:.1f}%")
        print(f"  Mean dE: {metrics['mean_dE']:.3f}")
        print(f"  Max dE: {metrics['max_dE']:.3f}")
        print(f"  J' range: {metrics['Jp_range'][0]:.1f}-{metrics['Jp_range'][1]:.1f}")
        print(f"  Monotonic: {mono} ({direction}, {violations} violations)")

        # Print first and last RGB for visual check
        print(f"  Start RGB: [{rgb[0, 0]:.4f}, {rgb[0, 1]:.4f}, {rgb[0, 2]:.4f}]")
        print(f"  End RGB:   [{rgb[-1, 0]:.4f}, {rgb[-1, 1]:.4f}, {rgb[-1, 2]:.4f}]")

        results[name] = (rgb, meta)

    return results


if __name__ == "__main__":
    results = design_all()

    print("\n\n=== Summary ===")
    header = f"{'Name':<16} {'CV%':>6} {'Mean dE':>8} {'Jp range':>12} {'Monotonic':>10}"
    print(header)
    print("-" * 55)
    for name, (rgb, meta) in results.items():
        m = compute_uniformity_metrics(rgb)
        mono, direction, _ = check_monotonic_lightness(rgb)
        Jp_lo, Jp_hi = m["Jp_range"]
        print(f"{name:<16} {m['cv_percent']:>5.1f}% {m['mean_dE']:>8.3f} {Jp_lo:>5.1f}-{Jp_hi:>5.1f} {str(mono):>10}")
