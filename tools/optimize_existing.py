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
    bezier_colormap,
    clip_to_gamut,
    compute_uniformity_metrics,
    find_most_vibrant_hue,
    BRAND_VIOLET_JPAPBP,
    BRAND_LIME_JPAPBP,
)


def optimize_sequential(rgb_list, n_control=6, n=256, force_monotonic=True,
                        linearize_lightness=True, dark_anchor=None,
                        light_anchor=None):
    """Optimize a sequential colormap via arc-length parameterization in CAM02-UCS.

    Parameters
    ----------
    rgb_list : list of [R, G, B]
    n_control : int
    n : int
    force_monotonic, linearize_lightness : bool
    dark_anchor, light_anchor : ndarray of shape (3,), optional
        CAM02-UCS (J', a', b') values to override the dark/light endpoint of
        the input path. Use to anchor an endpoint to a brand color.

    Returns
    -------
    rgb_optimized : ndarray, shape (n, 3)
    metadata : dict
    """
    rgb = np.asarray(rgb_list, dtype=np.float64)
    Jpapbp = srgb_to_cam02ucs(rgb)

    # Anchor endpoints to brand colors before any resampling. Replacing the
    # endpoint alone is not enough because the input path may pass through
    # J' values *outside* the new range (e.g. em.violet originally goes light
    # → midnight, passing through J'<57 before any anchor at J'=57). We
    # truncate intermediate points to stay strictly inside the new J' range
    # and substitute the anchor as the endpoint, so the resulting path is
    # monotonic in J' between the new endpoints.
    if dark_anchor is not None or light_anchor is not None:
        Jp = Jpapbp[:, 0]
        light_first = Jp[0] >= Jp[-1]
        light_end = light_anchor if light_anchor is not None else Jpapbp[0 if light_first else -1]
        dark_end = dark_anchor if dark_anchor is not None else Jpapbp[-1 if light_first else 0]
        light_Jp, dark_Jp = float(light_end[0]), float(dark_end[0])
        lo, hi = min(light_Jp, dark_Jp), max(light_Jp, dark_Jp)
        keep = (Jp > lo) & (Jp < hi)
        intermediate = Jpapbp[keep]
        # Sort intermediate so its J' progresses smoothly between the endpoints
        order = np.argsort(intermediate[:, 0])
        if light_first:
            order = order[::-1]
            Jpapbp = np.vstack([light_end[None], intermediate[order], dark_end[None]])
        else:
            Jpapbp = np.vstack([dark_end[None], intermediate[order], light_end[None]])

    # Arc-length parameterize in full 3D J'a'b' to get a clean reference path.
    Jpapbp_uniform = parameterize_by_arc_length(Jpapbp, n=n)

    # Skip isotonic regression when endpoints were anchored — the truncation +
    # sort above already produced a monotonic input, and isotonic regression
    # subtly shifts the endpoint J' away from the anchor.
    anchored = dark_anchor is not None or light_anchor is not None
    if force_monotonic and not anchored:
        direction = "increasing" if Jpapbp_uniform[-1, 0] >= Jpapbp_uniform[0, 0] else "decreasing"
        Jpapbp_uniform = enforce_monotonic_lightness(Jpapbp_uniform, direction)
    else:
        direction = "increasing" if Jpapbp_uniform[-1, 0] >= Jpapbp_uniform[0, 0] else "decreasing"

    # Fit a Bézier curve to the (a',b') path. This smooths chroma evolution.
    cp_ab, min_Jp, max_Jp = fit_bezier_to_path(Jpapbp_uniform, n_control=n_control)

    if linearize_lightness:
        # Construct the colormap from the smoothed chroma curve + linear J'.
        # bezier_colormap arc-length-parameterizes (a',b') and pairs it with
        # linear J', so both ΔJ' and Δ(a',b') are constant, giving low CV
        # *and* a straight lightness ramp.
        rgb_opt = bezier_colormap(cp_ab, min_Jp, max_Jp, n=n)
    else:
        # Legacy path: just convert + gamut-clip the arc-length samples.
        rgb_opt = cam02ucs_to_srgb(Jpapbp_uniform)
        rgb_opt = clip_to_gamut(Jpapbp_uniform, rgb_opt)
        if force_monotonic:
            Jpapbp_final = srgb_to_cam02ucs(rgb_opt)
            Jpapbp_final = enforce_monotonic_lightness(Jpapbp_final, direction)
            rgb_opt = cam02ucs_to_srgb(Jpapbp_final)
            rgb_opt = np.clip(rgb_opt, 0.0, 1.0)

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


def optimize_diverging(rgb_list, n_control_per_arm=5, n=256,
                       use_brand_endpoints=True,
                       equalize_endpoints=False, target_endpoint_Jp=None,
                       linearize_lightness=True):
    """Optimize a diverging colormap.

    Splits at the lightness extremum (midpoint) and processes each arm
    independently.

    With `use_brand_endpoints=True` (default), the violet and lime endpoints
    are anchored to the Earthmover brand colors (J'=57 / J'=86 respectively).
    The arms are asymmetric in J' span — the violet side is taller — but each
    arm is internally perceptually uniform (linear J', arc-length chroma) and
    both ends sit at brand-recognisable hues.

    With `equalize_endpoints=True`, both endpoints land at a common J'
    (`target_endpoint_Jp`, default: midpoint of the two original endpoints).
    The dark-end chroma is pushed to the gamut boundary near the brand hue
    via `find_most_vibrant_hue` to mitigate desaturation. Mutually exclusive
    with `use_brand_endpoints`.

    Parameters
    ----------
    rgb_list : list of [R, G, B]
    n_control_per_arm : int
    n : int
    use_brand_endpoints : bool
    equalize_endpoints : bool
    target_endpoint_Jp : float, optional
    linearize_lightness : bool

    Returns
    -------
    rgb_optimized : ndarray, shape (n, 3)
    metadata : dict
    """
    if use_brand_endpoints and equalize_endpoints:
        raise ValueError("use_brand_endpoints and equalize_endpoints are mutually exclusive")
    rgb = np.asarray(rgb_list, dtype=np.float64)
    Jpapbp = srgb_to_cam02ucs(rgb)
    Jp = Jpapbp[:, 0]

    # Find the midpoint (maximum lightness for this diverging map)
    mid_idx = np.argmax(Jp)
    peak_Jp = float(Jp[mid_idx])

    # Split into two arms (in CAM02-UCS)
    left = Jpapbp[:mid_idx + 1].copy()   # increasing in J' from left[0] to peak
    right = Jpapbp[mid_idx:].copy()      # decreasing in J' from peak to right[-1]

    if use_brand_endpoints:
        # Anchor the dark ends to the brand colors at their natural J'.
        left[0] = BRAND_VIOLET_JPAPBP
        right[-1] = BRAND_LIME_JPAPBP
        left_target_lo = float(BRAND_VIOLET_JPAPBP[0])
        right_target_lo = float(BRAND_LIME_JPAPBP[0])
    elif equalize_endpoints:
        if target_endpoint_Jp is None:
            target_endpoint_Jp = float((left[0, 0] + right[-1, 0]) / 2)
        left_target_lo = right_target_lo = target_endpoint_Jp
    else:
        left_target_lo = float(left[0, 0])
        right_target_lo = float(right[-1, 0])

    # Allocate samples proportional to each arm's 3D J'a'b' arc length, so
    # that chroma steps stay balanced even when the J' spans of the two arms
    # are very asymmetric (which they are in brand-endpoints mode: the lime
    # arm spans only ~11 J' units but moves substantially in chroma).
    left_len = float(compute_arc_length(left)[-1])
    right_len = float(compute_arc_length(right)[-1])
    total_len = left_len + right_len
    n_left = int(round(n * left_len / total_len)) + 1   # +1 for shared midpoint
    n_right = n - n_left + 1

    if linearize_lightness:
        # Smooth (a',b') with a Bézier fit, then build each arm with linear J'
        # and arc-length-uniform chroma → both ΔJ' and Δ(a',b') constant inside
        # an arm, giving low CV.
        left_arc = parameterize_by_arc_length(left, n=max(n_left, 64))
        right_arc = parameterize_by_arc_length(right, n=max(n_right, 64))
        cp_left_ab, _, _ = fit_bezier_to_path(left_arc, n_control=n_control_per_arm)
        cp_right_ab, _, _ = fit_bezier_to_path(right_arc, n_control=n_control_per_arm)

        if equalize_endpoints:
            # The dark-end J' was pulled away from the original; push chroma
            # to the gamut boundary near the brand hue to avoid muddy clipping.
            ap, bp = find_most_vibrant_hue(left_target_lo,
                                           cp_left_ab[0, 0], cp_left_ab[0, 1])
            cp_left_ab[0] = (ap, bp)
            ap, bp = find_most_vibrant_hue(right_target_lo,
                                           cp_right_ab[-1, 0], cp_right_ab[-1, 1])
            cp_right_ab[-1] = (ap, bp)
        # In brand-endpoints mode the endpoints are already brand-saturated
        # at their natural J', no rescue needed.

        rgb_left = bezier_colormap(cp_left_ab, left_target_lo, peak_Jp, n=n_left)
        rgb_right = bezier_colormap(cp_right_ab, peak_Jp, right_target_lo, n=n_right)
    else:
        left_uniform = parameterize_by_arc_length(left, n=n_left)
        right_uniform = parameterize_by_arc_length(right, n=n_right)
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

    # Fit Bézier control points for metadata. Re-fit from final J'a'b' so
    # metadata reflects the actual output (covers both linearize / legacy paths).
    final_Jpapbp = srgb_to_cam02ucs(rgb_opt)
    final_peak_idx = int(np.argmax(final_Jpapbp[:, 0]))
    cp_left_ab, min_Jp_left, max_Jp_left = fit_bezier_to_path(
        final_Jpapbp[: final_peak_idx + 1], n_control=n_control_per_arm
    )
    cp_right_ab, min_Jp_right, max_Jp_right = fit_bezier_to_path(
        final_Jpapbp[final_peak_idx:], n_control=n_control_per_arm
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

    # Single-hue maps that should terminate at a brand color rather than at
    # midnight / light grey, so the named hue actually appears at full brand
    # saturation in the output.
    brand_anchors = {
        "em.violet": {"dark_anchor": BRAND_VIOLET_JPAPBP},
    }

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
            rgb_opt, meta = optimize_sequential(
                rgb_list, n_control=n_ctrl, force_monotonic=True,
                linearize_lightness=True,
                **brand_anchors.get(name, {}),
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
