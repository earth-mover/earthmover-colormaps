"""Design colormaps from brand-color anchor points in CAM02-UCS.

`em.signal`, `em.diverging`, and `em.cycle` are cubic-spline curves through
the Earthmover brand colors at their natural J' values. The spline
interpolates through every anchor exactly, so brand violet and brand lime
appear at full saturation in the output, while the curve itself is
continuously differentiable (no visible kinks at the anchor joins).
"""

import numpy as np
from colorspace_utils import (
    smooth_anchored_colormap,
    compute_uniformity_metrics,
    check_monotonic_lightness,
    BRAND_MIDNIGHT_JPAPBP,
    BRAND_VIOLET_JPAPBP,
    BRAND_LIME_JPAPBP,
    BRAND_LIGHT_JPAPBP,
)


def design_lime(n=256):
    """em.lime: light grey → brand lime.

    Single-hue sequential. Index 0 is near-white, index 255 is the saturated
    Earthmover brand lime. Mirrors em.violet's white-to-brand pattern.
    """
    anchors = np.vstack([
        BRAND_LIGHT_JPAPBP,
        BRAND_LIME_JPAPBP,
    ])
    rgb = smooth_anchored_colormap(anchors, n=n)
    metadata = {
        "type": "sequential",
        "anchors_Jpapbp": anchors.tolist(),
        "spline_method": "cubic-spline",
        "uniform_colorspace": "CAM02-UCS",
    }
    return rgb, metadata


def design_bloom(n=256):
    """em.bloom: midnight → brand violet → light grey.

    A "violet-themed" sequential that passes through brand violet at full
    saturation in its middle. Wider J' range than em.violet (which only
    spans light → brand violet).
    """
    anchors = np.vstack([
        BRAND_MIDNIGHT_JPAPBP,
        BRAND_VIOLET_JPAPBP,
        BRAND_LIGHT_JPAPBP,
    ])
    rgb = smooth_anchored_colormap(anchors, n=n)
    metadata = {
        "type": "sequential",
        "anchors_Jpapbp": anchors.tolist(),
        "spline_method": "cubic-spline",
        "uniform_colorspace": "CAM02-UCS",
    }
    return rgb, metadata


def design_signal(n=256):
    """em.signal: midnight → brand violet → brand lime → light grey."""
    anchors = np.vstack([
        BRAND_MIDNIGHT_JPAPBP,
        BRAND_VIOLET_JPAPBP,
        BRAND_LIME_JPAPBP,
        BRAND_LIGHT_JPAPBP,
    ])
    rgb = smooth_anchored_colormap(anchors, n=n)
    metadata = {
        "type": "sequential",
        "anchors_Jpapbp": anchors.tolist(),
        "spline_method": "cubic-spline",
        "uniform_colorspace": "CAM02-UCS",
    }
    return rgb, metadata


def design_diverging(n=256):
    """em.diverging: brand violet ↔ light grey ↔ brand lime."""
    anchors = np.vstack([
        BRAND_VIOLET_JPAPBP,
        BRAND_LIGHT_JPAPBP,
        BRAND_LIME_JPAPBP,
    ])
    rgb = smooth_anchored_colormap(anchors, n=n)
    metadata = {
        "type": "diverging",
        "anchors_Jpapbp": anchors.tolist(),
        "spline_method": "cubic-spline",
        "uniform_colorspace": "CAM02-UCS",
    }
    return rgb, metadata


def design_cycle(n=256):
    """em.cycle: midnight → brand violet → light grey → brand lime → midnight."""
    anchors = np.vstack([
        BRAND_MIDNIGHT_JPAPBP,
        BRAND_VIOLET_JPAPBP,
        BRAND_LIGHT_JPAPBP,
        BRAND_LIME_JPAPBP,
        BRAND_MIDNIGHT_JPAPBP,
    ])
    rgb = smooth_anchored_colormap(anchors, n=n)
    metadata = {
        "type": "cyclic",
        "anchors_Jpapbp": anchors.tolist(),
        "spline_method": "cubic-spline",
        "uniform_colorspace": "CAM02-UCS",
    }
    return rgb, metadata


def design_all():
    """Design all from-scratch colormaps and report metrics."""
    designers = {
        "em.lime": design_lime,
        "em.signal": design_signal,
        "em.diverging": design_diverging,
        "em.bloom": design_bloom,
        "em.cycle": design_cycle,
    }
    results = {}
    for name, fn in designers.items():
        print(f"\nDesigning {name}...")
        rgb, meta = fn()
        m = compute_uniformity_metrics(rgb)
        mono, direction, violations = check_monotonic_lightness(rgb)
        Jp_lo, Jp_hi = m["Jp_range"]
        print(f"  CV: {m['cv_percent']:.1f}%   mean ΔE: {m['mean_dE']:.3f}   "
              f"max ΔE: {m['max_dE']:.3f}")
        print(f"  J' range: {Jp_lo:.1f}-{Jp_hi:.1f}   "
              f"monotonic: {mono} ({direction}, {violations})")
        print(f"  start RGB: [{rgb[0, 0]:.4f}, {rgb[0, 1]:.4f}, {rgb[0, 2]:.4f}]")
        print(f"  end RGB:   [{rgb[-1, 0]:.4f}, {rgb[-1, 1]:.4f}, {rgb[-1, 2]:.4f}]")
        results[name] = (rgb, meta)
    return results


if __name__ == "__main__":
    design_all()
