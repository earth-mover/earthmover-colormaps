"""Core utilities for colormap design in CAM02-UCS perceptual color space.

Provides conversion, arc-length parameterization, Bézier evaluation,
perceptual uniformity metrics, and gamut clipping — the building blocks
used by the optimization and design scripts.
"""

import numpy as np
from colorspacious import cspace_convert
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Color-space conversions
# ---------------------------------------------------------------------------

def srgb_to_cam02ucs(rgb):
    """Convert sRGB (0-1 float) array to CAM02-UCS J'a'b'.

    Parameters
    ----------
    rgb : array-like, shape (..., 3)
        sRGB values in [0, 1].

    Returns
    -------
    Jpapbp : ndarray, shape (..., 3)
        J', a', b' values in CAM02-UCS.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    # colorspacious expects sRGB1 (0-1 range)
    return cspace_convert(rgb, "sRGB1", "CAM02-UCS")


def cam02ucs_to_srgb(Jpapbp):
    """Convert CAM02-UCS J'a'b' to sRGB (0-1 float).

    Parameters
    ----------
    Jpapbp : array-like, shape (..., 3)

    Returns
    -------
    rgb : ndarray, shape (..., 3)
        sRGB values (may be outside [0, 1] if out of gamut).
    """
    Jpapbp = np.asarray(Jpapbp, dtype=np.float64)
    return cspace_convert(Jpapbp, "CAM02-UCS", "sRGB1")


# ---------------------------------------------------------------------------
# Arc-length parameterization
# ---------------------------------------------------------------------------

def compute_arc_length(points):
    """Cumulative Euclidean arc-length along a path.

    Parameters
    ----------
    points : ndarray, shape (N, D)
        Points along the path in any D-dimensional space.

    Returns
    -------
    s : ndarray, shape (N,)
        Cumulative arc-length, starting from 0.
    """
    points = np.asarray(points, dtype=np.float64)
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    s = np.zeros(len(points))
    s[1:] = np.cumsum(seg_lengths)
    return s


def parameterize_by_arc_length(Jpapbp, n=256, oversample=10000):
    """Re-sample a path to have uniform spacing in CAM02-UCS arc-length.

    Parameters
    ----------
    Jpapbp : ndarray, shape (M, 3)
        Path in CAM02-UCS (J', a', b').
    n : int
        Number of output samples.
    oversample : int
        Number of points to use for the smooth interpolation.

    Returns
    -------
    resampled : ndarray, shape (n, 3)
        Uniformly-spaced path in CAM02-UCS.
    """
    Jpapbp = np.asarray(Jpapbp, dtype=np.float64)
    M = len(Jpapbp)

    # Compute chord-length parameterization for the input
    s_input = compute_arc_length(Jpapbp)
    total_length = s_input[-1]
    if total_length == 0:
        return np.tile(Jpapbp[0], (n, 1))

    # Normalize to [0, 1]
    t_input = s_input / total_length

    # Fit cubic spline through the path
    # Use chord-length parameter as the independent variable
    spline = CubicSpline(t_input, Jpapbp, axis=0)

    # Oversample the spline
    t_fine = np.linspace(0, 1, oversample)
    path_fine = spline(t_fine)

    # Compute arc-length of the oversampled path
    s_fine = compute_arc_length(path_fine)
    total_arc = s_fine[-1]

    # Target arc-lengths for uniform spacing
    s_target = np.linspace(0, total_arc, n)

    # Interpolate to find parameter values at target arc-lengths
    t_at_target = np.interp(s_target, s_fine, t_fine)

    # Evaluate spline at these parameters
    resampled = spline(t_at_target)
    return resampled


# ---------------------------------------------------------------------------
# Bézier curve evaluation
# ---------------------------------------------------------------------------

def _bernstein(n, i, t):
    """Bernstein basis polynomial B_{i,n}(t)."""
    from scipy.special import comb
    return comb(n, i, exact=True) * (t ** i) * ((1 - t) ** (n - i))


def evaluate_bezier(control_points, n_eval=10000):
    """Evaluate a Bézier curve defined by control points.

    Parameters
    ----------
    control_points : ndarray, shape (K, D)
        K control points in D dimensions.
    n_eval : int
        Number of evaluation points.

    Returns
    -------
    curve : ndarray, shape (n_eval, D)
        Points along the Bézier curve.
    """
    control_points = np.asarray(control_points, dtype=np.float64)
    K = len(control_points)
    degree = K - 1
    t = np.linspace(0, 1, n_eval)

    curve = np.zeros((n_eval, control_points.shape[1]))
    for i in range(K):
        B = _bernstein(degree, i, t)
        curve += np.outer(B, control_points[i])

    return curve


def bezier_colormap(control_points_ab, min_Jp, max_Jp, n=256):
    """Create a colormap from Bézier control points in CAM02-UCS.

    The a'b' path is defined by a Bézier curve. J' varies linearly
    from min_Jp to max_Jp along the arc-length-parameterized path.

    Parameters
    ----------
    control_points_ab : ndarray, shape (K, 2)
        Bézier control points in (a', b') space.
    min_Jp : float
        Starting lightness (J').
    max_Jp : float
        Ending lightness (J').
    n : int
        Number of output colors.

    Returns
    -------
    rgb : ndarray, shape (n, 3)
        sRGB values in [0, 1], gamut-clipped.
    """
    control_points_ab = np.asarray(control_points_ab, dtype=np.float64)

    # Evaluate Bézier in a'b' at high resolution
    ab_fine = evaluate_bezier(control_points_ab, n_eval=10000)

    # Arc-length parameterize the a'b' path
    s = compute_arc_length(ab_fine)
    total_arc = s[-1]

    if total_arc == 0:
        # Degenerate case: single-hue colormap
        t_uniform = np.zeros(n)
    else:
        s_target = np.linspace(0, total_arc, n)
        t_uniform = np.interp(s_target, s, np.linspace(0, 1, len(ab_fine)))

    # Interpolate a'b' at uniform arc-length positions
    ab_uniform = np.column_stack([
        np.interp(t_uniform, np.linspace(0, 1, len(ab_fine)), ab_fine[:, 0]),
        np.interp(t_uniform, np.linspace(0, 1, len(ab_fine)), ab_fine[:, 1]),
    ])

    # Linear J' from min to max
    Jp = np.linspace(min_Jp, max_Jp, n)

    # Combine into J'a'b'
    Jpapbp = np.column_stack([Jp, ab_uniform])

    # Convert to sRGB and clip gamut
    rgb = cam02ucs_to_srgb(Jpapbp)
    rgb = clip_to_gamut(Jpapbp, rgb)

    return rgb


# ---------------------------------------------------------------------------
# Gamut clipping
# ---------------------------------------------------------------------------

def clip_to_gamut(Jpapbp, rgb=None, tol=1e-4):
    """Clip out-of-gamut colors by desaturating toward the achromatic axis.

    Holds J' fixed and binary-searches along the ray from (a'=0, b'=0)
    to the current (a', b') to find the maximum chroma that stays in gamut.

    Parameters
    ----------
    Jpapbp : ndarray, shape (N, 3)
        Colors in CAM02-UCS.
    rgb : ndarray, shape (N, 3), optional
        Pre-computed sRGB values. If None, computed from Jpapbp.
    tol : float
        Binary search tolerance for the chroma scaling factor.

    Returns
    -------
    rgb_clipped : ndarray, shape (N, 3)
        sRGB values in [0, 1].
    """
    Jpapbp = np.asarray(Jpapbp, dtype=np.float64)
    if rgb is None:
        rgb = cam02ucs_to_srgb(Jpapbp)
    else:
        rgb = np.asarray(rgb, dtype=np.float64).copy()

    out_of_gamut = np.any((rgb < -tol) | (rgb > 1 + tol), axis=1)

    for idx in np.where(out_of_gamut)[0]:
        Jp = Jpapbp[idx, 0]
        ap = Jpapbp[idx, 1]
        bp = Jpapbp[idx, 2]

        # Binary search: scale (a', b') by factor in [0, 1]
        lo, hi = 0.0, 1.0
        for _ in range(50):  # plenty of iterations for convergence
            mid = (lo + hi) / 2
            test = np.array([Jp, ap * mid, bp * mid])
            test_rgb = cam02ucs_to_srgb(test)
            if np.all(test_rgb >= -tol) and np.all(test_rgb <= 1 + tol):
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break

        final = np.array([Jp, ap * lo, bp * lo])
        rgb[idx] = cam02ucs_to_srgb(final)

    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Perceptual uniformity metrics
# ---------------------------------------------------------------------------

def compute_uniformity_metrics(rgb_table):
    """Compute perceptual uniformity metrics for a colormap.

    Parameters
    ----------
    rgb_table : array-like, shape (N, 3)
        sRGB values in [0, 1].

    Returns
    -------
    metrics : dict
        Keys: 'mean_dE', 'std_dE', 'cv_percent', 'max_dE', 'min_dE',
        'total_arc_length', 'Jp_range', 'Jp_monotonic'.
    """
    rgb = np.asarray(rgb_table, dtype=np.float64)
    Jpapbp = srgb_to_cam02ucs(rgb)

    # Step-wise ΔE in CAM02-UCS
    diffs = np.diff(Jpapbp, axis=0)
    step_dE = np.sqrt(np.sum(diffs ** 2, axis=1))

    mean_dE = np.mean(step_dE)
    std_dE = np.std(step_dE)
    cv = (std_dE / mean_dE * 100) if mean_dE > 0 else 0.0

    Jp = Jpapbp[:, 0]
    Jp_diffs = np.diff(Jp)
    tol = 1e-4  # tolerance for roundtrip noise
    monotonic = bool(np.all(Jp_diffs > -tol) or np.all(Jp_diffs < tol))

    return {
        "mean_dE": float(mean_dE),
        "std_dE": float(std_dE),
        "cv_percent": float(cv),
        "max_dE": float(np.max(step_dE)),
        "min_dE": float(np.min(step_dE)),
        "total_arc_length": float(np.sum(step_dE)),
        "Jp_range": (float(np.min(Jp)), float(np.max(Jp))),
        "Jp_monotonic": monotonic,
    }


def check_monotonic_lightness(rgb_table, space="CAM02-UCS", tol=1e-4):
    """Check whether J' is monotonic (within tolerance).

    Parameters
    ----------
    rgb_table : array-like, shape (N, 3)
    space : str
        Color space for lightness. Default CAM02-UCS.
    tol : float
        Tolerance for violations. Violations smaller than this in J' units
        are ignored (they arise from sRGB roundtrip noise).

    Returns
    -------
    monotonic : bool
    direction : str or None
        'increasing', 'decreasing', or None if not monotonic.
    violations : int
        Number of non-monotonic steps exceeding tolerance.
    """
    rgb = np.asarray(rgb_table, dtype=np.float64)
    Jpapbp = srgb_to_cam02ucs(rgb)
    Jp = Jpapbp[:, 0]
    diffs = np.diff(Jp)

    n_inc = np.sum(diffs > tol)
    n_dec = np.sum(diffs < -tol)

    if n_dec == 0:
        return True, "increasing", 0
    elif n_inc == 0:
        return True, "decreasing", 0
    else:
        if n_inc >= n_dec:
            return False, "increasing", int(n_dec)
        else:
            return False, "decreasing", int(n_inc)


# ---------------------------------------------------------------------------
# Monotonicity enforcement
# ---------------------------------------------------------------------------

def enforce_monotonic_lightness(Jpapbp, direction="increasing"):
    """Enforce monotonic J' by isotonic regression, keeping a'b' unchanged.

    Parameters
    ----------
    Jpapbp : ndarray, shape (N, 3)
        Path in CAM02-UCS.
    direction : str
        'increasing' or 'decreasing'.

    Returns
    -------
    Jpapbp_fixed : ndarray, shape (N, 3)
        Path with monotonic J'.
    """
    from scipy.optimize import isotonic_regression

    Jpapbp = np.asarray(Jpapbp, dtype=np.float64).copy()
    Jp = Jpapbp[:, 0].copy()

    if direction == "decreasing":
        Jp = -Jp

    result = isotonic_regression(Jp)
    Jp_mono = result.x

    if direction == "decreasing":
        Jp_mono = -Jp_mono

    Jpapbp[:, 0] = Jp_mono
    return Jpapbp


# ---------------------------------------------------------------------------
# Bézier fitting to existing paths
# ---------------------------------------------------------------------------

def fit_bezier_to_path(Jpapbp, n_control=6):
    """Fit Bézier control points to an existing path in CAM02-UCS.

    Uses least-squares optimization to find control points that minimize
    the sum of squared distances between the Bézier curve and the target path.

    Parameters
    ----------
    Jpapbp : ndarray, shape (N, 3)
        Target path in CAM02-UCS.
    n_control : int
        Number of Bézier control points.

    Returns
    -------
    control_points_ab : ndarray, shape (n_control, 2)
        Fitted control points in (a', b') space.
    min_Jp : float
        Starting lightness.
    max_Jp : float
        Ending lightness.
    """
    from scipy.optimize import minimize

    Jpapbp = np.asarray(Jpapbp, dtype=np.float64)
    N = len(Jpapbp)
    min_Jp = Jpapbp[0, 0]
    max_Jp = Jpapbp[-1, 0]

    # Target a'b' values
    target_ab = Jpapbp[:, 1:3]

    # Parameter values for target (uniform in [0, 1])
    t = np.linspace(0, 1, N)

    # Initial guess: evenly spaced along the target a'b' path
    indices = np.linspace(0, N - 1, n_control).astype(int)
    x0 = target_ab[indices].ravel()

    def objective(params):
        cp = params.reshape(n_control, 2)
        # Fix first and last control points to match endpoints
        cp[0] = target_ab[0]
        cp[-1] = target_ab[-1]

        # Evaluate Bézier at N points
        curve = evaluate_bezier(cp, n_eval=N)
        return np.sum((curve - target_ab) ** 2)

    result = minimize(objective, x0, method="L-BFGS-B",
                      options={"maxiter": 5000, "ftol": 1e-12})

    cp_fitted = result.x.reshape(n_control, 2)
    # Ensure endpoints match exactly
    cp_fitted[0] = target_ab[0]
    cp_fitted[-1] = target_ab[-1]

    return cp_fitted, float(min_Jp), float(max_Jp)
