"""
analytics/trajectory_smoothing.py
=================================
Savitzky-Golay and polynomial smoothing for ball trajectories.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import savgol_filter


def savgol_smooth_path(
    points: List[Tuple[float, float]],
    window: int = 7,
    polyorder: int = 2,
) -> List[Tuple[float, float]]:
    """Savitzky-Golay smoothing — removes zig-zag while preserving arc shape."""
    if len(points) < max(5, window):
        return points

    w = window if window % 2 == 1 else window + 1
    w = min(w, len(points) if len(points) % 2 == 1 else len(points) - 1)
    w = max(5, w)
    if w >= len(points):
        w = len(points) - 1 if (len(points) - 1) % 2 == 1 else len(points) - 2
    if w < 5:
        return points

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    po = min(polyorder, w - 2)
    xs_s = savgol_filter(xs, w, po)
    ys_s = savgol_filter(ys, w, po)
    return [(float(xs_s[i]), float(ys_s[i])) for i in range(len(points))]


def polynomial_fit_path(
    points: List[Tuple[float, float]],
    degree: int = 3,
) -> List[Tuple[float, float]]:
    """Fit polynomial x(t), y(t) for physically smooth cricket-ball motion."""
    n = len(points)
    if n < degree + 2:
        return points

    t = np.linspace(0.0, 1.0, n)
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    deg = min(degree, n - 1)
    px = np.polyfit(t, xs, deg)
    py = np.polyfit(t, ys, deg)
    xs_f = np.polyval(px, t)
    ys_f = np.polyval(py, t)
    return [(float(xs_f[i]), float(ys_f[i])) for i in range(n)]


def moving_average_path(
    points: List[Tuple[float, float]],
    window: int = 5,
) -> List[Tuple[float, float]]:
    if len(points) < window:
        return points
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    k = np.ones(window) / window
    xs_s = np.convolve(xs, k, mode="same")
    ys_s = np.convolve(ys, k, mode="same")
    return [(float(xs_s[i]), float(ys_s[i])) for i in range(len(points))]
