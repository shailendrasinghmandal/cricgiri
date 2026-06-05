"""
analytics/trajectory_physics.py
===============================
Physics-based cricket ball trajectory: gravity, bounce, swing, future path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PhysicsTrajectoryResult:
    """Parabolic fit + optional bounce + predicted future points."""
    fitted_pixels: List[Tuple[float, float]] = field(default_factory=list)
    future_pixels: List[Tuple[float, float]] = field(default_factory=list)
    bounce_index: Optional[int] = None
    release_velocity_ms: Tuple[float, float] = (0.0, 0.0)
    horizontal_deviation_m: float = 0.0
    gravity_m_s2: float = 9.81


def _parabolic_y(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x * x + b * x + c


def fit_parabolic_path(
    points: List[Tuple[float, float]],
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Fit y = ax² + bx + c in pixel space (screen coordinates)."""
    if len(points) < 4:
        return np.array([0.0, 0.0, 0.0]), points

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    try:
        coef = np.polyfit(xs, ys, 2)
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0, 0.0]), points

    ys_fit = np.polyval(coef, xs)
    fitted = [(float(xs[i]), float(ys_fit[i])) for i in range(len(xs))]
    return coef, fitted


def detect_bounce_index(
    points: List[Tuple[float, float]],
) -> Optional[int]:
    """Bounce ≈ local maximum screen-Y (ball lowest on pitch in bowler→batsman view)."""
    if len(points) < 5:
        return None
    ys = [p[1] for p in points]
    if ys[-1] < ys[0] - 5:
        return None  # upward camera
    margin = max(1, len(ys) // 8)
    segment = ys[margin:-margin] if len(ys) > 2 * margin else ys
    if not segment:
        return None
    local = int(np.argmax(segment))
    return margin + local


def predict_future_trajectory(
    points: List[Tuple[float, float]],
    fps: float = 30.0,
    pixels_per_meter: float = 38.0,
    n_future: int = 8,
    post_bounce_energy: float = 0.72,
) -> List[Tuple[float, float]]:
    """
    Extrapolate ball path using last velocity + gravity in pixel space.

    Used for hypothetical post-release / no-bat continuation overlays.
    """
    if len(points) < 3 or fps <= 0:
        return []

    p2, p1, p0 = points[-3], points[-2], points[-1]
    dt = 1.0 / fps
    vx = (p0[0] - p1[0]) / dt
    vy = (p0[1] - p1[1]) / dt
    ax, ay = 0.0, 0.0
    if len(points) >= 4:
        p3 = points[-4]
        vx_prev = (p1[0] - p2[0]) / dt
        vy_prev = (p1[1] - p2[1]) / dt
        ax = (vx - vx_prev) / dt
        ay = (vy - vy_prev) / dt

    g_px = (9.81 / max(pixels_per_meter, 1e-6)) * dt * dt
    x, y = p0[0], p0[1]
    out: List[Tuple[float, float]] = []
    bounce_i = detect_bounce_index(points)
    bounced = bounce_i is not None and bounce_i >= len(points) - 3

    for _ in range(n_future):
        vx += ax * dt
        vy += ay * dt + g_px
        if bounced:
            vy *= -post_bounce_energy
            bounced = False
        x += vx * dt
        y += vy * dt
        out.append((float(x), float(y)))
    return out


def estimate_swing_direction(
    world_points: List[Tuple[float, float, float]],
) -> str:
    """Horizontal deviation sign → inswing/outswing/none."""
    if len(world_points) < 4:
        return "none"
    xs = [p[0] for p in world_points]
    dev = xs[-1] - xs[0]
    if abs(dev) < 0.03:
        return "none"
    return "inswing" if dev < 0 else "outswing"
