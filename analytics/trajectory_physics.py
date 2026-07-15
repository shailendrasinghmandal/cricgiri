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


def _fit_velocity_accel(
    points: List[Tuple[float, float]],
    dt: float,
    window: int = 6,
) -> Tuple[float, float, float, float]:
    """
    Robust velocity / acceleration estimate from the tail of the path.

    Fits a low-order polynomial to the last ``window`` points in each axis and
    reads its derivative at the final sample, instead of a noisy 3-point finite
    difference. Returns (vx, vy, ax, ay) in px/s and px/s².
    """
    n = len(points)
    w = max(3, min(window, n))
    tail = points[-w:]
    t = np.arange(w, dtype=np.float64) * dt
    xs = np.array([p[0] for p in tail], dtype=np.float64)
    ys = np.array([p[1] for p in tail], dtype=np.float64)

    # Quadratic fit gives velocity + (constant) acceleration; fall back to linear.
    deg = 2 if w >= 4 else 1
    try:
        cx = np.polyfit(t, xs, deg)
        cy = np.polyfit(t, ys, deg)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 0.0, 0.0, 0.0

    t_end = t[-1]
    if deg == 2:
        vx = 2 * cx[0] * t_end + cx[1]
        vy = 2 * cy[0] * t_end + cy[1]
        ax = 2 * cx[0]
        ay = 2 * cy[0]
    else:
        vx, vy = cx[0], cy[0]
        ax = ay = 0.0
    return float(vx), float(vy), float(ax), float(ay)


def predict_future_trajectory_v2(
    points: List[Tuple[float, float]],
    fps: float = 30.0,
    pixels_per_meter: float = 38.0,
    n_future: int = 12,
    ground_y: Optional[float] = None,
    restitution: float = 0.55,
    horizontal_friction: float = 0.82,
    use_fitted_gravity: bool = True,
) -> List[Tuple[float, float]]:
    """
    Improved physics-based future extrapolation (pixel/screen space).

    Improvements over :func:`predict_future_trajectory`:
      * velocity/acceleration are read from a polynomial fit of the path tail
        (robust to jitter) rather than a single 3-point difference;
      * vertical acceleration can be taken from the *observed* motion
        (``use_fitted_gravity``) so swing/dip is honoured, clamped to a sane
        gravity band so noise can't invert the arc;
      * a real bounce model: when the ball reaches the pitch plane (``ground_y``,
        defaulting to the lowest observed screen-Y), vertical velocity reflects
        with ``restitution`` and horizontal velocity is scaled by
        ``horizontal_friction`` — and it can bounce more than once;
      * post-bounce continuation keeps producing a physically plausible arc
        instead of a single hard flip.

    Returns a list of future (x, y) pixel points (length ``n_future``).
    """
    if len(points) < 3 or fps <= 0:
        return []

    dt = 1.0 / fps
    vx, vy, ax_obs, ay_obs = _fit_velocity_accel(points, dt)

    # Gravity in pixel space (downward = +y on screen).
    g_px = (9.81 / max(pixels_per_meter, 1e-6))
    if use_fitted_gravity:
        # Trust the observed vertical acceleration but clamp to [0.4g, 2.5g]
        # so a noisy fit can't flip the ball upward or fling it down.
        ay = float(np.clip(ay_obs, 0.4 * g_px, 2.5 * g_px))
    else:
        ay = g_px
    ax = 0.0  # horizontal accel assumed ~0 (swing already baked into vx via fit)

    # Pitch plane: lowest point the ball has reached on screen, unless given.
    if ground_y is None:
        ground_y = max(p[1] for p in points)

    x, y = points[-1][0], points[-1][1]
    out: List[Tuple[float, float]] = []
    for _ in range(max(1, int(n_future))):
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        # Bounce when crossing the pitch plane while descending.
        if y >= ground_y and vy > 0:
            y = ground_y - (y - ground_y)  # reflect position above the plane
            vy = -abs(vy) * restitution
            vx *= horizontal_friction
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
