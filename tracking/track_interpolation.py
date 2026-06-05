"""
tracking/track_interpolation.py
===============================
Gap interpolation for missed detection frames (linear / cubic / spline).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from tracking.track_types import TrackPoint


def interpolate_track_gaps(
    points: List[TrackPoint],
    method: str = "cubic",
) -> List[TrackPoint]:
    """
    Fill missing frame indices between existing track points.

    Detected → Missing → Missing → Detected becomes smooth interpolated path.
    """
    if len(points) < 2:
        return points

    by_frame = {p.frame_idx: p for p in points}
    frames = sorted(by_frame.keys())
    filled: List[TrackPoint] = []

    for i in range(len(frames)):
        fi = frames[i]
        filled.append(by_frame[fi])
        if i + 1 >= len(frames):
            break

        fj = frames[i + 1]
        gap = fj - fi
        if gap <= 1:
            continue

        xs = np.array([by_frame[fi].x, by_frame[fj].x], dtype=np.float64)
        ys = np.array([by_frame[fi].y, by_frame[fj].y], dtype=np.float64)
        t_knots = np.array([0.0, 1.0], dtype=np.float64)

        for step in range(1, gap):
            t = step / float(gap)
            if method == "spline" and gap >= 4:
                # Use 4-point spline when enough context exists.
                ctx_frames = frames[max(0, i - 1) : min(len(frames), i + 3)]
                if len(ctx_frames) >= 3:
                    tx = np.array([float(f - fi) / gap for f in ctx_frames])
                    px = np.array([by_frame[f].x for f in ctx_frames])
                    py = np.array([by_frame[f].y for f in ctx_frames])
                    bx = float(np.interp(t, tx, px))
                    by = float(np.interp(t, tx, py))
                else:
                    bx = float(xs[0] + (xs[1] - xs[0]) * t)
                    by = float(ys[0] + (ys[1] - ys[0]) * t)
            elif method == "cubic" and gap >= 3:
                # Hermite-style cubic ease between endpoints.
                t2, t3 = t * t, t * t * t
                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2
                vx0 = (xs[1] - xs[0])
                vy0 = (ys[1] - ys[0])
                bx = h00 * xs[0] + h10 * vx0 + h01 * xs[1]
                by = h00 * ys[0] + h10 * vy0 + h01 * ys[1]
            else:
                bx = float(xs[0] + (xs[1] - xs[0]) * t)
                by = float(ys[0] + (ys[1] - ys[0]) * t)

            filled.append(TrackPoint(
                frame_idx=fi + step,
                x=bx,
                y=by,
                vx=0.0,
                vy=0.0,
                is_interpolated=True,
                confidence=0.0,
            ))

    return sorted(filled, key=lambda p: p.frame_idx)
