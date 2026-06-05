"""
Release and bat-impact phase detection for delivery trajectories.

Analyzes tracked points first, then downstream code draws the path only
from bowler release (ball leaves the hand) through bat contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from analytics.real_trajectory import TrajectoryPoint
from tracking.track_types import TrackPoint


@dataclass
class FlightPhases:
    """Analyzed delivery window for path rendering."""

    release_frame: int
    bat_impact_frame: int
    release_index: int
    bat_impact_index: int
    bounce_frame: Optional[int] = None

    @property
    def valid(self) -> bool:
        return (
            self.bat_impact_index >= self.release_index
            and self.bat_impact_frame >= self.release_frame
        )


def _to_trajectory_points(
    points: Sequence[Union[TrackPoint, TrajectoryPoint]],
) -> List[TrajectoryPoint]:
    out: List[TrajectoryPoint] = []
    for p in points:
        if isinstance(p, TrajectoryPoint):
            out.append(p)
        else:
            out.append(TrajectoryPoint(
                int(p.frame_idx),
                float(p.x),
                float(p.y),
                float(p.confidence),
                bool(p.is_interpolated),
                "tracker",
            ))
    return sorted(out, key=lambda q: q.frame_idx)


def find_release_index(
    points: List[TrajectoryPoint],
    min_step_px: float = 10.0,
) -> int:
    """
    First sustained outbound motion = ball left the bowler's hand.

    Drops static hand/arm detections before the ball accelerates down the pitch.
    """
    if len(points) < 2:
        return 0

    det_rows = [
        (p.frame_idx, p.x, p.y, max(p.confidence, 0.01))
        for p in points
        if not p.is_interpolated and p.confidence > 0
    ]
    if len(det_rows) < 2:
        det_rows = [(p.frame_idx, p.x, p.y, 0.05) for p in points]

    from analytics.real_trajectory import RealBallPathBuilder

    trimmed = RealBallPathBuilder._trim_to_release_start(det_rows, min_step_px=min_step_px)
    release_frame = int(trimmed[0][0])
    for i, p in enumerate(points):
        if p.frame_idx >= release_frame:
            return i
    return 0


def find_bat_impact_index(
    points: List[TrajectoryPoint],
    release_index: int = 0,
    post_bat_retreat_px: float = 20.0,
) -> int:
    """
    Last frame still on the incoming path toward the bat (before post-hit retreat
    or a hard mis-track jump).
    """
    n = len(points)
    if n < 2:
        return max(0, n - 1)

    pts = points[release_index:]
    if len(pts) < 2:
        return max(0, n - 1)

    ys = [p.y for p in pts]
    downward = ys[-1] >= ys[0] - 5.0
    local_n = len(pts)

    monotonic_toward_bat = all(
        ys[i + 1] >= ys[i] - 2.0 for i in range(local_n - 1)
    ) if local_n >= 3 else False
    if monotonic_toward_bat:
        for i in range(2, local_n):
            p = pts[i]
            if p.is_interpolated or p.confidence <= 0.0:
                continue
            if p.confidence < 0.22:
                return release_index + max(1, i - 1)
        for i in range(local_n - 1, 0, -1):
            p = pts[i]
            if not p.is_interpolated and p.confidence >= 0.22:
                return release_index + i
        return release_index + local_n - 1

    mid_lo = max(1, int(local_n * 0.12))
    mid_hi = max(mid_lo + 2, int(local_n * 0.70))
    mid_seg = ys[mid_lo:mid_hi]
    if len(mid_seg) >= 2:
        bounce_local = mid_lo + (
            int(np.argmax(mid_seg)) if downward else int(np.argmin(mid_seg))
        )
    else:
        bounce_local = int(np.argmax(ys) if downward else np.argmin(ys))

    for i in range(bounce_local + 2, local_n):
        p = pts[i]
        if p.is_interpolated or p.confidence <= 0.0:
            continue
        if p.confidence < 0.22:
            return release_index + max(bounce_local + 1, i)

    after_bounce = pts[bounce_local:]
    if len(after_bounce) >= 2:
        ab_ys = [p.y for p in after_bounce]
        peak_off = int(np.argmax(ab_ys) if downward else np.argmin(ab_ys))
        peak_local = bounce_local + peak_off
    else:
        peak_local = bounce_local

    peak_y = ys[peak_local]
    end_local = peak_local

    for i in range(peak_local + 1, local_n):
        y = ys[i]
        prev_y = ys[i - 1]
        if downward:
            if y >= prev_y - 4.0 and y >= peak_y - 12.0:
                end_local = i
                peak_y = max(peak_y, y)
            else:
                break
        else:
            if y <= prev_y + 4.0 and y <= peak_y + 12.0:
                end_local = i
                peak_y = min(peak_y, y)
            else:
                break

    for i in range(peak_local, local_n):
        y = ys[i]
        if downward and y < peak_y - post_bat_retreat_px:
            end_local = min(end_local, max(bounce_local + 1, i - 1))
            break
        if not downward and y > peak_y + post_bat_retreat_px:
            end_local = min(end_local, max(bounce_local + 1, i - 1))
            break

    steps = [
        math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)
        for i in range(1, local_n)
    ]
    med = max(4.0, float(np.median(steps)) if steps else 10.0)
    search_start = max(bounce_local + 3, int(local_n * 0.55))

    for i in range(search_start, local_n):
        if i < 1:
            continue
        step = steps[i - 1]
        dy = pts[i].y - pts[i - 1].y
        dys = [pts[k].y - pts[k - 1].y for k in range(max(1, i - 3), i)]
        med_ad = max(2.0, float(np.median([abs(d) for d in dys])) if dys else 0.0)

        if step > max(65.0, med * 4.5) and abs(dy) > max(35.0, med_ad * 3.0):
            end_local = max(bounce_local + 1, i - 1)
            break

        if i >= max(bounce_local + 2, int(local_n * 0.65)):
            if downward and dy < -max(12.0, med_ad * 2.0) and pts[i].y < peak_y - post_bat_retreat_px * 0.5:
                end_local = max(bounce_local + 1, i - 1)
                break
            if not downward and dy > max(12.0, med_ad * 2.0) and pts[i].y > peak_y + post_bat_retreat_px * 0.5:
                end_local = max(bounce_local + 1, i - 1)
                break

    return release_index + end_local


def analyze_flight_phases(
    points: Sequence[Union[TrackPoint, TrajectoryPoint]],
    post_bat_retreat_px: float = 20.0,
    min_release_step_px: float = 10.0,
) -> FlightPhases:
    """Analyze release and bat-impact frames before building the visible path."""
    traj = _to_trajectory_points(points)
    if not traj:
        return FlightPhases(0, 0, 0, 0, None)

    release_index = find_release_index(traj, min_step_px=min_release_step_px)
    bat_index = find_bat_impact_index(
        traj,
        release_index=release_index,
        post_bat_retreat_px=post_bat_retreat_px,
    )
    bat_index = max(release_index, min(bat_index, len(traj) - 1))

    bounce_frame: Optional[int] = None
    if bat_index > release_index:
        seg = traj[release_index: bat_index + 1]
        ys = [p.y for p in seg]
        downward = ys[-1] >= ys[0] - 5.0
        margin = max(1, len(seg) // 8)
        if len(seg) > 2 * margin:
            interior = ys[margin:-margin]
            bounce_local = margin + (
                int(np.argmax(interior)) if downward else int(np.argmin(interior))
            )
            bounce_frame = seg[bounce_local].frame_idx

    return FlightPhases(
        release_frame=int(traj[release_index].frame_idx),
        bat_impact_frame=int(traj[bat_index].frame_idx),
        release_index=int(release_index),
        bat_impact_index=int(bat_index),
        bounce_frame=bounce_frame,
    )


def slice_track_to_phases(
    points: List[TrackPoint],
    phases: FlightPhases,
) -> List[TrackPoint]:
    """Keep only release → bat contact inclusive."""
    if not points or not phases.valid:
        return points
    return points[phases.release_index: phases.bat_impact_index + 1]
