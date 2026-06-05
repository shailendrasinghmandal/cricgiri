"""
tracking/byte_associate.py
==========================
ByteTrack-style two-stage detection association for single cricket ball track.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from tracking.pro_kalman import ProBallKalman, build_association_costs, hungarian_pick_best
from tracking.detection import Detection


def split_high_low(
    detections: List[Detection],
    high_thresh: float,
    low_thresh: float,
) -> Tuple[List[Detection], List[Detection]]:
    """ByteTrack split: high-confidence first, then low-confidence recovery pool."""
    high = [d for d in detections if d.conf >= high_thresh]
    low = [
        d for d in detections
        if low_thresh <= d.conf < high_thresh
    ]
    return high, low


def associate_detections(
    kf: ProBallKalman,
    detections: List[Detection],
    *,
    high_thresh: float,
    low_thresh: float,
    max_step_px: float,
    last_pos: Optional[Tuple[float, float]],
    last_frame_idx: int,
    frame_idx: int,
) -> Optional[Detection]:
    """
    Two-stage match: try high-conf detections, then low-conf near prediction.
    Returns best gated detection or None.
    """
    if not detections:
        return None

    if not kf.initialized:
        boot = sorted(detections, key=lambda d: (-d.conf, -d.cy))
        return boot[0]

    high, low = split_high_low(detections, high_thresh, low_thresh)
    gap = max(1, frame_idx - last_frame_idx) if last_frame_idx is not None else 1
    max_step = max(max_step_px * gap, 40.0 + 28.0 * gap)

    for pool in (high, low):
        if not pool:
            continue
        gated = [d for d in pool if kf.gated(d.cx, d.cy)]
        if not gated:
            continue
        if last_pos is not None:
            gated = [
                d for d in gated
                if math.hypot(d.cx - last_pos[0], d.cy - last_pos[1]) <= max_step * 1.4
            ]
        if not gated:
            continue
        tuples = [(d.cx, d.cy, d.conf) for d in gated]
        cost = build_association_costs(tuples, kf)
        _, col = hungarian_pick_best(cost)
        if col < 0 or col >= len(gated) or cost[0, col] >= 1e5:
            continue
        return gated[col]
    return None
