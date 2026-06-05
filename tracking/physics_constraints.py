"""
Physics and geometry constraints for cricket ball detections / track points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tracking.track_ball import Detection


@dataclass
class PhysicsConstraintConfig:
    """Pixel-space limits tuned for ~30 fps broadcast / net footage."""

    max_speed_px_per_frame: float = 95.0
    max_accel_px_per_frame2: float = 45.0
    min_box_px: float = 2.5
    max_box_px: float = 56.0
    min_aspect: float = 0.28
    max_aspect: float = 3.2
    min_area_px: float = 12.0
    max_area_px: float = 2800.0


class PhysicsMotionFilter:
    """Reject impossible boxes and inconsistent motion jumps."""

    def __init__(self, cfg: Optional[PhysicsConstraintConfig] = None) -> None:
        self.cfg = cfg or PhysicsConstraintConfig()
        self._last: Optional[Tuple[float, float, float, float, float]] = None
        # x, y, vx, vy, frame

    def reset(self) -> None:
        self._last = None

    def valid_geometry(self, det: Detection) -> bool:
        c = self.cfg
        if det.w < c.min_box_px or det.h < c.min_box_px:
            return False
        if det.w > c.max_box_px or det.h > c.max_box_px:
            return False
        aspect = det.w / max(det.h, 1e-6)
        if aspect < c.min_aspect or aspect > c.max_aspect:
            return False
        area = det.w * det.h
        if area < c.min_area_px or area > c.max_area_px:
            return False
        return True

    def valid_motion(
        self,
        cx: float,
        cy: float,
        frame_idx: int,
        *,
        scale_gap: int = 1,
    ) -> bool:
        c = self.cfg
        gap = max(1, int(scale_gap))
        if self._last is None:
            return True

        lx, ly, lvx, lvy, lfi = self._last
        dt = max(1, frame_idx - lfi)
        if dt <= 0:
            return False

        vx = (cx - lx) / dt
        vy = (cy - ly) / dt
        speed = math.hypot(vx, vy)
        if speed > c.max_speed_px_per_frame * gap:
            return False

        ax = (vx - lvx) / dt
        ay = (vy - lvy) / dt
        if math.hypot(ax, ay) > c.max_accel_px_per_frame2 * gap:
            return False

        return True

    def observe(self, cx: float, cy: float, frame_idx: int) -> None:
        if self._last is None:
            vx = vy = 0.0
        else:
            lx, ly, _, _, lfi = self._last
            dt = max(1, frame_idx - lfi)
            vx = (cx - lx) / dt
            vy = (cy - ly) / dt
        self._last = (float(cx), float(cy), float(vx), float(vy), int(frame_idx))

    def filter_detections(
        self,
        detections: List[Detection],
        frame_idx: int,
        *,
        predicted_xy: Optional[Tuple[float, float]] = None,
        max_radial_from_pred: float = 120.0,
    ) -> List[Detection]:
        out: List[Detection] = []
        for det in detections:
            if not self.valid_geometry(det):
                continue
            if predicted_xy is not None:
                if math.hypot(det.cx - predicted_xy[0], det.cy - predicted_xy[1]) > max_radial_from_pred:
                    continue
            if not self.valid_motion(det.cx, det.cy, frame_idx):
                continue
            out.append(det)
        if out:
            best = max(out, key=lambda d: d.conf)
            self.observe(best.cx, best.cy, frame_idx)
        return out
