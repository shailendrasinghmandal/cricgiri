"""
observed_path.py
================
Real ball trajectory from YOLO detections only (release → bat impact).

No polynomial extrapolation, splines, Bezier prediction, or future paths.
Kalman coast points are never used as trajectory knots — only detector centers.
Short linear bridges connect two real detections when the model misses 1–N frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from analytics.visualizer import prepare_observed_trajectory_points


@dataclass
class RejectedObservation:
    frame_idx: int
    x: float
    y: float
    confidence: float
    reason: str


@dataclass
class ObservedPathResult:
    """Output of the real-path builder for one delivery."""

    raw_pixels: List[Tuple[float, float]]
    filtered_pixels: List[Tuple[float, float]]
    smooth_pixels: List[Tuple[float, float]]
    frame_indices: List[int]
    confidences: List[float]
    velocities: List[Tuple[float, float]]
    rejected: List[RejectedObservation] = field(default_factory=list)
    mean_confidence: float = 0.0
    bridge_points_inserted: int = 0

    @property
    def rejected_pixels(self) -> List[Tuple[float, float, str]]:
        return [(r.x, r.y, r.reason) for r in self.rejected]


def _pick_nearest_to_prediction(
    cands: List[Tuple[float, float, float]],
    pred_x: float,
    pred_y: float,
    max_dist_px: float,
) -> Optional[Tuple[float, float, float]]:
    best: Optional[Tuple[float, float, float]] = None
    best_d = float("inf")
    for cx, cy, conf in cands:
        d = math.hypot(cx - pred_x, cy - pred_y)
        if d > max_dist_px:
            continue
        score = d - conf * 20.0
        if score < best_d:
            best_d = score
            best = (cx, cy, conf)
    return best


def _velocity_px_per_frame(
    path: List[Tuple[int, float, float, float]],
) -> Tuple[float, float]:
    if len(path) < 2:
        return (0.0, 0.0)
    f0, x0, y0, _ = path[-2]
    f1, x1, y1, _ = path[-1]
    dt = max(1, f1 - f0)
    return ((x1 - x0) / dt, (y1 - y0) / dt)


class RealObservedPathBuilder:
    """
    ByteTrack-style nearest-neighbour association over per-frame YOLO centers,
    with motion / confidence outlier rejection and bat-impact truncation.
    """

    def __init__(
        self,
        min_confidence: float = 0.03,
        max_step_px: float = 95.0,
        max_bridge_gap_frames: int = 12,
        min_confidence_primary: float = 0.08,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_step_px = max_step_px
        self.max_bridge_gap_frames = max_bridge_gap_frames
        self.min_confidence_primary = min_confidence_primary

    def build(
        self,
        frame_start: int,
        frame_end: int,
        raw_ball_cache: Dict[int, List[Tuple[float, float, float]]],
        track_points: Optional[List[dict]] = None,
    ) -> ObservedPathResult:
        rejected: List[RejectedObservation] = []
        associated: List[Tuple[int, float, float, float]] = []

        # Seed track real centers when cache is empty for a frame.
        track_by_frame: Dict[int, Tuple[float, float, float]] = {}
        if track_points:
            for p in track_points:
                if p.get("is_interpolated", False):
                    continue
                conf = float(p.get("confidence", 0.0))
                if conf <= 0.0:
                    continue
                fi = int(p.get("frame_idx", -1))
                if fi < 0:
                    continue
                track_by_frame[fi] = (float(p["x"]), float(p["y"]), conf)

        for fi in range(int(frame_start), int(frame_end) + 1):
            cands = list(raw_ball_cache.get(fi, []))
            if not cands and fi in track_by_frame:
                tx, ty, tc = track_by_frame[fi]
                cands = [(tx, ty, tc)]

            if not cands:
                continue

            for cx, cy, conf in cands:
                if conf < self.min_confidence:
                    rejected.append(RejectedObservation(
                        fi, float(cx), float(cy), float(conf), "low_confidence",
                    ))

            viable = [
                (cx, cy, conf) for cx, cy, conf in cands
                if conf >= self.min_confidence
            ]
            if not viable:
                continue

            if not associated:
                best = max(viable, key=lambda c: c[2])
                associated.append((fi, float(best[0]), float(best[1]), float(best[2])))
                continue

            vx, vy = _velocity_px_per_frame(associated)
            last_fi, lx, ly, _ = associated[-1]
            gap = max(1, fi - last_fi)
            pred_x = lx + vx * gap
            pred_y = ly + vy * gap
            tol = min(self.max_step_px * gap, 50.0 + 22.0 * gap)

            pick = _pick_nearest_to_prediction(viable, pred_x, pred_y, tol)
            if pick is None:
                for cx, cy, conf in viable:
                    d = math.hypot(cx - lx, cy - ly)
                    if d > tol:
                        rejected.append(RejectedObservation(
                            fi, float(cx), float(cy), float(conf), "unrealistic_jump",
                        ))
                continue

            cx, cy, conf = pick
            step = math.hypot(cx - lx, cy - ly)
            if step > tol:
                rejected.append(RejectedObservation(
                    fi, float(cx), float(cy), float(conf), "unrealistic_jump",
                ))
                continue

            # Do not reject direction reversals — natural at bounce and seam movement.

            associated.append((fi, float(cx), float(cy), float(conf)))

        if len(associated) < 2:
            raw = [(x, y) for _, x, y, _ in associated]
            smooth = prepare_observed_trajectory_points(raw) if raw else []
            confs = [c for _, _, _, c in associated]
            return ObservedPathResult(
                raw_pixels=raw,
                filtered_pixels=raw,
                smooth_pixels=smooth,
                frame_indices=[f for f, _, _, _ in associated],
                confidences=confs,
                velocities=[],
                rejected=rejected,
                mean_confidence=float(sum(confs) / len(confs)) if confs else 0.0,
            )

        filtered: List[Tuple[int, float, float, float]] = [associated[0]]
        for pt in associated[1:]:
            fi, cx, cy, conf = pt
            lf, lx, ly, _ = filtered[-1]
            gap = fi - lf
            step = math.hypot(cx - lx, cy - ly)
            if conf < self.min_confidence_primary and step > self.max_step_px * 0.65:
                rejected.append(RejectedObservation(
                    fi, cx, cy, conf, "low_confidence",
                ))
                continue
            if step > self.max_step_px * max(1, gap):
                rejected.append(RejectedObservation(
                    fi, cx, cy, conf, "unrealistic_jump",
                ))
                continue
            filtered.append(pt)

        knot_pixels = [(x, y) for _, x, y, _ in filtered]

        raw_pixels: List[Tuple[float, float]] = []
        frame_indices: List[int] = []
        confidences: List[float] = []
        bridge_inserted = 0

        for k, (fi, cx, cy, conf) in enumerate(filtered):
            if k == 0:
                raw_pixels.append((cx, cy))
                frame_indices.append(fi)
                confidences.append(conf)
                continue

            prev_fi, prev_x, prev_y, _ = filtered[k - 1]
            gap = fi - prev_fi
            if 1 < gap <= self.max_bridge_gap_frames + 1:
                for step in range(1, gap):
                    t = step / float(gap)
                    bx = prev_x + (cx - prev_x) * t
                    by = prev_y + (cy - prev_y) * t
                    raw_pixels.append((float(bx), float(by)))
                    frame_indices.append(int(prev_fi + step))
                    confidences.append(0.0)
                    bridge_inserted += 1
            raw_pixels.append((cx, cy))
            frame_indices.append(fi)
            confidences.append(conf)

        # Split at non-physical jumps and keep strongest continuous segment.
        if len(raw_pixels) >= 4:
            segments: List[Tuple[int, int]] = []
            start = 0
            for i in range(1, len(raw_pixels)):
                step = math.hypot(
                    raw_pixels[i][0] - raw_pixels[i - 1][0],
                    raw_pixels[i][1] - raw_pixels[i - 1][1],
                )
                # Hard break for implausible segment jumps.
                # Keep strict so re-tracked later objects do not join main flight.
                if step > 60.0:
                    if (i - start) >= 2:
                        segments.append((start, i))
                    start = i
            if (len(raw_pixels) - start) >= 2:
                segments.append((start, len(raw_pixels)))

            if len(segments) > 1:
                def _seg_score(a: int, b: int) -> float:
                    seg = raw_pixels[a:b]
                    disp = math.hypot(seg[-1][0] - seg[0][0], seg[-1][1] - seg[0][1])
                    return float((b - a) * 40.0 + disp)

                best_a, best_b = max(segments, key=lambda s: _seg_score(s[0], s[1]))
                raw_pixels = raw_pixels[best_a:best_b]
                frame_indices = frame_indices[best_a:best_b]
                confidences = confidences[best_a:best_b]

        # Remove sharp local spikes while preserving bounce as a smooth bend.
        if len(raw_pixels) >= 5:
            cleaned: List[Tuple[float, float]] = [raw_pixels[0], raw_pixels[1]]
            cleaned_fi: List[int] = [frame_indices[0], frame_indices[1]]
            cleaned_cf: List[float] = [confidences[0], confidences[1]]
            for i in range(2, len(raw_pixels) - 1):
                p0 = np.asarray(cleaned[-2], dtype=np.float64)
                p1 = np.asarray(cleaned[-1], dtype=np.float64)
                p2 = np.asarray(raw_pixels[i], dtype=np.float64)
                p3 = np.asarray(raw_pixels[i + 1], dtype=np.float64)
                v1 = p1 - p0
                v2 = p2 - p1
                v3 = p3 - p2
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                n3 = float(np.linalg.norm(v3))
                drop = False
                if n1 > 2.0 and n2 > 2.0:
                    cos12 = float(np.dot(v1, v2) / (n1 * n2))
                    # Very sharp reversal + large local step => spike/outlier.
                    if cos12 < -0.70 and n2 > 18.0:
                        drop = True
                if not drop and n2 > 2.0 and n3 > 2.0:
                    cos23 = float(np.dot(v2, v3) / (n2 * n3))
                    if cos23 < -0.70 and n2 > 18.0:
                        drop = True
                if drop:
                    rejected.append(RejectedObservation(
                        int(frame_indices[i]),
                        float(raw_pixels[i][0]),
                        float(raw_pixels[i][1]),
                        float(confidences[i]) if i < len(confidences) else 0.0,
                        "bad_motion_direction",
                    ))
                    continue
                cleaned.append(raw_pixels[i])
                cleaned_fi.append(frame_indices[i])
                cleaned_cf.append(confidences[i])
            cleaned.append(raw_pixels[-1])
            cleaned_fi.append(frame_indices[-1])
            cleaned_cf.append(confidences[-1])
            raw_pixels = cleaned
            frame_indices = cleaned_fi
            confidences = cleaned_cf

        velocities: List[Tuple[float, float]] = []
        for i in range(len(raw_pixels)):
            if i == 0:
                velocities.append((0.0, 0.0))
            else:
                velocities.append((
                    raw_pixels[i][0] - raw_pixels[i - 1][0],
                    raw_pixels[i][1] - raw_pixels[i - 1][1],
                ))

        smooth_pixels = prepare_observed_trajectory_points(raw_pixels)
        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        return ObservedPathResult(
            raw_pixels=knot_pixels,
            filtered_pixels=raw_pixels,
            smooth_pixels=smooth_pixels,
            frame_indices=frame_indices,
            confidences=confidences,
            velocities=velocities,
            rejected=rejected,
            mean_confidence=mean_conf,
            bridge_points_inserted=bridge_inserted,
        )
