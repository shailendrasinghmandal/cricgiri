"""
real_trajectory.py
==================
Real cricket ball path: release → bat impact only.

Pipeline:
  YOLO centers (+ tracker real centers) → nearest-neighbor association
  → outlier rejection → linear gap bridge → Kalman jitter stabilize
  → trajectory buffer → light EMA/MA → bat-impact stop → polylines
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from analytics.trajectory_smoothing import savgol_smooth_path
from tracking.track_ball import _BallKalman


@dataclass
class TrajectoryPoint:
    frame_idx: int
    x: float
    y: float
    confidence: float
    is_interpolated: bool = False
    source: str = "detection"


@dataclass
class RejectedPoint:
    frame_idx: int
    x: float
    y: float
    confidence: float
    reason: str


@dataclass
class RealTrajectoryResult:
    points: List[TrajectoryPoint] = field(default_factory=list)
    render_pixels: List[Tuple[float, float]] = field(default_factory=list)
    raw_pixels: List[Tuple[float, float]] = field(default_factory=list)
    filtered_pixels: List[Tuple[float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    rejected: List[RejectedPoint] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)
    mean_confidence: float = 0.0
    stopped_at_frame: Optional[int] = None
    bridge_points_inserted: int = 0
    release_frame: Optional[int] = None
    bat_impact_frame: Optional[int] = None
    bounce_frame: Optional[int] = None

    @property
    def rejected_pixels(self) -> List[Tuple[float, float, str]]:
        return [(r.x, r.y, r.reason) for r in self.rejected]


def _velocity_px_per_frame(
    path: List[Tuple[int, float, float, float]],
) -> Tuple[float, float]:
    if len(path) < 2:
        return (0.0, 0.0)
    f0, x0, y0, _ = path[-2]
    f1, x1, y1, _ = path[-1]
    dt = max(1, f1 - f0)
    return ((x1 - x0) / dt, (y1 - y0) / dt)


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


class RealBallPathBuilder:
    """Build visible trajectory from real detections only."""

    def __init__(
        self,
        min_confidence: float = 0.03,
        min_confidence_primary: float = 0.08,
        max_step_px: float = 88.0,
        max_bridge_gap_frames: int = 8,
        ema_alpha: float = 0.38,
        smooth_window: int = 3,
        render_max_upward_px: float = 12.0,
        post_bat_retreat_px: float = 20.0,
        savgol_window: int = 7,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_confidence_primary = min_confidence_primary
        self.max_step_px = max_step_px
        self.max_bridge_gap_frames = max_bridge_gap_frames
        self.ema_alpha = ema_alpha
        self.smooth_window = smooth_window
        self.render_max_upward_px = render_max_upward_px
        self.post_bat_retreat_px = post_bat_retreat_px
        self.savgol_window = savgol_window

    @staticmethod
    def _trim_to_release_start(
        filtered: List[Tuple[int, float, float, float]],
        min_step_px: float = 10.0,
        sustain_steps: int = 2,
    ) -> List[Tuple[int, float, float, float]]:
        """Drop static pre-release points (hand/arm) before sustained ball flight."""
        if len(filtered) < 3:
            return filtered

        start = 0
        for i in range(1, len(filtered)):
            step = math.hypot(
                filtered[i][1] - filtered[i - 1][1],
                filtered[i][2] - filtered[i - 1][2],
            )
            if step < min_step_px:
                continue
            ok = 1
            for j in range(i, min(len(filtered) - 1, i + sustain_steps)):
                s2 = math.hypot(
                    filtered[j + 1][1] - filtered[j][1],
                    filtered[j + 1][2] - filtered[j][2],
                )
                if s2 >= min_step_px * 0.65:
                    ok += 1
            if ok >= sustain_steps:
                start = max(0, i - 1)
                break

        while start < len(filtered) - 1:
            step = math.hypot(
                filtered[start + 1][1] - filtered[start][1],
                filtered[start + 1][2] - filtered[start][2],
            )
            if step >= min_step_px * 0.5:
                break
            start += 1

        return filtered[start:]

    def _extend_filtered_toward_bat(
        self,
        filtered: List[Tuple[int, float, float, float]],
        frame_end: int,
        raw_ball_cache: Dict[int, List[Tuple[float, float, float]]],
    ) -> List[Tuple[int, float, float, float]]:
        """Add post-bounce cache detections so the arc reaches the batsman."""
        if len(filtered) < 2:
            return filtered

        last_fi, lx, ly, _ = filtered[-1]
        downward = True
        if len(filtered) >= 2:
            downward = filtered[-1][2] >= filtered[0][2] - 5.0

        extended = list(filtered)
        anchor = (lx, ly)
        max_extra_frames = 28
        stall = 0

        for fi in range(int(last_fi) + 1, int(frame_end) + 1):
            if fi - last_fi > max_extra_frames:
                break
            cands = [
                (cx, cy, conf)
                for cx, cy, conf in (raw_ball_cache.get(fi) or [])
                if conf >= self.min_confidence
            ]
            if not cands:
                stall += 1
                if stall >= 6:
                    break
                continue
            stall = 0

            if downward:
                pick = max(cands, key=lambda c: (c[2], c[1]))
            else:
                pick = min(cands, key=lambda c: (c[2], -c[1]))

            cx, cy, conf = pick
            step = math.hypot(cx - anchor[0], cy - anchor[1])
            if step > self.max_step_px * max(1, fi - extended[-1][0]):
                continue
            if step < 2.0 and conf < 0.2:
                continue

            extended.append((fi, float(cx), float(cy), float(conf)))
            anchor = (cx, cy)

        return extended

    def build(
        self,
        frame_start: int,
        frame_end: int,
        raw_ball_cache: Dict[int, List[Tuple[float, float, float]]],
        fps: float = 30.0,
        track_points: Optional[List[dict]] = None,
    ) -> RealTrajectoryResult:
        rejected: List[RejectedPoint] = []
        associated: List[Tuple[int, float, float, float]] = []

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
                    rejected.append(RejectedPoint(
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
                    if math.hypot(cx - lx, cy - ly) > tol:
                        rejected.append(RejectedPoint(
                            fi, float(cx), float(cy), float(conf), "unrealistic_jump",
                        ))
                continue

            cx, cy, conf = pick
            if math.hypot(cx - lx, cy - ly) > tol:
                rejected.append(RejectedPoint(
                    fi, float(cx), float(cy), float(conf), "unrealistic_jump",
                ))
                continue

            associated.append((fi, float(cx), float(cy), float(conf)))

        if len(associated) < 2:
            return RealTrajectoryResult(rejected=rejected)

        # Second-pass outlier gate
        filtered: List[Tuple[int, float, float, float]] = [associated[0]]
        for pt in associated[1:]:
            fi, cx, cy, conf = pt
            lf, lx, ly, _ = filtered[-1]
            gap = max(1, fi - lf)
            step = math.hypot(cx - lx, cy - ly)
            if conf < self.min_confidence_primary and step > self.max_step_px * 0.65:
                rejected.append(RejectedPoint(fi, cx, cy, conf, "low_confidence"))
                continue
            if step > self.max_step_px * gap:
                rejected.append(RejectedPoint(fi, cx, cy, conf, "impossible_velocity"))
                continue
            filtered.append(pt)

        filtered = self._trim_to_release_start(filtered)
        if len(filtered) < 2:
            return RealTrajectoryResult(rejected=rejected)

        filtered = self._extend_filtered_toward_bat(
            filtered, int(frame_end), raw_ball_cache,
        )

        # Linear gap bridge between real knots (not extrapolation ahead of ball)
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
            max_bridge = self.max_bridge_gap_frames + 1
            if k >= max(1, len(filtered) // 2):
                max_bridge = max(max_bridge, 22)
            if 1 < gap <= max_bridge:
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

        # Keep strongest continuous segment if track breaks
        if len(raw_pixels) >= 4:
            segments: List[Tuple[int, int]] = []
            start = 0
            for i in range(1, len(raw_pixels)):
                step = math.hypot(
                    raw_pixels[i][0] - raw_pixels[i - 1][0],
                    raw_pixels[i][1] - raw_pixels[i - 1][1],
                )
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
                    y_span = abs(seg[-1][1] - seg[0][1])
                    max_y = max(p[1] for p in seg)
                    # Prefer the segment that reaches the batsman (deepest Y), not a short bowler-end stub.
                    return float((b - a) * 40.0 + disp + y_span * 4.0 + max_y * 2.5)

                best_a, best_b = max(segments, key=lambda s: _seg_score(s[0], s[1]))
                raw_pixels = raw_pixels[best_a:best_b]
                frame_indices = frame_indices[best_a:best_b]
                confidences = confidences[best_a:best_b]

        # Spike removal (direction mismatch)
        if len(raw_pixels) >= 5:
            cleaned_px: List[Tuple[float, float]] = [raw_pixels[0], raw_pixels[1]]
            cleaned_fi: List[int] = [frame_indices[0], frame_indices[1]]
            cleaned_cf: List[float] = [confidences[0], confidences[1]]
            for i in range(2, len(raw_pixels) - 1):
                p0 = np.asarray(cleaned_px[-2], dtype=np.float64)
                p1 = np.asarray(cleaned_px[-1], dtype=np.float64)
                p2 = np.asarray(raw_pixels[i], dtype=np.float64)
                p3 = np.asarray(raw_pixels[i + 1], dtype=np.float64)
                v1, v2, v3 = p1 - p0, p2 - p1, p3 - p2
                n1, n2, n3 = map(float, (np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)))
                drop = False
                if n1 > 2.0 and n2 > 2.0 and float(np.dot(v1, v2) / (n1 * n2)) < -0.70 and n2 > 18.0:
                    drop = True
                if not drop and n2 > 2.0 and n3 > 2.0 and float(np.dot(v2, v3) / (n2 * n3)) < -0.70 and n2 > 18.0:
                    drop = True
                if drop:
                    rejected.append(RejectedPoint(
                        int(frame_indices[i]), float(raw_pixels[i][0]), float(raw_pixels[i][1]),
                        float(confidences[i]) if i < len(confidences) else 0.0,
                        "direction_mismatch",
                    ))
                    continue
                cleaned_px.append(raw_pixels[i])
                cleaned_fi.append(frame_indices[i])
                cleaned_cf.append(confidences[i])
            cleaned_px.append(raw_pixels[-1])
            cleaned_fi.append(frame_indices[-1])
            cleaned_cf.append(confidences[-1])
            raw_pixels = cleaned_px
            frame_indices = cleaned_fi
            confidences = cleaned_cf

        points = [
            TrajectoryPoint(
                frame_indices[i], raw_pixels[i][0], raw_pixels[i][1],
                confidences[i] if i < len(confidences) else 0.0,
                confidences[i] <= 0.0, "bridge" if confidences[i] <= 0.0 else "detection",
            )
            for i in range(len(raw_pixels))
        ]

        knot_raw = [(x, y) for _, x, y, _ in filtered]
        points = self._truncate_at_bat_impact(points)
        frame_indices = [p.frame_idx for p in points]
        points = self._kalman_stabilize(points)
        render_px, velocities = self._light_smooth(points)

        confs = [p.confidence for p in points if p.confidence > 0]
        return RealTrajectoryResult(
            points=points,
            render_pixels=render_px,
            raw_pixels=knot_raw,
            filtered_pixels=[(p.x, p.y) for p in points if not p.is_interpolated],
            frame_indices=frame_indices,
            rejected=rejected,
            velocities=velocities,
            mean_confidence=float(sum(confs) / len(confs)) if confs else 0.0,
            stopped_at_frame=points[-1].frame_idx if points else None,
            bridge_points_inserted=bridge_inserted,
        )

    def _kalman_stabilize(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        if len(points) < 2:
            return points
        kf = _BallKalman()
        out: List[TrajectoryPoint] = []
        for p in points:
            if not kf.initialized:
                kf.init(p.x, p.y)
                out.append(p)
                continue
            kf.predict()
            x, y, _, _ = kf.correct(p.x, p.y)
            bx = 0.75 * p.x + 0.25 * x
            by = 0.75 * p.y + 0.25 * y
            out.append(TrajectoryPoint(
                p.frame_idx, bx, by, p.confidence, p.is_interpolated, "kalman_stabilized",
            ))
        return out

    def _reject_upward_spikes(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Drop segments that move up-screen when the ball should travel toward the batsman."""
        if len(points) < 4:
            return points

        ys = [p.y for p in points]
        downward = ys[-1] >= ys[0] - 5.0
        med_dy = float(np.median([ys[i] - ys[i - 1] for i in range(1, len(ys))]))
        toward_bat = med_dy >= -2.0 if downward else med_dy <= 2.0
        if not toward_bat:
            return points

        cleaned: List[TrajectoryPoint] = [points[0]]
        for i in range(1, len(points)):
            prev = cleaned[-1]
            cur = points[i]
            dy = cur.y - prev.y
            gap = max(1, cur.frame_idx - prev.frame_idx)
            dy_per = dy / gap

            if downward and dy_per < -self.render_max_upward_px:
                continue
            if not downward and dy_per > self.render_max_upward_px:
                continue
            cleaned.append(cur)

        return cleaned if len(cleaned) >= 2 else points

    def _bridge_real_knots(
        self, knots: List[TrajectoryPoint],
    ) -> List[TrajectoryPoint]:
        """Linear bridge only across short gaps; never invent upward detours."""
        if len(knots) < 2:
            return knots

        ys = [k.y for k in knots]
        downward = ys[-1] >= ys[0] - 5.0
        out: List[TrajectoryPoint] = [knots[0]]

        for nxt in knots[1:]:
            prev = out[-1]
            gap = nxt.frame_idx - prev.frame_idx
            if gap <= 1:
                out.append(nxt)
                continue
            if gap > self.max_bridge_gap_frames:
                out.append(nxt)
                continue

            dy = (nxt.y - prev.y) / max(1, gap)
            if downward and (dy < -self.render_max_upward_px * 0.5 or nxt.y < prev.y - self.post_bat_retreat_px):
                break
            if not downward and (dy > self.render_max_upward_px * 0.5 or nxt.y > prev.y + self.post_bat_retreat_px):
                break

            for step in range(1, gap):
                t = step / float(gap)
                bx = prev.x + (nxt.x - prev.x) * t
                by = prev.y + (nxt.y - prev.y) * t
                out.append(TrajectoryPoint(
                    int(prev.frame_idx + step),
                    float(bx), float(by), 0.0, True, "bridge",
                ))
            out.append(nxt)

        return out

    def _drop_bottom_phantom_points(
        self,
        points: List[TrajectoryPoint],
        frame_h: int = 0,
    ) -> List[TrajectoryPoint]:
        """Remove bottom-edge false detections (scoreboard / crease clutter)."""
        if len(points) < 3:
            return points

        ys = [p.y for p in points]
        y_span = max(ys) - min(ys)
        if y_span < 40.0:
            return points

        fh = max(480, int(frame_h or int(max(ys) + 80)))
        bottom_edge = fh * 0.82
        mid_pitch = [p for p in points if p.y < bottom_edge - 40.0]
        if not mid_pitch:
            return points

        cap_y = max(p.y for p in mid_pitch) + 35.0
        return [p for p in points if p.y <= cap_y]

    def _truncate_at_bat_zone(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        End arc at bat: keep release→deepest point toward batsman, drop post-hit retreat.
        """
        n = len(points)
        if n < 4:
            return points

        ys = [p.y for p in points]
        downward = ys[-1] >= ys[0] - 5.0
        flight_start = max(1, int(n * 0.30))
        tail = ys[flight_start:]
        if not tail:
            return points

        peak_i = flight_start + (
            int(np.argmax(tail)) if downward else int(np.argmin(tail))
        )
        peak_y = ys[peak_i]
        end_i = peak_i

        # Prefer ending on the last strong real detection near the bat.
        real_peak_i = peak_i
        for i, p in enumerate(points):
            if p.is_interpolated or p.confidence <= 0:
                continue
            if downward and p.y >= peak_y - self.post_bat_retreat_px:
                if p.y >= ys[real_peak_i] - 2.0:
                    real_peak_i = i
            elif not downward and p.y <= peak_y + self.post_bat_retreat_px:
                if p.y <= ys[real_peak_i] + 2.0:
                    real_peak_i = i
        end_i = max(end_i, real_peak_i)

        for i in range(peak_i + 1, n):
            y = ys[i]
            if downward:
                if y >= peak_y - 8.0:
                    end_i = i
                    peak_y = max(peak_y, y)
                elif y < peak_y - self.post_bat_retreat_px:
                    break
            else:
                if y <= peak_y + 8.0:
                    end_i = i
                    peak_y = min(peak_y, y)
                elif y > peak_y + self.post_bat_retreat_px:
                    break

        return points[: end_i + 1]

    def _truncate_at_bat_impact(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """End arc at bat contact — never cut at bounce, only late mis-track jumps."""
        points = self._truncate_at_bat_zone(points)
        n = len(points)
        if n < 5:
            return points

        ys = [p.y for p in points]
        downward = ys[-1] >= ys[0] - 5.0
        margin = max(1, n // 8)
        if margin > 0 and (n - 2 * margin) >= 3:
            interior = ys[margin:-margin]
            bounce_i = margin + (
                int(np.argmax(interior)) if downward else int(np.argmin(interior))
            )
        else:
            bounce_i = int(np.argmax(ys) if downward else np.argmin(ys))

        # Bat search only in the last ~15% after bounce (toward batsman).
        search_start = max(bounce_i + 10, int(n * 0.85))
        steps = [
            math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y)
            for i in range(1, n)
        ]
        med = max(4.0, float(np.median(steps)) if steps else 10.0)

        for i in range(search_start, n):
            if i < int(n * 0.78):
                continue
            step = steps[i - 1] if i > 0 else 0.0
            dy = points[i].y - points[i - 1].y
            dys = [points[k].y - points[k - 1].y for k in range(max(1, i - 3), i)]
            med_ad = max(2.0, float(np.median([abs(d) for d in dys])) if dys else 0.0)

            if step > max(70.0, med * 5.0) and abs(dy) > max(45.0, med_ad * 3.5):
                return points[: max(bounce_i + 4, i)]
        return points

    def _light_smooth(
        self, points: List[TrajectoryPoint],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        if not points:
            return [], []

        xs = np.array([p.x for p in points], dtype=np.float64)
        ys = np.array([p.y for p in points], dtype=np.float64)

        ex, ey = xs[0], ys[0]
        for i in range(1, len(xs)):
            ex = self.ema_alpha * xs[i] + (1.0 - self.ema_alpha) * ex
            ey = self.ema_alpha * ys[i] + (1.0 - self.ema_alpha) * ey
            xs[i], ys[i] = ex, ey

        w = max(3, self.smooth_window)
        if len(xs) >= w:
            kernel = np.ones(w) / w
            xs = np.convolve(xs, kernel, mode="same")
            ys = np.convolve(ys, kernel, mode="same")

        render = [(float(xs[i]), float(ys[i])) for i in range(len(xs))]
        if len(render) >= max(5, self.savgol_window):
            render = savgol_smooth_path(render, window=self.savgol_window, polyorder=2)
        velocities: List[Tuple[float, float]] = [(0.0, 0.0)]
        for i in range(1, len(render)):
            gap = max(1, points[i].frame_idx - points[i - 1].frame_idx)
            velocities.append((
                (render[i][0] - render[i - 1][0]) / gap,
                (render[i][1] - render[i - 1][1]) / gap,
            ))
        return render, velocities


    def _fill_gaps_from_cache(
        self,
        points: List[TrajectoryPoint],
        raw_ball_cache: Optional[Dict[int, List[Tuple[float, float, float]]]],
    ) -> List[TrajectoryPoint]:
        """Insert weak cache detections only into short gaps between trusted tracker knots."""
        if not raw_ball_cache or len(points) < 2:
            return points

        out: List[TrajectoryPoint] = [points[0]]
        for nxt in points[1:]:
            prev = out[-1]
            gap = int(nxt.frame_idx) - int(prev.frame_idx)
            if 1 < gap <= 6:
                anchor = (prev.x, prev.y)
                for fi in range(int(prev.frame_idx) + 1, int(nxt.frame_idx)):
                    cands = [
                        (cx, cy, conf)
                        for cx, cy, conf in (raw_ball_cache.get(fi) or [])
                        if conf >= max(0.06, self.min_confidence_primary * 0.72)
                    ]
                    if not cands:
                        continue
                    pick = min(
                        cands,
                        key=lambda c: math.hypot(c[0] - anchor[0], c[1] - anchor[1]) - c[2] * 8.0,
                    )
                    cx, cy, conf = pick
                    step = math.hypot(cx - anchor[0], cy - anchor[1])
                    if step > self.max_step_px * max(1, fi - int(prev.frame_idx)):
                        continue
                    if cy > anchor[1] + 25.0 and fi > int(prev.frame_idx) + 1:
                        continue
                    out.append(TrajectoryPoint(
                        fi, float(cx), float(cy), float(conf), False, "cache_gap",
                    ))
                    anchor = (cx, cy)
            out.append(nxt)
        return out

    def build_from_tracker(
        self,
        track_points: List[dict],
        frame_h: int = 0,
        raw_ball_cache: Optional[Dict[int, List[Tuple[float, float, float]]]] = None,
    ) -> RealTrajectoryResult:
        """Render path from Kalman delivery track: release trim, bat stop, light smooth."""
        ordered = sorted(track_points, key=lambda p: int(p.get("frame_idx", 0)))
        if len(ordered) < 2:
            return RealTrajectoryResult()

        trimmed_dicts: List[dict] = []
        for i, p in enumerate(ordered):
            if i == 0:
                trimmed_dicts.append(p)
                continue
            step = math.hypot(
                float(p.get("x", 0)) - float(ordered[i - 1].get("x", 0)),
                float(p.get("y", 0)) - float(ordered[i - 1].get("y", 0)),
            )
            if len(trimmed_dicts) < 3 and step < 8.0 and bool(p.get("is_interpolated", False)):
                continue
            trimmed_dicts.append(p)
        if len(trimmed_dicts) >= 2:
            ordered = trimmed_dicts

        points: List[TrajectoryPoint] = []
        for p in ordered:
            points.append(TrajectoryPoint(
                int(p.get("frame_idx", 0)),
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("confidence", 0.0)),
                bool(p.get("is_interpolated", False)),
                "tracker",
            ))

        det_rows = [
            (p.frame_idx, p.x, p.y, max(p.confidence, 0.01))
            for p in points
            if not p.is_interpolated and p.confidence > 0
        ]
        from analytics.delivery_phases import analyze_flight_phases

        phases = analyze_flight_phases(
            points,
            post_bat_retreat_px=self.post_bat_retreat_px,
            min_release_step_px=10.0,
        )
        if phases.valid:
            points = points[phases.release_index: phases.bat_impact_index + 1]
        elif len(det_rows) >= 3:
            trimmed = self._trim_to_release_start(det_rows, min_step_px=10.0)
            keep_from = trimmed[0][0]
            points = [p for p in points if p.frame_idx >= keep_from]
            points = self._truncate_at_bat_impact(points)
        else:
            points = self._truncate_at_bat_impact(points)

        knot_raw = [(p.x, p.y) for p in points if not p.is_interpolated]
        points = self._fill_gaps_from_cache(points, raw_ball_cache)
        points = self._drop_bottom_phantom_points(points, frame_h=frame_h)
        # Re-apply bat stop after gap fill (extensions must not pass the bat).
        phases_after = analyze_flight_phases(
            points,
            post_bat_retreat_px=self.post_bat_retreat_px,
        )
        if phases_after.valid:
            points = points[phases_after.release_index: phases_after.bat_impact_index + 1]
        else:
            points = self._truncate_at_bat_impact(points)
        points = self._kalman_stabilize(points)
        points = self._reject_upward_spikes(points)
        render_px, velocities = self._light_smooth(points)
        confs = [p.confidence for p in points if p.confidence > 0 and not p.is_interpolated]

        final_phases = analyze_flight_phases(points, post_bat_retreat_px=self.post_bat_retreat_px)
        return RealTrajectoryResult(
            points=points,
            render_pixels=render_px,
            raw_pixels=knot_raw,
            filtered_pixels=[(p.x, p.y) for p in points if not p.is_interpolated],
            frame_indices=[p.frame_idx for p in points],
            velocities=velocities,
            mean_confidence=float(sum(confs) / len(confs)) if confs else 0.0,
            stopped_at_frame=points[-1].frame_idx if points else None,
            release_frame=final_phases.release_frame if final_phases.valid else None,
            bat_impact_frame=final_phases.bat_impact_frame if final_phases.valid else None,
            bounce_frame=final_phases.bounce_frame,
        )


def build_real_path_from_cache(
    frame_start: int,
    frame_end: int,
    raw_ball_cache: Dict[int, List[Tuple[float, float, float]]],
    fps: float = 30.0,
    track_points: Optional[List[dict]] = None,
) -> RealTrajectoryResult:
    return RealBallPathBuilder().build(
        frame_start, frame_end, raw_ball_cache, fps, track_points,
    )


def to_observed_path_result(result: RealTrajectoryResult) -> "ObservedPathResult":
    from analytics.observed_path import ObservedPathResult, RejectedObservation

    return ObservedPathResult(
        raw_pixels=result.raw_pixels,
        filtered_pixels=result.filtered_pixels,
        smooth_pixels=result.render_pixels,
        frame_indices=result.frame_indices,
        confidences=[p.confidence for p in result.points],
        velocities=result.velocities,
        rejected=[
            RejectedObservation(r.frame_idx, r.x, r.y, r.confidence, r.reason)
            for r in result.rejected
        ],
        mean_confidence=result.mean_confidence,
        bridge_points_inserted=result.bridge_points_inserted,
    )


def slice_trajectory_to_frame(
    render_pixels: List[Tuple[float, float]],
    frame_indices: List[int],
    frame_idx: int,
) -> List[Tuple[float, float]]:
    if not render_pixels or not frame_indices:
        return render_pixels
    last_i = 0
    for i, fi in enumerate(frame_indices):
        if fi <= frame_idx:
            last_i = i
        else:
            break
    if last_i < 1:
        return []
    return render_pixels[: last_i + 1]
