"""
tracking/track_ball.py
======================
Professional Kalman ball tracker for cricket analytics.

Per-frame pipeline (SORT / broadcast style):
  YOLO detections → predict (6-state CA Kalman) → Mahalanobis gate
  → Hungarian pick → update | short coast → track point

Uses filterpy KalmanFilter + scipy Hungarian assignment.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from tracking.detection import Detection
from tracking.track_types import TrackPoint, TrackResult
from tracking.pro_kalman import (
    CHI2_GATE_99,
    ProBallKalman,
    build_association_costs,
    hungarian_pick_best,
)
from tracking.byte_associate import associate_detections
from tracking.optical_flow_tracker import OpticalFlowTracker
from tracking.track_interpolation import interpolate_track_gaps

logger = logging.getLogger(__name__)

# Backward-compatible alias used by analytics/real_trajectory.py
_BallKalman = ProBallKalman


class _TrackState(Enum):
    IDLE = "idle"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"


class TrackerPhase(Enum):
    """Public multi-stage tracker state for debug HUD / analytics."""

    IDLE = "IDLE"
    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    PREDICTING = "PREDICTING"
    LOST = "LOST"


# ---------------------------------------------------------------------------
# BallTracker — professional Kalman + Mahalanobis + Hungarian
# ---------------------------------------------------------------------------

class BallTracker:
    """
    Stateful per-delivery cricket ball tracker.

    Professional method:
      - 6-state constant-acceleration Kalman (filterpy)
      - Mahalanobis gating (chi-squared, 2-D)
      - Hungarian assignment when multiple YOLO candidates
      - SORT-style confirm (min_hits) and coast (max_age)
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_missing_frames: int = 8,
        confidence_threshold: float = 0.10,
        max_jump_px: float = 120.0,
        max_ball_bbox_px: float = 52.0,
        min_move_px: float = 1.0,
        low_conf_thresh: float = 0.03,
        low_conf_radius_px: float = 95.0,
        min_hits_to_confirm: int = 3,
        mahalanobis_gate: float = CHI2_GATE_99,
        enable_byte_track: bool = True,
        enable_optical_flow: bool = True,
        byte_low_ratio: float = 0.55,
        interpolation_method: str = "cubic",
        use_physics_filter: bool = False,
        physics_filter: Optional[Any] = None,
    ) -> None:
        self.fps = fps
        self.dt = 1.0
        self.max_missing = max(2, int(max_missing_frames))
        self.conf_thresh = confidence_threshold
        self.max_jump = max_jump_px
        self.max_bbox = max_ball_bbox_px
        self.min_move = min_move_px
        self.low_conf_thresh = low_conf_thresh
        self.low_conf_radius = low_conf_radius_px
        self.min_hits_to_confirm = max(2, int(min_hits_to_confirm))
        self.mahalanobis_gate = mahalanobis_gate
        self.enable_byte_track = enable_byte_track
        self.enable_optical_flow = enable_optical_flow
        self.byte_low_ratio = byte_low_ratio
        self.interpolation_method = interpolation_method

        self._flow = OpticalFlowTracker()
        self.use_physics_filter = bool(use_physics_filter)
        self._physics = physics_filter

        self._kf = ProBallKalman(
            dt=self.dt,
            process_noise=2.8,
            measurement_noise=5.5,
            gate_threshold=mahalanobis_gate,
        )
        self._points: List[TrackPoint] = []
        self._state = _TrackState.IDLE
        self._hits = 0
        self._age = 0
        self._lost_frames = 0
        self._last_pos: Optional[Tuple[float, float]] = None
        self._last_frame_idx: Optional[int] = None
        self._last_flow_conf: float = 0.0

    def reset(self) -> None:
        self._kf = ProBallKalman(
            dt=self.dt,
            process_noise=2.8,
            measurement_noise=5.5,
            gate_threshold=self.mahalanobis_gate,
        )
        self._points = []
        self._state = _TrackState.IDLE
        self._hits = 0
        self._age = 0
        self._lost_frames = 0
        self._last_pos = None
        self._last_frame_idx = None
        self._last_flow_conf = 0.0
        self._flow.reset()
        if self._physics is not None:
            self._physics.reset()
        logger.debug("BallTracker reset.")

    def get_phase(self) -> TrackerPhase:
        """Map internal FSM to production DETECTED / TRACKING / PREDICTING / LOST."""
        if self._state == _TrackState.IDLE:
            return TrackerPhase.IDLE
        if self._lost_frames > 0 and self._kf.initialized:
            if self._lost_frames >= self.max_missing:
                return TrackerPhase.LOST
            return TrackerPhase.PREDICTING
        if self._state == _TrackState.TENTATIVE:
            return TrackerPhase.DETECTED
        return TrackerPhase.TRACKING

    def get_predicted_position(self) -> Optional[Tuple[float, float]]:
        if not self._kf.initialized:
            return None
        x, y, _, _ = self._kf.peek_predict()
        return float(x), float(y)

    def get_state_kinematics(self) -> Tuple[float, float, float, float, float, float]:
        """Return x, y, vx, vy, ax, ay from Kalman state."""
        if not self._kf.initialized:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        x, y, vx, vy = self._kf.state_xy_v()
        s = self._kf._kf.x
        ax = float(s[4, 0]) if s.shape[0] > 4 else 0.0
        ay = float(s[5, 0]) if s.shape[0] > 5 else 0.0
        return float(x), float(y), float(vx), float(vy), ax, ay

    def update(
        self,
        frame_idx: int,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> Optional[TrackPoint]:
        gray: Optional[np.ndarray] = None
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        gap = 1
        if self._last_frame_idx is not None:
            gap = max(1, frame_idx - self._last_frame_idx)

        if self._kf.initialized and gap > 1:
            self._kf.predict_multi(gap - 1)
        elif self._kf.initialized:
            self._kf.predict()

        pred_xy = self.get_predicted_position()
        gated_dets = self._filter_detections(detections, frame_idx)
        matched = self._associate(gated_dets, frame_idx)

        if matched is not None:
            if gray is not None:
                self._flow.seed(gray, matched.cx, matched.cy)
            self._last_flow_conf = float(matched.conf)
            return self._update_with_detection(frame_idx, matched)

        # Optical flow fallback when YOLO misses but track is active.
        if (
            self.enable_optical_flow
            and gray is not None
            and self._kf.initialized
            and self._lost_frames < self.max_missing
            and self._hits >= self.min_hits_to_confirm
        ):
            if self._flow._point is None and self._last_pos is not None:
                self._flow.seed(gray, self._last_pos[0], self._last_pos[1])
            flow_hit = self._flow.track(gray, predicted=pred_xy)
            if flow_hit is not None:
                fx, fy, fconf = flow_hit
                flow_det = Detection(
                    frame_idx=frame_idx, cx=fx, cy=fy, conf=fconf, w=18.0, h=18.0,
                )
                self._last_flow_conf = fconf
                return self._update_with_detection(frame_idx, flow_det)

        # Re-acquire after extended gap.
        if (
            self._kf.initialized
            and self._lost_frames >= self.max_missing
            and gated_dets
            and self._hits >= self.min_hits_to_confirm
        ):
            recovered = self._associate(gated_dets, frame_idx)
            if recovered is not None:
                self._lost_frames = 0
                if gray is not None:
                    self._flow.seed(gray, recovered.cx, recovered.cy)
                return self._update_with_detection(frame_idx, recovered)

        if self._kf.initialized and self._lost_frames < self.max_missing:
            if self._state == _TrackState.TENTATIVE and self._hits < 2 and self._lost_frames >= 2:
                self.reset()
                return None
            return self._update_predicted(frame_idx)

        if self._kf.initialized:
            self._lost_frames += 1
            self._age += 1
        return None

    def finalize(self) -> TrackResult:
        points = list(self._points)
        if points and self.interpolation_method and self.interpolation_method not in ("off", "none", ""):
            dense = interpolate_track_gaps(points, method=self.interpolation_method)
            # Only keep densified track if it preserves real motion span.
            if len(dense) >= len(points):
                xs = [p.x for p in dense if not p.is_interpolated]
                ys = [p.y for p in dense if not p.is_interpolated]
                if xs and ys and (max(ys) - min(ys) >= 40.0 or max(xs) - min(xs) >= 40.0):
                    points = dense
        if not points:
            return TrackResult(points=[], confidence_mean=0.0, interpolated_pct=0.0)

        detected = [p for p in points if not p.is_interpolated]
        interpolated = [p for p in points if p.is_interpolated]
        conf_mean = float(np.mean([p.confidence for p in detected])) if detected else 0.0
        interp_pct = len(interpolated) / len(points) if points else 0.0

        logger.info(
            "BallTracker finalized | %d total pts | %d detected | %d interpolated | "
            "mean_conf=%.3f | interp_pct=%.1f%% | state=%s",
            len(points), len(detected), len(interpolated),
            conf_mean, interp_pct * 100, self._state.value,
        )
        return TrackResult(
            points=points,
            confidence_mean=round(conf_mean, 4),
            interpolated_pct=round(interp_pct, 4),
        )

    def draw_trajectory(self, frame: np.ndarray, max_tail: int = 60) -> np.ndarray:
        vis = frame.copy()
        pts = self._points[-max_tail:] if len(self._points) > max_tail else self._points
        if len(pts) < 2:
            return vis

        for i in range(1, len(pts)):
            t = i / (len(pts) - 1)
            alpha = 0.25 + 0.75 * t
            color = (int(200 * t), int(200 * t), 0) if pts[i].is_interpolated else (
                int(200 * (1 - t)), int(220 * t), 0
            )
            p1 = (int(pts[i - 1].x), int(pts[i - 1].y))
            p2 = (int(pts[i].x), int(pts[i].y))
            overlay = vis.copy()
            cv2.line(overlay, p1, p2, color, max(1, int(3 * t)), cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

        if pts:
            last = pts[-1]
            cx, cy = int(last.x), int(last.y)
            dot_color = (0, 140, 255) if last.is_interpolated else (0, 220, 80)
            cv2.circle(vis, (cx, cy), 8, (0, 30, 200), -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 16, dot_color, 2, cv2.LINE_AA)
        return vis

    # ------------------------------------------------------------------
    # Detection filtering
    # ------------------------------------------------------------------

    def _filter_detections(
        self, detections: List[Detection], frame_idx: int,
    ) -> List[Detection]:
        """Geometry + confidence + Mahalanobis pre-gate."""
        out: List[Detection] = []
        pred: Optional[Tuple[float, float]] = None
        if self._kf.initialized:
            px, py, _, _ = self._kf.peek_predict()
            pred = (px, py)

        physics_pool = detections
        if self.use_physics_filter and self._physics is not None:
            physics_pool = self._physics.filter_detections(
                detections,
                frame_idx,
                predicted_xy=pred,
            )

        for det in physics_pool:
            if det.w > self.max_bbox or det.h > self.max_bbox:
                continue
            if det.w > 0 and det.h > 0 and (det.w < 2.0 or det.h < 2.0):
                continue

            passes_primary = det.conf >= self.conf_thresh
            passes_adaptive = False
            if pred is not None:
                radial = math.hypot(det.cx - pred[0], det.cy - pred[1])
                if det.conf >= self.low_conf_thresh and radial <= self.low_conf_radius:
                    passes_adaptive = True
            if not (passes_primary or passes_adaptive):
                continue

            if self._kf.initialized and not self._kf.gated(det.cx, det.cy):
                if not passes_adaptive:
                    continue

            if self._last_pos is not None and self._last_frame_idx is not None:
                gap = max(1, frame_idx - self._last_frame_idx)
                max_step = self._max_step_for_gap(gap)
                anchor = pred if pred is not None else self._last_pos
                if math.hypot(det.cx - anchor[0], det.cy - anchor[1]) > max_step * 1.35:
                    continue

            out.append(det)
        return out

    def _associate(
        self, detections: List[Detection], frame_idx: int,
    ) -> Optional[Detection]:
        """Hungarian / ByteTrack two-stage assignment."""
        if not detections:
            return None

        if self.enable_byte_track and self._kf.initialized:
            low = max(self.low_conf_thresh, self.conf_thresh * self.byte_low_ratio)
            return associate_detections(
                self._kf,
                detections,
                high_thresh=self.conf_thresh,
                low_thresh=low,
                max_step_px=self.max_jump,
                last_pos=self._last_pos,
                last_frame_idx=self._last_frame_idx or frame_idx,
                frame_idx=frame_idx,
            )

        if not self._kf.initialized:
            boot = sorted(
                detections,
                key=lambda d: (-d.conf, -d.cy),
            )
            return boot[0]

        tuples = [(d.cx, d.cy, d.conf) for d in detections]
        cost = build_association_costs(tuples, self._kf)
        _, col = hungarian_pick_best(cost)
        if col < 0 or col >= len(detections):
            return None
        if cost[0, col] >= 1e5:
            return None
        return detections[col]

    def _max_step_for_gap(self, gap: int) -> float:
        gap = max(1, gap)
        base = min(self.max_jump * gap, 40.0 + 32.0 * gap)
        if self._kf.initialized:
            _, _, vx, vy = self._kf.peek_predict()
            speed = math.hypot(vx, vy)
            base = max(base, speed * gap * 1.8 + 20.0)
        return base

    def _update_with_detection(self, frame_idx: int, det: Detection) -> TrackPoint:
        self._lost_frames = 0
        self._age = 0
        self._hits += 1

        if not self._kf.initialized:
            self._kf.init(det.cx, det.cy)
            x, y, vx, vy = det.cx, det.cy, 0.0, 0.0
            self._state = _TrackState.TENTATIVE
        else:
            x, y, vx, vy = self._kf.correct(det.cx, det.cy)

        if self._hits >= self.min_hits_to_confirm:
            self._state = _TrackState.CONFIRMED

        return self._append(TrackPoint(
            frame_idx=frame_idx,
            x=x, y=y, vx=vx, vy=vy,
            is_interpolated=False,
            confidence=det.conf,
        ))

    def _update_predicted(self, frame_idx: int) -> TrackPoint:
        """Coast: state was already advanced in update() predict step."""
        self._lost_frames += 1
        self._age += 1
        x, y, vx, vy = self._kf.state_xy_v()
        return self._append(TrackPoint(
            frame_idx=frame_idx,
            x=x, y=y, vx=vx, vy=vy,
            is_interpolated=True,
            confidence=0.0,
        ))

    def _append(self, tp: TrackPoint) -> TrackPoint:
        if tp.is_interpolated and self._last_pos is not None:
            if math.hypot(tp.x - self._last_pos[0], tp.y - self._last_pos[1]) < self.min_move:
                return tp
        self._points.append(tp)
        self._last_pos = (tp.x, tp.y)
        self._last_frame_idx = tp.frame_idx
        return tp


# ---------------------------------------------------------------------------
# YOLO detection parser
# ---------------------------------------------------------------------------

def parse_yolo_detections(
    yolo_results,
    frame_idx: int,
    class_id: int = 0,
    min_conf: float = 0.25,
) -> List[Detection]:
    detections: List[Detection] = []
    try:
        boxes = yolo_results[0].boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls != class_id or conf < min_conf:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                frame_idx=frame_idx,
                cx=float(int((x1 + x2) / 2)),
                cy=float(int((y1 + y2) / 2)),
                conf=conf,
                w=x2 - x1,
                h=y2 - y1,
            ))
    except Exception as exc:
        logger.debug("parse_yolo_detections frame %d: %s", frame_idx, exc)
    return detections


if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO)
    tracker = BallTracker(fps=30.0, max_missing_frames=5)
    for f in range(60):
        dets = []
        if random.random() > 0.12:
            dets.append(Detection(
                frame_idx=f,
                cx=200.0 + f * 12.0 + random.gauss(0, 2),
                cy=400.0 + f * 4.0 + random.gauss(0, 2),
                conf=random.uniform(0.4, 0.98),
                w=22.0, h=22.0,
            ))
        tracker.update(f, dets)
    result = tracker.finalize()
    print(f"points={len(result.points)} conf={result.confidence_mean:.3f} "
          f"interp={result.interpolated_pct * 100:.1f}%")
