"""
analytics/bounce_detection.py
==============================
Detects the cricket ball bounce point from a smoothed trajectory.

Two complementary methods are used and their results fused:
  1. Lowest-point method    — the frame where pixel-Y (or world-Z proxy)
                              is at its minimum (ball closest to ground).
  2. Velocity-inversion method — the frame where vertical velocity (vy)
                                 changes sign from positive (descending in
                                 image-space) to negative (rising).

Author: Cricket Analytics Engine
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np

from tracking.track_ball import TrackPoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BounceResult:
    """Detected bounce point information."""
    frame_idx: int                # frame where bounce occurred
    pixel_x: float                # pixel X at bounce
    pixel_y: float                # pixel Y at bounce (highest value = lowest on screen)
    world_x: Optional[float]      # world lateral coordinate (metres, if calibrated)
    world_y: Optional[float]      # world longitudinal coordinate (metres, if calibrated)
    method: str                   # 'lowest_point' | 'velocity_inversion' | 'fused'
    confidence: float             # 0–1 confidence in the detection
    pre_bounce_angle_deg: float   # approach angle before bounce
    post_bounce_angle_deg: float  # departure angle after bounce

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# BounceDetector
# ---------------------------------------------------------------------------

class BounceDetector:
    """
    Identifies the bounce point from a list of TrackPoints.

    Typical usage
    -------------
        detector = BounceDetector()
        result = detector.detect(track_points, world_coords=world_list)
    """

    def __init__(
        self,
        min_frames_before_bounce: int = 4,
        min_frames_after_bounce: int = 2,
        smoothing_window: int = 5,
        velocity_sign_change_tolerance: int = 1,
    ):
        """
        Args:
            min_frames_before_bounce: Minimum trajectory frames before a
                                      valid bounce can be declared.
            min_frames_after_bounce:  Minimum frames after detected bounce
                                      to confirm it (avoids declaring bounce
                                      near end of clip).
            smoothing_window:         Rolling window for velocity smoothing.
            velocity_sign_change_tolerance: Consecutive frames for sign-flip
                                            confirmation (reduces false positives).
        """
        self.min_frames_before = min_frames_before_bounce
        self.min_frames_after = min_frames_after_bounce
        self.smooth_win = smoothing_window
        self.vel_tolerance = velocity_sign_change_tolerance

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def detect(
        self,
        track_points: List[TrackPoint],
        world_coords: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[BounceResult]:
        """
        Detect the bounce from a full-delivery TrackPoint list.

        Args:
            track_points:  Ordered list of TrackPoints for the delivery.
            world_coords:  Optional list of (wx, wy) metric tuples aligned
                           with track_points (from PitchCalibrator).

        Returns:
            BounceResult if a bounce is detected, else None.
        """
        n = len(track_points)
        if n < self.min_frames_before + self.min_frames_after:
            logger.warning(
                "Insufficient track points (%d) for bounce detection.", n
            )
            return None

        ys = np.array([tp.y for tp in track_points], dtype=np.float64)
        vys = np.array([tp.vy for tp in track_points], dtype=np.float64)

        # Smooth velocities to reduce noise
        vys_smooth = self._rolling_mean(vys, self.smooth_win)

        # Flight direction (down-screen vs. up-screen)
        is_increasing_y = np.mean(np.diff(ys[:8])) > 0 if len(ys) >= 8 else True

        # Method 1: lowest pixel-Y within valid window
        lp_idx = self._lowest_point_method(ys, is_increasing_y)

        # Method 2: vertical velocity sign inversion
        vi_idx = self._velocity_inversion_method(vys_smooth, is_increasing_y)

        # Fuse results
        bounce_idx, method, confidence = self._fuse(lp_idx, vi_idx, n)

        if bounce_idx is None:
            logger.info("No reliable bounce point detected.")
            return None

        tp = track_points[bounce_idx]

        # Resolve world coordinates
        wx, wy = None, None
        if world_coords and bounce_idx < len(world_coords):
            wx, wy = world_coords[bounce_idx]

        # Compute approach / departure angles
        pre_angle = self._compute_angle(track_points, bounce_idx, direction="pre")
        post_angle = self._compute_angle(track_points, bounce_idx, direction="post")

        result = BounceResult(
            frame_idx=tp.frame_idx,
            pixel_x=round(tp.x, 2),
            pixel_y=round(tp.y, 2),
            world_x=round(wx, 4) if wx is not None else None,
            world_y=round(wy, 4) if wy is not None else None,
            method=method,
            confidence=round(confidence, 4),
            pre_bounce_angle_deg=round(pre_angle, 2),
            post_bounce_angle_deg=round(post_angle, 2),
        )

        logger.info(
            "Bounce detected | frame=%d | pixel=(%.1f, %.1f) | method=%s | conf=%.3f",
            result.frame_idx, result.pixel_x, result.pixel_y,
            result.method, result.confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _lowest_point_method(self, ys: np.ndarray, is_increasing_y: bool = True) -> Optional[int]:
        """
        Find the index of the lowest physical point on screen (maximum pixel Y for down-screen,
        minimum pixel Y for up-screen). Restricts search to a valid interior window.
        """
        start = self.min_frames_before
        end = len(ys) - self.min_frames_after
        if start >= end:
            return None

        window = ys[start:end]
        if is_increasing_y:
            local_idx = int(np.argmax(window))
        else:
            local_idx = int(np.argmin(window))

        # Reject if the turning point is on the window boundaries (monotonic trend)
        if local_idx == 0 or local_idx == len(window) - 1:
            return None

        return local_idx + start

    def _velocity_inversion_method(self, vys_smooth: np.ndarray, is_increasing_y: bool = True) -> Optional[int]:
        """
        Find the first frame where vertical velocity transitions to a rebound phase
        (positive-to-negative for down-screen flight, negative-to-positive for up-screen).
        """
        consecutive = 0
        for i in range(self.min_frames_before, len(vys_smooth) - self.min_frames_after):
            if is_increasing_y:
                is_rebound = vys_smooth[i] < 0
            else:
                is_rebound = vys_smooth[i] > 0

            if is_rebound:
                consecutive += 1
                if consecutive >= self.vel_tolerance:
                    # Return the start of the sign-flip
                    return i - consecutive + 1
            else:
                consecutive = 0
        return None

    # ------------------------------------------------------------------
    # Fusion logic
    # ------------------------------------------------------------------

    def _fuse(
        self,
        lp_idx: Optional[int],
        vi_idx: Optional[int],
        total_frames: int,
    ) -> Tuple[Optional[int], str, float]:
        """
        Combine lowest-point and velocity-inversion results.

        Returns:
            (index, method_name, confidence)
        """
        if lp_idx is None and vi_idx is None:
            return None, "none", 0.0

        if lp_idx is None:
            return vi_idx, "velocity_inversion", 0.65

        if vi_idx is None:
            return lp_idx, "lowest_point", 0.65

        # Both detected — check agreement
        diff = abs(lp_idx - vi_idx)
        if diff <= 3:
            # Close agreement: use midpoint, high confidence
            fused_idx = (lp_idx + vi_idx) // 2
            confidence = min(0.95, 0.80 + (3 - diff) * 0.05)
            return fused_idx, "fused", confidence
        else:
            # Disagreement: trust velocity inversion more (physics-based)
            logger.debug(
                "Bounce methods disagree: lowest_point=%d velocity_inv=%d (diff=%d). "
                "Preferring velocity_inversion.",
                lp_idx, vi_idx, diff,
            )
            return vi_idx, "velocity_inversion", 0.55

    # ------------------------------------------------------------------
    # Angle computation
    # ------------------------------------------------------------------

    def _compute_angle(
        self,
        points: List[TrackPoint],
        bounce_idx: int,
        direction: str = "pre",
        window: int = 4,
    ) -> float:
        """
        Compute the trajectory angle (in degrees) approaching or departing
        the bounce point.

        Args:
            points:     Full TrackPoint list.
            bounce_idx: Index of detected bounce in `points`.
            direction:  'pre' (before bounce) or 'post' (after bounce).
            window:     Number of frames to average over.

        Returns:
            Angle in degrees from horizontal. Negative = downward.
        """
        if direction == "pre":
            idxs = range(max(0, bounce_idx - window), bounce_idx)
        else:
            idxs = range(bounce_idx + 1, min(len(points), bounce_idx + window + 1))

        idxs = list(idxs)
        if len(idxs) < 2:
            return 0.0

        xs = np.array([points[i].x for i in idxs])
        ys = np.array([points[i].y for i in idxs])

        # Fit line to get slope
        if xs[-1] == xs[0]:
            return 90.0

        slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0] + 1e-9)
        angle = np.degrees(np.arctan(slope))
        return float(angle)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Apply a simple rolling-mean smoothing to a 1-D array."""
        if window <= 1:
            return arr.copy()
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw_bounce_point(
        self,
        frame,
        bounce: BounceResult,
        color=(0, 140, 255),
        radius: int = 12,
    ):
        """
        Annotate the bounce point on a given BGR frame.

        Args:
            frame:   BGR numpy array.
            bounce:  BounceResult from `detect()`.
            color:   Circle colour (BGR).
            radius:  Circle radius in pixels.

        Returns:
            Annotated frame copy.
        """
        import cv2
        vis = frame.copy()
        cx, cy = int(bounce.pixel_x), int(bounce.pixel_y)
        cv2.circle(vis, (cx, cy), radius, color, 2, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 3, color, -1, cv2.LINE_AA)

        label = f"Bounce [{bounce.method}] {bounce.confidence:.2f}"
        cv2.putText(
            vis, label, (cx + 14, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )
        if bounce.world_x is not None:
            world_label = f"({bounce.world_x:.2f}m, {bounce.world_y:.2f}m)"
            cv2.putText(
                vis, world_label, (cx + 14, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1, cv2.LINE_AA,
            )
        return vis


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Simulate a parabolic-ish trajectory (ball descends then rises after bounce)
    from tracking.track_ball import TrackPoint

    pts = []
    for i in range(60):
        x = 200 + i * 9.0
        # Parabola with minimum around frame 35
        y = 50 + 5 * (i - 10) - 0.25 * (i - 35) ** 2 + 300
        vy_sim = 10 - 0.5 * i  # positive → descending, negative → ascending
        pts.append(TrackPoint(frame_idx=i, x=x, y=y, vx=9.0, vy=vy_sim))

    detector = BounceDetector()
    result = detector.detect(pts)
    if result:
        print(result.to_dict())
    else:
        print("No bounce detected.")