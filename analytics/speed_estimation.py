"""
analytics/speed_estimation.py
==============================
Estimates bowling speed from tracked world-coordinate trajectory.

Method (per PDF requirement):
    v = d / t

Where:
  - d = distance travelled in world coordinates (metres), clamped to pitch length
  - t = frame_count / fps (seconds)

The speed is estimated over the full release-to-bounce (or release-to-stump) segment.
No fake fallbacks or arbitrary speed mappings — pure physics only.

Author: Cricket Analytics Engine
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Physical constraints
PITCH_LENGTH_M: float = 20.12   # Maximum pitch length (bowling crease to batting crease)
MIN_SPEED_KMH: float = 30.0    # Absolute minimum realistic bowling speed
MAX_SPEED_KMH: float = 165.0   # Absolute maximum (world record ~161 km/h)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpeedResult:
    """Bowling speed estimation output."""
    speed_kmh: float               # estimated speed in km/h
    speed_mph: float               # same in mph
    speed_ms: float                # same in m/s
    distance_m: float              # world-space distance used for estimate
    duration_sec: float            # time over which distance was measured
    frames_used: int               # number of frames in estimation window
    method: str                    # 'physics_tof' | 'arc_length'
    confidence: float              # 0–1 quality indicator

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# SpeedEstimator
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """
    Estimates bowling speed from calibrated world-coordinate trajectory points.

    Uses the PDF-specified formula: v = d / t
    with pitch-length clamping to prevent homography distortion errors.
    """

    def __init__(
        self,
        fps: float = 30.0,
        release_segment_frames: int = 12,
        min_frames: int = 3,
        speed_multiplier: float = 1.0,
    ):
        """
        Args:
            fps:                     Video frame rate.
            release_segment_frames:  Number of frames to use from ball release
                                     for the primary speed estimate.
            min_frames:              Minimum frames required for any estimate.
            speed_multiplier:        Calibration multiplier for estimated speed.
        """
        self.fps = fps
        self.release_frames = release_segment_frames
        self.min_frames = min_frames
        self.speed_multiplier = speed_multiplier

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def estimate(
        self,
        world_points: List[Tuple[float, float]],
        bounce_frame_idx: Optional[int] = None,
        start_frame: int = 0,
        release_frame_idx: Optional[int] = None,
        bounce_frame: Optional[int] = None,
        bounce_y: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> Optional[SpeedResult]:
        """
        Estimate bowling speed using v = d / t.

        Strategy:
          1. If we have bounce_y and release/bounce frame info → use pure time-of-flight
          2. Otherwise → use arc length of world trajectory with pitch-length clamping

        Args:
            world_points:      Ordered list of (wx, wy) metric coordinates.
            bounce_frame_idx:  Index (into world_points) of the bounce frame.
            start_frame:       Offset into world_points.
            release_frame_idx: Real frame index of ball release.
            bounce_frame:      Real frame index of ball bounce.
            bounce_y:          Metric world y coordinate of bounce.
            fps:               Video frame rate override.
        """
        effective_fps = fps or self.fps
        n = len(world_points)

        if n < self.min_frames:
            logger.warning(
                "Not enough world points (%d) to estimate speed (min=%d).",
                n, self.min_frames,
            )
            return None

        # ─── Method 1: Pure time-of-flight (release frame → bounce frame) ───
        if (bounce_y is not None and release_frame_idx is not None
                and bounce_frame is not None and effective_fps > 0):

            frames_flight = bounce_frame - release_frame_idx
            if frames_flight > 0:
                duration_sec = frames_flight / effective_fps

                # Clamp bounce_y to realistic pitch length
                clamped_y = min(abs(bounce_y), PITCH_LENGTH_M)
                # Distance from release point (~1m from bowling crease) to bounce
                distance_m = max(1.0, clamped_y - 1.0)

                speed_ms = distance_m / duration_sec
                speed_kmh = speed_ms * 3.6
                speed_mph = speed_kmh * 0.621371

                if MIN_SPEED_KMH <= speed_kmh <= MAX_SPEED_KMH:
                    speed_kmh *= self.speed_multiplier
                    speed_mph *= self.speed_multiplier
                    speed_ms *= self.speed_multiplier

                    logger.info(
                        "Speed (physics_tof): %.1f km/h | dist=%.2fm | dur=%.3fs | frames=%d",
                        speed_kmh, distance_m, duration_sec, frames_flight
                    )
                    return SpeedResult(
                        speed_kmh=round(speed_kmh, 2),
                        speed_mph=round(speed_mph, 2),
                        speed_ms=round(speed_ms, 4),
                        distance_m=round(distance_m, 4),
                        duration_sec=round(duration_sec, 4),
                        frames_used=frames_flight,
                        method="physics_tof",
                        confidence=0.85,
                    )

        # ─── Method 2: Arc length with pitch-length clamping ───
        # Use the full trajectory arc length, but clamp total longitudinal
        # distance to PITCH_LENGTH_M to prevent homography overflow
        segment = world_points[start_frame:]

        # If we have a bounce index, use only pre-bounce segment
        if bounce_frame_idx is not None and bounce_frame_idx < len(segment):
            segment = segment[:bounce_frame_idx + 1]

        frames_used = len(segment)
        if frames_used < self.min_frames:
            segment = world_points
            frames_used = len(segment)

        # Compute raw arc length
        raw_distance = self._arc_length(segment)

        # Compute longitudinal span and clamp
        ys = [p[1] for p in segment]
        longitudinal_span = abs(max(ys) - min(ys))

        if longitudinal_span > PITCH_LENGTH_M:
            # Scale down the distance proportionally
            scale = PITCH_LENGTH_M / longitudinal_span
            distance_m = raw_distance * scale
            logger.info(
                "Clamping world distance: raw_span=%.1fm → clamped to %.1fm (scale=%.3f)",
                longitudinal_span, PITCH_LENGTH_M, scale
            )
        else:
            distance_m = raw_distance

        duration_sec = frames_used / effective_fps

        if duration_sec <= 0 or distance_m <= 0:
            logger.warning("Invalid distance/duration for speed estimation.")
            return None

        speed_ms = distance_m / duration_sec
        speed_kmh = speed_ms * 3.6
        speed_mph = speed_kmh * 0.621371
        method = "arc_length"

        # Hard sanity gate: if the computation produces a speed clearly outside
        # realistic cricket bowling (≥ MAX_SPEED_KMH or < MIN_SPEED_KMH), the
        # underlying distance/duration is unreliable — usually a homography
        # over-scale. Return None so the pipeline falls through to its
        # pixel-scale fallback and ultimately the coarse-velocity estimator,
        # rather than publishing a silently-clamped 165 that LOOKS plausible
        # but isn't measured.
        if speed_kmh >= MAX_SPEED_KMH:
            logger.warning(
                "Speed estimate %.1f km/h exceeds realistic max — returning None "
                "to let the pipeline use the pixel-scale fallback.",
                speed_kmh,
            )
            return None
        if speed_kmh < MIN_SPEED_KMH:
            logger.warning(
                "Speed estimate %.1f km/h below realistic min — returning None.",
                speed_kmh,
            )
            return None

        # Apply calibration multiplier
        speed_kmh *= self.speed_multiplier
        speed_mph *= self.speed_multiplier
        speed_ms *= self.speed_multiplier

        # Confidence based on method and number of frames
        frame_factor = min(1.0, frames_used / self.release_frames)
        confidence = 0.75 * frame_factor

        result = SpeedResult(
            speed_kmh=round(speed_kmh, 2),
            speed_mph=round(speed_mph, 2),
            speed_ms=round(speed_ms, 4),
            distance_m=round(distance_m, 4),
            duration_sec=round(duration_sec, 4),
            frames_used=frames_used,
            method=method,
            confidence=round(confidence, 4),
        )

        logger.info(
            "Speed estimate: %.1f km/h (%.1f mph) | method=%s | conf=%.2f",
            result.speed_kmh, result.speed_mph, method, confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Per-frame instantaneous speed
    # ------------------------------------------------------------------

    def frame_speeds(
        self,
        world_points: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Compute per-frame instantaneous speed in km/h.

        Useful for speed-over-time graphs and deceleration analysis.

        Args:
            world_points: (wx, wy) list.

        Returns:
            List of speeds in km/h (length = len(world_points) - 1).
        """
        speeds = []
        dt = 1.0 / self.fps
        for i in range(1, len(world_points)):
            dx = world_points[i][0] - world_points[i - 1][0]
            dy = world_points[i][1] - world_points[i - 1][1]
            dist = np.hypot(dx, dy)
            s_ms = dist / dt
            speeds.append(round(s_ms * 3.6, 2))
        return speeds

    # ------------------------------------------------------------------
    # Deceleration rate
    # ------------------------------------------------------------------

    def estimate_deceleration(
        self,
        world_points: List[Tuple[float, float]],
        pre_bounce_end: int,
        post_bounce_start: int,
    ) -> Optional[float]:
        """
        Estimate average speed loss due to pitch bounce (km/h drop).

        Args:
            world_points:       Full trajectory.
            pre_bounce_end:     Last frame index before bounce.
            post_bounce_start:  First frame index after bounce.

        Returns:
            Speed drop in km/h, or None if insufficient data.
        """
        pre_seg = world_points[max(0, pre_bounce_end - 5):pre_bounce_end]
        post_seg = world_points[post_bounce_start:post_bounce_start + 5]

        if len(pre_seg) < 2 or len(post_seg) < 2:
            return None

        pre_speed_result = self.estimate(pre_seg)
        post_speed_result = self.estimate(post_seg)

        if pre_speed_result is None or post_speed_result is None:
            return None

        drop = pre_speed_result.speed_kmh - post_speed_result.speed_kmh
        logger.info(
            "Speed on pitch contact: pre=%.1f km/h  post=%.1f km/h  drop=%.1f km/h",
            pre_speed_result.speed_kmh, post_speed_result.speed_kmh, drop,
        )
        return round(drop, 2)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _arc_length(world_points: List[Tuple[float, float]]) -> float:
        """
        Compute the total 2-D arc length (metres) of a world-coordinate path.
        """
        if len(world_points) < 2:
            return 0.0
        pts = np.array(world_points, dtype=np.float64)
        diffs = np.diff(pts, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return float(np.sum(segment_lengths))