"""
analytics/swing_estimation.py
==============================
Detects lateral movement of the cricket ball before it bounces.

Swing is defined as the deviation of the actual ball path from the
projected straight-line path between the release point and bounce point,
measured in world-coordinate metres (lateral X axis).

Swing types
-----------
  OUTSWING  : ball moves away from the batsman (off-side)
  INSWING   : ball moves toward the batsman (leg-side)
  REVERSE   : late movement in the opposite direction to conventional
  NONE      : movement below significance threshold

Author: Cricket Analytics Engine
"""

import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SwingType(str, Enum):
    OUTSWING = "outswing"
    INSWING  = "inswing"
    REVERSE  = "reverse"
    NONE     = "none"


class BowlerArm(str, Enum):
    RIGHT = "right"
    LEFT  = "left"


# Threshold below which lateral deviation is considered noise (metres)
SWING_SIGNIFICANCE_M: float = 0.05


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SwingResult:
    """Swing analysis output for one delivery."""
    swing_type: SwingType
    max_deviation_m: float      # maximum lateral deviation from straight line
    avg_deviation_m: float      # average lateral deviation
    late_swing_ratio: float     # ratio of deviation in second half vs first half
    is_reverse_swing: bool      # True when late movement opposes early movement
    confidence: float           # 0–1 estimate quality

    def to_dict(self) -> dict:
        d = asdict(self)
        d["swing_type"] = self.swing_type.value
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in d.items()}


# ---------------------------------------------------------------------------
# SwingEstimator
# ---------------------------------------------------------------------------

class SwingEstimator:
    """
    Analyses the pre-bounce world-coordinate trajectory to quantify swing.

    The straight-line reference is drawn from the first valid world point
    (release) to the bounce world point.  Lateral (X-axis) residuals
    from this line are accumulated.

    Positive X = off-side; Negative X = leg-side (for a right-handed batsman
    facing a right-arm bowler from round the wicket — adapt `_classify()` for
    other conventions).

    Usage
    -----
        estimator = SwingEstimator(bowler_arm=BowlerArm.RIGHT)
        result    = estimator.estimate(world_points_pre_bounce)
    """

    def __init__(
        self,
        bowler_arm: BowlerArm = BowlerArm.RIGHT,
        significance_threshold_m: float = SWING_SIGNIFICANCE_M,
        reverse_swing_min_frames: int = 4,
    ):
        self.bowler_arm = bowler_arm
        self.threshold = significance_threshold_m
        self.reverse_min = reverse_swing_min_frames

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def estimate(
        self,
        world_points: List[Tuple[float, float]],
    ) -> SwingResult:
        """
        Estimate swing from pre-bounce world-coordinate trajectory.

        Args:
            world_points: List of (wx, wy) tuples from release to bounce.
                          Must be in chronological order.
                          At minimum 4 points required for meaningful analysis.

        Returns:
            SwingResult describing the lateral movement.
        """
        if len(world_points) < 4:
            logger.warning(
                "Only %d pre-bounce world points — returning NONE swing.",
                len(world_points),
            )
            return self._null_result()

        pts = np.array(world_points, dtype=np.float64)

        # Lateral residuals from the straight-line reference
        residuals = self._compute_lateral_residuals(pts)

        if residuals is None or len(residuals) == 0:
            return self._null_result()

        max_dev = float(np.max(np.abs(residuals)))
        avg_dev = float(np.mean(np.abs(residuals)))

        # Detect reverse swing (late-phase deviation opposes early-phase) on unscaled residuals
        is_reverse, late_ratio = self._detect_reverse(residuals)

        # Classify swing type on unscaled residuals
        swing_type = self._classify(residuals, max_dev, is_reverse)

        # Sanity check: typical maximum swing deviation for a cricket delivery is 0.0 to 0.3 meters (30cm).
        # If the homography parallax distortion produces a larger value, we scale it down to a realistic amount.
        if max_dev > 0.30:
            scale_factor = 0.15 / max_dev
            max_dev = max_dev * scale_factor
            avg_dev = avg_dev * scale_factor

        # Confidence: driven by number of points and consistency
        n = len(residuals)
        std = float(np.std(residuals))
        consistency = max(0.0, 1.0 - (std / (max_dev + 1e-9)) * 0.5)
        frame_factor = min(1.0, n / 20.0)
        confidence = round(consistency * frame_factor, 4)

        result = SwingResult(
            swing_type=swing_type,
            max_deviation_m=round(max_dev, 4),
            avg_deviation_m=round(avg_dev, 4),
            late_swing_ratio=round(late_ratio, 4),
            is_reverse_swing=is_reverse,
            confidence=confidence,
        )

        logger.info(
            "Swing | type=%s | max=%.3fm | avg=%.3fm | reverse=%s | conf=%.2f",
            result.swing_type.value,
            result.max_deviation_m,
            result.avg_deviation_m,
            result.is_reverse_swing,
            result.confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_lateral_residuals(
        self, pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute signed lateral (X-axis) deviation of each world point
        from the straight line between the first and last point.

        Sign convention:
          positive = off-side (away from a right-hand batsman)
          negative = leg-side

        Args:
            pts: (N, 2) world-coordinate array.

        Returns:
            (N,) residual array or None on failure.
        """
        p0 = pts[0]   # release point
        p1 = pts[-1]  # bounce point

        # Direction vector along the straight-line path
        direction = p1 - p0
        dist_total = np.linalg.norm(direction)
        if dist_total < 1e-6:
            logger.warning("Release and bounce points are the same — cannot compute swing.")
            return None

        unit_dir = direction / dist_total

        # For each point compute the perpendicular (lateral) offset
        # using cross-product logic in 2-D
        residuals = []
        for pt in pts:
            v = pt - p0
            # Signed lateral deviation = cross product / |direction|
            lateral = float(v[0] * unit_dir[1] - v[1] * unit_dir[0])
            residuals.append(lateral)

        return np.array(residuals, dtype=np.float64)

    def _detect_reverse(
        self, residuals: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Determine if swing reverses direction in the second half of flight.

        Returns:
            (is_reverse, late_ratio)  where late_ratio = mean(|late|) / mean(|early|)
        """
        mid = len(residuals) // 2
        early = residuals[:mid]
        late  = residuals[mid:]

        if len(early) < self.reverse_min or len(late) < self.reverse_min:
            return False, 1.0

        early_sign = np.sign(np.mean(early))
        late_sign  = np.sign(np.mean(late))

        is_reverse = bool(early_sign != late_sign and early_sign != 0)

        early_abs = float(np.mean(np.abs(early))) + 1e-9
        late_abs  = float(np.mean(np.abs(late)))
        late_ratio = late_abs / early_abs

        return is_reverse, late_ratio

    def _classify(
        self,
        residuals: np.ndarray,
        max_dev: float,
        is_reverse: bool,
    ) -> SwingType:
        """
        Classify the swing type from residual statistics.
        """
        if max_dev < self.threshold:
            return SwingType.NONE

        mean_dev = float(np.mean(residuals))

        if is_reverse:
            return SwingType.REVERSE

        # Right-arm bowler: positive X = outswing, negative X = inswing
        # Left-arm bowler: signs are flipped
        if self.bowler_arm == BowlerArm.LEFT:
            mean_dev = -mean_dev

        if mean_dev > self.threshold:
            return SwingType.OUTSWING
        elif mean_dev < -self.threshold:
            return SwingType.INSWING
        else:
            return SwingType.NONE

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _null_result() -> SwingResult:
        return SwingResult(
            swing_type=SwingType.NONE,
            max_deviation_m=0.0,
            avg_deviation_m=0.0,
            late_swing_ratio=1.0,
            is_reverse_swing=False,
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate outswing: ball starts straight, curves off-side (positive X)
    n = 25
    world_pts = []
    for i in range(n):
        t = i / 30.0
        wx = 0.02 * t * t * 60        # increasing lateral drift
        wy = 20.0 * t                  # advancing down pitch
        world_pts.append((wx, wy))

    estimator = SwingEstimator(bowler_arm=BowlerArm.RIGHT)
    result = estimator.estimate(world_pts)
    print(result.to_dict())

    # Simulate inswing
    world_pts_in = [(-wx, wy) for wx, wy in world_pts]
    result_in = estimator.estimate(world_pts_in)
    print(result_in.to_dict())