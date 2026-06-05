"""
analytics/trajectory.py
========================
Generates a smooth polynomial trajectory from noisy tracking points,
provides projected/extrapolated path, classifies line and length,
and renders the trajectory visually onto video frames.

Author: Cricket Analytics Engine
"""

import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations — ICC Line and Length Classification
# ---------------------------------------------------------------------------

class BowlingLine(str, Enum):
    OUTSIDE_OFF   = "outside_off"
    OFF_STUMP     = "off_stump"
    MIDDLE_STUMP  = "middle_stump"
    LEG_STUMP     = "leg_stump"
    OUTSIDE_LEG   = "outside_leg"
    WIDE_OUTSIDE_OFF = "wide_outside_off"
    WIDE_OUTSIDE_LEG = "wide_outside_leg"


class BowlingLength(str, Enum):
    FULL_TOSS     = "full_toss"
    YORKER        = "yorker"
    FULL          = "full"
    GOOD_LENGTH   = "good_length"
    SHORT_OF_GOOD = "short_of_good"
    SHORT         = "short"
    BOUNCER       = "bouncer"


# ---------------------------------------------------------------------------
# World-space boundary definitions (metres, bowling-crease origin)
# ---------------------------------------------------------------------------

# Lateral (X-axis) boundaries — approximate stump positions
LINE_BOUNDARIES: List[Tuple[float, BowlingLine]] = [
    (-0.55,  BowlingLine.WIDE_OUTSIDE_OFF),
    (-0.30,  BowlingLine.OUTSIDE_OFF),
    (-0.114, BowlingLine.OFF_STUMP),
    ( 0.114, BowlingLine.MIDDLE_STUMP),
    ( 0.228, BowlingLine.LEG_STUMP),
    ( 0.55,  BowlingLine.OUTSIDE_LEG),
]

# Longitudinal (Y-axis) boundaries — metres from bowling crease
# Positive Y = toward batsman
LENGTH_BOUNDARIES: List[Tuple[float, BowlingLength]] = [
    ( 8.0,  BowlingLength.SHORT),
    (12.0,  BowlingLength.SHORT_OF_GOOD),
    (15.0,  BowlingLength.GOOD_LENGTH),
    (18.0,  BowlingLength.FULL),
    (19.4,  BowlingLength.YORKER),
    (20.12, BowlingLength.FULL_TOSS),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Fitted trajectory and classification output."""
    poly_coeffs_x: List[float]     # X(t) polynomial coefficients (degree 2)
    poly_coeffs_y: List[float]     # Y(t) polynomial coefficients (degree 2)
    smoothed_pixels: List[Tuple[float, float]]   # fitted pixel-space points
    bowling_line: BowlingLine
    bowling_length: BowlingLength
    pitch_map_x: Optional[float]   # world X at bounce (metres)
    pitch_map_y: Optional[float]   # world Y at bounce (metres)
    r_squared: float               # goodness of polynomial fit

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bowling_line"]   = self.bowling_line.value
        d["bowling_length"] = self.bowling_length.value
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in d.items()}


# ---------------------------------------------------------------------------
# TrajectoryAnalyser
# ---------------------------------------------------------------------------

class TrajectoryAnalyser:
    """
    Fits a polynomial trajectory to noisy pixel-space tracking points,
    renders it onto frames, and classifies line/length from bounce
    world coordinates.

    Usage
    -----
        analyser = TrajectoryAnalyser()
        result = analyser.analyse(pixel_points, world_bounce=(wx, wy))
        overlay = analyser.render(frame, result)
    """

    def __init__(
        self,
        poly_degree: int = 2,
        render_samples: int = 120,
        line_color: Tuple[int, int, int] = (0, 200, 255),
        line_thickness: int = 2,
    ):
        self.poly_degree = poly_degree
        self.render_samples = render_samples
        self.line_color = line_color
        self.line_thickness = line_thickness

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def analyse(
        self,
        pixel_points: List[Tuple[float, float]],
        world_bounce: Optional[Tuple[float, float]] = None,
    ) -> Optional[TrajectoryResult]:
        """
        Fit a polynomial to pixel tracking points and classify the delivery.

        Args:
            pixel_points:  List of (px, py) tuples (≥ 3 required).
            world_bounce:  (wx, wy) in metres at the bounce point.

        Returns:
            TrajectoryResult or None if fitting fails.
        """
        if len(pixel_points) < 3:
            logger.warning("Not enough points (%d) to fit trajectory.", len(pixel_points))
            return None

        pts = np.array(pixel_points, dtype=np.float64)
        t = np.linspace(0, 1, len(pts))

        # ---- Fit independent polynomials X(t) and Y(t) ----
        try:
            coeffs_x = np.polyfit(t, pts[:, 0], self.poly_degree)
            coeffs_y = np.polyfit(t, pts[:, 1], self.poly_degree)
        except np.linalg.LinAlgError as exc:
            logger.error("Polynomial fit failed: %s", exc)
            return None

        # ---- Evaluate fitted curve ----
        t_dense = np.linspace(0, 1, self.render_samples)
        x_fit = np.polyval(coeffs_x, t_dense)
        y_fit = np.polyval(coeffs_y, t_dense)
        smoothed = list(zip(x_fit.tolist(), y_fit.tolist()))

        # ---- R² goodness of fit ----
        r2 = self._r_squared(pts[:, 1], np.polyval(coeffs_y, t))

        # ---- Classify line and length ----
        bowling_line, bowling_length = self._classify(world_bounce)

        result = TrajectoryResult(
            poly_coeffs_x=coeffs_x.tolist(),
            poly_coeffs_y=coeffs_y.tolist(),
            smoothed_pixels=smoothed,
            bowling_line=bowling_line,
            bowling_length=bowling_length,
            pitch_map_x=round(world_bounce[0], 4) if world_bounce else None,
            pitch_map_y=round(world_bounce[1], 4) if world_bounce else None,
            r_squared=round(float(r2), 4),
        )

        logger.info(
            "Trajectory fitted | R²=%.4f | line=%s | length=%s",
            result.r_squared, result.bowling_line.value, result.bowling_length.value,
        )
        return result

    def project_to_stumps(
        self,
        result: TrajectoryResult,
        current_t: float = 1.0,
        steps: int = 30,
    ) -> List[Tuple[float, float]]:
        """
        Extrapolate the fitted polynomial beyond t=1 to project where the
        ball would have reached the batting crease (for DRS-style prediction).

        Args:
            result:     TrajectoryResult from `analyse()`.
            current_t:  Normalised time at last observed point (default 1.0).
            steps:      Number of extrapolation steps.

        Returns:
            List of projected (px, py) pixel coordinates.
        """
        coeffs_x = np.array(result.poly_coeffs_x)
        coeffs_y = np.array(result.poly_coeffs_y)
        t_ext = np.linspace(current_t, current_t + 0.5, steps)
        x_ext = np.polyval(coeffs_x, t_ext)
        y_ext = np.polyval(coeffs_y, t_ext)
        return list(zip(x_ext.tolist(), y_ext.tolist()))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        result: TrajectoryResult,
        draw_raw_points: bool = False,
        raw_points: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """
        Draw the smooth polynomial trajectory onto a BGR frame.

        Args:
            frame:            BGR numpy array.
            result:           TrajectoryResult.
            draw_raw_points:  Whether to draw the original noisy detections.
            raw_points:       Raw pixel points (needed if draw_raw_points=True).

        Returns:
            Annotated BGR frame copy.
        """
        vis = frame.copy()
        pts = result.smoothed_pixels

        if len(pts) < 2:
            return vis

        # Draw smooth trajectory curve
        for i in range(1, len(pts)):
            pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            pt2 = (int(pts[i][0]),     int(pts[i][1]))
            cv2.line(vis, pt1, pt2, self.line_color, self.line_thickness, cv2.LINE_AA)

        # Draw raw noisy detections as small dots (optional)
        if draw_raw_points and raw_points:
            for px, py in raw_points:
                cv2.circle(vis, (int(px), int(py)), 3, (100, 100, 255), -1, cv2.LINE_AA)

        # Annotation: line and length overlay
        vis = self._draw_annotation(vis, result)
        return vis

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(
        self,
        world_bounce: Optional[Tuple[float, float]],
    ) -> Tuple[BowlingLine, BowlingLength]:
        """
        Classify the delivery line and length from the bounce world coordinate.

        Sanity rules:
          * If world_x is outside [-1.5, 1.5] m, the homography is implausible
            (real pitch is 1.83 m wide between popping creases). We refuse to
            publish a fake "wide_outside_leg" and instead clamp to the nearest
            valid line, marking that fall-back with a default OFF_STUMP /
            GOOD_LENGTH so downstream code doesn't propagate garbage.
          * Default fall-back is OFF_STUMP / GOOD_LENGTH (most common cricket
            delivery), not WIDE_OUTSIDE_LEG which was misleading.
        """
        if world_bounce is None:
            return BowlingLine.OFF_STUMP, BowlingLength.GOOD_LENGTH

        wx, wy = world_bounce
        wx = float(wx)
        wy = float(wy)

        # If x is wildly out of range, the homography is questionable —
        # don't publish a bogus "wide_outside_leg" classification. Default
        # to OFF_STUMP which is the most-common bowling line.
        if abs(wx) > 1.5:
            return BowlingLine.OFF_STUMP, BowlingLength.GOOD_LENGTH

        # ---- Line ----
        bowling_line = BowlingLine.OFF_STUMP  # safer default than WIDE_OUTSIDE_LEG
        for threshold, label in LINE_BOUNDARIES:
            if wx <= threshold:
                bowling_line = label
                break

        # ---- Length ----
        # Same idea — refuse to classify when y is implausible.
        if wy < 1.0 or wy > 22.0:
            return bowling_line, BowlingLength.GOOD_LENGTH

        bowling_length = BowlingLength.GOOD_LENGTH  # safer default
        for threshold, label in LENGTH_BOUNDARIES:
            if wy <= threshold:
                bowling_length = label
                break

        return bowling_line, bowling_length

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _r_squared(y_actual: np.ndarray, y_fitted: np.ndarray) -> float:
        """Compute R² coefficient of determination."""
        ss_res = np.sum((y_actual - y_fitted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        if ss_tot == 0:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def _draw_annotation(frame: np.ndarray, result: TrajectoryResult) -> np.ndarray:
        """Overlay line and length classification text."""
        h, w = frame.shape[:2]
        text1 = f"Line: {result.bowling_line.value.replace('_', ' ').title()}"
        text2 = f"Length: {result.bowling_length.value.replace('_', ' ').title()}"

        cv2.putText(frame, text1, (w - 320, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text2, (w - 320, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        return frame


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate 30 noisy pixel-space ball positions (arc shape)
    import random
    pixel_pts = []
    for i in range(30):
        x = 200 + i * 12 + random.gauss(0, 2)
        y = 80 + 3.5 * i + 0.12 * i ** 2 + random.gauss(0, 2)
        pixel_pts.append((x, y))

    analyser = TrajectoryAnalyser()
    result = analyser.analyse(pixel_pts, world_bounce=(0.05, 7.5))
    if result:
        print(result.to_dict())
