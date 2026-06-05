

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Internal module imports — reuse existing project components as-is.
# ---------------------------------------------------------------------------
from analytics.bounce_detection import BounceDetector
from analytics.swing_estimation import SwingEstimator
from analytics.speed_estimation import SpeedEstimator

# TrackPoint is the shared coordinate type used across tracking/ and analytics/.
# Expected contract: an object (or dict) with at least (x, y, z, frame) fields.
from tracking.track_ball import TrackPoint

# ---------------------------------------------------------------------------
# Module-level logger — callers can customise level/handler from outside.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Absolute minimum points required to compute any meaningful analytics.
MIN_TRACK_POINTS: int = 10

#: Minimum points considered "high quality" for full confidence scores.
HIGH_QUALITY_THRESHOLD: int = 30

#: Weight applied per missing analytic when computing overall track quality.
MISSING_PENALTY: float = 0.15


# ---------------------------------------------------------------------------
# Internal result container (not part of the public API; used for clarity)
# ---------------------------------------------------------------------------

@dataclass
class _AnalyticsResult:
    """Typed intermediate container before serialising to the public dict."""

    speed_kmh: Optional[float] = None
    bounce: Optional[dict[str, Any]] = None
    swing: Optional[dict[str, Any]] = None
    track_quality: float = 0.0
    num_points: int = 0

    # Internal diagnostics — not exposed in public output.
    _warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the public analytics dictionary format."""
        return {
            "speed_kmh":     self.speed_kmh,
            "bounce":        self.bounce,
            "swing":         self.swing,
            "track_quality": round(self.track_quality, 4),
            "num_points":    self.num_points,
        }


# ---------------------------------------------------------------------------
# AnalyticsEngine
# ---------------------------------------------------------------------------

class AnalyticsEngine:
    """
    Orchestrates all analytics sub-modules for a single delivery.

    The engine is stateless between calls to `process_track`; the same
    instance can safely process multiple deliveries sequentially.

    Parameters
    ----------
    min_track_points : int, optional
        Override the default minimum track-point threshold.
    """

    def __init__(self, min_track_points: int = MIN_TRACK_POINTS) -> None:
        self._min_track_points = min_track_points

        # Instantiate each sub-analyser exactly once at construction time so
        # any expensive model loading happens up-front rather than per delivery.
        logger.debug("Initialising BounceDetector …")
        self._bounce_detector = BounceDetector()

        logger.debug("Initialising SwingEstimator …")
        self._swing_estimator = SwingEstimator()

        logger.debug("Initialising SpeedEstimator …")
        self._speed_estimator = SpeedEstimator()

        logger.info(
            "AnalyticsEngine ready (min_track_points=%d).",
            self._min_track_points,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_track(
        self,
        track_points: list[TrackPoint],
        fps: float = 30.0,
    ) -> dict[str, Any]:
        """
        Run the full analytics pipeline on a sequence of ball-track points.

        Parameters
        ----------
        track_points : list[TrackPoint]
            Ordered sequence of 3-D tracking points for a single delivery,
            as produced by ``tracking.track_ball``.
        fps : float, optional
            Camera frame rate used by the speed estimator.  Defaults to 30.0.

        Returns
        -------
        dict
            Structured analytics result with the following keys:

            - ``speed_kmh``    – estimated ball speed in km/h (float | None)
            - ``bounce``       – bounce analytics dict or None
            - ``swing``        – swing analytics dict or None
            - ``track_quality``– overall confidence score in [0.0, 1.0]
            - ``num_points``   – number of input track points

        Raises
        ------
        TypeError
            If ``track_points`` is not a list-like sequence.
        ValueError
            If ``fps`` is non-positive.
        """
        t_start = time.perf_counter()

        # ---- argument guards ------------------------------------------------
        if not hasattr(track_points, "__len__") and not hasattr(track_points, "__iter__"):
            raise TypeError(
                f"track_points must be a list-like sequence, got {type(track_points).__name__}."
            )
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}.")

        # Materialise to a plain list so we can index freely without risk of
        # consuming a generator or mutating the caller's data structure.
        points: list[TrackPoint] = list(track_points)
        result = _AnalyticsResult(num_points=len(points))

        logger.info(
            "process_track called: %d points @ %.1f fps.", len(points), fps
        )

        # ---- minimum length validation -------------------------------------
        if not self._validate_track_length(points, result):
            # Validation logs the reason and populates _warnings; return early.
            return result.to_dict()

        # ---- run analytics stages (each isolated behind try/except) ---------
        self._run_speed_estimation(points, fps, result)
        self._run_bounce_detection(points, fps, result)
        self._run_swing_estimation(points, fps, result)

        # ---- compute overall track quality score ----------------------------
        result.track_quality = self._compute_track_quality(points, result)

        elapsed_ms = (time.perf_counter() - t_start) * 1_000
        logger.info(
            "process_track complete in %.2f ms | speed=%.1f km/h | quality=%.3f",
            elapsed_ms,
            result.speed_kmh or 0.0,
            result.track_quality,
        )
        if result._warnings:
            logger.warning("Track warnings: %s", "; ".join(result._warnings))

        return result.to_dict()

    # ------------------------------------------------------------------
    # Private helpers — one per pipeline stage
    # ------------------------------------------------------------------

    def _validate_track_length(
        self,
        points: list[TrackPoint],
        result: _AnalyticsResult,
    ) -> bool:
        """
        Return True if the track has enough points to proceed.

        Populates ``result._warnings`` and logs the reason on failure.
        """
        if len(points) < self._min_track_points:
            msg = (
                f"Insufficient track points: {len(points)} < "
                f"{self._min_track_points} (minimum required)."
            )
            result._warnings.append(msg)
            logger.warning(msg)
            return False
        return True

    def _run_speed_estimation(
        self,
        points: list[TrackPoint],
        fps: float,
        result: _AnalyticsResult,
    ) -> None:
        """
        Invoke SpeedEstimator and store the result.

        On any exception the field is left as None and a warning is recorded
        so the remaining stages can still run.
        """
        try:
            speed = self._speed_estimator.estimate(points, fps=fps)
            if speed is not None and speed > 0.0:
                result.speed_kmh = float(speed)
                logger.debug("Speed estimated: %.2f km/h", result.speed_kmh)
            else:
                result._warnings.append("SpeedEstimator returned null/zero speed.")
                logger.debug("Speed estimation returned null or non-positive value.")
        except Exception as exc:  # noqa: BLE001
            msg = f"SpeedEstimator raised an unexpected error: {exc}"
            result._warnings.append(msg)
            logger.error(msg, exc_info=True)

    def _run_bounce_detection(
        self,
        points: list[TrackPoint],
        fps: float,
        result: _AnalyticsResult,
    ) -> None:
        """
        Invoke BounceDetector and store the result.

        Expected return value from BounceDetector.detect():
            {
                "detected":    bool,
                "frame":       int | None,
                "position":    (x, y, z) | None,
                "confidence":  float,
            }
        """
        try:
            bounce = self._bounce_detector.detect(points)
            if bounce is not None:
                result.bounce = bounce.to_dict()
                logger.debug(
                    "Bounce detection: detected=%s confidence=%.3f",
                    bounce.get("detected"),
                    bounce.get("confidence", 0.0),
                )
            else:
                result._warnings.append("BounceDetector returned None.")
        except Exception as exc:  # noqa: BLE001
            msg = f"BounceDetector raised an unexpected error: {exc}"
            result._warnings.append(msg)
            logger.error(msg, exc_info=True)

    def _run_swing_estimation(
        self,
        points: list[TrackPoint],
        fps: float,
        result: _AnalyticsResult,
    ) -> None:
        """
        Invoke SwingEstimator and store the result.

        Expected return value from SwingEstimator.estimate():
            {
                "swing_cm":    float | None,   # lateral deviation in cm
                "direction":   "in" | "out" | None,
                "confidence":  float,
            }
        """
        try:
            swing = self._swing_estimator.estimate(points)
            if swing is not None:
                result.swing = dict(swing)
                logger.debug(
                    "Swing estimation: %.2f cm (%s) confidence=%.3f",
                    swing.get("swing_cm") or 0.0,
                    swing.get("direction") or "N/A",
                    swing.get("confidence", 0.0),
                )
            else:
                result._warnings.append("SwingEstimator returned None.")
        except Exception as exc:  # noqa: BLE001
            msg = f"SwingEstimator raised an unexpected error: {exc}"
            result._warnings.append(msg)
            logger.error(msg, exc_info=True)

    # ------------------------------------------------------------------
    # Quality / confidence scoring
    # ------------------------------------------------------------------

    def _compute_track_quality(
        self,
        points: list[TrackPoint],
        result: _AnalyticsResult,
    ) -> float:
        """
        Compute a composite [0.0, 1.0] quality score for this delivery.

        Scoring logic
        -------------
        1. **Point-count factor** – scales linearly from 0 → 1 as track length
           grows from ``_min_track_points`` up to ``HIGH_QUALITY_THRESHOLD``.
        2. **Completeness factor** – starts at 1.0; deducted by
           ``MISSING_PENALTY`` for each of the three analytics that is absent
           (speed, bounce, swing).
        3. **Sub-module confidence** – if bounce and swing return a
           ``"confidence"`` key, their mean is blended in at 30 % weight.

        The final score is clamped to [0.0, 1.0].
        """
        n = len(points)

        # --- 1. point-count factor -------------------------------------------
        span = max(HIGH_QUALITY_THRESHOLD - self._min_track_points, 1)
        point_factor = min((n - self._min_track_points) / span, 1.0)

        # --- 2. completeness factor ------------------------------------------
        completeness = 1.0
        if result.speed_kmh is None:
            completeness -= MISSING_PENALTY
        if result.bounce is None:
            completeness -= MISSING_PENALTY
        if result.swing is None:
            completeness -= MISSING_PENALTY
        completeness = max(completeness, 0.0)

        # --- 3. sub-module confidence blend ----------------------------------
        sub_confidences: list[float] = []

        bounce_conf = (result.bounce or {}).get("confidence")
        if bounce_conf is not None:
            try:
                sub_confidences.append(float(bounce_conf))
            except (TypeError, ValueError):
                pass

        swing_conf = (result.swing or {}).get("confidence")
        if swing_conf is not None:
            try:
                sub_confidences.append(float(swing_conf))
            except (TypeError, ValueError):
                pass

        module_confidence = (
            sum(sub_confidences) / len(sub_confidences)
            if sub_confidences
            else 1.0  # neutral if sub-modules don't report confidence
        )

        # --- Weighted combination --------------------------------------------
        # point_factor:     40 %
        # completeness:     30 %
        # module_confidence: 30 %
        score = (
            0.40 * point_factor
            + 0.30 * completeness
            + 0.30 * module_confidence
        )

        return max(0.0, min(score, 1.0))