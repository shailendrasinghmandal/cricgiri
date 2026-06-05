"""
tracking/optical_flow_tracker.py
================================
Lucas-Kanade optical flow fallback when YOLO misses the ball.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


class OpticalFlowTracker:
    """Bounded LK flow — continues track through blur / brief occlusion."""

    def __init__(
        self,
        max_drift_px: float = 45.0,
        max_step_px: float = 55.0,
        win_size: int = 21,
    ) -> None:
        self.max_drift = max_drift_px
        self.max_step = max_step_px
        self._lk = dict(
            winSize=(win_size, win_size),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self._prev_gray: Optional[np.ndarray] = None
        self._point: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None
        self._point = None

    def seed(self, gray: np.ndarray, x: float, y: float) -> None:
        self._prev_gray = gray
        self._point = np.array([[x, y]], dtype=np.float32)

    def track(
        self,
        gray: np.ndarray,
        predicted: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Advance one frame. Returns (x, y, confidence) or None if flow failed.
        confidence is heuristic 0–1 based on LK error vs prediction.
        """
        if self._prev_gray is None or self._point is None:
            self._prev_gray = gray
            return None

        nxt, st, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._point, None, **self._lk,
        )
        self._prev_gray = gray

        if nxt is None or st is None or int(st[0][0]) != 1:
            return None

        cx, cy = float(nxt[0, 0]), float(nxt[0, 1])
        step = float(np.linalg.norm(nxt[0] - self._point[0]))
        if step > self.max_step:
            return None

        conf = 0.55
        if predicted is not None:
            drift = math.hypot(cx - predicted[0], cy - predicted[1])
            if drift > self.max_drift:
                return None
            conf = max(0.25, 0.85 - drift / max(self.max_drift, 1.0))

        if err is not None:
            e = float(err[0][0])
            if e > 40.0:
                return None
            conf = min(conf, max(0.2, 1.0 - e / 50.0))

        self._point = nxt
        return cx, cy, conf
