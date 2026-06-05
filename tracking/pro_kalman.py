"""
pro_kalman.py
=============
Professional sports-ball Kalman tracking core (broadcast / SORT-style).

Pipeline per frame:
  1. Predict state (constant-acceleration model, variable dt)
  2. Mahalanobis gate detections (chi-squared, 2-D position)
  3. Hungarian assignment (single active track → best gated detection)
  4. Kalman update on match, else short coast if track is confirmed

References: SORT, ByteTrack, Hawk-Eye-style CA models, Mahalanobis gating.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Chi-squared thresholds for 2-D position innovation (95% / 99%)
CHI2_GATE_95 = 5.991
CHI2_GATE_99 = 9.210


class ProBallKalman:
    """
    6-state constant-acceleration Kalman filter for cricket ball centre (x, y).

    State vector: [x, y, vx, vy, ax, ay]
    Measurement: [x, y]
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 2.5,
        measurement_noise: float = 6.0,
        gate_threshold: float = CHI2_GATE_99,
    ) -> None:
        self.dt = float(dt)
        self.gate_threshold = float(gate_threshold)
        self.initialized = False

        self._kf = KalmanFilter(dim_x=6, dim_z=2)
        self._kf.F = self._build_F(self.dt)
        self._kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)
        self._kf.Q = self._build_Q(process_noise)
        self._kf.R = np.eye(2, dtype=np.float64) * measurement_noise
        self._kf.P = np.eye(6, dtype=np.float64) * 500.0
        self._kf.x = np.zeros((6, 1), dtype=np.float64)

        self._last_innovation_norm = 0.0

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        dt = float(dt)
        dt2 = 0.5 * dt * dt
        return np.array([
            [1.0, 0.0, dt,  0.0, dt2, 0.0],
            [0.0, 1.0, 0.0, dt,  0.0, dt2],
            [0.0, 0.0, 1.0, 0.0, dt,  0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @staticmethod
    def _build_Q(process_noise: float) -> np.ndarray:
        """Block-diagonal process noise (position / velocity / acceleration)."""
        q = float(process_noise)
        return np.diag([
            q * 0.25, q * 0.25,
            q * 1.0,  q * 1.0,
            q * 2.5,  q * 2.5,
        ]).astype(np.float64)

    def init(self, cx: float, cy: float) -> None:
        self._kf.x = np.array(
            [[cx], [cy], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64,
        )
        self._kf.P = np.eye(6, dtype=np.float64) * 120.0
        self.initialized = True
        self._last_innovation_norm = 0.0

    def predict(self, dt: Optional[float] = None) -> Tuple[float, float, float, float]:
        """Advance filter one step (or with custom dt). Returns (x, y, vx, vy)."""
        if dt is not None and abs(dt - self.dt) > 1e-6:
            self._kf.F = self._build_F(dt)
        self._adapt_process_noise()
        self._kf.predict()
        x, y, vx, vy = self.state_xy_v()
        return x, y, vx, vy

    def predict_multi(self, steps: int) -> Tuple[float, float, float, float]:
        """Predict `steps` frames ahead (for frame gaps)."""
        steps = max(1, int(steps))
        for _ in range(steps):
            self.predict(self.dt)
        return self.state_xy_v()

    def peek_predict(self, dt: Optional[float] = None) -> Tuple[float, float, float, float]:
        """Non-destructive one-step prediction."""
        if not self.initialized:
            return (0.0, 0.0, 0.0, 0.0)
        F = self._build_F(dt or self.dt)
        x = self._kf.x.copy()
        x = F @ x
        return (
            float(x[0, 0]), float(x[1, 0]),
            float(x[2, 0]), float(x[3, 0]),
        )

    def correct(self, cx: float, cy: float) -> Tuple[float, float, float, float]:
        """Fuse measurement. Returns filtered (x, y, vx, vy)."""
        z = np.array([[cx], [cy]], dtype=np.float64)
        self._kf.update(z)
        y = np.asarray(self._kf.y).reshape(-1)
        self._last_innovation_norm = float(np.linalg.norm(y))
        return self.state_xy_v()

    def state_xy_v(self) -> Tuple[float, float, float, float]:
        s = self._kf.x
        return (
            float(s[0, 0]), float(s[1, 0]),
            float(s[2, 0]), float(s[3, 0]),
        )

    def mahalanobis_sq(self, cx: float, cy: float) -> float:
        """
        Squared Mahalanobis distance of (cx, cy) to the predicted measurement.
        Uses prior state (call after predict, or uses peek_predict state).
        """
        if not self.initialized:
            return float("inf")
        z = np.array([[cx], [cy]], dtype=np.float64)
        hx = self._kf.H @ self._kf.x
        y = z - hx
        S = self._kf.H @ self._kf.P @ self._kf.H.T + self._kf.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float("inf")
        d2 = float((y.T @ S_inv @ y)[0, 0])
        return max(0.0, d2)

    def gated(self, cx: float, cy: float) -> bool:
        return self.mahalanobis_sq(cx, cy) <= self.gate_threshold

    def _adapt_process_noise(self) -> None:
        """Raise Q slightly after large innovations (bounce / seam)."""
        base = 2.5
        boost = min(12.0, self._last_innovation_norm * 0.08)
        self._kf.Q = self._build_Q(base + boost)


def hungarian_pick_best(
    cost_matrix: np.ndarray,
) -> Tuple[int, int]:
    """
    Assign one track (row 0) to best detection column.
    Returns (row_idx, col_idx); col_idx=-1 if no feasible assignment.
    """
    if cost_matrix.size == 0:
        return 0, -1
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    if len(row_ind) == 0:
        return 0, -1
    return int(row_ind[0]), int(col_ind[0])


def build_association_costs(
    detections: List[Tuple[float, float, float]],
    kf: ProBallKalman,
    conf_weight: float = 0.15,
    big_cost: float = 1e6,
) -> np.ndarray:
    """
    Build 1 x N cost matrix for Hungarian assignment.

    detections: list of (cx, cy, confidence)
  Cost = Mahalanobis d² + confidence penalty; gated-out → big_cost.
    """
    if not detections:
        return np.zeros((1, 0), dtype=np.float64)

    costs = np.zeros((1, len(detections)), dtype=np.float64)
    for j, (cx, cy, conf) in enumerate(detections):
        d2 = kf.mahalanobis_sq(cx, cy)
        if d2 > kf.gate_threshold:
            costs[0, j] = big_cost
        else:
            costs[0, j] = d2 + conf_weight * (1.0 - float(conf)) * 10.0
    return costs
