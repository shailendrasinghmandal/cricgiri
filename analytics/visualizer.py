"""
analytics/visualizer.py
========================
Premium TV-Broadcast Visual Renderer for CricGiri Cricket Analytics.

Faithfully reproduces the FullTrack AI reference output style:
  • Translucent blue/indigo pitch corridor overlay (stump-to-stump)
  • Smooth light-blue trajectory arc (broadcast analysis flight curve)
  • Frosted-glass stat cards: Swing, Spin, Speed (top-left stack)
  • Clean white crease boundary markings at both ends
  • Ball-tracker dot with neon halo ring
  • Bounce target marker (concentric rings)

Author: CricGiri AI Premium Renderer
"""
from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color Palette (BGR format)
# ---------------------------------------------------------------------------
C_BLUE_TRAJ      = (255, 210, 70)      # Trajectory arc - broadcast light blue
C_BLUE_GLOW     = (255, 150, 30)      # Soft cyan/blue outer glow
C_BLUE_CORRIDOR = (180, 140, 60)      # Translucent corridor blue
C_WHITE         = (255, 255, 255)
C_GOLD          = (0, 180, 255)       # Gold accent for info icons
C_DARK_BG       = (30, 30, 35)        # Card backing
C_CARD_BORDER   = (120, 120, 130)     # Card border
C_LABEL_GRAY    = (180, 180, 180)     # Label text color
C_NEON_CYAN     = (255, 220, 0)       # Neon cyan accent
C_GREEN         = (80, 240, 80)       # Ball indicator
C_CREASE_WHITE  = (230, 230, 230)     # Crease line color
C_RED           = (40, 40, 230)       # Red color for bounce indicator and ball halo

FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_PLAIN      = cv2.FONT_HERSHEY_SIMPLEX


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "21e564",
            "runId": "trajectory-debug",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("debug-21e564.log", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


@dataclass
class VisualFrame:
    """Everything the premium renderer needs to annotate one frame."""
    trajectory_pixels:  List[Tuple[float, float]]   # smoothed observed path (drawn)
    trajectory_raw_pixels: List[Tuple[float, float]] = field(default_factory=list)
    trajectory_filtered_pixels: List[Tuple[float, float]] = field(default_factory=list)
    trajectory_rejected_pixels: List[Tuple[float, float, str]] = field(default_factory=list)
    trajectory_velocities: List[Tuple[float, float]] = field(default_factory=list)
    trajectory_accepted_count: int = 0
    trajectory_rejected_count: int = 0
    trajectory_debug: bool = False
    smooth_render_catmull: bool = False
    tracking_confidence: float = 0.0
    ball_pixel:         Optional[Tuple[float, float]] = None
    speed_kmh:          float = 0.0
    speed_mph:          float = 0.0
    speed_reliable:     bool  = True         # False => measurement was prior/implausible
    swing_deg:          float = 0.0          # lateral deviation (display value)
    spin_rpm:           float = 0.0          # drift/turn angle in degrees (NOT RPM per PDF)
    swing_label:        str   = "none"       # "outswing" | "inswing" | "reverse" | "none"
    bounce_pixel:       Optional[Tuple[float, float]] = None
    stump_pixels:       Optional[List[Tuple[float, float]]] = None
    pitch_top_left:     Optional[Tuple[int, int]] = None
    pitch_top_right:    Optional[Tuple[int, int]] = None
    pitch_bot_left:     Optional[Tuple[int, int]] = None
    pitch_bot_right:    Optional[Tuple[int, int]] = None
    frame_idx:          int = 0
    state:              str = "TRACKING"


# ---------------------------------------------------------------------------
# Pitch Corridor Overlay (Translucent blue/indigo)
# ---------------------------------------------------------------------------

def draw_pitch_corridor(
    frame:      np.ndarray,
    vis:        VisualFrame,
    alpha:      float = 0.18,
) -> np.ndarray:
    """Draw a translucent blue pitch corridor from bowler to batsman end."""
    corners = None
    if (vis.pitch_top_left and vis.pitch_top_right
            and vis.pitch_bot_left and vis.pitch_bot_right):
        corners = np.array([
            vis.pitch_top_left,
            vis.pitch_top_right,
            vis.pitch_bot_right,
            vis.pitch_bot_left,
        ], dtype=np.int32)

    elif vis.stump_pixels and len(vis.stump_pixels) >= 2:
        stumps_sorted = sorted(vis.stump_pixels, key=lambda p: p[0])
        left_x  = int(stumps_sorted[0][0]) - 35
        right_x = int(stumps_sorted[-1][0]) + 35
        bot_y   = int(max(p[1] for p in vis.stump_pixels)) + 20

        top_y = max(0, bot_y - int(frame.shape[0] * 0.70))
        corridor_shrink = int((bot_y - top_y) * 0.18)

        corners = np.array([
            [left_x  + corridor_shrink, top_y],
            [right_x - corridor_shrink, top_y],
            [right_x, bot_y],
            [left_x,  bot_y],
        ], dtype=np.int32)

    if corners is None:
        return frame

    # Transparent corridor fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [corners], C_BLUE_CORRIDOR)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


# ---------------------------------------------------------------------------
# Crease Boundary Lines (white markings at both ends)
# ---------------------------------------------------------------------------

def draw_virtual_stumps(frame: np.ndarray, vis: VisualFrame) -> np.ndarray:
    """Draw virtual stump markers if pixel coordinates are provided."""
    if vis.stump_pixels and len(vis.stump_pixels) >= 2:
        for (x, y) in vis.stump_pixels:
            cv2.circle(frame, (int(x), int(y)), 8, C_GREEN, -1, cv2.LINE_AA)
    return frame


def draw_crease_lines(
    frame:   np.ndarray,
    vis:     VisualFrame,
) -> np.ndarray:
    """Draw clean white crease boundary lines at bowler and batsman ends."""
    if not (vis.pitch_top_left and vis.pitch_top_right
            and vis.pitch_bot_left and vis.pitch_bot_right):
        return frame

    tl = vis.pitch_top_left
    tr = vis.pitch_top_right
    bl = vis.pitch_bot_left
    br = vis.pitch_bot_right

    # Bottom crease (bowler end) - full width
    cv2.line(frame, bl, br, C_CREASE_WHITE, 2, cv2.LINE_AA)

    # Top crease (batsman end) - full width
    cv2.line(frame, tl, tr, C_CREASE_WHITE, 2, cv2.LINE_AA)

    # Side boundary lines
    cv2.line(frame, bl, tl, C_CREASE_WHITE, 1, cv2.LINE_AA)
    cv2.line(frame, br, tr, C_CREASE_WHITE, 1, cv2.LINE_AA)

    # Popping crease guidelines (horizontal cross-lines near stumps)
    # Bottom crease stumps zone
    _draw_stump_marks(frame, bl, br, spacing=0.25)
    # Top crease stumps zone
    _draw_stump_marks(frame, tl, tr, spacing=0.25)

    return frame


def _draw_stump_marks(
    frame: np.ndarray,
    left: Tuple[int, int],
    right: Tuple[int, int],
    spacing: float = 0.25,
) -> None:
    """Draw short perpendicular stump marking ticks at crease line."""
    lx, ly = left
    rx, ry = right

    # Draw short perpendicular marks at 25%, 50%, 75%
    for frac in [spacing, 0.5, 1.0 - spacing]:
        mx = int(lx + frac * (rx - lx))
        my = int(ly + frac * (ry - ly))

        # Short perpendicular tick (8px each direction)
        dx = rx - lx
        dy = ry - ly
        length = math.sqrt(dx*dx + dy*dy) if (dx*dx + dy*dy) > 0 else 1.0
        # Normal vector (perpendicular to crease line)
        nx = -dy / length
        ny = dx / length
        tick_len = 12

        p1 = (int(mx + nx * tick_len), int(my + ny * tick_len))
        p2 = (int(mx - nx * tick_len), int(my - ny * tick_len))
        cv2.line(frame, p1, p2, C_CREASE_WHITE, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Smooth Light-Blue Trajectory Arc
# ---------------------------------------------------------------------------

def _ransac_parabola(arr: np.ndarray, inlier_thresh_px: float = 18.0,
                     iters: int = 200, min_inliers: int = 4) -> np.ndarray:
    """
    RANSAC fit a parabola in screen space and return the inlier subset,
    preserving the input (time) order so the curve renders correctly.

    Tries both `v = f(x)` and `v = f(y)` and picks the axis that yields more
    agreeing points — important because cricket deliveries can be mostly
    horizontal (release-to-batsman) or mostly vertical (high arm action,
    short pitch).

    This is the key noise filter: when the trajectory is built from a mix of
    high-confidence tracker points and permissive YOLO candidates, RANSAC
    drops anything that doesn't lie on the same ball-flight parabola
    (background hits, brief detections on body parts, post-impact frames).
    """
    n = len(arr)
    if n < 4:
        return arr

    def _ransac_axis(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, int, float]:
        rng = np.random.default_rng(seed=42)
        best_mask = np.zeros(n, dtype=bool)
        best_score = -1
        best_resid = float("inf")
        for _ in range(iters):
            try:
                idx = rng.choice(n, size=3, replace=False)
            except ValueError:
                break
            # Guarantee 3 distinct u-values, otherwise polyfit explodes
            if len(set(u[idx])) < 3:
                continue
            try:
                coeffs = np.polyfit(u[idx], v[idx], 2)
            except Exception:
                continue
            v_pred = np.polyval(coeffs, u)
            residual = np.abs(v_pred - v)
            mask = residual < inlier_thresh_px
            score = int(mask.sum())
            if score < 3:
                continue
            resid_sum = float(residual[mask].sum())
            if (score > best_score) or (score == best_score and resid_sum < best_resid):
                best_score = score
                best_mask = mask
                best_resid = resid_sum
        return best_mask, best_score, best_resid

    mask_x, score_x, resid_x = _ransac_axis(arr[:, 0], arr[:, 1])
    mask_y, score_y, resid_y = _ransac_axis(arr[:, 1], arr[:, 0])

    # Pick the axis that explains the most points; tiebreak by lower residual.
    if (score_x > score_y) or (score_x == score_y and resid_x <= resid_y):
        best_mask, best_score = mask_x, score_x
    else:
        best_mask, best_score = mask_y, score_y

    if best_score < max(min_inliers, max(4, n // 3)):
        # Not enough agreement — keep the original sequence so we still draw
        # something rather than nothing. Smoothing downstream handles the rest.
        return arr

    # Preserve the original (time-ordered) sequence of the inliers so the
    # bounce ascent/descent reads correctly downstream.
    return arr[best_mask]


def _pick_longest_moving_segment(arr: np.ndarray,
                                  min_step_px: float = 4.0,
                                  max_step_px: float = 80.0,
                                  min_len: int = 5) -> np.ndarray:
    """
    Pick the longest contiguous run of points whose per-frame step is
    consistent with a moving ball (between `min_step_px` and `max_step_px`).

    This is what saves videos where the tracker covers both a phantom
    stationary cluster (e.g. logos, fixed cameras) and the real ball flight.
    Without this split, the smoothed curve gets dragged through both.
    """
    n = len(arr)
    if n < min_len + 1:
        return arr

    diffs = np.diff(arr, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    valid = (steps >= min_step_px) & (steps <= max_step_px)

    # Each valid[i] means the segment between point i and i+1 is "moving"
    # at a ball-like pace. Find the longest run of True.
    best_start = 0
    best_end = 0  # exclusive index in arr terms
    cur_start = 0
    cur_len = 0
    for i, v in enumerate(valid):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > (best_end - best_start):
                best_start = cur_start
                best_end = i + 2  # +1 for inclusive segment, +1 for the endpoint
        else:
            cur_len = 0

    if (best_end - best_start) < min_len:
        return arr
    return arr[best_start:best_end]


def _strip_stationary_clusters(arr: np.ndarray,
                                min_run: int = 5,
                                stationary_step_px: float = 3.5) -> np.ndarray:
    """
    Remove long runs of nearly-stationary points. These appear when the
    detector latches onto a phantom (a logo, a light, a static object) and
    the tracker fills in dozens of consecutive frames at almost the same
    pixel. They pull the fitted curve off the real flight.

    A point is considered part of a stationary run when its step from the
    previous point is below `stationary_step_px`. Runs shorter than
    `min_run` are kept (they may be the natural slowdown around bounce).
    """
    n = len(arr)
    if n < min_run + 2:
        return arr

    diffs = np.diff(arr, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    # mark indices whose step in OR out is tiny
    keep = np.ones(n, dtype=bool)
    i = 0
    while i < len(steps):
        if steps[i] < stationary_step_px:
            j = i
            while j < len(steps) and steps[j] < stationary_step_px:
                j += 1
            run_len = j - i + 1  # inclusive count of stationary points
            if run_len >= min_run:
                # keep the first and last point of the run, drop the middle.
                # This preserves trajectory entry/exit while killing the cluster.
                keep[i + 1 : j] = False
            i = j
        else:
            i += 1
    return arr[keep]


def _estimate_bounce_idx(arr: np.ndarray) -> int:
    """
    Estimate bounce as an interior screen-Y extremum, not the path endpoints.

    Endpoints are often release or bat-contact frames; using them as bounce
    causes false early truncation right after the first segment.
    """
    n = len(arr)
    if n < 5:
        return int(np.argmax(arr[:, 1]))

    margin = max(1, n // 6)
    ys = arr[:, 1]
    downward = float(ys[-1]) >= float(ys[0]) - 5.0
    if margin > 0 and (n - 2 * margin) >= 3:
        interior = arr[margin:-margin]
        local = int(np.argmax(interior[:, 1]) if downward else np.argmin(interior[:, 1]))
        return margin + local
    if n >= 3:
        interior = arr[1:-1]
        local = int(np.argmax(interior[:, 1]) if downward else np.argmin(interior[:, 1]))
        return 1 + local
    return int(np.argmax(arr[:, 1]) if downward else np.argmin(arr[:, 1]))


def _longest_clean_segment(arr: np.ndarray) -> np.ndarray:
    """
    Trim only the TAIL after a non-physical jump in the LATTER half of the
    trajectory. Never cuts the head: release-area detection wobble is part of
    the natural delivery and the visualizer's smoothing absorbs it.

    A real pitch bounce never produces a single-frame step >= 5x the recent
    median, so this is bounce-safe.
    """
    n_local = len(arr)
    if n_local < 8:
        return arr
    diffs_local = np.diff(arr, axis=0)
    steps_local = np.linalg.norm(diffs_local, axis=1)
    if len(steps_local) < 4:
        return arr
    MIN_GATE_ABS = 70.0
    MIN_LEAD_MEDIAN = 4.0
    ABS_HARD_JUMP = 90.0
    jump_indices: List[int] = []
    jump_set: set = set()
    for i_local in range(1, len(steps_local)):
        step_val = float(steps_local[i_local])
        if step_val > ABS_HARD_JUMP:
            jump_indices.append(i_local)
            jump_set.add(i_local)
            continue
        clean = [
            float(steps_local[k])
            for k in range(max(0, i_local - 6), i_local)
            if k not in jump_set
        ]
        if not clean:
            continue
        ref = max(MIN_LEAD_MEDIAN, float(np.median(clean)))
        gate = max(MIN_GATE_ABS, ref * 5.0)
        if step_val > gate:
            jump_indices.append(i_local)
            jump_set.add(i_local)
    if not jump_indices:
        return arr

    half_idx = max(1, n_local // 2)
    late_jumps = [j for j in jump_indices if j >= half_idx]
    if not late_jumps:
        return arr
    last_jump = late_jumps[-1]
    keep_until = last_jump + 1  # keep up to and including arr[last_jump]
    if 5 <= keep_until < n_local:
        _debug_log(
            "H3",
            "visualizer.py:_truncate_at_bat_impact:late-jump-tail",
            "Trimmed tail at late non-physical jump (bat impact).",
            {
                "input_len": int(n_local),
                "keep_until": int(keep_until),
                "num_jumps_total": int(len(jump_indices)),
                "num_late_jumps": int(len(late_jumps)),
            },
        )
        return arr[:keep_until]
    return arr


def _repair_single_point_spikes(arr: np.ndarray, threshold_px: float = 80.0) -> np.ndarray:
    """
    Replace any single-point outlier that is >= threshold_px off the chord
    between its immediate neighbours with that chord midpoint. This removes
    visible loops caused by lone tracker glitches without losing any frame
    from the delivery. Bounce-safe.
    """
    if len(arr) < 4:
        return arr
    out = arr.copy()
    fixed = 0
    for k in range(1, len(out) - 1):
        p_prev = out[k - 1]
        p_next = out[k + 1]
        mid = (p_prev + p_next) * 0.5
        d = float(np.linalg.norm(out[k] - mid))
        if d >= threshold_px:
            out[k] = mid
            fixed += 1
    if fixed:
        _debug_log(
            "H3",
            "visualizer.py:_repair_single_point_spikes",
            "Repaired single-frame spike(s) by snapping to chord midpoint.",
            {"input_len": int(len(arr)), "fixed": int(fixed), "threshold_px": float(threshold_px)},
        )
    return out


def _truncate_at_bat_impact(arr: np.ndarray) -> np.ndarray:
    """
    Cut off at bat impact: sudden step jump or velocity direction reversal
    near the *end* of the delivery arc — never at the bounce kinematic change.
    """
    n = len(arr)
    if n < 4:
        return arr

    # Phase 0a: repair single-point spikes (lone tracker glitches that draw
    # a visible loop). This preserves all frames of the delivery.
    arr = _repair_single_point_spikes(arr, threshold_px=80.0)
    n = len(arr)
    if n < 4:
        return arr

    # Phase 0b: tail-only truncation at non-physical jumps in the latter half
    # (bat impact / false re-track). The head is never cut: release-area
    # detection wobble is part of the natural delivery.
    arr = _longest_clean_segment(arr)
    n = len(arr)
    if n < 4:
        return arr

    diffs = np.diff(arr, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    if len(steps) == 0:
        return arr

    median_step = float(np.median(steps))
    if median_step < 1.0:
        median_step = 1.0

    bounce_idx = _estimate_bounce_idx(arr)
    # Bounce reverses vertical motion; bat contact is usually in the last third.
    search_start = max(bounce_idx + 10, int(n * 0.58), 3)
    downward = float(arr[-1, 1]) >= float(arr[0, 1]) - 5.0

    # Post-bat only: large step + reversal of post-bounce screen-Y trend.
    for i in range(search_start, n):
        if i < 2 or i < int(n * 0.50):
            continue
        step_i = float(steps[i - 1])
        dy = float(arr[i, 1] - arr[i - 1, 1])
        if step_i < max(48.0, median_step * 2.8) or abs(dy) < max(38.0, median_step * 2.2):
            continue
        dys = [
            float(arr[k, 1] - arr[k - 1, 1])
            for k in range(max(1, i - 4), i)
        ]
        if not dys:
            continue
        med_dy = float(np.median(dys))
        med_abs = max(2.5, float(np.median([abs(d) for d in dys])))
        if med_dy * dy < 0.0 and abs(dy) > max(38.0, 2.5 * med_abs):
            return arr[:i]
        if downward:
            if med_dy < -0.8 and dy > max(22.0, 2.2 * med_abs):
                return arr[:i]
        else:
            if med_dy > 0.8 and dy < -max(22.0, 2.2 * med_abs):
                return arr[:i]

    for i in range(search_start, len(steps)):
        if i < int(n * 0.45):
            continue
        if steps[i] > max(55.0, median_step * 4.5):
            # #region agent log
            _debug_log(
                "H3",
                "visualizer.py:_truncate_at_bat_impact:step-jump",
                "Truncated trajectory due to post-bounce step jump.",
                {
                    "input_len": int(n),
                    "truncate_at_idx": int(i + 1),
                    "step_value": float(steps[i]),
                    "median_step": float(median_step),
                    "threshold": float(max(65.0, median_step * 5.5)),
                    "bounce_idx": int(bounce_idx),
                    "search_start": int(search_start),
                },
            )
            # #endregion
            return arr[: i + 1]

    # Direction reversal alone happens at bounce — require a large step too
    # so we do not chop the path at the bounce point.
    for i in range(search_start + 1, n - 1):
        if i < int(n * 0.45):
            continue
        v1 = arr[i] - arr[i - 1]
        v2 = arr[i + 1] - arr[i]
        s1 = float(np.linalg.norm(v1))
        s2 = float(np.linalg.norm(v2))
        if s1 < 4.0 or s2 < 4.0:
            continue
        cos_angle = float(np.dot(v1, v2) / (s1 * s2))
        step_after = float(np.linalg.norm(arr[i + 1] - arr[i]))
        if i < max(bounce_idx + 6, int(n * 0.55)):
            continue
        if cos_angle < -0.30 and step_after > max(40.0, median_step * 2.8):
            # #region agent log
            _debug_log(
                "H3",
                "visualizer.py:_truncate_at_bat_impact:direction-reversal",
                "Truncated trajectory due to sharp reversal + large step.",
                {
                    "input_len": int(n),
                    "truncate_at_idx": int(i + 1),
                    "cos_angle": float(cos_angle),
                    "step_after": float(step_after),
                    "median_step": float(median_step),
                    "threshold": float(max(40.0, median_step * 3.0)),
                    "bounce_idx": int(bounce_idx),
                },
            )
            # #endregion
            return arr[: i + 1]

    return arr


def _moving_average_smooth(
    arr: np.ndarray,
    window: int = 3,
    max_shift_px: float = 2.0,
) -> np.ndarray:
    """Light jitter reduction along the observed path only (no prediction)."""
    n = len(arr)
    if n < 3 or window < 3:
        return arr
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.empty_like(arr)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        sm = arr[lo:hi].mean(axis=0)
        delta = sm - arr[i]
        dist = float(np.linalg.norm(delta))
        if dist > max_shift_px and dist > 1e-6:
            sm = arr[i] + delta * (max_shift_px / dist)
        out[i] = sm
    return out


def prepare_observed_trajectory_points(
    trajectory: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Real ball path only: dedupe → stop at bat impact → light moving average.
    No RANSAC, parabola fit, or synthetic continuation.
    """
    if not trajectory:
        return []

    arr = _dedupe_points(np.asarray(trajectory, dtype=np.float64), min_dist_px=1.5)
    if len(arr) < 2:
        return [(float(p[0]), float(p[1])) for p in arr]

    before_truncate = int(len(arr))
    arr = _truncate_at_bat_impact(arr)
    arr_before_smooth = arr.copy()
    if len(arr) >= 3:
        # Slightly stronger smoothing for longer tracks to match the
        # broadcast-style continuity seen in reference videos, while keeping
        # points close to observed detections.
        smooth_window = 5 if len(arr) >= 14 else 3
        smooth_shift = 3.5 if len(arr) >= 14 else 2.0
        arr = _moving_average_smooth(arr, window=smooth_window, max_shift_px=smooth_shift)

    # #region agent log
    # Quantify how much smoothing moves points away from observed anchors.
    if len(arr_before_smooth) == len(arr) and len(arr) >= 2:
        deltas = np.linalg.norm(arr - arr_before_smooth, axis=1)
        _debug_log(
            "H8",
            "visualizer.py:prepare_observed_trajectory_points:smoothing-delta",
            "Measured per-point shift introduced by moving-average smoothing.",
            {
                "points": int(len(arr)),
                "avg_shift_px": float(np.mean(deltas)),
                "max_shift_px": float(np.max(deltas)),
            },
        )
    # #endregion

    if len(arr) != before_truncate:
        # #region agent log
        _debug_log(
            "H3",
            "visualizer.py:prepare_observed_trajectory_points:post-truncate",
            "Trajectory length changed after truncation/smoothing step.",
            {
                "before_len": int(before_truncate),
                "after_len": int(len(arr)),
                "first": (float(arr[0][0]), float(arr[0][1])) if len(arr) else None,
                "last": (float(arr[-1][0]), float(arr[-1][1])) if len(arr) else None,
            },
        )
        # #endregion

    return [(float(p[0]), float(p[1])) for p in arr]


def draw_trajectory_arc(
    frame:      np.ndarray,
    trajectory: List[Tuple[float, float]],
    degree:     int   = 3,
    n_smooth:   int   = 200,
    use_catmull: bool = False,
) -> np.ndarray:
    """
    Draw a glowing trail through ONLY real tracked points (release → bat).

    Default: linear densification (faithful to observed knots).
    ``use_catmull=True``: Catmull-Rom smoothing for production anti-jitter render.
    """
    del degree  # kept for call-site compatibility

    if len(trajectory) < 2:
        return frame

    arr = _dedupe_points(np.asarray(trajectory, dtype=np.float64), min_dist_px=0.8)
    if use_catmull and len(arr) >= 3:
        dense = _catmull_rom_chain(arr, samples_per_segment=16)
        dense = _resample_by_distance(dense, max(n_smooth, len(arr) * 10))
    else:
        dense = _resample_by_distance(arr, max(n_smooth, len(arr) * 8))
    if len(dense) < 2:
        return frame

    # Fading trail: older segments drawn with lower opacity
    n_seg = len(dense) - 1
    for i in range(n_seg):
        t = (i + 1) / max(1, n_seg)
        alpha_glow = 0.08 + 0.26 * t
        alpha_core = 0.12 + 0.38 * t
        seg = np.round(dense[i : i + 2]).astype(np.int32)
        if len(seg) < 2:
            continue
        glow = frame.copy()
        cv2.polylines(glow, [seg], False, C_BLUE_GLOW, 14, cv2.LINE_AA)
        cv2.addWeighted(glow, alpha_glow, frame, 1.0 - alpha_glow, 0, frame)
        core = frame.copy()
        cv2.polylines(core, [seg], False, C_BLUE_TRAJ, 7, cv2.LINE_AA)
        cv2.addWeighted(core, alpha_core, frame, 1.0 - alpha_core, 0, frame)

    smooth_pts = np.round(dense).astype(np.int32)
    cv2.polylines(frame, [smooth_pts], False, C_BLUE_TRAJ, 9, cv2.LINE_AA)
    cv2.polylines(frame, [smooth_pts], False, (255, 250, 220), 3, cv2.LINE_AA)
    return frame


def draw_trajectory_debug_overlay(
    frame: np.ndarray,
    vis: VisualFrame,
) -> np.ndarray:
    """
    Full trajectory debug HUD (--trajectory-debug):

    RED    = raw YOLO centers
    GREEN  = filtered / accepted centers
    BLUE   = final trajectory knots
    YELLOW = rejected detections (+ reason)
    """
    if not vis.trajectory_debug:
        return frame

    # Rejected (yellow)
    for item in vis.trajectory_rejected_pixels:
        if len(item) >= 3:
            x, y, reason = float(item[0]), float(item[1]), str(item[2])
        else:
            x, y, reason = float(item[0]), float(item[1]), "rejected"
        cv2.circle(frame, (int(x), int(y)), 5, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame, reason[:18], (int(x) + 6, int(y) - 4),
            FONT_PLAIN, 0.35, (0, 220, 255), 1, cv2.LINE_AA,
        )

    # Raw detections (red)
    for x, y in vis.trajectory_raw_pixels:
        cv2.circle(frame, (int(x), int(y)), 4, C_RED, -1, cv2.LINE_AA)

    # Filtered / accepted (green)
    for x, y in vis.trajectory_filtered_pixels:
        cv2.circle(frame, (int(x), int(y)), 4, C_GREEN, -1, cv2.LINE_AA)

    # Final trail knots (blue)
    for x, y in vis.trajectory_pixels:
        cv2.circle(frame, (int(x), int(y)), 3, (255, 120, 40), -1, cv2.LINE_AA)

    # Velocity vectors (cyan) from filtered points
    filt = vis.trajectory_filtered_pixels
    vels = vis.trajectory_velocities
    for i, (x, y) in enumerate(filt):
        if i >= len(vels):
            break
        vx, vy = vels[i]
        if abs(vx) + abs(vy) < 0.5:
            continue
        ex = int(x + vx * 3.0)
        ey = int(y + vy * 3.0)
        cv2.arrowedLine(
            frame, (int(x), int(y)), (ex, ey),
            C_NEON_CYAN, 1, cv2.LINE_AA, tipLength=0.35,
        )

    conf = vis.tracking_confidence
    n_acc = vis.trajectory_accepted_count or len(vis.trajectory_filtered_pixels)
    n_rej = vis.trajectory_rejected_count or len(vis.trajectory_rejected_pixels)
    buf_sz = len(vis.trajectory_pixels)
    lines = [
        f"frame {vis.frame_idx}  conf={conf:.2f}",
        f"accepted={n_acc}  rejected={n_rej}  buffer={buf_sz}",
        "RED=raw GREEN=filtered BLUE=trail YELLOW=rejected",
    ]
    box_h = 14 + len(lines) * 20
    cv2.rectangle(frame, (8, 8), (460, box_h), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (14, 28 + i * 20), FONT_PLAIN, 0.48, C_WHITE, 1, cv2.LINE_AA)
    return frame


def _dedupe_points(points: np.ndarray, min_dist_px: float = 2.0) -> np.ndarray:
    if len(points) <= 1:
        return points
    kept = [points[0]]
    for pt in points[1:]:
        if float(np.linalg.norm(pt - kept[-1])) >= min_dist_px:
            kept.append(pt)
    return np.asarray(kept, dtype=np.float64)


def _rdp_points(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification for stable broadcast splines."""
    if len(points) <= 2:
        return points

    start = points[0]
    end = points[-1]
    line = end - start
    line_len = float(np.linalg.norm(line))
    if line_len <= 1e-6:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        rel = points - start
        dists = np.abs(np.cross(line, rel) / line_len)

    idx = int(np.argmax(dists))
    if float(dists[idx]) > epsilon:
        left = _rdp_points(points[:idx + 1], epsilon)
        right = _rdp_points(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def _catmull_rom_chain(points: np.ndarray, samples_per_segment: int = 18) -> np.ndarray:
    """Centripetal Catmull-Rom chain that passes through the given knots."""
    if len(points) <= 2:
        return _resample_by_distance(points, max(2, samples_per_segment))

    pts = np.asarray(points, dtype=np.float64)
    out: List[np.ndarray] = []
    for i in range(len(pts) - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, len(pts) - 1)]

        for j in range(samples_per_segment):
            t = j / float(samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(point)
    out.append(pts[-1])
    return np.asarray(out, dtype=np.float64)


def _resample_by_distance(points: np.ndarray, n_samples: int) -> np.ndarray:
    points = _dedupe_points(np.asarray(points, dtype=np.float64), min_dist_px=0.5)
    if len(points) <= 2:
        return points

    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        return points

    samples = np.linspace(0.0, total, max(2, n_samples))
    xs = np.interp(samples, cumulative, points[:, 0])
    ys = np.interp(samples, cumulative, points[:, 1])
    return np.column_stack([xs, ys])


def _broadcast_quadratic_phase(points: np.ndarray, n_samples: int = 90) -> np.ndarray:
    """
    Broadcast-style flight phase: one clean quadratic passing through start,
    middle, and end. This removes detector jitter while keeping the curve
    anchored to the observed ball path.
    """
    pts = _dedupe_points(np.asarray(points, dtype=np.float64), min_dist_px=1.0)
    if len(pts) <= 2:
        return _resample_by_distance(pts, n_samples)

    p0 = pts[0]
    p2 = pts[-1]

    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(diffs)])
    mid_idx = int(np.searchsorted(cumulative, cumulative[-1] * 0.5))
    mid_idx = int(np.clip(mid_idx, 1, len(pts) - 2))
    pm = pts[mid_idx]

    control = 2.0 * pm - 0.5 * (p0 + p2)
    dist = max(float(np.linalg.norm(p2 - p0)), 1.0)
    max_offset = dist * 0.65
    line_mid = 0.5 * (p0 + p2)
    offset = control - line_mid
    offset_len = float(np.linalg.norm(offset))
    if offset_len > max_offset:
        control = line_mid + offset * (max_offset / offset_len)

    t = np.linspace(0.0, 1.0, max(3, n_samples))[:, None]
    curve = ((1.0 - t) ** 2) * p0 + 2.0 * (1.0 - t) * t * control + (t ** 2) * p2
    return curve


# ---------------------------------------------------------------------------
# Bounce Target Marker (concentric rings)
# ---------------------------------------------------------------------------

def draw_bounce_arrow(
    frame:  np.ndarray,
    pos:    Tuple[float, float],
) -> np.ndarray:
    """Draw a clean target bounce marker with concentric rings."""
    cx, cy = int(pos[0]), int(pos[1])

    # Outer ring
    cv2.circle(frame, (cx, cy), 16, C_BLUE_TRAJ, 2, cv2.LINE_AA)
    # Inner filled dot
    cv2.circle(frame, (cx, cy), 6, C_BLUE_GLOW, -1, cv2.LINE_AA)
    # White center
    cv2.circle(frame, (cx, cy), 2, C_WHITE, -1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Ball Position Indicator (neon halo ring)
# ---------------------------------------------------------------------------

def draw_ball_indicator(
    frame:     np.ndarray,
    pos:       Tuple[float, float],
    predicted: bool   = False,
) -> np.ndarray:
    """Draw a glowing tracking cursor with a surrounding neon ring."""
    cx, cy = int(pos[0]), int(pos[1])

    if predicted:
        outer_color = C_NEON_CYAN
        inner_color = (0, 180, 180)
    else:
        outer_color = C_BLUE_TRAJ
        inner_color = C_BLUE_GLOW

    # Outer neon ring (blended)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), 14, outer_color, 2, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), 5, inner_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Core white dot
    cv2.circle(frame, (cx, cy), 2, C_WHITE, -1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Frosted-Glass Stat Cards (Swing / Spin / Speed) - FullTrack AI Style
# ---------------------------------------------------------------------------

def draw_fulltrack_hud(
    frame:   np.ndarray,
    vis:     VisualFrame,
) -> np.ndarray:
    """Draw frosted-glass stat cards matching FullTrack AI reference style.

    Layout: Three stacked cards in top-left corner:
      [Swing]  0.2 sf   (i)
      [Spin]   1.8°     (i)
      [Speed]  56 km/h  ≡○
    """
    h, w = frame.shape[:2]

    # Card dimensions
    CARD_X = 20
    CARD_W = 180
    CARD_H = 65
    CARD_GAP = 6
    CARD_START_Y = 55

    cards = []

    # Card 1: Swing
    swing_val = vis.swing_deg
    swing_str = f"{swing_val:.1f}" if swing_val > 0 else "0.0"
    cards.append(("Swing", swing_str, "sf"))

    # Card 2: Spin (turn angle in degrees, per PDF: drift/turn estimation)
    spin_val = vis.spin_rpm
    spin_str = f"{spin_val:.1f}" if spin_val > 0 else "0.0"
    cards.append(("Drift", spin_str, "deg"))  # drift/turn angle in degrees (per PDF)

    # Card 3: Speed — only show a number when it is a real measurement.
    # A low-confidence "prior" (e.g. the 110 km/h fallback) must NOT be shown as
    # if it were measured, so it renders as "~est" instead of a misleading value.
    speed_val = vis.speed_kmh
    if speed_val > 0 and vis.speed_reliable:
        speed_str = f"{int(speed_val)}"
    elif speed_val > 0:
        speed_str = "~est"
    else:
        speed_str = "---"
    cards.append(("Speed", speed_str, "km/h"))

    for i, (label, value, unit) in enumerate(cards):
        y_top = CARD_START_Y + i * (CARD_H + CARD_GAP)
        y_bot = y_top + CARD_H

        # Frosted glass backing
        overlay = frame.copy()
        cv2.rectangle(overlay, (CARD_X, y_top), (CARD_X + CARD_W, y_bot), C_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Card border (subtle)
        cv2.rectangle(frame, (CARD_X, y_top), (CARD_X + CARD_W, y_bot),
                      C_CARD_BORDER, 1, cv2.LINE_AA)

        # Label (top of card, smaller text)
        cv2.putText(frame, label, (CARD_X + 12, y_top + 22),
                    FONT_PLAIN, 0.50, C_LABEL_GRAY, 1, cv2.LINE_AA)

        # Swing/Spin icon (↻-like, top-right of card)
        icon_x = CARD_X + CARD_W - 32
        icon_y = y_top + 18
        cv2.ellipse(frame, (icon_x, icon_y), (8, 8), 0, 30, 330, C_WHITE, 1, cv2.LINE_AA)
        # Arrow tip on the arc
        cv2.line(frame, (icon_x + 6, icon_y - 5), (icon_x + 9, icon_y - 2), C_WHITE, 1, cv2.LINE_AA)
        cv2.line(frame, (icon_x + 6, icon_y - 5), (icon_x + 3, icon_y - 2), C_WHITE, 1, cv2.LINE_AA)

        # Main value (large, bold)
        cv2.putText(frame, value, (CARD_X + 12, y_top + 52),
                    FONT, 0.72, C_WHITE, 1, cv2.LINE_AA)

        # Unit label (next to value)
        value_width = cv2.getTextSize(value, FONT, 0.72, 1)[0][0]
        cv2.putText(frame, f" {unit}", (CARD_X + 14 + value_width, y_top + 52),
                    FONT_PLAIN, 0.42, C_LABEL_GRAY, 1, cv2.LINE_AA)

        # Gold info circle icon (i)
        info_x = CARD_X + CARD_W - 28
        info_y = y_top + 48
        cv2.circle(frame, (info_x, info_y), 10, C_GOLD, -1, cv2.LINE_AA)
        cv2.putText(frame, "i", (info_x - 3, info_y + 5),
                    FONT_PLAIN, 0.40, C_DARK_BG, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Core Render Wrapper
# ---------------------------------------------------------------------------

def render_fulltrack_frame(
    frame: np.ndarray,
    vis:   VisualFrame,
) -> np.ndarray:
    """Apply premium TV-broadcast visual graphics to a single BGR video frame."""
    # 1. Translucent pitch corridor
    frame = draw_pitch_corridor(frame, vis)

    # 2. Virtual stump markers (if any)
    frame = draw_virtual_stumps(frame, vis)

    # 3. Crease boundary lines
    frame = draw_crease_lines(frame, vis)

    # 3. Observed ball path only (no prediction / parabolic fit)
    if len(vis.trajectory_pixels) >= 2:
        frame = draw_trajectory_arc(
            frame,
            vis.trajectory_pixels,
            use_catmull=bool(vis.smooth_render_catmull),
        )

    # 3b. Optional debug overlay
    frame = draw_trajectory_debug_overlay(frame, vis)

    # 4. Frosted-glass stat cards (Swing / Drift / Speed)
    if vis.state != "STANDBY" and (vis.ball_pixel is not None or vis.trajectory_pixels):
        frame = draw_fulltrack_hud(frame, vis)

    return frame
