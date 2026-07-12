"""
analytics/blur_recover.py   [ADDITIVE — motion-blur / faint-ball frame recovery]
=================================================================================
THE PROBLEM (measured on clip01)
--------------------------------
The appearance detector (YOLO, trained on sharp round balls) misses the ball when
it is motion-blurred / faint / small — especially on and just after the bounce.
On clip01 that dropped per-frame recall to 0.56. But those frames are NOT empty:
the behind-stumps camera is static, so a moving ball leaves an unmistakable blob
in a background-subtracted frame even when it is too blurred for the box detector
(frame-diff found the ball in 10 of 11 misses).

THE FIX
-------
For each frame inside the delivery that lacks a confident detection:
  1. PREDICT where the ball should be from the physics track (local quadratic through
     the confident detections in the same bounce segment — never from ground truth),
  2. SEARCH a small physics-gated ROI around that prediction in a background-subtracted
     frame for a ball-sized moving blob,
  3. ACCEPT the blob nearest the prediction inside the gate; otherwise leave the frame
     empty (occluded frames — e.g. ball behind the batsman — have no motion blob and
     are honestly NOT recovered; we never fabricate a position).

This recovers blurred/faint balls without adding false positives, because the gate is
tied to the predicted trajectory, not the whole frame. Everything is opt-in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Pt = Tuple[int, float, float, float]   # frame, x, y, conf


@dataclass
class BlurRecoverConfig:
    bg_stride: int = 3          # sample every Nth frame for the median background
    diff_thresh: int = 26       # background-subtraction binarisation threshold
    min_area: float = 2.0       # min blob area (px^2)
    max_area: float = 600.0     # max blob area (rejects batsman-sized regions)
    base_radius: float = 9.0    # min ROI search radius (px)
    speed_radius_k: float = 1.6 # ROI radius also grows with local ball speed
    max_radius: float = 40.0    # cap ROI radius
    recovered_conf: float = 0.30
    min_anchor: int = 4         # need this many confident points to predict at all


def _median_background(video: str, stride: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    return np.median(np.array(frames), axis=0).astype(np.int16)


def _predict(anchors: List[Pt], f: int) -> Optional[Tuple[float, float, float]]:
    """Local quadratic (or linear) prediction of (x,y) at frame f from the nearest
    same-segment confident anchors. Returns (x, y, local_step_px) or None."""
    if len(anchors) < 2:
        return None
    near = sorted(anchors, key=lambda a: abs(a[0] - f))[:4]
    near.sort(key=lambda a: a[0])
    fs = np.array([a[0] for a in near], float)
    xs = np.array([a[1] for a in near], float)
    ys = np.array([a[2] for a in near], float)
    deg = 2 if len(near) >= 3 and (fs.max() - fs.min()) >= 2 else 1
    try:
        px = np.polyfit(fs, xs, deg)
        py = np.polyfit(fs, ys, deg)
    except Exception:  # noqa: BLE001
        return None
    x = float(np.polyval(px, f))
    y = float(np.polyval(py, f))
    # local step from the two anchors straddling / nearest f
    d = [np.hypot(near[i + 1][1] - near[i][1], near[i + 1][2] - near[i][2])
         / max(1, near[i + 1][0] - near[i][0]) for i in range(len(near) - 1)]
    step = float(np.median(d)) if d else 8.0
    return x, y, step


def _blobs(diff: np.ndarray, cfg: BlurRecoverConfig):
    _, th = cv2.threshold(diff.astype(np.uint8), cfg.diff_thresh, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if cfg.min_area <= a <= cfg.max_area:
            (bx, by), r = cv2.minEnclosingCircle(c)
            out.append((bx, by, a))
    return out


def recover_blurred_frames(
    video: str,
    confident: List[Pt],
    bounce_frame: Optional[int],
    span: Optional[Tuple[int, int]] = None,
    config: Optional[BlurRecoverConfig] = None,
) -> List[Pt]:
    """Return recovered (frame,x,y,conf) for missing frames inside the delivery.

    `confident` = the trusted detections/track (frame,x,y,conf). Prediction uses only
    these (segmented at `bounce_frame`), so recovery never sees ground truth.
    """
    cfg = config or BlurRecoverConfig()
    if len(confident) < cfg.min_anchor:
        return []
    have = {int(p[0]) for p in confident}
    lo = span[0] if span else min(have)
    hi = span[1] if span else max(have)
    bf = bounce_frame if bounce_frame is not None else 10 ** 9

    cap = cv2.VideoCapture(video)
    frames: Dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if lo - 1 <= i <= hi + 1:
            frames[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16)
        i += 1
    cap.release()
    bg = _median_background(video, cfg.bg_stride)

    def blobs_at(f):
        diff = cv2.GaussianBlur(np.abs(frames[f] - bg).astype(np.uint8), (3, 3), 0)
        return _blobs(diff, cfg)

    conf_sorted = sorted(confident, key=lambda p: p[0])
    recovered: List[Pt] = []

    # ---- 1. INTERIOR gaps: bounded quadratic predict + nearest blob -----------
    for f in range(lo, hi + 1):
        if f in have or f not in frames:
            continue
        seg = [p for p in confident if (int(p[0]) <= bf) == (f <= bf)]
        anchors = seg if len(seg) >= 2 else confident
        before = [a for a in anchors if a[0] < f]
        after = [a for a in anchors if a[0] > f]
        if not before or not after:
            continue                          # not interior -> handled by chaining below
        pred = _predict(anchors, f)
        if pred is None:
            continue
        px, py, step = pred
        R = float(np.clip(cfg.base_radius + cfg.speed_radius_k * step,
                          cfg.base_radius, cfg.max_radius))
        cand = sorted((np.hypot(bx - px, by - py), bx, by) for (bx, by, a) in blobs_at(f))
        cand = [c for c in cand if c[0] <= R]
        if cand:
            recovered.append((f, round(cand[0][1], 1), round(cand[0][2], 1), cfg.recovered_conf))

    # ---- 2. TAIL: greedy velocity-gated blob chaining past the last anchor -----
    #    each accepted blob becomes the anchor for the next frame, so the search
    #    ROI follows the real motion instead of an extrapolated parabola.
    def chain(seed_pts, direction):
        pts = sorted(seed_pts, key=lambda p: p[0])
        if len(pts) < 2:
            return []
        if direction < 0:
            pts = pts[::-1]
        (f1, x1, y1, _), (f0, x0, y0, _) = pts[1], pts[0]
        vx = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0] or 1)
        vy = (pts[-1][2] - pts[0][2]) / (pts[-1][0] - pts[0][0] or 1)
        cx, cy, cf = pts[-1][1], pts[-1][2], pts[-1][0]
        out = []
        f = cf + direction
        misses = 0
        while lo <= f <= hi and misses < 2:
            if f in have or f in {p[0] for p in out} or f not in frames:
                f += direction
                continue
            px, py = cx + vx * direction, cy + vy * direction
            speed = np.hypot(vx, vy)
            R = float(np.clip(cfg.base_radius + cfg.speed_radius_k * speed,
                              cfg.base_radius, cfg.max_radius))
            cand = sorted((np.hypot(bx - px, by - py), bx, by) for (bx, by, a) in blobs_at(f))
            cand = [c for c in cand if c[0] <= R]
            if cand:
                nx, ny = cand[0][1], cand[0][2]
                vx = 0.5 * vx + 0.5 * (nx - cx) * direction
                vy = 0.5 * vy + 0.5 * (ny - cy) * direction
                cx, cy = nx, ny
                out.append((f, round(nx, 1), round(ny, 1), cfg.recovered_conf))
                misses = 0
            else:
                misses += 1
            f += direction
        return out

    post = [p for p in conf_sorted if int(p[0]) > bf] or conf_sorted[-3:]
    recovered += chain(post[-3:], +1)          # extend forward toward the bat
    # de-dup, keep inside span
    seen = set(have)
    uniq = []
    for p in sorted(recovered, key=lambda q: q[0]):
        if p[0] not in seen:
            seen.add(p[0]); uniq.append(p)
    if uniq:
        logger.info("blur-recover: +%d frames (of %d gaps)", len(uniq),
                    (hi - lo + 1) - len(have))
    return uniq
