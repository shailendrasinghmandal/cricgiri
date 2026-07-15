"""
Detect and filter pre-rendered broadcast-style trajectory overlays on input video.

Used when practice clips already contain a painted blue flight path (test_video9+).
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# BGR bands for CricGiri-style light blue arc + cyan glow (see visualizer palette)
def trajectory_overlay_mask_bgr(frame: np.ndarray) -> np.ndarray:
    """Binary mask of likely painted trajectory pixels."""
    if frame is None or frame.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    img = frame[:, :, :3] if frame.ndim == 3 and frame.shape[2] >= 3 else frame
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    light_blue = (b > 135) & (g > 95) & (r < 155) & (b > r + 18)
    cyan_glow = (b > 175) & (g > 130) & (r < 115) & (g > r)
    white_core = (b > 200) & (g > 200) & (r > 180)
    return (light_blue | cyan_glow | white_core).astype(np.uint8)


def painted_overlay_probe_mask(frame: np.ndarray) -> np.ndarray:
    """Strict mask for DETECTING a painted trajectory (not for inpainting).

    The general ``trajectory_overlay_mask_bgr`` (and earlier blue-based probes)
    matched bright sky / white surfaces / blue-teal colour casts, which made the
    probe false-positive on red-soil grounds and net-practice clips (~32-47%
    "painted" on footage with no overlay), wrongly forcing reference-arc mode and
    killing real ball detection.

    The one feature truly unique to a rendered output is the **bright painted RED
    arc** (BGR ≈ (14,29,164): very low green/blue, strongly red-dominant). Red
    CLAY SOIL is orange (much higher green: r-g≈82) and is excluded by the large
    r-g gap, so this mask fires on the painted arc but not on red grounds.
    """
    if frame is None or frame.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    img = frame[:, :, :3] if frame.ndim == 3 and frame.shape[2] >= 3 else frame
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    painted_red = (r > 120) & (g < 75) & (b < 85) & ((r - g) > 105) & ((r - b) > 110)
    return painted_red.astype(np.uint8)


def overlay_fraction(frame: np.ndarray) -> float:
    m = painted_overlay_probe_mask(frame)
    return float(m.sum()) / max(m.size, 1)


def probe_painted_trajectory_overlay(
    cap: cv2.VideoCapture,
    *,
    sample_stride: int = 4,
    max_samples: int = 80,
    trigger_fraction: float = 0.0025,
    rise_delta: float = 0.002,
) -> Tuple[bool, float]:
    """
    Return True if video likely contains a pre-drawn trajectory overlay.

    A painted arc / pitch corridor is drawn AFTER the delivery, so it is absent
    in the early frames and appears later — the painted fraction RISES over time.
    A whole-frame blue/teal colour cast (some net-practice clips) is present from
    frame 0 and stays roughly constant. Relying on absolute fraction alone made
    the probe false-positive on such casts (~32% "painted" on raw footage) and
    forced reference-arc mode, killing real ball detection.

    So we require both: the late-segment painted fraction clears ``trigger_fraction``
    AND it RISES from the early segment by at least ``rise_delta`` (i.e., the
    overlay actually appears, rather than a constant colour cast). ``peak`` is
    still returned for logging.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        return False, 0.0

    early_cut = total * 0.20
    late_cut = total * 0.55
    early: list = []
    late: list = []
    peak = 0.0
    sampled = 0
    for fi in range(0, total, max(1, sample_stride)):
        if sampled >= max_samples:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        frac = overlay_fraction(frame)
        peak = max(peak, frac)
        if fi <= early_cut:
            early.append(frac)
        elif fi >= late_cut:
            late.append(frac)
        sampled += 1

    early_mean = float(np.mean(early)) if early else 0.0
    late_mean = float(np.mean(late)) if late else 0.0
    rise = late_mean - early_mean
    is_painted = (late_mean >= trigger_fraction) and (rise >= rise_delta)
    return is_painted, peak


def suppress_trajectory_overlay(frame: np.ndarray) -> np.ndarray:
    """
    Inpaint painted trajectory pixels so YOLO sees the scene without the blue arc.
    """
    if frame is None or frame.size == 0:
        return frame
    m = trajectory_overlay_mask_bgr(frame)
    frac = float(m.sum()) / max(m.size, 1)
    if frac < 0.002:
        return frame
    mask_u8 = (m * 255).astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return frame
    if frac > 0.08:
        kernel = np.ones((7, 7), np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=2)
    radius = 7 if frac > 0.05 else 5
    return cv2.inpaint(frame, mask_u8, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def center_on_painted_trajectory(
    frame: np.ndarray,
    cx: float,
    cy: float,
    *,
    patch_radius: int = 4,
) -> bool:
    """True if detection center sits on painted trajectory pixels."""
    if frame is None:
        return False
    h, w = frame.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    if ix < 0 or iy < 0 or ix >= w or iy >= h:
        return False
    r = max(1, int(patch_radius))
    x1, x2 = max(0, ix - r), min(w, ix + r + 1)
    y1, y2 = max(0, iy - r), min(h, iy + r + 1)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    m = trajectory_overlay_mask_bgr(patch)
    return float(m.sum()) / max(m.size, 1) >= 0.35
