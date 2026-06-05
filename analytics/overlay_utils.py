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


def overlay_fraction(frame: np.ndarray) -> float:
    m = trajectory_overlay_mask_bgr(frame)
    return float(m.sum()) / max(m.size, 1)


def probe_painted_trajectory_overlay(
    cap: cv2.VideoCapture,
    *,
    sample_stride: int = 4,
    max_samples: int = 80,
    trigger_fraction: float = 0.0075,
) -> Tuple[bool, float]:
    """
    Return True if video likely contains a pre-drawn trajectory overlay.

    ``trigger_fraction`` = min fraction of pixels in any sampled frame (0.75%).
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        return False, 0.0

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
        sampled += 1

    return peak >= trigger_fraction, peak


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
