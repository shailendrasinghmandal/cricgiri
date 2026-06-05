"""
Shared heuristics for cricket-ball labels and detections.

Broadcast overlays (score bugs, logos, watermarks) often sit in corners or
the top strip. Used by label audit, clean-dataset builder, and live pipeline.
"""
from __future__ import annotations

CORNER_MARGIN = 0.12
TOP_STRIP = 0.15
MAX_BALL_AREA_FRAC = 0.03
MIN_BALL_AREA_FRAC = 0.00015

BALL_CLASS_IDS = frozenset({0, 3})


def is_watermark_like_norm(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    *,
    corner_margin: float = CORNER_MARGIN,
    top_strip: float = TOP_STRIP,
) -> bool:
    """True if a normalized YOLO box (0–1) looks like an overlay, not a ball."""
    area = bw * bh
    if area > MAX_BALL_AREA_FRAC or area < MIN_BALL_AREA_FRAC:
        return True
    in_corner_x = cx < corner_margin or cx > 1.0 - corner_margin
    in_corner_y = cy < corner_margin or cy > 1.0 - corner_margin
    if in_corner_x and in_corner_y:
        return True
    if cy < top_strip and in_corner_x:
        return True
    if cy < top_strip * 0.85:
        return True
    return False


def is_broadcast_overlay_pixel(
    cx: float,
    cy: float,
    w: float,
    h: float,
    frame_w: int,
    frame_h: int,
) -> bool:
    if frame_w <= 0 or frame_h <= 0:
        return False
    return is_watermark_like_norm(
        cx / frame_w,
        cy / frame_h,
        w / frame_w,
        h / frame_h,
    )


def label_suspicion_flags(cls: int, cx: float, cy: float, bw: float, bh: float) -> list[str]:
    flags: list[str] = []
    if cls not in BALL_CLASS_IDS:
        flags.append(f"wrong_class_{cls}")
    if is_watermark_like_norm(cx, cy, bw, bh):
        flags.append("overlay_or_bad_geometry")
    return flags
