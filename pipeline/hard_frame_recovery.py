"""
Hard-frame ball recovery: SAHI-style ROI tiling + light deblur + bbox enlargement.

Triggered only when normal YOLO misses or returns weak confidence on frames
where the tracker already knows the approximate ball position (blur / motion streak).

Not run on every frame — keeps latency acceptable for MVP video processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np

from tracking.detection import Detection
from tracking.yolo_inference import YoloInferenceConfig, run_ball_yolo, run_ball_yolo_ensemble

logger = logging.getLogger(__name__)


@dataclass
class HardFrameRecoveryConfig:
    enabled: bool = False
    """Run recovery when best detection conf is below this (or none)."""
    conf_trigger: float = 0.12
    tile_size: int = 320
    tile_overlap: float = 0.25
    roi_size: int = 480
    enable_deblur: bool = True
    bbox_enlarge_factor: float = 1.35
    tiled_imgsz: int = 640
    scan_conf_floor: float = 0.005


def sharpen_for_blur(frame: np.ndarray) -> np.ndarray:
    """Light unsharp mask — helps motion-blur streaks without heavy cost."""
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(frame, 1.45, blurred, -0.45, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _crop_roi(
    frame: np.ndarray,
    cx: float,
    cy: float,
    size: int,
) -> Tuple[np.ndarray, float, float]:
    h, w = frame.shape[:2]
    half = size // 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(w, int(cx + half))
    y2 = min(h, int(cy + half))
    if x2 - x1 < 64 or y2 - y1 < 64:
        return frame, 0.0, 0.0
    return frame[y1:y2, x1:x2].copy(), float(x1), float(y1)


def _tile_slices(
    w: int,
    h: int,
    tile_size: int,
    overlap: float,
) -> List[Tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) tiles covering the ROI."""
    step = max(32, int(tile_size * (1.0 - overlap)))
    tiles: List[Tuple[int, int, int, int]] = []
    y = 0
    while y < h:
        x = 0
        y2 = min(h, y + tile_size)
        if y2 - y < 48 and y > 0:
            break
        while x < w:
            x2 = min(w, x + tile_size)
            if x2 - x < 48 and x > 0:
                break
            tiles.append((x, y, x2, y2))
            if x2 >= w:
                break
            x += step
        if y2 >= h:
            break
        y += step
    return tiles or [(0, 0, w, h)]


def _dedupe(detections: List[Detection], dist_px: float = 20.0) -> List[Detection]:
    if len(detections) <= 1:
        return detections
    ordered = sorted(detections, key=lambda d: -d.conf)
    kept: List[Detection] = []
    for d in ordered:
        if all((d.cx - k.cx) ** 2 + (d.cy - k.cy) ** 2 > dist_px ** 2 for k in kept):
            kept.append(d)
    return kept


def enlarge_blur_centroid(det: Detection, factor: float) -> Detection:
    """
    Expand bbox symmetrically — motion blur often under-boxes the streak.
    Centroid stays fixed; confidence slightly discounted.
    """
    f = max(1.0, float(factor))
    return Detection(
        frame_idx=det.frame_idx,
        cx=det.cx,
        cy=det.cy,
        conf=float(det.conf) * 0.97,
        w=float(det.w) * f,
        h=float(det.h) * f,
    )


def run_tiled_yolo(
    model: Any,
    image: np.ndarray,
    frame_idx: int,
    ycfg: YoloInferenceConfig,
    *,
    min_conf: float,
    tile_size: int,
    tile_overlap: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    alt_model: Any = None,
    ensemble: bool = False,
) -> List[Detection]:
    """SAHI-style inference on overlapping tiles within an ROI."""
    ih, iw = image.shape[:2]
    all_dets: List[Detection] = []
    tiled_cfg = YoloInferenceConfig(
        conf=ycfg.conf,
        iou=ycfg.iou,
        agnostic_nms=ycfg.agnostic_nms,
        max_det=max(int(ycfg.max_det), 5),
        imgsz=int(ycfg.imgsz),
        device=ycfg.device,
        half=ycfg.half,
        augment=ycfg.augment,
        scan_conf_floor=min_conf,
        class_id=ycfg.class_id,
    )
    for x1, y1, x2, y2 in _tile_slices(iw, ih, tile_size, tile_overlap):
        tile = image[y1:y2, x1:x2]
        if tile.size == 0:
            continue
        if ensemble and alt_model is not None:
            dets = run_ball_yolo_ensemble(
                model, alt_model, tile, frame_idx, tiled_cfg, min_conf=min_conf,
            )
        else:
            dets = run_ball_yolo(model, tile, frame_idx, tiled_cfg, min_conf=min_conf)
        for d in dets:
            d.cx += x1 + offset_x
            d.cy += y1 + offset_y
            all_dets.append(d)
    return _dedupe(all_dets)


def should_trigger_hard_recovery(
    detections: List[Detection],
    cfg: HardFrameRecoveryConfig,
    tracker_pos: Optional[Tuple[float, float]],
) -> bool:
    if not cfg.enabled or tracker_pos is None:
        return False
    if not detections:
        return True
    best = max(detections, key=lambda d: d.conf)
    return float(best.conf) < float(cfg.conf_trigger)


def recover_hard_frame(
    model: Any,
    frame: np.ndarray,
    frame_idx: int,
    ycfg: YoloInferenceConfig,
    cfg: HardFrameRecoveryConfig,
    tracker_pos: Tuple[float, float],
    *,
    alt_model: Any = None,
    ensemble: bool = False,
) -> List[Detection]:
    """
    Run tiled + deblurred detection centred on the tracker's predicted position.
    """
    cx, cy = tracker_pos
    roi, ox, oy = _crop_roi(frame, cx, cy, cfg.roi_size)
    if cfg.enable_deblur:
        roi = sharpen_for_blur(roi)

    min_conf = min(float(cfg.scan_conf_floor), float(ycfg.scan_conf_floor))
    tiled_cfg = YoloInferenceConfig(
        conf=ycfg.conf,
        iou=ycfg.iou,
        agnostic_nms=ycfg.agnostic_nms,
        max_det=ycfg.max_det,
        imgsz=int(cfg.tiled_imgsz),
        device=ycfg.device,
        half=ycfg.half,
        augment=False,
        scan_conf_floor=min_conf,
        class_id=ycfg.class_id,
    )
    dets = run_tiled_yolo(
        model,
        roi,
        frame_idx,
        tiled_cfg,
        min_conf=min_conf,
        tile_size=int(cfg.tile_size),
        tile_overlap=float(cfg.tile_overlap),
        offset_x=ox,
        offset_y=oy,
        alt_model=alt_model,
        ensemble=ensemble,
    )
    if cfg.bbox_enlarge_factor > 1.0:
        dets = [enlarge_blur_centroid(d, cfg.bbox_enlarge_factor) for d in dets]

    if dets:
        logger.debug(
            "Hard-frame recovery frame=%d | tiles on ROI %dx%d | hits=%d best=%.3f",
            frame_idx, roi.shape[1], roi.shape[0], len(dets),
            max(d.conf for d in dets),
        )
    return dets


def merge_primary_and_recovery(
    primary: List[Detection],
    recovery: List[Detection],
    *,
    prefer_recovery_if_empty: bool = True,
) -> List[Detection]:
    if not recovery:
        return primary
    if not primary and prefer_recovery_if_empty:
        return recovery
    merged = list(primary) + list(recovery)
    return _dedupe(merged)
