"""
Centralized Ultralytics YOLO inference for cricket ball detection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, List, Optional

import numpy as np

from tracking.track_ball import Detection, parse_yolo_detections

logger = logging.getLogger(__name__)


@dataclass
class YoloInferenceConfig:
    """Production-oriented YOLO predict() parameters."""

    conf: float = 0.35
    iou: float = 0.45
    agnostic_nms: bool = True
    max_det: int = 3
    imgsz: int = 640
    device: Optional[str] = None
    half: bool = False
    augment: bool = False
    scan_conf_floor: float = 0.01
    class_id: int = 0


def run_ball_yolo(
    model: Any,
    frame: np.ndarray,
    frame_idx: int,
    cfg: YoloInferenceConfig,
    *,
    min_conf: Optional[float] = None,
) -> List[Detection]:
    """
    Run YOLO with consistent NMS settings and optional FP16.

    ``min_conf`` is the post-parse floor (can be lower than ``cfg.conf`` for
    permissive cache builds).
    """
    if model is None:
        return []

    floor = float(cfg.scan_conf_floor if min_conf is None else min_conf)
    predict_conf = min(float(cfg.conf), floor)

    kwargs = dict(
        conf=predict_conf,
        iou=float(cfg.iou),
        agnostic_nms=bool(cfg.agnostic_nms),
        max_det=int(cfg.max_det),
        imgsz=int(cfg.imgsz),
        device=cfg.device,
        verbose=False,
    )
    if cfg.half:
        kwargs["half"] = True
    if cfg.augment:
        kwargs["augment"] = True

    try:
        import torch
        with torch.no_grad():
            results = model(frame, **kwargs)
    except Exception:
        results = model(frame, **kwargs)

    return parse_yolo_detections(
        results,
        frame_idx,
        class_id=int(cfg.class_id),
        min_conf=floor,
    )


def merge_ball_detections(
    primary: List[Detection],
    secondary: List[Detection],
    *,
    merge_dist_px: float = 28.0,
    max_keep: int = 5,
) -> List[Detection]:
    """
    Combine detections from two YOLO weights (inference ensemble).

    Keeps the higher-confidence box when two detections overlap; adds non-overlapping
    boxes from the secondary model (e.g. blur-trained backup).
    """
    merged: List[Detection] = list(primary)
    for det in secondary:
        replaced = False
        for i, ex in enumerate(merged):
            if math.hypot(det.cx - ex.cx, det.cy - ex.cy) <= merge_dist_px:
                if det.conf > ex.conf:
                    merged[i] = det
                replaced = True
                break
        if not replaced:
            merged.append(det)
    merged.sort(key=lambda d: -float(d.conf))
    return merged[: max(1, int(max_keep))]


def run_ball_yolo_ensemble(
    primary_model: Any,
    secondary_model: Any,
    frame: np.ndarray,
    frame_idx: int,
    cfg: YoloInferenceConfig,
    *,
    min_conf: Optional[float] = None,
    secondary_class_id: int = 0,
) -> List[Detection]:
    """
    Run two weight files and merge detections at inference time.

    This is the practical substitute for averaging checkpoints: ``ball_best.pt``
    (1-class) and ``ball_best_backup.pt`` (4-class) share a backbone but not the
    same detection head, so a single merged ``.pt`` is unsafe without retraining.
    """
    if primary_model is None and secondary_model is None:
        return []
    d1: List[Detection] = []
    d2: List[Detection] = []
    if primary_model is not None:
        d1 = run_ball_yolo(primary_model, frame, frame_idx, cfg, min_conf=min_conf)
    if secondary_model is not None:
        sec_cfg = (
            replace(cfg, class_id=int(secondary_class_id))
            if int(secondary_class_id) != int(cfg.class_id)
            else cfg
        )
        d2 = run_ball_yolo(
            secondary_model, frame, frame_idx, sec_cfg, min_conf=min_conf,
        )
    return merge_ball_detections(d1, d2, max_keep=max(int(cfg.max_det), 5))
