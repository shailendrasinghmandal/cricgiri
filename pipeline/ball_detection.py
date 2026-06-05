"""
pipeline/ball_detection.py
============================
Production ball detection: high-res YOLO, ROI crop, TTA, dynamic confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.track_ball import Detection
from tracking.yolo_inference import (
    YoloInferenceConfig,
    run_ball_yolo,
    run_ball_yolo_ensemble,
)
from pipeline.hard_frame_recovery import (
    HardFrameRecoveryConfig,
    merge_primary_and_recovery,
    recover_hard_frame,
    should_trigger_hard_recovery,
)

logger = logging.getLogger(__name__)


@dataclass
class BallDetectionConfig:
    """Detection settings (mirrors PipelineConfig detection fields)."""
    ball_confidence: float = 0.15
    inference_imgsz: int = 1280
    device: Optional[str] = None
    scan_confidence: float = 0.01
    yolo_iou: float = 0.45
    yolo_agnostic_nms: bool = True
    yolo_max_det: int = 3
    half_precision: bool = False
    yolo_augment: bool = False
    enable_tta: bool = True
    enable_roi: bool = True
    roi_base_size: int = 280
    roi_max_size: int = 520
    roi_expand_on_miss: int = 40
    dynamic_confidence: bool = True
    dynamic_conf_min: float = 0.08
    dynamic_conf_max: float = 0.22
    tta_brightness_delta: int = 18
    hard_frame_recovery: HardFrameRecoveryConfig = field(
        default_factory=HardFrameRecoveryConfig,
    )


@dataclass
class ROIState:
    """Dynamic search window centred on last known ball."""
    cx: float = 0.0
    cy: float = 0.0
    size: int = 280
    active: bool = False
    miss_streak: int = 0


class BallDetector:
    """
    Wraps YOLO ball inference with ROI + TTA + dynamic thresholding.

    Flow:
      full frame OR ROI crop → YOLO @ imgsz → merge TTA variants → NMS-like dedupe
    """

    def __init__(
        self,
        model: Any,
        config: BallDetectionConfig,
        *,
        alt_model: Any = None,
    ) -> None:
        self.model = model
        self.alt_model = alt_model
        self.ensemble_enabled = False
        self.cfg = config
        self._roi = ROIState(size=config.roi_base_size)
        self._recent_confs: List[float] = []
        self._effective_conf = float(config.ball_confidence)

    def reset(self) -> None:
        self._roi = ROIState(size=self.cfg.roi_base_size)
        self._recent_confs.clear()
        self._effective_conf = float(self.cfg.ball_confidence)

    @property
    def effective_confidence(self) -> float:
        return self._effective_conf

    def _update_dynamic_conf(self, detections: List[Detection]) -> None:
        if not self.cfg.dynamic_confidence:
            return
        if detections:
            self._recent_confs.append(float(max(d.conf for d in detections)))
            self._recent_confs = self._recent_confs[-30:]
        if len(self._recent_confs) < 5:
            return
        med = float(np.median(self._recent_confs))
        # Weak recent hits → lower gate; strong hits → raise slightly.
        target = med * 0.85
        target = max(self.cfg.dynamic_conf_min, min(self.cfg.dynamic_conf_max, target))
        self._effective_conf = 0.7 * self._effective_conf + 0.3 * target

    def _run_yolo(
        self,
        image: np.ndarray,
        frame_idx: int,
        min_conf: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> List[Detection]:
        if self.model is None:
            return []
        ycfg = YoloInferenceConfig(
            conf=float(self.cfg.ball_confidence),
            iou=float(self.cfg.yolo_iou),
            agnostic_nms=bool(self.cfg.yolo_agnostic_nms),
            max_det=int(self.cfg.yolo_max_det),
            imgsz=int(self.cfg.inference_imgsz),
            device=self.cfg.device,
            half=bool(self.cfg.half_precision),
            augment=bool(self.cfg.yolo_augment),
            scan_conf_floor=float(min_conf),
        )
        if self.alt_model is not None and self.ensemble_enabled:
            dets = run_ball_yolo_ensemble(
                self.model,
                self.alt_model,
                image,
                frame_idx,
                ycfg,
                min_conf=min_conf,
                secondary_class_id=0,
            )
        else:
            dets = run_ball_yolo(self.model, image, frame_idx, ycfg, min_conf=min_conf)
        if offset_x or offset_y:
            for d in dets:
                d.cx += offset_x
                d.cy += offset_y
        return dets

    def _tta_variants(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float, float, bool]]:
        """Return (image, offset_x, offset_y, flip_x) variants for TTA."""
        variants: List[Tuple[np.ndarray, float, float, bool]] = [(frame, 0.0, 0.0, False)]
        if not self.cfg.enable_tta:
            return variants

        h, w = frame.shape[:2]
        # Horizontal flip (mirror x back after detect).
        variants.append((cv2.flip(frame, 1), 0.0, 0.0, True))

        # Brightness boost / cut.
        for delta in (-self.cfg.tta_brightness_delta, self.cfg.tta_brightness_delta):
            bright = np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            variants.append((bright, 0.0, 0.0, False))

        # Mild scale-up crop centre (helps tiny balls).
        scale = 1.15
        nh, nw = int(h * scale), int(w * scale)
        scaled = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        y0 = max(0, (nh - h) // 2)
        x0 = max(0, (nw - w) // 2)
        crop = scaled[y0 : y0 + h, x0 : x0 + w]
        if crop.shape[0] == h and crop.shape[1] == w:
            variants.append((crop, 0.0, 0.0, False))
        return variants

    @staticmethod
    def _dedupe(detections: List[Detection], dist_px: float = 22.0) -> List[Detection]:
        if len(detections) <= 1:
            return detections
        ordered = sorted(detections, key=lambda d: -d.conf)
        kept: List[Detection] = []
        for d in ordered:
            if all(
                (d.cx - k.cx) ** 2 + (d.cy - k.cy) ** 2 > dist_px ** 2
                for k in kept
            ):
                kept.append(d)
        return kept

    def _apply_flip(self, dets: List[Detection], frame_w: int) -> List[Detection]:
        out: List[Detection] = []
        for d in dets:
            out.append(Detection(
                frame_idx=d.frame_idx,
                cx=float(frame_w - 1 - d.cx),
                cy=d.cy,
                conf=d.conf,
                w=d.w,
                h=d.h,
            ))
        return out

    def _roi_crop(
        self, frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        h, w = frame.shape[:2]
        if not self._roi.active:
            return frame, 0.0, 0.0

        half = int(self._roi.size // 2)
        x1 = max(0, int(self._roi.cx - half))
        y1 = max(0, int(self._roi.cy - half))
        x2 = min(w, int(self._roi.cx + half))
        y2 = min(h, int(self._roi.cy + half))
        if x2 - x1 < 80 or y2 - y1 < 80:
            return frame, 0.0, 0.0
        crop = frame[y1:y2, x1:x2]
        return crop, float(x1), float(y1)

    def _update_roi(
        self,
        detections: List[Detection],
        frame_w: int,
        frame_h: int,
        tracker_pos: Optional[Tuple[float, float]],
    ) -> None:
        if not self.cfg.enable_roi:
            return

        anchor: Optional[Tuple[float, float]] = None
        if detections:
            best = max(detections, key=lambda d: d.conf)
            anchor = (best.cx, best.cy)
            self._roi.miss_streak = 0
        elif tracker_pos is not None:
            anchor = tracker_pos
            self._roi.miss_streak += 1

        if anchor is None:
            if self._roi.miss_streak > 8:
                self._roi.active = False
            return

        self._roi.active = True
        self._roi.cx = float(np.clip(anchor[0], 0, frame_w - 1))
        self._roi.cy = float(np.clip(anchor[1], 0, frame_h - 1))

        if self._roi.miss_streak > 0:
            grow = self.cfg.roi_expand_on_miss * min(self._roi.miss_streak, 4)
            self._roi.size = min(self.cfg.roi_max_size, self.cfg.roi_base_size + grow)
        else:
            self._roi.size = self.cfg.roi_base_size

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        tracker_pos: Optional[Tuple[float, float]] = None,
    ) -> List[Detection]:
        """Run detection; returns all candidates above scan_confidence."""
        h, w = frame.shape[:2]
        scan_conf = min(self.cfg.scan_confidence, self._effective_conf)

        infer_img, ox, oy = frame, 0.0, 0.0
        if self.cfg.enable_roi and (self._roi.active or tracker_pos is not None):
            if tracker_pos and not self._roi.active:
                self._roi.cx, self._roi.cy = tracker_pos
                self._roi.active = True
            infer_img, ox, oy = self._roi_crop(frame)

        all_dets: List[Detection] = []
        for variant, vx, vy, flipped in self._tta_variants(infer_img):
            dets = self._run_yolo(variant, frame_idx, scan_conf, ox + vx, oy + vy)
            if flipped:
                dets = self._apply_flip(dets, w)
            all_dets.extend(dets)

        merged = self._dedupe(all_dets)
        hf_cfg = self.cfg.hard_frame_recovery
        if should_trigger_hard_recovery(merged, hf_cfg, tracker_pos):
            ycfg = YoloInferenceConfig(
                conf=float(self.cfg.ball_confidence),
                iou=float(self.cfg.yolo_iou),
                agnostic_nms=bool(self.cfg.yolo_agnostic_nms),
                max_det=int(self.cfg.yolo_max_det),
                imgsz=int(self.cfg.inference_imgsz),
                device=self.cfg.device,
                half=bool(self.cfg.half_precision),
                augment=bool(self.cfg.yolo_augment),
                scan_conf_floor=float(scan_conf),
            )
            recovery = recover_hard_frame(
                self.model,
                frame,
                frame_idx,
                ycfg,
                hf_cfg,
                tracker_pos,  # type: ignore[arg-type]
                alt_model=self.alt_model if self.ensemble_enabled else None,
                ensemble=self.ensemble_enabled,
            )
            merged = merge_primary_and_recovery(merged, recovery)
            merged = self._dedupe(merged)
        self._update_dynamic_conf(merged)
        self._update_roi(merged, w, h, tracker_pos)
        return merged

    def roi_box(self, frame_shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int, int]]:
        if not self._roi.active:
            return None
        h, w = frame_shape[:2]
        half = int(self._roi.size // 2)
        x1 = max(0, int(self._roi.cx - half))
        y1 = max(0, int(self._roi.cy - half))
        x2 = min(w, int(self._roi.cx + half))
        y2 = min(h, int(self._roi.cy + half))
        return x1, y1, x2, y2
