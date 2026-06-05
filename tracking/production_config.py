"""
Production preset configuration for cricket ball pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tracking.physics_constraints import PhysicsConstraintConfig
from tracking.yolo_inference import YoloInferenceConfig


@dataclass
class ProductionTrackingConfig:
    """High-stability tracking defaults (fast to enable via --production)."""

    ball_confidence: float = 0.35
    scan_confidence: float = 0.01
    inference_imgsz: int = 1280
    yolo_iou: float = 0.45
    yolo_agnostic_nms: bool = True
    yolo_max_det: int = 3
    half_precision: bool = True

    max_missing_frames: int = 18
    enable_byte_track: bool = True
    enable_tracker_optical_flow: bool = True
    enable_predictive_roi: bool = True
    track_interpolation: str = "cubic"
    hybrid_optical_flow: bool = True

    smooth_render_catmull: bool = True
    savgol_window: int = 7
    ema_alpha: float = 0.42

    use_enhanced_detection: bool = True
    enable_tta: bool = False
    enable_roi_detect: bool = True
    dynamic_confidence: bool = True

    physics: PhysicsConstraintConfig = field(default_factory=PhysicsConstraintConfig)

    def yolo_config(self, device: str | None = None) -> YoloInferenceConfig:
        return YoloInferenceConfig(
            conf=float(self.ball_confidence),
            iou=float(self.yolo_iou),
            agnostic_nms=bool(self.yolo_agnostic_nms),
            max_det=int(self.yolo_max_det),
            imgsz=int(self.inference_imgsz),
            device=device,
            half=bool(self.half_precision),
            scan_conf_floor=float(self.scan_confidence),
        )


def apply_production_cli(args) -> None:
    """Map production preset onto argparse Namespace (non-destructive overrides)."""
    p = ProductionTrackingConfig()
    args.production = True
    args.ball_confidence = float(p.ball_confidence)
    args.inference_imgsz = int(p.inference_imgsz)
    args.yolo_iou = float(p.yolo_iou)
    args.yolo_agnostic_nms = bool(p.yolo_agnostic_nms)
    args.yolo_max_det = int(p.yolo_max_det)
    args.half_precision = bool(p.half_precision)
    args.max_missing_frames = int(p.max_missing_frames)
    args.byte_track = True
    args.tracker_optical_flow = True
    args.hybrid_optical_flow = True
    args.track_interpolation = str(p.track_interpolation)
    args.enhanced_detection = bool(p.use_enhanced_detection)
    args.tta = bool(p.enable_tta)
    args.roi_detect = bool(p.enable_roi_detect)
    args.dynamic_conf = bool(p.dynamic_confidence)
    args.enable_predictive_roi = bool(p.enable_predictive_roi)
    args.predictive_roi = bool(p.enable_predictive_roi)
    args.smooth_render_catmull = bool(p.smooth_render_catmull)
    args.savgol_window = int(p.savgol_window)
    args.yolo_iou = float(p.yolo_iou)
    args.half_precision = bool(p.half_precision)
