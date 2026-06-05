"""
High-accuracy preset: maximize trajectory + analytics scores (~90 target).

Balances recall (lower scan gate, cache→tracker, 1280px) with controlled
false-positive rejection (no aggressive production physics filter).
"""

from __future__ import annotations

from dataclasses import dataclass

from tracking.yolo_inference import YoloInferenceConfig


@dataclass
class HighAccuracyConfig:
    """Tuned from conf010_eval failure analysis (sparse YOLO + heavy bridging)."""

    ball_confidence: float = 0.14
    scan_confidence: float = 0.01
    tracker_low_conf: float = 0.06
    inference_imgsz: int = 1280
    yolo_iou: float = 0.45
    yolo_agnostic_nms: bool = True
    yolo_max_det: int = 5
    half_precision: bool = True
    yolo_augment: bool = True

    max_missing_frames: int = 10
    bridge_max_gap_frames: int = 5
    enable_cache_kalman_rebuild: bool = True
    cache_tracker_fallback: bool = True
    skip_phase_slice_when_sparse: bool = True
    min_real_for_phase_slice: int = 8

    enable_byte_track: bool = True
    enable_tracker_optical_flow: bool = True
    hybrid_optical_flow: bool = True
    optical_flow_max_gap_frames: int = 14
    track_interpolation: str = "off"

    use_enhanced_detection: bool = True
    enable_tta: bool = False
    enable_roi_detect: bool = True
    dynamic_confidence: bool = True
    enable_predictive_roi: bool = False

    savgol_window: int = 5
    ema_alpha: float = 0.44
    speed_tof_bounce_conf: float = 0.40

    def yolo_config(self, device: str | None = None) -> YoloInferenceConfig:
        return YoloInferenceConfig(
            conf=float(self.ball_confidence),
            iou=float(self.yolo_iou),
            agnostic_nms=bool(self.yolo_agnostic_nms),
            max_det=int(self.yolo_max_det),
            imgsz=int(self.inference_imgsz),
            device=device,
            half=bool(self.half_precision),
            augment=bool(self.yolo_augment),
            scan_conf_floor=float(self.scan_confidence),
        )


def apply_high_accuracy_cli(args) -> None:
    """Apply preset onto argparse Namespace."""
    h = HighAccuracyConfig()
    args.high_accuracy = True
    args.ball_confidence = float(h.ball_confidence)
    args.inference_imgsz = int(h.inference_imgsz)
    args.yolo_iou = float(h.yolo_iou)
    args.yolo_max_det = int(h.yolo_max_det)
    args.half_precision = bool(h.half_precision)
    args.max_missing_frames = int(h.max_missing_frames)
    args.byte_track = True
    args.tracker_optical_flow = True
    args.hybrid_optical_flow = True
    args.track_interpolation = str(h.track_interpolation)
    args.enhanced_detection = True
    args.tta = True
    args.roi_detect = True
    args.dynamic_conf = True
    args.savgol_window = int(h.savgol_window)
    args.skip_false_start_trim = True
    args.production = False
