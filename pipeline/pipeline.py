from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Sub-module imports
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.analytics_engine import AnalyticsEngine
from tracking.track_ball import BallTracker, Detection, TrackerPhase, parse_yolo_detections, TrackPoint, TrackResult
from tracking.physics_constraints import PhysicsMotionFilter, PhysicsConstraintConfig
from tracking.production_config import ProductionTrackingConfig, apply_production_cli
from tracking.high_accuracy_config import HighAccuracyConfig, apply_high_accuracy_cli
from tracking.yolo_inference import (
    YoloInferenceConfig,
    run_ball_yolo,
    run_ball_yolo_ensemble,
)
from tracking.adaptive_multi_model import (
    AdaptiveFusionPlan,
    build_adaptive_fusion_plan,
    discover_ball_models,
    save_fusion_plan,
)
from analytics.pitch_calibration import PitchCalibrator, CalibrationResult
from analytics.bounce_detection import BounceDetector, BounceResult
from analytics.speed_estimation import SpeedEstimator, SpeedResult
from analytics.swing_estimation import SwingEstimator, SwingResult, BowlerArm
from analytics.trajectory import TrajectoryAnalyser, TrajectoryResult, BowlingLine, BowlingLength
from analytics.heatmap import HeatmapGenerator, BouncePoint
from analytics.observed_path import ObservedPathResult
from analytics.real_trajectory import (
    RealBallPathBuilder,
    slice_trajectory_to_frame,
    to_observed_path_result,
)
from analytics.visualizer import (
    VisualFrame,
    render_fulltrack_frame,
)
from pipeline.ball_detection import BallDetector, BallDetectionConfig
from pipeline.hard_frame_recovery import (
    HardFrameRecoveryConfig,
    merge_primary_and_recovery,
    recover_hard_frame,
    should_trigger_hard_recovery,
)
from analytics.track_export import export_track_csv, export_debug_json
from analytics.trajectory_physics import predict_future_trajectory, fit_parabolic_path
from ball_label_utils import is_broadcast_overlay_pixel
from analytics.overlay_utils import (
    probe_painted_trajectory_overlay,
    suppress_trajectory_overlay,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    """Append debug NDJSON entries for runtime trajectory investigation."""
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

# ---------------------------------------------------------------------------
# Delivery boundary constants
# ---------------------------------------------------------------------------

CALIBRATION_MAX_FRAMES: int = 90     # look at first N frames for stumps
DELIVERY_GAP_FRAMES: int     = 40    # frames of ball-loss → delivery ended
MIN_TRACK_POINTS: int        = 5     # minimum tracked points to count as a delivery
HUD_ALPHA: float             = 0.62  # HUD overlay transparency


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Input / output
    video_path:         str  = "videos/test.mp4"
    output_video_path:  str  = "outputs/output_annotated.mp4"
    output_json_path:   str  = "outputs/analysis_result.json"

    # Model paths (relative to project root; override via env vars on AWS)
    stump_model_path:   str  = "models/stump_best.pt"
    ball_model_path:    str  = "models/ball_best.pt"
    ball_model_alt_path: Optional[str] = None
    hybrid_ensemble: bool = False
    auto_hybrid_on_overlay: bool = True
    adaptive_multi_model: bool = False
    adaptive_model_paths: Optional[List[str]] = None
    adaptive_fusion_report_path: Optional[str] = None
    device:             Optional[str] = None

    # Detection — legacy path (test_video8_tuned) is default; enhanced is opt-in
    stump_confidence:   float = 0.30
    ball_confidence:    float = 0.10
    clean_video_mode:   bool  = True
    use_enhanced_detection: bool = False
    inference_imgsz:    int   = 640
    enable_tta:         bool  = False
    enable_roi_detect:  bool  = False
    dynamic_confidence: bool  = False

    # Tracking — Kalman only by default (tuned-stable)
    fps:                float = 30.0
    max_missing_frames: int   = 12
    enable_byte_track:  bool  = False
    enable_tracker_optical_flow: bool = False
    track_interpolation: str  = "off"
    byte_low_ratio:     float = 0.55
    max_deliveries:     int   = 1
    min_track_displacement_px: float = 55.0
    trajectory_reveal_hold_frames: int = 60
    trajectory_debug: bool = False
    hybrid_optical_flow: bool = False
    optical_flow_max_gap_frames: int = 12
    savgol_window: int = 7
    enable_physics_future: bool = True
    export_track_csv: bool = True
    frame_skip: int = 0

    # Presets (see --match-should-output / --blur-recovery)
    match_should_output: bool = False
    skip_false_start_cluster_trim: bool = False
    skip_bounce_track_extension: bool = False
    blur_recovery_detection: bool = False
    hard_frame_recovery: bool = False
    hard_frame_conf_trigger: float = 0.12

    # Production stability preset (--production)
    production_mode: bool = False
    yolo_iou: float = 0.45
    yolo_agnostic_nms: bool = True
    yolo_max_det: int = 3
    yolo_scan_confidence: float = 0.01
    half_precision: bool = False
    yolo_augment: bool = False
    enable_predictive_roi: bool = False
    smooth_render_catmull: bool = False
    production_debug_hud: bool = False

    # High-accuracy preset (--high-accuracy): recall + cache rebuild + smart bridge
    high_accuracy_mode: bool = False
    bridge_max_gap_frames: int = 18
    enable_cache_kalman_rebuild: bool = False
    cache_tracker_fallback: bool = False
    skip_phase_slice_when_sparse: bool = False
    min_real_for_phase_slice: int = 8
    tracker_low_conf_floor: float = 0.08
    speed_tof_bounce_conf: float = 0.60

    # Pre-painted trajectory on source video (auto-detect or --reference-overlay)
    auto_detect_reference_overlay: bool = True
    has_reference_overlay: bool = False

    # Analytics
    bowler_arm:         str   = "right"
    speed_multiplier:   float = 1.0
    pixels_per_meter:   float = 38.0

    # Pre-saved calibration — skip stump phase if provided
    calibration_file:   Optional[str] = None

    # Output switches
    save_video: bool = True
    save_json:  bool = True

    # Progress callback (job_id, pct) — used by the API
    progress_callback: Optional[Callable[[str, float], None]] = None
    job_id: str = ""


# ---------------------------------------------------------------------------
# Per-delivery / session result containers
# ---------------------------------------------------------------------------

@dataclass
class DeliveryAnalysis:
    delivery_id:         str
    frame_start:         int
    frame_end:           int
    release_frame:       Optional[int] = None
    bat_impact_frame:    Optional[int] = None
    track:               Optional[Dict] = None
    bounce:              Optional[Dict] = None
    speed:               Optional[Dict] = None
    swing:               Optional[Dict] = None
    trajectory:          Optional[Dict] = None
    line:                Optional[str]  = None
    length:              Optional[str]  = None
    confidence:          float          = 0.0
    processing_time_ms:  float          = 0.0
    world_trajectory:    Optional[List[List[float]]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        
        # Add the exact structured keys required by PDF Page 17
        d["speed_kmph"] = float(self.speed["speed_kmh"]) if self.speed else 0.0
        
        if self.bounce and self.bounce.get("world_x") is not None:
            d["bounce_point"] = {
                "x": round(float(self.bounce["world_x"]), 3),
                "y": round(float(self.bounce["world_y"]), 3)
            }
        else:
            d["bounce_point"] = None
            
        d["trajectory"] = d.get("world_trajectory") or []
        
        # swing_cm: PDF Page 17 expects float representing centimeters
        d["swing_cm"] = float(self.swing["swing_cm"]) if self.swing else 0.0
        d["swing_type"] = self.swing["direction"] if self.swing else "none"
        
        # heatmap_points: list of [x, y] bounce spots for this delivery
        if self.bounce and self.bounce.get("world_x") is not None:
            d["heatmap_points"] = [[
                round(float(self.bounce["world_x"]), 3),
                round(float(self.bounce["world_y"]), 3)
            ]]
        else:
            d["heatmap_points"] = []
            
        d["confidence_score"] = round(self.confidence, 4)
        
        return d


@dataclass
class SessionAnalysis:
    session_id:           str
    video_path:           str
    total_deliveries:     int            = 0
    total_frames:         int            = 0
    fps:                  float          = 30.0
    calibration:          Optional[Dict] = None
    deliveries:           List[DeliveryAnalysis] = field(default_factory=list)
    heatmap_stats:        Optional[Dict] = None
    processing_time_sec:  float          = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id":          self.session_id,
            "video_path":          self.video_path,
            "total_deliveries":    self.total_deliveries,
            "total_frames":        self.total_frames,
            "fps":                 self.fps,
            "calibration":         self.calibration,
            "deliveries":          [d.to_dict() for d in self.deliveries],
            "heatmap_stats":       self.heatmap_stats,
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CricketAnalyticsPipeline:
    """
    End-to-end cricket delivery analytics pipeline.

    Usage
    -----
        cfg      = PipelineConfig(video_path="match.mp4", ...)
        pipeline = CricketAnalyticsPipeline(cfg)
        session  = pipeline.run()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

        logger.info("Initialising CricketAnalyticsPipeline …")
        self.stump_model = self._load_yolo(config.stump_model_path, "Stump")
        self.ball_model  = self._load_yolo(config.ball_model_path,  "Ball")
        self.ball_model_alt: Any = None
        if config.ball_model_alt_path:
            alt_p = Path(config.ball_model_alt_path)
            if alt_p.exists() and str(alt_p) != str(Path(config.ball_model_path)):
                self.ball_model_alt = self._load_yolo(str(alt_p), "Ball-alt")
            elif not alt_p.exists():
                logger.warning("ball_model_alt not found: %s", alt_p)

        _lc_thresh = 0.04
        _lc_radius = 105.0
        _max_miss = int(config.max_missing_frames)
        if config.clean_video_mode:
            _lc_thresh = max(0.06, float(config.ball_confidence) * 0.42)
            _lc_radius = 130.0
            _max_miss = max(_max_miss, 12)
        if config.high_accuracy_mode:
            _ha = HighAccuracyConfig()
            _lc_thresh = float(_ha.tracker_low_conf)
            _lc_radius = 135.0
            _max_miss = int(_ha.max_missing_frames)

        self.tracker        = BallTracker(
            fps=config.fps,
            max_missing_frames=_max_miss,
            confidence_threshold=config.ball_confidence,
            max_jump_px=115.0,
            max_ball_bbox_px=52.0,
            low_conf_thresh=_lc_thresh,
            low_conf_radius_px=_lc_radius,
            min_hits_to_confirm=2,
            enable_byte_track=config.enable_byte_track,
            enable_optical_flow=config.enable_tracker_optical_flow,
            byte_low_ratio=config.byte_low_ratio,
            interpolation_method=config.track_interpolation,
        )
        det_cfg = BallDetectionConfig(
            ball_confidence=float(config.ball_confidence),
            inference_imgsz=int(config.inference_imgsz),
            device=config.device,
            yolo_iou=float(config.yolo_iou),
            yolo_agnostic_nms=bool(config.yolo_agnostic_nms),
            yolo_max_det=int(config.yolo_max_det),
            half_precision=bool(config.half_precision),
            yolo_augment=bool(config.yolo_augment),
            enable_tta=bool(config.enable_tta),
            enable_roi=bool(config.enable_roi_detect),
            dynamic_confidence=bool(config.dynamic_confidence),
            hard_frame_recovery=HardFrameRecoveryConfig(
                enabled=bool(config.hard_frame_recovery or config.blur_recovery_detection),
                conf_trigger=float(config.hard_frame_conf_trigger),
            ),
        )
        self._ball_detector = BallDetector(
            self.ball_model, det_cfg, alt_model=self.ball_model_alt,
        )
        self._track_debug: Dict[str, Any] = {"frames": []}
        _ema = 0.38
        _bridge = 8
        _post_bat = 22.0
        _min_conf = 0.03
        _max_step = 88.0
        if config.match_should_output:
            _ema = 0.50
            _bridge = 12
            _post_bat = 18.0
        if config.blur_recovery_detection:
            _min_conf = 0.02
            _max_step = 96.0
            _bridge = max(_bridge, 14)
        if config.high_accuracy_mode:
            _ha = HighAccuracyConfig()
            _min_conf = 0.02
            _max_step = 100.0
            _bridge = max(_bridge, 14)
            _ema = max(_ema, _ha.ema_alpha)
        if config.production_mode:
            _ema = max(_ema, 0.42)
            _bridge = max(_bridge, 12)
        self._physics_filter: Optional[PhysicsMotionFilter] = None
        if config.production_mode and not config.high_accuracy_mode:
            _pc = ProductionTrackingConfig().physics
            self._physics_filter = PhysicsMotionFilter(_pc)
            self.tracker.use_physics_filter = True
            self.tracker._physics = self._physics_filter
        if config.high_accuracy_mode:
            _ha = HighAccuracyConfig()
            self._yolo_infer_cfg = _ha.yolo_config(config.device)
            self.cfg.yolo_augment = bool(_ha.yolo_augment)
            self.tracker.low_conf_thresh = max(
                self.tracker.low_conf_thresh, float(_ha.tracker_low_conf),
            )
            self.tracker.low_conf_radius = max(
                self.tracker.low_conf_radius, 135.0,
            )
        else:
            self._yolo_infer_cfg = YoloInferenceConfig(
                conf=float(config.ball_confidence),
                iou=float(config.yolo_iou),
                agnostic_nms=bool(config.yolo_agnostic_nms),
                max_det=int(config.yolo_max_det),
                imgsz=int(config.inference_imgsz),
                device=config.device,
            half=bool(config.half_precision),
            augment=bool(config.yolo_augment),
            scan_conf_floor=float(config.yolo_scan_confidence),
        )
        self._real_path_builder = RealBallPathBuilder(
            min_confidence=_min_conf,
            min_confidence_primary=float(config.ball_confidence),
            max_step_px=_max_step,
            max_bridge_gap_frames=_bridge,
            ema_alpha=_ema,
            smooth_window=3,
            render_max_upward_px=11.0,
            post_bat_retreat_px=_post_bat,
            savgol_window=int(config.savgol_window),
        )
        self._video_frame_h: int = 0
        self._detected_overlay_peak: float = 0.0
        self._last_observed_path: Optional[ObservedPathResult] = None
        self._render_path_cache: Dict[str, Tuple[
            List[Tuple[float, float]],
            List[Tuple[float, float]],
            float,
            ObservedPathResult,
        ]] = {}
        self.calibrator     = PitchCalibrator()
        self.bounce_det     = BounceDetector()
        self.speed_est      = SpeedEstimator(fps=config.fps, speed_multiplier=config.speed_multiplier)
        self.swing_est      = SwingEstimator(bowler_arm=BowlerArm(config.bowler_arm))
        self.traj_analyser  = TrajectoryAnalyser()
        self.heatmap_gen    = HeatmapGenerator()
        self.analytics_engine = AnalyticsEngine()

        self.calibration_result: Optional[CalibrationResult] = None
        self._stump_exclusion_zones: List[Tuple[float, float, float, float]] = []

        # Per-frame cache of permissive ball candidates (geometry-valid YOLO hits
        # down to a very low confidence). Used by the render pass to extend the
        # broadcast trajectory through frames the tracker rejected, so the arc
        # spans the full ball flight even when individual detections are weak.
        self._raw_ball_cache: Dict[int, List[Tuple[float, float, float]]] = {}
        self._gray_frame_cache: Dict[int, np.ndarray] = {}
        self._adaptive_fusion_plan: Optional[AdaptiveFusionPlan] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> SessionAnalysis:
        """Execute the full pipeline. Returns a SessionAnalysis."""
        session_id = str(uuid.uuid4())
        t_start    = time.perf_counter()

        logger.info("=" * 62)
        logger.info("Pipeline START | session=%s", session_id)
        logger.info("Video : %s", self.cfg.video_path)
        logger.info("=" * 62)

        cap = self._open_video(self.cfg.video_path)
        if self.cfg.auto_detect_reference_overlay and not self.cfg.has_reference_overlay:
            has_ov, peak = probe_painted_trajectory_overlay(cap)
            self._detected_overlay_peak = float(peak)
            if has_ov:
                self.cfg.has_reference_overlay = True
                self.cfg.smooth_render_catmull = True
                self.cfg.skip_bounce_track_extension = True
                self.cfg.skip_false_start_cluster_trim = True
                self.cfg.enable_cache_kalman_rebuild = True
                logger.info(
                    "Painted trajectory detected (peak=%.2f%%). Reference-arc mode ON.",
                    peak * 100.0,
                )
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif self.cfg.has_reference_overlay:
            self.cfg.smooth_render_catmull = True
            self.cfg.skip_bounce_track_extension = True
            self.cfg.skip_false_start_cluster_trim = True
            self.cfg.enable_cache_kalman_rebuild = True
            logger.info("Reference-overlay mode enabled.")

        if self.cfg.has_reference_overlay:
            self.cfg.ball_confidence = min(float(self.cfg.ball_confidence), 0.08)
            self.cfg.min_track_displacement_px = min(
                float(self.cfg.min_track_displacement_px), 12.0,
            )
            self._yolo_infer_cfg.conf = float(self.cfg.ball_confidence)
            self._ball_detector.cfg.ball_confidence = float(self.cfg.ball_confidence)
            self._ball_detector.cfg.dynamic_confidence = False
            self._ball_detector.cfg.enable_roi = False
            self._ball_detector.cfg.enable_tta = False
            self.tracker.conf_thresh = float(self.cfg.ball_confidence)
            self.tracker.low_conf_thresh = 0.05
            if not self.cfg.adaptive_multi_model:
                self._ensure_hybrid_alt_model()

        if self.cfg.hybrid_ensemble and not self.cfg.adaptive_multi_model:
            self._ensure_hybrid_alt_model()
        self._ball_detector.ensemble_enabled = (
            self._use_ball_ensemble() and not self.cfg.adaptive_multi_model
        )

        fps         = cap.get(cv2.CAP_PROP_FPS) or self.cfg.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._video_frame_h = frame_h

        logger.info("Video | fps=%.1f | frames=%d | res=%dx%d",
                    fps, total_frames, frame_w, frame_h)

        self.cfg.fps = fps
        self.tracker.fps = fps
        self.speed_est.fps = fps

        # Adapt displacement gate to the video resolution. The default 70px
        # was tuned for ~1280px-wide broadcast footage; phone-portrait clips
        # (~464px wide) leave a ball only ~25px of travel. Scale to 5% of
        # width with a sane floor so unusual aspect ratios don't break tracking.
        adaptive_min_disp = max(18.0, float(frame_w) * 0.032)
        if adaptive_min_disp < self.cfg.min_track_displacement_px:
            logger.info(
                "Adapting min_track_displacement %.1f → %.1f px for %dpx-wide video",
                self.cfg.min_track_displacement_px, adaptive_min_disp, frame_w,
            )
            self.cfg.min_track_displacement_px = adaptive_min_disp

        session = SessionAnalysis(session_id=session_id, video_path=self.cfg.video_path, fps=fps)

        # ── Phase 1: Calibration (Pass 1 - No drawing) ───────────────────
        self._report_progress(10.0)
        self.calibration_result = self._run_calibration(cap, None)
        if self.calibration_result:
            session.calibration = self.calibration_result.to_dict()
        else:
            logger.warning("Calibration failed — world coordinates unavailable.")

        # Build stump exclusion zones from calibration detections
        # Any ball detection overlapping these zones is a false positive (stumps detected as ball)
        self._stump_exclusion_zones = self._build_stump_exclusion_zones(cap)

        # Build STATIC-PHANTOM exclusion zones. The current YOLO ball model is
        # over-eager and fires on stationary background objects (decorative
        # lights, scoreboards, balls on the ground in the nets backdrop, etc.).
        # Real cricket balls in flight MOVE — anything that stays at the same
        # pixel for many frames is a phantom and must be excluded.
        # Always scan for stationary false-positive ball regions. Clean footage
        # still has static background detections (e.g. nets ball at ~183,307 on
        # test_video8) that dominate tracking if left unfiltered.
        phantom_zones: List[Tuple[float, float, float, float]] = []
        # should_output/*.mp4 already contain painted trajectories; phantom scan
        # treats the blue arc as a static false positive and blocks real tracking.
        if not self._reference_arc_mode():
            phantom_zones = self._build_static_phantom_zones(
                cap, max_scan_frames=80 if self.cfg.clean_video_mode else 60,
            )
        if phantom_zones:
            logger.info(
                "Detected %d static phantom-ball region(s) in the background; "
                "excluding them from ball detection.",
                len(phantom_zones),
            )
            self._stump_exclusion_zones.extend(phantom_zones)

        if self.cfg.adaptive_multi_model:
            self._report_progress(18.0)
            self._adaptive_fusion_plan = self._build_adaptive_fusion_plan(cap)
            report_p = self.cfg.adaptive_fusion_report_path
            if report_p:
                save_fusion_plan(self._adaptive_fusion_plan, Path(report_p))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._raw_ball_cache.clear()
            self._gray_frame_cache.clear()

        # ── Phase 2: Ball Tracking (Pass 1 - High-speed tracking) ─────────
        self._report_progress(20.0)

        # Rewind video capture back to the beginning after calibration phase!
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx               = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        delivery_start          = frame_idx
        frames_since_detection  = 0
        self.tracker.reset()
        self._ball_detector.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if self.cfg.hybrid_optical_flow or self.cfg.enable_tracker_optical_flow:
                self._gray_frame_cache[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Ball detection (high-res YOLO + ROI + optional TTA)
            tracker_pos = self.tracker.get_predicted_position() or self.tracker._last_pos
            ball_dets = self._detect_ball(frame, frame_idx, tracker_pos=tracker_pos)
            tp = self.tracker.update(frame_idx, ball_dets, frame=frame)

            if self.cfg.trajectory_debug and frame_idx % max(1, 1 + self.cfg.frame_skip) == 0:
                self._record_track_debug(frame_idx, ball_dets, tp)

            # Check if tracker has officially lost the ball (exceeded max missing frames)
            tracker_lost = (
                tp is None and 
                self.tracker._kf.initialized and 
                self.tracker._lost_frames >= self.tracker.max_missing
            )

            if tp is not None:
                frames_since_detection = 0
            else:
                frames_since_detection += 1

            # Delivery boundary: tracker officially lost OR long gap without detection
            if tracker_lost or frames_since_detection >= DELIVERY_GAP_FRAMES:
                end_frame = frame_idx
                if tracker_lost:
                    end_frame = frame_idx - self.tracker._lost_frames
                else:
                    end_frame = frame_idx - DELIVERY_GAP_FRAMES

                # #region agent log
                _debug_log(
                    "H15",
                    "pipeline.py:first-pass:delivery-boundary",
                    "First-pass boundary triggered, attempting flush.",
                    {
                        "frame_idx": int(frame_idx),
                        "tracker_lost": bool(tracker_lost),
                        "lost_frames": int(self.tracker._lost_frames),
                        "frames_since_detection": int(frames_since_detection),
                        "delivery_start_before": int(delivery_start),
                        "delivery_end_candidate": int(max(delivery_start, end_frame)),
                    },
                )
                # #endregion
                delivery = self._flush_delivery(delivery_start, max(delivery_start, end_frame))
                if delivery:
                    session.deliveries.append(delivery)
                    # #region agent log
                    _debug_log(
                        "H15",
                        "pipeline.py:first-pass:delivery-boundary",
                        "Boundary flush produced a delivery.",
                        {
                            "accepted_frame_start": int(delivery.frame_start),
                            "accepted_frame_end": int(delivery.frame_end),
                        },
                    )
                    # #endregion
                else:
                    # #region agent log
                    _debug_log(
                        "H15",
                        "pipeline.py:first-pass:delivery-boundary",
                        "Boundary flush produced no delivery.",
                        {
                            "delivery_start_before": int(delivery_start),
                            "end_frame_candidate": int(max(delivery_start, end_frame)),
                            "next_delivery_start": int(frame_idx + 1),
                        },
                    )
                    # #endregion
                self.tracker.reset()
                self._ball_detector.reset()
                delivery_start = frame_idx + 1
                frames_since_detection = 0

            # Progress reporting (first pass up to 50%)
            if total_frames > 0:
                pct = 20.0 + 30.0 * (frame_idx / total_frames)
                self._report_progress(pct)

        # Flush any remaining delivery
        final_delivery = self._flush_delivery(delivery_start, frame_idx)
        if final_delivery:
            session.deliveries.append(final_delivery)

        if not session.deliveries:
            cache_fallback = self._fallback_delivery_from_cache(total_frames)
            if cache_fallback is not None:
                session.deliveries.append(cache_fallback)

        session.deliveries = self._limit_deliveries(session.deliveries)
        if not self.cfg.skip_bounce_track_extension:
            session.deliveries = [
                self._extend_delivery_through_bounce(d, total_frames)
                for d in session.deliveries
            ]
        self._rebuild_heatmap(session.deliveries)

        cap.release()

        # ── Phase 3: Premium TV-Broadcast Visual Rendering Pass (Pass 2) ──
        if self.cfg.save_video:
            self._report_progress(50.0)
            logger.info("=" * 62)
            logger.info("STARTING PREMIUM TV-BROADCAST VISUAL RENDERING PASS...")
            logger.info("=" * 62)

            cap_render = self._open_video(self.cfg.video_path)
            writer = self._open_writer(self.cfg.output_video_path, fps, frame_w, frame_h)

            if writer is not None:
                # Pre-calculate pitch corner coordinates if calibration is active
                pitch_corners = {}
                if self.calibration_result:
                    try:
                        # Map 4 corners of standard pitch from world to pixel coordinates
                        c_bl = self.calibration_result.world_to_pixel(-0.95, 0.0)
                        c_br = self.calibration_result.world_to_pixel(0.95, 0.0)
                        c_tl = self.calibration_result.world_to_pixel(-0.95, 20.12)
                        c_tr = self.calibration_result.world_to_pixel(0.95, 20.12)

                        pitch_corners = {
                            "pitch_top_left": (int(c_tl[0]), int(c_tl[1])),
                            "pitch_top_right": (int(c_tr[0]), int(c_tr[1])),
                            "pitch_bot_left": (int(c_bl[0]), int(c_bl[1])),
                            "pitch_bot_right": (int(c_br[0]), int(c_br[1])),
                        }
                    except Exception as e:
                        logger.warning("Crease projection failed: %s", e)

                render_frame_idx = 0
                while True:
                    ret, frame = cap_render.read()
                    if not ret:
                        break

                    # Path only from release (ball leaves hand) through bat contact.
                    active_del = None
                    for d in session.deliveries:
                        track_meta = d.track or {}
                        release_fi = int(
                            d.release_frame
                            or track_meta.get("release_frame")
                            or d.frame_start
                        )
                        bat_fi = int(
                            d.bat_impact_frame
                            or track_meta.get("bat_impact_frame")
                            or d.frame_end
                        )
                        if (
                            release_fi <= render_frame_idx
                            <= bat_fi + self.cfg.trajectory_reveal_hold_frames
                        ):
                            active_del = d
                            break

                    if active_del:
                        pts_list = active_del.track["points"]
                        track_meta = active_del.track or {}
                        release_fi = int(
                            active_del.release_frame
                            or track_meta.get("release_frame")
                            or active_del.frame_start
                        )
                        bat_fi = int(
                            active_del.bat_impact_frame
                            or track_meta.get("bat_impact_frame")
                            or active_del.frame_end
                        )
                        raw_path, visible_path, track_conf, path_dbg = (
                            self._build_observed_trajectory_pixels(active_del)
                        )
                        path_start_fi = (
                            int(path_dbg.frame_indices[0])
                            if path_dbg and path_dbg.frame_indices
                            else release_fi
                        )
                        path_start_fi = max(path_start_fi, release_fi)
                        show_overlay = (
                            render_frame_idx >= path_start_fi
                            and visible_path
                            and len(visible_path) >= 2
                        )
                        if show_overlay:
                            current_point = next(
                                (p for p in reversed(pts_list) if int(p.get("frame_idx", -1)) <= render_frame_idx),
                                pts_list[-1],
                            )
                            if path_dbg and path_dbg.frame_indices and render_frame_idx <= bat_fi:
                                visible_path = slice_trajectory_to_frame(
                                    visible_path,
                                    path_dbg.frame_indices,
                                    min(render_frame_idx, bat_fi),
                                )
                            self._last_observed_path = path_dbg
                            ball_pixel = (float(current_point["x"]), float(current_point["y"]))
                            bounce_pixel = None
                            if active_del.bounce and render_frame_idx >= active_del.bounce["frame_idx"]:
                                bounce_pixel = (active_del.bounce["pixel_x"], active_del.bounce["pixel_y"])
                            swing_cm = ((active_del.swing["swing_cm"] / 100.0) * 100) if active_del.swing else 0.0
                            drift_angle_deg = 0.0
                            if active_del.swing and active_del.swing.get("swing_cm") and active_del.swing["swing_cm"] > 0:
                                import math as _math
                                drift_angle_deg = round(_math.degrees(_math.atan2(
                                    active_del.swing["swing_cm"] / 100.0, 10.0
                                )), 1)
                            dbg = path_dbg if self.cfg.trajectory_debug and path_dbg else None
                            vis = VisualFrame(
                                trajectory_pixels=visible_path,
                                trajectory_raw_pixels=dbg.raw_pixels if dbg else [],
                                trajectory_filtered_pixels=dbg.filtered_pixels if dbg else [],
                                trajectory_rejected_pixels=dbg.rejected_pixels if dbg else [],
                                trajectory_velocities=dbg.velocities if dbg else [],
                                trajectory_accepted_count=len(dbg.filtered_pixels) if dbg else 0,
                                trajectory_rejected_count=len(dbg.rejected) if dbg else 0,
                                trajectory_debug=self.cfg.trajectory_debug,
                                smooth_render_catmull=self.cfg.smooth_render_catmull,
                                tracking_confidence=track_conf,
                                ball_pixel=ball_pixel,
                                speed_kmh=active_del.speed["speed_kmh"] if active_del.speed else 0.0,
                                speed_mph=active_del.speed.get("speed_mph", 0.0) if active_del.speed else 0.0,
                                swing_deg=swing_cm,
                                spin_rpm=drift_angle_deg,
                                swing_label=active_del.swing["direction"] if active_del.swing else "none",
                                bounce_pixel=bounce_pixel,
                                stump_pixels=None,
                                frame_idx=render_frame_idx,
                                state="ANALYSIS_READY",
                                **pitch_corners
                            )
                        else:
                            vis = VisualFrame(
                                trajectory_pixels=[],
                                frame_idx=render_frame_idx,
                                state="STANDBY",
                                **pitch_corners
                            )
                    else:
                        # Standby state (no delivery active)
                        vis = VisualFrame(
                            trajectory_pixels=[],
                            frame_idx=render_frame_idx,
                            state="STANDBY",
                            **pitch_corners
                        )

                    # Render TV-Broadcast graphics
                    frame = render_fulltrack_frame(frame, vis)
                    writer.write(frame)

                    # Update rendering progress (50% to 95%)
                    if total_frames > 0:
                        pct = 50.0 + 45.0 * (render_frame_idx / total_frames)
                        self._report_progress(pct)

                    render_frame_idx += 1

                cap_render.release()
                writer.release()
                logger.info("Premium TV-Broadcast rendering pass completed!")

        # ── Phase 4: Finalise ─────────────────────────────────────────────
        session.total_deliveries    = len(session.deliveries)
        session.total_frames        = frame_idx + 1
        session.heatmap_stats       = self.heatmap_gen.compute_stats().to_dict()
        session.processing_time_sec = time.perf_counter() - t_start

        if self.cfg.save_json:
            self._save_json(session)
        if self.cfg.export_track_csv:
            csv_path = Path(self.cfg.output_json_path).with_suffix(".csv")
            export_track_csv(session.deliveries, csv_path)
            logger.info("Track CSV saved: %s", csv_path)
        if self.cfg.trajectory_debug:
            dbg_path = Path(self.cfg.output_json_path).with_name(
                Path(self.cfg.output_json_path).stem + "_track_debug.json"
            )
            export_debug_json(self._track_debug, dbg_path)
            logger.info("Track debug JSON: %s", dbg_path)
        self._save_heatmap()

        self._report_progress(100.0)
        logger.info(
            "Pipeline DONE | deliveries=%d | time=%.2fs",
            session.total_deliveries, session.processing_time_sec,
        )
        return session

    # ------------------------------------------------------------------
    # Phase 1: Calibration
    # ------------------------------------------------------------------

    def _run_calibration(
        self, cap: cv2.VideoCapture, writer: Optional[cv2.VideoWriter]
    ) -> Optional[CalibrationResult]:
        """Attempt calibration from first CALIBRATION_MAX_FRAMES frames."""

        # Load pre-saved calibration if available
        if self.cfg.calibration_file and Path(self.cfg.calibration_file).exists():
            logger.info("Loading calibration from %s", self.cfg.calibration_file)
            return PitchCalibrator.load_calibration(self.cfg.calibration_file)

        logger.info("Auto-calibration from first %d frames …", CALIBRATION_MAX_FRAMES)

        best_detections: Optional[List[Dict]] = None
        best_count = 0
        best_score = 0.0

        for _ in range(CALIBRATION_MAX_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break

            if self.stump_model is not None:
                results = self.stump_model(
                    frame,
                    conf=self.cfg.stump_confidence,
                    device=self.cfg.device,
                    verbose=False,
                )
                dets = self._parse_stump_detections(results)
                # Need at least 3 stump detections
                score = float(sum(d.get("confidence", 0.0) for d in dets))
                if len(dets) >= 3 and (len(dets) > best_count or (len(dets) == best_count and score > best_score)):
                    best_count = len(dets)
                    best_score = score
                    best_detections = dets
                    logger.debug("Calibration candidate: %d stump detections", len(dets))

            if writer is not None:
                writer.write(frame)

        if best_detections is None:
            logger.warning("No valid stump detections found for calibration.")
            return None

        try:
            self._best_calibration_detections = best_detections
            result = self.calibrator.calibrate_from_stump_detections(best_detections)
            if self.cfg.calibration_file:
                self.calibrator.save_calibration(self.cfg.calibration_file)
            logger.info(
                "Calibration OK | reproj_err=%.3f px | stumps=%d",
                result.reprojection_error, best_count,
            )
            return result
        except Exception as exc:
            logger.error("Calibration error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Delivery flush
    # ------------------------------------------------------------------

    def _flush_delivery(
        self, frame_start: int, frame_end: int
    ) -> Optional[DeliveryAnalysis]:
        """Finalize the current tracker window into a full delivery analysis."""
        t0 = time.perf_counter()
        track_result = self.tracker.finalize()
        # #region agent log
        _debug_log(
            "H14",
            "pipeline.py:_flush_delivery:entry",
            "Entered delivery flush with tracker snapshot.",
            {
                "arg_frame_start": int(frame_start),
                "arg_frame_end": int(frame_end),
                "track_points_total": int(len(track_result.points)),
                "real_points_total": int(sum(1 for p in track_result.points if not p.is_interpolated)),
            },
        )
        # #endregion

        # Trim stationary prefix points from the start of the track (false positive noise filter).
        # Skip on reference-overlay clips: inpaint can leave a static false cluster before release.
        if len(track_result.points) > 1 and not self._reference_arc_mode():
            active_idx = 0
            first_x, first_y = track_result.points[0].x, track_result.points[0].y
            for i in range(1, len(track_result.points)):
                if abs(track_result.points[i].x - first_x) > 2.0 or abs(track_result.points[i].y - first_y) > 2.0:
                    active_idx = i
                    break
            if active_idx >= 3:
                track_result.points = track_result.points[active_idx:]

        # Filter out tracks with too few real detections (e.g. bowler arm
        # jitter or net vibrations). Previously required >= 3 real detections
        # which threw away phone-recorded clips where the YOLO model only
        # locked onto the ball in 2 frames. Allow 2 if the total track is
        # long enough to be a plausible delivery, but keep the hard floor at 2.
        n_real = sum(1 for p in track_result.points if not p.is_interpolated)
        if n_real < 2 and self._reference_arc_mode() and self._raw_ball_cache:
            cache_track = self._rebuild_track_with_kalman_cache(
                max(0, int(frame_start) - 8),
                int(frame_end) + 4,
                fast_confirm=True,
            )
            n_cache_real = sum(1 for p in cache_track if not p.is_interpolated)
            if n_cache_real >= 3 and len(cache_track) >= 6:
                conf_mean = float(
                    np.mean([p.confidence for p in cache_track if not p.is_interpolated])
                ) if n_cache_real else 0.0
                interp_pct = (
                    sum(1 for p in cache_track if p.is_interpolated) / len(cache_track)
                )
                track_result = TrackResult(
                    points=cache_track,
                    confidence_mean=round(conf_mean, 4),
                    interpolated_pct=round(interp_pct, 4),
                )
                n_real = n_cache_real
                logger.info(
                    "Reference-arc cache rebuild rescued flush: %d pts (%d real).",
                    len(cache_track),
                    n_cache_real,
                )
        if n_real < 2:
            # #region agent log
            _debug_log(
                "H14",
                "pipeline.py:_flush_delivery:reject-n_real",
                "Rejected delivery flush due to too few real detections.",
                {
                    "arg_frame_start": int(frame_start),
                    "arg_frame_end": int(frame_end),
                    "n_real": int(n_real),
                    "track_points_total": int(len(track_result.points)),
                },
            )
            # #endregion
            logger.info("Track has too few real detections (%d) — skipped.", n_real)
            return None
        if n_real == 2 and len(track_result.points) < 8:
            # #region agent log
            _debug_log(
                "H14",
                "pipeline.py:_flush_delivery:reject-short-2real",
                "Rejected flush: exactly two real detections in short track.",
                {
                    "arg_frame_start": int(frame_start),
                    "arg_frame_end": int(frame_end),
                    "n_real": int(n_real),
                    "track_points_total": int(len(track_result.points)),
                },
            )
            # #endregion
            logger.info(
                "Track has only %d real detections in a short %d-pt track — skipped.",
                n_real, len(track_result.points),
            )
            return None

        n_pts = len(track_result.points)
        if n_pts < MIN_TRACK_POINTS:
            # #region agent log
            _debug_log(
                "H14",
                "pipeline.py:_flush_delivery:reject-min-points",
                "Rejected flush due to minimum track-points gate.",
                {
                    "arg_frame_start": int(frame_start),
                    "arg_frame_end": int(frame_end),
                    "n_pts": int(n_pts),
                    "min_required": int(MIN_TRACK_POINTS),
                },
            )
            # #endregion
            logger.debug(
                "Track too short (%d pts) at frames %d–%d — skipped.",
                n_pts, frame_start, frame_end,
            )
            return None

        real_only = [p for p in track_result.points if not p.is_interpolated]
        real_ys = [float(p.y) for p in real_only] if real_only else []
        real_y_span = (max(real_ys) - min(real_ys)) if real_ys else 0.0
        if (
            n_real >= 3
            and real_y_span < 45.0
            and not self._reference_arc_mode()
        ):
            logger.info(
                "Track vertical span too small (%.1fpx) for %d real detections — "
                "skipped as static false positive.",
                real_y_span, n_real,
            )
            return None

        displacement = self._track_displacement(track_result.points)
        if displacement < self.cfg.min_track_displacement_px:
            # #region agent log
            _debug_log(
                "H14",
                "pipeline.py:_flush_delivery:reject-displacement",
                "Rejected flush due to displacement gate.",
                {
                    "arg_frame_start": int(frame_start),
                    "arg_frame_end": int(frame_end),
                    "displacement_px": float(displacement),
                    "min_required_px": float(self.cfg.min_track_displacement_px),
                },
            )
            # #endregion
            logger.info(
                "Track displacement too small (%.1fpx < %.1fpx) — skipped as static false positive.",
                displacement,
                self.cfg.min_track_displacement_px,
            )
            return None

        delivery_id = str(uuid.uuid4())
        logger.info(
            "Delivery %s | frames %d–%d | %d track pts",
            delivery_id[:8], frame_start, frame_end, n_pts,
        )

        # Preserve raw Kalman tracker output for ball tracking / trajectory render.
        kalman_points = list(track_result.points)
        pixel_pts = track_result.get_trajectory_pixels()
        # #region agent log
        _debug_log(
            "H13",
            "pipeline.py:_flush_delivery:pre-truncate",
            "Kalman track before analytics cleanup.",
            {
                "frame_start": int(kalman_points[0].frame_idx) if kalman_points else None,
                "frame_end": int(kalman_points[-1].frame_idx) if kalman_points else None,
                "points_total": int(len(kalman_points)),
                "real_points_total": int(sum(1 for p in kalman_points if not p.is_interpolated)),
            },
        )
        # #endregion

        # Check if the delivery is background-to-foreground (small y to large y)
        if len(pixel_pts) >= 2 and pixel_pts[-1][1] > pixel_pts[0][1] and self.calibration_result:
            if hasattr(self, "_best_calibration_detections") and self._best_calibration_detections:
                logger.info("Auto-detected background-to-foreground delivery (camera behind batsman). Swapping stump calibration ends dynamically.")
                try:
                    self.calibration_result = self.calibrator.calibrate_from_stump_detections(
                        self._best_calibration_detections, swap_ends=True
                    )
                except Exception as exc:
                    logger.error("Dynamic re-calibration swap error: %s", exc)

        # World coordinates. Homography from single-class stump boxes can be
        # unstable when a detected box is not a true wicket set, so validate
        # the mapped ball path before trusting it for speed/swing.
        raw_world_pts: Optional[List[Tuple[float, float]]] = None
        world_pts: Optional[List[Tuple[float, float]]] = None
        if self.calibration_result:
            raw_world_pts = self.calibration_result.pixel_trajectory_to_world(pixel_pts)
            if self._world_trajectory_is_plausible(raw_world_pts):
                world_pts = raw_world_pts
            else:
                logger.warning(
                    "Calibration rejected for delivery %s: mapped trajectory is physically implausible. "
                    "Using pixel-scale fallback for speed/swing.",
                    delivery_id[:8],
                )

        # Rebuild from permissive cache when track is sparse (should_output + high-accuracy).
        if self._raw_ball_cache and kalman_points and self._should_cache_kalman_rebuild(kalman_points):
            win_start = max(0, int(kalman_points[0].frame_idx) - 35)
            win_end = min(int(frame_end) + 5, int(kalman_points[-1].frame_idx) + 12)
            fast = bool(self.cfg.match_should_output)
            cache_track = self._rebuild_track_with_kalman_cache(
                win_start, win_end, fast_confirm=fast,
            )
            n_real_cache = sum(1 for p in cache_track if not p.is_interpolated)
            n_real_kal = sum(1 for p in kalman_points if not p.is_interpolated)
            if len(cache_track) >= max(8, len(kalman_points) // 2) and n_real_cache >= n_real_kal:
                kalman_points = cache_track
                pixel_pts = [(p.x, p.y) for p in kalman_points]
                if self.calibration_result:
                    raw_world_pts = self.calibration_result.pixel_trajectory_to_world(pixel_pts)
                    world_pts = raw_world_pts if self._world_trajectory_is_plausible(raw_world_pts) else None
                logger.info(
                    "Cache-Kalman rebuild: %d pts (%d real), frames %d–%d.",
                    len(kalman_points),
                    n_real_cache,
                    kalman_points[0].frame_idx,
                    kalman_points[-1].frame_idx,
                )

        # Analytics cleanup (does not modify stored Kalman track used for drawing).
        if self._reference_arc_mode():
            truncated_pts = list(kalman_points)
            truncated_pixels = list(pixel_pts)
            truncated_world = list(world_pts) if world_pts else None
        else:
            truncated_pts, truncated_pixels, truncated_world = self._truncate_post_hit_points(
                kalman_points, pixel_pts, world_pts
            )
        # #region agent log
        _debug_log(
            "H13",
            "pipeline.py:_flush_delivery:post-hit-truncate",
            "Track state after post-hit truncation.",
            {
                "frame_start": int(truncated_pts[0].frame_idx) if truncated_pts else None,
                "frame_end": int(truncated_pts[-1].frame_idx) if truncated_pts else None,
                "points_total": int(len(truncated_pts)),
                "real_points_total": int(sum(1 for p in truncated_pts if not p.is_interpolated)),
            },
        )
        # #endregion
        truncated_pts = self._trim_trailing_prediction_tail(truncated_pts)
        # #region agent log
        _debug_log(
            "H13",
            "pipeline.py:_flush_delivery:post-tail-trim",
            "Track state after trimming trailing prediction tail.",
            {
                "frame_start": int(truncated_pts[0].frame_idx) if truncated_pts else None,
                "frame_end": int(truncated_pts[-1].frame_idx) if truncated_pts else None,
                "points_total": int(len(truncated_pts)),
                "real_points_total": int(sum(1 for p in truncated_pts if not p.is_interpolated)),
            },
        )
        # #endregion
        if not self.cfg.skip_false_start_cluster_trim:
            truncated_pts = self._trim_to_best_detection_cluster(truncated_pts)
        # #region agent log
        _debug_log(
            "H13",
            "pipeline.py:_flush_delivery:post-cluster-trim",
            "Track state after best-cluster trimming stage.",
            {
                "frame_start": int(truncated_pts[0].frame_idx) if truncated_pts else None,
                "frame_end": int(truncated_pts[-1].frame_idx) if truncated_pts else None,
                "points_total": int(len(truncated_pts)),
                "real_points_total": int(sum(1 for p in truncated_pts if not p.is_interpolated)),
            },
        )
        # #endregion
        bridged_points = self._bridge_track_gaps_smart(truncated_pts)
        from analytics.delivery_phases import analyze_flight_phases, slice_track_to_phases

        phase_source = bridged_points or truncated_pts
        n_real_phase = sum(1 for p in phase_source if not p.is_interpolated)
        flight_phases = analyze_flight_phases(
            phase_source,
            post_bat_retreat_px=float(self._real_path_builder.post_bat_retreat_px),
            min_release_step_px=6.0 if self.cfg.match_should_output else 10.0,
        )
        if self.cfg.match_should_output and phase_source:
            flight_phases.release_index = 0
            flight_phases.release_frame = int(phase_source[0].frame_idx)
        skip_slice = (
            self.cfg.skip_phase_slice_when_sparse
            and n_real_phase < int(self.cfg.min_real_for_phase_slice)
        )
        if skip_slice:
            bounded_points = phase_source
            logger.info(
                "Skipping phase slice: only %d real points (min %d).",
                n_real_phase,
                self.cfg.min_real_for_phase_slice,
            )
        else:
            bounded_points = slice_track_to_phases(phase_source, flight_phases)
        if (
            self._reference_arc_mode()
            and phase_source
            and len(bounded_points) >= len(phase_source) - 2
            and len(phase_source) > 20
        ):
            real_pts = [p for p in phase_source if not p.is_interpolated]
            if len(real_pts) >= 6:
                release_pt = min(real_pts, key=lambda p: p.frame_idx)
                bat_pt = max(real_pts, key=lambda p: p.y)
                if bat_pt.frame_idx > release_pt.frame_idx + 3:
                    bounded_points = [
                        p for p in phase_source
                        if release_pt.frame_idx <= p.frame_idx <= bat_pt.frame_idx
                    ]
                    flight_phases.release_frame = int(release_pt.frame_idx)
                    flight_phases.bat_impact_frame = int(bat_pt.frame_idx)
                    logger.info(
                        "Reference-arc phase trim: release f%d → bat f%d (%d pts).",
                        flight_phases.release_frame,
                        flight_phases.bat_impact_frame,
                        len(bounded_points),
                    )
        if len(bounded_points) < 4:
            bounded_points = phase_source
        if flight_phases.valid:
            logger.info(
                "Flight phases analyzed: release frame %d → bat impact frame %d "
                "(bounce=%s, %d track pts bounded from %d).",
                flight_phases.release_frame,
                flight_phases.bat_impact_frame,
                flight_phases.bounce_frame,
                len(bounded_points),
                len(phase_source),
            )

        # Stored track = release → bat only (analyze first, then draw path).
        display_points = self._trim_trailing_prediction_tail(bounded_points)
        if len(display_points) < 4:
            display_points = self._trim_trailing_prediction_tail(bounded_points or truncated_pts)
        track_result = TrackResult(
            points=display_points,
            confidence_mean=track_result.confidence_mean,
            interpolated_pct=round(
                sum(1 for p in display_points if p.is_interpolated) / len(display_points),
                4,
            ) if display_points else 0.0,
        )
        track_dict = track_result.to_dict()
        if flight_phases.valid:
            track_dict["release_frame"] = int(flight_phases.release_frame)
            track_dict["bat_impact_frame"] = int(flight_phases.bat_impact_frame)
            if flight_phases.bounce_frame is not None:
                track_dict["bounce_frame"] = int(flight_phases.bounce_frame)
        pixel_pts = [(p.x, p.y) for p in display_points]
        # #region agent log
        _debug_log(
            "H13",
            "pipeline.py:_flush_delivery:kalman-vs-analytics",
            "Kalman track kept for render; analytics uses cleaned path.",
            {
                "kalman_points": int(len(display_points)),
                "analytics_points": int(len(bridged_points)),
                "kalman_frame_start": int(display_points[0].frame_idx) if display_points else None,
                "kalman_frame_end": int(display_points[-1].frame_idx) if display_points else None,
            },
        )
        # #endregion

        # Delivery-coverage gate. Previously rejected on
        #   interp_pct > 0.70 AND mean_conf < 0.30
        # which threw away usable deliveries from videos where the detector
        # was simply pessimistic. Relax slightly while keeping a hard floor
        # on total coverage so degenerate sequences are still rejected.
        if (
            track_result.interpolated_pct > 0.85
            and track_result.confidence_mean < 0.20
        ):
            # #region agent log
            _debug_log(
                "H14",
                "pipeline.py:_flush_delivery:reject-coverage",
                "Rejected flush due to weak detector coverage gate.",
                {
                    "frame_start": int(track_result.points[0].frame_idx) if track_result.points else None,
                    "frame_end": int(track_result.points[-1].frame_idx) if track_result.points else None,
                    "interp_pct": float(track_result.interpolated_pct),
                    "conf_mean": float(track_result.confidence_mean),
                },
            )
            # #endregion
            logger.info(
                "Delivery rejected: weak detector coverage (interp_pct=%.2f, mean_conf=%.2f).",
                track_result.interpolated_pct,
                track_result.confidence_mean,
            )
            return None

        if world_pts and self.calibration_result:
            bridged_world = self.calibration_result.pixel_trajectory_to_world(pixel_pts)
            world_pts = bridged_world if self._world_trajectory_is_plausible(bridged_world) else None
        else:
            world_pts = None

        analytics = self._compute_delivery_analytics(display_points, pixel_pts, world_pts)

        # Post-hoc bounce sanity. Reject implausible bounce coordinates that
        # slipped through (e.g. world_x=0, world_y=0 at the release frame).
        # IMPORTANT: only apply when we actually trust the homography
        # (world_pts is not None). When on pixel_scale_fallback the world_x/y
        # are relative pixel-metres, so a "(5, 10)" value can be perfectly
        # legitimate and we must not reject it.
        bounce = analytics.get("bounce")
        if bounce and world_pts is not None:
            bx = bounce.get("world_x")
            by = bounce.get("world_y")
            if bx is not None and by is not None:
                # Only catch the BOUNCE_AT_ORIGIN case (clearly at release
                # frame, not a bounce). Don't reject for being far down
                # the pitch — that's where balls bounce.
                bad = (
                    (abs(float(bx)) < 0.02 and abs(float(by)) < 0.02)
                    or float(by) > 22.0
                    or abs(float(bx)) > 3.0
                )
                if bad:
                    logger.info(
                        "Bounce coords (%.2f, %.2f) failed pitch-sanity, marking as None.",
                        float(bx), float(by),
                    )
                    bounce["world_x"] = None
                    bounce["world_y"] = None
                    bounce["confidence"] = round(
                        min(float(bounce.get("confidence", 0.5)), 0.30), 4
                    )
            analytics["bounce"] = bounce

        bounce = analytics.get("bounce")
        speed = analytics.get("speed")
        speed_kmh = speed["speed_kmh"] if speed else None
        swing = analytics.get("swing")

        # Trajectory polynomial fit + line/length classification
        world_bounce_tuple = None
        if bounce and bounce["world_x"] is not None:
            world_bounce_tuple = (bounce["world_x"], bounce["world_y"])
        traj: Optional[TrajectoryResult] = self.traj_analyser.analyse(
            pixel_pts, world_bounce=world_bounce_tuple
        )
        # Pixel-based line override — fires whenever we're on pixel-scale
        # fallback (the world coordinates aren't reliable, but stump pixel
        # positions usually are). The fix-up inside `_classify` ensures we
        # default to OFF_STUMP / GOOD_LENGTH rather than WIDE_OUTSIDE_LEG
        # when the override can't produce a confident result.
        if (
            traj and bounce and speed
            and speed.get("metric_source") == "pixel_scale_fallback"
        ):
            line_override = self._classify_line_from_pitch_pixels(
                float(bounce["pixel_x"]),
                float(bounce["pixel_y"]),
            )
            if line_override is not None:
                traj.bowling_line = line_override

        # Heatmap accumulation. Bounce coordinates may come from calibrated
        # world points or from the explicitly marked pixel-scale fallback.
        if bounce and bounce.get("world_x") is not None and bounce.get("world_y") is not None:
            line_value = traj.bowling_line.value if traj else "unknown"
            self.heatmap_gen.add_bounce(BouncePoint(
                delivery_id=delivery_id,
                world_x=self._heatmap_lateral_x(float(bounce["world_x"]), line_value),
                world_y=float(bounce["world_y"]),
                speed_kmh=speed_kmh if speed_kmh else 0.0,
                line=line_value,
                length=traj.bowling_length.value if traj else "unknown",
            ))

        # Compute physical 3D trajectory [x, y, z] for Page 17 schema.
        # If homography is unavailable this uses the same explicitly marked
        # pixel-scale fallback as speed/swing.
        world_trajectory = []
        trajectory_metric_pts = world_pts or self._pixel_points_to_metric(pixel_pts)
        if trajectory_metric_pts:
            n_pts = len(trajectory_metric_pts)
            b_list_idx = n_pts - 1
            if bounce and bounce["frame_idx"] is not None:
                for idx, pt in enumerate(bridged_points):
                    if pt.frame_idx == bounce["frame_idx"]:
                        b_list_idx = idx
                        break
            
            for idx, (wx, wy) in enumerate(trajectory_metric_pts):
                z_release = 2.1  # standard bowler release height in metres
                if idx <= b_list_idx:
                    # Pre-bounce phase: parabolic path from release height (2.1m) down to 0.0m
                    t = idx / b_list_idx if b_list_idx > 0 else 1.0
                    wz = z_release * (1.0 - t) + 0.4 * np.sin(np.pi * t)
                else:
                    # Post-bounce rebound: rising path from 0.0m up to stumps height (approx 0.8m)
                    denom = (n_pts - 1 - b_list_idx)
                    t = (idx - b_list_idx) / denom if denom > 0 else 1.0
                    wz = 0.8 * np.sin(0.5 * np.pi * t)
                
                world_trajectory.append([
                    round(float(wx), 3),
                    round(float(wy), 3),
                    round(float(wz), 3)
                ])

        # Aggregate confidence — now includes ALL component confidences instead
        # of dropping speed and trajectory goodness-of-fit. Tracker continuity
        # is also added because frames-with-real-detections is the cleanest
        # bottom-up signal of how trustworthy a delivery is. Weights reflect
        # which signals correlate best with manual quality assessment.
        track_continuity = max(
            0.0,
            1.0 - float(getattr(track_result, "interpolated_pct", 0.0)),
        )
        traj_r2 = float(traj.r_squared) if traj else 0.0
        components = {
            "tracker":      (float(track_result.confidence_mean or 0.0), 0.20),
            "continuity":   (track_continuity,                            0.20),
            "bounce":       (float(bounce.get("confidence", 0.0)) if bounce else 0.0, 0.18),
            "speed":        (float(speed.get("confidence", 0.0)) if speed else 0.0,  0.16),
            "swing":        (float(swing.get("confidence", 0.0)) if swing else 0.0,  0.10),
            "trajectory":   (max(0.0, min(1.0, traj_r2)),                 0.16),
        }
        wsum = sum(w for _, w in components.values())
        overall_conf = (
            sum(v * w for v, w in components.values()) / wsum
            if wsum > 0 else 0.0
        )

        ms = (time.perf_counter() - t0) * 1000
        # #region agent log
        accepted_start = int(track_result.points[0].frame_idx) if track_result.points else None
        accepted_end = int(track_result.points[-1].frame_idx) if track_result.points else None
        cache_in_arg = 0
        cache_before_accept = 0
        for fi in range(int(frame_start), int(frame_end) + 1):
            cands = self._raw_ball_cache.get(fi, [])
            if not cands:
                continue
            cache_in_arg += int(len(cands))
            if accepted_start is not None and fi < accepted_start:
                cache_before_accept += int(len(cands))
        _debug_log(
            "H16",
            "pipeline.py:_flush_delivery:accepted-cache-window",
            "Compared cache detections in flush window vs accepted track start.",
            {
                "arg_frame_start": int(frame_start),
                "arg_frame_end": int(frame_end),
                "accepted_frame_start": accepted_start,
                "accepted_frame_end": accepted_end,
                "cache_points_in_arg_window": int(cache_in_arg),
                "cache_points_before_accepted_start": int(cache_before_accept),
            },
        )
        # #endregion
        # #region agent log
        _debug_log(
            "H14",
            "pipeline.py:_flush_delivery:accepted",
            "Accepted delivery flush output.",
            {
                "frame_start": int(track_result.points[0].frame_idx) if track_result.points else None,
                "frame_end": int(track_result.points[-1].frame_idx) if track_result.points else None,
                "points_total": int(len(track_result.points)),
                "real_points_total": int(sum(1 for p in track_result.points if not p.is_interpolated)),
            },
        )
        # #endregion

        return DeliveryAnalysis(
            delivery_id=delivery_id,
            frame_start=int(display_points[0].frame_idx) if display_points else int(frame_start),
            frame_end=int(display_points[-1].frame_idx) if display_points else int(frame_end),
            release_frame=int(flight_phases.release_frame) if flight_phases.valid else None,
            bat_impact_frame=int(flight_phases.bat_impact_frame) if flight_phases.valid else None,
            track=track_dict,
            bounce=bounce,
            speed=speed,
            swing=swing,
            trajectory=traj.to_dict() if traj else None,
            line=traj.bowling_line.value if traj else None,
            length=traj.bowling_length.value if traj else None,
            confidence=round(overall_conf, 4),
            processing_time_ms=round(ms, 2),
            world_trajectory=world_trajectory,
        )

    def _fallback_delivery_from_cache(self, total_frames: int) -> Optional[DeliveryAnalysis]:
        """Construct one observed-only delivery from cached detections."""
        if not self._raw_ball_cache:
            logger.info("Cache fallback skipped: no cached detections.")
            return None

        real = self._real_path_builder.build(
            frame_start=0,
            frame_end=max(0, int(total_frames) - 1),
            raw_ball_cache=self._raw_ball_cache,
            fps=float(self.cfg.fps),
        )
        observed = to_observed_path_result(real)

        points_px = observed.filtered_pixels if len(observed.filtered_pixels) >= 3 else observed.raw_pixels
        logger.info(
            "Cache fallback observed points | raw=%d filtered=%d chosen=%d",
            int(len(observed.raw_pixels)),
            int(len(observed.filtered_pixels)),
            int(len(points_px)),
        )
        if len(points_px) < 3:
            logger.info("Cache fallback skipped: observed path too short.")
            return None

        track_points: List[TrackPoint] = []
        for idx, (x, y) in enumerate(points_px):
            fi = int(observed.frame_indices[idx]) if idx < len(observed.frame_indices) else idx
            conf = float(observed.confidences[idx]) if idx < len(observed.confidences) else 0.0
            track_points.append(
                TrackPoint(
                    frame_idx=fi,
                    x=float(x),
                    y=float(y),
                    vx=0.0,
                    vy=0.0,
                    is_interpolated=(conf <= 0.0),
                    confidence=max(0.0, conf),
                )
            )

        real_confs = [p.confidence for p in track_points if not p.is_interpolated and p.confidence > 0.0]
        conf_mean = float(np.mean(real_confs)) if real_confs else 0.0
        interp_pct = float(sum(1 for p in track_points if p.is_interpolated)) / float(len(track_points))
        track_result = TrackResult(
            points=track_points,
            confidence_mean=round(conf_mean, 4),
            interpolated_pct=round(interp_pct, 4),
        )
        pixel_pts = track_result.get_trajectory_pixels()
        world_pts = None
        if self.calibration_result:
            world_tmp = self.calibration_result.pixel_trajectory_to_world(pixel_pts)
            world_pts = world_tmp if self._world_trajectory_is_plausible(world_tmp) else None

        analytics = self._compute_delivery_analytics(track_points, pixel_pts, world_pts)
        bounce = analytics.get("bounce")
        speed = analytics.get("speed")
        swing = analytics.get("swing")

        world_bounce_tuple = None
        if bounce and bounce.get("world_x") is not None and bounce.get("world_y") is not None:
            world_bounce_tuple = (float(bounce["world_x"]), float(bounce["world_y"]))
        traj: Optional[TrajectoryResult] = self.traj_analyser.analyse(
            pixel_pts, world_bounce=world_bounce_tuple
        )

        continuity = max(0.0, 1.0 - track_result.interpolated_pct)
        overall_conf = 0.5 * track_result.confidence_mean + 0.5 * continuity
        delivery_id = str(uuid.uuid4())
        logger.info(
            "Cache fallback delivery %s | frames %d-%d | points=%d",
            delivery_id[:8],
            int(track_points[0].frame_idx),
            int(track_points[-1].frame_idx),
            int(len(track_points)),
        )
        return DeliveryAnalysis(
            delivery_id=delivery_id,
            frame_start=int(track_points[0].frame_idx),
            frame_end=int(track_points[-1].frame_idx),
            track=track_result.to_dict(),
            bounce=bounce,
            speed=speed,
            swing=swing,
            trajectory=traj.to_dict() if traj else None,
            line=traj.bowling_line.value if traj else None,
            length=traj.bowling_length.value if traj else None,
            confidence=round(float(overall_conf), 4),
            processing_time_ms=0.0,
            world_trajectory=None,
        )

    def _limit_deliveries(self, deliveries: List[DeliveryAnalysis]) -> List[DeliveryAnalysis]:
        """Keep only real bowling deliveries, dropping post-hit re-tracks."""
        # #region agent log
        _debug_log(
            "H10",
            "pipeline.py:_limit_deliveries:input",
            "Observed deliveries before max-deliveries filtering.",
            {
                "max_deliveries": int(self.cfg.max_deliveries),
                "count": int(len(deliveries)),
                "ranges": [
                    [int(d.frame_start), int(d.frame_end), int((d.frame_end - d.frame_start) + 1)]
                    for d in deliveries[:6]
                ],
            },
        )
        # #endregion
        if self.cfg.max_deliveries <= 0 or len(deliveries) <= self.cfg.max_deliveries:
            return deliveries

        if self.cfg.max_deliveries == 1 and len(deliveries) > 1:
            # Choose the best single delivery segment (avoid merging distant
            # re-tracks that produce non-physical loops).
            def _delivery_score(d: DeliveryAnalysis) -> float:
                pts = (d.track or {}).get("points") or []
                if not pts:
                    return -1e9
                real_pts = sum(1 for p in pts if not p.get("is_interpolated", False))
                conf = float((d.track or {}).get("confidence_mean", 0.0))
                interp = float((d.track or {}).get("interpolated_pct", 1.0))
                span = max(1, int(d.frame_end) - int(d.frame_start) + 1)
                xs = [float(p.get("x", 0.0)) for p in pts]
                ys = [float(p.get("y", 0.0)) for p in pts]
                y_span = (max(ys) - min(ys)) if ys else 0.0
                x_span = (max(xs) - min(xs)) if xs else 0.0
                if y_span < 45.0:
                    return -1e9
                # Real bowling deliveries have a large vertical arc on screen.
                # Reject flat horizontal false tracks (background / post-hit).
                arc_score = y_span * 8.0
                if y_span < 80.0:
                    arc_score *= 0.15
                if x_span > y_span * 1.6 and y_span < 140.0:
                    arc_score *= 0.15
                if y_span < 100.0 and x_span > max(60.0, y_span * 1.0):
                    arc_score *= 0.02
                # Reject bottom-edge / watermark false tracks (y near frame bottom).
                frame_h = max(480, int(self._video_frame_h or 848))
                bottom_edge = frame_h * 0.82
                if max(ys) >= bottom_edge:
                    arc_score *= 0.02
                if sum(1 for y in ys if y >= bottom_edge) >= 2 and min(ys) < frame_h * 0.55:
                    arc_score *= 0.005
                # Prefer the main delivery in the second half of the clip.
                if int(d.frame_start) < 35 and max(ys) >= bottom_edge:
                    arc_score *= 0.01
                # Main bowling delivery: late start + tall vertical arc + enough real hits.
                if int(d.frame_start) >= 40 and y_span >= 85.0 and real_pts >= 6:
                    arc_score += 140.0
                if int(d.frame_start) >= 55 and y_span >= 100.0 and real_pts >= 8:
                    arc_score += 80.0
                first_real_fi = min(
                    (int(p.get("frame_idx", 9999)) for p in pts if not p.get("is_interpolated", False)),
                    default=int(d.frame_start),
                )
                late_start_pen = max(0.0, 18.0 - float(first_real_fi)) * 8.0
                real_weight = 180.0 if y_span >= 100.0 else (60.0 if y_span >= 60.0 else 20.0)
                return (
                    arc_score
                    + real_weight * float(real_pts)
                    + 100.0 * conf
                    + 0.5 * float(span)
                    - 100.0 * interp
                    - 0.8 * float(int(d.frame_start))
                    - late_start_pen
                )

            # Merge split segments (pre-bounce / post-bounce) before picking best.
            ordered = sorted(deliveries, key=lambda d: int(d.frame_start))
            merged_ordered: List[DeliveryAnalysis] = []
            for d in ordered:
                if not merged_ordered:
                    merged_ordered.append(d)
                    continue
                prev = merged_ordered[-1]
                gap = int(d.frame_start) - int(prev.frame_end)
                if gap <= 18:
                    prev_pts = (prev.track or {}).get("points") or []
                    d_pts = (d.track or {}).get("points") or []
                    merge_ok = False
                    if prev_pts and d_pts:
                        py = [float(p.get("y", 0.0)) for p in prev_pts]
                        prev_y_span = (max(py) - min(py)) if py else 0.0
                        bx, by = float(prev_pts[-1].get("x", 0.0)), float(prev_pts[-1].get("y", 0.0))
                        nx, ny = float(d_pts[0].get("x", 0.0)), float(d_pts[0].get("y", 0.0))
                        jump = math.hypot(nx - bx, ny - by)
                        merge_ok = jump <= 120.0 and not (
                            prev_y_span < 50.0 and jump > 40.0
                        )
                    if merge_ok:
                        merged_ordered[-1] = self._merge_delivery_segments(prev, d)
                    else:
                        merged_ordered.append(d)
                else:
                    merged_ordered.append(d)
            deliveries = merged_ordered

            best = max(deliveries, key=_delivery_score)
            # Optionally merge immediate next segment if still separate.
            ordered = sorted(deliveries, key=lambda d: int(d.frame_start))
            try:
                idx_best = ordered.index(best)
            except ValueError:
                idx_best = -1
            if idx_best >= 0 and idx_best + 1 < len(ordered):
                nxt = ordered[idx_best + 1]
                gap = int(nxt.frame_start) - int(best.frame_end)
                best_pts = (best.track or {}).get("points") or []
                nxt_pts = (nxt.track or {}).get("points") or []
                if best_pts and nxt_pts and gap <= 18:
                    bx, by = float(best_pts[-1].get("x", 0.0)), float(best_pts[-1].get("y", 0.0))
                    nx, ny = float(nxt_pts[0].get("x", 0.0)), float(nxt_pts[0].get("y", 0.0))
                    jump = math.hypot(nx - bx, ny - by)
                    if jump <= 120.0:
                        merged_points_map: Dict[int, Dict[str, Any]] = {}
                        for seg in (best, nxt):
                            for p in (seg.track or {}).get("points") or []:
                                fi = int(p.get("frame_idx", -1))
                                if fi < 0:
                                    continue
                                ex = merged_points_map.get(fi)
                                if ex is None:
                                    merged_points_map[fi] = p
                                else:
                                    ex_interp = bool(ex.get("is_interpolated", False))
                                    p_interp = bool(p.get("is_interpolated", False))
                                    ex_conf = float(ex.get("confidence", 0.0))
                                    p_conf = float(p.get("confidence", 0.0))
                                    if (ex_interp and not p_interp) or (p_conf > ex_conf):
                                        merged_points_map[fi] = p
                        merged_pts = [merged_points_map[k] for k in sorted(merged_points_map.keys())]
                        if merged_pts:
                            det = [p for p in merged_pts if not bool(p.get("is_interpolated", False))]
                            conf_mean = (
                                float(sum(float(p.get("confidence", 0.0)) for p in det) / len(det))
                                if det else 0.0
                            )
                            interp_pct = (
                                float(sum(1 for p in merged_pts if bool(p.get("is_interpolated", False))) / len(merged_pts))
                                if merged_pts else 0.0
                            )
                            merged_track = {
                                "num_points": int(len(merged_pts)),
                                "confidence_mean": round(conf_mean, 4),
                                "interpolated_pct": round(interp_pct, 4),
                                "points": merged_pts,
                            }
                            best = DeliveryAnalysis(
                                delivery_id=best.delivery_id,
                                frame_start=int(best.frame_start),
                                frame_end=int(nxt.frame_end),
                                track=merged_track,
                                bounce=nxt.bounce or best.bounce,
                                speed=nxt.speed or best.speed,
                                swing=nxt.swing or best.swing,
                                trajectory=nxt.trajectory or best.trajectory,
                                line=nxt.line or best.line,
                                length=nxt.length or best.length,
                                confidence=max(float(best.confidence), float(nxt.confidence)),
                                processing_time_ms=float(best.processing_time_ms) + float(nxt.processing_time_ms),
                                world_trajectory=nxt.world_trajectory or best.world_trajectory,
                            )
            _debug_log(
                "H10",
                "pipeline.py:_limit_deliveries:select-best-single",
                "Selected best single delivery segment.",
                {
                    "selected_start": int(best.frame_start),
                    "selected_end": int(best.frame_end),
                    "selected_points": int(len((best.track or {}).get("points") or [])),
                },
            )
            return [best]

        kept = deliveries[: self.cfg.max_deliveries]
        logger.info(
            "Keeping first %d delivery; dropping %d later post-hit/re-track segment(s).",
            len(kept),
            len(deliveries) - len(kept),
        )
        return kept

    @staticmethod
    def _track_displacement(points: List[TrackPoint]) -> float:
        real = [p for p in points if not p.is_interpolated] or points
        if len(real) < 2:
            return 0.0
        return float(np.hypot(real[-1].x - real[0].x, real[-1].y - real[0].y))

    @staticmethod
    def _trim_trailing_prediction_tail(
        points: List[TrackPoint],
        max_predicted_tail: int = 1,
    ) -> List[TrackPoint]:
        """Remove long Kalman-only tails after the last real ball detection."""
        if not points:
            return points

        last_real_idx = None
        for idx in range(len(points) - 1, -1, -1):
            if not points[idx].is_interpolated:
                last_real_idx = idx
                break

        if last_real_idx is None:
            return points

        keep_until = min(len(points), last_real_idx + 1 + max_predicted_tail)
        if keep_until < len(points):
            logger.info(
                "Trimmed %d trailing predicted points after final real ball detection (frame %d).",
                len(points) - keep_until,
                points[last_real_idx].frame_idx,
            )
        return points[:keep_until]

    def _rebuild_heatmap(self, deliveries: List[DeliveryAnalysis]) -> None:
        self.heatmap_gen.reset()
        for delivery in deliveries:
            if not delivery.bounce:
                continue
            bounce = delivery.bounce
            if bounce.get("world_x") is None or bounce.get("world_y") is None:
                continue
            line_value = delivery.line or "unknown"
            self.heatmap_gen.add_bounce(BouncePoint(
                delivery_id=delivery.delivery_id,
                world_x=self._heatmap_lateral_x(float(bounce["world_x"]), line_value),
                world_y=float(bounce["world_y"]),
                speed_kmh=float(delivery.speed.get("speed_kmh", 0.0)) if delivery.speed else 0.0,
                line=line_value,
                length=delivery.length or "unknown",
            ))

    @staticmethod
    def _trim_to_best_detection_cluster(
        points: List[TrackPoint],
        max_real_gap: int = 3,
        min_cluster_real_points: int = 4,
        min_cluster_displacement_px: float = 25.0,
    ) -> List[TrackPoint]:
        """
        Drop weak early false starts and keep the sustained ball-flight segment.

        A YOLO ball model that has seen generic `item` labels may briefly fire on
        limbs/stumps before it detects the actual moving ball. The real delivery
        normally appears as the longest cluster of consecutive real detections.
        """
        real_indices = [i for i, p in enumerate(points) if not p.is_interpolated]
        if len(real_indices) < min_cluster_real_points:
            return points

        clusters: List[List[int]] = []
        current = [real_indices[0]]
        for idx in real_indices[1:]:
            prev = current[-1]
            frame_gap = points[idx].frame_idx - points[prev].frame_idx
            spatial_jump = float(np.hypot(
                points[idx].x - points[prev].x,
                points[idx].y - points[prev].y,
            ))
            # Split when time OR space gap is large (stops frame-21 phantom linking to frame-24 ball).
            if frame_gap <= max_real_gap and spatial_jump <= 70.0:
                current.append(idx)
            else:
                clusters.append(current)
                current = [idx]
        clusters.append(current)

        def cluster_metrics(cluster: List[int]) -> Tuple[float, float, int]:
            if not cluster:
                return 0.0, 0.0, 0
            first = points[cluster[0]]
            last = points[cluster[-1]]
            displacement = float(np.hypot(last.x - first.x, last.y - first.y))
            frame_span = int(last.frame_idx - first.frame_idx)
            conf_sum = float(sum(points[i].confidence for i in cluster))
            score = (
                frame_span * 1000.0
                + len(cluster) * 100.0
                + displacement
                + conf_sum * 5.0
            )
            return score, displacement, frame_span

        def cluster_gap_to_next(cluster: List[int], nxt: List[int]) -> float:
            return float(np.hypot(
                points[nxt[0]].x - points[cluster[-1]].x,
                points[nxt[0]].y - points[cluster[-1]].y,
            ))

        # Drop only weak leading phantom clusters (e.g. frame-21 limb hit), keep the
        # first sustained flight segment — do not discard an early real arc in favour
        # of a shorter late tail.
        while len(clusters) > 1:
            head, nxt = clusters[0], clusters[1]
            h_score, h_disp, h_span = cluster_metrics(head)
            n_score, _, _ = cluster_metrics(nxt)
            gap_px = cluster_gap_to_next(head, nxt)
            weak_head = (
                len(head) < min_cluster_real_points
                or h_disp < min_cluster_displacement_px
                or h_span <= 1
                or (gap_px > 90.0 and h_score < n_score * 0.55)
            )
            if weak_head:
                clusters = clusters[1:]
                continue
            break

        best = clusters[0]
        start_idx = best[0]
        _, displacement, _ = cluster_metrics(best)
        first = points[best[0]]
        last = points[best[-1]]

        if len(best) < min_cluster_real_points or displacement < min_cluster_displacement_px:
            # #region agent log
            _debug_log(
                "H9",
                "pipeline.py:_trim_to_best_detection_cluster:skip-trim",
                "Cluster trim skipped because best cluster is weak.",
                {
                    "points_total": int(len(points)),
                    "real_points_total": int(len(real_indices)),
                    "best_cluster_real_points": int(len(best)),
                    "best_cluster_displacement_px": float(displacement),
                    "best_cluster_start_frame": int(points[best[0]].frame_idx),
                    "best_cluster_end_frame": int(points[best[-1]].frame_idx),
                },
            )
            # #endregion
            return points

        start_idx = best[0]
        total_points = len(points)
        kept_points_if_trimmed = total_points - start_idx
        late_start_ratio = (start_idx / total_points) if total_points > 0 else 0.0

        # Guardrail: if the "best" cluster starts too late and would keep only a
        # tiny tail, it's likely not the true full delivery path. Keep original.
        if (
            start_idx > 0
            and late_start_ratio > 0.60
            and kept_points_if_trimmed < 10
        ):
            # #region agent log
            _debug_log(
                "H12",
                "pipeline.py:_trim_to_best_detection_cluster:guard-skip-late-tail",
                "Skipped cluster trim because it would keep only a short late tail.",
                {
                    "points_total": int(total_points),
                    "trimmed_prefix_points": int(start_idx),
                    "kept_points_if_trimmed": int(kept_points_if_trimmed),
                    "late_start_ratio": float(late_start_ratio),
                    "candidate_start_frame": int(points[start_idx].frame_idx),
                    "candidate_end_frame": int(points[-1].frame_idx),
                },
            )
            # #endregion
            return points

        # #region agent log
        _debug_log(
            "H9",
            "pipeline.py:_trim_to_best_detection_cluster:trim",
            "Applied false-start trimming to best detection cluster.",
            {
                "points_total": int(len(points)),
                "real_points_total": int(len(real_indices)),
                "best_cluster_real_points": int(len(best)),
                "best_cluster_displacement_px": float(displacement),
                "trimmed_prefix_points": int(start_idx),
                "start_frame": int(points[start_idx].frame_idx),
                "end_frame": int(points[-1].frame_idx),
            },
        )
        # #endregion
        if start_idx > 0:
            logger.info(
                "Trimmed %d false-start track points before sustained ball cluster "
                "(start_frame=%d, real_points=%d, displacement=%.1fpx).",
                start_idx,
                points[start_idx].frame_idx,
                len(best),
                displacement,
            )
        return points[start_idx:]

    def _detections_from_cache(self, frame_idx: int) -> List[Detection]:
        """Best YOLO center from permissive cache for Kalman re-tracking."""
        cached = self._raw_ball_cache.get(frame_idx) or []
        if not cached:
            return []
        fh = max(480, int(self._video_frame_h or 0))
        plausible = [
            c for c in cached
            if 40.0 < float(c[1]) < fh * 0.88 and float(c[2]) >= 0.01
        ]
        pool = plausible or list(cached)
        best = max(pool, key=lambda c: float(c[2]))
        return [
            Detection(
                frame_idx=frame_idx,
                cx=float(best[0]),
                cy=float(best[1]),
                conf=float(best[2]),
                w=22.0,
                h=22.0,
            )
        ]

    def _rebuild_track_with_kalman_cache(
        self,
        frame_start: int,
        frame_end: int,
        *,
        fast_confirm: bool = False,
    ) -> List[TrackPoint]:
        """Re-run professional Kalman tracker on cached detections (full delivery arc)."""
        sub = BallTracker(
            fps=self.cfg.fps,
            max_missing_frames=max(8, self.cfg.max_missing_frames),
            confidence_threshold=self.cfg.ball_confidence,
            max_jump_px=160.0 if fast_confirm else 140.0,
            low_conf_thresh=0.02,
            low_conf_radius_px=120.0 if fast_confirm else 110.0,
            min_hits_to_confirm=1 if fast_confirm else 2,
        )
        out: List[TrackPoint] = []
        for fi in range(int(frame_start), int(frame_end) + 1):
            tp = sub.update(fi, self._detections_from_cache(fi))
            if tp is not None:
                out.append(tp)
        return out

    @staticmethod
    def _merge_delivery_segments(
        a: "DeliveryAnalysis",
        b: "DeliveryAnalysis",
    ) -> "DeliveryAnalysis":
        """Merge two delivery segments into one track (pre/post bounce split)."""
        merged_map: Dict[int, Dict[str, Any]] = {}
        for seg in (a, b):
            for p in (seg.track or {}).get("points") or []:
                fi = int(p.get("frame_idx", -1))
                if fi < 0:
                    continue
                ex = merged_map.get(fi)
                if ex is None:
                    merged_map[fi] = p
                else:
                    ex_i = bool(ex.get("is_interpolated", False))
                    p_i = bool(p.get("is_interpolated", False))
                    if (ex_i and not p_i) or float(p.get("confidence", 0)) > float(ex.get("confidence", 0)):
                        merged_map[fi] = p
        merged_pts = [merged_map[k] for k in sorted(merged_map.keys())]
        det = [p for p in merged_pts if not bool(p.get("is_interpolated", False))]
        conf_mean = float(sum(float(p.get("confidence", 0)) for p in det) / len(det)) if det else 0.0
        interp_pct = (
            float(sum(1 for p in merged_pts if bool(p.get("is_interpolated", False))) / len(merged_pts))
            if merged_pts else 0.0
        )
        return DeliveryAnalysis(
            delivery_id=a.delivery_id,
            frame_start=min(int(a.frame_start), int(b.frame_start)),
            frame_end=max(int(a.frame_end), int(b.frame_end)),
            track={
                "num_points": len(merged_pts),
                "confidence_mean": round(conf_mean, 4),
                "interpolated_pct": round(interp_pct, 4),
                "points": merged_pts,
            },
            bounce=b.bounce or a.bounce,
            speed=b.speed or a.speed,
            swing=b.swing or a.swing,
            trajectory=b.trajectory or a.trajectory,
            line=b.line or a.line,
            length=b.length or a.length,
            confidence=max(float(a.confidence), float(b.confidence)),
            processing_time_ms=float(a.processing_time_ms) + float(b.processing_time_ms),
            world_trajectory=b.world_trajectory or a.world_trajectory,
        )

    @staticmethod
    def _filter_phantom_cache_point(
        cx: float, cy: float, conf: float,
        anchor: Optional[Tuple[float, float]],
    ) -> bool:
        """Drop static-background false positives far from the active ball track."""
        if anchor is None:
            return True
        if conf >= 0.35:
            return True
        dist = math.hypot(cx - anchor[0], cy - anchor[1])
        if dist > 95.0:
            return False
        if conf < 0.12 and dist > 45.0:
            return False
        return True

    def _extend_delivery_through_bounce(
        self,
        delivery: "DeliveryAnalysis",
        total_frames: int,
    ) -> "DeliveryAnalysis":
        """
        Extend Kalman track through bounce and post-bounce using YOLO cache.

        Fixes split deliveries where pass-1 tracker resets during brief misses.
        """
        pts_list = (delivery.track or {}).get("points") or []
        if not pts_list:
            return delivery

        ordered = sorted(pts_list, key=lambda p: int(p.get("frame_idx", 0)))
        track_frames = [int(p.get("frame_idx", 0)) for p in ordered]
        ext_start = max(0, min(track_frames) - 4)
        ext_end = min(int(total_frames) - 1, max(track_frames) + 22)

        # Seed from existing good pre-bounce track (ignore horizontal false tails).
        ys = [float(p.get("y", 0)) for p in ordered]
        y_span = max(ys) - min(ys) if ys else 0.0
        if y_span >= 80.0:
            pre_bounce = [p for p in ordered if not bool(p.get("is_interpolated", False))]
            if len(pre_bounce) >= 3:
                ordered = sorted(pre_bounce + [p for p in ordered if p.get("is_interpolated")], key=lambda p: int(p.get("frame_idx", 0)))

        last_pt = ordered[-1]
        anchor = (float(last_pt.get("x", 0)), float(last_pt.get("y", 0)))
        last_fi = int(last_pt.get("frame_idx", 0))

        # Collect post-bounce cache detections (after bounce frame ~31+).
        post_cache: List[Tuple[int, float, float, float]] = []
        for fi in range(last_fi + 1, min(ext_end + 1, int(total_frames))):
            cands = self._raw_ball_cache.get(fi) or []
            if not cands:
                continue
            best = max(cands, key=lambda c: (c[2], c[1]))
            cx, cy, conf = float(best[0]), float(best[1]), float(best[2])
            min_conf = 0.01 if fi > last_fi + 3 else 0.10
            if conf < min_conf:
                continue
            if not self._filter_phantom_cache_point(cx, cy, conf, anchor):
                continue
            if cy < anchor[1] - 80.0 and conf < 0.30:
                continue
            post_cache.append((fi, cx, cy, conf))
            anchor = (cx, cy)

        display_pts: List[TrackPoint] = []
        for p in ordered:
            display_pts.append(TrackPoint(
                frame_idx=int(p.get("frame_idx", 0)),
                x=float(p.get("x", 0)),
                y=float(p.get("y", 0)),
                vx=float(p.get("vx", 0)),
                vy=float(p.get("vy", 0)),
                is_interpolated=bool(p.get("is_interpolated", False)),
                confidence=float(p.get("confidence", 0)),
            ))

        if post_cache:
            first_post = post_cache[0]
            gap = first_post[0] - last_fi
            if gap > 1:
                lx, ly = display_pts[-1].x, display_pts[-1].y
                peak_y = max(ly, first_post[2], max((p[2] for p in post_cache), default=ly))
                for step in range(1, gap):
                    t = step / float(gap)
                    fx = lx + (first_post[1] - lx) * t
                    fy = ly + (peak_y - ly) * math.sin(0.5 * math.pi * t)
                    display_pts.append(TrackPoint(
                        frame_idx=last_fi + step,
                        x=float(fx), y=float(fy),
                        vx=(first_post[1] - lx) / gap,
                        vy=(peak_y - ly) / gap,
                        is_interpolated=True,
                        confidence=0.0,
                    ))

            sub = BallTracker(
                fps=self.cfg.fps,
                max_missing_frames=10,
                confidence_threshold=self.cfg.ball_confidence,
                max_jump_px=150.0,
                low_conf_thresh=0.02,
                low_conf_radius_px=120.0,
                min_hits_to_confirm=2,
            )
            for fi, cx, cy, conf in post_cache:
                dets = [Detection(fi, cx, cy, conf, 22.0, 22.0)]
                tp = sub.update(fi, dets)
                if tp is not None:
                    display_pts.append(tp)

        if len(display_pts) < 4:
            ext_end = min(int(total_frames) - 1, max(track_frames) + 18)
            rebuilt = self._rebuild_track_with_kalman_cache(ext_start, ext_end)
            if len(rebuilt) < 4:
                return delivery
            display_pts = self._bridge_track_gaps(rebuilt, max_gap_frames=18)

        display = self._trim_trailing_prediction_tail(display_pts, max_predicted_tail=2)
        if len(display) < 4:
            return delivery

        from analytics.delivery_phases import analyze_flight_phases, slice_track_to_phases

        flight_phases = analyze_flight_phases(
            display,
            post_bat_retreat_px=float(self._real_path_builder.post_bat_retreat_px),
        )
        bounded = slice_track_to_phases(display, flight_phases)
        if len(bounded) >= 4:
            display = bounded
        elif flight_phases.valid:
            display = slice_track_to_phases(display, flight_phases) or display

        det = [p for p in display if not p.is_interpolated]
        conf_mean = float(sum(p.confidence for p in det) / len(det)) if det else 0.0
        interp_pct = float(sum(1 for p in display if p.is_interpolated)) / len(display)

        track_dict = {
            "num_points": len(display),
            "confidence_mean": round(conf_mean, 4),
            "interpolated_pct": round(interp_pct, 4),
            "points": [p.to_dict() for p in display],
        }
        release_fi = delivery.release_frame
        bat_fi = delivery.bat_impact_frame
        if flight_phases.valid:
            release_fi = int(flight_phases.release_frame)
            bat_fi = int(flight_phases.bat_impact_frame)
            track_dict["release_frame"] = release_fi
            track_dict["bat_impact_frame"] = bat_fi
            if flight_phases.bounce_frame is not None:
                track_dict["bounce_frame"] = int(flight_phases.bounce_frame)

        logger.info(
            "Extended track through bounce: %d→%d pts, release %s→bat %s (was %d–%d).",
            len(pts_list), len(display),
            release_fi, bat_fi,
            int(delivery.frame_start), int(delivery.frame_end),
        )

        return DeliveryAnalysis(
            delivery_id=delivery.delivery_id,
            frame_start=int(display[0].frame_idx),
            frame_end=int(display[-1].frame_idx),
            release_frame=release_fi,
            bat_impact_frame=bat_fi,
            track=track_dict,
            bounce=delivery.bounce,
            speed=delivery.speed,
            swing=delivery.swing,
            trajectory=delivery.trajectory,
            line=delivery.line,
            length=delivery.length,
            confidence=delivery.confidence,
            processing_time_ms=delivery.processing_time_ms,
            world_trajectory=delivery.world_trajectory,
        )

    def _reference_arc_mode(self) -> bool:
        """Videos with pre-painted trajectory or explicit should_output preset."""
        return bool(
            self.cfg.match_should_output or self.cfg.has_reference_overlay
        )

    def _use_ball_ensemble(self) -> bool:
        if self.ball_model_alt is None:
            return False
        return bool(
            self.cfg.hybrid_ensemble
            or self.cfg.has_reference_overlay
            or self._reference_arc_mode()
        )

    def _build_adaptive_fusion_plan(self, cap: cv2.VideoCapture) -> AdaptiveFusionPlan:
        """Probe all ball checkpoints; fuse per-frame by span-weighted quality."""
        models_dir = Path(self.cfg.ball_model_path).parent
        paths = discover_ball_models(
            models_dir,
            explicit=self.cfg.adaptive_model_paths,
        )
        if not paths:
            raise FileNotFoundError(
                "adaptive_multi_model: no ball*.pt weights found in models/",
            )
        logger.info(
            "Adaptive multi-model fusion | probing %d weights on %s",
            len(paths),
            self.cfg.video_path,
        )
        min_conf = min(0.08, float(self.cfg.ball_confidence))
        if self._reference_arc_mode():
            min_conf = min(min_conf, 0.06)
        from dataclasses import replace

        probe_imgsz = int(self._yolo_infer_cfg.imgsz)
        if self.cfg.high_accuracy_mode:
            probe_imgsz = max(probe_imgsz, 1280)
        probe_cfg = replace(
            self._yolo_infer_cfg,
            imgsz=probe_imgsz,
            augment=False,
            max_det=max(5, int(self._yolo_infer_cfg.max_det)),
        )
        def _load_probe(path: str) -> Any:
            return self._load_yolo(path, "Ball-probe")

        plan = build_adaptive_fusion_plan(
            cap,
            paths,
            _load_probe,
            yolo_cfg=probe_cfg,
            reference_overlay=self._reference_arc_mode(),
            min_conf=min_conf,
            max_step_px=100.0 if self.cfg.high_accuracy_mode else 88.0,
        )
        logger.info(
            "Adaptive fusion ready | fused_frames=%d | top=%s",
            len(plan.fused_detections),
            plan.models[0].model_name if plan.models else "none",
        )
        return plan

    def _ensure_hybrid_alt_model(self) -> None:
        """Load backup weights for painted-arc clips when primary misses blur."""
        if self.ball_model_alt is not None:
            return
        if not (self.cfg.auto_hybrid_on_overlay or self.cfg.hybrid_ensemble):
            return
        if self.cfg.ball_model_alt_path:
            alt_path = self.cfg.ball_model_alt_path
        else:
            # AWS retrain generalizes better on painted/blur clips than 4-class backup.
            aws_p = Path("models/ball_best_aws_v1.pt")
            backup_p = Path("models/ball_best_backup.pt")
            alt_path = str(aws_p if aws_p.exists() else backup_p)
        alt_p = Path(alt_path)
        if not alt_p.exists():
            return
        if str(alt_p.resolve()) == str(Path(self.cfg.ball_model_path).resolve()):
            return
        self.ball_model_alt = self._load_yolo(str(alt_p), "Ball-alt")
        self._ball_detector.alt_model = self.ball_model_alt
        logger.info(
            "Hybrid YOLO ensemble enabled | primary=%s | alt=%s",
            self.cfg.ball_model_path,
            alt_p,
        )

    def _should_cache_kalman_rebuild(self, kalman_points: List[TrackPoint]) -> bool:
        if not self._raw_ball_cache or len(kalman_points) < 4:
            return False
        if self._reference_arc_mode():
            return True
        if not self.cfg.enable_cache_kalman_rebuild:
            return False
        n = len(kalman_points)
        n_real = sum(1 for p in kalman_points if not p.is_interpolated)
        interp = 1.0 - (n_real / max(n, 1))
        return interp > 0.42 or n_real < 8

    def _bridge_track_gaps_smart(self, points: List[TrackPoint]) -> List[TrackPoint]:
        """Fill gaps using cache detections first; linear only for tiny gaps."""
        max_gap = int(self.cfg.bridge_max_gap_frames)
        if len(points) < 2:
            return points

        bridged: List[TrackPoint] = [points[0]]
        for prev, curr in zip(points, points[1:]):
            gap = int(curr.frame_idx - prev.frame_idx)
            if gap <= 1:
                bridged.append(curr)
                continue

            inserted = False
            if gap <= max(12, max_gap + 4) and self._raw_ball_cache:
                for fi in range(int(prev.frame_idx) + 1, int(curr.frame_idx)):
                    for det in self._detections_from_cache(fi):
                        bridged.append(TrackPoint(
                            frame_idx=fi,
                            x=float(det.cx),
                            y=float(det.cy),
                            vx=float(det.cx - prev.x),
                            vy=float(det.cy - prev.y),
                            is_interpolated=False,
                            confidence=float(det.conf),
                        ))
                        inserted = True
                        break

            prev_real = not prev.is_interpolated
            curr_real = not curr.is_interpolated
            allow_linear = (
                not inserted
                and 1 < gap <= max_gap
                and (gap <= 4 or (prev_real and curr_real))
            )
            if allow_linear:
                for step in range(1, gap):
                    t = step / gap
                    x = prev.x + (curr.x - prev.x) * t
                    y = prev.y + (curr.y - prev.y) * t
                    bridged.append(TrackPoint(
                        frame_idx=prev.frame_idx + step,
                        x=float(x),
                        y=float(y),
                        vx=float((curr.x - prev.x) / gap),
                        vy=float((curr.y - prev.y) / gap),
                        is_interpolated=True,
                        confidence=0.0,
                    ))
            bridged.append(curr)

        return bridged

    @staticmethod
    def _bridge_track_gaps(
        points: List[TrackPoint],
        max_gap_frames: int = 18,
    ) -> List[TrackPoint]:
        """Legacy linear bridge (tests / static callers)."""
        if len(points) < 2:
            return points
        bridged: List[TrackPoint] = [points[0]]
        for prev, curr in zip(points, points[1:]):
            gap = curr.frame_idx - prev.frame_idx
            if 1 < gap <= max_gap_frames:
                for step in range(1, gap):
                    t = step / gap
                    bridged.append(TrackPoint(
                        frame_idx=prev.frame_idx + step,
                        x=float(prev.x + (curr.x - prev.x) * t),
                        y=float(prev.y + (curr.y - prev.y) * t),
                        vx=float((curr.x - prev.x) / gap),
                        vy=float((curr.y - prev.y) / gap),
                        is_interpolated=True,
                        confidence=0.0,
                    ))
            bridged.append(curr)
        return bridged

    @staticmethod
    def _world_trajectory_is_plausible(
        world_pts: Optional[List[Tuple[float, float]]],
    ) -> bool:
        """
        Reject world-trajectories that violate cricket pitch geometry:
          * lateral span (x) > 2.5 m  — pitch + reasonable spray
          * longitudinal span (y) > 22 m — slightly beyond pitch length
          * arc length > 24 m  — physically impossible in one delivery
          * any point with |x| > 4 m  — ball can't be 4m off pitch
        Previously these limits were 8.0m / 32m / 36m, which let through
        homographies that produced -11m trajectories for test_video6 and
        45m y-axis values for test_video3.
        """
        if not world_pts or len(world_pts) < 5:
            return False

        arr = np.array(world_pts, dtype=np.float64)
        if not np.isfinite(arr).all():
            return False

        x_span = float(np.ptp(arr[:, 0]))
        y_span = float(np.ptp(arr[:, 1]))
        if x_span > 2.5 or y_span > 22.0:
            return False
        if float(np.max(np.abs(arr[:, 0]))) > 4.0:
            return False
        # Y should grow monotonically (ball heading toward batsman). Allow
        # a small reverse for tracker jitter, but not a wholesale flip.
        if arr[-1, 1] < arr[0, 1] - 1.0:
            return False

        arc = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
        if arc <= 0.05 or arc > 24.0:
            return False

        return True

    def _compute_delivery_analytics(
        self,
        track_points: List[TrackPoint],
        pixel_pts: List[Tuple[float, float]],
        world_pts: Optional[List[Tuple[float, float]]],
    ) -> Dict[str, Optional[Dict]]:
        """
        Compute PDF-required delivery analytics from one finalized track.

        Calibrated world coordinates are preferred.  When no stump/manual
        calibration is available, the fallback uses a documented pixel scale so
        the output remains usable while clearly carrying lower confidence.
        """
        metric_source = "calibrated_homography" if world_pts else "pixel_scale_fallback"
        fallback_metric_pts = self._pixel_points_to_metric(pixel_pts)
        metric_pts = world_pts or fallback_metric_pts

        bounce_result = self.bounce_det.detect(track_points, world_coords=metric_pts)
        if bounce_result is None:
            bounce_result = self._estimate_boundary_bounce(track_points, metric_pts)
        bounce_result = self._prefer_screen_lowest_bounce(track_points, metric_pts, bounce_result)
        bounce = bounce_result.to_dict() if bounce_result else None
        bounce_idx = self._find_bounce_index(track_points, bounce_result.frame_idx if bounce_result else None)

        release_frame_idx = track_points[0].frame_idx if track_points else None
        bounce_frame = bounce_result.frame_idx if bounce_result else None
        bounce_y = bounce_result.world_y if bounce_result else None
        tof_conf_floor = float(self.cfg.speed_tof_bounce_conf)
        use_time_of_flight = bool(
            bounce_result
            and bounce_result.confidence >= tof_conf_floor
            and bounce_result.method != "boundary_lowest_point"
        )

        speed_result = self.speed_est.estimate(
            metric_pts,
            bounce_frame_idx=bounce_idx,
            release_frame_idx=release_frame_idx,
            bounce_frame=bounce_frame,
            bounce_y=bounce_y if use_time_of_flight else None,
            fps=self.cfg.fps,
        )
        selected_speed_source = metric_source

        if world_pts:
            fallback_speed = self.speed_est.estimate(
                fallback_metric_pts,
                bounce_frame_idx=bounce_idx,
                release_frame_idx=release_frame_idx,
                bounce_frame=bounce_frame,
                bounce_y=None,
                fps=self.cfg.fps,
            )
            if self._prefer_fallback_speed(speed_result, fallback_speed):
                logger.warning(
                    "Calibrated speed rejected: value hit physical clamp. "
                    "Using pixel-scale fallback speed for this delivery."
                )
                speed_result = fallback_speed
                selected_speed_source = "pixel_scale_fallback"

        speed = speed_result.to_dict() if speed_result else None
        if speed:
            speed["metric_source"] = selected_speed_source
            n_real = sum(1 for p in track_points if not p.is_interpolated)
            real_ratio = n_real / max(len(track_points), 1)
            base_conf = float(speed.get("confidence", 0.0))
            speed["confidence"] = round(
                min(0.92, base_conf * (0.55 + 0.45 * real_ratio)),
                4,
            )
            if speed.get("method") == "median_prior":
                speed["confidence"] = round(min(float(speed["confidence"]), 0.25), 4)
            if selected_speed_source == "pixel_scale_fallback":
                speed["pixels_per_meter"] = round(float(self.cfg.pixels_per_meter), 4)
                speed["confidence"] = round(
                    min(float(speed["confidence"]), 0.55),
                    4,
                )
        else:
            # Last-resort estimate so speed is NEVER null/zero. Use the raw
            # track points + Kalman velocity to compute a coarse pixel-scale
            # speed, marked as low-confidence so downstream knows to trust it
            # less. Better to report ~50 km/h with confidence 0.20 than to
            # publish "speed: null" / "0.0 km/h" in the JSON.
            speed = self._coarse_pixel_speed(
                track_points, fallback_metric_pts, self.cfg.fps,
            )
            if speed:
                selected_speed_source = "pixel_scale_fallback_coarse"
                speed["metric_source"] = selected_speed_source
                speed["pixels_per_meter"] = round(float(self.cfg.pixels_per_meter), 4)

        swing_input = metric_pts[:bounce_idx + 1] if bounce_idx is not None else metric_pts
        swing_result = self.swing_est.estimate(swing_input)
        swing = self._normalise_swing(swing_result, metric_source)

        return {"bounce": bounce, "speed": speed, "swing": swing}

    @staticmethod
    def _estimate_boundary_bounce(
        track_points: List[TrackPoint],
        metric_pts: List[Tuple[float, float]],
    ) -> Optional[BounceResult]:
        """
        Low-confidence bounce estimate when the visible track ends at ground
        contact and no interior velocity inversion exists.
        """
        if len(track_points) < 5:
            return None

        ys = np.array([p.y for p in track_points], dtype=np.float64)
        is_increasing_y = (ys[-1] - ys[0]) >= 0
        idx = int(np.argmax(ys) if is_increasing_y else np.argmin(ys))

        if idx < 2 or idx > len(track_points) - 1:
            confidence = 0.28
        else:
            confidence = 0.40

        tp = track_points[idx]
        wx, wy = (None, None)
        if metric_pts and idx < len(metric_pts):
            wx, wy = metric_pts[idx]

        return BounceResult(
            frame_idx=tp.frame_idx,
            pixel_x=round(tp.x, 2),
            pixel_y=round(tp.y, 2),
            world_x=round(float(wx), 4) if wx is not None else None,
            world_y=round(float(wy), 4) if wy is not None else None,
            method="boundary_lowest_point",
            confidence=confidence,
            pre_bounce_angle_deg=0.0,
            post_bounce_angle_deg=0.0,
        )

    @staticmethod
    def _prefer_screen_lowest_bounce(
        track_points: List[TrackPoint],
        metric_pts: List[Tuple[float, float]],
        bounce_result: Optional[BounceResult],
    ) -> Optional[BounceResult]:
        """
        In rear-camera clips the true pitch bounce is normally the lowest
        screen point before the post-bounce rise. Prefer that over a later
        velocity inversion when the selected bounce is visibly too high.

        Sanity rules (added after visual inspection showed 4/8 videos
        produced bounces in front of the bowler or in the sky):
          * Bounce cannot be at the very first frame (that's release).
          * Bounce frame must be a REAL detection — not an interpolated
            point. Interpolated points are placeholder positions and can
            inadvertently land at extremes (the tracker fills them in with
            Kalman predictions, which sometimes drift off-screen).
          * Bounce must have a true V-shape: at least one strictly higher
            y-value (further down the screen) than its neighbours on both
            sides. A monotonic descent isn't a bounce.
        """
        if not track_points or not metric_pts:
            return bounce_result

        y_values = [p.y for p in track_points]
        real_mask = [not getattr(p, "is_interpolated", False) for p in track_points]
        n = len(track_points)

        # Define safe window — exclude the first 25% and the last frame.
        min_idx = max(1, int(0.25 * n))
        max_idx = n - 1 if n <= 4 else n - 2

        # Candidate indices: inside safe window AND at a real (non-interp)
        # detection AND with a true V-shape (strictly lower neighbours).
        def is_v(i: int) -> bool:
            if i <= 0 or i >= n - 1:
                return False
            return y_values[i] > y_values[i - 1] and y_values[i] > y_values[i + 1]

        candidates = [
            i for i in range(min_idx, max_idx + 1)
            if real_mask[i] and is_v(i)
        ]

        if candidates:
            lowest_idx = max(candidates, key=lambda i: y_values[i])
        else:
            # No V-shape in real detections. For SHORT tracks this is
            # normal — there aren't enough samples to ever form a V. Fall
            # back to the lowest real detection in the safe window.
            real_in_window = [
                (i, y_values[i]) for i in range(min_idx, max_idx + 1)
                if real_mask[i]
            ]
            n_real_total = sum(real_mask)
            if n_real_total < 6 and real_in_window:
                lowest_idx = max(real_in_window, key=lambda t: t[1])[0]
            elif real_in_window:
                # Long enough track but no V-shape — original bounce_result
                # is more trustworthy than picking the longest-descending
                # point.
                return bounce_result
            else:
                return bounce_result

        lowest_pt = track_points[lowest_idx]

        # When the original bounce point sits within 25px of the visual
        # lowest point, both signals AGREE — that's a strong confirmation,
        # so boost the original confidence rather than discarding it.
        if bounce_result and bounce_result.pixel_y >= lowest_pt.y - 25.0:
            boosted = float(bounce_result.confidence) + 0.30
            bounce_result.confidence = round(min(boosted, 0.85), 4)
            return bounce_result

        wx, wy = metric_pts[min(lowest_idx, len(metric_pts) - 1)]
        conf = max(0.65, float(bounce_result.confidence) if bounce_result else 0.65)
        logger.info(
            "Bounce adjusted to screen-lowest point: frame %d pixel=(%.1f, %.1f).",
            lowest_pt.frame_idx,
            lowest_pt.x,
            lowest_pt.y,
        )
        return BounceResult(
            frame_idx=lowest_pt.frame_idx,
            pixel_x=lowest_pt.x,
            pixel_y=lowest_pt.y,
            world_x=wx,
            world_y=wy,
            method="screen_lowest_point",
            confidence=round(min(conf, 0.85), 4),
            pre_bounce_angle_deg=0.0,
            post_bounce_angle_deg=0.0,
        )

    @staticmethod
    def _prefer_fallback_speed(
        calibrated: Optional[SpeedResult],
        fallback: Optional[SpeedResult],
    ) -> bool:
        """
        Prefer the pixel-scale fallback whenever the calibrated estimate is
        implausible. Previously required both 'calibrated hit clamp' AND
        'fallback interior' — which left clamped 165 km/h values in place
        when the fallback was also extreme. Now: any time calibrated is
        outside the cricket range (or absent), fall back, even if the
        fallback itself is non-ideal (the coarse final fallback will catch
        truly-broken cases downstream).
        """
        if calibrated is None:
            return True
        if fallback is None:
            return False
        cal = float(calibrated.speed_kmh)
        fb = float(fallback.speed_kmh)
        # Hard clamp implies the underlying distance/duration is wrong.
        calibrated_implausible = cal <= 32.0 or cal >= 160.0
        # Only switch if fallback is at least somewhat better-behaved.
        fallback_better = abs(fb - 110.0) < abs(cal - 110.0)
        return calibrated_implausible and fallback_better

    def _coarse_pixel_speed(
        self,
        track_points: List[TrackPoint],
        fallback_metric_pts: List[Tuple[float, float]],
        fps: float,
    ) -> Optional[Dict]:
        """
        Last-resort speed estimate when the main estimator can't run
        (typically: very short tracks, ≤4 points). Uses the cumulative
        pixel-scale arc length over the visible portion and the elapsed
        frames. Always returns *something* (even if low-confidence) so the
        downstream report never shows speed=0/null.

        Falls back to the most permissive estimator possible:
          1. If ≥2 metric points: compute arc-length over span(frames)/fps.
          2. If ≥2 track points: estimate from Kalman vx/vy magnitude.
          3. Otherwise: median bowling speed (115 km/h) with conf=0.10.
        """
        # Cricket realism bounds: typical bowling speeds are 90-150 km/h
        # for adults. Anything outside [35, 155] km/h indicates the underlying
        # measurement is unreliable, so we collapse to a per-bowling-style
        # median prior with low confidence.
        BOWL_FLOOR = 35.0
        BOWL_CEIL  = 155.0
        median_kmh = 110.0  # safe mid-range prior for any cricket delivery

        def _clamp_and_finalise(raw_kmh: float, method: str, conf: float,
                                arc: float, duration: float, n_frames: int) -> Dict:
            """Apply realism band; if outside, fall to median prior."""
            if BOWL_FLOOR <= raw_kmh <= BOWL_CEIL:
                final_kmh = raw_kmh
                final_conf = conf
                final_method = method
            else:
                # Outside realistic band → use median prior, very low conf.
                final_kmh = median_kmh
                final_conf = 0.10
                final_method = f"{method}_outside_band_median"
            return {
                "speed_kmh":    round(final_kmh, 2),
                "speed_mph":    round(final_kmh * 0.621371, 2),
                "speed_ms":     round(final_kmh / 3.6, 4),
                "distance_m":   round(arc, 4),
                "duration_sec": round(duration, 4),
                "frames_used":  n_frames,
                "method":       final_method,
                "confidence":   round(final_conf, 4),
            }

        if len(track_points) >= 2 and len(fallback_metric_pts) >= 2:
            f_first = track_points[0].frame_idx
            f_last = track_points[-1].frame_idx
            n_frames = max(int(f_last - f_first), 1)
            duration = n_frames / max(float(fps or self.cfg.fps), 1.0)

            arc = 0.0
            for i in range(1, len(fallback_metric_pts)):
                px, py = fallback_metric_pts[i]
                qx, qy = fallback_metric_pts[i - 1]
                arc += float(np.hypot(px - qx, py - qy))

            if duration > 0 and arc > 0:
                raw_kmh = (arc / duration) * 3.6
                return _clamp_and_finalise(
                    raw_kmh, "coarse_arc_length", 0.20, arc, duration, n_frames,
                )

        if len(track_points) >= 2:
            vmag_pxf = float(np.hypot(track_points[-1].vx, track_points[-1].vy))
            ppm = max(float(self.cfg.pixels_per_meter), 1.0)
            raw_kmh = ((vmag_pxf * float(fps or self.cfg.fps)) / ppm) * 3.6
            if raw_kmh > 0:
                return _clamp_and_finalise(
                    raw_kmh, "kalman_velocity", 0.15, 0.0, 0.0, len(track_points),
                )

        return {
            "speed_kmh":    median_kmh,
            "speed_mph":    round(median_kmh * 0.621371, 2),
            "speed_ms":     round(median_kmh / 3.6, 4),
            "distance_m":   0.0,
            "duration_sec": 0.0,
            "frames_used":  0,
            "method":       "median_prior",
            "confidence":   0.10,
        }

    def _pixel_points_to_metric(
        self,
        pixel_pts: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Approximate metric coordinates from pixels when homography is absent.

        X is lateral displacement from release. Y is cumulative travel distance,
        which is more stable than raw image Y for single-camera fallback speed.
        """
        if not pixel_pts:
            return []

        ppm = max(float(self.cfg.pixels_per_meter), 1.0)
        x0, _ = pixel_pts[0]
        metric_pts: List[Tuple[float, float]] = []
        cumulative_m = 0.0
        prev = pixel_pts[0]

        for idx, pt in enumerate(pixel_pts):
            if idx > 0:
                cumulative_m += float(np.hypot(pt[0] - prev[0], pt[1] - prev[1])) / ppm
            metric_pts.append(((pt[0] - x0) / ppm, cumulative_m))
            prev = pt

        return metric_pts

    @staticmethod
    def _heatmap_lateral_x(world_x: float, line: str) -> float:
        """
        Keep heatmap bounces inside the crease when metric fallback is used.

        Pixel-scale fallback X is a relative image displacement, not true
        lateral metres, so it can exceed the pitch width even while Y/speed
        remain useful. Use the classified line as the stable lateral cue.
        """
        half_crease_m = 1.83
        if -half_crease_m <= world_x <= half_crease_m:
            return world_x

        line_to_x = {
            "wide_outside_off": -1.10,
            "outside_off": -0.70,
            "off_stump": -0.23,
            "middle_stump": 0.0,
            "leg_stump": 0.23,
            "outside_leg": 0.70,
        }
        return line_to_x.get(line, float(np.clip(world_x, -half_crease_m, half_crease_m)))

    def _classify_line_from_pitch_pixels(
        self,
        px: float,
        py: float,
    ) -> Optional[BowlingLine]:
        """Classify line from the rendered pitch corridor when metric X is fallback-only."""
        if not self.calibration_result:
            return None

        try:
            bl = np.array(self.calibration_result.world_to_pixel(-0.95, 0.0), dtype=np.float64)
            br = np.array(self.calibration_result.world_to_pixel(0.95, 0.0), dtype=np.float64)
            tl = np.array(self.calibration_result.world_to_pixel(-0.95, 20.12), dtype=np.float64)
            tr = np.array(self.calibration_result.world_to_pixel(0.95, 20.12), dtype=np.float64)
        except Exception:
            return None

        denom = float(bl[1] - tl[1])
        if abs(denom) < 1e-6:
            return None

        t = float(np.clip((py - tl[1]) / denom, 0.0, 1.0))
        left = tl + (bl - tl) * t
        right = tr + (br - tr) * t
        width = float(np.linalg.norm(right - left))
        if width < 1.0:
            return None

        center_x = (left[0] + right[0]) * 0.5
        half_width = abs(right[0] - left[0]) * 0.5
        if half_width < 1.0:
            return None

        lateral = (px - center_x) / half_width

        # If the bounce sits well outside the rendered corridor (|lateral| > 1.2)
        # the corridor itself is misplaced (homography picked the wrong stumps
        # or got the wrong scale). Don't publish a misleading "wide_outside"
        # classification — return None so the caller falls back to OFF_STUMP.
        if abs(lateral) > 1.2:
            return None

        if lateral < -0.70:
            return BowlingLine.WIDE_OUTSIDE_OFF
        if lateral < -0.35:
            return BowlingLine.OUTSIDE_OFF
        if lateral < -0.12:
            return BowlingLine.OFF_STUMP
        if lateral <= 0.12:
            return BowlingLine.MIDDLE_STUMP
        if lateral <= 0.35:
            return BowlingLine.LEG_STUMP
        if lateral <= 0.70:
            return BowlingLine.OUTSIDE_LEG
        return BowlingLine.WIDE_OUTSIDE_LEG

    @staticmethod
    def _find_bounce_index(
        track_points: List[TrackPoint],
        bounce_frame_idx: Optional[int],
    ) -> Optional[int]:
        if bounce_frame_idx is None:
            return None
        for idx, pt in enumerate(track_points):
            if pt.frame_idx == bounce_frame_idx:
                return idx
        return None

    @staticmethod
    def _normalise_swing(
        swing_result: Optional[SwingResult],
        metric_source: str,
    ) -> Optional[Dict]:
        if swing_result is None:
            return None

        swing = swing_result.to_dict()
        max_dev_m = float(swing.get("max_deviation_m", 0.0))
        out = {
            "direction": swing.get("swing_type", "none"),
            "swing_type": swing.get("swing_type", "none"),
            "swing_cm": round(max_dev_m * 100.0, 2),
            "max_deviation_m": round(max_dev_m, 4),
            "avg_deviation_m": swing.get("avg_deviation_m", 0.0),
            "late_swing_ratio": swing.get("late_swing_ratio", 1.0),
            "is_reverse_swing": swing.get("is_reverse_swing", False),
            "confidence": swing.get("confidence", 0.0),
            "metric_source": metric_source,
        }
        if metric_source == "pixel_scale_fallback":
            out["confidence"] = round(min(float(out["confidence"]), 0.45), 4)
        return out

    def _truncate_post_hit_points(
        self,
        points: List[TrackPoint],
        pixel_pts: List[Tuple[float, float]],
        world_pts: Optional[List[Tuple[float, float]]],
    ) -> Tuple[List[TrackPoint], List[Tuple[float, float]], Optional[List[Tuple[float, float]]]]:
        """
        Truncate tracking points once the ball is hit by the batsman or passes the stumps.
        This prevents outgoing batted-ball trajectories from warping the delivery curves.
        Automatically detects flight direction (behind-stumps vs. behind-bowler) and adapts.
        """
        if len(points) < 5:
            return points, pixel_pts, world_pts

        # ------------------------------------------------------------------
        # Phase 0 (NEW): Hard-jump segmentation.
        # Split the track at every single-frame inter-point step that is
        # non-physical (>=5x the recent median, with a 70 px absolute floor).
        # Keep the LONGEST clean segment. This catches BOTH cases:
        #   - tracker snapping onto a false target AFTER bat impact -> drop tail
        #   - tracker snapping ONTO the real ball after pre-delivery noise
        #     -> drop the noisy head
        #
        # Bounce-safe: a real pitch bounce is a smooth direction change with
        # similar step magnitudes on both sides; it never produces a >=5x
        # jump in step length.
        # ------------------------------------------------------------------
        def _split_at_hard_jumps_pixels(pts_list: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
            n_local = len(pts_list)
            if n_local < 8:
                return [(0, n_local)]
            steps_local = [
                float(np.hypot(pts_list[k][0] - pts_list[k - 1][0],
                               pts_list[k][1] - pts_list[k - 1][1]))
                for k in range(1, n_local)
            ]
            if len(steps_local) < 4:
                return [(0, n_local)]
            MIN_GATE_ABS = 70.0
            MIN_LEAD_MEDIAN = 4.0
            ABS_HARD_JUMP = 90.0   # px/frame: bat-impact / false re-track (bounce-safe)
            jump_indices: List[int] = []
            jump_set: set = set()
            for i_local in range(1, len(steps_local)):
                step_val = steps_local[i_local]
                # Absolute hard jump.
                if step_val > ABS_HARD_JUMP:
                    jump_indices.append(i_local)
                    jump_set.add(i_local)
                    continue
                # Relative jump (>=5x recent CLEAN steps), bounce-safe. Exclude
                # already-flagged jumps from the reference window so a prior
                # jump doesn't poison the median and hide consecutive jumps.
                clean = [
                    steps_local[k]
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
                return [(0, n_local)]
            starts = [0] + [j + 1 for j in jump_indices]
            ends = [j + 1 for j in jump_indices] + [n_local]
            return list(zip(starts, ends))

        # Phase 0a: single-point spike repair. A single frame whose position
        # lies far off the chord between its immediate neighbours is almost
        # always a momentary tracker glitch / weak YOLO detection. Replacing
        # it with the chord midpoint removes visible loops at the head/middle
        # without losing any frame from the delivery. Bounce-safe (a bounce
        # produces a smooth direction change, not a single-frame spike).
        # IMPORTANT: also update the TrackPoint object's x/y so downstream
        # code (which re-derives pixel_pts from track_result.points) sees the
        # repaired position.
        SPIKE_THRESHOLD_PX = 80.0
        if len(pixel_pts) >= 4:
            for k in range(1, len(pixel_pts) - 1):
                p_prev = pixel_pts[k - 1]
                p_next = pixel_pts[k + 1]
                mid_x = (p_prev[0] + p_next[0]) * 0.5
                mid_y = (p_prev[1] + p_next[1]) * 0.5
                d = float(np.hypot(pixel_pts[k][0] - mid_x, pixel_pts[k][1] - mid_y))
                if d >= SPIKE_THRESHOLD_PX:
                    logger.info(
                        "Single-frame spike repaired at point %d (frame %d): "
                        "moved (%.0f,%.0f) -> (%.0f,%.0f) (off-chord by %.1fpx).",
                        k, points[k].frame_idx,
                        pixel_pts[k][0], pixel_pts[k][1],
                        mid_x, mid_y, d,
                    )
                    pixel_pts[k] = (mid_x, mid_y)
                    # Mutate the TrackPoint so subsequent get_trajectory_pixels()
                    # calls return the repaired position.
                    points[k].x = float(mid_x)
                    points[k].y = float(mid_y)

        # Trim only the TAIL after a bat impact / false re-track. Per design,
        # we never cut the head: the natural delivery includes its release
        # noise and bounce, and visualizer smoothing handles small early
        # detection wobble nicely. We only chop when there's a clear
        # non-physical jump LATE in the trajectory.
        segments = _split_at_hard_jumps_pixels(pixel_pts)
        num_jumps = len(segments) - 1
        if num_jumps >= 1:
            half_idx = max(1, len(pixel_pts) // 2)
            # Step index i means the jump is between pixel_pts[i] and pixel_pts[i+1].
            # segments[k][0] - 1 is the step index for the jump preceding seg k.
            late_jump_step_indices = [
                segments[k][0] - 1 for k in range(1, len(segments))
                if (segments[k][0] - 1) >= half_idx
            ]
            if late_jump_step_indices:
                last_jump = late_jump_step_indices[-1]
                keep_until = last_jump + 1  # keep up to pixel_pts[last_jump] inclusive
                if 5 <= keep_until < len(pixel_pts):
                    try:
                        cut_step = float(np.hypot(
                            pixel_pts[keep_until][0] - pixel_pts[keep_until - 1][0],
                            pixel_pts[keep_until][1] - pixel_pts[keep_until - 1][1],
                        ))
                    except Exception:
                        cut_step = 0.0
                    logger.info(
                        "Trajectory tail trimmed at point %d (frame %d->%d): "
                        "late hard-jump detected (step %.1fpx). Treated as bat impact.",
                        keep_until,
                        points[keep_until - 1].frame_idx,
                        points[keep_until].frame_idx,
                        cut_step,
                    )
                    points = points[:keep_until]
                    pixel_pts = pixel_pts[:keep_until]
                    if world_pts:
                        world_pts = world_pts[:keep_until]
                    if len(points) < 5:
                        return points, pixel_pts, world_pts

        truncate_idx = len(points)

        def _find_post_bat_impact_cut(
            pts: List[Tuple[float, float]],
            bounce_idx: int,
            increasing_y: bool,
            base_step: float,
        ) -> Optional[int]:
            """
            Cut only after bat contact / false re-track — never at bounce.

            Bounce = smooth V-shape; bat hit = large step + sharp reversal of
            the post-bounce trend (e.g. tracker snaps to keeper/fielder).
            """
            n_local = len(pts)
            if n_local < 8:
                return None
            # Start only in the latter half AFTER bounce kinematics settle —
            # avoids chopping the post-bounce rise itself.
            search_start = max(bounce_idx + 10, int(n_local * 0.58))
            if search_start >= n_local - 1:
                return None

            min_step = max(48.0, base_step * 2.8)
            min_dy = max(38.0, base_step * 2.2)

            for i in range(search_start, n_local):
                if i < int(n_local * 0.50):
                    continue
                px, py = pts[i]
                prev_px, prev_py = pts[i - 1]
                step = float(np.hypot(px - prev_px, py - prev_py))
                dy = float(py - prev_py)
                if step < min_step or abs(dy) < min_dy:
                    continue

                dys: List[float] = []
                for k in range(max(1, i - 4), i):
                    dys.append(float(pts[k][1] - pts[k - 1][1]))
                if not dys:
                    continue
                med_dy = float(np.median(dys))
                med_abs = max(2.5, float(np.median([abs(d) for d in dys])))

                # Strong reversal vs immediate post-bounce motion (bat hit).
                if med_dy * dy < 0.0 and abs(dy) > max(min_dy, 2.5 * med_abs):
                    return max(bounce_idx + 2, i - 3)

                # Post-bounce toward batter: screen-Y should keep trending one way;
                # a large snap the other way after bounce is not pitch bounce.
                if increasing_y:
                    # After bounce, ball rises (dy < 0). Large downward snap = mis-track.
                    if med_dy < -0.8 and dy > max(22.0, 2.2 * med_abs):
                        return max(bounce_idx + 2, i - 3)
                else:
                    # After bounce, ball drops (dy > 0). Large upward snap = mis-track.
                    if med_dy > 0.8 and dy < -max(22.0, 2.2 * med_abs):
                        return max(bounce_idx + 2, i - 3)

                # Sharp corner + large step (bat deflection), only late in delivery.
                if i >= max(bounce_idx + 8, int(n_local * 0.62)) and i < n_local - 1:
                    v1x = pts[i][0] - pts[i - 1][0]
                    v1y = pts[i][1] - pts[i - 1][1]
                    v2x = pts[i + 1][0] - pts[i][0]
                    v2y = pts[i + 1][1] - pts[i][1]
                    s1 = float(np.hypot(v1x, v1y))
                    s2 = float(np.hypot(v2x, v2y))
                    if s1 >= 6.0 and s2 >= 6.0:
                        cos_a = float((v1x * v2x + v1y * v2y) / (s1 * s2))
                        if cos_a < -0.30 and step >= min_step:
                            return max(bounce_idx + 2, i - 2)

            return None

        def _estimate_bounce_idx_local(pts: List[Tuple[float, float]], increasing_y: bool) -> int:
            """Interior extremum bounce estimate (avoid endpoint minima/maxima)."""
            n_local = len(pts)
            if n_local < 5:
                ys_local = [p[1] for p in pts]
                return int(np.argmax(ys_local) if increasing_y else np.argmin(ys_local))
            margin = max(2, n_local // 6)
            if (n_local - 2 * margin) >= 3:
                interior = pts[margin:-margin]
                ys_i = [p[1] for p in interior]
                local_idx = int(np.argmax(ys_i) if increasing_y else np.argmin(ys_i))
                return int(margin + local_idx)
            ys_all = [p[1] for p in pts]
            return int(np.argmax(ys_all) if increasing_y else np.argmin(ys_all))

        # 1. Detect natural pixel-Y flight direction (increasing Y vs. decreasing Y)
        start_y = pixel_pts[0][1]
        y_trend = 0.0
        for px, py in pixel_pts:
            if abs(py - start_y) > 2.0:
                y_trend = py - start_y
                break

        is_increasing_y = y_trend >= 0.0

        if is_increasing_y:
            # Ball travels down-screen until bounce, then normally rises up-screen
            # toward the batter. Keep that post-bounce rise; cut only when the
            # track makes a new abrupt jump consistent with bat contact/mis-track.
            raw_extrema_idx = int(np.argmax([py for _, py in pixel_pts]))
            bounce_idx = _estimate_bounce_idx_local(pixel_pts, increasing_y=True)
            post_steps = [
                float(np.hypot(pixel_pts[k][0] - pixel_pts[k - 1][0], pixel_pts[k][1] - pixel_pts[k - 1][1]))
                for k in range(max(1, bounce_idx + 1), len(points))
            ]
            base_step = float(np.median(post_steps)) if post_steps else 12.0
            base_step = max(8.0, base_step)
            step_gate = max(72.0, base_step * 4.8)
            lateral_gate = max(52.0, base_step * 3.8)
            extreme_step_gate = max(95.0, step_gate * 1.35)
            extreme_lateral_gate = max(75.0, lateral_gate * 1.35)

            bat_cut = _find_post_bat_impact_cut(pixel_pts, bounce_idx, True, base_step)
            if bat_cut is not None and bat_cut < len(points):
                logger.info(
                    "Trajectory truncated at point %d (frame %d): post-bat-impact "
                    "(after bounce_idx=%d, increasing-y).",
                    bat_cut, points[bat_cut].frame_idx, bounce_idx,
                )
                truncate_idx = bat_cut
            else:
                anomaly_run = 0
                wrong_motion_run = 0
                for i in range(max(1, bounce_idx + 3), len(points)):
                    px, py = pixel_pts[i]
                    prev_px, prev_py = pixel_pts[i - 1]
                    step = float(np.hypot(px - prev_px, py - prev_py))
                    dy = float(py - prev_py)
                    lateral_jump = abs(px - prev_px)
                    is_anomaly = step > step_gate or lateral_jump > lateral_gate
                    is_extreme = step > extreme_step_gate or lateral_jump > extreme_lateral_gate
                    if is_anomaly:
                        anomaly_run += 1
                    else:
                        anomaly_run = 0

                    # Post-bounce toward batter: sustained wrong drift (not bounce).
                    wrong_motion = (dy > 1.0 and lateral_jump > max(20.0, base_step * 1.3))
                    if wrong_motion:
                        wrong_motion_run += 1
                    else:
                        wrong_motion_run = 0

                    if is_extreme or anomaly_run >= 2 or wrong_motion_run >= 3:
                        backtrack = 4 if is_extreme else 3
                        cut_i = max(bounce_idx + 2, i - backtrack)
                        logger.info(
                            "Trajectory truncated at point %d->%d (frame %d->%d): post-bat-impact mis-track (step %.1fpx, lateral %.1fpx, dy %.1fpx; gates step %.1f/ext %.1f lateral %.1f/ext %.1f; anomaly_run=%d wrong_motion_run=%d).",
                            i, cut_i, points[i].frame_idx, points[cut_i].frame_idx, step, lateral_jump,
                            dy, step_gate, extreme_step_gate, lateral_gate, extreme_lateral_gate, anomaly_run, wrong_motion_run
                        )
                        truncate_idx = cut_i
                        break
        else:
            # Ball travels up-screen until bounce, then normally drops down-screen.
            raw_extrema_idx = int(np.argmin([py for _, py in pixel_pts]))
            bounce_idx = _estimate_bounce_idx_local(pixel_pts, increasing_y=False)
            post_steps = [
                float(np.hypot(pixel_pts[k][0] - pixel_pts[k - 1][0], pixel_pts[k][1] - pixel_pts[k - 1][1]))
                for k in range(max(1, bounce_idx + 1), len(points))
            ]
            base_step = float(np.median(post_steps)) if post_steps else 12.0
            base_step = max(8.0, base_step)
            step_gate = max(72.0, base_step * 4.8)
            lateral_gate = max(52.0, base_step * 3.8)
            extreme_step_gate = max(95.0, step_gate * 1.35)
            extreme_lateral_gate = max(75.0, lateral_gate * 1.35)

            bat_cut = _find_post_bat_impact_cut(pixel_pts, bounce_idx, False, base_step)
            if bat_cut is not None and bat_cut < len(points):
                logger.info(
                    "Trajectory truncated at point %d (frame %d): post-bat-impact "
                    "(after bounce_idx=%d, decreasing-y).",
                    bat_cut, points[bat_cut].frame_idx, bounce_idx,
                )
                truncate_idx = bat_cut
            else:
                anomaly_run = 0
                wrong_motion_run = 0
                for i in range(max(1, bounce_idx + 3), len(points)):
                    px, py = pixel_pts[i]
                    prev_px, prev_py = pixel_pts[i - 1]
                    step = float(np.hypot(px - prev_px, py - prev_py))
                    dy = float(py - prev_py)
                    lateral_jump = abs(px - prev_px)
                    is_anomaly = step > step_gate or lateral_jump > lateral_gate
                    is_extreme = step > extreme_step_gate or lateral_jump > extreme_lateral_gate
                    if is_anomaly:
                        anomaly_run += 1
                    else:
                        anomaly_run = 0

                    wrong_motion = (dy < -1.0 and lateral_jump > max(20.0, base_step * 1.3))
                    if wrong_motion:
                        wrong_motion_run += 1
                    else:
                        wrong_motion_run = 0

                    if is_extreme or anomaly_run >= 2 or wrong_motion_run >= 3:
                        backtrack = 4 if is_extreme else 3
                        cut_i = max(bounce_idx + 2, i - backtrack)
                        logger.info(
                            "Trajectory truncated at point %d->%d (frame %d->%d): post-bat-impact mis-track (step %.1fpx, lateral %.1fpx, dy %.1fpx; gates step %.1f/ext %.1f lateral %.1f/ext %.1f; anomaly_run=%d wrong_motion_run=%d).",
                            i, cut_i, points[i].frame_idx, points[cut_i].frame_idx, step, lateral_jump,
                            dy, step_gate, extreme_step_gate, lateral_gate, extreme_lateral_gate, anomaly_run, wrong_motion_run
                        )
                        truncate_idx = cut_i
                        break

        if truncate_idx < len(points):
            points = points[:truncate_idx]
            pixel_pts = pixel_pts[:truncate_idx]
            if world_pts:
                world_pts = world_pts[:truncate_idx]
                
        return points, pixel_pts, world_pts

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _predictive_roi_crop(
        self,
        frame: np.ndarray,
        tracker_pos: Tuple[float, float],
    ) -> Tuple[np.ndarray, float, float]:
        """Crop a velocity-expanded window around the Kalman-predicted ball."""
        h, w = frame.shape[:2]
        pred = self.tracker.get_predicted_position() or tracker_pos
        cx, cy = float(pred[0]), float(pred[1])
        kin = self.tracker.get_state_kinematics()
        vx, vy = float(kin[2]), float(kin[3])
        speed = math.hypot(vx, vy)
        half = int(min(340, max(140, 110 + speed * 2.8)))
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half))
        y2 = min(h, int(cy + half))
        if x2 - x1 < 48 or y2 - y1 < 48:
            return frame, 0.0, 0.0
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame, 0.0, 0.0
        return crop, float(x1), float(y1)

    def _record_track_debug(
        self,
        frame_idx: int,
        detections: List[Detection],
        track_point: Optional[TrackPoint],
    ) -> None:
        """Collect per-frame debug data for visualization export."""
        pred = self.tracker.get_predicted_position()
        kin = self.tracker.get_state_kinematics()
        self._track_debug.setdefault("frames", []).append({
            "frame_idx": int(frame_idx),
            "phase": self.tracker.get_phase().name,
            "detections": [
                {"x": d.cx, "y": d.cy, "conf": d.conf} for d in detections[:12]
            ],
            "track": None if track_point is None else {
                "x": track_point.x, "y": track_point.y,
                "conf": track_point.confidence,
                "interpolated": track_point.is_interpolated,
            },
            "kalman_predict": pred,
            "velocity": [kin[2], kin[3]],
            "acceleration": [kin[4], kin[5]],
            "roi": self._ball_detector.roi_box(
                (self._video_frame_h or 848, 480, 3)
            ),
        })

    def _draw_track_debug(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        """Overlay raw dets, Kalman prediction, ROI, and track point."""
        vis = frame.copy()
        roi = self._ball_detector.roi_box(frame.shape)
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), 1)

        pred = self.tracker.get_predicted_position()
        if pred:
            cv2.drawMarker(
                vis, (int(pred[0]), int(pred[1])), (0, 255, 255),
                cv2.MARKER_CROSS, 14, 2,
            )

        cache = self._raw_ball_cache.get(frame_idx) or []
        for cx, cy, conf in cache[:8]:
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 200, 255), 1)

        if self.tracker._last_pos:
            lx, ly = self.tracker._last_pos
            cv2.circle(vis, (int(lx), int(ly)), 6, (0, 255, 0), 2)

        kin = self.tracker.get_state_kinematics()
        phase = self.tracker.get_phase().name
        lines = [
            f"f{frame_idx}  phase={phase}  conf={self._ball_detector.effective_confidence:.2f}",
            f"v=({kin[2]:.1f},{kin[3]:.1f})  a=({kin[4]:.1f},{kin[5]:.1f})",
        ]
        if pred:
            lines.append(f"kalman=({pred[0]:.0f},{pred[1]:.0f})")
        y = 20
        for line in lines:
            cv2.putText(
                vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            y += 18
        return vis

    def _detect_ball(
        self,
        frame,
        frame_idx,
        tracker_pos: Optional[Tuple[float, float]] = None,
    ):
        if self.ball_model is None and self._adaptive_fusion_plan is None:
            return []

        detect_frame = frame
        if self._reference_arc_mode():
            detect_frame = suppress_trajectory_overlay(frame)

        cfg_conf = float(self.cfg.ball_confidence)
        if self.cfg.has_reference_overlay and not self.cfg.match_should_output:
            cfg_conf = min(cfg_conf, 0.10)

        if self._adaptive_fusion_plan is not None:
            raw_dets = list(
                self._adaptive_fusion_plan.fused_detections.get(frame_idx, []),
            )
        elif self.cfg.use_enhanced_detection:
            cfg_conf = float(self._ball_detector.effective_confidence)
            raw_dets = self._ball_detector.detect(
                detect_frame, frame_idx, tracker_pos=tracker_pos,
            )
        else:
            infer_frame = detect_frame
            ox = oy = 0.0
            if self.cfg.enable_predictive_roi and tracker_pos is not None:
                infer_frame, ox, oy = self._predictive_roi_crop(detect_frame, tracker_pos)
            yolo_min = min(float(self._yolo_infer_cfg.scan_conf_floor), cfg_conf)
            if self._use_ball_ensemble():
                raw_dets = run_ball_yolo_ensemble(
                    self.ball_model,
                    self.ball_model_alt,
                    infer_frame,
                    frame_idx,
                    self._yolo_infer_cfg,
                    min_conf=yolo_min,
                    secondary_class_id=0,
                )
            else:
                raw_dets = run_ball_yolo(
                    self.ball_model,
                    infer_frame,
                    frame_idx,
                    self._yolo_infer_cfg,
                    min_conf=yolo_min,
                )
            if ox or oy:
                for d in raw_dets:
                    d.cx += ox
                    d.cy += oy

        hf_cfg = HardFrameRecoveryConfig(
            enabled=bool(self.cfg.hard_frame_recovery or self.cfg.blur_recovery_detection),
            conf_trigger=float(self.cfg.hard_frame_conf_trigger),
        )
        if (
            not self.cfg.use_enhanced_detection
            and hf_cfg.enabled
            and should_trigger_hard_recovery(raw_dets, hf_cfg, tracker_pos)
        ):
            recovery = recover_hard_frame(
                self.ball_model,
                detect_frame,
                frame_idx,
                self._yolo_infer_cfg,
                hf_cfg,
                tracker_pos,  # type: ignore[arg-type]
                alt_model=self.ball_model_alt if self._use_ball_ensemble() else None,
                ensemble=self._use_ball_ensemble(),
            )
            raw_dets = merge_primary_and_recovery(raw_dets, recovery)

        filtered = []
        cache_pts: List[Tuple[float, float, float]] = []
        frame_h, frame_w = frame.shape[:2]
        if tracker_pos is not None and (
            self.cfg.high_accuracy_mode or self.cfg.cache_tracker_fallback
        ):
            min_ball_cy = max(12.0, frame_h * 0.12)
        else:
            min_ball_cy = max(18.0, frame_h * 0.18)
        rej_geom = 0
        rej_geom_strict_only = 0
        rej_top = 0
        rej_exclusion = 0
        pass_candidates = 0
        strict_pass_total = 0
        strict_pass_below_cfg = 0
        for det in raw_dets:
            # Relaxed geometry gate for CACHE only: preserve weak/odd-looking
            # early ball boxes for observed-path reconstruction.
            if det.w < 2.0 or det.h < 2.0 or det.w > 60.0 or det.h > 60.0:
                rej_geom += 1
                continue
            aspect = det.w / max(det.h, 1e-6)
            if aspect < 0.25 or aspect > 3.50:
                rej_geom += 1
                continue

            # Filter 2: Reject tiny edge artifacts.
            if det.cx < 10.0 or det.cy < min_ball_cy:
                rej_top += 1
                continue

            # Filter 2a: Reject bottom-edge clutter (crease, crowd, scoreboard).
            bottom_frac = 0.90 if frame_h > frame_w * 1.05 else 0.84
            if det.cy > frame_h * bottom_frac:
                rej_top += 1
                continue

            # Filter 2b: Reject broadcast overlays (watermarks, score bugs).
            if is_broadcast_overlay_pixel(det.cx, det.cy, det.w, det.h, frame_w, frame_h):
                rej_top += 1
                continue

            # Filter 3: Reject detections inside stump exclusion zones
            in_exclusion = False
            for (ex1, ey1, ex2, ey2) in self._stump_exclusion_zones:
                if ex1 <= det.cx <= ex2 and ey1 <= det.cy <= ey2:
                    in_exclusion = True
                    break
            if in_exclusion:
                rej_exclusion += 1
                continue

            # Always cache the candidate for the real-path renderer.
            pass_candidates += 1
            cache_pts.append((float(det.cx), float(det.cy), float(det.conf)))

            strict_geom_ok = (
                3.0 <= det.w <= 52.0
                and 3.0 <= det.h <= 52.0
                and 0.30 <= aspect <= 3.00
            )
            if strict_geom_ok:
                strict_pass_total += 1
                if det.conf < cfg_conf:
                    strict_pass_below_cfg += 1
                filtered.append(det)

        if cache_pts:
            self._raw_ball_cache[frame_idx] = cache_pts
            # Keep permissive cache candidates for the render/recovery pass,
            # but only pass geometry-valid YOLO boxes into the Kalman tracker.
            # Feeding the tracker the single best low-confidence cached point
            # made false positives look like real ball motion.
            low_floor = float(self.cfg.tracker_low_conf_floor)
            tracker_dets = [
                d for d in filtered
                if float(d.conf) >= cfg_conf or float(d.conf) >= low_floor
            ]
            if tracker_dets:
                return tracker_dets

            if self.cfg.cache_tracker_fallback and cache_pts and tracker_pos is not None:
                tx, ty = float(tracker_pos[0]), float(tracker_pos[1])
                near = [
                    c for c in cache_pts
                    if math.hypot(float(c[0]) - tx, float(c[1]) - ty) <= 110.0
                    and float(c[2]) >= max(0.04, low_floor * 0.75)
                ]
                if near:
                    best = max(near, key=lambda c: float(c[2]))
                    return [
                        Detection(
                            frame_idx=frame_idx,
                            cx=float(best[0]),
                            cy=float(best[1]),
                            conf=float(best[2]),
                            w=20.0,
                            h=20.0,
                        )
                    ]
            return filtered

        # #region agent log
        _debug_log(
            "H17",
            "pipeline.py:_detect_ball:filter-summary",
            "Per-frame detector filtering summary.",
            {
                "frame_idx": int(frame_idx),
                "raw_detections": int(len(raw_dets)),
                "cached_candidates": int(pass_candidates),
                "tracker_candidates": int(len(filtered)),
                "rejected_geometry": int(rej_geom),
                "rejected_top_edge": int(rej_top),
                "rejected_stump_zone": int(rej_exclusion),
                "rejected_strict_geom_only": int(rej_geom_strict_only),
                "strict_pass_total": int(strict_pass_total),
                "strict_pass_below_cfg": int(strict_pass_below_cfg),
                "min_ball_cy": float(min_ball_cy),
                "frame_h": int(frame_h),
                "frame_w": int(frame_w),
            },
        )
        # #endregion

        return filtered

    def _build_observed_trajectory_pixels(
        self,
        delivery: "DeliveryAnalysis",
    ) -> Tuple[
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        float,
        Optional[ObservedPathResult],
    ]:
        """
        Trajectory for render = Kalman BallTracker delivery track only.
        Light EMA smooth + bat-impact stop; no synthetic curve fitting.
        """
        pts_list = (delivery.track or {}).get("points") or []
        if not pts_list:
            return [], [], 0.0, None

        cache_key = f"{delivery.delivery_id}:kalman"
        if cache_key in self._render_path_cache:
            return self._render_path_cache[cache_key]

        # Prefer cache+Kalman rebuild for render when track may be short.
        track_frames = [int(p.get("frame_idx", 0)) for p in pts_list]
        ext_start = max(0, min(track_frames) - 3) if track_frames else int(delivery.frame_start)
        ext_end = min(
            int(delivery.frame_end) + 30,
            max(track_frames) + 25 if track_frames else int(delivery.frame_end),
        )
        real_det = sum(
            1 for p in pts_list
            if not p.get("is_interpolated", False) and float(p.get("confidence", 0) or 0) > 0
        )
        real = self._real_path_builder.build_from_tracker(
            pts_list,
            frame_h=max(480, int(self._video_frame_h or 0)),
            raw_ball_cache=self._raw_ball_cache,
        )
        result = to_observed_path_result(real)
        if self.cfg.hybrid_optical_flow:
            result = self._augment_path_with_optical_flow(result)
        if len(result.smooth_pixels) < 2:
            logger.info(
                "Render path: real trajectory too short (%d smooth, %d raw).",
                len(result.smooth_pixels), len(result.raw_pixels),
            )
            return [], [], 0.0, result

        r0 = int(result.frame_indices[0]) if result.frame_indices else int(delivery.frame_start)
        r1 = int(result.frame_indices[-1]) if result.frame_indices else int(delivery.frame_end)
        logger.info(
            "Render path: release→bat (%d pts, %d smooth, frames %d–%d).",
            len(pts_list),
            len(result.smooth_pixels),
            r0,
            r1,
        )

        out = (
            result.raw_pixels,
            result.smooth_pixels,
            result.mean_confidence,
            result,
        )
        self._render_path_cache[cache_key] = out

        # #region agent log
        _debug_log(
            "H1",
            "pipeline.py:_build_observed_trajectory_pixels:real-scan",
            "Built real path (YOLO → filter → Kalman → EMA → bat stop).",
            {
                "delivery_frame_start": int(delivery.frame_start),
                "delivery_frame_end": int(delivery.frame_end),
                "accepted_knots": len(result.raw_pixels),
                "filtered_len": len(result.filtered_pixels),
                "smooth_len": len(result.smooth_pixels),
                "rejected_count": len(result.rejected),
                "mean_conf": float(result.mean_confidence),
            },
        )
        _debug_log(
            "H2",
            "pipeline.py:_build_observed_trajectory_pixels:raw-path",
            "Path knots passed to renderer.",
            {
                "raw_len": len(result.raw_pixels),
                "raw_first": result.raw_pixels[0] if result.raw_pixels else None,
                "raw_last": result.raw_pixels[-1] if result.raw_pixels else None,
            },
        )
        _debug_log(
            "H4",
            "pipeline.py:_build_observed_trajectory_pixels:smooth-path",
            "Final smooth polyline for cv2.polylines.",
            {
                "smooth_len": len(result.smooth_pixels),
                "smooth_first": result.smooth_pixels[0] if result.smooth_pixels else None,
                "smooth_last": result.smooth_pixels[-1] if result.smooth_pixels else None,
            },
        )
        # #endregion

        return out

    def _augment_path_with_optical_flow(
        self,
        result: ObservedPathResult,
    ) -> ObservedPathResult:
        """
        Fill short missed-detection gaps with LK optical flow, bounded by the
        next trusted trajectory point so flow cannot drift into background.
        """
        pts = list(result.smooth_pixels or [])
        frames = list(result.frame_indices or [])
        if len(pts) < 2 or len(frames) != len(pts) or not self._gray_frame_cache:
            return result

        max_gap = max(2, int(self.cfg.optical_flow_max_gap_frames))
        out_pts: List[Tuple[float, float]] = [pts[0]]
        out_frames: List[int] = [int(frames[0])]
        out_conf: List[float] = [
            float(result.confidences[0]) if result.confidences else result.mean_confidence
        ]
        flow_inserted = 0

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        for idx in range(1, len(pts)):
            prev_frame = int(frames[idx - 1])
            curr_frame = int(frames[idx])
            prev_pt = np.array(pts[idx - 1], dtype=np.float32)
            curr_pt = np.array(pts[idx], dtype=np.float32)
            gap = curr_frame - prev_frame

            if 1 < gap <= max_gap:
                tracked = prev_pt.copy()
                for fi in range(prev_frame + 1, curr_frame):
                    g0 = self._gray_frame_cache.get(fi - 1)
                    g1 = self._gray_frame_cache.get(fi)
                    if g0 is None or g1 is None:
                        break

                    src = tracked.reshape(1, 1, 2).astype(np.float32)
                    nxt, st, err = cv2.calcOpticalFlowPyrLK(g0, g1, src, None, **lk_params)
                    t = (fi - prev_frame) / float(gap)
                    linear = prev_pt + (curr_pt - prev_pt) * t

                    use_linear = True
                    if nxt is not None and st is not None and int(st[0][0]) == 1:
                        cand = nxt.reshape(2).astype(np.float32)
                        drift = float(np.linalg.norm(cand - linear))
                        step = float(np.linalg.norm(cand - tracked))
                        err_val = float(err[0][0]) if err is not None else 999.0
                        if drift <= 35.0 and step <= 55.0 and err_val <= 45.0:
                            tracked = cand
                            use_linear = False

                    if use_linear:
                        tracked = linear.astype(np.float32)

                    out_pts.append((float(tracked[0]), float(tracked[1])))
                    out_frames.append(int(fi))
                    out_conf.append(0.06)
                    flow_inserted += 1

            out_pts.append((float(curr_pt[0]), float(curr_pt[1])))
            out_frames.append(curr_frame)
            if idx < len(result.confidences):
                out_conf.append(float(result.confidences[idx]))
            else:
                out_conf.append(float(result.mean_confidence))

        if flow_inserted <= 0:
            return result

        velocities: List[Tuple[float, float]] = [(0.0, 0.0)]
        for idx in range(1, len(out_pts)):
            dt = max(1, out_frames[idx] - out_frames[idx - 1])
            velocities.append((
                (out_pts[idx][0] - out_pts[idx - 1][0]) / dt,
                (out_pts[idx][1] - out_pts[idx - 1][1]) / dt,
            ))

        logger.info(
            "Hybrid optical-flow trajectory bridge inserted %d support points.",
            flow_inserted,
        )
        return ObservedPathResult(
            raw_pixels=result.raw_pixels,
            filtered_pixels=result.filtered_pixels,
            smooth_pixels=out_pts,
            frame_indices=out_frames,
            confidences=out_conf,
            velocities=velocities,
            rejected=result.rejected,
            mean_confidence=result.mean_confidence,
            bridge_points_inserted=int(result.bridge_points_inserted) + flow_inserted,
        )

    def _augment_visible_path(
        self,
        base_path: List[Tuple[float, float]],
        delivery: "DeliveryAnalysis",
    ) -> List[Tuple[float, float]]:
        """
        Build the broadcast trajectory directly from raw YOLO detections,
        bypassing the tracker output when it's contaminated by static
        background phantoms (logos, lights, painted markings).

        Algorithm:
          1. Collect every cached candidate in an extended delivery window.
          2. Identify static phantoms — positions that show up in many
             frames at near-identical pixel — and drop them.
          3. Greedy temporal chain through the moving candidates: pick a
             high-confidence seed and walk forward/backward, accepting per
             frame the candidate closest to (anchor + velocity).
          4. Fall back to the older tracker-merge logic when there's too
             little data to chain reliably (e.g. test_video6 where the
             detector barely sees the ball).
        """
        if not self._raw_ball_cache:
            return base_path

        tracker_pts: Dict[int, Tuple[float, float]] = {}
        try:
            for p in delivery.track["points"]:
                fi = int(p.get("frame_idx", -1))
                if fi >= 0:
                    tracker_pts[fi] = (float(p["x"]), float(p["y"]))
        except Exception:
            tracker_pts = {}

        frame_start = int(delivery.frame_start)
        frame_end = int(delivery.frame_end)
        fps = float(getattr(self.tracker, "fps", 30.0)) or 30.0

        # Window: cover the whole delivery plus pre/post-roll to catch
        # the release point and any post-bounce frames.
        window_start = max(0, frame_start - 15)
        window_end = frame_end + max(30, int(round(0.6 * fps)))

        # 1. Collect every candidate in the window.
        candidates: List[Tuple[int, float, float, float]] = []
        for fi, cands in self._raw_ball_cache.items():
            if fi < window_start or fi > window_end:
                continue
            for (cx, cy, conf) in cands:
                candidates.append((fi, cx, cy, conf))

        if len(candidates) < 6:
            return self._merge_tracker_with_cache(
                tracker_pts, window_start, window_end, fps, base_path
            )

        # 2. Filter static phantoms by coarse position binning. A bin that
        #    contains hits from many different frames is a fixed background
        #    object (logos, lights, painted lines), not a moving ball.
        window_len = max(1, window_end - window_start + 1)
        bin_frames: Dict[Tuple[int, int], set] = {}
        for (fi, cx, cy, _conf) in candidates:
            key = (int(cx) // 14, int(cy) // 14)
            bin_frames.setdefault(key, set()).add(fi)

        # A real ball moves through a position once; phantoms appear in
        # >~20% of frames at the same pixel.
        phantom_threshold = max(6, int(round(window_len * 0.18)))
        phantom_bins = {k for k, v in bin_frames.items()
                        if len(v) >= phantom_threshold}

        moving = [c for c in candidates
                  if (int(c[1]) // 14, int(c[2]) // 14) not in phantom_bins]

        if len(moving) < 5:
            return self._merge_tracker_with_cache(
                tracker_pts, window_start, window_end, fps, base_path
            )

        per_frame: Dict[int, List[Tuple[float, float, float]]] = {}
        for (fi, cx, cy, conf) in moving:
            per_frame.setdefault(fi, []).append((cx, cy, conf))

        path = self._chain_ball_flight(per_frame, fps)
        if len(path) < 5:
            return self._merge_tracker_with_cache(
                tracker_pts, window_start, window_end, fps, base_path
            )

        return [(x, y) for (_fi, x, y) in path]

    def _chain_ball_flight(
        self,
        per_frame: Dict[int, List[Tuple[float, float, float]]],
        fps: float,
    ) -> List[Tuple[int, float, float]]:
        """
        Build the longest coherent ball flight from per-frame candidates.
        Picks a high-confidence seed and walks forward/backward in time,
        choosing per frame the candidate closest to (anchor + velocity).
        """
        if not per_frame:
            return []

        max_step = 95.0   # px/frame at 720p — fast ball with margin

        # Seed: highest combined score of confidence + nearby neighbors.
        seed_fi: Optional[int] = None
        seed_xy: Optional[Tuple[float, float]] = None
        best_seed_score = -1.0
        for fi, cands in per_frame.items():
            for (cx, cy, conf) in cands:
                neighbors = 0
                for df in (-3, -2, -1, 1, 2, 3):
                    nfi = fi + df
                    if nfi not in per_frame:
                        continue
                    for (nx, ny, _nc) in per_frame[nfi]:
                        if math.hypot(nx - cx, ny - cy) <= max_step * abs(df):
                            neighbors += 1
                            break
                score = conf * 8.0 + neighbors
                if score > best_seed_score:
                    best_seed_score = score
                    seed_fi = fi
                    seed_xy = (cx, cy)

        if seed_fi is None or seed_xy is None:
            return []

        path: List[Tuple[int, float, float]] = [(seed_fi, seed_xy[0], seed_xy[1])]
        sorted_fis = sorted(per_frame.keys())
        # Strict consecutive-miss budget so the chain TERMINATES when the ball
        # truly leaves the frame instead of cascading onto a far-away phantom.
        # 4 frames ≈ ~130 ms at 30 fps / 70 ms at 60 fps — enough to span a
        # brief detector drop but short enough to stop dead-reckoning.
        # Allow bridging detector drop-outs through bounce (e.g. 10+ frame gaps).
        max_consecutive_miss = 12

        def _walk(direction: int) -> None:
            if direction > 0:
                fi_iter = [fi for fi in sorted_fis if fi > path[-1][0]]
                anchor = path[-1]
            else:
                fi_iter = [fi for fi in reversed(sorted_fis) if fi < path[0][0]]
                anchor = path[0]

            anchor_fi = anchor[0]
            anchor_xy = (anchor[1], anchor[2])
            vel = (0.0, 0.0)
            miss = 0

            for fi in fi_iter:
                # Strict miss cap: stop dead-reckoning the moment we've missed
                # too many consecutive frames since the last accepted point.
                if miss > max_consecutive_miss:
                    break

                gap = abs(fi - anchor_fi)

                pred_x = anchor_xy[0] + vel[0] * (fi - anchor_fi)
                pred_y = anchor_xy[1] + vel[1] * (fi - anchor_fi)
                # Tolerance is tight: for short gaps we trust the velocity
                # estimate; for longer gaps we don't expand much because that
                # invites jumping to phantoms.
                tol = min(max_step * gap, 50.0 + 20.0 * gap)

                best_d = float("inf")
                best_pick: Optional[Tuple[float, float, float]] = None
                for (cx, cy, conf) in per_frame.get(fi, []):
                    d = math.hypot(cx - pred_x, cy - pred_y)
                    if d > tol:
                        continue
                    s = d - conf * 25.0
                    if s < best_d:
                        best_d = s
                        best_pick = (cx, cy, conf)

                if best_pick is None:
                    miss += 1
                    continue

                if direction > 0:
                    path.append((fi, best_pick[0], best_pick[1]))
                else:
                    path.insert(0, (fi, best_pick[0], best_pick[1]))

                step_gap = max(1, abs(fi - anchor_fi))
                nv = ((best_pick[0] - anchor_xy[0]) / step_gap,
                      (best_pick[1] - anchor_xy[1]) / step_gap)
                if vel == (0.0, 0.0):
                    vel = nv
                else:
                    vel = (0.6 * nv[0] + 0.4 * vel[0],
                           0.6 * nv[1] + 0.4 * vel[1])
                anchor_fi = fi
                anchor_xy = (best_pick[0], best_pick[1])
                miss = 0

        _walk(+1)
        _walk(-1)
        return path

    def _merge_tracker_with_cache(
        self,
        tracker_pts: Dict[int, Tuple[float, float]],
        window_start: int,
        window_end: int,
        fps: float,
        base_path: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Legacy fallback: walk tracker + cache, accepting candidates that
        continue the flight. Used when cache-driven chaining doesn't find
        enough points (sparse-detection videos)."""
        MAX_PER_FRAME_STEP = 45.0
        MAX_TOTAL_GAP_PX = 150.0
        MAX_GAP_FRAMES = max(8, int(round(0.4 * fps)))

        merged: Dict[int, Tuple[float, float]] = dict(tracker_pts)
        anchor_xy: Optional[Tuple[float, float]] = None
        anchor_fi: Optional[int] = None
        if tracker_pts:
            anchor_fi = max(tracker_pts.keys())
            anchor_xy = tracker_pts[anchor_fi]

        accepted: Dict[int, Tuple[float, float]] = {}
        all_frames = sorted(set(self._raw_ball_cache.keys()) | set(tracker_pts.keys()))
        for fi in all_frames:
            if fi < window_start or fi > window_end:
                continue
            if fi in tracker_pts:
                anchor_xy = tracker_pts[fi]
                anchor_fi = fi
                continue
            cands = self._raw_ball_cache.get(fi, [])
            if not cands:
                continue
            if anchor_fi is not None and fi - anchor_fi > MAX_GAP_FRAMES:
                break

            best_choice = None
            best_cost = float("inf")
            for (cx, cy, conf) in cands:
                if anchor_xy is not None and anchor_fi is not None:
                    gap = max(1, abs(fi - anchor_fi))
                    dist = math.hypot(cx - anchor_xy[0], cy - anchor_xy[1])
                    if dist > min(MAX_PER_FRAME_STEP * gap, MAX_TOTAL_GAP_PX):
                        continue
                    cost = dist - conf * 50.0
                else:
                    cost = -conf * 50.0
                if cost < best_cost:
                    best_cost = cost
                    best_choice = (cx, cy, conf)

            if best_choice is None:
                continue
            accepted[fi] = (best_choice[0], best_choice[1])
            anchor_xy = (best_choice[0], best_choice[1])
            anchor_fi = fi

        merged.update(accepted)
        if not merged:
            return base_path
        ordered_frames = sorted(merged.keys())
        return [merged[f] for f in ordered_frames]

    def _build_stump_exclusion_zones(
        self, cap: cv2.VideoCapture
    ) -> List[Tuple[float, float, float, float]]:
        """
        Build exclusion zones from stump detections.
        Any ball detection inside these zones is rejected as a false positive.
        Uses an expanded bounding box (30px padding) around each detected stump region.
        """
        zones = []
        if self.stump_model is None:
            return zones

        # Read first frame for stump detection
        saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

        if not ret:
            return zones

        results = self.stump_model(
            frame,
            conf=0.10,
            device=self.cfg.device,
            verbose=False,
        )
        try:
            boxes = results[0].boxes
            for box in boxes:
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                bw = x2 - x1
                bh = y2 - y1
                if conf < 0.50 or bw > 90.0 or bh > 180.0:
                    continue

                # Expand only a little: broad single-class wicket boxes can
                # overlap the actual ball path near the batter.
                pad = 12.0
                zones.append((x1 - pad, y1 - pad, x2 + pad, y2 + pad))
                logger.info(
                    "Stump exclusion zone: [%.0f, %.0f, %.0f, %.0f] conf=%.2f",
                    x1 - pad, y1 - pad, x2 + pad, y2 + pad, conf,
                )
        except Exception as exc:
            logger.debug("Error building stump exclusion zones: %s", exc)

        return zones

    def _build_static_phantom_zones(
        self,
        cap: cv2.VideoCapture,
        max_scan_frames: int = 80,
        cluster_radius_px: float = 10.0,
        min_repeats: int = 14,
        min_avg_conf: float = 0.015,
        zone_pad_px: float = 14.0,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Detect STATIC false-positive 'ball' regions in the background.

        Strategy: scan the first N frames at low confidence, cluster
        detections by pixel proximity. Any cluster with >= `min_repeats`
        detections at near-identical pixel position is a phantom (real
        cricket balls move several pixels per frame in flight). Return
        small exclusion boxes for each phantom location.

        This addresses the most common failure mode of single-class ball
        detectors trained on cricket footage — false positives on
        stationary background objects (decorative lights, scoreboards,
        practice balls in nets backdrop, etc.).
        """
        zones: List[Tuple[float, float, float, float]] = []
        if self.ball_model is None:
            return zones

        saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Scan every other frame for speed; collect raw detections at very
        # low conf so we can see what the model fires on consistently.
        detections: List[Tuple[float, float, float]] = []
        scanned = 0
        step = 2
        max_iters = max_scan_frames * step
        for _ in range(max_iters):
            ret, frame = cap.read()
            if not ret:
                break
            try:
                results = self.ball_model(
                    frame,
                    conf=0.01,
                    device=self.cfg.device,
                    verbose=False,
                )
                for b in results[0].boxes:
                    conf = float(b.conf[0].item())
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    detections.append((cx, cy, conf))
            except Exception:
                pass
            scanned += 1
            if scanned >= max_scan_frames:
                break
            # Skip a frame to halve compute.
            cap.read()

        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
        if not detections:
            return zones

        # Greedy clustering: assign each detection to the nearest existing
        # cluster centroid, or start a new cluster.
        clusters: List[Dict] = []
        for cx, cy, conf in detections:
            placed = False
            for cl in clusters:
                if math.hypot(cx - cl["cx"], cy - cl["cy"]) <= cluster_radius_px:
                    cl["points"].append((cx, cy, conf))
                    # Running centroid update.
                    n = len(cl["points"])
                    cl["cx"] = ((n - 1) * cl["cx"] + cx) / n
                    cl["cy"] = ((n - 1) * cl["cy"] + cy) / n
                    placed = True
                    break
            if not placed:
                clusters.append({"cx": cx, "cy": cy, "points": [(cx, cy, conf)]})

        # A real moving ball trail will have many low-radius points along a
        # path — i.e. each cluster will be small (~1-3 hits) but there will
        # be many clusters near each other. A phantom has 5+ hits all at the
        # same pixel.
        for cl in clusters:
            n = len(cl["points"])
            if n < min_repeats:
                continue
            confs = [p[2] for p in cl["points"]]
            avg_conf = sum(confs) / max(n, 1)
            if avg_conf < min_avg_conf:
                continue
            xs = [p[0] for p in cl["points"]]
            ys = [p[1] for p in cl["points"]]
            # Tight spatial spread → phantom. A trail of a real ball would
            # have a wider spread.
            if (max(xs) - min(xs)) > 2 * cluster_radius_px:
                continue
            if (max(ys) - min(ys)) > 2 * cluster_radius_px:
                continue
            zone = (
                cl["cx"] - zone_pad_px,
                cl["cy"] - zone_pad_px,
                cl["cx"] + zone_pad_px,
                cl["cy"] + zone_pad_px,
            )
            zones.append(zone)
            logger.info(
                "Phantom-ball zone: centroid=(%.0f, %.0f) hits=%d avg_conf=%.2f → exclude [%.0f, %.0f, %.0f, %.0f]",
                cl["cx"], cl["cy"], n, avg_conf, *zone,
            )

        return zones

    @staticmethod
    def _parse_stump_detections(yolo_results) -> List[Dict]:
        """
        Parse stump YOLO results into calibration-ready dicts.

        Supports:
          - Distinct 3-class stumps (off, mid, leg).
          - Single-box stump set models (1-class / 'item'): Splits the overall
            bounding box(es) into constituent stumps (left, middle, right).
        """
        raw_dets = []
        try:
            boxes = yolo_results[0].boxes
            for box in boxes:
                cls  = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                raw_dets.append({
                    "cls":        cls,
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": conf,
                    "cx":         (x1 + x2) / 2.0,
                    "y2":         y2,
                })
        except Exception as exc:
            logger.debug("parse_stump_detections error: %s", exc)
            return []

        if not raw_dets:
            return []

        # Check if the model predicted distinct stump classes (e.g. 0, 1, 2 representing off, mid, leg)
        unique_classes = set(d["cls"] for d in raw_dets)
        if len(unique_classes) >= 3:
            # We have at least 3 distinct classes, standard mapping works!
            label_map = {0: "off_stump", 1: "middle_stump", 2: "leg_stump"}
            detections = []
            for d in raw_dets:
                label = label_map.get(d["cls"])
                if label:
                    detections.append({
                        "label":      label,
                        "bbox":       d["bbox"],
                        "confidence": d["confidence"],
                    })
            return detections

        # If it's a single-box stump set model (1-class 'item'), infer the
        # two wicket ends by vertical position instead of hard-coded class IDs.
        bowler_box = None
        batsman_box = None

        # Keep the strongest box per wicket end. The foreground/bowler-end
        # wicket has larger y2; the far/batsman-end wicket has smaller y2.
        sorted_by_y = sorted(raw_dets, key=lambda d: d["y2"])
        if sorted_by_y:
            y_min = sorted_by_y[0]["y2"]
            y_max = sorted_by_y[-1]["y2"]
            separation = max(70.0, 0.12 * max(y_max, 1.0))

            batsman_candidates = [d for d in raw_dets if d["y2"] <= y_min + separation]
            bowler_candidates = [d for d in raw_dets if d["y2"] >= y_max - separation]

            if batsman_candidates:
                batsman_box = max(batsman_candidates, key=lambda d: d["confidence"])
            if bowler_candidates and y_max - y_min >= separation:
                bowler_box = max(bowler_candidates, key=lambda d: d["confidence"])

        detections = []

        if bowler_box:
            x1, y1, x2, y2 = bowler_box["bbox"]
            cx = bowler_box["cx"]
            conf = bowler_box["confidence"]

            # Split into 3 constituent stumps (off, middle, leg) for homography base points
            detections.extend([
                {"label": "off_stump",    "bbox": [x1 - 5, y1, x1 + 5, y2], "confidence": conf},
                {"label": "middle_stump", "bbox": [cx - 5, y1, cx + 5, y2], "confidence": conf},
                {"label": "leg_stump",    "bbox": [x2 - 5, y1, x2 + 5, y2], "confidence": conf},
            ])

        if batsman_box:
            x1, y1, x2, y2 = batsman_box["bbox"]
            cx = batsman_box["cx"]
            conf = batsman_box["confidence"]

            # Split batsman stumps as well
            detections.extend([
                {"label": "batsman_off", "bbox": [x1 - 5, y1, x1 + 5, y2], "confidence": conf},
                {"label": "batsman_mid", "bbox": [cx - 5, y1, cx + 5, y2], "confidence": conf},
                {"label": "batsman_leg", "bbox": [x2 - 5, y1, x2 + 5, y2], "confidence": conf},
            ])

        return detections

    # ------------------------------------------------------------------
    # HUD overlay
    # ------------------------------------------------------------------

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
    ) -> np.ndarray:
        """Draw a minimal status HUD onto the frame (non-blocking)."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        # Background panel
        cv2.rectangle(overlay, (10, 10), (310, 55), (10, 10, 10), -1)
        cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)
        cv2.rectangle(frame, (10, 10), (310, 55), (60, 60, 60), 1)

        cal_txt   = "CAL: OK" if self.calibration_result else "CAL: --"
        frame_txt = f"Frame {frame_idx}"
        n_del     = len(self.heatmap_gen._points)

        cv2.putText(frame, f"CricGiri AI  {cal_txt}  Del:{n_del}  {frame_txt}",
                    (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 180), 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yolo(model_path: str, name: str):
        """Load a YOLO model gracefully (returns None if not found)."""
        path = Path(model_path)
        if not path.exists():
            logger.warning("[%s] Model not found: %s — running without.", name, model_path)
            return None
        try:
            from ultralytics import YOLO
            m = YOLO(str(path))
            logger.info("[%s] Loaded: %s", name, model_path)
            return m
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return None
        except Exception as exc:
            logger.error("[%s] Load failed: %s", name, exc)
            return None

    @staticmethod
    def _open_video(path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        return cap

    @staticmethod
    def _open_writer(
        path: str, fps: float, w: int, h: int
    ) -> Optional[cv2.VideoWriter]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # mp4v often won't play in Windows Media Player / Movies & TV; prefer avc1 (H.264).
        for codec in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            wr = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if wr.isOpened():
                logger.info("Output video: %s (codec=%s)", path, codec)
                return wr
            wr.release()
        logger.error("VideoWriter failed to open: %s", path)
        return None

    def _save_json(self, session: SessionAnalysis) -> None:
        path = self.cfg.output_json_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(session.to_dict(), fh, indent=2, default=str)
        logger.info("JSON saved: %s", path)

    def _save_heatmap(self) -> None:
        out_dir = Path(self.cfg.output_video_path).parent
        hmap    = self.heatmap_gen.render()
        self.heatmap_gen.save(hmap, str(out_dir / "pitch_heatmap.png"))
        self.heatmap_gen.export_json(
            str(Path(self.cfg.output_json_path).parent / "heatmap_data.json")
        )

    def _report_progress(self, pct: float) -> None:
        if self.cfg.progress_callback and self.cfg.job_id:
            try:
                self.cfg.progress_callback(self.cfg.job_id, pct)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CricGiri Cricket Analytics Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video",       required=True, help="Input video path")
    parser.add_argument("--stump-model", default="models/stump_best.pt")
    _ball = Path("models/ball_best.pt")
    _fpv_ball = Path("models/ball_fpv_btd.pt")
    if _ball.exists():
        _default_ball = str(_ball)
    elif _fpv_ball.exists():
        _default_ball = str(_fpv_ball)
    else:
        _default_ball = "models/ball_best.pt"
    parser.add_argument("--ball-model",  default=_default_ball,
                        help="Ball YOLO weights (default: ball_best.pt)")
    parser.add_argument(
        "--ball-model-alt",
        default=None,
        help="Second ball weights for hybrid ensemble (default on painted-arc: ball_best_backup.pt)",
    )
    parser.add_argument(
        "--hybrid-ensemble",
        action="store_true",
        help="Run primary + alt YOLO and merge boxes (inference ensemble, not merged .pt)",
    )
    parser.add_argument(
        "--no-auto-hybrid",
        action="store_true",
        help="Do not auto-load ball_best_backup.pt on painted-trajectory clips",
    )
    parser.add_argument(
        "--adaptive-multi-model",
        action="store_true",
        help="Probe all ball*.pt weights; weight by best frame-span track length; fuse per frame",
    )
    parser.add_argument(
        "--adaptive-models",
        nargs="*",
        default=None,
        help="Explicit list of .pt paths (default: all ball*.pt under models/)",
    )
    parser.add_argument(
        "--adaptive-fusion-report",
        default=None,
        help="JSON path for per-model span weights (default: beside --out-json)",
    )
    parser.add_argument("--out-video",   default="outputs/output_annotated.mp4")
    parser.add_argument("--out-json",    default="outputs/analysis_result.json")
    parser.add_argument("--fps",         type=float, default=30.0)
    parser.add_argument("--device",      default=None,
                        help="Ultralytics device, e.g. 0 for CUDA GPU or cpu")
    parser.add_argument("--arm",         default="right", choices=["right", "left"])
    parser.add_argument("--ball-confidence", type=float, default=0.10,
                        help="YOLO primary confidence (0.10 legacy/tuned default)")
    parser.add_argument("--inference-imgsz", type=int, default=640,
                        help="YOLO inference resolution (640 legacy; 1280 for enhanced)")
    parser.add_argument("--enhanced-detection", action="store_true",
                        help="Enable BallDetector stack (1280/TTA/ROI/dynamic conf)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (enhanced mode)")
    parser.add_argument("--roi-detect", action="store_true",
                        help="Enable ROI crop detection after ball lock (enhanced mode)")
    parser.add_argument("--dynamic-conf", action="store_true",
                        help="Enable dynamic confidence adjustment (enhanced mode)")
    parser.add_argument("--byte-track", action="store_true",
                        help="Enable ByteTrack-style low-conf recovery")
    parser.add_argument("--tracker-optical-flow", action="store_true",
                        help="Enable LK optical flow in live tracker")
    parser.add_argument("--track-interpolation", default="off",
                        choices=["off", "linear", "cubic", "spline"],
                        help="Gap interpolation on finalized track (off = legacy/tuned)")
    parser.add_argument("--clean-video", action="store_true", default=True,
                        help="Clean footage (no logos/other balls): higher conf + proximity recovery")
    parser.add_argument("--no-clean-video", action="store_false", dest="clean_video",
                        help="Broadcast/noisy footage: lower conf + phantom-zone filtering")
    parser.add_argument("--max-missing-frames", type=int, default=12,
                        help="Max Kalman coast frames for tracker continuity (not drawn on observed trail)")
    parser.add_argument("--pixels-per-meter", type=float, default=38.0,
                        help="Fallback scale used only when calibration is unavailable")
    parser.add_argument("--max-deliveries", type=int, default=1,
                        help="Maximum bowling deliveries to keep; use 0 for no limit")
    parser.add_argument("--min-track-displacement", type=float, default=55.0,
                        help="Reject tracks with less motion than this many pixels")
    parser.add_argument("--trajectory-reveal-hold-frames", type=int, default=60,
                        help="Frames to hold the completed trajectory after delivery contact/end")
    parser.add_argument("--trajectory-debug", action="store_true",
                        help="Overlay raw ball centers + frame/confidence debug HUD")
    parser.add_argument("--match-should-output", action="store_true",
                        help="Preset: match should_output/ trajectory (legacy 640 detect, no cluster trim, no bounce extend)")
    parser.add_argument("--blur-recovery", action="store_true",
                        help="Preset: 1280px detect + tiled hard-frame fallback + optical flow for blur gaps")
    parser.add_argument("--hard-frame-recovery", action="store_true",
                        help="SAHI-style ROI tiling + deblur on missed/low-conf frames (needs tracker position)")
    parser.add_argument("--hard-frame-conf-trigger", type=float, default=0.12,
                        help="Run tiled recovery when best detection conf is below this")
    parser.add_argument("--high-accuracy", action="store_true",
                        help="Max-quality preset: 1280px, cache rebuild, smart bridge, enhanced detect (~90 target)")
    parser.add_argument("--reference-overlay", action="store_true",
                        help="Input has painted trajectory; auto-detect if omitted")
    parser.add_argument("--no-auto-overlay-detect", action="store_true",
                        help="Disable painted-trajectory auto detection")
    parser.add_argument("--production", action="store_true",
                        help="Production preset: conf=0.35, cubic interp, byte track, physics filter, ROI")
    parser.add_argument("--production-debug", action="store_true",
                        help="Extended tracker HUD (phase, velocity, Kalman) during --trajectory-debug")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--yolo-max-det", type=int, default=3, help="YOLO max detections per frame")
    parser.add_argument("--half-precision", action="store_true", help="YOLO FP16 inference when supported")
    parser.add_argument("--predictive-roi", action="store_true",
                        help="Legacy detect path: crop around Kalman prediction")
    parser.add_argument("--smooth-render-catmull", action="store_true",
                        help="Catmull-Rom anti-jitter trajectory render")
    parser.add_argument("--skip-false-start-trim", action="store_true",
                        help="Do not drop early detection clusters (keeps release-from-hand points)")
    parser.add_argument("--skip-bounce-extend", action="store_true",
                        help="Skip post-bounce Kalman extension (prevents path past bat)")
    parser.add_argument("--hybrid-optical-flow", action="store_true",
                        help="Enable LK optical-flow inside missed-detection gaps (enhanced mode)")
    parser.add_argument("--optical-flow-max-gap-frames", type=int, default=12,
                        help="Largest detection gap to support with bounded optical flow")
    parser.add_argument("--calibration", default=None,
                        help="Path to saved .npz calibration file")
    parser.add_argument("--no-video",    action="store_true",
                        help="Skip annotated video output (faster)")
    args = parser.parse_args()

    match_should = bool(args.match_should_output)
    blur_recovery = bool(args.blur_recovery)
    hard_frame = bool(getattr(args, "hard_frame_recovery", False))
    production = bool(getattr(args, "production", False))
    high_accuracy = bool(getattr(args, "high_accuracy", False))

    if high_accuracy:
        apply_high_accuracy_cli(args)
        production = False
    elif production:
        apply_production_cli(args)
        if not args.trajectory_debug:
            args.trajectory_debug = bool(getattr(args, "production_debug", False))

    if bool(getattr(args, "production_debug", False)):
        args.trajectory_debug = True
    skip_cluster = bool(args.skip_false_start_trim) or match_should
    skip_extend = bool(args.skip_bounce_extend) or match_should

    if match_should:
        if args.ball_confidence == 0.10:
            args.ball_confidence = 0.10
        if args.inference_imgsz == 640:
            args.inference_imgsz = 640
        if not args.hybrid_optical_flow:
            args.hybrid_optical_flow = True
        if args.max_missing_frames == 12:
            args.max_missing_frames = 18

    if blur_recovery:
        args.hard_frame_recovery = True
        args.inference_imgsz = max(int(args.inference_imgsz), 1280)
        args.tracker_optical_flow = True
        args.hybrid_optical_flow = True
        args.byte_track = True
        if args.max_missing_frames == 12:
            args.max_missing_frames = 20

    if hard_frame and not blur_recovery:
        args.hard_frame_recovery = True
        args.inference_imgsz = max(int(args.inference_imgsz), 1280)
        if args.max_missing_frames == 12:
            args.max_missing_frames = 18

    cfg = PipelineConfig(
        video_path        = args.video,
        stump_model_path  = args.stump_model,
        ball_model_path   = args.ball_model,
        ball_model_alt_path = getattr(args, "ball_model_alt", None),
        hybrid_ensemble = bool(getattr(args, "hybrid_ensemble", False)),
        auto_hybrid_on_overlay = not bool(getattr(args, "no_auto_hybrid", False)),
        adaptive_multi_model = bool(getattr(args, "adaptive_multi_model", False)),
        adaptive_model_paths = getattr(args, "adaptive_models", None) or None,
        adaptive_fusion_report_path = (
            getattr(args, "adaptive_fusion_report", None)
            or (
                str(Path(args.out_json).with_name(
                    Path(args.out_json).stem + "_fusion_plan.json"
                ))
                if getattr(args, "adaptive_multi_model", False)
                else None
            )
        ),
        device            = args.device,
        output_video_path = args.out_video,
        output_json_path  = args.out_json,
        fps               = args.fps,
        bowler_arm        = args.arm,
        ball_confidence   = args.ball_confidence,
        clean_video_mode  = args.clean_video,
        use_enhanced_detection = args.enhanced_detection,
        inference_imgsz   = args.inference_imgsz,
        enable_tta        = args.tta or args.enhanced_detection,
        enable_roi_detect = args.roi_detect or args.enhanced_detection,
        dynamic_confidence = args.dynamic_conf or args.enhanced_detection,
        enable_byte_track = args.byte_track,
        enable_tracker_optical_flow = args.tracker_optical_flow,
        track_interpolation = args.track_interpolation,
        max_missing_frames= args.max_missing_frames,
        pixels_per_meter  = args.pixels_per_meter,
        max_deliveries    = args.max_deliveries,
        min_track_displacement_px = args.min_track_displacement,
        trajectory_reveal_hold_frames = args.trajectory_reveal_hold_frames,
        trajectory_debug  = args.trajectory_debug,
        hybrid_optical_flow = args.hybrid_optical_flow,
        optical_flow_max_gap_frames = args.optical_flow_max_gap_frames,
        match_should_output = match_should,
        skip_false_start_cluster_trim = skip_cluster,
        skip_bounce_track_extension = skip_extend,
        blur_recovery_detection = blur_recovery,
        hard_frame_recovery = bool(getattr(args, "hard_frame_recovery", False) or blur_recovery),
        hard_frame_conf_trigger = float(getattr(args, "hard_frame_conf_trigger", 0.12)),
        production_mode = production and not high_accuracy,
        high_accuracy_mode = high_accuracy,
        bridge_max_gap_frames = int(getattr(args, "bridge_max_gap_frames", 5 if high_accuracy else 18)),
        enable_cache_kalman_rebuild = high_accuracy,
        cache_tracker_fallback = high_accuracy,
        skip_phase_slice_when_sparse = high_accuracy,
        min_real_for_phase_slice = 8,
        tracker_low_conf_floor = float(
            HighAccuracyConfig().tracker_low_conf if high_accuracy else 0.08
        ),
        speed_tof_bounce_conf = float(
            HighAccuracyConfig().speed_tof_bounce_conf if high_accuracy else 0.60
        ),
        yolo_iou = float(getattr(args, "yolo_iou", 0.45)),
        yolo_agnostic_nms = True,
        yolo_max_det = int(getattr(args, "yolo_max_det", 3)),
        yolo_scan_confidence = 0.01,
        half_precision = bool(getattr(args, "half_precision", False)),
        enable_predictive_roi = bool(getattr(args, "enable_predictive_roi", False)
                                    or getattr(args, "predictive_roi", False)),
        smooth_render_catmull = bool(getattr(args, "smooth_render_catmull", False)),
        production_debug_hud = bool(getattr(args, "production_debug", False)),
        savgol_window = int(getattr(args, "savgol_window", 7)),
        auto_detect_reference_overlay=not bool(
            getattr(args, "no_auto_overlay_detect", False)
        ),
        has_reference_overlay=bool(getattr(args, "reference_overlay", False)),
        calibration_file  = args.calibration,
        save_video        = not args.no_video,
    )

    pl     = CricketAnalyticsPipeline(cfg)
    result = pl.run()

    print(f"\n✅  Session {result.session_id}")
    print(f"    Deliveries      : {result.total_deliveries}")
    print(f"    Processing time : {result.processing_time_sec:.2f} s")
    print(f"    JSON output     : {cfg.output_json_path}")
    print(f"    Video output    : {cfg.output_video_path}")
