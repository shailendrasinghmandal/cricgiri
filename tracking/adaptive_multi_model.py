"""
Adaptive multi-model ball fusion.

1. Run each ball YOLO checkpoint on the video (probe pass).
2. Chain detections per model → count track points; find best frame span per model.
3. Weight models by span quality (more linked points → higher weight).
4. Per frame, pick the detection from the highest (model_weight × confidence) source.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from analytics.real_trajectory import _pick_nearest_to_prediction
from analytics.overlay_utils import suppress_trajectory_overlay
from tracking.detection import Detection
from tracking.yolo_inference import YoloInferenceConfig, run_ball_yolo

logger = logging.getLogger(__name__)

# Skip huge / non-standard ball weights in auto-discovery.
_SKIP_NAMES = frozenset({
    "ball_best_leather_new.pt",
    "ball_kushagra_train5.pt",
})
_MAX_MODEL_BYTES = 80_000_000


@dataclass
class ModelSpanInfo:
    model_path: str
    model_name: str
    span_start: int
    span_end: int
    path_points: int
    span_points: int
    model_weight: float = 0.0
    class_id: int = 0
    path_frames: List[int] = field(default_factory=list)


@dataclass
class AdaptiveFusionPlan:
    models: List[ModelSpanInfo] = field(default_factory=list)
    fused_detections: Dict[int, List[Detection]] = field(default_factory=dict)
    per_model_cache: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = field(
        default_factory=dict,
    )
    total_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "models": [asdict(m) for m in self.models],
            "fused_frame_count": len(self.fused_detections),
        }


def discover_ball_models(
    models_dir: Path,
    *,
    explicit: Optional[List[str]] = None,
) -> List[Path]:
    if explicit:
        out = [Path(p) for p in explicit if Path(p).exists()]
        return out
    found: List[Path] = []
    for p in sorted(models_dir.glob("ball*.pt")):
        if p.name in _SKIP_NAMES:
            continue
        if p.stat().st_size > _MAX_MODEL_BYTES:
            continue
        found.append(p)
    # Stable preferred order
    order = {
        "ball_best.pt": 0,
        "ball_best_aws_v1.pt": 1,
        "ball_best_backup.pt": 2,
        "ball_best_legacy.pt": 3,
    }
    found.sort(key=lambda x: (order.get(x.name, 99), x.name))
    return found


def _relaxed_cache_point(
    det: Detection,
    frame_h: int,
    frame_w: int,
) -> bool:
    if det.w < 2.0 or det.h < 2.0 or det.w > 60.0 or det.h > 60.0:
        return False
    aspect = det.w / max(det.h, 1e-6)
    if aspect < 0.25 or aspect > 3.50:
        return False
    if det.cx < 10.0 or det.cy < max(12.0, frame_h * 0.12):
        return False
    bottom_frac = 0.90 if frame_h > frame_w * 1.05 else 0.84
    if det.cy > frame_h * bottom_frac:
        return False
    return True


def _probe_video_with_model(
    cap: cv2.VideoCapture,
    model: Any,
    *,
    yolo_cfg: YoloInferenceConfig,
    class_id: int,
    reference_overlay: bool,
    min_conf: float,
    progress: Optional[Callable[[float], None]] = None,
) -> Dict[int, List[Tuple[float, float, float]]]:
    cache: Dict[int, List[Tuple[float, float, float]]] = {}
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cfg = YoloInferenceConfig(
        conf=yolo_cfg.conf,
        iou=yolo_cfg.iou,
        agnostic_nms=yolo_cfg.agnostic_nms,
        max_det=max(5, int(yolo_cfg.max_det)),
        imgsz=yolo_cfg.imgsz,
        device=yolo_cfg.device,
        half=yolo_cfg.half,
        augment=yolo_cfg.augment,
        scan_conf_floor=yolo_cfg.scan_conf_floor,
        class_id=int(class_id),
    )
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        infer = (
            suppress_trajectory_overlay(frame)
            if reference_overlay
            else frame
        )
        fh, fw = frame.shape[:2]
        dets = run_ball_yolo(model, infer, fi, cfg, min_conf=min_conf)
        pts: List[Tuple[float, float, float]] = []
        for d in dets:
            if not _relaxed_cache_point(d, fh, fw):
                continue
            pts.append((float(d.cx), float(d.cy), float(d.conf)))
        if pts:
            cache[fi] = pts
        fi += 1
        if progress and fi % 10 == 0:
            progress(min(99.0, 100.0 * fi / total))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cache


def _build_associated_path(
    cache: Dict[int, List[Tuple[float, float, float]]],
    frame_start: int,
    frame_end: int,
    *,
    max_step_px: float = 96.0,
    min_conf: float = 0.02,
) -> List[Tuple[int, float, float, float]]:
    path: List[Tuple[int, float, float, float]] = []
    pred_x = pred_y = None
    vx = vy = 0.0

    for fi in range(int(frame_start), int(frame_end) + 1):
        raw = [
            (cx, cy, conf)
            for cx, cy, conf in (cache.get(fi) or [])
            if conf >= min_conf
        ]
        if not raw:
            continue

        if pred_x is None:
            cx, cy, conf = max(raw, key=lambda c: c[2])
        else:
            picked = _pick_nearest_to_prediction(
                raw, float(pred_x), float(pred_y), max_step_px,
            )
            if picked is None:
                cx, cy, conf = max(raw, key=lambda c: c[2])
                if math.hypot(cx - pred_x, cy - pred_y) > max_step_px * 1.8:
                    continue
            else:
                cx, cy, conf = picked

        if path:
            pf, px, py, _ = path[-1]
            dt = max(1, fi - pf)
            vx = 0.6 * vx + 0.4 * (cx - px) / dt
            vy = 0.6 * vy + 0.4 * (cy - py) / dt
        path.append((fi, float(cx), float(cy), float(conf)))
        pred_x = cx + vx
        pred_y = cy + vy

    return path


def _longest_span_with_gaps(
    frames: List[int],
    *,
    max_gap: int = 2,
) -> Tuple[int, int, int]:
    if not frames:
        return (0, -1, 0)
    frames = sorted(set(int(f) for f in frames))
    best_s, best_e, best_n = frames[0], frames[0], 1
    cur_s = frames[0]
    cur_n = 1
    prev = frames[0]
    for f in frames[1:]:
        if f - prev <= max_gap + 1:
            cur_n += 1
            prev = f
        else:
            if cur_n > best_n:
                best_s, best_e, best_n = cur_s, prev, cur_n
            cur_s = f
            cur_n = 1
            prev = f
    if cur_n > best_n:
        best_s, best_e, best_n = cur_s, prev, cur_n
    return best_s, best_e, best_n


def _det_from_tuple(fi: int, cx: float, cy: float, conf: float) -> Detection:
    return Detection(
        frame_idx=int(fi),
        cx=float(cx),
        cy=float(cy),
        conf=float(conf),
        w=12.0,
        h=12.0,
    )


def _frame_model_boost(info: ModelSpanInfo, frame_idx: int) -> float:
    """Higher inside the model's best span; moderate on other linked path frames."""
    if info.span_start <= frame_idx <= info.span_end:
        boost = 1.0
    elif frame_idx in info.path_frames:
        boost = 0.55
    else:
        boost = 0.25
    return boost


def _fuse_frame_detections(
    frame_idx: int,
    model_infos: List[ModelSpanInfo],
    per_model_cache: Dict[str, Dict[int, List[Tuple[float, float, float]]]],
) -> List[Detection]:
    best: Optional[Detection] = None
    best_score = -1.0
    extras: List[Tuple[float, Detection]] = []

    for info in model_infos:
        if info.model_weight <= 0.0:
            continue
        boost = _frame_model_boost(info, frame_idx)
        cands = per_model_cache.get(info.model_name, {}).get(frame_idx) or []
        for cx, cy, conf in cands:
            score = float(info.model_weight) * boost * float(conf)
            det = _det_from_tuple(frame_idx, cx, cy, conf)
            extras.append((score, det))
            if score > best_score:
                best_score = score
                best = det

    if best is None:
        return []
    extras.sort(key=lambda x: -x[0])
    out = [best]
    for score, det in extras[1:3]:
        if math.hypot(det.cx - out[0].cx, det.cy - out[0].cy) > 35.0:
            out.append(det)
    return out


def build_adaptive_fusion_plan(
    cap: cv2.VideoCapture,
    model_paths: List[Path],
    load_model: Callable[[str], Any],
    *,
    yolo_cfg: YoloInferenceConfig,
    reference_overlay: bool = False,
    min_conf: float = 0.06,
    max_step_px: float = 96.0,
    progress: Optional[Callable[[str, float], None]] = None,
) -> AdaptiveFusionPlan:
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_end = total_frames - 1
    per_model_cache: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = {}
    model_infos: List[ModelSpanInfo] = []

    for i, mp in enumerate(model_paths):
        name = mp.name
        if progress:
            progress(f"probe:{name}", 100.0 * i / max(1, len(model_paths)))
        logger.info("Adaptive fusion probe | model=%s", mp)
        model = load_model(str(mp))
        class_id = 0
        try:
            names = getattr(model, "names", None) or {}
            if isinstance(names, dict) and len(names) > 1:
                for k, v in names.items():
                    if str(v).lower() in ("ball", "item", "sports ball"):
                        class_id = int(k)
                        break
        except Exception:
            pass

        cache = _probe_video_with_model(
            cap,
            model,
            yolo_cfg=yolo_cfg,
            class_id=class_id,
            reference_overlay=reference_overlay,
            min_conf=min_conf,
        )
        per_model_cache[name] = cache
        path = _build_associated_path(
            cache, 0, frame_end, max_step_px=max_step_px, min_conf=min_conf * 0.5,
        )
        span_s, span_e, span_n = _longest_span_with_gaps([p[0] for p in path])
        if span_n <= 0 and path:
            span_s, span_e = path[0][0], path[-1][0]
            span_n = len(path)

        model_infos.append(
            ModelSpanInfo(
                model_path=str(mp),
                model_name=name,
                span_start=int(span_s),
                span_end=int(span_e),
                path_points=len(path),
                span_points=int(span_n),
                class_id=int(class_id),
                path_frames=[int(p[0]) for p in path],
            )
        )

    max_span = max((m.span_points for m in model_infos), default=0)
    max_path = max((m.path_points for m in model_infos), default=0)
    denom = max(1, max_span)
    for m in model_infos:
        span_score = m.span_points / denom
        path_score = m.path_points / max(1, max_path)
        m.model_weight = round(0.15 + 0.55 * span_score + 0.30 * path_score, 4)

    # Re-normalize weights to sum ≈ 1
    wsum = sum(m.model_weight for m in model_infos) or 1.0
    for m in model_infos:
        m.model_weight = round(m.model_weight / wsum, 4)

    model_infos.sort(key=lambda x: -x.model_weight)
    logger.info(
        "Adaptive model weights: %s",
        ", ".join(f"{m.model_name}={m.model_weight:.2f}({m.span_start}-{m.span_end},{m.span_points}pts)" for m in model_infos),
    )

    frame_owner: Dict[int, str] = {}
    available = set(range(0, frame_end + 1))
    for info in sorted(model_infos, key=lambda x: -x.span_points):
        for fi in range(info.span_start, info.span_end + 1):
            if fi in available:
                frame_owner[fi] = info.model_name
                available.discard(fi)

    fused: Dict[int, List[Detection]] = {}
    for fi in range(0, frame_end + 1):
        chosen: Optional[Detection] = None
        owner = frame_owner.get(fi)
        if owner:
            cands = per_model_cache.get(owner, {}).get(fi) or []
            if cands:
                cx, cy, conf = max(cands, key=lambda c: c[2])
                chosen = _det_from_tuple(fi, cx, cy, conf)
        if chosen is None:
            soft = _fuse_frame_detections(fi, model_infos, per_model_cache)
            if soft:
                chosen = soft[0]
        if chosen is not None:
            fused[fi] = [chosen]

    return AdaptiveFusionPlan(
        models=model_infos,
        fused_detections=fused,
        per_model_cache=per_model_cache,
        total_frames=total_frames,
    )


def save_fusion_plan(plan: AdaptiveFusionPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
