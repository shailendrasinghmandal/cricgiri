"""Production JSON response builder for client integrations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.jobs import JobRecord
from api.schemas import JobStatus


def _first_delivery(result: Dict[str, Any]) -> Dict[str, Any]:
    dels = result.get("deliveries") or []
    return dels[0] if dels else {}


def build_production_payload(job: JobRecord) -> Dict[str, Any]:
    """Structured response matching deployment spec."""
    base: Dict[str, Any] = {
        "status": job.status.value if isinstance(job.status, JobStatus) else str(job.status),
        "job_id": job.job_id,
        "video_filename": job.video_filename,
        "progress_pct": job.progress_pct,
        "processing_time_sec": None,
        "trajectory": [],
        "bounce_point": None,
        "speed_kmph": None,
        "speed": None,
        "confidence": 0.0,
        "line": None,
        "length": None,
        "swing_type": None,
        "swing_cm": None,
        "heatmap_points": [],
        "output_video": None,
        "result_json": None,
        "future_trajectory": [],
        "error": job.error,
    }

    if job.status != JobStatus.completed or not job.result:
        return base

    d = _first_delivery(job.result)
    traj = d.get("trajectory") or []
    swing = d.get("swing") or {}

    base.update({
        "status": "success",
        "processing_time_sec": job.result.get("processing_time_sec"),
        "trajectory": traj,
        "bounce_point": d.get("bounce_point"),
        "speed_kmph": d.get("speed_kmph"),
        "speed": f"{d.get('speed_kmph')} km/h" if d.get("speed_kmph") else None,
        "confidence": d.get("confidence_score", 0.0),
        "line": d.get("line"),
        "length": d.get("length"),
        "swing_type": d.get("swing_type"),
        "swing_cm": d.get("swing_cm"),
        "heatmap_points": d.get("heatmap_points") or [],
        "output_video": f"/api/v1/analysis/{job.job_id}/video",
        "result_json": f"/api/v1/analysis/{job.job_id}/result",
        "future_trajectory": _extract_future_trajectory(d),
        "total_deliveries": job.result.get("total_deliveries", 0),
        "session_id": job.result.get("session_id"),
    })
    return base


def _extract_future_trajectory(delivery: Dict[str, Any]) -> List[List[float]]:
    """Future path points from trajectory physics block if present."""
    traj_block = delivery.get("trajectory") or {}
    if isinstance(traj_block, dict):
        fut = traj_block.get("future_points") or traj_block.get("predicted_path") or []
        if fut:
            return fut
    swing = delivery.get("swing") or {}
    if isinstance(swing, dict) and swing.get("future_trajectory"):
        return swing["future_trajectory"]
    return []
