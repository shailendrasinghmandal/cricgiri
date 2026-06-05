"""Request/response models for the analytics API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional  # noqa: F401 used by ModelInfoResponse

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class AnalyzeVideoResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued
    message: str = "Video queued for analysis"


class DeliveryResult(BaseModel):
    delivery_id: str
    speed_kmph: float = 0.0
    bounce_point: Optional[Dict[str, float]] = None
    trajectory: List[List[float]] = Field(default_factory=list)
    line: Optional[str] = None
    length: Optional[str] = None
    swing_cm: float = 0.0
    swing_type: str = "none"
    heatmap_points: List[List[float]] = Field(default_factory=list)
    confidence_score: float = 0.0


class AnalysisResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_pct: float = 0.0
    video_filename: Optional[str] = None
    error: Optional[str] = None
    session_id: Optional[str] = None
    total_deliveries: int = 0
    processing_time_sec: Optional[float] = None
    deliveries: List[DeliveryResult] = Field(default_factory=list)
    heatmap_stats: Optional[Dict[str, Any]] = None
    annotated_video_url: Optional[str] = None
    result_json_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    ball_model: str
    stump_model: str


class ModelInfoResponse(BaseModel):
    model_version: str
    ball_weights: str
    stump_weights: str
    device: Optional[str] = None
    half_precision: bool = False
    inference_imgsz: int = 640
    ball_confidence: float = 0.10
    bowler_arm: str = "right"
    max_upload_mb: int = 200
    endpoints: Dict[str, str] = Field(default_factory=dict)


class AnalyzeJobResponse(BaseModel):
    status: str = "queued"
    job_id: str
    message: str = "Video queued. Poll GET /analyze/{job_id} for results."
    poll_url: str = ""
