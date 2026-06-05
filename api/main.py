"""
CricGiri Cricket Analytics API — production deployment entry point.

Primary endpoints:
  GET  /health
  GET  /model-info
  POST /analyze              → upload video, returns job_id
  GET  /analyze/{job_id}     → production JSON (poll until status=success)

Legacy / PDF aliases preserved under /api/v1/...
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from api.jobs import job_to_response, worker
from api.logging_config import setup_logging
from api.production import build_production_payload
from api.schemas import (
    AnalyzeJobResponse,
    AnalyzeVideoResponse,
    AnalysisResponse,
    HealthResponse,
    JobStatus,
    ModelInfoResponse,
)
from api.settings import ROOT, settings

setup_logging(settings.log_dir, debug=settings.debug)
logger = logging.getLogger(__name__)

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid or missing X-API-Key header")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    worker.start()
    logger.info(
        "%s v%s | model=%s | device=%s",
        settings.app_name,
        settings.app_version,
        settings.model_version,
        settings.device or "auto",
    )
    yield
    worker.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production cricket bowling analytics API — YOLO tracking, trajectory, speed, bounce.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_upload(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(400, "Filename required")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")
    return ext


async def _queue_video(
    file: UploadFile,
    *,
    bowler_arm: Optional[str] = None,
    save_video: Optional[bool] = None,
) -> AnalyzeJobResponse:
    ext = _validate_upload(file)
    content = await file.read()
    max_bytes = int(settings.max_upload_mb) * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")

    if bowler_arm in ("right", "left"):
        settings.bowler_arm = bowler_arm
    if save_video is not None:
        settings.save_annotated_video = save_video

    safe_name = Path(file.filename or "upload").name
    job_id = str(uuid.uuid4())
    upload_path = settings.upload_dir / f"{job_id}{ext}"
    upload_path.write_bytes(content)

    worker.store.create(
        job_id=job_id,
        video_filename=safe_name,
        video_path=str(upload_path),
        out_dir=settings.output_dir,
    )
    worker.enqueue(job_id)
    logger.info("Queued %s | %s | %.1f MB", job_id, safe_name, len(content) / 1e6)

    return AnalyzeJobResponse(
        status="queued",
        job_id=job_id,
        poll_url=f"/analyze/{job_id}",
    )


# ── Production endpoints ─────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Production"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    ball = ROOT / settings.ball_model_path
    stump = ROOT / settings.stump_model_path
    ok = ball.exists() and stump.exists()
    return HealthResponse(
        status="ok" if ok else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        ball_model=str(ball),
        stump_model=str(stump),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Production"])
async def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=settings.model_version,
        ball_weights=str(ROOT / settings.ball_model_path),
        stump_weights=str(ROOT / settings.stump_model_path),
        device=settings.device,
        half_precision=settings.use_half_precision,
        inference_imgsz=settings.inference_imgsz,
        ball_confidence=settings.ball_confidence,
        bowler_arm=settings.bowler_arm,
        max_upload_mb=settings.max_upload_mb,
        endpoints={
            "analyze": "POST /analyze",
            "poll": "GET /analyze/{job_id}",
            "video": "GET /api/v1/analysis/{job_id}/video",
            "docs": "GET /docs",
            "health": "GET /health",
        },
    )


@app.post("/analyze", response_model=AnalyzeJobResponse, tags=["Production"])
async def analyze(
    file: UploadFile = File(...),
    bowler_arm: Optional[str] = Form(default=None),
    save_video: Optional[bool] = Form(default=True),
    _: None = Depends(_verify_api_key),
):
    """Upload cricket video → async analysis. Poll GET /analyze/{job_id}."""
    return await _queue_video(file, bowler_arm=bowler_arm, save_video=save_video)


@app.get("/analyze/{job_id}", tags=["Production"])
async def get_analyze_result(job_id: str):
    """Production JSON: trajectory, bounce, speed, confidence, output video URL."""
    job = worker.store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return JSONResponse(build_production_payload(job))


# ── Legacy / detailed endpoints ──────────────────────────────────────────────

@app.post("/api/v1/analyze-video", response_model=AnalyzeVideoResponse, tags=["Legacy"])
@app.post("/analyze-video", response_model=AnalyzeVideoResponse, tags=["Legacy"])
async def analyze_video_legacy(
    file: UploadFile = File(...),
    bowler_arm: Optional[str] = Form(default=None),
    save_video: Optional[bool] = Form(default=None),
):
    resp = await _queue_video(file, bowler_arm=bowler_arm, save_video=save_video)
    return AnalyzeVideoResponse(job_id=resp.job_id, status=JobStatus.queued)


@app.get("/api/v1/analysis/{job_id}", response_model=AnalysisResponse, tags=["Legacy"])
@app.get("/analysis/{job_id}", response_model=AnalysisResponse, tags=["Legacy"])
async def get_analysis(job_id: str) -> AnalysisResponse:
    job = worker.store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return job_to_response(job, base_url=settings.public_base_url.rstrip("/"))


@app.get("/api/v1/analysis/{job_id}/video", tags=["Output"])
async def download_video(job_id: str):
    job = worker.store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = Path(job.output_video_path)
    if not path.exists():
        raise HTTPException(404, "Annotated video not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}_annotated.mp4")


@app.get("/api/v1/analysis/{job_id}/result", tags=["Output"])
async def download_result_json(job_id: str):
    job = worker.store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = Path(job.output_json_path)
    if not path.exists():
        raise HTTPException(404, "Result JSON not ready")
    return FileResponse(path, media_type="application/json", filename=f"{job_id}.json")


@app.delete("/api/v1/analysis/{job_id}", tags=["Admin"])
async def delete_job(job_id: str, _: None = Depends(_verify_api_key)):
    job = worker.store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    for p in (job.video_path, job.output_json_path, job.output_video_path):
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass
    return JSONResponse({"deleted": job_id})
