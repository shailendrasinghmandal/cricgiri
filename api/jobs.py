"""In-memory job store + background pipeline worker."""

from __future__ import annotations

import gc
import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

from api.schemas import AnalysisResponse, DeliveryResult, JobStatus
from api.settings import ApiSettings, settings

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus = JobStatus.queued
    progress_pct: float = 0.0
    video_filename: str = ""
    video_path: str = ""
    output_json_path: str = ""
    output_video_path: str = ""
    error: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, *, job_id: str, video_filename: str, video_path: str, out_dir: Path) -> JobRecord:
        record = JobRecord(
            job_id=job_id,
            video_filename=video_filename,
            video_path=video_path,
            output_json_path=str(out_dir / f"{job_id}.json"),
            output_video_path=str(out_dir / f"{job_id}_annotated.mp4"),
        )
        with self._lock:
            self._jobs[job_id] = record
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        progress_pct: Optional[float] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            if status is not None:
                rec.status = status
            if progress_pct is not None:
                rec.progress_pct = progress_pct
            if error is not None:
                rec.error = error
            if result is not None:
                rec.result = result
            if status in (JobStatus.completed, JobStatus.failed):
                rec.completed_at = datetime.now(timezone.utc).isoformat()


def _delivery_from_dict(d: Dict[str, Any]) -> DeliveryResult:
    return DeliveryResult(
        delivery_id=str(d.get("delivery_id", "")),
        speed_kmph=float(d.get("speed_kmph") or 0.0),
        bounce_point=d.get("bounce_point"),
        trajectory=d.get("trajectory") or [],
        line=d.get("line"),
        length=d.get("length"),
        swing_cm=float(d.get("swing_cm") or 0.0),
        swing_type=str(d.get("swing_type") or "none"),
        heatmap_points=d.get("heatmap_points") or [],
        confidence_score=float(d.get("confidence_score") or 0.0),
    )


def job_to_response(job: JobRecord, *, base_url: str = "") -> AnalysisResponse:
    deliveries: List[DeliveryResult] = []
    session_id = None
    total_deliveries = 0
    processing_time_sec = None
    heatmap_stats = None

    if job.result:
        session_id = job.result.get("session_id")
        total_deliveries = int(job.result.get("total_deliveries") or 0)
        processing_time_sec = job.result.get("processing_time_sec")
        heatmap_stats = job.result.get("heatmap_stats")
        for d in job.result.get("deliveries") or []:
            deliveries.append(_delivery_from_dict(d))

    annotated_url = None
    json_url = None
    if job.status == JobStatus.completed:
        if Path(job.output_video_path).exists():
            annotated_url = f"{base_url}/api/v1/analysis/{job.job_id}/video"
        if Path(job.output_json_path).exists():
            json_url = f"{base_url}/api/v1/analysis/{job.job_id}/result"

    return AnalysisResponse(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        video_filename=job.video_filename,
        error=job.error,
        session_id=session_id,
        total_deliveries=total_deliveries,
        processing_time_sec=processing_time_sec,
        deliveries=deliveries,
        heatmap_stats=heatmap_stats,
        annotated_video_url=annotated_url,
        result_json_url=json_url,
    )


class PipelineWorker:
    """Single-worker queue — reuses one loaded pipeline instance."""

    def __init__(self, api_settings: ApiSettings | None = None) -> None:
        self.settings = api_settings or settings
        self.store = JobStore()
        self._queue: Queue[str] = Queue()
        self._pipeline: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pipeline-worker")
        self._thread.start()
        logger.info("Pipeline worker started")

    def stop(self) -> None:
        self._stop.set()

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        from pipeline.pipeline import CricketAnalyticsPipeline, PipelineConfig

        root = Path(__file__).resolve().parent.parent
        cfg = PipelineConfig(
            video_path=str(root / "videos" / "test.mp4"),
            ball_model_path=str(root / self.settings.ball_model_path),
            stump_model_path=str(root / self.settings.stump_model_path),
            device=self.settings.device,
            ball_confidence=float(self.settings.ball_confidence),
            inference_imgsz=int(self.settings.inference_imgsz),
            half_precision=bool(self.settings.use_half_precision),
            save_video=False,
            save_json=False,
        )
        logger.info("Loading analytics pipeline (models warmup)…")
        self._pipeline = CricketAnalyticsPipeline(cfg)
        logger.info("Pipeline ready")
        return self._pipeline

    def _progress_cb(self, job_id: str) -> Callable[[str, float], None]:
        def _cb(_jid: str, pct: float) -> None:
            self.store.update(job_id, progress_pct=float(pct))
        return _cb

    def _run_job(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if not job:
            return

        self.store.update(job_id, status=JobStatus.processing, progress_pct=5.0)
        try:
            pipeline = self._ensure_pipeline()
            cfg = pipeline.cfg
            cfg.video_path = job.video_path
            cfg.output_json_path = job.output_json_path
            cfg.output_video_path = job.output_video_path
            cfg.job_id = job_id
            cfg.bowler_arm = self.settings.bowler_arm
            cfg.ball_confidence = float(self.settings.ball_confidence)
            cfg.device = self.settings.device
            cfg.save_video = bool(self.settings.save_annotated_video)
            cfg.save_json = True
            cfg.blur_recovery_detection = bool(self.settings.blur_recovery)
            cfg.hard_frame_recovery = bool(self.settings.blur_recovery)
            cfg.inference_imgsz = int(self.settings.inference_imgsz)
            cfg.half_precision = bool(self.settings.use_half_precision)
            cfg.progress_callback = self._progress_cb(job_id)

            # Clear per-video caches so consecutive API jobs stay isolated.
            if hasattr(pipeline, "_raw_ball_cache"):
                pipeline._raw_ball_cache.clear()
            if hasattr(pipeline, "_gray_frame_cache"):
                pipeline._gray_frame_cache.clear()
            if hasattr(pipeline, "_render_path_cache"):
                pipeline._render_path_cache.clear()

            session = pipeline.run()
            result = session.to_dict()
            self.store.update(
                job_id,
                status=JobStatus.completed,
                progress_pct=100.0,
                result=result,
            )
            logger.info("Job %s completed | deliveries=%d", job_id, session.total_deliveries)
            gc.collect()
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self.store.update(
                job_id,
                status=JobStatus.failed,
                error=str(exc),
                progress_pct=0.0,
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._run_job(job_id)
            except Exception:
                logger.error("Worker crash on job %s:\n%s", job_id, traceback.format_exc())
            finally:
                self._queue.task_done()


worker = PipelineWorker()
