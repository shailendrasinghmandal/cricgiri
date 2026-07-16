"""
CricGiri delivery analytics HTTP API (+ backend webhook) — PIPELINE engine.

DEPLOYED AS ``api/delivery_api.py`` (kept as pipeline_api.py in the repo so it does
not collide with the older script-engine API of the same name). It is a DROP-IN
replacement for the previously shipped delivery API: identical module path, routes,
request shapes, webhook payloads, passthrough meta, SSRF guard, /status polling and
WebSocket protocol — so the backend integration and the Dockerfile CMD
(``uvicorn api.delivery_api:app --host 0.0.0.0 --port 7860``) need no changes.

The ONLY difference is the engine underneath: analysis runs through the PIPELINE
(pipeline.pipeline.CricketAnalyticsPipeline) — the product — instead of the
single-video script engine, so `result` is the pipeline's documented delivery JSON
(see DELIVERY_API_RESPONSE_FORMAT.md: swing_sf / spin_factor / trajectory_matrices
/ confidence_pct ...) and detection uses the ball_ft_t4 + ball_best_leather_new
ensemble by default.

Endpoints
---------
  GET  /                  redirect to interactive docs (/docs)
  GET  /health            liveness + which models/device are loaded
  GET  /status/{job_id}   poll an async (webhook) job: processing | done | error
  POST /analyze           multipart/form OR application/json
      { "video_url": "...", "webhook_url": "...", "pitch_length": 20.12,
        "service": "...", "request_id": "...", "clip_id": "...", "session_id": "..." }
      * raw file upload (`video`) -> ALWAYS synchronous, returns the result JSON.
      * `video_url`               -> ALWAYS async: 202 {"status":"accepted","job_id"}
        now, result POSTed to `webhook_url` (defaults to DEFAULT_WEBHOOK_URL) later.
      The four passthrough keys are echoed verbatim in the 202, /status and webhook.
      header:  X-API-Key: <key>   (only if API_KEY env is set)
  WS   /ws/analyze        same analysis streamed with live progress

Run:
  python -m uvicorn api.delivery_api:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from fastapi import (BackgroundTasks, FastAPI, HTTPException, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cricgiri.pipeline_api")

API_KEY = os.environ.get("API_KEY") or None
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Fixed webhook callback for the CricGiri backend integration — the backend does
# not pass webhook_url per-request; this is used whenever one isn't given.
DEFAULT_WEBHOOK_URL = os.environ.get(
    "CRICGIRI_WEBHOOK_URL", "https://aistg.cricgiri.com/webhook/session-clip")

# Keys echoed straight back to the caller (202 body, /status, webhook payload) so
# the backend can correlate a callback with the request that produced it.
_PASSTHROUGH_KEYS = ("service", "request_id", "clip_id", "session_id")

# The YOLO models are shared singletons and torch inference is NOT thread-safe, so
# analysis is serialised: one video at a time (on a single GPU this is also the
# throughput cap anyway).
_INFER_LOCK = threading.Lock()

# In-memory job store for async (webhook) requests so GET /status/{job_id} works as
# a polling fallback alongside the webhook push. Single uvicorn worker -> a plain
# dict + lock is enough.
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_SECONDS = 24 * 3600

# The pipeline loads ~250 MB of weights, so it is built ONCE and reused across
# requests; per-video state is reset in _run_pipeline().
_PIPELINE: Any = None
_PIPELINE_LOCK = threading.Lock()


def _set_job(job_id: str, payload: dict) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id] = {**payload, "updated_at": time.time()}


def _get_job(job_id: str) -> Optional[dict]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _prune_old_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    with _JOBS_LOCK:
        for jid in [j for j, v in _JOBS.items() if v.get("updated_at", 0) < cutoff]:
            del _JOBS[jid]


app = FastAPI(
    title="CricGiri Pipeline Analytics API",
    version="2.0.0",
    description="Upload a cricket-delivery video (or a video_url + webhook); receive "
                "trajectory, bounce, line, length, speed, swing and spin analytics as JSON.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def _check_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key header")


def _is_safe_public_url(url: str) -> bool:
    """SSRF guard: only http(s) URLs whose host resolves to a PUBLIC IP.

    video_url / webhook_url are caller-controlled and are fetched/posted to
    server-side, so without this a caller could point either at internal services
    or the cloud metadata endpoint (169.254.169.254).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def _download_video(url: str, dest: Path, max_mb: int) -> None:
    """Stream `url` to `dest`, aborting once the size cap is exceeded."""
    max_bytes = max_mb * 1024 * 1024
    size = 0
    with requests.get(url, stream=True, timeout=(10, 120)) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"file too large (> {max_mb} MB)")
                f.write(chunk)
    if size == 0:
        raise ValueError("empty download")


def _post_webhook(url: str, payload: dict) -> None:
    """Best-effort delivery: a few retries with backoff, then give up and log."""
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code < 300:
                return
            logger.warning("webhook %s returned HTTP %s (attempt %d/3)",
                           url, resp.status_code, attempt + 1)
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("webhook POST to %s failed (attempt %d/3): %s", url, attempt + 1, exc)
        if attempt < 2:
            time.sleep(2 * (attempt + 1))
    logger.error("webhook delivery to %s permanently failed for job_id=%s",
                 url, payload.get("job_id"))


def _get_pipeline():
    """Build the pipeline once (loads ~250 MB of weights) and reuse it."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            from pipeline.pipeline import CricketAnalyticsPipeline, PipelineConfig
            logger.info("loading pipeline (ensemble weights)…")
            _PIPELINE = CricketAnalyticsPipeline(PipelineConfig(
                video_path=str(ROOT / "videos" / "_warm.mp4"),
                save_video=False, save_json=False,
            ))
            logger.info("pipeline ready | %s + %s | conf %s | imgsz %s",
                        Path(_PIPELINE.cfg.ball_model_path).name,
                        Path(_PIPELINE.cfg.ball_model_alt_path or "-").name,
                        _PIPELINE.cfg.ball_confidence, _PIPELINE.cfg.inference_imgsz)
        return _PIPELINE


def _run_pipeline(video: Path, pitch_length_m: float, *, save_video: bool = False,
                  out_video: Optional[str] = None,
                  progress_cb=None) -> dict:
    """Analyse ONE video with the shared pipeline. Caller must hold _INFER_LOCK."""
    p = _get_pipeline()
    cfg = p.cfg
    cfg.video_path = str(video)
    cfg.pitch_length_m = float(pitch_length_m)
    cfg.save_video = bool(save_video)
    cfg.save_json = False
    cfg.output_video_path = out_video or str(ROOT / "outputs" / f"{video.stem}.mp4")
    cfg.progress_callback = progress_cb
    # _report_progress() only fires when BOTH progress_callback and job_id are set
    # (pipeline.py: `if self.cfg.progress_callback and self.cfg.job_id`), so without
    # a job_id the WebSocket would silently emit no progress at all.
    cfg.job_id = video.stem
    # Clear per-video caches so consecutive requests stay isolated.
    for attr in ("_raw_ball_cache", "_gray_frame_cache", "_render_path_cache"):
        cache = getattr(p, attr, None)
        if hasattr(cache, "clear"):
            cache.clear()
    return p.run().to_dict()


def _stage_for(pct: float) -> str:
    """Coarse stage label for the WS progress messages.

    The pipeline reports one overall percentage (detection runs to ~50%, rendering
    50-95%), so the stage is derived from it — the message SHAPE stays identical to
    the previously deployed API ({"status":"processing","stage":...,"pct":...}).
    """
    if pct < 50:
        return "detecting_ball"
    if pct < 95:
        return "rendering"
    return "finalizing"


def _process_and_notify(tmp: Path, video_url: str, pitch_length: float,
                        webhook_url: str, meta: dict, job_id: str) -> None:
    """Background body: download -> analyse -> POST the outcome to webhook_url."""
    try:
        _download_video(video_url, tmp, MAX_MB)
        t0 = time.perf_counter()
        with _INFER_LOCK:
            result = _run_pipeline(tmp, pitch_length)
        result["processing_sec"] = round(time.perf_counter() - t0, 1)
        payload = {"status": "done", "job_id": job_id, "result": result, **meta}
    except Exception as exc:                                       # noqa: BLE001
        logger.exception("background analysis failed for %s (job_id=%s)", video_url, job_id)
        payload = {"status": "error", "job_id": job_id,
                   "message": f"{type(exc).__name__}: {exc}", **meta}
    finally:
        tmp.unlink(missing_ok=True)
    _set_job(job_id, payload)
    _post_webhook(webhook_url, payload)


@app.on_event("startup")
def _warm() -> None:
    """Load the weights ONCE at startup, not on the first request."""
    try:
        _get_pipeline()
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("pipeline warm-up deferred: %s", exc)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    loaded = _PIPELINE is not None
    cfg = _PIPELINE.cfg if loaded else None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "engine": "pipeline.pipeline.CricketAnalyticsPipeline",
        "ball_models": ([Path(cfg.ball_model_path).name,
                         Path(cfg.ball_model_alt_path).name] if loaded and cfg.ball_model_alt_path
                        else ([Path(cfg.ball_model_path).name] if loaded else [])),
        "ensemble": bool(cfg.hybrid_ensemble) if loaded else None,
        "device": (getattr(cfg, "device", None) or "auto") if loaded else None,
        "pitch_length_default_m": float(cfg.pitch_length_m) if loaded else 20.12,
        "webhook_default": DEFAULT_WEBHOOK_URL,
        "pipeline_version": "offline_mapping+physics_gate+reconstruction",
    }


@app.get("/status/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    """Poll an async (webhook) job: processing / done / error.

    A fallback alongside the webhook push — e.g. if the caller's receiver missed
    the callback. State is kept in memory for _JOB_TTL_SECONDS, then pruned.
    """
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job_id '{job_id}'")
    return JSONResponse(job)


@app.post("/analyze")
async def analyze(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """Analyse one delivery video and return the delivery JSON.

    Accepts a multipart file upload (synchronous) OR a `video_url` (always async:
    202 + job_id now, result POSTed to `webhook_url` when done — defaulting to
    DEFAULT_WEBHOOK_URL). Speed scales with `pitch_length` (metres) — pass the
    real pitch length when known.
    """
    x_api_key = request.headers.get("x-api-key")
    _check_key(x_api_key)

    video: Optional[UploadFile] = None
    video_url: Optional[str] = None
    webhook_url: Optional[str] = None
    pitch_length = 20.12
    meta: dict = {}

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(400, "JSON body must be an object")
        video_url = body.get("video_url") or None
        webhook_url = body.get("webhook_url") or None
        if body.get("pitch_length") is not None:
            try:
                pitch_length = float(body["pitch_length"])
            except (TypeError, ValueError):
                raise HTTPException(400, "pitch_length must be a number")
        meta = {k: body[k] for k in _PASSTHROUGH_KEYS if k in body}
    else:
        form = await request.form()
        raw_video = form.get("video")
        # request.form() returns starlette's UploadFile (a base class of fastapi's,
        # not an instance of it) — so identify it by exclusion: form values are
        # either `str` or some UploadFile.
        video = raw_video if raw_video is not None and not isinstance(raw_video, str) else None
        raw_url = form.get("video_url")
        video_url = raw_url if isinstance(raw_url, str) and raw_url else None
        raw_webhook = form.get("webhook_url")
        webhook_url = raw_webhook if isinstance(raw_webhook, str) and raw_webhook else None
        raw_pitch = form.get("pitch_length")
        if raw_pitch:
            try:
                pitch_length = float(raw_pitch)
            except (TypeError, ValueError):
                raise HTTPException(400, "pitch_length must be a number")
        meta = {k: form[k] for k in _PASSTHROUGH_KEYS
                if isinstance(form.get(k), str) and form.get(k)}

    has_file = video is not None and bool(video.filename)
    has_url = bool(video_url)
    if has_url and not webhook_url:
        webhook_url = DEFAULT_WEBHOOK_URL
    if has_file and has_url:
        raise HTTPException(400, "Provide only one of `video` or `video_url`, not both")
    if not has_file and not has_url:
        raise HTTPException(400, "Provide either `video` (file upload) or `video_url` (link)")
    if webhook_url and has_file:
        raise HTTPException(400, "webhook_url is only supported with video_url, not a file upload")
    if has_url and not _is_safe_public_url(video_url):
        raise HTTPException(400, "video_url must be a publicly reachable http(s) URL")
    if webhook_url and not _is_safe_public_url(webhook_url):
        raise HTTPException(400, "webhook_url must be a publicly reachable http(s) URL")

    ext = (Path(urlparse(video_url).path).suffix.lower() if has_url
           else Path(video.filename or "").suffix.lower())
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    up_dir = ROOT / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    tmp = up_dir / f"upload_{uuid.uuid4().hex[:10]}{ext}"

    # ── async (webhook) path ───────────────────────────────────────────────
    if webhook_url:
        job_id = uuid.uuid4().hex
        _prune_old_jobs()
        _set_job(job_id, {"status": "processing", "job_id": job_id, **meta})
        logger.info("job accepted: job_id=%s video_url=%s pitch_length=%s meta=%s",
                    job_id, video_url, pitch_length, meta)
        background_tasks.add_task(_process_and_notify, tmp, video_url,
                                  pitch_length, webhook_url, meta, job_id)
        return JSONResponse({"status": "accepted", "job_id": job_id, **meta},
                            status_code=202, background=background_tasks)

    # ── synchronous (file upload) path ─────────────────────────────────────
    try:
        data = await video.read()
        if not data:
            raise HTTPException(400, "Empty upload")
        if len(data) > MAX_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large (> {MAX_MB} MB)")
        tmp.write_bytes(data)

        t0 = time.perf_counter()

        def _run() -> dict:
            with _INFER_LOCK:
                return _run_pipeline(tmp, float(pitch_length))

        try:
            result = await run_in_threadpool(_run)
        except Exception as exc:                                   # noqa: BLE001
            logger.exception("analysis failed for %s", video.filename)
            raise HTTPException(500, f"analysis failed: {type(exc).__name__}: {exc}")
    finally:
        tmp.unlink(missing_ok=True)

    result["processing_sec"] = round(time.perf_counter() - t0, 1)
    result.update(meta)
    return JSONResponse(result)


@app.websocket("/ws/analyze")
async def analyze_ws(websocket: WebSocket) -> None:
    """Same analysis as POST /analyze, streamed over a WebSocket with live progress.

    Protocol (unchanged from the previously deployed API — existing clients work):
      1. Connect (optionally with header ``X-API-Key: <key>``).
      2. Send ONE text frame: JSON metadata
         ``{"filename": "clip.mp4", "pitch_length": 20.12, "api_key": "<key>"}``
         (``api_key`` in the JSON is a fallback for WS clients that cannot set
         custom headers on the handshake; either the header or this field must
         match if an API key is configured.)
      3. Send ONE binary frame: the complete video file bytes.
      4. Receive JSON text frames until one has ``"status": "done"`` or
         ``"status": "error"``:
           {"status": "processing", "stage": "detecting_ball", "pct": 42.0}
           {"status": "done", "result": {...same schema as POST /analyze...}}
           {"status": "error", "message": "..."}
      The server closes the connection after the final message.
    """
    await websocket.accept()
    tmp: Optional[Path] = None
    try:
        try:
            meta = json.loads(await websocket.receive_text())
        except Exception:                                          # noqa: BLE001
            await websocket.send_json({
                "status": "error",
                "message": "expected a JSON text message first: "
                           '{"filename": "clip.mp4", "pitch_length": 20.12, "api_key": "..."}',
            })
            return

        supplied_key = meta.get("api_key") or websocket.headers.get("x-api-key")
        if API_KEY and supplied_key != API_KEY:
            await websocket.send_json({"status": "error", "message": "invalid or missing API key"})
            return

        filename = str(meta.get("filename") or "upload.mp4")
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            await websocket.send_json({
                "status": "error",
                "message": f"unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}",
            })
            return
        try:
            pitch_length = float(meta.get("pitch_length", 20.12))
        except (TypeError, ValueError):
            await websocket.send_json({"status": "error", "message": "pitch_length must be a number"})
            return

        try:
            data = await websocket.receive_bytes()
        except Exception:                                          # noqa: BLE001
            await websocket.send_json({
                "status": "error",
                "message": "expected one binary frame with the video bytes after the metadata",
            })
            return
        if not data:
            await websocket.send_json({"status": "error", "message": "empty video payload"})
            return
        if len(data) > MAX_MB * 1024 * 1024:
            await websocket.send_json({"status": "error", "message": f"file too large (> {MAX_MB} MB)"})
            return

        up_dir = ROOT / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        tmp = up_dir / f"upload_{uuid.uuid4().hex[:10]}{ext}"
        tmp.write_bytes(data)

        # Progress flows from the worker thread (the pipeline runs synchronously,
        # off the event loop) via a thread-safe queue.Queue; this coroutine drains
        # it and forwards each item as a WS message while awaiting the result.
        progress_q: "queue.Queue[tuple]" = queue.Queue()

        def _progress_cb(_job_id: str, pct: float) -> None:
            progress_q.put((_stage_for(float(pct)), float(pct)))

        def _run() -> dict:
            with _INFER_LOCK:
                return _run_pipeline(tmp, pitch_length, progress_cb=_progress_cb)

        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _run)
        try:
            while not future.done():
                try:
                    stage, pct = progress_q.get(timeout=0.2)
                    await websocket.send_json({"status": "processing", "stage": stage,
                                               "pct": round(pct, 1)})
                except queue.Empty:
                    await asyncio.sleep(0.05)
            # Drain any progress queued right before completion.
            while not progress_q.empty():
                stage, pct = progress_q.get_nowait()
                await websocket.send_json({"status": "processing", "stage": stage,
                                           "pct": round(pct, 1)})
            result = future.result()
            result["processing_sec"] = round(time.perf_counter() - t0, 1)
            await websocket.send_json({"status": "done", "result": result})
        except Exception as exc:                                   # noqa: BLE001
            logger.exception("WS analysis failed for %s", filename)
            await websocket.send_json({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
    except WebSocketDisconnect:
        logger.info("WS client disconnected before analysis completed")
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        try:
            await websocket.close()
        except Exception:                                          # noqa: BLE001
            pass
