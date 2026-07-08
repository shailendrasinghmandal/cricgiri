"""
api/delivery_api.py — CricGiri single-video delivery analytics HTTP API.
=======================================================================
A thin, self-contained HTTP wrapper around the PROVEN offline analysis engine
(scripts/run_demo_testing.analyze_video): low-conf ensemble detection -> offline
motion-consistency mapping -> physics-validity gate -> static-cluster guard ->
homography analytics -> release->bounce time-of-flight speed. Returns the exact
delivery schema (same as testing_result_new/clipNN.json).

Endpoints
---------
  GET  /health          liveness + which device/model is loaded
  GET  /                 redirect to interactive docs (/docs)
  POST /analyze          multipart upload -> schema-compliant delivery JSON
      form fields:  video=<file> (required), pitch_length=<float> (default 20.12)
      header:       X-API-Key: <key>   (only if API_KEY env is set)

Run (GPU venv):
  ./venv/Scripts/python.exe -m uvicorn api.delivery_api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import importlib.util
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

ROOT = Path(__file__).resolve().parent.parent

# Load the analysis engine (scripts/run_demo_testing.py) as an importable module.
_spec = importlib.util.spec_from_file_location(
    "cricgiri_engine", ROOT / "scripts" / "run_demo_testing.py")
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

# Optional config from the existing settings module (API key / size cap); fall back
# to sensible defaults if it is unavailable so this app runs stand-alone.
try:
    from api.settings import settings
    API_KEY = settings.api_key
    MAX_MB = settings.max_upload_mb
except Exception:
    API_KEY, MAX_MB = None, 200

logger = logging.getLogger("cricgiri.delivery_api")
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(
    title="CricGiri Delivery Analytics API",
    version="1.0.0",
    description="Upload a single cricket-delivery video; receive trajectory, bounce, "
                "line, length, speed and swing analytics as JSON.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def _check_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key header")


@app.on_event("startup")
def _warm() -> None:
    """Load the ball + stump models ONCE at startup (not on the first request)."""
    try:
        engine.load_engine()
        logger.info("CricGiri engine warmed (device=%s)", engine._ENGINE["device"])
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("engine warm-up deferred: %s", exc)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": engine._ENGINE["models"] is not None,
        "device": engine._ENGINE["device"],
        "pitch_length_default_m": engine.PITCH_LEN,
        "pipeline_version": "offline_mapping+physics_gate+reconstruction",
    }


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(..., description="Cricket delivery video (mp4/mov/avi/mkv/webm)"),
    pitch_length: float = Form(20.12, description="Real stump-to-stump pitch length in metres"),
    x_api_key: Optional[str] = Header(default=None),
) -> JSONResponse:
    """Analyse one delivery video and return the schema-compliant JSON.

    Speed accuracy scales with `pitch_length` — pass the real pitch length when known.
    """
    _check_key(x_api_key)
    ext = Path(video.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")
    data = await video.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (> {MAX_MB} MB)")

    up_dir = ROOT / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    tmp = up_dir / f"upload_{uuid.uuid4().hex[:10]}{ext}"
    tmp.write_bytes(data)

    t0 = time.perf_counter()

    def _run() -> dict:
        # engine.analyze_video is heavy + not thread-safe on the GPU; the engine's
        # own singletons + a unique work-id per call keep files from colliding, and
        # run_in_threadpool keeps the event loop free. Single GPU -> effectively serial.
        return engine.analyze_video(str(tmp), pitch_length=float(pitch_length))

    try:
        result = await run_in_threadpool(_run)
    except Exception as exc:                                   # noqa: BLE001
        logger.exception("analysis failed for %s", video.filename)
        raise HTTPException(500, f"analysis failed: {type(exc).__name__}: {exc}")
    finally:
        tmp.unlink(missing_ok=True)

    result["processing_sec"] = round(time.perf_counter() - t0, 1)
    return JSONResponse(result)
