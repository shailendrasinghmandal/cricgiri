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

import asyncio
import importlib.util
import logging
import threading
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (FastAPI, File, Form, Header, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

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
OUTPUT_DIR = ROOT / "outputs" / "delivery_api"
UI_OUTPUT_DIR = ROOT / "outputs" / "delivery_api_ui"

# The ball/stump YOLO models are shared singletons and torch inference is NOT thread-safe;
# run_in_threadpool would otherwise run requests concurrently on the same model. Serialise
# analysis so one video is processed at a time (single GPU -> this is also the throughput cap).
_INFER_LOCK = threading.Lock()

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
    return RedirectResponse("/ui")


def _engine_status() -> dict:
    return {
        "models": engine._ENGINE.get("model_paths") or [],
        "device": engine._ENGINE.get("device"),
        "conf": engine.CONF,
        "imgsz": engine.IMGSZ,
        "pitch_length_m": engine.PITCH_LEN,
        "engine": "api.delivery_api -> scripts.run_demo_testing.analyze_video",
    }


def _ui_page(message: str = "", download_name: Optional[str] = None, preview: str = "") -> HTMLResponse:
    status = _engine_status()
    models = ", ".join(status["models"]) if status["models"] else "loading on first request"
    msg = f"<p class='ok'>{message}</p>" if message else ""
    link = (
        f"<p>"
        f"<a class='btn' href='/ui/download/{download_name}' download>Download JSON</a> "
        f"<a class='btn secondary' href='/ui/view/{download_name}' target='_blank'>Open JSON</a>"
        f"</p>"
        f"<p class='path'>Saved at: {UI_OUTPUT_DIR / download_name}</p>"
        if download_name
        else ""
    )
    pre = f"<pre>{preview[:12000]}</pre>" if preview else ""
    return HTMLResponse(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CricGiri Local 8000 API</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#101820; color:#f7f9fb; margin:32px; }}
    main {{ max-width: 900px; margin:auto; }}
    .box {{ background:#172330; border:1px solid #2d4055; border-radius:8px; padding:18px; margin:14px 0; }}
    label {{ display:block; font-weight:700; margin:14px 0 6px; }}
    input {{ font-size:16px; }}
    input[type=number] {{ padding:8px; width:120px; }}
    button,.btn {{ background:#ffd34d; color:#111; border:0; padding:11px 16px; border-radius:6px; font-weight:800; text-decoration:none; cursor:pointer; }}
    .secondary {{ background:#9fd3ff; }}
    .ok {{ color:#9dffb4; font-weight:800; }}
    .path {{ color:#c7d3df; font-size:14px; }}
    pre {{ background:#06101d; padding:14px; border-radius:8px; overflow:auto; max-height:420px; white-space:pre-wrap; }}
  </style>
</head>
<body>
<main>
  <h1>CricGiri Local 8000 Upload</h1>
  <div class="box">
    <b>Same later 8000 engine:</b> {status["engine"]}<br>
    <b>Models:</b> {models}<br>
    <b>Config:</b> conf={status["conf"]}, imgsz={status["imgsz"]}, pitch_length_m={status["pitch_length_m"]}
  </div>
  <div class="box">
    <form method="post" action="/ui/upload" enctype="multipart/form-data">
      <label>Video</label>
      <input type="file" name="video" accept=".mp4,.mov,.avi,.mkv,.webm" required>
      <label>Pitch length in metres &nbsp;<span style="font-weight:400;color:#9fd3ff">(changes the speed — default full pitch 20.12 m)</span></label>
      <input type="number" name="pitch_length" value="20.12" step="0.01" min="1" max="30">
      <p style="color:#c7d3df;font-size:13px;margin:4px 0 0">Speed scales with this: a shorter pitch → lower km/h; longer → higher. Bounce &amp; length also shift.</p>
      <p><button type="submit">Run 8000 Engine</button></p>
    </form>
  </div>
  {msg}
  {link}
  {pre}
</main>
</body>
</html>""")


@app.get("/ui", response_class=HTMLResponse)
def upload_ui() -> HTMLResponse:
    return _ui_page()


@app.post("/ui/upload", response_class=HTMLResponse)
async def upload_ui_run(
    video: UploadFile = File(..., description="Cricket delivery video"),
    pitch_length: float = Form(20.12, description="Real pitch length in metres"),
) -> HTMLResponse:
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
    UI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(video.filename or "upload").stem
    result_id = f"{safe_stem}_{uuid.uuid4().hex[:8]}"
    tmp = up_dir / f"ui_{result_id}{ext}"
    out_json = UI_OUTPUT_DIR / f"{result_id}.json"
    tmp.write_bytes(data)
    t0 = time.perf_counter()

    def _run() -> dict:
        with _INFER_LOCK:
            result = engine.analyze_video(
                str(tmp),
                pitch_length=float(pitch_length),
                work_id=f"ui_{result_id}",
                cleanup=True,
            )
            result["result_id"] = result_id
            result["uploaded_filename"] = video.filename
            return result

    try:
        result = await run_in_threadpool(_run)
    except Exception as exc:                                   # noqa: BLE001
        logger.exception("UI analysis failed for %s", video.filename)
        raise HTTPException(500, f"analysis failed: {type(exc).__name__}: {exc}")
    finally:
        tmp.unlink(missing_ok=True)

    result["processing_sec"] = round(time.perf_counter() - t0, 1)
    import json as _json
    pretty = _json.dumps(result, indent=2, ensure_ascii=False, default=str)
    out_json.write_text(pretty, encoding="utf-8")
    return _ui_page(
        message=f"Done. Saved JSON: {out_json.name}",
        download_name=out_json.name,
        preview=pretty,
    )


@app.get("/ui/download/{filename}")
def upload_ui_download(filename: str) -> FileResponse:
    safe = Path(filename).name
    path = UI_OUTPUT_DIR / safe
    if not path.exists() or path.suffix.lower() != ".json":
        raise HTTPException(404, "JSON result not found")
    return FileResponse(path, media_type="application/json", filename=safe)


@app.get("/ui/view/{filename}")
def upload_ui_view(filename: str) -> Response:
    safe = Path(filename).name
    path = UI_OUTPUT_DIR / safe
    if not path.exists() or path.suffix.lower() != ".json":
        raise HTTPException(404, "JSON result not found")
    return Response(path.read_text(encoding="utf-8"), media_type="application/json")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": engine._ENGINE["models"] is not None,
        "ball_models": engine._ENGINE.get("model_paths") or ["models/ball_ft_t4.pt"],
        "device": engine._ENGINE["device"],
        "pitch_length_default_m": engine.PITCH_LEN,
        "pipeline_version": "offline_mapping+physics_gate+reconstruction",
    }


@app.get("/analysis/{result_id}/video")
def get_output_video(result_id: str) -> FileResponse:
    """Return the trajectory-rendered output video for a previous /analyze call."""
    safe_id = "".join(ch for ch in result_id if ch.isalnum() or ch in ("-", "_"))
    if safe_id != result_id:
        raise HTTPException(400, "Invalid result id")
    path = OUTPUT_DIR / safe_id / "trajectory_web.mp4"
    if not path.exists():
        path = OUTPUT_DIR / safe_id / "trajectory.mp4"
    if not path.exists():
        raise HTTPException(404, "Output video not found or not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"{result_id}_trajectory.mp4")


def _make_output_video(stem: str, result_id: str) -> Optional[str]:
    """Render the detected trajectory video and return its local API URL."""
    out_dir = OUTPUT_DIR / result_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "trajectory.mp4"

    # Prefer the corrected renderer used for the cleaner/full trajectory demos:
    # it remaps the full video and renders a segmented physics arc instead of
    # only drawing a spline through the sparse JSON track points.
    try:
        from pipeline_corrected import run_corrected

        corrected = run_corrected(
            ROOT / "videos" / f"{stem}.mp4",
            out=raw_path,
            extend=0.15,
            cleanup_staged=False,
            capture=True,
        )
        if corrected.get("ok"):
            web = corrected.get("web_mp4")
            if web:
                web_path = Path(web)
                target = out_dir / "trajectory_web.mp4"
                if web_path.exists() and web_path != target:
                    shutil.copy2(web_path, target)
            return f"/analysis/{result_id}/video"
        logger.warning("corrected video render failed: %s", corrected.get("error"))
    except Exception:  # noqa: BLE001
        logger.warning("corrected video render skipped", exc_info=True)

    # Fallback: render the same clean mapped points used by the JSON result.
    pts = engine.dr.read_points(stem)
    render_pts = engine.clean_for_render(pts)
    ok, _ = engine.render_clean(stem, render_pts, raw_path)
    if not ok or not raw_path.exists():
        return None

    try:
        from api.video_utils import transcode_to_web

        web_path = transcode_to_web(raw_path)
        if web_path and web_path.exists():
            target = out_dir / "trajectory_web.mp4"
            if web_path != target:
                shutil.copy2(web_path, target)
    except Exception:  # noqa: BLE001
        logger.warning("video transcode skipped", exc_info=True)

    return f"/analysis/{result_id}/video"


def _cleanup_engine_temp(stem: str) -> None:
    (ROOT / "videos" / f"{stem}.mp4").unlink(missing_ok=True)
    shutil.rmtree(ROOT / "outputs" / "mapped" / stem, ignore_errors=True)
    shutil.rmtree(ROOT / "outputs" / "detections" / stem, ignore_errors=True)


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(..., description="Cricket delivery video (mp4/mov/avi/mkv/webm)"),
    pitch_length: float = Form(20.12, description="Real stump-to-stump pitch length in metres"),
    pitch_length_yards: Optional[float] = Form(
        default=None,
        description="Optional pitch length in yards. If provided, overrides pitch_length metres.",
    ),
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
    result_id = uuid.uuid4().hex[:12]
    work_id = f"api_{result_id}"
    pitch_length_m = float(pitch_length_yards) * 0.9144 if pitch_length_yards else float(pitch_length)

    t0 = time.perf_counter()

    def _run() -> dict:
        # Heavy + GPU-bound. run_in_threadpool keeps the event loop free; _INFER_LOCK
        # serialises the actual inference (shared, non-thread-safe models); the unique
        # per-call work-id keeps each request's temp files isolated.
        with _INFER_LOCK:
            result = engine.analyze_video(
                str(tmp),
                pitch_length=pitch_length_m,
                work_id=work_id,
                cleanup=False,
            )
            video_url = None
            if int(result.get("total_deliveries") or 0) > 0:
                video_url = _make_output_video(work_id, result_id)
            result["result_id"] = result_id
            result["output_video"] = video_url
            result["output_video_url"] = video_url
            return result

    try:
        result = await run_in_threadpool(_run)
    except Exception as exc:                                   # noqa: BLE001
        logger.exception("analysis failed for %s", video.filename)
        raise HTTPException(500, f"analysis failed: {type(exc).__name__}: {exc}")
    finally:
        tmp.unlink(missing_ok=True)
        _cleanup_engine_temp(work_id)

    result["processing_sec"] = round(time.perf_counter() - t0, 1)
    # Pretty-print (indent + stable key order) so the response is human-readable when
    # the team inspects it directly; harmless to programmatic consumers.
    import json as _json
    return Response(
        content=_json.dumps(result, indent=2, ensure_ascii=False, default=str),
        media_type="application/json",
    )


@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket) -> None:
    """Live WebSocket analysis: connect, send the video, receive progress + result.

    Protocol
    --------
      1. Client connects to   ws(s)://<host>/ws/analyze?pitch_length=20.12
      2. Client sends the whole video file as ONE binary message.
      3. Server streams JSON text messages:
             {"type":"status","stage":"received","progress":5}
             {"type":"status","stage":"analyzing","progress":10..90}   (heartbeat)
             {"type":"result","progress":100,"result":{...delivery JSON...}}
         then closes. On failure: {"type":"error","message":"..."}.

    `pitch_length` may be given as a query param, or as an initial JSON *text*
    message {"pitch_length": 20.12} sent before the binary video.
    """
    await websocket.accept()
    tmp: Optional[Path] = None
    try:
        # Optional auth (only if API_KEY is configured): ?api_key=...
        if API_KEY and websocket.query_params.get("api_key") != API_KEY:
            await websocket.send_json({"type": "error", "message": "invalid or missing api_key"})
            await websocket.close(code=1008)
            return

        pitch_length = float(websocket.query_params.get("pitch_length", engine.PITCH_LEN))

        # First frame may be JSON metadata (text) or the video itself (binary).
        first = await websocket.receive()
        video_bytes: Optional[bytes] = first.get("bytes")
        if video_bytes is None and first.get("text"):
            try:
                meta = __import__("json").loads(first["text"])
                pitch_length = float(meta.get("pitch_length", pitch_length))
            except Exception:                                          # noqa: BLE001
                pass
            nxt = await websocket.receive()                            # then the binary video
            video_bytes = nxt.get("bytes")

        if not video_bytes:
            await websocket.send_json({"type": "error", "message": "no video bytes received"})
            await websocket.close(code=1003)
            return

        await websocket.send_json({
            "type": "status", "stage": "received", "progress": 5,
            "message": f"received {len(video_bytes)} bytes",
        })

        up_dir = ROOT / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        tmp = up_dir / f"ws_{uuid.uuid4().hex[:10]}.mp4"
        tmp.write_bytes(video_bytes)

        # Run the (blocking, GPU-bound) analysis in a worker thread; _INFER_LOCK
        # serialises the shared, non-thread-safe models. Heartbeat progress is
        # coarse (the engine runs as one call) but keeps the client's bar moving.
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()

        def _run() -> dict:
            with _INFER_LOCK:
                return engine.analyze_video(str(tmp), pitch_length=float(pitch_length))

        task = loop.run_in_executor(None, _run)

        pct = 10
        await websocket.send_json({"type": "status", "stage": "analyzing", "progress": pct})
        while not task.done():
            await asyncio.sleep(1.5)
            if pct < 90:
                pct += 5
            await websocket.send_json({"type": "status", "stage": "analyzing", "progress": pct})

        result = await task
        result["processing_sec"] = round(time.perf_counter() - t0, 1)
        await websocket.send_json({"type": "result", "progress": 100, "result": result})
        await websocket.close()

    except WebSocketDisconnect:
        logger.info("ws client disconnected")
    except Exception as exc:                                           # noqa: BLE001
        logger.exception("ws analysis failed")
        try:
            await websocket.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            await websocket.close(code=1011)
        except Exception:                                              # noqa: BLE001
            pass
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
