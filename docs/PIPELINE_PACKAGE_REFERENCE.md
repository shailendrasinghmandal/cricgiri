# INTERNAL ENGINEERING REFERENCE

**CricGiri Pipeline Package** — `cricgiri_pipeline_package.zip` (213 MB)
Engine: `pipeline/pipeline.py` · API: `api/delivery_api.py` · 3 model weights

---

## 1. TL;DR — the things that matter most

1. **This package is a DROP-IN replacement for the previously delivered zip.** Same module path (`uvicorn api.delivery_api:app`), same port (**7860**), same routes, same webhook payloads, same passthrough meta, same WebSocket protocol. **The backend integration needs no changes.** The only difference is the engine underneath.
2. **The engine is now `pipeline/pipeline.py` (the product), not the single-video script engine.** So `result` carries the full documented delivery JSON — `swing_sf`, `spin_factor` / `spin_degree`, `trajectory_matrices`, `confidence_pct` — see §5.3.
3. **All three model weights are mandatory** (§6). Removing `ball_best_leather_new.pt` silently destroys the release phase; removing `stump_best.pt` nulls speed/line/length/world coordinates.
4. **`scripts/` is REQUIRED, not optional.** `pipeline.py` loads `scripts/physics_gate_v2.py` *by path at runtime*, and that file imports `scripts/delivery_reconstruction.py` on import. If either is absent the physics gate **degrades silently** — no error, the physics fields just become meaningless defaults. Verify with the check in §4.
5. **`--workers 1` is required.** Models are shared singletons (inference serialised by a lock) and the `/status` job store is an in-memory, process-local dict.
6. **The numbers are approximate.** Measured against hand-labelled ground truth: track recall ≈ 0.32, `speed_kmph` rests on a release frame the tracker picks up late, and 13 post-bounce ball positions are missed. Read §7 before putting numbers in front of a client.
7. **This package ships no `venv/` and no `.git/`** — that is deliberate. It is 213 MB instead of ~900 MB and runs on Windows, macOS and Linux; dependencies are installed on the target host for that platform.

---

## 2. Package map

| Path | Files | Purpose |
|---|---|---|
| `pipeline/` | 5 | **The engine.** `pipeline.py` (~5k lines): tracking → analytics → response |
| `analytics/` | 22 | bounce detection, pitch calibration, speed, swing, the video overlay |
| `tracking/` | 16 | ball tracker (Kalman/SORT), YOLO inference, physics constraints |
| `scripts/` | 99 | **`physics_gate_v2.py` + `delivery_reconstruction.py` are REQUIRED** (§1.4). The rest is research tooling carried along for safety |
| `api/` | 2 | `delivery_api.py` (the HTTP API + webhook) and `__init__.py` |
| `config/` | 1 | `tracking_defaults.yaml` |
| `models/` | 3 | the weights (§6) |
| `run_pipeline.py` | 1 | CLI entry point — one video → trajectory video + JSON |
| `Dockerfile` | 1 | same CMD/port as the deployed image |
| `outputs/ videos/ logs/` | — | runtime scratch; results land in `outputs/` |

No `venv/`, no `.git/`, no platform binaries (`.dll` / `.so` / `.dylib`).

---

## 3. API layer — `api/delivery_api.py`

This is what the Dockerfile's CMD runs:
`uvicorn api.delivery_api:app --host 0.0.0.0 --port 7860 --workers 1`

The pipeline is built **once** at startup (`@app.on_event("startup")`) — it loads ~250 MB of weights — and reused for every request. Per-video state is reset on each call.

**Concurrency:** a single `threading.Lock()` (`_INFER_LOCK`) serialises all inference — the YOLO models are shared singletons, torch inference isn't thread-safe, and there is one GPU, so this is also the real throughput cap (one video at a time). Requests run via `run_in_threadpool` / `BackgroundTasks` so the event loop stays responsive.

### 3.1 Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | redirect to `/docs` |
| GET | `/health` | liveness + which models/device are loaded |
| GET | `/status/{job_id}` | poll an async (webhook) job |
| POST | `/analyze` | file upload (**sync**) or `video_url` (**async + webhook**) |
| WS | `/ws/analyze` | same analysis, streamed with live progress |

### 3.2 Webhook implementation (the part most relevant to backend integration)

**Trigger:** `POST /analyze` accepts either a raw file upload (**always synchronous** — returns the result JSON directly) or a `video_url` (**always asynchronous** — returns `202 Accepted` immediately and POSTs the result to a webhook when done).

```
DEFAULT_WEBHOOK_URL = "https://aistg.cricgiri.com/webhook/session-clip"
```
Overridable via the `CRICGIRI_WEBHOOK_URL` env var.

**Request shape** (multipart form *or* `application/json`):
```json
{ "video_url": "https://.../clip.mp4", "webhook_url": "https://your-backend/callback",
  "pitch_length": 20.12,
  "service": "...", "request_id": "...", "clip_id": "...", "session_id": "..." }
```
`webhook_url` is **optional** — omitted, `DEFAULT_WEBHOOK_URL` is used. The four passthrough keys (`service`, `request_id`, `clip_id`, `session_id`) are pure metadata: whatever the caller sends is echoed back verbatim in the `202`, in `/status`, and in the final webhook payload. The server never interprets them.

**Security guard on both `video_url` and `webhook_url`** — `_is_safe_public_url()` is an SSRF guard: it resolves the hostname and rejects private / loopback / link-local / reserved / multicast / unspecified IPs (blocks internal-network and cloud-metadata targets). Both URLs are checked **before** any download or POST happens.

**Async flow** (`_process_and_notify`):
```python
try:
    _download_video(video_url, tmp, MAX_MB)
    with _INFER_LOCK:
        result = _run_pipeline(tmp, pitch_length)
    result["processing_sec"] = round(time.perf_counter() - t0, 1)
    payload = {"status": "done", "job_id": job_id, "result": result, **meta}
except Exception as exc:
    payload = {"status": "error", "job_id": job_id,
               "message": f"{type(exc).__name__}: {exc}", **meta}
finally:
    tmp.unlink(missing_ok=True)
_set_job(job_id, payload)
_post_webhook(webhook_url, payload)
```

**Delivery guarantees:** `_post_webhook()` is best-effort with retry — up to **3 attempts, 2 s / 4 s backoff**, treating any non-2xx or exception as failure. After 3 failures it logs an error and gives up. There is **no dead-letter queue** and no retry beyond process lifetime.

**Polling fallback:** every async job is also tracked in an in-memory dict (`_JOBS`), so `GET /status/{job_id}` works even if the receiver missed the callback. Jobs are pruned after `_JOB_TTL_SECONDS = 24 * 3600` (**24 h**). This store is process-local and does not survive a restart — it relies on `--workers 1`.

**Webhook payload shapes:**
```json
// success
{"status": "done",  "job_id": "...", "result": { ...full delivery JSON, see 5.3... }, "...meta..."}
// failure
{"status": "error", "job_id": "...", "message": "ValueError: ...", "...meta..."}
```

### 3.3 `WS /ws/analyze` — WebSocket variant

Same analysis, streamed — for a frontend progress bar.

1. Client connects (optionally with header `X-API-Key: <key>`).
2. Client sends **ONE text frame**: `{"filename": "clip.mp4", "pitch_length": 20.12, "api_key": "..."}` (`api_key` in the JSON is a fallback for clients that cannot set handshake headers).
3. Client sends **ONE binary frame**: the raw video bytes.
4. Server streams `{"status": "processing", "stage": "...", "pct": ...}` messages, then a final `{"status": "done", "result": {...}}` or `{"status": "error", "message": "..."}`, then closes.

Progress comes from the pipeline's own `progress_callback`; a thread-safe `queue.Queue` bridges the worker thread (running off the event loop) to the async send loop. `stage` is derived from the pipeline's single overall percentage (`detecting_ball` < 50, `rendering` < 95, then `finalizing`).

> **Gotcha:** the pipeline only emits progress when **both** `progress_callback` **and** `job_id` are set (`if self.cfg.progress_callback and self.cfg.job_id`). The API sets `cfg.job_id` for this reason — without it the socket connects fine and streams **zero** progress, silently.

### 3.4 Config — environment variables

| Var | Default | Purpose |
|---|---|---|
| `CRICGIRI_WEBHOOK_URL` | `https://aistg.cricgiri.com/webhook/session-clip` | default callback |
| `API_KEY` | unset | if set, requires `X-API-Key` (or `api_key` in the WS meta frame) |
| `MAX_UPLOAD_MB` | `200` | upload / download size cap |

---

## 4. Deployment

```bash
unzip cricgiri_pipeline_package.zip -d cricgiri && cd cricgiri
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate

# PyTorch FIRST — pick the build matching the host:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # NVIDIA

pip install -r requirements.txt
python -m uvicorn api.delivery_api:app --host 0.0.0.0 --port 7860 --workers 1
```
Docker (same CMD/port as the deployed image):
```bash
docker build -t cricgiri . && docker run -p 7860:7860 cricgiri
```
Linux hosts also need: `apt-get install -y libgl1 libglib2.0-0 ffmpeg`

**Verify before the first real clip** — this is the silent-failure check:
```bash
python -c "from pipeline.pipeline import _PHYSICS_FILTER; print('physics gate loaded:', _PHYSICS_FILTER is not None)"
# must print True. False => scripts/ is missing or incomplete (see 1.4)
```
CLI (no server): `python run_pipeline.py clip.mp4 --pitch-length-yards 22`

---

## 5. The engine — `pipeline/pipeline.py`

### 5.1 Bootstrap

`CricketAnalyticsPipeline(PipelineConfig())` loads the **ensemble** — `ball_ft_t4.pt` (primary) + `ball_best_leather_new.pt` (alt) — plus `stump_best.pt`. Ensemble detection is enabled at runtime (`hybrid_ensemble=True` → `_ball_detector.ensemble_enabled`). Construct `PipelineConfig()` plainly and you get this; **overriding `ball_model_path` / `ball_model_alt_path` / `inference_imgsz` turns off the tuned configuration.**

### 5.2 `run()` — step by step

| Step | What it does |
|---|---|
| 1. Detection | Both ball models run per frame at conf **0.05**, imgsz **1280**. Low conf = high recall on a small, fast ball; false positives are removed later |
| 2. Tracking | Kalman/SORT tracker builds the delivery track (`min_hits_to_confirm=3`) |
| 3. Release guard | Drops a small leading group (≤3 pts) cut off by a long gap (≥6 frames) — a delivery cannot teleport |
| 4. Physics gate | `scripts/physics_gate_v2.physics_filter` removes x-reversals and gap-jumps → `physics_verdict`, `physics_removed_points`, `physically_valid` |
| 5. Calibration | `stump_best.pt` finds the stumps → pitch homography → world/metric values |
| 6. Analytics | bounce, line, length, speed (release→bounce time-of-flight), swing/spin |
| 7. Response | `SessionAnalysis.to_dict()` assembles the documented JSON (§5.3) |
| 8. Render | (optional) trajectory video: ball arc, pitch corridor, Swing/Drift/Speed panel |

### 5.3 Output JSON schema

Top level: `source_video`, `fps`, `total_frames`, `total_deliveries`, `pipeline_version`, `detection_conf_threshold` (+ note), `pitch_length_yards`, `deliveries[]`, `processing_sec`.

Per delivery (36 fields) — see `DELIVERY_API_RESPONSE_FORMAT.md` for the full contract:

| Field | Meaning |
|---|---|
| `track` | `num_points`, `average_confidence` (mean YOLO conf of **detected** points; interpolated excluded), `physics_removed_points`, `physics_verdict`, `post_bounce_recovered` |
| `bounce` / `bounce_world` / `bounce_point` | bounce in pixels / in real pitch metres |
| `world_trajectory` / `ball_flight_position` | `[x_m, y_m, z_m]` — lateral, down-pitch, height |
| `trajectory_3d` / `trajectory_pixels` | same path with `frame_index` + `time_sec` (animation / overlay) |
| `trajectory_matrices` / `model_matrix` / `matrix_convention` | 4×4 row-major poses for a 3D renderer |
| `line` | `{label, confidence, reliability}` — `wide_off`, `outside_off`, `off_stump`, `middle_stump`, `leg_stump`, `down_leg`, `wide_leg` |
| `length` | `{label, confidence, distance_from_batsman_m}` — **only** `yorker`, `full_length`, `good_length`, `short_length` |
| `speed` / `speed_kmph` | release→bounce speed, km/h |
| `swing_sf` / `swing_factor` | 0–1 factor (**not** centimetres) |
| `spin_factor` / `spin_degree` / `spin_unit` / `spin_status` | path-curvature proxy — **not** measured RPM |
| `swing_type` | `inswing` / `outswing` / `straight` |
| `confidence_score` / `confidence_pct` / `confidence_label` | headline trust — **show these** |
| `raw_confidence_score` | internal intermediate — do not display |

`trajectory_source` is `"frontend_scaled_path"`: `x_m`/`y_m` are the tracked path rescaled onto the pitch and `z_m` is a **physics-shaped estimate** — a single camera cannot measure height. The arc *shape* is right; the height number is indicative.

### 5.4 Key tunables (`PipelineConfig`)

| Field | Default | Note |
|---|---|---|
| `ball_model_path` | `models/ball_ft_t4.pt` | primary detector |
| `ball_model_alt_path` | `models/ball_best_leather_new.pt` | **the ensemble** — do not remove |
| `hybrid_ensemble` | `True` | enables 2-model detection |
| `ball_confidence` | `0.05` | acceptance floor, **not** a quality score |
| `inference_imgsz` | `1280` | at 640 the ball is unresolvable |
| `pitch_length_m` | `20.12` | scales speed + every down-pitch value |
| `save_video` / `save_json` | `True` | outputs |

### 5.5 Fragile spots worth knowing before you touch this code

* **`scripts/` deleted → the physics gate fails silently.** `_load_physics_gate()` catches the ImportError and returns `None`; the physics fields then revert to defaults with no error. Always check `_PHYSICS_FILTER is not None`.
* **`cfg.job_id` unset → no WebSocket progress at all** (§3.3).
* **The speed realism band is applied to the *published* value.** The calibration multiplier is applied **before** the `[30, 165]` km/h gate; if reordered, a multiplier can publish a speed above the max (this is how a 167 km/h — faster than the world record — can appear). `167` is **not hardcoded anywhere**; it is computed `distance ÷ time`.
* **Label sets are normalised at the response layer** (`normalise_line()` / `normalise_length()`). The internal enums have more buckets (`full_toss`, `short_of_good`, `bouncer`, `wide_outside_off`…). If you add an enum value, add it to the map or it becomes `"unknown"`.
* **`api/settings.py` (not shipped here) defaults `inference_imgsz=640`** — anything routing the pipeline through that config silently halves detection resolution.
* **More than one uvicorn worker breaks `/status`** (process-local job store).

---

## 6. The three model weights — all mandatory

| Model | Size | Role | If removed |
|---|---|---|---|
| `ball_ft_t4.pt` | 90 MB | primary ball detector | ball barely detected |
| `ball_best_leather_new.pt` | 156 MB | 2nd detector (**ensemble**) | **release phase vanishes** — measured: this model alone finds the ball at f54–f58; `ft_t4` only picks it up at f59 |
| `stump_best.pt` | 6 MB | stumps → pitch homography | `speed`, `line`, `length` and all world coordinates go **null** |

Measured effect of the ensemble on one clip: **8 pts / 67.9 km/h** (single model) → **12 pts / 86.7 km/h** (ensemble).

---

## 7. Measured accuracy — read this before quoting numbers

Scored against hand-clicked ground truth (`gt/clip01.csv`, 25 ball positions, frames 54–86, bounce f65):

| | Value |
|---|---|
| Track recall | **0.32** — finds about a third of the visible ball |
| Precision | 0.67 |
| Positional accuracy when correct | **~2 px** |
| False positives in the track | 4 (f66–69, where GT shows no visible ball) |
| Missed after the bounce | **13 positions** (f70–86) — more than the whole tracked delivery |

* **`speed_kmph` is approximate.** Speed is measured release→bounce, and the tracker confirms the track ~4 frames after the true release (`min_hits_to_confirm=3` discards the pre-confirmation detections — the detector *does* find the ball at f54/56/57 with conf 0.356/0.118/0.617). The known fix is to backfill those points on confirmation.
* **`line` is indicative only** — calibrated off stump width, so confidence is capped (~0.45).
* **`length` is metric-reliable** and more trustworthy.
* **`z_m` is an estimate, not a measurement** (§5.3).

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `physics gate loaded: False` | `scripts/` missing or incomplete — re-extract |
| `speed: null`, short track | a ball model was deleted from `models/` |
| `line`/`length`/world all null | `stump_best.pt` missing, or stumps not visible in the clip |
| `/status/{job_id}` → 404 for a live job | more than one uvicorn worker, or the process restarted |
| Webhook never arrives | check the SSRF guard accepted the URL; then the 3 retries are exhausted — use `/status` |
| Very slow (minutes/clip) | no GPU → CPU. Install the CUDA torch build |
| `ImportError: libGL.so.1` | Linux: `apt-get install -y libgl1 libglib2.0-0 ffmpeg` |
| `total_deliveries: 0` | no ball track in that clip — footage-limited, not an error |
| Out of memory | `PipelineConfig(inference_imgsz=960)` — costs accuracy |

> **Known issue (pre-existing, replicated deliberately):** the SSRF guard rejects a URL if **any** resolved address is non-public. `aistg.cricgiri.com` also resolves to a NAT64 address (`64:ff9b::/96`) which Python marks *reserved* — so on a NAT64/DNS64 network the guard blocks the default webhook. It works on the current host; worth revisiting.
