# CricGiri Pipeline — Server Setup Guide

Self-contained package: **all code + all model weights**. Nothing else to download.
Run a cricket video through it and get a **trajectory video** + **delivery JSON**.

---

## 1. What's in the package

```
run_pipeline.py                    <- the entry point you run
pipeline/                          <- the analysis pipeline (the product)
analytics/                         <- bounce, calibration, speed, visualiser
tracking/                          <- ball tracker, YOLO inference
scripts/                           <- REQUIRED (see note below)
config/                            <- tracking_defaults.yaml
models/ball_ft_t4.pt               (90 MB)  primary ball detector
models/ball_best_leather_new.pt    (156 MB) 2nd ball detector — ENSEMBLE
models/stump_best.pt               (6 MB)   stump detector
requirements.txt
DELIVERY_API_RESPONSE_FORMAT.md    <- the JSON contract
outputs/  videos/  logs/           <- created for you (results land in outputs/)
```

> ### ⚠️ Do not delete anything — every part is load-bearing
> * **All three `models/*.pt` are mandatory.**
>   * Remove `ball_best_leather_new.pt` → the release phase is never detected
>     (this model alone finds the ball at release), tracks are short, `speed` degrades.
>   * Remove `stump_best.pt` → no pitch calibration → `speed`, `line`, `length`
>     and all world coordinates go null.
> * **`scripts/` is required, not optional.** `pipeline/pipeline.py` loads
>   `scripts/physics_gate_v2.py` **by path at runtime**, and that file loads
>   `scripts/delivery_reconstruction.py` when imported. If either is missing the
>   physics gate **silently degrades** (`physics_verdict` / `physically_valid`
>   become meaningless defaults) — no error is raised. Keep the folder structure intact.

---

## 2. Requirements

* **Python 3.10 or 3.11**
* **~4 GB RAM** minimum
* **NVIDIA GPU (optional but recommended)** — ~30–60 s per clip.
  CPU-only works but is ~1–3 min per clip.
* ~1.5 GB disk for dependencies + 250 MB for this package

---

## 3. Install

```bash
unzip cricgiri_pipeline_package.zip -d cricgiri
cd cricgiri

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# --- PyTorch first: pick the build that matches the server ---
# CPU-only server:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU server (CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# --- everything else ---
pip install -r requirements.txt
```

### Linux servers only — OpenCV system libs
`opencv-python-headless` still needs a couple of system libraries:
```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 ffmpeg
```

---

## 4. Verify the install (do this before your first real clip)

```bash
python -c "import torch, cv2, ultralytics; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python -c "from pipeline.pipeline import CricketAnalyticsPipeline, PipelineConfig; c=PipelineConfig(); print('primary:', c.ball_model_path); print('alt    :', c.ball_model_alt_path); print('ensemble:', c.hybrid_ensemble, '| conf', c.ball_confidence, '| imgsz', c.inference_imgsz)"
```
Expected:
```
primary: models/ball_ft_t4.pt
alt    : models/ball_best_leather_new.pt
ensemble: True | conf 0.05 | imgsz 1280
```
Confirm the **physics gate** actually loaded (this is the silent-failure check):
```bash
python -c "from pipeline.pipeline import _PHYSICS_FILTER; print('physics gate loaded:', _PHYSICS_FILTER is not None)"
```
Must print **`True`**. If it prints `False`, `scripts/` is missing or incomplete.

---

## 5. Run a video

```bash
python run_pipeline.py clip.mp4
```
Output:
```
JSON  -> outputs/result.json
VIDEO -> outputs/result.mp4
```

Options:
```bash
python run_pipeline.py clip.mp4 --pitch-length-yards 22     # REAL pitch length (14-22)
python run_pipeline.py clip.mp4 --out-json my.json --out-video my.mp4
python run_pipeline.py clip.mp4 --no-video                  # JSON only (faster)
```

> **`--pitch-length-yards` matters.** It sets the metric scale: speed and every
> down-pitch value scale with it. Pass the **actual** pitch length of the footage.
> Default is 22 (a full pitch).

### Use it from your own code
```python
from pipeline.pipeline import CricketAnalyticsPipeline, PipelineConfig

cfg = PipelineConfig(
    video_path="clip.mp4",
    output_video_path="outputs/result.mp4",
    output_json_path="outputs/result.json",
    pitch_length_m=20.12,      # 22 yards
    save_video=True, save_json=True,
)
result = CricketAnalyticsPipeline(cfg).run().to_dict()
print(result["deliveries"][0]["speed_kmph"])
```
> Construct `PipelineConfig()` plainly and you get the ensemble automatically.
> **Do not override** `ball_model_path` / `ball_model_alt_path` / `inference_imgsz`
> unless you intend to — overriding them turns off the tuned configuration.

---

## 5b. Run it as an HTTP API (+ backend webhook)

This package is a **drop-in replacement** for the previously delivered API — same
module path, port, routes, request shapes, webhook payloads and WebSocket protocol.
**The backend integration needs no changes.** Only the engine underneath differs
(the pipeline, so `result` carries `swing_sf` / `spin_factor` / `trajectory_matrices`
/ `confidence_pct`, and detection uses the 2-model ensemble).

```bash
pip install -r requirements.txt        # includes fastapi/uvicorn/requests/websockets
python -m uvicorn api.delivery_api:app --host 0.0.0.0 --port 7860 --workers 1
```
Or with Docker (same CMD/port as before):
```bash
docker build -t cricgiri . && docker run -p 7860:7860 cricgiri
```

> **`--workers 1` is required.** The YOLO models are shared singletons (inference is
> serialised by a lock) and the `/status` job store is an in-memory, process-local
> dict. More than one worker breaks `/status`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | redirect to `/docs` |
| GET | `/health` | liveness + which models/device are loaded |
| GET | `/status/{job_id}` | poll an async (webhook) job: `processing` / `done` / `error` |
| POST | `/analyze` | file upload (**sync**) or `video_url` (**async + webhook**) |
| WS | `/ws/analyze` | same analysis, streamed with live progress |

### Webhook flow (`video_url` → always async)
```bash
curl -X POST http://localhost:7860/analyze -H 'Content-Type: application/json' -d '{
  "video_url":   "https://.../clip.mp4",
  "webhook_url": "https://your-backend/callback",
  "pitch_length": 20.12,
  "service": "...", "request_id": "...", "clip_id": "...", "session_id": "..."
}'
# -> 202 {"status":"accepted","job_id":"...", ...meta}
```
When it finishes, the result is POSTed to `webhook_url`:
```jsonc
// success
{"status":"done",  "job_id":"...", "result": { ...delivery JSON... }, ...meta}
// failure
{"status":"error", "job_id":"...", "message":"ValueError: ...",       ...meta}
```
* `webhook_url` is **optional** — defaults to `CRICGIRI_WEBHOOK_URL`
  (`https://aistg.cricgiri.com/webhook/session-clip`).
* The four keys `service` / `request_id` / `clip_id` / `session_id` are **passthrough
  metadata**: echoed verbatim in the 202, in `/status`, and in the webhook. Never interpreted.
* Delivery is **best-effort**: 3 attempts, 2 s / 4 s backoff, then logged and dropped —
  no dead-letter queue. `GET /status/{job_id}` is the fallback (kept **24 h**).
* Raw file uploads stay **synchronous**; `webhook_url` is rejected with a file upload.
* **SSRF guard:** `video_url` and `webhook_url` must resolve to a *public* IP —
  private/loopback/link-local/reserved/metadata targets are rejected.

### WebSocket
1. Connect (optionally `X-API-Key`).
2. Send **one text frame**: `{"filename":"clip.mp4","pitch_length":20.12,"api_key":"..."}`
3. Send **one binary frame**: the video bytes.
4. Receive: `{"status":"processing","stage":"detecting_ball","pct":42.0}` … then
   `{"status":"done","result":{...}}` or `{"status":"error","message":"..."}`.

### Env vars
| Var | Default | Purpose |
|---|---|---|
| `CRICGIRI_WEBHOOK_URL` | `https://aistg.cricgiri.com/webhook/session-clip` | default callback |
| `API_KEY` | unset | if set, requires `X-API-Key` |
| `MAX_UPLOAD_MB` | `200` | upload / download size cap |

---

## 6. What you get

`outputs/result.json` — full delivery JSON (see `DELIVERY_API_RESPONSE_FORMAT.md`):
`track`, `bounce`, `bounce_world`, `world_trajectory`, `trajectory_3d`,
`trajectory_matrices`, `line`, `length`, `speed_kmph`, `swing_sf`,
`spin_factor` / `spin_degree`, `confidence_pct` / `confidence_label`.

Label sets are fixed:
* **length** — `yorker` · `full_length` · `good_length` · `short_length`
* **line** — `wide_off` · `outside_off` · `off_stump` · `middle_stump` · `leg_stump` · `down_leg` · `wide_leg`
* **swing_type** — `inswing` · `outswing` · `straight`

`outputs/result.mp4` — the clip with the ball arc, pitch corridor, and a
Swing / Drift / Speed panel showing **the same numbers as the JSON**.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `physics gate loaded: False` | `scripts/` missing or incomplete — re-extract the package. |
| `speed: null`, short track | A model was deleted from `models/`. All three are required. |
| `line`/`length`/world values null | `stump_best.pt` missing, or stumps not visible in the clip. |
| Very slow (minutes/clip) | No GPU → running on CPU. Install the CUDA torch build. |
| `ImportError: libGL.so.1` | Linux: `apt-get install -y libgl1 libglib2.0-0 ffmpeg` |
| `total_deliveries: 0` | No ball track found in that clip (not an error — footage-limited). |
| Out of memory | Lower resolution: `PipelineConfig(inference_imgsz=960)` (costs accuracy). |

---

## 8. Known accuracy limits (measured against hand-labelled ground truth)

Be aware of these before putting numbers in front of a client:

* **`speed_kmph` is approximate.** The tracker does not always lock on at the
  true release frame, and speed is measured release→bounce, so it can read low.
* **Post-bounce tracking is incomplete.** On the labelled clip, 13 ball positions
  after the bounce were visible but not tracked.
* **Track recall ≈ 0.32** on the labelled clip (it finds about a third of the
  visible ball positions). Positions it *does* report are accurate to ~2 px.
* `line` is calibrated off stump width, so its confidence is capped (~0.45) and
  it is labelled `"indicative"`. `length` is metric-reliable and more trustworthy.
* `z_m` (height) in `world_trajectory` is a **physics-shaped estimate**, not a
  measurement — a single camera cannot measure height. `trajectory_source`
  marks the path as `frontend_scaled_path` for this reason.
