# CricGiri Delivery Analytics API

A single-endpoint HTTP API around the proven offline analysis engine
(`scripts/run_demo_testing.analyze_video`): low-conf ensemble detection → offline
motion-consistency mapping → physics gate → static-cluster guard → homography
analytics → release→bounce time-of-flight speed. Returns the delivery JSON schema
(identical to `testing_result_new/clipNN.json`).

## Run

```bash
# GPU venv (RTX 4050). Models load once at startup.
./venv/Scripts/python.exe -m uvicorn api.delivery_api:app --host 0.0.0.0 --port 8000
```

Interactive docs: `http://<host>:8000/docs`

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness + device/model loaded |
| GET | `/` | redirect to `/docs` |
| POST | `/analyze` | upload one delivery video → schema JSON |

### `POST /analyze`

Multipart form:
- `video` (file, required) — `.mp4/.mov/.avi/.mkv/.webm`, ≤ 200 MB
- `pitch_length` (float, optional, default `20.12`) — real stump-to-stump length in
  metres. **This is the speed scale knob** — pass the true pitch length for best speed
  accuracy (a 10% error in pitch length ≈ 10% error in speed).

Header (only if `API_KEY` env is set): `X-API-Key: <key>`

```bash
curl -X POST http://localhost:8000/analyze \
  -F "video=@delivery.mp4" \
  -F "pitch_length=20.12"
```

## Responses

**Success (`200`)** — the full delivery schema plus `processing_sec`:

```json
{
  "source_video": "delivery.mp4",
  "fps": 60.0,
  "total_frames": 198,
  "total_deliveries": 1,
  "pipeline_version": "offline_mapping+physics_gate+reconstruction",
  "detection_confidence": 0.05,
  "processing_sec": 30.9,
  "deliveries": [
    {
      "delivery_id": "delivery_7adda1",
      "frame_start": 46, "frame_end": 75,
      "track": {"num_points": 17, "average_confidence": 0.535,
                "physics_removed_points": 0, "physics_verdict": "valid",
                "post_bounce_recovered": false},
      "bounce": {"frame_index": 70, "x_pixel": 198.4, "y_pixel": 337.0},
      "bounce_world": {"x_m": -0.684, "y_m": 16.243},
      "bounce_point": {"x": -0.684, "y": 16.243},
      "world_trajectory": [[-1.27, 19.09, 0.0], "..."],
      "ball_flight_position": [[-1.27, 19.09, 0.0], "..."],
      "line": {"label": "wide_off", "confidence": 0.1, "reliability": "indicative"},
      "length": {"label": "good_length", "confidence": 0.14, "distance_from_batsman_m": 3.88},
      "speed": {"kmph": 137.2, "confidence": 0.85, "status": "estimated"},
      "speed_kmph": 137.2,
      "swing_cm": 4.3, "swing_type": "outswing",
      "swing_confidence": 0.2, "swing_status": "indicative_direction_only",
      "heatmap_points": [[-0.684, 16.243]],
      "physically_valid": true, "confidence_score": 0.36
    }
  ]
}
```

**No delivery (`200`)** — a stationary false positive or too few ball points:

```json
{ "source_video": "clip.mp4", "total_deliveries": 0,
  "status": "NO_TRACK", "reason": "static_cluster", "deliveries": [] }
```

**Errors:** `400` unsupported/empty file · `401` bad API key · `413` too large · `500` failure.

## Field notes (honesty)

- **`world_trajectory` / `ball_flight_position`** — `[x_m (lateral), y_m (down-pitch),
  z_m]`. `z_m` is `0.0` (the path is reported on the pitch plane; monocular height is not
  reliably recoverable). Airborne points that project to non-physical world coords via the
  ground homography are filtered out, so this can be short/empty on poorly-calibrated clips.
- **`line` / `swing`** — lateral (off-stump-width) calibration is weak, so these are
  `indicative` only.
- **`length` / `bounce`** — the bounce is on the ground plane, so down-pitch metrics are
  metric-reliable when the stumps are detected.
- **`speed`** — release→bounce time-of-flight; `status: estimated`; accuracy improves when
  the true `pitch_length` is supplied.

## Notes

- Models load once at startup; GPU inference is serialised (single RTX 4050). Each request
  is ~20–35 s. For high throughput, front with a queue / multiple GPU workers.
- Production weights (`ball_ft_t4.pt` + leather ensemble, `stump_best.pt`) are read-only.
- Config via env: `API_KEY` (enable auth), `MAX_UPLOAD_MB` (see `api/settings.py`).
