---
title: CricGiri Delivery Analytics API
emoji: 🏏
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# CricGiri Delivery Analytics API

Upload a single cricket-delivery video → receive trajectory, bounce, line, length,
speed and swing analytics as JSON.

## Endpoints

- `GET  /health` — liveness + which model/device is loaded
- `GET  /docs` — interactive test page (upload a video, see the JSON live)
- `POST /analyze` — `multipart/form-data`: `video=<file>`, `pitch_length=<float, default 20.12>`

## Example

```bash
curl -X POST "https://<your-space>.hf.space/analyze" \
  -F "video=@clip.mp4" \
  -F "pitch_length=20.12"
```

Returns the delivery JSON (`source_video`, `fps`, `deliveries[]` with `track`,
`bounce`, `world_trajectory`, `line`, `length`, `speed`, `swing`, `confidence_score`).

This Space runs on **CPU** — a clip takes ~1–3 minutes. The Space sleeps after
inactivity and wakes on the next request (~30 s cold start).
