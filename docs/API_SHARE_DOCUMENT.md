# CricGiri Cricket Analytics API
### Client Integration Document · Version 1.0

---

## 1. Overview

The **CricGiri Analytics API** is a REST service that analyzes cricket bowling practice videos and returns AI-powered delivery metrics:

| Output | Description |
|---|---|
| Bowling speed | Estimated speed in km/h |
| Bounce point | Where the ball bounced on the pitch (world coordinates) |
| Line | off_stump, middle_stump, leg_stump, outside_off, outside_leg |
| Length | yorker, full, good_length, short, bouncer, etc. |
| Swing | inswing / outswing / none + magnitude in cm |
| Trajectory | Ball path as `[x, y, z]` world coordinates |
| Heatmap points | Bounce locations for pitch visualization |
| Annotated video | Optional MP4 with trajectory overlay |

**Technology:** FastAPI (Python) · YOLOv8 ball detection · Kalman tracking

---

## 2. Base URL

Replace `BASE_URL` with the server address provided by the CricGiri team.

| Environment | Example |
|---|---|
| Local / LAN testing | `http://192.168.1.10:8000` |
| Production (when deployed) | `https://api.cricgiri.com` |

**Interactive docs (Swagger UI):**  
`{BASE_URL}/docs`

**Health check:**  
`{BASE_URL}/api/v1/health`

> **Note:** `localhost` only works on the server machine. Clients must use the server's IP or domain.

---

## 3. Authentication

**MVP:** No API key required.

Production deployments may add API keys or JWT in a future version.

---

## 4. Integration Flow

```
Client App
    │
    ▼
POST /api/v1/analyze-video  (upload MP4/MOV)
    │
    ▼
Response: { "job_id": "...", "status": "queued" }
    │
    ▼
Poll every 3–5 seconds:
GET /api/v1/analysis/{job_id}
    │
    ├── status = "processing"  → wait and poll again
    ├── status = "completed"   → read deliveries[] analytics
    └── status = "failed"      → read error field
    │
    ▼
Optional: GET /api/v1/analysis/{job_id}/video
    → download annotated MP4
```

---

## 5. API Endpoints

### 5.1 Health Check

**`GET /api/v1/health`**

Use to verify the API and models are loaded.

**Response (200):**
```json
{
  "status": "ok",
  "app": "CricGiri Analytics API",
  "version": "1.0.0",
  "ball_model": "models/ball_best.pt",
  "stump_model": "models/stump_best.pt"
}
```

---

### 5.2 Upload Video for Analysis

**`POST /api/v1/analyze-video`**

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Bowling video (mp4, mov, avi, mkv, webm) |
| `bowler_arm` | string | No | `right` (default) or `left` |
| `save_video` | boolean | No | `true` (default) — generate annotated MP4 |

**cURL example:**
```bash
curl -X POST "{BASE_URL}/api/v1/analyze-video" \
  -F "file=@bowling_practice.mp4" \
  -F "bowler_arm=right"
```

**Response (200):**
```json
{
  "job_id": "cb18fd2a-7240-4fd8-a7e5-5374784ee374",
  "status": "queued",
  "message": "Video queued for analysis"
}
```

**Errors:**

| Code | Meaning |
|---|---|
| 400 | Invalid file format or missing filename |
| 413 | File exceeds 200 MB upload limit |

---

### 5.3 Get Analysis Status & Results

**`GET /api/v1/analysis/{job_id}`**

Poll this endpoint until `status` is `completed` or `failed`.

**Response while processing (200):**
```json
{
  "job_id": "cb18fd2a-7240-4fd8-a7e5-5374784ee374",
  "status": "processing",
  "progress_pct": 50.0,
  "video_filename": "test.mp4",
  "total_deliveries": 0,
  "deliveries": []
}
```

**Response when completed (200) — real verified sample:**
```json
{
  "job_id": "cb18fd2a-7240-4fd8-a7e5-5374784ee374",
  "status": "completed",
  "progress_pct": 100.0,
  "video_filename": "test.mp4",
  "session_id": "dea5974d-2f3a-4702-825b-a61d2b59883a",
  "total_deliveries": 1,
  "processing_time_sec": 11.49,
  "deliveries": [
    {
      "delivery_id": "28551f81-97a1-4f51-8e1c-e651e3964cea",
      "speed_kmph": 110.0,
      "bounce_point": { "x": -0.055, "y": 0.416 },
      "trajectory": [
        [0.0, 0.0, 2.1],
        [0.0, 0.026, 2.067],
        [-0.055, 0.416, 0.0]
      ],
      "line": "middle_stump",
      "length": "good_length",
      "swing_cm": 3.58,
      "swing_type": "none",
      "heatmap_points": [[-0.055, 0.416]],
      "confidence_score": 0.567
    }
  ],
  "heatmap_stats": {
    "total_deliveries": 1,
    "hottest_zone_x_m": -0.055,
    "hottest_zone_y_m": 0.402,
    "zone_distribution": { "good_length": 1 }
  },
  "annotated_video_url": "/api/v1/analysis/cb18fd2a-7240-4fd8-a7e5-5374784ee374/video",
  "result_json_url": "/api/v1/analysis/cb18fd2a-7240-4fd8-a7e5-5374784ee374/result"
}
```

**Response when failed (200):**
```json
{
  "job_id": "...",
  "status": "failed",
  "error": "Description of what went wrong",
  "deliveries": []
}
```

**Errors:**

| Code | Meaning |
|---|---|
| 404 | Job ID not found |

---

### 5.4 Download Annotated Video

**`GET /api/v1/analysis/{job_id}/video`**

Returns the annotated MP4 (trajectory overlay) when processing is complete.

**Example:**
```
{BASE_URL}/api/v1/analysis/cb18fd2a-7240-4fd8-a7e5-5374784ee374/video
```

**Response:** `video/mp4` file download

---

### 5.5 Download Full Result JSON

**`GET /api/v1/analysis/{job_id}/result`**

Returns the complete raw analysis JSON file.

---

## 6. Delivery Fields Reference

| Field | Type | Description |
|---|---|---|
| `delivery_id` | string | Unique ID for this delivery |
| `speed_kmph` | float | Bowling speed in km/h |
| `bounce_point` | object | `{ "x": metres, "y": metres }` on pitch |
| `trajectory` | array | List of `[x, y, z]` world coordinates |
| `line` | string | `off_stump`, `middle_stump`, `leg_stump`, `outside_off`, `outside_leg` |
| `length` | string | `yorker`, `full`, `good_length`, `short`, `bouncer`, etc. |
| `swing_cm` | float | Lateral movement in centimetres |
| `swing_type` | string | `inswing`, `outswing`, or `none` |
| `heatmap_points` | array | `[[x, y], ...]` bounce spots for heatmap UI |
| `confidence_score` | float | 0.0–1.0 overall confidence |

---

## 7. Job Status Values

| Status | Meaning | Client action |
|---|---|---|
| `queued` | Video received, waiting to process | Poll |
| `processing` | AI pipeline running | Poll (check `progress_pct`) |
| `completed` | Done — results in `deliveries[]` | Display analytics |
| `failed` | Error occurred | Show `error` message |

**Recommended poll interval:** 3–5 seconds  
**Typical processing time:** 10–90 seconds per video (depends on length and hardware)

---

## 8. Mobile / Frontend Integration (Pseudocode)

```javascript
// Step 1: Upload
const form = new FormData();
form.append("file", videoFile);
form.append("bowler_arm", "right");

const uploadRes = await fetch(`${BASE_URL}/api/v1/analyze-video`, {
  method: "POST",
  body: form,
});
const { job_id } = await uploadRes.json();

// Step 2: Poll until done
let result;
while (true) {
  const res = await fetch(`${BASE_URL}/api/v1/analysis/${job_id}`);
  result = await res.json();
  if (result.status === "completed") break;
  if (result.status === "failed") throw new Error(result.error);
  await sleep(4000);
}

// Step 3: Use analytics
const delivery = result.deliveries[0];
console.log("Speed:", delivery.speed_kmph, "km/h");
console.log("Line:", delivery.line);
console.log("Length:", delivery.length);

// Step 4: Optional annotated video
const videoUrl = `${BASE_URL}${result.annotated_video_url}`;
```

---

## 9. Video Requirements

| Requirement | Detail |
|---|---|
| Format | MP4, MOV (recommended), AVI, MKV, WebM |
| Max file size | 200 MB |
| Camera angle | Behind stumps, tripod-mounted |
| Content | Single bowling delivery or practice session |
| Visibility | Full pitch visible, bowling side in frame |

---

## 10. Legacy Endpoints (PDF aliases)

These mirror the PDF specification and behave identically:

| PDF spec | Actual endpoint |
|---|---|
| `POST /analyze-video` | `POST /api/v1/analyze-video` |
| `GET /analysis/{job_id}` | `GET /api/v1/analysis/{job_id}` |

---

## 11. Testing Without Code

1. Open `{BASE_URL}/docs` in a browser
2. Expand **POST /api/v1/analyze-video**
3. Click **Try it out**
4. Upload a bowling MP4, set `bowler_arm` to `right`
5. Click **Execute** — copy the `job_id`
6. Expand **GET /api/v1/analysis/{job_id}**
7. Paste `job_id`, click **Execute** — view full analytics JSON

---

## 12. Limits & Notes (MVP)

- Processing is **async** — upload returns immediately, results via polling
- One video processed at a time per server instance (queue-based)
- Speed and trajectory are **approximate** (single-camera analytics)
- Spin RPM is **not** included in MVP
- Results are stored on server until manually deleted

---

## 13. Support & Contact

| Item | Detail |
|---|---|
| API version | 1.0.0 |
| Framework | FastAPI + Uvicorn |
| Model | `ball_best.pt` (custom YOLOv8 cricket ball) |
| Verified test | Job `cb18fd2a-7240-4fd8-a7e5-5374784ee374` on `test.mp4` |

For server URL, deployment access, or integration support, contact the CricGiri engineering team.

---

*Document generated for CricGiri Cricket Analytics · FastAPI REST API v1.0*
