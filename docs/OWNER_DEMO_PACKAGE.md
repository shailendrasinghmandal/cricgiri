# CricGiri Model Demo — For Company Owner

**Prepared by:** Shailendra Singh  
**Date:** June 2026  
**What this is:** Live AI cricket bowling analytics — upload a video, get speed, bounce, line, length, swing, trajectory + annotated video.

---

## FASTEST WAY TO TEST (2 minutes)

### Step 1 — Open this link in browser (while API is running on office Wi‑Fi)

```
http://192.168.29.107:8000/docs
```

This opens the **live interactive API**. No coding needed.

### Step 2 — Test the model

1. Click **`POST /api/v1/analyze-video`** → **Try it out**
2. Upload any bowling MP4 from `videos/` folder (e.g. `test.mp4`)
3. Set `bowler_arm` = **right**
4. Click **Execute**
5. Copy the **`job_id`** from the response
6. Open **`GET /api/v1/analysis/{job_id}`** → paste job_id → **Execute**
7. See full analytics: speed, bounce, line, length, swing, trajectory

### Step 3 — Download annotated video (optional)

Open in browser (replace JOB_ID):
```
http://192.168.29.107:8000/api/v1/analysis/JOB_ID/video
```

---

## WHAT TO SHARE WITH OWNER (copy-paste message)

```
Subject: CricGiri AI Bowling Analytics — Live Demo Ready

Hi,

The cricket ball analytics model is ready for testing. Here's what was built:

✅ Ball detection (YOLOv8 custom model)
✅ Ball tracking (Kalman filter)
✅ Speed estimation (km/h)
✅ Bounce point detection
✅ Line & length classification
✅ Swing estimation
✅ Trajectory + heatmap data
✅ FastAPI REST API for mobile app integration

LIVE TEST (same office Wi‑Fi):
→ http://192.168.29.107:8000/docs

Upload a bowling video and get full analytics in ~15 seconds.

Verified sample result (test.mp4):
• Speed: 110 km/h
• Line: middle_stump | Length: good_length
• Confidence: 0.57

Full API documentation attached: docs/API_SHARE_DOCUMENT.md

— Shailendra
```

---

## PRE-RECORDED PROOF (if owner cannot access live API)

Share these files directly from the project folder:

| What | File path |
|---|---|
| **Annotated demo video** | `api_storage/results/cb18fd2a-7240-4fd8-a7e5-5374784ee374_annotated.mp4` |
| **Analytics JSON** | `api_storage/results/cb18fd2a-7240-4fd8-a7e5-5374784ee374.json` |
| **Verified API response** | `outputs/api_sample_verify.json` |
| **All test videos output** | `outputs/hard_frame_recovery/` (13 annotated MP4s) |
| **API integration doc** | `docs/API_SHARE_DOCUMENT.md` |

---

## VERIFIED SAMPLE RESULT (test.mp4)

```json
{
  "status": "completed",
  "speed_kmph": 110.0,
  "line": "middle_stump",
  "length": "good_length",
  "swing_type": "none",
  "bounce_point": { "x": -0.055, "y": 0.416 },
  "confidence_score": 0.567,
  "processing_time_sec": 11.5
}
```

---

## BEFORE OWNER TESTS — YOU MUST DO THIS

On your PC, run in terminal:

```bash
cd "d:\cricket_final\cricket project"
venv\Scripts\python.exe run_api.py
```

Keep terminal open. Then share the link:
```
http://192.168.29.107:8000/docs
```

Owner must be on **same Wi‑Fi** as your PC.

---

## IF OWNER IS REMOTE (not same Wi‑Fi)

Option A — Send pre-recorded files:
- Annotated MP4 from `api_storage/results/`
- `docs/API_SHARE_DOCUMENT.md`

Option B — Deploy API to cloud (AWS/Render) and share production URL (future step).

---

## SYSTEM ARCHITECTURE (one line for owner)

```
Mobile App → FastAPI → YOLO Ball Detection → Kalman Tracking → Analytics JSON + Annotated Video
```

---

*CricGiri Cricket Analytics · Model: ball_best.pt · API v1.0*
