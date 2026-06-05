# CricGiri Production Deployment Guide

**Deadline-ready · Permanent public API · Reuses existing `ball_best.pt` pipeline**

---

## Hosting choice (read this first)

| Option | Time to deploy | Permanent URL | GPU | Best for |
|---|---|---|---|---|
| **Render.com** | **~30 min** | `https://cricgiri-api.onrender.com` | No (CPU) | **Tomorrow deadline** |
| **AWS EC2** | ~2 hours | Elastic IP + domain | Yes (g4dn) | Production + GPU |
| **Docker on VPS** | ~1 hour | Your domain | Optional | Full control |

**Recommendation for tomorrow:** Deploy to **Render** (Starter plan $7/mo = always on).  
**Recommendation for production GPU:** **AWS EC2 g4dn.xlarge** + Elastic IP.

---

## 1. Folder structure (production)

```
cricket project/
├── api/                    # FastAPI application
│   ├── main.py             # Routes: /health, /model-info, /analyze
│   ├── jobs.py             # Background pipeline worker
│   ├── settings.py         # Env configuration
│   └── production.py       # Client JSON builder
├── models/
│   ├── ball_best.pt        # Your trained YOLO weights
│   └── stump_best.pt
├── pipeline/               # Existing analytics pipeline (unchanged)
├── uploads/                # Incoming videos (server)
├── outputs/api/            # Processed JSON + MP4
├── logs/                   # api.log
├── deploy/
│   ├── Dockerfile
│   ├── render.yaml         # Render one-click deploy
│   ├── nginx/
│   ├── systemd/
│   └── aws/ec2_setup.sh
├── requirements-prod.txt
└── run_api.py
```

---

## 2. API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model-info` | Model version + settings |
| POST | `/analyze` | Upload video → `job_id` |
| GET | `/analyze/{job_id}` | Poll results (production JSON) |
| GET | `/api/v1/analysis/{job_id}/video` | Download annotated MP4 |
| GET | `/docs` | Swagger UI for client testing |

---

## 3. Production response format

**POST /analyze** returns:
```json
{
  "status": "queued",
  "job_id": "uuid",
  "poll_url": "/analyze/uuid"
}
```

**GET /analyze/{job_id}** when complete:
```json
{
  "status": "success",
  "job_id": "...",
  "speed_kmph": 110.0,
  "speed": "110.0 km/h",
  "confidence": 0.567,
  "bounce_point": {"x": -0.055, "y": 0.416},
  "trajectory": [[0.0, 0.0, 2.1], ...],
  "line": "middle_stump",
  "length": "good_length",
  "swing_type": "none",
  "swing_cm": 3.58,
  "output_video": "/api/v1/analysis/{job_id}/video",
  "processing_time_sec": 11.5
}
```

---

## 4. FASTEST DEPLOY — Render.com (permanent URL today)

### Step 1 — Push code to GitHub
```bash
git add .
git commit -m "Production API deployment"
git push origin main
```

### Step 2 — Create Render account
1. Go to https://render.com
2. New → **Web Service** → Connect GitHub repo
3. Settings:
   - **Runtime:** Docker
   - **Dockerfile path:** `deploy/Dockerfile`
   - **Plan:** Starter ($7/mo — always on, no sleep)
   - **Health check:** `/health`

### Step 3 — Environment variables (Render dashboard)
```
MODEL_VERSION=ball_best_v1
BALL_MODEL_PATH=models/ball_best.pt
SAVE_ANNOTATED_VIDEO=true
MAX_UPLOAD_MB=200
PUBLIC_BASE_URL=https://YOUR-SERVICE.onrender.com
```

### Step 4 — Deploy
Render builds Docker image and gives permanent URL:
```
https://cricgiri.onrender.com
```

### Step 5 — Share with owner
```
https://cricgiri.onrender.com/docs
```

**Custom domain (optional):** Render dashboard → Settings → Custom Domain → `api.cricgiri.com`

---

## 5. AWS EC2 deploy (GPU + permanent IP)

### Step 1 — Launch EC2
- AMI: Ubuntu 22.04
- Instance: `g4dn.xlarge` (GPU) or `t3.large` (CPU)
- Storage: 50 GB
- Security group: ports **22, 80, 443**

### Step 2 — Elastic IP
EC2 → Elastic IPs → Allocate → Associate with instance

### Step 3 — Upload project
```bash
scp -r "cricket project" ubuntu@ELASTIC_IP:/opt/cricgiri
```

### Step 4 — Run setup
```bash
ssh ubuntu@ELASTIC_IP
cd /opt/cricgiri
cp .env.production.example .env
# edit .env — set DEVICE=0 for GPU
sudo bash deploy/aws/ec2_setup.sh
```

### Step 5 — DNS
Point `api.yourdomain.com` A-record → Elastic IP

### Step 6 — SSL
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

**Permanent URL:** `https://api.yourdomain.com/docs`

---

## 6. Docker local test (before cloud)

```bash
cd "cricket project"
docker build -f deploy/Dockerfile -t cricgiri-api .
docker run -p 8000:8000 -v ./uploads:/app/uploads cricgiri-api
```

Or:
```bash
docker compose -f deploy/docker-compose.yml up --build
```

Test: http://localhost:8000/health

---

## 7. systemd — keep API always running (EC2)

```bash
sudo systemctl start cricgiri-api
sudo systemctl enable cricgiri-api    # start on boot
sudo systemctl status cricgiri-api
sudo systemctl restart cricgiri-api   # after code/model update
sudo journalctl -u cricgiri-api -f    # live logs
```

---

## 8. Example curl commands

```bash
BASE=https://api.yourdomain.com

# Health
curl $BASE/health

# Model info
curl $BASE/model-info

# Upload video
curl -X POST $BASE/analyze \
  -F "file=@bowling.mp4" \
  -F "bowler_arm=right"

# Poll results (replace JOB_ID)
curl $BASE/analyze/JOB_ID

# Download annotated video
curl -O $BASE/api/v1/analysis/JOB_ID/video
```

---

## 9. Update model weights later

```bash
# 1. Copy new weights to server
scp models/ball_best_new.pt ubuntu@SERVER:/opt/cricgiri/models/ball_best.pt

# 2. Update version label in .env
MODEL_VERSION=ball_best_v2

# 3. Restart (reloads YOLO on next job — pipeline singleton refreshes on restart)
sudo systemctl restart cricgiri-api
```

On Render: push new model to git → auto-redeploy, or upload via Render shell.

---

## 10. Security

| Setting | How |
|---|---|
| API key | Set `API_KEY=secret` in `.env` — clients send `X-API-Key: secret` |
| Upload limit | `MAX_UPLOAD_MB=200` |
| HTTPS | Nginx + certbot (EC2) or Render auto-SSL |
| File cleanup | Delete old jobs via `DELETE /api/v1/analysis/{job_id}` |

---

## 11. Performance

| Optimization | Config |
|---|---|
| GPU | `DEVICE=0` |
| Half precision | `USE_HALF_PRECISION=true` (GPU only) |
| Model loaded once | Pipeline singleton in `api/jobs.py` |
| Single worker | `--workers 1` (required for shared model) |
| Memory cleanup | `gc.collect()` after each job |

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| 502 timeout | Increase nginx `proxy_read_timeout 600s` |
| Out of memory | Use larger instance or `SAVE_ANNOTATED_VIDEO=false` |
| Model not found | Verify `models/ball_best.pt` in Docker image |
| Slow inference | Enable GPU + `USE_HALF_PRECISION=true` |

---

## 13. What to share with client/owner

```
Live API:  https://YOUR-PERMANENT-URL/docs
Health:    https://YOUR-PERMANENT-URL/health
Upload:    POST /analyze
Results:   GET /analyze/{job_id}
```

---

*CricGiri Analytics API v1.0 · FastAPI + Uvicorn · ball_best.pt*
