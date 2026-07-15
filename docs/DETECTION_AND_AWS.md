# Ball detection model, tuning, and AWS GPU training

## Which model is used?

| Component | File | Type |
|-----------|------|------|
| **Ball** | `models/ball_best.pt` | YOLOv8 (Ultralytics), **single class: ball** (~22 MB) |
| **Stumps** | `models/stump_best.pt` | YOLOv8, calibration only |

There is no separate “kernel” setting in the app. The network uses learned convolution filters inside YOLO. What you control at runtime:

| Knob | Effect on blur / missed ball |
|------|------------------------------|
| `--inference-imgsz` | **1280** keeps tiny/fast balls larger in the image (best for blur) |
| `--ball-confidence` | Primary gate (default **0.10**); scan still runs at **0.01** |
| `--enhanced-detection` | ROI crop + TTA (flip/brightness) + dynamic confidence |
| `--byte-track` | Re-associates **low-confidence** boxes near predicted position |
| `--tracker-optical-flow` | LK flow between frames when YOLO misses |
| `--hybrid-optical-flow` | Fills short gaps in the **drawn path** from cache |
| `--max-missing-frames` | How long Kalman coasts without a detection (default 12; use **18–20** for blur) |

Training-side blur robustness is in `train.py`: `BLUR=0.25`, `HSV_*`, `imgsz=1280`.

## Why the ball is “missing”

Typical causes in this project:

1. **Motion blur** — ball smear → low YOLO confidence → below `ball_confidence`
2. **Small object** — at 640px inference the ball can be only a few pixels
3. **False-start trim** — early real points dropped as “phantom” (use `--skip-false-start-trim` or `--match-should-output`)
4. **Strict geometry filter** — box rejected for tracker but may still be in **cache** for path recovery
5. **Post-bat extension** — extra points after bat contact (use `--skip-bounce-extend`)

Diagnose per frame:

```powershell
.\venv\Scripts\python.exe _diagnose_video5_frames.py
```

## Presets

### Match `should_output/` trajectories (same path timing, your blue HUD)

```powershell
.\venv\Scripts\python.exe -m pipeline.pipeline `
  --video should_output\video3.mp4 `
  --match-should-output `
  --out-video outputs\should_match\video3_match.mp4 `
  --out-json outputs\should_match\video3_match.json
```

Batch + compare to `outputs/should_video*_smooth.json`:

```powershell
.\venv\Scripts\python.exe scripts\run_should_output_match.py
```

### Blur / fast ball recovery (test videos, night nets)

```powershell
.\venv\Scripts\python.exe -m pipeline.pipeline `
  --video videos\test_video5.mp4 `
  --blur-recovery `
  --ball-confidence 0.08 `
  --out-video outputs\hybrid\test_video5_blur.mp4 `
  --out-json outputs\hybrid\test_video5_blur.json
```

## AWS GPU server (Tesla T4)

**Server:** `ec2-user@52.220.231.57`  
**Key:** `YOLOv8-model-training.ppk` (project root)  
**Access:** Office IP only — send your **public IP** to Moksh to open the firewall.

This environment cannot SSH to your server from here. On your PC:

### WinSCP

1. New site → Host `52.220.231.57`, User `ec2-user`
2. Advanced → SSH → Authentication → Private key file → select `YOLOv8-model-training.ppk`
3. Upload folder: `dataset/ball_finetune/` (or `dataset/ball_clean/`)
4. Upload: `train.py`, `dataset/.../data.yaml`, `models/ball_best.pt`

### PuTTY

1. Host `ec2-user@52.220.231.57`
2. Connection → SSH → Auth → Browse `.ppk`
3. Terminal:

```bash
cd ~/cricket   # or your upload path
python3 -m venv venv && source venv/bin/activate
pip install ultralytics torch torchvision
python train.py --data dataset/ball_finetune/data.yaml --epochs 50 --imgsz 1280 --device 0
```

4. Download `runs/ball_detector/weights/best.pt` → replace `models/ball_best.pt` locally.

### Fine-tune dataset already prepared

- `dataset/ball_finetune/` — 136 labeled + negatives, train/val split
- Command: see `dataset/ball_finetune/data.yaml`

## Target output

`should_output/video1-4.mp4` are **reference renders** (golden path + graphics).  
Reference tracks: `outputs/should_video1_smooth.json` … `should_video4_smooth.json`.

Goal: same **release → bat** path on the same frames; keep **your** pipeline colors/HUD (already in `analytics/visualizer.py`).
