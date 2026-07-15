# YOLO configuration for best cricket ball accuracy

Accuracy comes from **two layers**:

1. **Model quality** (training data + weights + `imgsz=1280`)
2. **Inference + tracking** (confidence, NMS, augment, Kalman, optical flow)

You already use YOLOv8 via Ultralytics. This guide maps each knob to cricket ball detection.

---

## 1. Inference settings (runtime — change without retraining)

Centralized in `tracking/yolo_inference.py` and CLI flags.

| Parameter | Best for cricket ball | Why |
|-----------|----------------------|-----|
| **imgsz** | **1280** (never 640 for production) | Ball is ~10–25 px at 720p; 640 shrinks it below reliable size |
| **conf** | **0.08–0.14** (tracker gate) | Lower = more recall on blur; too low = crowd/stump FPs |
| **scan_conf_floor** | **0.01** | Keeps weak boxes in cache for path recovery |
| **iou** | **0.45** | NMS overlap; 0.5+ merges duplicate balls |
| **agnostic_nms** | **True** | Single class; avoids class-split NMS bugs |
| **max_det** | **3–5** | One ball; allow 3 for motion-blur duplicates before NMS |
| **half** | **True** on GPU (T4+) | ~1.5× faster, tiny recall loss |
| **augment** | **True** (`--high-accuracy`) | Ultralytics multi-scale TTA at predict; +recall on hard frames |

### Recommended commands

**Maximum recall (blur, fast ball):**

```powershell
.\venv\Scripts\python.exe -m pipeline.pipeline `
  --video "videos\your_clip.mp4" `
  --high-accuracy `
  --ball-confidence 0.10 `
  --inference-imgsz 1280
```

**Fewer false positives (clean nets):**

```powershell
.\venv\Scripts\python.exe -m pipeline.pipeline `
  --video "videos\your_clip.mp4" `
  --high-accuracy `
  --ball-confidence 0.14 `
  --yolo-iou 0.45
```

**Do not use `--production` alone for recall** — `conf=0.35` is tuned for stability, not missing-frame recovery.

### Tuning `conf` without retraining

| Symptom | Action |
|---------|--------|
| Ball often missing | Lower `--ball-confidence` by **0.02** (floor ~0.06) |
| Wrong objects tracked | Raise by **0.03**; enable `--no-clean-video` on broadcast |
| Path jittery but detections OK | Fix tracking (`--high-accuracy`), not only conf |

---

## 2. Model / training (retrain for real gains)

Training script: `train.py`  
Dataset: **`dataset/ball_clean`** or **`dataset/ball_hard_2000`** (hard failure frames).

### Critical training settings (already in `train.py`)

| Setting | Value | Note |
|---------|-------|------|
| imgsz | **1280** | Must match inference |
| SCALE | **0.25** | Do not shrink tiny balls (was 0.6 — bad) |
| ERASING | **0** | Erasing deletes 15 px balls |
| FLIPUD | **0** | Gravity direction |
| BLUR | **0.25** | Motion blur (now passed to `model.train`) |
| single_cls | **True** | Only on **ball-only** YAML |
| patience | **40** | Early stop |

### AWS / local train command

```bash
python train.py \
  --data dataset/ball_hard_2000/data.yaml \
  --epochs 80 \
  --batch 8 \
  --imgsz 1280 \
  --device 0
```

Then copy `runs/ball_detector/.../weights/best.pt` → `models/ball_best.pt`.

### Should you change model size?

| Model | Speed | Small object recall |
|-------|-------|---------------------|
| yolov8n | Fastest | OK (current `ball_best.pt` likely n/s) |
| **yolov8s** | Medium | **Better** for 10–20 px balls |
| yolov8m | Slow | Diminishing returns for one class |

For a **retrain from scratch** on lots of data:

```bash
python train.py --scratch --model yolov8s.pt --data dataset/ball_clean/data.yaml --epochs 200 --imgsz 1280
```

For **fine-tune** (recommended, fastest):

```bash
python train.py --data dataset/ball_finetune/data.yaml --epochs 50 --imgsz 1280
```

### Data beats architecture

Target metrics on validation:

- **mAP@50 ≥ 0.80** (good)
- **Recall ≥ 0.75** (most important for tracking)

Add frames where the pipeline missed the ball:

1. Run pipeline with `--trajectory-debug`
2. Export missed frames → label ball only
3. Merge into `dataset/ball_hard_2000` or `ball_finetune`
4. Retrain from `models/ball_best.pt`

---

## 3. What YOLO cannot fix alone

Even a perfect detector needs the rest of the stack:

| Problem | Fix |
|---------|-----|
| 1–3 frame gaps | `--high-accuracy` (byte track, optical flow, cache rebuild) |
| Painted trajectory on video | Auto reference-arc + inpaint (built-in) |
| Speed wrong | Calibration / `pixels_per_meter` |
| Path past bat | Phase detection (release → bat) |

---

## 4. Validation workflow

After new `ball_best.pt`:

```powershell
# 1) Frame-level spot check
.\venv\Scripts\python.exe _check_model_on_frames.py

# 2) Full pipeline + scores
.\venv\Scripts\python.exe -m pipeline.pipeline --video videos\test_video5.mp4 --high-accuracy --out-json outputs\eval.json
.\venv\Scripts\python.exe _accuracy_report.py outputs

# 3) Compare to previous weights (keep backup of old ball_best.pt)
```

---

## 5. Quick reference — `--high-accuracy` YOLO bundle

Enabled automatically:

```text
imgsz=1280, conf=0.14, iou=0.45, agnostic_nms=True, max_det=5,
half=True, augment=True, scan_floor=0.01
```

Plus tracking: byte track, optical flow, cache Kalman rebuild, smart bridge.

This is the best **out-of-the-box** configuration without a new training run.
