# Production ball tracking tuning

This guide covers the **production stability preset** added on top of the existing YOLO + Kalman pipeline. It does not replace your fine-tuned `ball_best.pt` weights.

## Quick start

```powershell
.\venv\Scripts\python.exe -m pipeline.pipeline `
  --video "videos\your_clip.mp4" `
  --production `
  --out-json outputs\session.json `
  --out-video outputs\session.mp4
```

For debug overlays (phase, velocity, Kalman, raw dets):

```powershell
.\venv\Scripts\python.exe -m pipeline.pipeline `
  --video "videos\your_clip.mp4" `
  --production --production-debug `
  --trajectory-debug
```

## What `--production` enables

| Area | Setting |
|------|---------|
| YOLO | `conf=0.35`, `iou=0.45`, `agnostic_nms=True`, `max_det=3`, `imgsz=1280` |
| Detection | Enhanced mode: ROI crop, dynamic confidence (TTA off by default) |
| Tracker | ByteTrack recovery, LK optical flow, cubic gap interpolation |
| Missed frames | Up to 18 Kalman coast frames + hybrid optical flow in gaps |
| Physics | `PhysicsMotionFilter` — box geometry, max speed/accel, prediction gate |
| Path builder | Stronger EMA + Savitzky–Golay (`savgol_window=7`) |
| Render | Catmull-Rom smooth arc (`--smooth-render-catmull`, on in preset) |

Legacy behaviour (low `conf=0.10`, 640px) is unchanged unless you pass `--production`.

## Recommended inference parameters (cricket ball)

Start here after fine-tuning on hard frames:

```text
conf              = 0.35
iou               = 0.45
agnostic_nms      = True
max_det           = 3
inference_imgsz   = 1280
scan_conf_floor   = 0.01   # permissive cache for path rebuild
half_precision    = True    # if GPU supports FP16
```

CLI equivalents: `--ball-confidence 0.35 --yolo-iou 0.45 --yolo-max-det 3 --inference-imgsz 1280 --half-precision`

## Tuning confidence

**Too many false positives (crowd, stumps, ads):**

- Raise `--ball-confidence` in steps of **0.05** (e.g. 0.35 → 0.40).
- Keep `scan_conf_floor` at **0.01** for observed-path cache unless you see phantom cache points.
- Enable `--clean-video` only on truly clean net footage.

**Too many missed frames (blur, fast ball):**

- Lower confidence slightly (0.30) **or** use `--blur-recovery` (1280 + TTA + lower conf).
- Increase `--max-missing-frames` (e.g. 20).
- Do **not** combine aggressive `--blur-recovery` with `--match-should-output` on annotated reference MP4s.

**Dynamic confidence** (production default): adapts gate between `dynamic_conf_min` (0.08) and `dynamic_conf_max` (0.22) from recent hit quality.

## Tuning smoothing

| Knob | Effect |
|------|--------|
| `--track-interpolation cubic` | Fills 1–5 frame gaps in finalized track |
| `savgol_window` (default 7 in production) | Path builder noise reduction; try 5 (snappier) or 9 (smoother) |
| `--smooth-render-catmull` | Visual-only spline; does not change analytics JSON |
| Kalman `max_missing_frames` | How long tracker predicts before LOST |

If the blue arc lags the ball on screen, reduce Catmull strength by disabling `--smooth-render-catmull` or lower `savgol_window`.

## Physics filter

Configured in `tracking/physics_constraints.py` via `ProductionTrackingConfig.physics`:

- `max_speed_px_per_frame` — reject teleports (default ~95 @ 30fps)
- `max_accel_px_per_frame2` — reject impossible kicks (default ~45)
- Aspect ratio and box area gates for non-ball blobs

Tighten speed if you still see jumps; loosen slightly on very high FPS or 4K crops.

## Tracker states (debug)

`IDLE` → `DETECTED` → `TRACKING` → `PREDICTING` (coast) → `LOST`

Short occlusions stay in `PREDICTING` until `max_missing_frames` is exceeded.

## Fine-tuning workflow

1. Run with `--production --trajectory-debug --no-video` on failure clips.
2. Export hard frames to your YOLO dataset (existing scripts).
3. Retrain from latest `models/ball_best.pt`.
4. Swap weights: `--ball-model models/ball_best.pt` (no code change).

## Preset comparison

| Preset | Use when |
|--------|----------|
| `--high-accuracy` | **Target ~90 scores**: 1280px, cache rebuild, smart bridge, conf 0.14 |
| `--production` | Stable analytics, fewer FPs, smooth path |
| `--blur-recovery` | Heavy motion blur, many 1–3 frame drops |
| `--match-should-output` | Compare to golden `should_output/` overlays (not for re-detecting painted MP4s) |

### `--high-accuracy` (recommended for your goal)

```powershell
.\venv\Scripts\python.exe -m pipeline.pipeline --video "videos\clip.mp4" --high-accuracy
```

Batch + score:

```powershell
.\venv\Scripts\python.exe _run_high_accuracy_eval.py
```

Changes vs default: conf **0.14**, imgsz **1280**, TTA+ROI, byte track + optical flow, **cache→Kalman rebuild** when interpolation &gt; 42%, **cache→tracker fallback** when strict detect empty, **smart bridge** (cache fill then linear only for gaps ≤5), skip phase slice when &lt;8 real points, speed confidence scaled by real-detection ratio.

## Module map

- `tracking/yolo_inference.py` — centralized `predict()` + `torch.no_grad()`
- `tracking/physics_constraints.py` — geometry + motion gates
- `tracking/production_config.py` — preset + `apply_production_cli`
- `tracking/track_ball.py` — Kalman, ByteTrack, phases, interpolation
- `pipeline/ball_detection.py` — ROI + TTA + dynamic conf
- `analytics/real_trajectory.py` — observed path + bounce extension
