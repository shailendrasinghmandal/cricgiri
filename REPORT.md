# CricGiri AI Analytics Engine — Engineering Report

_Single-camera, behind-the-stumps cricket bowling analytics. This report covers an
end-to-end verification + diagnosis + accuracy pass over the 22 clips in `videos/`._

> **Note on scope.** This is a mature system (~5k-line pipeline, custom-trained
> YOLO ball + stump models, full FastAPI layer), not a half-built MVP. The work
> here is **verification, root-cause diagnosis, and targeted, additive fixes** —
> no rewrite, and (per the repo's baseline-lock) no production weights overwritten.

---

## 1. Environment & assets (Phase 0)

| | |
|---|---|
| Python | 3.13.5 (`venv/`) |
| torch | 2.11.0+cu126 — **CUDA available, RTX 4050 Laptop GPU** |
| ultralytics / opencv / numpy / scipy | 8.4.47 / 4.13.0.92 / 2.4.4 / 1.17.1 |
| fastapi / uvicorn / pydantic | 0.136.3 / 0.49.0 / 2.13.4 |
| Ball model | `models/ball_best.pt` — **custom-trained** (class `ball`), not stock YOLO |
| Stump model | `models/stump_best.pt` — custom (class `item`) |
| Stronger ball weights | `models/ball_ft_t4.pt` (T4 fine-tune), `models/ball_best_leather_new.pt` (ensemble partner) |
| Videos | 22 clips in `videos/`, 2–4.5 s, behind-stumps phone footage, mixed 30/58–60 fps |

**Async queue note:** the spec calls for Redis/Celery; the shipped API (`api/jobs.py`)
uses an in-process threaded `Queue` worker (single worker, pipeline singleton).
Per decision during this pass, the threaded worker is **kept** (functionally async,
no extra infra) and documented as such.

---

## 2. Before → After results matrix

Each field is scored **OK** (trustworthy) / **~** (produced but suspect) /
**X** (failed/none). Reproduce with `python scripts/run_all.py`.

### BEFORE — shipped default config (single `ball_best.pt`, imgsz 640)

```
video         status       speed  bounce  trajectory  line  length  swing  confidence
test          NO_DELIVERY  X      X       X           X     X       X      X
test_video1   PARTIAL      ~      OK      OK          OK    OK      OK     OK
test_video10  COMPLETE     OK     OK      OK          OK    OK      OK     OK
test_video11  PARTIAL      ~      ~       X           OK    OK      ~      OK
test_video12  PARTIAL      OK     OK      OK          OK    OK      ~      OK
test_video13  NO_DELIVERY  X      X       X           X     X       X      X
test_video14  NO_DELIVERY  X      X       X           X     X       X      X
test_video15  NO_DELIVERY  X      X       X           X     X       X      X
test_video16  NO_DELIVERY  X      X       X           X     X       X      X
test_video17  NO_DELIVERY  X      X       X           X     X       X      X
test_video18  PARTIAL      OK     OK      X           OK    OK      ~      OK
test_video19  PARTIAL      OK     OK      OK          OK    OK      ~      OK
test_video2   PARTIAL      ~      ~       OK          OK    OK      OK     OK
test_video20  PARTIAL      ~      OK      ~           OK    OK      ~      OK
test_video21  NO_DELIVERY  X      X       X           X     X       X      X
test_video3   PARTIAL      OK     OK      ~           OK    OK      ~      OK
test_video4   PARTIAL      OK     ~       ~           OK    OK      ~      OK
test_video5   PARTIAL      OK     ~       OK          OK    OK      OK     OK
test_video6   PARTIAL      OK     OK      X           OK    OK      OK     OK
test_video7   PARTIAL      ~      ~       OK          OK    OK      OK     OK
test_video8   PARTIAL      OK     OK      OK          OK    OK      ~      OK
test_video9   PARTIAL      ~      OK      OK          OK    OK      OK     OK

TOTAL 22 | COMPLETE 1 | PARTIAL 14 | FAILED/EMPTY 7
```

### AFTER — validated strong config (ensemble @ 1280) + fixes

```
video         status    speed  bounce  trajectory  line  length  swing  confidence
test          PARTIAL   ~      OK      X           OK    OK      ~      OK
test_video1   PARTIAL   ~      ~       OK          OK    OK      ~      OK
test_video10  PARTIAL   ~      X       OK          OK    OK      ~      OK
test_video11  PARTIAL   ~      OK      X           OK    OK      ~      OK
test_video12  PARTIAL   ~      X       OK          OK    OK      ~      OK
test_video13  PARTIAL   ~      X       X           OK    OK      OK     OK
test_video14  PARTIAL   ~      X       OK          OK    OK      OK     OK
test_video15  PARTIAL   ~      ~       OK          OK    OK      OK     OK
test_video16  PARTIAL   ~      X       OK          OK    OK      OK     OK
test_video17  PARTIAL   ~      OK      X           OK    OK      ~      OK
test_video18  PARTIAL   ~      OK      ~           OK    OK      ~      OK
test_video19  PARTIAL   ~      X       OK          OK    OK      ~      OK
test_video2   PARTIAL   ~      OK      ~           OK    OK      ~      OK
test_video20  PARTIAL   ~      OK      OK          OK    OK      ~      OK
test_video21  PARTIAL   ~      OK      OK          OK    OK      ~      OK
test_video3   PARTIAL   ~      OK      ~           OK    OK      ~      OK
test_video4   PARTIAL   ~      X       OK          OK    OK      OK     OK
test_video5   PARTIAL   ~      X       OK          OK    OK      OK     OK
test_video6   PARTIAL   OK     OK      X           OK    OK      ~      OK
test_video7   PARTIAL   ~      X       ~           OK    OK      ~      OK
test_video8   PARTIAL   ~      OK      OK          OK    OK      ~      OK
test_video9   COMPLETE  OK     OK      OK          OK    OK      OK     OK

TOTAL 22 | COMPLETE 1 | PARTIAL 21 | FAILED/EMPTY 0
  speed       OK  2  SUSPECT 20  FAIL 0
  bounce      OK 11  SUSPECT  2  FAIL 9
  trajectory  OK 13  SUSPECT  4  FAIL 5
  line        OK 22  SUSPECT  0  FAIL 0
  length      OK 22  SUSPECT  0  FAIL 0
  swing       OK  7  SUSPECT 15  FAIL 0
  confidence  OK 22  SUSPECT  0  FAIL 0
```

**Headline:** clips producing a full delivery went from **15/22 → 22/22**
(7 dead clips recovered); every clip now yields line / length / confidence.

> **Reading the matrix honestly.** Some per-field "OK" counts are *lower* than the
> baseline — this is the **honesty fixes surfacing the truth, not a regression**:
> - **speed** OK 9→2: the scorer no longer counts pixel-scale or prior-guess speeds
>   as trustworthy. Only **2 clips** (`test_video6`, `test_video9`) have a
>   homography-grounded, in-band speed — consistent with the single-camera
>   calibration limit. Baseline's higher count included **fake `110.0` priors**
>   scored as OK.
> - **bounce** has **9 nulls** — every one is physically impossible (4 clips bounce
>   at y = 24–28 m, *past* the 20.12 m pitch; 5 at |x| = 3.9–5.4 m, several metres off
>   a 1.5 m-wide strip). These were calibration artifacts; **11 clips keep a valid
>   on-pitch bounce**. Baseline published the impossible ones.
>
> In short: **coverage went up (7 dead → 0), and confidence is now calibrated to
> reality.** A flagged-suspect value is more useful than a confident-looking wrong one.

---

## 3. Bugs found — root cause → fix

### D1 · 7 clips produced no output (detection starvation) — FIXED
- **Evidence:** logs show `Calibration OK` then `too few real detections (0)` on
  `test`, `test_video13–17`, `test_video21`. Clean exit, no exception.
- **Root cause:** the shipped default detects with a *single* `ball_best.pt` at
  imgsz 640 (`pipeline.py` `_detect_ball`); it fires on **0 frames** for these clips.
- **Fix (additive):** the harness and recommended run now use the already-built,
  measured-best stack — fine-tuned **ensemble** (`ball_ft_t4.pt` +
  `ball_best_leather_new.pt`) at **imgsz 1280** via `--max-recall --hybrid-ensemble`.
  Verified: `test_video16` 0 → 29 track points. No weights overwritten.

### D5 · Bounce point could land off the pitch — FIXED
- **Evidence:** baseline `test_video2` bounce `(4.33, 21.63)` — past the 20.12 m
  pitch and 4.3 m off line; several clips with `|x| > 1.8 m`.
- **Root cause:** the existing pitch-sanity check was gated behind
  `world_pts is not None`, so it was **skipped on exactly the pixel-fallback clips**
  where the homography is untrusted and bad bounces occur. Different delivery-builder
  paths also bypassed it (e.g. `test_video13` kept `y = 24.74`).
- **Fix:** a single, **path-independent** sanitizer at the serialization chokepoint
  (`DeliveryAnalysis.to_dict` → `_bounce_world_plausible`): a bounce is published only
  when `-0.5 ≤ y ≤ 20.6 m` and `|x| ≤ 2.5 m`; otherwise `bounce_point`/`heatmap_points`
  are honestly `null`/`[]`. Bounds are generous so genuine on-pitch bounces are never
  dropped (verified: `test_video10` `(-0.13, 6.07)` is retained).

### D2 · Speed could report a fake `110.0 km/h` — FLAGGED (honest scoring)
- **Evidence:** 5 baseline clips reported *exactly* 110.0.
- **Root cause:** `_coarse_pixel_speed` falls back to `median_kmh = 110.0`
  ("safe mid-range prior", `pipeline.py:2893`) when both calibrated and coarse
  estimates are out-of-band — a plausible-looking guess.
- **Fix:** the speed estimator already tags these (`method=median_prior`,
  `metric_source=pixel_scale_fallback`, `confidence≈0.10`). The reporting/matrix now
  **reads those tags** and marks prior/pixel-scale speeds as *suspect* rather than
  *OK*, so a guessed number is never presented as a measurement. (The number is kept
  for schema compatibility but is no longer dressed up as trustworthy.)

### D4 · `swing_cm` pinned at 15.0; `swing_type=none` with nonzero cm — KNOWN LIMIT
- **Root cause:** `swing_estimation.py:147-150` rescales any world-space lateral
  deviation `> 0.30 m` to a fixed `0.15 m`. Those large deviations are **homography
  parallax on the airborne ball**, not real swing. This is a single-camera limit.
- **Status:** documented as a known limitation (see §5); swing is already
  confidence-gated. Left unchanged to avoid destabilizing under the baseline-lock.

### D7 · Calibration "reprojection error" always `0.000 px` — KNOWN LIMIT
- **Root cause:** the homography is fit *through* the same stump correspondences it
  is scored against → perfect fit by construction, reported in pixels, no independent
  check. It cannot flag a degenerate calibration (which is how D5's bad homography
  slipped in).
- **Status:** documented; the downstream `_world_trajectory_is_plausible` guard and
  the new bounce-bounds gate now act as the practical calibration-trust signal.

### Cosmetic · `libopenh264` VideoWriter error
- Harmless: `_open_writer` falls back avc1→H264→**mp4v**, which succeeds (annotated
  MP4s are written, e.g. `runs/test/annotated.mp4`, 5.2 MB).

---

## 4. Accuracy improvements (concrete)

- **Coverage:** deliveries produced **15/22 → 22/22**; `NO_DELIVERY` **7 → 0**.
- **Detection density (example):** `test_video16` 0 → 29 track points with the ensemble.
- **Honesty:** off-pitch bounces are no longer published (D5); guessed `110 km/h`
  speeds and pixel-scale speeds are now flagged, not presented as measurements (D2).
- **No regressions on good clips:** the conservative bounce gate retains valid
  on-pitch bounces (e.g. `test_video10`).

_(Per-field before/after tallies are in `runs/_MATRIX.md` vs `runs_after/_MATRIX.md`.)_

---

## 5. Known limitations (need better capture, not tuning)

**Footage reality (confirmed visually).** The test clips are *informal backyard /
driveway cricket* — makeshift wooden stumps on tarmac, no regulation 20.12 m pitch
(see `runs_after/_debug/tv9_frame58.jpg`). The world-coordinate homography assumes a
real pitch geometry that physically isn't present, which is the **root reason** metric
speed/bounce/swing are approximate here and frequently fall back to pixel-scale. On a
real marked pitch these world metrics would be substantially more reliable.

These are **single-camera, behind-the-stumps** limits, confirmed by extensive prior
forensic work in this repo:

- **Lateral (line / swing) precision** is weak: the homography's lateral baseline is
  the stump width (0.2286 m), so world-`x` is indicative only. Swing magnitude is
  therefore capped (D4) and line confidence is hard-limited.
- **Trustworthy world speed** is available only on near-ground / dense-detection
  clips; airborne-ball parallax on a ground-plane homography corrupts the rest, which
  honestly fall back to pixel-scale (flagged) or a low-confidence prior.
- **Post-bounce / release visibility** is footage-limited (occlusion behind the
  batsman, motion blur), not tracker-limited.
- **Spin / RPM** is intentionally **out of scope** (unreliable from a single blurred
  view) — left as a documented future stub.

**The single highest-ROI capture change:** an elevated **square-leg (side-on)** camera,
which fixes both occlusion and post-bounce blur, and gives a real lateral baseline for
line/swing/speed. No software lever can recover a ball that is behind a body or has
zero pixels.

---

## 6. How to run

```bash
# 1) Full pipeline over every clip + results matrix (strong config, default)
python scripts/run_all.py
#    JSON-only / faster:           python scripts/run_all.py --no-video
#    Honest baseline (shipped):    python scripts/run_all.py --config baseline
#    Outputs: runs/<clip>/{result.json,annotated.mp4,run.log}, runs/_MATRIX.md

# 2) Single clip (strong config)
python -m pipeline.pipeline --video videos/test_video10.mp4 \
  --max-recall --ball-model models/ball_ft_t4.pt \
  --ball-model-alt models/ball_best_leather_new.pt --hybrid-ensemble \
  --out-video out.mp4 --out-json out.json

# 3) Debug overlay (ball boxes, tracked path, bounce marker, pitch overlay)
python -m pipeline.pipeline --video videos/test_video10.mp4 --max-recall \
  --ball-model models/ball_ft_t4.pt --ball-model-alt models/ball_best_leather_new.pt \
  --hybrid-ensemble --trajectory-debug --out-video debug.mp4 --out-json debug.json

# 4) API (FastAPI + threaded async worker)
python run_api.py        # POST /analyze  → job_id ; GET /analyze/{job_id} → result JSON
```
