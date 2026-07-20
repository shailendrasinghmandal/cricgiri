# CricGiri Engine — Change Log & Handover

Everything changed in the delivery engine, why, and what your team needs to do.
All changes are in the offline/ensemble engine used by the API:
`scripts/run_demo_testing.py` (plus two helpers).

---

## TL;DR for the team

1. **Pull `main`** from `shailendrasinghmandal/cricgiri`.
2. **Four engine modules were previously missing from git** — a fresh clone could not run
   the engine at all. They're now committed (§5). This alone may explain earlier failures.
3. **Model weights are NOT in this repo.** Get them from
   `INFINOIDTECHNOLOGIES/Cricgiri_Codebase` (Git LFS — run `git lfs install && git lfs pull`).
4. **Re-generate any saved JSONs.** Old JSON files still contain the pre-fix trajectories.
   Re-run the videos through the engine to get the corrected output.
5. **Frontend: delete any correction code.** The JSON is now directly renderable —
   see `FRONTEND_RENDERING_GUIDE.md`.

---

## 1. Rendering fixes — the JSON is now directly renderable

Previously the client had to repair the JSON before drawing it. Symptoms everyone hit:
zig-zag paths, the ball crossing the pitch centre line, and bounces at the bowler's feet.
Fixed at the source.

| # | Was | Now | Fixes |
|---|---|---|---|
| 1.1 | Height sampled at an **integer frame** — several densely-sampled points shared one frame and got the same height | Height is continuous in flight **phase** | staircase → **zig-zag** |
| 1.2 | Height fell on a straight ramp `h·(1−t)` | Gravity curve **`h·(1−t²)`** — holds height after release, then steepens into the bounce | ball "sliding" into the bounce instead of arcing |
| 1.3 | Lateral centred on the **ball's own mean**, so the true side was lost | Lateral anchored to the side from the **`line` label** (leg=+x, off=−x); drift clamped so it cannot cross the centre | ball on the **wrong side** / crossing the centre line; two clips landing on the same side |
| 1.4 | Bounce position derived **after** the curve, so they could disagree | Bounce **phase validated first**; if it implies a bounce at the release end it falls back to the position implied by the **`length` label** | **bounce at the bowler's feet** |
| 1.5 | `bounce_world` came from the bounce frame | `bounce_world` = the arc's **actual lowest point** (forced to z=0) | marker not sitting under the drawn curve |

**Guarantees now enforced:** exactly one ground contact; `1.5 < bounce y_m < pitch length`;
ball stays on one side; points evenly spaced ~0.7 m (no straight chord across a gap).

## 2. Detection / recovery fixes

Clips that previously returned **0 deliveries** but did contain a ball.

**2.1 Physics gate direction bug** — `scripts/physics_gate_v2.py`
The gate decided the ball's travel direction with `sign(median(diff(x)))`. When the
detector re-reports a ball at rest it jitters by hundredths of a pixel, and those
near-zero steps **outvoted the real strides**, so the gate concluded the ball travelled
backwards, flagged every genuine step as a reversal, and deleted the whole flight —
keeping only the jitter. Now only steps with real motion (>1px) vote on direction.
→ recovered clips that had a clean 18-frame arc at 0.80 confidence being reported as "static".

**2.2 Static-cluster gate was resolution-blind** — `run_demo_testing.py`
A fixed 45–60 px threshold, tuned on 720px-wide footage, wrongly rejected real short
flights in 478px portrait clips (a 32px arc is 6.8% of that frame, not "static").
Now scales with frame width (5%, min 24px). Reject-only gate, so it cannot regress a
clip that already passed.

**2.3 RANSAC fallback for compact tracks** — `run_demo_testing.py` + `map_trajectory.py`
`min_spread_px=120`, `min_step_px=9` and the 60px seed-motion gate are absolute pixels;
on small clips they discard a genuine slow/compact flight. A **fallback pass with
resolution-scaled gates** now runs *only when the strict pass yields <4 points*, so
working clips are untouched by construction. `ransac_trajectory()` gained a
`min_seed_motion_px` parameter for this.

**2.4 Gap densification** — trajectories are resampled evenly along the pitch, so a
mid-flight detection gap no longer renders as one long straight chord ("cut").

## 3. Output schema changes

**3.1 One confidence only.** Removed `track.average_confidence`, `line.confidence`,
`length.confidence`, `speed.confidence`, `swing_confidence`, and the top-level
`detection_conf_threshold` (+ its note). The response now exposes **only**
`confidence_score`, `confidence_pct`, `confidence_label` (plus `physically_valid`).

**3.2 Pitch length now affects speed.** The pitch-length input was inert. Speed is
`distance ÷ time` and pitch length sets the distance scale, so the published speed now
scales with it (a shorter pitch → proportionally lower km/h). Default 20.12 m is
unchanged. Only the arc-*estimated* speed is scaled — the homography-measured speed
already carries pitch length, so it is not double-counted.

**3.3 Length is fixed to the video.** Length is classified against the standard 20.12 m
pitch, so changing the pitch-length setting **no longer flips the category** — a yorker
stays a yorker, a good length stays good. Only speed responds to pitch.

## 4. Rejected — do not re-implement

**Static-FP / watermark filter.** Dropping detections that fire at the same pixel across
many frames (to kill the "Filmed on FULLTRACK AI" logo) was implemented and **reverted**:
on these clips it stripped 50–70% of detections and broke previously-good clips (15, 20,
27 lost their bounce). A comment marks this in the source. Don't re-add it without a
per-clip A/B.

## 5. Files

| File | Change |
|---|---|
| `scripts/run_demo_testing.py` | all engine fixes above |
| `scripts/map_trajectory.py` | `min_seed_motion_px` parameter |
| `scripts/physics_gate_v2.py` | direction-vote fix **(was untracked)** |
| `scripts/calibrated_speed.py` | **was untracked** |
| `scripts/analytics_v2.py` | **was untracked** |
| `scripts/delivery_reconstruction.py` | **was untracked** |
| `api/delivery_api.py` | local API; pitch-length input wired through |
| `FRONTEND_RENDERING_GUIDE.md` | new — how to render the JSON |
| `ENGINE_CHANGES.md` | this file |

The four "was untracked" modules are imported by `run_demo_testing.py`. Before this
commit they existed only on the dev machine, so **any clone of this repo had a
non-functional engine**.

## 6. Running it

```bash
# API (upload a video, download JSON):
python -m uvicorn api.delivery_api:app --host 0.0.0.0 --port 8000
#   UI:     http://localhost:8000/ui
#   REST:   POST /analyze   (form: video=@clip.mp4, pitch_length=20.12)

# One video in code:
import importlib.util
spec = importlib.util.spec_from_file_location("engine", "scripts/run_demo_testing.py")
engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
result = engine.analyze_video("clip.mp4", pitch_length=20.12)
```

Ensemble: `ball_ft_t4.pt` + `ball_best_leather_new.pt`, stumps via `stump_best.pt`,
conf 0.05, imgsz 1280. GPU auto-detected.

## 7. Known limitations (honest status)

- **`length` can be misclassified.** A yorker has been reported as `good_length`/
  `short_length` when the bounce is detected late/poorly. The rendered bounce always
  agrees with the reported label, but the label itself can be wrong versus the video.
  This is bounce-detection accuracy — not a rendering bug.
- **Speed is an estimate, not a measurement.** On this footage the homography
  time-of-flight usually fails, so the published speed is a plausible arc-based estimate
  (`status: "estimated"`). True per-ball speed needs side-on footage with visible stumps.
- **Lateral is indicative.** Sideways calibration comes from stump *width* (0.2286 m), a
  very short baseline, so the ball's *side* is reliable but the exact metres are not.
- **Some clips genuinely have no ball flight** (occlusion, motion blur, ball not in
  frame). Those return `total_deliveries: 0` — that is correct behaviour, not a failure.
