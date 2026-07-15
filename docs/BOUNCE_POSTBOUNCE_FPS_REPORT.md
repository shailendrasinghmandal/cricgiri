# Bounce Precision · Post-Bounce Tracking · High-FPS Evaluation

**Evidence-first report. Every number below is measured from on-disk detections,
the mapped arc, and the source video — nothing is assumed or retrained.**

Reproduce:

```bash
python scripts/bounce_postbounce_diagnostics.py   # bounce + post-bounce + FPS CSVs
python scripts/build_high_fps_test.py             # dataset/high_fps_test cohort
python scripts/render_diagnostic_visuals.py       # visual examples
```

Deliverables produced:

| File | What |
|---|---|
| `outputs/bounce_analysis.csv` | per-clip bounce frame, coords, supporting detections, confidence |
| `outputs/bounce_failure_analysis.csv` | per-clip measurement of the 7 candidate failure causes |
| `outputs/post_bounce_analysis.csv` | per-clip raw / recovered / mapped post-bounce + A/B/C |
| `outputs/fps_comparison.csv` | 30fps vs 58–60fps, same metrics |
| `dataset/high_fps_test/` | high-FPS benchmark cohort + manifest |
| `outputs/diagnostics_visuals/_MONTAGE.jpg` | annotated visual examples |

Cohort: 22 deliveries — 15 at ~30fps, 7 at 58–60fps. All are amateur phone /
net-practice clips at ~720×1280 (one 464×832).

---

## TL;DR — the evidence-based answers

1. **How accurate is bounce detection?** A lowest-point can be placed on
   **22/22** clips, but it is **backed by a real detected "V" + ≥2 ball
   detections on only 3/22 (14%)**. The rest are geometric guesses on a flat arc.

2. **Why does post-bounce tracking fail?** It splits three ways:
   **32% recoverable** (the ball *was* detected after the bounce and the mapper
   dropped it), **27% genuinely not detected**, **27% only static clutter** in
   the post-bounce window. So roughly a third is a free mapper fix, a third is a
   true detector miss.

3. **How much does FPS affect quality?** On this footage, **almost none.**
   58–60fps gives **2.15× more detections per second** but the **usable ball
   detections per delivery stay flat (13.0 vs 12.7)** and bounce/post quality
   does not improve. More frames of the same blurry, low-res ball ≠ more usable
   ball positions.

4. **Can 60/120fps reach FullTrack quality with the current model?** Not from
   frame rate alone. The limiter is **per-frame detectability** (motion-blurred,
   small, low-contrast ball), not how often we sample it. Higher FPS only helps
   if each frame is *also* sharper/higher-res (i.e. better footage **and** a
   detector that survives blur).

5. **What is the current ceiling?** With this footage + model, a fully
   detection-supported delivery (clear release, a real bounce V, and a detected
   rise) happens on about **3/22 (~14%)** of clips. That is the realistic ceiling
   today — the remaining clips don't contain enough *usable* ball pixels for a
   professional-grade arc.

---

## PART 1 — Bounce precision

`outputs/bounce_analysis.csv`. The bounce is the lowest on-screen point of the
real ball path. "Supported" requires a measurable vertical turn (`mag ≥ 6 px`)
**and** ≥2 real ball detections within ±3 frames / 60 px.

| Metric | Value |
|---|---|
| Bounce geometrically located | 22 / 22 |
| **Bounce detection-supported** | **3 / 22 (14%)** |
| Supported clips | `test_video1`, `test_video5`, `test_video17` |

Only `test_video5` reaches full confidence (`mag=40, support=5, conf=1.0`).
`test_video1` (`conf=0.51`) and `test_video17` (`conf=0.30`) are partial.
On the other 19 clips the "bounce" is placed at the arc's lowest sample but the
arc shows **no measurable downward-then-upward turn** — there is nothing there to
trust.

### Why bounce detection is weak — measured, not guessed

`outputs/bounce_failure_analysis.csv`. Each of the seven candidate causes was
measured per clip. Primary cause tally:

| Primary cause | Clips |
|---|---|
| `flat_arc_no_vertex` (dets exist near the bottom, but no real V) | 14 |
| `motion_blur` (ball box at bounce ≫ blurrier than mid-flight) | 4 |
| `ok(supported)` | 2 |
| `too_late` (arc bounce frame after the raw lowest detection) | 1 |
| `sparse_dets` (<2 real detections around the bounce) | 1 |

Per-cause hit counts (a clip can trip several):

| Candidate cause | Clips tripped |
|---|---|
| Sparse detections around bounce | 1 |
| Detector loses ball near ground | 0 |
| Motion blur at bounce | 5 |
| Ground-colour similarity (vs same ball mid-flight) | **0** |
| Bounce estimated too early / late | early 0 · late 1 |
| Spline smoothing hiding the bounce | **0** |

**What this rules OUT (important):**

- **Spline smoothing is NOT hiding the bounce (0 clips).** The raw detections do
  not reach materially lower than the smoothed arc — there is no V being rounded
  off. The bounce is missing from the *data*, not the smoothing.
- **Ground-colour similarity is NOT a bounce-specific cause (0 clips).** Measured
  *relative to the same ball mid-flight*, the ball is no harder to separate from
  the background at the bounce than elsewhere. (An earlier absolute metric flagged
  every clip — that was a thresholding artifact and was corrected to a ratio.)
- **Near-ground detection loss is not the driver either (0 clips):** when the arc
  reaches the bottom band, coverage there is not selectively worse.

**What it points TO:** the bounce fails because the **real ball arc is too short
/ too flat to contain a bounce vertex** (14 clips) and, secondarily, because the
**ball is motion-blurred** around the bounce (4–5 clips). Both are *detection-
density / footage* problems, not tracking or smoothing problems.

---

## PART 2 — Post-bounce tracking

`outputs/post_bounce_analysis.csv`. For each clip we take the post-bounce window
(≤12 frames after the pivot), count raw detections there, and test whether they
form a **moving** ball (recoverable) or are static clutter.

| Class | Meaning | Clips |
|---|---|---|
| **A — visible but filtered** | detector saw a *moving* post-bounce ball; mapper dropped it → **free fix** | **7 (32%)** |
| **B — not detected** | detector fired ~nothing post-bounce → needs detection work | 6 (27%) |
| **C — static / clutter** | detections exist but don't move like a ball | 6 (27%) |
| ambiguous | 1–2 weak detections | 3 (14%) |

- **A (recoverable):** `test_video2, 5, 6, 9, 10, 11, 17`
- **B (true miss):** `test_video4, 8, 15, 18, 20, 21`

**Before / after recovery (post-bounce frames across the cohort):**

| | Post-bounce ball frames |
|---|---|
| Mapped arc only (before) | 46 |
| + recovered moving detections (after) | **98** (+52) |
| Mean post-bounce coverage after recovery | **0.295** |

So the existing recovery roughly **doubles** the post-bounce evidence we keep —
but mean coverage is still **~0.30**, i.e. most deliveries remain incomplete
after the bounce. The 32% in class **A** can be improved purely in the mapper
(no detector change). The 27% in class **B** cannot — the ball is simply not in
the detections.

> Decision rule the data supports: **first harvest class A** (mapper/association
> fix, zero detector cost). Only class B justifies detector work, and only on the
> kind of footage where the ball is actually present in the pixels.

---

## PART 3 — High-FPS evaluation (the critical question)

`outputs/fps_comparison.csv`. Same metrics, two cohorts.

| Group | n | dets/sec | **real ball dets / delivery** | bounce-supported | post-recoverable (A) | post coverage |
|---|---|---|---|---|---|---|
| ~30 fps | 15 | 27.2 | **12.7** | 20% | 27% | 0.31 |
| 58–60 fps | 7 | **58.4** | **13.0** | 0% | 43% | 0.27 |

**Reading it honestly:**

- Frame rate does exactly what it should to raw sampling: **dets/sec scales 2.15×**
  with FPS (27→58).
- But the quantity that actually builds the trajectory — **usable ball detections
  per delivery — is flat (13.0 vs 12.7).** Higher FPS sliced the same blurry
  flight into more frames; each extra frame did not yield an extra *usable* ball
  position.
- Bounce support and post-bounce coverage do **not** improve with FPS (bounce
  support is 0% in the hi-FPS group, though that subset is tiny — see caveat).

**Conclusion:** On consumer phone footage, **FPS is not the bottleneck.**
The bottleneck is **per-frame detectability** — a small, fast, motion-blurred ball
on a ~720p handheld frame. This is consistent with Part 1 (flat arc + blur) and
Part 2 (class B true misses).

### Caveats (stated plainly)

- The "high-FPS" cohort is **still amateur phone footage** (720×1280, handheld),
  just at 60fps. It proves *"more FPS on the same quality footage doesn't help"*.
  It does **not** prove that **professional fixed-camera, high-resolution 120fps**
  footage wouldn't help — that is the untested upper half of the hypothesis.
- Only 3 clips are fully bounce-supported, so the per-group bounce-support % is
  small-sample and noisy. The robust signals are dets/sec (clean 2.15×) and the
  flat per-delivery inlier count.

### Why we did not auto-download YouTube clips

Building a *valid* high-FPS benchmark needs clips that (a) are genuinely high-FPS
after YouTube re-encode (often dropped to 30/60), and (b) show the **full ball
flight** (release→bounce→batter) from a fixed side-on/behind camera. The common
"side-on slow-motion bowling action" clips are **biomechanics shots that crop at
release and never show the ball's flight or bounce** — useless for this pipeline.
Rather than guess at copyrighted URLs and pass unsuitable footage off as a
benchmark, we:

1. built the cohort from our **7 verified, real 58–60fps deliveries** that *do*
   contain full ball flight (`dataset/high_fps_test/`), and
2. shipped `scripts/fetch_high_fps.py` (URL-driven, verifies real FPS after
   download) + `scripts/run_high_fps_pipeline.py` so any vetted clip you trust
   can be dropped in and measured the same way.

---

## Visual examples

`outputs/diagnostics_visuals/_MONTAGE.jpg`
(red = raw detections · green = on-arc ball · cyan = recovered post-bounce · white X = bounce)

- `test_video5` (30fps) — supported bounce + recovered rise (the good case).
- `test_video17` (30fps) — strong post-bounce recovery (cov 2.0), bounce slightly late.
- `test_video2` (60fps) — a full red arc of raw detections the mapper dropped:
  textbook **class A "visible but filtered"**.
- `test_video13` / `test_video20` — sparse / motion-blur clips: flat arc, no real
  bounce, little or nothing post-bounce (**class B**).

---

## Honest conclusions & highest-impact path

**The primary bottleneck is footage/detection, in this order:**

1. **Per-frame ball detectability** (motion blur + small, low-contrast ball on
   ~720p). This caps everything upstream of tracking. *(Evidence: flat arcs on
   19/22, blur on 4–5, FPS doubles dets/sec but not usable dets/delivery.)*
2. **Mapper over-filtering of real post-bounce detections** — 32% of clips have a
   recoverable moving ball that was discarded. *(Free win, no detector change.)*
3. **Frame rate** is **not** a material lever on current footage quality.

**What is NOT the problem (measured):** spline smoothing (0), ground colour (0),
near-ground selective loss (0), and — per the brief — the trackers/Kalman/MHT/
optical-flow/release-corridor are not implicated by any of these measurements.

**Recommended order of work (highest impact first):**

1. **Harvest class-A post-bounce detections in the mapper/association** (recover
   the 7 clips already shown recoverable; lift mean post-coverage above ~0.30).
   Zero detector cost.
2. **Improve per-frame detection of the blurred/near-bounce ball** — the only
   thing that raises the 14% supported-bounce rate. Best done with *better
   footage first* (higher shutter speed to kill motion blur, higher resolution),
   then, if needed, blur-augmented detector fine-tuning targeted at class-B clips.
3. **Only then** re-test FPS — and only on footage that is *also* sharp and
   high-res. On equal-quality footage, FPS gave us nothing.

**Bottom line:** Higher FPS will not, by itself, produce FullTrack-style
trajectories with the current footage and model. The decisive levers are (a)
recovering the post-bounce detections we already have, and (b) increasing how
many frames contain a *usable* ball — which is a footage-quality and detection
problem, not a frame-rate or tracking problem.
