# Pre-Detection Ball Enhancement — Evidence Report

**Question asked:** before training the detector again, can we make the ball
*easier to see* (super-resolution / deblur / contrast / edge) **inside dynamic
ROIs** and thereby recover the detections the current pipeline misses —
especially the motion-blurred, tiny, low-contrast balls near and after the bounce?

**Short answer (measured, not assumed):**

> Enhancement is **not** the highest-impact path with the current (frozen)
> detector. Across 231 frames the current pipeline missed, the best enhancement
> (FSRCNN ×4) recovered **16 / 231 (6.9%)** — only **+2 frames** over simply
> re-running the detector on a *zoomed raw ROI* (14 / 231, 6.1%). Heavy deblur
> (Wiener, Richardson–Lucy, deblur combos) **hurt** recovery. The visual
> examples show that at most missed frames **there is no ball to recover at the
> expected location** — it is occluded, out of the interpolated path, or
> degraded past the point pixel-enhancement can fix. The one robust, cheap win
> is the **dual-pass ROI re-detector** (zoom + light upscale) for the
> **post-bounce** region.

Everything below is reproducible from scripts in `scripts/` and the CSVs in `outputs/`.

---

## How this was measured

| Pass | What | Source |
|------|------|--------|
| **Baseline = current pipeline** | production ensemble `ball_ft_t4 + ball_best_leather_new` @ imgsz1280 | cached `outputs/detections/<clip>/detections.csv` |
| **Target frames** | frames where the ball is *expected* (on the mapped arc, plus a 12-frame post-bounce window) but the current pipeline produced **no** detection within `VALID` px | derived |
| **Enhanced pass** | same ensemble re-run on a **dynamic ROI** around the expected location, after each enhancement | `scripts/ball_enhancement_benchmark.py` |
| **Recovery** | a detection inside the ROI, within `VALID` px of the expected ball location (validated against the trajectory, so false positives are not counted) | derived |

A `raw` ROI pass (crop + re-detect, **no** enhancement) isolates the *zoom*
effect from the *enhancement* effect. No detector was retrained — this answers
"do the missing balls already exist visually?" first, exactly as requested.

19 clips produced target frames (2 clips had the ball fully covered already, 1
errored on cache); **231 missed frames** were tested against **14 enhancement
methods**.

---

## Stage 1 — Visual analysis of failed detections

`outputs/ball_visibility_report.csv` (283 frame rows). Measured on the **real
ball box** for detected frames, and at the **expected location** for missing
frames (the ball itself can't be measured when it isn't found).

**A) Detected balls, by region (measured on the actual box):**

| region | n | ball px | blur (Lap var) | contrast (std) | ball-vs-bg | edge |
|--------|---|---------|----------------|----------------|-----------|------|
| release | 86 | 51.5 | 521 | 37.8 | **15.8** | 74.5 |
| mid | 10 | 15.1 | 455 | 28.2 | 12.4 | 67.3 |
| bounce | 73 | 32.6 | 591 | 34.9 | 15.1 | 79.0 |
| post | 85 | 81.3 | 599 | 42.1 | 12.8 | 81.7 |

**C) Missing frames — ball-vs-bg contrast at the expected location (lower = blends in):**

| region | nMiss | ball-vs-bg | blur@loc | brightness |
|--------|-------|-----------|----------|------------|
| release | 4 | **53.2** | 2838 | 137 |
| mid | 2 | 20.6 | 2786 | 157 |
| bounce | 12 | **17.7** | 799 | 135 |
| post | 11 | **11.0** | 363 | 96 |

**Reading it honestly:**
- When the ball *is* detected, it is reasonably visible everywhere — blur and
  edge sharpness are roughly constant across regions. So "the detected balls are
  blurry" is **not** what is happening.
- The decisive, trustworthy signal is **ball-vs-background contrast collapsing
  from release → bounce → post (≈53 → 18 → 11)**. Missed frames cluster exactly
  where the ball blends into the pitch/outfield.
- We **cannot** directly measure a missing ball's blur (it isn't localizable),
  so the "motion blur" hypothesis is supported only indirectly (from the earlier
  `bounce_failure_analysis` blur-ratio), not by this report.

---

## Stage 2 — Enhancement detection benchmark

`outputs/enhancement_benchmark.csv`. 231 missed frames, recoveries validated
against the trajectory. `dBlur/dEdge/dBg` are the change in ball-region visual
quality vs the raw ROI (for super-res, `dBlur` drops are a per-pixel artifact of
interpolation, not real blurring — the real SR gain is the ×2/×4 size increase).

| method | family | recovered / 231 | rate | rel | bnc | post | ms/ROI | dEdge | dBg (contrast) |
|--------|--------|-----------------|------|-----|-----|------|--------|-------|------|
| **fsrcnn4x** | super-res | **16** | 6.9% | 1 | 2 | 12 | 539 | −40 | +0.08 |
| **raw (zoom only)** | baseline | **14** | 6.1% | 1 | 2 | 11 | 500 | 0 | 0 |
| espcn2x | super-res | 14 | 6.1% | 1 | 2 | 11 | 524 | −24 | +0.01 |
| gamma | contrast | 14 | 6.1% | 1 | 2 | 11 | 501 | −10 | −1.4 |
| cubic2x | super-res | 12 | 5.2% | 1 | 2 | 9 | 492 | −23 | +0.04 |
| fsrcnn2x | super-res | 12 | 5.2% | 1 | 2 | 9 | 522 | −24 | +0.05 |
| clahe | contrast | 11 | 4.8% | 1 | 1 | 9 | 510 | +35 | +1.3 |
| unsharp | deblur | 10 | 4.3% | 2 | 2 | 6 | 503 | +26 | +0.08 |
| wiener | deblur | 9 | 3.9% | 0 | 0 | 8 | 604 | +146 | −0.5 |
| local_contrast | contrast | 8 | 3.5% | 1 | 1 | 6 | 508 | +34 | +3.1 |
| edge | edge | 7 | 3.0% | 1 | 0 | 6 | 506 | +22 | +0.7 |
| combo_full | combo | 7 | 3.0% | 0 | 1 | 6 | 528 | +16 | +1.7 |
| richardson_lucy | deblur | 5 | 2.2% | 0 | 1 | 4 | 528 | +24 | +0.05 |
| combo_deblur | combo | 0 | 0.0% | 0 | 0 | 0 | 787 | +126 | +0.5 |

**Per-family verdict:**
- **Super-resolution** — the only family that *matched or slightly beat* zoom-only.
  FSRCNN ×4 = best at **+2 frames** over raw ROI. OpenCV DNN SR (FSRCNN/ESPCN) ≈
  cubic — the learned models give no meaningful edge over bicubic here. Real-ESRGAN
  was **not** run (heavy torch/weights, ~seconds/ROI); given that even ×4 DNN SR
  added only 2 frames, it is not worth the cost.
- **Contrast** — CLAHE/gamma/local-contrast genuinely raise ball-vs-bg contrast
  (`dBg` up to +3.1) but did **not** translate into more detections; several
  landed below zoom-only.
- **Deblur** — Wiener and Richardson–Lucy **reduced** recovery (9 and 5 vs 14)
  and cost more. The deblur combo recovered **0**. Deconvolution amplifies pitch
  texture into ball-like artifacts that the validator (and detector) reject.
- **Edge** — no help.

**Detections gained → coverage (`outputs/detection_gains.csv`, best method FSRCNN ×4):**

- mean arc coverage **0.454 → 0.489 (+0.035)** — a ~3.5-point bump.
- almost entirely in the **post-bounce** region (best method recovered 12 of 16
  in post).
- The standout clip is `test_video2` (60 fps): post-bounce 16 → 20, coverage
  0.727 → **0.818**. The recoverable frames concentrate where the ball is
  *actually present* and merely small — i.e. exactly the dual-pass ROI use case.

---

## Visual examples

`outputs/enhancement_examples/_MONTAGE.jpg` — for each missed frame: `raw |
cubic2x | fsrcnn2x | clahe | unsharp | combo_full`.

![enhancement examples](../outputs/enhancement_examples/_MONTAGE.jpg)

This montage is the most important evidence: at the majority of missed frames the
ROI (centered on the *expected* ball location) shows **grass, nets, sandy pitch,
or a motion-blurred batsman/stumps with no separable ball**. Enhancement makes
the texture crisper but does not reveal a ball that isn't there. The few good
cases (e.g. `test_video11 f43`, a faint dark blob CLAHE sharpens) are exactly the
6% the dual-pass recovers.

---

## Runtime costs

On **CPU**, every method is ~**0.5 s per ROI** (Wiener 0.6 s, deblur-combo 0.8 s).
A delivery has ~10–20 missed frames, so a dual-pass adds ~5–10 s/clip on CPU
(far less on GPU, and only on the missed frames). Heavy deblur doubles cost for
**negative** benefit and should not be used.

---

## Answers to the success criteria

1. **Do the missing detections already exist visually but are hidden by blur/low
   contrast?** Mostly **no**. The dominant cause of a missed frame is that the
   ball is genuinely absent/occluded/degraded at the expected location, not that
   a clear ball is sitting there waiting for a contrast boost. Where it *does*
   exist (~6%), zoom + light SR recovers it.
2. **Does enhancement beat training as the primary pipeline?** **No.** The
   success bar was "enhancement produces more detections than training would."
   Enhancement produced **+2 frames over a zoom-only re-detect** and **+0.035
   coverage** total. That does not clear the bar, so enhancement does **not**
   become the new primary pipeline.
3. **What actually helps?** A **lightweight dual-pass ROI re-detector** (crop the
   predicted/bounce/post-bounce region, optional ×2–×4 upscale, re-run the
   detector). It recovers ~6% of missed frames cheaply, almost all post-bounce —
   the original weakness — with no retraining.
4. **The real ceiling:** the frozen detector is also penalised because enhanced
   ROIs are *out-of-distribution* (it was trained on native frames). Contrast and
   edge metrics improve, but the model can't capitalise. To convert recovered
   *visual information* into detections you would have to **train on
   upscaled/enhanced ball crops** — i.e. enhancement only pays off *with* a
   targeted training cycle, not instead of one.

---

## Recommended highest-impact path (evidence-ordered)

1. **Ship the dual-pass ROI re-detector for post-bounce only** (optional FSRCNN
   ×2–×4 or bicubic upscale). Cheap, retrain-free, +coverage where it matters.
   Skip Wiener/RL/edge/combos entirely.
2. **Harvest, don't enhance-at-inference:** the ~6% recoverable post-bounce
   frames + the genuinely-present-but-tiny balls are the right **training data**.
   A short fine-tune on upscaled/low-contrast post-bounce crops will beat
   inference-time enhancement, because it fixes the out-of-distribution problem.
3. **Stop chasing pixel recovery for the rest:** the montage proves most misses
   have no recoverable ball. Further gains there need a higher-FPS/closer camera,
   not a filter.

---

## Files

| Deliverable | Path |
|-------------|------|
| Visibility report | `outputs/ball_visibility_report.csv` |
| Enhancement benchmark (all methods) | `outputs/enhancement_benchmark.csv` |
| Super-res / deblur / contrast | same CSV, `family` column |
| Detection / coverage gains per clip | `outputs/detection_gains.csv` |
| Visual examples | `outputs/enhancement_examples/` |
| Enhancement primitives | `analytics/ball_enhancement.py` |
| Benchmark runner | `scripts/ball_enhancement_benchmark.py` |
| Stage-1 analyzer | `scripts/ball_visibility_report.py` |
| Example renderer | `scripts/render_enhancement_examples.py` |
