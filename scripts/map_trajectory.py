"""
scripts/map_trajectory.py
=========================
OFFLINE global trajectory mapping from raw detections.

The diagnosis (scripts/dump_ball_detections.py) showed the detector DOES find the
ball — the detection dots trace the correct path — but the ONLINE tracker commits
frame-by-frame and gets pulled off by false positives (batsman, nets, banner,
sightscreen). This tool instead looks at ALL detections at once and finds the
single smooth, motion-consistent arc that the real ball follows, rejecting
everything that does not fit it.

Method
------
1. Run the ball detector on every frame (ensemble), collect all candidate
   detections (frame, x, y, conf).
2. RANSAC over a TEMPORAL polynomial model x(t), y(t) (t = frame index):
   the real ball moves smoothly, so its detections all lie near a low-order
   curve in time; false positives (static batsman cluster, scattered net/banner
   hits) do not. We sample triplets from DISTINCT, spread-out frames, fit
   quadratics x(t)/y(t), and count inliers from distinct frames.
3. Guard against latching onto a STATIC cluster (a stationary FP gives many
   detections at one spot): reject models whose inlier arc does not actually
   MOVE (min spatial spread) and prefer longer frame coverage.
4. Refit the inliers (confidence-weighted, optionally cubic to capture the
   bounce) -> the clean mapped trajectory.
5. Render an overlay MP4 (faded raw detections + bright inlier ball points +
   the smooth fitted curve drawn progressively) and a single _MAPPED.jpg.

Usage
-----
    python scripts/map_trajectory.py --video videos/test_video5.mp4
    python scripts/map_trajectory.py --video videos/test_video5.mp4 --conf 0.10 --inlier-px 34
    python scripts/map_trajectory.py --all            # every clip in videos/
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Det = Tuple[int, float, float, float]   # frame, x, y, conf


def load_models(primary: str, alt):
    """alt may be a single path or a list of paths (multi-model ensemble)."""
    from ultralytics import YOLO
    models = [YOLO(primary)]
    alts = alt if isinstance(alt, (list, tuple)) else [alt]
    for a in alts:
        if a and Path(a).exists():
            models.append(YOLO(a))
    return models


# Color-assist: detect a saturated, round ball (yellow tennis OR red/dark leather)
# that a white-trained YOLO model misses. These candidates are ADDED to the YOLO
# pool; the existing motion-RANSAC then rejects static colour false positives.
# "off" | "yellow" | "red" | "both"  — measured: red proposer lifts a red-leather
# clip's per-frame recall 0.14 -> 0.24 (clip02) with no training.
_COLOR_ASSIST = False
_COLOR_MODE = "both"


def color_ball_candidates(frame) -> List[tuple]:
    """Return [(x, y, conf)] for saturated round ball-like blobs (tennis and/or
    red leather), controlled by _COLOR_MODE."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = []
    if _COLOR_MODE in ("yellow", "both"):
        masks.append(((20, 70, 90), (55, 255, 255), 20, 1500))       # tennis
    if _COLOR_MODE in ("red", "both"):
        # red leather wraps hue 0 — two bands. Smaller/looser area (balls read tiny).
        masks.append(((0, 60, 40), (12, 255, 220), 3, 500))
        masks.append(((168, 60, 40), (180, 255, 220), 3, 500))
    out = []
    for lo, hi, amin, amax in masks:
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for ct in cnts:
            a = cv2.contourArea(ct)
            if a < amin or a > amax:
                continue
            (x, y), r = cv2.minEnclosingCircle(ct)
            if r <= 0:
                continue
            circ = a / (math.pi * r * r)
            if circ < 0.5:                       # round-ish only
                continue
            out.append((float(x), float(y), 0.30 + 0.4 * min(1.0, circ)))
    return out


def detect_all(models, video: str, conf: float, imgsz: int, device: str,
               augment: bool = False) -> Tuple[List[Det], int, int, int, float]:
    """`augment` = YOLO test-time augmentation (multi-scale/flip). Slower but
    lifts recall on small/blurred balls (measured clip01 0.72 -> 0.76, no training)."""
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dets: List[Det] = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cand = []
        for m in models:
            r = m.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False,
                          augment=augment)[0]
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cand.append(((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf[0])))
        if _COLOR_ASSIST:
            cand.extend(color_ball_candidates(frame))
        # dedupe within frame
        cand.sort(key=lambda d: -d[2])
        kept = []
        for c in cand:
            if all((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 > 22 ** 2 for k in kept):
                kept.append(c)
        for (x, y, c) in kept:
            dets.append((f, x, y, c))
        f += 1
    cap.release()
    return dets, W, H, total, fps


def _polyfit_eval(ts, xs, ys, deg, t_query):
    px = np.polyfit(ts, xs, deg)
    py = np.polyfit(ts, ys, deg)
    return np.polyval(px, t_query), np.polyval(py, t_query), px, py


def _median_step(inl_idx, frames, X, Y) -> float:
    """Median consecutive-frame displacement of an inlier set (px/frame).
    A flying ball is large (~15-40); a static FP cluster (batsman) is ~0-5."""
    order = sorted(inl_idx, key=lambda i: frames[i])
    steps = []
    for a, b in zip(order, order[1:]):
        df = max(1.0, frames[b] - frames[a])
        steps.append(math.hypot(X[b] - X[a], Y[b] - Y[a]) / df)
    return float(np.median(steps)) if steps else 0.0


def _centrality(inl_idx, X, Y, W, H) -> float:
    """Mean centrality of an inlier set (1 = centre, 0 = frame border). Real
    deliveries travel through the central play area; FP columns sit on the net /
    pole structures at the frame EDGES."""
    if W <= 0 or H <= 0:
        return 1.0
    cs = []
    for i in inl_idx:
        fx = min(X[i], W - X[i]) / (W / 2.0)     # 0 at left/right edge, 1 centre
        fy = min(Y[i], H - Y[i]) / (H / 2.0)
        cs.append(max(0.0, min(fx, 1.0)) * 0.7 + max(0.0, min(fy, 1.0)) * 0.3)
    return float(np.mean(cs)) if cs else 1.0


def ransac_trajectory(dets: List[Det], inlier_px: float, W: int = 0, H: int = 0,
                      iters: int = 4000,
                      min_frames: int = 5, min_spread_px: float = 120.0,
                      min_step_px: float = 9.0,
                      seed: int = 0,
                      seed_spread_cap: float = 12.0) -> Tuple[List[Det], Optional[np.ndarray], Optional[np.ndarray]]:
    """Find the largest set of detections consistent with a smooth MOVING ball
    trajectory (quadratic x(t), y(t)). Guards against static FP clusters
    (batsman/sightscreen) by requiring real per-frame motion. Returns
    (inliers, px, py).

    `seed_spread_cap` caps the minimum seed-triplet frame spread. The spread
    requirement used to be 0.18*span of the WHOLE clip, which for a compact
    delivery (~20-30 frames) inside a long clip (~200 frames) demanded a ~35-frame
    seed spread — larger than the delivery itself, so no seed triplet from within
    the real ball arc could ever be sampled and its (correct) quadratic was never
    hypothesised (clip07 truncated 26->7 pts, losing the bounce). Capping at ~12
    frames lets compact deliveries be seeded while still rejecting tight FP
    clusters (which span only a few frames)."""
    if len(dets) < min_frames:
        return dets, None, None, 0.0, 1.0
    rng = np.random.default_rng(seed)
    frames = np.array([d[0] for d in dets], dtype=float)
    X = np.array([d[1] for d in dets]); Y = np.array([d[2] for d in dets])
    C = np.array([d[3] for d in dets])
    uniq_frames = sorted(set(int(f) for f in frames))
    if len(uniq_frames) < 3:
        return dets, None, None, 0.0, 1.0
    t0, t1 = min(frames), max(frames)
    span = max(1.0, t1 - t0)
    tn = (frames - t0) / span                      # normalised time
    min_seed_spread = min(0.18 * span, seed_spread_cap)   # capped for long clips

    best_inliers: List[int] = []
    best_score = -1.0
    best_px = best_py = None
    for _ in range(iters):
        # sample 3 detections from 3 distinct, spread-out frames
        fa, fb, fc = sorted(rng.choice(uniq_frames, size=3, replace=False))
        if (fc - fa) < min_seed_spread:            # need temporal spread (capped)
            continue
        idx = []
        for ff in (fa, fb, fc):
            cands = [i for i in range(len(dets)) if int(frames[i]) == ff]
            idx.append(max(cands, key=lambda i: C[i]))   # highest-conf in that frame
        ts = tn[idx]; xs = X[idx]; ys = Y[idx]
        # require the seed triplet itself to MOVE (reject static seeds early)
        if math.hypot(xs.max() - xs.min(), ys.max() - ys.min()) < 60.0:
            continue
        try:
            xq, yq, px, py = _polyfit_eval(ts, xs, ys, 2, tn)
        except Exception:
            continue
        d2 = (xq - X) ** 2 + (yq - Y) ** 2
        inl = np.where(d2 <= inlier_px ** 2)[0]
        # keep at most one inlier per frame (the closest) -> distinct-frame count
        per_frame = {}
        for i in inl:
            ff = int(frames[i])
            if ff not in per_frame or d2[i] < d2[per_frame[ff]]:
                per_frame[ff] = i
        inl_idx = list(per_frame.values())
        if len(inl_idx) < min_frames:
            continue
        # reject STATIC clusters two ways:
        # (1) the inlier arc must span a large image region (a real flight),
        xr = X[inl_idx].max() - X[inl_idx].min()
        yr = Y[inl_idx].max() - Y[inl_idx].min()
        spread = math.hypot(xr, yr)
        if spread < min_spread_px:
            continue
        # (2) the ball must actually MOVE frame-to-frame (kills batsman cluster).
        mstep = _median_step(inl_idx, frames, X, Y)
        if mstep < min_step_px:
            continue
        # (3) edge penalty: arcs hugging the frame border are usually FP columns
        # on the nets/poles, not a ball travelling through the central play area.
        cen = _centrality(inl_idx, X, Y, W, H)
        if cen < 0.18:                  # almost entirely on the border -> reject
            continue
        # score rewards long, well-moving, confident, CENTRAL arcs
        score = (len(inl_idx) * (1.0 + 0.4 * min(1.0, mstep / 25.0))
                 * (1.0 + spread / 600.0) * (0.4 + 0.6 * cen))
        if score > best_score:
            best_score = score; best_inliers = inl_idx; best_px, best_py = px, py

    if not best_inliers:
        return [], None, None, 0.0, 1.0
    # TEMPORAL-CONTIGUITY filter: the real ball flight is continuous in time.
    # Keep only the largest run of inlier frames with no gap > max_gap; this
    # drops isolated far-in-time FP inliers (e.g. a net-pole hit at frame 1).
    order = sorted(best_inliers, key=lambda i: frames[i])
    max_gap = 10
    runs = [[order[0]]]
    for i in order[1:]:
        if frames[i] - frames[runs[-1][-1]] <= max_gap:
            runs[-1].append(i)
        else:
            runs.append([i])
    run = max(runs, key=len)
    inliers = [dets[i] for i in sorted(run, key=lambda i: frames[i])]
    # confidence-weighted PARABOLA refit (deg 2 = clean arc, no cubic loop).
    fi = np.array([d[0] for d in inliers], dtype=float)
    ti = (fi - t0) / span
    xi = np.array([d[1] for d in inliers]); yi = np.array([d[2] for d in inliers])
    wi = np.clip(np.array([d[3] for d in inliers]), 0.05, 1.0)
    degx = 2 if len(inliers) >= 3 else 1
    # y can be cubic ONLY when the arc is long enough to show a real bounce
    f_span = fi.max() - fi.min()
    degy = 3 if (len(inliers) >= 8 and f_span >= 18) else 2
    try:
        px = np.polyfit(ti, xi, degx, w=wi); py = np.polyfit(ti, yi, degy, w=wi)
    except Exception:
        px, py = best_px, best_py
    # return the global (t0, span) used for the fit so the renderer can evaluate
    # the polynomial in the SAME normalised-time coordinates.
    return inliers, px, py, float(t0), float(span)


def _robust_quadratic(fi, vals, w, deg, resid_sigma=2.5):
    """Confidence-weighted polynomial fit with one robust outlier-rejection
    pass (drop points > resid_sigma residual-stds, refit). Returns coeffs over
    LOCAL normalised time t in [0,1]."""
    fi = np.asarray(fi, float); vals = np.asarray(vals, float); w = np.asarray(w, float)
    t0 = fi.min(); span = max(1.0, fi.max() - fi.min())
    t = (fi - t0) / span
    d = min(deg, max(1, len(fi) - 1))
    c = np.polyfit(t, vals, d, w=w)
    if len(fi) >= d + 3:
        resid = vals - np.polyval(c, t)
        s = np.std(resid) or 1.0
        keep = np.abs(resid) <= resid_sigma * s
        if keep.sum() >= d + 1 and keep.sum() < len(fi):
            c = np.polyfit(t[keep], vals[keep], d, w=w[keep])
    return c, t0, span


def fit_segment(pts: List[Det], n: int = 80, degx: int = 2, degy: int = 2):
    """Robust weighted quadratic fit of one trajectory phase. Returns sampled
    (x,y) polyline across the segment's frame range (the curve FOLLOWS the points
    — quadratic, so it cannot loop or overshoot like a cubic)."""
    if len(pts) < 2:
        return [(p[1], p[2]) for p in pts]
    fi = [p[0] for p in pts]; xi = [p[1] for p in pts]; yi = [p[2] for p in pts]
    wi = np.clip([p[3] for p in pts], 0.05, 1.0)
    cx, t0, span = _robust_quadratic(fi, xi, wi, degx)
    cy, _, _ = _robust_quadratic(fi, yi, wi, degy)
    fs = np.linspace(min(fi), max(fi), n)
    ts = (fs - t0) / span
    xs = np.polyval(cx, ts); ys = np.polyval(cy, ts)
    return list(zip(xs.tolist(), ys.tolist()))


def detect_bounce_index(pts: List[Det]) -> Optional[int]:
    """Find the interior bounce index = where vertical motion REVERSES (the ball
    descends to the pitch then rises toward the bat). Returns None if no clear
    bounce is present in the detected window (then we DON'T force one)."""
    n = len(pts)
    if n < 6:
        return None
    ys = np.array([p[2] for p in pts], float)
    # smooth lightly to avoid latching onto single-frame noise
    k = np.ones(3) / 3.0
    ysm = np.convolve(ys, k, mode="same")
    i = int(np.argmax(ysm))                 # lowest point on screen (max y)
    if i < 2 or i > n - 3:
        return None
    rise = ysm[i] - ysm[max(0, i - 3)]      # came down into the bounce
    fall = ysm[i] - ysm[min(n - 1, i + 3)]  # went up after the bounce
    if rise > 6.0 and fall > 6.0:
        return i
    return None


def segmented_fit(inliers: List[Det]):
    """SEGMENTED physics-consistent fit. Splits at the bounce and fits each phase
    with its own robust quadratic (projectile-like), instead of forcing one
    global curve through the bounce. Returns (segments, bounce_xy) where segments
    is a list of (polyline, label, color)."""
    inliers = sorted(inliers, key=lambda d: d[0])
    if len(inliers) < 3:
        return [([(p[1], p[2]) for p in inliers], "raw", (0, 220, 255))], None
    bi = detect_bounce_index(inliers)
    if bi is not None and bi >= 2 and bi <= len(inliers) - 3:
        pre = inliers[: bi + 1]             # include bounce point in both
        post = inliers[bi:]                 # -> continuity at the bounce
        segs = [
            (fit_segment(pre), "pre-bounce", (0, 235, 255)),    # yellow-cyan
            (fit_segment(post), "post-bounce", (60, 220, 60)),  # green
        ]
        bounce_xy = (inliers[bi][1], inliers[bi][2])
        return segs, bounce_xy
    # no clear bounce -> single honest quadratic (no forced kink)
    return [(fit_segment(inliers), "single", (0, 220, 255))], None


def curve_points(px, py, f_lo, f_hi, t0, span, n=160):
    # sample frames across the inlier span, convert to the fit's normalised time
    fs = np.linspace(f_lo, f_hi, n)
    ts = (fs - t0) / span
    xs = np.polyval(px, ts); ys = np.polyval(py, ts)
    return list(zip(xs.tolist(), ys.tolist()))


def _draw_segments(img, segments, upto_frame=None, f_lo=None, f_hi=None):
    """Draw each fitted phase as its own polyline (pre-bounce / post-bounce)."""
    for (poly, label, col) in segments:
        pts = [(int(x), int(y)) for (x, y) in poly]
        if upto_frame is not None and f_lo is not None and f_hi is not None and f_hi > f_lo:
            kk = max(0, min(len(pts), int((upto_frame - f_lo) / (f_hi - f_lo) * len(pts))))
            pts = pts[:kk]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, 3, cv2.LINE_AA)


def render(video: str, dets: List[Det], inliers: List[Det], segments, bounce_xy,
           out_mp4: Path, out_img: Path):
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    inl_xy = {d[0]: (d[1], d[2]) for d in inliers}
    f_lo = min((d[0] for d in inliers), default=0)
    f_hi = max((d[0] for d in inliers), default=0)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    base = None
    f = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if base is None:
            base = frame.copy()
        for (df, x, y, c) in dets:                      # faded raw dets (incl FPs)
            if df == f:
                cv2.circle(frame, (int(x), int(y)), 4, (130, 130, 130), 1, cv2.LINE_AA)
        _draw_segments(frame, segments, upto_frame=f, f_lo=f_lo, f_hi=f_hi)
        if bounce_xy is not None:                       # bounce split marker
            cv2.drawMarker(frame, (int(bounce_xy[0]), int(bounce_xy[1])), (40, 40, 235),
                           cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        if f in inl_xy:                                 # current ball point
            x, y = inl_xy[f]
            cv2.circle(frame, (int(x), int(y)), 7, (0, 80, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 11, (0, 220, 255), 2, cv2.LINE_AA)
        seglab = "+".join(s[1] for s in segments)
        txt = f"SEGMENTED fit  frame {f}  pts={len(inliers)}  [{seglab}]"
        cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(frame)
        f += 1
    cap.release(); vw.release()

    if base is not None:                                # single mapped still
        for (df, x, y, c) in dets:
            cv2.circle(base, (int(x), int(y)), 3, (130, 130, 130), 1, cv2.LINE_AA)
        _draw_segments(base, segments)
        if bounce_xy is not None:
            cv2.drawMarker(base, (int(bounce_xy[0]), int(bounce_xy[1])), (40, 40, 235),
                           cv2.MARKER_TILTED_CROSS, 26, 3, cv2.LINE_AA)
            cv2.putText(base, "bounce", (int(bounce_xy[0]) + 12, int(bounce_xy[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 235), 2, cv2.LINE_AA)
        for d in inliers:
            cv2.circle(base, (int(d[1]), int(d[2])), 6, (0, 80, 255), -1, cv2.LINE_AA)
        seglab = "+".join(s[1] for s in segments)
        msg = f"{Path(video).stem}: {len(inliers)} ball pts | segmented[{seglab}] (grey=raw incl FPs)"
        cv2.putText(base, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(base, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_img), base)


def process(video: str, models, conf: float, imgsz: int, device: str, inlier_px: float):
    name = Path(video).stem
    out_dir = ROOT / "outputs" / "mapped" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dets, W, H, total, fps = detect_all(models, video, conf, imgsz, device)
    inliers, px, py, t0, span = ransac_trajectory(dets, inlier_px, W=W, H=H)
    segments, bounce_xy = segmented_fit(inliers)
    render(video, dets, inliers, segments, bounce_xy,
           out_dir / f"{name}_mapped.mp4", out_dir / "_MAPPED.jpg")
    with open(out_dir / "mapped_path.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "x", "y", "conf"]); w.writerows(inliers)
    # Persist the RAW detections from THIS SAME pass so post-bounce recovery
    # (delivery_reconstruction.recover_post_bounce) can read them and they are
    # guaranteed consistent with mapped_path.csv (no cross-run artifact mismatch).
    det_dir = ROOT / "outputs" / "detections" / name
    det_dir.mkdir(parents=True, exist_ok=True)
    with open(det_dir / "detections.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "x", "y", "conf"])
        w.writerows([[d[0], round(d[1], 1), round(d[2], 1), round(d[3], 3)] for d in dets])
    rej = len(dets) - len(inliers)
    seglab = "+".join(s[1] for s in segments)
    print(f"  {name}: raw_dets={len(dets)}  ball_arc_pts={len(inliers)}  rejected_FP={rej}  "
          f"frames={(inliers[0][0] if inliers else '-')}..{(inliers[-1][0] if inliers else '-')}  "
          f"fit=[{seglab}]  bounce={'yes' if bounce_xy else 'no'}")
    print(f"    -> {out_dir/'_MAPPED.jpg'}  |  {out_dir/(name+'_mapped.mp4')}")
    return len(inliers)


def main():
    ap = argparse.ArgumentParser(description="Offline global trajectory mapping from detections")
    ap.add_argument("--video", default=None)
    ap.add_argument("--all", action="store_true", help="process every clip in videos/")
    ap.add_argument("--model", default=str(ROOT / "models" / "ball_ft_t4.pt"))
    ap.add_argument("--alt-model", nargs="*", default=[str(ROOT / "models" / "ball_best_leather_new.pt")],
                    help="one or more secondary models for a multi-model ensemble")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--inlier-px", type=float, default=34.0)
    ap.add_argument("--color-assist", action="store_true",
                    help="also detect a bright saturated ball by colour (e.g. yellow tennis ball)")
    args = ap.parse_args()
    global _COLOR_ASSIST
    _COLOR_ASSIST = bool(args.color_assist)

    vids: List[str] = []
    if args.all:
        vids = sorted(glob.glob(str(ROOT / "videos" / "*.mp4")))
    elif args.video:
        vids = [args.video]
    else:
        print("pass --video <path> or --all"); return

    models = load_models(args.model, args.alt_model)
    print(f"Mapping {len(vids)} clip(s) | conf={args.conf} inlier={args.inlier_px}px")
    for v in vids:
        try:
            process(v, models, args.conf, args.imgsz, args.device, args.inlier_px)
        except Exception as e:
            import traceback; print(f"  {Path(v).stem}: ERROR {e}"); traceback.print_exc()


if __name__ == "__main__":
    main()
