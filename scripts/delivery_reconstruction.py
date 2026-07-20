"""
scripts/delivery_reconstruction.py
==================================
DELIVERY RECONSTRUCTION ENGINE — maximise CORRECT trajectory completeness.

A cricket delivery is ONE event: Release -> Flight -> Bounce -> Rise -> Batter.
The detector gives sparse, sometimes-incomplete observations of it. This engine
reconstructs the hidden flight between the real detections, with two priorities:

  * POST-BOUNCE RECOVERY (highest): when pre-bounce + bounce + a few post-bounce
    detections exist, continue the rise toward the batter using the post-bounce
    projectile (estimated outgoing angle + energy), bounded by confidence.
  * RELEASE RECOVERY: extend the descent back toward release when the descent
    direction is well established.

Rules
-----
  * Detections are PRIMARY. Physics only CONSTRAINS the gaps between them.
  * Extensions are CONFIDENCE-SCALED and clamped at the parabola vertex, so we
    never turn the ball over or fabricate impossible motion.
  * Low confidence => stay short and safe. High confidence => longer reach.

Outputs
-------
  outputs/final/<clip>.mp4 / .jpg     broadcast trajectory (premium, no markers)
  outputs/debug/<clip>.jpg            raw / cleaned / bounce / reconstructed
  outputs/final/_DELIVERY_METRICS.csv per-clip completion metrics
  outputs/final/_ALL_FINAL_MONTAGE.jpg

Usage
-----
    python scripts/delivery_reconstruction.py --clip test_video5
    python scripts/delivery_reconstruction.py --all
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE = (255, 90, 20)
GLOW = (255, 150, 70)
FINAL = ROOT / "outputs" / "final"
DEBUG = ROOT / "outputs" / "debug"


def read_points(clip: str):
    p = ROOT / "outputs" / "mapped" / clip / "mapped_path.csv"
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(open(p)):
        try:
            out.append((int(float(r["frame"])), float(r["x"]), float(r["y"]),
                        float(r.get("conf", 1.0))))
        except (KeyError, ValueError):
            continue
    return sorted(out, key=lambda d: d[0])


def read_raw(clip: str):
    p = ROOT / "outputs" / "detections" / clip / "detections.csv"
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(open(p)):
        try:
            out.append((int(float(r["frame"])), float(r["x"]), float(r["y"]), float(r["conf"])))
        except (KeyError, ValueError):
            continue
    return sorted(out, key=lambda d: d[0])


def _read_release_recovered(clip):
    """ROI multi-scale release detections saved by scripts/release_recovery.py."""
    p = ROOT / "outputs" / "mapped" / clip / "release_recovered.csv"
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(open(p)):
        try:
            out.append((int(float(r["frame"])), float(r["x"]), float(r["y"]), float(r["conf"])))
        except (KeyError, ValueError):
            continue
    return out


def recover_pre_release(clip, start_frame, start_xy, win=15, max_speed=42.0, near=60.0):
    """Recover REAL release/early-flight ball detections the mapper discarded
    BEFORE the trajectory starts. Combines (a) raw detections already in
    detections.csv and (b) the ROI multi-scale points from release_recovery.py,
    accepting only a MOVING chain consistent with the early flight."""
    cand = {}
    for (f, x, y, c) in read_raw(clip):
        if start_frame - win <= f < start_frame and \
           math.hypot(x - start_xy[0], y - start_xy[1]) <= max_speed * (start_frame - f) + near:
            if f not in cand or c > cand[f][3]:
                cand[f] = (f, x, y, c)
    for (f, x, y, c) in _read_release_recovered(clip):   # ROI-recovered (targeted)
        if f < start_frame and f not in cand:
            cand[f] = (f, x, y, c)
    pre = [cand[k] for k in sorted(cand)]
    if len(pre) < 2:
        return []
    P = np.array([(p[1], p[2]) for p in pre])
    spread = math.hypot(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    fr = np.array([p[0] for p in pre])
    steps = [np.linalg.norm(P[i] - P[i - 1]) / max(1, fr[i] - fr[i - 1]) for i in range(1, len(P))]
    med = float(np.median(steps)) if steps else 0.0
    return pre if (spread > 35 and med > 6) else []


def recover_post_bounce(clip, pivot_frame, pivot_xy, win=12, max_speed=42.0, near=60.0):
    """Recover REAL post-bounce ball detections the offline mapper discarded.
    Looks in the raw detections just after the pivot (bounce / descent-end), takes
    one per frame (highest conf), velocity-bounded from the pivot, and accepts the
    chain ONLY if it actually MOVES (spread+step) — a static FP is rejected, a
    rising ball is kept. Returns the recovered (frame,x,y,conf) points."""
    raw = read_raw(clip)
    if not raw:
        return []
    per = {}
    for (f, x, y, c) in raw:
        if not (pivot_frame < f <= pivot_frame + win):
            continue
        if math.hypot(x - pivot_xy[0], y - pivot_xy[1]) > max_speed * (f - pivot_frame) + near:
            continue
        if f not in per or c > per[f][3]:
            per[f] = (f, x, y, c)
    post = [per[k] for k in sorted(per)]
    if len(post) < 3:
        return []
    P = np.array([(p[1], p[2]) for p in post])
    spread = math.hypot(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    fr = np.array([p[0] for p in post])
    steps = [np.linalg.norm(P[i] - P[i - 1]) / max(1, fr[i] - fr[i - 1]) for i in range(1, len(P))]
    med = float(np.median(steps)) if steps else 0.0
    if spread > 45 and med > 7:            # a real MOVING ball -> recover it
        return post
    return []


# ── cleaning ─────────────────────────────────────────────────────────────────
def clean_track(points):
    if len(points) < 4:
        return points
    P = list(points)
    xy = np.array([(p[1], p[2]) for p in P], float)
    fr = np.array([p[0] for p in P], float)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    med = float(np.median(steps)) or 1.0
    keep = [0]
    for i in range(1, len(P) - 1):
        mid = (xy[i - 1] + xy[i + 1]) / 2.0
        spike = np.linalg.norm(xy[i] - mid) > 3.0 * med + 8.0
        v_in = np.linalg.norm(xy[i] - xy[i - 1]) / max(1.0, fr[i] - fr[i - 1])
        v_out = np.linalg.norm(xy[i + 1] - xy[i]) / max(1.0, fr[i + 1] - fr[i])
        jump = (v_in > 6 * v_out + 6) or (v_out > 6 * v_in + 6)
        if not (spike or jump):
            keep.append(i)
    keep.append(len(P) - 1)
    return [P[i] for i in keep]


def detect_bounce_index(pts):
    ys = np.array([p[2] for p in pts], float)
    n = len(ys)
    if n < 6:
        return None
    ysm = np.convolve(ys, np.ones(3) / 3.0, mode="same")
    i = int(np.argmax(ysm))
    if i < 2 or i > n - 3:
        return None
    rise = ysm[i] - ysm[max(0, i - 3)]; fall = ysm[i] - ysm[min(n - 1, i + 3)]
    if rise > 6 and fall > 6:
        return i, float(min(rise, fall))
    return None


# ── delivery confidence ──────────────────────────────────────────────────────
def delivery_confidence(clean, has_bounce, bounce_mag=0.0):
    n = len(clean)
    c_count = min(1.0, n / 12.0)
    frames = np.array([p[0] for p in clean], float)
    if len(frames) > 2:
        gaps = np.diff(frames)
        c_space = 1.0 / (1.0 + (np.std(gaps) / (np.mean(gaps) + 1e-6)))
    else:
        c_space = 0.5
    c_conf = float(np.mean([p[3] for p in clean])) if clean else 0.0
    c_bounce = min(1.0, 0.4 + bounce_mag / 40.0) if has_bounce else 0.4
    conf = 0.35 * c_count + 0.20 * c_space + 0.20 * c_bounce + 0.25 * min(1.0, c_conf * 2)
    return float(max(0.0, min(1.0, conf)))


# ── projectile fit + confidence-scaled, vertex-clamped extension ─────────────
def fit_projectile(seg, anchor=None, anchor_at="end", extend_at="start", ext_px=0.0, n=70):
    """Fit a projectile parabola through the segment's detections (DATA range
    only), then CONTINUE conservatively along the chosen end's tangent by ext_px
    pixels (a straight continuation that cannot turn the ball over).
      anchor_at: which end the bounce anchor pins to ('end' or 'start')
      extend_at: which end to tangent-extend ('end' or 'start')."""
    fr = np.array([p[0] for p in seg], float)
    xs = np.array([p[1] for p in seg], float)
    ys = np.array([p[2] for p in seg], float)
    w = np.clip([p[3] for p in seg], 0.15, 1.0)
    if anchor is not None:
        bf = fr.max() + 1 if anchor_at == "end" else fr.min() - 1
        fr = np.append(fr, bf); xs = np.append(xs, anchor[0]); ys = np.append(ys, anchor[1])
        w = np.append(w, w.max() * 6.0)
    order = np.argsort(fr)
    fr, xs, ys, w = fr[order], xs[order], ys[order], w[order]
    t0 = fr.min(); span = max(1.0, fr.max() - fr.min())
    t = (fr - t0) / span
    deg = 2 if len(t) >= 3 else 1
    try:
        cx = np.polyfit(t, xs, deg, w=w); cy = np.polyfit(t, ys, deg, w=w)
    except Exception:
        cx = np.polyfit(t, xs, 1, w=w); cy = np.polyfit(t, ys, 1, w=w)
    ts = np.linspace(0.0, 1.0, n)
    main = list(zip(np.polyval(cx, ts).tolist(), np.polyval(cy, ts).tolist()))
    if ext_px > 0 and len(main) >= 2:
        main = _tangent_extend(main, ext_px, extend_at)
    return main, (cx, cy)


def _tangent_extend(curve, ext_px, which_out, step=6.0):
    """Continue the curve straight along its end tangent for ext_px pixels."""
    p = [np.array(q, float) for q in curve]
    out = list(curve)
    nstep = max(1, int(ext_px / step))
    if which_out == "end":
        d = p[-1] - p[-2]
        if np.linalg.norm(d) > 1e-6:
            d = d / np.linalg.norm(d)
            for k in range(1, nstep + 1):
                out.append(tuple(p[-1] + d * step * k))
    else:
        d = p[0] - p[1]
        if np.linalg.norm(d) > 1e-6:
            d = d / np.linalg.norm(d)
            pre = [tuple(p[0] + d * step * k) for k in range(nstep, 0, -1)]
            out = pre + out
    return out


def reconstruct(points, clip=None):
    clean = clean_track(points)
    if len(clean) < 2:
        xy = [(p[1], p[2]) for p in clean]
        return {"down": xy, "up": [], "clean": clean, "bounce": None, "conf": 0.2,
                "n_pre": len(clean), "n_post": 0, "recovered": 0, "recovered_pre": 0}
    # RELEASE RECOVERY: prepend real early-flight ball detections the mapper
    # dropped before the trajectory started (moving chain only, no static FP).
    rec_pre = recover_pre_release(clip, int(clean[0][0]), (clean[0][1], clean[0][2])) if clip else []
    if rec_pre:
        have = {int(p[0]) for p in clean}
        clean = sorted(rec_pre + [p for p in clean], key=lambda p: p[0])
        clean = [p for i, p in enumerate(clean) if i == 0 or int(p[0]) != int(clean[i - 1][0])]
    # mapped-arc bounce (if any)
    b = detect_bounce_index(clean)
    bi = None
    if b is not None and 2 <= b[0] <= len(clean) - 3:
        bi = b[0]

    # PIVOT = the bounce (if mapped) else the descent end. Recover REAL moving
    # post-bounce detections the mapper dropped, from raw detections.
    if bi is not None:
        pivot_xy = (clean[bi][1], clean[bi][2]); pivot_f = clean[bi][0]
        mapped_post = clean[bi + 1:]            # post-bounce points already in the arc
    else:
        pivot_xy = (clean[-1][1], clean[-1][2]); pivot_f = clean[-1][0]
        mapped_post = []
    recovered = recover_post_bounce(clip, pivot_f, pivot_xy) if clip else []

    # combine REAL post-bounce points from both sources (dedupe by frame)
    by_f = {int(p[0]): p for p in mapped_post}
    for p in recovered:
        by_f.setdefault(int(p[0]), p)
    post_all = [by_f[k] for k in sorted(by_f)]

    rise_ok = False
    if len(post_all) >= 2:
        rise_px = pivot_xy[1] - min(p[2] for p in post_all)
        spread = math.hypot(np.ptp([p[1] for p in post_all]), np.ptp([p[2] for p in post_all]))
        rise_ok = rise_px > 10.0 or spread > 30.0

    if rise_ok:
        # REAL-detection-supported rise (mapped + recovered dots, no fabrication).
        bmag = b[1] if bi is not None else 20.0
        conf = delivery_confidence(clean + recovered, True, bmag)
        pre = clean[: bi + 1] if bi is not None else clean
        ext_release = 18 + 40 * conf
        down, cd = fit_projectile(pre, anchor=pivot_xy, anchor_at="end",
                                  extend_at="start", ext_px=ext_release)
        up_seg = [(pivot_f, pivot_xy[0], pivot_xy[1], 1.0)] + post_all
        ext_batter = min(45.0, 15 + 30 * conf)
        up, cu = fit_projectile(up_seg, anchor=None, extend_at="end", ext_px=ext_batter)
        return {"down": down, "up": up, "clean": clean, "bounce": pivot_xy, "conf": conf,
                "ang_in": _ang(cd, 1.0), "ang_out": _ang(cu, 0.0),
                "n_pre": len(pre), "n_post": len(post_all), "recovered": len(recovered), "recovered_pre": len(rec_pre)}

    # No recoverable post-bounce ball -> ONE smooth conservative descent.
    conf = delivery_confidence(clean, False)
    ext = 16 + 40 * conf
    down, _ = fit_projectile(clean, anchor=None, extend_at="end", ext_px=ext)
    return {"down": down, "up": [], "clean": clean, "bounce": None,
            "conf": conf, "n_pre": len(clean), "n_post": 0, "recovered": 0, "recovered_pre": len(rec_pre)}


def _prepend_release(ext_curve, main_curve):
    """Keep only the release-side extension that lies before the main curve start."""
    if not ext_curve or not main_curve:
        return main_curve
    start = np.array(main_curve[0])
    pre = [p for p in ext_curve if np.linalg.norm(np.array(p) - start) > 6]
    # keep the portion of ext that extends backward (first ~25%)
    pre = pre[: max(0, len(ext_curve) // 4)]
    return pre + main_curve


def _ang(coeffs, at_t):
    cx, cy = coeffs
    return math.degrees(math.atan2(np.polyval(np.polyder(cy), at_t), np.polyval(np.polyder(cx), at_t)))


# ── completion metrics ───────────────────────────────────────────────────────
def metrics(points, R):
    clean = R["clean"]; curve = R["down"] + R["up"]
    if len(clean) < 2 or len(curve) < 2:
        return dict(trajectory_length=0, continuity=0.0, release_cov=0.0,
                    post_bounce_cov=0.0, bounce_acc=0.0, completion=0.0, conf=round(R["conf"], 2))
    cp = np.array(curve)
    traj_len = float(np.sum(np.linalg.norm(np.diff(cp, axis=0), axis=1)))
    frames = sorted(int(p[0]) for p in clean)
    span = max(1, frames[-1] - frames[0])
    covered = sum(1 for f in range(frames[0], frames[-1] + 1)
                  if any(abs(f - rf) <= 1 for rf in frames))
    continuity = covered / (span + 1)
    # release coverage: detections present in the first 30% of the flight window
    rel_cut = frames[0] + 0.30 * span
    release_cov = min(1.0, sum(1 for f in frames if f <= rel_cut) / 2.0)
    # post-bounce coverage
    post_bounce_cov = min(1.0, R["n_post"] / 4.0) if R["bounce"] else 0.0
    bounce_acc = 1.0 if R["bounce"] else 0.0
    # completion: has release-ish start + (bounce + post) ideally
    completion = 0.4 * min(1.0, release_cov) + (0.6 * post_bounce_cov if R["bounce"] else 0.2 * continuity)
    return dict(trajectory_length=round(traj_len, 1), continuity=round(continuity, 2),
                release_cov=round(release_cov, 2), post_bounce_cov=round(post_bounce_cov, 2),
                bounce_acc=round(bounce_acc, 2), completion=round(min(1.0, completion), 2),
                conf=round(R["conf"], 2))


# ── premium render (trajectory only) ─────────────────────────────────────────
def draw_arc(frame, curve, thick):
    if len(curve) < 2:
        return
    H = frame.shape[0]
    pts = [(int(x), int(y)) for (x, y) in curve]
    mask = np.zeros_like(frame)
    for a, b in zip(pts, pts[1:]):
        cv2.line(mask, a, b, GLOW, thick + 9, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), 7, 7)
    frame[:] = np.clip(frame.astype(np.int16) + (mask.astype(np.int16) * 0.55), 0, 255).astype(np.uint8)
    m = len(pts)
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        f = i / max(1, m - 1)
        persp = 0.65 + 0.55 * ((a[1] + b[1]) * 0.5 / max(1, H))
        tw = max(2, int(thick * persp))
        if f < 0.10:
            seg = frame.copy(); cv2.line(seg, a, b, BLUE, tw, cv2.LINE_AA)
            al = 0.30 + (f / 0.10) * 0.70
            cv2.addWeighted(seg, al, frame, 1 - al, 0, frame)
        else:
            cv2.line(frame, a, b, BLUE, tw, cv2.LINE_AA)


def save_debug(base, raw, R, path):
    img = base.copy()
    curve = R["down"] + R["up"]
    for a, b in zip([(int(x), int(y)) for x, y in curve], [(int(x), int(y)) for x, y in curve][1:]):
        cv2.line(img, a, b, BLUE, 3, cv2.LINE_AA)
    for p in raw:
        cv2.circle(img, (int(p[1]), int(p[2])), 6, (40, 40, 235), -1, cv2.LINE_AA)
    for p in R["clean"]:
        cv2.circle(img, (int(p[1]), int(p[2])), 4, (60, 220, 60), 1, cv2.LINE_AA)
    if R["bounce"]:
        cv2.drawMarker(img, (int(R["bounce"][0]), int(R["bounce"][1])), (255, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 24, 2, cv2.LINE_AA)
    t = f"conf {R['conf']:.2f}  pre {R['n_pre']}  post {R['n_post']}  (red=raw green=clean X=bounce)"
    cv2.putText(img, t, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, t, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def render(clip, thick):
    pts = read_points(clip)
    if len(pts) < 3:
        print(f"  {clip}: only {len(pts)} pts — skip"); return None
    video = ROOT / "videos" / f"{clip}.mp4"
    if not video.exists():
        print(f"  {clip}: no video"); return None
    R = reconstruct(pts, clip=clip)
    M = metrics(pts, R)
    curve = R["down"] + R["up"]
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f_lo = pts[0][0]
    FINAL.mkdir(parents=True, exist_ok=True); DEBUG.mkdir(parents=True, exist_ok=True)
    out_mp4 = FINAL / f"{clip}.mp4"
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    base = None; f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if base is None:
            base = frame.copy()
        if f >= f_lo - 3:
            draw_arc(frame, curve, thick)
        vw.write(frame); f += 1
    cap.release(); vw.release()
    if base is not None:
        ci = base.copy(); draw_arc(ci, curve, thick)
        cv2.imwrite(str(FINAL / f"{clip}.jpg"), ci)
        save_debug(base, pts, R, DEBUG / f"{clip}.jpg")
    row = dict(clip=clip, **M, bounce=("yes" if R["bounce"] else "no"),
               n_pre=R["n_pre"], n_post=R["n_post"])
    print(f"  {clip}: conf={M['conf']} completion={M['completion']} cont={M['continuity']} "
          f"len={M['trajectory_length']} post_cov={M['post_bounce_cov']} bounce={row['bounce']}")
    return row


def montage():
    imgs = sorted(p for p in FINAL.glob("*.jpg") if not p.name.startswith("_"))
    tiles = []
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]; tw = 300; th = int(h * tw / w)
        tiles.append(cv2.resize(im, (tw, th)))
    if not tiles:
        return
    th = max(t.shape[0] for t in tiles); tw = tiles[0].shape[1]
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cols = 5; rows = math.ceil(len(tiles) / cols)
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(str(FINAL / "_ALL_FINAL_MONTAGE.jpg"), sheet)


def main():
    ap = argparse.ArgumentParser(description="Delivery reconstruction engine")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--thick", type=int, default=10)
    args = ap.parse_args()
    if args.all:
        clips = [p.parent.name for p in sorted((ROOT / "outputs" / "mapped").glob("*/mapped_path.csv"))]
    elif args.clip:
        clips = [args.clip]
    else:
        print("pass --clip or --all"); return
    print(f"Delivery reconstruction for {len(clips)} clip(s)")
    rows = []
    for c in clips:
        r = render(c, args.thick)
        if r:
            rows.append(r)
    if rows:
        fields = ["clip", "conf", "completion", "continuity", "trajectory_length",
                  "release_cov", "post_bounce_cov", "bounce_acc", "bounce", "n_pre", "n_post"]
        with open(FINAL / "_DELIVERY_METRICS.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
            for r in rows:
                w.writerow(r)
        mean_comp = round(sum(r["completion"] for r in rows) / len(rows), 3)
        mean_cont = round(sum(r["continuity"] for r in rows) / len(rows), 3)
        print(f"\n=== {len(rows)} deliveries | mean completion={mean_comp} continuity={mean_cont} ===")
        print(f"  metrics -> {FINAL/'_DELIVERY_METRICS.csv'}")
        if len(rows) > 1:
            montage()


if __name__ == "__main__":
    main()
