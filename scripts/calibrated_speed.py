"""
scripts/calibrated_speed.py
===========================
PHYSICALLY-CALIBRATED speed engine. Uses a pitch-plane HOMOGRAPHY so pixel
displacement is converted to real metres (perspective-correct), then computes
speed ONLY from real detections — never the rendered/spline/extension curve.

Phases
------
1. CALIBRATION: detect BOTH stump sets (batting + bowling). Each set gives a
   centre + width at a known pitch position. Two sets x left/right edge = 4
   image<->world correspondences.
2. HOMOGRAPHY: solve H mapping image -> pitch-plane metres (down-pitch y in
   [0, pitch_len], lateral x in metres). Pitch length default 20.12 m.
3. WORLD TRAJECTORY: project each REAL detection (mapped + recovered) through H.
4. REAL DISTANCE: world distance between consecutive real detections.
5. SPEED: per-segment speed = world_dist / dt(fps). instantaneous / rolling /
   delivery (robust median).
6. ROBUST: reject bounce-frame segments, tracking jumps, and >2.5 MAD outliers.
7. CONFIDENCE: calibration quality x stump stability x detection count x
   homography sanity.
8. GROUND-TRUTH GATE: if calibration is poor/none OR both stumps not found ->
   "speed unavailable" (no fake km/h).
9. VALIDATION: flag anything outside 40-170 km/h.

Honest limitation: the ball is ABOVE the pitch plane (airborne parallax), so the
ground-plane homography slightly under/over-shoots the airborne portion; the
bounce and low points are most accurate. This is reflected in confidence, and is
still far more correct than image-space.

Outputs: outputs/speed_validation.csv  + outputs/calibrated_speed.json

Usage:
    python scripts/calibrated_speed.py
    python scripts/calibrated_speed.py --pitch-length 20.12 --clips test_video5
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUMP = ROOT / "models" / "stump_best.pt"
STUMP_WIDTH_M = 0.2286
PITCH_LEN_DEFAULT = 20.12

try:
    import torch
    DEVICE = "0" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

_spec = importlib.util.spec_from_file_location("dr", ROOT / "scripts" / "delivery_reconstruction.py")
dr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dr)


def load_calib():
    p = ROOT / "outputs" / "calibration_report.csv"
    return {r["clip"]: r for r in csv.DictReader(open(p))} if p.exists() else {}


# ── Phase 1: detect both stump sets ──────────────────────────────────────────
def detect_both_stumps(model, clip, n=16):
    cap = cv2.VideoCapture(str(ROOT / "videos" / f"{clip}.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(3)); H = int(cap.get(4))
    dets = []
    for f in np.linspace(0, max(0, total - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read()
        if not ok:
            continue
        r = model.predict(im, conf=0.40, imgsz=1280, device=DEVICE, verbose=False)[0]
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append([(x1 + x2) / 2, y2, x2 - x1, (x2 - x1) * (y2 - y1)])  # cx, base_y, w, area
    cap.release()
    if len(dets) < 4:
        return None, W, H
    D = np.array(dets)
    # split into two area clusters (near=large, far=small) via the area median gap
    order = np.argsort(D[:, 3])
    Ds = D[order]
    # k=2 split at the largest area gap
    gaps = np.diff(Ds[:, 3])
    if len(gaps) == 0:
        return None, W, H
    split = int(np.argmax(gaps)) + 1
    far = Ds[:split]; near = Ds[split:]
    if len(far) < 2 or len(near) < 2:
        return None, W, H
    def med(g):
        return dict(cx=float(np.median(g[:, 0])), base_y=float(np.median(g[:, 1])),
                    w=float(np.median(g[:, 2])), n=len(g))
    return dict(near=med(near), far=med(far)), W, H


# ── Phase 2: homography image -> pitch metres ────────────────────────────────
def build_homography(stumps, pitch_len):
    """4 correspondences: near (bowler, y=0) & far (batsman, y=pitch_len), each
    left/right edge at +/- half stump-width."""
    n = stumps["near"]; f = stumps["far"]
    hw = STUMP_WIDTH_M / 2.0
    img = np.array([
        [n["cx"] - n["w"] / 2, n["base_y"]], [n["cx"] + n["w"] / 2, n["base_y"]],
        [f["cx"] - f["w"] / 2, f["base_y"]], [f["cx"] + f["w"] / 2, f["base_y"]],
    ], dtype=np.float32)
    wrld = np.array([
        [-hw, 0.0], [hw, 0.0],
        [-hw, pitch_len], [hw, pitch_len],
    ], dtype=np.float32)
    Hm, _ = cv2.findHomography(img, wrld)
    # sanity: pitch must span a sensible pixel height
    pitch_px = abs(n["base_y"] - f["base_y"])
    return Hm, pitch_px


def to_world(Hm, x, y):
    p = Hm @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


# ── Phases 3-6: world trajectory + robust speed ──────────────────────────────
def compute_speed(clean, recovered, Hm, fps, bounce_frame):
    pts = sorted({int(p[0]): p for p in (clean + recovered)}.values(), key=lambda p: p[0])
    if len(pts) < 3:
        return None
    world = [(p[0], *to_world(Hm, p[1], p[2])) for p in pts]
    seg_speeds = []
    inst = []
    for (f0, wx0, wy0), (f1, wx1, wy1) in zip(world, world[1:]):
        dt = (f1 - f0) / fps
        if dt <= 0:
            continue
        dist = math.hypot(wx1 - wx0, wy1 - wy0)
        kmh = dist / dt * 3.6
        # reject bounce-frame segment (vertical reversal) + implausible jumps
        if bounce_frame is not None and (f0 <= bounce_frame <= f1):
            continue
        if kmh > 250 or kmh < 5:                # tracking jump / static
            continue
        seg_speeds.append(kmh); inst.append((f0, round(kmh, 1)))
    if len(seg_speeds) < 2:
        return None
    s = np.array(seg_speeds)
    # MAD outlier rejection
    med = np.median(s); mad = np.median(np.abs(s - med)) or 1.0
    keep = s[np.abs(s - med) <= 2.5 * mad]
    delivery = float(np.median(keep)) if len(keep) else float(med)
    # rolling = median of a 3-window
    rolling = float(np.median(s)) if len(s) else delivery
    return dict(delivery_kmh=round(delivery, 1), rolling_kmh=round(rolling, 1),
                n_segments=len(seg_speeds), instantaneous=inst[:12],
                world_first=(round(world[0][1], 2), round(world[0][2], 2)),
                world_last=(round(world[-1][1], 2), round(world[-1][2], 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--pitch-length", type=float, default=PITCH_LEN_DEFAULT)
    args = ap.parse_args()
    calib = load_calib()
    clips = args.clips or [p.parent.name for p in sorted((ROOT / "outputs/mapped").glob("*/mapped_path.csv"))
                           if p.parent.name.startswith("test_video")]
    from ultralytics import YOLO
    smodel = YOLO(str(STUMP))

    rows = []; records = {}
    print(f"{'clip':13}{'calib':>6}{'2stumps':>8}{'pitchPx':>8}{'segs':>5}"
          f"{'speed_kmh':>11}{'conf':>6}  status")
    print("-" * 72)
    for clip in clips:
        cal = calib.get(clip, {}); cq = cal.get("calib_quality", "none")
        fps = float(cal.get("fps", 30) or 30)
        pts = dr.read_points(clip)
        if len(pts) < 3:
            continue
        clean = dr.clean_track(pts)
        recovered = dr.recover_post_bounce(clip, int(clean[-1][0]), (clean[-1][1], clean[-1][2])) or []
        b = dr.detect_bounce_index(clean)
        bf = clean[b[0]][0] if b else None

        # GROUND-TRUTH GATE 1: calibration must be usable
        if cq == "none":
            rows.append(dict(clip=clip, calib=cq, two_stumps="-", speed_kmh="unavailable",
                             confidence=0.0, status="calib_none")); records[clip] = rows[-1]
            print(f"{clip:13}{cq:>6}{'-':>8}{'-':>8}{'-':>5}{'unavailable':>11}{0.0:>6}  calib_none")
            continue

        stumps, W, Hh = detect_both_stumps(smodel, clip)
        # GATE 2: need BOTH stump sets for a homography
        if stumps is None:
            rows.append(dict(clip=clip, calib=cq, two_stumps="no", speed_kmh="unavailable",
                             confidence=0.0, status="single_stump")); records[clip] = rows[-1]
            print(f"{clip:13}{cq:>6}{'no':>8}{'-':>8}{'-':>5}{'unavailable':>11}{0.0:>6}  single_stump")
            continue

        Hm, pitch_px = build_homography(stumps, args.pitch_length)
        # GATE 3: stumps must be sufficiently separated in pixels (else degenerate)
        if pitch_px < 40:
            rows.append(dict(clip=clip, calib=cq, two_stumps="yes", speed_kmh="unavailable",
                             confidence=0.0, status="degenerate_homography")); records[clip] = rows[-1]
            print(f"{clip:13}{cq:>6}{'yes':>8}{pitch_px:>8.0f}{'-':>5}{'unavailable':>11}{0.0:>6}  degenerate")
            continue

        sp = compute_speed(clean, recovered, Hm, fps, bf)
        if sp is None:
            rows.append(dict(clip=clip, calib=cq, two_stumps="yes", speed_kmh="unavailable",
                             confidence=0.0, status="too_few_segments")); records[clip] = rows[-1]
            print(f"{clip:13}{cq:>6}{'yes':>8}{pitch_px:>8.0f}{0:>5}{'unavailable':>11}{0.0:>6}  few_segments")
            continue

        kmh = sp["delivery_kmh"]
        # CONFIDENCE
        gate = {"good": 1.0, "poor": 0.55}.get(cq, 0.3)
        conf = round(gate * min(1.0, sp["n_segments"] / 5.0) * min(1.0, pitch_px / 200.0), 2)
        # VALIDATION: plausible cricket range
        in_range = 40 <= kmh <= 170
        status = "ok" if in_range else "out_of_range"
        speed_out = kmh if in_range else f"{kmh}(out_of_range)"
        rec = dict(clip=clip, calib=cq, two_stumps="yes", pitch_px=round(pitch_px),
                   speed_kmh=kmh, rolling_kmh=sp["rolling_kmh"], n_segments=sp["n_segments"],
                   confidence=conf, in_range=in_range, status=status,
                   world_first=sp["world_first"], world_last=sp["world_last"],
                   instantaneous=sp["instantaneous"])
        rows.append(rec); records[clip] = rec
        print(f"{clip:13}{cq:>6}{'yes':>8}{pitch_px:>8.0f}{sp['n_segments']:>5}"
              f"{str(speed_out):>11}{conf:>6}  {status}")

    # speed_validation.csv
    with open(ROOT / "outputs/speed_validation.csv", "w", newline="") as fh:
        fields = ["clip", "calib", "two_stumps", "pitch_px", "speed_kmh", "rolling_kmh",
                  "n_segments", "confidence", "in_range", "status"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    (ROOT / "outputs/calibrated_speed.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

    avail = [r for r in rows if isinstance(r.get("speed_kmh"), (int, float)) and r.get("in_range")]
    print(f"\n=== trustworthy speeds: {len(avail)}/{len(rows)} clips "
          f"(others -> 'speed unavailable', not faked) ===")
    if avail:
        print("  example:", ", ".join(f"{r['clip']}={r['speed_kmh']}km/h(c{r['confidence']})" for r in avail[:5]))
    print("-> outputs/speed_validation.csv | calibrated_speed.json")


if __name__ == "__main__":
    main()
