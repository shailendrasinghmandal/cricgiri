"""
scripts/analytics_v2.py
=======================
Cricket intelligence V2 — bounce / line / length with confidence, plus pitch map
and a per-clip bowling report. No new detections, no speed.

Key principle (why this is more reliable than speed): the BOUNCE is on the
GROUND PLANE, so the pitch homography maps it to world metres EXACTLY (an
airborne ball suffers parallax; the bounce does not). So world-space bounce ->
line and length are trustworthy on calibrated clips.

Phases
------
1. Bounce V2  : fuse min-height + velocity-reversal + curvature-peak + trajectory
                consistency + support count -> bounce frame/xy + confidence.
2. World bounce: project the refined bounce through the homography (exact on the
                 ground) -> bounce_x_m (lateral), bounce_y_m (down-pitch).
3. Line V2    : bounce_x_m vs stump line -> off/middle/leg/outside/wide + conf.
4. Length V2  : distance of bounce from the BATSMAN'S crease (metres) ->
                yorker/full/good/short/bouncer + conf. (Uses world metres, not
                the detected frame, so it is not biased by late detection.)
5. Delivery quality 0-100: calibration + density + bounce support + continuity +
   line conf + length conf.
6. Pitch map  : aggregate all world bounces -> pitch_map.png + pitch_map_data.csv.
7. Bowling report: bowling_report.json per clip (line/length/bounce/speed/conf).
8. Validation : high- vs low-confidence deliveries.

Outputs: outputs/bounce_v2.csv, line_v2.csv, length_v2.csv, pitch_map.png,
         pitch_map_data.csv, bowling_report.json
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
PITCH_LEN = 20.12
STUMP_W = 0.2286


def _imp(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


dr = _imp("dr", ROOT / "scripts" / "delivery_reconstruction.py")
cs = _imp("cs", ROOT / "scripts" / "calibrated_speed.py")


def load_calib():
    p = ROOT / "outputs" / "calibration_report.csv"
    return {r["clip"]: r for r in csv.DictReader(open(p))} if p.exists() else {}


def load_speed():
    p = ROOT / "outputs" / "calibrated_speed.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ── Phase 1: bounce V2 (5-evidence fusion) ───────────────────────────────────
def bounce_v2(clean, recovered):
    pts = sorted({int(p[0]): p for p in (clean + recovered)}.values(), key=lambda p: p[0])
    if len(pts) < 4:
        return None
    xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
    fr = np.array([p[0] for p in pts])
    vy = np.gradient(ys)
    i_min = int(np.argmax(ys))                                   # 1 lowest point
    i_rev = next((i for i in range(1, len(vy)) if vy[i - 1] > 0 and vy[i] <= 0), None)  # 2 reversal
    i_cur = int(np.argmax(np.abs(np.gradient(vy)))) if len(vy) > 2 else i_min            # 3 curvature peak
    cand = [i_min, i_cur] + ([i_rev] if i_rev is not None else [])
    bi = int(round(np.median(cand)))
    bx, by, bf = float(xs[bi]), float(ys[bi]), int(fr[bi])
    # 4 trajectory consistency: residual of bounce to a quadratic through pre-pts
    consistency = 1.0
    pre = [(p[0], p[1], p[2]) for p in pts if p[0] <= bf]
    if len(pre) >= 3:
        f = np.array([p[0] for p in pre], float)
        cyf = np.polyfit(f - f[0], [p[2] for p in pre], 2)
        resid = abs(by - np.polyval(cyf, bf - f[0]))
        consistency = max(0.0, 1.0 - resid / 40.0)
    # 5 support count
    support = sum(1 for p in pts if abs(p[0] - bf) <= 2 and math.hypot(p[1] - bx, p[2] - by) <= 55)
    agree = max(0.0, 1.0 - (max(cand) - min(cand)) / 6.0)
    conf = round(min(1.0, support / 4.0) * (0.4 + 0.3 * agree + 0.3 * consistency), 2)
    return dict(frame=bf, x=round(bx, 1), y=round(by, 1), support=support,
                agreement=round(agree, 2), consistency=round(consistency, 2), confidence=conf)


# ── line / length from world coords ──────────────────────────────────────────
def classify_line_world(x_m):
    a = abs(x_m)
    if a <= 0.04:
        return "middle_stump"
    if a <= STUMP_W / 2:
        return "off_stump" if x_m < 0 else "leg_stump"     # RH-batter convention
    if a <= 0.40:
        return "outside_off" if x_m < 0 else "down_leg"
    return "wide_off" if x_m < 0 else "wide_leg"


def classify_length_world(dist_from_batsman_m):
    """Length by how far the bounce is from the BATSMAN'S crease (metres).
    Standard cricket lengths (RH reference)."""
    d = dist_from_batsman_m
    if d < 1.0:
        return "yorker"
    if d < 2.5:
        return "full"
    if d < 6.0:
        return "good_length"
    if d < 8.0:
        return "short_of_good"
    if d < 10.0:
        return "short"
    return "bouncer"          # pitched in the bowler's half -> rears up at batsman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--pitch-length", type=float, default=PITCH_LEN)
    args = ap.parse_args()
    calib = load_calib(); speed = load_speed()
    clips = args.clips or [p.parent.name for p in sorted((ROOT / "outputs/mapped").glob("*/mapped_path.csv"))
                           if p.parent.name.startswith("test_video")]
    from ultralytics import YOLO
    smodel = YOLO(str(cs.STUMP))

    bounce_rows = []; line_rows = []; length_rows = []; report = {}; pitch_pts = []
    print(f"{'clip':13}{'calib':>6}{'bounceCf':>9}{'line':>14}{'length':>15}{'quality':>8}")
    print("-" * 70)
    for clip in clips:
        cal = calib.get(clip, {}); cq = cal.get("calib_quality", "none")
        gate = {"good": 1.0, "poor": 0.55}.get(cq, 0.25)
        pts = dr.read_points(clip)
        if len(pts) < 3:
            continue
        clean = dr.clean_track(pts)
        recovered = dr.recover_post_bounce(clip, int(clean[-1][0]), (clean[-1][1], clean[-1][2])) or []
        rb = bounce_v2(clean, recovered)
        # Phase 4: independent bounce from the reconstructed pre-bounce trajectory.
        # Cross-checking it against bounce_v2 makes length depend on the fitted
        # descent, not on a single detected frame.
        recon_bounce = None
        try:
            R = dr.reconstruct(pts, clip)
            if R and R.get("bounce"):
                recon_bounce = R["bounce"]
        except Exception:
            recon_bounce = None

        # world-space bounce via homography (exact on ground plane)
        bx_m = by_m = None; line_lab = "unknown"; line_cf = 0.0
        len_lab = "unknown"; len_cf = 0.0
        if rb and cq in ("good", "poor"):
            stumps, W, H = cs.detect_both_stumps(smodel, clip)
            if stumps is not None:
                Hm, pitch_px = cs.build_homography(stumps, args.pitch_length)
                if pitch_px >= 40:
                    wx, wy = cs.to_world(Hm, rb["x"], rb["y"])
                    bx_m = round(float(wx), 3); by_m = round(float(wy), 3)
                    # length: distance of bounce from batsman crease (far stumps = pitch_len).
                    # Down-pitch is the LONG calibration baseline (20.12 m) -> reliable.
                    dist_bat = max(0.0, args.pitch_length - by_m)
                    # length conf lower if bounce_y is out of [0, pitch] (airborne/parallax)
                    in_pitch = -1.0 <= by_m <= args.pitch_length + 2.0
                    # Phase 4 cross-check: does the reconstructed pre-bounce
                    # trajectory's bounce agree with bounce_v2? Agreement -> the
                    # length is supported by the fitted descent, not one frame.
                    recon_agree = 1.0
                    if recon_bounce is not None:
                        d_rb = math.hypot(recon_bounce[0] - rb["x"], recon_bounce[1] - rb["y"])
                        recon_agree = max(0.5, 1.0 - d_rb / 80.0)   # 0px->1.0, 80px->0.5
                    if in_pitch:
                        len_lab = classify_length_world(dist_bat)
                        len_cf = round(gate * rb["confidence"] * 0.85 * recon_agree, 2)
                    else:
                        len_lab = "uncertain"          # bounce world-y off the pitch -> don't assert
                        len_cf = round(gate * rb["confidence"] * 0.25, 2)
                    # LINE from lateral world x. HONEST CAVEAT: the lateral homography is
                    # calibrated off the stump width (0.2286 m) -- an ~88x shorter baseline
                    # than the down-pitch calibration (20.12 m). A few-px stump-edge error
                    # blows up into a large lateral-metre error, so LINE is INDICATIVE only.
                    # Apply a lateral-baseline penalty and hard confidence cap.
                    LATERAL_PENALTY, LINE_CAP = 0.6, 0.45
                    line_lab = classify_line_world(bx_m)
                    line_cf = round(min(LINE_CAP, gate * rb["confidence"] * LATERAL_PENALTY), 2)
                    if in_pitch and abs(bx_m) < 1.5:    # only map plausible, on-pitch bounces
                        pitch_pts.append((clip, bx_m, dist_bat, len_cf, len_lab))

        # delivery quality 0-100
        dens = min(1.0, len(clean) / 12.0)
        cont = min(1.0, len(clean) / max(1, clean[-1][0] - clean[0][0] + 1) * 1.5)
        bsup = (rb["confidence"] if rb else 0.0)
        q = 100 * (0.25 * gate + 0.15 * dens + 0.15 * cont + 0.20 * bsup
                   + 0.125 * line_cf + 0.125 * len_cf)
        quality = round(q)

        sp = speed.get(clip, {})
        spd = sp.get("speed_kmh") if isinstance(sp.get("speed_kmh"), (int, float)) and sp.get("in_range") else None
        report[clip] = dict(
            calibration_quality=cq,
            bounce=rb or {"confidence": 0.0},
            bounce_world={"x_m": bx_m, "y_m": by_m} if bx_m is not None else None,
            line={"label": line_lab, "confidence": line_cf, "reliability": "indicative"},
            length={"label": len_lab, "confidence": len_cf, "reliability": "metric" if len_cf >= 0.4 else "low"},
            speed={"kmph": spd, "confidence": sp.get("confidence", 0.0)} if spd else {"kmph": None, "status": "unavailable"},
            delivery_quality=quality,
        )
        bounce_rows.append(dict(clip=clip, **(rb or {}), x_m=bx_m, y_m=by_m, calib=cq))
        line_rows.append(dict(clip=clip, line=line_lab, x_m=bx_m, confidence=line_cf, calib=cq))
        length_rows.append(dict(clip=clip, length=len_lab,
                                dist_from_batsman_m=(round(args.pitch_length - by_m, 2) if by_m is not None else None),
                                confidence=len_cf, calib=cq))
        print(f"{clip:13}{cq:>6}{(rb['confidence'] if rb else 0):>9}{line_lab:>14}({line_cf:.2f})"
              f"{len_lab:>11}({len_cf:.2f}){quality:>8}")

    # CSVs
    def wcsv(name, rows, fields):
        with open(ROOT / "outputs" / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
            for r in rows:
                w.writerow(r)
    wcsv("bounce_v2.csv", bounce_rows, ["clip", "frame", "x", "y", "x_m", "y_m", "support", "agreement", "consistency", "confidence", "calib"])
    wcsv("line_v2.csv", line_rows, ["clip", "line", "x_m", "confidence", "calib"])
    wcsv("length_v2.csv", length_rows, ["clip", "length", "dist_from_batsman_m", "confidence", "calib"])
    report_out = {
        "_methodology": {
            "length": "World-space: bounce projected through pitch homography; distance "
                      "from batsman crease (down-pitch). Long calibration baseline "
                      "(20.12 m) -> METRIC reliability. NOT derived from the detected "
                      "bounce frame index, so no late-detection 'yorker' bias.",
            "line": "World-space lateral x of bounce. Calibrated off stump width "
                    "(0.2286 m) -- ~88x shorter baseline -> high variance. INDICATIVE "
                    "only; confidence hard-capped at 0.45.",
            "bounce": "5-evidence fusion: min-height + velocity-reversal + curvature-peak "
                      "+ pre-bounce trajectory consistency + support count.",
            "speed": "From calibrated_speed.json (airborne parallax-limited); reported "
                     "only where in_range, else 'unavailable'.",
            "gating": "All metric analytics gated by calibration_report.csv quality "
                      "(good=1.0 / poor=0.55 / none=0.25). Bounces with world-y off the "
                      "pitch are marked 'uncertain' rather than asserted.",
        },
        "clips": report,
    }
    (ROOT / "outputs" / "bowling_report.json").write_text(json.dumps(report_out, indent=2, default=str), encoding="utf-8")

    # Phase 6: pitch map
    draw_pitch_map(pitch_pts, args.pitch_length)
    with open(ROOT / "outputs" / "pitch_map_data.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "bounce_x_m", "dist_from_batsman_m", "confidence", "length"])
        for r in pitch_pts:
            w.writerow(r)

    # Phase 8: validation
    hi = [c for c, r in report.items() if r["delivery_quality"] >= 55]
    lo = [c for c, r in report.items() if r["delivery_quality"] < 35]
    print(f"\n=== VALIDATION ===")
    print(f"  high-confidence deliveries ({len(hi)}): {', '.join(sorted(hi)) or '-'}")
    print(f"  low-confidence deliveries  ({len(lo)}): {', '.join(sorted(lo)) or '-'}")
    print(f"  pitch-mapped bounces: {len(pitch_pts)}")
    print("-> outputs/bounce_v2.csv | line_v2.csv | length_v2.csv | bowling_report.json | pitch_map.png")


def draw_pitch_map(pitch_pts, pitch_len):
    """Top-down pitch: 22 yards long, ~3m wide play area. Plot bounce (x_m, dist
    from batsman). Stumps at both ends, good-length band shaded."""
    Wm, Lm = 3.0, pitch_len
    scale = 60                      # px per metre (length); width scaled to fit
    H = int(Lm * scale * 0.45) + 80
    Wpx = 360
    img = np.full((H, Wpx, 3), 40, np.uint8)
    x0 = Wpx // 2
    def to_px(x_m, dist_m):
        px = int(x0 + x_m / (Wm / 2) * (Wpx * 0.42))
        py = int(40 + (dist_m / Lm) * (H - 80))
        return px, py
    # pitch strip
    cv2.rectangle(img, (int(x0 - Wpx * 0.30), 40), (int(x0 + Wpx * 0.30), H - 40), (70, 90, 70), -1)
    # good-length band (2.5-6 m from batsman)
    p1 = to_px(0, 2.5); p2 = to_px(0, 6.0)
    cv2.rectangle(img, (int(x0 - Wpx * 0.30), p1[1]), (int(x0 + Wpx * 0.30), p2[1]), (60, 120, 60), -1)
    # stumps: batsman (dist 0, top), bowler (dist pitch_len, bottom)
    for dist in (0.0, pitch_len):
        c = to_px(0, dist)
        for dx in (-0.11, 0, 0.11):
            cv2.line(img, to_px(dx, dist), (to_px(dx, dist)[0], to_px(dx, dist)[1] - 16), (200, 200, 230), 2)
    cv2.putText(img, "BATSMAN", (x0 - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, "good length", (x0 + 10, (p1[1] + p2[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
    # bounces
    for (clip, x_m, dist, conf, lab) in pitch_pts:
        px, py = to_px(x_m, dist)
        col = (0, int(80 + 175 * conf), int(255 * (1 - conf)))   # red(low)->green(high)
        cv2.circle(img, (px, py), 7, col, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Pitch map: {len(pitch_pts)} bounces (green=high conf)", (8, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(ROOT / "outputs" / "pitch_map.png"), img)


if __name__ == "__main__":
    main()
