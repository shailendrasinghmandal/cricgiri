"""
scripts/speed_audit.py   [ADDITIVE — Lever 3: speed error-budget + hardening]
============================================================================
FullTrack's weakest metric is speed (systematic overestimate, radar-corrected in
post). This audits OUR speed end-to-end against a labelled clip and reports how many
km/h each stage contributes, so we fix the biggest one first.

Stages audited (release->bounce time-of-flight: speed = distance / time):
  1. RELEASE / first-tracked-frame  -> sets the flight TIME (bounce_f - release_f)
  2. FRAME TIMING (fps)             -> is phone video actually constant fps?
  3. CALIBRATION SCALE (homography) -> sets the down-pitch DISTANCE in metres

Also compares the fragile two-frame endpoint method to a robust multi-frame
release-window fit (least-squares down-pitch speed over the pre-bounce points),
which the brief prefers.

  venv/Scripts/python.exe scripts/speed_audit.py --stem clip01
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PITCH_LEN = 20.12


def _imp(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


def load_gt(stem):
    rows = {int(r["frame"]): (float(r["x"]), float(r["y"]), int(r["visible"]), int(r["is_bounce"]))
            for r in csv.DictReader(open(ROOT / "gt" / f"{stem}.csv"))}
    vis = {f: (x, y) for f, (x, y, v, b) in rows.items() if v}
    bounce = next(((f, x, y) for f, (x, y, v, b) in rows.items() if b and v), None)
    return vis, bounce


def fps_constancy(video):
    """Read per-frame timestamps; report nominal fps vs the actual inter-frame delta
    distribution (phone video can be variable-frame-rate)."""
    cap = cv2.VideoCapture(str(video))
    nominal = cap.get(cv2.CAP_PROP_FPS)
    ts = []
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    ts = np.array([t for t in ts if t > 0])
    if len(ts) < 3:
        return nominal, None, None
    dt = np.diff(ts)
    dt = dt[(dt > 0) & (dt < 1000)]
    return nominal, float(np.mean(dt)), float(np.std(dt))


def speed_tof(dist_m, frames_flight, fps):
    if frames_flight <= 0:
        return None
    return dist_m / (frames_flight / fps) * 3.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="clip01")
    args = ap.parse_args()
    stem = args.stem

    cs = _imp("cs", ROOT / "scripts" / "calibrated_speed.py")
    vis, bounce = load_gt(stem)
    if not bounce:
        raise SystemExit("no bounce in GT")
    video = ROOT / "eval" / f"{stem}.mp4"

    # homography for the down-pitch distance
    tmp = ROOT / "videos" / f"{stem}.mp4"; made = False
    if not tmp.exists():
        shutil.copy(video, tmp); made = True
    try:
        from ultralytics import YOLO
        st, _, _ = cs.detect_both_stumps(YOLO(str(cs.STUMP)), stem)
        Hm, ppx = cs.build_homography(st, PITCH_LEN) if st is not None else (None, 0)
    finally:
        if made:
            tmp.unlink(missing_ok=True)
    if Hm is None:
        raise SystemExit("no homography")

    nominal_fps, mean_dt, std_dt = fps_constancy(video)
    fps = 1000.0 / mean_dt if mean_dt else nominal_fps

    # GT geometry
    rf = min(vis); bf = bounce[0]; flight = bf - rf
    bx, by = cs.to_world(Hm, bounce[1], bounce[2])
    dist = max(1.0, min(abs(by), PITCH_LEN) - 1.0)
    base_speed = speed_tof(dist, flight, fps)

    print(f"\n=== SPEED AUDIT — {stem} ===")
    print(f"release frame {rf}, bounce frame {bf}, flight {flight} frames | "
          f"down-pitch dist {dist:.2f} m | fps {fps:.1f}")
    print(f"baseline TOF speed = {base_speed:.1f} km/h" if base_speed else "TOF undefined")

    print("\n--- 1. FRAME TIMING (fps constancy) ---")
    print(f"nominal fps {nominal_fps:.2f} | measured mean inter-frame {mean_dt:.2f} ms "
          f"(std {std_dt:.2f} ms)")
    jitter_pct = (std_dt / mean_dt * 100) if mean_dt else 0
    print(f"frame-timing jitter {jitter_pct:.1f}%  -> ~{jitter_pct/100*base_speed:.1f} km/h "
          f"if uncorrected" if base_speed else "")
    print(f"  verdict: {'VARIABLE fps — timing contributes error' if jitter_pct > 3 else 'effectively constant fps — negligible'}")

    print("\n--- 2. RELEASE / BOUNCE FRAME sensitivity ---")
    for df in (1, 2, 3):
        s_lo = speed_tof(dist, flight + df, fps); s_hi = speed_tof(dist, max(1, flight - df), fps)
        print(f"  release off by +/-{df} frame(s): {s_hi:.0f} .. {s_lo:.0f} km/h "
              f"(+/-{abs(s_hi - base_speed):.0f} km/h)")
    per_frame = abs(speed_tof(dist, flight - 1, fps) - base_speed)
    print(f"  => ~{per_frame:.0f} km/h PER FRAME of release error  (dominant term)")

    print("\n--- 3. CALIBRATION SCALE sensitivity ---")
    for pct in (5, 10, 20):
        s = speed_tof(dist * (1 + pct / 100), flight, fps)
        print(f"  distance off by +{pct}%: {s:.0f} km/h (+{s - base_speed:.0f} km/h)")

    print("\n--- 4. ROBUST multi-frame window vs two-frame endpoints ---")
    # least-squares down-pitch speed over the pre-bounce GT points (uses ALL points)
    pre = sorted((f, *cs.to_world(Hm, x, y)) for f, (x, y) in vis.items() if f <= bf)
    if len(pre) >= 4:
        fr = np.array([p[0] for p in pre], float); ym = np.array([abs(p[2]) for p in pre], float)
        # robust: clip to physical pitch, fit line ym = a*frame + b
        ym = np.clip(ym, 0, PITCH_LEN)
        a, b = np.polyfit(fr, ym, 1)
        win_speed = abs(a) * fps * 3.6
        print(f"  window fit over {len(pre)} pre-bounce pts: slope {a:.3f} m/frame "
              f"-> {win_speed:.0f} km/h")
        print(f"  two-frame endpoint TOF: {base_speed:.0f} km/h")
        print(f"  => window method is less sensitive to a single wrong endpoint frame")

    print("\n=== ERROR BUDGET (ranked) ===")
    budget = [("release/bounce frame", per_frame * 2, "±2 frame typical detection error"),
              ("calibration scale", speed_tof(dist * 1.1, flight, fps) - base_speed, "±10% homography"),
              ("frame timing (fps)", jitter_pct / 100 * base_speed, f"{jitter_pct:.1f}% jitter")]
    budget.sort(key=lambda x: -abs(x[1]))
    for name, km, note in budget:
        print(f"  {name:22} ~{abs(km):5.0f} km/h   ({note})")
    print("\nFIX PRIORITY: the top row is the biggest lever. For release/bounce that means "
          "a COMPLETE track (don't let detection miss the early release frames) and/or the "
          "window-fit above instead of two-frame endpoints.")


if __name__ == "__main__":
    main()
