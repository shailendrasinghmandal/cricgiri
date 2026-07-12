"""
scripts/calib_correct.py   [ADDITIVE — Lever 1: bias / calibration correction]
==============================================================================
Mirror what FullTrack AI does in post: measure the SYSTEMATIC error of each output
metric against hand-labelled ground truth, and (optionally) correct a consistent
bias with an offset/scale fit on a TRAIN split and reported on a HELD-OUT split.

Ground-truthable metrics from gt/<stem>.csv (per-frame ball px + bounce flag):
  * bounce_x_m   (lateral, drives LINE)      pipeline vs homography(GT bounce px)
  * bounce_y_m   (down-pitch, drives LENGTH) pipeline vs homography(GT bounce px)
  * speed_kmph   pipeline vs GT release->bounce time-of-flight (geometry, not radar)

For each metric we report mean SIGNED error (bias), std (spread), and whether the
error looks constant (offset) or value-scaling (scale). With N clips we do
LEAVE-ONE-OUT: fit the correction on the other clips, apply to the held-out clip,
and show before/after |error|. Everything is read-only + reversible; the pipeline
is untouched (this is a post-hoc report / optional corrector).

  venv/Scripts/python.exe scripts/calib_correct.py                 # measure + LOO
  venv/Scripts/python.exe scripts/calib_correct.py --calib-correct # print fitted offsets/scales
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "gt"
PITCH_LEN = 20.12


def _imp(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


def load_gt(stem):
    rows = {int(r["frame"]): (float(r["x"]), float(r["y"]), int(r["visible"]), int(r["is_bounce"]))
            for r in csv.DictReader(open(GT_DIR / f"{stem}.csv"))}
    vis = {f: (x, y) for f, (x, y, v, b) in rows.items() if v}
    bounce = next(((f, x, y) for f, (x, y, v, b) in rows.items() if b and v), None)
    return vis, bounce


def gt_reference(stem, vis, bounce, cs, av2, fps):
    """GT-derived world metrics via the same stump homography the pipeline uses."""
    tmp = ROOT / "videos" / f"{stem}.mp4"
    made = False
    if not tmp.exists():
        shutil.copy(ROOT / "eval" / f"{stem}.mp4", tmp); made = True
    try:
        from ultralytics import YOLO
        smodel = YOLO(str(cs.STUMP))
        st, _, _ = cs.detect_both_stumps(smodel, stem)
        if st is None:
            return None
        Hm, ppx = cs.build_homography(st, PITCH_LEN)
        if ppx < 40:
            return None
    finally:
        if made:
            tmp.unlink(missing_ok=True)
    if not bounce:
        return None
    bx, by = cs.to_world(Hm, bounce[1], bounce[2])
    # GT speed = release->bounce time-of-flight from the GT track (geometry, not radar)
    f0 = min(vis); bf = bounce[0]
    dist = max(1.0, min(abs(by), PITCH_LEN) - 1.0)
    dur = max(1, bf - f0) / fps
    gt_speed = dist / dur * 3.6
    return dict(bounce_x_m=round(float(bx), 3), bounce_y_m=round(float(by), 3),
                length_m=round(float(max(0.0, PITCH_LEN - by)), 2), speed_kmph=round(gt_speed, 1))


def pipeline_output(stem, rdt):
    res = rdt.analyze_video(str(ROOT / "eval" / f"{stem}.mp4"))
    if not res.get("deliveries"):
        return None
    d = res["deliveries"][0]
    wb = d.get("bounce_world") or {}
    return dict(bounce_x_m=wb.get("x_m"), bounce_y_m=wb.get("y_m"),
                length_m=(d.get("length") or {}).get("distance_from_batsman_m"),
                speed_kmph=d.get("speed_kmph"))


METRICS = ["speed_kmph", "bounce_x_m", "bounce_y_m", "length_m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-correct", action="store_true",
                    help="print the fitted offset/scale corrections")
    args = ap.parse_args()

    rdt = _imp("rdt", ROOT / "scripts" / "run_demo_testing.py")
    cs = rdt.cs; av2 = rdt.av2
    import cv2

    stems = sorted(p.stem for p in GT_DIR.glob("*.csv"))
    per = {}                                            # stem -> dict(metric -> (pred, gt))
    for stem in stems:
        vis, bounce = load_gt(stem)
        cap = cv2.VideoCapture(str(ROOT / "eval" / f"{stem}.mp4")); fps = cap.get(5) or 60.0; cap.release()
        gt = gt_reference(stem, vis, bounce, cs, av2, fps)
        pr = pipeline_output(stem, rdt)
        if not gt or not pr:
            print(f"[skip] {stem}: gt={bool(gt)} pred={bool(pr)}")
            continue
        per[stem] = {m: (pr.get(m), gt.get(m)) for m in METRICS}

    if not per:
        raise SystemExit("no clips with both pipeline output and GT reference")

    # ── raw bias table ─────────────────────────────────────────────────────
    print("\n=== RAW ERROR vs GROUND TRUTH (per clip) ===")
    print(f"{'clip':10}" + "".join(f"{m:>16}" for m in METRICS))
    for stem in per:
        cells = []
        for m in METRICS:
            p, g = per[stem][m]
            cells.append(f"{p}->{g}" if (p is not None and g is not None) else "-")
        print(f"{stem:10}" + "".join(f"{c:>16}" for c in cells))

    print("\n=== SYSTEMATIC BIAS (mean signed error = pred - gt) ===")
    print(f"{'metric':14}{'n':>4}{'bias(mean)':>13}{'spread(std)':>13}{'verdict':>26}")
    biases = {}
    for m in METRICS:
        errs = [per[s][m][0] - per[s][m][1] for s in per
                if per[s][m][0] is not None and per[s][m][1] is not None]
        if not errs:
            print(f"{m:14}{0:>4}{'no data':>13}"); continue
        bias = float(np.mean(errs)); spread = float(np.std(errs))
        biases[m] = bias
        verdict = ("consistent -> correctable" if abs(bias) > spread * 0.8 and len(errs) > 1
                   else "noisy/uncertain (need more clips)")
        print(f"{m:14}{len(errs):>4}{bias:>+13.2f}{spread:>13.2f}{verdict:>26}")

    # ── leave-one-out offset correction ────────────────────────────────────
    if len(per) >= 2:
        print("\n=== LEAVE-ONE-OUT OFFSET CORRECTION (before -> after |error|) ===")
        for m in METRICS:
            rows = [(s, per[s][m][0], per[s][m][1]) for s in per
                    if per[s][m][0] is not None and per[s][m][1] is not None]
            if len(rows) < 2:
                continue
            before, after = [], []
            for i, (s, p, g) in enumerate(rows):
                others = [rows[j][1] - rows[j][2] for j in range(len(rows)) if j != i]
                off = float(np.mean(others))            # bias learned from the OTHER clips
                before.append(abs(p - g)); after.append(abs((p - off) - g))
            print(f"  {m:12} mean|err| {np.mean(before):6.2f} -> {np.mean(after):6.2f}"
                  f"   ({'improved' if np.mean(after) < np.mean(before) else 'no gain'})")

    if args.calib_correct:
        print("\n=== FITTED CORRECTIONS (apply as: corrected = pred - offset) ===")
        for m, b in biases.items():
            print(f"  {m:12} offset = {b:+.2f}")
        print("  NOTE: fit on all clips; with only", len(per),
              "clips these are indicative — refit as you label more.")


if __name__ == "__main__":
    main()
