"""
scripts/eval_harness.py   [ADDITIVE — Phase 0 measurement harness]
===================================================================
Objective before/after evaluation against hand-clicked ground truth
(created with scripts/gt_label.py -> gt/<stem>.csv).

Per delivery it reports:
  * detector_recall / detector_precision   (raw per-frame detections vs GT)
  * track_recall, missed_frames            (final delivery track vs GT)
  * false_positives_raw / false_positives_track
  * traj_rmse_px  and  traj_rmse_mm        (world mm via the stump homography;
       NOTE the homography is a GROUND-PLANE map — airborne points carry
       parallax, so mm RMSE is exact only near the bounce. Both numbers are
       printed; pixel RMSE is the parallax-free one.)
  * bounce_frame_err / bounce_err_px / bounce_err_mm

Artifacts read (produced by the offline pipeline / run_demo_testing):
  outputs/detections/<stem>/detections.csv    raw ensemble detections
  outputs/mapped/<stem>/mapped_path.csv       final clean track
If missing, they are generated in-process with the production config.

Usage:
  venv/Scripts/python.exe scripts/eval_harness.py --video testing/clip.mp4 --tag baseline
  venv/Scripts/python.exe scripts/eval_harness.py --video ... --tag phase1
  venv/Scripts/python.exe scripts/eval_harness.py --compare baseline phase1
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "gt"
OUT_DIR = ROOT / "outputs" / "eval_harness"
MATCH_PX_DEFAULT = 14.0
PITCH_LEN = 20.12


def _imp(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def load_gt(stem: str):
    p = GT_DIR / f"{stem}.csv"
    if not p.exists():
        raise SystemExit(f"[ERROR] no ground truth at {p} — run scripts/gt_label.py first")
    rows = list(csv.DictReader(open(p)))
    gt = {int(r["frame"]): (float(r["x"]), float(r["y"]), int(r["visible"]), int(r["is_bounce"]))
          for r in rows}
    return gt


def load_csv_pts(p: Path, fcol="frame"):
    if not p.exists():
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        f = int(float(r[fcol]))
        out.setdefault(f, []).append((float(r["x"]), float(r["y"])))
    return out


def ensure_artifacts(video: Path, stem: str, ball_model=None, alt_model=None, force=False):
    """Generate detections + mapped track with the production offline config.

    ball_model/alt_model override the default ensemble so a candidate (e.g. a
    blur-finetuned model) can be evaluated. force=True regenerates even if present."""
    det_p = ROOT / "outputs" / "detections" / stem / "detections.csv"
    map_p = ROOT / "outputs" / "mapped" / stem / "mapped_path.csv"
    if det_p.exists() and map_p.exists() and not force and ball_model is None:
        return det_p, map_p
    mt = _imp("mt", ROOT / "scripts" / "map_trajectory.py")
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    primary = ball_model or str(ROOT / "models" / "ball_ft_t4.pt")
    alt = alt_model or str(ROOT / "models" / "ball_best_leather_new.pt")
    models = mt.load_models(primary, [alt])
    dets, W, H, total, fps = mt.detect_all(models, str(video), 0.05, 1280, device)
    inl, *_ = mt.ransac_trajectory(dets, 34.0, W=W, H=H)
    det_p.parent.mkdir(parents=True, exist_ok=True)
    with open(det_p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "x", "y", "conf"])
        w.writerows([[d[0], round(d[1], 1), round(d[2], 1), round(d[3], 3)] for d in dets])
    map_p.parent.mkdir(parents=True, exist_ok=True)
    with open(map_p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "x", "y", "conf"]); w.writerows(inl)
    return det_p, map_p


def homography_for(video: Path, stem: str):
    """Ground-plane homography via the existing stump calibration (or None)."""
    cs = _imp("cs", ROOT / "scripts" / "calibrated_speed.py")
    from ultralytics import YOLO
    tmp = ROOT / "videos" / f"{stem}.mp4"
    made = False
    if not tmp.exists():
        tmp.parent.mkdir(exist_ok=True)
        shutil.copy(video, tmp)
        made = True
    try:
        smodel = YOLO(str(cs.STUMP))
        st, _, _ = cs.detect_both_stumps(smodel, stem)
        if st is None:
            return None, None
        Hm, ppx = cs.build_homography(st, PITCH_LEN)
        return (Hm if ppx >= 40 else None), cs
    finally:
        if made:
            tmp.unlink(missing_ok=True)


def evaluate(video: Path, match_px: float, ball_model=None, force=False) -> dict:
    stem = video.stem
    gt = load_gt(stem)
    det_p, map_p = ensure_artifacts(video, stem, ball_model=ball_model, force=force)
    dets = load_csv_pts(det_p)
    track = load_csv_pts(map_p)
    track_1 = {f: pts[0] for f, pts in track.items()}         # one point per frame

    vis = {f: (x, y) for f, (x, y, v, b) in gt.items() if v}
    gt_bounce = next(((f, x, y) for f, (x, y, v, b) in gt.items() if b and v), None)

    # detector recall / precision on labeled frames
    det_hit = det_fp = 0
    for f, (gx, gy) in vis.items():
        cand = dets.get(f, [])
        if any(math.hypot(x - gx, y - gy) <= match_px for x, y in cand):
            det_hit += 1
    all_lab = set(gt.keys())
    total_dets_on_labeled = 0
    for f in all_lab:
        for (x, y) in dets.get(f, []):
            total_dets_on_labeled += 1
            g = vis.get(f)
            if g is None or math.hypot(x - g[0], y - g[1]) > match_px:
                det_fp += 1

    # track vs GT
    tr_hit, missed, tr_fp, resid_px = 0, [], 0, []
    world_pairs = []
    for f, (gx, gy) in vis.items():
        t = track_1.get(f)
        if t is None:
            missed.append(f)
            continue
        d = math.hypot(t[0] - gx, t[1] - gy)
        if d <= match_px:
            tr_hit += 1
        resid_px.append(d)
        world_pairs.append((f, t, (gx, gy)))
    for f, t in track_1.items():
        g = vis.get(f)
        if g is not None and math.hypot(t[0] - g[0], t[1] - g[1]) > match_px:
            tr_fp += 1

    # world-mm RMSE (parallax caveat) + bounce error
    Hm, cs = homography_for(video, stem)
    rmse_mm = bounce_err_mm = None
    if Hm is not None:
        w_res = []
        for f, t, g in world_pairs:
            tw = cs.to_world(Hm, t[0], t[1]); gw = cs.to_world(Hm, g[0], g[1])
            w_res.append(math.hypot(tw[0] - gw[0], tw[1] - gw[1]))
        if w_res:
            rmse_mm = round(float(np.sqrt(np.mean(np.square(w_res)))) * 1000, 1)

    # bounce prediction from the track (same detector production uses)
    dr = _imp("dr", ROOT / "scripts" / "delivery_reconstruction.py")
    tr_sorted = sorted((f, x, y, 1.0) for f, (x, y) in track_1.items())
    bi = dr.detect_bounce_index([list(p) for p in tr_sorted]) if len(tr_sorted) >= 6 else None
    if isinstance(bi, (tuple, list)):            # some versions return (index, ...)
        bi = bi[0] if bi else None
    bounce_frame_err = bounce_err_px = None
    if gt_bounce and bi is not None and 0 <= bi < len(tr_sorted):
        bf, bx, by = tr_sorted[bi][0], tr_sorted[bi][1], tr_sorted[bi][2]
        bounce_frame_err = int(abs(bf - gt_bounce[0]))
        bounce_err_px = round(math.hypot(bx - gt_bounce[1], by - gt_bounce[2]), 1)
        if Hm is not None:
            pw = cs.to_world(Hm, bx, by); gw = cs.to_world(Hm, gt_bounce[1], gt_bounce[2])
            bounce_err_mm = round(math.hypot(pw[0] - gw[0], pw[1] - gw[1]) * 1000, 1)

    n_vis = len(vis)
    return dict(
        clip=stem, labeled_frames=len(gt), visible_frames=n_vis, match_px=match_px,
        detector_recall=round(det_hit / n_vis, 3) if n_vis else None,
        detector_precision=round((total_dets_on_labeled - det_fp) / total_dets_on_labeled, 3)
            if total_dets_on_labeled else None,
        track_recall=round(tr_hit / n_vis, 3) if n_vis else None,
        missed_frames=len(missed),
        false_positives_raw=det_fp, false_positives_track=tr_fp,
        traj_rmse_px=round(float(np.sqrt(np.mean(np.square(resid_px)))), 1) if resid_px else None,
        traj_rmse_mm=rmse_mm,
        bounce_frame_err=bounce_frame_err, bounce_err_px=bounce_err_px,
        bounce_err_mm=bounce_err_mm,
        homography=("ok" if Hm is not None else "unavailable"),
    )


METRICS = ["detector_recall", "detector_precision", "track_recall", "missed_frames",
           "false_positives_raw", "false_positives_track", "traj_rmse_px",
           "traj_rmse_mm", "bounce_frame_err", "bounce_err_px", "bounce_err_mm"]


def print_table(rows: list[dict], title: str):
    print(f"\n=== {title} ===")
    hdr = ["clip"] + METRICS
    widths = [max(len(h), 8) for h in hdr]
    print("  ".join(h[:w].ljust(w) for h, w in zip(hdr, widths)))
    for r in rows:
        cells = [str(r.get("clip", ""))[:widths[0]].ljust(widths[0])]
        for m, w in zip(METRICS, widths[1:]):
            v = r.get(m)
            cells.append(("-" if v is None else str(v)).ljust(w))
        print("  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", default=[], help="video path (repeatable)")
    ap.add_argument("--gt-dir", default=None, help="evaluate every clip that has a gt csv")
    ap.add_argument("--videos-dir", default="testing", help="where clips live for --gt-dir mode")
    ap.add_argument("--tag", default="baseline", help="name this run for later --compare")
    ap.add_argument("--match-px", type=float, default=MATCH_PX_DEFAULT)
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    ap.add_argument("--ball-model", default=None,
                    help="override the primary ball model (e.g. a blur-finetuned .pt) "
                         "and force detection regeneration for this run")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        a, b = args.compare
        ra = json.loads((OUT_DIR / f"{a}.json").read_text())
        rb = json.loads((OUT_DIR / f"{b}.json").read_text())
        byclip = {r["clip"]: r for r in rb}
        print(f"\n=== BEFORE ({a})  vs  AFTER ({b}) ===")
        for r in ra:
            o = byclip.get(r["clip"])
            if not o:
                continue
            print(f"\n{r['clip']}:")
            for m in METRICS:
                va, vb = r.get(m), o.get(m)
                delta = ""
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    delta = f"  (d {vb - va:+.3g})"
                print(f"  {m:24} {va} -> {vb}{delta}")
        return

    vids = [Path(v) for v in args.video]
    if args.gt_dir or not vids:
        vdir = ROOT / args.videos_dir
        vids = [vdir / f"{p.stem}.mp4" for p in sorted(GT_DIR.glob("*.csv"))
                if (vdir / f"{p.stem}.mp4").exists()]
    if not vids:
        raise SystemExit("no videos to evaluate (label some clips with gt_label.py first)")

    rows = [evaluate(v, args.match_px, ball_model=args.ball_model,
                     force=bool(args.ball_model)) for v in vids]
    print_table(rows, f"run '{args.tag}'  (match radius {args.match_px}px)")
    (OUT_DIR / f"{args.tag}.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {OUT_DIR / (args.tag + '.json')}")
    print("NOTE: traj_rmse_mm uses the ground-plane homography — exact at the bounce, "
          "parallax-inflated for airborne points. Use traj_rmse_px for airborne accuracy.")


if __name__ == "__main__":
    main()
