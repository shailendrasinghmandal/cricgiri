"""
scripts/run_demo_testing.py   [ADDITIVE — demo output for testing/ videos]
=========================================================================
Produces the BEST-quality output for every clip in testing/, using our PROVEN
components (NOT the old online tracker that gave jumpy tracks):

  low-conf(0.05) ensemble detection (GPU)  -> high recall, FULLTRACK is able to,
  offline global motion-consistency mapping -> rejects FP, clean ball arc,
  physics-validity gate                      -> no impossible points,
  segmented projectile reconstruction        -> smooth release->bat curve + render,
  homography analytics                        -> bounce / line / length / speed.

Per clip -> testing_result_new/clipNN/:
  trajectory.mp4      broadcast trajectory video (premium, no debug)
  track_overlay.jpg   detected ball points + arc on a frame (proof)
  clipNN.json         full PDF-aligned analytics JSON
  clipNN.csv          per-point track table
Plus testing_result_new/manifest.csv + summary.md.

Run on GPU: venv/Scripts/python.exe scripts/run_demo_testing.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

import os

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "venv" / "Scripts" / "python.exe"
SRC = ROOT / "testing"
DST = ROOT / "testing_result_new"
# Defaults = the validated local (GPU) config. On a small CPU cloud host, override via
# env so the API fits free-tier RAM/CPU: CRICGIRI_IMGSZ=640 and CRICGIRI_BALL_MODELS=ball_best.pt.
CONF = float(os.environ.get("CRICGIRI_CONF", "0.05"))   # LOW conf = high recall
IMGSZ = int(os.environ.get("CRICGIRI_IMGSZ", "1280"))
PITCH_LEN = float(os.environ.get("CRICGIRI_PITCH_LEN", "20.12"))
# Ball detector ensemble (comma-separated names under models/). Missing files are skipped;
# if none are present it falls back to the committed models/ball_best.pt so a cloud image
# that only carries the small committed weights still runs.
_BALL_MODELS_ENV = os.environ.get(
    "CRICGIRI_BALL_MODELS", "ball_ft_t4.pt,ball_best_leather_new.pt")


def _imp(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


dr = _imp("dr", ROOT / "scripts" / "delivery_reconstruction.py")
cs = _imp("cs", ROOT / "scripts" / "calibrated_speed.py")
av2 = _imp("av2", ROOT / "scripts" / "analytics_v2.py")
pg = _imp("pg", ROOT / "scripts" / "physics_gate_v2.py")
mt = _imp("mt", ROOT / "scripts" / "map_trajectory.py")   # in-process mapping for the API

import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from analytics.speed_estimation import SpeedEstimator   # robust release->bounce v=d/t


def clean_for_render(pts):
    """Keep the REAL detected ball points all the way to the bat (release ->
    bounce -> bat). Only remove physically-impossible points (horizontal-direction
    reversals / gap-jumps via the physics gate). No truncation at the bounce, no
    synthetic points — just the detected coordinates."""
    pts = sorted(pts, key=lambda p: p[0])
    if len(pts) < 4:
        return pts
    filt, _, _, _ = pg.physics_filter(pts)
    return filt if len(filt) >= 3 else pts


def smooth_xy(pts):
    """Catmull-Rom interpolating spline THROUGH the detected points (release ->
    bounce -> bat). Local & interpolating, so it never overshoots into loops/
    V-hooks the way a global smoothing spline does."""
    P = [np.array([p[1], p[2]], float) for p in pts]
    Q = [P[0]]
    for p in P[1:]:
        if np.hypot(*(p - Q[-1])) > 1.5:           # drop near-duplicate coords
            Q.append(p)
    P = Q
    if len(P) < 3:
        return [tuple(p) for p in P]
    pad = [P[0]] + P + [P[-1]]
    out = []
    for i in range(1, len(pad) - 2):
        p0, p1, p2, p3 = pad[i - 1], pad[i], pad[i + 1], pad[i + 2]
        for t in np.linspace(0, 1, 24, endpoint=False):
            t2, t3 = t * t, t * t * t
            pt = 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append((float(pt[0]), float(pt[1])))
    out.append((float(P[-1][0]), float(P[-1][1])))
    return out


def render_clean(stem, render_pts, out_path, thick=8):
    """Draw a smooth spline THROUGH the detected points (release -> bounce -> bat).
    No reconstruct/parabola extension (that caused the V-hook)."""
    video = ROOT / "videos" / f"{stem}.mp4"
    if not video.exists() or len(render_pts) < 3:
        return False, None
    curve = smooth_xy(render_pts)
    if len(curve) < 2:
        return False, None
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0
    f_lo = render_pts[0][0]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f >= f_lo - 3:
            dr.draw_arc(frame, curve, thick)
        vw.write(frame); f += 1
    cap.release(); vw.release()
    return True, None


def detect_all_frames(models, video, device):
    cap = cv2.VideoCapture(str(video))
    rows = []; f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        for m in models:
            r = m.predict(im, conf=CONF, imgsz=IMGSZ, device=device, verbose=False)[0]
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                rows.append([f, round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1),
                             round(float(b.conf[0]), 3), round(x2 - x1, 1), round(y2 - y1, 1)])
        f += 1
    cap.release()
    return rows, f


def analytics_for(stem, render_pts, fps, W, H, smodel, device, pitch_len=PITCH_LEN):
    """bounce / line / length / speed from the CLEAN track (no noisy recovery).

    `pitch_len` (metres) is the down-pitch scale used to build the homography AND
    the release->bounce speed distance. Supplying the REAL pitch length for a clip
    directly sets the metric scale, so speed accuracy scales with how well you know it.
    """
    clean = list(render_pts); recovered = []
    rb = av2.bounce_v2(clean, recovered)
    out = {"bounce": None, "line": {"label": "unknown", "confidence": 0.0},
           "length": {"label": "unknown", "confidence": 0.0},
           "speed": {"kmph": None, "status": "unavailable"}, "world_bounce": None}
    # stumps + homography
    st, Wd, Hd = cs.detect_both_stumps(smodel, stem)
    Hm = None; ppx = None
    if st is not None:
        Hm, ppx = cs.build_homography(st, pitch_len)
        if ppx < 40:
            Hm = None
    if rb:
        out["bounce"] = rb
        if Hm is not None:
            wx, wy = cs.to_world(Hm, rb["x"], rb["y"])
            out["world_bounce"] = {"x_m": round(float(wx), 3), "y_m": round(float(wy), 3)}
            in_pitch = -1.0 <= wy <= pitch_len + 2
            dist_bat = max(0.0, pitch_len - wy)
            if in_pitch:
                out["length"] = {"label": av2.classify_length_world(dist_bat),
                                 "confidence": round(0.85 * rb["confidence"], 2),
                                 "dist_from_batsman_m": round(dist_bat, 2)}
            else:
                out["length"] = {"label": "uncertain", "confidence": round(0.25 * rb["confidence"], 2)}
            out["line"] = {"label": av2.classify_line_world(wx),
                           "confidence": round(min(0.45, 0.6 * rb["confidence"]), 2),
                           "reliability": "indicative"}
    # speed — release->bounce TIME-OF-FLIGHT (robust to airborne parallax).
    #   distance = down-pitch travel to the bounce (the bounce is ON the ground plane,
    #     so its world-y is homography-exact) minus a ~1 m release offset,
    #   time     = (bounce_frame - release_frame) / fps.
    # The per-segment homography method was replaced here: airborne ball points map to
    # wildly wrong world positions (parallax), giving 1e3-1e5 km/h segments that the
    # sanity band then discards -> speed=None. Time-of-flight sidesteps that entirely.
    if Hm is not None and rb is not None and out["world_bounce"] is not None and len(render_pts) >= 3:
        try:
            world_pts = [cs.to_world(Hm, p[1], p[2]) for p in render_pts]
            release_frame = int(render_pts[0][0])
            bounce_frame = int(rb.get("frame", render_pts[-1][0]))
            bounce_wy = abs(out["world_bounce"]["y_m"])
            est = SpeedEstimator(fps=fps).estimate(
                world_points=world_pts,
                release_frame_idx=release_frame,
                bounce_frame=bounce_frame,
                bounce_y=bounce_wy,
                fps=fps)
            if est is not None:
                out["speed"] = {"kmph": round(est.speed_kmh, 1), "method": est.method,
                                "confidence": est.confidence, "distance_m": est.distance_m,
                                "duration_sec": est.duration_sec, "pitch_length_m": pitch_len,
                                "status": "ok"}
        except Exception:
            pass
    return out, clean, recovered, rb, Hm, ppx


def build_delivery_result(source_name, fps, total_frames, id_label, stem, all_pts, an,
                          removed, verdict, recovered, Hm, ppx, pitch_len, conf=CONF):
    """Assemble ONE video's schema-compliant result dict (top-level + single delivery).
    Shared by the batch demo (main) and the single-video API engine (analyze_video)."""
    conf_mean = round(float(np.mean([p[3] for p in all_pts])), 3) if all_pts else 0.0
    rbf = an["bounce"]; wb = an["world_bounce"]
    bframe = int(rbf["frame"]) if rbf else (int(all_pts[-1][0]) if all_pts else 0)

    # world-space trajectory: ground-plane projection via the pitch homography. Airborne
    # points suffer parallax and map to non-physical world coords; keep only points inside
    # the real pitch bounds (|x|<=3.5 m, -1<=y<=pitch+3 m). z=0 (reported on the plane).
    world_traj = []
    if Hm is not None:
        for p in all_pts:
            wx, wy = cs.to_world(Hm, p[1], p[2])
            if abs(wx) <= 3.5 and -1.0 <= wy <= pitch_len + 3.0:
                world_traj.append([round(float(wx), 2), round(float(wy), 2), 0.0])

    # swing = systematic lateral bend of the PRE-bounce path in PIXEL space (robust; no
    # airborne-parallax explosion), scaled to cm via the homography ppx. Indicative only.
    swing_cm, swing_type = 0.0, "straight"
    pre = [(int(p[0]), float(p[1])) for p in all_pts if int(p[0]) <= bframe]
    if ppx and len(pre) >= 4:
        fr = np.array([q[0] for q in pre], float); px = np.array([q[1] for q in pre], float)
        cx2 = np.polyfit(fr, px, 2); cx1 = np.polyfit(fr, px, 1)
        dev_px = float(np.max(np.abs(np.polyval(cx2, fr) - np.polyval(cx1, fr))))
        swing_cm = round(min(25.0, dev_px / ppx * 100), 1)
        swing_type = ("inswing" if cx2[0] > 0 else "outswing") if swing_cm >= 2.5 else "straight"

    bounce_px = (dict(frame_index=int(rbf["frame"]), x_pixel=round(float(rbf["x"]), 1),
                      y_pixel=round(float(rbf["y"]), 1)) if rbf else None)
    bounce_world = (dict(x_m=wb["x_m"], y_m=wb["y_m"]) if wb else None)
    bounce_point = (dict(x=wb["x_m"], y=wb["y_m"]) if wb else None)
    heatmap = ([[wb["x_m"], wb["y_m"]]] if wb else [])

    sp = an["speed"]; spd_kmph = sp.get("kmph")
    speed_block = dict(kmph=spd_kmph, confidence=sp.get("confidence", 0.3),
                       status=("estimated" if spd_kmph is not None else "unavailable"))
    ln = an["line"]; lg = an["length"]
    line_block = dict(label=ln.get("label"), confidence=ln.get("confidence", 0.0),
                      reliability=ln.get("reliability", "indicative"))
    length_block = dict(label=lg.get("label"), confidence=lg.get("confidence", 0.0),
                        distance_from_batsman_m=lg.get("dist_from_batsman_m"))

    physically_valid = (removed == 0)
    cfs = [c for c in [line_block["confidence"], length_block["confidence"],
                       (speed_block["confidence"] if spd_kmph is not None else None)] if c]
    confidence_score = round((float(np.mean(cfs)) if cfs else 0.0) * (1.0 if physically_valid else 0.7), 2)
    did = f"{id_label}_" + hashlib.md5(
        f"{stem}:{all_pts[0][0]}-{all_pts[-1][0]}".encode()).hexdigest()[:6]

    delivery = dict(
        delivery_id=did,
        frame_start=int(all_pts[0][0]), frame_end=int(all_pts[-1][0]),
        track=dict(num_points=len(all_pts), average_confidence=conf_mean,
                   physics_removed_points=removed, physics_verdict=verdict,
                   post_bounce_recovered=bool(recovered)),
        bounce=bounce_px, bounce_world=bounce_world, bounce_point=bounce_point,
        world_trajectory=world_traj, ball_flight_position=world_traj,
        line=line_block, length=length_block,
        speed=speed_block, speed_kmph=spd_kmph,
        swing_cm=swing_cm, swing_type=swing_type, swing_confidence=0.2,
        swing_status="indicative_direction_only",
        heatmap_points=heatmap,
        physically_valid=physically_valid, confidence_score=confidence_score)
    return dict(
        source_video=source_name, fps=round(fps, 2), total_frames=total_frames,
        total_deliveries=1, pipeline_version="offline_mapping+physics_gate+reconstruction",
        detection_confidence=conf, deliveries=[delivery])


def _no_track_result(source_name, fps, total_frames, reason, conf=CONF):
    """Schema-shaped response when no valid delivery is found (0 deliveries)."""
    return dict(
        source_video=source_name, fps=round(fps or 0.0, 2), total_frames=int(total_frames or 0),
        total_deliveries=0, pipeline_version="offline_mapping+physics_gate+reconstruction",
        detection_confidence=conf, status="NO_TRACK", reason=reason, deliveries=[])


# Lazily-loaded singletons so the API loads the models ONCE, not per request.
_ENGINE = {"models": None, "smodel": None, "device": None}


def load_engine():
    """Load (once) the ball ensemble + stump model + device for the analysis engine.
    Uses CRICGIRI_BALL_MODELS if the files exist, else falls back to the committed
    models/ball_best.pt so a minimal cloud image still runs."""
    if _ENGINE["models"] is None:
        import torch
        from ultralytics import YOLO
        _ENGINE["device"] = "0" if torch.cuda.is_available() else "cpu"
        names = [m.strip() for m in _BALL_MODELS_ENV.split(",") if m.strip()]
        paths = [ROOT / "models" / n for n in names if (ROOT / "models" / n).exists()]
        if not paths:
            fallback = ROOT / "models" / "ball_best.pt"
            if fallback.exists():
                paths = [fallback]
        if not paths:
            raise RuntimeError("no ball model weights found under models/")
        _ENGINE["models"] = [YOLO(str(p)) for p in paths]
        _ENGINE["smodel"] = YOLO(str(cs.STUMP))
    return _ENGINE["models"], _ENGINE["smodel"], _ENGINE["device"]


def analyze_video(video_path, pitch_length=PITCH_LEN, conf=CONF, work_id=None, cleanup=True):
    """Analyse ONE cricket-delivery video and return the schema-compliant result dict.

    This is the single-video engine behind the HTTP API. It runs the exact proven demo
    pipeline (low-conf ensemble detection -> offline motion-consistency mapping ->
    physics-validity gate -> static-cluster guard -> homography analytics -> time-of-flight
    speed) and assembles the same JSON as the batch demo. Production weights are read-only.
    """
    import uuid
    models, smodel, device = load_engine()
    stem = work_id or ("api_" + uuid.uuid4().hex[:10])
    vsrc = Path(video_path)
    vdst = ROOT / "videos" / f"{stem}.mp4"
    vdst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(vsrc, vdst)
    try:
        # IN-PROCESS offline mapping using the WARMED ensemble (no per-request subprocess
        # model reload). detect_all returns W/H/frame-count/fps; ransac_trajectory picks the
        # motion-consistent ball arc. Persist mapped_path.csv + detections.csv (same pass,
        # so they can never mismatch) for read_points / analytics / recovery downstream.
        dets, W, H, N, fps = mt.detect_all(models, str(vdst), conf, IMGSZ, device)
        inliers, _px, _py, _t0, _span = mt.ransac_trajectory(dets, 34.0, W=W, H=H)
        map_dir = ROOT / "outputs" / "mapped" / stem; map_dir.mkdir(parents=True, exist_ok=True)
        with open(map_dir / "mapped_path.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["frame", "x", "y", "conf"]); w.writerows(inliers)
        det_dir = ROOT / "outputs" / "detections" / stem; det_dir.mkdir(parents=True, exist_ok=True)
        with open(det_dir / "detections.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["frame", "x", "y", "conf"])
            w.writerows([[d[0], round(d[1], 1), round(d[2], 1), round(d[3], 3)] for d in dets])

        pts = dr.read_points(stem)
        if len(pts) < 4:
            return _no_track_result(vsrc.name, fps, N, "too_few_points", conf)
        render_pts = clean_for_render(pts)
        _xy = np.array([(p[1], p[2]) for p in render_pts], float)
        _spread = float(np.hypot(np.ptp(_xy[:, 0]), np.ptp(_xy[:, 1]))) if len(_xy) else 0.0
        if _spread < 60.0:
            return _no_track_result(vsrc.name, fps, N, "static_cluster", conf)
        with open(map_dir / "mapped_path.csv", "w", newline="") as fh:   # write back the cleaned arc
            w = csv.writer(fh); w.writerow(["frame", "x", "y", "conf"])
            w.writerows([[int(p[0]), p[1], p[2], p[3]] for p in render_pts])
        _, removed, verdict, _ = pg.physics_filter(render_pts)
        an, clean, recovered, rb, Hm, ppx = analytics_for(stem, render_pts, fps, W, H, smodel, device, pitch_length)
        all_pts = sorted({int(p[0]): p for p in (clean + recovered)}.values(), key=lambda p: p[0])
        return build_delivery_result(vsrc.name, fps, N, "delivery", stem, all_pts, an,
                                     removed, verdict, recovered, Hm, ppx, pitch_length, conf)
    finally:
        if cleanup:
            vdst.unlink(missing_ok=True)
            shutil.rmtree(ROOT / "outputs" / "mapped" / stem, ignore_errors=True)
            shutil.rmtree(ROOT / "outputs" / "detections" / stem, ignore_errors=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Best-quality demo output for testing/ clips.")
    ap.add_argument("--pitch-length", type=float, default=PITCH_LEN,
                    help="Real down-pitch length in metres (stump-to-stump). Default 20.12 "
                         "(full pitch). Supply the ACTUAL pitch length for a clip to set the "
                         "metric scale correctly — speed accuracy scales with this.")
    args = ap.parse_args()
    pitch_len = float(args.pitch_length)

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    import torch
    from ultralytics import YOLO
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  conf={CONF}  imgsz={IMGSZ}  pitch_len={pitch_len}m")
    models = [YOLO(str(ROOT / "models" / m)) for m in ("ball_ft_t4.pt", "ball_best_leather_new.pt")]
    smodel = YOLO(str(cs.STUMP))

    vids = sorted(SRC.glob("*.mp4"))
    manifest = []
    for i, v in enumerate(vids, 1):
        stem = f"demo{i:02d}"; cd = DST / f"clip{i:02d}"; cd.mkdir(exist_ok=True)
        try:
            shutil.copy(v, ROOT / "videos" / f"{stem}.mp4")
            cap = cv2.VideoCapture(str(v)); fps = cap.get(5) or 30.0
            W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7)); cap.release()
            print(f"\n[{i}/{len(vids)}] {v.name[:40]}  {W}x{H} {N}f {fps:.0f}fps", flush=True)

            # 1. offline mapping (low conf) — does its own ensemble detection +
            #    motion-consistency mapping. (No separate detection pass needed:
            #    analytics + render use the clean mapped points, not raw recovery.)
            subprocess.run([str(PY), str(ROOT / "scripts/map_trajectory.py"), "--video",
                            str(ROOT / "videos" / f"{stem}.mp4"), "--conf", str(CONF)],
                           capture_output=True, text=True, timeout=400)
            pts = dr.read_points(stem)
            if len(pts) < 4:
                manifest.append(dict(clip=f"clip{i:02d}", source=v.name, fps=round(fps, 1), frames=N,
                                     status="NO_TRACK", track_points=len(pts)))
                print("   NO_TRACK (too few points)", flush=True); continue
            # 3. DEMO-CLEAN the points (drop V-hook noise) and write back so BOTH the
            #    render and the analytics use the clean arc.
            render_pts = clean_for_render(pts)
            # 3a. STATIC-CLUSTER GUARD: a real delivery travels hundreds of px across the
            #     frame; a stationary high-conf false positive (e.g. clip08 = 4 pts all at
            #     42,446) spans ~0. Reject sub-60px tracks as NO_TRACK so a static FP can
            #     never masquerade as a delivery. (All real testing clips span >=100px.)
            _xy = np.array([(p[1], p[2]) for p in render_pts], float)
            _spread = float(np.hypot(np.ptp(_xy[:, 0]), np.ptp(_xy[:, 1]))) if len(_xy) else 0.0
            if _spread < 60.0:
                manifest.append(dict(clip=f"clip{i:02d}", source=v.name, fps=round(fps, 1), frames=N,
                                     status="NO_TRACK", track_points=len(render_pts)))
                print(f"   NO_TRACK (static cluster, spread={_spread:.0f}px)", flush=True); continue
            # NOTE: post-bounce recovery (dr.recover_post_bounce) was evaluated here to
            # extend the arc past the bounce toward the bat. It recovered the real rise on
            # some clips (clip07 11->20) but the recover_post_bounce velocity bound
            # (max_speed*dt+near) loosens over its window and admits DISTANT STATIC
            # false-positive clusters on others (clip03/06 got a static tail at ~203,134;
            # clip05 a static cluster at 272,27). A small window rejects the static FP but
            # also rejects the genuine rise. No single setting is safe across all clips, so
            # recovery is intentionally NOT enabled — it needs an extrapolation-tube guard
            # (accept a recovered point only if it continues the fitted arc), a separate
            # A/B task. The RANSAC seed-spread-cap fix already extends compact deliveries
            # with zero regression.
            mp_csv = ROOT / "outputs" / "mapped" / stem / "mapped_path.csv"
            with open(mp_csv, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["frame", "x", "y", "conf"])
                w.writerows([[int(p[0]), p[1], p[2], p[3]] for p in render_pts])
            filtered, removed, verdict, _ = pg.physics_filter(render_pts)
            # 4. render the CLEAN arc directly (no recovery -> no V-hook re-added)
            ok_r, R = render_clean(stem, render_pts, cd / "trajectory.mp4")
            if not ok_r:
                R = dr.reconstruct(render_pts, None)
            # 5. analytics (on the same clean arc)
            an, clean, recovered, rb, Hm, ppx = analytics_for(stem, render_pts, fps, W, H, smodel, device, pitch_len)
        except Exception as e:
            import traceback
            manifest.append(dict(clip=f"clip{i:02d}", source=v.name, status=f"ERROR:{type(e).__name__}"))
            print(f"   ERROR on {stem}: {e}\n{traceback.format_exc()}", flush=True); continue

        # ── assemble the schema-compliant result (shared with the API engine) ──
        all_pts = sorted({int(p[0]): p for p in (clean + recovered)}.values(), key=lambda p: p[0])
        tp = [dict(frame_idx=int(p[0]), x=round(p[1], 1), y=round(p[2], 1), conf=round(p[3], 3)) for p in all_pts]
        result = build_delivery_result(v.name, fps, N, f"clip{i:02d}", stem, all_pts, an,
                                       removed, verdict, recovered, Hm, ppx, pitch_len)
        conf_mean = result["deliveries"][0]["track"]["average_confidence"]
        (cd / f"clip{i:02d}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        with open(cd / f"clip{i:02d}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["frame_idx", "x", "y", "conf"]); w.writeheader(); w.writerows(tp)
        # copy renders
        fin = ROOT / "outputs" / "final" / f"{stem}.mp4"
        if fin.exists():
            shutil.copy(fin, cd / "trajectory.mp4")
        mp = ROOT / "outputs" / "mapped" / stem / "_MAPPED.jpg"
        if mp.exists():
            shutil.copy(mp, cd / "track_overlay.jpg")
        manifest.append(dict(clip=f"clip{i:02d}", source=v.name, fps=round(fps, 1), frames=N,
                             track_points=len(tp), bounce=("yes" if rb else "no"),
                             line=an["line"]["label"], length=an["length"]["label"],
                             speed_kmph=an["speed"].get("kmph"), conf_mean=conf_mean,
                             physics_removed=removed, status="OK"))
        print(f"   OK: {len(tp)} pts, bounce={'y' if rb else 'n'}, line={an['line']['label']}, "
              f"length={an['length']['label']}, speed={an['speed'].get('kmph')}")

    # manifest + summary
    fields = ["clip", "source", "fps", "frames", "track_points", "bounce", "line", "length",
              "speed_kmph", "conf_mean", "physics_removed", "status"]
    with open(DST / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in manifest:
            w.writerow(r)
    ok = [m for m in manifest if m.get("status") == "OK"]
    md = ["# Testing Result (NEW) — best-quality pipeline output\n",
          f"Pipeline: **low-conf({CONF}) ensemble detection -> offline motion-consistency mapping -> "
          "physics-validity gate -> segmented projectile reconstruction -> homography analytics.**\n",
          f"Clips processed: {len(manifest)} | usable tracks: {len(ok)}\n",
          "| clip | track pts | bounce | line | length | speed | status |",
          "|---|---|---|---|---|---|---|"]
    for m in manifest:
        md.append(f"| {m['clip']} | {m.get('track_points','-')} | {m.get('bounce','-')} | "
                  f"{m.get('line','-')} | {m.get('length','-')} | {m.get('speed_kmph','-')} | {m['status']} |")
    md.append("\nEach clipNN/ folder: `trajectory.mp4`, `track_overlay.jpg`, `clipNN.json`, `clipNN.csv`.")
    (DST / "summary.md").write_text("\n".join(md), encoding="utf-8")

    # cleanup temp videos
    for i in range(1, len(vids) + 1):
        (ROOT / "videos" / f"demo{i:02d}.mp4").unlink(missing_ok=True)
    print(f"\nDONE -> {DST}  ({len(ok)}/{len(manifest)} usable)")


if __name__ == "__main__":
    main()
