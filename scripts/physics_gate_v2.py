"""
scripts/physics_gate_v2.py   [ADDITIVE EXPERIMENT — physics-validity gate]
=========================================================================
A real cricket ball follows projectile physics: it travels in ONE horizontal
direction from release to bat, and falls under gravity (one descent, an optional
single bounce, then a rise). It NEVER reverses horizontal direction in mid-air.

The baseline reconstruction fits a smooth spline through whatever points it
tracked — with no physics check — so a single false-positive (e.g. the ball after
it was hit, or a moving person) bends the drawn curve into an impossible shape
(see should_output clip #57: a clean descent + one stray point 10 frames later
became a C-hook). This gate rejects points that violate ball physics BEFORE the
curve is drawn. Production reconstruction is untouched; this wraps it.

Checks
------
1. MONOTONIC HORIZONTAL MOTION — the ball keeps moving one way across frame.
   A point that reverses the dominant x-direction beyond tolerance is rejected
   (this is what kills the #57 outlier).
2. TEMPORAL-GAP JUMP — a large frame gap combined with a velocity-inconsistent
   spatial jump is rejected.
3. VERTICAL TURNING POINTS — a valid path falls (y up in image) with at most one
   reversal (the bounce). Many reversals => flagged 'suspect'.

Verdict per clip: valid / suspect / invalid, plus how many points were removed.

Outputs: outputs/physics_gate_ab.csv
Run:  venv/Scripts/python.exe scripts/physics_gate_v2.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _imp(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


dr = _imp("dr", ROOT / "scripts" / "delivery_reconstruction.py")


def physics_filter(pts):
    """Keep only points consistent with one-directional, gravity-driven flight.
    Returns (filtered, removed_count, verdict, reasons)."""
    pts = sorted(pts, key=lambda p: p[0])
    if len(pts) < 4:
        return pts, 0, "too_few", []
    xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
    fs = np.array([p[0] for p in pts])
    xrange = float(xs.max() - xs.min()); yrange = float(ys.max() - ys.min())
    # Dominant horizontal direction from the robust early-half slope.
    # Only steps that are REAL motion get a vote: a detector that re-reports the
    # same ball at rest jitters by a few hundredths of a pixel, and those
    # near-zero steps can out-number the true strides and hand the median their
    # (meaningless) sign -- which flips xdir and makes the genuine flight read as
    # one long reversal. Fall back to net early-half displacement if nothing moves.
    half = max(2, len(pts) // 2)
    _dx = np.diff(xs[:half + 1])
    _moving = _dx[np.abs(_dx) > 1.0]
    if _moving.size:
        xdir = np.sign(np.median(_moving))
    else:
        xdir = np.sign(xs[half] - xs[0]) if half >= 2 else 0.0
    # x is meaningful only if horizontal span is a real fraction of the motion
    horizontal = xrange > 0.30 * yrange and xrange > 25
    xtol = 0.08 * xrange + 8
    keep = [0]; reasons = []
    for i in range(1, len(pts)):
        j = keep[-1]
        ddx = xs[i] - xs[j]; ddf = fs[i] - fs[j]
        # 1. horizontal-direction reversal
        if horizontal and xdir != 0 and np.sign(ddx) == -xdir and abs(ddx) > xtol:
            reasons.append(f"f{int(fs[i])}:x_reversal({ddx:+.0f})"); continue
        # 2. big temporal gap with a large, direction-wrong jump
        if ddf >= 6 and horizontal and xdir != 0 and np.sign(ddx) == -xdir and abs(ddx) > 0.20 * xrange:
            reasons.append(f"f{int(fs[i])}:gap_jump(df{int(ddf)})"); continue
        keep.append(i)
    filtered = [pts[k] for k in keep]
    # 3. vertical turning points on the filtered set (noise-robust, smoothed)
    fy = np.array([p[2] for p in filtered])
    turns = 0
    if len(fy) >= 5:
        sm = np.convolve(fy, np.ones(3) / 3, mode="valid")
        dvy = np.sign(np.diff(sm))
        turns = int(np.sum(np.abs(np.diff(dvy)) > 0))
    removed = len(pts) - len(filtered)
    if removed == 0 and turns <= 1:
        verdict = "valid"
    elif turns >= 3:
        verdict = "suspect"
    else:
        verdict = "cleaned" if removed > 0 else "valid"
    return filtered, removed, verdict, reasons


def build_track_json(curve, path):
    pts = [{"x": float(x), "y": float(y), "frame_idx": -1, "is_interpolated": False} for (x, y) in curve]
    Path(path).write_text(json.dumps({"deliveries": [{"track": {"points": pts}}]}))


def gated_curve(clip):
    """Baseline vs physics-gated reconstruction curve for a clip."""
    pts = dr.read_points(clip)
    Rb = dr.reconstruct(pts, clip)
    base_curve = (Rb.get("down") or []) + (Rb.get("up") or [])
    filtered, removed, verdict, reasons = physics_filter(pts)
    # reconstruct from filtered points WITHOUT raw recovery (clip=None) so a
    # rejected outlier cannot be re-introduced by the recovery step.
    Rg = dr.reconstruct(filtered, None)
    gate_curve = (Rg.get("down") or []) + (Rg.get("up") or [])
    return base_curve, gate_curve, removed, verdict, reasons, len(pts), len(filtered)


def main():
    # the should_output clips with a red reference arc (for accuracy before/after)
    REF = {"EDE06032-9744-42C5-88FB-851E4BC48CB6": "EDE",
           "SavedVideo_1776714003623": "#39_open",
           "SavedVideo_1776831946956": "#57_backyard"}
    test_clips = sorted([p.parent.name for p in (ROOT / "outputs/mapped").glob("*/mapped_path.csv")
                         if p.parent.name.startswith("test_video")],
                        key=lambda c: int(c.replace("test_video", "")))

    rows = []
    print("=== PHYSICS GATE on all clips (removed points + verdict) ===")
    print(f"{'clip':40}{'pts':>5}{'kept':>5}{'removed':>8}  verdict   reasons")
    print("-" * 90)
    for clip in test_clips + list(REF):
        _, _, removed, verdict, reasons, n, k = gated_curve(clip)
        nm = REF.get(clip, clip)
        rows.append(dict(clip=nm, points=n, kept=k, removed=removed, verdict=verdict,
                         reasons="; ".join(reasons)))
        flag = "  <== REMOVED" if removed else ""
        print(f"{nm:40}{n:>5}{k:>5}{removed:>8}  {verdict:8}  {'; '.join(reasons)}{flag}")

    with open(OUT / "physics_gate_ab.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # accuracy before/after vs the red reference arc, for the 3 should_output clips
    print("\n=== REFERENCE-ARC MATCH before vs after gate (should_output) ===")
    import subprocess
    for clip, nm in REF.items():
        base_curve, gate_curve, removed, verdict, reasons, n, k = gated_curve(clip)
        res = {}
        for tag, curve in (("base", base_curve), ("gate", gate_curve)):
            jp = ROOT / f"tmp/_pg_{tag}_{clip}.json"
            build_track_json(curve, jp)
            try:
                p = subprocess.run([str(ROOT / "venv/Scripts/python.exe"),
                                    str(ROOT / "scripts/compare_to_reference_arc.py"),
                                    "--video", str(ROOT / f"should_output/{clip}.mp4"),
                                    "--track", str(jp),
                                    "--out-prefix", str(ROOT / f"tmp/_pg_{tag}_{clip}")],
                                   capture_output=True, text=True, timeout=180)
                import re
                m = re.search(r'"path_to_arc_mean_px":\s*([\d.eE+]+)', p.stdout)
                res[tag] = float(m.group(1)) if m else None
            except Exception as e:
                res[tag] = None
        def fmt(v):
            return "no-ref" if v is None or v > 1e9 else f"{v:.1f}px"
        print(f"  {nm:16} removed={removed:>2}  base {fmt(res.get('base')):>9}  ->  gated {fmt(res.get('gate')):>9}")
    print("\n-> outputs/physics_gate_ab.csv  | tmp/_pg_*overlay.jpg")


if __name__ == "__main__":
    main()
