"""
scripts/run_all.py
==================
Phase 1 / Phase 4 regression harness.

Runs the shipped pipeline (`python -m pipeline.pipeline`) over every clip in a
video folder (default: videos/), one subprocess per clip so a single crash does
not abort the batch. Per clip it writes:

    runs/<stem>/result.json      pipeline analysis JSON
    runs/<stem>/annotated.mp4     annotated render
    runs/<stem>/run.log           stdout+stderr tail + status + timing

Then it builds a results matrix (rows = videos, columns = output fields) marking
each field  OK / SUSPECT / FAIL  and prints it, also saving:

    runs/_MATRIX.md       human-readable matrix
    runs/_summary.json    machine-readable per-clip + per-field results

Usage
-----
    python scripts/run_all.py                 # all of videos/, default config
    python scripts/run_all.py --videos videos --limit 3
    python scripts/run_all.py --extra-args "--max-recall"   # pass-through flags
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

# Real-world pitch constants (metres) used only for sanity-scoring the outputs.
PITCH_LEN_M = 20.12        # stump-to-stump
PITCH_HALF_WIDTH_M = 1.83  # half of the 3.05 m pitch width (generous)
SPEED_MIN, SPEED_MAX = 60.0, 160.0   # plausible delivery speed band (km/h)

OK, SUSPECT, FAIL = "OK", "SUSPECT", "FAIL"
GLYPH = {OK: "[OK]", SUSPECT: "[~]", FAIL: "[X]"}

FIELDS = ["speed", "bounce", "trajectory", "line", "length", "swing", "confidence"]


def _score_delivery(d: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """Return {field: (status, detail)} for one delivery dict."""
    out: Dict[str, Tuple[str, str]] = {}

    # speed — reliability-aware, not just value-in-band. A number alone is not
    # trustworthy: a fixed median prior, or a pixel-scale (non-homography)
    # estimate, is indicative only and must be flagged even when it lands in band.
    spd = float(d.get("speed_kmph") or 0.0)
    sp = d.get("speed") or {}
    method = str(sp.get("method") or "")
    metric_src = str(sp.get("metric_source") or "")
    is_prior = ("median_prior" in method) or ("outside_band_median" in method)
    is_pixel = metric_src == "pixel_scale_fallback"
    if spd <= 0:
        out["speed"] = (FAIL, "0")
    elif is_prior:
        out["speed"] = (SUSPECT, f"{spd:.1f} prior-guess")
    elif not (SPEED_MIN <= spd <= SPEED_MAX):
        out["speed"] = (SUSPECT, f"{spd:.1f} out-of-band")
    elif is_pixel:
        out["speed"] = (SUSPECT, f"{spd:.1f} pixel-scale")
    else:
        out["speed"] = (OK, f"{spd:.1f} calibrated")

    # bounce
    bp = d.get("bounce_point")
    if not bp:
        out["bounce"] = (FAIL, "none")
    else:
        bx, by = float(bp.get("x", 0)), float(bp.get("y", 0))
        on_pitch = (0.0 <= by <= PITCH_LEN_M) and (abs(bx) <= PITCH_HALF_WIDTH_M)
        out["bounce"] = (OK if on_pitch else SUSPECT, f"({bx:.2f},{by:.2f})")

    # trajectory
    traj = d.get("trajectory") or []
    n = len(traj)
    if n == 0:
        out["trajectory"] = (FAIL, "0pts")
    elif n < 5:
        out["trajectory"] = (SUSPECT, f"{n}pts")
    else:
        out["trajectory"] = (OK, f"{n}pts")

    # line / length
    for f in ("line", "length"):
        v = d.get(f)
        if not v or str(v).lower() in ("none", "null", "unknown"):
            out[f] = (FAIL, str(v))
        elif "uncertain" in str(v).lower():
            out[f] = (SUSPECT, str(v))
        else:
            out[f] = (OK, str(v))

    # swing
    st = str(d.get("swing_type") or "none").lower()
    sc = float(d.get("swing_cm") or 0.0)
    if st in ("none", "null", "") or sc == 0.0:
        out["swing"] = (SUSPECT, f"{st}/{sc:.1f}cm")
    else:
        out["swing"] = (OK, f"{st}/{sc:.1f}cm")

    # confidence
    cf = float(d.get("confidence_score") or 0.0)
    if cf <= 0.0:
        out["confidence"] = (FAIL, "0")
    elif cf >= 1.0:
        out["confidence"] = (SUSPECT, f"{cf:.2f}")
    else:
        out["confidence"] = (OK, f"{cf:.2f}")

    return out


def run_one(video: Path, out_dir: Path, extra_args: List[str]) -> Dict[str, Any]:
    stem = video.stem
    vd = out_dir / stem
    vd.mkdir(parents=True, exist_ok=True)
    out_json = vd / "result.json"
    out_video = vd / "annotated.mp4"

    cmd = [
        str(PY), "-u", "-m", "pipeline.pipeline",
        "--video", str(video),
        "--out-video", str(out_video),
        "--out-json", str(out_json),
        *extra_args,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    elapsed = time.perf_counter() - t0

    log = vd / "run.log"
    tail_out = (proc.stdout or "")[-4000:]
    tail_err = (proc.stderr or "")[-4000:]
    log.write_text(
        f"cmd: {' '.join(cmd)}\n"
        f"returncode: {proc.returncode}\n"
        f"elapsed_sec: {elapsed:.1f}\n"
        f"\n===== STDOUT (tail) =====\n{tail_out}\n"
        f"\n===== STDERR (tail) =====\n{tail_err}\n",
        encoding="utf-8",
    )

    rec: Dict[str, Any] = {
        "video": stem,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "json_exists": out_json.exists(),
        "video_exists": out_video.exists(),
        "status": "",
        "n_deliveries": 0,
        "error": "",
        "fields": {},
    }

    if proc.returncode != 0:
        rec["status"] = "FAIL"
        rec["error"] = tail_err.strip().splitlines()[-1] if tail_err.strip() else "nonzero exit"
        return rec

    if not out_json.exists():
        rec["status"] = "NO_JSON"
        rec["error"] = "pipeline exited 0 but produced no JSON"
        return rec

    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as exc:
        rec["status"] = "BAD_JSON"
        rec["error"] = str(exc)
        return rec

    deliveries = data.get("deliveries") or []
    rec["n_deliveries"] = len(deliveries)
    if not deliveries:
        rec["status"] = "NO_DELIVERY"
        rec["fields"] = {f: [FAIL, "no-delivery"] for f in FIELDS}
        return rec

    # Score the first (primary) delivery.
    scored = _score_delivery(deliveries[0])
    rec["fields"] = {f: [scored[f][0], scored[f][1]] for f in FIELDS}
    statuses = [scored[f][0] for f in FIELDS]
    if FAIL in statuses or SUSPECT in statuses:
        rec["status"] = "PARTIAL"
    else:
        rec["status"] = "COMPLETE"
    return rec


def build_matrix(records: List[Dict[str, Any]]) -> str:
    hdr = ["video", "status", "time(s)", "dels", *FIELDS]
    widths = {h: len(h) for h in hdr}
    rows: List[List[str]] = []
    for r in records:
        row = [
            r["video"],
            r["status"],
            f"{r['elapsed_sec']:.0f}",
            str(r["n_deliveries"]),
        ]
        for f in FIELDS:
            st = r["fields"].get(f, [FAIL, ""])[0]
            row.append(GLYPH.get(st, st))
        rows.append(row)
        for h, cell in zip(hdr, row):
            widths[h] = max(widths[h], len(cell))

    def fmt(cells: List[str]) -> str:
        return "  ".join(c.ljust(widths[h]) for h, c in zip(hdr, cells))

    lines = [fmt(hdr), fmt(["-" * widths[h] for h in hdr])]
    lines += [fmt(r) for r in rows]

    # Tally
    n = len(records)
    complete = sum(1 for r in records if r["status"] == "COMPLETE")
    partial = sum(1 for r in records if r["status"] == "PARTIAL")
    failed = sum(1 for r in records if r["status"] in ("FAIL", "NO_JSON", "BAD_JSON", "NO_DELIVERY"))
    lines.append("")
    lines.append(f"TOTAL {n} | COMPLETE {complete} | PARTIAL {partial} | FAILED/EMPTY {failed}")
    for f in FIELDS:
        ok = sum(1 for r in records if r["fields"].get(f, [FAIL])[0] == OK)
        sus = sum(1 for r in records if r["fields"].get(f, [FAIL])[0] == SUSPECT)
        bad = sum(1 for r in records if r["fields"].get(f, [FAIL])[0] == FAIL)
        lines.append(f"  {f:11s}  OK {ok:2d}  SUSPECT {sus:2d}  FAIL {bad:2d}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run pipeline over a video folder and build a results matrix.")
    ap.add_argument("--videos", default="videos", help="Folder of .mp4 clips")
    ap.add_argument("--out", default="runs", help="Output root (runs/<stem>/)")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N clips (0 = all)")
    ap.add_argument(
        "--config", choices=["strong", "baseline"], default="strong",
        help="strong = validated ensemble (ft_t4 + leather_new) at 1280 with max-recall "
             "(recovers detection-starved clips); baseline = shipped single-model 640 default",
    )
    ap.add_argument("--no-video", action="store_true",
                    help="Skip the annotated render pass (faster; JSON-only matrix)")
    ap.add_argument("--extra-args", default="", help="Extra flags appended to the pipeline CLI")
    args = ap.parse_args()

    vdir = ROOT / args.videos
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validated strong config = the measured-best detection stack (additive; no
    # production weights overwritten — just selects models that already exist).
    STRONG = [
        "--max-recall",
        "--ball-model", "models/ball_ft_t4.pt",
        "--ball-model-alt", "models/ball_best_leather_new.pt",
        "--hybrid-ensemble",
    ]
    extra: List[str] = list(STRONG) if args.config == "strong" else []
    if args.no_video:
        extra.append("--no-video")
    if args.extra_args.strip():
        extra += args.extra_args.split()

    videos = sorted(vdir.glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"No .mp4 found in {vdir}")
        sys.exit(1)

    print(f"Running pipeline over {len(videos)} clips from {vdir} ...")
    if extra:
        print(f"Extra args: {extra}")

    records: List[Dict[str, Any]] = []
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v.name} ...", flush=True)
        rec = run_one(v, out_dir, extra)
        print(f"    -> {rec['status']} ({rec['elapsed_sec']:.0f}s, {rec['n_deliveries']} deliveries)", flush=True)
        records.append(rec)

    matrix = build_matrix(records)
    print("\n" + matrix)

    (out_dir / "_MATRIX.md").write_text(
        "# Pipeline results matrix\n\n```\n" + matrix + "\n```\n", encoding="utf-8"
    )
    (out_dir / "_summary.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {out_dir/'_MATRIX.md'}  and  {out_dir/'_summary.json'}")


if __name__ == "__main__":
    main()
