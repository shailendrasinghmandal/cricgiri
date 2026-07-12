"""
scripts/run_phase4a.py   [ADDITIVE — Phase 4a orchestrator: blur-aug finetune + eval]
====================================================================================
Runs the whole detector-blur-robustness experiment unattended, then measures it
against the locked baseline on the hand-labelled clip(s):

  1. build a blur/motion-degraded copy of dataset/ball_clean (if not already there)
  2. fine-tune models/ball_ft_t4.pt on it (low-LR blur profile)
  3. evaluate the resulting model on eval/clip01.mp4 vs gt/clip01.csv  (tag 'phase4a')
  4. print BEFORE(baseline) vs AFTER(phase4a) and an accept/reject verdict
     (accept only if detector_recall strictly improves and false-positives don't blow up)

Nothing production is overwritten: the candidate stays in runs/… and models/staging/.
Run:  venv/Scripts/python.exe scripts/run_phase4a.py
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
BLUR_DS = ROOT / "dataset" / "ball_blur_aug"
EVAL_VIDEO = ROOT / "eval" / "clip01.mp4"
OUT = ROOT / "outputs" / "eval_harness"


def run(cmd, **kw):
    print(f"\n>> {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise SystemExit(f"[phase4a] step failed (exit {r.returncode}): {cmd[:3]} ...")
    return r


def main():
    t0 = dt.datetime.now()
    print(f"[phase4a] start {t0:%Y-%m-%d %H:%M:%S}")

    # 1 ── build blur-augmented dataset (skip if already built) ────────────────
    if (BLUR_DS / "data.yaml").exists():
        print(f"[phase4a] blur dataset already present: {BLUR_DS}")
    else:
        run([PY, str(ROOT / "scripts/build_blur_hard_dataset.py"),
             "--src", str(ROOT / "dataset/ball_clean"), "--out", str(BLUR_DS),
             "--variants", "2", "--motion-min", "9", "--motion-max", "31",
             "--keep-original"])

    # 2 ── fine-tune from ball_ft_t4 on the blurred set ────────────────────────
    #   imgsz 640 (not 1280) + workers 2 + cache=False keeps a 19k-image set inside
    #   16GB RAM — 1280 OOMs the cv2 augmentation workers on this box.
    run([PY, str(ROOT / "train.py"), "--finetune",
         "--model", str(ROOT / "models/ball_ft_t4.pt"),
         "--data", str(BLUR_DS / "data.yaml"),
         "--epochs", "20", "--imgsz", "640", "--batch", "8",
         "--workers", "2", "--cache", "False"])

    # locate the freshest best.pt and stage it
    runs = sorted((ROOT / "runs" / "ball_detector").rglob("weights/best.pt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit("[phase4a] training produced no best.pt")
    best = runs[0]
    staging = ROOT / "models" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    cand = staging / f"ball_blur_ft_{t0:%Y%m%d_%H%M}.pt"
    shutil.copy2(best, cand)
    print(f"[phase4a] candidate staged: {cand}")

    # 3 ── evaluate candidate on the labelled clip ─────────────────────────────
    run([PY, str(ROOT / "scripts/eval_harness.py"),
         "--video", str(EVAL_VIDEO), "--tag", "phase4a", "--ball-model", str(cand)])

    # 4 ── verdict ─────────────────────────────────────────────────────────────
    base = {r["clip"]: r for r in json.loads((OUT / "baseline.json").read_text())}
    new = {r["clip"]: r for r in json.loads((OUT / "phase4a.json").read_text())}
    print("\n" + "=" * 64)
    print("PHASE 4A RESULT  (blur-augmentation finetune)")
    print("=" * 64)
    accepted_any = False
    for clip, nb in new.items():
        ob = base.get(clip)
        if not ob:
            continue
        dr0, dr1 = ob["detector_recall"], nb["detector_recall"]
        fp0, fp1 = ob["false_positives_raw"], nb["false_positives_raw"]
        tr0, tr1 = ob["track_recall"], nb["track_recall"]
        acc = (dr1 > dr0) and (fp1 <= max(5, fp0) * 1.25)
        accepted_any = accepted_any or acc
        print(f"  {clip}: detector_recall {dr0:.2f} -> {dr1:.2f}  "
              f"| track_recall {tr0:.2f} -> {tr1:.2f}  | raw_FP {fp0} -> {fp1}  "
              f"=> {'ACCEPT' if acc else 'REJECT'}")
    print("-" * 64)
    print("VERDICT:", "candidate IMPROVES recall — keep for wider eval"
          if accepted_any else
          "no recall gain — baseline kept (blur-finetune did not help this footage)")
    print(f"[phase4a] done in {(dt.datetime.now() - t0)}")
    print("candidate weights:", cand)


if __name__ == "__main__":
    sys.exit(main())
