"""
scripts/extract_frames_for_annotation.py   [ADDITIVE — annotation prep]
======================================================================
Extract every frame of one or more testing clips into eval/frames/<stem>/ and
write a per-clip annotation template, so they can be hand-labelled (any tool) and
fed back through scripts/gt_from_annotations.py.

Usage:
  venv/Scripts/python.exe scripts/extract_frames_for_annotation.py --clips 2 3 4 5 6
  venv/Scripts/python.exe scripts/extract_frames_for_annotation.py --all
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "testing"
EVAL = ROOT / "eval"


def extract(video: Path, stem: str) -> int:
    outdir = EVAL / "frames" / stem
    outdir.mkdir(parents=True, exist_ok=True)
    shutil_dst = EVAL / f"{stem}.mp4"
    if not shutil_dst.exists():
        import shutil
        shutil.copy(video, shutil_dst)
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5)
    rows = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        name = f"frame_{i:05d}.jpg"
        cv2.imwrite(str(outdir / name), f)
        rows.append([i, name, "", "", "", ""])
        i += 1
    cap.release()
    with open(EVAL / f"{stem}_annotation_template.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "filename", "x", "y", "visible", "is_bounce"])
        w.writerows(rows)
    print(f"  {stem}: {i} frames  {W}x{H} @ {fps:.0f}fps  -> eval/frames/{stem}/")
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, nargs="*", default=[],
                    help="1-based clip numbers in sorted testing/ order")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    vids = sorted(SRC.glob("*.mp4"))
    if not vids:
        raise SystemExit(f"no clips in {SRC}")
    if args.all:
        idxs = list(range(1, len(vids) + 1))
    else:
        idxs = args.clips or [2, 3, 4, 5, 6]

    print(f"extracting {len(idxs)} clip(s) from {SRC} ...")
    for n in idxs:
        if not (1 <= n <= len(vids)):
            print(f"  clip {n}: out of range (have {len(vids)})"); continue
        extract(vids[n - 1], f"clip{n:02d}")
    print("\nAnnotate the ball in eval/frames/clipNN/, then convert with:")
    print("  venv/Scripts/python.exe scripts/gt_from_annotations.py \\")
    print("     --yolo-labels <labels_dir> --images eval/frames/clipNN --stem clipNN --bounce-frame <N>")


if __name__ == "__main__":
    main()
