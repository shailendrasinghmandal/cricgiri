"""
scripts/gt_from_annotations.py   [ADDITIVE — Phase 0 annotation -> GT converter]
================================================================================
Turn whatever you annotate into the harness ground-truth file gt/<stem>.csv
(columns: frame,x,y,visible,is_bounce). Two input styles:

1) FILLED TEMPLATE  (the CSV shipped alongside the extracted frames)
   Edit eval/<stem>_annotation_template.csv and fill x,y (pixel centre of the
   ball) and visible (1/0) per frame; put is_bounce=1 on the single bounce frame.
   Leave x,y blank OR set visible=0 for frames where the ball is not visible.
       venv/Scripts/python.exe scripts/gt_from_annotations.py \
           --template eval/clip01_annotation_template.csv --stem clip01

2) YOLO LABELS  (from LabelImg / CVAT / Roboflow exported over the frames)
   A folder of frame_00042.txt files ("<cls> <cx> <cy> <w> <h>" normalised);
   the ball box centre becomes x,y. Needs the frame images for W,H.
       venv/Scripts/python.exe scripts/gt_from_annotations.py \
           --yolo-labels eval/frames/clip01/labels --images eval/frames/clip01 \
           --stem clip01 --bounce-frame 70

Frame index is parsed from the file name (the digits), so annotate only the
frames you want — unlabelled frames are simply absent (treated as "no GT here").
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "gt"


def _frame_num(name: str) -> int | None:
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else None


def from_template(path: Path, stem: str) -> dict:
    gt = {}
    for r in csv.DictReader(open(path)):
        f = _frame_num(r.get("filename") or "")
        if f is None and r.get("frame", "").strip().isdigit():
            f = int(r["frame"])
        if f is None:
            continue
        x = (r.get("x") or "").strip()
        y = (r.get("y") or "").strip()
        vis_raw = (r.get("visible") or "").strip()
        bnc = 1 if (r.get("is_bounce") or "").strip() in ("1", "y", "yes", "true") else 0
        if x and y:
            visible = 1 if vis_raw in ("", "1", "y", "yes", "true") else 0
            gt[f] = (float(x), float(y), visible, bnc)
        elif vis_raw in ("0", "n", "no", "false"):
            gt[f] = (0.0, 0.0, 0, bnc)
        # blank row with no coords and no explicit 0 -> skip (unlabelled)
    return gt


def from_yolo(labels_dir: Path, images_dir: Path, bounce_frame: int | None) -> dict:
    gt = {}
    for txt in sorted(labels_dir.glob("*.txt")):
        f = _frame_num(txt.name)
        if f is None:
            continue
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = images_dir / f"{txt.stem}{ext}"
            if p.exists():
                img = p
                break
        if img is None:
            continue
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        lines = [ln.split() for ln in txt.read_text().splitlines() if ln.strip()]
        if not lines:
            gt[f] = (0.0, 0.0, 0, 0)          # empty label = ball not visible
            continue
        # highest box (smallest area assumption not needed; take first ball box)
        cls_cx_cy = lines[0]
        cx, cy = float(cls_cx_cy[1]) * W, float(cls_cx_cy[2]) * H
        gt[f] = (round(cx, 1), round(cy, 1), 1, 1 if f == bounce_frame else 0)
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="clip stem -> gt/<stem>.csv")
    ap.add_argument("--template", help="filled annotation template CSV")
    ap.add_argument("--yolo-labels", help="folder of YOLO .txt labels")
    ap.add_argument("--images", help="folder of the frame images (for YOLO W,H)")
    ap.add_argument("--bounce-frame", type=int, default=None,
                    help="(YOLO mode) frame index of the bounce")
    args = ap.parse_args()

    if args.template:
        gt = from_template(Path(args.template), args.stem)
    elif args.yolo_labels:
        if not args.images:
            raise SystemExit("--images is required with --yolo-labels")
        gt = from_yolo(Path(args.yolo_labels), Path(args.images), args.bounce_frame)
    else:
        raise SystemExit("give --template or --yolo-labels")

    if not gt:
        raise SystemExit("no labels parsed — check the input")
    GT_DIR.mkdir(exist_ok=True)
    out = GT_DIR / f"{args.stem}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x", "y", "visible", "is_bounce"])
        for f in sorted(gt):
            x, y, vis, bnc = gt[f]
            w.writerow([f, x, y, vis, bnc])
    vis_n = sum(1 for v in gt.values() if v[2])
    bnc_n = sum(1 for v in gt.values() if v[3])
    print(f"wrote {len(gt)} rows ({vis_n} visible, {bnc_n} bounce) -> {out}")
    if bnc_n == 0:
        print("  note: no bounce frame marked — bounce-error metrics will be skipped.")


if __name__ == "__main__":
    main()
