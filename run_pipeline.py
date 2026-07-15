#!/usr/bin/env python
"""
run_pipeline.py — analyse ONE cricket video with the CricGiri pipeline.

Produces:
  * a trajectory video (ball arc + Swing/Drift/Speed panel)
  * the delivery JSON (see DELIVERY_API_RESPONSE_FORMAT.md)

Usage:
    python run_pipeline.py clip.mp4
    python run_pipeline.py clip.mp4 --pitch-length-yards 22
    python run_pipeline.py clip.mp4 --out-video out.mp4 --out-json out.json
    python run_pipeline.py clip.mp4 --no-video          # JSON only (faster)

Uses the ball_ft_t4 + ball_best_leather_new ENSEMBLE by default — both weights
are required for a complete track (the leather model finds the release phase).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

YARDS_TO_M = 0.9144


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the CricGiri cricket pipeline on a video.")
    ap.add_argument("video", help="input video (mp4/mov/avi/mkv/webm)")
    ap.add_argument("--out-video", default="outputs/result.mp4", help="rendered trajectory video")
    ap.add_argument("--out-json", default="outputs/result.json", help="delivery JSON")
    ap.add_argument("--pitch-length-yards", type=float, default=22.0,
                    help="REAL pitch length in yards (14-22). Scales speed and all "
                         "down-pitch values — set it to the actual pitch. Default 22.")
    ap.add_argument("--no-video", action="store_true", help="skip rendering (JSON only, faster)")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        return 2

    # Fail loudly and early if the weights are missing, rather than producing
    # a silently degraded result.
    missing = [m for m in ("ball_ft_t4.pt", "ball_best_leather_new.pt", "stump_best.pt")
               if not (Path("models") / m).exists()]
    if missing:
        print(f"ERROR: missing model weights in models/: {missing}", file=sys.stderr)
        print("       All three are required. Re-extract the package.", file=sys.stderr)
        return 2

    from pipeline.pipeline import CricketAnalyticsPipeline, PipelineConfig

    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        video_path=str(video),
        output_video_path=args.out_video,
        output_json_path=args.out_json,
        pitch_length_m=float(args.pitch_length_yards) * YARDS_TO_M,
        save_video=not args.no_video,
        save_json=True,
    )

    print(f"video   : {video}")
    print(f"models  : {Path(cfg.ball_model_path).name} + {Path(cfg.ball_model_alt_path).name} "
          f"(ensemble) + {Path(cfg.stump_model_path).name}")
    print(f"detect  : conf={cfg.ball_confidence} imgsz={cfg.inference_imgsz}")
    print(f"pitch   : {args.pitch_length_yards} yd ({cfg.pitch_length_m:.2f} m)")
    print("running …")

    session = CricketAnalyticsPipeline(cfg).run()
    result = session.to_dict()

    print()
    print(f"deliveries: {result.get('total_deliveries', 0)}")
    for d in result.get("deliveries") or []:
        print(f"  {d['delivery_id']}: speed={d['speed_kmph']} km/h  "
              f"line={d['line']['label']}  length={d['length']['label']}  "
              f"track={d['track']['num_points']} pts  "
              f"confidence={d['confidence_pct']}% {d['confidence_label']}")
    print()
    print(f"JSON  -> {args.out_json}")
    if not args.no_video:
        print(f"VIDEO -> {args.out_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
