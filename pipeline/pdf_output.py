"""
Convert the full pipeline JSON to the exact PDF Page-17 "Final Structured
Response" format from the CricGiri Balling Model Shalender specification.

The PDF specifies per-delivery shape:
    {
      "delivery_id":      str,
      "speed_kmph":       float,
      "bounce_point":     {"x": float, "y": float},
      "trajectory":       [[x, y, z], ...],
      "line":             str,
      "length":           str,
      "swing_cm":         float,
      "swing_type":       str,
      "heatmap_points":   [[x, y], ...],
      "confidence_score": float
    }

This module does NOT touch the main pipeline — it just reads the full JSON
and writes a clean PDF-compliant copy alongside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _delivery_to_pdf(delivery: Dict[str, Any]) -> Dict[str, Any]:
    """Map a full delivery record to PDF Page-17 schema."""
    speed = delivery.get("speed") or {}
    bounce = delivery.get("bounce") or {}
    swing = delivery.get("swing") or {}

    # Prefer the PDF-named convenience fields the pipeline already adds
    # when they are present (set in DeliveryAnalysis.to_dict).
    speed_kmph = float(
        delivery.get("speed_kmph")
        or speed.get("speed_kmh")
        or 0.0
    )

    bounce_point = delivery.get("bounce_point")
    if bounce_point is None and bounce.get("world_x") is not None:
        bounce_point = {
            "x": round(float(bounce["world_x"]), 3),
            "y": round(float(bounce["world_y"]), 3),
        }

    # trajectory: prefer the world (x,y,z) trajectory from the pipeline
    trajectory = (
        delivery.get("trajectory")
        or delivery.get("world_trajectory")
        or []
    )
    # Make sure each row is [x, y, z]; pad with 0.0 if z missing.
    norm_traj: List[List[float]] = []
    for row in trajectory:
        if not row:
            continue
        if len(row) >= 3:
            norm_traj.append([
                round(float(row[0]), 3),
                round(float(row[1]), 3),
                round(float(row[2]), 3),
            ])
        elif len(row) == 2:
            norm_traj.append([
                round(float(row[0]), 3),
                round(float(row[1]), 3),
                0.0,
            ])

    line = str(delivery.get("line") or "unknown")
    length = str(delivery.get("length") or "unknown")

    swing_cm = float(
        delivery.get("swing_cm")
        or swing.get("swing_cm")
        or 0.0
    )
    swing_type = str(
        delivery.get("swing_type")
        or swing.get("direction")
        or "none"
    )

    heatmap_points = delivery.get("heatmap_points")
    if not heatmap_points and bounce_point is not None:
        heatmap_points = [[bounce_point["x"], bounce_point["y"]]]
    elif not heatmap_points:
        heatmap_points = []

    confidence_score = round(
        float(delivery.get("confidence_score") or delivery.get("confidence") or 0.0),
        4,
    )

    return {
        "delivery_id":      str(delivery.get("delivery_id") or ""),
        "speed_kmph":       round(speed_kmph, 2),
        "bounce_point":     bounce_point,
        "trajectory":       norm_traj,
        "line":             line,
        "length":           length,
        "swing_cm":         round(swing_cm, 2),
        "swing_type":       swing_type,
        "heatmap_points":   heatmap_points,
        "confidence_score": confidence_score,
    }


def convert_session_to_pdf(full_json_path: Path, out_path: Path) -> Dict[str, Any]:
    """
    Read a full pipeline JSON and write the PDF-format version.

    Output shape:
        {
          "video":      str,
          "fps":        float,
          "deliveries": [ <PDF per-delivery dict>, ... ]
        }
    """
    raw = json.loads(full_json_path.read_text(encoding="utf-8"))
    deliveries = [
        _delivery_to_pdf(d) for d in (raw.get("deliveries") or [])
    ]
    pdf_doc = {
        "video":      str(raw.get("video_path") or full_json_path.stem),
        "fps":        float(raw.get("fps") or 0.0),
        "deliveries": deliveries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pdf_doc, indent=2), encoding="utf-8")
    return pdf_doc


def convert_outputs_directory(
    outputs_dir: Path = Path("outputs"),
    pdf_subdir: str = "pdf_format",
) -> List[Dict[str, Any]]:
    """Convert every main_pipeline_*_latest.json into a PDF-format JSON."""
    target_dir = outputs_dir / pdf_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for src in sorted(outputs_dir.glob("main_pipeline_*_latest.json")):
        stem = src.stem.replace("main_pipeline_", "").replace("_latest", "")
        dst = target_dir / f"{stem}.json"
        doc = convert_session_to_pdf(src, dst)
        n = len(doc["deliveries"])
        summaries.append({
            "video": doc["video"],
            "deliveries": n,
            "output": str(dst),
        })
        print(f"  {stem:<22}  {n} deliveries  ->  {dst}")
    return summaries


if __name__ == "__main__":
    print("Converting pipeline JSON to PDF Page-17 format ...")
    convert_outputs_directory()
    print("Done.")
