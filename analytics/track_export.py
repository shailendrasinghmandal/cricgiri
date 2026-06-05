"""
analytics/track_export.py
=========================
Export ball track to CSV and extended debug JSON.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def export_track_csv(
    deliveries: List[Any],
    output_path: str | Path,
) -> Path:
    """Write per-frame ball coordinates CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for d in deliveries:
        pts = (getattr(d, "track", None) or {}).get("points") if hasattr(d, "track") else None
        if pts is None and isinstance(d, dict):
            pts = (d.get("track") or {}).get("points")
        if not pts:
            continue
        did = getattr(d, "delivery_id", None) or d.get("delivery_id", "")
        for p in pts:
            rows.append({
                "delivery_id": did,
                "frame_idx": p.get("frame_idx"),
                "x": p.get("x"),
                "y": p.get("y"),
                "vx": p.get("vx"),
                "vy": p.get("vy"),
                "confidence": p.get("confidence"),
                "is_interpolated": p.get("is_interpolated"),
            })

    with out.open("w", newline="", encoding="utf-8") as fh:
        if not rows:
            writer = csv.DictWriter(
                fh,
                fieldnames=["delivery_id", "frame_idx", "x", "y", "vx", "vy", "confidence", "is_interpolated"],
            )
            writer.writeheader()
        else:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out


def export_debug_json(
    debug_payload: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    return out
