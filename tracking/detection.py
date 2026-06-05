"""Shared detection dataclass (avoids circular imports)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """Raw YOLO detection for a single frame."""
    frame_idx: int
    cx: float
    cy: float
    conf: float
    w: float = 0.0
    h: float = 0.0
