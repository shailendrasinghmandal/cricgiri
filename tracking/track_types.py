"""Shared track dataclasses (avoids circular imports)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Tuple


@dataclass
class TrackPoint:
    """Single-frame tracking result after Kalman fusion."""
    frame_idx: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    is_interpolated: bool = False
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in asdict(self).items()
        }


@dataclass
class TrackResult:
    """Immutable delivery-level tracking result."""
    points: List[TrackPoint] = field(default_factory=list)
    confidence_mean: float = 0.0
    interpolated_pct: float = 0.0

    def get_trajectory_pixels(self) -> List[Tuple[float, float]]:
        return [(tp.x, tp.y) for tp in self.points]

    def get_velocities(self) -> List[Tuple[float, float]]:
        return [(tp.vx, tp.vy) for tp in self.points]

    def to_dict(self) -> dict:
        return {
            "num_points": len(self.points),
            "confidence_mean": round(self.confidence_mean, 4),
            "interpolated_pct": round(self.interpolated_pct, 4),
            "points": [tp.to_dict() for tp in self.points],
        }
