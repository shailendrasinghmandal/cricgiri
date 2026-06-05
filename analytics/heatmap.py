"""
analytics/heatmap.py
====================
Accumulates ball bounce locations across multiple deliveries and
generates pitch heatmap overlays using OpenCV and numpy.

Supports:
  - Per-delivery bounce point storage
  - Kernel density estimation (KDE) heatmap generation
  - Overlay onto a pitch diagram PNG/frame
  - Zone statistics export

Author: Cricket Analytics Engine
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pitch canvas dimensions
# ---------------------------------------------------------------------------

# Default heatmap canvas size in pixels (portrait orientation, pitch top-down view)
CANVAS_H: int = 800   # pixels high (bowling crease → batting crease)
CANVAS_W: int = 400   # pixels wide (off-side → leg-side)

# World-to-canvas scale factors
# Pitch is PITCH_LENGTH_M × CREASE_WIDTH_M in world space
PITCH_LENGTH_M: float = 20.12
CREASE_WIDTH_M: float = 3.66

SCALE_Y: float = CANVAS_H / PITCH_LENGTH_M   # px per metre (longitudinal)
SCALE_X: float = CANVAS_W / CREASE_WIDTH_M   # px per metre (lateral)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BouncePoint:
    delivery_id: str
    world_x: float       # lateral (metres, off-side negative)
    world_y: float       # longitudinal (metres, bowling crease = 0)
    speed_kmh: float
    line: str
    length: str


@dataclass
class HeatmapStats:
    total_deliveries: int
    hottest_zone_x_m: float      # world X of density peak
    hottest_zone_y_m: float      # world Y of density peak
    zone_distribution: Dict[str, int]   # length zone → count

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# HeatmapGenerator
# ---------------------------------------------------------------------------

class HeatmapGenerator:
    """
    Accumulates bounce points and renders a smooth KDE-based pitch heatmap.

    Usage
    -----
        gen = HeatmapGenerator()
        gen.add_bounce(BouncePoint(...))
        heatmap_img = gen.render()
        gen.save(heatmap_img, "outputs/heatmap.png")
    """

    def __init__(
        self,
        canvas_h: int = CANVAS_H,
        canvas_w: int = CANVAS_W,
        kernel_size: int = 45,
        colormap: int = cv2.COLORMAP_JET,
        heatmap_alpha: float = 0.65,
    ):
        """
        Args:
            canvas_h:       Heatmap image height in pixels.
            canvas_w:       Heatmap image width in pixels.
            kernel_size:    Gaussian blur kernel size (controls spread).
            colormap:       OpenCV colormap for false-colour rendering.
            heatmap_alpha:  Blend alpha when overlaying on background.
        """
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.colormap = colormap
        self.alpha = heatmap_alpha

        self._points: List[BouncePoint] = []
        self._accumulator: np.ndarray = np.zeros(
            (canvas_h, canvas_w), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_bounce(self, bp: BouncePoint) -> None:
        """Add a single delivery bounce point to the accumulator."""
        px, py = self._world_to_canvas(bp.world_x, bp.world_y)

        if not (0 <= px < self.canvas_w and 0 <= py < self.canvas_h):
            logger.debug(
                "Bounce point (%.2f, %.2f) maps outside canvas — skipped.",
                bp.world_x, bp.world_y,
            )
            return

        # Add a unit impulse at the pixel location
        self._accumulator[py, px] += 1.0
        self._points.append(bp)
        logger.debug(
            "Bounce added | delivery=%s | world=(%.3f, %.3f) | canvas=(%d, %d)",
            bp.delivery_id, bp.world_x, bp.world_y, px, py,
        )

    def add_bounces_batch(self, points: List[BouncePoint]) -> None:
        """Batch-add a list of bounce points."""
        for bp in points:
            self.add_bounce(bp)
        logger.info("Batch added %d bounce points.", len(points))

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._points.clear()
        self._accumulator[:] = 0.0
        logger.info("HeatmapGenerator reset.")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate a smooth heatmap image, optionally blended with a
        pitch background diagram.

        Args:
            background: BGR image of pitch diagram (same size as canvas).
                        If None, a plain grass-green background is generated.

        Returns:
            BGR heatmap image array.
        """
        if np.sum(self._accumulator) == 0:
            logger.warning("No bounce points accumulated — returning blank heatmap.")
            return self._blank_background()

        # ---- Gaussian blur to create smooth KDE-style density ----
        smoothed = cv2.GaussianBlur(
            self._accumulator,
            (self.kernel_size, self.kernel_size),
            sigmaX=0,
        )

        # ---- Normalise to 0–255 ----
        norm = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX)
        norm_u8 = norm.astype(np.uint8)

        # ---- Apply false colour ----
        coloured = cv2.applyColorMap(norm_u8, self.colormap)

        # ---- Blend with background ----
        if background is None:
            background = self._blank_background()
        else:
            background = cv2.resize(background, (self.canvas_w, self.canvas_h))

        mask = (norm_u8 > 0).astype(np.float32)[:, :, np.newaxis]
        blended = (
            background.astype(np.float32) * (1.0 - mask * self.alpha)
            + coloured.astype(np.float32) * (mask * self.alpha)
        ).clip(0, 255).astype(np.uint8)

        # ---- Overlay pitch markings ----
        blended = self._draw_pitch_markings(blended)

        # ---- Draw individual bounce dots ----
        blended = self._draw_bounce_dots(blended)

        logger.info(
            "Heatmap rendered | %d deliveries | canvas=%dx%d",
            len(self._points), self.canvas_w, self.canvas_h,
        )
        return blended

    def render_to_video_frame(
        self,
        video_frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Embed the heatmap into a corner of a video frame.

        Args:
            video_frame: Full-size BGR video frame.
            roi:         (x, y, w, h) region to place the heatmap.
                         Defaults to top-right corner thumbnail.

        Returns:
            Annotated video frame.
        """
        h, w = video_frame.shape[:2]
        if roi is None:
            thumb_w, thumb_h = int(w * 0.18), int(h * 0.35)
            roi = (w - thumb_w - 20, 20, thumb_w, thumb_h)

        rx, ry, rw, rh = roi
        heatmap = self.render()
        heatmap_resized = cv2.resize(heatmap, (rw, rh))

        out = video_frame.copy()
        out[ry:ry + rh, rx:rx + rw] = heatmap_resized

        # Draw border
        cv2.rectangle(out, (rx - 1, ry - 1), (rx + rw + 1, ry + rh + 1), (255, 255, 255), 1)
        cv2.putText(out, "Pitch Heatmap", (rx, ry - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_stats(self) -> HeatmapStats:
        """Return summary statistics across all accumulated deliveries."""
        if not self._points:
            return HeatmapStats(0, 0.0, 0.0, {})

        # Find density peak in world space
        smoothed = cv2.GaussianBlur(
            self._accumulator,
            (self.kernel_size, self.kernel_size),
            sigmaX=0,
        )
        peak_py, peak_px = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        wx_peak, wy_peak = self._canvas_to_world(int(peak_px), int(peak_py))

        # Zone distribution by bowling length
        zone_dist: Dict[str, int] = {}
        for bp in self._points:
            zone_dist[bp.length] = zone_dist.get(bp.length, 0) + 1

        return HeatmapStats(
            total_deliveries=len(self._points),
            hottest_zone_x_m=round(wx_peak, 3),
            hottest_zone_y_m=round(wy_peak, 3),
            zone_distribution=zone_dist,
        )

    def export_json(self, path: str) -> None:
        """Serialise all bounce points to JSON."""
        data = {
            "total_deliveries": len(self._points),
            "stats": self.compute_stats().to_dict(),
            "bounce_points": [asdict(bp) for bp in self._points],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Heatmap data exported to %s", path)

    def save(self, image: np.ndarray, path: str) -> None:
        """Save a rendered heatmap image to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        logger.info("Heatmap saved to %s", path)

    # ------------------------------------------------------------------
    # Coordinate utilities
    # ------------------------------------------------------------------

    def _world_to_canvas(self, wx: float, wy: float) -> Tuple[int, int]:
        """
        Convert world metres (wx, wy) to canvas pixel (px, py).

        Canvas layout:
          - Top    = bowling crease end (wy = 0)
          - Bottom = batting crease end (wy = PITCH_LENGTH_M)
          - Left   = off-side (wx = -CREASE_WIDTH_M/2)
          - Right  = leg-side (wx = +CREASE_WIDTH_M/2)
        """
        px = int((wx + CREASE_WIDTH_M / 2) * SCALE_X)
        py = int(wy * SCALE_Y)
        return px, py

    def _canvas_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Inverse of _world_to_canvas."""
        wx = px / SCALE_X - CREASE_WIDTH_M / 2
        wy = py / SCALE_Y
        return wx, wy

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _blank_background(self) -> np.ndarray:
        """Create a plain pitch-green background."""
        bg = np.full((self.canvas_h, self.canvas_w, 3), (34, 90, 34), dtype=np.uint8)
        return bg

    def _draw_pitch_markings(self, img: np.ndarray) -> np.ndarray:
        """Overlay ICC pitch line markings (creases, stumps)."""
        # ---- Batting crease (bottom) ----
        batt_y = self.canvas_h - 10
        cv2.line(img, (0, batt_y), (self.canvas_w, batt_y), (255, 255, 255), 1)
        cv2.putText(img, "Batting Crease", (5, batt_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # ---- Bowling crease (top) ----
        bowl_y = 10
        cv2.line(img, (0, bowl_y), (self.canvas_w, bowl_y), (255, 255, 255), 1)
        cv2.putText(img, "Bowling Crease", (5, bowl_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # ---- Stump lines (off, middle, leg) ----
        stump_xs = [-0.228, 0.0, 0.228]
        for sx in stump_xs:
            spx, _ = self._world_to_canvas(sx, 0)
            cv2.line(img, (spx, 0), (spx, self.canvas_h), (220, 220, 100), 1)

        # ---- Good-length zone marker ----
        gl_y1 = int(5.0 * SCALE_Y)
        gl_y2 = int(8.0 * SCALE_Y)
        cv2.rectangle(img, (0, gl_y1), (self.canvas_w, gl_y2), (255, 255, 0), 1)
        cv2.putText(img, "Good Length", (3, gl_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)
        return img

    def _draw_bounce_dots(self, img: np.ndarray) -> np.ndarray:
        """Draw individual delivery bounce dots."""
        for bp in self._points:
            px, py = self._world_to_canvas(bp.world_x, bp.world_y)
            if 0 <= px < self.canvas_w and 0 <= py < self.canvas_h:
                cv2.circle(img, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
        return img


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import uuid

    logging.basicConfig(level=logging.INFO)

    gen = HeatmapGenerator()

    # Simulate 20 deliveries landing mostly in the good-length zone
    for _ in range(20):
        bp = BouncePoint(
            delivery_id=str(uuid.uuid4())[:8],
            world_x=random.gauss(0.0, 0.15),
            world_y=random.gauss(7.0, 1.5),
            speed_kmh=random.uniform(120, 145),
            line=random.choice(["off_stump", "middle_stump", "outside_off"]),
            length=random.choice(["good_length", "short_of_good", "full"]),
        )
        gen.add_bounce(bp)

    img = gen.render()
    gen.save(img, "/tmp/test_heatmap.png")
    print(gen.compute_stats().to_dict())
    print("Heatmap saved to /tmp/test_heatmap.png")