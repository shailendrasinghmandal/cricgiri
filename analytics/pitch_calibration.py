"""
analytics/pitch_calibration.py
===============================
Converts pixel coordinates to real-world (metric) coordinates using
stump positions as calibration anchors and OpenCV homography.

The pitch coordinate system is defined as:
  - Origin (0, 0) at the bowling-crease centre
  - X-axis : lateral (off-side positive)
  - Y-axis : longitudinal (batting-crease direction)
  - Z-axis : vertical (not modelled in 2-D homography; handled separately)

Author: Cricket Analytics Engine
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants  (ICC regulation pitch dimensions in metres)
# ---------------------------------------------------------------------------

PITCH_LENGTH_M: float = 20.12   # bowling crease to batting crease
STUMP_SPACING_M: float = 0.228  # centre-to-centre (off, middle, leg)
STUMP_HEIGHT_M: float = 0.711   # ground to bail groove
CREASE_WIDTH_M: float = 3.66    # overall crease line half-width (approx)

# ---------------------------------------------------------------------------
# Calibration result dataclass
# ---------------------------------------------------------------------------

class CalibrationResult:
    """
    Holds the homography matrix and exposes coordinate transform methods.
    """

    def __init__(
        self,
        H: np.ndarray,
        H_inv: np.ndarray,
        pixel_src: np.ndarray,
        world_dst: np.ndarray,
        reprojection_error: float,
    ):
        self.H = H                             # pixel → world
        self.H_inv = H_inv                     # world → pixel
        self.pixel_src = pixel_src             # calibration points (pixels)
        self.world_dst = world_dst             # calibration points (metres)
        self.reprojection_error = reprojection_error

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """
        Map a single pixel point to world (metre) coordinates.

        Args:
            px, py: Pixel coordinates.

        Returns:
            (wx, wy) in metres.
        """
        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H)
        wx, wy = float(result[0, 0, 0]), float(result[0, 0, 1])
        return wx, wy

    def world_to_pixel(self, wx: float, wy: float) -> Tuple[float, float]:
        """
        Map a world (metre) point back to pixel coordinates.

        Args:
            wx, wy: World coordinates in metres.

        Returns:
            (px, py) in pixels.
        """
        pt = np.array([[[wx, wy]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H_inv)
        px, py = float(result[0, 0, 0]), float(result[0, 0, 1])
        return px, py

    def pixel_trajectory_to_world(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Batch-transform a trajectory list from pixel to world space.

        Args:
            points: List of (px, py) tuples.

        Returns:
            List of (wx, wy) tuples in metres.
        """
        if not points:
            return []
        arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(arr, self.H)
        return [(float(p[0, 0]), float(p[0, 1])) for p in transformed]

    def to_dict(self) -> dict:
        return {
            "reprojection_error_px": round(self.reprojection_error, 4),
            "homography_matrix": self.H.tolist(),
        }


# ---------------------------------------------------------------------------
# PitchCalibrator
# ---------------------------------------------------------------------------

class PitchCalibrator:
    """
    Calibrates the camera-to-pitch homography using detected stump positions.

    Calibration workflow
    --------------------
    1. Detect off-stump, middle-stump, leg-stump pixel positions from a
       still frame (or manually annotate them).
    2. Provide the corresponding known world (metre) coordinates.
    3. Call `calibrate()` to compute the homography.
    4. Use the returned `CalibrationResult` for all coordinate transforms.

    Known-point layout (default, bowling-end camera):
        Pixel source            World (metres, bowling-crease origin)
        off-stump base          (-0.228,  0.0)
        middle-stump base       ( 0.0,    0.0)
        leg-stump base          (+0.228,  0.0)
        off-stump popping crease  (-0.228, -1.22)   [popping crease = 1.22 m]
        leg-stump popping crease  (+0.228, -1.22)
    """

    # Default world coordinates for a standard 5-point calibration
    # (bowling-end stumps as seen from behind the wicket)
    DEFAULT_WORLD_POINTS: np.ndarray = np.array([
        [-STUMP_SPACING_M,        0.0  ],   # off-stump base
        [ 0.0,                    0.0  ],   # middle-stump base
        [ STUMP_SPACING_M,        0.0  ],   # leg-stump base
        [-STUMP_SPACING_M,       -1.22 ],   # off-stump popping crease
        [ STUMP_SPACING_M,       -1.22 ],   # leg-stump popping crease
    ], dtype=np.float32)

    def __init__(self, method: int = cv2.RANSAC, ransac_reproj_threshold: float = 3.0):
        """
        Args:
            method: OpenCV homography estimation method.
            ransac_reproj_threshold: Max reprojection error for RANSAC inliers.
        """
        self.method = method
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self._result: Optional[CalibrationResult] = None

    # ------------------------------------------------------------------
    # Primary calibration method
    # ------------------------------------------------------------------

    def calibrate(
        self,
        pixel_points: np.ndarray,
        world_points: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """
        Compute homography from pixel → world given calibration anchor points.

        Args:
            pixel_points: (N, 2) array of pixel coordinates matching world_points.
            world_points: (N, 2) array of world (metre) coordinates.
                          If None, DEFAULT_WORLD_POINTS is used (N must equal 5).

        Returns:
            CalibrationResult with transform methods.

        Raises:
            ValueError: If fewer than 4 point pairs are supplied.
            RuntimeError: If OpenCV fails to compute a valid homography.
        """
        if world_points is None:
            world_points = self.DEFAULT_WORLD_POINTS

        pixel_points = np.array(pixel_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)

        if len(pixel_points) < 4:
            raise ValueError(
                f"At least 4 point pairs required for homography; got {len(pixel_points)}."
            )
        if pixel_points.shape != world_points.shape:
            raise ValueError(
                f"pixel_points shape {pixel_points.shape} != world_points shape {world_points.shape}"
            )

        logger.info(
            "Computing homography from %d calibration points …", len(pixel_points)
        )

        # Compute H: pixel → world
        H, mask = cv2.findHomography(
            pixel_points,
            world_points,
            method=self.method,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )

        if H is None:
            raise RuntimeError("cv2.findHomography returned None — calibration failed.")

        # Compute inverse H: world → pixel
        H_inv = np.linalg.inv(H)

        # Compute mean reprojection error on inliers
        reprojection_error = self._compute_reprojection_error(
            pixel_points, world_points, H, mask
        )

        logger.info(
            "Homography computed. Mean reprojection error: %.3f px", reprojection_error
        )
        if reprojection_error > 5.0:
            logger.warning(
                "High reprojection error (%.3f px) — calibration quality may be poor.",
                reprojection_error,
            )

        self._result = CalibrationResult(
            H=H,
            H_inv=H_inv,
            pixel_src=pixel_points,
            world_dst=world_points,
            reprojection_error=reprojection_error,
        )
        return self._result

    def calibrate_from_stump_detections(
        self,
        stump_detections: List[Dict],
        world_points: Optional[np.ndarray] = None,
        swap_ends: bool = False,
    ) -> CalibrationResult:
        """
        Derive pixel anchor points from YOLO stump bounding-box detections.

        Expected detection dict keys:
            label  : 'off_stump' | 'middle_stump' | 'leg_stump' | 'off_crease' | 'leg_crease' | 'batsman_off' | 'batsman_mid' | 'batsman_leg'
            bbox   : [x1, y1, x2, y2]
        """
        pixel_pts = []
        world_pts = []

        detection_map = {d["label"]: d for d in stump_detections}

        # Define all potential labels and their corresponding world coordinates
        if not swap_ends:
            label_world_coords = {
                "off_stump":   [-STUMP_SPACING_M,        0.0  ],
                "middle_stump": [ 0.0,                    0.0  ],
                "leg_stump":    [ STUMP_SPACING_M,        0.0  ],
                "off_crease":   [-STUMP_SPACING_M,       -1.22 ],
                "leg_crease":   [ STUMP_SPACING_M,       -1.22 ],
                "batsman_off":  [-STUMP_SPACING_M,       20.12 ],
                "batsman_mid":  [ 0.0,                   20.12 ],
                "batsman_leg":  [ STUMP_SPACING_M,       20.12 ],
            }
        else:
            label_world_coords = {
                "off_stump":   [-STUMP_SPACING_M,       20.12 ],
                "middle_stump": [ 0.0,                   20.12 ],
                "leg_stump":    [ STUMP_SPACING_M,       20.12 ],
                "batsman_off":  [-STUMP_SPACING_M,        0.0  ],
                "batsman_mid":  [ 0.0,                    0.0  ],
                "batsman_leg":  [ STUMP_SPACING_M,        0.0  ],
            }

        # Extract whatever detected points match our labels
        for label, coord in label_world_coords.items():
            if label in detection_map:
                bbox = detection_map[label]["bbox"]
                bx = (bbox[0] + bbox[2]) / 2.0
                # Use bottom of bounding box as the base point
                by = float(bbox[3])
                pixel_pts.append([bx, by])
                world_pts.append(coord)

        # Fallback: If we only have the 3 bowler stumps, we cannot compute perspective homography (requires >= 4 points).
        # We project 2 virtual popping crease points (off_crease and leg_crease) using standard camera perspective.
        if len(pixel_pts) == 3 and "middle_stump" in detection_map and "off_stump" in detection_map and "leg_stump" in detection_map:
            off_bbox = detection_map["off_stump"]["bbox"]
            leg_bbox = detection_map["leg_stump"]["bbox"]
            mid_bbox = detection_map["middle_stump"]["bbox"]

            # Estimate stump height in pixels
            h_stumps = np.mean([
                off_bbox[3] - off_bbox[1],
                leg_bbox[3] - leg_bbox[1],
                mid_bbox[3] - mid_bbox[1]
            ])

            # The popping crease is closer to the camera (lower down on screen)
            # We project the crease 1.22 meters in front of the stumps, which is approx 1.7x stump height on screen.
            dy = h_stumps * 1.7
            
            # Extract bases
            bx_off, by_off = pixel_pts[0]
            bx_leg, by_leg = pixel_pts[2]
            
            # Since the camera perspective widens closer to the lens:
            cx_screen = pixel_pts[1][0]  # middle stump X
            bx_off_crease = cx_screen + (bx_off - cx_screen) * 1.28
            bx_leg_crease = cx_screen + (bx_leg - cx_screen) * 1.28
            
            by_crease = np.mean([by_off, by_leg]) + dy

            # Append the 2 projected popping crease points
            pixel_pts.append([bx_off_crease, by_crease])
            world_pts.append([-STUMP_SPACING_M, -1.22])

            pixel_pts.append([bx_leg_crease, by_crease])
            world_pts.append([STUMP_SPACING_M, -1.22])

        pixel_arr = np.array(pixel_pts, dtype=np.float32)
        world_arr = np.array(world_pts, dtype=np.float32)

        return self.calibrate(pixel_arr, world_arr)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def draw_calibration_overlay(
        self,
        frame: np.ndarray,
        result: Optional[CalibrationResult] = None,
    ) -> np.ndarray:
        """
        Draw calibration anchor points on a frame for visual verification.

        Args:
            frame:  BGR frame.
            result: CalibrationResult (uses self._result if None).

        Returns:
            Annotated BGR frame copy.
        """
        result = result or self._result
        if result is None:
            logger.warning("No calibration result available for overlay.")
            return frame.copy()

        vis = frame.copy()
        labels = ["Off", "Mid", "Leg", "Off-C", "Leg-C"]

        for i, (px, py) in enumerate(result.pixel_src):
            cx, cy = int(px), int(py)
            cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                vis,
                labels[i] if i < len(labels) else str(i),
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Error label
        cv2.putText(
            vis,
            f"Reproj err: {result.reprojection_error:.2f}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        return vis

    @staticmethod
    def _compute_reprojection_error(
        src: np.ndarray,
        dst: np.ndarray,
        H: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> float:
        """Mean reprojection error in pixels for inlier points."""
        src_h = src.reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(src_h, H).reshape(-1, 2)
        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
            if inlier_mask.sum() == 0:
                inlier_mask = np.ones(len(src), dtype=bool)
        else:
            inlier_mask = np.ones(len(src), dtype=bool)

        errors = np.linalg.norm(projected[inlier_mask] - dst[inlier_mask], axis=1)
        return float(np.mean(errors))

    def save_calibration(self, path: str) -> None:
        """Persist the calibration matrix to disk (numpy .npz)."""
        if self._result is None:
            raise RuntimeError("No calibration to save. Run calibrate() first.")
        np.savez(
            path,
            H=self._result.H,
            H_inv=self._result.H_inv,
            pixel_src=self._result.pixel_src,
            world_dst=self._result.world_dst,
            reprojection_error=np.array([self._result.reprojection_error]),
        )
        logger.info("Calibration saved to %s", path)

    @classmethod
    def load_calibration(cls, path: str) -> CalibrationResult:
        """Load a previously saved calibration from a .npz file."""
        data = np.load(path)
        result = CalibrationResult(
            H=data["H"],
            H_inv=data["H_inv"],
            pixel_src=data["pixel_src"],
            world_dst=data["world_dst"],
            reprojection_error=float(data["reprojection_error"][0]),
        )
        logger.info(
            "Calibration loaded from %s (reproj err: %.3f px)",
            path,
            result.reprojection_error,
        )
        return result


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Simulate detected stump pixel positions on a 1920×1080 frame
    mock_pixel_points = np.array([
        [880, 650],   # off-stump base
        [960, 655],   # middle-stump base
        [1040, 650],  # leg-stump base
        [870, 540],   # off-stump popping crease
        [1050, 540],  # leg-stump popping crease
    ], dtype=np.float32)

    calibrator = PitchCalibrator()
    result = calibrator.calibrate(mock_pixel_points)

    print("Homography H:")
    print(result.H)
    print(f"Reprojection error: {result.reprojection_error:.4f} px")

    # Test round-trip accuracy
    wx, wy = result.pixel_to_world(960, 655)
    print(f"Middle-stump pixel (960, 655) → world ({wx:.4f}, {wy:.4f}) m")

    back_px, back_py = result.world_to_pixel(wx, wy)
    print(f"World ({wx:.4f}, {wy:.4f}) → pixel ({back_px:.1f}, {back_py:.1f})")