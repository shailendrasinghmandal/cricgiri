
import math
import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION  ← adjust these constants for your video / setup
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PATH   = "runs/detect/train-4/weights/best.pt"
INPUT_VIDEO  = "videos/test.mp4"
OUTPUT_VIDEO = "output/ball_analytics.mp4"

# Detection filters
CONF_THRESHOLD   = 0.25          # minimum YOLO confidence
MAX_BALL_SIZE_PX = 120           # ignore huge detections (reflections, stumps)
MAX_JUMP_PX      = 350           # maximum plausible inter-frame displacement
BALL_CLASS_ID    = 0             # class index for "cricket ball" in your model

# Tracking
MAX_LOST_FRAMES     = 10         # Kalman prediction frames before SEARCHING
MIN_MOVE_PX         = 4          # minimum movement to add trajectory point
MAX_TRAJECTORY_LEN  = 80         # kept points for drawing / analysis

# Speed estimation
PIXELS_PER_METER    = 95.0       # calibrate: pixel_length_of_known_object / meters
#   Example calibrations:
#     20 m pitch visible as 800 px  →  800 / 20  = 40.0   (default)
#     Cricket ball diameter 22 cm / 28 px  →  28 / 0.22 ≈ 127
#     Adjust to your specific camera angle.
SPEED_SMOOTH_WINDOW = 9          # rolling-average window (frames)

# Swing classification
SWING_MIN_FRAMES     = 15        # need at least this many trajectory points
SWING_PIXEL_THRESH   = 10        # horizontal displacement (px) to call a swing
CURVATURE_CONFIRM    = 0.55      # fraction of path segments that must agree


# ═══════════════════════════════════════════════════════════════════════════
# ❷  KALMAN FILTER
# ═══════════════════════════════════════════════════════════════════════════

class BallKalmanFilter:
    """
    4-state constant-velocity Kalman filter: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)

        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)

        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.initialized            = False

    def init(self, cx: int, cy: int) -> None:
        self.kf.statePost = np.array([[cx], [cy], [0.], [0.]], dtype=np.float32)
        self.initialized  = True

    def predict(self) -> tuple[int, int]:
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])

    def correct(self, cx: int, cy: int) -> tuple[int, int]:
        m = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(m)
        s = self.kf.statePost
        return int(s[0][0]), int(s[1][0])


# ═══════════════════════════════════════════════════════════════════════════
# ❸  SPEED CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

class SpeedCalculator:
    """
    Converts per-frame pixel displacement into real-world speed (km/h).

    Uses a rolling average over `window` frames to produce a stable,
    non-fluctuating readout.  Also tracks a final (settled) delivery speed.
    """

    def __init__(self, fps: float, pixels_per_meter: float = PIXELS_PER_METER,
                 window: int = SPEED_SMOOTH_WINDOW):
        self.fps              = fps
        self.pixels_per_meter = pixels_per_meter
        self._buf             = deque(maxlen=window)
        self.final_speed_kmh  = 0.0
        self._peak_kmh        = 0.0

    def update(self, prev: tuple | None, curr: tuple | None) -> float:
        """Returns smoothed speed in km/h."""
        if prev is None or curr is None:
            return self.smooth_kmh

        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        px_per_frame = math.hypot(dx, dy)
        mps          = (px_per_frame * self.fps) / self.pixels_per_meter
        kmh          = mps * 3.6
        self._buf.append(kmh)

        smooth = self.smooth_kmh
        if smooth > self._peak_kmh:
            self._peak_kmh     = smooth
            self.final_speed_kmh = smooth
        return smooth

    @property
    def smooth_kmh(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ❹  SWING CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

class SwingClassifier:
    """
    Analyses the horizontal component of the ball trajectory to classify:
        "Left Swing"  – ball curves consistently to the left
        "Right Swing" – ball curves consistently to the right
        "Straight"    – minimal lateral movement

    Algorithm
    ─────────
    1. Collect the full trajectory list (deque of (x, y) tuples).
    2. Need at least SWING_MIN_FRAMES points to make a decision.
    3. Compare first-quarter mean X vs last-quarter mean X.
    4. Additionally check frame-by-frame horizontal direction consistency.
    5. A swing label is confirmed only when both checks agree.
    """

    def __init__(self):
        self._label    = "Straight"
        self._conf     = 0.0           # internal confidence [0..1]
        self._lock_ctr = 0             # freeze label once confident

    def update(self, trajectory: deque) -> str:
        pts = list(trajectory)
        n   = len(pts)

        if n < SWING_MIN_FRAMES:
            return self._label

        # ── Quarter-split X-coordinate comparison ────────────────
        q     = max(n // 4, 3)
        x_early = np.mean([p[0] for p in pts[:q]])
        x_late  = np.mean([p[0] for p in pts[-q:]])
        net_dx  = x_late - x_early       # positive = moved right, negative = left

        # ── Frame-by-frame directional consistency ────────────────
        dx_signs = []
        for i in range(1, n):
            ddx = pts[i][0] - pts[i - 1][0]
            if abs(ddx) > 1:             # ignore sub-pixel jitter
                dx_signs.append(np.sign(ddx))

        if len(dx_signs) < 6:
            return self._label

        right_frac = dx_signs.count(1)  / len(dx_signs)
        left_frac  = dx_signs.count(-1) / len(dx_signs)

        # ── Classification logic ──────────────────────────────────
        if (net_dx > SWING_PIXEL_THRESH and right_frac > CURVATURE_CONFIRM):
            label = "Right Swing"
            conf  = right_frac
        elif (net_dx < -SWING_PIXEL_THRESH and left_frac > CURVATURE_CONFIRM):
            label = "Left Swing"
            conf  = left_frac
        else:
            label = "Straight"
            conf  = 1.0 - max(right_frac, left_frac)

        # ── Stability: lock when confident ───────────────────────
        if conf > 0.65:
            self._lock_ctr += 1
        else:
            self._lock_ctr = max(0, self._lock_ctr - 1)

        if self._lock_ctr >= 5:
            self._label = label   # stabilised — stop flickering
        elif self._lock_ctr == 0:
            self._label = label   # reset allowed only when confidence drops

        return self._label

    @property
    def label(self) -> str:
        return self._label


# ═══════════════════════════════════════════════════════════════════════════
# ❺  DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

# Colour palette (BGR)
_GREEN      = (0,   220,  80)
_YELLOW     = (0,   220, 220)
_RED        = (0,    40, 220)
_ORANGE     = (0,   140, 255)
_CYAN       = (220, 200,   0)
_WHITE      = (220, 220, 220)
_DIM        = (100, 100, 100)
_PANEL_BG   = (14,  14,   14)
_BORDER     = (70,  70,   70)


def draw_trajectory(frame: np.ndarray, points: deque) -> None:
    """
    Render a fading colour-gradient trajectory.
    Oldest points → blue/dim; newest → bright green.
    """
    pts = list(points)
    n   = len(pts)
    if n < 2:
        return

    for i in range(1, n):
        t      = i / (n - 1)                        # 0 → 1 along path
        alpha  = 0.25 + 0.75 * t
        blue   = int(200 * (1 - t))
        green  = int(220 * t)
        red    = 0
        color  = (blue, green, red)
        thick  = max(1, int(4 * t))
        over   = frame.copy()
        cv2.line(over, pts[i - 1], pts[i], color, thick)
        cv2.addWeighted(over, alpha, frame, 1 - alpha, 0, frame)


def draw_velocity_arrow(frame: np.ndarray,
                        curr: tuple[int, int],
                        prev: tuple[int, int] | None) -> None:
    """Directional arrow proportional to speed."""
    if prev is None:
        return
    dx, dy = curr[0] - prev[0], curr[1] - prev[1]
    mag = math.hypot(dx, dy)
    if mag < 2:
        return
    scale  = min(55.0 / max(mag, 1), 4.5)
    end    = (int(curr[0] + dx * scale), int(curr[1] + dy * scale))
    cv2.arrowedLine(frame, curr, end, _ORANGE, 2, tipLength=0.4)


def draw_ball(frame: np.ndarray,
              center: tuple[int, int],
              radius: int,
              conf: float,
              predicted: bool = False) -> None:
    """
    Tracking box + centre dot + highlight ring.
    Green = detected  |  Cyan = Kalman-predicted
    """
    cx, cy   = center
    half     = max(radius + 14, 38)
    x1, y1   = cx - half, cy - half
    x2, y2   = cx + half, cy + half

    if predicted:
        box_color  = _CYAN
        label      = "Predicted"
        thickness  = 2
    else:
        box_color  = _GREEN
        label      = f"Ball  {conf:.2f}"
        thickness  = 3

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
    cv2.circle(frame, center, 8,  _RED,    -1)   # solid red dot
    cv2.circle(frame, center, 20, _YELLOW,  2)   # yellow ring

    cv2.putText(frame, label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)


def _delivery_color(label: str) -> tuple[int, int, int]:
    return {
        "Right Swing": (60,  200, 255),
        "Left Swing" : (255, 160,  60),
        "Straight"   : (80,  220,  80),
    }.get(label, _WHITE)


def draw_hud(frame: np.ndarray,
             speed_kmh: float,
             delivery: str,
             state: str) -> None:
    """
    Semi-transparent analytics HUD in the top-left corner.

    ┌─────────────────────────┐
    │  BALL ANALYTICS         │
    │  SPEED   xxx.x km/h     │
    │  ████████░░░░░░░        │  ← speed bar
    │  DELIVERY  Right Swing  │
    │  STATUS   TRACKING      │
    └─────────────────────────┘
    """
    PX, PY = 14, 14
    PW, PH = 272, 130

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (PX, PY), (PX + PW, PY + PH), _PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    # Border
    cv2.rectangle(frame, (PX, PY), (PX + PW, PY + PH), _BORDER, 1)

    def label(text, x_off, y_off, color=_DIM, scale=0.42, font=cv2.FONT_HERSHEY_SIMPLEX):
        cv2.putText(frame, text,
                    (PX + x_off, PY + y_off),
                    font, scale, color, 1, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "BALL ANALYTICS",
                (PX + 10, PY + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Speed row ────────────────────────────────────────────────
    sp_color = _GREEN if state == "TRACKING" else _DIM
    label("SPEED",              10, 50)
    cv2.putText(frame, f"{speed_kmh:6.1f} km/h",
                (PX + 72, PY + 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.56, sp_color, 1, cv2.LINE_AA)

    # Speed bar
    bar_w    = PW - 20
    bar_fill = int(min(speed_kmh / 220.0, 1.0) * bar_w)
    cv2.rectangle(frame, (PX + 10, PY + 57), (PX + 10 + bar_w, PY + 65), (38, 38, 38), -1)
    if bar_fill > 0:
        cv2.rectangle(frame, (PX + 10, PY + 57), (PX + 10 + bar_fill, PY + 65), sp_color, -1)

    # ── Delivery row ─────────────────────────────────────────────
    d_color = _delivery_color(delivery)
    label("DELIVERY",           10, 90)
    cv2.putText(frame, delivery,
                (PX + 90, PY + 90),
                cv2.FONT_HERSHEY_DUPLEX, 0.56, d_color, 1, cv2.LINE_AA)

    # ── Status row ───────────────────────────────────────────────
    st_color = {
        "TRACKING"  : _GREEN,
        "PREDICTING": _CYAN,
        "SEARCHING" : (60, 80, 200),
    }.get(state, _DIM)
    label("STATUS",             10, 118)
    cv2.putText(frame, state,
                (PX + 78, PY + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, st_color, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
# ❻  MAIN TRACKING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs("output", exist_ok=True)

    # ── Model & video ────────────────────────────────────────────
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {INPUT_VIDEO}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(fps),
        (width, height),
    )

    # ── Trackers ─────────────────────────────────────────────────
    kf           = BallKalmanFilter()
    speed_calc   = SpeedCalculator(fps=fps)
    swing_cls    = SwingClassifier()

    trajectory      = deque(maxlen=MAX_TRAJECTORY_LEN)
    prev_center     : tuple[int, int] | None = None
    lost_frames     = 0
    ball_radius     = 20
    current_speed   = 0.0
    delivery_label  = "Straight"

    # ── Frame loop ───────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        boxes   = results[0].boxes

        # ── Pick best detection ───────────────────────────────────
        best_ball : tuple | None = None
        best_dist : float        = float("inf")

        for box in boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != BALL_CLASS_ID:
                continue
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1

            # Reject over-sized detections (not a cricket ball)
            if bw > MAX_BALL_SIZE_PX or bh > MAX_BALL_SIZE_PX:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r  = (bw + bh) // 4

            # Proximity filter — prefer detection closest to last known pos
            if prev_center is None:
                best_ball = (cx, cy, r, conf)
                break

            dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
            if dist < best_dist and dist < MAX_JUMP_PX:
                best_dist = dist
                best_ball = (cx, cy, r, conf)

        # ── Update Kalman + metrics ───────────────────────────────
        if best_ball is not None:
            raw_cx, raw_cy, raw_r, conf = best_ball
            ball_radius = max(raw_r, 10)
            lost_frames = 0

            if not kf.initialized:
                kf.init(raw_cx, raw_cy)
                smooth = (raw_cx, raw_cy)
            else:
                kf.predict()
                smooth = kf.correct(raw_cx, raw_cy)

            current_speed  = speed_calc.update(prev_center, smooth)
            delivery_label = swing_cls.update(trajectory)

            old_center  = prev_center
            prev_center = smooth

            _append_trajectory(trajectory, smooth)

            draw_trajectory(frame, trajectory)
            draw_ball(frame, smooth, ball_radius, conf, predicted=False)
            draw_velocity_arrow(frame, smooth, old_center)

            state = "TRACKING"

        elif kf.initialized and lost_frames < MAX_LOST_FRAMES:
            # ── Kalman prediction (ball momentarily invisible) ────
            lost_frames += 1
            smooth       = kf.predict()

            current_speed  = speed_calc.update(prev_center, smooth)
            delivery_label = swing_cls.update(trajectory)

            old_center  = prev_center
            prev_center = smooth

            _append_trajectory(trajectory, smooth)

            draw_trajectory(frame, trajectory)
            draw_ball(frame, smooth, ball_radius, conf=0.0, predicted=True)
            draw_velocity_arrow(frame, smooth, old_center)

            state = "PREDICTING"

        else:
            # ── Delivery completed ───────────────────────────────
         lost_frames += 1

         draw_trajectory(frame, trajectory)

         if len(trajectory) > 15:
          state = "COMPLETED"
         else:
          state = "SEARCHING"

        # ── HUD overlay ──────────────────────────────────────────
        display_speed = (
            speed_calc.final_speed_kmh
            if speed_calc.final_speed_kmh > 0
            else current_speed
        )

        draw_hud(
            frame,
            display_speed,
            delivery_label,
            state
        )

        out.write(frame)
        cv2.imshow("Cricket Ball Analytics", frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit early
            break



    # ── Cleanup ──────────────────────────────────────────────────
        # ──────────────────────────────────────────────
    # FINAL SUMMARY SCREEN
    # ──────────────────────────────────────────────

    summary_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Background
    summary_frame[:] = (15, 15, 15)

    # Title
    cv2.putText(
        summary_frame,
        "FINAL ANALYTICS",
        (width // 2 - 200, height // 2 - 120),
        cv2.FONT_HERSHEY_DUPLEX,
        1.6,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    # Speed
    cv2.putText(
        summary_frame,
        f"Speed      : {speed_calc.final_speed_kmh:.1f} km/h",
        (width // 2 - 200, height // 2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    # Delivery
    cv2.putText(
        summary_frame,
        f"Delivery   : {swing_cls.label}",
        (width // 2 - 200, height // 2 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 0),
        3,
        cv2.LINE_AA
    )

    # Status
    cv2.putText(
        summary_frame,
        "Status     : COMPLETED",
        (width // 2 - 200, height // 2 + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 200, 255),
        3,
        cv2.LINE_AA
    )

    # Show for 3 seconds
    summary_frames = int(fps * 3)

    for _ in range(summary_frames):

        out.write(summary_frame)

        cv2.imshow(
            "Cricket Ball Analytics",
            summary_frame
        )

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Output saved → {OUTPUT_VIDEO}")
    print(f"  Final delivery speed : {speed_calc.final_speed_kmh:.1f} km/h")
    print(f"  Delivery type        : {swing_cls.label}")


# ═══════════════════════════════════════════════════════════════════════════
# ❼  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _append_trajectory(traj: deque, pt: tuple[int, int]) -> None:
    """Append only if the ball moved enough (avoids stuttering dots)."""
    if not traj:
        traj.append(pt)
        return
    if math.hypot(pt[0] - traj[-1][0], pt[1] - traj[-1][1]) > MIN_MOVE_PX:
        traj.append(pt)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
