"""
analytics/world3d_fit.py   [ADDITIVE — research: monocular 3D ball reconstruction]
==================================================================================
THE PROBLEM
-----------
A single behind-stumps camera gives a *ground-plane* homography H (image <-> pitch
at z=0). Projecting an AIRBORNE ball pixel back through H assumes z=0, so it lands
far down the ray from the true position — the parallax error that makes airborne
"world" coordinates useless (I measured points at y = -58 m earlier). So the naive
"work in world coordinates" idea fails for the flight.

THE FIX (this module)
---------------------
Recover a FULL 3x4 camera projection matrix P (not just the ground homography) from
stump correspondences: the stumps give known 3D points at BOTH the ground (base,
z=0) and the top (z=0.71 m), which is exactly the vertical reference a homography
lacks. With P we can *project* any 3D point to the image. We then fit a physically
real 3D ballistic trajectory (gravity in world-z, linear down-pitch + swing) whose
REPROJECTION through P matches the observed ball pixels, anchored by the bounce
touching the ground (z=0). This is standard monocular 3D reconstruction made
solvable by the physics prior: a parabola has ~7 parameters but a delivery gives
15-25 pixel observations (2 eqns each) -> heavily over-determined.

WHAT YOU GET vs the naive method
--------------------------------
* a real 3D world arc (X down-pitch, Y lateral, Z height) for the WHOLE flight,
* release height, bounce location and speed in true metres,
* every frame reprojects onto the observed pixels (so it is verifiable),
* honest error bars: accuracy is bounded by how well the stumps calibrate P, which
  for close-together far stumps is the limiting factor (quantified in the self-test).

STUMP_HEIGHT_M = top of stumps above ground.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

STUMP_WIDTH_M = 0.2286     # outer stump-to-stump width (matches calibrated_speed)
STUMP_HEIGHT_M = 0.71      # ground to top of stumps
G_MS2 = 9.81               # gravity


# ─────────────────────────────────────────────────────────────────────────────
# Camera calibration: full 3x4 projection matrix via DLT
# ─────────────────────────────────────────────────────────────────────────────
def dlt_calibrate(world_pts: np.ndarray, img_pts: np.ndarray) -> np.ndarray:
    """Solve x ~ P X for the 3x4 projection matrix P from >=6 3D<->2D pairs.

    world_pts: (N,3) metres ; img_pts: (N,2) pixels. Returns P (3,4)."""
    world_pts = np.asarray(world_pts, float)
    img_pts = np.asarray(img_pts, float)
    n = len(world_pts)
    if n < 6:
        raise ValueError("need >=6 correspondences to solve a full projection matrix")
    A = []
    for (X, Y, Z), (u, v) in zip(world_pts, img_pts):
        Xh = [X, Y, Z, 1.0]
        A.append([*Xh, 0, 0, 0, 0, *(-u * np.array(Xh))])
        A.append([0, 0, 0, 0, *Xh, *(-v * np.array(Xh))])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    return P / P[2, 3] if abs(P[2, 3]) > 1e-9 else P


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D world point(s) (…,3) to pixels (…,2)."""
    X = np.atleast_2d(X)
    Xh = np.hstack([X, np.ones((len(X), 1))])
    uvw = Xh @ P.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv


def stump_projection_matrix(near: dict, far: dict, pitch_len: float,
                            near_top_y: float, far_top_y: float) -> np.ndarray:
    """Build P from the 4 stump verticals: base (z=0) + top (z=STUMP_HEIGHT_M) of the
    near (down-pitch y=0) and far (y=pitch_len) stumps. `near`/`far` are the dicts
    from calibrated_speed.detect_both_stumps (cx, base_y, w); *_top_y are the box tops."""
    hw = STUMP_WIDTH_M / 2.0
    world, img = [], []
    for st, y_far, top_y in ((near, 0.0, near_top_y), (far, pitch_len, far_top_y)):
        cx, by, w = st["cx"], st["base_y"], st["w"]
        # base corners on the ground, top corners at stump height (same x extent)
        world += [[-hw, y_far, 0.0], [hw, y_far, 0.0],
                  [-hw, y_far, STUMP_HEIGHT_M], [hw, y_far, STUMP_HEIGHT_M]]
        img += [[cx - w / 2, by], [cx + w / 2, by],
                [cx - w / 2, top_y], [cx + w / 2, top_y]]
    return dlt_calibrate(np.array(world), np.array(img))


# ─────────────────────────────────────────────────────────────────────────────
# Naive baseline: ground-plane back-projection (what the current pipeline does)
# ─────────────────────────────────────────────────────────────────────────────
def ground_backproject(P: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Back-project pixels assuming z=0 (the parallax-prone method). Uses the
    ground homography implied by P (columns for X,Y,1)."""
    H = P[:, [0, 1, 3]]
    Hinv = np.linalg.inv(H)
    uv = np.atleast_2d(uv)
    out = []
    for u, v in uv:
        w = Hinv @ np.array([u, v, 1.0])
        out.append([w[0] / w[2], w[1] / w[2], 0.0])
    return np.array(out)


# ─────────────────────────────────────────────────────────────────────────────
# Physics-constrained monocular 3D ballistic fit
# ─────────────────────────────────────────────────────────────────────────────
BALL_DIAM_M = 0.072       # cricket ball diameter (~7.2 cm)


def decompose_camera(P: np.ndarray):
    """RQ-decompose P=K[R|t] -> K (intrinsics), R, camera centre C, and a P
    normalised so that row-3 gives metric camera-frame depth."""
    M = P[:, :3]
    K, R = np.linalg.qr(np.linalg.inv(M))          # RQ via QR of inverse
    K = np.linalg.inv(K)
    R = np.linalg.inv(R)
    # fix signs so K has positive diagonal
    Tsg = np.diag(np.sign(np.diag(K)))
    K = K @ Tsg; R = Tsg @ R
    K /= K[2, 2]
    C = -np.linalg.inv(M) @ P[:, 3]
    return K, R, C


@dataclass
class World3DConfig:
    reproj_reject_px: float = 8.0     # drop a detection whose reprojection error exceeds this
    bounce_anchor_w: float = 20.0     # weight on the "bounce touches ground (z=0)" constraint
    max_release_h: float = 2.6        # sanity clamp on release height (m)
    fit_gravity: bool = False         # if True, let g float (else fixed real gravity)
    size_depth_w: float = 6.0         # weight on the ball-apparent-size depth constraint
                                      #  (the cue that breaks the behind-stumps degeneracy)


def _model_3d(params, t, tb, g):
    """Segmented ballistic world path. t frames relative to release; tb=bounce frame.
    params = [X0,Vx, Y0,Vy, Z0,Vz, Vz2]  (X lateral, Y down-pitch, Z height)."""
    X0, Vx, Y0, Vy, Z0, Vz, Vz2 = params
    X = X0 + Vx * t
    Y = Y0 + Vy * t
    pre = t <= tb
    Z = np.empty_like(t)
    Z[pre] = Z0 + Vz * t[pre] - 0.5 * g * t[pre] ** 2
    # post-bounce: start at ground (z=0) at tb, rise then fall
    tp = t[~pre] - tb
    Z[~pre] = 0.0 + Vz2 * tp - 0.5 * g * tp ** 2
    return np.stack([X, Y, np.maximum(Z, 0.0 * Z)], axis=1)


def fit_ballistic_3d(P, frames, uv, fps, bounce_frame=None, pitch_len=20.12,
                     ball_px_diam=None, config: Optional[World3DConfig] = None) -> dict:
    """Reconstruct the 3D world trajectory whose reprojection matches `uv`.

    frames: (N,) int ; uv: (N,2) observed ball pixels ; fps for gravity scaling.
    ball_px_diam: optional (N,) apparent ball diameter in px per frame — the DEPTH
      cue that breaks the behind-stumps monocular degeneracy. When given, the fit
      also matches predicted camera-depth to (focal * real_diam / apparent_diam).
    Returns dict(world_pts, reproj_rmse_px, release_height_m, bounce_xy_m, params, kept).
    """
    cfg = config or World3DConfig()
    frames = np.asarray(frames, float)
    uv = np.asarray(uv, float)
    f0 = frames.min()
    t = frames - f0
    tb = (bounce_frame - f0) if bounce_frame is not None else (t.max() * 0.6)
    g = G_MS2 / (fps ** 2)                       # world metres per frame^2

    # camera decomposition for the size->depth constraint
    K, Rcam, Ccam = decompose_camera(P)
    fpx = float((abs(K[0, 0]) + abs(K[1, 1])) / 2.0)
    meas_depth = None
    if ball_px_diam is not None:
        d = np.asarray(ball_px_diam, float)
        meas_depth = fpx * BALL_DIAM_M / np.clip(d, 1e-3, None)   # metres from camera

    # initialise from ground back-projection (rough) + physical priors
    gb = ground_backproject(P, uv)
    Y0 = 0.5                                       # release ~0.5 m down-pitch from bowler stump
    Vy = max(0.05, (pitch_len * 0.8) / max(1.0, t.max()))
    init = [float(np.median(gb[:, 0])) * 0.1, 0.0, Y0, Vy, 2.0, 0.0, 6.0]

    def resid(p, mask=None):
        m = np.ones(len(t), bool) if mask is None else mask
        pred = _model_3d(p, t[m], tb, g)
        reproj = project(P, pred)
        r = list((reproj - uv[m]).ravel())
        zb = (p[4] + p[5] * tb - 0.5 * g * tb ** 2)
        r.append(cfg.bounce_anchor_w * zb)        # bounce touches ground
        if meas_depth is not None:                # size->depth constraint
            depth_pred = (pred - Ccam) @ Rcam[2]  # camera-frame depth of each point
            r += list(cfg.size_depth_w * (depth_pred - meas_depth[m]))
        return np.array(r)

    sol = least_squares(lambda p: resid(p), init, method="lm", max_nfev=6000)
    pred = _model_3d(sol.x, t, tb, g)
    reproj = project(P, pred)
    err = np.linalg.norm(reproj - uv, axis=1)

    # per-detection outlier rejection + one refit on inliers
    keep = err <= cfg.reproj_reject_px
    if keep.sum() >= 6 and keep.sum() < len(err):
        sol = least_squares(lambda p: resid(p, keep), sol.x, method="lm", max_nfev=6000)
        pred = _model_3d(sol.x, t, tb, g)
        reproj = project(P, pred)
        err = np.linalg.norm(reproj - uv, axis=1)
        keep = err <= cfg.reproj_reject_px

    rmse = float(np.sqrt(np.mean(err[keep] ** 2))) if keep.any() else float(np.sqrt(np.mean(err ** 2)))
    Z0 = sol.x[4]
    bounce_xy = _model_3d(sol.x, np.array([tb]), tb, g)[0]
    return dict(
        world_pts=pred, frames=frames.astype(int), reproj=reproj,
        reproj_rmse_px=round(rmse, 2),
        release_height_m=round(float(Z0), 2),
        bounce_xy_m=(round(float(bounce_xy[0]), 2), round(float(bounce_xy[1]), 2)),
        kept=keep, params=sol.x, gravity_per_frame2=g,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: synthetic camera + 3D delivery -> project -> reconstruct -> measure
# proves (a) the 3D fit recovers the real world arc, (b) how badly the naive
# ground back-projection fails for airborne points.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    fps = 60.0
    pitch = 20.12

    # --- ground-truth camera: behind bowler, ~2.4 m high, looking down pitch ---
    def look_at(eye, target, up=np.array([0, 0, 1.0])):
        f = target - eye; f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        Rt = np.stack([r, -u, f])
        t = -Rt @ eye
        return Rt, t
    eye = np.array([0.0, -3.0, 2.4]); target = np.array([0.0, 12.0, 0.4])
    Rt, tt = look_at(eye, target)
    fpx = 900.0; cxp, cyp = 239.0, 425.0            # ~478x850 frame
    K = np.array([[fpx, 0, cxp], [0, fpx, cyp], [0, 0, 1.0]])
    P_true = K @ np.hstack([Rt, tt.reshape(3, 1)])

    # --- ground-truth 3D delivery: release 2.1m, descend, bounce ~15m, rise ---
    N = 22; fr = np.arange(46, 46 + N); t = (fr - fr[0]).astype(float)
    tb = 15.0; g = G_MS2 / fps ** 2
    X = 0.15 - 0.01 * t                              # slight swing
    Y = 0.6 + (16.0 / N) * t                         # down the pitch
    Z = np.where(t <= tb, 2.1 + 0.02 * t - 0.5 * g * t ** 2,
                 np.maximum(0, 5.5 * (t - tb) - 0.5 * g * (t - tb) ** 2))
    gt3d = np.stack([X, Y, Z], axis=1)
    uv = project(P_true, gt3d) + rng.normal(0, 1.2, (N, 2))   # +noise

    # --- recover P from stump correspondences (base+top of near/far) ---
    hw = STUMP_WIDTH_M / 2.0
    sw, sh = STUMP_WIDTH_M, STUMP_HEIGHT_M
    stump_world = np.array([
        [-hw, 0, 0], [hw, 0, 0], [-hw, 0, sh], [hw, 0, sh],
        [-hw, pitch, 0], [hw, pitch, 0], [-hw, pitch, sh], [hw, pitch, sh]])
    stump_img = project(P_true, stump_world) + rng.normal(0, 0.5, (8, 2))
    P_est = dlt_calibrate(stump_world, stump_img)

    # simulate the ball's apparent diameter (depth cue) = f * D / camera-depth
    K_t, R_t, C_t = decompose_camera(P_true)
    ft = float((abs(K_t[0, 0]) + abs(K_t[1, 1])) / 2.0)
    depth_true = (gt3d - C_t) @ R_t[2]
    ball_px = ft * BALL_DIAM_M / depth_true + rng.normal(0, 0.4, N)   # +size noise

    res_no = fit_ballistic_3d(P_est, fr, uv, fps, bounce_frame=int(fr[0] + tb), pitch_len=pitch)
    res_sz = fit_ballistic_3d(P_est, fr, uv, fps, bounce_frame=int(fr[0] + tb),
                              pitch_len=pitch, ball_px_diam=ball_px)

    def rmse3(a, b):
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
    naive = ground_backproject(P_est, uv)            # the current pipeline's method
    print("\n=== MONOCULAR 3D RECONSTRUCTION — synthetic proof ===")
    print(f"observations: {N} frames, +1.2px noise, behind-stumps camera (down-pitch axis)")
    print(f"{'method':38}{'3D world RMSE':>15}{'release h':>11}{'bounce (X,Y)':>18}")
    print("-" * 82)
    print(f"{'naive ground back-projection (now)':38}{rmse3(naive, gt3d):>13.1f} m"
          f"{'-':>11}{'-':>18}")
    print(f"{'physics 3D fit, reprojection only':38}{rmse3(res_no['world_pts'], gt3d):>13.1f} m"
          f"{res_no['release_height_m']:>9} m{str(res_no['bounce_xy_m']):>18}")
    print(f"{'physics 3D fit + BALL-SIZE depth cue':38}{rmse3(res_sz['world_pts'], gt3d):>13.1f} m"
          f"{res_sz['release_height_m']:>9} m{str(res_sz['bounce_xy_m']):>18}")
    print("-" * 82)
    print(f"truth: release 2.10 m, bounce (X,Y)=({X[int(tb)]:.2f},{Y[int(tb)]:.2f}) m")
    print(f"reproj RMSE: no-size {res_no['reproj_rmse_px']}px | with-size {res_sz['reproj_rmse_px']}px")
