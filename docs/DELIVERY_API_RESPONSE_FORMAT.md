# CricGiri Delivery API — Response Format Reference

**Endpoint:** `POST /analyze` (multipart form)
**Content-Type of response:** `application/json` (pretty-printed / indented)

### Request

| Form field | Required | Type | Notes |
|---|---|---|---|
| `video` | yes | file | Cricket delivery clip: `.mp4 / .mov / .avi / .mkv / .webm` |
| `pitch_length_yards` | optional | number | Real pitch length in yards (14–22). Scales the whole world output. Default **22**. |
| `pitch_length` | optional | number | Alternative: pitch length in **metres**. Ignored if `pitch_length_yards` is sent. |

> Speed and all down-pitch (`y_m`) values scale with pitch length. Send the **actual** pitch length of the footage for correct numbers. Processing takes ~40–120 s per clip (full detection pipeline on GPU).

---

## Top-level object

| Field | Type | Meaning |
|---|---|---|
| `source_video` | string | Uploaded file name (internal). |
| `fps` | number | Frames per second of the clip. |
| `total_frames` | int | Frame count of the clip. |
| `total_deliveries` | int | Number of deliveries detected (currently 1 per clip). `0` ⇒ no valid ball track (see `status`/`reason`). |
| `pipeline_version` | string | Engine version tag. |
| `detection_conf_threshold` | number | **Detection acceptance floor (a setting, NOT a quality score).** `0.05` is deliberately low so the small, fast ball is not missed. |
| `detection_conf_threshold_note` | string | Reminder that the above is a threshold, not a confidence. |
| `pitch_length_yards` | number | Pitch length used (root-level fallback; also per-delivery). |
| `deliveries` | array | One delivery object (below). |
| `result_id` | string | Unique id for this analysis. |
| `output_video`, `output_video_url` | string | Path to the rendered trajectory video: `GET /analysis/{result_id}/video`. |
| `processing_sec` | number | Server processing time. |

If no ball is found: `total_deliveries: 0`, `deliveries: []`, plus `status: "NO_TRACK"` and a `reason`.

---

## Delivery object (`deliveries[0]`)

### Identity & pitch
| Field | Type | Meaning |
|---|---|---|
| `delivery_id` | string | Unique id for this delivery. |
| `pitch_length_m` | number | Pitch length used, in metres. |
| `pitch_length_yards` | number | Pitch length used, in yards. |
| `frame_start`, `frame_end` | int | Ball track's first/last frame. |

### Track quality
| Field | Type | Meaning |
|---|---|---|
| `track.num_points` | int | Number of ball positions in the track. |
| `track.average_confidence` | number (0–1) | **Real detection quality** — mean YOLO confidence of the accepted ball points (e.g. `0.74`). |
| `track.physics_removed_points` | int | Points rejected by the physics-validity gate (0 = clean). |
| `track.physics_verdict` | string | `"valid"` when the arc is physically consistent. |
| `track.post_bounce_recovered` | bool | Whether points after the bounce were recovered. |

### Bounce
| Field | Type | Meaning |
|---|---|---|
| `bounce` | object / null | Pixel-space bounce: `{frame_index, x_pixel, y_pixel}`. `null` = full toss (no pitch). |
| `bounce_world` | object / null | World-space bounce `{x_m, y_m}`. **Presence ⇒ draw the bounce marker.** `null` ⇒ full toss. |
| `bounce_point` | object / null | Same as `bounce_world` as `{x, y}` (legacy alias). |

### Trajectory — **four equivalent forms of the same path**
| Field | Type | Use for |
|---|---|---|
| `world_trajectory` | array of `[x_m, y_m, z_m]` | **Primary 3D source of truth.** See axis meaning below. |
| `ball_flight_position` | array of `[x_m, y_m, z_m]` | Alias of `world_trajectory`. |
| `trajectory_3d` | array of objects | Same points with `frame_index`, `time_sec`, `x_m`, `y_m`, `z_m` — use when you want **time** per point (animation). |
| `trajectory_pixels` | array of objects | 2D pixel path `{frame_index, time_sec, x_pixel, y_pixel}` — for overlaying on the original video. |
| `trajectory_matrices` | array of 4×4 | Each point as a homogeneous **pose matrix** for 3D engines (see below). |
| `model_matrix` | 4×4 | Scene placement transform for the whole trajectory (identity by default). |
| `matrix_convention` | string | Documents the matrix layout & axes. |

**Axis meaning (all trajectory forms):**
- `x_m` — lateral offset from the centre line, in metres (− = one side, + = the other).
- `y_m` — distance down the pitch from the bowler's stumps, in metres (`0 → pitch_length_yards × 0.9144`).
- `z_m` — height above the ground, in metres. Full at release → `0` at the bounce → rises after.

> **Honesty note:** `x_m` and `y_m` are *measured* from the video. `z_m` (height) is a *physics-shaped estimate* — a single camera cannot truly measure height. The arc **shape** is correct (rise, dip at bounce, kick-up); the exact height number is indicative, not DRS-grade.

### The 4×4 matrices (`trajectory_matrices`)
One 4×4 per trajectory point, for a 3D (360°-orbitable) renderer.
- **Layout:** row-major 4×4. Rows 0–2 = `[right | up | forward | translation]`; row 3 = `[0,0,0,1]`.
- **Position:** the last value of rows 0–2 (column 3) is the ball's `[x, y, z]`.
- **Orientation:** `forward` = direction of travel; `up` ≈ world +Z (height); `right` = forward × up.
- Directly consumable as a three.js `Matrix4` / Unity pose. Derived from `world_trajectory` — same data, engine-ready form.

> **Frontend must confirm two conventions** or the arc may render rotated/flipped:
> 1. **Row-major** (used here) vs column-major.
> 2. **Axis mapping** `x=lateral, y=down-pitch, z=height` (z-up) vs the engine's own (three.js is y-up).
> Both are one-line adjustments on our side — tell us the engine and we'll match it.

### Line / Length / Speed
| Field | Type | Meaning |
|---|---|---|
| `line.label` | string | `straight / wide_off / wide_leg / ...`. |
| `line.confidence`, `line.reliability` | number / string | **Indicative only** — line is calibrated off stump width, so confidence is capped (~0.45). Show the label tagged "indicative"; do not surface the decimal. |
| `length.label` | string | `full / good_length / short / ...`. |
| `length.confidence` | number | Length is metric-reliable (higher confidence, e.g. 0.83). |
| `length.distance_from_batsman_m` | number | Bounce distance from the batsman, metres. |
| `speed.kmph` / `speed_kmph` | number | Release→bounce speed. Two forms provided. |
| `speed.confidence`, `speed.status` | number / string | Quality of the speed estimate. Scales with the pitch length you sent. |

### Swing / Spin — factors, not centimetres
| Field | Type | Meaning |
|---|---|---|
| `swing_sf` / `swing_factor` | number (0–1) | Swing magnitude as a **factor** (no cm). |
| `swing_type` | string | `inswing / outswing / straight` (direction hint). |
| `swing_confidence`, `swing_status` | number / string | **Indicative direction only** — single-camera swing is not precisely measurable. |
| `spin_factor` | number (0–1) | Path-curvature spin proxy. |
| `spin_degree` | number | Same, expressed in degrees. |
| `spin_unit`, `spin_status` | string | Documents that spin is a curvature proxy, not measured RPM. |

### Confidence — what to show the client
| Field | Type | Meaning |
|---|---|---|
| `confidence_score` | number (0–1) | **Overall delivery quality headline** (e.g. `0.92`). |
| `confidence_pct` | int | Same as a percentage (e.g. `92`) — **show this**. |
| `confidence_label` | string | `High / Medium / Low` — **show this**. |
| `raw_confidence_score` | number | Internal intermediate — not for display. |
| `physically_valid` | bool | Whether the arc passed the physics gate. |
| `heatmap_points` | array | `[x_m, y_m]` bounce spot(s) for a pitch heatmap. |

---

## Recommended display mapping (frontend)

| Show to the client | Field |
|---|---|
| 3D trajectory | `world_trajectory` (or `trajectory_matrices` if the engine wants poses) |
| Bounce marker | `bounce_world` (only if present) |
| Overall trust badge | `confidence_pct` + `confidence_label` → e.g. **"92% · High"** |
| Speed | `speed_kmph` |
| Length | `length.label` |
| Line | `line.label` tagged *"indicative"* |
| Swing | `swing_type` (+ `swing_sf`) tagged *"indicative"* |

**Do NOT show the client:** `detection_conf_threshold`, `raw_confidence_score`, or the raw sub-confidence decimals (`line.confidence`, `swing_confidence`). They are internal or intentionally indicative.

---

## Trimmed example (arrays shortened)

```json
{
  "source_video": "clip.mp4",
  "fps": 30,
  "total_frames": 123,
  "total_deliveries": 1,
  "detection_conf_threshold": 0.05,
  "detection_conf_threshold_note": "detection acceptance floor, not a quality score",
  "pitch_length_yards": 22,
  "deliveries": [
    {
      "delivery_id": "delivery_ea63b4",
      "pitch_length_m": 20.12,
      "pitch_length_yards": 22,
      "frame_start": 23,
      "frame_end": 54,
      "track": { "num_points": 29, "average_confidence": 0.737,
                 "physics_removed_points": 0, "physics_verdict": "valid",
                 "post_bounce_recovered": false },
      "bounce_world": { "x_m": -0.09, "y_m": 12.95 },
      "world_trajectory": [
        [0.55, 1.0, 1.95],
        [0.09, 5.18, 1.27],
        [-0.09, 12.95, 0.0],
        [-0.21, 19.52, 0.65]
      ],
      "trajectory_matrices": [
        [[1,0,0,0.55],[0,0.1644,0.9864,1.0],[0,0.9864,-0.1644,1.95],[0,0,0,1]]
      ],
      "model_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "matrix_convention": "row_major_4x4; rows0-2=[right|up|forward|translation], row3=[0,0,0,1]; coords=[x_lateral_m,y_downpitch_m,z_height_m]",
      "line": { "label": "wide_leg", "confidence": 0.45, "reliability": "indicative" },
      "length": { "label": "good_length", "confidence": 0.83, "distance_from_batsman_m": 2.83 },
      "speed_kmph": 88,
      "speed": { "kmph": 88, "confidence": 0.85, "status": "estimated" },
      "swing_sf": 0.204,
      "swing_type": "inswing",
      "spin_degree": 9.2,
      "confidence_score": 0.92,
      "confidence_pct": 92,
      "confidence_label": "High"
    }
  ]
}
```
