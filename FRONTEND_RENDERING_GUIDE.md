# CricGiri — Rendering a Delivery JSON in 3D (Frontend Guide)

How to turn the API's delivery JSON into a clean Hawk-Eye-style 3D trajectory.

> **The JSON is directly renderable.** Plot `world_trajectory` as given — do not smooth
> it, do not re-derive the bounce, do not flip the lateral axis. The engine already
> guarantees a smooth arc, a bounce that touches the ground, and the correct side.
> If a render looks wrong, check §5 before changing the data.

---

## 1. Coordinate system

Each point in `world_trajectory` is `[x_m, y_m, z_m]`, all in **metres**:

| axis | meaning | range |
|---|---|---|
| `x_m` | **lateral** — sideways offset from the pitch centre line | ~ −1.0 … +1.0 |
| `y_m` | **down-pitch** — distance from the **bowler's** end | 0 … pitch length (20.12) |
| `z_m` | **height** above the ground | 0 … ~2.0 |

- `y_m = 0` → bowler's end, `y_m = 20.12` → batsman's end. The batsman always stands at
  the **far end**, so larger `y_m` = closer to the batsman.
- `z_m = 0` → the ball is on the ground. This happens exactly once: at the bounce.
- **Sign convention: leg side = +x, off side = −x.** This matches the `line` label.
  If your scene renders mirrored, negate `x` once in `toScene` — do not patch the data.

## 2. Scene mapping

Sort by `y_m` ascending (bowler → batsman) before drawing; array order isn't guaranteed.
Then map to a Y-up scene with the pitch running along Z:

```js
const halfLen = pitchLengthM / 2;              // 20.12 / 2 = 10.06
function toScene(p) {
  return {
    x: p[0],                    // lateral
    y: p[2] + 0.02,             // height (+2cm so the line isn't buried in the turf)
    z: -halfLen + p[1],         // depth: bowler at -halfLen, batsman at +halfLen
  };
}

const ordered = [...delivery.world_trajectory].sort((a, b) => a[1] - b[1]);
const points  = ordered.map(toScene);
```

`pitchLengthM = pitch_length_yards * 0.9144` (22 yd → 20.12 m).

## 3. What to draw

1. **Pitch strip**, plus a centre line from the bowler's stumps to the batsman's stumps.
2. **Trajectory** — a tube/line through `points`. A Catmull-Rom or similar interpolating
   spline is fine; the points are already dense (~0.7 m apart) and smooth, so a plain
   polyline also looks correct. **Do not fit your own curve to them.**
3. **Bounce marker** — a flat disc on the ground:
   ```js
   const bx = delivery.bounce_world.x_m;
   const bz = -halfLen + delivery.bounce_world.y_m;
   drawDisc({ x: bx, y: 0.001, z: bz, radius: 0.18 });
   ```
   `bounce_world` is guaranteed to be the arc's lowest point, so the disc always sits
   directly under the trajectory.
4. **Ball** at the last point, at its own height (`points.at(-1)`).

## 4. What the engine guarantees

You can rely on these — they're enforced in `scripts/run_demo_testing.py`:

| Guarantee | Why it matters |
|---|---|
| Height is a **continuous gravity curve** `h·(1−t²)` | ball holds height after release then steepens into the bounce — no staircase, no zig-zag, no "slide" |
| **Exactly one ground contact**, `z_m = 0` at the bounce | the ball visibly pitches |
| `bounce_world` = the arc's **lowest point** | marker can never disagree with the drawn curve |
| Bounce is **never at the release end** (`y_m > 1.5`) | validated; falls back to the position implied by `length` if detection is degenerate |
| Ball **stays on one side** of the centre line | side comes from the `line` label, clamped so drift can't cross over |
| Points evenly spaced ~0.7 m in `y_m` | no long straight chord jumping across a detection gap |

## 5. Validation (smoke test, not correction)

If a render looks wrong, assert these against the JSON. A failure means an engine bug —
report it rather than patching in the client, otherwise every client re-implements the
same workaround:

- [ ] **Single ground contact** — `min(z_m) === 0`, and only one point at 0.
- [ ] **Bounce in range** — `1.5 < bounce_world.y_m < pitchLengthM`.
- [ ] **Marker matches curve** — `bounce_world.x_m` equals the x of the lowest point.
- [ ] **One side only** — `min(x_m)` and `max(x_m)` do not straddle 0.
- [ ] **No zig-zag** — the sign of `Δx` reverses at most once (one gentle swing).
- [ ] **Arc, not slide** — a quarter of the way to the bounce the ball is still near its
      release height (≳75% of it), not already halfway down.
- [ ] **No gaps** — consecutive `y_m` differ by ≲1 m.

## 6. Fields you need

```
world_trajectory    [[x_m, y_m, z_m], ...]   the path — render as-is
bounce_world        { x_m, y_m }             where it pitched (arc's lowest point)
line.label          wide_off | off_stump | middle_stump | leg_stump | wide_leg | down_leg
length.label        yorker | full_length | good_length | short_length
speed_kmph          number                   display only
confidence_pct      0-100                    display only (single confidence field)
pitch_length_yards  22.0                     metres = yards * 0.9144
```

If `deliveries` is empty, no ball flight was detected in that video — show "no delivery
detected", don't attempt to render.

## 7. Known limitation (data accuracy, not rendering)

`length.label` can occasionally be misclassified — we have seen a yorker reported as
`good_length` — because it depends on where the bounce is detected. The trajectory will
still render cleanly and consistently with whatever label is reported; the label and the
rendered bounce always agree with each other. This is a detection-accuracy issue on the
engine side and is not something to compensate for in the frontend.
