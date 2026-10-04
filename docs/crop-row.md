# Crop-row guidance under field conditions

A worked example that applies Gauntlet to **vision-based row guidance**:
a vehicle follows a crop row using only a forward camera, and two
perception models are compared under dust, glare, motion blur, weeds
and curved rows.

> **Synthetic scenes.** `env: crop-row` renders a procedurally generated
> field (rows of plants, weeds, soil, sky) with a pinhole camera in
> numpy. It is not field imagery and the numbers below say nothing about
> any real product. The point is the method: the same suite, report and
> comparison apply unchanged to a real perception model evaluated
> against logged or simulated field images.

![Conditions rendered by env: crop-row](https://raw.githubusercontent.com/mhussainahmad/gauntlet/main/docs/assets/crop-row-conditions.png)

## The task

| | |
|---|---|
| Vehicle | Constant 1.5 m/s, kinematic yaw-rate steering, 0.1 s control period. |
| Rows | 0.76 m (30-inch) spacing, plants every 0.15 m with gaps and size variation. |
| Camera | 1.6 m high, pitched 28° down, 75° horizontal field of view, 96 × 128 RGB. |
| Observation | The image only. Ground-truth lateral and heading error are in `info`. |
| Success | 100 steps (15 m) without leaving the ±0.25 m row corridor, ending within 0.08 m of the row centre. |

Both policies share the same steering law
(`gauntlet.policy.crop_row.row_steering`), so any difference between
them is perception.

## The two models

**Classical** (`--policy crop-row-classical`,
`src/gauntlet/policy/crop_row.py`). Excess-green index, Otsu threshold,
windowed row tracking from the bottom of the image, projection through
the calibrated camera and a least-squares line fit. Tuned on clean
scenes only, then frozen.

**CNN** (`examples/crop_row/`). Three conv layers and two dense layers
regressing lateral and heading error from the image. Trained by
`train_cnn.py` on 15,000 frames from **clean scenes only** (no dust,
glare, blur, weeds or curvature), reaching 2.7 mm / 0.0017 rad
validation error. The weights ship as `cnn_weights.npz` and inference is
plain numpy, so evaluating it needs no torch.

## The suite

`examples/suites/crop-row-field.yaml`: dust {0, 0.5, 1} × glare {0, 1}
× blur {0, 1} × weeds {0, 1} × curvature {0, 0.1 m⁻¹} = 48 cells,
10 seeded episodes each, 480 rollouts per model.

```bash
uv run gauntlet run examples/suites/crop-row-field.yaml \
  --policy crop-row-classical --out out/crop-classical --n-workers 4
PYTHONPATH=examples/crop_row uv run gauntlet run examples/suites/crop-row-field.yaml \
  --policy cnn_policy:make_policy --out out/crop-cnn --n-workers 4
uv run gauntlet compare out/crop-classical/episodes.json out/crop-cnn/episodes.json
uv run gauntlet diff out/crop-classical/report.json out/crop-cnn/report.json
```

## Results

Both models are perfect on the clean cell (10/10).

| | Classical | CNN |
|---|---|---|
| **Overall success** | **40.2%** | **43.1%** |
| Dust 0 / 0.5 / 1.0 | 68% / 50% / **3%** | 50% / 50% / 29% |
| Glare 0 / 1.0 | 47% / 33% | 50% / 36% |
| Motion blur 0 / 1.0 | 38% / 43% | 45% / 42% |
| Weeds 0 / 1.0 | 60% / **20%** | 45% / 42% |
| Curvature 0 / 0.1 m⁻¹ | 47% / 34% | 86% / **0%** |
| Largest first-order Sobol index | dust 0.31, weeds 0.16 | curvature 0.76 |

Per-axis rates are marginals over the other axes (each value covers
the other four conditions); 95% Wilson intervals are in the reports.

**A single number would say the CNN is a safe swap.** Overall success
moves by +2.9 points. The paired comparison (same seeds, McNemar)
tells a different story: **10 cells regress and 12 improve**.

- **The CNN never handles a curved row.** Every cell with
  `row_curvature = 0.1` drops to 0%, including the clean-image curved
  cell that the classical detector passes 10/10. The training set had
  only straight rows; nothing in an aggregate metric on the clean
  validation set reveals it.
- **The classical detector is blinded by dust and confused by weeds.**
  Dense dust collapses the excess-green contrast below its detection
  floor (3% success), and weeds between the rows pull the tracked
  centroid sideways. The CNN, which learned a whole-image regression,
  is far less sensitive to both.
- **Motion blur barely matters to either**, because the blur runs
  along the rows, the direction the row geometry does not depend on.
  An axis whose real-world importance you assumed can turn out flat;
  the report shows that too.

What to do next follows directly: add curved rows to the CNN's training
data (or gate it to straight-row operation), and give the classical
pipeline a dust fallback. Then rerun the suite and `gauntlet compare`
the new checkpoint against this one.

## Relating it to real systems

| In this example | In a field perception stack |
|---|---|
| Procedural frames | Logged camera frames replayed per condition, or a photoreal simulator |
| `dust_density`, `glare_intensity`, `motion_blur`, `weed_density` | The same, tagged on logged data or synthesised on clean frames |
| `row_curvature` | Headland turns, contour planting, pivot circles |
| `inference_delay_jitter` (works here too) | Compute latency on the embedded controller |
| `gauntlet compare` between two checkpoints | Release gate for a model update before it ships to machines |
