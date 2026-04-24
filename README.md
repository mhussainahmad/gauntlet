# Gauntlet

> An evaluation harness for learned robot policies.

Gauntlet answers a single question for VLA / diffusion / scripted policies:

> *"How does this policy fail, and has the latest checkpoint regressed against the last one?"*

It wraps any policy behind a uniform adapter, runs it across a parameterized
suite of MuJoCo perturbations (lighting, camera pose, textures, clutter,
initial conditions), and produces a structured report that **breaks failures
down by axis** instead of hiding them in an aggregate mean.

See [`GAUNTLET_SPEC.md`](./GAUNTLET_SPEC.md) for the full design.

---

## Status

Phase 1 MVP is feature-complete: tabletop env, perturbation axes, parallel
runner, breakdown-first HTML report, and the `gauntlet run / report /
compare` CLI. Phase 2 starts on real-policy adapters and runtime
observability. The public surface is still unstable.

## Backends

Gauntlet ships four simulator backends. The Suite YAML's `env:` key
is the dispatch: `tabletop` uses MuJoCo (default, ships in the core
install); `tabletop-pybullet`, `tabletop-genesis`, and
`tabletop-isaac` each live behind an optional extra.

| `env:` slug          | Simulator | Install                                  | Observations |
|----------------------|-----------|------------------------------------------|--------------|
| `tabletop`           | MuJoCo    | `uv sync` (core)                         | State + render-on-demand |
| `tabletop-pybullet`  | PyBullet  | `uv sync --extra pybullet`               | State + render-on-demand |
| `tabletop-genesis`   | Genesis   | `uv sync --extra genesis`                | State + render-on-demand |
| `tabletop-isaac`     | Isaac Sim | `uv sync --extra isaac` (GPU required)   | State-only (rendering follow-up) |

The four backends share action/observation spaces byte-for-byte and
the canonical 7 perturbation axes. They are **not** numerically
identical: same policy + same seed on `tabletop` vs `tabletop-pybullet`
vs `tabletop-genesis` vs `tabletop-isaac` produces semantically similar
but numerically different trajectories. Running `gauntlet compare`
across backends measures simulator drift, not policy regression; the
CLI requires `--allow-cross-backend` to proceed.

The `tabletop-isaac` backend wraps NVIDIA Omniverse Kit and **requires
a CUDA-capable RTX-class GPU at runtime**. The `[isaac]` extra resolves
on CPU-only machines but the Kit bootstrap inside `IsaacSimTabletopEnv.__init__`
fails without a GPU. CI tests use a `sys.modules`-injected fake
`isaacsim` namespace and do NOT install this extra; live execution
needs a developer GPU workstation. The state-only first cut declares
the four cosmetic axes (`lighting_intensity`, `camera_offset_x`,
`camera_offset_y`, `object_texture`) `VISUAL_ONLY_AXES` so cosmetic-only
sweeps are rejected at suite-load time on this backend until the
rendering follow-up RFC lands.

Image observations are available on all three backends via
`render_in_obs=True` / `render_size=(H, W)` on the env constructor
(`TabletopEnv`, `PyBulletTabletopEnv`, or `GenesisTabletopEnv`).
PyBullet uses a headless, deterministic TINY rasteriser; Genesis uses
its default CPU Rasterizer (pyrender-backed). The emitted `obs["image"]`
Box has shape / dtype / bounds byte-identical across backends, so VLA
adapters (OpenVLA, SmolVLA) work on any of them by swapping only the
env factory. Pixel values explicitly differ (different rasterisers —
semantic parity only). All seven perturbation axes produce observable
deltas on the rendered image on every backend; `VISUAL_ONLY_AXES` is
empty everywhere.

For multi-view policies (SmolVLA, ACT, Diffusion Policy — anything
that consumes paired wrist + side + overhead frames), pass
`cameras=[CameraSpec(...), ...]` to `TabletopEnv` or
`PyBulletTabletopEnv` instead. Each spec lands in
`obs["images"][name]`; `obs["image"]` stays populated as an alias to
the first camera so single-view consumers (the runner's video
recorder, OpenVLA-style adapters) keep working unchanged. The
single-camera default (`cameras=None`) is byte-identical to the
phase-1 contract — see
[`docs/polish-exploration-multi-camera.md`](./docs/polish-exploration-multi-camera.md)
for the full design and
[`examples/evaluate_multi_camera.py`](./examples/evaluate_multi_camera.py)
for a worked example.

See [`docs/phase2-rfc-005-pybullet-adapter.md`](./docs/phase2-rfc-005-pybullet-adapter.md)
for the full PyBullet backend design,
[`docs/phase2-rfc-006-pybullet-rendering.md`](./docs/phase2-rfc-006-pybullet-rendering.md)
for PyBullet's image-observation follow-up,
[`docs/phase2-rfc-007-genesis-adapter.md`](./docs/phase2-rfc-007-genesis-adapter.md)
for the Genesis backend design,
[`docs/phase2-rfc-008-genesis-rendering.md`](./docs/phase2-rfc-008-genesis-rendering.md)
for the Genesis image-observation follow-up, and
[`docs/phase2-rfc-009-isaac-sim-adapter.md`](./docs/phase2-rfc-009-isaac-sim-adapter.md)
for the Isaac Sim backend design.

## Quickstart

Three commands reproduce the end-to-end example against the bundled
smoke suite (3 lighting intensities x 2 cube textures x 4 episodes = 24
rollouts; finishes in seconds on a laptop):

```bash
uv sync
uv run gauntlet run examples/suites/tabletop-smoke.yaml --policy random --out out/
open out/report.html  # macOS: open ; Linux: xdg-open ; Windows: start
```

Artefacts land in `out/`: `episodes.json` (one record per rollout),
`report.json` (analysed breakdowns), and `report.html` — a self-contained
report leading with the failure-clusters table, then per-axis bar charts,
then 2D heatmaps of axis combinations. The smoke suite is intentionally
tiny; for the canonical 4-axis x 144-cell x 1440-rollout shape, swap the
YAML path for `examples/suites/tabletop-basic-v1.yaml`. See
[`GAUNTLET_SPEC.md`](./GAUNTLET_SPEC.md) for the full design and
[`examples/evaluate_random_policy.py`](./examples/evaluate_random_policy.py)
for the equivalent invocation via the public Python API. For the
Genesis backend, `uv sync --extra genesis` then
[`examples/evaluate_random_policy_genesis.py`](./examples/evaluate_random_policy_genesis.py)
drives the same smoke suite against `tabletop-genesis`.

The Suite YAML's `sampling:` key picks the perturbation grid strategy:
the default `cartesian` enumerates the full Cartesian product of the
declared axes (the historical behaviour, byte-identical to every
existing suite), while `latin_hypercube` draws `n_samples` points using
McKay 1979 Latin Hypercube Sampling — every axis covers `n_samples`
distinct strata regardless of dimensionality. For five axes at five
steps, cartesian = 3,125 cells; LHS at `n_samples: 32` covers the same
hypercube with strictly higher marginal coverage at ~98x fewer
rollouts. See
[`examples/suites/tabletop-lhs-smoke.yaml`](./examples/suites/tabletop-lhs-smoke.yaml)
and [`examples/evaluate_random_policy_lhs.py`](./examples/evaluate_random_policy_lhs.py)
for an end-to-end demo. Sobol (`sampling: sobol`) is reserved in the
schema for a follow-up; LHS is the supported quasi-random alternative
today.

Once you have multiple runs (different seeds, policy revisions, or
backends), `gauntlet aggregate <runs-dir> --out fleet/` rolls every
`report.json` recursively under `<runs-dir>` into a single fleet
meta-report — `fleet/fleet_report.json` plus a self-contained
`fleet/fleet_report.html` leading with the persistent failure
clusters that survive across runs (clusters appearing in at least
`--persistence-threshold` of the runs, default `0.5`). See
[`examples/aggregate_runs.py`](./examples/aggregate_runs.py) for the
equivalent invocation via the Python API and
[`docs/phase3-rfc-019-fleet-aggregate.md`](./docs/phase3-rfc-019-fleet-aggregate.md)
for the algorithm.

### Using a real VLA

- Install the HF extras: `uv sync --extra hf` (pulls torch / transformers / pillow; core installs stay torch-free).
- See [`examples/evaluate_openvla.py`](./examples/evaluate_openvla.py) for the ≤20-line OpenVLA-7B factory.
- Image-conditioned policies need a rendered frame — construct `TabletopEnv(render_in_obs=True)` so `obs["image"]` is emitted.
- SmolVLA: `uv sync --extra lerobot`; see [`examples/evaluate_smolvla.py`](./examples/evaluate_smolvla.py). For the PyBullet-backed equivalent, `uv sync --extra lerobot --extra pybullet` and run [`examples/evaluate_smolvla_pybullet.py`](./examples/evaluate_smolvla_pybullet.py).
- SmolVLA-base is pretrained on SO-100 (6-D joint) whereas TabletopEnv is 7-D EE-twist+gripper — zero-shot success is ~0% by embodiment mismatch; fine-tune on TabletopEnv-compatible data for meaningful evaluation.

### Runtime drift detection

Optional Phase 2 add-on (`[monitor]` extra). Given a reference sweep of a
known-good policy, fit a small observation autoencoder and score a
candidate sweep's trajectories against it — per-episode reconstruction
error + per-dim action-std surface OOD rollouts. `gauntlet run
--record-trajectories <dir>` dumps per-episode NPZ sidecars;
`gauntlet monitor train <dir> --out <ae_dir>` fits the AE; `gauntlet
monitor score <episodes.json> <dir> --ae <ae_dir> --out drift.json`
writes the sidecar. The three-step workflow is scripted end-to-end in
[`examples/evaluate_with_drift.py`](./examples/evaluate_with_drift.py);
`drift.json` is optional and orthogonal to `report.json`.

### ROS 2 integration

Optional Phase 2 add-on (`[ros2]` extra). Two halves wire gauntlet into a
ROS 2 graph:

- `gauntlet ros2 publish episodes.json --topic /gauntlet/episodes`
  serialises each Episode as JSON inside `std_msgs/msg/String` and
  publishes one message per Episode. Useful for fleet-wide failure-mode
  aggregation across many real robots running gauntlet evaluations.
- `gauntlet ros2 record --topic /robot/joint_states --out trajectory.jsonl
  --duration 30` subscribes to a real robot's topic and dumps each
  received message to a JSONL file on disk. Useful for the "real robots
  with logging" half of `GAUNTLET_SPEC.md` §7.

Because `rclpy` is **not** distributed via PyPI in its official form, the
`[ros2]` extra is empty — `uv sync --extra ros2` is a no-op beyond the
dev tooling. Install ROS 2 (Humble or Jazzy) via your system package
manager, e.g. `sudo apt install ros-humble-rclpy`, or run inside the
official Docker image (`docker run -it osrf/ros:humble-desktop`), then
source the relevant `setup.bash` before invoking `gauntlet ros2`. The
`--dry-run` flag on `gauntlet ros2 publish` short-circuits the rclpy
import so you can preview the JSON payloads without installing ROS 2.

The publisher / recorder API is documented in
[`docs/phase2-rfc-010-ros2-integration.md`](./docs/phase2-rfc-010-ros2-integration.md);
see [`examples/publish_episodes_to_ros2.py`](./examples/publish_episodes_to_ros2.py)
for the equivalent invocation via the public Python API.

### Debugging failures with replay

Once a run has flagged an episode as failing, `gauntlet replay` re-
simulates exactly that rollout with the same seed, optionally nudging
one axis off the original grid:

```bash
uv run gauntlet replay out/episodes.json \
  --suite examples/suites/tabletop-smoke.yaml \
  --policy scripted \
  --episode-id 3:1 \
  --override lighting_intensity=1.2 \
  --out out/replay.json
```

Zero-override replay is bit-identical to the original episode; any
deviation points at a real reproducibility bug. See
[`examples/replay_failure.py`](./examples/replay_failure.py) for the
equivalent library call.

### Recording rollout videos

Failure analytics are far more actionable when a human can *watch*
the broken rollout. Opt in to the `[video]` extra to dump one MP4 per
episode and surface inline `<video>` thumbnails in the failure-
clusters table of the HTML report:

```bash
uv sync --extra video
uv run python examples/evaluate_random_policy_with_video.py --out out
# Open out/report.html — the failure-clusters table now embeds
# clickable thumbnails of every failed rollout.
```

The `[video]` extra pulls `imageio[ffmpeg]`, which bundles a static
ffmpeg binary — no system ffmpeg install required. Pass
`--only-failures` to suppress MP4 writes for successful episodes
(saves disk on long sweeps). The Runner asserts the env was
constructed with `render_in_obs=True` when `record_video=True`; the
example wires that automatically.

## Development

```bash
# Sync deps (creates .venv, installs everything in pyproject + dev group).
uv sync

# Lint, type-check, test.
uv run ruff check .
uv run mypy
uv run pytest
```

## Project layout

```
src/gauntlet/
  policy/    # Policy adapter protocol + reference wrappers (Random, Scripted, HF, LeRobot)
  env/       # Parameterized envs — MuJoCo (core) + PyBullet ([pybullet] extra) + Genesis ([genesis] extra)
  suite/     # YAML-defined perturbation grid suites
  runner/    # Parallel rollout orchestration + seed management
  report/    # Failure analysis + HTML/JSON generation
  monitor/   # Runtime drift detection + action-entropy ([monitor] extra)
  replay/    # Single-episode replay with axis overrides
  ros2/      # ROS 2 publisher + recorder ([ros2] extra; rclpy via apt/Docker)
  cli.py     # gauntlet run / report / compare / monitor / replay / ros2
```

## License

MIT.
