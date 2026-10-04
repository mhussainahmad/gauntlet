# Gauntlet

[![CI](https://github.com/mhussainahmad/gauntlet/actions/workflows/ci.yml/badge.svg)](https://github.com/mhussainahmad/gauntlet/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gauntlet-robotics.svg)](https://pypi.org/project/gauntlet-robotics/)
[![Python](https://img.shields.io/pypi/pyversions/gauntlet-robotics.svg)](https://pypi.org/project/gauntlet-robotics/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**Regression testing and failure analysis for learned robot policies.**

Gauntlet answers one question for VLA, diffusion, and scripted policies:

> *How does this policy fail, and has the new checkpoint regressed against the last one?*

It wraps any policy behind a small adapter, runs it across a seeded grid
of simulator perturbations (lighting, camera pose, clutter, object pose,
actuation latency, sensor corruption, instruction paraphrase), and
writes a report that **breaks failures down by condition** instead of
averaging them away.

![Gauntlet report: failure clusters, per-axis sensitivity and success rates](https://raw.githubusercontent.com/mhussainahmad/gauntlet/main/docs/assets/report-field-conditions.png)

<sub>Report from the [field-conditions example](./docs/field-robustness.md):
the baseline is perfect up to 100 ms of control latency and falls to
47% at 200 ms. Every failure cluster is a latency cluster.</sub>

## Why

A success rate is a mean, and a mean hides the thing you care about.
Two checkpoints at 80% can fail in completely different places, and a
fine-tune that gains two points on the clean scene can lose its entire
safety margin on one axis. Gauntlet makes the per-condition picture the
default output:

- **Failure clusters first.** Axis combinations whose failure rate is
  well above baseline, with Wilson CIs, ranked by lift.
- **Paired comparisons.** `gauntlet compare` / `gauntlet diff` line two
  runs up cell by cell (common random numbers + McNemar when seeds
  match) and flag regressions, not just a headline delta.
- **Sensitivity indices.** Per-axis first- and total-order Sobol
  indices, corrected for rollout seed noise.
- **Reproducible.** Every episode is determined by
  `(suite, cell, seed)`; `gauntlet replay` re-simulates any one
  bit-for-bit, optionally with one axis nudged.

## See a real report

Every [GitHub Release](https://github.com/mhussainahmad/gauntlet/releases/latest)
ships `reference-benchmark.zip`: two policies (a closed-loop controller
and a degraded copy of it) on the smoke suite and the
[field-conditions suite](./docs/field-robustness.md), with the
`compare` / `diff` deltas. Unzip and open any `report.html`.

## Install

```bash
pip install gauntlet-robotics            # core: MuJoCo, torch-free
pip install 'gauntlet-robotics[hf]'      # + OpenVLA / HuggingFace adapter
pip install 'gauntlet-robotics[lerobot]' # + SmolVLA / pi0 / diffusion adapters
```

Other extras: `pybullet`, `genesis`, `isaac` (backends), `monitor`
(drift detection), `video`, `ros2`. Python ≥ 3.11.

## Quickstart

```bash
git clone https://github.com/mhussainahmad/gauntlet && cd gauntlet
uv sync

# 1. One policy, one suite: 24 rollouts, a few seconds.
uv run gauntlet run examples/suites/tabletop-smoke.yaml --policy random --out out/smoke
xdg-open out/smoke/report.html        # macOS: open

# 2. Baseline vs. regressed checkpoint on field-style conditions: 2 x 720 rollouts, ~15 s.
uv run python scripts/generate_reference_benchmark.py \
  --suite examples/suites/tabletop-field-conditions.yaml --out out/field
cat out/field/diff.txt
```

Each run writes `episodes.json` (one record per rollout), `report.json`
(the analysis) and a self-contained `report.html`.

### Bring your own policy

A policy is anything with `act(obs) -> action`:

```python
# my_policy.py
import numpy as np

class MyPolicy:
    def __init__(self) -> None:
        self.model = load_my_checkpoint()          # your code

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        # obs: ee_pos, cube_pos, target_pos, ... (+ "image" with render_in_obs=True)
        return self.model(obs["ee_pos"], obs["cube_pos"])  # 7-D EE twist + gripper

def make_policy() -> MyPolicy:
    return MyPolicy()
```

```bash
uv run gauntlet run examples/suites/tabletop-basic-v1.yaml \
  --policy my_policy:make_policy --out out/mine
uv run gauntlet compare out/last_week/episodes.json out/mine/episodes.json
```

Third-party packages can also register policies and envs through entry
points; see [`docs/plugin-development.md`](./docs/plugin-development.md).

## What's in the box

| | |
|---|---|
| **Backends** | MuJoCo (core), PyBullet, Genesis, Isaac Sim. Same action/observation spaces and axes; `compare` refuses cross-backend diffs unless asked. |
| **Perturbation axes** | Lighting, camera offset and full extrinsics, object texture / pose / class swap, distractors, OOD initial state, actuation latency, image corruption, colour shift, instruction paraphrase. |
| **Sampling** | Cartesian grids, Latin hypercube, Sobol, worst-case search; `gauntlet suite plan` sizes episodes-per-cell for a target effect. |
| **Policy adapters** | Random, scripted, OpenVLA (HF), SmolVLA / pi0 / diffusion (LeRobot), GR00T, RDT, Decision Transformer. |
| **Analysis** | Failure clusters, Wilson CIs, paired compare, per-cell diff, Sobol indices, behavioural metrics (time, path length, jerk), safety counters, `gauntlet bisect` across checkpoints. |
| **Operations** | Rollout caching, MP4 recording, runtime drift detection, ROS 2 publish/record, fleet aggregation, static dashboard, real-to-sim scene ingestion. |

Details for each are in the **[user guide](./docs/guide.md)**. Design
decisions are recorded as RFCs and design notes under [`docs/`](./docs/).

## Stability

`0.2.x` is on PyPI as `gauntlet-robotics`. The public API, on-disk
schemas and CLI flags follow [Semantic Versioning](https://semver.org/);
the contract is in [`docs/stability.md`](./docs/stability.md). Pin
`gauntlet-robotics>=0.2,<0.3`.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy                 # --strict
uv run pytest               # ~1,700 torch-free tests; extras run in their own CI jobs
```

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) and the
[property-test notes](./docs/guide.md#property-tests).

## Project layout

```
src/gauntlet/
  policy/      # Policy adapter protocol + reference wrappers (Random, Scripted, HF, LeRobot)
  env/         # Parameterized envs — MuJoCo (core) + PyBullet/Genesis/Isaac (extras)
  suite/       # YAML-defined perturbation grid suites (cartesian / LHS / Sobol)
  runner/      # Parallel rollout orchestration + seed management + cache
  report/      # Per-run failure analysis + HTML/JSON generation
  monitor/     # Runtime drift detection + action-entropy ([monitor] extra)
  replay/      # Single-episode replay with axis overrides
  ros2/        # ROS 2 publisher + recorder ([ros2] extra; rclpy via apt/Docker)
  diff/        # Structured per-axis report deltas powering `gauntlet diff`
  aggregate/   # Fleet-wide failure-mode clustering across many runs
  dashboard/   # Self-contained static SPA indexing every report.json
  realsim/     # Real-to-sim scene ingestion + renderers (nearest-frame, gaussian splat)
  plugins.py   # Entry-point discovery for third-party policies / envs
  cli.py       # gauntlet run / report / compare / diff / aggregate /
               # dashboard / realsim / monitor / replay / ros2
```

## License

MIT, see [LICENSE](./LICENSE).
