# API reference

Authoritative reference for the public Python API exposed by the
`gauntlet` package. The CLI (`gauntlet --help`) wraps everything below;
this page is for callers driving the harness from a notebook, a custom
script, or a third-party orchestrator.

Scope: every symbol in `gauntlet`'s `__all__` that a downstream
consumer is expected to touch. Internals (anything `_`-prefixed) are
excluded — they may change between releases without notice.

For the conceptual model — perturbation axes, suite YAML grammar,
report schema, the seven-axis surface — see `GAUNTLET_SPEC.md` and the
RFCs cross-linked from each section below.

## Contents

- [Environment Protocol and backends](#environment-protocol-and-backends)
- [Suite (perturbation grid)](#suite-perturbation-grid)
- [Policy adapters](#policy-adapters)
- [Runner (parallel rollout)](#runner-parallel-rollout)
- [Episode (rollout result)](#episode-rollout-result)
- [Report (failure analysis)](#report-failure-analysis)
- [Diff (per-axis structured delta)](#diff-per-axis-structured-delta)
- [Compare (regression verdict + cross-backend drift)](#compare-regression-verdict--cross-backend-drift)
- [Bisect (cross-checkpoint regression search)](#bisect-cross-checkpoint-regression-search)
- [Aggregate (fleet meta-report)](#aggregate-fleet-meta-report)
- [Dashboard (static SPA)](#dashboard-static-spa)
- [Replay (single-episode re-simulation)](#replay-single-episode-re-simulation)
- [Monitor (drift detector)](#monitor-drift-detector)
- [ROS 2 bridge](#ros-2-bridge)
- [Real-to-sim (scene reconstruction)](#real-to-sim-scene-reconstruction)
- [Security (path / YAML guards)](#security-path--yaml-guards)
- [Plugins (third-party env / policy)](#plugins-third-party-env--policy)

---

## Environment Protocol and backends

```python
from gauntlet.env import (
    GauntletEnv, CameraSpec,
    AXIS_NAMES, PerturbationAxis, axis_for,
    N_DISTRACTOR_SLOTS,
)
```

`GauntletEnv` is a structural `typing.Protocol` (RFC-005 §3) — any
object exposing `reset`, `step`, `set_perturbation`, `restore_baseline`,
`close`, `observation_space`, `action_space`, and a class-level
`AXIS_NAMES: frozenset[str]` satisfies it. The Runner duck-types
against the Protocol; backends do not subclass anything. Per the
contract, `reset()` MUST: (1) call `restore_baseline()`; (2) apply
seed-driven randomisation; (3) apply queued perturbations; (4) clear
the queue.

`CameraSpec(name, pose, size)` is a `NamedTuple` describing one render
camera. `pose = (x, y, z, rx, ry, rz)` is metres + MuJoCo-XYZ Euler
radians. `size = (H, W)`. See
`docs/polish-exploration-multi-camera.md` for the full contract.

`AXIS_NAMES` is the canonical ordered tuple of every scalar
perturbation axis name the harness understands (the original seven
plus B-32 / B-31 / B-05 / B-06 / B-42 / B-43 / B-38 additions). Stable
contract — Suite YAML keys, Runner cell ids, Report axis labels all
key off it. `PerturbationAxis(name, kind, sampler, bounds)` is the
frozen record one axis is described by; `axis_for(name)` returns the
default-configured `PerturbationAxis` registered under `name`. See
`gauntlet.env.perturbation` for the per-axis constructor surface
(`lighting_intensity`, `camera_offset_x`, …, `inference_delay_jitter`).
`N_DISTRACTOR_SLOTS` is the integer cap on the `distractor_count`
axis; downstream code reads it instead of hard-coding the bound.

### `TabletopEnv`

```python
from gauntlet.env import TabletopEnv

env = TabletopEnv(max_steps=200, n_substeps=5,
                  render_in_obs=False, render_size=(224, 224),
                  cameras=None)
```

MuJoCo pick-and-place reference backend. Mocap-driven floating
end-effector, snap-grasp model, success when cube XY lies within
`TARGET_RADIUS` of the target site. State-only by default;
`render_in_obs=True` adds `obs["image"]` from the `main` camera, and a
non-empty `cameras` list injects extra `<camera>` elements and emits
`obs["images"][name]` per spec (multi-camera; first spec aliased to
`obs["image"]`). Declares all seven canonical perturbation axes.

### `PyBulletTabletopEnv`

```python
from gauntlet.env.pybullet import PyBulletTabletopEnv

env = PyBulletTabletopEnv(max_steps=200, render_in_obs=False,
                          render_size=(224, 224), cameras=None)
```

PyBullet reimplementation of `TabletopEnv` (RFC-005 / RFC-006). Same
observation surface and axis names; uses the headless TINY renderer for
images. Requires the `[pybullet]` extra.

### `GenesisTabletopEnv`

```python
from gauntlet.env.genesis import GenesisTabletopEnv

env = GenesisTabletopEnv(max_steps=200, n_substeps=5,
                         render_in_obs=False, render_size=(224, 224))
```

Genesis backend (RFC-007). State-only first cut; `render_in_obs` is
accepted for API parity — image rendering lands in the follow-up RFC.
Boots `gs.init` once per worker process. Requires the `[genesis]`
extra.

### `IsaacSimTabletopEnv`

```python
from gauntlet.env.isaac import IsaacSimTabletopEnv

env = IsaacSimTabletopEnv(max_steps=200)
```

Isaac Sim backend (RFC-009). State-only — Omniverse Kit camera wiring
deferred. Construction boots `SimulationApp({"headless": True})`
(~5-s amortised cost per worker). Declares the 5 state-only axes
(camera and texture axes are visual-only and excluded). Requires the
`[isaac]` extra and a CUDA-capable RTX-class GPU.

### `MobileTabletopEnv` (B-13)

```python
from gauntlet.env import MobileTabletopEnv

env = MobileTabletopEnv()
# Action: [dx, dy, dz, drx, dry, drz, gripper, base_vx, base_vy, base_omega_z]
```

Composition wrapper around `TabletopEnv` adding a kinematic SE(2) base.
The first 7 action slots are the inner pick twist; the trailing 3 slots
are integrated kinematically (`base_pose <- base_pose + (vx, vy,
omega) * BASE_DT`). Observation is every key the inner env publishes
plus `pose: Box(shape=(3,))`, the integrated `(x, y, theta)`. Success
is `nav_done AND picked` — base XY within `NAV_RADIUS` of a fixed
base-frame target table AND the inner env reporting a successful
grasp. Phase-1 scope: the wrapper does NOT forward perturbations to
the inner env (`AXIS_NAMES` is empty); the freejoint-mounted-arm and
collision rebake are deferred. Registered as `tabletop-mobile`.

### Env registry

```python
from gauntlet.env.registry import register_env, get_env_factory, registered_envs
```

The Runner calls `get_env_factory(suite.env)` when no explicit
`env_factory` is supplied. Third-party packages register their backends
by importing `register_env(name, factory)` at package init or via the
`gauntlet.envs` entry-point group (see [Plugins](#plugins-third-party-env--policy)).

### Gymnasium registration

```python
import gauntlet  # triggers register_envs()
import gymnasium as gym

env = gym.make("gauntlet/Tabletop-v0")
```

Importing `gauntlet` registers every backend with `gymnasium`'s global
registry. Heavy backends use string `entry_point`s so this does not
import torch / pybullet / genesis / isaacsim. Idempotent.

---

## Suite (perturbation grid)

```python
from gauntlet.suite import (
    Suite, SuiteCell, AxisSpec, SamplingMode, SAMPLING_MODES,
    load_suite, load_suite_from_string,
)
```

`Suite` is a Pydantic model: `name`, `env`, `episodes_per_cell`,
optional `seed`, optional `n_samples`, `sampling`
(`"cartesian" | "latin_hypercube" | "sobol"`), and `axes:
dict[str, AxisSpec]`. `extra="forbid"`.

`AxisSpec` is one of two shapes (never both):

- continuous: `{low, high, steps}` — `steps` evenly spaced inclusive
  points (midpoint when `steps == 1`);
- categorical: `{values: [...]}` — explicit enumeration.

`SuiteCell(index, values)` is one point in the grid; `Suite.cells()`
returns the full enumeration in stable, deterministic order.

### `load_suite(path)` / `load_suite_from_string(text)`

```python
from gauntlet.suite import load_suite

suite = load_suite("examples/suites/lighting_sweep.yaml")
```

YAML loader with full Pydantic validation. `BUILTIN_BACKEND_IMPORTS`
maps env slugs (`tabletop-pybullet`, `tabletop-genesis`,
`tabletop-isaac`) to the package the loader lazy-imports on demand.

### Samplers

```python
from gauntlet.suite.sampling import build_sampler, CartesianSampler, Sampler
from gauntlet.suite.lhs import LatinHypercubeSampler, lhs_unit_cube
from gauntlet.suite.sobol import SobolSampler, sobol_unit_cube, MAX_DIMS
```

`build_sampler(mode)` returns the strategy keyed by `Suite.sampling`.
LHS uses McKay 1979 stratification; Sobol uses Joe-Kuo 6.21201
direction numbers (21-dimension cap). Both are deterministic given the
RNG `Suite.cells()` seeds from `suite.seed`. See
`docs/polish-exploration-lhs-sampling.md` and
`docs/polish-exploration-sobol-sampler.md`.

---

## Policy adapters

```python
from gauntlet.policy import (
    Policy, ResettablePolicy, SamplablePolicy, Action, Observation,
    RandomPolicy, ScriptedPolicy, DEFAULT_PICK_AND_PLACE_TRAJECTORY,
    HuggingFacePolicy, LeRobotPolicy,
    PolicySpecError, resolve_policy_factory,
)
```

`Policy` is the minimal `runtime_checkable` Protocol: `act(obs) ->
action`. `ResettablePolicy` adds `reset(rng)` — the Runner calls it at
the start of every episode when present so stochastic policies re-seed
deterministically. `SamplablePolicy` (B-18) adds
`act_n(obs, n=8) -> Sequence[Action]`: the runner draws `n`
independent actions at a handful of trajectory steps to compute the
mode-collapse metric stored on `Episode.action_variance`.
Implementations MUST keep `act_n` side-effect free on per-episode
state (snapshot + restore the RNG / chunk-queue cursor) so the
measured rollout's actions remain identical to an unmeasured rollout
with the same `Episode.seed` — preserving the determinism that B-08
CRN paired-compare and B-39 bisect rely on. Greedy / open-loop
policies (Scripted, OpenVLA-greedy) deliberately do not implement
this Protocol; the runner keys off `isinstance` and writes
`Episode.action_variance=None`.

### `RandomPolicy`

```python
RandomPolicy(action_dim=7, action_low=-1.0, action_high=1.0, seed=None)
```

Uniform-random baseline. Determinism anchored on `seed`; the Runner's
per-episode RNG replaces it via `reset()`. Use as the zero-information
reference point against which a learned policy's success rate is
judged.

### `ScriptedPolicy`

```python
ScriptedPolicy(trajectory=None, loop=False)
```

Open-loop playback of a `(steps, action_dim)` array.
`trajectory=None` selects `DEFAULT_PICK_AND_PLACE_TRAJECTORY` (the
canonical 7-DoF pick-and-place stub). Last-frame hold by default;
`loop=True` wraps.

### `HuggingFacePolicy`

```python
HuggingFacePolicy(repo_id, instruction, *,
                  unnorm_key=None, device=None, dtype="bfloat16",
                  image_obs_key="image", processor_kwargs=None,
                  model_kwargs=None)
```

Adapter for OpenVLA-style Vision2Seq HF models (RFC-001). `act(obs)`
extracts `obs[image_obs_key]` as `uint8 (H, W, 3)`, builds a PIL
image, runs the processor with the cached prompt, and calls
`predict_action`. Flips the gripper convention (OpenVLA `[0, 1]` →
TabletopEnv `[+1 open, -1 close]`); emits a `RuntimeWarning` if any
twist coord exceeds `[-1, 1]` so an unnorm-key mismatch surfaces
loudly. Requires the `[hf]` extra.

### `LeRobotPolicy`

```python
LeRobotPolicy(repo_id, instruction, *,
              device=None, dtype="bfloat16", image_obs_key="image",
              camera_keys=("observation.images.camera1", ...),
              state_obs_keys=("ee_pos", "gripper"),
              action_remap=None,
              preprocessor_overrides=None,
              postprocessor_overrides=None)
```

Adapter for lerobot SmolVLA (RFC-002). Default `action_remap` pads
SO-100 joints into TabletopEnv twist (warns once). `reset()` flushes
the SmolVLA action-chunk queue — critical for correctness; without it
episode N starts on the tail of episode N-1's cached chunk. Requires
the `[lerobot]` extra.

### `resolve_policy_factory(spec)`

```python
factory = resolve_policy_factory("random")             # built-in
factory = resolve_policy_factory("scripted")
factory = resolve_policy_factory("my_pkg.policies:make_policy")
```

Resolves a policy spec string to a zero-arg factory. Raises
`PolicySpecError` (a `ValueError` subclass) on bad specs.

---

## Runner (parallel rollout)

```python
from gauntlet.runner import Runner, Episode, WorkItem, execute_one
```

### `Runner`

```python
Runner(*, n_workers=1, env_factory=None, start_method="spawn",
       trajectory_dir=None, record_video=False, video_dir=None,
       video_fps=30, record_only_failures=False,
       cache_dir=None, policy_id=None, max_steps=None)

episodes = runner.run(policy_factory=..., suite=...)
```

Orchestrates `(cell × episode)` rollouts. `n_workers=1` runs in-process
(skips spawn overhead, readable tracebacks); `n_workers >= 2` uses a
`spawn`-context `multiprocessing.pool.Pool` and the per-item function
is the same in both paths so output is bit-identical for the same
inputs. `env_factory=None` dispatches via `gauntlet.env.registry` on
`suite.env`; an explicit factory wins. `trajectory_dir` opts into NPZ
trajectory dumps; `record_video` opts into MP4 dumps (requires the
`[video]` extra); `cache_dir` opts into the per-Episode rollout cache
keyed on `(suite, axis_config, env_seed, policy_id, max_steps)` —
required `max_steps` because the cache key needs it and `GauntletEnv`
exposes no public getter. `cache_stats()` returns
`{"hits", "misses", "puts"}`.

Seed derivation: `master = SeedSequence(suite.seed); cell_seqs =
master.spawn(n_cells); episode_seq = cell_seqs[i].spawn(eps_per_cell)[j]`.
Each episode gets a uint32 env seed (stored on `Episode.seed`) and an
independent `np.random.default_rng(episode_seq)` for the policy
stream.

`run()` returns Episodes sorted by `(cell_index, episode_index)`,
regardless of worker completion order.

### `execute_one(env, policy_factory, work_item, *, trajectory_dir, video_config)`

The per-episode primitive shared by both execution paths. Public so
`gauntlet.replay.replay_one` can reuse it; ordinary callers go through
`Runner.run`.

---

## Episode (rollout result)

```python
from gauntlet.runner import Episode
```

Pydantic model; `extra="forbid"`. Fields:

- identity: `suite_name`, `cell_index`, `episode_index`, `seed`,
  `perturbation_config: dict[str, float]`;
- outcome: `success`, `terminated`, `truncated`, `step_count`,
  `total_reward`;
- `metadata: dict[str, float | int | str | bool]` — free-form bag;
  the Runner uses it to echo `master_seed` for None-seed runs.
- `video_path: str | None` — relative MP4 path when
  `Runner(record_video=True)`; `None` otherwise. Always relative so
  the HTML report can embed via `<video src="...">` without a server.

`ser_json_inf_nan="strings"` so NaN/Inf reward round-trips through
JSON.

---

## Report (failure analysis)

```python
from gauntlet.report import (
    Report, AxisBreakdown, CellBreakdown, FailureCluster, Heatmap2D,
    build_report, render_html, write_html,
)
```

### `build_report(episodes, *, suite_env=None) -> Report`

```python
from gauntlet.report import build_report, write_html

report = build_report(episodes, suite_env=suite.env)
write_html(report, "out/report.html")
```

Pure function — no I/O. `Report` field order is intentional:
breakdowns first (`per_axis`, `per_cell`, `failure_clusters`,
`heatmap_2d`), scalar means last (`overall_success_rate`,
`overall_failure_rate`). Per spec §6: never aggregate away failures.
All float axis values appearing as dict keys are float-normalized to 9
dp at construction time so floating-point drift never splits a grid
value across two buckets.

`FailureCluster` flags axis-pair value combinations whose failure rate
is at least `cluster_multiple` times the baseline. `Heatmap2D` carries
the success-rate matrix for a pair of axes; keys in
`Report.heatmap_2d` follow `f"{axis_x}__{axis_y}"`.

### `write_html(report, path)` / `render_html(report) -> str`

Renders the per-run HTML artifact. `write_html` is the file-writing
wrapper; `render_html` returns the HTML string for in-memory use.

---

## Diff (per-axis structured delta)

```python
from gauntlet.diff import (
    ReportDiff, AxisDelta, CellFlip, ClusterDelta,
    Verdict, VERDICT_REGRESSED, VERDICT_IMPROVED, VERDICT_WITHIN_NOISE,
    diff_reports, render_text,
    # Paired-CRN (B-08) — opt-in, requires shared master_seed on both runs.
    PairedComparison, PairedCellDelta, McNemarResult, PairingError,
    pair_episodes, mcnemar_test, paired_delta_ci, compute_paired_cells,
)
```

### `diff_reports(report_a, report_b, *, a_label, b_label, cell_flip_threshold, cluster_intensify_threshold) -> ReportDiff`

```python
from gauntlet.diff import diff_reports, render_text

result = diff_reports(report_a, report_b, a_label="run_a", b_label="run_b",
                      cell_flip_threshold=0.1, cluster_intensify_threshold=0.5)
print(render_text(result))
```

`git diff`-style structured delta between two `Report`s. Surfaces
per-axis-value rate deltas (`AxisDelta`), per-cell success-rate flips
(`CellFlip`), and the failure-cluster set difference (added /
removed / `ClusterDelta` for shared clusters whose lift rose). Use
when `gauntlet compare`'s yes/no verdict is too coarse and you need
"what moved?". `render_text` produces a plain-text rendering; the CLI
wraps it with rich color.

### Regression-vs-noise verdict (B-20)

`Verdict` is `Literal["regressed", "improved", "within_noise"]` plus
the three constants `VERDICT_REGRESSED`, `VERDICT_IMPROVED`,
`VERDICT_WITHIN_NOISE`. Each `CellFlip` carries a `verdict` field set
by `diff_reports`: a flip whose magnitude exceeds
`cell_flip_threshold` *and* whose Wilson / paired-CRN bracket
excludes zero is tagged `regressed` or `improved`; otherwise it is
tagged `within_noise`. CI gates should key off `regressed`, not the
raw delta sign.

### Paired-CRN comparison (B-08, B-39)

```python
from gauntlet.diff import compute_paired_cells, mcnemar_test, PairingError
from gauntlet.runner import Episode  # episode lists from two paired runs

paired = compute_paired_cells(episodes_a, episodes_b)  # list[PairedCellDelta]
worst = sorted(paired, key=lambda p: p.delta_ci_high)[:5]
```

When two Runs share a `master_seed`, the Runner derives identical
per-episode env seeds for every paired `(cell_index, episode_index)`
key. `compute_paired_cells` pairs the resulting episodes, runs
`mcnemar_test(b, c)` over the discordant counts, and returns one
`PairedCellDelta` per cell with the Newcombe-Tango paired CI on
`b_success_rate - a_success_rate`. The CI bracket shrinks roughly
2-4x relative to two independent Wilson intervals — the variance
reduction that `gauntlet bisect` (B-39) consumes. `pair_episodes` is
the lower-level helper exposed for callers that want the discordant
pairs directly. Raises `PairingError` (a `ValueError` subclass) when
the two episode lists do not share a master seed or do not align
key-for-key.

---

## Compare (regression verdict + cross-backend drift)

```python
from gauntlet.compare import (
    to_github_summary,
    AxisDrift, DriftMap, DriftMapError,
    compute_drift_map, render_drift_map_table, top_axis_drifts,
)
```

`gauntlet compare` is the binary regression-gate CLI; this module
surfaces two structured outputs that ride on top of its
`compare.json` payload.

### `to_github_summary(compare_result) -> str` (B-24)

```python
import json
from gauntlet.compare import to_github_summary

with open("compare.json") as fh:
    payload = json.load(fh)
summary_md = to_github_summary(payload)
# Append to GITHUB_STEP_SUMMARY in CI:
# echo "$summary_md" >> "$GITHUB_STEP_SUMMARY"
```

Renders a `compare.json` payload as a markdown blob suitable for
`$GITHUB_STEP_SUMMARY` — the GitHub Actions surface that lets a CI
job emit a rich rendered report alongside the standard logs. Tables
of regressed cells and intensified failure clusters land directly in
the run summary tab without an extra artefact upload.

### `compute_drift_map(report_a, report_b, ...) -> DriftMap` (B-29)

```python
from gauntlet.compare import compute_drift_map, render_drift_map_table

drift = compute_drift_map(sim_report, real_report,
                          a_label="sim", b_label="real")
print(render_drift_map_table(drift))
top = top_axis_drifts(drift, limit=5)  # list[AxisDrift]
```

Cross-backend embodiment-transfer scoring inspired by SIMPLER. For
every shared `(axis, axis_value)` cell across two `Report`s, emits an
`AxisDrift` (`a_rate`, `b_rate`, `delta`, `abs_delta`) so a fleet
operator can answer "where does the sim/real gap concentrate?".
`top_axis_drifts(drift, limit=N)` returns the top-N drifts ranked by
`abs_delta`. `render_drift_map_table` renders a sorted markdown table
suitable for stdout or a PR comment. `DriftMapError` (a `ValueError`
subclass) signals incompatible report shapes (different axis schemas,
different env families). See `docs/phase3-rfc-019-fleet-aggregate.md`
for the framing.

### Sim-vs-real correlation (B-28)

```python
from gauntlet.aggregate import (
    AxisTransfer, SimRealReport, compute_sim_real_correlation,
)

corr = compute_sim_real_correlation(sim_episodes, real_episodes,
                                    pair_key="instance_id")
```

`compute_sim_real_correlation` ranks each axis by how well its sim
success rate predicts its real-rollout outcome (Spearman + per-axis
Pearson). `SimRealReport` is the top-level pydantic model;
`AxisTransfer` is one row per axis. The default pairing key is
`instance_id` from `Episode.metadata`; pass `pair_key=` to use a
different metadata field. SureSim-style framing — see the B-28
backlog entry.

---

## Bisect (cross-checkpoint regression search)

```python
from gauntlet.bisect import (
    BisectError, BisectResult, BisectStep, RunnerFactory, bisect,
)
```

### `bisect(checkpoints, resolver, *, suite, target_cell_id, runner_factory=Runner, episodes_per_step=None) -> BisectResult` (B-39)

```python
from gauntlet.bisect import bisect
from gauntlet.suite import load_suite

suite = load_suite("examples/suites/cube_stack.yaml")
result = bisect(
    checkpoints=["v1-good", "v2", "v3", "v4-bad"],
    resolver=lambda ckpt: lambda: load_my_policy(ckpt),
    suite=suite,
    target_cell_id=12,
    episodes_per_step=64,
)
print(result.first_bad)         # e.g. "v3"
for step in result.steps:
    print(step.checkpoint, step.delta.delta_ci_high)
```

Binary-searches an ordered checkpoint list `[good, *intermediates,
bad]` for the first checkpoint at which the named target cell's
success rate dropped significantly below the known-good baseline. At
each midpoint the engine runs the resolved policy on `suite`,
filters episodes to `target_cell_id`, and consults
`gauntlet.diff.paired.compute_paired_cells` for the paired CI. The
collapse rule is one-sided: `delta_ci_high < 0` ⟹ midpoint
significantly worse, move `hi = mid`. Episodes-per-cell can be
overridden per step via `episodes_per_step` (defaults to
`suite.episodes_per_cell`).

`RunnerFactory` is the `Protocol` for the Runner constructor —
defaults to the built-in `gauntlet.runner.Runner`, but downstream
callers can wire a custom executor (subprocess pool, remote worker
shim) by satisfying its single-method shape. `BisectError` (a
`ValueError` subclass) signals bad inputs (target cell missing from
the suite, fewer than 2 checkpoints) or the degenerate
all-discordant-zero case. Each `BisectStep` row records the
`checkpoint` queried, the resulting `PairedCellDelta`, and which
search interval boundary moved. `BisectResult.first_bad` is the
final answer; `result.steps` is the full audit trail. See
`docs/backlog.md` B-39 for the anti-feature framing (no
weight-interpolation, list-resolution-bound).

---

## Aggregate (fleet meta-report)

```python
from gauntlet.aggregate import (
    FleetReport, FleetRun,
    aggregate_reports, aggregate_directory, discover_run_files,
    render_fleet_html, write_fleet_html,
)
```

### `aggregate_directory(directory, *, persistence_threshold=0.5) -> FleetReport`

```python
from gauntlet.aggregate import aggregate_directory, write_fleet_html

fleet = aggregate_directory("runs/", persistence_threshold=0.5)
write_fleet_html(fleet, "fleet/fleet_report.html")
```

Recursively scans `directory` for `report.json` files and emits the
fleet roll-up. `persistence_threshold` is the minimum fraction of
runs a failure cluster must appear in to be flagged as persistent —
the headline cross-run signal. See
`docs/phase3-rfc-019-fleet-aggregate.md`.

---

## Dashboard (static SPA)

```python
from gauntlet.dashboard import (
    build_dashboard, build_dashboard_index, discover_reports,
)
```

### `build_dashboard(reports_dir, out, *, title="Gauntlet Dashboard")`

```python
from gauntlet.dashboard import build_dashboard

build_dashboard("runs/", "dashboard/", title="Cube-stacking sweep")
```

Materialises a self-contained dashboard SPA — `index.html` +
`dashboard.js` + `dashboard.css` — into `out`. The run index is
embedded inline as a JSON literal so the SPA opens from a `file://`
path with no server (sidesteps Chromium's CORS rejection of
same-origin file fetches). `build_dashboard_index(paths, base_dir)`
is the pure-data helper exposed for testing and notebook use. See
`docs/phase3-rfc-020-web-dashboard.md`.

---

## Replay (single-episode re-simulation)

```python
from gauntlet.replay import (
    replay_one, parse_override, validate_overrides, OverrideError,
)
```

### `replay_one(*, target, suite, policy_factory, overrides, env_factory=None) -> Episode`

```python
from gauntlet.replay import replay_one

replayed = replay_one(target=ep, suite=suite,
                      policy_factory=resolve_policy_factory("scripted"),
                      overrides={"lighting_intensity": 0.4})
```

In-process re-simulation of one Episode with optional axis overrides.
Uses the original `seed` from `target` so the rollout is bit-identical
absent overrides. `parse_override("AXIS=VALUE")` is the CLI-side
parser, exported for notebook callers. `OverrideError` is a
`ValueError` subclass. See
`docs/phase2-rfc-004-trajectory-replay.md`.

---

## Monitor (drift detector)

```python
from gauntlet.monitor import (
    PerEpisodeDrift, DriftReport,
    ActionEntropyStats, action_entropy,
    ConformalFailureDetector,
    StateAutoencoder, train_ae, score_drift,  # require [monitor] extra
)
```

Torch-free symbols (`PerEpisodeDrift`, `DriftReport`,
`action_entropy`, `ActionEntropyStats`, `ConformalFailureDetector`)
are eagerly re-exported. Torch-backed symbols (`StateAutoencoder`,
`train_ae`, `score_drift`) are lazy — accessing them on a torch-free
install raises a clean `ImportError` with the install hint.

### `ConformalFailureDetector` (B-01)

```python
from gauntlet.monitor import ConformalFailureDetector

# Calibrate on a held-out set of *successful* episodes from the same
# (policy, env) pair. Episodes must come from a SamplablePolicy run
# so Episode.action_variance is populated.
det = ConformalFailureDetector.fit(calibration_episodes, alpha=0.05)
score, alarm = det.score(candidate_episode)
det.to_dict()                  # JSON-serialisable round-trip artefact
```

Split-conformal failure-prediction detector keyed off
`Episode.action_variance` (the B-18 mode-collapse signal). Computes
the calibration-quantile threshold under the FIPER-style finite-
sample correction (`ceil((n + 1) * (1 - alpha)) / n`-quantile of the
calibration scores). At score time returns a per-episode
`(failure_score, failure_alarm)` pair: the score is the candidate's
`action_variance` divided by the threshold (>1 ⟹ more uncertain than
calibration); the alarm is `score > 1.0`. Pure numpy — no torch, no
scipy. Asymmetric by design: greedy policies leave
`Episode.action_variance` as `None` and the detector skips them
rather than fabricating a 0.0 signal. See `docs/backlog.md` B-01 for
the FIPER / FAIL-Detect references.

### `train_ae(trajectory_dir, *, out_dir, reference_suite, latent_dim=8, epochs=50, batch_size=256, lr=1e-3, seed=0)`

Fit a `StateAutoencoder` on the reference proprio trajectories
emitted by `Runner(trajectory_dir=...)`. Writes `weights.pt`,
`normalization.json`, `config.json` into `out_dir`.

### `score_drift(episodes_path, trajectory_dir, ae_dir, *, top_k=10) -> DriftReport`

Score a candidate sweep's per-episode reconstruction error against
the trained AE and emit a `DriftReport` (`top_k` most-OOD episode
indices, candidate vs reference reconstruction-error mean / p95).

See `docs/phase2-rfc-003-drift-detector.md`.

---

## ROS 2 bridge

```python
from gauntlet.ros2 import (
    Ros2EpisodePayload,
    Ros2EpisodePublisher, Ros2RolloutRecorder,  # require rclpy on PATH
)
```

`Ros2EpisodePayload` is the rclpy-free pydantic payload (eagerly
re-exported). The publisher / recorder are lazy — accessing them
without rclpy raises an `ImportError` with the apt / Docker install
hint.

### `Ros2EpisodePublisher`

```python
with Ros2EpisodePublisher(topic="/gauntlet/episodes",
                          node_name="gauntlet_episode_publisher",
                          qos_depth=10) as publisher:
    for ep in episodes:
        publisher.publish_episode(ep)
```

Publishes each `Episode` as JSON-inside-`std_msgs/msg/String` (RFC-010
§5). `publish_episode` returns the constructed `Ros2EpisodePayload`
so callers can audit the wire payload. `close()` is idempotent;
`__enter__` / `__exit__` wire the same.

### `Ros2RolloutRecorder`

```python
with Ros2RolloutRecorder(topic="/robot/joint_states",
                         out_path=Path("trajectory.jsonl"),
                         duration_s=30.0) as recorder:
    n = recorder.spin_until_done()
```

Subscribes to `topic` and dumps received messages as JSONL
(`{"timestamp", "topic", "data"}` per line; `data = str(msg)`,
lossy-but-generic). `duration_s=0.0` spins forever until `Ctrl-C`.

See `docs/phase2-rfc-010-ros2-integration.md`.

---

## Real-to-sim (scene reconstruction)

```python
from gauntlet.realsim import (
    Scene, CameraFrame, CameraIntrinsics, Pose,
    SCENE_SCHEMA_VERSION, MANIFEST_FILENAME,
    IMAGE_MAGIC_BYTES, POSE_BOTTOM_ROW_TOLERANCE,
    ingest_frames, save_scene, load_scene,
    IngestionError, SceneIOError,
    RealSimRenderer, RendererFactory, RendererRegistryError,
    register_renderer, get_renderer, list_renderers,
)
```

`IMAGE_MAGIC_BYTES` is the `dict[str, bytes]` mapping from
human-readable container name (`"png"`, `"jpeg"`, `"ppm-binary"`,
`"ppm-ascii"`) to the magic-byte prefix the ingest pipeline checks
before accepting a frame — magic-byte sniffing rather than Pillow /
imageio import on the hot path. `POSE_BOTTOM_ROW_TOLERANCE` is the
absolute tolerance the schema validator applies to the bottom row of
each `Pose` 4x4 matrix (`[0, 0, 0, 1]` to within this much) before
rejecting a non-rigid transform. `RendererFactory` is the
`Callable[[], RealSimRenderer]` type alias — the value
`register_renderer(name, factory)` accepts and `get_renderer(name)`
returns.

### `ingest_frames(frames_dir, calib_path, *, source=None) -> Scene`

```python
from gauntlet.realsim import ingest_frames, save_scene

scene = ingest_frames("frames/", "calib.json", source="robot42-log7")
manifest_path = save_scene(scene, "scene_dir/", frames_dir="frames/", symlink=False)
```

Validates a directory of camera frames + a calibration JSON
(intrinsics + per-frame poses) and produces a `Scene`. `save_scene`
writes `manifest.json` plus frame copies (or symlinks); `load_scene`
round-trips it. `IngestionError` / `SceneIOError` subclass
`ValueError`. See `docs/phase3-rfc-021-real-to-sim-stub.md`.

### `RealSimRenderer` Protocol + registry

`RealSimRenderer` is a `runtime_checkable` Protocol with one method:
`render(scene, viewpoint, intrinsics) -> np.ndarray` (HxWx3 uint8
RGB). The first concrete implementation (gaussian splatting) is
deferred. The registry is module-local, not part of
`gauntlet.plugins`, until a concrete renderer ships and a follow-up
RFC promotes it.

---

## Plugins (third-party env / policy)

```python
from gauntlet.plugins import (
    POLICY_ENTRY_POINT_GROUP, ENV_ENTRY_POINT_GROUP,
    discover_policy_plugins, discover_env_plugins,
    warn_on_collision,
)
```

Third-party packages register policies under the `gauntlet.policies`
entry-point group and envs under `gauntlet.envs`. `discover_*` reads
the entry-point table (cached via `lru_cache`); failed `ep.load()`
calls are wrapped in a `RuntimeWarning` and dropped from the returned
dict so one broken plugin does not take the harness down.
`warn_on_collision` flags identity-mismatched name collisions between
built-ins and plugins. The built-in registries (`gauntlet.policy.registry`,
`gauntlet.env.registry`) enforce built-in-wins precedence one layer
up. See `docs/polish-exploration-plugin-system.md` and
`docs/plugin-development.md`.
