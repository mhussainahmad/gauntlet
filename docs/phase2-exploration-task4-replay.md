# Phase 2 Exploration: Task 4 — Trajectory Replay

**Date:** 2026-04-22
**Task:** Touch-point map for `gauntlet replay`.
**Scope:** Read-only. No code written.

---

## Q1: Does `Episode` have stable, reproducible identity?

**YES — identity is complete and deterministic.**

Identity tuple: `(suite_name, cell_index, episode_index, seed, perturbation_config)` — all fields already exist on the `Episode` pydantic model at `src/gauntlet/runner/episode.py:38-60`.

- `cell_index` and `episode_index` are stable ordinals assigned in `Suite.cells()` enumeration order (`src/gauntlet/suite/schema.py:262-280`).
- `seed` is derived from the master seed via `SeedSequence.spawn` and stored on every Episode (`src/gauntlet/runner/worker.py:156-165`).
- `perturbation_config: dict[str, float]` is a copy of the cell's axis values.
- Output ordering is "a stable contract downstream code keys off" (`src/gauntlet/runner/runner.py:50-54`).

**Minimum schema additions: 0 fields.** The existing tuple is sufficient to drive an exact bit-identical rerun.

---

## Q2: Can the Runner execute a single episode in-process?

**YES — fully supported, zero blockers.**

- `Runner(n_workers=1)` branches to `_run_in_process(policy_factory, work_items)` (`src/gauntlet/runner/runner.py:175-176, 249-264`), which loads the env once and iterates work items sequentially. No `multiprocessing.Pool`, no `spawn` context, no pickle boundary.
- `Suite` is a `pydantic.BaseModel` (`src/gauntlet/suite/schema.py:184-214`). Construct programmatically: `Suite(name="...", env="tabletop", episodes_per_cell=1, seed=..., axes={...})`. No YAML loader required.
- A 1-cell Suite falls out naturally: each axis declared with a single categorical value (`AxisSpec(values=[x])`) → `Suite.cells()` yields one cell (`src/gauntlet/suite/schema.py:276-280`). No "explicit single-cell" schema support needed; Cartesian product of `[v1] × [v2] × ...` is one tuple.

**Replay workflow:** load Episode JSON → build 1-cell Suite from `perturbation_config` (each axis → categorical with its value, with one axis overridden if the user passed `--override`) → `Runner(n_workers=1).run(policy_factory, suite)` → one Episode.

---

## Q3: Perturbation flow (Suite → Runner → Env)

**Seven canonical axis names** accepted by `TabletopEnv.set_perturbation` (`src/gauntlet/env/perturbation/__init__.py:81-89` + `src/gauntlet/env/tabletop.py:78-88`):

1. `lighting_intensity`
2. `camera_offset_x`
3. `camera_offset_y`
4. `object_texture`
5. `object_initial_pose_x`
6. `object_initial_pose_y`
7. `distractor_count`

**Value shape:** `(name: str, value: float)` — all floats, even for `object_texture` (categorical index) and `distractor_count` (int count). `src/gauntlet/suite/schema.py:60-63`.

**Flow:**

1. `Suite.axes` declares `AxisSpec` per axis; `cells()` builds `SuiteCell.values: Mapping[str, float]` for each grid point.
2. `Runner` copies to `WorkItem.perturbation_values: dict[str, float]` (`src/gauntlet/runner/runner.py:238`).
3. Worker's `_execute_one` loops `for name, value in item.perturbation_values.items(): env.set_perturbation(name, value)` (`src/gauntlet/runner/worker.py:189-191`).
4. `TabletopEnv.set_perturbation(name, value)` validates `name in _KNOWN_AXIS_NAMES` and queues `_pending_perturbations[name] = value` (`src/gauntlet/env/tabletop.py:331-339`).
5. On `reset()`, the env applies each pending value via `_apply_one_perturbation`.

**Range validation is sparse:** only `distractor_count` has hard bounds `[0, N_DISTRACTOR_SLOTS=10]` (`src/gauntlet/env/tabletop.py:60-64, 333-338`). The other six axes trust the caller — `replay`'s `--override` parser should add per-axis sanity checks to catch typos / out-of-range values before the simulation starts.

---

## Minimum schema additions needed

**Zero.** The existing `Episode` fields already carry the full identity tuple and the full perturbation config.

## Single-episode-in-process feasibility

**YES.** `Runner(n_workers=1)` already short-circuits to `_run_in_process`.

## Replay implementation footprint estimate

- **~40-60 lines of new code** for the core `replay_episode()` function.
- **+CLI glue** (`gauntlet replay` typer subcommand).
- **Files touched:** 1 new file (e.g. `src/gauntlet/replay.py` or `src/gauntlet/tools/replay.py`), 1 CLI extension.
- **Schema changes:** none.
- **Runner changes:** none (reuse the `n_workers=1` in-process path).

The smallest task of Phase 2.
