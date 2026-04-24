# Phase 2 Task 9 Exploration: Isaac Sim Backend

**Date:** 2026-04-24
**Branch:** phase-2/isaac-sim-adapter
**Base:** cea727b (main, post-Task-8)
**Target simulator:** [Isaac Sim](https://developer.nvidia.com/isaac/sim) — NVIDIA Omniverse Kit, distributed as `isaacsim` on PyPI since 2024.

This document is the measurement / decision pass that constrains RFC-009.
The RFC itself ships under `docs/phase2-rfc-009-isaac-sim-adapter.md`.

---

## Q1: Does `isaacsim` install cleanly against the existing dep tree?

**Short answer:** No — not on the CPU-only `ubuntu-latest` runner the rest
of the extras matrix targets, and not without a 5-15 GB transitive download
even when CUDA is available. This is the dominant constraint that shapes
every other decision in this doc.

| Fact | Citation / observation |
|---|---|
| `isaacsim` is on PyPI (since Isaac Sim 4.x, ~2024) | https://pypi.org/project/isaacsim/ |
| Hard CUDA / GPU runtime requirement | Wraps Omniverse Kit, which requires NVIDIA RTX-class GPU + ≥525 driver. Even `import isaacsim` triggers Kit bootstrap that probes for a GPU. |
| Footprint | `isaacsim` is a small shim (~1 MB) that pulls `isaacsim-kernel`, `isaacsim-core`, `isaacsim-app`, `isaacsim-asset`, ... — full install is ~15 GB once Kit and its asset cache are on disk. |
| `Requires-Python` | 3.10 / 3.11 only on Isaac Sim 4.x; 3.11 on 4.5+. |
| `pip install isaacsim` on a CPU runner | Wheels resolve, but every meaningful import (`from isaacsim import SimulationApp` and below) raises a Kit bootstrap error or segfaults probing for the GPU. |
| Comparison vs other extras' install footprint | `[hf]` ~3 GB (torch + transformers); `[lerobot]` ~4 GB; `[pybullet]` ~50 MB; `[genesis]` ~1 GB. `[isaac]` is the heaviest by an order of magnitude. |

**Consequence:** unlike `[genesis]` and `[pybullet]`, the `[isaac]` extra
cannot be tested end-to-end on `ubuntu-latest`. We pick a different
strategy from those RFCs: **mock-based unit tests in the default
torch-free CI job, no live `isaacsim` round-trip in CI**. RFC-009 §8
formalises this.

---

## Q2: Python matrix

| Python | `isaacsim` 4.5+ wheel | CI strategy |
|---|---|---|
| 3.11 | wheel published | Default-job mocked tests (no `isaacsim` install). |
| 3.12 | wheel published on Isaac Sim 4.5+ | Default-job mocked tests. |
| 3.13 | not yet published | N/A. |

Because the `[isaac]` extra never gets installed in CI under this PR's
scope, the matrix point is moot for pin-validation purposes — the pin
`isaacsim>=4.5,<5` gates only `uv sync --extra isaac` on a developer's
GPU workstation.

---

## Q3: Why state-only (deferring rendering)

Three independent reasons all point the same way:

1. **Genesis state-only-first-cut precedent (RFC-007 §2 non-goals).**
   The previous backend RFC carved rendering into a follow-up
   (RFC-008). Mirroring that staging keeps each PR small enough to
   review and matches the codebase's existing review-shape pattern.
2. **Headless rendering on Isaac Sim requires a different bootstrap
   path** (`SimulationApp({"renderer": "RaytracedLighting", "headless":
   True})`) and the cosmetic axes (`lighting_intensity`, `object_texture`,
   `camera_offset_{x,y}`) wire into Omniverse USD prim attributes
   (`UsdLuxDistantLight.intensityAttr`, `UsdShade.MaterialBindingAPI`,
   `UsdGeomCamera.set_world_pose`) — a different design surface than
   the PyBullet `getCameraImage` or Genesis `cam.render` path. Worth a
   dedicated RFC.
3. **Mock-driven tests get progressively harder as the surface grows.**
   A state-only adapter touches ~10 `omni.isaac.core` symbols. Adding
   rendering would push that to ~25 and force the conftest fake to
   model USD prim hierarchies. State-only stops at the smallest
   useful contract.

The cosmetic axes are still part of `AXIS_NAMES` (canonical 7) but
declared `VISUAL_ONLY_AXES` so the suite loader's
`_reject_purely_visual_suites` guard rejects cosmetic-only sweeps on
this backend — same shape Genesis had pre-RFC-008 and PyBullet had
pre-RFC-006.

---

## Q4: Mocking strategy for unit tests

The constraint: tests must run in the default torch-free CI job, which
does NOT install `isaacsim`. So we inject a fake `isaacsim` namespace
into `sys.modules` BEFORE importing `gauntlet.env.isaac.tabletop_isaac`,
construct the adapter under the fake, and assert on the surface
contract (spaces, axis-name validation, perturbation-call dispatch,
reset/step ordering).

The plan, prototyped in §Q4.1 and pinned in RFC-009 §8:

1. **`tests/isaac/conftest.py`** — module-scoped autouse fixture
   `_install_fake_isaacsim` that populates
   `sys.modules["isaacsim"]`, `sys.modules["isaacsim.core"]`,
   `sys.modules["isaacsim.core.api"]`, `sys.modules["isaacsim.core.api.objects"]`,
   `sys.modules["omni"]`, `sys.modules["omni.isaac"]`,
   `sys.modules["omni.isaac.core"]` with a `MagicMock` whose
   `SimulationApp.__init__` is a no-op and whose `World` /
   `DynamicCuboid` / `VisualCuboid` / `XFormPrim` factories return
   stubs with the methods the adapter calls (`reset`, `step`,
   `get_world_pose`, `set_world_pose`, `add`).

2. **The fake objects must produce real numpy values for
   `get_world_pose`** so the adapter's `_build_obs` can pass through
   numpy casting without `MagicMock` leaking into a numpy array (which
   would fail `dtype=float64` validation). The fixture exposes a
   helper for tests to override the next pose returned by a given
   prim.

3. **Per-test pose injection.** For state-effecting axis tests
   (`object_initial_pose_x`), the test queues the perturbation, calls
   `env.reset(seed=...)`, and asserts that the FAKE
   `cube_prim.set_world_pose` was called with the right tuple. The
   tests do not assert on rendered pixels, sim physics, or downstream
   GPU state — only on the adapter -> `omni.isaac.core` call surface.

### Q4.1 Adapter <-> isaacsim touchpoints (locked, ~10 symbols)

These are the only `isaacsim` / `omni.isaac.core` symbols the adapter
imports / calls. The mock surface in `conftest.py` only has to cover
these, which keeps the fake under 80 lines.

| Touchpoint | Adapter use |
|---|---|
| `isaacsim.SimulationApp({"headless": True})` | Bootstrap Kit (lazy, in `__init__`). |
| `omni.isaac.core.World()` | The simulation world singleton. |
| `world.reset()` | Per-episode reset. |
| `world.step(render=False)` | Per-control-step physics advance (n_substeps loop). |
| `world.scene.add(prim)` | Register a prim with the scene graph. |
| `omni.isaac.core.objects.DynamicCuboid(prim_path, position, size)` | The cube. |
| `omni.isaac.core.objects.VisualCuboid(...)` | EE marker (kinematic). |
| `omni.isaac.core.objects.FixedCuboid(...)` | Table-top + distractors. |
| `prim.get_world_pose() -> (np.ndarray pos, np.ndarray quat)` | Per-step obs. |
| `prim.set_world_pose(position=..., orientation=...)` | Reset + perturbation writes. |

The Genesis adapter touches a comparable footprint (`gs.Scene`,
`scene.add_entity`, `entity.set_pos/get_pos`, etc.); the count is
intentionally similar so the surface is comparable.

---

## Q5: CI gating decision

Three options surveyed:

| Option | Outcome |
|---|---|
| (a) Real `pip install isaacsim` on `ubuntu-latest` and run live tests. | Fails — Kit bootstrap segfaults without a GPU. Job permanently red. |
| (b) Self-hosted GPU runner. | Out of scope and out of budget for this PR. |
| (c) Default-job mocked tests + a placeholder `isaac-tests` job that's allowed to fail. | Picked. Mocked tests give us coverage; placeholder job exists for the day a GPU runner is wired up. |

**RFC-009 §8 picks option (c).** The placeholder job runs
`uv sync --extra isaac --group isaac-dev` (will fail on
`ubuntu-latest`; that's expected and gated by `continue-on-error: true`)
and then `pytest -m isaac --co -q || [ $? -eq 5 ]` (zero tests is
fine — pytest's exit-5 is silenced). The only purpose of the job is to
keep the wiring visible so a future GPU-enabled worker run is one
config-change away.

The mocked tests live in `tests/isaac/` and are NOT marked
`@pytest.mark.isaac` — they run in the default torch-free CI job and
have to pass there. The `isaac` marker is reserved for the future
real-runtime tests that will be gated to the GPU runner.

---

## Q6: Adapter scope (locked)

State-only; no rendering; ImportError-cleanly when `isaacsim` is
absent; mocked unit tests in the default job. Mirrors the Genesis
state-only first cut at the public-surface level:

* `IsaacSimTabletopEnv(*, max_steps=200)` — same kwarg signature as
  the state-only Genesis env. No `render_in_obs` kwarg this PR.
* `observation_space: spaces.Dict` with the same five keys as
  state-only Genesis: `cube_pos` (3,), `cube_quat` (4,), `ee_pos`
  (3,), `gripper` (1,), `target_pos` (3,) — all `dtype=np.float64`.
  `step` lives on `info`, not obs.
* `action_space: spaces.Box(low=-1.0, high=1.0, shape=(7,),
  dtype=np.float64)` — 7-D EE-twist+gripper, identical to MuJoCo /
  PyBullet / Genesis.
* `AXIS_NAMES: ClassVar[frozenset[str]]` — the canonical 7.
* `VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset({
  "lighting_intensity", "camera_offset_x", "camera_offset_y",
  "object_texture"})` — the four cosmetic axes (state-only first
  cut). Triggers the suite loader's cosmetic-only-sweep rejection.
* `set_perturbation(name, value)`, `restore_baseline()`, `reset()`,
  `step()`, `close()` matching the `Env` Protocol in
  `src/gauntlet/env/base.py`.

Wired into `gauntlet.env.registry` as `"tabletop-isaac"`, lazily
imported from `src/gauntlet/env/isaac/__init__.py` with the
two-part-ish guard pattern Genesis uses (single ImportError this time
because Isaac Sim doesn't have an undeclared transitive runtime dep).

---

## Q7: Registration footprint

**New files:**

* `docs/phase2-rfc-009-isaac-sim-adapter.md` — the RFC.
* `src/gauntlet/env/isaac/__init__.py` — guard import + register.
* `src/gauntlet/env/isaac/tabletop_isaac.py` — `IsaacSimTabletopEnv`.
* `tests/isaac/__init__.py` — empty marker.
* `tests/isaac/conftest.py` — fake `isaacsim` namespace fixture.
* `tests/isaac/test_import_guards_isaac.py` — ImportError-without-extra coverage.
* `tests/isaac/test_env_isaac.py` — full state-obs / spaces / reset / step / set_perturbation tests under the mock.
* `tests/isaac/test_suite_loader_isaac.py` — loader accepts state-only sweep, rejects cosmetic-only sweep.
* `examples/evaluate_random_policy_isaac.py` — Genesis example mirror.

**Modified files:**

* `pyproject.toml` — `[isaac]` extra (`isaacsim>=4.5,<5`); `[dependency-groups] isaac-dev`; `[tool.pytest.ini_options]` markers += `isaac`; `[[tool.mypy.overrides]]` for `isaacsim`, `omni.*`, `pxr.*`. The default `lint-typecheck-test` job's `-m "not …"` exclusion grows by `and not isaac`.
* `src/gauntlet/suite/schema.py` — `BUILTIN_BACKEND_IMPORTS["tabletop-isaac"] = "gauntlet.env.isaac"`.
* `src/gauntlet/suite/loader.py` — `_EXTRA_FOR_MODULE["gauntlet.env.isaac"] = "isaac"`.
* `.github/workflows/ci.yml` — new `isaac-tests` placeholder job (continue-on-error). Default job's pytest exclusion list grows.
* `README.md` — backends table gets a `tabletop-isaac` row.

**Unmodified by design:**

* `src/gauntlet/env/base.py`, `src/gauntlet/env/registry.py`,
  `src/gauntlet/runner/*`, `src/gauntlet/suite/loader.py` (loader
  cosmetic-only-sweep rejection works for free off `VISUAL_ONLY_AXES`
  via `_visual_only_axes_of`) — Protocol seam from RFC-005 holds.
* MuJoCo / PyBullet / Genesis adapters — no cross-backend changes.

---

## Q8: Risk register

1. **Isaac Sim 4.x -> 5 API churn.** Pin `isaacsim>=4.5,<5` (same one-major
   ceiling as `[hf]`, `[lerobot]`, `[pybullet]`). When 5 lands, revisit.
2. **Mock-based tests can drift from real behaviour.** Mitigation: the
   surface is small (~10 symbols, §Q4.1); per-axis tests assert on
   call shape (positional args, kwargs, target prim) so a real-API
   rename will fail loudly during a manual GPU-workstation smoke. We
   document in RFC-009 §11 the procedure for an external contributor
   on a GPU box to validate the adapter against real Isaac Sim.
3. **`SimulationApp` is a process-global singleton with destructive
   `close()`.** The adapter's `close()` is a best-effort drop (matches
   Genesis's `close`); follow-up tests under a real GPU runner may need
   to skip the `close()` to avoid invalidating the global Kit state
   between test functions. Documented but not blocking.
4. **`isinstance(env, GauntletEnv)` under the mock** — verified in §Q4
   prototype: the adapter's `observation_space` and `action_space` are
   plain `gymnasium.spaces` instances, not mocked, so the runtime
   Protocol check passes regardless of how the underlying Kit
   primitives are stubbed. Pinned in RFC-009 §8 test list.
5. **`pytest -m isaac` exit-5 in the placeholder CI job.** No tests
   carry the `isaac` marker in this PR (the `isaac` marker is
   reserved). Use `pytest --co -q -m isaac || [ $? -eq 5 ]` so the
   step succeeds on zero-collection.

---

## 150-word executive summary

`isaacsim` is on PyPI but requires CUDA + an NVIDIA RTX-class GPU at
runtime; the wheels resolve on `ubuntu-latest` but every meaningful
import segfaults probing for a GPU. So the `[isaac]` extra cannot be
tested end-to-end in CI on the existing runner pool. We pick a
mock-based contract: tests in `tests/isaac/` inject a fake `isaacsim`
+ `omni.isaac.core` namespace into `sys.modules` before importing the
adapter, then assert on the public surface (spaces, axis dispatch,
reset/step ordering, per-axis prim-call shape). The state-only
contract mirrors `GenesisTabletopEnv`'s pre-RFC-008 shape: same five
state-obs keys, same 7-D action, same canonical 7 axes, four cosmetic
axes flagged `VISUAL_ONLY_AXES` so the suite loader rejects
cosmetic-only sweeps. CI gets a placeholder `isaac-tests` job that's
`continue-on-error: true` until a GPU runner exists. New backend
registers as `"tabletop-isaac"`; canonical install hint is
`uv sync --extra isaac`.
