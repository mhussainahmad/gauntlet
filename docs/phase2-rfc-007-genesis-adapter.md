# Phase 2 RFC 007 — Genesis backend adapter

- **Status**: Draft
- **Phase**: 2, Task 7 (`GAUNTLET_SPEC.md` §7: "Additional simulators: Isaac Sim, Genesis, PyBullet adapters.")
- **Author**: architecture agent
- **Date**: 2026-04-23
- **Supersedes**: n/a
- **References**:
  - `docs/phase2-rfc-001-huggingface-policy.md` (`[hf]` extras pattern, torch-free core rule).
  - `docs/phase2-rfc-003-drift-detector.md` (`[monitor]` extra — additive-torch precedent).
  - `docs/phase2-rfc-005-pybullet-adapter.md` (`GauntletEnv` Protocol + registry, state-only-first-cut template this RFC reuses).
  - `docs/phase2-rfc-006-pybullet-rendering.md` (shape of the follow-up rendering RFC this one defers to).
  - `docs/phase2-exploration-task7-genesis-backend.md` (the measurement pass that constrains §4 / §5 / §6).

---

## 1. Summary

RFC-005 introduced `GauntletEnv` + `gauntlet.env.registry` as the backend-agnostic seam, and landed PyBullet first because it was the easiest non-MuJoCo simulator to get right. RFC-005 §1 and §14 called out Genesis as a separate follow-up because its API was churning. Genesis has now stabilised enough at 0.4.x (CPU-first, bit-identical same-process determinism, primitive-friendly scene API — all measured in the exploration doc) that a third backend becomes worth the trouble.

This RFC adds `"tabletop-genesis"` behind a new `[genesis]` extra, reusing the `GauntletEnv` Protocol, the registry, and the state-only-first-cut pattern from RFC-005. The core plumbing is unchanged: `base.py`, `registry.py`, the Runner, the Suite loader already dispatch through the Protocol. Only three code paths touch core surface: (a) `pyproject.toml` gains a `[genesis]` extra + `[dependency-groups] genesis-dev` + a new `genesis` pytest marker, (b) the core `rich` ceiling bumps from `<15` to `<16` to accommodate Genesis's transitive `rich 15.0.0` (empirically verified, genesis-world declares no upper bound), and (c) CI gains a `genesis-tests` job mirroring `pybullet-tests`.

Two install-time deviations vs RFC-005 are honesty-flagged up front (§4). Genesis does **not** declare `torch` in `install_requires` — it delegates to users so they can pick CPU vs CUDA wheels — so the `[genesis]` extra explicitly lists both `genesis-world` and `torch>=2.2,<3`, and the `gauntlet.env.genesis/__init__.py` guard handles both missing cases with a single clear message. The second is the `rich` ceiling bump just mentioned.

## 2. Goals / non-goals

### Goals

- Ship `gauntlet.env.genesis.tabletop_genesis.GenesisTabletopEnv` satisfying `GauntletEnv` (from RFC-005) structurally — no new protocol methods.
- All seven canonical perturbation axes validated on `set_perturbation`; state-affecting axes (`object_initial_pose_{x,y}`, `distractor_count`) wire through to scene state; cosmetic axes (`lighting_intensity`, `camera_offset_{x,y}`, `object_texture`) queue + validate and are listed in `VISUAL_ONLY_AXES` pending a follow-up rendering RFC — same state-only-first-cut carve-out RFC-005 used (§6).
- Same 7-D continuous action space, same observation-dict keys, same max-steps/default-substeps as `TabletopEnv` and `PyBulletTabletopEnv`. Policies written for the other backends swap in without code changes.
- Keep `gauntlet.core` Genesis-free: importing `gauntlet`, `gauntlet.env` (excluding the `genesis/` subpackage), `gauntlet.policy`, `gauntlet.runner`, `gauntlet.suite`, `gauntlet.report` must not transitively import `genesis` or `torch`. The `[genesis]` extra is lazily imported from inside `gauntlet.env.genesis.__init__` with a clear install-hint `ImportError`.
- Same-process bit-determinism: `(master_seed, cell, ep)` produces bit-identical episode output on `tabletop-genesis` the way it does on MuJoCo and PyBullet (measured: 0.0 abs-diff across 20 steps — exploration §Q5).
- `mypy --strict` passes whether or not `[genesis]` is installed (needs the same `[[tool.mypy.overrides]]` trick RFC-005 used for `pybullet`/`pybullet_data`).
- `uv sync --extra genesis` enables the backend in one command; composes cleanly with `--extra hf`, `--extra lerobot`, `--extra monitor`, `--extra pybullet`.

### Non-goals

- **Isaac Sim support.** Same reasoning as RFC-005 §2. Remains deferred.
- **Image rendering on Genesis** — cosmetic axes state-only on first cut, follow-up RFC will add `render_in_obs=True` (same staging as RFC-005 → RFC-006 for PyBullet). §9 sketches it.
- **Cross-backend numerical parity.** Same policy + same seed on `tabletop` vs `tabletop-pybullet` vs `tabletop-genesis` produces three different trajectories. Documented, not a bug. Same wording as RFC-005 §7.4.
- **URDF robot arm + IK.** MuJoCo uses a mocap body; PyBullet uses a fixed constraint; Genesis uses a direct `entity.set_pos()` teleport each step on a small "EE" box. No arm asset across any backend.
- **New perturbation axes.** The canonical 7 are the contract.
- **GPU backend.** `gs.init(backend=gs.cuda)` is untested — CI is CPU-only and Genesis's `gs.cpu` is first-class supported. Users who want GPU set their own backend at runtime; no code in this RFC keys off GPU-specific behaviour.

## 3. Protocol / registry reuse

No changes. RFC-005 already extracted `GauntletEnv` as a `runtime_checkable` Protocol and `gauntlet.env.registry` as a plain `dict[str, Callable]`. `GenesisTabletopEnv` satisfies the Protocol structurally — same `AXIS_NAMES: ClassVar[frozenset[str]]`, same `observation_space`/`action_space`, same `reset`/`step`/`set_perturbation`/`restore_baseline`/`close` surface. Registration is a two-line call inside `gauntlet.env.genesis.__init__` under the key `"tabletop-genesis"`.

## 4. `[genesis]` extra — pin + dep-tree decisions

### 4.1 Extras definition (proposed)

```toml
[project.optional-dependencies]
# Independent extra for the Genesis backend (RFC-007).
# genesis-world is pure-Python + some native extensions, CPU-first, but
# imports torch at module scope without declaring it in install_requires
# (upstream delegates to users so they can pick CPU/CUDA/ROCm wheels).
# We declare torch explicitly so `uv sync --extra genesis` always
# produces a working import.
genesis = [
    "genesis-world>=0.4,<0.5",
    "torch>=2.2,<3",
]

[dependency-groups]
genesis-dev = [
    {include-group = "dev"},
    "pytest-mock>=3.12,<4",
]
```

### 4.2 Pin rationale

- `genesis-world>=0.4,<0.5` — tighter than the other extras' one-major-ceiling (`[hf]` is `<5`, `[lerobot]` is `<1`, `[pybullet]` is `<4`) because the 0.4.x → 0.5 boundary is expected to include breaking API changes (scene/entity API churn is the dominant upstream risk surfaced in exploration §Q8). Intentional: catch breakage at the extras seam, not inside user evaluation runs.
- `torch>=2.2,<3` — matches the `[hf]` and `[monitor]` floors for cross-extras compositional cleanness. A user running `uv sync --extra hf --extra genesis` gets a single resolved torch. No change to existing extras needed.
- No `transformers`, no `lerobot`, no `pillow` in `[genesis]`. Genesis brings its own visualisation stack (`pyglet`, `trimesh`, `vtk`). We do not depend on any of them — they come along as transitive deps of `genesis-world` but no `gauntlet.env.genesis` code imports them.

### 4.3 Core-dep ceiling bump: `rich<15` → `rich<16`

Genesis's transitive resolution picks `rich 15.0.0`; Genesis declares no upper bound. Our core currently pins `rich>=13.7,<15`. Without the bump, `uv sync --extra genesis` fails to resolve. The bump is safe for the existing CLI colour-output code (no Rich 14→15 API usage that we know of — the CLI uses only `rich.console.Console`, `rich.traceback`, `rich.text.Text`, all stable since Rich 12). This is the only core-dep change this RFC forces.

### 4.4 Size / install footprint

`genesis-world` itself is 82.6 MB (one wheel, `py3-none-any`). The transitive tree adds ~1 GB uncompressed — vtk, scikit-image, moviepy, pymeshlab are the heaviest. Comparable to `[lerobot]` (which pulls torch + transformers + lerobot's own assets). CI job runtime is expected to be dominated by the `uv sync` step (~60 s cold cache); the test step itself runs fast because scene-build is cached after the first env instantiation within a pytest session (exploration §Q4).

### 4.5 Python support matrix

| Python | Wheel | CI |
|---|---|---|
| 3.11 | ✅ | ✅ in `genesis-tests` job matrix |
| 3.12 | ✅ | ✅ in `genesis-tests` job matrix |
| 3.13 | ✅ | not in matrix (other extras also test only 3.11/3.12) |
| 3.14 | ❌ `Requires-Python: <3.14` | N/A — ceiling set upstream |

### 4.6 Interaction with the Runner's `spawn` start method

`genesis-world` initialises global state via `gs.init(backend=...)`. Each worker subprocess is a fresh Python interpreter (spawn, not fork), so `gs.init` must run once in each worker. This happens naturally because `GenesisTabletopEnv.__init__` calls `gs.init` — the Runner's env-factory contract is "construct one env per worker", which amortises the ~4-s init cost across every episode the worker handles. The factory itself is module-level (pickles cleanly under spawn), same pattern RFC-005 used.

`gs.init` is idempotent on repeated calls within one process (verified empirically in the exploration pass); the adapter guards against double-init just in case.

## 5. Scene layout

Mirrors `PyBulletTabletopEnv` (RFC-005 §5) at the entity inventory level, adapted to Genesis primitives:

```
gs.Scene
├── gs.morphs.Plane()                             # ground, fixed
├── gs.morphs.Box  (table-top, fixed, 1.0×1.0×0.04)
├── gs.morphs.Box  (cube, movable, 0.05³)          # the pick target
├── gs.morphs.Box  (EE, kinematic-by-teleport, 0.02³)
├── gs.morphs.Cylinder (target disc, fixed visual, radius 0.05)
└── gs.morphs.Box × 10 (distractors, pre-allocated, teleported on/off)
```

Key differences vs PyBullet:

- **No URDF, no MJCF** — all entities are Genesis primitives (`Box`, `Plane`, `Cylinder`). Keeps `scene.build()` at the ~5-s minimum and the repo ships no Genesis-specific URDF/MJCF asset. The only optional asset would be a cube-face texture, deferred to the rendering follow-up (image swap is state-invisible).
- **No kinematic constraint primitive.** Genesis has no `createConstraint` equivalent. The EE body is driven by calling `entity.set_pos()` every physics step — verified to work (50 steps, 78 ms, no physics glitches on the cube — exploration §Q6). Semantically equivalent to PyBullet's `p.changeConstraint` loop.
- **Distractor bodies exist always.** Disabling means `set_pos((x, y, -10.0))` (below the ground plane and outside every camera frustum); enabling means `set_pos((x, y, rest_z))`. §6.7 expands.
- **Materials / surfaces at build time only.** Genesis's `gs.surfaces.*` objects are attached at `scene.add_entity(...)`; post-build surface mutation is not a stable API (exploration §Q6). The cube gets the default surface; the "alt" texture for `object_texture` is deferred to the rendering follow-up RFC.

Scene construction happens in `__init__`. `scene.reset()` + targeted `set_pos` / `set_quat` calls handle the per-episode state restoration inside `restore_baseline()` and the per-reset perturbation application inside `reset()` — same four-step ordering the Protocol demands (RFC-005 §3.2): `restore_baseline` → seed randomisation → apply queued perturbations → clear queue.

### 5.1 Physics configuration

- `dt`: default Genesis timestep (`1/100 s` for the rigid solver). We run `n_substeps=5` by default, same as `TabletopEnv`, giving a 50 Hz outer control loop. `SimOptions(dt=0.01, substeps=5)` passed at `gs.Scene(...)`.
- Gravity: `(0.0, 0.0, -9.81)` — Genesis default; kept.
- Solver: Genesis's default rigid solver (positional, iterative). Not configurable in the same way as MuJoCo's `integrator=implicit` / PyBullet's `num_solver_iterations`. This is one of the numerical-parity deltas (§7.3).

## 6. The 7 axes on Genesis

Five groups of decisions. Documented upfront, implemented in order.

### 6.1 `lighting_intensity` — cosmetic, deferred

Genesis's `scene.add_light(pos, dir, color, intensity, directional, castshadow, cutoff, attenuation)` is only called at scene construction. Post-build intensity mutation is not a stable public API at 0.4.6 (exploration §Q6). **State-only first cut: the axis queues + validates, stores the value on `self._light_intensity`, and is declared a member of `VISUAL_ONLY_AXES`.** The follow-up rendering RFC will (a) pin a Genesis minor pin that exposes `light.set_intensity()`, or (b) rebuild-on-change inside an image-only render path. Same shape RFC-005 used for the same axis on PyBullet before RFC-006.

### 6.2 `camera_offset_x`, `camera_offset_y` — cosmetic, deferred

`scene.add_camera(res, pos, lookat, fov)` plus `cam.set_pose(pos=..., lookat=...)` is stable and works post-build. But on the state-only first cut, **no camera is instantiated** (obs dict has no image key). The offset is queued + validated + stored on `self._cam_offset` and is a member of `VISUAL_ONLY_AXES`. Follow-up rendering RFC: creates the camera in `__init__`, wires `set_pose` into the perturbation branch. Zero-cost scaffold (store the offset now; rendering RFC reads it later).

### 6.3 `object_texture` — cosmetic, deferred

No runtime material-ID or surface swap at 0.4.6 (exploration §Q6). State-only first cut: queues + validates + stores on `self._texture_choice`, `VISUAL_ONLY_AXES` membership. Follow-up rendering RFC will either author two cube entities (one visible per episode via teleport) or wait for upstream `surface.set_color()`.

### 6.4 `object_initial_pose_x`, `object_initial_pose_y` — state-affecting, full parity

Direct: after `scene.reset()` and the default seed-driven randomisation, call `self._cube.set_pos((value, 0.0, cube_rest_z))` (for x) or `(…, value, …)` (for y). Happens inside `reset()`'s "apply queued perturbations" step, matching the MuJoCo semantic of "override the randomised qpos with an axis-driven value".

### 6.5 `distractor_count` — state-affecting, teleport-away semantic

10 distractors pre-allocated at build (static ring around workspace, same positions RFC-005 §5.1 chose):

- When `count == N` (0 ≤ N ≤ 10), the first `N` distractors `set_pos((x_i, y_i, rest_z))` and the remaining `10 - N` `set_pos((x_i, y_i, -10.0))`. Position-driven only — Genesis has no per-entity visibility or collision-enabled flag at 0.4.6.
- Numerical deviation from MuJoCo: MuJoCo's branch also disables collision (`contype = 0`) on hidden distractors. Genesis's teleport-away distractors are physically present at `z = -10.0` where nothing can touch them; observationally equivalent but not mechanism-identical. **Documented deviation; not a correctness regression for any workload that doesn't drive the EE below the ground plane.**

Value validation (`_validate_distractor_count`): requires `round(value)` ∈ `[0, 10]`, same integer check PyBullet does (RFC-005 §6.7).

### 6.6 Summary table

| Axis | Category on Genesis | `VISUAL_ONLY_AXES` | Follow-up RFC |
|---|---|---|---|
| `lighting_intensity` | cosmetic | ✅ | ✅ (rendering) |
| `camera_offset_x` | cosmetic | ✅ | ✅ (rendering) |
| `camera_offset_y` | cosmetic | ✅ | ✅ (rendering) |
| `object_texture` | cosmetic | ✅ | ✅ (rendering) |
| `object_initial_pose_x` | state-affecting | ❌ | — |
| `object_initial_pose_y` | state-affecting | ❌ | — |
| `distractor_count` | state-affecting | ❌ | — |

Three of seven axes produce state-observable effects on the first cut. The four cosmetic axes' branches execute (storing values), are reachable via `set_perturbation`, and validate input — but produce no obs-dict delta until rendering lands. The Suite loader's existing "all-VISUAL_ONLY_AXES sweep → error" guard (RFC-005 §12, already in `suite/loader.py`) applies here too.

### 6.7 `AXIS_NAMES`

```python
AXIS_NAMES: ClassVar[frozenset[str]] = frozenset({
    "lighting_intensity", "camera_offset_x", "camera_offset_y",
    "object_texture", "object_initial_pose_x", "object_initial_pose_y",
    "distractor_count",
})
```

Byte-identical to `TabletopEnv.AXIS_NAMES` and `PyBulletTabletopEnv.AXIS_NAMES`. Canonical 7.

## 7. Determinism, parity, and the "Genesis-Genesis-only" compare contract

### 7.1 Same-process determinism (verified)

Exploration §Q5: two rollouts in the same process, same initial pose, same 20 steps → max abs position diff of 0.0 on the cube. The Runner relies on `(backend, master_seed, cell, ep)` determinism per-backend; Genesis supplies it.

### 7.2 Cross-process determinism (scope-caveated)

Not verified across subprocesses. RFC-005 made the same caveat for PyBullet. The Runner's worker-subprocess determinism is guaranteed by seed threading at the Python level (episode seeds are hashed from master + cell + ep); the *simulator's* cross-process determinism depends on numerical libraries' FP-ordering sensitivity. In practice: the `genesis` dependency stack (torch CPU, numpy) is deterministic on CPU with single-threaded default BLAS, but we do not ship a CI guard for this. If it becomes a real user complaint, a follow-up test can be added; it does not gate this RFC.

### 7.3 Cross-backend numerical non-parity

Same policy + same seed on `tabletop` vs `tabletop-pybullet` vs `tabletop-genesis` produces three different trajectories. All three are internally deterministic; none match the other two at the float level. `gauntlet compare` across backends measures simulator drift, not policy regression. This RFC does **not** try to match MuJoCo or PyBullet pixel-for-pixel or pose-for-pose; it matches at the observation-keys / action-space / reward-signal / success-criterion contract level, and the cube + EE behaviour is semantically plausible.

### 7.4 Success criterion

Same as the other backends: cube center within `TARGET_RADIUS = 0.05 m` of `target_pos` for one timestep. Termination semantics identical.

### 7.5 Quaternion convention

Genesis's entity quaternions are returned via `entity.get_quat()` as torch tensors. The convention (wxyz vs xyzw) is wxyz for Genesis's stable API — same as MuJoCo, unlike PyBullet. No conversion helper needed; the `_build_obs` just `.cpu().numpy()`s the tensor. (If this is wrong at test time, the fix is a one-liner in `_build_obs` — same shape RFC-005 §7.3 had to add for PyBullet's xyzw.)

## 8. Test matrix

All new tests live in `tests/genesis/` and are marked `@pytest.mark.genesis`. The default (torch-free) CI job excludes this marker via the existing `-m "not hf and not lerobot and not monitor and not pybullet"` pattern — which the RFC extends to `"not hf and not lerobot and not monitor and not pybullet and not genesis"`.

| Test | Verifies |
|---|---|
| `test_protocol_conformance` | `GenesisTabletopEnv` is an instance of `GauntletEnv` via `isinstance` (runtime-checkable Protocol). |
| `test_spaces_parity_with_mujoco` | Same `action_space` shape/dtype/bounds and same `observation_space` keys/shapes as `TabletopEnv` (for an instance created with `render_in_obs=False`). |
| `test_reset_seed_determinism` | Two `env.reset(seed=42)` + 20 steps produce bit-identical obs stacks. Exploration §Q5 verified empirically; this codifies it. |
| `test_set_perturbation_validates_axis_names` | Unknown axis → `ValueError` naming the axis. Covers all 7 known names. |
| `test_set_perturbation_validates_distractor_count_range` | Integer `[0, 10]` bound, same as PyBullet's. |
| `test_pose_perturbation_affects_cube_obs` | `set_perturbation("object_initial_pose_x", 0.1)` + `reset` → cube obs X reflects 0.1 (within a tolerance, not bit-equal). |
| `test_distractor_count_teleport_semantics` | With `count=3`, three distractors are at `rest_z`, seven at `-10.0`; verified via entity `get_pos()`. |
| `test_restore_baseline_is_idempotent` | Calling `restore_baseline()` twice has the same effect as once. |
| `test_cosmetic_axes_queue_store_and_clear` | Each `VISUAL_ONLY` axis (four of them) stores its queued value during the "apply perturbations" phase and clears the pending queue afterwards. Verified via inspection of private state — accepted test coupling, matches the PyBullet equivalent. |
| `test_suite_loader_rejects_cosmetic_only_sweep` | A Suite YAML naming only `VISUAL_ONLY` axes on `tabletop-genesis` is rejected at load time with the same clear message RFC-005 §12 specified. |
| `test_import_guard_missing_genesis_world` | `patch.dict(sys.modules, {"genesis": None})` → `import gauntlet.env.genesis` raises `ImportError` with the install hint. |
| `test_import_guard_missing_torch` | `patch.dict(sys.modules, {"torch": None})` → same, with torch-specific message. |

No rendering tests — rendering is out of scope. No cross-backend numerical comparison tests — explicitly non-goal (§7.3).

## 9. Follow-up RFC sketch (image rendering)

Will be RFC-008. Scope: add `render_in_obs=True` + `render_size=(H, W)` to `GenesisTabletopEnv`, wire `scene.add_camera` in `__init__`, wire `cam.set_pose` + `cam.render` for the four cosmetic axes. Unblocks VLA adapters (`HuggingFacePolicy`, `LeRobotPolicy`) on Genesis. Empties `VISUAL_ONLY_AXES` (mirrors what RFC-006 did to `PyBulletTabletopEnv.VISUAL_ONLY_AXES`). Drops no code landed by this RFC — only adds on top.

## 10. CI footprint

New job `genesis-tests` in `.github/workflows/ci.yml`, byte-pattern-copied from `pybullet-tests`:

```yaml
genesis-tests:
  name: genesis backend tests (py${{ matrix.python-version }})
  runs-on: ubuntu-latest
  strategy:
    fail-fast: false
    matrix:
      python-version: ["3.11", "3.12"]
  steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v3
      with:
        enable-cache: true
        cache-dependency-glob: "uv.lock"
    - name: Set up Python ${{ matrix.python-version }}
      run: uv python install ${{ matrix.python-version }}
    - name: Sync dependencies (genesis extra + genesis-dev group)
      run: >-
        uv sync
        --extra genesis
        --group genesis-dev
        --python ${{ matrix.python-version }}
    - name: Run genesis-marked tests
      run: uv run pytest -m genesis -q
```

Default `lint-typecheck-test` job's `pytest` `-m` string extends to exclude `genesis`. No other job changes.

## 11. Open questions

- **Q1: Does `gs.init` need backend override for CI?** CI workers are CPU-only. `gs.init(backend=gs.cpu)` is the explicit choice the adapter will make. No env-var override needed for CI.
- **Q2: What happens if two `GenesisTabletopEnv` instances are created in one process?** `gs.init` is process-global. Second instance's `gs.init` is a no-op. Test: create two envs sequentially, check both work. Expected: works (exploration §Q6 built two scenes in one process).
- **Q3: Can the `[genesis]` + `[hf]` + `[lerobot]` + `[pybullet]` all-on combo resolve?** Expected yes — torch pins line up (all `>=2.2,<3`), rich bump is the only core dep move. Documented as a check inside `genesis-tests` via `uv sync --extra genesis --extra hf --extra pybullet` dry-run step optional follow-up; not a gate for this RFC.

## 12. Implementation steps (atomic commits)

Mirrors RFC-005's step cadence to keep review easy. One commit per row, each passes `ruff check` + `ruff format --check` + `mypy --strict` (without the extras installed) + the narrow pytest slice it adds.

1. Exploration doc (`docs/phase2-exploration-task7-genesis-backend.md`). **Landed.**
2. RFC (this document).
3. `pyproject.toml`: add `[genesis]` extra, `genesis-dev` dev group, `genesis` pytest marker, `[[tool.mypy.overrides]]` for `genesis.*`, core `rich` ceiling bump `<15` → `<16`.
4. `.github/workflows/ci.yml`: add `genesis-tests` job; extend the default job's `-m "not ..."` exclusion to include `genesis`.
5. `src/gauntlet/env/genesis/__init__.py`: two-part import guard (genesis-world + torch), `register_env("tabletop-genesis", ...)`. No `GenesisTabletopEnv` body yet — empty-env stub that satisfies the Protocol just enough for `isinstance` to pass.
6. `src/gauntlet/env/genesis/tabletop_genesis.py`: scaffold — `__init__` with `gs.init` + `gs.Scene` construction + entity allocation + `scene.build`; `action_space`/`observation_space`; `close()`. Raises `NotImplementedError` on `step` / `reset` / `set_perturbation` / `restore_baseline` for now. Smoke test: instance constructs, exposes the right spaces and axis names.
7. `reset()` + `restore_baseline()` + `_build_obs` — state obs parity with `TabletopEnv`. No perturbations yet. Test: reset with a seed, 20-step identical-rollout determinism codified.
8. `step()` + reward + termination — full state-only rollout loop. Test: rollout a `RandomPolicy` for 50 steps, verify `step` returns the right shapes.
9. `set_perturbation` + `_validate_*` + pending-queue plumbing — axis validation only, no branches yet. Test: unknown-axis rejection, `distractor_count` range rejection.
10. The 7 perturbation branches — state-affecting first (`object_initial_pose_{x,y}`, `distractor_count`), cosmetic axes stored on self + declared `VISUAL_ONLY_AXES`. Test: pose + distractor state-visible, cosmetic axes leave obs unchanged.
11. Import-guard tests (`test_import_guards_genesis.py`) — patch-based, mirror `test_import_guards.py`'s PyBullet/hf/lerobot/monitor cases.
12. README snippet + example (`examples/evaluate_random_policy_genesis.py`) — mirrors `examples/evaluate_smolvla_pybullet.py` but for `RandomPolicy` on Genesis, CPU-only.

Each commit atomic, each passes lint/type/tests for its scope. One PR at the end (branch: `phase-2/genesis-backend`; base: `main`).

## 13. Decisions I'm deliberately making vs deferring

**Making now:**
- `genesis-world>=0.4,<0.5` tight pin (expected breaking 0.5 bump).
- `torch>=2.2,<3` explicit in `[genesis]`.
- `rich<16` core bump — documented, minimal blast radius.
- Teleport-away distractor semantics.
- State-only first cut; cosmetic axes `VISUAL_ONLY_AXES`-flagged.

**Deferring (follow-up RFC):**
- Image-obs path (`render_in_obs=True`), `scene.add_camera` wiring, cosmetic-axis mutation. RFC-008.
- GPU backend support — no code in this RFC rejects it, just isn't tested.
- Cross-process determinism proof — not gating.
- Genesis 0.5 upgrade — when it ships, revisit.
