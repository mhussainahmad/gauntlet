# Phase 2 RFC 009 — Isaac Sim backend adapter (state-only)

- **Status**: Draft
- **Phase**: 2, Task 9 (`GAUNTLET_SPEC.md` §7: "Additional simulators: Isaac Sim, Genesis, PyBullet adapters.")
- **Author**: implementation agent
- **Date**: 2026-04-24
- **Supersedes**: n/a
- **References**:
  - `docs/phase2-rfc-005-pybullet-adapter.md` (`GauntletEnv` Protocol + registry, state-only-first-cut template).
  - `docs/phase2-rfc-007-genesis-adapter.md` (most recent simulator-adapter RFC; this RFC mirrors its structure).
  - `docs/phase2-exploration-task9-isaac-sim-adapter.md` (the measurement pass that constrains §4 / §6 / §8).

---

## 1. Summary

RFC-005 introduced the backend-agnostic `GauntletEnv` Protocol +
`gauntlet.env.registry` and landed PyBullet as the second backend.
RFC-007 mirrored that staging for Genesis. This RFC adds the third:
**Isaac Sim**, registered as `"tabletop-isaac"` behind a new `[isaac]`
extra, satisfying the same `GauntletEnv` surface structurally — no
new protocol methods, no core-API changes.

The dominant constraint, surfaced in the exploration pass, is that
`isaacsim` (NVIDIA's PyPI distribution since 2024) requires CUDA + an
RTX-class GPU at runtime. The PyPI wheel resolves on the
`ubuntu-latest` GitHub runner the rest of the extras matrix uses, but
every meaningful import (`from isaacsim import SimulationApp` etc.)
segfaults or raises a Kit bootstrap error probing for a GPU.

This means the `[isaac]` extra cannot be tested end-to-end in CI on
the existing runner pool. We diverge from the RFC-005 / RFC-007
pattern in one place only: **tests live in `tests/isaac/`, run in the
default torch-free CI job, and use a `sys.modules`-injected fake
`isaacsim` namespace** to exercise the adapter contract without ever
running real Kit. A placeholder `isaac-tests` CI job exists for the
day a GPU runner is wired in but is `continue-on-error: true` and
silences pytest's exit-5 on zero collection.

State-only first cut: cosmetic axes (`lighting_intensity`,
`object_texture`, `camera_offset_{x,y}`) are part of `AXIS_NAMES`
(canonical 7) but declared `VISUAL_ONLY_AXES`. The suite loader's
existing `_reject_purely_visual_suites` guard fires for free off this
classvar — same shape Genesis had pre-RFC-008. Image rendering on
Isaac Sim is deferred to a follow-up RFC.

## 2. Goals / non-goals

### Goals

- Ship `gauntlet.env.isaac.tabletop_isaac.IsaacSimTabletopEnv`
  satisfying `GauntletEnv` (from RFC-005) structurally — no new
  protocol methods.
- All seven canonical perturbation axes accepted on
  `set_perturbation`; state-affecting axes
  (`object_initial_pose_{x,y}`, `distractor_count`) wire through to
  the underlying Isaac Sim prim state; cosmetic axes
  (`lighting_intensity`, `object_texture`, `camera_offset_{x,y}`)
  queue + validate + store on shadow attributes and are members of
  `VISUAL_ONLY_AXES` pending a follow-up rendering RFC.
- Same 7-D continuous action space and same five-key state-obs dict
  as `TabletopEnv` / `PyBulletTabletopEnv` / `GenesisTabletopEnv` (the
  state-only `render_in_obs=False` shape). Policies written for the
  other backends compose without code changes at the obs/action
  contract.
- Keep `gauntlet.core` Isaac-Sim-free: importing `gauntlet`,
  `gauntlet.env` (excluding the `isaac/` subpackage), `gauntlet.policy`,
  `gauntlet.runner`, `gauntlet.suite`, `gauntlet.report` must not
  transitively import `isaacsim` or `omni`.
- The `[isaac]` extra is lazily imported from inside
  `gauntlet.env.isaac.__init__` with a clear install-hint
  `ImportError` when `isaacsim` is absent.
- `mypy --strict` passes whether or not `[isaac]` is installed —
  mirrors the same `[[tool.mypy.overrides]]` pattern PyBullet and
  Genesis use, extended to `isaacsim`, `omni`, `omni.*`, `pxr`,
  `pxr.*`.
- `uv sync --extra isaac` enables the backend in one command on a
  developer's GPU workstation; composes cleanly with `--extra hf`,
  `--extra lerobot`, `--extra monitor`, `--extra pybullet`,
  `--extra genesis` at the pin-resolution level.

### Non-goals

- **Live `isaacsim` round-trip in CI.** Out of scope; needs a GPU
  runner (§8). The placeholder job is the wiring-only stub.
- **Image rendering on Isaac Sim** — cosmetic axes are state-only on
  this first cut; follow-up RFC will land `render_in_obs=True` mirror
  of RFC-006 / RFC-008. §11 sketches the shape.
- **Cross-backend numerical parity.** Same as RFC-005 §7.4 / RFC-007
  §7.3 — same policy + same seed on `tabletop` vs `tabletop-pybullet`
  vs `tabletop-genesis` vs `tabletop-isaac` produces different
  trajectories. Documented, not a bug.
- **URDF / USD robot arm + IK.** Same carve-out the other backends
  made — the EE is a kinematic prim teleported via `set_world_pose`
  every control step; no arm asset.
- **New perturbation axes.** The canonical 7 are the contract.
- **Replacing the live `pip install isaacsim` + GPU-runner CI as a
  follow-up.** Worth doing, but unrelated to the adapter shape —
  flagged in §11.
- **Real-physics validation against MuJoCo.** Mock-based tests cannot
  do this; documented in §8.2 as a "GPU-workstation manual smoke" not
  a CI gate.

## 3. Protocol / registry reuse

No protocol changes. `GauntletEnv` (a `runtime_checkable` Protocol in
`src/gauntlet/env/base.py`) is the contract; `IsaacSimTabletopEnv`
satisfies it structurally — same `AXIS_NAMES: ClassVar[frozenset[str]]`,
same `observation_space` / `action_space`, same `reset` / `step` /
`set_perturbation` / `restore_baseline` / `close` surface.

Registration is a two-line call inside `gauntlet.env.isaac.__init__`
under the key `"tabletop-isaac"`, mirroring the
`tabletop-pybullet` / `tabletop-genesis` registrations. Two existing
modules grow a one-line entry to wire the lazy-import + install-hint
path:

* `src/gauntlet/suite/schema.py` — `BUILTIN_BACKEND_IMPORTS["tabletop-isaac"] = "gauntlet.env.isaac"`.
* `src/gauntlet/suite/loader.py` — `_EXTRA_FOR_MODULE["gauntlet.env.isaac"] = "isaac"`.

The cosmetic-only-sweep rejection in
`gauntlet.suite.loader._reject_purely_visual_suites` reads
`VISUAL_ONLY_AXES` off the registered factory; because
`IsaacSimTabletopEnv.VISUAL_ONLY_AXES` is non-empty on this first
cut, the rejection happens for free. No loader code changes.

## 4. `[isaac]` extra — pin + dep-tree decisions

### 4.1 Extras definition

```toml
[project.optional-dependencies]
# Independent extra for the Isaac Sim backend (RFC-009).
# isaacsim wraps NVIDIA Omniverse Kit and requires a CUDA-capable
# RTX-class GPU at runtime; the PyPI wheel resolves on CPU-only
# machines but every meaningful import probes for the GPU and fails.
# So `uv sync --extra isaac` is intended for a developer's GPU
# workstation, not the CI runner. CI tests use a sys.modules-injected
# fake `isaacsim` namespace and do NOT install this extra (RFC-009 §8).
isaac = [
    "isaacsim>=5.0,<6",
]

[dependency-groups]
isaac-dev = [
    {include-group = "dev"},
    "pytest-mock>=3.12,<4",
]
```

### 4.2 Pin rationale

- `isaacsim>=5.0,<6` — the 4.x line is **cp310-only** on the NVIDIA
  index, so it cannot satisfy the project's `requires-python = ">=3.11"`
  floor. 5.0 is the first release that publishes cp311 wheels; 6.x
  jumped to cp312-only and is incompatible with 3.11. The one-major
  ceiling matches `[hf]` (`<5`), `[lerobot]` (`<1`), `[pybullet]`
  (`<4`), `[genesis]` (`<0.5`).
- No `torch` in `[isaac]`. Isaac Sim ships its own torch through the
  `isaacsim-core` transitive dep; we do not need to declare it.
- No `numpy` / `gymnasium` in `[isaac]` — both already pinned in core.

### 4.2a NVIDIA package index — `tool.uv` configuration

The PyPI placeholders for `isaacsim` and its `isaacsim-*` /
`omniverse-kit` transitives are wheel-stub shims that download the
actual binaries from NVIDIA's index at install time. uv's resolver
needs to see the NVIDIA index too, otherwise `uv lock` fails trying to
build the placeholder. So `pyproject.toml` declares:

```toml
[[tool.uv.index]]
name = "nvidia"
url = "https://pypi.nvidia.com/"
explicit = true

[tool.uv.sources]
isaacsim = { index = "nvidia" }
# ...one entry per isaacsim-* + omniverse-kit transitive
```

The `explicit = true` flag prevents the NVIDIA index from leaking into
the resolution of unrelated packages — only packages explicitly
sourced via `tool.uv.sources` consult that index. CI builds that do
NOT install the `[isaac]` extra are unaffected (the index is named
but never consulted because no listed package is needed).

### 4.3 Core-dep ceilings — no changes needed

Unlike RFC-007 (which had to bump `rich<15` to `<16` to admit
genesis-world's transitive `rich`), `isaacsim` does not push any
core dep over a current ceiling. Verified via `pip install --dry-run
isaacsim` against the current `pyproject.toml`: every overlap fits
under existing ceilings. No core-dep churn.

### 4.4 Size / install footprint

`isaacsim` itself is a thin shim (~1 MB) but pulls
`isaacsim-kernel`, `isaacsim-core`, `isaacsim-app`, `isaacsim-asset`,
... — full install is ~5-15 GB once Kit binaries and the asset cache
land. Comparable to `[lerobot]` (~4 GB) at the small end and an order
of magnitude heavier than `[pybullet]` (~50 MB) at the large end. CI
never installs this extra so the cost lands only on developer
workstations.

### 4.5 Python support matrix

| Python | Wheel published | CI strategy |
|---|---|---|
| 3.11 | yes | default-job mocked tests; placeholder `isaac-tests` job tries `uv sync --extra isaac` (allowed to fail) |
| 3.12 | yes (Isaac Sim 4.5+) | same |
| 3.13 | not yet | N/A |

### 4.6 Interaction with the Runner's `spawn` start method

When the extra IS installed (developer GPU workstation), `isaacsim`
initialises a process-global `SimulationApp` singleton at first
import. Each Runner worker is a fresh Python interpreter under spawn,
so the singleton is constructed fresh per worker — same shape Genesis
took with `gs.init`. The adapter's `__init__` does the bootstrap
inside the worker process so the cost (multi-second Kit boot) is paid
once per worker and amortised across every episode that worker
handles.

When the extra is NOT installed (CI / mocked tests), the bootstrap
never runs because the fake `isaacsim` namespace shortcuts every
call.

## 5. Scene layout

Mirrors `GenesisTabletopEnv` (RFC-007 §5) at the entity inventory
level, adapted to Omniverse USD prims via `omni.isaac.core` factories:

```
World (omni.isaac.core)
├── DefaultGroundPlane                              # ground, fixed
├── FixedCuboid (table-top, 1.0x1.0x0.04)
├── DynamicCuboid (cube, movable, 0.05^3)
├── VisualCuboid  (EE, kinematic-by-teleport, 0.02^3)
├── VisualCuboid  (target marker, 0.05 radius equivalent)
└── FixedCuboid x 10 (distractors, pre-allocated, teleported on/off)
```

Key shape decisions vs other backends:

- **No URDF / USD robot asset.** Same trade Genesis (RFC-007 §5) and
  PyBullet (RFC-005 §7.1) made — the EE is a small kinematic
  `VisualCuboid` whose pose is overwritten via `set_world_pose` each
  control step. Avoids shipping a Franka USD asset in the repo and
  keeps Kit construction cheap.
- **Distractor pre-allocation + teleport-away semantic.** Same
  approach as Genesis. `omni.isaac.core` doesn't expose a stable
  per-prim visibility-toggle that survives reset cleanly; teleport to
  `z = -10.0` is the cleanest cross-backend semantic.
- **Two cubes pre-allocated for `object_texture`** is deferred to the
  follow-up rendering RFC; the state-only first cut uses a single
  cube and stores `object_texture` on a shadow attribute only.

### 5.1 Physics configuration

- World physics dt: Isaac Sim default (`1/60 s`), with `n_substeps=5`
  matching MuJoCo / PyBullet / Genesis for a 12 Hz outer loop.
- Gravity: `(0.0, 0.0, -9.81)` — Isaac Sim default; kept.
- Solver: PhysX default. Different from MuJoCo's MJ solver and from
  PyBullet's iterative solver — one of the numerical-parity deltas
  (§7.3).

## 6. The 7 axes on Isaac Sim

Same five-row split RFC-007 §6 used. Documented upfront, implemented
in step 6 (one commit).

### 6.1 `lighting_intensity` — cosmetic, deferred

Lives on USD `UsdLuxDistantLight.intensityAttr` once a render path
exists. State-only first cut: queues + validates + stores on
`self._light_intensity`; member of `VISUAL_ONLY_AXES`. Follow-up
rendering RFC will wire to the prim attribute.

### 6.2 `camera_offset_x`, `camera_offset_y` — cosmetic, deferred

Lives on `UsdGeomCamera.set_world_pose` once a render path exists.
State-only first cut: queues + validates + stores on
`self._cam_offset` (`np.zeros(2)`); member of `VISUAL_ONLY_AXES`.

### 6.3 `object_texture` — cosmetic, deferred

Lives on `UsdShade.MaterialBindingAPI` in the eventual rendering
RFC. State-only first cut: queues + validates + stores on
`self._texture_choice`; member of `VISUAL_ONLY_AXES`.

### 6.4 `object_initial_pose_x`, `object_initial_pose_y` — state-affecting, full parity

Direct: after `world.reset()` and the seed-driven cube XY randomisation,
call `self._cube.set_world_pose(position=np.array([value, y, z]),
orientation=...)` for X (or `[x, value, z]` for Y). Same semantic as
the other backends.

### 6.5 `distractor_count` — state-affecting, teleport-away semantic

10 distractors pre-allocated in `__init__`. When `count == N`
(0 ≤ N ≤ 10), the first N teleport to their rest XY at the
table-top Z; the remaining 10 - N teleport to `z = -10.0`. Same
shape Genesis (§6.5) and PyBullet use. Value validation
(`distractor_count` axis): `round(value) ∈ [0, 10]`.

### 6.6 Summary table

| Axis | Category on Isaac Sim | `VISUAL_ONLY_AXES` | Follow-up RFC |
|---|---|---|---|
| `lighting_intensity` | cosmetic | yes | rendering |
| `camera_offset_x` | cosmetic | yes | rendering |
| `camera_offset_y` | cosmetic | yes | rendering |
| `object_texture` | cosmetic | yes | rendering |
| `object_initial_pose_x` | state-affecting | no | — |
| `object_initial_pose_y` | state-affecting | no | — |
| `distractor_count` | state-affecting | no | — |

Three of seven axes produce state-observable effects on this first
cut. The four cosmetic axes' branches execute (storing values on
shadow attributes), are reachable via `set_perturbation`, and validate
input — but produce no obs-dict delta until rendering lands. The
existing `_reject_purely_visual_suites` guard fires off this set
without code changes.

### 6.7 `AXIS_NAMES`

```python
AXIS_NAMES: ClassVar[frozenset[str]] = frozenset({
    "lighting_intensity", "camera_offset_x", "camera_offset_y",
    "object_texture", "object_initial_pose_x", "object_initial_pose_y",
    "distractor_count",
})
```

Byte-identical to every other backend's `AXIS_NAMES`. Canonical 7.

## 7. Determinism, parity, and the "Isaac-Sim-Isaac-Sim-only" compare contract

### 7.1 Same-process determinism

Under the mocked tests, `reset(seed=s)` twice from a fresh adapter
produces byte-identical observations because the mocks are
deterministic and the adapter's seed-driven RNG is
`np.random.default_rng(seed)`. Test
`test_reset_seed_determinism_under_mock` codifies this.

Under real Isaac Sim (developer GPU workstation), determinism depends
on PhysX configuration and is not part of this RFC's CI gate. The
real-runtime determinism property is best-effort and has the same
"single process, single seed" caveat the other simulator-adapter RFCs
made.

### 7.2 Cross-process determinism — explicit non-goal

Same caveat as RFC-005 §7 / RFC-007 §7.2. Worker-subprocess
determinism is guaranteed at the Python level (episode seeds are
hashed from master + cell + ep); the simulator's cross-process
determinism is best-effort.

### 7.3 Cross-backend numerical non-parity

Same policy + same seed on `tabletop` vs `tabletop-pybullet` vs
`tabletop-genesis` vs `tabletop-isaac` produces four different
trajectories. All four are internally semantically plausible; none
match the others at the float level. `gauntlet compare` across
backends measures simulator drift, not policy regression. Same
wording as the other adapter RFCs.

### 7.4 Success criterion

Same as the other backends: cube center within `TARGET_RADIUS = 0.05 m`
of `target_pos` for one timestep. Termination semantics identical.

### 7.5 Quaternion convention

Isaac Sim's `omni.isaac.core` `prim.get_world_pose()` returns
`(position: np.ndarray (3,), orientation: np.ndarray (4,))` with
orientation in **wxyz** order (Isaac Sim convention since 4.x).
Matches MuJoCo and Genesis; no conversion helper needed. (PyBullet's
xyzw stays the odd one out.)

## 8. Test matrix

All new tests live in `tests/isaac/`. **They are NOT marked
`@pytest.mark.isaac`** — that marker is reserved for the
currently-empty bucket of real-runtime tests that will live behind a
GPU runner. The mocked tests run in the **default** torch-free CI
job; this is the deliberate divergence from the RFC-005/007 pattern
called out in §1.

The fake `isaacsim` namespace is installed via a module-scoped
autouse fixture in `tests/isaac/conftest.py` so every test file
inherits a consistent surface. The fixture pokes
`sys.modules["isaacsim"]`, `sys.modules["isaacsim.core"]`,
`sys.modules["isaacsim.core.api"]`, `sys.modules["isaacsim.core.api.objects"]`,
`sys.modules["omni"]`, `sys.modules["omni.isaac"]`,
`sys.modules["omni.isaac.core"]` with stubs whose factories return
objects with the methods the adapter calls (`reset`, `step`,
`get_world_pose`, `set_world_pose`, `add`).

| Test file | Verifies |
|---|---|
| `test_import_guards_isaac.py` | Without the fake namespace AND with `isaacsim` absent, importing `gauntlet.env.isaac` raises `ImportError` whose message names `uv sync --extra isaac`. |
| `test_env_isaac.py::test_protocol_conformance` | `IsaacSimTabletopEnv` is an instance of `GauntletEnv` via `isinstance`. |
| `test_env_isaac.py::test_spaces_parity_with_genesis` | Same `action_space` shape/dtype/bounds and same five-key state-obs dict as `GenesisTabletopEnv(render_in_obs=False)`. |
| `test_env_isaac.py::test_axis_names_are_canonical_seven` | `AXIS_NAMES` matches the canonical 7. |
| `test_env_isaac.py::test_visual_only_axes_are_four_cosmetic` | `VISUAL_ONLY_AXES` is exactly the four cosmetic axes. |
| `test_env_isaac.py::test_reset_seed_determinism_under_mock` | `env.reset(seed=42)` twice + 5 mocked steps → byte-identical obs. |
| `test_env_isaac.py::test_reset_returns_expected_shapes_and_dtypes` | Shapes / dtypes / info keys match Genesis. |
| `test_env_isaac.py::test_step_contract` | `step` returns `(obs, reward, terminated, truncated, info)` with the right Python types. |
| `test_env_isaac.py::test_set_perturbation_rejects_unknown_axis` | Unknown axis → `ValueError` naming the axis. |
| `test_env_isaac.py::test_set_perturbation_rejects_out_of_range_distractor_count` | Integer `[0, 10]` bound. |
| `test_env_isaac.py::test_set_perturbation_accepts_every_known_axis` | All seven canonical axes accepted. |
| `test_env_isaac.py::test_object_initial_pose_x_calls_cube_set_world_pose` | The state-affecting axis lands on the right mocked prim with the right value. |
| `test_env_isaac.py::test_object_initial_pose_y_calls_cube_set_world_pose` | Counterpart for Y. |
| `test_env_isaac.py::test_distractor_count_teleport_semantics` | Mocked distractor prims see the right `set_world_pose` calls. |
| `test_env_isaac.py::test_cosmetic_axes_store_on_shadows_only` | Each `VISUAL_ONLY` axis stores on its shadow attribute and does NOT call any prim. |
| `test_env_isaac.py::test_restore_baseline_re_hides_all_distractors` | Post-`restore_baseline`, every distractor mocked prim has been told to teleport to `z = -10.0`. |
| `test_env_isaac.py::test_close_is_idempotent` | `close()` twice doesn't raise. |
| `test_suite_loader_isaac.py::test_loader_accepts_state_only_isaac_sweep` | A YAML naming `env: tabletop-isaac` with state-only axes loads. |
| `test_suite_loader_isaac.py::test_loader_rejects_cosmetic_only_isaac_sweep` | A YAML with only cosmetic axes is rejected with a clear message naming the axes. |
| `test_suite_loader_isaac.py::test_loader_accepts_mixed_axes_isaac_sweep` | A YAML mixing cosmetic + state axes loads. |

No rendering tests — out of scope. No real-Isaac-Sim round-trip tests
— gated to a future GPU-runner CI follow-up.

### 8.2 Manual GPU-workstation smoke (documented, not a CI gate)

A contributor with a GPU workstation can validate the adapter against
real Isaac Sim by:

```
uv sync --extra isaac --group isaac-dev
uv run python examples/evaluate_random_policy_isaac.py
```

The example mirrors `examples/evaluate_random_policy_genesis.py` and
runs `RandomPolicy` against the bundled
`examples/suites/tabletop-isaac-smoke.yaml`. A failure here surfaces
real-API drift the mocks couldn't catch.

## 9. CI footprint

New `isaac-tests` job in `.github/workflows/ci.yml`. Pattern-copied
from `genesis-tests` with two deviations:

1. **`continue-on-error: true`** at the job level so a failing
   `uv sync --extra isaac` step does not block PR merge.
2. **`pytest -m isaac --co -q || [ $? -eq 5 ]`** so the test step
   succeeds on zero-collection (no tests carry the `isaac` marker in
   this PR; the marker is reserved for the future real-runtime
   bucket).

```yaml
isaac-tests:
  name: isaac backend tests (py${{ matrix.python-version }})
  runs-on: ubuntu-latest
  continue-on-error: true
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
    - name: Sync dependencies (isaac extra + isaac-dev group)
      run: >-
        uv sync
        --extra isaac
        --group isaac-dev
        --python ${{ matrix.python-version }}
      continue-on-error: true
    - name: Run isaac-marked tests (zero-collect tolerated)
      run: |
        uv run pytest -m isaac --co -q || [ $? -eq 5 ]
      continue-on-error: true
```

Default `lint-typecheck-test` job's `pytest` `-m` exclusion grows by
`and not isaac`. Mocked tests in `tests/isaac/` are unmarked so they
DO run in the default job — that is the whole reason for the
divergence from RFC-005/007.

## 10. Open questions

- **Q1: Should the placeholder `isaac-tests` job run on every PR or
  only `workflow_dispatch`?** Default: every PR with
  `continue-on-error: true`. Trivial to flip later; the wiring stays
  visible.
- **Q2: When a GPU runner lands, does `isaac-tests` move from
  `ubuntu-latest` to that runner?** Yes. The `runs-on:` line is the
  only delta; the steps are already shaped for the real-install path.
- **Q3: Does the mock conftest leak into other test files (`tests/`)?**
  No — fixture is scoped to `tests/isaac/` only, and uses
  `monkeypatch.setitem` so `sys.modules` is restored at fixture
  teardown.
- **Q4: When does `VISUAL_ONLY_AXES` drop to `frozenset()` on
  `IsaacSimTabletopEnv`?** When the rendering follow-up RFC lands —
  same shape Genesis went through (RFC-007 → RFC-008) and PyBullet
  before that (RFC-005 → RFC-006).

## 11. Future work (rendering RFC sketch)

A follow-up RFC (call it RFC-010 for now) will add `render_in_obs=True` /
`render_size=(H, W)` to `IsaacSimTabletopEnv`, wiring the
`SimulationApp` `headless=True, renderer="RaytracedLighting"`
bootstrap, an `omni.replicator.core` camera capture pipeline, and the
four cosmetic axes' prim-attribute mutations. Empties
`VISUAL_ONLY_AXES`. Mirrors the staging RFC-008 used for Genesis. Will
need a GPU runner CI job to test for real; until that runner exists,
that follow-up is also mock-driven (with a richer fake surface).

A separate follow-up should add a self-hosted GPU GitHub Actions
runner to the project's workflow infrastructure so the placeholder
job can flip to live execution and start catching real API drift.

## 12. Implementation steps (atomic commits)

Mirrors RFC-007 §12 cadence to keep review easy. One commit per row.
Each commit passes `ruff check` + `ruff format --check` +
`mypy --strict` (without the extra installed) + the narrow pytest
slice it adds.

1. Exploration doc (`docs/phase2-exploration-task9-isaac-sim-adapter.md`). **Landed.**
2. RFC (this document).
3. `pyproject.toml`: `[isaac]` extra (`isaacsim>=5.0,<6`), `isaac-dev`
   dev group, `isaac` pytest marker, `[[tool.mypy.overrides]]` for
   `isaacsim`, `omni.*`, `pxr.*`. No code yet.
4. `src/gauntlet/env/isaac/__init__.py` (lazy guard +
   `register_env`), `src/gauntlet/env/isaac/tabletop_isaac.py`
   scaffold (constants + class skeleton + spaces +
   `AXIS_NAMES` / `VISUAL_ONLY_AXES`). `src/gauntlet/suite/schema.py`
   gets `BUILTIN_BACKEND_IMPORTS["tabletop-isaac"] = ...`.
   `src/gauntlet/suite/loader.py` gets `_EXTRA_FOR_MODULE` entry.
   `tests/isaac/__init__.py` empty.
   `tests/isaac/conftest.py` with the fake-`isaacsim` autouse fixture.
   `tests/isaac/test_import_guards_isaac.py` covering
   ImportError-without-extra.
5. State-obs `IsaacSimTabletopEnv` body — full reset/step/`_build_obs`
   under the mocked surface. Tests in `tests/isaac/test_env_isaac.py`:
   protocol conformance, space parity, axis-names, deterministic
   reset, step contract.
6. `set_perturbation` + `restore_baseline` — all 7 axes, four flagged
   `VISUAL_ONLY_AXES`. Tests cover per-axis mock-call assertions and
   `restore_baseline` re-hides distractors.
7. Suite loader integration — `tests/isaac/test_suite_loader_isaac.py`
   asserting state-only sweep loads, cosmetic-only sweep rejected.
8. CI job `isaac-tests` in `.github/workflows/ci.yml` +
   default-job exclusion grows by `and not isaac`.
9. README backends section — append `tabletop-isaac` row. Mention the
   "GPU required for real install" caveat.
10. Example `examples/evaluate_random_policy_isaac.py` mirroring
    `examples/evaluate_random_policy_genesis.py`. Imports cleanly when
    `isaacsim` is absent (no top-level `import isaacsim`).

## 13. Decisions I'm deliberately making vs deferring

**Making now:**

- `isaacsim>=5.0,<6` pin (one-major ceiling; matches the cadence of
  the other extras).
- State-only first cut; cosmetic axes flagged `VISUAL_ONLY_AXES`.
- Mock-driven CI tests in `tests/isaac/` running in the default job
  (DIVERGES from RFC-005/007's marker-gated pattern; explained in §1
  and §8).
- Placeholder `isaac-tests` job with `continue-on-error: true`.
- Same teleport-away distractor semantic as Genesis.

**Deferring (follow-up RFC):**

- Image-obs path (`render_in_obs=True`), camera + lighting +
  material wiring. RFC-010-ish.
- GPU-runner CI for real-Isaac-Sim execution. Workflow-infrastructure
  follow-up.
- `isaacsim 5.x` upgrade — when it ships, revisit pin.
- Cross-process determinism proof — not gating; matches the staging
  the other adapter RFCs took.
