# Phase 2 RFC 005 — PyBullet adapter + `GauntletEnv` protocol extraction

- **Status**: Draft
- **Phase**: 2, Task 5 (`GAUNTLET_SPEC.md` §7: "Additional simulators: Isaac Sim, Genesis, PyBullet adapters.")
- **Author**: architect agent
- **Date**: 2026-04-22
- **Supersedes**: n/a
- **References**:
  - `docs/phase2-rfc-001-huggingface-policy.md` (`[hf]` extras-group pattern, torch-free core rule, lazy-import-with-install-hint template).
  - `docs/phase2-rfc-002-lerobot-smolvla.md` (one extra per capability; independent version pins).
  - `docs/phase2-rfc-003-drift-detector.md` (`[monitor]` extra precedent; "core-import stays cheap" invariant).
  - `docs/phase2-exploration-task5-pybullet-env-registry.md` (public-surface inventory of `TabletopEnv` this RFC builds on).

---

## 1. Summary

`GAUNTLET_SPEC.md` §7 lumps three new simulators into one Phase 2 item — Isaac Sim, Genesis, PyBullet. Two of the three are hostile to a clean first cut: Isaac Sim needs a proprietary NVIDIA stack (Omniverse / Kit, GPU-only, licensing friction) and Genesis is both heavy and its public API is still churning. PyBullet is the pragmatic first non-MuJoCo backend — CPU-only, PyPI-distributed, stable API, deterministic solver — so this RFC lands **PyBullet first** and extracts the shared plumbing (a `GauntletEnv` Protocol plus a registry) so Isaac Sim and Genesis can land as sibling RFCs (RFC-006, RFC-007) without a second refactor.

Concretely this RFC: (a) defines `GauntletEnv` as a `runtime_checkable` Protocol living in `gauntlet.env.base`, (b) introduces `gauntlet.env.registry` with `register_env` / `get_env_factory`, (c) registers `"tabletop"` as a built-in backed by the existing `TabletopEnv`, (d) adds `"tabletop-pybullet"` behind a new `[pybullet]` extra mirroring the RFC-001/002/003 pattern, and (e) ships state-only observation parity — rendered pixels on PyBullet are deferred to a follow-up RFC. The `Policy` protocol, `Episode` / `Report` schemas, and the Runner's bit-determinism contract are all unchanged. Core code changes are minimal but non-zero: `suite/schema.py::SUPPORTED_ENVS`, `suite/loader.py`, and `runner/worker.py` all rewire to the registry and to the Protocol (so they no longer name `TabletopEnv` directly).

One honesty caveat is foregrounded in §7: the two purely-visual axes (`lighting_intensity`, `object_texture`) mutate the PyBullet scene but produce byte-identical observations on a state-only obs dict. The branches are implemented on day one (parity in *application*), but observable effect on success rates is zero until the image-rendering follow-up RFC lands. A sweep that varies only those two axes will show every cell identical; the RFC proposes rejecting such suites at load time with a clear error (§12). A second honesty caveat: a given seed on `tabletop-pybullet` produces trajectories semantically similar but numerically different from the same seed on `tabletop` — cross-backend `gauntlet compare` measures backend drift, not policy regression (§8).

## 2. Goals / non-goals

### Goals

- Ship `gauntlet.env.base.GauntletEnv` — a `runtime_checkable` Protocol that the Runner and Suite-loader dispatch through, so the harness is simulator-agnostic end-to-end.
- Ship `gauntlet.env.registry` with `register_env(name, factory)` / `get_env_factory(name)` / `registered_envs()`. Built-ins register at import time; third-party backends register via an explicit `register_env` call inside their own `__init__.py` (guarded by the matching ImportError — same pattern as `[hf]`/`[lerobot]`/`[monitor]`).
- Ship `gauntlet.env.pybullet.tabletop_pybullet.PyBulletTabletopEnv` — a state-only PyBullet backend that satisfies `GauntletEnv`, supports all seven canonical perturbation axes at the `set_perturbation` level, and drives a floating mocap-analogue end-effector via a fixed constraint (§5).
- Keep `gauntlet.core` pybullet-free: importing `gauntlet`, `gauntlet.env` (excluding the `pybullet/` subpackage), `gauntlet.policy`, `gauntlet.runner`, `gauntlet.suite`, `gauntlet.report` must not transitively import `pybullet`. The `[pybullet]` extra is lazily imported inside `PyBulletTabletopEnv.__init__` with a clear install-hint ImportError. Mirrors §6 of the spec.
- Keep the `Policy` protocol, `ResettablePolicy`, `Episode`, `Report`, and the Runner's determinism contract unchanged. Cross-module blast radius limited to the three files named in §1 plus the `env/__init__.py` that does registration.
- Keep `mypy --strict` passing whether or not the `[pybullet]` extra is installed.
- `uv sync --extra pybullet` — one command to enable. Composes with `--extra hf`, `--extra lerobot`, `--extra monitor`.
- Preserve bit-determinism within a backend: a given `(backend, master_seed, cell, ep)` produces bit-identical episode outputs on PyBullet the same way it does on MuJoCo (§8).

### Non-goals

- **Isaac Sim support.** Requires Omniverse/Kit, GPU-only, proprietary licensing. Deferred to **RFC-006** (§14 appendix sketches the shape).
- **Genesis support.** API is still in flux as of April 2026 (rapid `pip` churn, breaking changes between minor versions). Deferred to **RFC-007** (§14 appendix sketches the shape).
- **Image rendering for VLA adapters on PyBullet.** `TabletopEnv(render_in_obs=True, render_size=(H, W))` landed in RFC-001 Task 1 for MuJoCo. Re-implementing it on PyBullet (a tuned off-screen `getCameraImage` pipeline, deterministic framebuffer, matched camera intrinsics to keep vision checkpoints evaluable) is its own body of work; split into a follow-up RFC. The first PyBullet release ships state-only.
- **Full behavioural parity on the two purely-visual axes.** `lighting_intensity` and `object_texture` mutate the scene but do not change state-only observations. Branches are implemented; the observable effect is zero on state-only obs. See §7 "visual-axis honesty caveat" and §12 open question Q1 for the load-time-rejection default.
- **Cross-backend numerical parity.** Same policy + same seed on `tabletop` vs `tabletop-pybullet` does **not** produce the same trajectory (different solvers, different contact models, different integrator step semantics). See §8.
- **New perturbation axes.** The canonical 7 are the contract; no axis additions in this RFC.
- **Third-party backend plugin discovery via `entry_points`.** Import-time `register_env` from a first-party subpackage is simpler and sufficient for the three-backend near-term roadmap. An `entry_points` discovery layer is a straightforward future addition once an external team ships a fourth backend (footnoted in §3).
- **A URDF robot arm + IK solver.** The MuJoCo env uses a mocap body — no kinematic chain. PyBullet gets the same abstraction via a fixed constraint (§5). No arm asset, no IK dependency.

## 3. `GauntletEnv` protocol + registry design

### 3.1 Does `TabletopEnv` already satisfy `GauntletEnv`?

**Almost — one additive change.** The current surface has everything the Protocol needs (`observation_space`, `action_space`, `reset`, `step`, `set_perturbation`, `restore_baseline`, `close`), and the method signatures line up exactly. The one missing piece is a **class-level** `AXIS_NAMES: ClassVar[frozenset[str]]` declaring which axes the backend supports. Today `TabletopEnv` carries that set as a *module-private* constant (`_KNOWN_AXIS_NAMES` on line 78 of `env/tabletop.py`), not a class attribute — so the Suite loader can't introspect "does the selected backend support axis X?" without hard-coding the mapping.

Additive fix: promote `_KNOWN_AXIS_NAMES` to `TabletopEnv.AXIS_NAMES: ClassVar[frozenset[str]] = frozenset(AXIS_NAMES)`, re-point the existing check in `set_perturbation` at `type(self).AXIS_NAMES`, and drop the module-private alias. Zero behavioural change. `PyBulletTabletopEnv.AXIS_NAMES` is the same frozenset (the whole point is parity).

### 3.2 The Protocol

Proposed surface (in `gauntlet.env.base`, mirroring the `Policy` Protocol pattern from `gauntlet.policy.base`):

```python
@runtime_checkable
class GauntletEnv(Protocol):
    AXIS_NAMES: ClassVar[frozenset[str]]

    observation_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]: ...

    def step(
        self, action: Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]: ...

    def set_perturbation(self, name: str, value: float) -> None: ...
    def restore_baseline(self) -> None: ...
    def close(self) -> None: ...
```

`render` is intentionally **not** on the Protocol: state-only envs (first PyBullet cut) don't need it, and the image-rendering follow-up RFC will decide whether to promote it to the Protocol or gate it on a separate `RenderableGauntletEnv` sub-Protocol.

### 3.3 Behavioural contract (semantic, not implementation-tied)

Spelled out in the Protocol's class docstring so backends don't overfit to MuJoCo's in-place mutation model or PyBullet's `resetSimulation` idiom:

- `set_perturbation(name, value)` **queues** a named scalar — it does not take effect immediately. Raises `ValueError` for `name not in type(self).AXIS_NAMES`. Same validation rules as `TabletopEnv` (e.g. `distractor_count` integer bound) apply per-backend.
- `restore_baseline()` makes the env observationally equivalent to its post-`__init__` state (not bit-identical internal C state — that's a MuJoCo-only property). Called by the Runner between episodes (see `runner/worker.py::_execute_one`). Does **not** clear the queued perturbations — that is the input to the next reset.
- `reset(seed=...)` MUST (in order): (1) call `restore_baseline()` internally, (2) apply seed-driven state randomisation, (3) apply every queued perturbation on top, (4) clear the pending-perturbation queue. This is the ordering the Runner already relies on (`TabletopEnv.reset` implements it on lines 397–457) and the PyBullet backend must replicate — reversing steps 2 and 3 would silently change determinism semantics.
- `seed` is the **only** entropy source for a reset. Pass-through of `gymnasium.Env.reset`'s standard contract.

### 3.4 The registry

`gauntlet.env.registry`:

```python
_REGISTRY: dict[str, Callable[..., GauntletEnv]] = {}

def register_env(name: str, factory: Callable[..., GauntletEnv]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"env {name!r} already registered")
    _REGISTRY[name] = factory

def get_env_factory(name: str) -> Callable[..., GauntletEnv]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown env {name!r}; registered: {sorted(_REGISTRY)}"
        ) from exc

def registered_envs() -> frozenset[str]: ...
```

Built-in registration happens in `gauntlet.env.__init__`:

```python
# src/gauntlet/env/__init__.py
from gauntlet.env.registry import register_env
from gauntlet.env.tabletop import TabletopEnv

register_env("tabletop", TabletopEnv)

# PyBullet registers itself — NOT imported here; its __init__ does the work
# behind the [pybullet] extra's ImportError guard. Users opt in via:
#     from gauntlet.env import pybullet  # noqa: F401  -- registers on import
# and the Suite loader does this exact import when env == "tabletop-pybullet"
# (see §11).
```

### 3.5 Registration mechanism — import-time, not `entry_points`

**Decision: explicit `register_env` call from the backend's own `__init__.py`, triggered by an import from the Suite loader.**

Why not `entry_points` / `setup.cfg`:

- **No current third-party backend.** All three near-term backends (PyBullet, Isaac Sim, Genesis) are first-party and ship in this repo behind extras. Import-time registration is one file per backend; `entry_points` adds a `pkg_resources` discovery pass on every CLI invocation.
- **Mirrors every other Phase 2 adapter.** RFC-001/002/003 all chose "lazy-import from a subpackage on demand, behind an ImportError with a clear install hint." Keeping the env-plugin path consistent reduces the cognitive overhead of the codebase.
- **Deferred, not foreclosed.** An `entry_points` group `"gauntlet.envs"` can be added later (strictly additive — `registered_envs()` can union the two sources) without re-litigating this RFC.

The concrete flow: when `Suite.env == "tabletop-pybullet"` the loader does `import gauntlet.env.pybullet` which runs `register_env("tabletop-pybullet", PyBulletTabletopEnv)`. The import guard inside `gauntlet.env.pybullet.__init__` wraps the `import pybullet` call and raises a clean ImportError with `uv sync --extra pybullet` in the message if the extra isn't installed — same pattern as `HuggingFacePolicy`.

## 4. `[pybullet]` extras placement + `pyproject.toml` diff

### 4.1 Extras decision: new `[pybullet]` extra, confirmed

Straight application of the RFC-001/002/003 precedent:

- **(A) New `[pybullet]` extra.** Users who only run MuJoCo skip it; users who want both backends `uv sync --extra pybullet`; composes with every other extra. **Chosen.**
- **(B) Core dep.** Rejected — violates §6 ("small deps"), roughly doubles cold-install size, and the first-cut PyBullet env is state-only (no pixels) so it's entirely optional on day one.
- **(C) Separate distribution `gauntlet-pybullet`.** Rejected on the same grounds RFC-001 §3 / RFC-002 §3 / RFC-003 §3 already rejected this pattern — premature, adds release plumbing before the first external user.
- **(D) Fold into an umbrella `[sim]` extra alongside future Isaac/Genesis.** Rejected — same version-pin-independence argument as RFC-002 §3. PyBullet's release cadence is very different from Isaac Sim's (quarterly vs Omniverse-tied) and Genesis's (weekly-ish on `main`). One extra per backend.

### 4.2 `pyproject.toml` diff fragments

```toml
# EXISTING from RFC-001/002/003 — unchanged.
[project.optional-dependencies]
hf       = [ ... ]
lerobot  = [ ... ]
monitor  = [ ... ]

# NEW — independent extra for the PyBullet backend.
# pybullet is a pure-Python/C++ package on PyPI, CPU-only, no CUDA needed,
# wheels on manylinux/macOS/Windows. The 3.2 floor matches the first release
# with the stable constraint + `getPhysicsEngineParameters` API surface
# this RFC depends on (see §8 determinism knobs). Ceiling one major above
# matches the RFC-001 convention for catching surprise upstream breakage.
pybullet = [
    "pybullet>=3.2,<4",
]

# NEW — dev-group analogue of hf-dev / lerobot-dev, for the pybullet pytest job.
[dependency-groups]
pybullet-dev = [
    {include-group = "dev"},
    "pytest-mock>=3.12,<4",
]

# NEW mypy override — let mypy import-check env/pybullet/* even when the
# extra isn't installed. pybullet ships no type stubs; Any is permitted at
# the FFI boundary per §6.
[[tool.mypy.overrides]]
module = ["pybullet", "pybullet_data"]
ignore_missing_imports = true

# NEW pytest marker.
[tool.pytest.ini_options]
markers = [
    "hf: tests that require the [hf] extra",
    "lerobot: tests that require the [lerobot] extra",
    "monitor: tests that require the [monitor] extra",
    "pybullet: tests that require the [pybullet] extra (pybullet>=3.2)",
]
```

### 4.3 File placement

- `src/gauntlet/env/base.py` — `GauntletEnv` Protocol + `Observation` / `Action` re-exports (new).
- `src/gauntlet/env/registry.py` — `register_env`, `get_env_factory`, `registered_envs` (new).
- `src/gauntlet/env/__init__.py` — registers `"tabletop"` (mutated — one import + one call).
- `src/gauntlet/env/pybullet/__init__.py` — imports `pybullet` with install-hint ImportError guard; on success, calls `register_env("tabletop-pybullet", PyBulletTabletopEnv)` (new).
- `src/gauntlet/env/pybullet/tabletop_pybullet.py` — the backend (new).
- `src/gauntlet/env/pybullet/assets/` — a small textured plane + cube + distractor URDFs + a table URDF (new; all hand-authored minimal URDFs, no external downloads).
- Tests: `tests/pybullet/test_env_pybullet.py`, `tests/pybullet/test_perturbation_pybullet.py`, `tests/pybullet/test_determinism_pybullet.py`, all marked `@pytest.mark.pybullet` and living under `tests/pybullet/` to mirror the RFC-001 `tests/hf/` layout.

## 5. `PyBulletTabletopEnv` scene layout

### 5.1 Scene

Matches the MuJoCo MJCF semantically, but authored as PyBullet primitives + minimal URDFs:

- **Plane**: `p.loadURDF("plane.urdf")` from `pybullet_data` (stdlib asset).
- **Table**: a box collision+visual shape built via `p.createMultiBody`, matching the MuJoCo `table.pos.z = 0.4` / top-thickness `0.02` / half-extents `(0.5, 0.5)`. Kinematic (zero mass).
- **Cube**: box shape half-extent 0.025, mass 0.1, friction matching MuJoCo defaults. Stored as `self._cube_id`. Two textures pre-loaded at `__init__` for the `object_texture` swap (see §6.3).
- **End-effector body**: a small kinematic multi-body (tiny visual sphere, no collision) — the **mocap analogue**. Driven by a `createConstraint(..., jointType=p.JOINT_FIXED)` that we tug around via `changeConstraint(pivot, orn, maxForce=...)` each step. This is option (A) from the brief — see §7.1 for the full rationale.
- **Distractors**: 10 pre-loaded small cubes at fixed baseline positions with alpha=0, collision masked off. `distractor_count` reveals the first N.
- **Target**: a visual-only flat disc (collision shape `GEOM_CYLINDER`, mass 0, `collisionFilterPair` disabled so it never pushes the cube). Indexed as `self._target_id`. Moved via `resetBasePositionAndOrientation` in `reset`.
- **Light / shadow**: PyBullet's headless (`DIRECT`) renderer lighting is controlled through `configureDebugVisualizer` (GUI-only in practice) and the image-pipeline `lightDirection`/`lightDiffuseCoeff` args to `getCameraImage` (headless). Since this RFC ships state-only, the `lighting_intensity` branch stores the requested intensity on `self._light_intensity` and is consumed by `getCameraImage` in the follow-up rendering RFC (see §6.1).

### 5.2 Determinism setup (constructor-time)

```python
self._client = p.connect(p.DIRECT)
p.setPhysicsEngineParameter(
    fixedTimeStep=1.0 / 240.0,          # matches PyBullet default, explicit.
    numSolverIterations=10,
    numSubSteps=0,                       # single-step per mj_step equivalent.
    deterministicOverlappingPairs=1,
    physicsClientId=self._client,
)
p.setGravity(0, 0, -9.81, physicsClientId=self._client)
```

`restore_baseline` on PyBullet is `resetSimulation(physicsClientId=...)` followed by a scene rebuild from cached construction-time IDs — the behavioural-equivalence contract from §3.3 is satisfied without trying to mirror MuJoCo's in-place-snapshot model.

## 6. Perturbation parity table (all 7 axes)

| Axis | MuJoCo (TabletopEnv) | PyBullet (PyBulletTabletopEnv) | Observable on state-only obs? |
|---|---|---|---|
| `lighting_intensity` | `model.light_diffuse[0] = [v, v, v]` | Store `self._light_intensity = v`; applied to `getCameraImage(lightDiffuseCoeff=v, ...)` on the rendering path. Headless DIRECT mode has no runtime light API, so state-only runs are a no-op at the physics level. | **No (cosmetic on state-only)** |
| `camera_offset_x` | `model.cam_pos[cam] = baseline + [v,0,0]` | Store `self._cam_eye = baseline_eye + [v,0,0]`; applied at `computeViewMatrix(cameraEyePosition=...)` call inside the rendering helper. | No (cosmetic on state-only) |
| `camera_offset_y` | Analogous on Y | Analogous on Y | No (cosmetic on state-only) |
| `object_texture` | `model.geom_matid[cube] = default_or_alt` | `p.changeVisualShape(cube_id, -1, textureUniqueId=_tex_default_or_alt)` with two `p.loadTexture(...)` IDs cached at construction. | **No (cosmetic on state-only)** |
| `object_initial_pose_x` | `data.qpos[cube_qpos_adr + 0] = v` | `p.resetBasePositionAndOrientation(cube_id, [v, y_rand, z_rest], identity_quat)` — overrides the random XY just like MuJoCo. | Yes |
| `object_initial_pose_y` | Analogous on Y | Analogous on Y | Yes |
| `distractor_count` | Per-slot rgba.alpha + `contype`/`conaffinity` toggle | Per-slot: `changeVisualShape(alpha=1.0)` + `setCollisionFilterGroupMask(mask=1)` for i<N; reset both to baseline (alpha=0, mask=0) otherwise. Does not call `loadURDF`/`removeBody` per-episode — 10 slots are pre-loaded and toggled, matching the MuJoCo distractor-pool pattern on `TabletopEnv._distractor_geom_ids`. | Yes (only when a distractor blocks the cube path or collides during grasp) |

### 6.1 Per-axis edge cases

- **`lighting_intensity`**: PyBullet's `p.configureDebugVisualizer` lighting knobs only affect the GUI client. Headless (`DIRECT`) rendering uses `getCameraImage` arguments. For this state-only release, we store the requested intensity on an attribute and plumb it through the future rendering path; no physics effect. The follow-up rendering RFC must decide whether to add a `COV_ENABLE_SHADOWS` toggle or stick to the `lightDiffuseCoeff` path — out of scope here. See §12 Q1 for how suites that vary only visual axes are handled.
- **`camera_offset_x` / `camera_offset_y`**: camera pose on PyBullet is a rendering-time concept (there is no "camera body" in the physics scene — the view matrix is built on demand). Like `lighting_intensity`, values are stored on `self._cam_eye` and consumed in the rendering helper.
- **`object_texture`**: PyBullet textures are loaded via `p.loadTexture(file_path)` and bound via `p.changeVisualShape(..., textureUniqueId=...)`. Two 2×2 pixel PNGs ship in `env/pybullet/assets/textures/` (one red, one green, matching the MuJoCo material pair). Like `lighting_intensity`, no state-level effect on state-only obs.
- **`object_initial_pose_x` / `_y`**: the only subtle point is z-height. MuJoCo's `TabletopEnv` keeps cube z at `_CUBE_REST_Z = _TABLE_TOP_Z + _CUBE_HALF = 0.445 m`. PyBullet uses the same constant (same table dims, same cube dims).
- **`distractor_count`**: the MuJoCo implementation toggles per-geom `contype`/`conaffinity` (0 = ghost, 1 = active). PyBullet's analogue is `p.setCollisionFilterGroupMask` — group `1` means "participates in contact", group `0` means "ghost". Invisible-and-ghosted distractors cannot perturb the cube; visible-and-solid ones can. The `distractor_count` validator clamps to `[0, 10]` — exact parity with `TabletopEnv.N_DISTRACTOR_SLOTS`.

### 6.2 The visual-axis honesty caveat

Two of the seven axes (`lighting_intensity`, `object_texture`) only change *rendered* scene content, not proprioception. On a state-only observation dict they mutate the PyBullet scene but produce byte-identical observations — which means a sweep that varies **only** those axes will report every cell as observationally identical, and per-cell success rates will be pairwise-equal up to physics noise (which is zero, because the physics didn't change either).

This is not a bug; it is the consequence of shipping state-only first. Options for how the harness communicates it to users (§12 Q1 picks the default):

1. **Reject at Suite-load time** — the loader errors if a suite declares only cosmetic axes on a state-only backend. Clear, opinionated, matches §6's "never hide failures from users."
2. **Warn at env-construct time** — softer; the run still happens but emits a one-line warning.
3. **Silent no-op** — rejected. Looks like a harness bug.

**Default: (1), loader-side rejection**. The `PyBulletTabletopEnv` exposes a class-level `VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset({"lighting_intensity", "object_texture"})` and the Suite loader cross-checks that at least one axis outside this set is varied when the target backend declares `VISUAL_ONLY_AXES`. `TabletopEnv` declares the same attribute as an empty frozenset (MuJoCo's renderer is state-free but the axes *do* reach the renderer through `render()` calls from `HuggingFacePolicy` via RFC-001's `render_in_obs`), so the check is a true PyBullet-first-release artefact that self-heals when the rendering RFC lands.

## 7. Action + observation parity

### 7.1 Action-control analogue — constraint-based, not IK

The `TabletopEnv` uses a MuJoCo mocap body: no kinematic chain, no actuators, the end-effector position and orientation are *declared* each step and physics never fights them. Two PyBullet analogues were considered:

- **(A) Fixed constraint + `changeConstraint`**. Create a small kinematic multi-body at construction time, pin a fixed `createConstraint(..., jointType=p.JOINT_FIXED, ...)` between the world and that body, and each step call `p.changeConstraint(constraint_id, jointChildPivot=new_pos, jointChildFrameOrientation=new_quat, maxForce=LARGE)`. This is the closest thing PyBullet has to a mocap: the body is driven to the target pose each step through a hard constraint, the 7-D action layout stays byte-identical (`[dx, dy, dz, drx, dry, drz, gripper]`, same `MAX_LINEAR_STEP`/`MAX_ANGULAR_STEP` scales, same snap gripper), and no URDF arm is needed.
- **(B) Inverse kinematics + joint position targets**. Would require shipping a URDF robot arm (even a synthetic 6-DoF stick arm), calling `p.calculateInverseKinematics` every step, and threading position-controlled joint targets via `setJointMotorControlArray`. Introduces a kinematic chain with reachability-failure modes the MuJoCo env does not have, adds asset weight, and gives a different 7-D→IK mapping that would bifurcate policy expectations across backends.

**Decision: (A), constraint-based.** It is the direct mocap analogue, keeps the action space byte-identical, ships no arm asset, has no IK-failure modes, and reuses the exact MAX_LINEAR_STEP / MAX_ANGULAR_STEP / GRIPPER_OPEN / GRIPPER_CLOSED class constants. The grasp model (snap-to-EE when gripper closes within `GRASP_RADIUS`) is replayed via the same `_snap_cube_to_ee` pattern — after `p.stepSimulation` we `resetBasePositionAndOrientation(cube_id, ee_pos, ee_quat)` and zero the cube's velocities via `resetBaseVelocity`. Same override-physics-after-step trick the MuJoCo env uses on lines 497–500 of `tabletop.py`.

### 7.2 Observation-space parity

`observation_space` is byte-identical to `TabletopEnv`:

```python
spaces.Dict({
    "cube_pos":   Box((3,), float64),
    "cube_quat":  Box((4,), float64),   # wxyz order — see §7.3
    "ee_pos":     Box((3,), float64),
    "gripper":    Box((1,), float64),   # -1 closed, +1 open
    "target_pos": Box((3,), float64),
})
```

Shapes and dtypes match. `RandomPolicy` and `ScriptedPolicy` run on the PyBullet backend unchanged.

### 7.3 The one subtle wire-format pin: quaternion order

MuJoCo uses **wxyz** quaternion order throughout (see `TabletopEnv._build_obs`). PyBullet uses **xyzw** order everywhere (`getBasePositionAndOrientation`, `getQuaternionFromEuler`). The backend must convert once in `_build_obs` — `cube_quat = np.array([pb_quat[3], pb_quat[0], pb_quat[1], pb_quat[2]])` — so downstream consumers see MuJoCo-order quats regardless of backend. This is documented in the backend's docstring and asserted in tests (§9 case 6).

### 7.4 Cross-backend numerical parity: explicitly NO

Same policy + same seed on `tabletop` vs `tabletop-pybullet` produces **different trajectories**. Different solvers (MuJoCo's `CG/Newton` vs PyBullet's `btSequentialImpulseConstraintSolver`), different contact parameters, different integrator step semantics, different friction models. "Semantically similar, numerically different" is the honest summary. The `gauntlet compare` command, when fed a report from each backend, measures simulator-induced drift — **not** policy regression. The implementation must document this in both the env docstring and the follow-up `gauntlet compare` CLI help text (see §11 and §12 Q2).

## 8. Determinism notes

### 8.1 Within-backend bit-determinism — achievable

PyBullet's solver is deterministic given:

- `p.connect(p.DIRECT)` (no GUI, no multi-threaded renderer).
- `p.setPhysicsEngineParameter(fixedTimeStep=1/240, numSolverIterations=10, deterministicOverlappingPairs=1)` locked at construction and never mutated.
- No random-number calls inside the physics loop (the backend's RNG is the `np.random.Generator` seeded from `reset(seed=...)`, used only for cube/target XY randomisation).
- `resetSimulation` called inside `restore_baseline` so every episode starts from a fresh, fully-re-created scene — no residual internal state from prior episodes.

Given those knobs, `reset(seed=42)` twice yields identical observations, `(reset + fixed action sequence)` yields identical trajectories, and the Runner's existing bit-determinism contract (`runner/worker.py::_execute_one`) holds unchanged: a single `master_seed` still fully reproduces every episode.

### 8.2 Across-backend: not achievable, not attempted

Per §7.4.

### 8.3 OS / Python / library version sensitivity

PyBullet's solver is numerically stable across recent Linux/macOS/Windows wheels on x86-64, but aarch64 wheels (Apple Silicon) have historically shown sub-ULP differences in contact impulses. CI runs `ubuntu-latest` x86-64 only — bit-determinism promises are CI-scoped. A cross-arch canary is noted in §9.

## 9. Test plan

New marker `@pytest.mark.pybullet`, new subdir `tests/pybullet/`. Default `pytest` (the `-m 'not hf and not lerobot and not monitor and not pybullet'` default job) de-selects everything.

### 9.1 Unit / smoke tests (run in the new `pybullet-tests` CI job)

1. **Protocol conformance**: `assert isinstance(PyBulletTabletopEnv(), GauntletEnv)` — the `runtime_checkable` Protocol check. Also do it for `TabletopEnv` to lock the Protocol contract in both directions.
2. **Registry round-trip**: `get_env_factory("tabletop-pybullet")()` returns an instance whose type satisfies `GauntletEnv`. Double-registration (`register_env("tabletop-pybullet", X)` twice) raises `ValueError`.
3. **Reset determinism**: call `env.reset(seed=42)` twice; the returned obs dicts must be bit-identical (`np.array_equal` on every key).
4. **Step determinism**: same seed, a fixed 20-step random-but-seeded action sequence → `np.array_equal` on every obs returned from step.
5. **Action-/observation-space parity**: `env.action_space == TabletopEnv().action_space` by shape+dtype+bounds; `env.observation_space.keys() == {"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"}` with matching shapes/dtypes.
6. **Quat order**: run 10 steps with a rotation-only action, assert `obs["cube_quat"][0]` matches the expected w-component (not the PyBullet native xyzw's w in position 3).
7. **Per-axis application** (seven tests, one per axis):
   - State-effecting axes (`object_initial_pose_x`, `_y`, `distractor_count`) — assert the world-state change (cube at the requested XY after reset; N distractors visible and solid).
   - Cosmetic axes (`lighting_intensity`, `object_texture`, `camera_offset_x`, `_y`) — assert the internal attribute is set (state-only obs, cannot assert observable effect in this release). Flag these with a `xfail`-style comment pointing at the follow-up rendering RFC. Also assert they are no-ops on the observation dict (the state-only guarantee).
8. **`restore_baseline`**: set all seven axes, call `reset(seed=s)`, then `set_perturbation` nothing + `reset(seed=s)` — the observation must match the baseline-s trajectory.
9. **End-to-end rollout smoke**: 10 rollouts with `RandomPolicy`, seed 0..9 — no crashes, no NaN/Inf in any observation, `obs["cube_pos"][2]` stays ≥ table top minus 1cm tolerance (catches cube falling through the table from a bad collision mask).
10. **Runner integration**: run a small suite (two axes × two steps × 2 episodes) through `Runner.run(env_name="tabletop-pybullet")` — resulting `Episode` objects pass the existing schema validators (`gauntlet.runner.episode.Episode.model_validate`).
11. **Load-time visual-axis rejection** (§6.2 / §12 Q1): a suite varying only `lighting_intensity` against `env: tabletop-pybullet` raises a `pydantic.ValidationError` at `load_suite_from_string` time with a message pointing at the rendering RFC.
12. **Extra-missing ImportError**: monkeypatch `sys.modules['pybullet'] = None` and assert importing `gauntlet.env.pybullet` raises `ImportError` whose message contains `uv sync --extra pybullet`.

### 9.2 Mocking strategy

Most tests construct a real `PyBulletTabletopEnv` — the `[pybullet]` extra is installed in the `pybullet-tests` CI job, so there is no pickle/sim boundary to mock. Tests that need to exercise the ImportError path (case 12) monkeypatch at module-import time. There is no MagicMock-the-sim pattern; PyBullet's DIRECT mode is fast enough (roughly 10 ms/step on CPU) that a full rollout-per-test is tolerable.

### 9.3 CI

One new job: `pybullet-tests`. `uv sync --extra pybullet --group pybullet-dev` → `uv run pytest -m pybullet`. Runs on `ubuntu-latest` x86-64 (aarch64 deferred, see §8.3). Default job continues to run `pytest -m 'not hf and not lerobot and not monitor and not pybullet'` and must stay green without the `pybullet` wheel installed — the enforcement of §6's "core stays small."

## 10. Module layout

```
src/gauntlet/
├── env/
│   ├── __init__.py                    # MUTATED: registers "tabletop"
│   ├── base.py                        # NEW: GauntletEnv Protocol
│   ├── registry.py                    # NEW: register_env / get_env_factory
│   ├── tabletop.py                    # TOUCHED: AXIS_NAMES class attr; dropped _KNOWN_AXIS_NAMES
│   ├── assets/
│   │   └── tabletop.xml               # unchanged
│   ├── perturbation/                  # unchanged
│   └── pybullet/                      # NEW SUBPACKAGE
│       ├── __init__.py                # NEW: ImportError guard; calls register_env
│       ├── tabletop_pybullet.py       # NEW: PyBulletTabletopEnv
│       └── assets/
│           ├── table.urdf             # NEW (hand-authored minimal URDF)
│           ├── cube.urdf              # NEW
│           ├── distractor.urdf        # NEW
│           └── textures/
│               ├── cube_default.png   # NEW (2×2 red)
│               └── cube_alt.png       # NEW (2×2 green)
├── suite/
│   ├── schema.py                      # MUTATED: SUPPORTED_ENVS now registry-backed
│   └── loader.py                      # MUTATED: triggers backend import on env dispatch
├── runner/
│   └── worker.py                      # MUTATED: env: GauntletEnv (was TabletopEnv)
└── ...
```

## 11. CLI + suite YAML syntax

### 11.1 YAML

No new keys. The existing `env:` string is the registry key:

```yaml
name: tabletop-basic-v1-pybullet
env: tabletop-pybullet          # was "tabletop"; now one of registered_envs()
axes:
  object_initial_pose_x: {low: -0.1, high: 0.1, steps: 3}
  distractor_count:      {low: 0, high: 4, steps: 3}
episodes_per_cell: 5
```

`"tabletop"` continues to work (it is registered by `gauntlet.env.__init__`). `"tabletop-pybullet"` triggers the registry path described in §3.4.

### 11.2 Loader dispatch + extra-missing error

`suite/loader.py::_validate` grows a small step: after the pydantic validation, if `suite.env` is not yet in `registered_envs()`, attempt the canonical import for the built-in backends (a small hard-coded `_BUILTIN_BACKEND_IMPORTS: dict[str, str]` — `{"tabletop-pybullet": "gauntlet.env.pybullet"}`). If the import itself raises, surface a single clear error:

```
unknown env 'tabletop-pybullet'; the matching extra is not installed.
Install with:
    uv sync --extra pybullet
or, for a plain pip env:
    pip install 'gauntlet[pybullet]'
```

If the import succeeds but the backend still isn't registered, raise the generic `get_env_factory` `ValueError` (with `sorted(registered_envs())` listed).

### 11.3 CLI

`gauntlet run`, `gauntlet report`, `gauntlet compare` are unchanged at the command-surface level. One UX polish: `gauntlet run` picks up the selected env name from the suite YAML and (on startup, before worker fork) imports the matching backend in the parent process so the import-hint error surfaces once in the parent's stderr rather than N times across workers. `gauntlet compare` gains one defensive check — if the two reports' `suite.env` differ, emit a loud warning and a `--i-know-what-i-am-doing` flag (see §12 Q2).

## 12. Open questions with defaults

- **Q1 — How to handle suites that vary only cosmetic axes on a state-only backend?** (See §6.2.) **Default: reject at Suite-load time** with a clear error pointing at the follow-up rendering RFC. Alternatives (warn, silent-no-op) are listed for the implementer's reference. **This is the top open question** — the UX consequences are user-facing.
- **Q2 — `gauntlet compare` across different backends.** `compare(a, b)` where `a.suite.env != b.suite.env` measures simulator drift, not policy regression. **Default: emit a loud warning, exit nonzero unless `--allow-cross-backend` is passed.**
- **Q3 — Headless light control knob.** PyBullet's headless `getCameraImage` takes `lightDiffuseCoeff` but not a full RGB vector. The MuJoCo axis is a scalar today, so the two line up trivially; if the rendering RFC ever upgrades the axis to a 3-vector, the backend signature must change. **Default: keep it a scalar forever; if per-channel lighting becomes a real requirement, it's a new axis, not a widening of `lighting_intensity`.**
- **Q4 — Per-episode `resetSimulation` cost.** Rebuilding the scene per-reset costs a handful of ms; tolerable at our step counts (200 max × 5 ms each ≈ 1 s per episode). **Default: eat the cost for now.** Revisit only if profiling shows the setup is >5% of wall time.
- **Q5 — Mass / friction defaults.** MuJoCo's defaults vs PyBullet's defaults are different. Since cross-backend numerical parity is already out of scope (§7.4), the PyBullet URDFs pick "reasonable for the task" values (mass 0.1 kg, friction 1.0) and document them. **Default: reasonable-defaults, document, don't chase MuJoCo-number-matching.**
- **Q6 — URDF asset paths at install time.** `pybullet_data.getDataPath()` gives a plane URDF for free, but our four custom URDFs (`table.urdf`, `cube.urdf`, `distractor.urdf`, and the two textures) must ship with the wheel. **Default: `[tool.uv_build] sources = ["src"]` + `[tool.setuptools.package-data]`-equivalent include — tested in case 12 of §9.**

## 13. Implementation checklist (~13 commits)

Sized to land as one §9-cadence PR. Each bullet maps to one commit:

1. Add `[project.optional-dependencies] pybullet = [...]`, `[dependency-groups] pybullet-dev`, the `pybullet` mypy override, and the `pybullet` pytest marker to `pyproject.toml`. Regenerate `uv.lock`.
2. Add `src/gauntlet/env/base.py` with the `GauntletEnv` Protocol and docstring (§3.2–§3.3 verbatim).
3. Add `src/gauntlet/env/registry.py` with `register_env` / `get_env_factory` / `registered_envs`. Unit tests for the registry in `tests/test_env_registry.py` (unmarked — pure Python).
4. Touch `src/gauntlet/env/tabletop.py`: promote `_KNOWN_AXIS_NAMES` to `TabletopEnv.AXIS_NAMES: ClassVar[frozenset[str]]`; add `VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset()`. Touch `env/__init__.py` to `register_env("tabletop", TabletopEnv)`.
5. Mutate `src/gauntlet/suite/schema.py`: replace `SUPPORTED_ENVS` literal with a `_env_supported` validator that consults `registered_envs()` plus the backends-not-yet-imported dispatch table. Mutate `suite/loader.py::_validate` with the import-on-demand logic and the install-hint error.
6. Mutate `src/gauntlet/runner/worker.py`: change the `TabletopEnv` type hints to `GauntletEnv`; the runtime behaviour is unchanged because `TabletopEnv` satisfies the Protocol. Add a short test proving a mock-`GauntletEnv` can drive the Runner end-to-end (protects against future breaks).
7. Add `src/gauntlet/env/pybullet/__init__.py` with the ImportError guard and the `register_env` call. Add a failing-import unit test (§9.1 case 12).
8. Ship asset files: `env/pybullet/assets/table.urdf`, `cube.urdf`, `distractor.urdf`, `textures/cube_default.png`, `textures/cube_alt.png`. Wire package-data inclusion.
9. Add `src/gauntlet/env/pybullet/tabletop_pybullet.py` with construction, `restore_baseline`, `reset`, `step`, `set_perturbation`, `_apply_ee_command` (constraint-based §7.1), `_snap_cube_to_ee`, `_build_obs`, `close`, plus the `AXIS_NAMES` / `VISUAL_ONLY_AXES` ClassVars.
10. Wire the seven per-axis branches of `_apply_one_perturbation` per the table in §6. State-effecting branches commit functional code; cosmetic branches set attributes with a comment pointing at the rendering RFC.
11. Write `tests/pybullet/test_env_pybullet.py` (cases 1–5, 9, 10), `tests/pybullet/test_perturbation_pybullet.py` (cases 7, 8), `tests/pybullet/test_determinism_pybullet.py` (cases 3, 4). All `@pytest.mark.pybullet`.
12. Add `tests/pybullet/test_loader_pybullet.py` (case 11, visual-axis rejection) and the `gauntlet compare` cross-backend warning (Q2). Also the protocol-conformance test for `TabletopEnv` itself (case 1 part b).
13. Add a `pybullet-tests` job to `.github/workflows/ci.yml` mirroring `hf-tests` from RFC-001 §3. Update the README's "Backends" section to list `"tabletop"` and `"tabletop-pybullet"` with the `uv sync --extra pybullet` one-liner.

Feasibility check against the ~12–15 budget: the seven per-axis branches are cheap (cosmetic axes are two-line attribute writes; state-effecting axes are three-line PyBullet calls). The heavy lifting is the scene wiring (commit 9) and the URDF authoring (commit 8). Budget hits 13 commits as listed; the budget is tight, so we **defer full observable parity on the two cosmetic axes to the rendering follow-up RFC and land the branches as stubbed attribute writes here** — the §6 table is honest about this.

## 14. Appendix: Isaac Sim + Genesis follow-up RFC sketches

### 14.1 RFC-006 sketch — `"tabletop-isaac"` via Isaac Sim

- **Extras**: `[isaac]`. Likely torch-heavy (Isaac Sim's `omni.isaac.core` ships a torch-using tensor-API tier) and GPU-only; the extras group documents a pre-install shim that the user drives manually (Omniverse Kit launcher or `pip install isaacsim`-style wheel once the public PyPI path stabilises). **Will not ship on public CI** — licensing and GPU-runner cost preclude it. The harness side is identical: `register_env("tabletop-isaac", IsaacTabletopEnv)` inside `gauntlet.env.isaac.__init__`, Protocol-conformant backend, same 7-axis `set_perturbation` branches. Perturbation mappings: lighting via USD `Distant_Light.intensity`, camera via Isaac's `Camera.set_world_pose`, textures via USD material override, pose/distractor via USD primitive attributes. Constraint-based mocap EE maps to Isaac's `RigidPrim.set_world_pose` under kinematic mode. Determinism knob: `SimulationContext` with `use_gpu_pipeline=False` and fixed `physics_dt`. Open question the RFC must answer: does the USD scene graph ship as a single `.usda` file in the repo or regenerated programmatically at `reset`? (Recommendation: single-file asset, analogous to the MuJoCo MJCF — keeps diffable scene authoring cheap.)
- **Blocker the RFC must surface**: Isaac Sim's Python API is `asyncio`-flavoured in places (`omni.kit.async_engine`); the Runner's synchronous `_execute_one` may need a thin `asyncio.run(...)` wrapper at the backend boundary — a contained wart, not a Protocol change.

### 14.2 RFC-007 sketch — `"tabletop-genesis"` via Genesis

- **Extras**: `[genesis]`. `genesis-world` (the PyPI name as of April 2026) pulls torch + CUDA, so CI compatibility matches the `[hf]` pattern: mock at the backend seam where possible, gate a real-Genesis smoke test behind `workflow_dispatch`. Harness side: `register_env("tabletop-genesis", GenesisTabletopEnv)` inside `gauntlet.env.genesis.__init__`, Protocol-conformant. Genesis has first-class parallel-env APIs which the RFC must decide whether to expose (the `Runner` currently parallelises via `multiprocessing`; Genesis's in-process parallelism is an optimisation opportunity, not a requirement). Perturbation mappings: lighting via `scene.add_light(...)` / runtime mutation of `light.intensity`, camera via `scene.add_camera(...)`, textures via `entity.set_material(...)`, pose via `entity.set_pos`, distractors via pre-spawned entity pool with visibility toggles.
- **Blocker the RFC must surface**: Genesis's API has been churning (breaking changes between minor versions in early 2026). The RFC should pin a specific Genesis version, treat that pin as the tested contract, and explicitly promise nothing about forward-compatibility until the API settles. The `Protocol`-based harness insulates users — a Genesis backend update is one-file-isolated.
