# Polish exploration: multi-camera observation support

Status: exploration / pre-implementation
Owner: polish/multi-camera-obs branch

## 1. Why this matters (the domain win)

Real-robot manipulation policies almost never run on a single fixed
overhead camera. The de-facto pattern across LeRobot SmolVLA, Diffusion
Policy, ACT, and most production VLA stacks is **two to four
synchronised RGB streams** — typically a wrist-mounted camera that
follows the gripper, plus one or two side/overhead cameras for global
context. SmolVLA's released checkpoints, for example, expect
`observation.images.camera1`, `observation.images.camera2`,
`observation.images.camera3` (`src/gauntlet/policy/lerobot.py:163-165`).

The current harness exposes a single `obs["image"]` per env when
constructed with `render_in_obs=True`. Multi-view policies that
gauntlet aspires to evaluate honestly therefore have three bad
options today:

1. **Lossy downsampling** — concatenate the same single view three
   times and pretend it's three cameras. Defeats the purpose: the
   wrist camera carries information no fixed view does, and a
   policy that trained on it cannot recover from this.
2. **Patch the env** — fork `TabletopEnv` per evaluation. Off-spec,
   and every fork has to re-derive the perturbation contract.
3. **Skip multi-view policies entirely** — silently narrows the
   harness's scope away from exactly the policies that matter for
   real-robot evaluation.

This RFC adds a **first-class, opt-in** multi-camera surface that
keeps the single-camera default byte-identical (sacred per the task
spec; the existing render tests at `tests/test_env.py:430-481` pin
this), and lights up an `obs["images"]: dict[str, np.ndarray]` path
for callers that want it.

## 2. Public API

`gauntlet.env.base.CameraSpec` (new):

```python
class CameraSpec(NamedTuple):
    name: str
    pose: tuple[float, float, float, float, float, float]
    size: tuple[int, int]   # (height, width) — matches render_size
```

* `name` — the key under which this camera's frame appears in
  `obs["images"][name]`. Recommended values: `"wrist"`, `"top"`,
  `"side"`, `"front"` to match LeRobot conventions, but any
  non-empty unique string is accepted.
* `pose = (x, y, z, rx, ry, rz)` — **world-frame** position in
  metres + **MuJoCo-convention XYZ Euler angles in radians**. The
  camera looks along its local `-Z` axis, with local `+Y` as up
  (standard MuJoCo `<camera>` semantics; see MJCF reference). This
  is the same convention as the existing `<camera name="main">`
  declaration in `src/gauntlet/env/assets/tabletop.xml`. We pin
  Euler XYZ rather than quat because the multi-camera *use case*
  is humans typing camera positions into a config file; quaternions
  are awful to author by hand.
* `size = (H, W)` — the rendered image shape, matching the existing
  `render_size` convention (height-first to match MuJoCo's
  `Renderer(model, height=H, width=W)` constructor and the existing
  `obs["image"]` shape).

`Env.__init__(..., cameras: list[CameraSpec] | None = None)`:

* `cameras=None` (default): **byte-identical Phase 1 behaviour**.
  The legacy `render_in_obs=True` / `render_size=...` path keeps
  working; nothing else changes. The single-cam default-unchanged
  test in `tests/test_multi_camera.py::test_single_camera_default_unchanged`
  pins this.
* `cameras=[CameraSpec(...), ...]`: a per-camera dict appears at
  `obs["images"][name]`. The legacy `obs["image"]` key is **also**
  populated, aliased to the **first** camera's frame, so existing
  consumers (the runner's video recording at
  `src/gauntlet/runner/worker.py:417`, the
  `LeRobotPolicy._get_image_from_obs` lookup, and any downstream
  user code) keep working unmodified.
* `cameras=[]` (empty list): treated identically to `cameras=None`.
  Forbidding it would break ergonomics for callers that build the
  list dynamically; falling back to legacy behaviour is the
  least-surprise option and matches what `cameras=None` does.

### Interaction with the legacy `render_in_obs` kwarg

`cameras` takes precedence. When both are passed, the legacy
`render_in_obs` and `render_size` are silently ignored — the
`cameras` list fully describes the rendering surface. We considered
raising on the conflict but decided against it: callers migrating
from the legacy API will frequently leave `render_in_obs=True` in
place out of muscle memory, and a hard error there hurts more than
it helps. A FutureWarning is overkill for a kwarg this niche.

### Duplicate / invalid `cameras` validation

* Duplicate `name` values raise `ValueError("duplicate camera name: ...")`
  at `__init__` time.
* `name == ""` raises `ValueError("camera name must be non-empty")`.
* `size` with non-positive entries raises `ValueError`.
* The `pose` tuple is type-checked structurally (NamedTuple), no
  range validation — there's no "wrong" camera position physically.

## 3. Backwards-compatibility strategy

| Surface | Default (`cameras=None`) | Multi-cam (`cameras=[...]`) |
|---|---|---|
| `obs.keys()` | unchanged (Phase 1 set + `image` if `render_in_obs`) | Phase 1 set + `image` (alias) + `images` |
| `obs["image"]` | uint8 (H, W, 3) when `render_in_obs=True`, else absent | uint8 (H, W, 3) — first camera's frame |
| `obs["images"]` | absent | `dict[str, NDArray[uint8]]`, one entry per `CameraSpec` |
| `observation_space["image"]` | Box (H, W, 3) uint8 when `render_in_obs=True` | Box matching first camera's shape |
| `observation_space["images"]` | absent | `gymnasium.spaces.Dict({name: Box(H, W, 3)})` |
| `set_perturbation(...)` axes | unchanged | unchanged (`camera_offset_*` still moves the legacy `main` camera; multi-cam poses are NOT perturbed in v1 — see open questions §6) |

The byte-identity guarantee is verified by
`tests/test_multi_camera.py::test_single_camera_default_unchanged`,
which constructs `TabletopEnv()` (no kwargs) and pins both the obs
key set and the obs-space contents against a frozen golden frozenset
(same shape as the existing `_PHASE_1_OBS_KEYS` at
`tests/test_env.py:433`).

### Runner / trajectory recording interaction

The Runner's worker (`src/gauntlet/runner/worker.py:411`) iterates
`obs.items()` when `record_trajectory=True` and forces every value
through `np.asarray(v, dtype=np.float64)`. A nested
`obs["images"]: dict[...]` value will raise a clear `TypeError`
("float() argument must be a string or a real number, not 'dict'")
on the first trajectory append. This is intentional and acceptable:

* The runner is owned by a sibling agent (sibling branch
  `polish/incremental-cache`), so we cannot extend its iteration
  logic in this PR.
* The single-camera path through `obs["image"]` (uint8, gets cast
  to float64) keeps working when `record_trajectory=True`. Multi-
  camera users who need full per-step trajectories should record
  the first cam through the legacy alias and reconstruct other
  views from `seed + perturbation` if needed.
* `record_video=True` continues to work in multi-cam mode because
  the worker reads `obs["image"]` (the first-cam alias), not
  `obs["images"]`.

A follow-up could extend the worker to skip non-array obs values
when recording trajectories; out of scope for this PR.

## 4. Per-backend support matrix

| Backend | Multi-cam in this PR | Notes |
|---|---|---|
| MuJoCo (`tabletop.py`) | **YES** | Per-spec `<camera>` injected into the MJCF string at `__init__` (`from_xml_string` instead of `from_xml_path`), one cached `mujoco.Renderer` per spec because sizes differ. The legacy `main` camera is preserved alongside, so `camera_offset_*` perturbations and the legacy `render_in_obs` codepath are untouched. |
| PyBullet (`tabletop_pybullet.py`) | deferred | PyBullet is the easiest backend to extend (`getCameraImage` is per-call, no scene-graph state to add) and is a strong candidate for the next polish task. Marker file: a follow-up RFC will track. |
| Genesis (`tabletop_genesis.py`) | deferred | Multi-cam in Genesis means one `scene.add_camera(...)` per spec at scene-build time (`tabletop_genesis.py:389`). The current camera is wired into perturbation axes via `_apply_camera_pose` (`tabletop_genesis.py:695`); a multi-cam refactor needs to disentangle the perturbed legacy camera from new fixed cameras, plus the build-time scene cost grows linearly with cam count. Larger surface; deferred so this PR stays atomic. |
| Isaac Sim (`isaac/tabletop_isaac.py`) | deferred / mocked | Isaac adapter is a smoke-test mock today (Phase 2 §10). Multi-cam wiring there is a no-op until the real backend lands. |

The MuJoCo-only PR is already a real value-add: MuJoCo is the
default backend and the one used by every example in the README,
the existing render tests, and the SmolVLA / HuggingFace policy
adapters' integration tests.

## 5. `CameraSpec` design choice — NamedTuple vs dataclass vs pydantic

* **NamedTuple** (chosen): zero new deps, immutable, hashable,
  supports tuple unpacking (`x, y, z, rx, ry, rz = spec.pose`),
  cheap to import (typed module already imports `NamedTuple` for
  other Phase 2 surfaces). mypy strict-friendly.
* dataclass: equivalent runtime semantics, but mutable by default
  (`frozen=True` adds boilerplate), and subscript syntax doesn't
  work the way users expect for "config-like" objects.
* pydantic: gauntlet has zero pydantic deps today; adding one for
  a 3-field structural type is wildly disproportionate.

NamedTuple wins on every axis here.

## 6. Open questions (deferred)

1. **Should the `camera_offset_x` / `camera_offset_y` perturbations
   apply to all multi-cam cameras or only the legacy `main`?**
   This PR keeps the legacy semantics: those axes still only move
   `main`. A follow-up could add `camera_offset_x__wrist` etc., but
   the per-axis-name proliferation is ugly and the use case (jittering
   multiple cameras in lockstep) is not currently asked for.
2. **Per-camera intrinsics (FOV, near/far)?** Hardcoded to MuJoCo
   defaults via the injected `<camera>` element in v1. Adding them
   to `CameraSpec` is straightforward when a real caller asks.
3. **Should `cameras` be wired through the Suite YAML loader?**
   Not in this PR — the Suite loader (`src/gauntlet/suite/loader.py`)
   only constructs envs via the registry, and the multi-cam axis
   is more naturally a per-`env_factory` concern (i.e. the user's
   suite Python file constructs the env with the cameras they
   want). Adding `cameras: ...` to the YAML schema is a separate
   conversation.
4. **Should `_PHASE_1_OBS_KEYS` move to a shared module?** Currently
   inlined in three test files. Out of scope; a tiny test-helper
   refactor.
