# Phase 2 RFC 008 — Genesis image observations

- **Status**: Draft
- **Phase**: 2, Task 8 (closes the image-rendering non-goal from RFC-007 §2 / §6 / §9).
- **Author**: architect agent
- **Date**: 2026-04-23
- **Supersedes**: n/a. Builds on RFC-006 and RFC-007 without amending either.
- **References**:
  - `docs/phase2-rfc-001-huggingface-policy.md` §5 (MuJoCo `render_in_obs=True` / `render_size` precedent).
  - `docs/phase2-rfc-006-pybullet-rendering.md` (the PyBullet rendering precedent this RFC mirrors section-for-section).
  - `docs/phase2-rfc-007-genesis-adapter.md` §6 / §9 (state-only gap, VISUAL_ONLY_AXES rejection, §9 sketch for this RFC).
  - `docs/phase2-exploration-task8-genesis-rendering.md` (Rasterizer measurements, private-API survey, texture-swap design space).

---

## 1. Summary

`GenesisTabletopEnv` ships state-only (RFC-007 §2 non-goal, §6 carve-out).
The four cosmetic axes — `lighting_intensity`, `object_texture`,
`camera_offset_x`, `camera_offset_y` — reach the env's
`_apply_one_perturbation` dispatcher and land on instance attributes
(`_light_intensity`, `_texture_choice`, `_cam_offset`), but nothing
consumes them and the Suite loader rejects any sweep whose every axis is
cosmetic. Image-conditioned policies (`HuggingFacePolicy`,
`LeRobotPolicy` — RFC-001/002) cannot be evaluated on the Genesis
backend today because there is no `obs["image"]` path.

This RFC adds a headless, deterministic image-observation path to
`GenesisTabletopEnv`:

1. New constructor kwargs `render_in_obs: bool = False` and
   `render_size: tuple[int, int] = (224, 224)`, identical surface and
   defaults to `TabletopEnv` (`env/tabletop.py:160-161`) and
   `PyBulletTabletopEnv` (RFC-006 §3.1). Default stays off so the
   RFC-007 state-only contract is byte-identical.
2. A pinhole camera added at `__init__` time via `scene.add_camera(res,
   pos, lookat, up, fov, near, far)` when `render_in_obs=True`, using
   the shared `_CAM_*` constants that match MuJoCo and PyBullet. Not
   added when `render_in_obs=False` — zero new scene-graph entities on
   the state-only default path.
3. A `_render_obs_image` method that, before each call, flushes pending
   rigid transforms through
   `scene.visualizer.rasterizer._context.update_rigid()` and then
   returns `cam.render(rgb=True)[0]` as a uint8 `(H, W, 3)` array —
   shape / dtype / bounds byte-identical to MuJoCo (exploration §2.3,
   §2.4).
4. `_build_obs` emits that array under `obs["image"]` when enabled;
   `observation_space["image"]` is a `Box(0, 255, (H, W, 3), uint8)` —
   identical to `TabletopEnv` (`env/tabletop.py:232-237`) and
   `PyBulletTabletopEnv` (RFC-006 §3.2).
5. The four cosmetic axes become **observable on `obs["image"]`** via:
   - `lighting_intensity` → scales the default directional light's
     intensity by writing
     `scene.visualizer.rasterizer._context._scene.light_nodes[0].light.intensity = BASELINE_LIGHT_INTENSITY * self._light_intensity`.
     Private-API hop documented (exploration §4), pinned by the
     existing `genesis-world>=0.4,<0.5` constraint.
   - `camera_offset_x` / `camera_offset_y` → `cam.set_pose(pos=eye + offset, lookat=_CAM_TARGET, up=_CAM_UP)` post-build (exploration §3.1).
   - `object_texture` → two cubes are allocated at `__init__`: a "red"
     cube at the baseline rest pose and a "green" cube at
     `_DISTRACTOR_HIDDEN_Z = -10.0`. The perturbation branch swaps the
     two cubes' positions and the `self._cube` / `self._cube_alt`
     handles, so the downstream physics / grasp code keeps working on
     `self._cube` unchanged (exploration §5).
6. `GenesisTabletopEnv.VISUAL_ONLY_AXES` drops to `frozenset()` — exact
   parity with `TabletopEnv` and (post-RFC-006) `PyBulletTabletopEnv`.
   The loader's `_reject_purely_visual_suites` becomes a no-op on
   `tabletop-genesis` — cosmetic-only sweeps load.

What this RFC does **not** do (§2):

- Does not change the CLI or the Suite schema. Users enable rendering
  via an env factory, exactly like `examples/evaluate_openvla.py` does
  for MuJoCo and `examples/evaluate_smolvla_pybullet.py` does for
  PyBullet (RFC-006 §8 Q3).
- Does not chase cross-backend pixel parity (RFC-007 §7.3 holds —
  semantic parity only: same camera pose, same scene layout, same
  light direction).
- Does not wire `shadow` as a perturbation knob. Genesis's default
  `VisOptions.shadow=True` stays on (exploration §7, §10 Q3).
- Does not expose depth / segmentation / normal channels. `cam.render`
  stays at `rgb=True` only.
- Does not add GPU-rendering support. The path is CPU Rasterizer-only,
  same backend pin RFC-007 set (`gs.init(backend=gs.cpu)`).

## 2. Goals / non-goals

### Goals

- Match `TabletopEnv`'s and `PyBulletTabletopEnv`'s `render_in_obs`
  surface (kwargs, defaults, obs key, space shape/dtype/bounds)
  byte-for-byte. VLA policies that work on MuJoCo via RFC-001
  `render_in_obs=True` and on PyBullet via RFC-006 must work on
  Genesis by swapping only the env factory (e.g.
  `partial(GenesisTabletopEnv, render_in_obs=True)`), no policy
  changes.
- Consume all four cosmetic axes in the render path. A sweep varying
  only `lighting_intensity` and `object_texture` — previously rejected
  at load time on `tabletop-genesis` — now loads and produces distinct
  pixels per cell.
- Preserve bit-determinism. `reset(seed=s)` twice on two envs produces
  `np.array_equal`-equal `obs["image"]` arrays. Same seed + same
  action sequence produces the same per-step image. Matches
  within-backend contract from RFC-007 §8.1.
- Keep the state-only default byte-identical to the RFC-007 contract.
  `render_in_obs=False` (the default) produces an observation dict
  with exactly the existing five keys, and no camera is attached to
  the scene.
- Keep `gauntlet.core` Genesis-free. The rendering code lives entirely
  inside `gauntlet.env.genesis.tabletop_genesis`; no new module-level
  imports outside the `[genesis]` extra. The private-API hop reaches
  `scene.visualizer.rasterizer._context` which is a runtime attribute
  on the already-imported `gs.Scene` — no new imports.
- Keep `mypy --strict` green whether or not the `[genesis]` extra is
  installed. RFC-007's `[[tool.mypy.overrides]]` on `genesis.*`
  already covers the new call sites.

### Non-goals

- **CLI / YAML changes.** Pre-existing gap addressed by the
  `suite.env`-dispatch bug-fix (already landed, commit `619b99a`). No
  new surface this RFC.
- **Cross-backend pixel parity.** RFC-007 §7.3 holds. A test asserts
  shape / dtype parity with MuJoCo but explicitly not pixel equality.
- **Shadow rendering knob.** Deferred (exploration §10 Q3).
- **Depth / segmentation / normal channels.** Not emitted.
- **FOV / near / far perturbations.** Camera intrinsics are fixed at
  construction time. Not in the canonical seven axes.
- **Apple Silicon / non-x86-64 determinism.** CI is x86-64 Linux.
  Cross-arch pixel-determinism is not asserted (exploration §7.1,
  RFC-006 §5.1).
- **Genesis 0.5 upgrade.** Handled by a separate RFC when the minor
  version lands. The `genesis-world<0.5` pin from RFC-007 §3 holds.

## 3. The surface

### 3.1 Constructor

```python
class GenesisTabletopEnv:
    def __init__(
        self,
        *,
        max_steps: int = 200,
        n_substeps: int = 5,
        render_in_obs: bool = False,
        render_size: tuple[int, int] = (224, 224),
    ) -> None:
        ...
```

Validation matches `TabletopEnv.__init__` line-for-line:
`render_in_obs=True` with a non-positive `render_size` raises
`ValueError("render_size must be a (height, width) of positive ints; got ...")`.
Byte-copy of the string so any future change to the MuJoCo validator
must land in all three places, enforced by the paired test in §7 case 2.

### 3.2 `observation_space`

```python
obs_spaces: dict[str, gym.spaces.Space[Any]] = {
    "cube_pos":   spaces.Box(...),  # unchanged
    "cube_quat":  spaces.Box(...),
    "ee_pos":     spaces.Box(...),
    "gripper":    spaces.Box(...),
    "target_pos": spaces.Box(...),
}
if self._render_in_obs:
    h, w = self._render_size
    obs_spaces["image"] = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
self.observation_space = spaces.Dict(obs_spaces)
```

The `"image"` entry is conditional — present iff `render_in_obs=True`.
Matches `TabletopEnv.observation_space` (`env/tabletop.py:232-237`) and
`PyBulletTabletopEnv` (RFC-006 §3.2) exactly.

### 3.3 `_render_obs_image` method

```python
def _render_obs_image(self) -> NDArray[np.uint8]:
    """Render the scene camera into a uint8 (H, W, 3) array. Deterministic,
    headless (CPU Rasterizer), consumes the four cosmetic axes.

    Pushes any pending rigid-transform updates into the Rasterizer
    context before rendering. Without this flush, set_pos calls that
    have not been followed by scene.step() are invisible to the renderer
    (exploration §2.4).
    """
    self._scene.visualizer.rasterizer._context.update_rigid()
    rgb = self._camera.render(rgb=True)[0]
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
```

`update_rigid()` is a private-API call on a 0.4.6-stable internal
attribute. It is the only per-frame work beyond `cam.render`. The
explicit `np.asarray(..., dtype=np.uint8)` + `np.ascontiguousarray` is
a cheap belt-and-braces guard matching `env/tabletop.py:653` — Genesis
already returns uint8 contiguous, so the operations are near-free.

### 3.4 New module-level constants

Added to `env/genesis/tabletop_genesis.py` next to the existing
`_TABLE_TOP_Z` block:

```python
# Camera pose — semantically matches MuJoCo's main camera
# (`assets/tabletop.xml:23`, pos="0.6 -0.6 0.8", target=(0, 0, 0.42)),
# exact values RFC-006 §3.4 derived for PyBullet parity. Cross-backend
# numerical pixel parity is NOT a goal (RFC-007 §7.3).
_CAM_EYE_BASELINE: tuple[float, float, float] = (0.6, -0.6, 0.8)
_CAM_TARGET:       tuple[float, float, float] = (0.0,  0.0, 0.42)
_CAM_UP:           tuple[float, float, float] = (0.0,  0.0, 1.0)
_CAM_FOV:          float = 45.0
_CAM_NEAR:         float = 0.01
_CAM_FAR:          float = 5.0

# Genesis's default directional light (populated by VisOptions.lights)
# ships at intensity=5.0 on 0.4.6. The lighting_intensity axis is a
# scalar multiplier on this baseline — axis value 1.0 = unchanged
# scene. Semantic match to MuJoCo's light stack and to PyBullet's
# lightDiffuseCoeff (RFC-006 §3.3).
_BASELINE_LIGHT_INTENSITY: float = 5.0

_DEFAULT_RENDER_SIZE: tuple[int, int] = (224, 224)  # matches TabletopEnv + PyBulletTabletopEnv
```

### 3.5 Cube-pair allocation (for `object_texture`)

Today `__init__` allocates a single `self._cube` at
`tabletop_genesis.py:264`. This RFC replaces that with a cube pair:

```python
# Two cubes: red is the "default" texture, green is the "alt". The
# object_texture axis swaps which is visible. The hidden cube lives at
# _DISTRACTOR_HIDDEN_Z — same teleport-away semantic as
# distractor_count (RFC-007 §6.5). self._cube always points at the
# "active" (rest-pose) cube; self._cube_alt points at the hidden one.
self._cube_red: Any = self._scene.add_entity(
    gs.morphs.Box(pos=(0.0, 0.0, _CUBE_REST_Z), size=(2*_CUBE_HALF,)*3),
    surface=gs.surfaces.Default(
        diffuse_texture=gs.surfaces.ColorTexture(color=(1.0, 0.2, 0.2)),
    ),
)
self._cube_green: Any = self._scene.add_entity(
    gs.morphs.Box(pos=(0.0, 0.0, _DISTRACTOR_HIDDEN_Z), size=(2*_CUBE_HALF,)*3),
    surface=gs.surfaces.Default(
        diffuse_texture=gs.surfaces.ColorTexture(color=(0.2, 1.0, 0.2)),
    ),
    material=gs.materials.Rigid(gravity_compensation=1.0),
)
self._cube: Any = self._cube_red
self._cube_alt: Any = self._cube_green
```

`gravity_compensation=1.0` on the hidden cube keeps it from falling
through the plane at `z=-10.0`. The active cube remains dynamic (no
gravity comp) so the existing physics / grasp simulation is unchanged.

`self._cube` stays the canonical handle that `_cube_pos`, `_cube_quat`,
`_snap_cube_to_ee`, and reset's cube teleport all use — no call-site
changes downstream.

### 3.6 Conditional camera at `__init__`

```python
if self._render_in_obs:
    self._camera: Any = self._scene.add_camera(
        res=(self._render_size[1], self._render_size[0]),  # genesis takes (W, H)
        pos=_CAM_EYE_BASELINE,
        lookat=_CAM_TARGET,
        up=_CAM_UP,
        fov=_CAM_FOV,
        near=_CAM_NEAR,
        far=_CAM_FAR,
        GUI=False,
    )
else:
    self._camera = None

self._scene.build()
```

Genesis's `add_camera(res=...)` takes `(width, height)`, not `(H, W)`;
our `render_size` stores `(H, W)` to match MuJoCo. The swap happens at
the single `add_camera` call site, tested in §7 case 1 with an
asymmetric resolution.

`render_in_obs=False` omits `add_camera` entirely — the scene graph and
the rendering cost are both zero for state-only runs. Measured
(exploration §2.1): even an unused attached camera is ~free in step
time, so this is correctness-before-optimisation rather than a measured
bottleneck.

### 3.7 Per-axis branch updates

The cosmetic branches in `_apply_one_perturbation` gain real effects:

```python
elif name == "lighting_intensity":
    self._light_intensity = float(value)
    if self._render_in_obs:
        self._scene.visualizer.rasterizer._context._scene \
            .light_nodes[0].light.intensity = \
            _BASELINE_LIGHT_INTENSITY * self._light_intensity
elif name == "camera_offset_x":
    self._cam_offset[0] = float(value)
    if self._render_in_obs:
        self._apply_camera_pose()
elif name == "camera_offset_y":
    self._cam_offset[1] = float(value)
    if self._render_in_obs:
        self._apply_camera_pose()
elif name == "object_texture":
    new_choice = 1 if round(float(value)) != 0 else 0
    if new_choice != self._texture_choice:
        # Swap active + alt cube handles and positions.
        self._cube.set_pos((0.0, 0.0, _DISTRACTOR_HIDDEN_Z))
        self._cube, self._cube_alt = self._cube_alt, self._cube
        # The new active cube position will be re-set by reset() after
        # this axis branch; the tactical set_pos below keeps the render
        # honest if this branch is the final pre-render action.
        self._cube.set_pos((0.0, 0.0, _CUBE_REST_Z))
    self._texture_choice = new_choice
```

`_apply_camera_pose` is a one-liner helper:

```python
def _apply_camera_pose(self) -> None:
    eye = (
        _CAM_EYE_BASELINE[0] + float(self._cam_offset[0]),
        _CAM_EYE_BASELINE[1] + float(self._cam_offset[1]),
        _CAM_EYE_BASELINE[2],
    )
    self._camera.set_pose(pos=eye, lookat=_CAM_TARGET, up=_CAM_UP)
```

All four branches short-circuit cleanly under `render_in_obs=False`:
they still store `self._light_intensity`, `self._cam_offset`,
`self._texture_choice` so a future `render_in_obs=True` session on the
same env would see the queued state — but the cube-swap
`set_pos` calls fire unconditionally because they touch physical scene
state (the alt cube's teleport is observable via its `get_pos`, so the
swap must happen for state-only callers too, same way
`distractor_count` teleports are always applied).

### 3.8 `VISUAL_ONLY_AXES` → empty

```python
VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset()
```

Was `frozenset({"lighting_intensity", "camera_offset_x",
"camera_offset_y", "object_texture"})`. Dropping it to empty aligns
Genesis with MuJoCo and (post-RFC-006) PyBullet. The Suite loader's
`_reject_purely_visual_suites` becomes a no-op on `tabletop-genesis` —
cosmetic-only sweeps load. Intentionally removes the existing
test-backed rejection (§7 case 7 flips the assertion).

**Honesty caveat preserved at the docstring**: a user running a
cosmetic-only sweep with `render_in_obs=False` will still see
pairwise-identical state obs across cells, since state-only obs is
untouched by the cosmetic axes. Same property `TabletopEnv` and
`PyBulletTabletopEnv` have; not a bug. Docstring of
`GenesisTabletopEnv.VISUAL_ONLY_AXES` points at this.

## 4. Determinism contract

Exploration §7 argues Rasterizer is bit-deterministic given fixed scene
graph, fixed light intensity, fixed camera pose, no RNG at render time.
All four inputs are pinned at render time:

- **Scene transforms** are either baseline poses from `__init__` or set
  via `set_pos` in `reset()` / `_apply_one_perturbation`. No RNG
  touches these after the seed is consumed by the `reset()` randomisation
  pass. `update_rigid()` (§3.3) flushes them before each render.
- **Camera pose** is a pure function of `self._cam_offset`, which is
  set from the perturbation queue (no RNG).
- **Light intensity** is a pure function of `self._light_intensity`,
  same pattern.
- **Cube handle** (`self._cube` vs `self._cube_alt`) is a pure function
  of `self._texture_choice`, same pattern.

Net: `reset(seed=s)` twice on two `GenesisTabletopEnv` instances
produces bit-identical `obs["image"]` arrays. Covered by §7 case 3.

### 4.1 `restore_baseline` interaction

`restore_baseline` today (`tabletop_genesis.py:505`) resets the
visual-axis shadows to baseline. This RFC extends it to also (a) reset
the pyrender light intensity to `_BASELINE_LIGHT_INTENSITY * 1.0`, (b)
reset the camera pose to `(_CAM_EYE_BASELINE, _CAM_TARGET, _CAM_UP)`,
(c) swap the cube handles back so `self._cube == self._cube_red` and
the red cube's pose is at rest, green at hidden. All three resets are
guarded by `if self._render_in_obs:` for the pyrender + camera resets;
the cube-handle reset happens unconditionally (same reason as §3.7 —
state callers observe via `get_pos`).

No new baseline-snapshot plumbing needed — the constants carry the
baseline values directly.

## 5. Per-axis effect table

Update of RFC-007 §6's last column (what changes on `obs`) under
`render_in_obs=True`:

| Axis | State obs effect | `obs["image"]` effect (`render_in_obs=True`) |
|---|---|---|
| `lighting_intensity` | None (§3.7 caveat — no state obs change) | **Yes** — directional light intensity scales. Uniform brightness change. |
| `camera_offset_x` | None | **Yes** — `cam.set_pose` shifts eye. |
| `camera_offset_y` | None | **Yes** — `cam.set_pose` shifts eye. |
| `object_texture` | None | **Yes** — active cube is red (0) or green (1). |
| `object_initial_pose_x` | Yes (cube_pos.x) | Yes (scene layout moves). |
| `object_initial_pose_y` | Yes (cube_pos.y) | Yes. |
| `distractor_count` | Yes (cube interaction with revealed distractors) | Yes — revealed distractors visible. |

All seven axes have at least one observable effect on at least one
observation mode. Matches the post-RFC-006 parity table for
`PyBulletTabletopEnv` and the MuJoCo reference.

## 6. Implementation checklist (atomic commits)

Each bullet → one commit. Sized to the §9 one-task-one-PR cadence.

1. **Exploration doc.** `docs/phase2-exploration-task8-genesis-rendering.md`. **Already shipped on this branch** (`c530a42`).
2. **RFC doc.** This file. The commit introducing this RFC.
3. **Camera + light constants.** Add `_CAM_*` / `_BASELINE_LIGHT_INTENSITY` / `_DEFAULT_RENDER_SIZE` to `env/genesis/tabletop_genesis.py` as module-level constants. Pure additive, no behaviour change. No new tests (constants exercised by later commits).
4. **`render_in_obs` / `render_size` kwargs + `observation_space`.** Widen `__init__`, validate inputs (same error strings as `TabletopEnv`), add `self._render_in_obs` / `self._render_size` state, extend `observation_space` conditionally. No camera add yet (conditional `self._camera = None`). One test: `GenesisTabletopEnv(render_in_obs=True, render_size=(64, 96)).observation_space["image"].shape == (64, 96, 3)`.
5. **Conditional camera at `__init__` + `_render_obs_image` + `_build_obs` guard.** Add the camera when `render_in_obs=True`, implement `_render_obs_image` per §3.3, call it from `_build_obs` when enabled. Add tests for shape/dtype (§7 case 1) and within-run determinism (§7 case 3).
6. **Cube-pair allocation + handle-swap scaffold.** Replace `self._cube` with the `self._cube_red` / `self._cube_green` pair and the pointer `self._cube` + `self._cube_alt` (§3.5). `_apply_one_perturbation("object_texture", ...)` branch gains the swap body (§3.7). `restore_baseline` gains the cube-handle reset. No image-sensitivity tests yet; state-side test: cube physics state after swap is at the expected pose.
7. **Cosmetic-axis render wiring.** Wire `lighting_intensity` private-API write + `camera_offset_{x,y}` `_apply_camera_pose` + `object_texture` cube-swap into the render path (§3.7). `restore_baseline` also resets the pyrender light + camera pose. Add `tests/genesis/test_render_genesis.py` with the four axis-sensitivity tests (§7 case 5).
8. **Cross-backend shape parity test.** `obs["image"]` shape / dtype / bounds match `TabletopEnv(render_in_obs=True).observation_space["image"]` exactly (§7 case 2). Also match `PyBulletTabletopEnv` when that extra is available.
9. **`VISUAL_ONLY_AXES → frozenset()`.** Touch the ClassVar on `GenesisTabletopEnv`. Update `tests/genesis/test_env_genesis.py::<the cosmetic-axis state-invariant test>` — either flip its assertion (obs image now *differs*) or retire it and defer all cosmetic coverage to `test_render_genesis.py`. Update / rename the existing loader test that asserts cosmetic-only sweeps are rejected on `tabletop-genesis` into a "now-loads" acceptance test.
10. **RandomPolicy rendering smoke** in `test_render_genesis.py`: 3 rollouts at 64×64, `render_in_obs=True`, asserts every `obs["image"]` is valid `uint8 (64, 64, 3)` (§7 case 8).
11. **README update + example touch-up.** Two sentences in the Backends section: image obs is available on `tabletop-genesis` via `render_in_obs=True`; semantic parity (not pixel parity) with MuJoCo and PyBullet. Extend `examples/evaluate_random_policy_genesis.py` to run with `render_in_obs=True` through a `partial(GenesisTabletopEnv, render_in_obs=True)` factory.

Budget hits ~11 commits; tight vs the 10-ish target RFC-006 hit. No new
assets, no new subpackage, no new CI job — the existing `genesis-tests`
job picks up the new tests by the `@pytest.mark.genesis` marker.

## 7. Test plan

New file `tests/genesis/test_render_genesis.py`, marker
`@pytest.mark.genesis`, CI job `genesis-tests` covers it.

1. **Shape / dtype / range.** `render_in_obs=True, render_size=(H, W)` → `obs["image"].shape == (H, W, 3)`, `dtype == uint8`, `min >= 0`, `max <= 255`. Image absent when `render_in_obs=False`. Uses an asymmetric `(64, 96)` to catch H/W transposition bugs.
2. **Observation-space parity with MuJoCo.** Construct both `TabletopEnv(render_in_obs=True, render_size=(H, W))` and `GenesisTabletopEnv(render_in_obs=True, render_size=(H, W))`; their `observation_space["image"]` are equal under `gym.spaces.Box.__eq__` (low, high, shape, dtype). Pixel values explicitly **not** compared.
3. **Within-run determinism.** Two independent `GenesisTabletopEnv` instances, both `reset(seed=42)`, assert `np.array_equal` on `obs["image"]`.
4. **Post-step determinism.** Same seed, fixed 20-step rng-seeded action sequence on two instances → `obs["image"]` at step 20 matches byte-for-byte.
5. **Axis sensitivity** — four tests, each `render_size=(64, 64)` to keep CI cheap:
   - `lighting_intensity=0.3` vs `=1.5` → `not np.array_equal`.
   - `object_texture=0.0` vs `=1.0` → `not np.array_equal`.
   - `camera_offset_x=-0.05` vs `=+0.05` → `not np.array_equal`.
   - `camera_offset_y=-0.05` vs `=+0.05` → `not np.array_equal`.
6. **State-only default preserved.** `GenesisTabletopEnv()` (default `render_in_obs=False`) → `"image"` not in `obs`; five existing state keys unchanged. Locks the RFC-007 state-only contract.
7. **Loader rejection relaxed.** `tests/genesis/test_env_genesis.py::<test_rejects_cosmetic_only_sweep_on_genesis>` (or wherever the assertion lives today) flips — a cosmetic-only sweep on `tabletop-genesis` now `load_suite_from_string`s without raising. Fresh parallel test: a 2-axis cosmetic-only Genesis sweep loads cleanly.
8. **End-to-end smoke.** `partial(GenesisTabletopEnv, render_in_obs=True, render_size=(64, 64))` passed into a `Runner`; three rollouts of `RandomPolicy` on a 2×2 axis grid at 2 episodes-per-cell. Every emitted `Episode` carries `success in (True, False)`; no exceptions. Does not need CLI dispatch (library-level).

### 7.1 Mocking strategy

Real `GenesisTabletopEnv` instances everywhere. Rasterizer first-render
compile is ~24 s at 224×224 / ~100 ms at 64×64 (exploration §2.2); it
is paid once per pytest process regardless of how many tests construct
envs. Subsequent renders are ~10 ms at 64×64. The eight tests above
collectively do under ~100 renders. Total added test time budget: under
30 s on the `genesis-tests` job. No MagicMock.

### 7.2 CI

No new job. The existing `genesis-tests` job
(`.github/workflows/ci.yml`, added in RFC-007 §10) picks up the new
tests via the shared `@pytest.mark.genesis` marker.

## 8. Open questions with defaults

- **Q1 — Enable rendering from the CLI?** Not in this RFC. Users
  construct the env factory explicitly, matching MuJoCo and PyBullet.
  Same answer as RFC-006 §8 Q1.
- **Q2 — Match MuJoCo / PyBullet camera pixel-exactly?** No. Different
  rasterisers, different internal math. Semantic parity (same pose,
  same layout, same lighting direction) is the goal. The test plan
  explicitly avoids asserting pixel equality. Same answer as RFC-006
  §8 Q2.
- **Q3 — Should `render_size` be a suite-level YAML field?** No.
  Camera intrinsics are a backend-factory concern; moving them to YAML
  would double up with the existing `partial` mechanism. Same answer
  as RFC-006 §8 Q3.
- **Q4 — Multiprocessing worker semantics.** Each Runner worker
  subprocess constructs its own `GenesisTabletopEnv`, its own
  `gs.Scene`, its own camera, its own pyrender context. First-render
  shader compile is paid per-worker (~24 s at 224×224), amortised across
  all episodes that worker runs. `scene.visualizer.rasterizer._context`
  is scene-local — no cross-worker state. Mirrors RFC-006 §8 Q4.
- **Q5 — Private-API stability on `_context._scene.light_nodes`.** The
  `genesis-world<0.5` pin from RFC-007 holds this surface stable. If
  0.5 rearranges the internals, a runtime `AttributeError` from
  `_render_obs_image` or `restore_baseline` would be obvious; the
  adapter's docstring names the pin as the contract. A future RFC can
  re-wire to a public API when Genesis adds one. Not a gate for 0.4.x.
- **Q6 — Why not `scene.step()` to flush transforms?** `step()`
  advances sim time by `dt * n_substeps = 50 ms`, corrupting reset-time
  state for any free body under gravity. `update_rigid()` is a
  pure push of `set_pos` / `set_quat` transforms into the Rasterizer
  context, no physics integration.
- **Q7 — Do we cache the camera pose across `set_pose` calls?** No
  need. `cam.set_pose` is cheap (<100 µs measured) compared to the
  ~10 ms render itself. Mirrors RFC-006 §8 Q4 (viewMatrix not cached).
- **Q8 — Do we raise if a user hits the cosmetic-only +
  `render_in_obs=False` pairing?** No. Library usage choice, not a bug.
  Same answer as RFC-006 §8 Q6.

## 9. Future work surfaced (out of scope here)

- **Isaac Sim backend.** Still deferred (Phase 2 §7, RFC-005 §2, RFC-007 §2).
- **Genesis public `light.set_intensity` / material swap API.** Upstream
  feature request. If 0.5 lands it, a follow-up can swap our private
  call site for the public one.
- **Shadow / specular / normal-map knobs as perturbation axes.** None
  is in the canonical seven.
- **Cross-backend image-space numerical parity.** Explicitly not a
  goal (RFC-007 §7.3).
- **GPU Rasterizer path (CUDA via BatchRenderer).** Not this RFC.
  RFC-007's CPU-only stance stands.
