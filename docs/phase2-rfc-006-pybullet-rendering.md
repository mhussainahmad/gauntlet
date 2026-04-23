# Phase 2 RFC 006 — PyBullet image observations

- **Status**: Draft
- **Phase**: 2, Task 6 (closes the image-rendering non-goal from RFC-005 §2 / §6.2 / §12 Q1).
- **Author**: architect agent
- **Date**: 2026-04-23
- **Supersedes**: n/a. Builds on RFC-005 without amending it.
- **References**:
  - `docs/phase2-rfc-001-huggingface-policy.md` §5 (MuJoCo `render_in_obs=True` / `render_size` precedent).
  - `docs/phase2-rfc-005-pybullet-adapter.md` §6.2 / §7.4 / §12 Q1 (state-only gap, cross-backend numerical non-parity, the VISUAL_ONLY_AXES rejection).
  - `docs/phase2-exploration-task6-pybullet-rendering.md` (API survey, camera framing, option space for `VISUAL_ONLY_AXES`).

---

## 1. Summary

`PyBulletTabletopEnv` ships state-only (RFC-005 §2 non-goal #3). The four
cosmetic axes — `lighting_intensity`, `object_texture`, `camera_offset_x`,
`camera_offset_y` — reach the env's `_apply_one_perturbation` dispatcher and
land on instance attributes (`_light_intensity`, `_current_texture_id`,
`_cam_eye_offset`), but nothing consumes them and the Suite loader
(`loader.py::_reject_purely_visual_suites`) rejects any sweep whose every
axis is cosmetic. Image-conditioned policies (`HuggingFacePolicy`,
`LeRobotPolicy` — RFC-001/002) cannot be evaluated on the PyBullet backend
today because there is no `obs["image"]` path.

This RFC adds a headless, deterministic image-observation path to
`PyBulletTabletopEnv`:

1. New constructor kwargs `render_in_obs: bool = False` and
   `render_size: tuple[int, int] = (224, 224)`, identical surface and defaults
   to `TabletopEnv` (`env/tabletop.py:160-161`). Default stays off so the
   Phase-1 state-only contract is byte-identical.
2. New `_render_obs_image` method using `p.getCameraImage(..., renderer=p.ER_TINY_RENDERER, shadow=0, flags=p.ER_NO_SEGMENTATION_MASK)` — the only headless-safe, deterministic rendering path (exploration §2).
3. `_build_obs` emits a uint8 `(H, W, 3)` array under `obs["image"]` when enabled; `observation_space["image"]` is a `Box(0, 255, (H, W, 3), uint8)` — shape / dtype / bounds byte-identical to MuJoCo.
4. The four cosmetic axes become **observable on `obs["image"]`** through the render path: `lightDiffuseCoeff` ← `self._light_intensity`; `viewMatrix` ← baseline eye + `self._cam_eye_offset`; cube texture is already rebound in `_apply_one_perturbation("object_texture", ...)` and picked up automatically by the renderer.
5. `PyBulletTabletopEnv.VISUAL_ONLY_AXES` drops to `frozenset()` — exact parity with `TabletopEnv`. The loader's rejection becomes a no-op on PyBullet (exploration §4, **option A**).

What this RFC does **not** do (§2):

- Does not wire the CLI `gauntlet run` subcommand to dispatch by `suite.env`
  (the CLI today runs MuJoCo for every suite regardless of `env:` — a
  pre-existing gap from RFC-005 §11.3 that is orthogonal to rendering and
  will ship as its own bug-fix PR).
- Does not add a `render_in_obs` YAML field (users enable rendering via their
  env factory, exactly like `examples/evaluate_openvla.py` does for MuJoCo).
- Does not chase cross-backend pixel parity (RFC-005 §7.4 holds: semantic
  parity only — same camera pose, same scene layout, same light direction).
- Does not add shadow rendering (exploration §2.3 — shadows are
  non-deterministic on some pybullet wheels; deferred unless a future axis
  needs it).

## 2. Goals / non-goals

### Goals

- Match `TabletopEnv`'s `render_in_obs` surface (kwargs, defaults, obs key,
  space shape/dtype/bounds) byte-for-byte. VLA policies that work on MuJoCo
  via RFC-001 `render_in_obs=True` must work on PyBullet by swapping only
  the env factory (e.g. `partial(PyBulletTabletopEnv, render_in_obs=True)`),
  no policy changes.
- Consume all four cosmetic axes in the render path. A sweep varying only
  `lighting_intensity` and `object_texture` — previously rejected at load
  time on `tabletop-pybullet` — now loads and produces distinct pixels per
  cell.
- Preserve bit-determinism. `reset(seed=s)` twice produces
  `np.array_equal`-equal `obs["image"]` arrays. Same seed + same action
  sequence produces the same per-step image. Matches the within-backend
  contract from RFC-005 §8.
- Keep the state-only default byte-identical to the existing PyBullet
  contract. `render_in_obs=False` (the default) produces an observation
  dict with exactly the existing five keys and no new work at step time.
- Keep `gauntlet.core` pybullet-free. The rendering code lives entirely
  inside `gauntlet.env.pybullet.tabletop_pybullet`; no new module-level
  imports outside the `[pybullet]` extra.
- Keep `mypy --strict` green whether or not the `[pybullet]` extra is
  installed.

### Non-goals

- **CLI `gauntlet run` dispatch by `suite.env`.** Pre-existing gap; will
  land on its own branch. Called out in §11.
- **YAML `render_in_obs` field.** Not added. Users enable rendering by
  constructing a custom env factory, exactly mirroring
  `examples/evaluate_openvla.py`.
- **Cross-backend pixel parity.** RFC-005 §7.4 holds. A test asserts
  shape / dtype parity with MuJoCo but explicitly not pixel equality.
- **Shadow rendering.** Deferred (exploration §2.3). If a future perturbation
  axis needs shadow control, that axis enters a new RFC.
- **Depth or segmentation channels.** Not emitted. The extra passes cost
  rasterisation time and no current policy consumes them.
- **FOV / near / far perturbations.** Camera intrinsics are fixed at
  construction time. Not in the canonical seven axes (RFC-005 §6) — not in
  scope.
- **Apple Silicon determinism.** CI is x86-64 Linux. Cross-arch
  pixel-determinism is not asserted (exploration §5.1).

## 3. The surface

### 3.1 Constructor

```python
class PyBulletTabletopEnv:
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

Validation and error messages match `TabletopEnv.__init__` line-for-line:
`render_in_obs=True` with a non-positive `render_size` raises the same
`ValueError("render_size must be a (height, width) of positive ints; got ...")`
string. This is a deliberate byte-copy — any future change to the MuJoCo
validator must land in both places, enforced by the paired test in §9 case 2.

### 3.2 `observation_space`

```python
obs_spaces = {
    "cube_pos":   spaces.Box(...),  # unchanged
    "cube_quat":  spaces.Box(...),
    "ee_pos":     spaces.Box(...),
    "gripper":    spaces.Box(...),
    "target_pos": spaces.Box(...),
}
if self._render_in_obs:
    h, w = self._render_size
    obs_spaces["image"] = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
```

The `"image"` entry is conditional — present iff `render_in_obs=True`. This
matches `TabletopEnv.observation_space` exactly (`env/tabletop.py:232-237`).

### 3.3 `_render_obs_image` method

```python
def _render_obs_image(self) -> NDArray[np.uint8]:
    """Render the main camera into a uint8 (H, W, 3) array. Deterministic,
    headless (ER_TINY_RENDERER), consumes the four cosmetic axes.
    """
    h, w = self._render_size
    eye = (
        _CAM_EYE_BASELINE[0] + float(self._cam_eye_offset[0]),
        _CAM_EYE_BASELINE[1] + float(self._cam_eye_offset[1]),
        _CAM_EYE_BASELINE[2],
    )
    view = p.computeViewMatrix(
        cameraEyePosition=list(eye),
        cameraTargetPosition=list(_CAM_TARGET),
        cameraUpVector=list(_CAM_UP),
    )
    proj = p.computeProjectionMatrixFOV(
        fov=_CAM_FOV, aspect=float(w) / float(h), nearVal=_CAM_NEAR, farVal=_CAM_FAR,
    )
    _, _, rgb, _, _ = p.getCameraImage(
        width=w,
        height=h,
        viewMatrix=view,
        projectionMatrix=proj,
        lightDirection=[0.0, 0.0, 1.0],
        lightColor=[1.0, 1.0, 1.0],
        lightDiffuseCoeff=float(self._light_intensity),
        lightAmbientCoeff=_CAM_LIGHT_AMBIENT,
        lightSpecularCoeff=0.0,
        shadow=0,
        flags=p.ER_NO_SEGMENTATION_MASK,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=self._client,
    )
    # getCameraImage returns (H, W, 4) uint8; strip alpha for MuJoCo parity.
    arr = np.asarray(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    return np.ascontiguousarray(arr)
```

Module-level camera constants (§3.4) are the only new top-of-file additions.

### 3.4 New module-level constants

```python
# Camera pose — semantically matches TabletopEnv's `<camera name="main" pos="0.6 -0.6 0.8" xyaxes="1 1 0 0 0 1"/>`
# in assets/tabletop.xml. Cross-backend numerical pixel parity is NOT a goal (RFC-005 §7.4).
_CAM_EYE_BASELINE: tuple[float, float, float] = (0.6, -0.6, 0.8)
_CAM_TARGET:       tuple[float, float, float] = (0.0,  0.0, 0.42)  # table-top centre
_CAM_UP:           tuple[float, float, float] = (0.0,  0.0, 1.0)
_CAM_FOV:          float = 45.0
_CAM_NEAR:         float = 0.01
_CAM_FAR:          float = 5.0
_CAM_LIGHT_AMBIENT: float = 0.3  # matches MuJoCo headlight ambient (assets/tabletop.xml:5)

_DEFAULT_RENDER_SIZE: tuple[int, int] = (224, 224)  # matches TabletopEnv default
```

### 3.5 `VISUAL_ONLY_AXES` → empty

```python
VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset()
```

Was `frozenset({"lighting_intensity", "object_texture"})`. Dropping it to
empty aligns PyBullet with MuJoCo (`env/tabletop.py:130`): "the renderer
consumes those axes through `render_in_obs` adapters." That is now true on
both backends. The Suite loader's `_reject_purely_visual_suites` becomes a
no-op on `tabletop-pybullet` — cosmetic-only sweeps load. This intentionally
removes the existing test-backed rejection (§9 case 8 updates the
corresponding assertion).

**Honesty caveat preserved at the docstring**: a user running a cosmetic-only
sweep with `render_in_obs=False` will still see pairwise-identical cells,
since state-only obs is untouched by the cosmetic axes. This is the same
property `TabletopEnv` has had since Phase 1 and is not a bug — the docstring
of `PyBulletTabletopEnv.VISUAL_ONLY_AXES` points at the same invariant.

## 4. Determinism contract

Exploration §5 argues `ER_TINY_RENDERER` is bit-deterministic given fixed
`(viewMatrix, projectionMatrix, scene-state, lightDiffuseCoeff)` and
`shadow=0`. All four inputs are pinned at render time:

- `projectionMatrix` is constant per-instance (fixed at construction from
  `_CAM_FOV` / `_CAM_NEAR` / `_CAM_FAR` / aspect from `render_size`).
- `viewMatrix` depends only on `self._cam_eye_offset`, set in
  `_apply_one_perturbation` from the pending-perturbation queue (no RNG).
- Scene state is the physics state, which the existing Phase 2 Task 5 tests
  already demonstrate is bit-deterministic within a backend (`tests/pybullet/test_determinism_pybullet.py`).
- `lightDiffuseCoeff` is `self._light_intensity`, set the same way.
- `shadow=0` is unconditional.

Net: `reset(seed=s)` twice produces bit-identical `obs["image"]`. Covered by
§9 case 3.

### 4.1 restore_baseline interaction

`restore_baseline()` already resets `_light_intensity = 1.0` and
`_cam_eye_offset = np.zeros(2)` at `tabletop_pybullet.py:519-521` and
rebuilds the scene via `_build_scene()`, which rebinds the default cube
texture at `:294`. All three render-input reset paths exist; no new
baseline-snapshot plumbing needed.

## 5. Per-axis effect table

Update of RFC-005 §6's last column (what changes on `obs`) under
`render_in_obs=True`:

| Axis | State obs effect | `obs["image"]` effect (`render_in_obs=True`) |
|---|---|---|
| `lighting_intensity` | None (§6.2 caveat) | **Yes** — `lightDiffuseCoeff` shifts. Uniform brightness change. |
| `camera_offset_x` | None | **Yes** — view matrix shifts; scene parallax. |
| `camera_offset_y` | None | **Yes** — view matrix shifts; scene parallax. |
| `object_texture` | None | **Yes** — cube's bound texture UID flipped (red default / green alt). |
| `object_initial_pose_x` | Yes (cube_pos.x) | Yes (scene layout moves). |
| `object_initial_pose_y` | Yes (cube_pos.y) | Yes. |
| `distractor_count` | Yes (only on collision) | Yes — revealed distractors appear as coloured boxes. |

All seven axes have at least one observable effect on at least one observation
mode — the RFC-005 §6.2 caveat is closed.

## 6. Implementation checklist (~9 commits)

Each bullet → one commit. Sized to the RFC-005 §9 one-task-one-PR cadence.

1. **Exploration doc.** `docs/phase2-exploration-task6-pybullet-rendering.md`. **Already shipped on this branch** (`813614b`).
2. **RFC doc.** This file. The commit introducing this RFC.
3. **Camera + light constants.** Add `_CAM_*` / `_DEFAULT_RENDER_SIZE` to `env/pybullet/tabletop_pybullet.py` as module-level constants. Pure additive, no behaviour change. No new tests (constants exercised by later commits).
4. **`render_in_obs` / `render_size` kwargs.** Widen `__init__`, validate inputs (same error strings as `TabletopEnv`), add `self._render_in_obs` / `self._render_size` state, extend `observation_space` conditionally. No render call yet. One test: `PyBulletTabletopEnv(render_in_obs=True, render_size=(64, 96)).observation_space["image"].shape == (64, 96, 3)`.
5. **`_render_obs_image`.** Implement per §3.3. Called only from `_build_obs` guarded by `self._render_in_obs`. Add tests for shape/dtype (§9 case 1) and within-run determinism (§9 case 3).
6. **Axis sensitivity tests.** `tests/pybullet/test_render_pybullet.py` — four tests, one per cosmetic axis, each asserting distinct pixels between two reset/step pairs varying only that axis (§9 case 5).
7. **Cross-backend shape parity test.** `obs["image"]` shape/dtype/bounds match `TabletopEnv(render_in_obs=True).observation_space["image"]` exactly (§9 case 2). This is the cross-backend contract the VLA policies rely on.
8. **`VISUAL_ONLY_AXES → frozenset()`.** Touch the ClassVar on `PyBulletTabletopEnv`. Update `tests/pybullet/test_loader_pybullet.py::test_rejects_suite_whose_every_axis_is_visual_only` to assert the sweep now **loads** (test rename + assertion flip). Add a new test: a cosmetic-only sweep loads cleanly.
9. **RandomPolicy rendering smoke.** `tests/pybullet/test_render_pybullet.py::test_random_policy_smoke_with_rendering` — 3 rollouts, `render_in_obs=True`, asserts every `obs["image"]` is valid uint8 `(224, 224, 3)` (§9 case 8).
10. **README update.** Two sentences in the Backends section: image obs is available on `tabletop-pybullet` via `render_in_obs=True`, semantic parity (not pixel parity) with MuJoCo. Same section points to a new example file.
11. **Example file.** `examples/evaluate_smolvla_pybullet.py` — minimal "PyBullet analogue of `evaluate_smolvla.py`" using `partial(PyBulletTabletopEnv, render_in_obs=True, render_size=(512, 512))`. Exercises the full integration path end-to-end (library-level; doesn't need CLI dispatch).

Budget hits ~11 commits; tight vs the 10-ish target. No new assets, no new
subpackage, no new CI job — the existing `pybullet-tests` job picks up the
new tests by the `@pytest.mark.pybullet` marker.

## 7. Test plan

New file `tests/pybullet/test_render_pybullet.py`, marker
`@pytest.mark.pybullet`, CI job `pybullet-tests` covers it.

1. **Shape / dtype / range.** `render_in_obs=True, render_size=(H, W)` → `obs["image"].shape == (H, W, 3)`, `dtype == uint8`, `min >= 0`, `max <= 255`. Image absent when `render_in_obs=False`.
2. **Observation-space parity with MuJoCo.** Construct both `TabletopEnv(render_in_obs=True, render_size=(H, W))` and `PyBulletTabletopEnv(render_in_obs=True, render_size=(H, W))`; their `observation_space["image"]` are equal under `gym.spaces.Box.__eq__` (low, high, shape, dtype). Pixel values explicitly **not** compared (RFC-005 §7.4).
3. **Within-run determinism.** Two independent `PyBulletTabletopEnv` instances, both `reset(seed=42)`, assert `np.array_equal` on `obs["image"]`.
4. **Post-step determinism.** Same seed, fixed 20-step rng-seeded action sequence on two instances → `obs["image"]` at step 20 matches byte-for-byte.
5. **Axis sensitivity** — four tests:
   - `lighting_intensity=0.3` vs `=1.5` → `not np.array_equal`.
   - `object_texture=0.0` vs `=1.0` → `not np.array_equal`.
   - `camera_offset_x=-0.05` vs `=+0.05` → `not np.array_equal`.
   - `camera_offset_y=-0.05` vs `=+0.05` → `not np.array_equal`.
6. **State-only default preserved.** `PyBulletTabletopEnv()` (default `render_in_obs=False`) → `"image"` not in `obs`; five other keys unchanged. Locks the Phase-1 state-only contract (§2 non-goal preservation).
7. **Loader rejection relaxed.** `tests/pybullet/test_loader_pybullet.py::test_rejects_suite_whose_every_axis_is_visual_only` → renamed to `test_accepts_suite_with_only_cosmetic_axes`, asserts the same YAML now `load_suite_from_string`s without raising. Fresh test `test_mujoco_backend_accepts_cosmetic_only_suites` (already exists; unchanged) now has a twin on the PyBullet side.
8. **End-to-end smoke.** `partial(PyBulletTabletopEnv, render_in_obs=True)` passed into a `Runner`; three rollouts of `RandomPolicy` on a 2×2 axis grid at 2 episodes-per-cell. Every emitted `Episode` carries `success in (True, False)`; no exceptions. Does not need CLI dispatch (library-level).

### 7.1 Mocking strategy

Real `PyBulletTabletopEnv` instances everywhere. `ER_TINY_RENDERER` at
224×224 costs ~10–25 ms per `_render_obs_image` call; the eight tests above
collectively do ≤~50 renders. Total added test time budget: well under 5
seconds on the existing `pybullet-tests` job. No MagicMock.

### 7.2 CI

No new job. The existing `pybullet-tests` job (`.github/workflows/ci.yml`)
picks up the new tests via the shared marker.

## 8. Open questions with defaults

- **Q1 — Enable rendering from the CLI?** Not in this RFC. Users construct the env factory explicitly, matching the MuJoCo precedent (`examples/evaluate_openvla.py`). A `--render-in-obs` CLI flag can land alongside the `suite.env`-dispatch bug-fix.
- **Q2 — Match MuJoCo camera pixel-exactly?** No. Different rasterisers, different internal math. Semantic parity (same pose, same layout, same light) is the goal. The test plan explicitly avoids asserting pixel equality.
- **Q3 — Should `render_size` be a suite-level YAML field?** No. Camera intrinsics are a backend-factory concern; moving them to YAML would double up the resolution knob with the existing `partial` mechanism. Users who want 512×512 write `partial(PyBulletTabletopEnv, render_in_obs=True, render_size=(512, 512))` — same pattern as `examples/evaluate_smolvla.py`.
- **Q4 — Do we cache the `viewMatrix` / `projectionMatrix`?** `projectionMatrix` yes (constant per instance — compute once in `__init__`, cache on `self._proj_matrix`). `viewMatrix` no — it depends on the camera offset which is set per-reset; the compute is cheap (< 10 µs) compared to the render.
- **Q5 — Per-step alpha stripping cost?** `arr[:, :, :3]` is a view, not a copy. `np.ascontiguousarray` upgrades to a contiguous view-or-copy; the contiguous copy is ≈ 150 kB / 224×224×3 — one frame worth of memcpy per step, negligible next to the render pass itself.
- **Q6 — Do we raise if a user hits the cosmetic-only + `render_in_obs=False` pairing?** No. That is a library usage choice, not a bug — matches MuJoCo's today. Documented in the class docstring; no runtime warning.

## 9. Future work surfaced (out of scope here)

- **CLI `suite.env` dispatch.** Separate bug-fix PR.
- **CLI `--render-in-obs` flag.** After the dispatch fix, probably as a small
  follow-up on the same branch chain.
- **Genesis backend.** RFC-007 — the next substantive Phase-2 task after
  this one lands. Will use the same `render_in_obs` / `render_size`
  surface; the Protocol does not change.
- **Shadow rendering as an optional knob.** If a future axis needs it.
- **Depth / segmentation channels.** When a policy adapter actually
  consumes them; none does today.
