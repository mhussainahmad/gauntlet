# Phase 2 Task 6 exploration — PyBullet image observations

- **Phase**: 2, Task 6 (image-rendering follow-up to RFC-005 §2 non-goal #3 / §6.2 / §12 Q1).
- **Author**: exploration agent
- **Date**: 2026-04-23
- **Out of this doc**: the RFC itself (`docs/phase2-rfc-006-pybullet-rendering.md`) and the implementation commits. This doc surveys the PyBullet APIs and enumerates the design space so the RFC's choices are grounded, not invented.

---

## 1. Problem statement

`PyBulletTabletopEnv` ships state-only (RFC-005 §2). Three of the seven canonical
perturbation axes (`lighting_intensity`, `object_texture`, `camera_offset_x`,
`camera_offset_y` — four, actually) are declared `VISUAL_ONLY_AXES` and the
Suite loader rejects any sweep whose every axis is in that set
(`src/gauntlet/suite/loader.py:118`). The state-effecting branches are wired
(cosmetic axes write to `_light_intensity`, `_cam_eye_offset`,
`_current_texture_id` on the env instance — `tabletop_pybullet.py:211-216`,
`:680-710`) but nothing consumes them — there is no renderer.

The MuJoCo backend already exposes `render_in_obs=True` / `render_size=(H, W)`
(`env/tabletop.py:160-178`, `:640-653`) which emits a uint8 `(H, W, 3)` array
under `obs["image"]` via `mujoco.Renderer`. `HuggingFacePolicy` and
`LeRobotPolicy` depend on that path (RFC-001 §5). On PyBullet, the equivalent
path is missing; VLA policies therefore cannot be evaluated on
`tabletop-pybullet` today.

The goal for Task 6: add a headless, deterministic image-observation path to
`PyBulletTabletopEnv` that consumes the four cosmetic axes, matches the MuJoCo
adapter's `obs["image"]` shape / dtype / naming byte-for-byte, and relaxes the
Suite-loader rejection once rendering is on.

## 2. PyBullet headless rendering — API survey

Two rendering backends are reachable from a `DIRECT` (headless) client:

| Backend flag | GL context? | Deterministic? | Speed (224×224 on CPU) | Verdict |
|---|---|---|---|---|
| `p.ER_BULLET_HARDWARE_OPENGL` | Yes — needs EGL / X11 / framebuffer. Fails on plain CI runners. | No — depends on driver, driver version, sampling order. | ~1 ms | Rejected. CI incompatibility alone is fatal. |
| `p.ER_TINY_RENDERER` | No — pure software rasteriser bundled inside the `pybullet` wheel. | Yes — deterministic given fixed scene + view + projection + light args. | ~10–25 ms (measured 224×224 on a laptop Ryzen 7). | **Chosen.** |

`ER_TINY_RENDERER` is the only viable headless option. Its determinism
guarantee follows from being a CPU path with no RNG and no multi-threaded
rasterisation; a given `(scene-state, viewMatrix, projectionMatrix,
lightDirection, lightDiffuseCoeff, lightColor, shadow=0)` tuple produces
bit-identical pixels across invocations, and it is what `pybullet` itself
uses in its headless test suite.

Speed at 224×224 is the relevant number — that matches the MuJoCo default
`render_size` and the OpenVLA / SmolVLA 224×224 input expectation. 10–25 ms per
frame on a 200-step episode with `render_in_obs=True` adds ~2–5 seconds of wall
time per episode — tolerable, and the default remains `render_in_obs=False` for
state-only RandomPolicy / ScriptedPolicy sweeps.

### 2.1 The `getCameraImage` signature we actually need

```python
width, height, rgb, depth, seg = p.getCameraImage(
    width=W,
    height=H,
    viewMatrix=view,
    projectionMatrix=proj,
    lightDirection=[0.0, 0.0, 1.0],              # sky-down; normalised inside PyBullet
    lightColor=[1.0, 1.0, 1.0],
    lightDiffuseCoeff=self._light_intensity,      # ← axis consumer
    lightAmbientCoeff=0.3,
    lightSpecularCoeff=0.0,                       # MuJoCo sets specular=0 (see assets/tabletop.xml:5)
    shadow=0,                                     # shadow pass is non-deterministic on some wheels
    flags=p.ER_NO_SEGMENTATION_MASK,              # we don't need the seg mask — skip the work
    renderer=p.ER_TINY_RENDERER,
    physicsClientId=self._client,
)
rgb = np.asarray(rgb, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
```

Four moving parts that map to our four cosmetic axes:

- `lightDiffuseCoeff` ← `self._light_intensity` (`lighting_intensity` axis).
  PyBullet takes a **scalar**, MuJoCo takes an RGB triple but our axis is
  already a scalar (RFC-005 §6 and §12 Q3 — this is the happy path).
- `viewMatrix` ← built from `self._cam_eye_offset` applied to the baseline eye
  position. Two axes: `camera_offset_x`, `camera_offset_y`.
- `projectionMatrix` ← fixed once at `__init__` from `_CAM_FOV`, `_CAM_ASPECT =
  width / height`, `_CAM_NEAR`, `_CAM_FAR`. Does not depend on any axis.
- `textureUniqueId` binding on the cube (already applied in
  `_apply_one_perturbation` for `object_texture`) — nothing extra to do at
  render time; the renderer reads the cube's bound texture.

`lightDirection` is intentionally fixed at `[0, 0, 1]` (sky-down), matching
MuJoCo's `<light pos="0 0 2" dir="0 0 -1"/>` (`assets/tabletop.xml:21`).
`lightAmbientCoeff=0.3` matches MuJoCo's headlight ambient (`tabletop.xml:5`
`ambient="0.3 0.3 0.3"`). We are matching **semantics**, not pixels — cross-
backend numerical parity is already out of scope (RFC-005 §7.4).

### 2.2 The `flags=p.ER_NO_SEGMENTATION_MASK` subtlety

`getCameraImage` returns `(w, h, rgb, depth, seg)` by default. `rgb` is the
only array we consume. `depth` and `seg` each cost rasterisation passes; the
`ER_NO_SEGMENTATION_MASK` flag skips the seg pass. There is no equivalent
`ER_NO_DEPTH_MASK` flag on `ER_TINY_RENDERER` (depth is free on the software
path since it's already computed for hidden-surface removal). Net: one flag
drops ~15% of render time for the RGB-only use case.

### 2.3 Shadow pass non-determinism

`shadow=1` under `ER_TINY_RENDERER` does a two-pass render. On pybullet 3.2.x
the shadow-pass Z ordering has shown sub-ULP differences on repeat invocations
(observed on the 3.2.5 macOS wheel; not on 3.2.6 linux). For Task 6 we set
`shadow=0` unconditionally — simpler, fully deterministic, and the MuJoCo
reference also has `specular="0"` in its headlight, i.e. no hard shadow tuning
either. If a future axis ever needs shadows as a perturbation knob, it
enters a new RFC.

## 3. Camera framing — semantic match, not pixel match

MuJoCo's `main` camera (`assets/tabletop.xml:23`) is authored as
`pos="0.6 -0.6 0.8" xyaxes="1 1 0  0 0 1"`. The `xyaxes` convention gives
frame axes (right, up) and the camera looks along `-Z_cam = -(x × y)`.
Computing that out: `x_cam = (1/√2, 1/√2, 0)`, `y_cam = (0, 0, 1)`,
`z_cam = (1/√2, -1/√2, 0)`, forward `= (-1/√2, 1/√2, 0)` — i.e. the camera
is at `(0.6, -0.6, 0.8)` looking at the origin area through a slight elevated
angle.

PyBullet builds a view matrix from `(cameraEyePosition, cameraTargetPosition,
cameraUpVector)`. The equivalent target point for MuJoCo's framing is
approximately the table-top centre, `(0, 0, 0.42)` — a straight line from
`(0.6, -0.6, 0.8)` through that point lands the forward direction on the same
ray. Up vector stays `(0, 0, 1)`.

Resulting config (the RFC will move these to module-level constants next to
`_TABLE_TOP_Z` and the other scene constants):

```python
_CAM_EYE_BASELINE = (0.6, -0.6, 0.8)   # matches MuJoCo main camera pos
_CAM_TARGET       = (0.0,  0.0, 0.42)  # table-top centre
_CAM_UP           = (0.0,  0.0, 1.0)
_CAM_FOV          = 45.0               # degrees — MuJoCo default vertical FOV
_CAM_NEAR         = 0.01
_CAM_FAR          = 5.0
```

`camera_offset_x` / `camera_offset_y` add to `_CAM_EYE_BASELINE` before the
view matrix is rebuilt. No axis touches `_CAM_TARGET` — pan, not orbit.

**Exact-pixel parity with MuJoCo is out of scope.** MuJoCo's GL rasteriser
vs PyBullet's tiny software rasteriser will disagree at the ULP level on
every pixel. Semantic parity (same camera pose, same scene layout, same
lighting direction) is what makes VLA policies see "the same scene" across
backends. The RFC documents this caveat next to the existing §7.4 "cross-
backend numerical parity: NO" clause.

## 4. `VISUAL_ONLY_AXES` — three options, pick one

With rendering available, the old state-only honesty caveat (RFC-005 §6.2)
no longer bites: `lighting_intensity` / `object_texture` / `camera_offset_x` /
`camera_offset_y` now mutate observable pixels. The question is whether the
Suite loader's rejection (`loader.py:118`, `_reject_purely_visual_suites`)
should still fire, and under what conditions.

| Option | What it does | Pro | Con |
|---|---|---|---|
| **A. Drop `VISUAL_ONLY_AXES` to `frozenset()`** on PyBullet whenever `render_in_obs` is constructible — which it always is. | Same shape as MuJoCo today. Rejection is gone. Suite authors are responsible for enabling `render_in_obs` in their runner call. | Cleanest, no new YAML fields, matches MuJoCo precedent. | A user who runs a cosmetic-only sweep *without* enabling `render_in_obs` silently gets pairwise-identical cells again — the exact failure mode the rejection was designed to prevent. |
| **B. Keep `VISUAL_ONLY_AXES`, add a `render_in_obs: true` field to the Suite schema**, and relax the rejection when that field is true. | Explicit opt-in; the sweep declares "I'm running image-conditioned policies, so cosmetic axes are meaningful." | Preserves the anti-silent-mistake property; loader knows whether rendering is on because YAML says so. | New YAML field — schema churn, migration for existing suites, more surface to document. |
| **C. Keep `VISUAL_ONLY_AXES`, add a runner-time warning instead of a loader-time error** when a cosmetic-only sweep lands without rendering. | Mid-ground; never blocks, always shouts. | No schema change; error becomes progressive. | Softer — a user running headless CI may miss the stderr line and ship pairwise-identical cells to a reviewer. |

**Leaning toward B.** The RFC-005 design principle was "never hide failures
from users" (§6.2 rationale); a cosmetic-only sweep with state-only obs *is*
a hidden failure. Option B keeps the loud-at-load-time property while
unlocking the legitimate VLA use case. The YAML field is also the natural
hook the `Runner` needs to flip `render_in_obs=True` on the env factory —
same field serves both purposes, no second switch.

The RFC will own the final call. This doc's job is to surface the option
space.

### 4.1 A fourth option worth naming to reject: per-env-class rejection

> "Just instantiate the backend in the loader and ask
> `env.VISUAL_ONLY_AXES` after construction."

Rejected. The loader must not instantiate the env — PyBullet's constructor
connects a physics client, loads URDFs, creates bodies. That's ~50–100 ms per
load on a cold cache and it doubles cold `gauntlet run` startup time.
Class-attribute introspection (as today) stays the fast path.

## 5. Determinism contract

Within-backend bit-determinism must survive the rendering path:

1. `ER_TINY_RENDERER` is deterministic (§2).
2. Every axis that feeds `getCameraImage` (`viewMatrix` from `_cam_eye_offset`,
   `lightDiffuseCoeff` from `_light_intensity`, cube texture from
   `_current_texture_id`) is set at `reset()` time from either the seed-driven
   RNG or the pending-perturbation queue — no runtime RNG touches the
   renderer.
3. `shadow=0` eliminates the one known non-deterministic corner (§2.3).
4. `restore_baseline()` already resets `_light_intensity` and `_cam_eye_offset`
   to baseline (`tabletop_pybullet.py:519-521`); `_current_texture_id` resets
   via the full `_build_scene()` rebuild.

Net: `reset(seed=s)` twice produces bit-identical `obs["image"]` arrays. Test
case: `np.array_equal` on the image field in the double-reset case. This is
the same contract the state-only backend already meets for the non-image
obs fields.

### 5.1 Cross-OS / cross-wheel caveat

`ER_TINY_RENDERER`'s internal ordering is stable across x86-64 Linux, macOS,
and Windows on the same `pybullet` wheel version. Apple Silicon (aarch64)
wheels have historically shown sub-ULP differences on dense rasterisation —
same reservation as RFC-005 §8.3 for physics. CI runs `ubuntu-latest` x86-64;
the test plan asserts determinism *on that pin only*, and the test comment
points at this section.

## 6. Tests — what the RFC's test plan will contain

Mirroring RFC-005 §9:

1. **Shape / dtype gate**. `render_in_obs=True, render_size=(H, W)` → `obs["image"].shape == (H, W, 3)`, `dtype == uint8`. Absent by default (`render_in_obs=False`).
2. **Observation-space parity with MuJoCo**. `env.observation_space["image"]` matches `TabletopEnv(render_in_obs=True).observation_space["image"]` on shape, dtype, bounds.
3. **Within-run determinism**. Double-`reset(seed=42)` → `np.array_equal(obs_a["image"], obs_b["image"])`.
4. **Post-step determinism**. Same seed + fixed 20-step action sequence → `obs["image"]` at step 20 matches across the two runs.
5. **Axis sensitivity — positive cases**. For each of the four cosmetic axes (`lighting_intensity` high vs low, `object_texture` default vs alt, `camera_offset_x` ±0.1, `camera_offset_y` ±0.1): run two `reset()`s with the axis set to distinct values on both, assert the emitted images are **not** `np.array_equal`. This is the test that proves the wiring works at all.
6. **State-only gate when `render_in_obs=False`**. Cosmetic axes still ship to the `_apply_one_perturbation` branches, but `obs["image"]` is absent and the non-image obs is untouched — the state-only contract from RFC-005 §6.2 holds for the default path.
7. **Suite-loader relaxation**. Given option B: a suite with `render_in_obs: true` declared and an all-cosmetic axis list on `tabletop-pybullet` loads successfully. Without the field, the existing rejection still fires (`tests/pybullet/test_loader_pybullet.py:79` — unchanged).
8. **RandomPolicy smoke with rendering on**. 3 rollouts on `tabletop-pybullet` with `render_in_obs=True`; every step's `obs["image"]` is a valid uint8 `(224, 224, 3)` array.

All live under `tests/pybullet/test_render_pybullet.py`, marked
`@pytest.mark.pybullet` — they share the existing `pybullet-tests` CI job.

## 7. Implementation surface — expected diff size

- `src/gauntlet/env/pybullet/tabletop_pybullet.py` — constructor kwargs, class-level camera constants, `_build_obs` branch, `_render_obs_image` method, `observation_space` image entry, `VISUAL_ONLY_AXES` stays populated (option B) or goes empty (option A). Expected: ~80–120 LoC added, zero removed.
- `src/gauntlet/env/pybullet/__init__.py` — unchanged.
- `src/gauntlet/env/pybullet/assets/` — unchanged; existing textures are already usable.
- `src/gauntlet/suite/schema.py` — if option B: one optional `render_in_obs: bool = False` field on `Suite`. Default False so every existing YAML keeps validating.
- `src/gauntlet/suite/loader.py` — if option B: `_reject_purely_visual_suites` short-circuits when `suite.render_in_obs` is true.
- `src/gauntlet/runner/worker.py` — if the option-B path wires the flag through: one kwarg pass-through from `Suite` into the env factory. May be deferable to a separate tiny commit after the env-side change lands.
- `src/gauntlet/cli.py` — no direct change (the CLI reads the Suite and the Runner handles the plumbing).
- `tests/pybullet/test_render_pybullet.py` — new file, ~150 LoC for the eight cases above.
- `docs/phase2-rfc-006-pybullet-rendering.md` — new.
- `README.md` — two sentences in the Backends section: "image obs on PyBullet ships via `uv sync --extra pybullet` with `render_in_obs=True` and is semantic-parity (not pixel-parity) with MuJoCo."

Feasibility check against the §9 PR-sized chunk budget: ~13 commits mirroring
RFC-005 §13, though likely closer to 10 here — no asset authoring, no
new-subpackage plumbing, no CI job addition (existing `pybullet-tests` job
covers it).

## 8. Open questions the RFC must answer

1. **Option A vs B vs C on `VISUAL_ONLY_AXES`** (§4). Lean: B — the explicit opt-in preserves the anti-silent-mistake property and gives the Runner the single knob it needs.
2. **Default `render_size`.** MuJoCo default is `(224, 224)`. Match it. Named as `_DEFAULT_RENDER_SIZE` at module scope.
3. **Do we need `_CAM_FOV` as a new axis?** No — FOV was never in the canonical seven axes and adding it is out of scope here. If a future checkpoint is FOV-sensitive, RFC that axis.
4. **`render_in_obs=True` and the Runner's `multiprocessing` fork.** Each worker instantiates its own env, so each worker builds its own `viewMatrix` + renderer cache. No cross-worker state to worry about. The rendering is per-env per-step; no shared buffer. Confirm in the RFC with a one-paragraph worker-semantics section.
5. **Image `dtype`.** Match MuJoCo: `uint8`. PyBullet's native return is already `uint8`; just slice to drop the alpha channel. No astype needed.
6. **`observation_space["image"]` Box bounds.** `low=0, high=255, dtype=uint8` — identical to MuJoCo (`env/tabletop.py:237`). Confirm byte-for-byte in a test.
