# Phase 2 Task 8 exploration — Genesis image observations

- **Phase**: 2, Task 8 (image-rendering follow-up to RFC-007 §6 / §9).
- **Author**: exploration agent
- **Date**: 2026-04-23
- **Out of this doc**: the RFC itself (`docs/phase2-rfc-008-genesis-rendering.md`) and the implementation commits. This doc surveys the Genesis rendering APIs on the pinned `genesis-world==0.4.6` wheel and enumerates the design space so the RFC's choices are grounded, not invented.

---

## 1. Problem statement

`GenesisTabletopEnv` ships state-only (RFC-007 §2 non-goal, §6 carve-out).
Four of the seven canonical perturbation axes (`lighting_intensity`,
`object_texture`, `camera_offset_x`, `camera_offset_y`) are declared
`VISUAL_ONLY_AXES` (`tabletop_genesis.py:186-193`). Their `set_perturbation`
branches queue and validate, and `_apply_one_perturbation` stores them on
`self._light_intensity`, `self._cam_offset[0/1]`, `self._texture_choice`
(`:543-554`), but nothing consumes those values — there is no renderer.
The Suite loader rejects any sweep whose every axis is purely cosmetic
(`loader.py`), so `tabletop-genesis` cannot run a cosmetic-only sweep at
all, and VLA adapters (`HuggingFacePolicy`, `LeRobotPolicy` — RFC-001/002)
cannot be evaluated on the Genesis backend today.

The MuJoCo backend exposes `render_in_obs=True` / `render_size=(H, W)`
(`env/tabletop.py:160-178`, `:640-653`) via `mujoco.Renderer`. The
PyBullet backend matches it via `ER_TINY_RENDERER` (RFC-006 §3.3). Both
emit uint8 `(H, W, 3)` arrays under `obs["image"]`. On Genesis, the
equivalent path is missing.

The goal for Task 8: add a headless, deterministic image-observation path
to `GenesisTabletopEnv` that consumes all four cosmetic axes, matches the
MuJoCo / PyBullet `obs["image"]` shape / dtype / naming byte-for-byte, and
drops `VISUAL_ONLY_AXES` to `frozenset()` (same endpoint RFC-006 hit for
PyBullet).

## 2. Genesis rendering — API survey

Measurements below were run against `genesis-world==0.4.6` on CPU
(`gs.init(backend=gs.cpu)`) in this repo's `.venv`. A minimal scene
(`Plane` + one `Box`) plus a single 224×224 pinhole camera was built,
then rendered repeatedly. All numbers are from that setup unless noted.

### 2.1 Two renderer back-ends, one viable

Genesis 0.4.6 exposes three `RendererOptions`: `Rasterizer`, `RayTracer`,
and `BatchRenderer` (`genesis.options.renderers`). Measured behaviour:

| Renderer | Headless? | Deterministic? | Cold first render (224×224, CPU) | Hot render | Verdict |
|---|---|---|---|---|---|
| `Rasterizer` (**default**) | Yes — pyrender-backed software GL context, no display required. | Yes — back-to-back `cam.render()` returns bit-identical arrays (measured `np.array_equal` true). | ~24 s (one-time shader/JIT compile). | ~23 ms. | **Chosen.** |
| `RayTracer` | Yes, but pulls CUDA path tracer. | Yes on CPU. | Much slower; minutes for first frame at 224×224. | Seconds. | Rejected — CI budget. Also `add_light` on our scene raised `GenesisException: This method is only supported by BatchRenderer. Please use 'add_mesh_light' when using RayTracer.` on 0.4.6 — the light API is fractured across back-ends. |
| `BatchRenderer` | Yes, but pulls torch/CUDA batched renderer. | Not validated on CPU in this exploration. | Comparable to Rasterizer. | Comparable. | Rejected — torch-on-CPU bloat, off-menu; Rasterizer already meets the contract. |

**Rasterizer is the only viable default.** It is already Genesis's default
`renderer=` (verified — constructing `gs.Scene(show_viewer=False)` and
inspecting `type(s._visualizer._renderer).__name__` returned
`'Rasterizer'`), so no new option needs to be passed from our adapter —
just leave `renderer=` unspecified on `gs.Scene(...)`.

Its determinism is a property of the pure-CPU pyrender back-end with no
RNG at render time: given a fixed scene graph, fixed camera pose, fixed
light intensity, and no `castshadow` non-determinism in the default
directional light, two renders produce byte-identical output. The RFC's
determinism contract is built on this.

### 2.2 First-render cost is one-time per process

The measured ~24 s first-render latency at 224×224 is a shader /
taichi-JIT compile that happens once per Python process on first
`cam.render()`. Subsequent renders (same process, same or different
resolution) pay only ~10–25 ms at 224×224. Test fixtures that construct
and destroy multiple `GenesisTabletopEnv` instances inside one pytest
process amortise the cost — only the first env pays it. CI's
`genesis-tests` job (RFC-007 §10) runs `pytest -m genesis -q` which
spins up a single process; the cost is paid once.

For test files, we use small `render_size` (`(64, 64)`) to keep
first-render low; measured: ~100 ms. Production `render_size` stays
`(224, 224)` to match MuJoCo and the VLA input convention.

### 2.3 The `cam.render` signature we need

```python
rgb, depth, seg, normal = cam.render(
    rgb=True,
    depth=False,
    segmentation=False,
    colorize_seg=False,
    normal=False,
    antialiasing=False,
    force_render=False,
)
```

`cam.render` returns a 4-tuple. `rgb` is a contiguous `(H, W, 3)` uint8
`np.ndarray` — the exact shape/dtype contract MuJoCo and PyBullet
already satisfy. `depth` / `segmentation` / `normal` are `None` when
their kwargs are False, so the kwargs default to cheap. Setting
`rgb=True` with everything else default is our one and only call site.

`force_render=False` is fine — Genesis's Rasterizer already honours the
determinism contract without forcing a re-render; empirically, two
back-to-back `cam.render(rgb=True)[0]` calls with `force_render` omitted
return `np.array_equal`-equal arrays.

### 2.4 Scene-state flush before rendering

Non-obvious (this caught me during exploration): after
`entity.set_pos(...)`, a subsequent `cam.render()` **does not see the
updated transforms** unless either (a) a `scene.step()` has run since
the set_pos, or (b) the rendering context's rigid buffer has been
flushed manually. `reset()` on `GenesisTabletopEnv` today calls
`set_pos` on cube, target, EE, and all ten distractors without a
following `step()`, so a naive `cam.render()` post-reset would see
stale transforms.

The fix that avoids advancing sim time is
`self._scene.visualizer.rasterizer._context.update_rigid()`. Measured:
calling `update_rigid()` after a distractor teleport pushes the new
transforms into the Rasterizer and subsequent renders reflect them.
This is a Genesis-private API (the `_context`); we honesty-flag that
in the RFC and pin `genesis-world<0.5` (already the RFC-007 pin) so the
dependency is captured.

Alternative: call `scene.step()` once at the end of `reset()`. Rejected —
it advances sim time by `dt * n_substeps = 50 ms` per reset, which is
observable through `_step_count` drift and `cube.get_pos()` changing
under the physics solver's gravity integration for free-falling bodies.
Better to flush transforms without stepping.

## 3. Camera framing — semantic match to MuJoCo and PyBullet

MuJoCo's `main` camera (`assets/tabletop.xml:23`) is authored as
`pos="0.6 -0.6 0.8"` with `xyaxes="1 1 0  0 0 1"` — camera at
`(0.6, -0.6, 0.8)` looking at the table-top centre `(0, 0, 0.42)` with
up vector `(0, 0, 1)` (RFC-006 §3 derives this).

Genesis's `scene.add_camera` takes `(res, pos, lookat, up, fov, near,
far, ...)`. Plugging MuJoCo's framing directly:

```python
_CAM_EYE_BASELINE = (0.6, -0.6, 0.8)   # matches MuJoCo main camera pos
_CAM_TARGET       = (0.0,  0.0, 0.42)  # table-top centre
_CAM_UP           = (0.0,  0.0, 1.0)
_CAM_FOV          = 45.0               # degrees — MuJoCo default vertical FOV
_CAM_NEAR         = 0.01
_CAM_FAR          = 5.0
```

Byte-for-byte copy of RFC-006 §3.4. Cross-backend pixel parity is still
explicitly out of scope (RFC-007 §7.3) — semantic parity (same camera
pose, same scene layout, same light direction) only. The RFC documents
this next to the existing "cross-backend numerical parity: NO" clause.

### 3.1 Post-build camera mutation for `camera_offset_{x,y}`

Measured: `cam.set_pose(pos=..., lookat=..., up=...)` works post-build
(no scene rebuild required) and back-to-back renders after the same
`set_pose` are `np.array_equal`-equal. This is the mutation hook for
the two camera-offset axes. No new Genesis API needed beyond what 0.4.6
already exposes publicly.

## 4. The lighting knob — private-API mutation on the pyrender scene

Genesis 0.4.6's public `Scene.add_light(pos, dir, color, intensity, ...)`
raises `GenesisException: This method is only supported by
BatchRenderer` on a Rasterizer-backed scene (measured, literal error
message). The public light API is locked out on the back-end we must use.
`VisOptions.lights=[...]` works at Scene-construction time but not
post-build, so it can't implement a per-episode `lighting_intensity`
perturbation.

Two private-API alternatives:

| Option | What it mutates | Pixel effect? | Stability risk |
|---|---|---|---|
| `scene.visualizer.rasterizer._context._scene.light_nodes[0].light.intensity` | Scales the default directional light's intensity. Genesis auto-populates one directional light (dir=(-1,-1,-1), intensity=5.0) on every Rasterizer scene via `VisOptions.lights` default. Mutating `.intensity` post-build is picked up by the next render. | **Yes** — measured scene brightness changes. `intensity *= 2.0` shifted `mean` from 50 → 65 at 64×64. | Pyrender's `DirectionalLight.intensity` is a stable public attribute of pyrender itself; Genesis re-exports it. `_context` is the only private hop. |
| `scene.visualizer.rasterizer._context.ambient_light = (r, g, b)` (plus `ctx._scene.ambient_light = np.array([r, g, b, 1])`) | Scales the Rasterizer's ambient light term. | Yes — measured, 0.1 → 0.5 shifted `mean` 65 → 76. | Same private `_context` hop; Genesis has no public setter. |

**Chosen: mutate `light_nodes[0].light.intensity` only.** One knob is
enough to implement `lighting_intensity`; semantic match to MuJoCo's
`lightDiffuseCoeff`-style scalar and to PyBullet's `lightDiffuseCoeff`
(RFC-006 §3.3). Baseline value on the axis stays `1.0`; we treat the
axis as a scalar multiplier on the default `5.0` intensity — same
convention as PyBullet.

The RFC documents both the private-API hop and the `genesis<0.5` pin
that holds this surface stable. If Genesis 0.5 rearranges `_context`,
it enters a separate RFC (mirrors RFC-007's "upgrade to Genesis 0.5
when it ships" deferred decision).

### 4.1 Rejected alternative: rebuild-on-change

RFC-007 §9 sketch option (a) was "pin a Genesis minor that exposes
`light.set_intensity()`." 0.4.6 does not, and upstream has no near-term
roadmap for it. The §9 sketch alternative — "rebuild-on-change inside
an image-only render path" — was measured: `scene.build()` costs ~5 s on
first build and doesn't appreciably shrink for a smaller scene
(exploration Task 7 §Q4 confirmed). Rebuild-per-episode would blow the
eval budget by ~50× vs a hot-path render. Rejected.

## 5. The texture knob — two-cube teleport-away

Genesis 0.4.6 does **not** expose a post-build material or vertex-colour
mutation that flows through the Rasterizer (measured: setting
`mesh_nodes[-1].mesh.primitives[0].material.baseColorFactor` had no
effect on subsequent renders; `color` silently drops off
`gs.surfaces.Default(...)` because the field is `diffuse_texture`, not
`color`; `ColorTexture` is applied at build-time only). RFC-007 §6 and
the §Q6 exploration already documented this: "No runtime material-ID
or surface swap at 0.4.6."

The option space:

| Option | Shape | Pro | Con |
|---|---|---|---|
| **A. Two pre-built cubes, teleport-away swap.** At `__init__`, add a "red" cube at rest pose and a "green" cube at `_DISTRACTOR_HIDDEN_Z` (`-10.0`). The `object_texture` branch flips which one is at rest and which is hidden. | 1 extra entity at build time; zero extra entity per reset; one pair of `set_pos` calls per `object_texture` toggle. | Works on 0.4.6 unchanged. Uses the exact same teleport-away semantic `distractor_count` already uses (`tabletop_genesis.py:563-573`). No private-API dependency. Two full-quality rendered cubes. | Cube physics state is split across two entities — the "active" cube is the one `self._cube` points at; the "inactive" cube is inert at `z=-10.0`. `_snap_cube_to_ee` and grasp state must always act on the active cube handle. Branch swaps the handle. |
| B. Rebuild scene with alt `ColorTexture`. | One build per toggle. | Works without private API. | ~5 s rebuild per episode that toggles the axis. Prohibitive. |
| C. Private-API mutation of pyrender primitive vertex buffers. | Zero-cost swap. | Fastest. | Deep private API; measured baseColorFactor mutation did *not* flow through (§3 findings above). Non-starter. |
| D. Fail-closed on `object_texture` on Genesis. Keep the axis in `VISUAL_ONLY_AXES` pending upstream support. | n/a | No code risk. | Defeats the point — three of four cosmetic axes would ship, one still rejected. |

**Chosen: Option A (two-cube teleport).** It's the most direct
adaptation of the "teleport-away" pattern already in the codebase. The
cube-handle swap is a small code addition (three lines in
`_apply_one_perturbation("object_texture", ...)` and a `self._cube`
redirection) and keeps the per-reset cost inside one extra `set_pos`
per axis toggle.

### 5.1 Handle-swap semantics

`self._cube` today is the single box entity (`tabletop_genesis.py:264`).
Option A introduces `self._cube_red` + `self._cube_green`, a
`self._cube` pointer that always aliases the active one, and a
`self._cube_alt` pointer aliasing the hidden one.
`_apply_one_perturbation("object_texture", v)` swaps the two pointers
and issues two `set_pos` calls (active → rest pose, alt → hidden).
`restore_baseline` starts from "red is active, green is hidden." All
the downstream methods (`_cube_pos`, `_snap_cube_to_ee`, reset's cube
teleport) touch `self._cube` → no changes needed outside the
`_apply_one_perturbation` branch.

## 6. `VISUAL_ONLY_AXES` — same shape as RFC-006

RFC-006 §3.5 dropped PyBullet's `VISUAL_ONLY_AXES` to `frozenset()` and
flipped the Suite loader's rejection-test into an acceptance-test on
`tabletop-pybullet`. This RFC does the same for `tabletop-genesis`:
`GenesisTabletopEnv.VISUAL_ONLY_AXES = frozenset()`; the loader test
for the Genesis backend that today asserts cosmetic-only sweeps are
rejected (`tests/genesis/...`) is renamed to assert they now load.

The "user runs cosmetic-only sweep with `render_in_obs=False`" honesty
caveat is preserved on the class docstring — the sweep loads, the obs
is state-only, pairwise-identical cells result. This is the same
property MuJoCo and PyBullet have had since RFC-006. Not a bug.

## 7. Determinism contract

Within-instance bit-determinism must survive the rendering path:

1. Rasterizer is deterministic (§2.1).
2. All four cosmetic axes feed deterministic setters:
   - `lighting_intensity` → private-API scalar write, no RNG.
   - `camera_offset_x/y` → `cam.set_pose(pos=...)`, no RNG.
   - `object_texture` → two `set_pos` calls on already-built entities.
3. `shadow` stays at Genesis's default directional-light `castshadow=True`; the Rasterizer's shadow pass is CPU and deterministic on same-wheel x86-64 (same argument as RFC-006 §2.3 for PyBullet's `shadow=0`, one-level weaker since we cannot easily disable Genesis's shadow). If a CI flake ever surfaces from a shadow-pass Z-ordering issue, a follow-up can wire a `VisOptions(shadow=False)` knob — measured, `VisOptions.shadow` exists and is `True` by default. Not a gating concern for this RFC.
4. `restore_baseline` already resets `_light_intensity`, `_cam_offset`, `_texture_choice` (`tabletop_genesis.py:525-528`); this RFC adds the pyrender-light intensity reset (`.intensity = BASELINE_LIGHT_INTENSITY * 1.0`) and the camera pose reset (`cam.set_pose(pos=_CAM_EYE_BASELINE, lookat=_CAM_TARGET, up=_CAM_UP)`) in the same body.
5. `ctx.update_rigid()` before each render (§2.4) is pure transform-push, no RNG.

Net: `reset(seed=s)` twice on two separate `GenesisTabletopEnv`
instances produces bit-identical `obs["image"]` arrays. Test case:
`np.array_equal` on the image field in the double-reset case, mirroring
RFC-006 §9 case 3.

### 7.1 Cross-OS / cross-wheel caveat

Same reservation as RFC-006 §5.1 and RFC-007 §8.2 — Rasterizer is
x86-64 Linux / Windows deterministic on same-wheel; Apple Silicon and
cross-wheel are not asserted. CI runs `ubuntu-latest`, which is what
the test pin asserts.

## 8. Tests — what the RFC's test plan will contain

Mirroring RFC-006 §7 almost exactly (same eight cases), under
`tests/genesis/test_render_genesis.py`, marker `@pytest.mark.genesis`,
shared with the existing `genesis-tests` CI job (no new job).

1. **Shape / dtype gate.** `render_in_obs=True, render_size=(H, W)` → `obs["image"].shape == (H, W, 3)`, `dtype == uint8`, `min >= 0`, `max <= 255`. Image absent when `render_in_obs=False`.
2. **Observation-space parity with MuJoCo.** `GenesisTabletopEnv(render_in_obs=True).observation_space["image"]` equals `TabletopEnv(render_in_obs=True).observation_space["image"]` under `gym.spaces.Box.__eq__`. Pixel values explicitly not compared (RFC-007 §7.3).
3. **Within-run determinism.** Two `GenesisTabletopEnv` instances, both `reset(seed=42)`; `np.array_equal(obs_a["image"], obs_b["image"])` holds.
4. **Post-step determinism.** Same seed + fixed 20-step rng-seeded action sequence on two instances → `obs["image"]` at step 20 matches byte-for-byte.
5. **Axis sensitivity — four cases.** For each cosmetic axis (`lighting_intensity` 0.3 vs 1.5; `object_texture` 0 vs 1; `camera_offset_x` ±0.05; `camera_offset_y` ±0.05), two envs reset with the axis set to distinct values render **not** `np.array_equal` images. This is the test that proves the wiring works at all.
6. **State-only default preserved.** `GenesisTabletopEnv()` (default `render_in_obs=False`) → `"image"` not in `obs`; five existing state keys unchanged. Locks the RFC-007 state-only contract.
7. **Suite-loader relaxation.** Rename / flip the test that asserts a cosmetic-only sweep on `tabletop-genesis` is rejected; assert it now loads (mirrors RFC-006 §7 case 7). Add a new "cosmetic-only sweep loads on tabletop-genesis" test.
8. **RandomPolicy rendering smoke.** 3 rollouts, `render_in_obs=True`, `render_size=(64, 64)` (keep CI cheap), asserts every step's `obs["image"]` is a valid `uint8 (64, 64, 3)` array. Mirrors RFC-006 §7 case 8.

Total added test time budget at 64×64: well under 30 s on the existing
`genesis-tests` job. The one-time ~24-s cold-compile tax is paid once
per process; the 64×64 per-frame cost after that is <10 ms.

## 9. Implementation surface — expected diff size

- `src/gauntlet/env/genesis/tabletop_genesis.py` — constructor kwargs (+2 lines + validation), module-level camera constants (+10 lines), `_render_obs_image` method (+15 lines), `_build_obs` one-liner guard, `observation_space` image-key extension, four cosmetic-axis branches rewired to consume the new state, `VISUAL_ONLY_AXES` → `frozenset()` classvar, `restore_baseline` extension (camera + light reset + cube-handle reset), `self._cube_red` / `self._cube_green` / `self._cube_alt` plumbing + `self._cube` pointer swap. Expected: ~120–150 LoC added, ~5 removed.
- `src/gauntlet/env/genesis/__init__.py` — unchanged.
- `src/gauntlet/env/genesis/assets/` — does not exist; this RFC does not add one. Two cubes are `gs.morphs.Box` primitives with `gs.surfaces.Default(diffuse_texture=gs.surfaces.ColorTexture(color=...))`. No asset file.
- `src/gauntlet/suite/schema.py` — unchanged. No YAML field added (RFC-006 §8 Q3 rationale holds: camera intrinsics are a backend-factory concern, enabled via `partial(GenesisTabletopEnv, render_in_obs=True)`).
- `src/gauntlet/suite/loader.py` — unchanged. `_reject_purely_visual_suites` already picks `VISUAL_ONLY_AXES` up from the class dispatch; when it empties, the loader's rejection on `tabletop-genesis` becomes a no-op. No code change needed in the loader — only in the test that asserts the rejection fires (§8 case 7).
- `src/gauntlet/runner/worker.py` — unchanged.
- `src/gauntlet/cli.py` — unchanged.
- `tests/genesis/test_render_genesis.py` — new file, ~180 LoC for the eight cases above.
- `tests/genesis/test_env_genesis.py` — minor update: the cosmetic-axis tests that today assert "obs unchanged after cosmetic perturbation" either get marked `render_in_obs=False` explicitly, or retired in favour of the new `test_render_genesis.py` positive cases. This RFC leans retire-and-replace.
- `docs/phase2-rfc-008-genesis-rendering.md` — new.
- `README.md` — two sentences in the Backends section: "image obs on Genesis ships via `uv sync --extra genesis` with `render_in_obs=True` on the env constructor; semantic parity (not pixel parity) with MuJoCo/PyBullet."
- `examples/evaluate_random_policy_genesis.py` — extend to exercise `render_in_obs=True` via a `partial(GenesisTabletopEnv, render_in_obs=True)` factory, or alternatively a new example `examples/evaluate_smolvla_genesis.py` mirroring `evaluate_smolvla_pybullet.py`. The RFC picks the first (extend existing) to minimise file churn.

Feasibility check against the §9 PR-sized chunk budget: ~9–11 commits,
mirroring RFC-006. No new asset, no new subpackage, no new CI job — the
existing `genesis-tests` job covers it.

## 10. Open questions the RFC must answer

1. **Light-intensity knob — private API or rebuild?** Lean: private API (§4). Rebuild is ~5 s/episode; not viable.
2. **Default `render_size`.** `(224, 224)`. Matches MuJoCo + PyBullet. Named `_DEFAULT_RENDER_SIZE` at module scope.
3. **Do we need to disable Genesis's default shadow pass?** Not up front. The default `VisOptions.shadow = True` is observed deterministic on same-wheel; if CI ever flakes, revisit.
4. **Image `dtype`.** Match MuJoCo / PyBullet: `uint8`. `cam.render(rgb=True)[0]` already returns `uint8`. Zero-conversion.
5. **`observation_space["image"]` Box bounds.** `low=0, high=255, dtype=uint8`. Byte-for-byte identical to MuJoCo (`env/tabletop.py:237`) — tested in §8 case 2.
6. **`render_in_obs=True` and the Runner's `multiprocessing` fork.** Each worker instantiates its own env, its own `gs.Scene`, its own `cam`, pays its own first-render tax. No cross-worker state — the Rasterizer context is scene-local. Confirm in the RFC with a one-paragraph worker-semantics section, mirroring RFC-006 §8 Q4.
7. **Flush strategy — `ctx.update_rigid()` vs `scene.step()`.** `update_rigid()` (§2.4). Reason: `scene.step()` advances sim time 50 ms per reset and alters `cube.get_pos()` under gravity for any free body, silently corrupting within-instance determinism against the MuJoCo reference.
8. **Should the private-API dependency on `_context._scene.light_nodes` land a runtime check with a clear error if Genesis 0.5 removes it?** Yes. One `AttributeError`-guarded call site in `_render_obs_image` + `restore_baseline`, logging a hint to pin `genesis-world<0.5` and file an issue. This is the only part of the adapter that reaches inside a `_name` — honesty-flag it loudly.

## 11. What this doc does *not* commit to

- **Cross-backend pixel parity.** Out of scope (RFC-007 §7.3). Semantic parity only.
- **Genesis 0.5 upgrade.** If and when it ships. This RFC keeps the `genesis-world<0.5` pin from RFC-007.
- **Shadow-controlled perturbation axis.** Not in the canonical seven. Future RFC if ever needed.
- **Depth / segmentation channels in `obs`.** Not emitted. `cam.render` kwargs stay at `rgb=True` only.
- **FOV / near / far as axes.** Not in the canonical seven. Out of scope.
- **Isaac Sim.** Still deferred (Phase 2 scope, §7 of GAUNTLET_SPEC).

## 12. Summary — what the RFC will commit to

A Rasterizer-backed image-observation path on `GenesisTabletopEnv`,
matching MuJoCo and PyBullet's `render_in_obs=True` / `render_size=(H, W)`
surface byte-for-byte; four cosmetic axes wired through the render path
via the canonical MuJoCo/PyBullet framing, a private-API directional-light
intensity mutation (honesty-flagged, `genesis-world<0.5` pinned), a
two-cube teleport-away for `object_texture`, and `cam.set_pose` for both
camera-offset axes; `VISUAL_ONLY_AXES` drops to `frozenset()`; the Suite
loader's cosmetic-only rejection becomes a no-op on `tabletop-genesis`;
tests under `tests/genesis/test_render_genesis.py` on the existing
`genesis-tests` CI job. No new subpackage, no new asset, no new CI job.
One PR on branch `phase-2/genesis-rendering`.
