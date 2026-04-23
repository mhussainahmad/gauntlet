# Phase 2 Task 7 Exploration: Genesis Backend

**Date:** 2026-04-23
**Branch:** phase-2/genesis-backend
**Base:** b43ac07 (main, post-Task-6)
**Target simulator:** [Genesis](https://genesis-embodied-ai.github.io/) — `genesis-world==0.4.6` on PyPI.

---

## Q1: Does `genesis-world` install cleanly against the existing dep tree?

**Dry-run** (`pip install --dry-run 'genesis-world>=0.4,<1'` into a fresh cpython-3.12 venv) shows **no hard version conflicts with the core `gauntlet` package**, but two touch points warrant a deliberate decision in the RFC:

| Installed vs core | Outcome |
|---|---|
| `mujoco 3.7.0` (core: `>=3.2,<4`) | ✅ satisfies |
| `numpy 2.4.4` (core: `>=1.26,<3`) | ✅ satisfies |
| `pydantic 2.13.3` (core: `>=2.7,<3`) | ✅ satisfies |
| `pyyaml 6.0.3` (core: `>=6.0,<7`) | ✅ satisfies |
| `rich 15.0.0` (core: `>=13.7,<15`) | ⚠️ **exceeds ceiling** — `genesis-world` itself declares no `rich` pin, so pip resolved 15.0.0 because nothing capped it. Project ceiling needs bumping to `<16`. |

Genesis itself is a 82.6 MB pure-Python wheel (`py3-none-any`), `Requires-Python: <3.14,>=3.10`.

**Non-obvious finding — runtime dep not in install_requires:** `genesis-world` imports `torch` at package-level but does **not** declare it in `install_requires` (delegates to users so they can pick CPU/CUDA/ROCm variants). A plain `pip install genesis-world` then `import genesis` fails with `ModuleNotFoundError: No module named 'torch'`. Consequence: the `[genesis]` extra in `pyproject.toml` **must** list both `genesis-world` and `torch` — and the `gauntlet.env.genesis/__init__.py` guard must surface both missing-dep cases with a clear message (the PyBullet template handles a single `ImportError` source; this one has two).

---

## Q2: Python matrix

| Python | `genesis-world` wheel | Notes |
|---|---|---|
| 3.11 | ✅ (py3-none-any) | CI matrix — works. |
| 3.12 | ✅ (py3-none-any) | CI matrix — works. |
| 3.13 | ✅ (py3-none-any) | Not in CI matrix today. |
| 3.14 | ❌ `Requires-Python: <3.14` | Irrelevant to CI; laptop is 3.14 but `uv` manages its own 3.11/3.12. |

**Ceiling `<3.14` is documented upstream** ([`genesis-world` metadata](https://pypi.org/project/genesis-world/0.4.6/)). When 3.14 wheels ship, the `[genesis]` extra can be opened up — no code change needed because `gauntlet.env.genesis/__init__.py` catches `ImportError` at import time regardless.

---

## Q3: CPU backend viability

Genesis backends exposed as module-level constants: `gs.cpu`, `gs.gpu`, `gs.cuda`, `gs.metal`. `gs.init(backend=gs.cpu)` completes in ~4s (one-time per process, caches Taichi-style kernels thereafter).

**CPU is a first-class supported backend**, unlike (e.g.) Isaac Gym where "CPU" means "unsupported and broken". CI is CPU-only and Genesis does not downgrade functionality on CPU — it just runs slower.

---

## Q4: Scene build cost (the real runtime constraint)

`scene.build()` is Genesis's compile-the-scene-into-fused-kernels step. Measured on this laptop (cpython-3.12, cpu backend):

| Scene | `build()` time |
|---|---|
| 1st build — plane + 1 box | **4.87 s** |
| 2nd build — same shape (same process) | **0.91 s** |
| 3rd build — same shape (same process) | **0.75 s** |
| 1st build — plane + box + Franka MJCF | **~25 s** |

**Conclusions:**
1. **Build caching is effective** across scenes within one process — first build compiles kernels; subsequent builds reuse them. Worker-process reuse (multiprocessing `spawn` pool kept alive for the whole run) amortises the 5 s cost across all episodes.
2. **`scene.build()` is a per-env construction cost, NOT a per-episode cost.** The adapter must build once in `__init__` and use `scene.reset()` + entity-level setters between episodes. Never rebuild per episode.
3. **MJCF / mesh loading dominates** the 25 s spike for Franka. First-cut adapter uses primitive-only bodies (same trade the PyBullet adapter made — RFC-005 §7.1). This keeps env construction at ~5 s, comparable to the other extras-gated backends.

**Entity-level ops after build (verified)**: `set_pos`, `set_quat`, `set_dofs_position`, `get_pos`, `get_vel`, `set_friction`. These are the primitives used to express perturbations.

**Missing APIs (verified)**: no entity-level `set_visible` / `set_collision_enabled`. Consequence: the `distractor_count` axis cannot use MuJoCo's visibility-toggle semantics; it uses a **teleport-away** semantic (move disabled distractors to `z = -10.0`, out of both physics and render reach). Documented as a semantic deviation in the RFC, not a bug — the PyBullet adapter already carved out the same "different backend, semantically similar, documented delta" pattern (RFC-005 §7.4).

---

## Q5: Determinism from a single seed (§6 reproducibility requirement)

Ran the same scene twice in the same process (fresh `gs.Scene`, same initial pos + size, same 20 `step()` calls), captured cube position each step, compared:

```
determinism max abs diff (same scene, two rollouts): 0.0
```

**Bit-identical across rollouts** on the CPU backend. The reproducibility property gauntlet relies on (`(suite, axis_config, seed)` → identical trajectory) holds for Genesis on CPU. This property **is not claimed across backends** (MuJoCo vs PyBullet already don't match — RFC-005 §7.4; Genesis joins them as a third distinct numerical regime).

**Not verified but flagged in RFC**: determinism across *processes* (two subprocesses, same seed). The PyBullet RFC noted the same caveat. Runner-level seed threading is already worker-local per episode; cross-process numerical parity is not a user-facing guarantee.

---

## Q6: Scene / entity API surface relevant to the 7 axes

Confirmed empirically:

| Axis | Genesis API | Maps to |
|---|---|---|
| `lighting_intensity` | `scene.add_light(pos, dir, color, intensity, ...)` exists at scene construction. Post-build intensity mutation: **not a stable API**. | State-only first cut: stored in `self._light_intensity` for the follow-up rendering RFC (same treatment the PyBullet adapter gave this axis pre-RFC-006). Marked `VISUAL_ONLY_AXES`. |
| `camera_offset_x/y` | `scene.add_camera(...)` returns a camera with `set_pose(pos=..., lookat=...)`. | State-only first cut: camera exists only for the follow-up rendering RFC; offset stored but unused by state obs. Marked `VISUAL_ONLY_AXES`. |
| `object_texture` | No runtime material-ID swap in Genesis's stable surface API. `gs.surfaces.ColorTexture(color=...)` is set at entity creation. | State-only first cut: stored, marked `VISUAL_ONLY_AXES`. A mid-episode texture swap lives with the rendering follow-up. |
| `object_initial_pose_x/y` | `cube.set_pos((x, y, z))` after `scene.reset()`, before `scene.step()`. | Direct — same semantic as MuJoCo `qpos` overwrite. |
| `distractor_count` | Pre-allocate 10 distractor boxes at build. "Disable" → `set_pos((x, y, -10.0))`; "enable" → `set_pos((x, y, cube_rest_z))`. | **Teleport-away semantic** (documented deviation). No visibility API exists; this is the cleanest equivalent. Matches the PyBullet adapter's pre-allocation pattern (RFC-005 §5.2). |

Floating-EE kinematic control (not a native "kinematic constraint" in Genesis): verified by teleporting an EE box via `set_pos` each of 50 steps while the physics sim advanced — 78 ms total, no physics glitches, cube untouched. Same control pattern the PyBullet adapter uses via `p.changeConstraint` (RFC-005 §7.1). The floating-EE abstraction (no IK, no arm) extends across all three backends.

---

## Q7: Registration footprint

Mirrors RFC-005 exactly:

**New files:**
- `docs/phase2-rfc-007-genesis-adapter.md` (the next document — §4 will cover dep-tree decisions, §5 scene layout, §6 axis semantics, §7 test matrix, §8 CI footprint).
- `src/gauntlet/env/genesis/__init__.py` — two-part guard import (`genesis-world` + `torch`), then `register_env("tabletop-genesis", …)`.
- `src/gauntlet/env/genesis/tabletop_genesis.py` — `GenesisTabletopEnv` class.
- `tests/genesis/test_env_genesis.py` — per-axis contract tests, marked `@pytest.mark.genesis`.

**Modified files:**
- `pyproject.toml` — `[genesis]` extra (`genesis-world>=0.4,<0.5` + `torch>=2.2,<3`); `[dependency-groups] genesis-dev`; `[tool.pytest.ini_options] markers` += `genesis`; `[tool.ruff]` / `[tool.mypy]` overrides for third-party no-stubs (`genesis`, `genesis.*`). **Core `rich` ceiling bumped `<15 → <16`** (see Q1).
- `.github/workflows/ci.yml` — new `genesis-tests` job mirroring `pybullet-tests`.
- `src/gauntlet/suite/loader.py` (if the missing-extra message needs a `genesis` branch — the existing dispatch may already cover any unknown `env:` key generically).

**Unmodified (by design):**
- `src/gauntlet/env/base.py`, `src/gauntlet/env/registry.py`, `src/gauntlet/runner/*` — the protocol extracted in Task 5 is backend-agnostic; Genesis slots in without touching it.
- `src/gauntlet/env/tabletop.py` (MuJoCo), `src/gauntlet/env/pybullet/*` — no cross-backend changes.

---

## Q8: Risk register

1. **Genesis 0.4.x API churn.** 0.4.6 is the latest on PyPI; the project is < 1 year old. Pin `>=0.4,<0.5` (tighter than `[hf]`'s `>=4.40,<5` or `[lerobot]`'s `>=0.4,<1`) to catch API breakage at the extras seam early. RFC §4 restates the pin rationale.
2. **Torch as undeclared runtime dep.** Genesis 0.4.6 imports `torch` but does not list it in `install_requires`. The `[genesis]` extra declares it explicitly. The import guard catches both `genesis` missing and `torch` missing with a unified message.
3. **Scene-build time (~5 s minimum, ~25 s with MJCF).** First-cut avoids MJCF entirely (primitives only). Env construction fits inside the Runner's existing per-worker-process amortisation.
4. **Cross-process determinism not verified.** Same-process determinism is 0.0-diff. Cross-process is not in scope (no other backend makes that claim either).
5. **Visual-axis semantics deferred to follow-up RFC.** `lighting_intensity`, `camera_offset_{x,y}`, `object_texture` are declared `VISUAL_ONLY_AXES` in the state-only first cut — queued + validated but do not mutate state obs. Mirrors PyBullet's pre-RFC-006 shape.
6. **`distractor_count` is teleport-away, not visibility-off.** Documented in RFC §6. Semantically similar, numerically distinct (no collision with a far-below-ground body; same as the visibility-off branch in MuJoCo, which also disables collisions).

---

## 150-word executive summary

`genesis-world==0.4.6` installs cleanly on Python 3.11–3.13, CPU-first, bit-identical same-process determinism verified (0.0 abs-diff over 20 steps). Two install-time deviations vs the existing extras pattern: (1) Genesis does **not** declare `torch` in `install_requires`, so the `[genesis]` extra carries both `genesis-world` and `torch`, and the subpackage `__init__` guards against both missing; (2) Genesis pulls `rich 15.0.0`, requiring a core-dep ceiling bump `rich<15 → rich<16`. `scene.build()` is 5 s with primitives (~25 s if a Franka MJCF is loaded), so the first cut is primitive-only, matching the PyBullet adapter's no-URDF-arm carve-out (RFC-005 §7.1). Two axes diverge semantically: `distractor_count` is **teleport-away** (Genesis has no entity-visibility API), and `lighting_intensity`/`object_texture`/`camera_offset_{x,y}` are declared `VISUAL_ONLY_AXES` pending a follow-up rendering RFC — same shape PyBullet had pre-RFC-006. Backend registers as `tabletop-genesis`, mirroring `tabletop-pybullet`.
