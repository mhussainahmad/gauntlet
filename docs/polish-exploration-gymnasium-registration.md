# Polish exploration — Gymnasium env registration

Status: exploration. Targets `gymnasium.make("gauntlet/Tabletop-v0")` and the
three sibling backend ids. Scope: ship `gym.register(...)` plumbing for all
four shipped backends; preserve every existing direct-import code path.

## Why this matters

`gauntlet` ships four `gymnasium.Env` adapters (MuJoCo `tabletop`, PyBullet,
Genesis, Isaac Sim) and a thin internal name->factory registry
(`gauntlet.env.registry`) used by the Suite loader and the CLI. None of those
backends are reachable through `gymnasium.make(...)` — the standard
construction surface every other RL/IL ecosystem assumes:

* `stable-baselines3` examples open with `env = gym.make(env_id)`.
* `RLlib`'s `config.environment(env=env_id)` resolves through `gym.make`.
* `CleanRL`'s reference implementations all do `gym.make(args.env_id)`.
* `TorchRL`'s `GymEnv(env_name)` wraps the same constructor.
* Most third-party wrappers (`gym.wrappers.RecordEpisodeStatistics`,
  `RecordVideo`, `PixelObservationWrapper`) compose around envs that came
  from `gym.make`.

Right now a researcher who wants to drop a gauntlet adapter into any of those
pipelines has to write the glue:

```python
# Today — no gym.make affordance
from gauntlet.env.tabletop import TabletopEnv
env = TabletopEnv()
```

After this PR:

```python
# After — standard ecosystem affordance
import gauntlet  # noqa: F401  (registers env ids on import)
import gymnasium as gym
env = gym.make("gauntlet/Tabletop-v0")
```

That's a small, purely additive wiring win — but it removes a real friction
point for every downstream user who already lives inside the `gym.make`
ecosystem.

## Namespace decision: `gauntlet/<Backend>-v0`

Gymnasium 1.0 supports namespaced env ids of the form
`namespace/Name-vN`. The convention is:

* `Atari/...`, `MiniGrid/...`, `Robosuite/...`, `Adroit/...` etc. for
  third-party suites.
* The namespace is the project / package; the second segment names the
  individual env in CamelCase; `vN` versions the *task semantics* (action
  space, observation space, reward, success criterion) so a downstream
  baseline pinned to `v0` keeps reproducing the same numbers if `v1` ever
  ships.

We pick `gauntlet/` as the namespace. The four registered ids:

| id                              | backend                              |
| ------------------------------- | ------------------------------------ |
| `gauntlet/Tabletop-v0`          | MuJoCo (core dep)                    |
| `gauntlet/TabletopPyBullet-v0`  | PyBullet (`[pybullet]` extra)        |
| `gauntlet/TabletopGenesis-v0`   | Genesis (`[genesis]` extra)          |
| `gauntlet/TabletopIsaac-v0`     | Isaac Sim (`[isaac]` extra)          |

`-v0` is the right starting version: every backend's action layout, observation
keys, reward, and success criterion are pinned in tests today (cross-backend
determinism audit, Phase 2.5 Task 17). Any future change to that surface
becomes `-v1` and the original baseline keeps working.

Rejected: a flat `Tabletop-v0` / `TabletopPyBullet-v0` (no namespace). Gym's
default registry already has hundreds of unprefixed ids; using a namespace
avoids any accidental collision and matches every modern third-party suite.

## Backends-as-entry-points: lazy import for the heavy three

The MuJoCo backend is a core dep — `import gauntlet.env.tabletop` is always
safe. The other three sit behind optional extras:

* `[pybullet]` -> `pybullet`, `pybullet_data`
* `[genesis]`  -> `genesis-world`, `torch`
* `[isaac]`    -> `isaacsim` (and indirectly Omniverse Kit + a CUDA GPU)

Each backend's subpackage `__init__.py` does an `ImportError`-guarded
`import` of its heavy dep so that, on a machine without the extra installed,
`from gauntlet.env.pybullet import PyBulletTabletopEnv` raises a clean
"install hint" error rather than a confusing
`ModuleNotFoundError: No module named 'pybullet'`.

That guard machinery is exactly why we MUST register the heavy backends with
**string `entry_point`s**, not class references:

```python
gym.register(
    id="gauntlet/TabletopPyBullet-v0",
    entry_point="gauntlet.env.pybullet.tabletop_pybullet:PyBulletTabletopEnv",
    max_episode_steps=200,
)
```

Gymnasium resolves the string lazily inside `gym.make(...)` — *only* when the
user actually constructs that env. A bare `import gauntlet` therefore:

1. Triggers `register_envs()` -> four `gym.register(...)` calls.
2. None of the four touches `pybullet` / `genesis` / `isaacsim`.
3. The Tabletop (MuJoCo) string is also a string for consistency, but
   `mujoco` is a core dep so resolution is free either way.

A user without the `[pybullet]` extra can `import gauntlet` and
`gym.make("gauntlet/Tabletop-v0")` with zero ImportError; trying to
`gym.make("gauntlet/TabletopPyBullet-v0")` is what triggers the install-hint.

That mirrors the existing pattern: the internal `gauntlet.env.registry` is
*also* lazy at this granularity (the Suite loader imports
`gauntlet.env.pybullet` only when the YAML asks for it). The new gym
registration extends the same hygiene to the gym-make code path.

## Idempotency design

`gym.register(...)` with an already-registered id emits a warning in
gymnasium 1.0+ and may overwrite the existing entry. That's a problem for
us: `register_envs()` is called from `gauntlet/__init__.py` so the warning
would fire on every `import gauntlet` after the first (e.g. when a test
re-imports the package, or when a user explicitly calls `register_envs()`
to be safe).

The fix is two layers:

1. **Per-id check.** `_register_one(env_id, entry_point, ...)` early-returns
   if `env_id in gymnasium.envs.registry`. This is the correctness
   guarantee — robust to any caller pattern, robust to gym version.
2. **Module-level fast-path.** A `_REGISTERED: bool` flag short-circuits
   `register_envs()` after the first successful run so the four dict
   lookups are skipped on the hot path. Cheap and harmless if it ever
   gets out of sync with the gym registry (the per-id check still wins).

Both layers compose: bypass the bool, the per-id check still works; clear
the registry in a test, clear the bool, re-registration is clean.

## Backwards compatibility

Three rules, all preserved unchanged:

1. **Direct imports still work.**
   `from gauntlet.env.tabletop import TabletopEnv` and
   `TabletopEnv()` produce the same env they always did. Same for the
   three backend subpackages. A regression test pins this.
2. **Internal registry unchanged.** `gauntlet.env.registry` (the
   name->factory dict used by the Suite loader and the CLI) is untouched.
   Existing `tests/test_env_registry.py` passes unchanged.
3. **No new dependencies.** `gymnasium` is already a core dep
   (`gymnasium>=1.0,<2`); `gym.register` is core API.

The new affordance is purely additive. Anything that worked yesterday works
today.

## Open questions (and the chosen answers)

* **Should we register an env with `render_in_obs=True` by default?** No.
  Each adapter's existing `__init__` default (which is `render_in_obs=False`
  on the backends that have a render mode flag) is what every existing
  rollout uses; flipping the default at the gym layer would create a silent
  per-backend API split. Users who want pixels in obs pass it through
  `gym.make(..., render_in_obs=True)`.
* **Should `register_envs()` also register a "smoke" version with
  `max_steps=20` for fast tests?** No. Every shipped adapter takes
  `max_steps` as a constructor kwarg and `gym.make("gauntlet/Tabletop-v0",
  max_episode_steps=20)` already does the right thing through Gymnasium's
  `TimeLimit` wrapper. Adding a parallel `Tabletop-v0-smoke` id would
  double the surface for zero new capability.
* **Should we add `kwargs={...}` to the registration?** No. Every adapter's
  constructor has sensible defaults today; any keyword the user wants to
  override goes through `gym.make("...", foo=bar)`. Carrying default
  kwargs in the registration would make the gym surface drift away from
  the direct-import surface.
* **Should `register_envs()` be public or private?** Public. Re-export it
  from `gauntlet.__init__` so users who explicitly want to control when
  registration happens (e.g. inside a `pytest` `autouse` fixture, or in a
  multiprocessing worker that imports `gauntlet` lazily) have the lever.
  The default — register on import — is the gymnasium-ecosystem convention.

## Test plan

1. `tests/test_gym_registration.py::test_make_tabletop_constructs_mujoco_env`
   — `gym.make("gauntlet/Tabletop-v0")` returns an env whose `.unwrapped`
   is an instance of `TabletopEnv`.
2. `test_register_envs_is_idempotent` — call `register_envs()` twice; no
   exception, no duplicate entry, `gym.envs.registry` count unchanged.
3. `test_all_four_ids_registered_after_import` — after `import gauntlet`,
   all four ids appear in `gym.envs.registry`.
4. `test_heavy_backend_registration_is_lazy` — run `import gauntlet` in a
   subprocess (clean interpreter, no test pollution) and assert that
   `gauntlet.env.pybullet`, `gauntlet.env.genesis`, `gauntlet.env.isaac`
   are NOT in `sys.modules` afterwards. Subprocess is the only honest
   test here — pop-from-`sys.modules` would falsely succeed if any earlier
   collection step imported the heavy module.
5. `test_existing_direct_import_still_works` — backwards-compat regression
   pinning `from gauntlet.env.tabletop import TabletopEnv; TabletopEnv()`.
6. The existing `tests/test_env_registry.py` suite passes unchanged.

## Out of scope

* The PyBullet / Genesis / Isaac heavy-extras tests (`tests/pybullet/`,
  `tests/genesis/`, `tests/isaac/`) gain no new cases here. The lazy-import
  test asserts those backends are *not* loaded; verifying that
  `gym.make("gauntlet/TabletopPyBullet-v0")` constructs a real env on a
  machine with the extra installed is left to a follow-up CI job (or to
  the user's own pipeline).
* Gymnasium's `EnvSpec.kwargs`-based parameterisation (e.g. registering
  `gauntlet/Tabletop-v0-render` with `kwargs={"render_in_obs": True}`)
  is intentionally not used — see the open-questions section above.
