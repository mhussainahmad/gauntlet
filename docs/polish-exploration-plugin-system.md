# Polish exploration: plugin system for third-party policies + envs

Status: exploration / pre-implementation
Owner: phase-3/plugin-system branch

## 1. Why this matters (the ecosystem win)

`gauntlet` ships a fixed set of policy + env adapters today. To grow into
an evaluation standard the way `gymnasium` and `stable-baselines3` did,
third-party packages must be able to register their own adapters
**without modifying gauntlet's source tree**. The Python idiom for that
is `[project.entry-points]` discovery via
`importlib.metadata.entry_points()` — pip-installable plugins drop their
class behind a named entry-point string and gauntlet picks it up at
import-cheap discovery time.

The motivating example — entirely hypothetical — is a downstream
package called `gauntlet-rl-baselines` that wraps `stable-baselines3`.
Its `pyproject.toml` would carry:

```toml
[project.entry-points."gauntlet.policies"]
sb3 = "gauntlet_rl_baselines.adapter:SBAdapter"
```

After `pip install gauntlet-rl-baselines`, the user types
`gauntlet run --suite suites/reach.yaml --policy sb3` and the harness
finds `SBAdapter` via `importlib.metadata.entry_points(group="gauntlet.policies")`
and constructs it. No fork of gauntlet, no PR, no monkeypatch.

## 2. Entry-point group naming convention

Two groups:

* `gauntlet.policies` — values are dotted import paths to a `Policy`
  subclass (or any zero-arg callable returning a `Policy`).
  Names map to the `--policy` CLI argument.
* `gauntlet.envs` — values are dotted import paths to a class
  satisfying the `GauntletEnv` protocol. Names map to the `env:` key in
  Suite YAML and to `gauntlet.env.registry.get_env_factory(name)`.

Group names follow the `<package>.<plural-noun>` convention used by
`gymnasium.envs`, `pytest11`, and `flask.commands`. The plural is
deliberate — `gauntlet.policy` would clash with the existing
`src/gauntlet/policy/` import path on tooling that conflates the two.

## 3. Built-in vs plugin precedence rule

**Built-ins always win on collision.**

Reasoning: a third-party plugin that grabs the name `random` and
silently shadows the in-tree `RandomPolicy` would be a debugging trap,
particularly because the swap would be visible only on machines where
that plugin happened to be `pip install`-ed. The deterministic rule is:

1. The built-in registry table (`POLICY_REGISTRY`, `ENV_REGISTRY`)
   is the source of truth for first-party names.
2. Plugin discovery runs *after* a built-in lookup miss.
3. If a plugin entry-point name collides with a built-in name **and**
   the loaded class is not the same object as the built-in, a
   `RuntimeWarning` is emitted naming the conflicting distribution and
   the built-in is used.
4. Self-collisions caused by gauntlet's own dogfooded entry points
   (the entry point loads the same class object the built-in table
   already holds) are silent — checked via `is`, not name equality.

## 4. Lazy-load failure model

Entry-point discovery is cheap: `entry_points(group=...)` just reads the
installed packages' metadata, no plugin Python code runs. Only when a
user actually references a plugin name does `ep.load()` execute the
plugin's import side effects.

If `ep.load()` raises (`ImportError` because the plugin's own deps are
missing, `AttributeError` because the entry point points at a moved
class, anything else), `gauntlet.plugins` wraps the exception in a
clear `RuntimeError` of the form:

```
Plugin 'sb3' from 'gauntlet-rl-baselines' failed to load: <reason>
```

…and drops the plugin from the discovery dict. Gauntlet itself stays
operational; the user sees a targeted error pointing at the plugin's
distribution name (queried via `ep.dist.name`) so they can file a bug
upstream.

## 5. Dogfood: built-in adapters as entry points

We also register gauntlet's own first-party adapters under both groups
in `pyproject.toml`. After `uv pip install -e .` the entry points
materialize and discovery returns the built-in classes alongside any
real third-party plugins. This eats our own dogfood — if the plugin
mechanism is broken on a fresh checkout, the discovery smoke test fails
loudly.

The four built-in policies registered:

```toml
[project.entry-points."gauntlet.policies"]
random = "gauntlet.policy.random:RandomPolicy"
scripted = "gauntlet.policy.scripted:ScriptedPolicy"
huggingface = "gauntlet.policy.huggingface:HuggingFacePolicy"
lerobot = "gauntlet.policy.lerobot:LeRobotPolicy"
```

The four built-in envs registered:

```toml
[project.entry-points."gauntlet.envs"]
tabletop = "gauntlet.env.tabletop:TabletopEnv"
tabletop-pybullet = "gauntlet.env.pybullet.tabletop_pybullet:PyBulletTabletopEnv"
tabletop-genesis = "gauntlet.env.genesis.tabletop_genesis:GenesisTabletopEnv"
tabletop-isaac = "gauntlet.env.isaac.tabletop_isaac:IsaacSimTabletopEnv"
```

The heavy backends (huggingface / lerobot / pybullet / genesis / isaac)
stay behind their existing optional extras — their entry-point strings
are registered but `ep.load()` only fires on demand. A user without the
`[hf]` extra never imports torch just because `entry_points()` is called.

## 6. Special case: `random` and the `action_dim` partial

`gauntlet.policy.registry.resolve_policy_factory("random")` today
returns `partial(RandomPolicy, action_dim=7)` because `RandomPolicy`
needs that argument at construction time. The plugin entry-point string
points at the bare class, so resolving `"random"` through the plugin
path would crash on a missing `action_dim`.

Resolution: the legacy short-circuits in `resolve_policy_factory`
(`"random"`, `"scripted"`, `"module:attr"`) stay byte-identical. The
plugin discovery is consulted **only** when the spec is a bare word
that does not match a built-in shortcut. That covers the
`gauntlet run --policy sb3` motivating example without breaking the
existing `--policy random` partial.

## 7. Public surface (new module: `gauntlet.plugins`)

```python
def discover_policy_plugins() -> dict[str, type[Policy]]: ...
def discover_env_plugins() -> dict[str, type[GauntletEnv]]: ...
```

Both functions are `@lru_cache(maxsize=1)`-cached. Tests that mock
`importlib.metadata.entry_points` must call `<fn>.cache_clear()` in
fixture teardown.

## 8. Open questions / non-goals

* **Versioning**: entry-point names have no semver. We document in
  `docs/plugin-development.md` the convention that breaking changes to
  a plugin's adapter API require a new entry-point name (`sb3` →
  `sb3-v2`), not a silent class swap.
* **Validation**: we do *not* runtime-check that a discovered class
  satisfies the `Policy` / `GauntletEnv` protocol at discovery time —
  that would force an import of the plugin code. The structural check
  happens when the runner actually calls `policy.act()` /
  `env.reset()`. Fast, lazy, lossy.
* **Out of scope** (Phase 3+): plugin manifests beyond a single class
  string; per-plugin configuration; entry-point-discovered Suite YAML
  templates.
