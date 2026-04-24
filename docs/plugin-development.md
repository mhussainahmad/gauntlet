# Plugin development guide

`gauntlet` uses Python's standard
[`importlib.metadata` entry-point mechanism](https://packaging.python.org/en/latest/specifications/entry-points/)
so any pip-installable package can register policy or env adapters
without modifying gauntlet's source. This guide walks through writing,
publishing, and testing a plugin.

## Overview

Two entry-point groups, both consumed by
:mod:`gauntlet.plugins`:

| Group              | What it registers                                                              | Resolved by                                                  |
|--------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------|
| `gauntlet.policies`| A class (or zero-arg callable) returning a `gauntlet.policy.base.Policy`       | `gauntlet.policy.registry.resolve_policy_factory`            |
| `gauntlet.envs`    | A class returning a `gauntlet.env.base.GauntletEnv`                            | `gauntlet.env.registry.resolve_env_factory`                  |

Built-in adapters always win on collision. Plugin discovery is lazy:
the entry-point string is registered at install time, but the plugin's
Python code only imports when a user actually references the plugin
name. Failed imports are wrapped in a `RuntimeWarning` and the plugin
is dropped from the registry — gauntlet itself stays operational.

## Writing a `Policy` plugin

A policy adapter is any class satisfying the
`gauntlet.policy.base.Policy` Protocol — i.e. a single
`act(obs: Mapping[str, NDArray]) -> NDArray[np.float64]` method. If your
adapter needs per-episode state, also implement
`reset(rng: np.random.Generator) -> None` (the `ResettablePolicy`
Protocol).

### Minimal example

```python
# src/my_gauntlet_plugin/sb3_adapter.py
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO


class SBAdapter:
    """Adapt a stable-baselines3 PPO model to the gauntlet Policy Protocol."""

    def __init__(self, model_path: str = "ppo_default.zip") -> None:
        self._model = PPO.load(model_path)

    def act(self, obs: Mapping[str, NDArray[Any]]) -> NDArray[np.float64]:
        action, _ = self._model.predict(obs["state"], deterministic=True)
        return np.asarray(action, dtype=np.float64)
```

### Register the plugin in your package's `pyproject.toml`

```toml
[project]
name = "my-gauntlet-plugin"
version = "0.1.0"
dependencies = [
    "gauntlet>=0.1,<1",
    "stable-baselines3>=2.0,<3",
]

[project.entry-points."gauntlet.policies"]
sb3 = "my_gauntlet_plugin.sb3_adapter:SBAdapter"
```

After `pip install my-gauntlet-plugin`, the user runs:

```
gauntlet run --suite suites/reach.yaml --policy sb3
```

…and gauntlet resolves `sb3` through the plugin path.

### Constructor arguments

The plugin path treats the registered entry as a **zero-arg factory**.
If your adapter needs constructor arguments, register a zero-arg
callable instead of the class:

```python
# src/my_gauntlet_plugin/factories.py
from my_gauntlet_plugin.sb3_adapter import SBAdapter

def make_default_sb3() -> SBAdapter:
    return SBAdapter(model_path="checkpoints/policy_v3.zip")
```

```toml
[project.entry-points."gauntlet.policies"]
sb3-v3 = "my_gauntlet_plugin.factories:make_default_sb3"
```

For more configurability, expose multiple entry-point names —
`sb3-small`, `sb3-large`, etc. — each pointing at a distinct factory.

## Writing an `Env` plugin

An env adapter is any class satisfying the
`gauntlet.env.base.GauntletEnv` Protocol. The full surface is
documented in that module; the short version is:

* `reset(*, seed=None, options=None) -> (obs, info)`
* `step(action) -> (obs, reward, terminated, truncated, info)`
* `set_perturbation(name, value) -> None` (queues for next reset)
* `restore_baseline() -> None`
* `close() -> None`
* `AXIS_NAMES: ClassVar[frozenset[str]]`
* `observation_space`, `action_space`

### Minimal example

```python
# src/my_gauntlet_plugin/cartpole_env.py
from typing import Any, ClassVar

import gymnasium as gym
from numpy.typing import NDArray


class GauntletCartPole:
    """Wrap gymnasium CartPole-v1 with the GauntletEnv interface."""

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init__(self) -> None:
        self._env = gym.make("CartPole-v1")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._initial_obs: NDArray[Any] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        return {"state": obs}, info

    def step(self, action: NDArray[Any]) -> tuple[
        dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]
    ]:
        obs, reward, terminated, truncated, info = self._env.step(int(action[0]))
        return {"state": obs}, float(reward), terminated, truncated, info

    def set_perturbation(self, name: str, value: float) -> None:
        raise ValueError(f"unsupported perturbation axis: {name!r}")

    def restore_baseline(self) -> None:
        pass

    def close(self) -> None:
        self._env.close()
```

```toml
[project.entry-points."gauntlet.envs"]
cartpole = "my_gauntlet_plugin.cartpole_env:GauntletCartPole"
```

After install:

```yaml
# suites/cartpole.yaml
env: cartpole
episodes:
  - name: baseline
    seed: 0
```

## Lazy-import recommendation

**Do not import heavy dependencies at module top-level.** Entry-point
discovery is allowed to be cheap; users without your plugin's heavy
deps installed should not see torch / pybullet / etc. imported just
because gauntlet enumerated entry points.

The wrong way:

```python
# my_gauntlet_plugin/__init__.py
import torch  # imported every time entry_points() runs
import stable_baselines3
```

The right way — defer to the adapter file:

```python
# my_gauntlet_plugin/__init__.py
# (intentionally empty)

# my_gauntlet_plugin/sb3_adapter.py
import torch  # only imported if a user resolves the `sb3` plugin
import stable_baselines3
```

If your adapter file has unavoidable side-effecting imports, wrap them
with a clear `ImportError` describing the missing extra:

```python
try:
    import stable_baselines3
except ImportError as exc:
    raise ImportError(
        "my-gauntlet-plugin requires stable-baselines3; "
        "install with: pip install 'my-gauntlet-plugin[sb3]'"
    ) from exc
```

`gauntlet.plugins` will catch the `ImportError`, wrap it in a
`RuntimeWarning` naming your distribution, and drop the plugin so the
rest of the harness keeps working.

## Versioning + entry-point names

Entry-point names have no semver. `gauntlet` treats the name as an
opaque string and trusts that the loaded class satisfies the
`Policy` / `GauntletEnv` Protocol. To version your plugin's API:

* Keep the *name* stable for the lifetime of an API contract.
  `sb3 = "...:SBAdapter"` should keep meaning "the canonical PPO
  adapter that takes no constructor args".
* Bump the entry-point name when you make a breaking change:
  `sb3 → sb3-v2` rather than silently swapping the class behind `sb3`.
* Document supported gauntlet versions in your plugin's README.
  `gauntlet>=0.1,<1` is the conventional pin against the current
  pre-release line.

## Testing your plugin

Two recipes, depending on whether you want fast unit tests or a real
integration test.

### Recipe 1 — Mock the discovery layer (fast)

```python
# tests/test_my_plugin.py
from unittest.mock import patch

from gauntlet.policy.registry import resolve_policy
from gauntlet.plugins import discover_policy_plugins

from my_gauntlet_plugin.sb3_adapter import SBAdapter


def test_sb3_adapter_resolves_via_plugin_path() -> None:
    discover_policy_plugins.cache_clear()
    with patch(
        "gauntlet.plugins.discover_policy_plugins",
        return_value={"sb3": SBAdapter},
    ):
        cls = resolve_policy("sb3")
    assert cls is SBAdapter
```

### Recipe 2 — Real install in a venv (integration)

In CI, set up a job that:

1. `pip install -e .` your plugin repo (this materialises the entry
   points).
2. `pip install gauntlet`.
3. Runs `python -c "from gauntlet.plugins import discover_policy_plugins; print(discover_policy_plugins())"` and asserts your plugin name appears.
4. Runs `gauntlet run --suite suites/integration.yaml --policy sb3`
   end-to-end on a tiny test suite.

This is the gold standard but slow — keep it on a single CI job, not
in every developer's local pre-commit.

### Verifying the structural Protocol fit

`gauntlet.policy.base.Policy` and `gauntlet.env.base.GauntletEnv` are
both `runtime_checkable` Protocols, so:

```python
from gauntlet.env.base import GauntletEnv
from my_gauntlet_plugin.cartpole_env import GauntletCartPole


def test_cartpole_satisfies_protocol() -> None:
    env = GauntletCartPole()
    try:
        assert isinstance(env, GauntletEnv)
    finally:
        env.close()
```

Failures here surface a missing method or attribute before the runner
catches it on first `reset()`.

## Diagnostic commands

To see every plugin gauntlet has loaded in your environment:

```python
from gauntlet.plugins import discover_policy_plugins, discover_env_plugins

print("policies:", sorted(discover_policy_plugins()))
print("envs:", sorted(discover_env_plugins()))
```

If your plugin name is missing:

* Confirm `pip show my-gauntlet-plugin` lists the package.
* Confirm `python -c "from importlib.metadata import entry_points; print(list(entry_points(group='gauntlet.policies')))"` includes your name.
* If the plugin's `load()` raises, a `RuntimeWarning` is emitted —
  re-run with `python -W default` to surface it.
