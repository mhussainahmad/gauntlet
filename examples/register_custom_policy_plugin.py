"""Demonstrate gauntlet's third-party policy-plugin registration path.

Usage:
    uv run python examples/register_custom_policy_plugin.py

Real-world plugin registration happens at *package install time*: a
third-party distribution declares its policy class under the
``gauntlet.policies`` entry-point group in its ``pyproject.toml`` and
``pip install`` makes that class discoverable via
:func:`importlib.metadata.entry_points`.

The pyproject.toml snippet a third-party package would ship looks
like this::

    # In the third-party plugin package's pyproject.toml:
    [project]
    name = "my-gauntlet-plugin"
    version = "0.1.0"
    dependencies = ["gauntlet"]

    [project.entry-points."gauntlet.policies"]
    my-constant = "my_gauntlet_plugin.policies:MyConstantPolicy"

After ``pip install my-gauntlet-plugin`` the harness's
:func:`gauntlet.policy.resolve_policy` resolver picks the plugin up
automatically — no gauntlet code change needed.

Because that flow requires a real package install we cannot
reproduce it inside a single example file. Instead this demo uses the
**programmatic shortcut** that
:mod:`gauntlet.plugins`'s own unit tests use: monkey-patch
``gauntlet.plugins.entry_points`` to return a hand-built
:class:`importlib.metadata.EntryPoint` table, then invalidate the
:func:`functools.lru_cache` on :func:`discover_policy_plugins`.

This is a demo path — a real plugin should always be a properly
installed package.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from gauntlet.plugins import POLICY_ENTRY_POINT_GROUP, discover_policy_plugins
from gauntlet.policy import Policy, resolve_policy_factory
from gauntlet.policy.base import Action, Observation
from gauntlet.policy.registry import resolve_policy

__all__ = ["MyConstantPolicy", "main"]


# ---------------------------------------------------------------------------
# Step 1: define the third-party policy class.
# ---------------------------------------------------------------------------


class MyConstantPolicy:
    """Toy policy that emits a constant zero action of fixed length.

    Satisfies the :class:`gauntlet.policy.base.Policy` protocol — the
    Runner duck-types via :func:`isinstance(obj, Policy)` so this
    plain class is enough; no inheritance required. A real plugin
    would do meaningful inference inside :meth:`act`.

    The class is deliberately zero-arg constructible so the registry
    can use the class itself as the factory (the documented contract
    for plugin entry points — see ``docs/plugin-development.md``).
    """

    def __init__(self, action_dim: int = 7) -> None:
        self._action_dim = action_dim

    def act(self, obs: Observation) -> Action:
        del obs  # Constant policy ignores observations.
        return np.zeros(self._action_dim, dtype=np.float64)


# ---------------------------------------------------------------------------
# Step 2: build a fake EntryPoint table for the demo.
# ---------------------------------------------------------------------------


class _FakeDist:
    """Minimal stand-in for :class:`importlib.metadata.Distribution`.

    :mod:`gauntlet.plugins` only reads ``.name`` off ``EntryPoint.dist``
    so a duck-typed shim is enough — same approach
    ``tests/test_plugins.py`` uses.
    """

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeEntryPoint:
    """Duck-typed :class:`importlib.metadata.EntryPoint`.

    The real EntryPoint class is an immutable NamedTuple whose
    ``load()`` looks up ``module:attr`` via :mod:`importlib`. Tests
    (and this example) bypass that lookup by returning the target
    object directly from :meth:`load`.
    """

    def __init__(self, name: str, target: type[Policy], dist_name: str) -> None:
        self.name = name
        self._target = target
        self.dist = _FakeDist(dist_name)

    def load(self) -> type[Policy]:
        return self._target


def _fake_entry_points_factory(
    policy_eps: list[_FakeEntryPoint],
) -> Any:
    """Build the side_effect for :func:`unittest.mock.patch`.

    :mod:`gauntlet.plugins` calls ``entry_points(group=...)`` only —
    we honour that signature and dispatch by group name. Other groups
    (e.g. ``gauntlet.envs``) get an empty list so the harness keeps
    working if the rest of the codebase calls into discovery.
    """

    def _fake(*, group: str) -> Iterable[_FakeEntryPoint]:
        if group == POLICY_ENTRY_POINT_GROUP:
            return policy_eps
        return []

    return _fake


# ---------------------------------------------------------------------------
# Step 3: drive the demo.
# ---------------------------------------------------------------------------


def main() -> None:
    """Register :class:`MyConstantPolicy` programmatically and look it up.

    The whole demo runs inside a single :func:`unittest.mock.patch`
    context so the fake entry-point table is only visible while the
    demo is running. The :func:`functools.lru_cache` on
    :func:`discover_policy_plugins` is invalidated before and after
    the patch so neither this script nor anything that imports it
    later sees stale state.
    """
    plugin_name = "my-constant"
    fake_eps = [
        _FakeEntryPoint(
            name=plugin_name,
            target=cast(type[Policy], MyConstantPolicy),
            dist_name="my-gauntlet-plugin",
        )
    ]

    # Cache invalidation guards: same pattern tests/test_plugins.py
    # uses, and required because discover_policy_plugins is decorated
    # with functools.lru_cache(maxsize=1).
    discover_policy_plugins.cache_clear()
    try:
        with patch(
            "gauntlet.plugins.entry_points",
            side_effect=_fake_entry_points_factory(fake_eps),
        ):
            # 3a: discover_policy_plugins surfaces the plugin.
            registry = discover_policy_plugins()
            assert plugin_name in registry, (
                f"expected {plugin_name!r} in plugin registry; got {sorted(registry)}"
            )
            print(f"discover_policy_plugins() found {len(registry)} plugin(s): {sorted(registry)}")

            # 3b: resolve_policy returns the *class* by name. Built-ins
            # (random, scripted) always win on collision; bare-word
            # plugin names like ours fall through to the registry.
            cls = resolve_policy(plugin_name)
            assert cls is MyConstantPolicy
            print(f"resolve_policy({plugin_name!r}) -> {cls.__module__}.{cls.__name__}")

            # 3c: resolve_policy_factory returns a zero-arg callable
            # producing a fresh policy instance — the same contract
            # the CLI's ``--policy <name>`` flag uses to plug a plugin
            # into Runner.run(...).
            factory = resolve_policy_factory(plugin_name)
            policy = factory()
            assert isinstance(policy, Policy)
            obs: Observation = {"state": np.zeros(3, dtype=np.float64)}
            action = policy.act(obs)
            print(
                f"resolve_policy_factory({plugin_name!r})() -> "
                f"instance with action.shape={action.shape}, "
                f"action.dtype={action.dtype}, sum(action)={action.sum():.1f}"
            )
    finally:
        # Always invalidate the cache on the way out so a downstream
        # caller in the same Python session does not see the fake
        # entry points.
        discover_policy_plugins.cache_clear()

    print(
        "Demo complete. Real plugins ship via the pyproject.toml "
        '[project.entry-points."gauntlet.policies"] table — see this '
        "script's module docstring for the snippet."
    )


def _smoke_assert_action_dim_default() -> None:
    """Quick sanity check the policy class is healthy outside the demo.

    Kept tiny on purpose — example files are not tests, but a one-line
    construct + act keeps the demo honest if MyConstantPolicy ever
    drifts from the Policy protocol.
    """
    pol = MyConstantPolicy()
    out: NDArray[np.float64] = pol.act({"state": np.zeros(3, dtype=np.float64)})
    assert out.shape == (7,)
    assert out.dtype == np.float64


if __name__ == "__main__":  # pragma: no cover
    _smoke_assert_action_dim_default()
    main()
