"""Unit tests for the env registry (Phase 2 Task 5 step 3, RFC-005 §3.4).

These tests pin the registry's public surface — register_env /
get_env_factory / registered_envs — independent of any real backend.
They use freshly-namespaced test names (``_testreg_*``) so the
built-in ``tabletop`` registration from ``gauntlet.env.__init__`` does
not collide, and so the monotonically-growing module-level registry is
safe under the default session-scoped conftest.

Phase 3 (plugin-system polish task) extended the suite with
``resolve_env_factory`` plugin-fallthrough behaviour mirroring the
``resolve_policy`` tests in ``tests/test_policy_registry.py``.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast
from unittest.mock import patch

import pytest

from gauntlet.env import registry as registry_mod
from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import (
    get_env_factory,
    register_env,
    registered_envs,
    resolve_env_factory,
)

_NAME_COUNTER = itertools.count()


def _unique(prefix: str) -> str:
    """Registry entries are global; scope test names per-invocation."""
    return f"_testreg_{prefix}_{next(_NAME_COUNTER)}"


class _StubEnv:
    """Minimal placeholder — registry does not inspect its return type."""


def _factory(**_: Any) -> _StubEnv:
    return _StubEnv()


# The registry stores ``Callable[..., GauntletEnv]`` at its type boundary, but
# the stub only needs to round-trip through the dict — never satisfies the
# full Protocol. ``cast`` documents the deliberate laxity at the test seam.
_stub_factory: Callable[..., GauntletEnv] = cast(Callable[..., GauntletEnv], _factory)


def test_register_then_retrieve_round_trips() -> None:
    name = _unique("roundtrip")
    register_env(name, _stub_factory)
    assert get_env_factory(name) is _stub_factory


def test_registered_envs_reflects_registration() -> None:
    name = _unique("listed")
    assert name not in registered_envs()
    register_env(name, _stub_factory)
    assert name in registered_envs()


def test_registered_envs_returns_frozenset() -> None:
    result = registered_envs()
    assert isinstance(result, frozenset)


def test_double_registration_raises_value_error() -> None:
    name = _unique("dup")
    register_env(name, _stub_factory)
    with pytest.raises(ValueError, match=name):
        register_env(name, _stub_factory)


def test_get_env_factory_unknown_raises_with_sorted_registered_list() -> None:
    with pytest.raises(ValueError, match="unknown env"):
        get_env_factory("_testreg_does_not_exist_sentinel")

    # Register two entries and confirm the error message sorts them.
    a = _unique("sortedA")
    b = _unique("sortedB")
    register_env(a, _stub_factory)
    register_env(b, _stub_factory)
    with pytest.raises(ValueError) as excinfo:
        get_env_factory("_testreg_still_missing_sentinel")
    msg = str(excinfo.value)
    assert a in msg and b in msg
    # sorted() — ``a`` index must appear before ``b`` index in the message.
    assert msg.index(a) < msg.index(b)


def test_factory_call_returns_stub_instance() -> None:
    """The registry stores callables, not instances — calling must work."""
    name = _unique("callable")
    register_env(name, _stub_factory)
    factory = get_env_factory(name)
    env = factory()
    assert isinstance(env, _StubEnv)


def test_module_exposes_only_public_api() -> None:
    """Public surface includes the four documented callables."""
    public = {n for n in dir(registry_mod) if not n.startswith("_")}
    assert {
        "register_env",
        "get_env_factory",
        "registered_envs",
        "resolve_env_factory",
    } <= public


# ──────────────────────────────────────────────────────────────────────
# Phase 3 — plugin fallthrough via resolve_env_factory.
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def _isolated_env_plugin_cache() -> Iterator[None]:
    """Clear the env plugin discovery lru_cache between tests."""
    from gauntlet.plugins import discover_env_plugins

    discover_env_plugins.cache_clear()
    yield
    discover_env_plugins.cache_clear()


def _patch_env_plugins(plugins: Mapping[str, type[Any]]) -> Any:
    """Patch :func:`gauntlet.plugins.discover_env_plugins` to return ``plugins``."""
    return patch(
        "gauntlet.plugins.discover_env_plugins",
        return_value=plugins,
    )


class _FakePluginEnv:
    """Duck-typed env class returned by tests' fake plugin entries."""

    AXIS_NAMES: frozenset[str] = frozenset()


def test_resolve_env_factory_returns_builtin_when_present(
    _isolated_env_plugin_cache: None,
) -> None:
    """The first-party ``tabletop`` registration wins over plugin discovery."""
    # ``tabletop`` is registered by ``gauntlet.env.__init__`` at import time.
    factory = resolve_env_factory("tabletop")
    # Identity round-trip with the imperative table — proves no plugin shim.
    assert factory is get_env_factory("tabletop")


def test_resolve_env_factory_falls_through_to_plugin(
    _isolated_env_plugin_cache: None,
) -> None:
    """An unknown built-in name resolves through the plugin table."""
    fake_plugins = {"third-party-env": _FakePluginEnv}
    with _patch_env_plugins(fake_plugins):
        factory = resolve_env_factory("third-party-env")
    # The cast in resolve_env_factory widens the class to a Callable; calling
    # it constructs an instance of the fake.
    instance = factory()
    assert isinstance(instance, _FakePluginEnv)


def test_resolve_env_factory_unknown_name_lists_available(
    _isolated_env_plugin_cache: None,
) -> None:
    """Unknown name surfaces both built-in and plugin names in the error."""
    with (
        _patch_env_plugins({"third-party-env": _FakePluginEnv}),
        pytest.raises(ValueError, match="unknown env") as excinfo,
    ):
        resolve_env_factory("definitely-not-here")
    msg = str(excinfo.value)
    assert "tabletop" in msg
    assert "third-party-env" in msg


def test_resolve_env_factory_collision_warns_and_keeps_builtin(
    _isolated_env_plugin_cache: None,
) -> None:
    """A plugin shadowing ``tabletop`` warns + keeps the built-in factory."""
    fake_plugins = {"tabletop": _FakePluginEnv}
    builtin = get_env_factory("tabletop")
    with (
        _patch_env_plugins(fake_plugins),
        pytest.warns(RuntimeWarning, match="shadows the built-in 'tabletop'"),
    ):
        result = resolve_env_factory("tabletop")
    assert result is builtin


def test_resolve_env_factory_dogfood_collision_silent(
    _isolated_env_plugin_cache: None,
) -> None:
    """Identity collision (gauntlet's own dogfood entry points) must NOT warn.

    The dogfooded ``tabletop`` entry point loads the same class object
    the imperative ``register_env`` already holds, so the
    identity-based collision check should stay silent.
    """
    builtin = get_env_factory("tabletop")
    fake_plugins = {"tabletop": cast(type[Any], builtin)}
    with (
        _patch_env_plugins(fake_plugins),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error", RuntimeWarning)
        result = resolve_env_factory("tabletop")
    assert result is builtin
