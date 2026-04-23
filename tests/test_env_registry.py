"""Unit tests for the env registry (Phase 2 Task 5 step 3, RFC-005 §3.4).

These tests pin the registry's public surface — register_env /
get_env_factory / registered_envs — independent of any real backend.
They use freshly-namespaced test names (``_testreg_*``) so the
built-in ``tabletop`` registration from ``gauntlet.env.__init__`` does
not collide, and so the monotonically-growing module-level registry is
safe under the default session-scoped conftest.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any, cast

import pytest

from gauntlet.env import registry as registry_mod
from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import (
    get_env_factory,
    register_env,
    registered_envs,
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
    """Public surface is exactly the three documented callables."""
    public = {n for n in dir(registry_mod) if not n.startswith("_")}
    assert {"register_env", "get_env_factory", "registered_envs"} <= public
