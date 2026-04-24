"""Direct unit tests for ``gauntlet.policy.registry.resolve_policy_factory``.

Phase 2.5 Task 11. The CLI tests in ``tests/test_cli.py`` already cover
the happy paths (``"random"``, ``"module:attr"``) and one error path
(unknown spec). This file fills in the under-covered ``_resolve_module_attr``
branches: empty / multi-colon spec, missing attribute, non-callable
attribute, unimportable module, and the empty-string guard at the
top-level resolver.

Phase 3 (plugin-system polish task) extended the suite with
``POLICY_REGISTRY`` introspection + plugin-fallthrough behaviour for
both :func:`resolve_policy_factory` and the new :func:`resolve_policy`.

All tests are pure unit checks — no env, no Runner — and run in well
under 50 ms each.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from typing import Any, cast
from unittest.mock import patch

import pytest

from gauntlet.policy.random import RandomPolicy
from gauntlet.policy.registry import (
    POLICY_REGISTRY,
    PolicySpecError,
    resolve_policy,
    resolve_policy_factory,
)
from gauntlet.policy.scripted import ScriptedPolicy

# Module-level non-callable + zero-arg-callable used by the
# ``module:attr`` error-path tests below. They must live at module scope
# so ``importlib.import_module`` finds them.

_NOT_A_CALLABLE: int = 42


def _ok_policy_factory() -> RandomPolicy:
    return RandomPolicy(action_dim=7)


# ──────────────────────────────────────────────────────────────────────
# Happy paths.
# ──────────────────────────────────────────────────────────────────────


def test_resolve_random_returns_random_policy_factory() -> None:
    factory = resolve_policy_factory("random")
    policy = factory()
    assert isinstance(policy, RandomPolicy)


def test_resolve_scripted_returns_scripted_policy_class() -> None:
    factory = resolve_policy_factory("scripted")
    policy = factory()
    assert isinstance(policy, ScriptedPolicy)


def test_resolve_module_attr_happy_path() -> None:
    factory = resolve_policy_factory("tests.test_policy_registry:_ok_policy_factory")
    assert isinstance(factory(), RandomPolicy)


def test_resolve_strips_surrounding_whitespace() -> None:
    factory = resolve_policy_factory("  random  ")
    assert isinstance(factory(), RandomPolicy)


# ──────────────────────────────────────────────────────────────────────
# Error paths — top-level resolve_policy_factory.
# ──────────────────────────────────────────────────────────────────────


def test_empty_string_spec_rejected() -> None:
    with pytest.raises(PolicySpecError, match="non-empty"):
        resolve_policy_factory("")


def test_whitespace_only_spec_rejected() -> None:
    with pytest.raises(PolicySpecError, match="non-empty"):
        resolve_policy_factory("   ")


def test_unknown_shortcut_rejected() -> None:
    """A bare word that is neither ``random`` / ``scripted`` nor
    contains ``:`` is rejected with the ``unknown policy spec`` hint."""
    with pytest.raises(PolicySpecError, match="unknown policy spec"):
        resolve_policy_factory("policy-name-without-colon")


# ──────────────────────────────────────────────────────────────────────
# Error paths — _resolve_module_attr.
# ──────────────────────────────────────────────────────────────────────


def test_module_spec_without_colon_rejected() -> None:
    """A spec whose left/right halves cannot be parsed because there
    are zero colons hits the top-level ``unknown`` branch (covered
    above). A spec with TWO colons hits the
    ``_resolve_module_attr`` count check."""
    with pytest.raises(PolicySpecError, match="exactly one ':'"):
        resolve_policy_factory("a:b:c")


def test_module_spec_empty_module_path() -> None:
    with pytest.raises(PolicySpecError, match="non-empty"):
        resolve_policy_factory(":attr")


def test_module_spec_empty_attribute_name() -> None:
    with pytest.raises(PolicySpecError, match="non-empty"):
        resolve_policy_factory("module:")


def test_module_spec_unimportable_module() -> None:
    with pytest.raises(PolicySpecError, match="could not import module"):
        resolve_policy_factory("definitely_not_a_real_module_42:attr")


def test_module_spec_missing_attribute() -> None:
    with pytest.raises(PolicySpecError, match="has no attribute"):
        resolve_policy_factory("tests.test_policy_registry:does_not_exist")


def test_module_spec_attribute_is_not_callable() -> None:
    with pytest.raises(PolicySpecError, match="not callable"):
        resolve_policy_factory("tests.test_policy_registry:_NOT_A_CALLABLE")


# ──────────────────────────────────────────────────────────────────────
# PolicySpecError is a ValueError subclass — catchable both ways.
# ──────────────────────────────────────────────────────────────────────


def test_policy_spec_error_is_value_error() -> None:
    """Existing ``except ValueError`` handlers must keep working."""
    with pytest.raises(ValueError):
        resolve_policy_factory("")


# ──────────────────────────────────────────────────────────────────────
# Phase 3 — built-in registry table + plugin fallthrough.
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def _isolated_plugin_cache() -> Iterator[None]:
    """Clear the plugin discovery lru_cache so per-test mocks take effect."""
    from gauntlet.plugins import discover_policy_plugins

    discover_policy_plugins.cache_clear()
    yield
    discover_policy_plugins.cache_clear()


def _patch_policy_plugins(plugins: Mapping[str, type[Any]]) -> Any:
    """Patch :func:`gauntlet.plugins.discover_policy_plugins` to return ``plugins``."""
    return patch(
        "gauntlet.plugins.discover_policy_plugins",
        return_value=plugins,
    )


class _ZeroArgFakePolicy:
    """Dummy zero-arg policy used to exercise plugin-resolution paths."""

    def act(self, obs: dict[str, Any]) -> Any:
        raise NotImplementedError


def test_policy_registry_contains_builtins() -> None:
    """The built-in table must hold the two first-party adapters."""
    assert set(POLICY_REGISTRY) == {"random", "scripted"}
    assert POLICY_REGISTRY["random"] is cast(Any, RandomPolicy)
    assert POLICY_REGISTRY["scripted"] is cast(Any, ScriptedPolicy)


def test_resolve_policy_returns_builtin_class(_isolated_plugin_cache: None) -> None:
    """``resolve_policy`` returns the class itself, not a factory."""
    assert resolve_policy("random") is cast(Any, RandomPolicy)
    assert resolve_policy("scripted") is cast(Any, ScriptedPolicy)


def test_resolve_policy_returns_plugin_class_on_unknown_builtin(
    _isolated_plugin_cache: None,
) -> None:
    """A plugin name resolves to its loaded class when no built-in shadows it."""
    fake_plugins = {"third-party": _ZeroArgFakePolicy}
    with _patch_policy_plugins(fake_plugins):
        result = resolve_policy("third-party")
    assert result is cast(Any, _ZeroArgFakePolicy)


def test_resolve_policy_unknown_name_lists_available(
    _isolated_plugin_cache: None,
) -> None:
    """Unknown name surfaces a helpful 'available:' list."""
    with (
        _patch_policy_plugins({"sb3": _ZeroArgFakePolicy}),
        pytest.raises(PolicySpecError, match="unknown policy") as excinfo,
    ):
        resolve_policy("does-not-exist")
    msg = str(excinfo.value)
    assert "random" in msg and "scripted" in msg and "sb3" in msg


def test_resolve_policy_collision_warns_and_keeps_builtin(
    _isolated_plugin_cache: None,
) -> None:
    """A plugin shadowing a built-in must warn but return the built-in class."""
    fake_plugins = {"random": _ZeroArgFakePolicy}
    with (
        _patch_policy_plugins(fake_plugins),
        pytest.warns(RuntimeWarning, match="shadows the built-in 'random'"),
    ):
        result = resolve_policy("random")
    assert result is cast(Any, RandomPolicy)


def test_resolve_policy_dogfood_collision_silent(
    _isolated_plugin_cache: None,
) -> None:
    """Identity collision (gauntlet's own dogfood entry points) must NOT warn."""
    # Same class object the built-in table holds — the dogfood case.
    fake_plugins = {"random": cast(type, RandomPolicy)}
    with _patch_policy_plugins(fake_plugins), warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = resolve_policy("random")
    assert result is cast(Any, RandomPolicy)


def test_resolve_policy_factory_falls_through_to_plugin(
    _isolated_plugin_cache: None,
) -> None:
    """``--policy <plugin-name>`` resolves a plugin's bare class as a zero-arg factory."""
    fake_plugins = {"sb3": _ZeroArgFakePolicy}
    with _patch_policy_plugins(fake_plugins):
        factory = resolve_policy_factory("sb3")
    instance = factory()
    assert isinstance(instance, _ZeroArgFakePolicy)


def test_resolve_policy_factory_unknown_lists_plugins_in_error(
    _isolated_plugin_cache: None,
) -> None:
    """Unknown bare-word with no matching plugin must surface installed plugin names."""
    with (
        _patch_policy_plugins({"sb3": _ZeroArgFakePolicy}),
        pytest.raises(PolicySpecError, match="unknown policy spec") as excinfo,
    ):
        resolve_policy_factory("missing")
    assert "sb3" in str(excinfo.value)


def test_resolve_policy_factory_random_keeps_action_dim_partial(
    _isolated_plugin_cache: None,
) -> None:
    """`random` must keep its `partial(action_dim=7)` wrapper, NOT route through plugins.

    A plugin entry-point string for `random` points at the bare class,
    which would crash on a missing ``action_dim`` if resolved zero-arg.
    The legacy short-circuit guards against that even when the plugin
    table contains a "random" entry (the dogfood case).
    """
    fake_plugins = {"random": cast(type, RandomPolicy)}
    with _patch_policy_plugins(fake_plugins):
        factory = resolve_policy_factory("random")
    policy = factory()
    assert isinstance(policy, RandomPolicy)
    # The action_dim default is wired in (the partial would crash if not).
    assert policy.action_dim == 7
