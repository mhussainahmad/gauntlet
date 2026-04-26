"""Unit tests for :mod:`gauntlet.plugins` (entry-point discovery).

These tests inject fake plugins by patching
:func:`importlib.metadata.entry_points` so they exercise discovery,
failure-wrapping, and dedupe without needing a real pip install. The
``_clear_caches`` autouse fixture invalidates the
:func:`functools.lru_cache` on the discovery helpers so each test starts
clean.

Phase 3 / plugin-system polish task.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from typing import Any, cast
from unittest.mock import patch

import pytest

from gauntlet import plugins
from gauntlet.plugins import (
    ENV_ENTRY_POINT_GROUP,
    POLICY_ENTRY_POINT_GROUP,
    discover_env_plugins,
    discover_policy_plugins,
    warn_on_collision,
)

# ──────────────────────────────────────────────────────────────────────
# Test fixtures.
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_plugin_caches() -> Iterator[None]:
    """Invalidate the lru_cache on each discover_* helper before + after.

    Without this, the first test's mocked entry points would leak into
    every subsequent test in the module.
    """
    discover_policy_plugins.cache_clear()
    discover_env_plugins.cache_clear()
    yield
    discover_policy_plugins.cache_clear()
    discover_env_plugins.cache_clear()


class _FakeDist:
    """Minimal stand-in for ``EntryPoint.dist`` used in tests."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeEntryPoint:
    """Minimal duck-typed EntryPoint.

    Real :class:`importlib.metadata.EntryPoint` instances are immutable
    NamedTuples; constructing one with a populated ``dist`` is awkward
    in tests. A duck type is enough here because :mod:`gauntlet.plugins`
    only touches ``.name``, ``.dist``, and ``.load()``.
    """

    def __init__(self, name: str, target: Any, dist_name: str = "fake-pkg") -> None:
        self.name = name
        self._target = target
        self.dist = _FakeDist(dist_name)

    def load(self) -> Any:
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def _patch_entry_points(
    *,
    policy_eps: list[_FakeEntryPoint] | None = None,
    env_eps: list[_FakeEntryPoint] | None = None,
) -> Any:
    """Return a context manager patching :func:`importlib.metadata.entry_points`.

    The mocked function inspects the ``group=`` kwarg and returns the
    matching list (or an empty list for unknown groups). Real Python
    code paths inside :mod:`gauntlet.plugins` call ``entry_points`` with
    ``group=...`` only — see the source.
    """
    policy_eps = policy_eps or []
    env_eps = env_eps or []

    def _fake(*, group: str) -> list[_FakeEntryPoint]:
        if group == POLICY_ENTRY_POINT_GROUP:
            return policy_eps
        if group == ENV_ENTRY_POINT_GROUP:
            return env_eps
        return []

    return patch("gauntlet.plugins.entry_points", side_effect=_fake)


# ──────────────────────────────────────────────────────────────────────
# Sample fake plugin classes.
# ──────────────────────────────────────────────────────────────────────


class _FakePolicy:
    def act(self, obs: dict[str, Any]) -> Any:
        raise NotImplementedError


class _FakeEnv:
    AXIS_NAMES: frozenset[str] = frozenset()


# ──────────────────────────────────────────────────────────────────────
# discover_policy_plugins — happy path.
# ──────────────────────────────────────────────────────────────────────


def test_discover_policy_plugins_returns_loaded_classes() -> None:
    eps = [_FakeEntryPoint("fake-policy", _FakePolicy, dist_name="fake-pkg")]
    with _patch_entry_points(policy_eps=eps):
        result = discover_policy_plugins()
    assert set(result) == {"fake-policy"}
    assert result["fake-policy"] is cast(Any, _FakePolicy)


def test_discover_policy_plugins_empty_group_returns_empty_dict() -> None:
    with _patch_entry_points():
        result = discover_policy_plugins()
    assert result == {}


# ──────────────────────────────────────────────────────────────────────
# discover_env_plugins — happy path.
# ──────────────────────────────────────────────────────────────────────


def test_discover_env_plugins_returns_loaded_classes() -> None:
    eps = [_FakeEntryPoint("fake-env", _FakeEnv, dist_name="fake-env-pkg")]
    with _patch_entry_points(env_eps=eps):
        result = discover_env_plugins()
    assert set(result) == {"fake-env"}
    assert result["fake-env"] is cast(Any, _FakeEnv)


def test_discover_env_plugins_empty_group_returns_empty_dict() -> None:
    with _patch_entry_points():
        result = discover_env_plugins()
    assert result == {}


# ──────────────────────────────────────────────────────────────────────
# Failure wrapping — ImportError-raising plugins are dropped + warned.
# ──────────────────────────────────────────────────────────────────────


def test_discover_drops_plugin_that_raises_importerror() -> None:
    """A plugin whose load() raises must be dropped, with a clear warning."""
    good = _FakeEntryPoint("good", _FakePolicy, dist_name="good-pkg")
    bad = _FakeEntryPoint(
        "bad",
        ImportError("missing torch"),
        dist_name="bad-pkg",
    )
    with (
        _patch_entry_points(policy_eps=[good, bad]),
        pytest.warns(RuntimeWarning, match=r"Plugin 'bad' from 'bad-pkg' failed to load"),
    ):
        result = discover_policy_plugins()
    # 'good' survives, 'bad' is dropped.
    assert "good" in result
    assert "bad" not in result


def test_discover_drops_plugin_that_raises_arbitrary_exception() -> None:
    """Non-ImportError exceptions are also wrapped + dropped — never crash."""
    bad = _FakeEntryPoint(
        "exploding",
        RuntimeError("kaboom"),
        dist_name="boom-pkg",
    )
    with (
        _patch_entry_points(env_eps=[bad]),
        pytest.warns(RuntimeWarning, match=r"failed to load"),
    ):
        result = discover_env_plugins()
    assert result == {}


def test_failed_plugin_does_not_break_other_plugins() -> None:
    """Discovery is best-effort: one bad plugin must not poison the rest."""
    good_a = _FakeEntryPoint("alpha", _FakePolicy, dist_name="alpha-pkg")
    bad = _FakeEntryPoint(
        "beta",
        ImportError("nope"),
        dist_name="beta-pkg",
    )
    good_c = _FakeEntryPoint("gamma", _FakePolicy, dist_name="gamma-pkg")
    with _patch_entry_points(policy_eps=[good_a, bad, good_c]), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = discover_policy_plugins()
    assert set(result) == {"alpha", "gamma"}


# ──────────────────────────────────────────────────────────────────────
# Duplicate name handling.
# ──────────────────────────────────────────────────────────────────────


def test_duplicate_plugin_names_keep_first_and_warn() -> None:
    """Two distributions registering the same plugin name: first wins, warn on second."""
    first = _FakeEntryPoint("collide", _FakePolicy, dist_name="first-pkg")

    class _OtherPolicy:
        def act(self, obs: dict[str, Any]) -> Any:
            raise NotImplementedError

    second = _FakeEntryPoint("collide", _OtherPolicy, dist_name="second-pkg")
    with (
        _patch_entry_points(policy_eps=[first, second]),
        pytest.warns(
            RuntimeWarning,
            match=r"registered by both 'first-pkg' and 'second-pkg'",
        ),
    ):
        result = discover_policy_plugins()
    assert set(result) == {"collide"}
    assert result["collide"] is cast(Any, _FakePolicy)


# ──────────────────────────────────────────────────────────────────────
# lru_cache behaviour.
# ──────────────────────────────────────────────────────────────────────


def test_discover_policy_plugins_is_cached() -> None:
    """A second call with no cache_clear() must NOT re-invoke entry_points()."""
    eps = [_FakeEntryPoint("once", _FakePolicy)]
    with _patch_entry_points(policy_eps=eps) as mocked:
        first = discover_policy_plugins()
        second = discover_policy_plugins()
        # Identity, not just equality — the dict is reused.
        assert first is second
        # entry_points() was called once across both discover_* invocations.
        assert mocked.call_count == 1


def test_cache_clear_lets_tests_inject_new_plugins() -> None:
    """Tests rely on cache_clear() to swap mocked entry points."""
    first_eps = [_FakeEntryPoint("v1", _FakePolicy)]
    with _patch_entry_points(policy_eps=first_eps):
        first = discover_policy_plugins()
    discover_policy_plugins.cache_clear()
    second_eps = [_FakeEntryPoint("v2", _FakePolicy)]
    with _patch_entry_points(policy_eps=second_eps):
        second = discover_policy_plugins()
    assert set(first) == {"v1"}
    assert set(second) == {"v2"}


# ──────────────────────────────────────────────────────────────────────
# warn_on_collision — built-in vs plugin shadowing.
# ──────────────────────────────────────────────────────────────────────


def test_warn_on_collision_silent_when_objects_are_identical() -> None:
    """Dogfood case: gauntlet's own entry points load the built-in class itself."""
    builtin = _FakePolicy
    plugin = _FakePolicy  # same object — the dogfood case
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Must not raise (i.e., must not warn).
        warn_on_collision(
            name="random",
            group=POLICY_ENTRY_POINT_GROUP,
            builtin_obj=builtin,
            plugin_obj=plugin,
            plugin_dist="gauntlet",
        )


def test_warn_on_collision_warns_when_objects_differ() -> None:
    """A third-party shadow of a built-in name must emit a clear warning."""
    builtin = _FakePolicy

    class _RogueRandom:
        def act(self, obs: dict[str, Any]) -> Any:
            raise NotImplementedError

    with pytest.warns(RuntimeWarning, match=r"shadows the built-in 'random'"):
        warn_on_collision(
            name="random",
            group=POLICY_ENTRY_POINT_GROUP,
            builtin_obj=builtin,
            plugin_obj=_RogueRandom,
            plugin_dist="rogue-pkg",
        )


# ──────────────────────────────────────────────────────────────────────
# Public surface guard.
# ──────────────────────────────────────────────────────────────────────


def test_module_public_surface() -> None:
    """The public surface is exactly what's documented.

    The four newer ``gauntlet.{axes,samplers,sinks,cli}`` groups are
    asserted alongside the original two — keeping all six name pairs
    in one place is the cheapest regression guard against an
    accidental ``__all__`` drift when a future polish task adds a
    seventh group.
    """
    expected = {
        # Original two.
        "POLICY_ENTRY_POINT_GROUP",
        "ENV_ENTRY_POINT_GROUP",
        "discover_policy_plugins",
        "discover_env_plugins",
        "warn_on_collision",
        # Polish-task expansion.
        "AXIS_ENTRY_POINT_GROUP",
        "SAMPLER_ENTRY_POINT_GROUP",
        "SINK_ENTRY_POINT_GROUP",
        "CLI_ENTRY_POINT_GROUP",
        "discover_axis_plugins",
        "discover_sampler_plugins",
        "discover_sink_plugins",
        "discover_cli_plugins",
    }
    assert expected == set(plugins.__all__)


# ──────────────────────────────────────────────────────────────────────
# Real EntryPoint type compat — sanity check we accept the stdlib type.
# ──────────────────────────────────────────────────────────────────────


def test_real_entrypoint_type_is_accepted() -> None:
    """Smoke-check that real importlib.metadata.EntryPoint instances also work.

    The fake type is a duck — this test ensures we did not accidentally
    rely on a fake-only attribute. Builds a real EntryPoint and feeds
    it through the load path with no dist (the ``<unknown>`` fallback).
    """
    # ``EntryPoint(name, value, group)`` resolves on .load() by importing
    # the dotted path. Point at a real, cheap stdlib callable.
    real_ep = EntryPoint(
        name="real-test",
        value="builtins:list",
        group=POLICY_ENTRY_POINT_GROUP,
    )

    def _fake(*, group: str) -> list[EntryPoint]:
        if group == POLICY_ENTRY_POINT_GROUP:
            return [real_ep]
        return []

    with patch("gauntlet.plugins.entry_points", side_effect=_fake):
        result = discover_policy_plugins()
    assert set(result) == {"real-test"}
    assert result["real-test"] is cast(Any, list)


# ──────────────────────────────────────────────────────────────────────
# End-to-end: gauntlet's own dogfood entry points materialize.
# ──────────────────────────────────────────────────────────────────────


def test_dogfood_policies_discovered_after_install() -> None:
    """The ``[project.entry-points."gauntlet.policies"]`` block in pyproject.toml
    registers gauntlet's own four built-in adapters. After ``uv pip install -e .``
    they must surface through :func:`discover_policy_plugins`.

    This is the eat-your-own-dogfood smoke test promised by the polish
    task: if the plugin mechanism is broken on a fresh checkout, this
    fires before any third-party plugin author hits the same bug.

    Failure-tolerant: ``huggingface`` / ``lerobot`` may fail to load on
    a default-job venv (their extras aren't installed); we accept that
    by checking for a *subset* of the always-available built-ins
    (``random`` + ``scripted``).
    """
    discover_policy_plugins.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plugins = discover_policy_plugins()
    # ``random`` and ``scripted`` are torch-free and always load.
    assert {"random", "scripted"} <= set(plugins), (
        f"Dogfood entry points missing — got {sorted(plugins)}. "
        "Did you re-install after pyproject.toml changes? "
        "Try: uv pip install -e ."
    )


def test_dogfood_envs_discovered_after_install() -> None:
    """The first-party ``tabletop`` env entry point must materialize.

    Heavy backends (`tabletop-pybullet`, `-genesis`, `-isaac`) drop on
    a default-job venv because their extras aren't installed —
    accepting this matches the lazy-load failure model.
    """
    discover_env_plugins.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plugins = discover_env_plugins()
    assert "tabletop" in plugins, (
        f"Dogfood env entry point missing — got {sorted(plugins)}. "
        "Did you re-install after pyproject.toml changes? "
        "Try: uv pip install -e ."
    )
