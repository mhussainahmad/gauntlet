"""Entry-point discovery for third-party policies and envs.

See ``docs/polish-exploration-plugin-system.md`` for the full design;
quick recap:

* ``gauntlet.policies`` and ``gauntlet.envs`` are the two
  ``[project.entry-points]`` groups we read.
* Discovery is cached via :func:`functools.lru_cache` — the entry-point
  table is fixed for the lifetime of the process under normal use.
  Tests that mock :func:`importlib.metadata.entry_points` must call
  :meth:`discover_policy_plugins.cache_clear` (and the env equivalent)
  in fixture teardown.
* ``ep.load()`` failures are wrapped in a clear :class:`RuntimeError`
  and the offending entry point is dropped from the returned dict — the
  rest of gauntlet keeps working.

This module deliberately does NOT consult the built-in registry tables.
Built-in-vs-plugin precedence (built-ins always win on collision) is
enforced one layer up — see :mod:`gauntlet.policy.registry` and
:mod:`gauntlet.env.registry` resolvers.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from importlib.metadata import EntryPoint, entry_points
from typing import Any, cast

from gauntlet.env.base import GauntletEnv
from gauntlet.policy.base import Policy

__all__ = [
    "ENV_ENTRY_POINT_GROUP",
    "POLICY_ENTRY_POINT_GROUP",
    "discover_env_plugins",
    "discover_policy_plugins",
    "warn_on_collision",
]

POLICY_ENTRY_POINT_GROUP = "gauntlet.policies"
ENV_ENTRY_POINT_GROUP = "gauntlet.envs"


def _entry_point_dist_name(ep: EntryPoint) -> str:
    """Return the distribution name owning ``ep``, or ``'<unknown>'``.

    ``EntryPoint.dist`` is populated when the entry point comes from
    an installed distribution — the normal case. Hand-built EntryPoint
    instances (e.g. in tests) may leave it ``None``; we fall back to
    a placeholder so error messages never crash on an attribute lookup.
    """
    dist = getattr(ep, "dist", None)
    if dist is None:
        return "<unknown>"
    name = getattr(dist, "name", None)
    if isinstance(name, str) and name:
        return name
    return "<unknown>"


def _load_entry_points(group: str) -> dict[str, Any]:
    """Discover and load every entry point in ``group``.

    Returns a ``name -> loaded-object`` dict. Entry points that raise
    on :meth:`EntryPoint.load` are wrapped in a clear
    :class:`RuntimeWarning` (so the failure is visible) and dropped from
    the returned dict. Gauntlet itself stays operational either way.

    Duplicate names within ``group`` (rare — would require two different
    distributions both registering the same name) keep the first-seen
    entry; subsequent duplicates emit a :class:`RuntimeWarning` naming
    both distributions so the user can resolve the conflict by
    uninstalling one.
    """
    discovered: dict[str, Any] = {}
    seen_dists: dict[str, str] = {}
    for ep in entry_points(group=group):
        if ep.name in discovered:
            other = seen_dists.get(ep.name, "<unknown>")
            warnings.warn(
                f"gauntlet plugin name {ep.name!r} is registered by both "
                f"{other!r} and {_entry_point_dist_name(ep)!r}; using the "
                f"first ({other!r}). Uninstall one to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        try:
            obj = ep.load()
        except Exception as exc:  # wrap *any* plugin failure
            warnings.warn(
                f"Plugin {ep.name!r} from {_entry_point_dist_name(ep)!r} "
                f"failed to load: {exc!r}. Dropping from registry.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        discovered[ep.name] = obj
        seen_dists[ep.name] = _entry_point_dist_name(ep)
    return discovered


@lru_cache(maxsize=1)
def discover_policy_plugins() -> dict[str, type[Policy]]:
    """Return ``name -> Policy class`` for every installed policy plugin.

    Reads the ``gauntlet.policies`` entry-point group. Failed entry
    points are skipped with a :class:`RuntimeWarning`; the returned
    dict only contains successfully-loaded plugins.

    The result is cached for the lifetime of the process. Tests that
    inject fake entry points via :func:`unittest.mock.patch` on
    :func:`importlib.metadata.entry_points` must call
    :meth:`discover_policy_plugins.cache_clear` in fixture teardown
    to avoid leaking the mocked result into other tests.
    """
    raw = _load_entry_points(POLICY_ENTRY_POINT_GROUP)
    # We do not runtime-check that each loaded object satisfies the
    # ``Policy`` Protocol — that would require constructing an instance,
    # which is exactly the kind of side effect entry-point discovery is
    # meant to defer. The structural check happens at first
    # ``policy.act(obs)`` inside the Runner. The cast widens the
    # ``Any`` table to the documented public type.
    return cast(dict[str, type[Policy]], raw)


@lru_cache(maxsize=1)
def discover_env_plugins() -> dict[str, type[GauntletEnv]]:
    """Return ``name -> GauntletEnv class`` for every installed env plugin.

    Reads the ``gauntlet.envs`` entry-point group. Failed entry points
    are skipped with a :class:`RuntimeWarning`; the returned dict only
    contains successfully-loaded plugins.

    See :func:`discover_policy_plugins` for cache-invalidation guidance.
    """
    raw = _load_entry_points(ENV_ENTRY_POINT_GROUP)
    # See discover_policy_plugins() for the cast rationale.
    return cast(dict[str, type[GauntletEnv]], raw)


def warn_on_collision(
    *,
    name: str,
    group: str,
    builtin_obj: object,
    plugin_obj: object,
    plugin_dist: str,
) -> None:
    """Emit a :class:`RuntimeWarning` if a plugin shadows a built-in.

    Called by the policy / env registry resolvers when a plugin entry
    point and a built-in name coincide. The check is **identity-based**
    (``is``) — gauntlet's own dogfooded entry points load the same class
    object the built-in table already holds, so they never warn. A real
    third-party shadow loads a different class and triggers the warning.
    """
    if plugin_obj is builtin_obj:
        return
    warnings.warn(
        f"plugin {name!r} from {plugin_dist!r} (group {group!r}) shadows the "
        f"built-in {name!r}; using the built-in. Rename the plugin's entry "
        f"point or uninstall it to silence this warning.",
        RuntimeWarning,
        stacklevel=3,
    )
