"""Env registry — name → factory mapping, populated at import time.

See ``docs/phase2-rfc-005-pybullet-adapter.md`` §3.4 and §3.5 for the
historical rationale (first-party backends register from their own
subpackage ``__init__.py``). Phase 3 (plugin-system polish task) layered
``[project.entry-points]`` discovery on top — see
``docs/polish-exploration-plugin-system.md`` and :mod:`gauntlet.plugins`.

Resolution order for :func:`resolve_env_factory` (the new public
resolver that callers should prefer over the bare
:func:`get_env_factory`):

1. The first-party imperative registry populated by
   ``register_env`` from each backend subpackage's ``__init__.py``.
   These wins on collision — built-ins are never silently shadowed.
2. The third-party ``gauntlet.envs`` entry-point group.

The stored values are ``Callable[..., GauntletEnv]`` — factories, not
instances — so a Runner worker can construct a fresh env per subprocess
without the registry paying for construction at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from gauntlet.env.base import GauntletEnv

_REGISTRY: dict[str, Callable[..., GauntletEnv]] = {}


def register_env(name: str, factory: Callable[..., GauntletEnv]) -> None:
    """Register ``factory`` under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is already registered. Re-registration under the same
        name is a programming error (two backends claiming the same
        ``env:`` key in a Suite YAML) and is loud by design. If you need
        to swap factories during a test, use a unique name and reach into
        the registry from test code.
    """
    if name in _REGISTRY:
        raise ValueError(f"env {name!r} already registered")
    _REGISTRY[name] = factory


def get_env_factory(name: str) -> Callable[..., GauntletEnv]:
    """Return the factory previously registered under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not registered. The error message includes the
        sorted list of currently-registered names to help the user spot
        typos and missing-extra cases (the Suite loader intercepts the
        missing-extra case separately and rewrites the message — see
        ``gauntlet.suite.loader`` in step 5).
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown env {name!r}; registered: {sorted(_REGISTRY)}") from exc


def registered_envs() -> frozenset[str]:
    """Return an immutable snapshot of currently-registered env names."""
    return frozenset(_REGISTRY)


def resolve_env_factory(name: str) -> Callable[..., GauntletEnv]:
    """Look up an env factory by name from built-ins or plugins.

    Built-ins (the imperative ``register_env`` table) win on collision.
    If ``name`` is unknown to the built-in table, the resolver consults
    :func:`gauntlet.plugins.discover_env_plugins` (the
    ``gauntlet.envs`` entry-point group) and returns the loaded class
    cast to the factory type. The runner / suite loader will surface
    any structural mismatch on first :meth:`reset`.

    A plugin shadowing a built-in name emits a :class:`RuntimeWarning`
    naming the plugin's distribution and uses the built-in. Identity
    collisions (gauntlet's own dogfooded entry points loading the same
    class object the built-in table holds) are silent — checked via
    ``is``, not name equality.

    Raises
    ------
    ValueError
        If ``name`` matches neither a built-in nor a successfully-loaded
        plugin. The message includes the union of available names.
    """
    builtin = _REGISTRY.get(name)
    # Lazy import — avoids any chance of an import cycle between
    # ``gauntlet.env`` (which calls ``register_env`` at import) and
    # ``gauntlet.plugins``.
    from gauntlet.plugins import (
        ENV_ENTRY_POINT_GROUP,
        discover_env_plugins,
        warn_on_collision,
    )

    plugins = discover_env_plugins()
    if builtin is not None:
        plugin_obj = plugins.get(name)
        if plugin_obj is not None:
            warn_on_collision(
                name=name,
                group=ENV_ENTRY_POINT_GROUP,
                builtin_obj=builtin,
                plugin_obj=plugin_obj,
                plugin_dist="<plugin>",
            )
        return builtin
    plugin_obj = plugins.get(name)
    if plugin_obj is not None:
        # Plugin classes satisfy GauntletEnv structurally; the cast
        # documents the deliberate static widening at the registration
        # seam. Same pattern as the imperative ``register_env`` calls
        # in ``gauntlet.env.__init__`` and the backend subpackages.
        return cast(Callable[..., GauntletEnv], plugin_obj)
    available = sorted({*_REGISTRY, *plugins})
    raise ValueError(
        f"unknown env {name!r}; available (built-ins + plugins): {available}",
    )
