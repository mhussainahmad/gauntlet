"""Policy spec → ``policy_factory`` resolver.

The CLI accepts a ``--policy`` string in one of three shapes:

* ``"random"`` — built-in shortcut. Resolves to a 7-DoF
  :class:`gauntlet.policy.random.RandomPolicy`.
* ``"scripted"`` — built-in shortcut. Resolves to the default
  :class:`gauntlet.policy.scripted.ScriptedPolicy`.
* ``"module.path:attr"`` — dotted import of a zero-arg callable that
  returns a :class:`gauntlet.policy.base.Policy`. Used to plug user
  policies into the harness.

The resolved factory is intentionally a top-level callable
(:class:`functools.partial` over a class, or the class itself, or the
imported attribute). Lambdas / closures would not pickle under the
``spawn`` start method that :class:`gauntlet.runner.Runner` requires for
``n_workers >= 2``; using ``partial`` keeps the parallel path alive even
though the unit tests only exercise ``-w 1``.

Plugin support
--------------
A fourth shape — a bare word that is neither ``"random"`` nor
``"scripted"`` — is consulted against the third-party plugin registry
(``gauntlet.policies`` ``[project.entry-points]`` group). Built-ins
always win on collision (see :mod:`gauntlet.plugins`). Plugin-resolved
classes are treated as zero-arg factories, matching the existing
``module.path:attr`` contract. See
``docs/polish-exploration-plugin-system.md`` for the full design.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import partial
from typing import cast

from gauntlet.policy.base import Policy
from gauntlet.policy.random import RandomPolicy
from gauntlet.policy.scripted import ScriptedPolicy

__all__ = [
    "POLICY_REGISTRY",
    "PolicySpecError",
    "resolve_policy",
    "resolve_policy_factory",
]


# Action dimension for the Phase 1 tabletop env. Matches
# ``TabletopEnv``'s 7-DoF action layout (see env/tabletop.py).
_DEFAULT_ACTION_DIM = 7


# Built-in policy table. The plugin discovery layer (`gauntlet.plugins`)
# treats this as the source of truth for first-party names — plugins
# never overwrite an entry here. Names match the ``--policy`` CLI
# shortcuts and the dogfooded ``[project.entry-points."gauntlet.policies"]``
# section of ``pyproject.toml``.
#
# ``RandomPolicy`` / ``ScriptedPolicy`` need no constructor args (or
# accept all-defaults), so the class itself serves as a zero-arg
# factory in the plugin path. The legacy ``resolve_policy_factory``
# keeps wrapping ``RandomPolicy`` in ``partial(action_dim=7)`` because
# the CLI's ``--policy random`` contract is "explicit 7-DoF default".
POLICY_REGISTRY: dict[str, type[Policy]] = {
    "random": cast(type[Policy], RandomPolicy),
    "scripted": cast(type[Policy], ScriptedPolicy),
}


class PolicySpecError(ValueError):
    """Raised when a ``--policy`` string cannot be resolved.

    Subclasses :class:`ValueError` so existing ``except ValueError``
    handlers in callers (e.g. tests) keep working, but the dedicated
    type lets the CLI distinguish a malformed spec from other errors
    bubbling out of policy construction.
    """


def _resolve_module_attr(spec: str) -> Callable[[], Policy]:
    """Resolve ``"module.path:attr"`` to the imported callable.

    The resulting object must be a zero-arg callable returning a Policy.
    We do not validate the return value — the Runner will surface any
    type mismatch on first call — but we *do* validate the spec format
    and the import so the user gets a clean error at CLI time, not deep
    inside the worker pool.
    """
    if spec.count(":") != 1:
        raise PolicySpecError(
            f"policy spec {spec!r}: module form must be 'module.path:attr' "
            "(exactly one ':' separating module and attribute)"
        )
    module_path, attr_name = spec.split(":", 1)
    if not module_path or not attr_name:
        raise PolicySpecError(
            f"policy spec {spec!r}: both module path and attribute name must be non-empty"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise PolicySpecError(
            f"policy spec {spec!r}: could not import module {module_path!r}: {exc}"
        ) from exc
    try:
        attr = getattr(module, attr_name)
    except AttributeError as exc:
        raise PolicySpecError(
            f"policy spec {spec!r}: module {module_path!r} has no attribute {attr_name!r}"
        ) from exc
    if not callable(attr):
        raise PolicySpecError(
            f"policy spec {spec!r}: {module_path}.{attr_name} is not callable "
            f"(got {type(attr).__name__})"
        )
    # mypy can't see through getattr; the runtime check above guards us.
    factory: Callable[[], Policy] = attr
    return factory


def resolve_policy(name: str) -> type[Policy]:
    """Look up a policy *class* by name from built-ins or plugins.

    Distinct from :func:`resolve_policy_factory` — that returns a
    zero-arg factory (with the ``action_dim=7`` partial baked in for
    ``"random"``); this returns the class itself, suitable for
    library-level callers that want to introspect or subclass the
    adapter.

    Built-ins win on collision: if a plugin registers a name already
    present in :data:`POLICY_REGISTRY`, the built-in class is returned
    and a :class:`RuntimeWarning` is emitted (unless the plugin entry
    loads the same class object — the dogfood case).

    Raises:
        PolicySpecError: If ``name`` matches neither a built-in nor a
            successfully-loaded plugin.
    """
    if not name or not name.strip():
        raise PolicySpecError("policy name must be a non-empty string")
    name = name.strip()
    builtin = POLICY_REGISTRY.get(name)
    # Lazy import — avoids a cycle (``gauntlet.plugins`` imports
    # ``gauntlet.policy.base``) and keeps registry imports cheap on
    # cold paths that never touch plugins.
    from gauntlet.plugins import (
        POLICY_ENTRY_POINT_GROUP,
        discover_policy_plugins,
        warn_on_collision,
    )

    plugins = discover_policy_plugins()
    if builtin is not None:
        plugin_cls = plugins.get(name)
        if plugin_cls is not None:
            warn_on_collision(
                name=name,
                group=POLICY_ENTRY_POINT_GROUP,
                builtin_obj=builtin,
                plugin_obj=plugin_cls,
                plugin_dist="<plugin>",
            )
        return builtin
    plugin_cls = plugins.get(name)
    if plugin_cls is not None:
        return plugin_cls
    available = sorted({*POLICY_REGISTRY, *plugins})
    raise PolicySpecError(
        f"unknown policy {name!r}; available: {available}",
    )


def resolve_policy_factory(spec: str) -> Callable[[], Policy]:
    """Turn a ``--policy`` CLI string into a zero-arg policy factory.

    Args:
        spec: One of ``"random"``, ``"scripted"``, ``"module.path:attr"``,
            or a bare-word plugin name registered under the
            ``gauntlet.policies`` entry-point group.

    Returns:
        Zero-arg callable that returns a fresh :class:`Policy` on each
        call. Picklable under :mod:`multiprocessing` ``spawn`` so the
        same factory works for ``n_workers == 1`` and ``n_workers >= 2``.

    Raises:
        PolicySpecError: If ``spec`` is empty, has the wrong shape, or
            references an unimportable module / missing attribute /
            unknown plugin.
    """
    if not spec or not spec.strip():
        raise PolicySpecError("policy spec must be a non-empty string")
    spec = spec.strip()
    if spec == "random":
        # ``partial`` over a class pickles cleanly; a lambda does not.
        return partial(RandomPolicy, action_dim=_DEFAULT_ACTION_DIM)
    if spec == "scripted":
        # The class is itself a zero-arg callable (all kwargs default).
        return ScriptedPolicy
    if ":" in spec:
        return _resolve_module_attr(spec)
    # Plugin fallthrough — lazy import to avoid touching importlib.metadata
    # on the hot CLI path when only a built-in shortcut is used.
    from gauntlet.plugins import discover_policy_plugins

    plugins = discover_policy_plugins()
    if spec in plugins:
        # Plugin classes are expected to be zero-arg constructible
        # (same contract as ``module:attr`` factories today). If the
        # class needs arguments the user must register a callable
        # instead — see ``docs/plugin-development.md``.
        return cast(Callable[[], Policy], plugins[spec])
    raise PolicySpecError(
        f"unknown policy spec {spec!r}: expected 'random', 'scripted', "
        f"'module.path:attr', or a registered plugin name "
        f"(installed plugins: {sorted(plugins)})"
    )
