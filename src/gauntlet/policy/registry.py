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
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import partial

from gauntlet.policy.base import Policy
from gauntlet.policy.random import RandomPolicy
from gauntlet.policy.scripted import ScriptedPolicy

__all__ = ["PolicySpecError", "resolve_policy_factory"]


# Action dimension for the Phase 1 tabletop env. Matches
# ``TabletopEnv``'s 7-DoF action layout (see env/tabletop.py).
_DEFAULT_ACTION_DIM = 7


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


def resolve_policy_factory(spec: str) -> Callable[[], Policy]:
    """Turn a ``--policy`` CLI string into a zero-arg policy factory.

    Args:
        spec: One of ``"random"``, ``"scripted"``, or ``"module.path:attr"``.

    Returns:
        Zero-arg callable that returns a fresh :class:`Policy` on each
        call. Picklable under :mod:`multiprocessing` ``spawn`` so the
        same factory works for ``n_workers == 1`` and ``n_workers >= 2``.

    Raises:
        PolicySpecError: If ``spec`` is empty, has the wrong shape, or
            references an unimportable module / missing attribute.
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
    raise PolicySpecError(
        f"unknown policy spec {spec!r}: expected 'random', 'scripted', or 'module.path:attr'"
    )
