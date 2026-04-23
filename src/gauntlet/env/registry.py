"""Env registry — name → factory mapping, populated at import time.

See ``docs/phase2-rfc-005-pybullet-adapter.md`` §3.4 and §3.5 for the full
rationale. Summary: first-party backends register themselves from their
own subpackage ``__init__.py`` (``gauntlet.env`` for the MuJoCo built-in,
``gauntlet.env.pybullet`` for the PyBullet backend). The registry stays
plain-Python — no ``entry_points`` discovery — because all near-term
backends ship in-tree behind optional extras.

The stored values are ``Callable[..., GauntletEnv]`` — factories, not
instances — so a Runner worker can construct a fresh env per subprocess
without the registry paying for construction at import time.
"""

from __future__ import annotations

from collections.abc import Callable

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
