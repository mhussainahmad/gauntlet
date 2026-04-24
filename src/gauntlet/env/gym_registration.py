"""Register gauntlet env adapters with the gymnasium global registry.

See ``docs/polish-exploration-gymnasium-registration.md`` for the full
rationale. Summary:

* The four shipped adapters (MuJoCo / PyBullet / Genesis / Isaac Sim)
  each implement :class:`gymnasium.Env` but, prior to this module, were
  only constructible by direct class import. Wiring them through
  :func:`gymnasium.register` lets a downstream user say
  ``gym.make("gauntlet/Tabletop-v0")`` — the standard ecosystem affordance
  every wrapper / RL framework / training script assumes.
* Heavy backends are registered with **string ``entry_point``s** so the
  ``[pybullet]`` / ``[genesis]`` / ``[isaac]`` subpackages are NOT imported
  at registration time. Gymnasium resolves the string lazily inside
  ``gym.make(...)``; a user without the optional extra can
  ``import gauntlet`` and use the MuJoCo backend with zero ImportError.
* Registration is **idempotent**. ``register_envs()`` can be called any
  number of times — repeated calls are a no-op. The per-id check
  (``env_id in gymnasium.envs.registry``) is the correctness mechanism;
  the module-level ``_REGISTERED`` flag is a fast-path short-circuit.

The internal :mod:`gauntlet.env.registry` (name->factory dict used by
the Suite loader and the CLI) is **untouched** — this module is purely
additive. Direct-import patterns
(``from gauntlet.env.tabletop import TabletopEnv``) keep working
unchanged.
"""

from __future__ import annotations

import gymnasium as gym

__all__ = ["register_envs"]


# Default per-episode step cap, kept in sync with each adapter's
# ``__init__(..., max_steps: int = 200, ...)`` default. Surfacing it
# through gymnasium's ``max_episode_steps`` lets ``gym.make(...)`` install
# the standard :class:`gymnasium.wrappers.TimeLimit` automatically — users
# can override per-call via ``gym.make("...", max_episode_steps=N)``.
_DEFAULT_MAX_EPISODE_STEPS: int = 200

# Module-level fast-path. After the first successful ``register_envs()``
# call the four dict-membership checks below are skipped. Independent of
# the per-id idempotency check inside ``_register_one`` — that one is the
# correctness guarantee; this is the optimisation.
_REGISTERED: bool = False


def _register_one(
    env_id: str,
    entry_point: str,
    *,
    max_episode_steps: int | None = None,
) -> None:
    """Register a single env id, no-op if already present.

    The per-id ``env_id in gym.envs.registry`` check makes this safe to
    call from any caller pattern: the module-level fast-path can be
    bypassed (e.g. by a test that toggles ``_REGISTERED``) and this still
    refuses to double-register. ``gym.register`` itself emits a warning
    on duplicate registration in gymnasium 1.0+; the early return keeps
    that warning silent on every ``import gauntlet`` after the first.
    """
    if env_id in gym.envs.registry:
        return
    gym.register(
        id=env_id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
    )


def register_envs() -> None:
    """Register all gauntlet env adapters with the gymnasium registry.

    Idempotent — repeated calls are no-ops. Safe to call from a
    multiprocessing worker, from inside a pytest fixture, or implicitly
    via ``import gauntlet``.

    The four registered ids (see the module docstring for the full table):

    * ``gauntlet/Tabletop-v0`` — MuJoCo (core dep, always available).
    * ``gauntlet/TabletopPyBullet-v0`` — requires the ``[pybullet]`` extra.
    * ``gauntlet/TabletopGenesis-v0`` — requires the ``[genesis]`` extra.
    * ``gauntlet/TabletopIsaac-v0`` — requires the ``[isaac]`` extra and
      a CUDA-capable GPU at construction time.

    Heavy backends are registered with string ``entry_point``s — gymnasium
    resolves the import lazily inside ``gym.make(...)``, so registration
    itself never triggers the optional-extra ImportError. A user without
    the extra can ``import gauntlet`` and use the MuJoCo backend; only
    ``gym.make("gauntlet/Tabletop<HeavyBackend>-v0")`` triggers the
    heavy import (and the corresponding install-hint error if the extra
    is absent).
    """
    global _REGISTERED
    if _REGISTERED:
        return

    _register_one(
        "gauntlet/Tabletop-v0",
        "gauntlet.env.tabletop:TabletopEnv",
        max_episode_steps=_DEFAULT_MAX_EPISODE_STEPS,
    )

    _REGISTERED = True
