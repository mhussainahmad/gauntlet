"""Env-agnostic Protocol that the Runner + Suite loader dispatch through.

This module is import-cheap on purpose: no simulator imports live here, so
``gauntlet.env.base`` stays safe to import from ``gauntlet.core`` (see
``GAUNTLET_SPEC.md`` §6 "small deps"). Backend-specific modules
(``gauntlet.env.tabletop`` for MuJoCo, ``gauntlet.env.pybullet`` for PyBullet)
register themselves via :mod:`gauntlet.env.registry` and satisfy this Protocol
structurally — there is no base class to inherit from.

See ``docs/phase2-rfc-005-pybullet-adapter.md`` §3 for the full rationale.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

import gymnasium as gym
from numpy.typing import NDArray

# Observation / action type aliases shared by every backend. Kept here (not in
# ``gauntlet.env.tabletop``) so second-party backends can type against the
# Protocol without importing MuJoCo.
#
# ``Any`` inside ``NDArray`` mirrors ``gauntlet.policy.base`` — externally
# every backend returns float64 arrays, but the policy Protocol was declared
# liberally to absorb minor dtype drift at the policy/env seam.
Observation = dict[str, NDArray[Any]]
Action = NDArray[Any]


@runtime_checkable
class GauntletEnv(Protocol):
    """Structural env contract the Runner and Suite-loader dispatch through.

    Any object that exposes this surface — including but not limited to
    :class:`gauntlet.env.tabletop.TabletopEnv` — is accepted by
    :mod:`gauntlet.runner.worker`. The Protocol is intentionally a **subset**
    of :class:`gymnasium.Env`: ``render`` is deliberately omitted so state-only
    backends (first-cut PyBullet) do not need to implement it. A separate
    ``RenderableGauntletEnv`` sub-Protocol is planned for the image-rendering
    follow-up RFC.

    Behavioural contract
    --------------------
    ``set_perturbation(name, value)``
        **Queues** a named scalar on the env — it does not take effect
        immediately. Raises :class:`ValueError` for
        ``name not in type(self).AXIS_NAMES``. Per-backend validation rules
        (integer bounds on ``distractor_count``, etc.) apply on top of the
        name check.

    ``restore_baseline()``
        Makes the env **observationally** equivalent to its post-``__init__``
        state — not bit-identical internal C state (that is a MuJoCo-only
        property). Called by the Runner between episodes
        (see ``runner/worker.py::_execute_one``). Does **not** clear the
        queued perturbations — those are the input to the next reset.

    ``reset(seed=...)``
        MUST, in order:

        1. call ``restore_baseline()`` internally,
        2. apply seed-driven state randomisation,
        3. apply every queued perturbation on top,
        4. clear the pending-perturbation queue.

        The Runner already relies on this ordering
        (``TabletopEnv.reset`` implements it today). Reversing steps 2 and 3
        silently changes determinism semantics; backends MUST NOT do that.
        ``seed`` is the only entropy source for a reset.

    ``close()``
        Releases any simulator-held resources. Idempotent.

    ``AXIS_NAMES``
        Class-level :class:`frozenset` of axis names this backend accepts via
        ``set_perturbation``. Exposed as a :class:`~typing.ClassVar` so the
        Suite loader can introspect axis support per backend without
        constructing the env. Every backend that implements the canonical 7
        axes declares the same frozenset.
    """

    AXIS_NAMES: ClassVar[frozenset[str]]

    observation_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]: ...

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]: ...

    def set_perturbation(self, name: str, value: float) -> None: ...

    def restore_baseline(self) -> None: ...

    def close(self) -> None: ...


__all__ = ["Action", "GauntletEnv", "Observation"]
