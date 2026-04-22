"""Policy adapter protocol.

See GAUNTLET_SPEC.md §3 (core abstractions) and §6 (design principles).

A :class:`Policy` is the narrowest possible adapter: a single ``act`` method
that maps an observation to an action. Policies that need per-episode state
reset additionally satisfy :class:`ResettablePolicy`; the runner detects this
via :func:`isinstance` and calls ``reset`` at the start of each rollout.

Both protocols are ``runtime_checkable`` so the runner can duck-type check
without importing concrete classes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Action",
    "Observation",
    "Policy",
    "ResettablePolicy",
]

# Observations bridge the MuJoCo / gymnasium FFI boundary and may carry mixed
# dtypes (uint8 images, float32 proprio, etc.). Per spec §6, ``Any`` inside
# ``NDArray`` is permitted at the FFI boundary only.
Observation: TypeAlias = Mapping[str, NDArray[Any]]

# Actions are continuous control vectors. We standardise on float64 so the
# MuJoCo control buffer (``mjData.ctrl``) accepts them without a cast.
Action: TypeAlias = NDArray[np.float64]


@runtime_checkable
class Policy(Protocol):
    """Minimal policy adapter.

    A policy maps a (possibly multi-modal) observation to an action vector.
    Policies may be stateful internally; stateful policies should also
    satisfy :class:`ResettablePolicy`.
    """

    def act(self, obs: Observation) -> Action:
        """Return an action for the current observation."""
        ...


@runtime_checkable
class ResettablePolicy(Protocol):
    """Policy with per-episode reset.

    The runner calls :meth:`reset` at the start of each episode, passing the
    episode's deterministic RNG so stochastic policies can re-seed and
    scripted policies can rewind their step counter.
    """

    def act(self, obs: Observation) -> Action:
        """Return an action for the current observation."""
        ...

    def reset(self, rng: np.random.Generator) -> None:
        """Reset per-episode state using the supplied RNG."""
        ...
