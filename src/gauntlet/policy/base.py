"""Policy adapter protocol.

See GAUNTLET_SPEC.md §3 (core abstractions) and §6 (design principles).

A :class:`Policy` is the narrowest possible adapter: a single ``act`` method
that maps an observation to an action. Policies that need per-episode state
reset additionally satisfy :class:`ResettablePolicy`; the runner detects this
via :func:`isinstance` and calls ``reset`` at the start of each rollout.

A :class:`SamplablePolicy` is the optional B-18 extension: a policy that can
draw N independently-sampled actions for the *same* observation without
mutating its own internal state. Detected by ``isinstance`` in the worker;
greedy / open-loop policies (Scripted, OpenVLA-greedy) that legitimately
have nothing stochastic to sample do NOT implement this Protocol and are
honestly reported as ``Episode.action_variance=None`` rather than the
misleading ``0.0``. See ``docs/backlog.md`` B-18 — the column being
``None`` for greedy policies is a documented anti-feature, not a bug.

All three protocols are ``runtime_checkable`` so the runner can duck-type
check without importing concrete classes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Action",
    "Observation",
    "Policy",
    "ResettablePolicy",
    "SamplablePolicy",
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


@runtime_checkable
class SamplablePolicy(Protocol):
    """Policy that can sample N independent actions for the same observation.

    The B-18 mode-collapse metric draws ``N`` actions from the policy at a
    handful of trajectory steps and reports per-axis variance averaged
    across those steps as :attr:`gauntlet.runner.Episode.action_variance`.

    Implementers MUST guarantee :meth:`act_n` is *side-effect free* on the
    policy's per-episode state — calling it must not advance an RNG, pop
    a chunk-queue entry, or step a scripted cursor. Otherwise the
    measured rollout's actions would diverge from an un-measured rollout
    with the same seed and break :attr:`gauntlet.runner.Episode.seed`
    reproducibility (the contract that lets B-08 CRN paired-compare
    work). The reference implementations on :class:`RandomPolicy`,
    :class:`gauntlet.policy.lerobot.LeRobotPolicy`, and
    :class:`gauntlet.policy.pi0.Pi0Policy` snapshot internal state
    before sampling and restore it after.

    Greedy / deterministic policies (Scripted, OpenVLA-greedy) do NOT
    implement this Protocol — the runner detects the absence via
    ``isinstance`` and writes ``Episode.action_variance=None``. Reporting
    ``0.0`` for "greedy" would be honest in some sense but conflates
    "actually has zero spread" with "not measured here", and the B-18
    spec's anti-feature note explicitly favours the former: only
    sampleable policies expose this column.
    """

    def act(self, obs: Observation) -> Action:
        """Return one action for the current observation."""
        ...

    def act_n(self, obs: Observation, n: int = 8) -> Sequence[Action]:
        """Return ``n`` independently-sampled actions for the same ``obs``.

        Implementations MUST NOT mutate per-episode state. Returned
        sequence has length ``n``; each element has the same shape and
        dtype as :meth:`act`. ``n >= 1``.
        """
        ...
