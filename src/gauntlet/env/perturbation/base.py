"""Perturbation axis base + sampler protocol.

See ``GAUNTLET_SPEC.md`` §3 (a perturbation axis is "a named, seedable
parameter with a sampling distribution") and §6 (reproducibility is a
hard rule — every sampler takes an explicit ``np.random.Generator``).

Design (Option B from the Phase 1 task 4 brief):
:class:`PerturbationAxis` is a frozen data object — a name plus default
sampling bounds — and the *environment* owns the apply logic via
:meth:`gauntlet.env.tabletop.TabletopEnv.set_perturbation`. Axes are
produced by :class:`AxisSampler` instances, which take a
:class:`numpy.random.Generator` and return a scalar.

Why Option B:

* The env already has the FFI knowledge (which model field to mutate,
  which baseline to delta from). Pushing that knowledge out into per-axis
  classes would couple every axis to ``mujoco`` internals.
* Axes become trivial to construct from YAML in Task 5 — just a name +
  bounds — without any subclass dispatch.
* Tests can construct axes without needing a live env around.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "AXIS_KIND_CATEGORICAL",
    "AXIS_KIND_CONTINUOUS",
    "AXIS_KIND_INT",
    "AxisKind",
    "AxisSampler",
    "PerturbationAxis",
    "make_categorical_sampler",
    "make_continuous_sampler",
    "make_int_sampler",
]


AxisKind = str  # "continuous" | "int" | "categorical"

AXIS_KIND_CONTINUOUS: Final[AxisKind] = "continuous"
AXIS_KIND_INT: Final[AxisKind] = "int"
AXIS_KIND_CATEGORICAL: Final[AxisKind] = "categorical"


@runtime_checkable
class AxisSampler(Protocol):
    """Stateless sampler: ``(rng) -> value``.

    Stateless because the only state is the RNG itself. Reproducibility
    follows directly: the same ``np.random.Generator`` (seeded with the
    same int) yields the same sequence of samples.
    """

    def __call__(self, rng: np.random.Generator) -> float:
        """Draw one scalar sample from this axis's distribution."""
        ...


def make_continuous_sampler(low: float, high: float) -> AxisSampler:
    """Return a uniform float sampler on ``[low, high]``.

    Raises:
        ValueError: if ``low > high``.
    """
    if low > high:
        raise ValueError(f"low must be <= high; got low={low}, high={high}")
    lo = float(low)
    hi = float(high)

    def _sample(rng: np.random.Generator) -> float:
        return float(rng.uniform(lo, hi))

    return _sample


def make_int_sampler(low: int, high: int) -> AxisSampler:
    """Return a uniform integer sampler on ``[low, high]`` (inclusive).

    Returns a float so the protocol stays uniform across axis kinds; the
    env-side handler rounds to int.
    """
    if low > high:
        raise ValueError(f"low must be <= high; got low={low}, high={high}")
    lo = int(low)
    hi = int(high)

    def _sample(rng: np.random.Generator) -> float:
        # rng.integers high is exclusive; +1 makes it inclusive.
        return float(int(rng.integers(lo, hi + 1)))

    return _sample


def make_categorical_sampler(values: tuple[float, ...]) -> AxisSampler:
    """Return a uniform categorical sampler over the supplied finite values.

    Raises:
        ValueError: if ``values`` is empty.
    """
    if len(values) == 0:
        raise ValueError("categorical sampler needs at least one value")
    choices = tuple(float(v) for v in values)

    def _sample(rng: np.random.Generator) -> float:
        idx = int(rng.integers(0, len(choices)))
        return choices[idx]

    return _sample


@dataclass(frozen=True)
class PerturbationAxis:
    """Named, seedable scalar perturbation axis.

    Attributes:
        name: One of :data:`gauntlet.env.perturbation.AXIS_NAMES`. Validated
            against that registry by the package ``__init__``; constructing
            an axis directly does not re-validate (use the registered
            constructors in :mod:`gauntlet.env.perturbation.axes`).
        kind: One of ``"continuous"`` / ``"int"`` / ``"categorical"``.
            Documents how the env-side handler interprets the value.
        sampler: Callable ``(rng) -> float`` producing one sample.
        low: Lower bound of the sampling range. Informational for
            categorical axes (set to the min legal value).
        high: Upper bound. Informational for categorical axes.
    """

    name: str
    kind: AxisKind
    sampler: AxisSampler
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one value from this axis's distribution.

        Convenience wrapper around :attr:`sampler`. Reproducibility:
        ``axis.sample(np.random.default_rng(s))`` is bit-identical across
        runs for fixed ``s``.
        """
        return self.sampler(rng)


# Internal alias to keep type signatures readable in axes.py.
SamplerFactory = Callable[..., AxisSampler]
