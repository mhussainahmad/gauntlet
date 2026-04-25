"""Sampler protocol + concrete strategies for :class:`Suite.cells`.

See ``docs/polish-exploration-lhs-sampling.md`` for the design rationale.
This module is the dispatch layer behind :attr:`Suite.sampling`:

* ``"cartesian"`` (default) — :class:`CartesianSampler`, the existing
  :func:`itertools.product` enumeration. Byte-identical to the
  pre-LHS behaviour.
* ``"latin_hypercube"`` — :class:`LatinHypercubeSampler` (added in a
  follow-up step in this same task series).
* ``"sobol"`` — :class:`SobolSampler` (Joe-Kuo 6.21201
  low-discrepancy sequence; see ``docs/polish-exploration-sobol-sampler.md``).

The :class:`Sampler` protocol is intentionally minimal: ``sample(suite,
rng)`` returns a list of :class:`SuiteCell` records. The Runner is
unaware of which sampler produced the list — every downstream consumer
keys off :attr:`SuiteCell.index` exactly as before.

The RNG passed to :meth:`Sampler.sample` is owned by the caller. For
the default :meth:`Suite.cells` entry point, it is seeded from
:attr:`Suite.seed` (or OS entropy when ``seed is None``) so two
:meth:`Suite.cells` calls on the same suite produce the same cells.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from gauntlet.suite.schema import Suite, SuiteCell

__all__ = [
    "CartesianSampler",
    "Sampler",
    "build_sampler",
]


@runtime_checkable
class Sampler(Protocol):
    """Strategy: turn a :class:`Suite` into a flat list of grid cells.

    Implementations must be pure (no I/O, no global state) and
    deterministic for a fixed ``rng`` — re-running with the same
    :class:`numpy.random.Generator` (seeded with the same int) must
    yield the same list.

    Cartesian sampling ignores ``rng`` because the grid is fully
    determined by per-axis enumerations; LHS / Sobol consume it.
    """

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        """Return every cell this strategy emits for ``suite``."""
        ...


class CartesianSampler:
    """The pre-LHS Cartesian-grid enumeration, factored out as a strategy.

    Wraps the existing :func:`itertools.product` over per-axis
    :meth:`AxisSpec.enumerate` calls. Every aspect of the output is
    byte-identical to the historical :meth:`Suite.cells` body — same
    axis order, same rightmost-axis-varies-fastest sequence, same
    zero-based contiguous :attr:`SuiteCell.index` values.

    ``rng`` is unused (cartesian sampling is deterministic from the
    suite alone) but kept on the signature so this class satisfies
    :class:`Sampler`.
    """

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        # Imported lazily to avoid a circular import: schema imports
        # this module to dispatch ``Suite.cells``, and this method is
        # only called from inside that dispatch path.
        from gauntlet.suite.schema import SuiteCell

        del rng  # cartesian sampling does not consume entropy
        axis_names = tuple(suite.axes.keys())
        per_axis_values = tuple(spec.enumerate() for spec in suite.axes.values())
        out: list[SuiteCell] = []
        for index, combo in enumerate(itertools.product(*per_axis_values)):
            mapping: dict[str, float] = dict(zip(axis_names, combo, strict=True))
            out.append(SuiteCell(index=index, values=mapping))
        return out


def build_sampler(mode: str) -> Sampler:
    """Return the :class:`Sampler` for a given :attr:`Suite.sampling` value.

    Raises:
        ValueError: if ``mode`` is not one of the recognised strategies.
            The :class:`Suite` schema already restricts ``sampling`` to
            the literal set, so this branch is defence-in-depth for
            direct callers that bypass schema validation.
    """
    if mode == "cartesian":
        return CartesianSampler()
    if mode == "latin_hypercube":
        # Local import dodges the schema <-> sampling cycle: schema
        # imports this module to dispatch ``Suite.cells``, and the
        # sampler subclasses each defer their ``SuiteCell`` import to
        # call time for the same reason.
        from gauntlet.suite.lhs import LatinHypercubeSampler

        return LatinHypercubeSampler()
    if mode == "sobol":
        # Same lazy-import rationale as LHS above.
        from gauntlet.suite.sobol import SobolSampler

        return SobolSampler()
    raise ValueError(
        f"unknown sampling mode {mode!r}; expected one of "
        "{'cartesian', 'latin_hypercube', 'sobol'}",
    )
