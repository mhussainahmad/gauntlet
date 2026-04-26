"""Cross-checkpoint regression bisection — B-39.

``gauntlet bisect`` answers the "which checkpoint introduced this
regression?" question that every fleet operator asks when iterating on
a training run. Given an ordered list of checkpoints
``[good, *intermediates, bad]`` and a single failing target cell, the
engine binary-searches the list and returns the first checkpoint at
which the target cell's success rate dropped significantly below the
known-good baseline.

The variance-reduction story re-uses :mod:`gauntlet.diff.paired`
(B-08): every pairwise comparison runs both sides of the suite under
the same master seed, so per-episode env seeds are identical across
``(cell_index, episode_index)`` and the McNemar / Newcombe-Tango
paired-CI shrinks the residual variance ~2-4x relative to two
independent Wilson intervals. The bisect decision rule reads the
target cell's :class:`~gauntlet.diff.paired.PairedCellDelta` and
collapses the search interval whenever the upper bound of that paired
delta lies strictly below zero (midpoint significantly worse than
good).

Anti-feature framing (mirrors the backlog entry verbatim):

* Weight interpolation between checkpoints is **not** supported — the
  general case requires saved intermediate checkpoints, which most
  training runs don't keep. The user supplies the discrete checkpoint
  list and the engine linear-search-binary-searches over it. This
  loses the bisect-y appeal of git-bisect (you only get as much
  resolution as you saved checkpoints) but is honest: there is no
  weight-space interpolation that works across model architectures.
* The bisect signal collapses if the target cell has zero discordant
  pairs in some midpoint comparison. The CI path returns ``None`` for
  the upper bound in that degenerate case and we keep searching.

Public surface:

* :class:`BisectStep` — one ``(checkpoint, target-cell delta)`` row.
* :class:`BisectResult` — top-level artefact (first-bad checkpoint,
  the ordered step list, the final delta).
* :func:`bisect` — pure engine; takes an ordered checkpoint list, a
  resolver callable mapping checkpoint id -> zero-arg policy factory,
  a :class:`~gauntlet.suite.Suite`, and a target cell id; returns
  :class:`BisectResult`.

The CLI subcommand lives in :mod:`gauntlet.bisect.cli` and is
registered against the top-level ``gauntlet`` Typer app from
:mod:`gauntlet.cli`.
"""

from __future__ import annotations

from gauntlet.bisect.bisect import BisectError as BisectError
from gauntlet.bisect.bisect import BisectResult as BisectResult
from gauntlet.bisect.bisect import BisectStep as BisectStep
from gauntlet.bisect.bisect import RunnerFactory as RunnerFactory
from gauntlet.bisect.bisect import bisect as bisect

__all__ = [
    "BisectError",
    "BisectResult",
    "BisectStep",
    "RunnerFactory",
    "bisect",
]
