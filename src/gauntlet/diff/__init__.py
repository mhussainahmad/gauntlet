"""Structured per-axis diff between two :class:`gauntlet.report.Report` runs.

``gauntlet compare`` returns a binary regression verdict — useful for CI
gates but uninformative when you're iterating on a checkpoint and want to
know *what* moved. This package powers ``gauntlet diff``: a
``git diff``-style structured delta surfacing axis-value rate changes,
per-cell success-flip events, and the failure-cluster set difference.

Public surface:

* :class:`ReportDiff` — top-level Pydantic model.
* :class:`AxisDelta` — per-axis-value rate deltas.
* :class:`CellFlip` — a per-cell success-rate change exceeding a threshold.
* :class:`ClusterDelta` — a shared failure cluster whose lift rose by at
  least ``cluster_intensify_threshold``.
* :func:`diff_reports` — pure function ``(Report, Report) -> ReportDiff``.
* :func:`render_text` — ``git diff``-style plain-text rendering of a
  :class:`ReportDiff` (CLI applies rich color on top).
"""

from __future__ import annotations

from gauntlet.diff.diff import AxisDelta as AxisDelta
from gauntlet.diff.diff import CellFlip as CellFlip
from gauntlet.diff.diff import ClusterDelta as ClusterDelta
from gauntlet.diff.diff import ReportDiff as ReportDiff
from gauntlet.diff.diff import diff_reports as diff_reports
from gauntlet.diff.paired import McNemarResult as McNemarResult
from gauntlet.diff.paired import PairedCellDelta as PairedCellDelta
from gauntlet.diff.paired import PairedComparison as PairedComparison
from gauntlet.diff.paired import PairingError as PairingError
from gauntlet.diff.paired import compute_paired_cells as compute_paired_cells
from gauntlet.diff.paired import mcnemar_test as mcnemar_test
from gauntlet.diff.paired import pair_episodes as pair_episodes
from gauntlet.diff.paired import paired_delta_ci as paired_delta_ci
from gauntlet.diff.render import render_text as render_text

__all__ = [
    "AxisDelta",
    "CellFlip",
    "ClusterDelta",
    "McNemarResult",
    "PairedCellDelta",
    "PairedComparison",
    "PairingError",
    "ReportDiff",
    "compute_paired_cells",
    "diff_reports",
    "mcnemar_test",
    "pair_episodes",
    "paired_delta_ci",
    "render_text",
]
