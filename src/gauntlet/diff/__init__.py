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
from gauntlet.diff.render import render_text as render_text

__all__ = [
    "AxisDelta",
    "CellFlip",
    "ClusterDelta",
    "ReportDiff",
    "diff_reports",
    "render_text",
]
