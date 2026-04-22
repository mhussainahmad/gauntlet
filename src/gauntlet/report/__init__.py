"""Failure-analysis report — see ``GAUNTLET_SPEC.md`` §5 task 7.

Public surface:

* :class:`Report` — top-level Pydantic model produced by
  :func:`build_report`. Carries the breakdowns first, the scalar mean
  second (per spec §6: "never aggregate away failures").
* :class:`AxisBreakdown` — per-axis marginal success rates.
* :class:`CellBreakdown` — per-cell aggregates.
* :class:`FailureCluster` — axis-pair value combinations whose failure
  rate is at least ``cluster_multiple`` times the baseline failure rate.
* :class:`Heatmap2D` — 2D success-rate matrix for a pair of axes.
* :func:`build_report` — pure function ``list[Episode] -> Report``.

Phase 1 task 8 (the HTML generator) consumes :class:`Report` directly;
nothing in this package performs I/O.
"""

from __future__ import annotations

from gauntlet.report.analyze import build_report as build_report
from gauntlet.report.html import render_html as render_html
from gauntlet.report.html import write_html as write_html
from gauntlet.report.schema import AxisBreakdown as AxisBreakdown
from gauntlet.report.schema import CellBreakdown as CellBreakdown
from gauntlet.report.schema import FailureCluster as FailureCluster
from gauntlet.report.schema import Heatmap2D as Heatmap2D
from gauntlet.report.schema import Report as Report

__all__ = [
    "AxisBreakdown",
    "CellBreakdown",
    "FailureCluster",
    "Heatmap2D",
    "Report",
    "build_report",
    "render_html",
    "write_html",
]
