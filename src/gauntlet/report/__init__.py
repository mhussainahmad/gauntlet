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

from gauntlet.report.abstention import compute_abstention_metrics as compute_abstention_metrics
from gauntlet.report.analyze import build_report as build_report
from gauntlet.report.html import render_html as render_html
from gauntlet.report.html import write_html as write_html
from gauntlet.report.schema import AbstentionMetrics as AbstentionMetrics
from gauntlet.report.schema import AxisBreakdown as AxisBreakdown
from gauntlet.report.schema import CellBreakdown as CellBreakdown
from gauntlet.report.schema import FailureCluster as FailureCluster
from gauntlet.report.schema import Heatmap2D as Heatmap2D
from gauntlet.report.schema import Report as Report
from gauntlet.report.schema import SensitivityIndex as SensitivityIndex
from gauntlet.report.trajectory_taxonomy import TaxonomyError as TaxonomyError
from gauntlet.report.trajectory_taxonomy import TaxonomyResult as TaxonomyResult
from gauntlet.report.trajectory_taxonomy import TrajectoryCluster as TrajectoryCluster
from gauntlet.report.trajectory_taxonomy import (
    cluster_failed_trajectories as cluster_failed_trajectories,
)

__all__ = [
    "AbstentionMetrics",
    "AxisBreakdown",
    "CellBreakdown",
    "FailureCluster",
    "Heatmap2D",
    "Report",
    "SensitivityIndex",
    "TaxonomyError",
    "TaxonomyResult",
    "TrajectoryCluster",
    "build_report",
    "cluster_failed_trajectories",
    "compute_abstention_metrics",
    "render_html",
    "write_html",
]
