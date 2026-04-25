"""Compare-side polish modules — see backlog B-24, B-29.

Public surface (kept minimal; the core compare logic still lives in
:mod:`gauntlet.cli` ``_build_compare`` for now):

* :func:`to_github_summary` — render a ``compare.json`` payload as a
  GitHub Actions ``$GITHUB_STEP_SUMMARY``-compatible markdown blob.
* :func:`compute_drift_map` — cross-backend embodiment-transfer drift
  scoring (B-29). See :mod:`gauntlet.compare.drift_map` for the
  per-axis-value drift map schema.
"""

from __future__ import annotations

from gauntlet.compare.drift_map import (
    AxisDrift as AxisDrift,
)
from gauntlet.compare.drift_map import (
    DriftMap as DriftMap,
)
from gauntlet.compare.drift_map import (
    DriftMapError as DriftMapError,
)
from gauntlet.compare.drift_map import (
    compute_drift_map as compute_drift_map,
)
from gauntlet.compare.drift_map import (
    render_drift_map_table as render_drift_map_table,
)
from gauntlet.compare.drift_map import (
    top_axis_drifts as top_axis_drifts,
)
from gauntlet.compare.github_summary import to_github_summary as to_github_summary

__all__ = [
    "AxisDrift",
    "DriftMap",
    "DriftMapError",
    "compute_drift_map",
    "render_drift_map_table",
    "to_github_summary",
    "top_axis_drifts",
]
