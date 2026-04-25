"""Fleet-wide aggregation — see ``GAUNTLET_SPEC.md`` §7 and
``docs/phase3-rfc-019-fleet-aggregate.md``.

Public surface (built up across commits — schema, then aggregation,
then HTML rendering):

* :class:`FleetRun` — one row of the fleet roll-up (per discovered
  ``report.json``).
* :class:`FleetReport` — the meta-report aggregated across N runs.
* :func:`aggregate_reports` — pure function ``list[Report] -> FleetReport``.
* :func:`aggregate_directory` — discovery + aggregation in one call.
* :func:`discover_run_files` — recursive ``report.json`` glob.
"""

from __future__ import annotations

from gauntlet.aggregate.analyze import aggregate_directory as aggregate_directory
from gauntlet.aggregate.analyze import aggregate_reports as aggregate_reports
from gauntlet.aggregate.analyze import discover_run_files as discover_run_files
from gauntlet.aggregate.html import render_fleet_html as render_fleet_html
from gauntlet.aggregate.html import write_fleet_html as write_fleet_html
from gauntlet.aggregate.schema import FleetReport as FleetReport
from gauntlet.aggregate.schema import FleetRun as FleetRun
from gauntlet.aggregate.sim_real import AxisTransfer as AxisTransfer
from gauntlet.aggregate.sim_real import SimRealReport as SimRealReport
from gauntlet.aggregate.sim_real import (
    compute_sim_real_correlation as compute_sim_real_correlation,
)

__all__ = [
    "AxisTransfer",
    "FleetReport",
    "FleetRun",
    "SimRealReport",
    "aggregate_directory",
    "aggregate_reports",
    "compute_sim_real_correlation",
    "discover_run_files",
    "render_fleet_html",
    "write_fleet_html",
]
