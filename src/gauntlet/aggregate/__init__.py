"""Fleet-wide aggregation — see ``GAUNTLET_SPEC.md`` §7 and
``docs/phase3-rfc-019-fleet-aggregate.md``.

Public surface (built up across commits — schema first, then
:func:`aggregate_reports`, then :func:`render_fleet_html`):

* :class:`FleetRun` — one row of the fleet roll-up (per discovered
  ``report.json``).
* :class:`FleetReport` — the meta-report aggregated across N runs.
"""

from __future__ import annotations

from gauntlet.aggregate.schema import FleetReport as FleetReport
from gauntlet.aggregate.schema import FleetRun as FleetRun

__all__ = [
    "FleetReport",
    "FleetRun",
]
