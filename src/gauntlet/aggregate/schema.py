"""Pydantic models for the fleet-level meta-report.

See ``docs/phase3-rfc-019-fleet-aggregate.md`` for the design and
``GAUNTLET_SPEC.md`` §7 for the spec call-out.

The fleet schema is a *consumer* of :class:`gauntlet.report.Report` and
:class:`gauntlet.runner.Episode` — neither is changed by this module.
:class:`FleetReport` is what ``gauntlet aggregate`` writes to
``fleet_report.json``; :class:`FleetRun` is one entry per discovered
``report.json`` (one per evaluation run).

Design notes:

* Every model uses ``ConfigDict(extra="forbid", ser_json_inf_nan="strings")``
  matching the recent hotfix #18 — silent additions are a contract
  violation, and non-finite floats round-trip cleanly through JSON.
* Field order on :class:`FleetReport` is breakdown-first (per spec §6
  "never aggregate away failures"): ``persistent_failure_clusters``
  surfaces above the scalar means.
* :class:`FleetRun.source_file` is stored *relative to the scan root*,
  not absolute, so the resulting ``fleet_report.json`` is portable
  (RFC §3).
* :attr:`FleetReport.persistent_failure_clusters` reuses
  :class:`gauntlet.report.schema.FailureCluster` — no parallel type
  hierarchy. ``failure_rate`` and ``lift`` on those clusters are
  re-computed against the *fleet baseline* (not any single run's
  baseline); see RFC §4.2.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gauntlet.report.schema import AxisBreakdown, FailureCluster

__all__ = ["FleetReport", "FleetRun"]


class FleetRun(BaseModel):
    """One row of the fleet roll-up — metadata for a single discovered run.

    Built from one ``report.json`` (a serialised
    :class:`gauntlet.report.Report`). ``source_file`` is the path to
    that file *relative to the scan root* the user passed to
    ``gauntlet aggregate`` — keeping it relative makes the resulting
    ``fleet_report.json`` portable (you can ``tar`` the run tree and
    open it on a different machine).

    ``policy_label`` is a free-form string. The aggregator derives it
    from the run's directory name (the parent of ``report.json``)
    because that's the only piece of "where this run came from"
    information available from the existing :class:`Report` schema.
    Future work could let users pass a ``--label`` flag to
    ``gauntlet run`` so the label is recorded explicitly; until then,
    the directory name is the convention.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    run_id: str
    policy_label: str
    suite_name: str
    suite_env: str | None
    n_episodes: int
    n_success: int
    success_rate: float
    source_file: str


class FleetReport(BaseModel):
    """Top-level fleet meta-report.

    Built by :func:`gauntlet.aggregate.aggregate_reports` (or its
    directory-scanning sibling :func:`aggregate_directory`) from a
    list of per-run :class:`gauntlet.report.Report` objects.

    Field order is intentional, per spec §6:

    1. ``runs`` + counts — context, not headlines.
    2. ``per_axis_aggregate`` — the breakdown surface.
    3. ``persistent_failure_clusters`` — the headline (which axis
       combinations break the fleet, sorted by lift descending).
    4. ``cross_run_success_distribution`` — per-suite success-rate
       distribution across runs (the "regression scoreboard").
    5. ``persistence_threshold`` — the parameter that produced the
       cluster list, echoed for reproducibility.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    runs: list[FleetRun]
    n_runs: int
    n_total_episodes: int
    per_axis_aggregate: dict[str, AxisBreakdown]
    persistent_failure_clusters: list[FailureCluster]
    cross_run_success_distribution: dict[str, list[float]]
    persistence_threshold: float
    fleet_baseline_failure_rate: float
    mean_success_rate: float
    std_success_rate: float
    # Set of distinct ``suite_name`` values appearing in the input. A
    # multi-suite directory is allowed (RFC §5) — this lets downstream
    # consumers detect it without re-scanning ``runs``.
    suite_names: list[str] = Field(default_factory=list)
