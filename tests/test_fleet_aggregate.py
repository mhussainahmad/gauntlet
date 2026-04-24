"""Fleet-aggregate tests — see ``docs/phase3-rfc-019-fleet-aggregate.md``.

Phase 3 Task 19 ships ``gauntlet aggregate`` and the
:class:`gauntlet.aggregate.FleetReport` schema. Tests live in this one
module to keep the contract one place: schema round-trip,
``aggregate_reports`` semantics, persistent-cluster algorithm, and
HTML rendering. The CLI surface has its own module
(:mod:`tests.test_cli_aggregate`).

This file grows across the commit chain — first the schema tests
(this commit), then the analyze + cluster tests, then the HTML / XSS
tests.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from gauntlet.aggregate import FleetReport, FleetRun
from gauntlet.report.schema import AxisBreakdown, FailureCluster

# ---------------------------------------------------------------------------
# Schema — round-trip + extra="forbid" + non-finite floats.
# ---------------------------------------------------------------------------


def test_fleet_run_round_trips_through_json() -> None:
    fr = FleetRun(
        run_id="run-001",
        policy_label="policy-001",
        suite_name="tiny-fleet",
        suite_env="tabletop",
        n_episodes=42,
        n_success=30,
        success_rate=30 / 42,
        source_file="run-001/report.json",
    )
    rt = FleetRun.model_validate_json(fr.model_dump_json())
    assert rt == fr


def test_fleet_run_rejects_extra_fields() -> None:
    """``extra="forbid"`` is the contract — silent additions are bugs."""
    payload = {
        "run_id": "r",
        "policy_label": "p",
        "suite_name": "s",
        "suite_env": None,
        "n_episodes": 1,
        "n_success": 1,
        "success_rate": 1.0,
        "source_file": "r/report.json",
        "extra_field_that_should_not_exist": "boom",
    }
    with pytest.raises(ValidationError):
        FleetRun.model_validate(payload)


def test_fleet_report_rejects_extra_fields() -> None:
    payload = {
        "runs": [],
        "n_runs": 0,
        "n_total_episodes": 0,
        "per_axis_aggregate": {},
        "persistent_failure_clusters": [],
        "cross_run_success_distribution": {},
        "persistence_threshold": 0.5,
        "fleet_baseline_failure_rate": 0.0,
        "mean_success_rate": 0.0,
        "std_success_rate": 0.0,
        "suite_names": [],
        "extra_field": 1,
    }
    with pytest.raises(ValidationError):
        FleetReport.model_validate(payload)


def test_fleet_report_round_trips_nan_and_inf() -> None:
    """Mirror of hotfix #18 — non-finite floats round-trip cleanly."""
    fr = FleetReport(
        runs=[],
        n_runs=0,
        n_total_episodes=0,
        per_axis_aggregate={
            "lighting_intensity": AxisBreakdown(
                name="lighting_intensity",
                rates={0.3: float("nan")},
                counts={0.3: 0},
                successes={0.3: 0},
            ),
        },
        persistent_failure_clusters=[
            FailureCluster(
                axes={"lighting_intensity": 0.3, "object_texture": 1.0},
                n_episodes=10,
                n_success=0,
                failure_rate=1.0,
                lift=float("inf"),
            ),
        ],
        cross_run_success_distribution={"tiny-fleet": [float("nan"), 1.0]},
        persistence_threshold=0.5,
        fleet_baseline_failure_rate=float("nan"),
        mean_success_rate=float("nan"),
        std_success_rate=float("nan"),
        suite_names=["tiny-fleet"],
    )
    rt = FleetReport.model_validate_json(fr.model_dump_json())
    assert math.isnan(rt.per_axis_aggregate["lighting_intensity"].rates[0.3])
    assert math.isinf(rt.persistent_failure_clusters[0].lift)
    assert math.isnan(rt.cross_run_success_distribution["tiny-fleet"][0])
    assert math.isnan(rt.fleet_baseline_failure_rate)
