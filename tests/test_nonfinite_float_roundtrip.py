"""Regression: ``Episode`` and ``Report`` must round-trip JSON for non-finite floats.

Pydantic 2's default float serializer emits ``null`` for ``float('nan')`` /
``float('inf')`` and the validator then refuses ``None`` for ``float`` fields,
so any episode or report carrying a non-finite value (NaN reward from a broken
policy, ``float('nan')`` heat-map cells for empty grid positions) silently
becomes unloadable. The fix wires ``ser_json_inf_nan="strings"`` into every
schema's ``ConfigDict``; this test pins it.

Surfaced by the property tests added in ``tests/test_property_episode_roundtrip.py``.
"""

from __future__ import annotations

import math

from gauntlet.report.schema import (
    AxisBreakdown,
    CellBreakdown,
    FailureCluster,
    Heatmap2D,
    Report,
)
from gauntlet.runner.episode import Episode


def test_episode_roundtrips_nan_and_inf() -> None:
    ep = Episode(
        suite_name="x",
        cell_index=0,
        episode_index=0,
        seed=1,
        perturbation_config={"a": float("nan")},
        success=True,
        terminated=True,
        truncated=False,
        step_count=1,
        total_reward=float("inf"),
    )
    js = ep.model_dump_json()
    rt = Episode.model_validate_json(js)
    assert math.isnan(rt.perturbation_config["a"])
    assert math.isinf(rt.total_reward) and rt.total_reward > 0


def test_episode_roundtrips_negative_inf() -> None:
    ep = Episode(
        suite_name="x",
        cell_index=0,
        episode_index=0,
        seed=1,
        perturbation_config={},
        success=False,
        terminated=False,
        truncated=True,
        step_count=0,
        total_reward=float("-inf"),
    )
    rt = Episode.model_validate_json(ep.model_dump_json())
    assert math.isinf(rt.total_reward) and rt.total_reward < 0


def test_heatmap2d_roundtrips_nan_cells() -> None:
    """The heat-map's empty cells are deliberately NaN — see
    :class:`gauntlet.report.schema.Heatmap2D` docstring."""
    hm = Heatmap2D(
        axis_x="lighting",
        axis_y="texture",
        x_values=[0.3, 1.5],
        y_values=[0.0, 1.0],
        success_rate=[[1.0, float("nan")], [float("nan"), 0.0]],
    )
    rt = Heatmap2D.model_validate_json(hm.model_dump_json())
    assert rt.success_rate[0][0] == 1.0
    assert math.isnan(rt.success_rate[0][1])
    assert math.isnan(rt.success_rate[1][0])
    assert rt.success_rate[1][1] == 0.0


def test_axis_breakdown_roundtrips_nan_rate() -> None:
    ab = AxisBreakdown(
        name="lighting",
        rates={0.3: float("nan")},
        counts={0.3: 0},
        successes={0.3: 0},
    )
    rt = AxisBreakdown.model_validate_json(ab.model_dump_json())
    assert math.isnan(rt.rates[0.3])


def test_cell_breakdown_roundtrips_inf_rate() -> None:
    cb = CellBreakdown(
        cell_index=0,
        perturbation_config={"a": float("nan")},
        n_episodes=0,
        n_success=0,
        success_rate=float("nan"),
    )
    rt = CellBreakdown.model_validate_json(cb.model_dump_json())
    assert math.isnan(rt.success_rate)
    assert math.isnan(rt.perturbation_config["a"])


def test_failure_cluster_roundtrips_inf_lift() -> None:
    fc = FailureCluster(
        axes={"lighting": 0.3, "texture": 1.0},
        n_episodes=10,
        n_success=0,
        failure_rate=1.0,
        lift=float("inf"),
    )
    rt = FailureCluster.model_validate_json(fc.model_dump_json())
    assert math.isinf(rt.lift) and rt.lift > 0


def test_report_roundtrips_nan_overall_rate() -> None:
    rep = Report(
        suite_name="x",
        suite_env=None,
        n_episodes=0,
        n_success=0,
        per_axis=[],
        per_cell=[],
        failure_clusters=[],
        heatmap_2d={},
        overall_success_rate=float("nan"),
        overall_failure_rate=float("nan"),
        cluster_multiple=2.0,
    )
    rt = Report.model_validate_json(rep.model_dump_json())
    assert math.isnan(rt.overall_success_rate)
    assert math.isnan(rt.overall_failure_rate)
