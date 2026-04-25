"""Unit tests for :mod:`gauntlet.diff` — see ``polish/gauntlet-diff``.

The tests are pure-Python: synthetic :class:`gauntlet.report.Report`
objects are hand-constructed (no Runner / MuJoCo dependency) so the
suite stays fast and focused on the differ semantics.
"""

from __future__ import annotations

import json

import pytest

from gauntlet.diff import (
    AxisDelta,
    CellFlip,
    ClusterDelta,
    ReportDiff,
    diff_reports,
    render_text,
)
from gauntlet.report import (
    AxisBreakdown,
    CellBreakdown,
    FailureCluster,
    Report,
)

# ──────────────────────────────────────────────────────────────────────
# Report builders.
# ──────────────────────────────────────────────────────────────────────


def _axis(name: str, rates: dict[float, float]) -> AxisBreakdown:
    """Build an :class:`AxisBreakdown` with synthetic counts.

    ``counts`` and ``successes`` are derived from ``rates`` by assuming
    10 episodes per axis value — fine for the differ's purposes since
    the differ never inspects them.
    """
    counts = dict.fromkeys(rates, 10)
    successes = {value: round(rate * 10) for value, rate in rates.items()}
    return AxisBreakdown(name=name, rates=dict(rates), counts=counts, successes=successes)


def _cell(
    *, cell_index: int, perturbation_config: dict[str, float], success_rate: float
) -> CellBreakdown:
    n_episodes = 10
    n_success = round(success_rate * n_episodes)
    return CellBreakdown(
        cell_index=cell_index,
        perturbation_config=dict(perturbation_config),
        n_episodes=n_episodes,
        n_success=n_success,
        success_rate=success_rate,
    )


def _cluster(*, axes: dict[str, float], failure_rate: float, lift: float) -> FailureCluster:
    n_episodes = 5
    n_success = round((1.0 - failure_rate) * n_episodes)
    return FailureCluster(
        axes=dict(axes),
        n_episodes=n_episodes,
        n_success=n_success,
        failure_rate=failure_rate,
        lift=lift,
    )


def _report(
    *,
    suite_name: str = "synthetic",
    overall_success_rate: float,
    n_episodes: int = 100,
    per_axis: list[AxisBreakdown] | None = None,
    per_cell: list[CellBreakdown] | None = None,
    failure_clusters: list[FailureCluster] | None = None,
) -> Report:
    n_success = round(overall_success_rate * n_episodes)
    return Report(
        suite_name=suite_name,
        n_episodes=n_episodes,
        n_success=n_success,
        per_axis=per_axis if per_axis is not None else [],
        per_cell=per_cell if per_cell is not None else [],
        failure_clusters=failure_clusters if failure_clusters is not None else [],
        heatmap_2d={},
        overall_success_rate=overall_success_rate,
        overall_failure_rate=1.0 - overall_success_rate,
        cluster_multiple=2.0,
    )


# ──────────────────────────────────────────────────────────────────────
# 1 — identical reports → empty deltas.
# ──────────────────────────────────────────────────────────────────────


def test_diff_identical_reports_yields_empty_deltas() -> None:
    rep = _report(
        overall_success_rate=0.8,
        per_axis=[_axis("lighting_intensity", {0.3: 0.9, 1.5: 0.7})],
        per_cell=[
            _cell(cell_index=0, perturbation_config={"lighting_intensity": 0.3}, success_rate=0.9),
            _cell(cell_index=1, perturbation_config={"lighting_intensity": 1.5}, success_rate=0.7),
        ],
        failure_clusters=[
            _cluster(
                axes={"lighting_intensity": 1.5, "object_mass": 2.0},
                failure_rate=0.6,
                lift=3.0,
            ),
        ],
    )

    diff = diff_reports(rep, rep)

    assert diff.overall_success_rate_delta == 0.0
    assert diff.n_episodes_delta == 0
    assert diff.cell_flips == []
    assert diff.cluster_added == []
    assert diff.cluster_removed == []
    assert diff.cluster_intensified == []
    # Per-axis deltas are present (one AxisDelta per shared axis) but every
    # rate delta is exactly zero.
    assert set(diff.axis_deltas.keys()) == {"lighting_intensity"}
    assert all(v == 0.0 for v in diff.axis_deltas["lighting_intensity"].rate_deltas.values())


# ──────────────────────────────────────────────────────────────────────
# 2 — overall + axis deltas surface correctly.
# ──────────────────────────────────────────────────────────────────────


def test_diff_overall_and_axis_deltas() -> None:
    a = _report(
        overall_success_rate=0.8,
        n_episodes=100,
        per_axis=[_axis("lighting_intensity", {0.3: 0.9, 1.5: 0.7})],
    )
    b = _report(
        overall_success_rate=0.6,
        n_episodes=120,
        per_axis=[_axis("lighting_intensity", {0.3: 0.85, 1.5: 0.4})],
    )

    diff = diff_reports(a, b)

    assert diff.n_episodes_delta == 20
    assert diff.overall_success_rate_delta == pytest.approx(-0.2)
    rate_deltas = diff.axis_deltas["lighting_intensity"].rate_deltas
    assert rate_deltas[0.3] == pytest.approx(-0.05)
    assert rate_deltas[1.5] == pytest.approx(-0.3)


# ──────────────────────────────────────────────────────────────────────
# 3 — cell flip surfacing & threshold.
# ──────────────────────────────────────────────────────────────────────


def test_diff_cell_flips_threshold_two_sided() -> None:
    a = _report(
        overall_success_rate=0.5,
        per_cell=[
            _cell(cell_index=0, perturbation_config={"axis_a": 0.0}, success_rate=1.0),
            _cell(cell_index=1, perturbation_config={"axis_a": 1.0}, success_rate=0.5),
            _cell(cell_index=2, perturbation_config={"axis_a": 2.0}, success_rate=0.5),
        ],
    )
    b = _report(
        overall_success_rate=0.5,
        per_cell=[
            # cell 0: regression of -0.5 — counted.
            _cell(cell_index=0, perturbation_config={"axis_a": 0.0}, success_rate=0.5),
            # cell 1: improvement of +0.4 — counted (>= 0.10).
            _cell(cell_index=1, perturbation_config={"axis_a": 1.0}, success_rate=0.9),
            # cell 2: drift of +0.05 — under threshold, NOT counted.
            _cell(cell_index=2, perturbation_config={"axis_a": 2.0}, success_rate=0.55),
        ],
    )

    diff = diff_reports(a, b, cell_flip_threshold=0.10)

    flipped_indices = sorted(f.cell_index for f in diff.cell_flips)
    assert flipped_indices == [0, 1]
    by_cell = {f.cell_index: f for f in diff.cell_flips}
    assert by_cell[0].direction == "regressed"
    assert by_cell[1].direction == "improved"
    # Sort order: regressions (negative delta) before improvements.
    assert diff.cell_flips[0].direction == "regressed"
    assert diff.cell_flips[-1].direction == "improved"


# ──────────────────────────────────────────────────────────────────────
# 4 — cluster added / removed / intensified.
# ──────────────────────────────────────────────────────────────────────


def test_diff_cluster_set_difference_and_intensification() -> None:
    common_cluster_a = _cluster(axes={"axis_a": 1.0, "axis_b": 0.5}, failure_rate=0.4, lift=1.5)
    common_cluster_b = _cluster(axes={"axis_a": 1.0, "axis_b": 0.5}, failure_rate=0.8, lift=2.5)
    only_in_a = _cluster(axes={"axis_a": 0.0, "axis_b": 0.5}, failure_rate=0.7, lift=2.0)
    only_in_b = _cluster(axes={"axis_a": 2.0, "axis_b": 0.5}, failure_rate=0.9, lift=3.5)

    a = _report(
        overall_success_rate=0.6,
        failure_clusters=[common_cluster_a, only_in_a],
    )
    b = _report(
        overall_success_rate=0.5,
        failure_clusters=[common_cluster_b, only_in_b],
    )

    diff = diff_reports(a, b, cluster_intensify_threshold=0.5)

    assert len(diff.cluster_added) == 1
    assert diff.cluster_added[0].axes == {"axis_a": 2.0, "axis_b": 0.5}
    assert len(diff.cluster_removed) == 1
    assert diff.cluster_removed[0].axes == {"axis_a": 0.0, "axis_b": 0.5}
    # common cluster: lift went 1.5 -> 2.5 (delta = 1.0, >= 0.5 threshold).
    assert len(diff.cluster_intensified) == 1
    intensified = diff.cluster_intensified[0]
    assert intensified.axes == {"axis_a": 1.0, "axis_b": 0.5}
    assert intensified.delta == pytest.approx(1.0)


def test_diff_cluster_intensify_threshold_gate() -> None:
    """Intensification under the threshold is dropped."""
    a = _report(
        overall_success_rate=0.6,
        failure_clusters=[
            _cluster(axes={"x": 1.0, "y": 0.5}, failure_rate=0.4, lift=1.5),
        ],
    )
    b = _report(
        overall_success_rate=0.55,
        failure_clusters=[
            # Lift rose by 0.2 — below the default 0.5 threshold.
            _cluster(axes={"x": 1.0, "y": 0.5}, failure_rate=0.5, lift=1.7),
        ],
    )
    diff = diff_reports(a, b)
    assert diff.cluster_intensified == []


def test_diff_cluster_de_intensification_not_reported() -> None:
    """Lift dropped → improvement, but ``cluster_intensified`` is one-sided."""
    a = _report(
        overall_success_rate=0.5,
        failure_clusters=[
            _cluster(axes={"x": 1.0, "y": 0.5}, failure_rate=0.8, lift=3.0),
        ],
    )
    b = _report(
        overall_success_rate=0.7,
        failure_clusters=[
            _cluster(axes={"x": 1.0, "y": 0.5}, failure_rate=0.4, lift=1.5),
        ],
    )
    diff = diff_reports(a, b, cluster_intensify_threshold=0.5)
    assert diff.cluster_intensified == []


# ──────────────────────────────────────────────────────────────────────
# 5 — labels, suite names, validation.
# ──────────────────────────────────────────────────────────────────────


def test_diff_propagates_labels_and_suite_names() -> None:
    a = _report(suite_name="suite-a", overall_success_rate=0.8)
    b = _report(suite_name="suite-b", overall_success_rate=0.7)

    diff = diff_reports(a, b, a_label="first.json", b_label="second.json")

    assert diff.a_label == "first.json"
    assert diff.b_label == "second.json"
    assert diff.a_suite_name == "suite-a"
    assert diff.b_suite_name == "suite-b"


def test_diff_rejects_negative_thresholds() -> None:
    rep = _report(overall_success_rate=0.5)
    with pytest.raises(ValueError, match="cell_flip_threshold"):
        diff_reports(rep, rep, cell_flip_threshold=-0.1)
    with pytest.raises(ValueError, match="cluster_intensify_threshold"):
        diff_reports(rep, rep, cluster_intensify_threshold=-0.1)


# ──────────────────────────────────────────────────────────────────────
# 6 — JSON round-trip.
# ──────────────────────────────────────────────────────────────────────


def test_diff_round_trips_through_json() -> None:
    a = _report(
        overall_success_rate=0.8,
        per_axis=[_axis("axis_a", {0.0: 0.9, 1.0: 0.7})],
        per_cell=[
            _cell(cell_index=0, perturbation_config={"axis_a": 0.0}, success_rate=1.0),
        ],
        failure_clusters=[
            _cluster(axes={"axis_a": 1.0, "axis_b": 0.5}, failure_rate=0.6, lift=2.0),
        ],
    )
    b = _report(
        overall_success_rate=0.5,
        per_axis=[_axis("axis_a", {0.0: 0.4, 1.0: 0.6})],
        per_cell=[
            _cell(cell_index=0, perturbation_config={"axis_a": 0.0}, success_rate=0.4),
        ],
        failure_clusters=[
            _cluster(axes={"axis_a": 1.0, "axis_b": 0.5}, failure_rate=0.95, lift=4.0),
        ],
    )
    diff = diff_reports(a, b)
    serialized = diff.model_dump_json()
    restored = ReportDiff.model_validate_json(serialized)
    assert restored == diff


# ──────────────────────────────────────────────────────────────────────
# 7 — render_text basics.
# ──────────────────────────────────────────────────────────────────────


def test_render_text_contains_expected_markers() -> None:
    a = _report(
        overall_success_rate=0.8,
        per_axis=[_axis("lighting_intensity", {0.3: 0.9, 1.5: 0.7})],
        per_cell=[
            _cell(cell_index=0, perturbation_config={"lighting_intensity": 0.3}, success_rate=0.9),
            _cell(cell_index=1, perturbation_config={"lighting_intensity": 1.5}, success_rate=0.7),
        ],
        failure_clusters=[
            _cluster(
                axes={"lighting_intensity": 1.5, "object_mass": 2.0},
                failure_rate=0.6,
                lift=2.0,
            ),
        ],
    )
    b = _report(
        overall_success_rate=0.55,
        per_axis=[_axis("lighting_intensity", {0.3: 0.9, 1.5: 0.2})],
        per_cell=[
            _cell(cell_index=0, perturbation_config={"lighting_intensity": 0.3}, success_rate=0.9),
            _cell(cell_index=1, perturbation_config={"lighting_intensity": 1.5}, success_rate=0.2),
        ],
        failure_clusters=[
            _cluster(
                axes={"lighting_intensity": 1.5, "object_mass": 2.0},
                failure_rate=0.9,
                lift=3.5,
            ),
        ],
    )
    diff = diff_reports(a, b, a_label="checkpoint-001.json", b_label="checkpoint-002.json")
    text = render_text(diff)

    # Header lines mention both labels.
    assert "checkpoint-001.json" in text
    assert "checkpoint-002.json" in text

    # +/- markers anchor the diff style; signs always include sign.
    assert "-" in text
    # Axis name appears in the per-axis section.
    assert "lighting_intensity" in text
    # The regressed cell line surfaces the perturbation_config value.
    assert "1.5" in text
    # Cluster intensification surfaces the multi-axis identity.
    assert "object_mass" in text


def test_render_text_is_pure_plaintext() -> None:
    """No ANSI escape codes — color is applied at the CLI layer."""
    a = _report(overall_success_rate=0.8)
    b = _report(overall_success_rate=0.5)
    text = render_text(diff_reports(a, b))
    assert "\x1b[" not in text


def test_render_text_handles_empty_diff() -> None:
    rep = _report(overall_success_rate=0.8)
    text = render_text(diff_reports(rep, rep))
    assert "no changes" in text.lower() or "no differences" in text.lower()


# ──────────────────────────────────────────────────────────────────────
# 8 — non-shared axes are skipped (different schemas → no axis delta).
# ──────────────────────────────────────────────────────────────────────


def test_diff_skips_axes_present_in_only_one_report() -> None:
    a = _report(
        overall_success_rate=0.8,
        per_axis=[_axis("axis_a", {0.3: 0.9, 1.5: 0.7})],
    )
    b = _report(
        overall_success_rate=0.5,
        per_axis=[
            _axis("axis_a", {0.3: 0.5, 1.5: 0.5}),
            _axis("axis_b", {0.0: 0.9}),
        ],
    )
    diff = diff_reports(a, b)
    # Only the shared axis surfaces.
    assert set(diff.axis_deltas.keys()) == {"axis_a"}


def test_diff_skips_axis_values_present_in_only_one_report() -> None:
    """Per-axis-value deltas only span the intersection of axis values."""
    a = _report(
        overall_success_rate=0.8,
        per_axis=[_axis("axis_a", {0.3: 0.9, 1.5: 0.7})],
    )
    b = _report(
        overall_success_rate=0.5,
        per_axis=[_axis("axis_a", {0.3: 0.5, 1.5: 0.5, 2.5: 0.4})],
    )
    diff = diff_reports(a, b)
    # 2.5 is only in b → skipped; 0.3 and 1.5 are present in both.
    assert set(diff.axis_deltas["axis_a"].rate_deltas.keys()) == {0.3, 1.5}


# ──────────────────────────────────────────────────────────────────────
# 9 — type / forbid-extra discipline.
# ──────────────────────────────────────────────────────────────────────


def test_axis_delta_forbids_extra_fields() -> None:
    with pytest.raises(ValueError):
        AxisDelta.model_validate({"name": "x", "rate_deltas": {0.0: 0.1}, "extra": 1})


def test_cell_flip_direction_is_literal() -> None:
    with pytest.raises(ValueError):
        CellFlip.model_validate(
            {
                "cell_index": 0,
                "perturbation_config": {"x": 0.0},
                "a_success_rate": 1.0,
                "b_success_rate": 0.0,
                "direction": "sideways",
            }
        )


def test_cluster_delta_serialises_to_json_dict() -> None:
    cd = ClusterDelta(axes={"x": 1.0, "y": 2.0}, a_lift=1.0, b_lift=3.0, delta=2.0)
    payload = json.loads(cd.model_dump_json())
    assert payload["delta"] == 2.0
