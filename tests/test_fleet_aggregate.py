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

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from gauntlet.aggregate import (
    FleetReport,
    FleetRun,
    aggregate_directory,
    aggregate_reports,
    discover_run_files,
)
from gauntlet.report import Report, build_report
from gauntlet.report.schema import AxisBreakdown, FailureCluster
from gauntlet.runner.episode import Episode

# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------


def _ep(
    *,
    suite_name: str = "tiny-fleet",
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
    seed: int = 0,
) -> Episode:
    """Build an Episode without touching MuJoCo."""
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=seed,
        perturbation_config=dict(config),
        success=success,
        terminated=success,
        truncated=False,
        step_count=5,
        total_reward=1.0 if success else 0.0,
    )


def _grid_episodes(
    *,
    suite_name: str,
    failing_cluster: tuple[float, float] | None,
) -> list[Episode]:
    """Build a tidy 3x3 grid of episodes for two axes.

    ``failing_cluster``, when set, marks one (lighting, texture) cell as
    100% failure (5 episodes). Other cells succeed 100%. Cluster
    fingerprinting tests build runs with overlapping / disjoint
    clusters this way.
    """
    eps: list[Episode] = []
    cell_idx = 0
    for lighting in (0.3, 0.6, 0.9):
        for texture in (0.0, 1.0, 2.0):
            cell_eps = 5
            for ep_i in range(cell_eps):
                config = {"lighting_intensity": lighting, "object_texture": texture}
                if failing_cluster is not None and (lighting, texture) == failing_cluster:
                    success = False
                else:
                    success = True
                eps.append(
                    _ep(
                        suite_name=suite_name,
                        cell_index=cell_idx,
                        episode_index=ep_i,
                        success=success,
                        config=config,
                        seed=cell_idx * 100 + ep_i,
                    )
                )
            cell_idx += 1
    return eps


def _report_for(
    *,
    suite_name: str = "tiny-fleet",
    failing_cluster: tuple[float, float] | None,
) -> Report:
    return build_report(_grid_episodes(suite_name=suite_name, failing_cluster=failing_cluster))


def _write_report_dir(
    base: Path,
    rep: Report,
    *,
    run_name: str,
) -> Path:
    """Mirror the on-disk layout produced by ``gauntlet run --out``."""
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "report.json").write_text(
        json.dumps(rep.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_dir / "report.json"


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


# ---------------------------------------------------------------------------
# Discovery + basic aggregation.
# ---------------------------------------------------------------------------


def test_discover_run_files_recurses_and_sorts(tmp_path: Path) -> None:
    rep = _report_for(failing_cluster=None)
    _write_report_dir(tmp_path, rep, run_name="run-b")
    _write_report_dir(tmp_path, rep, run_name="run-a")
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_report_dir(nested, rep, run_name="run-c")
    found = discover_run_files(tmp_path)
    # Sorted, recursive, no extras.
    assert [p.name for p in found] == ["report.json", "report.json", "report.json"]
    assert len(found) == 3
    rels = sorted(p.relative_to(tmp_path).as_posix() for p in found)
    assert rels == [
        "nested/run-c/report.json",
        "run-a/report.json",
        "run-b/report.json",
    ]


def test_discover_run_files_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert discover_run_files(tmp_path) == []


def test_discover_run_files_nonexistent_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_run_files(tmp_path / "does-not-exist")


def test_aggregate_reports_basic_counts() -> None:
    """Per-axis counts/successes sum across runs."""
    rep_a = _report_for(failing_cluster=None)  # 45 episodes, 100% success
    rep_b = _report_for(failing_cluster=(0.3, 0.0))  # 45 eps, 5 fail
    fleet = aggregate_reports([rep_a, rep_b])
    assert fleet.n_runs == 2
    assert fleet.n_total_episodes == 90
    # 5 failures / 90 = 0.0555...
    assert fleet.fleet_baseline_failure_rate == pytest.approx(5 / 90)
    light = fleet.per_axis_aggregate["lighting_intensity"]
    assert sum(light.counts.values()) == 90
    assert sum(light.successes.values()) == 85


def test_aggregate_reports_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_reports([])


def test_aggregate_reports_invalid_threshold_raises() -> None:
    rep = _report_for(failing_cluster=None)
    with pytest.raises(ValueError):
        aggregate_reports([rep], persistence_threshold=-0.01)
    with pytest.raises(ValueError):
        aggregate_reports([rep], persistence_threshold=1.5)


def test_aggregate_directory_round_trips_through_json(tmp_path: Path) -> None:
    rep_a = _report_for(failing_cluster=(0.3, 0.0))
    rep_b = _report_for(failing_cluster=(0.3, 0.0))
    _write_report_dir(tmp_path, rep_a, run_name="run-a")
    _write_report_dir(tmp_path, rep_b, run_name="run-b")
    fleet = aggregate_directory(tmp_path)
    js = fleet.model_dump_json()
    rt = FleetReport.model_validate_json(js)
    assert rt.n_runs == 2
    assert rt.runs[0].source_file == "run-a/report.json"


def test_aggregate_directory_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        aggregate_directory(tmp_path)


def test_aggregate_directory_malformed_report_errors_with_path(tmp_path: Path) -> None:
    bad = tmp_path / "broken" / "report.json"
    bad.parent.mkdir()
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        aggregate_directory(tmp_path)
    assert "broken/report.json" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Persistent-cluster algorithm.
# ---------------------------------------------------------------------------


def test_persistent_cluster_appears_in_majority_of_runs() -> None:
    """3 runs share a cluster; 2 runs do not. With threshold 0.5 it's persistent."""
    shared = (0.3, 0.0)
    other = (0.9, 2.0)
    reps = [
        _report_for(failing_cluster=shared),
        _report_for(failing_cluster=shared),
        _report_for(failing_cluster=shared),
        _report_for(failing_cluster=other),
        _report_for(failing_cluster=other),
    ]
    fleet = aggregate_reports(reps, persistence_threshold=0.5)
    fingerprints = {tuple(sorted(c.axes.items())) for c in fleet.persistent_failure_clusters}
    assert (
        ("lighting_intensity", 0.3),
        ("object_texture", 0.0),
    ) in fingerprints
    # The 2-of-5 cluster is below the 0.5 threshold.
    assert (
        ("lighting_intensity", 0.9),
        ("object_texture", 2.0),
    ) not in fingerprints


def test_persistent_cluster_threshold_is_inclusive() -> None:
    """A cluster appearing in exactly 50% of 4 runs is included at threshold=0.5."""
    shared = (0.3, 0.0)
    reps = [
        _report_for(failing_cluster=shared),
        _report_for(failing_cluster=shared),
        _report_for(failing_cluster=None),
        _report_for(failing_cluster=None),
    ]
    fleet = aggregate_reports(reps, persistence_threshold=0.5)
    fingerprints = {tuple(sorted(c.axes.items())) for c in fleet.persistent_failure_clusters}
    assert (
        ("lighting_intensity", 0.3),
        ("object_texture", 0.0),
    ) in fingerprints
    # And at 0.51 it is excluded (2/4 = 0.5 < 0.51).
    fleet_strict = aggregate_reports(reps, persistence_threshold=0.51)
    fingerprints_strict = {
        tuple(sorted(c.axes.items())) for c in fleet_strict.persistent_failure_clusters
    }
    assert (
        ("lighting_intensity", 0.3),
        ("object_texture", 0.0),
    ) not in fingerprints_strict


def test_persistent_cluster_lift_uses_fleet_baseline() -> None:
    """The cluster's lift is recomputed against the fleet baseline."""
    shared = (0.3, 0.0)
    reps = [_report_for(failing_cluster=shared) for _ in range(3)]
    fleet = aggregate_reports(reps, persistence_threshold=0.5)
    # 3 runs * 5 failed episodes = 15 fails out of 3*45 = 135 total.
    assert fleet.fleet_baseline_failure_rate == pytest.approx(15 / 135)
    cluster = next(
        c
        for c in fleet.persistent_failure_clusters
        if tuple(sorted(c.axes.items())) == (("lighting_intensity", 0.3), ("object_texture", 0.0))
    )
    # Cluster: 3 runs * 5 failed = 15 episodes, 0 successes, failure_rate = 1.0.
    assert cluster.n_episodes == 15
    assert cluster.n_success == 0
    assert cluster.failure_rate == pytest.approx(1.0)
    assert cluster.lift == pytest.approx(1.0 / fleet.fleet_baseline_failure_rate)


def test_persistent_cluster_fingerprint_invariant_under_axis_order() -> None:
    """Sorted-name fingerprint matches even if per-run axis insertion order differs.

    Build two reports manually with the same cluster but axes inserted
    in different orders (``{a: 0, b: 1}`` vs ``{b: 1, a: 0}``).
    """
    base_eps = _grid_episodes(suite_name="s", failing_cluster=(0.3, 0.0))
    rep_a = build_report(base_eps)
    eps_b: list[Episode] = []
    cell_idx = 0
    for texture in (0.0, 1.0, 2.0):
        for lighting in (0.3, 0.6, 0.9):
            for ep_i in range(5):
                success = not (lighting == 0.3 and texture == 0.0)
                eps_b.append(
                    _ep(
                        suite_name="s",
                        cell_index=cell_idx,
                        episode_index=ep_i,
                        success=success,
                        config={"object_texture": texture, "lighting_intensity": lighting},
                        seed=cell_idx * 100 + ep_i,
                    )
                )
            cell_idx += 1
    rep_b = build_report(eps_b)
    fleet = aggregate_reports([rep_a, rep_b], persistence_threshold=0.5)
    fingerprints = {tuple(sorted(c.axes.items())) for c in fleet.persistent_failure_clusters}
    expected = (("lighting_intensity", 0.3), ("object_texture", 0.0))
    assert expected in fingerprints
    matching = [
        c for c in fleet.persistent_failure_clusters if tuple(sorted(c.axes.items())) == expected
    ]
    assert len(matching) == 1
    # And it pooled both runs' episodes (2 runs * 5 failed eps each).
    assert matching[0].n_episodes == 10


def test_persistent_clusters_sorted_by_lift_then_failure_rate() -> None:
    """Stable presentation: lift desc, then failure rate desc."""
    reps = []
    for _ in range(3):
        eps = _grid_episodes(suite_name="s", failing_cluster=(0.3, 0.0))
        # Add some half-failures at (0.6, 1.0).
        for ep_i in range(5):
            eps.append(
                _ep(
                    suite_name="s",
                    cell_index=99,
                    episode_index=ep_i,
                    success=ep_i >= 3,  # 3/5 fail
                    config={"lighting_intensity": 0.6, "object_texture": 1.0},
                    seed=10000 + ep_i,
                )
            )
        reps.append(build_report(eps))
    fleet = aggregate_reports(reps, persistence_threshold=0.5)
    lifts = [c.lift for c in fleet.persistent_failure_clusters]
    assert lifts == sorted(lifts, reverse=True)


# ---------------------------------------------------------------------------
# Mean / std / cross-run distribution.
# ---------------------------------------------------------------------------


def test_cross_run_success_distribution_keyed_by_suite_name() -> None:
    rep_a = _report_for(suite_name="suite-a", failing_cluster=None)
    rep_b = _report_for(suite_name="suite-a", failing_cluster=(0.3, 0.0))
    rep_c = _report_for(suite_name="suite-b", failing_cluster=None)
    fleet = aggregate_reports([rep_a, rep_b, rep_c])
    assert set(fleet.cross_run_success_distribution.keys()) == {"suite-a", "suite-b"}
    assert len(fleet.cross_run_success_distribution["suite-a"]) == 2
    assert len(fleet.cross_run_success_distribution["suite-b"]) == 1
    assert sorted(fleet.suite_names) == ["suite-a", "suite-b"]


def test_mean_and_std_success_rate_match_statistics() -> None:
    import statistics

    rep_a = _report_for(failing_cluster=None)
    rep_b = _report_for(failing_cluster=(0.3, 0.0))
    rep_c = _report_for(failing_cluster=(0.6, 1.0))
    fleet = aggregate_reports([rep_a, rep_b, rep_c])
    rates = [
        rep_a.overall_success_rate,
        rep_b.overall_success_rate,
        rep_c.overall_success_rate,
    ]
    assert fleet.mean_success_rate == pytest.approx(statistics.fmean(rates))
    assert fleet.std_success_rate == pytest.approx(statistics.pstdev(rates))


def test_single_run_std_is_zero() -> None:
    fleet = aggregate_reports([_report_for(failing_cluster=None)])
    assert fleet.std_success_rate == 0.0


# ---------------------------------------------------------------------------
# HTML rendering — structural + XSS.
# ---------------------------------------------------------------------------


from gauntlet.aggregate import render_fleet_html, write_fleet_html  # noqa: E402


def _make_clean_fleet() -> FleetReport:
    rep_a = _report_for(failing_cluster=(0.3, 0.0))
    rep_b = _report_for(failing_cluster=(0.3, 0.0))
    return aggregate_reports([rep_a, rep_b], persistence_threshold=0.5)


def test_render_fleet_html_starts_with_doctype() -> None:
    fleet = _make_clean_fleet()
    html = render_fleet_html(fleet)
    assert html.startswith("<!DOCTYPE html>")


def test_render_fleet_html_contains_expected_sections() -> None:
    fleet = _make_clean_fleet()
    html = render_fleet_html(fleet)
    # Summary card metrics — labels, not values.
    assert "Runs" in html
    assert "Total episodes" in html
    assert "Mean success" in html
    # Persistent-cluster table heading.
    assert "Persistent failure clusters" in html
    # Per-axis chart canvases (one per aggregate axis).
    for axis in fleet.per_axis_aggregate:
        assert f'id="fleet-axis-{axis}"' in html
    # Per-run table header.
    assert "Per-run breakdown" in html


def test_write_fleet_html_writes_a_file(tmp_path: Path) -> None:
    fleet = _make_clean_fleet()
    out = tmp_path / "fleet_report.html"
    write_fleet_html(fleet, out)
    assert out.is_file()
    body = out.read_text(encoding="utf-8")
    assert body.startswith("<!DOCTYPE html>")
    assert body.endswith("</html>\n") or body.endswith("</html>")


# XSS payload — same shape as tests/test_security_html_report.py.
_XSS = "<script>alert(1)</script>"


def test_xss_in_policy_label_is_escaped() -> None:
    """An attacker-controlled policy_label MUST be HTML-escaped."""
    fr = FleetRun(
        run_id="r0",
        policy_label=_XSS,
        suite_name="suite",
        suite_env=None,
        n_episodes=1,
        n_success=1,
        success_rate=1.0,
        source_file="r0/report.json",
    )
    fleet = FleetReport(
        runs=[fr],
        n_runs=1,
        n_total_episodes=1,
        per_axis_aggregate={},
        persistent_failure_clusters=[],
        cross_run_success_distribution={"suite": [1.0]},
        persistence_threshold=0.5,
        fleet_baseline_failure_rate=0.0,
        mean_success_rate=1.0,
        std_success_rate=0.0,
        suite_names=["suite"],
    )
    html = render_fleet_html(fleet)
    assert _XSS not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_xss_in_source_file_is_escaped() -> None:
    fr = FleetRun(
        run_id="r0",
        policy_label="p",
        suite_name="suite",
        suite_env=None,
        n_episodes=1,
        n_success=1,
        success_rate=1.0,
        source_file=_XSS,
    )
    fleet = FleetReport(
        runs=[fr],
        n_runs=1,
        n_total_episodes=1,
        per_axis_aggregate={},
        persistent_failure_clusters=[],
        cross_run_success_distribution={"suite": [1.0]},
        persistence_threshold=0.5,
        fleet_baseline_failure_rate=0.0,
        mean_success_rate=1.0,
        std_success_rate=0.0,
        suite_names=["suite"],
    )
    html = render_fleet_html(fleet)
    assert _XSS not in html


def test_xss_in_axis_name_is_escaped() -> None:
    """Axis name surfaces in chart title and per-axis table."""
    breakdown = AxisBreakdown(
        name=_XSS,
        rates={0.0: 1.0, 1.0: 0.0},
        counts={0.0: 1, 1.0: 1},
        successes={0.0: 1, 1.0: 0},
    )
    fleet = FleetReport(
        runs=[],
        n_runs=0,
        n_total_episodes=0,
        per_axis_aggregate={_XSS: breakdown},
        persistent_failure_clusters=[],
        cross_run_success_distribution={},
        persistence_threshold=0.5,
        fleet_baseline_failure_rate=0.0,
        mean_success_rate=0.0,
        std_success_rate=0.0,
        suite_names=[],
    )
    html = render_fleet_html(fleet)
    assert _XSS not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_xss_in_cluster_axis_name_is_escaped() -> None:
    cluster = FailureCluster(
        axes={_XSS: 0.3, "second_axis": 1.0},
        n_episodes=10,
        n_success=0,
        failure_rate=1.0,
        lift=10.0,
    )
    fleet = FleetReport(
        runs=[],
        n_runs=0,
        n_total_episodes=0,
        per_axis_aggregate={},
        persistent_failure_clusters=[cluster],
        cross_run_success_distribution={},
        persistence_threshold=0.5,
        fleet_baseline_failure_rate=0.1,
        mean_success_rate=0.0,
        std_success_rate=0.0,
        suite_names=[],
    )
    html = render_fleet_html(fleet)
    assert _XSS not in html


def test_embedded_json_block_does_not_break_out_of_script() -> None:
    """``</script>`` inside an attacker-controlled string must be escaped
    inside the embedded ``<script id="fleet-data">`` block.
    """
    import re

    payload = "</script><script>alert(5)</script>"
    fr = FleetRun(
        run_id="r0",
        policy_label=payload,
        suite_name="suite",
        suite_env=None,
        n_episodes=1,
        n_success=1,
        success_rate=1.0,
        source_file="r0/report.json",
    )
    fleet = FleetReport(
        runs=[fr],
        n_runs=1,
        n_total_episodes=1,
        per_axis_aggregate={},
        persistent_failure_clusters=[],
        cross_run_success_distribution={"suite": [1.0]},
        persistence_threshold=0.5,
        fleet_baseline_failure_rate=0.0,
        mean_success_rate=1.0,
        std_success_rate=0.0,
        suite_names=["suite"],
    )
    html = render_fleet_html(fleet)
    match = re.search(
        r'<script id="fleet-data" type="application/json">(?P<body>.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None, "fleet-data script block missing"
    assert "</script>" not in match.group("body")
