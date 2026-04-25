"""Dashboard builder tests — see Phase 3 Task 20 / RFC 020.

Pin the contract for the index extraction (per-run summary +
per-axis aggregates + filter dimensions), the embedded JSON literal,
and the on-disk artefact (``index.html`` + ``dashboard.js`` +
``dashboard.css`` in the output directory). The CLI surface has its
own module (:mod:`tests.test_cli_dashboard`).

This file grows across the commit chain — first the index tests
(step 2), then template assertions (step 3), then JS / CSS asset
copy assertions (steps 4 / 5), then the filter-UI assertions
(step 7).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gauntlet.dashboard import (
    build_dashboard_index,
    discover_reports,
)
from gauntlet.report import Report, build_report
from gauntlet.runner.episode import Episode

# ---------------------------------------------------------------------------
# Shared fixtures — build mini reports without touching MuJoCo / pybullet.
# ---------------------------------------------------------------------------


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
    seed: int = 0,
    suite_name: str = "tiny-fleet",
) -> Episode:
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


def _grid_report(
    *,
    suite_name: str = "tiny-fleet",
    suite_env: str | None = "tabletop",
    failing_cluster: tuple[float, float] | None = None,
) -> Report:
    """3 x 3 grid x 5 episodes = 45 episodes per report."""
    eps: list[Episode] = []
    cell_idx = 0
    for lighting in (0.3, 0.6, 0.9):
        for texture in (0.0, 1.0, 2.0):
            for ep_i in range(5):
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
    return build_report(eps, suite_env=suite_env)


def _write_run(
    base: Path,
    rep: Report,
    *,
    run_name: str,
    sibling_html: bool = False,
) -> Path:
    """Mirror the on-disk layout produced by ``gauntlet run --out``."""
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    report_json = run_dir / "report.json"
    report_json.write_text(
        json.dumps(rep.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if sibling_html:
        (run_dir / "report.html").write_text(
            "<html><body>placeholder</body></html>", encoding="utf-8"
        )
    return report_json


# ---------------------------------------------------------------------------
# Discovery.
# ---------------------------------------------------------------------------


def test_discover_reports_recurses_and_sorts(tmp_path: Path) -> None:
    rep = _grid_report()
    _write_run(tmp_path, rep, run_name="run-b")
    _write_run(tmp_path, rep, run_name="run-a")
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_run(nested, rep, run_name="run-c")
    found = discover_reports(tmp_path)
    rels = sorted(p.relative_to(tmp_path).as_posix() for p in found)
    assert rels == [
        "nested/run-c/report.json",
        "run-a/report.json",
        "run-b/report.json",
    ]


def test_discover_reports_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert discover_reports(tmp_path) == []


def test_discover_reports_nonexistent_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_reports(tmp_path / "missing")


# ---------------------------------------------------------------------------
# Index extraction — runs / summary / per-axis / filter dims.
# ---------------------------------------------------------------------------


def test_build_dashboard_index_runs_carry_expected_fields(tmp_path: Path) -> None:
    _write_run(tmp_path, _grid_report(), run_name="run-a", sibling_html=True)
    _write_run(tmp_path, _grid_report(failing_cluster=(0.3, 0.0)), run_name="run-b")
    paths = discover_reports(tmp_path)
    index = build_dashboard_index(paths, base_dir=tmp_path)

    assert len(index["runs"]) == 2
    a, b = index["runs"]
    assert a["run_id"] == "run-a"
    assert a["policy_label"] == "run-a"
    assert a["suite_name"] == "tiny-fleet"
    assert a["env"] == "tabletop"
    assert a["n_episodes"] == 45
    assert a["n_success"] == 45
    assert a["success_rate"] == 1.0
    assert a["source_file"] == "run-a/report.json"
    # Sibling report.html is recorded as a relative href.
    assert a["report_html"] == "run-a/report.html"
    # mtime captured at discovery time — must be a finite float (epoch seconds).
    assert isinstance(a["mtime"], float)
    assert a["mtime"] > 0

    # The b run had a 5/45 failing cluster.
    assert b["n_success"] == 40
    assert b["success_rate"] == pytest.approx(40 / 45)
    # No sibling report.html written for run-b.
    assert b["report_html"] is None


def test_build_dashboard_index_summary_scalars(tmp_path: Path) -> None:
    _write_run(tmp_path, _grid_report(), run_name="run-a")
    _write_run(tmp_path, _grid_report(failing_cluster=(0.3, 0.0)), run_name="run-b")
    paths = discover_reports(tmp_path)
    summary = build_dashboard_index(paths, base_dir=tmp_path)["summary"]
    assert summary["n_runs"] == 2
    assert summary["n_total_episodes"] == 90
    # mean of [1.0, 40/45] = (1.0 + 40/45) / 2
    assert summary["mean_success_rate"] == pytest.approx((1.0 + 40 / 45) / 2)
    # population std = sqrt(((1.0 - mean)^2 + (40/45 - mean)^2) / 2)
    assert summary["std_success_rate"] > 0


def test_build_dashboard_index_per_axis_aggregates(tmp_path: Path) -> None:
    _write_run(tmp_path, _grid_report(), run_name="run-a")
    _write_run(tmp_path, _grid_report(failing_cluster=(0.3, 0.0)), run_name="run-b")
    paths = discover_reports(tmp_path)
    agg = build_dashboard_index(paths, base_dir=tmp_path)["per_axis_aggregate"]

    # Both axes survive aggregation; counts pool across runs.
    assert set(agg.keys()) == {"lighting_intensity", "object_texture"}
    light = agg["lighting_intensity"]
    # 45 episodes per run x 2 runs = 90 total across the lighting axis.
    assert sum(light["counts"].values()) == 90
    # Three lighting buckets, 30 episodes each (15/run x 2 runs).
    assert light["counts"] == {"0.3": 30.0, "0.6": 30.0, "0.9": 30.0}
    # 5 failures sit in the (0.3, 0.0) bucket so the lighting=0.3 marginal
    # rate dips below 1.0 while lighting=0.6 and 0.9 stay at 1.0.
    assert light["rates"]["0.3"] < 1.0
    assert light["rates"]["0.6"] == 1.0
    assert light["rates"]["0.9"] == 1.0


def test_build_dashboard_index_filter_dimensions(tmp_path: Path) -> None:
    """`envs` / `suite_names` / `policy_labels` are sorted-unique."""
    _write_run(tmp_path, _grid_report(suite_name="suite-x"), run_name="run-a")
    _write_run(tmp_path, _grid_report(suite_name="suite-y"), run_name="run-b")
    _write_run(tmp_path, _grid_report(suite_name="suite-x"), run_name="run-c")
    paths = discover_reports(tmp_path)
    index = build_dashboard_index(paths, base_dir=tmp_path)

    assert index["envs"] == ["tabletop"]
    assert index["suite_names"] == ["suite-x", "suite-y"]
    assert index["policy_labels"] == ["run-a", "run-b", "run-c"]


def test_build_dashboard_index_handles_none_env(tmp_path: Path) -> None:
    """Reports with ``suite_env=None`` (pre RFC-005) are not in `envs`."""
    _write_run(
        tmp_path,
        _grid_report(suite_env=None),
        run_name="legacy",
    )
    paths = discover_reports(tmp_path)
    index = build_dashboard_index(paths, base_dir=tmp_path)
    assert index["envs"] == []
    assert index["runs"][0]["env"] is None


def test_build_dashboard_index_malformed_json_includes_path(tmp_path: Path) -> None:
    bad = tmp_path / "broken" / "report.json"
    bad.parent.mkdir()
    bad.write_text("{not valid", encoding="utf-8")
    paths = discover_reports(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        build_dashboard_index(paths, base_dir=tmp_path)
    assert "broken/report.json" in str(exc_info.value)


def test_build_dashboard_index_invalid_report_includes_path(tmp_path: Path) -> None:
    bad = tmp_path / "wrong" / "report.json"
    bad.parent.mkdir()
    bad.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    paths = discover_reports(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        build_dashboard_index(paths, base_dir=tmp_path)
    assert "wrong/report.json" in str(exc_info.value)
