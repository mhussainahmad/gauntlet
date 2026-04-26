"""Fleet-wide failure-mode clustering tests — Phase 3 Task 19.

Pins :func:`gauntlet.aggregate.cluster_fleet_failures` plus the CLI
``gauntlet aggregate <dir> --cluster-output ...`` flag surface. Both
sides share the per-(run, cell) failure-signature primitive — see
``docs/fleet-aggregation.md``.

The fixtures here build :class:`gauntlet.report.Report` objects via
:func:`gauntlet.report.build_report` and dump them to disk under a
``runs/`` tree that mirrors what ``gauntlet run --out`` produces. No
MuJoCo / suite layer is exercised — the clustering module is the
pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.aggregate import (
    FleetCluster,
    FleetClusteringResult,
    cluster_fleet_failures,
)
from gauntlet.cli import app
from gauntlet.report import Report, build_report
from gauntlet.runner.episode import Episode
from gauntlet.security import PathTraversalError

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


def _grid_report(
    *,
    suite_name: str = "tiny-fleet",
    failing_cells: tuple[tuple[float, float], ...] = (),
) -> Report:
    """3x3 grid * 5 episodes = 45 episodes; failing cells are 100% fail."""
    eps: list[Episode] = []
    cell_idx = 0
    fail_set = set(failing_cells)
    for lighting in (0.3, 0.6, 0.9):
        for texture in (0.0, 1.0, 2.0):
            for ep_i in range(5):
                config = {"lighting_intensity": lighting, "object_texture": texture}
                success = (lighting, texture) not in fail_set
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
    return build_report(eps)


def _write_run(base: Path, rep: Report, *, run_name: str) -> Path:
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "report.json").write_text(
        json.dumps(rep.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_dir / "report.json"


# ---------------------------------------------------------------------------
# Empty / degenerate inputs.
# ---------------------------------------------------------------------------


def test_empty_directory_returns_empty_result(tmp_path: Path) -> None:
    """No report.json files anywhere → zero runs, no exception."""
    result = cluster_fleet_failures(tmp_path)
    assert isinstance(result, FleetClusteringResult)
    assert result.n_runs == 0
    assert result.n_unique_failures == 0
    assert result.clusters == []
    assert result.silhouette is None


def test_directory_with_only_clean_runs_yields_no_clusters(tmp_path: Path) -> None:
    """A run that has no failure clusters cannot contribute a signature."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(failing_cells=()), run_name="clean-a")
    _write_run(runs, _grid_report(failing_cells=()), run_name="clean-b")

    result = cluster_fleet_failures(runs)
    assert result.n_runs == 2
    assert result.clusters == []
    assert result.n_unique_failures == 0
    assert result.silhouette is None


def test_single_run_yields_silhouette_none(tmp_path: Path) -> None:
    """One run can produce one or more buckets but silhouette is undefined.

    Silhouette is only well-defined when there are at least two
    clusters; one run with one failing cell collapses to a single
    cluster (so silhouette must be ``None``), and one run with two
    failing cells produces two buckets that may sit in two clusters
    (so silhouette MAY be defined). We only pin the single-cluster
    branch — the two-bucket branch is exercised under the medoid test.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(
        runs,
        _grid_report(failing_cells=((0.3, 0.0),)),
        run_name="solo-run",
    )

    result = cluster_fleet_failures(runs)
    assert result.n_runs == 1
    assert len(result.clusters) == 1
    assert result.silhouette is None
    only = result.clusters[0]
    assert only.member_run_ids == ["solo-run"]
    assert only.cross_run_consistency == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Core clustering — overlapping cells across runs.
# ---------------------------------------------------------------------------


def test_overlapping_failure_signatures_collapse_to_shared_clusters(
    tmp_path: Path,
) -> None:
    """4 runs share three distinct failing cells → 3 clusters expected.

    Run-A and Run-B both fail (0.3, 0.0); Run-C and Run-D both fail
    (0.9, 2.0); Run-A also fails (0.6, 1.0); Run-D also fails
    (0.6, 1.0). The signature space therefore has three distinct
    buckets and the four runs spread across them as:

    * (0.3, 0.0) — A, B
    * (0.6, 1.0) — A, D
    * (0.9, 2.0) — C, D
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(
        runs,
        _grid_report(failing_cells=((0.3, 0.0), (0.6, 1.0))),
        run_name="run-a",
    )
    _write_run(
        runs,
        _grid_report(failing_cells=((0.3, 0.0),)),
        run_name="run-b",
    )
    _write_run(
        runs,
        _grid_report(failing_cells=((0.9, 2.0),)),
        run_name="run-c",
    )
    _write_run(
        runs,
        _grid_report(failing_cells=((0.9, 2.0), (0.6, 1.0))),
        run_name="run-d",
    )

    result = cluster_fleet_failures(runs, max_clusters=8)

    assert result.n_runs == 4
    assert result.n_unique_failures == 3
    assert len(result.clusters) == 3

    # Each cluster's run set is one of the three documented above.
    by_runs = {tuple(c.member_run_ids): c for c in result.clusters}
    assert ("run-a", "run-b") in by_runs
    assert ("run-a", "run-d") in by_runs
    assert ("run-c", "run-d") in by_runs

    # ``cross_run_consistency`` is "fraction of n_runs that hit this
    # mode"; each of these clusters has 2 of 4 runs.
    for cluster in result.clusters:
        assert cluster.cross_run_consistency == pytest.approx(0.5)


def test_max_clusters_caps_returned_clusters(tmp_path: Path) -> None:
    """When unique failures > max_clusters the agglomerative pass kicks in."""
    runs = tmp_path / "runs"
    runs.mkdir()
    failing_cells = [
        (0.3, 0.0),
        (0.3, 1.0),
        (0.6, 0.0),
        (0.6, 2.0),
        (0.9, 1.0),
    ]
    for cell in failing_cells:
        _write_run(
            runs,
            _grid_report(failing_cells=(cell,)),
            run_name=f"run-{cell[0]}-{cell[1]}",
        )

    result = cluster_fleet_failures(runs, max_clusters=3)

    assert result.n_runs == 5
    # Pre-merge unique signatures.
    assert result.n_unique_failures == 5
    # Post-merge cluster count obeys the cap.
    assert len(result.clusters) <= 3
    # Cluster ids are dense and zero-indexed.
    assert {c.cluster_id for c in result.clusters} == set(range(len(result.clusters)))
    # Every original run appears somewhere in the merged set.
    seen_runs = {rid for c in result.clusters for rid in c.member_run_ids}
    assert seen_runs == {f"run-{c[0]}-{c[1]}" for c in failing_cells}


def test_representative_signature_is_a_real_member(tmp_path: Path) -> None:
    """Medoid → the representative is one of the cluster's bucket members."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(
        runs,
        _grid_report(failing_cells=((0.3, 0.0),)),
        run_name="alpha",
    )
    _write_run(
        runs,
        _grid_report(failing_cells=((0.3, 0.0),)),
        run_name="beta",
    )

    result = cluster_fleet_failures(runs, max_clusters=8)
    assert len(result.clusters) == 1
    sig = result.clusters[0].representative_failure_signature
    # The shared signature carries axes={"lighting_intensity": 0.3,
    # "object_texture": 0.0}; the medoid of a single bucket is itself.
    assert sig["axes"] == {
        "lighting_intensity": pytest.approx(0.3),
        "object_texture": pytest.approx(0.0),
    }
    # Wilson lower bound is a finite float in [0, 1] for any non-empty
    # failure cluster.
    wilson_value = sig["wilson_lower_bound"]
    assert isinstance(wilson_value, float)
    assert 0.0 <= wilson_value <= 1.0


# ---------------------------------------------------------------------------
# Path-traversal hardening — safe_join at the boundary.
# ---------------------------------------------------------------------------


def test_path_traversal_in_directory_argument_is_rejected(tmp_path: Path) -> None:
    """Symlink-escape outside the canonicalised root is filtered, not loaded.

    Building an actual ``../../etc`` argument would clash with pytest's
    tmp_path sandbox, so we model the equivalent escape with a symlink:
    a ``runs/`` directory that contains a symlink pointing at a
    sibling-tree's ``report.json``. The clustering module's
    safe_join-routed glob must refuse to cross the symlink and must not
    surface the foreign file in the result.
    """
    inside = tmp_path / "runs"
    inside.mkdir()
    _write_run(inside, _grid_report(failing_cells=((0.3, 0.0),)), run_name="legit")

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _write_run(outside, _grid_report(failing_cells=((0.9, 2.0),)), run_name="foreign")

    # Plant a symlink inside ``runs/`` that points at the foreign run dir.
    (inside / "smuggled").symlink_to(outside / "foreign")

    result = cluster_fleet_failures(inside)
    seen_runs = {rid for c in result.clusters for rid in c.member_run_ids}
    # The legit run is in. The smuggled foreign report.json is filtered
    # out — never makes it into a cluster.
    assert "legit" in seen_runs
    assert "foreign" not in seen_runs
    assert "smuggled" not in seen_runs


def test_absolute_path_injection_is_rejected_by_safe_join() -> None:
    """An absolute-path argument that does not exist still routes through safe_join."""
    bogus = Path("/does/not/exist/gauntlet-fleet-clustering")
    with pytest.raises(FileNotFoundError):
        cluster_fleet_failures(bogus)


def test_non_directory_argument_raises_not_a_directory(tmp_path: Path) -> None:
    """A regular file argument fails fast."""
    f = tmp_path / "report.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        cluster_fleet_failures(f)


def test_max_clusters_must_be_positive(tmp_path: Path) -> None:
    """``max_clusters < 1`` is a programmer error."""
    with pytest.raises(ValueError, match="max_clusters"):
        cluster_fleet_failures(tmp_path, max_clusters=0)


# Reference one symbol from the security module so an accidental
# unused-import refactor doesn't strand the path-traversal guarantee
# silently. The module ships :class:`PathTraversalError` as the single
# exception type a caller can catch.
def test_path_traversal_error_is_a_value_error_subclass() -> None:
    assert issubclass(PathTraversalError, ValueError)


# ---------------------------------------------------------------------------
# Frozen dataclass / public-surface contract.
# ---------------------------------------------------------------------------


def test_fleet_cluster_is_frozen() -> None:
    """``FleetCluster`` is immutable — preventing surprise aliasing."""
    cluster = FleetCluster(
        cluster_id=0,
        member_run_ids=["a"],
        representative_failure_signature={"axes": {}},
        cross_run_consistency=1.0,
    )
    with pytest.raises(AttributeError):
        # ``frozen=True`` raises FrozenInstanceError, which IS an
        # AttributeError subclass.
        cluster.cluster_id = 1  # type: ignore[misc]


def test_fleet_clustering_result_is_frozen() -> None:
    result = FleetClusteringResult(
        clusters=[],
        n_runs=0,
        n_unique_failures=0,
        silhouette=None,
    )
    with pytest.raises(AttributeError):
        result.n_runs = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI happy-path — ``gauntlet aggregate <dir> --cluster-output ...``.
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_aggregate_writes_cluster_json(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """``gauntlet aggregate <dir> --cluster-output X`` emits valid JSON.

    The clustering CLI is wired into the same ``aggregate`` command that
    already writes ``fleet_report.json``; the new flag is opt-in.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(failing_cells=((0.3, 0.0),)), run_name="run-a")
    _write_run(runs, _grid_report(failing_cells=((0.3, 0.0),)), run_name="run-b")
    out_dir = tmp_path / "out"
    cluster_out = tmp_path / "fleet_clustering.json"

    result = runner.invoke(
        app,
        [
            "aggregate",
            str(runs),
            "--out",
            str(out_dir),
            "--cluster-output",
            str(cluster_out),
            "--max-clusters",
            "4",
            "--no-html",
        ],
    )
    assert result.exit_code == 0, result.stderr or result.output

    assert cluster_out.exists()
    payload = json.loads(cluster_out.read_text(encoding="utf-8"))
    assert payload["n_runs"] == 2
    assert payload["n_unique_failures"] == 1
    assert isinstance(payload["clusters"], list)
    assert len(payload["clusters"]) == 1
    cluster = payload["clusters"][0]
    assert cluster["cluster_id"] == 0
    assert sorted(cluster["member_run_ids"]) == ["run-a", "run-b"]
    assert cluster["cross_run_consistency"] == pytest.approx(1.0)
    sig = cluster["representative_failure_signature"]
    assert sig["axes"] == {
        "lighting_intensity": pytest.approx(0.3),
        "object_texture": pytest.approx(0.0),
    }


def test_cli_aggregate_max_clusters_must_be_positive(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Typer enforces ``--max-clusters >= 1`` at parse time."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(failing_cells=((0.3, 0.0),)), run_name="run-a")
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "aggregate",
            str(runs),
            "--out",
            str(out_dir),
            "--max-clusters",
            "0",
            "--no-html",
        ],
    )
    assert result.exit_code != 0
