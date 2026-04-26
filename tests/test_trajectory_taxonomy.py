"""Tests for B-17 failure-mode taxonomy via trajectory clustering.

Covers the public surface of :mod:`gauntlet.report.trajectory_taxonomy`:

* Three synthetic shape families produce three clusters with the
  euclidean fallback distance (deterministic, dep-free).
* Exemplar selection picks the medoid (the member whose summed intra-
  cluster distance to the others is smallest).
* Silhouette score reads above 0.5 on well-separated synthetic data.
* Successful episodes are filtered out; missing NPZs degrade
  gracefully.
* Requesting ``distance="dtw"`` without the ``[trajectory-taxonomy]``
  extra raises a clean :class:`TaxonomyError` (subclass of ``ImportError``)
  with the install hint, when the extra is not installed.
* The public dataclasses carry NO prose-label field — the anti-feature
  framing in ``docs/backlog.md`` B-17 is asserted via dataclass-field
  introspection so a future "helpful" prose-label addition trips CI.

Pure numpy + pytest. No Runner, no torch.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.report import Report, build_report, render_html
from gauntlet.report.trajectory_taxonomy import (
    TaxonomyError,
    TaxonomyResult,
    TrajectoryCluster,
    cluster_failed_trajectories,
    episode_id,
    trajectory_path_for_episode,
)
from gauntlet.runner.episode import Episode

_SUITE = "test-suite-taxonomy"

# Each family is a (T, 7) action stream signature; clustering should
# group all episodes within a family together.
_FAMILY_A_SEED = 11
_FAMILY_B_SEED = 22
_FAMILY_C_SEED = 33

# DTW availability — used to skip the DTW-specific tests when the
# extra isn't installed and to drive the "graceful import error"
# negative test.
_DTW_AVAILABLE = importlib.util.find_spec("dtw") is not None


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
) -> Episode:
    return Episode(
        suite_name=_SUITE,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=cell_index * 100 + episode_index,
        perturbation_config={},
        success=success,
        terminated=success,
        truncated=False,
        step_count=20,
        total_reward=1.0 if success else 0.0,
    )


def _shape_a(rng: np.random.Generator, n_steps: int = 20) -> NDArray[np.float64]:
    """Linear ramp on the first action dim, zero elsewhere — `pick-and-lift` shape."""
    arr = np.zeros((n_steps, 7), dtype=np.float64)
    arr[:, 0] = np.linspace(0.0, 1.0, n_steps)
    arr += rng.normal(scale=0.02, size=arr.shape)
    return arr


def _shape_b(rng: np.random.Generator, n_steps: int = 20) -> NDArray[np.float64]:
    """Sinusoidal sweep on the second action dim — `oscillate` shape."""
    arr = np.zeros((n_steps, 7), dtype=np.float64)
    arr[:, 1] = np.sin(np.linspace(0.0, 4 * np.pi, n_steps))
    arr += rng.normal(scale=0.02, size=arr.shape)
    return arr


def _shape_c(rng: np.random.Generator, n_steps: int = 20) -> NDArray[np.float64]:
    """Step on the gripper dim, zero elsewhere — `clamp-and-hold` shape."""
    arr = np.zeros((n_steps, 7), dtype=np.float64)
    arr[n_steps // 2 :, 6] = 1.0
    arr += rng.normal(scale=0.02, size=arr.shape)
    return arr


def _write_npz(
    trajectory_dir: Path,
    *,
    cell_index: int,
    episode_index: int,
    actions: NDArray[np.float64],
) -> None:
    """Mimic :func:`gauntlet.runner.worker.write_trajectory_npz`'s on-disk shape.

    We only need the ``action`` array — the loader ignores everything
    else — but we write the seed / index scalars too so the file is
    indistinguishable from a real worker dump.
    """
    path = trajectory_dir / f"cell_{cell_index:04d}_ep_{episode_index:04d}.npz"
    np.savez_compressed(
        path,
        action=actions,
        seed=np.asarray(cell_index * 100 + episode_index, dtype=np.int64),
        cell_index=np.asarray(cell_index, dtype=np.int64),
        episode_index=np.asarray(episode_index, dtype=np.int64),
        obs_state=np.zeros((actions.shape[0], 3), dtype=np.float64),
    )


def _seed_dataset(
    trajectory_dir: Path,
    n_per_family: int = 4,
) -> list[Episode]:
    """Write ``3 * n_per_family`` failed episode NPZs in three shape families.

    Returns the corresponding :class:`Episode` list (all ``success=False``).
    Cell indices are assigned per-family (0, 1, 2) so the
    ``cell_NNNN_ep_NNNN`` ids cluster lexicographically by family — but
    the clustering must NOT depend on that ordering (the test asserts
    the *episode-id grouping* is the family grouping, regardless of
    initial input order).
    """
    episodes: list[Episode] = []
    for family_idx, (seed, shape_fn) in enumerate(
        [
            (_FAMILY_A_SEED, _shape_a),
            (_FAMILY_B_SEED, _shape_b),
            (_FAMILY_C_SEED, _shape_c),
        ]
    ):
        rng = np.random.default_rng(seed)
        for ep_idx in range(n_per_family):
            actions = shape_fn(rng)
            _write_npz(
                trajectory_dir,
                cell_index=family_idx,
                episode_index=ep_idx,
                actions=actions,
            )
            episodes.append(
                _ep(cell_index=family_idx, episode_index=ep_idx, success=False),
            )
    return episodes


# ── Public-API surface contracts ─────────────────────────────────────


def test_dataclasses_have_no_prose_label_fields() -> None:
    """B-17 anti-feature: NO prose-label field on the public surface.

    The backlog explicitly says "Auto-labels are LLM-bait — easy to
    ship 'knocked target off table' labels that are confidently wrong.
    Better to label by exemplar episode index than by prose." This
    test enforces that the dataclasses carry no string field whose
    name suggests a human-readable description, so a future
    "helpful" addition trips CI before review.
    """
    cluster_field_names = {f.name for f in dataclasses.fields(TrajectoryCluster)}
    result_field_names = {f.name for f in dataclasses.fields(TaxonomyResult)}
    forbidden_substrings = ("label", "description", "summary", "name", "text", "prose")
    for fname in cluster_field_names | result_field_names:
        # ``cluster_id`` is the integer ordinal, which contains "id" but
        # is the exemplar-index handle by design — explicitly allowed.
        if fname == "cluster_id":
            continue
        # ``exemplar_episode_id`` and ``member_episode_ids`` carry the
        # ``cell_NNNN_ep_NNNN`` machine ids the backlog endorses.
        if fname in {"exemplar_episode_id", "member_episode_ids"}:
            continue
        lowered = fname.lower()
        assert not any(sub in lowered for sub in forbidden_substrings), (
            f"forbidden prose-label field: {fname!r}"
        )


def test_episode_id_matches_npz_filename_format() -> None:
    """``episode_id`` produces the same string as the NPZ filename stem."""
    assert episode_id(0, 0) == "cell_0000_ep_0000"
    assert episode_id(7, 42) == "cell_0007_ep_0042"


def test_trajectory_path_for_episode_matches_runner_format(tmp_path: Path) -> None:
    """The path mirrors :func:`gauntlet.runner.worker.trajectory_path_for`."""
    ep = _ep(cell_index=3, episode_index=5, success=False)
    path = trajectory_path_for_episode(tmp_path, ep)
    assert path == tmp_path / "cell_0003_ep_0005.npz"


# ── Clustering behaviour (euclidean — dep-free) ──────────────────────


def test_three_shape_families_yield_three_clusters(tmp_path: Path) -> None:
    """Synthetic A/B/C trajectories cluster into exactly three groups.

    The euclidean fallback is sufficient on the well-separated
    synthetic data — this is the deterministic, dep-free contract.
    """
    episodes = _seed_dataset(tmp_path, n_per_family=4)

    result = cluster_failed_trajectories(
        tmp_path,
        episodes,
        distance="euclidean",
    )

    assert isinstance(result, TaxonomyResult)
    assert len(result.clusters) == 3
    assert sum(len(c.member_episode_ids) for c in result.clusters) == 12

    # Each cluster's members must come from a single family (cell index).
    for cluster in result.clusters:
        cells = {member.split("_")[1] for member in cluster.member_episode_ids}
        assert len(cells) == 1, (
            f"cluster {cluster.cluster_id} mixed families: {cluster.member_episode_ids}"
        )


def test_silhouette_score_above_0_5_on_well_separated_data(tmp_path: Path) -> None:
    """Well-separated synthetic shapes give silhouette > 0.5."""
    episodes = _seed_dataset(tmp_path, n_per_family=4)
    result = cluster_failed_trajectories(
        tmp_path,
        episodes,
        distance="euclidean",
    )
    assert result.silhouette is not None
    assert result.silhouette > 0.5, (
        f"silhouette score too low for well-separated data: {result.silhouette!r}"
    )


def test_exemplar_is_the_medoid(tmp_path: Path) -> None:
    """Exemplar episode is the member with the smallest sum of intra-cluster distances.

    We hand-craft a three-member cluster where one member is exactly
    the average of the other two — that one is the medoid by
    construction and must be picked as the exemplar.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(15, 7))
    # Member 0 is offset by +1 on dim 0; member 2 is offset by -1 on
    # dim 0; member 1 is the mean of the two — the medoid.
    member_0 = base + np.eye(15, 7)[0:15] * 0.0
    member_0[:, 0] += 1.0
    member_2 = base.copy()
    member_2[:, 0] -= 1.0
    member_1 = (member_0 + member_2) / 2.0

    _write_npz(tmp_path, cell_index=0, episode_index=0, actions=member_0)
    _write_npz(tmp_path, cell_index=0, episode_index=1, actions=member_1)
    _write_npz(tmp_path, cell_index=0, episode_index=2, actions=member_2)

    episodes = [_ep(cell_index=0, episode_index=i, success=False) for i in range(3)]
    result = cluster_failed_trajectories(
        tmp_path,
        episodes,
        n_clusters=1,
        distance="euclidean",
    )
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.exemplar_episode_id == "cell_0000_ep_0001", (
        f"medoid must be member 1; got {cluster.exemplar_episode_id!r}"
    )


def test_successful_episodes_are_filtered_out(tmp_path: Path) -> None:
    """``cluster_failed_trajectories`` ignores successful episodes by name."""
    rng = np.random.default_rng(0)
    actions = _shape_a(rng)
    _write_npz(tmp_path, cell_index=0, episode_index=0, actions=actions)
    _write_npz(tmp_path, cell_index=0, episode_index=1, actions=actions)

    episodes = [
        _ep(cell_index=0, episode_index=0, success=True),
        _ep(cell_index=0, episode_index=1, success=False),
    ]
    result = cluster_failed_trajectories(
        tmp_path,
        episodes,
        n_clusters=1,
        distance="euclidean",
    )
    assert len(result.clusters) == 1
    assert result.clusters[0].member_episode_ids == ("cell_0000_ep_0001",)


def test_missing_npz_degrades_gracefully(tmp_path: Path) -> None:
    """Episodes whose NPZ is absent on disk are silently dropped."""
    rng = np.random.default_rng(0)
    actions = _shape_a(rng)
    # Only write episode 0; episode 1's NPZ is intentionally absent.
    _write_npz(tmp_path, cell_index=0, episode_index=0, actions=actions)

    episodes = [
        _ep(cell_index=0, episode_index=0, success=False),
        _ep(cell_index=0, episode_index=1, success=False),
    ]
    result = cluster_failed_trajectories(
        tmp_path,
        episodes,
        distance="euclidean",
    )
    # Only the available episode survives.
    assert len(result.clusters) == 1
    assert result.clusters[0].member_episode_ids == ("cell_0000_ep_0000",)


def test_no_failed_episodes_returns_empty_result(tmp_path: Path) -> None:
    """All-success input gives an empty cluster list, undefined silhouette."""
    rng = np.random.default_rng(0)
    actions = _shape_a(rng)
    _write_npz(tmp_path, cell_index=0, episode_index=0, actions=actions)

    episodes = [_ep(cell_index=0, episode_index=0, success=True)]
    result = cluster_failed_trajectories(tmp_path, episodes, distance="euclidean")
    assert result.clusters == ()
    assert result.silhouette is None


def test_clusters_ordered_by_descending_member_count(tmp_path: Path) -> None:
    """The biggest cluster appears first — readable report ordering."""
    # Family A: 5 members; Family B: 2 members.
    for ep_idx in range(5):
        _write_npz(
            tmp_path,
            cell_index=0,
            episode_index=ep_idx,
            actions=_shape_a(np.random.default_rng(ep_idx)),
        )
    for ep_idx in range(2):
        _write_npz(
            tmp_path,
            cell_index=1,
            episode_index=ep_idx,
            actions=_shape_b(np.random.default_rng(100 + ep_idx)),
        )
    episodes = [_ep(cell_index=0, episode_index=i, success=False) for i in range(5)]
    episodes += [_ep(cell_index=1, episode_index=i, success=False) for i in range(2)]
    result = cluster_failed_trajectories(
        tmp_path,
        episodes,
        n_clusters=2,
        distance="euclidean",
    )
    assert len(result.clusters) == 2
    assert len(result.clusters[0].member_episode_ids) == 5
    assert len(result.clusters[1].member_episode_ids) == 2
    # Renumbered cluster ids match the rendered order.
    assert result.clusters[0].cluster_id == 0
    assert result.clusters[1].cluster_id == 1


# ── DTW negative path: extra-not-installed → TaxonomyError ───────────


@pytest.mark.skipif(_DTW_AVAILABLE, reason="dtw-python is installed")
def test_dtw_without_extra_raises_taxonomy_error(tmp_path: Path) -> None:
    """Asking for DTW without the extra raises :class:`TaxonomyError`.

    Skipped when the extra IS installed — only meaningful on the
    default-extras-free venv.
    """
    rng = np.random.default_rng(0)
    _write_npz(
        tmp_path,
        cell_index=0,
        episode_index=0,
        actions=_shape_a(rng),
    )
    _write_npz(
        tmp_path,
        cell_index=0,
        episode_index=1,
        actions=_shape_a(rng),
    )
    episodes = [_ep(cell_index=0, episode_index=i, success=False) for i in range(2)]

    with pytest.raises(TaxonomyError) as excinfo:
        cluster_failed_trajectories(
            tmp_path,
            episodes,
            n_clusters=1,
            distance="dtw",
        )
    # Must be an ImportError-derived class so callers can catch generic.
    assert isinstance(excinfo.value, ImportError)
    # Install hint is required so the user knows how to fix it.
    assert "trajectory-taxonomy" in str(excinfo.value)


@pytest.mark.skipif(not _DTW_AVAILABLE, reason="dtw-python not installed")
def test_dtw_path_clusters_three_families(tmp_path: Path) -> None:
    """DTW path produces the same three-cluster result as euclidean.

    Skipped when the extra is NOT installed — the default-extras-free
    venv exercises the negative test above instead.
    """
    episodes = _seed_dataset(tmp_path, n_per_family=4)
    result = cluster_failed_trajectories(
        tmp_path,
        episodes,
        distance="dtw",
    )
    assert len(result.clusters) == 3


# ── HTML render integration ───────────────────────────────────────────


def _build_minimal_report(episodes: list[Episode]) -> Report:
    """Build a :class:`Report` from a list of :class:`Episode`."""
    return build_report(episodes)


def test_render_html_without_trajectory_dir_omits_taxonomy_section(tmp_path: Path) -> None:
    """Default `render_html(report)` (no second arg) renders no taxonomy section.

    Backward-compat contract: existing callers see no taxonomy
    section at all (it is absent entirely).
    """
    episodes = _seed_dataset(tmp_path, n_per_family=2)
    report = _build_minimal_report(episodes)
    html = render_html(report)
    assert "Failure-mode taxonomy" not in html
    assert "Failure-Mode Taxonomy" not in html


def test_render_html_with_trajectory_dir_renders_taxonomy_table(tmp_path: Path) -> None:
    """When `trajectory_dir` + `episodes` are passed, the taxonomy table renders.

    The exemplar episode id (a `cell_NNNN_ep_NNNN` string) appears in
    the output — confirming the medoid-by-id labelling reached the
    rendered HTML.
    """
    import re

    episodes = _seed_dataset(tmp_path, n_per_family=3)
    report = _build_minimal_report(episodes)
    html = render_html(report, trajectory_dir=tmp_path, episodes=episodes)
    assert "Failure-mode taxonomy" in html
    # All exemplar ids follow the `cell_NNNN_ep_NNNN` pattern; at
    # least one such id must appear in the rendered HTML.
    assert re.search(r"cell_\d{4}_ep_\d{4}", html), (
        "expected at least one cell_NNNN_ep_NNNN exemplar id in HTML"
    )


def test_render_html_missing_trajectory_dir_renders_unavailable_notice(tmp_path: Path) -> None:
    """A missing trajectory dir collapses to the one-line notice.

    Per the B-17 spec: "if no trajectories were dumped, render a one-
    line 'Trajectory taxonomy unavailable — re-run with `--trajectory-
    dir`' notice rather than crashing."
    """
    missing = tmp_path / "no-such-dir"
    # No NPZs anywhere; report has at least one failed episode.
    episodes = [
        _ep(cell_index=0, episode_index=0, success=False),
        _ep(cell_index=0, episode_index=1, success=True),
    ]
    report = _build_minimal_report(episodes)
    html = render_html(report, trajectory_dir=missing, episodes=episodes)
    assert "Trajectory taxonomy unavailable" in html
    assert "--trajectory-dir" in html
    # Notice path: NO table column headers from the taxonomy table.
    assert "Mean Intra-Cluster Distance" not in html


def test_render_html_no_failures_renders_unavailable_notice(tmp_path: Path) -> None:
    """All-success run with a trajectory_dir falls back to the notice."""
    rng = np.random.default_rng(0)
    actions = _shape_a(rng)
    _write_npz(tmp_path, cell_index=0, episode_index=0, actions=actions)
    episodes = [_ep(cell_index=0, episode_index=0, success=True)]
    report = _build_minimal_report(episodes)
    html = render_html(report, trajectory_dir=tmp_path, episodes=episodes)
    assert "Trajectory taxonomy unavailable" in html
    assert "no failures to cluster" in html.lower()
