"""Failure-mode taxonomy via trajectory clustering (B-17).

Group *failed* episodes by trajectory similarity so the report surfaces
"these failures look the same regardless of axis-config" — orthogonal
to the existing axis-config failure clusters in
:mod:`gauntlet.report.analyze`. The two answer different questions:

* axis-config clusters: *which* perturbation values failed.
* trajectory clusters: *how* the rollouts unfolded (in action space).

Refs: "Unsupervised Discovery of Failure Taxonomies from Deployment
Logs" (arXiv 2506.06570), RoboFail / RoboMD (arXiv 2412.02818).

──────────────────────────────────────────────────────────────────────
Anti-feature note (per ``docs/backlog.md`` B-17)
──────────────────────────────────────────────────────────────────────

The backlog calls out that auto-generated prose labels ("knocked target
off table", "dropped before lift") are LLM-bait — easy to ship
confidently-wrong descriptions. The public surface here therefore
labels each cluster by an **exemplar episode index** (the medoid's
``cell_NNNN_ep_NNNN`` identifier), never by prose. The dataclasses
intentionally have NO ``cluster_label_text`` / ``description`` field;
the test suite asserts this so a future "helpful" addition that
introduces a prose label trips CI.

──────────────────────────────────────────────────────────────────────
Trajectory representation
──────────────────────────────────────────────────────────────────────

Each per-episode NPZ produced by :func:`gauntlet.runner.worker.write_trajectory_npz`
carries an ``action`` array of shape ``(T, 7)`` and a varying set of
``obs_<key>`` arrays whose key set depends on the backend. We cluster
on the ``action`` array because:

* ``action`` is guaranteed present across every backend.
* ``action`` is a fixed-width 7-D space — no per-backend obs-key
  divergence to reconcile.
* "How the policy moved" is exactly the failure-shape question the
  backlog entry phrases ("dropped before lift", "knocked off table"
  are both action-shape signatures).

──────────────────────────────────────────────────────────────────────
Distance metric
──────────────────────────────────────────────────────────────────────

* ``"dtw"`` (default, requires the ``[trajectory-taxonomy]`` extra):
  Dynamic time warping handles variable-length trajectories natively.
  Backed by ``dtw-python`` — a single transitive dep (numpy), much
  lighter than ``tslearn``. Lazy-imported inside the DTW branch so
  the default install path never touches it.
* ``"euclidean"``: fallback that does not need the extra. Trajectories
  are truncated to the shortest length in the failed set, then a
  flat L2 over the (T, 7) array. Documented as the cruder option.

──────────────────────────────────────────────────────────────────────
Clustering
──────────────────────────────────────────────────────────────────────

We compute a full pairwise distance matrix (N is the number of
*failed* episodes — small in practice; a 1440-episode suite at 30%
failure rate gives ~430 episodes, comfortably under the O(N^2) ceiling)
and run a deterministic agglomerative average-linkage cluster on it.
No scipy / sklearn dependency — implemented inline.

When ``n_clusters`` is ``None`` we silhouette-search over
``k ∈ [2, min(8, N-1)]`` and pick the k with the highest silhouette
score on the precomputed distance matrix (Rousseeuw 1987). The search
range is bounded so a tiny failure set (e.g. 3 episodes) does not
trigger pathological partitions.

──────────────────────────────────────────────────────────────────────
Exemplar selection
──────────────────────────────────────────────────────────────────────

Each cluster's exemplar is the **medoid** — the member with the
smallest total intra-cluster distance to the other members. Ties are
broken by the canonical ``cell_NNNN_ep_NNNN`` lexicographic order so
the result is deterministic across runs. This is the user-facing
identifier the report renders as the cluster's name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from gauntlet.runner.episode import Episode

__all__ = [
    "TaxonomyError",
    "TaxonomyResult",
    "TrajectoryCluster",
    "cluster_failed_trajectories",
    "episode_id",
    "trajectory_path_for_episode",
]


Distance = Literal["dtw", "euclidean"]


class TaxonomyError(ImportError):
    """Raised when the [trajectory-taxonomy] extra is requested but absent.

    Subclass of :class:`ImportError` so callers can ``except ImportError``
    and fall back to the euclidean path. The message points at the exact
    ``pip install`` invocation that fixes it.
    """


@dataclass(frozen=True, slots=True)
class TrajectoryCluster:
    """One taxonomy cluster of failed trajectories.

    Attributes:
        cluster_id: Zero-based cluster ordinal in the result.
        member_episode_ids: ``cell_NNNN_ep_NNNN`` identifiers for every
            member, sorted lexicographically for determinism.
        exemplar_episode_id: The medoid identifier — the member whose
            summed intra-cluster distance to the others is smallest.
            Ties broken lexicographically.
        intra_cluster_distance: Mean pairwise distance among the members.
            ``0.0`` for a singleton.
    """

    cluster_id: int
    member_episode_ids: tuple[str, ...]
    exemplar_episode_id: str
    intra_cluster_distance: float


@dataclass(frozen=True, slots=True)
class TaxonomyResult:
    """Result of :func:`cluster_failed_trajectories`.

    Attributes:
        clusters: One :class:`TrajectoryCluster` per group, ordered by
            descending member count then by ``exemplar_episode_id``.
        silhouette: Mean silhouette score over the chosen partition.
            ``None`` when the score is undefined (one cluster, or fewer
            than two failed episodes total).
    """

    clusters: tuple[TrajectoryCluster, ...]
    silhouette: float | None


def episode_id(cell_index: int, episode_index: int) -> str:
    """Canonical episode identifier matching the trajectory NPZ filename.

    Mirrors :func:`gauntlet.runner.worker.trajectory_path_for` so the
    string the report renders is the same one the user grep's against
    on disk.
    """
    return f"cell_{cell_index:04d}_ep_{episode_index:04d}"


def trajectory_path_for_episode(trajectory_dir: Path, ep: Episode) -> Path:
    """Return the canonical NPZ path for *ep*'s trajectory.

    Matches :func:`gauntlet.runner.worker.trajectory_path_for` exactly.
    """
    return trajectory_dir / f"{episode_id(ep.cell_index, ep.episode_index)}.npz"


def _load_action_trajectory(npz_path: Path) -> NDArray[np.float64]:
    """Load the ``action`` array from one trajectory NPZ.

    Returns the action array as ``(T, 7)`` float64. Other keys
    (``obs_*``, ``seed``, indices) are ignored — only the action stream
    is consumed by the clustering metric (see module docstring).
    """
    with np.load(npz_path) as npz:
        actions = np.asarray(npz["action"], dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(
            f"trajectory NPZ {npz_path.name}: expected (T, 7) action array, "
            f"got shape {actions.shape}"
        )
    return actions


def _euclidean_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """L2 distance between two (T, 7) action arrays.

    Different lengths are reconciled by truncation to ``min(T_a, T_b)``;
    extra timesteps on the longer trajectory are dropped. This is the
    cruder fallback when DTW is unavailable.
    """
    t = min(a.shape[0], b.shape[0])
    if t == 0:
        return 0.0
    diff = a[:t] - b[:t]
    return float(np.sqrt(np.sum(diff * diff)))


def _dtw_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """DTW distance between two (T, 7) action arrays.

    Lazy import of :mod:`dtw` (the ``dtw-python`` package). Raises
    :class:`TaxonomyError` with the install hint when the extra is
    missing — callers ``except ImportError`` to fall back.
    """
    try:
        from dtw import dtw as _dtw_fn
    except ImportError as exc:
        raise TaxonomyError(
            "DTW distance requires the [trajectory-taxonomy] extra. "
            "Install with: pip install 'gauntlet[trajectory-taxonomy]'"
        ) from exc
    # ``dtw_python``'s ``dtw`` returns a result object whose ``.distance``
    # attribute is the alignment cost. ``dist_method='euclidean'`` matches
    # the local-step metric used in the original Sakoe-Chiba formulation.
    # ``keep_internals=False`` avoids retaining the warping matrix —
    # we only need the scalar.
    result = _dtw_fn(a, b, dist_method="euclidean", keep_internals=False)
    return float(cast("float", result.distance))


def _pairwise_distances(
    trajectories: list[NDArray[np.float64]],
    distance: Distance,
) -> NDArray[np.float64]:
    """Build an (N, N) symmetric distance matrix.

    Diagonal is exactly zero. Distance is symmetric by construction —
    we compute the upper triangle only and mirror.
    """
    n = len(trajectories)
    matrix = np.zeros((n, n), dtype=np.float64)
    compute = _dtw_distance if distance == "dtw" else _euclidean_distance
    for i in range(n):
        for j in range(i + 1, n):
            d = compute(trajectories[i], trajectories[j])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def _agglomerative_clusters(
    distances: NDArray[np.float64],
    n_clusters: int,
) -> list[list[int]]:
    """Average-linkage agglomerative clustering on a distance matrix.

    Deterministic: ties on the merge step are broken by
    ``(min_index_a, min_index_b)`` lexicographic order. Returns a list
    of clusters (each a list of original row indices) of length
    exactly ``n_clusters`` — or fewer when the input has fewer rows.
    """
    n = distances.shape[0]
    if n == 0:
        return []
    if n_clusters >= n:
        return [[i] for i in range(n)]

    # Each "active" cluster is a list of original indices. We track
    # active clusters by their canonical sorted member tuple so the
    # tie-break is deterministic.
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > n_clusters:
        # Find the closest pair under average linkage.
        best_d = float("inf")
        best_pair = (0, 1)
        n_active = len(clusters)
        for a in range(n_active):
            for b in range(a + 1, n_active):
                # Average distance between members of cluster a and b.
                total = 0.0
                count = 0
                for i in clusters[a]:
                    for j in clusters[b]:
                        total += float(distances[i, j])
                        count += 1
                avg = total / count if count else 0.0
                if avg < best_d:
                    best_d = avg
                    best_pair = (a, b)
        a, b = best_pair
        merged = sorted(clusters[a] + clusters[b])
        # Remove higher index first so the lower index stays valid.
        del clusters[b]
        del clusters[a]
        clusters.append(merged)
        # Keep clusters sorted by their first member for deterministic
        # iteration order on subsequent rounds.
        clusters.sort(key=lambda members: members[0])
    return clusters


def _silhouette_score(
    distances: NDArray[np.float64],
    labels: list[int],
) -> float | None:
    """Mean silhouette score (Rousseeuw 1987) on a precomputed distance matrix.

    Returns ``None`` when fewer than two clusters exist or the dataset
    has fewer than two points (the score is undefined). Silhouette
    is in ``[-1, 1]`` — higher is better.
    """
    n = len(labels)
    unique = set(labels)
    if n < 2 or len(unique) < 2:
        return None
    label_arr = np.asarray(labels, dtype=np.int64)
    silhouettes: list[float] = []
    for i in range(n):
        own = label_arr == label_arr[i]
        own[i] = False
        if not own.any():
            # Singleton cluster — by Rousseeuw's convention contributes 0.
            silhouettes.append(0.0)
            continue
        a_i = float(np.mean(distances[i, own]))
        b_i = float("inf")
        for k in unique:
            if k == int(label_arr[i]):
                continue
            other = label_arr == k
            mean_d = float(np.mean(distances[i, other]))
            if mean_d < b_i:
                b_i = mean_d
        denom = max(a_i, b_i)
        silhouettes.append(0.0 if denom == 0.0 else (b_i - a_i) / denom)
    return float(np.mean(silhouettes))


def _labels_from_clusters(clusters: list[list[int]], n: int) -> list[int]:
    """Flatten ``[[idx, ...], ...]`` to a per-row label list."""
    labels = [0] * n
    for cluster_id, members in enumerate(clusters):
        for member in members:
            labels[member] = cluster_id
    return labels


def _select_n_clusters(
    distances: NDArray[np.float64],
    n_failed: int,
) -> tuple[int, list[list[int]], float | None]:
    """Silhouette-search ``k ∈ [2, min(8, n_failed - 1)]``.

    Returns ``(best_k, best_clusters, best_silhouette)``. When
    ``n_failed < 2`` (no meaningful partition exists) returns the
    trivial all-singletons partition with ``silhouette=None``.
    """
    if n_failed < 2:
        return 1, [[i] for i in range(n_failed)], None
    k_max = min(8, n_failed - 1)
    if k_max < 2:
        # Only one episode pair — trivially one cluster of two.
        return 1, [list(range(n_failed))], None
    best_k = 2
    best_score = float("-inf")
    best_clusters: list[list[int]] = []
    for k in range(2, k_max + 1):
        clusters = _agglomerative_clusters(distances, k)
        labels = _labels_from_clusters(clusters, n_failed)
        score = _silhouette_score(distances, labels)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_k = k
            best_clusters = clusters
    if not best_clusters:
        # No silhouette was computable — fall back to k=2.
        best_clusters = _agglomerative_clusters(distances, 2)
        return 2, best_clusters, None
    return best_k, best_clusters, float(best_score)


def _cluster_to_record(
    cluster_id: int,
    members: list[int],
    distances: NDArray[np.float64],
    ids: list[str],
) -> TrajectoryCluster:
    """Build a :class:`TrajectoryCluster` from row indices.

    Picks the medoid (smallest total intra-cluster distance) as the
    exemplar; ties broken by lexicographic ``episode_id`` so the
    result is deterministic across reruns.
    """
    members_sorted = sorted(members, key=lambda i: ids[i])
    if len(members_sorted) == 1:
        only = members_sorted[0]
        return TrajectoryCluster(
            cluster_id=cluster_id,
            member_episode_ids=(ids[only],),
            exemplar_episode_id=ids[only],
            intra_cluster_distance=0.0,
        )
    # Total distance from each member to all others in the cluster.
    totals: list[tuple[float, str, int]] = []
    for i in members_sorted:
        total = 0.0
        for j in members_sorted:
            if i == j:
                continue
            total += float(distances[i, j])
        totals.append((total, ids[i], i))
    # Smallest total wins; ``ids[i]`` is the lexicographic tie-break.
    totals.sort()
    exemplar_idx = totals[0][2]
    # Mean pairwise distance among the cluster members.
    pair_count = 0
    pair_sum = 0.0
    for a_idx, i in enumerate(members_sorted):
        for j in members_sorted[a_idx + 1 :]:
            pair_sum += float(distances[i, j])
            pair_count += 1
    intra = pair_sum / pair_count if pair_count else 0.0
    return TrajectoryCluster(
        cluster_id=cluster_id,
        member_episode_ids=tuple(ids[i] for i in members_sorted),
        exemplar_episode_id=ids[exemplar_idx],
        intra_cluster_distance=intra,
    )


def cluster_failed_trajectories(
    trajectory_dir: Path,
    episodes: list[Episode],
    n_clusters: int | None = None,
    distance: Distance = "dtw",
) -> TaxonomyResult:
    """Cluster the trajectories of failed episodes.

    Loads the per-episode action streams from ``trajectory_dir`` (one
    NPZ per episode following
    :func:`gauntlet.runner.worker.trajectory_path_for`'s scheme) and
    groups them by trajectory similarity. Episodes whose
    ``success`` is ``True`` are skipped — the function name promises
    *failed*-trajectory clustering. Episodes whose NPZ is missing on
    disk are likewise skipped (a partial trajectory dump should
    degrade gracefully, not crash the report).

    Args:
        trajectory_dir: Directory containing the per-episode NPZ files.
        episodes: All episodes in the report. Successes are filtered
            out internally.
        n_clusters: Number of clusters. When ``None``, silhouette-
            search picks ``k ∈ [2, min(8, n_failed - 1)]``.
        distance: ``"dtw"`` (default; requires the
            ``[trajectory-taxonomy]`` extra) or ``"euclidean"``
            (fallback, no extra).

    Returns:
        :class:`TaxonomyResult`. ``clusters`` may be empty (no failures
        in the dataset, or no NPZs available). ``silhouette`` is
        ``None`` when the score is undefined.

    Raises:
        TaxonomyError: When ``distance="dtw"`` is requested but the
            ``[trajectory-taxonomy]`` extra is not installed.
    """
    failed_episodes = [ep for ep in episodes if not ep.success]
    if not failed_episodes:
        return TaxonomyResult(clusters=(), silhouette=None)

    # Skip any failed episode whose NPZ is missing — partial dumps
    # are a documented possibility (e.g. a manual delete) and the
    # report should not crash.
    available: list[tuple[str, NDArray[np.float64]]] = []
    for ep in failed_episodes:
        path = trajectory_path_for_episode(trajectory_dir, ep)
        if not path.exists():
            continue
        ep_id = episode_id(ep.cell_index, ep.episode_index)
        available.append((ep_id, _load_action_trajectory(path)))

    if not available:
        return TaxonomyResult(clusters=(), silhouette=None)

    # Sort by id for fully deterministic order regardless of input
    # episode list ordering.
    available.sort(key=lambda pair: pair[0])
    ids = [pair[0] for pair in available]
    trajectories = [pair[1] for pair in available]

    n_failed = len(trajectories)
    if n_failed == 1:
        # One failure — a singleton cluster. Silhouette is undefined.
        only = TrajectoryCluster(
            cluster_id=0,
            member_episode_ids=(ids[0],),
            exemplar_episode_id=ids[0],
            intra_cluster_distance=0.0,
        )
        return TaxonomyResult(clusters=(only,), silhouette=None)

    distances = _pairwise_distances(trajectories, distance)

    if n_clusters is None:
        _, clusters_idx, silhouette = _select_n_clusters(distances, n_failed)
    else:
        k = max(1, min(n_clusters, n_failed))
        clusters_idx = _agglomerative_clusters(distances, k)
        if k >= 2:
            labels = _labels_from_clusters(clusters_idx, n_failed)
            silhouette = _silhouette_score(distances, labels)
        else:
            silhouette = None

    # Order clusters: largest first, then by exemplar id for stable
    # tie-break.
    records = [
        _cluster_to_record(cid, members, distances, ids) for cid, members in enumerate(clusters_idx)
    ]
    records.sort(
        key=lambda rec: (-len(rec.member_episode_ids), rec.exemplar_episode_id),
    )
    # Re-issue cluster ids so the visible IDs match the sort order.
    renumbered = tuple(
        TrajectoryCluster(
            cluster_id=i,
            member_episode_ids=rec.member_episode_ids,
            exemplar_episode_id=rec.exemplar_episode_id,
            intra_cluster_distance=rec.intra_cluster_distance,
        )
        for i, rec in enumerate(records)
    )
    return TaxonomyResult(clusters=renumbered, silhouette=silhouette)
