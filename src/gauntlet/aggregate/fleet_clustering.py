"""Fleet-wide failure-mode clustering — Phase 3 Task 19.

Layered on top of the existing :mod:`gauntlet.aggregate` pipeline. Where
:func:`gauntlet.aggregate.aggregate_reports` rolls counts up by axis,
*this* module looks at the **shape** of each run's failure clusters and
groups runs that fail in the same way — so a fleet of 30 checkpoints
collapses to "these five failure modes recur across the fleet" instead
of "30 separate cluster lists, good luck".

The unit being clustered is a per-(run, cell) **failure signature**:

* the run's failing axis-pair (sorted, normalised float values);
* the Wilson lower bound (B-03) on the cluster's failure rate — small
  jitter is rounded out via a coarse grid so two near-identical cells
  do not split across two clusters;
* a small set of behavioural-metric summaries (B-02 / B-21) so two
  runs that fail under the same axis-config but with different
  trajectories still separate.

Two passes:

1. **Hash-bucket pass** — equal signatures collapse into the same
   bucket. Runs that share a fingerprint go into the same cluster
   immediately; this is the cheap exact-match primitive.
2. **Agglomerative merge pass** — if the bucket count exceeds
   ``max_clusters``, merge nearest bucket pairs (single-linkage on the
   custom signature distance) until the count is at or below the cap.

The cluster's ``representative_failure_signature`` is the **medoid** —
the bucket signature with the minimum summed intra-cluster distance,
not an arbitrary first member (RFC §6 / task spec).

Empty inputs short-circuit cleanly:

* ``n_runs == 0`` → empty :class:`FleetClusteringResult` with
  ``silhouette = None``. No exception (the CLI happy-path test pins
  this).
* Single run / single bucket → silhouette is undefined and reported as
  ``None`` (the textbook silhouette assumes >=2 clusters).

No scipy dependency — Wilson math (the only floating-point sibling we
already ship) lives in :mod:`gauntlet.report.wilson` and the rest of
the aggregate package keeps the dep surface tight. The clustering math
here is closed-form / O(n^2) over the bucket count, not the run count;
in practice ``n_buckets <= n_runs * n_unique_failures_per_run`` and the
merge cap pulls it back down.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

from pydantic import ValidationError

from gauntlet.report.schema import FailureCluster, Report
from gauntlet.report.wilson import wilson_interval
from gauntlet.security import PathTraversalError, safe_join

__all__ = [
    "FleetCluster",
    "FleetClusteringResult",
    "cluster_fleet_failures",
]


# JSON-compatible recursive type alias used by the public surface
# (``representative_failure_signature`` payloads, the
# ``_result_to_payload`` writer). Mirrors the
# :mod:`gauntlet.aggregate.html._JsonValue` alias so the two halves of
# the aggregate package speak the same dump-format vocabulary, but kept
# local rather than imported to keep this module independent of the
# HTML renderer.
_JsonValue: TypeAlias = (
    float
    | int
    | str
    | bool
    | None
    | dict[str, "_JsonValue"]
    | list["_JsonValue"]
    | tuple["_JsonValue", ...]
)


# Axis values get rounded to this many decimal places so 1e-15 drift
# does not split a bucket. Mirrors :func:`gauntlet.report.analyze._norm`
# (we duplicate the constant locally to keep this module's import
# surface independent of the analyze layer's private helpers).
_AXIS_DECIMALS = 9

# Wilson lower bound rounding grid. 0.05 is coarse enough that
# (0.41, ..) and (0.43, ..) collapse, fine enough that (0.40, ..) and
# (0.55, ..) stay distinct. Documented in :doc:`docs/fleet-aggregation`.
_WILSON_GRID = 0.05

# Behavioural metric rounding grid. Behavioural means are dimensionless
# ratios in our schema (path-length ratio, jerk RMS, time-to-success
# fraction); a 0.1 grid keeps two runs distinct when they differ
# "noticeably" without splitting on the third significant digit.
_BEHAV_GRID = 0.1

# The behavioural metric set fed into the signature. Each is on
# :class:`FailureCluster` (B-02 / B-21). Order is fixed so the resulting
# tuple key is stable across Python versions / dict iteration orders.
_BEHAVIOURAL_FIELDS: tuple[str, ...] = (
    "mean_jerk_rms",
    "mean_path_length_ratio",
    "mean_time_to_success",
)


# ---------------------------------------------------------------------------
# Public dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetCluster:
    """One fleet-wide failure-mode cluster.

    Built by :func:`cluster_fleet_failures`. All members are immutable so
    a result can be cached / shared across threads without surprising
    aliasing.

    Attributes:
        cluster_id: Zero-indexed identifier, ``0..n_clusters-1``. Stable
            within a single :class:`FleetClusteringResult` (sorted by
            descending member count, then by signature for ties), so
            two callers that point at the same input directory see the
            same id-to-signature mapping.
        member_run_ids: Sorted list of run-ids (the parent directory
            name of each contributing ``report.json``, mirroring the
            existing :class:`gauntlet.aggregate.FleetRun.run_id`
            convention) that exhibit this failure mode. May contain a
            run multiple times only when a single run carries two
            cells with the same signature; we dedupe.
        representative_failure_signature: The medoid signature among
            the cluster's bucket members — the signature with the
            minimum summed intra-cluster distance. Returned as a
            JSON-friendly mapping so it round-trips cleanly through
            :func:`json.dumps` (axis values float, Wilson lower bound
            float, behavioural metric values float-or-None).
        cross_run_consistency: Fraction of total fleet runs that
            contribute to this cluster, ``[0.0, 1.0]``. ``1.0`` means
            every run in the fleet hits this failure mode; ``0.05``
            means it's a long-tail mode that bites one run in twenty.
            ``NaN`` is never produced — empty results carry zero
            clusters, and the divisor is always ``n_runs >= 1`` for any
            cluster that exists.
    """

    cluster_id: int
    member_run_ids: list[str]
    representative_failure_signature: dict[str, _JsonValue]
    cross_run_consistency: float


@dataclass(frozen=True)
class FleetClusteringResult:
    """Result of :func:`cluster_fleet_failures`.

    Attributes:
        clusters: Zero or more :class:`FleetCluster`, sorted by
            descending member count then by ``cluster_id``. Empty when
            the input directory has no ``report.json`` files OR when no
            run carries any failure cluster.
        n_runs: Total number of runs scanned (one per ``report.json``).
            Zero is allowed and short-circuits clustering.
        n_unique_failures: Number of distinct failure signatures
            observed BEFORE the agglomerative-merge pass collapsed the
            bucket count to ``max_clusters``. Lets a caller see "your
            fleet has 47 unique failure modes, we showed you the top
            8" without re-scanning.
        silhouette: Mean silhouette coefficient (Rousseeuw, 1987) over
            the post-merge clusters, ``[-1.0, 1.0]``. ``None`` when the
            metric is undefined (zero clusters, one cluster, or one
            unique signature). The score is informational — the
            clusters are useful even at low silhouette because the
            primary primitive is exact-signature equality.
    """

    clusters: list[FleetCluster]
    n_runs: int
    n_unique_failures: int
    silhouette: float | None


# ---------------------------------------------------------------------------
# Internal scratch types.
# ---------------------------------------------------------------------------


@dataclass
class _SignatureKey:
    """Hashable failure signature.

    Held as a dataclass (not :class:`tuple`) so the field semantics
    survive in tracebacks. The hash key is the ``(axes, wilson, behav)``
    tuple and equality is structural; we override ``__hash__`` /
    ``__eq__`` rather than relying on ``frozen=True`` so the
    behavioural-field tuple (which itself contains floats and Nones)
    stays compatible across Python versions.
    """

    # Sorted ((axis_name, normalised_value), ...) tuple. Tuple instead
    # of dict so the key is hashable.
    axes: tuple[tuple[str, float], ...]
    # Wilson lower bound on the failure rate, rounded to ``_WILSON_GRID``.
    wilson_lower: float
    # Behavioural metric tuple in ``_BEHAVIOURAL_FIELDS`` order. ``None``
    # entries are kept as None so a cluster with missing telemetry
    # doesn't collapse into one with measured-but-zero telemetry.
    behavioural: tuple[float | None, ...]

    def __hash__(self) -> int:
        return hash((self.axes, self.wilson_lower, self.behavioural))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SignatureKey):
            return NotImplemented
        return (
            self.axes == other.axes
            and self.wilson_lower == other.wilson_lower
            and self.behavioural == other.behavioural
        )

    def to_payload(self) -> dict[str, _JsonValue]:
        """Render to a JSON-friendly dict for the cluster representative."""
        axes_payload: dict[str, _JsonValue] = {name: value for name, value in self.axes}
        behav_payload: dict[str, _JsonValue] = {
            name: value for name, value in zip(_BEHAVIOURAL_FIELDS, self.behavioural, strict=True)
        }
        return {
            "axes": axes_payload,
            "wilson_lower_bound": self.wilson_lower,
            "behavioural_metric_summary": behav_payload,
        }


@dataclass
class _Bucket:
    """One signature bucket — pre-agglomeration scratch type."""

    signature: _SignatureKey
    run_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signature extraction.
# ---------------------------------------------------------------------------


def _round_axis_value(value: float) -> float:
    """Float-normalise an axis value so 1e-15 drift does not split a bucket."""
    if not math.isfinite(value):
        return value
    return round(value, _AXIS_DECIMALS)


def _round_to_grid(value: float | None, grid: float) -> float | None:
    """Round ``value`` to the nearest multiple of ``grid``.

    Returns ``None`` when the input is ``None`` (missing telemetry) or
    non-finite — propagating the absent-data sentinel so two clusters
    with missing measurements stay distinct from two clusters with the
    same measured value.
    """
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(value / grid) * grid


def _wilson_lower_for_cluster(cluster: FailureCluster) -> float:
    """Return the Wilson lower bound on the failure rate for this cluster.

    The schema's ``ci_low`` field is the canonical pre-computed value
    (B-03 attaches it at report-build time). For backwards-compat with
    pre-B-03 reports — which left ``ci_low = None`` — fall back to
    recomputing via :func:`wilson_interval` on the raw counts. ``n=0``
    (which the analyse layer never produces but the schema permits)
    rounds to ``0.0`` so the signature stays comparable.
    """
    if cluster.ci_low is not None and math.isfinite(cluster.ci_low):
        return cluster.ci_low
    n_failures = cluster.n_episodes - cluster.n_success
    low, _high = wilson_interval(n_failures, cluster.n_episodes)
    return low if low is not None else 0.0


def _signature_for_cluster(cluster: FailureCluster) -> _SignatureKey:
    """Extract the canonical failure signature for one cluster.

    The signature drives both the hash-bucket pass and the
    agglomerative merge — see the module docstring. Axes are sorted by
    name so two semantically-identical clusters with different
    insertion orders match.
    """
    axes = tuple(sorted((name, _round_axis_value(value)) for name, value in cluster.axes.items()))
    wilson_lower = _round_to_grid(_wilson_lower_for_cluster(cluster), _WILSON_GRID)
    # ``_round_to_grid`` only returns None for None / non-finite input;
    # ``_wilson_lower_for_cluster`` always yields a finite float, so the
    # cast back to float is safe. Defensive fallback to 0.0 keeps mypy
    # happy without an ``Any`` leak.
    wilson_value = wilson_lower if wilson_lower is not None else 0.0
    behavioural = tuple(
        _round_to_grid(getattr(cluster, name), _BEHAV_GRID) for name in _BEHAVIOURAL_FIELDS
    )
    return _SignatureKey(axes=axes, wilson_lower=wilson_value, behavioural=behavioural)


# ---------------------------------------------------------------------------
# Distance / silhouette helpers.
# ---------------------------------------------------------------------------


def _signature_distance(a: _SignatureKey, b: _SignatureKey) -> float:
    """Composite distance between two failure signatures.

    Sum of three normalised components:

    * **Axis Hamming** — count of (name, value) pairs that differ
      between ``a`` and ``b``, normalised by the union size. ``[0, 1]``.
      A signature with a missing axis name on one side counts as a
      mismatch.
    * **Wilson L1** — absolute difference between the two rounded
      Wilson lower bounds. ``[0, 1]`` (the bound itself is in [0, 1]).
    * **Behavioural L1** — sum of per-field absolute differences across
      :data:`_BEHAVIOURAL_FIELDS`. ``None`` on either side contributes
      ``0.5`` for that field (a "soft" mismatch — neither evidence of
      similarity nor full dissimilarity), and the sum is normalised by
      the field count so the contribution stays in ``[0, 1]``.

    The three components are equally weighted and the total is in
    ``[0, 3]``. The agglomerative merge uses single-linkage on this
    metric — the absolute scale matters less than the rank order.
    """
    # Axis component.
    axes_a = dict(a.axes)
    axes_b = dict(b.axes)
    union_keys = axes_a.keys() | axes_b.keys()
    if not union_keys:
        axis_dist = 0.0
    else:
        mismatches = sum(1 for k in union_keys if axes_a.get(k) != axes_b.get(k))
        axis_dist = mismatches / len(union_keys)

    # Wilson component — both values are already grid-rounded floats in
    # [0, 1].
    wilson_dist = abs(a.wilson_lower - b.wilson_lower)

    # Behavioural component.
    n_behav = len(_BEHAVIOURAL_FIELDS)
    behav_total = 0.0
    for av, bv in zip(a.behavioural, b.behavioural, strict=True):
        if av is None or bv is None:
            behav_total += 0.5
        else:
            behav_total += min(1.0, abs(av - bv))
    behav_dist = behav_total / n_behav if n_behav > 0 else 0.0

    return axis_dist + wilson_dist + behav_dist


def _silhouette(
    bucket_assignments: list[int],
    bucket_signatures: list[_SignatureKey],
) -> float | None:
    """Mean silhouette over the bucket-level clustering.

    ``bucket_assignments[i]`` is the cluster id for ``bucket_signatures[i]``.
    Standard Rousseeuw silhouette: per-bucket ``s = (b - a) / max(a, b)``
    where ``a`` is the mean intra-cluster distance and ``b`` is the
    minimum mean inter-cluster distance to another cluster. Returns
    ``None`` when the metric is undefined:

    * < 2 buckets total;
    * < 2 clusters total;
    * any cluster has only one bucket AND there's only one bucket pair
      to evaluate (the textbook ``s_i = 0`` for a singleton is
      well-defined; we keep it but only return ``None`` when the
      *whole result* is a single cluster).
    """
    n = len(bucket_signatures)
    if n < 2:
        return None
    cluster_ids = set(bucket_assignments)
    if len(cluster_ids) < 2:
        return None

    silhouettes: list[float] = []
    for i in range(n):
        own_cluster = bucket_assignments[i]
        # Mean distance to other members of own cluster.
        own_distances = [
            _signature_distance(bucket_signatures[i], bucket_signatures[j])
            for j in range(n)
            if j != i and bucket_assignments[j] == own_cluster
        ]
        if not own_distances:
            # Singleton bucket → s_i = 0 by convention.
            silhouettes.append(0.0)
            continue
        a_i = sum(own_distances) / len(own_distances)

        # Minimum mean distance to any other cluster.
        other_means: list[float] = []
        for cid in cluster_ids:
            if cid == own_cluster:
                continue
            members = [
                _signature_distance(bucket_signatures[i], bucket_signatures[j])
                for j in range(n)
                if bucket_assignments[j] == cid
            ]
            if members:
                other_means.append(sum(members) / len(members))
        if not other_means:
            silhouettes.append(0.0)
            continue
        b_i = min(other_means)

        denom = max(a_i, b_i)
        if denom == 0.0:
            silhouettes.append(0.0)
        else:
            silhouettes.append((b_i - a_i) / denom)

    return sum(silhouettes) / len(silhouettes)


# ---------------------------------------------------------------------------
# Bucket → cluster pipeline.
# ---------------------------------------------------------------------------


def _build_buckets(reports: Iterable[tuple[str, Report]]) -> list[_Bucket]:
    """Bucket every per-(run, cluster) signature by exact equality.

    ``reports`` yields ``(run_id, Report)`` pairs. A run can contribute
    multiple cluster signatures; each is bucketed independently, and a
    run that hits the same signature twice is recorded once per
    bucket (we dedupe member ids in :func:`cluster_fleet_failures`).
    """
    buckets: dict[_SignatureKey, _Bucket] = {}
    for run_id, report in reports:
        for cluster in report.failure_clusters:
            sig = _signature_for_cluster(cluster)
            entry = buckets.setdefault(sig, _Bucket(signature=sig))
            entry.run_ids.append(run_id)
    return list(buckets.values())


def _agglomerate(
    buckets: list[_Bucket],
    *,
    max_clusters: int,
) -> list[list[int]]:
    """Single-linkage agglomerative merge to ``max_clusters``.

    Returns a list of bucket-index lists — one per resulting cluster.
    When ``len(buckets) <= max_clusters`` no merging happens and the
    return is one list per bucket. ``max_clusters <= 0`` is rejected
    upstream; this helper assumes a positive cap.
    """
    n = len(buckets)
    # Each cluster is initially a singleton list of bucket indices.
    clusters: list[list[int]] = [[i] for i in range(n)]
    if n <= max_clusters:
        return clusters

    while len(clusters) > max_clusters:
        # Find the closest pair under single-linkage.
        best_i = 0
        best_j = 1
        best_dist = math.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Single linkage = min distance over all member pairs.
                pair_dist = math.inf
                for ai in clusters[i]:
                    for bj in clusters[j]:
                        d = _signature_distance(
                            buckets[ai].signature,
                            buckets[bj].signature,
                        )
                        if d < pair_dist:
                            pair_dist = d
                if pair_dist < best_dist:
                    best_dist = pair_dist
                    best_i = i
                    best_j = j
        # Merge ``best_j`` into ``best_i``; drop ``best_j``.
        clusters[best_i] = clusters[best_i] + clusters[best_j]
        clusters.pop(best_j)

    return clusters


def _medoid_signature(
    member_indices: list[int],
    buckets: list[_Bucket],
) -> _SignatureKey:
    """Pick the medoid signature — minimum summed intra-cluster distance.

    Ties are broken by the signature's natural sort (axis tuple, then
    Wilson lower, then behavioural tuple) so the choice is deterministic
    across runs. Single-member clusters short-circuit to that member's
    signature.
    """
    if len(member_indices) == 1:
        return buckets[member_indices[0]].signature

    best_idx = member_indices[0]
    best_score = math.inf
    # Sort the candidate index list so equal-score signatures resolve
    # deterministically — the loop visits them in stable order.
    for idx in sorted(
        member_indices,
        key=lambda k: (
            buckets[k].signature.axes,
            buckets[k].signature.wilson_lower,
            buckets[k].signature.behavioural,
        ),
    ):
        score = sum(
            _signature_distance(buckets[idx].signature, buckets[other].signature)
            for other in member_indices
            if other != idx
        )
        if score < best_score:
            best_score = score
            best_idx = idx
    return buckets[best_idx].signature


# ---------------------------------------------------------------------------
# I/O — directory scan + JSON load.
# ---------------------------------------------------------------------------


def _resolve_report_dir(report_dir: Path) -> Path:
    """Resolve and validate the user-supplied scan root.

    The clustering CLI accepts an arbitrary ``<dir>`` argument; we run
    it through :func:`gauntlet.security.safe_join` (with an empty parts
    tuple) to canonicalise it through the same resolver path the rest
    of the harness uses, then verify the directory exists. A traversal
    in the argument itself (e.g. ``../../etc``) doesn't escape any
    base in this call — but the canonicalised form is what every later
    ``safe_join`` boundary inside the loader uses, so the contract is
    consistent end-to-end.
    """
    resolved = safe_join(report_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"directory not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"not a directory: {resolved}")
    return resolved


def _iter_report_files(root: Path) -> list[Path]:
    """Recursively collect ``report.json`` files under ``root``.

    Each match is re-validated through :func:`safe_join` against the
    canonical root — this catches the symlink-out-of-tree case that
    bare ``rglob`` would happily follow.
    """
    matches: list[Path] = []
    for candidate in sorted(root.rglob("report.json")):
        # Compute the relative path so we can re-route through
        # safe_join — this rejects any candidate whose resolved form
        # leaves ``root`` (symlink escape, etc.).
        try:
            rel = candidate.resolve(strict=False).relative_to(root)
        except ValueError:
            # ``relative_to`` raised — the candidate resolved outside
            # the root. Skip rather than raise so a single dangling
            # symlink in the tree doesn't sink the whole scan.
            continue
        try:
            verified = safe_join(root, *rel.parts)
        except PathTraversalError:
            continue
        matches.append(verified)
    return matches


def _load_report(path: Path, root: Path) -> tuple[str, Report]:
    """Load one ``report.json`` and derive its ``run_id``.

    ``run_id`` is the path of the parent directory relative to
    ``root`` — mirroring :class:`gauntlet.aggregate.FleetRun.run_id`'s
    "the run dir name is the run id" convention but extending to nested
    layouts (the relative parent path stays unique even when two runs
    land under different sub-fleets).
    """
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{path.name}: invalid JSON ({exc.msg})") from exc
    try:
        report = Report.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{path}: not a valid report.json: {exc}") from exc

    parent = path.parent
    run_id = path.stem if parent == root else parent.relative_to(root).as_posix()
    return run_id, report


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def cluster_fleet_failures(
    report_dir: Path,
    max_clusters: int = 8,
) -> FleetClusteringResult:
    """Cluster failure modes across every ``report.json`` under ``report_dir``.

    Pipeline:

    1. Resolve / sandbox-validate the scan root via
       :func:`gauntlet.security.safe_join`.
    2. Discover every ``report.json`` under the root (recursive).
    3. For each report, extract one **failure signature** per failure
       cluster — see :func:`_signature_for_cluster`.
    4. Bucket signatures by exact equality. Runs that share a signature
       are pooled into the same bucket.
    5. If the bucket count exceeds ``max_clusters``, run a single-
       linkage agglomerative merge until the count fits the cap.
    6. For each surviving cluster, pick the medoid signature as the
       representative and compute ``cross_run_consistency`` as the
       distinct-member-count divided by the total run count.
    7. Compute the mean silhouette over the bucket-level assignment for
       reporting purposes.

    Args:
        report_dir: Directory to recursively scan for ``report.json``
            files. Both the directory path and every discovered file
            path are routed through :func:`safe_join` to defeat
            traversal / symlink-escape attacks.
        max_clusters: Hard cap on the number of clusters returned. The
            agglomerative pass merges nearest pairs until the bucket
            count fits. Must be ``>= 1``.

    Returns:
        A :class:`FleetClusteringResult`. Empty directory (or every
        report has zero failure clusters) yields ``clusters=[]``,
        ``n_unique_failures=0``, and ``silhouette=None`` with no
        exception. ``n_runs`` is always the count of ``report.json``
        files actually loaded.

    Raises:
        ValueError: ``max_clusters < 1``.
        FileNotFoundError: ``report_dir`` does not exist after
            :func:`safe_join` canonicalisation.
        NotADirectoryError: ``report_dir`` exists but is not a
            directory.
        PathTraversalError: ``report_dir`` (or a discovered file) is
            blocked by :func:`safe_join`.
    """
    if max_clusters < 1:
        raise ValueError(f"max_clusters must be >= 1; got {max_clusters}")

    root = _resolve_report_dir(report_dir)
    files = _iter_report_files(root)

    # Empty / no-files case → empty result, no exception.
    if not files:
        return FleetClusteringResult(
            clusters=[],
            n_runs=0,
            n_unique_failures=0,
            silhouette=None,
        )

    loaded: list[tuple[str, Report]] = [_load_report(p, root) for p in files]
    n_runs = len(loaded)

    buckets = _build_buckets(loaded)
    n_unique_failures = len(buckets)

    if not buckets:
        # Every loaded report was clean — no failure clusters at all.
        return FleetClusteringResult(
            clusters=[],
            n_runs=n_runs,
            n_unique_failures=0,
            silhouette=None,
        )

    cluster_member_buckets = _agglomerate(buckets, max_clusters=max_clusters)

    # Build the bucket-level assignment list for silhouette computation.
    bucket_assignments: list[int] = [0] * len(buckets)
    for cid, members in enumerate(cluster_member_buckets):
        for m in members:
            bucket_assignments[m] = cid
    bucket_signatures = [b.signature for b in buckets]
    silhouette = _silhouette(bucket_assignments, bucket_signatures)

    # Sort clusters by descending member count (the run-distinct count,
    # not the bucket count) then by the medoid signature for stable
    # tiebreaking — a 1-run cluster shouldn't outrank a 5-run cluster
    # on raw bucket count alone.
    enriched: list[tuple[int, list[str], _SignatureKey]] = []
    for raw_id, member_indices in enumerate(cluster_member_buckets):
        run_ids = sorted({rid for idx in member_indices for rid in buckets[idx].run_ids})
        medoid = _medoid_signature(member_indices, buckets)
        enriched.append((raw_id, run_ids, medoid))

    enriched.sort(
        key=lambda triple: (
            -len(triple[1]),
            triple[2].axes,
            triple[2].wilson_lower,
            triple[2].behavioural,
        )
    )

    clusters: list[FleetCluster] = []
    for new_id, (_raw_id, run_ids, medoid) in enumerate(enriched):
        consistency = len(run_ids) / n_runs if n_runs > 0 else 0.0
        clusters.append(
            FleetCluster(
                cluster_id=new_id,
                member_run_ids=list(run_ids),
                representative_failure_signature=medoid.to_payload(),
                cross_run_consistency=consistency,
            )
        )

    return FleetClusteringResult(
        clusters=clusters,
        n_runs=n_runs,
        n_unique_failures=n_unique_failures,
        silhouette=silhouette,
    )


def _result_to_payload(result: FleetClusteringResult) -> dict[str, _JsonValue]:
    """JSON-friendly dump of a :class:`FleetClusteringResult`.

    Used by :mod:`gauntlet.aggregate.cli` to write the clustering
    artefact alongside ``fleet_report.json``. Surfaced here so test
    fixtures can re-use it without re-importing the CLI helper.
    """
    cluster_payloads: list[_JsonValue] = []
    for c in result.clusters:
        cluster_payloads.append(
            {
                "cluster_id": c.cluster_id,
                "member_run_ids": list(c.member_run_ids),
                "representative_failure_signature": _coerce_payload(
                    c.representative_failure_signature,
                ),
                "cross_run_consistency": c.cross_run_consistency,
            }
        )
    return {
        "clusters": cluster_payloads,
        "n_runs": result.n_runs,
        "n_unique_failures": result.n_unique_failures,
        "silhouette": result.silhouette,
    }


def _coerce_payload(value: _JsonValue) -> _JsonValue:
    """Recursively coerce signature payload values for ``json.dumps``.

    Floats round-trip cleanly; ``None`` becomes ``null``; nested dicts
    get the same treatment. Tuple values (axis tuples in particular)
    become lists so the JSON encoder accepts them. Mapping types other
    than :class:`dict` (e.g. ``MappingProxyType``) get a dict copy so
    the encoder doesn't choke.
    """
    if isinstance(value, Mapping):
        return {str(k): _coerce_payload(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_coerce_payload(v) for v in value]
    if isinstance(value, list):
        return [_coerce_payload(v) for v in value]
    return value
