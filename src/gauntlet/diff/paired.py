"""Common-random-numbers (CRN) paired comparison statistics — B-08.

When ``gauntlet compare`` / ``diff`` runs against two policies on the
*same* suite under the *same* ``master_seed``, the
:class:`gauntlet.runner.Runner` already derives identical per-episode env
seeds from the ``(master_seed, cell_index, episode_index)`` lattice via
:meth:`numpy.random.SeedSequence.spawn`. The
:class:`~gauntlet.env.tabletop.TabletopEnv` already feeds that single
seed into both initial-state randomisation
(:attr:`TabletopEnv._rng`) and any policy-side stream (built from the
same SeedSequence node in :func:`gauntlet.runner.worker.execute_one`).

So the *paired* run already has paired observations end-to-end -- the
only remaining work is to *exploit* the pairing on the statistics side.
This module is that exploit:

* :func:`pair_episodes` lines up the two episode lists by
  ``(cell_index, episode_index)`` and rejects mismatched seeds (a hard
  signal that the runs were not actually paired).
* :func:`mcnemar_test` computes the closed-form McNemar chi-square (and
  the exact-binomial p-value when ``b + c < 25``) on the paired
  pass/fail contingency table for one cell.
* :func:`paired_delta_ci` returns the Newcombe / Tango Wald CI on the
  per-cell success-rate delta -- much tighter than the
  Wilson-on-A minus Wilson-on-B difference because it folds in the
  positive correlation between paired episodes.
* :func:`compute_paired_cells` produces a list of :class:`PairedCellDelta`
  for every shared cell -- the data structure
  :command:`gauntlet compare --paired` and :command:`gauntlet diff --paired`
  surface, and the artefact B-20 (regression-vs-noise attribution)
  consumes for tagging flips.

Variance-reduction intuition (Goldberg / SureSim):

    Var[X - Y]  =  Var[X] + Var[Y] - 2 * Cov[X, Y]

Under independent runs ``Cov = 0`` and the difference variance is the
sum. Under CRN the covariance is large and positive, which subtracts
back out -- empirically a 2-4x reduction in episodes-needed for the same
delta-confidence on tabletop sweeps. See "Beyond Binary Success"
(arxiv 2603.13616) and SureSim (arxiv 2510.04354).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from gauntlet.report.wilson import Z_95

__all__ = [
    "MCNEMAR_EXACT_THRESHOLD",
    "McNemarResult",
    "PairedCellDelta",
    "PairedComparison",
    "PairingError",
    "compute_paired_cells",
    "mcnemar_test",
    "pair_episodes",
    "paired_delta_ci",
]


# ----------------------------------------------------------------------
# Constants.
# ----------------------------------------------------------------------

# Below this many discordant pairs (b + c) the chi-square approximation
# under-covers and we fall back to the exact two-sided binomial. Threshold
# of 25 matches the standard textbook recommendation (Agresti, "Categorical
# Data Analysis", section 10.1.4) and SciPy's ``scipy.stats.contingency.mcnemar``
# default ``exact=True`` switching point.
MCNEMAR_EXACT_THRESHOLD: int = 25


# ----------------------------------------------------------------------
# Data shapes.
# ----------------------------------------------------------------------


class McNemarResult(BaseModel):
    """McNemar's test on a single cell's paired pass/fail table.

    ``b`` counts pairs where A succeeded and B failed (loss for B).
    ``c`` counts pairs where A failed and B succeeded (win for B).
    The ``a`` (both succeed) and ``d`` (both fail) cells are not part of
    the discordant count; they are excluded from the test by design --
    a paired pass/fail flip is what we care about.

    ``statistic`` is the chi-square value when ``exact=False`` and is
    ``nan`` for the exact-binomial path; ``p_value`` is always
    populated. ``exact`` reports which branch ran so downstream consumers
    can decide whether to display the chi-square or just the p-value.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    b: int
    c: int
    statistic: float
    p_value: float
    exact: bool


class PairedCellDelta(BaseModel):
    """One shared cell's paired CRN delta.

    The delta is ``b_success_rate - a_success_rate`` over the *paired*
    episode set (size ``n_paired``). The CI is the Newcombe / Tango
    Wald interval on the paired difference, which is much tighter than
    independently subtracting the two Wilson intervals.

    The ``mcnemar`` field carries the per-cell McNemar test on the
    pass/fail contingency table; downstream B-20 attribution turns
    ``p_value`` into a regressed/improved/within_noise tag.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    cell_index: int
    perturbation_config: dict[str, float]
    n_paired: int
    a_success_rate: float
    b_success_rate: float
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    mcnemar: McNemarResult


class PairedComparison(BaseModel):
    """Top-level CRN paired comparison artefact for a single (a, b) pair.

    Surfaced both inside the ``compare.json`` payload and inside the
    :class:`gauntlet.diff.ReportDiff` so consumers (the dashboard, B-20,
    a future GitHub-Actions summary) get one consistent shape.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    paired: bool = True
    master_seed: int
    suite_name: str
    n_cells: int
    n_paired_episodes: int
    cells: list[PairedCellDelta] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Pairing helpers.
# ----------------------------------------------------------------------


class PairingError(ValueError):
    """Two episode lists cannot be paired (master-seed or shape mismatch).

    Surfaced verbatim by the CLI ``--paired`` path so the user gets a
    clear, actionable message -- never silently degraded to an unpaired
    run, because that would defeat the user's explicit opt-in.
    """


def _episode_master_seed(episode: object) -> int:
    """Extract ``master_seed`` from an Episode-like object.

    Accepts either a :class:`gauntlet.runner.Episode` or any object that
    quacks like one (a dict from a hand-built test fixture, a pydantic
    model with the same shape). Pure helper so the paired stack does not
    take a hard dependency on the runner package -- the diff module is
    consumable from a torch-free install.
    """
    metadata = getattr(episode, "metadata", None)
    if metadata is None and isinstance(episode, Mapping):
        metadata = episode.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PairingError(
            f"episode is missing a metadata mapping; cannot extract master_seed "
            f"(got episode={episode!r})"
        )
    seed = metadata.get("master_seed")
    if not isinstance(seed, int):
        raise PairingError(
            f"episode.metadata.master_seed must be int; got {type(seed).__name__}={seed!r}"
        )
    return seed


def _episode_key(episode: object) -> tuple[int, int]:
    """``(cell_index, episode_index)`` lookup tuple."""
    cell_index = getattr(episode, "cell_index", None)
    episode_index = getattr(episode, "episode_index", None)
    if cell_index is None and isinstance(episode, Mapping):
        cell_index = episode.get("cell_index")
    if episode_index is None and isinstance(episode, Mapping):
        episode_index = episode.get("episode_index")
    if not isinstance(cell_index, int) or not isinstance(episode_index, int):
        raise PairingError(
            f"episode is missing (cell_index, episode_index) ints; got episode={episode!r}"
        )
    return (cell_index, episode_index)


def _episode_seed(episode: object) -> int:
    """Extract the per-episode env seed.

    The runner threads the same env seed into both
    :meth:`gauntlet.env.tabletop.TabletopEnv.reset` (initial-state RNG)
    and the policy stream, so equality of this scalar between paired
    sides is the on-disk proof that CRN actually held end-to-end.
    """
    seed = getattr(episode, "seed", None)
    if seed is None and isinstance(episode, Mapping):
        seed = episode.get("seed")
    if not isinstance(seed, int):
        raise PairingError(f"episode.seed must be int; got {type(seed).__name__}={seed!r}")
    return seed


def _episode_success(episode: object) -> bool:
    """Extract the boolean ``success`` flag (the McNemar input)."""
    success = getattr(episode, "success", None)
    if success is None and isinstance(episode, Mapping):
        success = episode.get("success")
    if not isinstance(success, bool):
        raise PairingError(
            f"episode.success must be bool; got {type(success).__name__}={success!r}"
        )
    return success


def _episode_perturbation_config(episode: object) -> Mapping[str, float]:
    """Extract the per-episode perturbation config (echoed onto delta rows)."""
    cfg = getattr(episode, "perturbation_config", None)
    if cfg is None and isinstance(episode, Mapping):
        cfg = episode.get("perturbation_config")
    if not isinstance(cfg, Mapping):
        raise PairingError(
            f"episode.perturbation_config must be a mapping; got {type(cfg).__name__}={cfg!r}"
        )
    return cfg


def pair_episodes(
    episodes_a: Sequence[object],
    episodes_b: Sequence[object],
) -> tuple[int, list[tuple[object, object]]]:
    """Line up two episode lists by ``(cell_index, episode_index)``.

    Verifies (in order):

    1. Both sides carry the same ``master_seed`` (recorded by the runner
       in every Episode's ``metadata["master_seed"]``). A mismatch is a
       hard error -- the runs were *not* paired and the user should drop
       ``--paired``.
    2. Every paired ``(cell, episode)`` index has the same env ``seed``
       on both sides. This proves the deterministic seed-derivation
       contract held end-to-end. A mismatch is a runner-level bug
       (somebody bypassed :func:`gauntlet.runner.worker.extract_env_seed`)
       and we surface it loudly rather than silently emit garbage stats.

    Returns:
        ``(master_seed, paired_pairs)``. ``paired_pairs`` is the
        intersection of both lists' ``(cell, episode)`` keys, sorted for
        downstream determinism. Cells present on only one side are
        silently dropped -- they are not paired observations and
        contribute no information to the CRN delta.

    Raises:
        PairingError: master-seed mismatch, non-int seeds, or per-episode
            seed mismatch on a shared ``(cell, episode)`` key.
    """
    if not episodes_a or not episodes_b:
        raise PairingError(
            f"both episode lists must be non-empty for paired comparison; "
            f"got len(a)={len(episodes_a)}, len(b)={len(episodes_b)}"
        )

    seed_a = _episode_master_seed(episodes_a[0])
    seed_b = _episode_master_seed(episodes_b[0])
    if seed_a != seed_b:
        raise PairingError(
            f"cannot pair: master_seed mismatch (run-a={seed_a}, run-b={seed_b}); "
            f"use --no-paired to compare independently"
        )

    by_key_a: dict[tuple[int, int], object] = {_episode_key(ep): ep for ep in episodes_a}
    by_key_b: dict[tuple[int, int], object] = {_episode_key(ep): ep for ep in episodes_b}

    shared_keys = sorted(by_key_a.keys() & by_key_b.keys())
    paired: list[tuple[object, object]] = []
    for key in shared_keys:
        ep_a = by_key_a[key]
        ep_b = by_key_b[key]
        seed_ep_a = _episode_seed(ep_a)
        seed_ep_b = _episode_seed(ep_b)
        if seed_ep_a != seed_ep_b:
            raise PairingError(
                f"cannot pair: env-seed mismatch at cell={key[0]}, episode={key[1]} "
                f"(run-a seed={seed_ep_a}, run-b seed={seed_ep_b}). The runs "
                f"share master_seed={seed_a} but the per-episode seeds differ -- "
                f"this is a runner-level bug; please file an issue with the suite."
            )
        paired.append((ep_a, ep_b))

    return (seed_a, paired)


# ----------------------------------------------------------------------
# Statistical primitives.
# ----------------------------------------------------------------------


def _binomial_two_sided_p(b: int, c: int) -> float:
    """Two-sided exact-binomial p-value for the McNemar fallback.

    Under H0 (no policy effect on the discordant pairs), each discordant
    pair is a fair coin flip and the count of "B improved" outcomes is
    ``Binomial(n=b+c, p=0.5)``. The standard two-sided p-value sums
    binomial probabilities at points at least as extreme (in absolute
    deviation from ``(b+c)/2``) as ``min(b, c)``, multiplied by 2 and
    clamped to 1.0 for symmetry.

    This is the textbook small-sample escape (Agresti section 10.1.4); it is
    closed-form and needs no scipy.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = 0.0
    log_half_pow_n = -n * math.log(2.0)
    for i in range(k + 1):
        log_term = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_pow_n
        tail += math.exp(log_term)
    p = 2.0 * tail
    return min(1.0, p)


def mcnemar_test(b: int, c: int) -> McNemarResult:
    """McNemar's test for paired pass/fail outcomes.

    Args:
        b: Pairs where A succeeded and B failed (regression for B).
        c: Pairs where A failed and B succeeded (improvement for B).

    Returns:
        :class:`McNemarResult`. Uses the closed-form chi-square
        ``(b - c)^2 / (b + c)`` with one degree of freedom when
        ``b + c >= MCNEMAR_EXACT_THRESHOLD``; below that switches to the
        exact-binomial two-sided p-value, which has correct small-sample
        coverage where the chi-square approximation under-covers.

    Raises:
        ValueError: if ``b`` or ``c`` is negative.
    """
    if b < 0 or c < 0:
        raise ValueError(f"b and c must be non-negative; got b={b}, c={c}")

    n = b + c
    if n == 0:
        return McNemarResult(b=b, c=c, statistic=0.0, p_value=1.0, exact=True)

    if n < MCNEMAR_EXACT_THRESHOLD:
        p = _binomial_two_sided_p(b, c)
        return McNemarResult(b=b, c=c, statistic=float("nan"), p_value=p, exact=True)

    statistic = ((b - c) ** 2) / n
    # ``math.erfc(x / sqrt(2)) == 2 * (1 - Phi(x))`` for x >= 0.
    p = math.erfc(math.sqrt(statistic) / math.sqrt(2.0))
    p = min(1.0, max(0.0, p))
    return McNemarResult(b=b, c=c, statistic=statistic, p_value=p, exact=False)


def paired_delta_ci(
    b: int,
    c: int,
    n_paired: int,
    *,
    z: float = Z_95,
) -> tuple[float, float]:
    """Newcombe / Tango Wald CI on the paired success-rate difference.

    The estimator is ``d = (c - b) / n_paired`` -- the change in B's
    success rate over the paired set, signed so a *regression for B*
    (more A-succeeds-B-fails than A-fails-B-succeeds) is negative. The
    variance under the paired-binomial is symmetric in ``b`` and ``c``::

        var[d]  =  ((b + c) - (c - b)^2 / n) / n^2

    The Wald interval is ``d +/- z * sqrt(var[d])``, clamped to
    ``[-1.0, 1.0]``. Standard paired-difference Wald interval -- Newcombe
    ("Improved confidence intervals for the difference between binomial
    proportions based on paired data", Statistics in Medicine, 1998)
    section 3.3 / Tango (1998).

    Args:
        b: Pairs where A succeeded, B failed (regression for B).
        c: Pairs where A failed, B succeeded (improvement for B).
        n_paired: Total number of paired observations (must satisfy
            ``n_paired >= b + c``).
        z: Two-sided standard normal quantile. Default :data:`Z_95`
            (95% interval).

    Returns:
        ``(ci_low, ci_high)`` clamped into ``[-1.0, 1.0]`` and signed
        so the bracket runs in the same direction as
        ``b_success_rate - a_success_rate`` from
        :func:`compute_paired_cells`.

    Raises:
        ValueError: if ``n_paired <= 0`` or ``n_paired < b + c`` or
            ``b`` or ``c`` is negative.
    """
    if n_paired <= 0:
        raise ValueError(f"n_paired must be > 0; got {n_paired}")
    if b < 0 or c < 0:
        raise ValueError(f"b and c must be non-negative; got b={b}, c={c}")
    if b + c > n_paired:
        raise ValueError(f"b + c ({b + c}) cannot exceed n_paired ({n_paired})")

    n_f = float(n_paired)
    delta = (c - b) / n_f
    radicand = max(0.0, ((b + c) - ((c - b) ** 2) / n_f) / (n_f * n_f))
    half = z * math.sqrt(radicand)
    return (max(-1.0, delta - half), min(1.0, delta + half))


# ----------------------------------------------------------------------
# Top-level: compute every paired cell delta for a (run-a, run-b) pair.
# ----------------------------------------------------------------------


def _normalize_perturbation(cfg: Mapping[str, float]) -> dict[str, float]:
    """Normalise float values to 9 decimal places.

    Mirrors :func:`gauntlet.diff.diff._norm` so the paired payload's
    cell identifiers round-trip identically to the
    :class:`~gauntlet.diff.diff.CellFlip` ones -- the dashboard / B-20
    consumer can join the two on ``perturbation_config`` without
    floating-point pain.
    """
    return {k: round(float(v), 9) for k, v in cfg.items()}


def compute_paired_cells(
    episodes_a: Sequence[object],
    episodes_b: Sequence[object],
    *,
    suite_name: str | None = None,
) -> PairedComparison:
    """Build the per-cell paired CRN delta artefact for a run pair.

    Pairs the episode lists via :func:`pair_episodes`, groups by
    ``cell_index``, and emits one :class:`PairedCellDelta` per shared
    cell. The McNemar test and the paired CI live on every entry.

    The output is sorted by ``cell_index`` ascending -- stable,
    determinism-friendly, and what the dashboard expects.

    Args:
        episodes_a: First run's :class:`gauntlet.runner.Episode` list
            (or any sequence of objects with the same field shape).
        episodes_b: Second run's episodes.
        suite_name: Optional label echoed onto :attr:`PairedComparison.suite_name`.
            When ``None`` the helper falls back to the first episode's
            ``suite_name`` (if present) or ``"unknown"``.

    Returns:
        :class:`PairedComparison` with one :class:`PairedCellDelta` per
        shared cell.

    Raises:
        PairingError: master-seed or env-seed mismatch.
    """
    master_seed, paired = pair_episodes(episodes_a, episodes_b)

    by_cell: dict[int, list[tuple[object, object]]] = {}
    for ep_a, ep_b in paired:
        cell_index = _episode_key(ep_a)[0]
        by_cell.setdefault(cell_index, []).append((ep_a, ep_b))

    cells: list[PairedCellDelta] = []
    for cell_index in sorted(by_cell.keys()):
        pairs = by_cell[cell_index]
        n_paired = len(pairs)
        a_cell = 0  # both succeed
        b_cell = 0  # A succeeds, B fails (regression for B)
        c_cell = 0  # A fails, B succeeds (improvement for B)
        d_cell = 0  # both fail
        for ep_a, ep_b in pairs:
            sa = _episode_success(ep_a)
            sb = _episode_success(ep_b)
            if sa and sb:
                a_cell += 1
            elif sa and not sb:
                b_cell += 1
            elif (not sa) and sb:
                c_cell += 1
            else:
                d_cell += 1

        a_success_rate = (a_cell + b_cell) / n_paired
        b_success_rate = (a_cell + c_cell) / n_paired
        delta = b_success_rate - a_success_rate
        ci_low, ci_high = paired_delta_ci(b_cell, c_cell, n_paired)
        mcnemar = mcnemar_test(b_cell, c_cell)
        perturbation_config = _normalize_perturbation(
            _episode_perturbation_config(pairs[0][0]),
        )
        cells.append(
            PairedCellDelta(
                cell_index=cell_index,
                perturbation_config=perturbation_config,
                n_paired=n_paired,
                a_success_rate=a_success_rate,
                b_success_rate=b_success_rate,
                delta=delta,
                delta_ci_low=ci_low,
                delta_ci_high=ci_high,
                mcnemar=mcnemar,
            )
        )

    resolved_suite_name: str
    if suite_name is not None:
        resolved_suite_name = suite_name
    else:
        first_name = getattr(episodes_a[0], "suite_name", None)
        if first_name is None and isinstance(episodes_a[0], Mapping):
            first_name = episodes_a[0].get("suite_name")
        resolved_suite_name = first_name if isinstance(first_name, str) else "unknown"

    return PairedComparison(
        paired=True,
        master_seed=master_seed,
        suite_name=resolved_suite_name,
        n_cells=len(cells),
        n_paired_episodes=len(paired),
        cells=cells,
    )
