"""Wilson score confidence intervals + sample-size planning (B-03, B-41).

Used by :mod:`gauntlet.report.analyze` (B-03) to attach a 95% CI to
every per-cell, per-axis-value, and failure-cluster success rate. The
report layer stores both the point estimate (``success_rate`` /
``failure_rate``) and the bracket so downstream consumers — HTML
rendering, dashboard, ``gauntlet diff`` regression-vs-noise attribution
(B-20) — can show "0.50 [0.19, 0.81]" without recomputing.

Closed-form Wilson is preferred over Clopper-Pearson here because:

* No new dependencies — :mod:`scipy.stats.binom` is the alternative,
  but ``scipy`` is not a project dependency and B-03's brief explicitly
  forbids adding heavy deps for a 10-line interval.
* Wilson has well-behaved coverage at small ``n`` and never returns
  ``[0, 0]`` or ``[1, 1]`` for ``k=0`` or ``k=n`` respectively (unlike
  the naive normal approximation), which matters because a single all-
  failed cell is exactly the kind of cell a failure-first report
  surfaces first.

Public entry points:

* :func:`wilson_interval` — the CI math used by the report layer.
* :func:`required_episodes` — B-41 sample-size calculator under the
  standard two-proportion z-test for independent samples.
* :func:`required_episodes_paired` — B-41 + B-08 paired-CRN-aware
  variant: applies the ``(1 - rho)`` variance reduction that shared-
  seed paired rollouts buy you.

Anti-feature warning (B-41 brief): the power calculator tells you the
*minimum* sample size needed to detect a binary success-rate gap of a
given size at a given power. Running exactly that many episodes per
cell optimises for the binary success summary statistic and *misses*
the long tail of failure modes that only show up at ten times that
count. The summary-vs-failures trap is exactly what GAUNTLET_SPEC.md
§1 ("failures over averages") is built to avoid. Treat the output as a
floor on episodes-per-cell, never a ceiling.
"""

from __future__ import annotations

import math
from statistics import NormalDist

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_PAIRED_RHO",
    "Z_95",
    "required_episodes",
    "required_episodes_paired",
    "wilson_interval",
]


# Default two-sided confidence level. Hard-coded as 0.95 per B-03 brief
# ("Wilson 95% by default") and per the precedent set by the
# statistics literature (Wilson, "Probable Inference, the Law of
# Succession, and Statistical Inference", 1927). Module-level constant
# so callers that need a different level can pass it explicitly without
# crawling through the analyze layer.
DEFAULT_CONFIDENCE: float = 0.95

# Two-sided z-score for the 95% Wilson interval. Pre-computed (vs.
# ``statistics.NormalDist().inv_cdf(0.975)`` at call time) so the
# common path stays allocation-free and trivially copy-paste-auditable
# against any statistics textbook.
Z_95: float = 1.959963984540054

# Default outcome-correlation for paired (CRN) sample-size planning.
# Empirical value for shared-seed paired rollouts under the B-08 paired
# comparison engine: ~0.5 outcome correlation between policy A and
# policy B episodes drawn from the same ``(cell_index, episode_index,
# master_seed)`` triple. Callers with measured correlation should pass
# their own ``rho`` to :func:`required_episodes_paired`. Bounded in
# ``[0, 1)``; ``rho >= 1`` would imply zero variance and infinite
# information per episode, which is degenerate.
DEFAULT_PAIRED_RHO: float = 0.5


def wilson_interval(
    successes: int,
    n: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float | None, float | None]:
    """Two-sided Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes (``0 <= successes <= n``).
        n: Number of trials. ``n == 0`` is allowed and returns
            ``(None, None)`` — a CI is undefined with no observations,
            and downstream pydantic schemas carry ``ci_low: float | None``
            for exactly this case.
        confidence: Two-sided coverage level in ``(0, 1)``. Default
            :data:`DEFAULT_CONFIDENCE` (0.95). The ``z`` quantile is
            computed via :class:`statistics.NormalDist` for any value
            other than 0.95, which uses the cached :data:`Z_95`.

    Returns:
        ``(ci_low, ci_high)`` clamped to ``[0.0, 1.0]``. Returns
        ``(None, None)`` when ``n == 0``.

    Raises:
        ValueError: if ``successes < 0`` or ``successes > n`` or
            ``confidence`` is not in ``(0, 1)``.

    Notes:
        Closed-form Wilson score interval:

            centre = (k + z² / 2) / (n + z²)
            half   = (z / (n + z²)) * sqrt(k(n-k)/n + z²/4)

        where ``k`` is ``successes`` and ``z`` is the standard-normal
        quantile at ``1 - (1 - confidence) / 2``. The result is clamped
        to ``[0.0, 1.0]`` to absorb floating-point arithmetic that can
        push the upper bound to ``1.0 + 1e-16`` for ``k == n``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    if successes < 0 or successes > n:
        raise ValueError(
            f"successes must satisfy 0 <= successes <= n; got successes={successes}, n={n}",
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"confidence must be in (0, 1); got {confidence}",
        )

    if n == 0:
        return (None, None)

    if confidence == DEFAULT_CONFIDENCE:
        z = Z_95
    else:
        # statistics.NormalDist is stdlib and exact enough for our
        # purposes (no scipy dependency). Two-sided coverage means we
        # ask for the upper tail at (1 + confidence) / 2.
        from statistics import NormalDist

        z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)

    z_squared = z * z
    denom = n + z_squared
    centre = (successes + z_squared / 2.0) / denom
    # ``successes * (n - successes) / n`` is k(n-k)/n; the second term
    # is z²/4. Together they are the variance of the score statistic.
    radicand = (successes * (n - successes)) / n + z_squared / 4.0
    # Floating-point can drive the radicand a hair below zero at the
    # boundary k=0 or k=n; clamp to keep ``sqrt`` real.
    half = (z / denom) * math.sqrt(max(radicand, 0.0))

    low = max(0.0, centre - half)
    high = min(1.0, centre + half)
    return (low, high)


# ──────────────────────────────────────────────────────────────────────
# Sample-size planning (B-41).
# ──────────────────────────────────────────────────────────────────────


def required_episodes(
    p1: float,
    p2: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Sample size per arm to detect a two-proportion gap at the given power.

    Closed-form pooled-variance two-proportion z-test:

        n = ((z_{alpha/2} + z_beta) * sqrt(2 * p_avg * (1 - p_avg)))^2
            / (p1 - p2)^2

    where ``p_avg = (p1 + p2) / 2`` and ``z_beta`` is the standard-
    normal quantile at ``power`` (so ``power = 0.8`` → ``z_beta ≈
    0.842``). The result is rounded up via :func:`math.ceil`. Returns
    the count *per arm* — i.e. ``required_episodes(0.5, 0.4) ≈ 389``
    means 389 episodes for the baseline policy AND 389 for the
    candidate.

    For paired (common-random-numbers, CRN) rollouts use
    :func:`required_episodes_paired`, which applies the ``(1 - rho)``
    variance reduction that shared-seed pairing buys you (B-08).

    Args:
        p1: Baseline success rate in ``[0, 1]``.
        p2: Candidate success rate in ``[0, 1]``. Must differ from
            ``p1``; ``p1 == p2`` would imply infinite samples needed
            and is rejected with :class:`ValueError`.
        alpha: Two-sided significance level in ``(0, 1)``. Default
            ``0.05`` (the standard choice; matches B-03's Wilson
            confidence default).
        power: Statistical power in ``(0, 1)``. Default ``0.8`` (the
            standard convention from Cohen, 1988).

    Returns:
        Minimum episodes per arm, ``>= 1``.

    Raises:
        ValueError: if ``p1 == p2``, if either rate is outside
            ``[0, 1]``, or if ``alpha`` / ``power`` is outside
            ``(0, 1)``.

    Notes:
        Anti-feature reminder (B-41): this answer is the *minimum*
        sample size needed to call the binary success-rate gap
        statistically significant. Running exactly that many episodes
        per cell silently buys you a tight binary summary and a wide-
        open blind spot for failure modes that surface only in the
        long tail. GAUNTLET_SPEC.md §1 ("failures over averages") is
        built around exactly this trap. Use the output as a floor.

    Examples:
        Detect a 10-percentage-point drop from 0.5 → 0.4 at the
        canonical alpha=0.05 / power=0.8::

            >>> required_episodes(0.5, 0.4)
            389
    """
    if not 0.0 <= p1 <= 1.0:
        raise ValueError(f"p1 must be in [0, 1]; got {p1}")
    if not 0.0 <= p2 <= 1.0:
        raise ValueError(f"p2 must be in [0, 1]; got {p2}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1); got {power}")
    if p1 == p2:
        raise ValueError(
            "p1 and p2 must differ — a zero effect size needs an "
            "infinite sample. Pass --detect-delta with a non-zero gap.",
        )

    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_beta = NormalDist().inv_cdf(power)
    p_avg = 0.5 * (p1 + p2)
    delta = abs(p1 - p2)
    n_float = ((z_alpha + z_beta) * math.sqrt(2.0 * p_avg * (1.0 - p_avg))) ** 2 / (delta * delta)
    return max(1, math.ceil(n_float))


def required_episodes_paired(
    p1: float,
    p2: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    rho: float = DEFAULT_PAIRED_RHO,
) -> int:
    """Paired-CRN-aware sample size — :func:`required_episodes` * ``(1 - rho)``.

    Common-random-numbers paired comparison (B-08) shares the
    underlying perturbation seeds across the two policy arms, which
    cancels the within-pair variance and reduces the effective sample
    size needed to call a difference significant. The reduction is the
    standard variance-of-paired-differences formula::

        n_paired = ceil(n_independent * (1 - rho))

    where ``rho`` is the *outcome* correlation between paired episodes
    (NOT the seed correlation, which is always 1.0 by construction —
    the two arms see the exact same cell / episode_index / seed).
    ``rho == 0`` recovers the independent-sample answer; ``rho == 0.5``
    halves it. ``rho`` near ``1.0`` would imply paired episodes succeed
    or fail in lock-step and almost no sample is needed; we accept any
    ``rho`` in ``[0, 1)`` but the empirical value for shared-seed
    rollouts is around :data:`DEFAULT_PAIRED_RHO` (0.5).

    Args:
        p1: Baseline success rate in ``[0, 1]``.
        p2: Candidate success rate in ``[0, 1]``.
        alpha: Two-sided significance level in ``(0, 1)``. Default
            ``0.05``.
        power: Statistical power in ``(0, 1)``. Default ``0.8``.
        rho: Outcome correlation between paired episodes, in
            ``[0, 1)``. Default :data:`DEFAULT_PAIRED_RHO`.

    Returns:
        Minimum episodes per cell under paired-CRN sampling, ``>= 1``.

    Raises:
        ValueError: same triggers as :func:`required_episodes`, plus
            ``rho`` outside ``[0, 1)``.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError(
            f"rho must be in [0, 1); got {rho}. rho == 1 would imply paired "
            "outcomes carry no extra information per pair — degenerate.",
        )
    n_indep = required_episodes(p1, p2, alpha=alpha, power=power)
    return max(1, math.ceil(n_indep * (1.0 - rho)))
