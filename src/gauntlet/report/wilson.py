"""Wilson score confidence intervals for a binomial proportion.

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

The single public entry point is :func:`wilson_interval`.
"""

from __future__ import annotations

import math

__all__ = [
    "DEFAULT_CONFIDENCE",
    "Z_95",
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
