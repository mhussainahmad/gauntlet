"""Per-axis Sobol sensitivity indices computed from rollout outcomes.

Implements the closed-form Saltelli decomposition of the success-rate
variance — see "Global Sensitivity Analysis: The Primer" (Saltelli et
al., 2008) — applied to a binary outcome ``Y in {0, 1}`` over the
perturbation grid ``X = (X_0, ..., X_{d-1})``:

* **First-order**  ``S_i = Var_X[E[Y | X_i]] / Var(Y)``. Read as "the
  fraction of the outcome variance explained by axis ``i`` alone".
* **Total-order**  ``S_T_i = 1 - Var_X[E[Y | X_~i]] / Var(Y)``. Read as
  "the fraction of variance that disappears when ``i`` is held fixed";
  always ``>= S_i`` and the gap is the share of variance carried by
  interactions involving axis ``i``.

Conditional means are weighted by the population that actually landed
in each bucket, not by the number of distinct bucket values — matters
when the suite is non-rectangular (e.g. Sobol-sampled, where bucket
counts are not uniform).

Edge cases:

* Axis with a single value contributes no variance by construction;
  both indices are pinned to ``0.0`` rather than the closed-form
  ``S_T_i`` (which would otherwise leak the "noise within the one
  cell" into a non-zero number and confuse the reader).
* All-success / all-fail run has ``Var(Y) == 0`` and the indices are
  undefined; both fields return ``None`` and the caller surfaces them
  as "no signal" rather than 0.

The module is pure: numpy only, no scipy.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from gauntlet.runner.episode import Episode

__all__ = ["compute_sobol_indices"]


def _conditional_variance(
    groups: Iterable[tuple[int, int]],
    overall_mean: float,
) -> float:
    """Population-weighted variance of conditional means.

    ``groups`` yields ``(n_in_group, n_success_in_group)`` pairs; the
    conditional mean for the group is ``n_success / n``. The returned
    value is ``Σ (n_g / N) * (mean_g - overall_mean) ** 2`` — the
    population-weighted (not equal-weighted) variance per Saltelli.
    """
    pairs = [(n, k) for n, k in groups if n > 0]
    if not pairs:
        return 0.0
    total = sum(n for n, _ in pairs)
    if total == 0:
        return 0.0
    weighted = 0.0
    for n, k in pairs:
        mean_g = k / n
        weighted += (n / total) * (mean_g - overall_mean) ** 2
    return weighted


def compute_sobol_indices(
    episodes: list[Episode],
    axis_names: tuple[str, ...],
) -> dict[str, tuple[float | None, float | None]]:
    """Return ``{axis_name: (first_order, total_order)}``.

    Each value is ``(None, None)`` when ``Var(Y) == 0`` (all-success or
    all-fail), and ``(0.0, 0.0)`` when the axis carries a single value.
    Otherwise the first-order index is in ``[0, 1]`` (clamped to absorb
    floating-point drift below zero) and the total-order index is in
    ``[0, 1]`` (clamped at both ends).
    """
    if not episodes or not axis_names:
        return {}

    y = np.asarray([1 if ep.success else 0 for ep in episodes], dtype=np.float64)
    overall = float(y.mean())
    var_y = float(y.var())

    out: dict[str, tuple[float | None, float | None]] = {}
    for axis in axis_names:
        # Group by this axis only (first-order) and by all-other-axes
        # (total-order) in one pass.
        first_groups: dict[float, list[int]] = defaultdict(lambda: [0, 0])
        total_groups: dict[tuple[tuple[str, float], ...], list[int]] = defaultdict(lambda: [0, 0])
        n_axis_seen = 0
        for ep in episodes:
            cfg = ep.perturbation_config
            if axis not in cfg:
                continue
            n_axis_seen += 1
            y_i = 1 if ep.success else 0
            v_axis = float(cfg[axis])
            bucket_first = first_groups[v_axis]
            bucket_first[0] += 1
            bucket_first[1] += y_i
            others = tuple(sorted((k, float(v)) for k, v in cfg.items() if k != axis))
            bucket_total = total_groups[others]
            bucket_total[0] += 1
            bucket_total[1] += y_i

        # Var(Y) == 0 → indices undefined (all-success or all-fail).
        if var_y == 0.0:
            out[axis] = (None, None)
            continue
        # Axis with a single value contributes no variance — pin to 0.
        if len(first_groups) <= 1:
            out[axis] = (0.0, 0.0)
            continue
        # Axis missing from every episode (defensive — Runner never
        # produces this) is also pinned at 0.
        if n_axis_seen == 0:
            out[axis] = (0.0, 0.0)
            continue

        var_first = _conditional_variance(
            ((n, k) for n, k in first_groups.values()),
            overall,
        )
        var_others = _conditional_variance(
            ((n, k) for n, k in total_groups.values()),
            overall,
        )
        s_i = var_first / var_y
        s_t = 1.0 - var_others / var_y
        # Clamp tiny negative drift to zero; clamp total-order to [0, 1]
        # so a one-cell-per-others bucket (where ``var_others == var_y``
        # exactly) reads as ``0.0``, not ``-0.0`` or ``1e-17``.
        out[axis] = (max(0.0, min(1.0, s_i)), max(0.0, min(1.0, s_t)))

    return out
