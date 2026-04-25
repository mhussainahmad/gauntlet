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

B-42 anti-feature warning (SO(3) bias)
--------------------------------------
The ``camera_extrinsics`` axis carries a 6-D pose delta (translation
+ XYZ-Euler rotation) collapsed to a categorical-by-index for the
cell-value channel. The closed-form Saltelli decomposition treats
each unique index as a discrete bucket — but the rotation half of
each bucket is a non-linearly-composing rotation in SO(3), so the
per-bucket conditional mean does NOT decompose into the per-rotation-
component contributions. The reported per-axis indices on
``camera_extrinsics`` are therefore an *indicator* of viewpoint
sensitivity, not a calibrated estimate. :func:`compute_sobol_indices`
emits a :class:`UserWarning` whenever the axis is present in the
input episode set so consumers (CLI, HTML report) can surface the
caveat to readers.

The module is pure: numpy only, no scipy.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from gauntlet.runner.episode import Episode

__all__ = ["CAMERA_EXTRINSICS_SO3_WARNING", "compute_sobol_indices"]


# B-42 — SO(3) bias warning. Surfaced once per
# :func:`compute_sobol_indices` call when the ``camera_extrinsics``
# axis is in the episode set; the HTML report consumer reads the
# warning text via :func:`warnings.catch_warnings` to render the
# banner row above the sensitivity-index chart.
CAMERA_EXTRINSICS_SO3_WARNING: str = (
    "Sobol indices on the 'camera_extrinsics' axis (B-42) are biased: "
    "each cell value is a categorical index into a structured 6-D pose "
    "delta (translation + XYZ-Euler rotation), and SO(3) rotations do "
    "NOT compose linearly — so the closed-form Saltelli decomposition "
    "treats every unique extrinsics bucket as discrete and discards the "
    "per-rotation-component variance structure. Treat the reported "
    "S_i / S_T_i on this axis as a viewpoint-sensitivity indicator, not "
    "a calibrated estimate. References: RoboView-Bias (arXiv 2509.22356), "
    "'Do You Know Where Your Camera Is?' (arXiv 2510.02268)."
)


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

    # B-42 — emit the SO(3)-bias warning once per call when the
    # ``camera_extrinsics`` axis is present in the requested axis set.
    # We check ``axis_names`` (not the per-episode config) because the
    # caller controls which axes get scored — checking the episode
    # config would also fire for a side-channel axis that the caller
    # explicitly excluded from this run's report.
    if "camera_extrinsics" in axis_names:
        warnings.warn(CAMERA_EXTRINSICS_SO3_WARNING, UserWarning, stacklevel=2)

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
