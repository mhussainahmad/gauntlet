"""Latin Hypercube Sampling (LHS) sampler — McKay/Beckman/Conover (1979).

See ``docs/polish-exploration-lhs-sampling.md`` for the design rationale
and the unit-cube → axis-value mapping rules.

The algorithm in six lines:

1. Take ``n = suite.n_samples`` samples across ``d = len(suite.axes)``
   axes.
2. Per axis, divide ``[0, 1)`` into ``n`` equal-width strata; draw one
   uniform sample inside each stratum: ``u_i = (i + uniform()) / n``.
3. Independently permute the per-axis stratum index — this decouples
   the marginals so each axis covers all ``n`` strata exactly once
   while the joint distribution is randomised.
4. Map each unit-cube point ``u in [0, 1)`` onto its axis spec:
   continuous → affine ``low + u * (high - low)``; integer-valued
   continuous (caught at the AxisSpec layer by sentinel handling)
   rounds to int via the affine map; categorical
   ``values[min(int(u * K), K - 1)]``.

Reproducibility: a fresh ``np.random.Generator(seed=s)`` produces the
same ``(n_samples, n_axes)`` matrix and therefore the same ``SuiteCell``
list. The ``Suite.cells`` dispatch layer seeds this RNG from
``suite.seed`` so two ``Suite.cells()`` calls on the same suite return
the same cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gauntlet.suite.schema import AxisSpec, Suite, SuiteCell

__all__ = ["LatinHypercubeSampler", "lhs_unit_cube"]


def lhs_unit_cube(
    n_samples: int,
    n_axes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw an LHS matrix in ``[0, 1)^n_axes`` with ``n_samples`` rows.

    The McKay 1979 stratification: each axis has every one of
    ``n_samples`` equal-width strata covered exactly once; per-axis
    permutations decorrelate the marginals.

    Args:
        n_samples: Number of points (rows). Must be ``>= 1``.
        n_axes: Number of dimensions (columns). Must be ``>= 1``.
        rng: numpy ``Generator`` — the only source of entropy.

    Returns:
        ``(n_samples, n_axes)`` float array. Every value lies in
        ``[0, 1)``; per axis, the column contains exactly one value
        in each stratum ``[i / n_samples, (i + 1) / n_samples)``.

    Raises:
        ValueError: if ``n_samples < 1`` or ``n_axes < 1``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}")
    if n_axes < 1:
        raise ValueError(f"n_axes must be >= 1; got {n_axes}")
    # Step 1 + 2: jittered stratum centres in [0, 1). The (i + uniform)
    # construction guarantees each row index ``i`` falls in stratum
    # ``[i/n, (i+1)/n)`` *before* the per-axis permutation reshuffles
    # which row gets which stratum.
    cuts = (np.arange(n_samples)[:, None] + rng.uniform(size=(n_samples, n_axes))) / n_samples
    # Step 3: independent permutations per axis. ``rng.shuffle`` is
    # in-place and reads from a single Generator, so the per-axis
    # streams are correlated through the RNG state — that is the
    # standard McKay construction (no additional decorrelation needed).
    for axis in range(n_axes):
        rng.shuffle(cuts[:, axis])
    return cuts


def _axis_value_from_unit(spec: AxisSpec, u: float) -> float:
    """Map a unit-cube draw onto the axis's value space.

    Three cases driven off the :class:`AxisSpec` shape (the same
    information the cartesian sampler keys off):

    * ``values is not None`` (categorical): one of the ``K = len(values)``
      categories, chosen by the unit interval being divided into ``K``
      equal-width sub-strata. Equivalent to ``values[min(int(u * K),
      K - 1)]``; the ``min`` guards against the (impossible-in-practice)
      ``u == 1.0`` boundary.
    * ``values is None`` and ``low == high``: degenerate continuous
      axis — single legal value. Returns ``low``.
    * ``values is None`` otherwise: affine map ``low + u * (high -
      low)``. Inclusive lower, exclusive upper, matching
      ``np.random.Generator.uniform(low, high)`` semantics.

    Integer axes (``distractor_count``) flow through the same affine
    branch — the env-side handler rounds to int, exactly as the
    cartesian path does (see :class:`gauntlet.env.perturbation.PerturbationAxis`).
    Returning a float keeps the :class:`SuiteCell.values` mapping
    homogeneous in type.
    """
    if spec.values is not None:
        choices = spec.values
        idx = min(int(u * len(choices)), len(choices) - 1)
        return float(choices[idx])
    # Continuous: low + high are guaranteed present (axis-level
    # validator) for non-cartesian YAMLs; ``steps`` is unused here
    # (LHS budget lives on the Suite, not on the axis).
    assert spec.low is not None
    assert spec.high is not None
    if spec.low == spec.high:
        return float(spec.low)
    return float(spec.low + u * (spec.high - spec.low))


class LatinHypercubeSampler:
    """McKay 1979 LHS sampler keyed off :attr:`Suite.n_samples`.

    Emits exactly ``suite.n_samples`` :class:`SuiteCell` records,
    each with one value per axis drawn from the axis spec via the
    unit-cube → axis-value mapping documented in
    :func:`_axis_value_from_unit`.

    The sampler is stateless; the only state it consumes is the RNG
    passed to :meth:`sample`. Two calls with two ``np.random.Generator``
    instances seeded with the same int produce identical lists.
    """

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        # Schema-level validators guarantee n_samples is a positive int
        # whenever ``sampling != "cartesian"`` reaches this method, but
        # we still narrow defensively — direct ``Suite(...)`` calls
        # bypass the loader-side path.
        from gauntlet.suite.schema import SuiteCell

        n = suite.n_samples
        if n is None:
            raise ValueError(
                "LatinHypercubeSampler requires Suite.n_samples; "
                "the schema validator should have caught this. "
                "Was the Suite constructed bypassing model_validate?",
            )
        axis_names = tuple(suite.axes.keys())
        axis_specs = tuple(suite.axes.values())
        unit = lhs_unit_cube(n, len(axis_names), rng)

        out: list[SuiteCell] = []
        for row_idx in range(n):
            mapping: dict[str, float] = {
                name: _axis_value_from_unit(spec, float(unit[row_idx, col_idx]))
                for col_idx, (name, spec) in enumerate(zip(axis_names, axis_specs, strict=True))
            }
            out.append(SuiteCell(index=row_idx, values=mapping))
        return out
