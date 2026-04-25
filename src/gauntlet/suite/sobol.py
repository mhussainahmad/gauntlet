"""Sobol low-discrepancy sequence sampler — Joe-Kuo 6.21201.

See ``docs/polish-exploration-sobol-sampler.md`` for the design note
and the rationale for the embedded direction-number table.

Algorithm (Bratley-Fox 1988 + Joe-Kuo 2008 direction numbers):

1. For each dimension, build the 32-element direction-number table
   ``V[j]`` from the inline ``(s, a, m_i)`` triple. Dimension 0 is
   the parameter-free van der Corput sequence
   (``V[0, i] = 2^(31 - i)``); dimensions 1..N use the Joe-Kuo
   recurrence:

       m_i = 2^s * m_{i-s} XOR m_{i-s}
                XOR \\sum_{k=1..s-1} 2^k * a_k * m_{i-k}

   where ``a_k`` is the ``k``-th bit of the integer ``a`` (MSB-first
   per the Joe-Kuo convention; ``a_0`` and ``a_s`` are implicitly 1
   and absorbed into the recurrence).

2. Iterate Gray-code style: for sample ``k >= 1``, find ``c`` =
   position of the lowest 0-bit of ``k - 1``; XOR ``V[:, c]`` into
   the running state ``X``; emit ``X / 2^32``.

The leading sample (``k = 0``) is the origin ``(0, 0, ..., 0)``;
``skip = 1`` (the default in :func:`sobol_unit_cube`) drops it
because dropping the leading origin improves uniformity for small
``n_samples`` (Owen 2003). See the design note for why we don't use
the popular ``skip = n_samples - 1`` instead.

Reproducibility: the Sobol sequence is **fully deterministic** —
the algorithm consumes no entropy. The :class:`SobolSampler` accepts
an :class:`numpy.random.Generator` purely to satisfy the
:class:`gauntlet.suite.sampling.Sampler` protocol; the generator is
ignored at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from gauntlet.suite.schema import AxisSpec, Suite, SuiteCell

__all__ = ["SobolSampler", "sobol_unit_cube"]


# Bit width of the direction-number representation. 32 bits gives
# every emitted point a denominator of 2^32 = 4_294_967_296, far more
# than enough resolution for perturbation grids that bottom out at
# float32 cameras and sensor noise. The Bratley-Fox iteration burns
# constant memory regardless of ``n_samples``.
_BITS: Final[int] = 32


# ============================================================================
# Joe-Kuo 6.21201 direction-number table — first 20 dimensions.
#
# Pulled verbatim from
# http://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201
# (public domain). The upstream file is whitespace-delimited
# ``d s a m_i...`` rows starting at ``d = 2`` (dimension 1 is the
# parameter-free van der Corput sequence and is hardcoded below).
#
# Each tuple is ``(s, a, m_init)``:
#   s       — degree of the primitive polynomial.
#   a       — coefficients of the polynomial as a single integer (the
#             MSB-first bit pattern; ``a_0`` and ``a_s`` are implicitly
#             1 and not stored).
#   m_init  — list of length ``s`` of initial direction numbers.
#
# 20 dimensions is comfortable headroom: harness perturbation grids
# rarely exceed 10 axes, and the validator raises a clear error
# pointing users at LHS if they need more.
# ============================================================================
_JOE_KUO_TABLE: Final[tuple[tuple[int, int, tuple[int, ...]], ...]] = (
    (1, 0, (1,)),  # d = 2
    (2, 1, (1, 3)),  # d = 3
    (3, 1, (1, 3, 1)),  # d = 4
    (3, 2, (1, 1, 1)),  # d = 5
    (4, 1, (1, 1, 3, 3)),  # d = 6
    (4, 4, (1, 3, 5, 13)),  # d = 7
    (5, 2, (1, 1, 5, 5, 17)),  # d = 8
    (5, 4, (1, 1, 5, 5, 5)),  # d = 9
    (5, 7, (1, 1, 7, 11, 19)),  # d = 10
    (5, 11, (1, 1, 5, 1, 1)),  # d = 11
    (5, 13, (1, 1, 1, 3, 11)),  # d = 12
    (5, 14, (1, 3, 5, 5, 31)),  # d = 13
    (6, 1, (1, 3, 3, 9, 7, 49)),  # d = 14
    (6, 13, (1, 1, 1, 15, 21, 21)),  # d = 15
    (6, 16, (1, 3, 1, 13, 27, 49)),  # d = 16
    (6, 19, (1, 1, 1, 15, 7, 5)),  # d = 17
    (6, 22, (1, 3, 1, 15, 13, 25)),  # d = 18
    (6, 25, (1, 1, 5, 5, 19, 61)),  # d = 19
    (7, 1, (1, 3, 7, 11, 23, 15, 103)),  # d = 20
    (7, 4, (1, 3, 7, 13, 13, 15, 69)),  # d = 21
)


# Total dimensions covered by the inline table: van der Corput (dim 0)
# plus the 20 Joe-Kuo rows. ``MAX_DIMS = 21`` is the user-facing
# capacity; any Suite requesting more raises a ``ValueError`` pointing
# at LHS as the unbounded-dimension alternative.
MAX_DIMS: Final[int] = 1 + len(_JOE_KUO_TABLE)


def _build_direction_numbers(n_dims: int) -> NDArray[np.uint64]:
    """Build the ``(n_dims, _BITS)`` direction-number matrix.

    Row 0 is van der Corput (``V[0, i] = 1 << (31 - i)``); rows
    ``1..n_dims-1`` are derived from :data:`_JOE_KUO_TABLE` via the
    Joe-Kuo recurrence described at the top of this module.

    The output is ``uint64`` purely for headroom on the
    ``2^k * m`` left-shifts inside the recurrence — each direction
    number itself fits in 32 bits.
    """
    v = np.zeros((n_dims, _BITS), dtype=np.uint64)
    # Dimension 0: van der Corput — V[0, i] = 2^(31 - i) for i in [0, 32).
    for i in range(_BITS):
        v[0, i] = np.uint64(1) << np.uint64(_BITS - 1 - i)
    # Dimensions 1..n_dims-1: Joe-Kuo recurrence.
    for dim in range(1, n_dims):
        s, a, m_init = _JOE_KUO_TABLE[dim - 1]
        # Build M[1..BITS] by extending the s initial values via the
        # recurrence. 1-indexed to match the Joe-Kuo paper notation
        # (M[0] is unused).
        m_ext = [0] * (_BITS + 1)
        for i in range(1, s + 1):
            m_ext[i] = m_init[i - 1]
        for i in range(s + 1, _BITS + 1):
            # Base term: 2^s * m_{i-s} XOR m_{i-s}
            val = ((1 << s) * m_ext[i - s]) ^ m_ext[i - s]
            # Middle terms: XOR in 2^k * a_k * m_{i-k} for k in 1..s-1.
            # Joe-Kuo packs a_1..a_{s-1} MSB-first into ``a``: bit
            # ``s - 1 - k`` (0-indexed) is ``a_k``.
            for k in range(1, s):
                a_k = (a >> (s - 1 - k)) & 1
                if a_k:
                    val ^= (1 << k) * m_ext[i - k]
            m_ext[i] = val
        # V[dim, i-1] = M[i] << (BITS - i) — promote the direction
        # numbers into the top bits of the 32-bit word so the
        # Gray-code XOR builds high-precision points.
        for i in range(1, _BITS + 1):
            v[dim, i - 1] = np.uint64(m_ext[i]) << np.uint64(_BITS - i)
    return v


def sobol_unit_cube(
    n_samples: int,
    n_dims: int,
    *,
    skip: int = 1,
) -> NDArray[np.float64]:
    """Generate ``n_samples`` Sobol points in ``[0, 1)^n_dims``.

    The Joe-Kuo 6.21201 sequence iterated via the standard Bratley-Fox
    1988 Gray-code construction. The leading ``skip`` points are
    discarded before returning — the default ``skip = 1`` drops the
    origin ``(0, 0, ..., 0)`` because Owen 2003 recommends it for
    small ``n_samples`` (it improves marginal uniformity).

    Args:
        n_samples: Number of points (rows). Must be ``>= 1``.
        n_dims: Number of dimensions (columns). Must be ``>= 1`` and
            ``<= MAX_DIMS = 21`` (the inline direction-number table
            covers 21 dimensions; for more, use LHS).
        skip: Number of leading sequence points to discard. Default
            ``1`` (drop the origin). ``0`` keeps the origin; larger
            values consume extra positions of the underlying
            sequence.

    Returns:
        ``(n_samples, n_dims)`` float64 array. Every value lies in
        ``[0, 1)`` — never exactly 1.0 (the algorithm divides by
        ``2^32`` and the high bit cannot be set after the
        Gray-code XOR keeps state below ``2^32``).

    Raises:
        ValueError: if ``n_samples < 1``, ``n_dims < 1``, ``n_dims >
            MAX_DIMS``, or ``skip < 0``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}")
    if n_dims < 1:
        raise ValueError(f"n_dims must be >= 1; got {n_dims}")
    if n_dims > MAX_DIMS:
        raise ValueError(
            f"Sobol sampler ships direction numbers for {MAX_DIMS} dimensions; "
            f"suite uses {n_dims}; extend the table or use LHS"
        )
    if skip < 0:
        raise ValueError(f"skip must be >= 0; got {skip}")

    v = _build_direction_numbers(n_dims)

    total = skip + n_samples
    out = np.empty((n_samples, n_dims), dtype=np.float64)
    state = np.zeros(n_dims, dtype=np.uint64)
    denom = float(1 << _BITS)
    write_idx = 0
    for k in range(total):
        if k > 0:
            # Gray-code position: lowest 0-bit of (k-1).
            kk = k - 1
            c = 0
            while (kk >> c) & 1:
                c += 1
            state ^= v[:, c]
        if k >= skip:
            # state is uint64 but the values are bounded by 2^32 - 1
            # (the direction-number XORs never set bits above bit 31),
            # so casting through float64 / 2^32 is exact.
            out[write_idx, :] = state.astype(np.float64) / denom
            write_idx += 1
    return out


def _axis_value_from_unit(spec: AxisSpec, u: float) -> float:
    """Map a unit-cube draw onto the axis's value space.

    Identical contract to :func:`gauntlet.suite.lhs._axis_value_from_unit`:

    * ``values is not None`` (categorical): ``values[min(int(u * K), K - 1)]``
      where ``K = len(values)``.
    * ``values is None`` and ``low == high``: degenerate; returns ``low``.
    * ``values is None`` otherwise: affine ``low + u * (high - low)``.
      ``u`` is always in ``[0, 1)`` so the upper bound is exclusive,
      matching :func:`numpy.random.Generator.uniform` semantics.
    """
    if spec.values is not None:
        choices = spec.values
        idx = min(int(u * len(choices)), len(choices) - 1)
        return float(choices[idx])
    assert spec.low is not None
    assert spec.high is not None
    if spec.low == spec.high:
        return float(spec.low)
    return float(spec.low + u * (spec.high - spec.low))


class SobolSampler:
    """Joe-Kuo Sobol sampler keyed off :attr:`Suite.n_samples`.

    Emits exactly ``suite.n_samples`` :class:`SuiteCell` records,
    each with one value per axis drawn from the axis spec via the
    unit-cube → axis-value mapping documented in
    :func:`_axis_value_from_unit`.

    The ``rng`` argument on :meth:`sample` is accepted for protocol
    compatibility with :class:`gauntlet.suite.sampling.Sampler` but
    is **ignored**: the Sobol sequence is fully deterministic, and
    seeding has no effect on the emitted points. Two
    ``SobolSampler.sample(suite, rng)`` calls — even with different
    ``rng`` values or different ``Suite.seed`` settings — produce
    byte-identical lists.

    Args:
        skip: Number of leading Sobol-sequence points to discard.
            Default ``1`` (drops the origin ``(0, ..., 0)`` per
            Owen 2003). See :func:`sobol_unit_cube` for the
            rationale.
    """

    def __init__(self, *, skip: int = 1) -> None:
        """Bind ``skip`` (number of leading sequence points to discard).

        Raises:
            ValueError: if ``skip < 0``.
        """
        if skip < 0:
            raise ValueError(f"skip must be >= 0; got {skip}")
        self._skip = skip

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        """Emit ``suite.n_samples`` Joe-Kuo Sobol cells.

        ``rng`` is accepted for protocol conformance and ignored — the
        Sobol sequence is fully deterministic.

        Raises:
            ValueError: when ``suite.n_samples is None``.
        """
        from gauntlet.suite.schema import SuiteCell

        del rng  # Sobol is deterministic; no entropy consumed.

        n = suite.n_samples
        if n is None:
            raise ValueError(
                "SobolSampler requires Suite.n_samples; "
                "the schema validator should have caught this. "
                "Was the Suite constructed bypassing model_validate?",
            )
        axis_names = tuple(suite.axes.keys())
        axis_specs = tuple(suite.axes.values())
        unit = sobol_unit_cube(n, len(axis_names), skip=self._skip)

        out: list[SuiteCell] = []
        for row_idx in range(n):
            mapping: dict[str, float] = {
                name: _axis_value_from_unit(spec, float(unit[row_idx, col_idx]))
                for col_idx, (name, spec) in enumerate(zip(axis_names, axis_specs, strict=True))
            }
            out.append(SuiteCell(index=row_idx, values=mapping))
        return out
