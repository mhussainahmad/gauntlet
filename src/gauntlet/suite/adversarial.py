"""Adversarial / failure-seeking sampler — Thompson-sampling bandit over bins.

See ``docs/backlog.md`` B-07 (RoboMD, arXiv 2412.02818) for the design
rationale: a pilot Sobol / LHS / cartesian run produces a
:class:`gauntlet.report.schema.Report`; this sampler bins the
perturbation hypercube, fits a per-bin Beta posterior over its observed
(success, failure) counts, and concentrates new samples in the bins the
posterior thinks are most likely to fail.

ANTI-FEATURE WARNING. Adversarial sampling deliberately biases coverage
toward known-failure regions of the perturbation space. The resulting
report is **not** a fair sample of the suite — every per-axis marginal
and per-cluster statistic is skewed by construction. Use this mode only
for failure-mode discovery; never quote its overall success rate as a
benchmark number. The suite loader prints a loud
:class:`UserWarning` whenever a YAML opts in.

Algorithm (one paragraph):

1. For each axis, build an integer bin index in ``[0, K_axis)`` where
   ``K_axis`` is :data:`_BINS_PER_CONTINUOUS_AXIS` for continuous axes
   and ``len(values)`` for categorical axes. Bin keys are tuples of
   per-axis indices, so the joint hypercube has at most
   ``prod(K_axis)`` bins.
2. Walk the pilot report's :attr:`Report.per_cell` list. For each
   pilot cell, map ``perturbation_config[axis]`` back to its bin
   index via :func:`_axis_value_to_bin`, and accumulate
   ``(successes, failures)`` for that bin.
3. For each new sample (``i`` in ``range(n_samples)``), draw one
   posterior failure-rate per bin from ``Beta(failures + 1, successes
   + 1)`` (uniform Beta(1, 1) prior on bins the pilot never visited),
   pick the bin with the highest sampled rate (Thompson sampling), and
   draw uniformly inside that bin's hypercube — same
   ``_axis_value_from_unit`` mapping as :class:`SobolSampler`.

Reproducibility: a fresh ``np.random.Generator(seed)`` produces the
same ``n_samples`` rows. Two ``Suite.cells()`` calls on the same
adversarial suite return the same cells.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from gauntlet.report.schema import Report
    from gauntlet.suite.schema import AxisSpec, Suite, SuiteCell

__all__ = [
    "ADVERSARIAL_WARNING",
    "AdversarialSampler",
    "load_pilot_report",
]


# Number of equal-width bins per continuous axis. Four is the smallest
# value that gives a non-trivial Thompson posterior surface (2 splits
# per axis = quartiles) without exploding the joint bin count for
# 5-7-axis suites: ``4 ^ 5 = 1024`` bins, ``4 ^ 7 = 16_384`` bins.
# Tunable through the constructor for callers that want finer cells.
_BINS_PER_CONTINUOUS_AXIS: Final[int] = 4


# Loud anti-feature warning — surfaced from the suite loader and from
# direct ``Suite.cells()`` calls so any path into adversarial sampling
# carries the same caveat. Keep the wording stable so test assertions
# can substring-match without hardcoding the full text.
ADVERSARIAL_WARNING: Final[str] = (
    "adversarial sampling biases coverage toward known-failure regions; "
    "resulting report is NOT a fair sample of the perturbation space "
    "-- use only for failure-mode discovery, not for benchmark reporting."
)


def load_pilot_report(path: Path | str) -> Report:
    """Load a pilot ``report.json`` from disk into a :class:`Report`.

    The loader uses :meth:`Report.model_validate_json` so old report
    files (pre-B-03 / pre-B-19 / pre-B-26) round-trip via the optional
    fields without rejection. Raises :class:`FileNotFoundError` on
    missing files; ``pydantic.ValidationError`` if the file exists but
    does not parse as a Report.
    """
    # Local import avoids a top-level cycle: gauntlet.report imports
    # gauntlet.suite types in some helpers.
    from gauntlet.report.schema import Report

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"adversarial sampler: pilot report not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return Report.model_validate_json(fh.read())


def _axis_value_to_bin(spec: AxisSpec, value: float, n_bins: int) -> int:
    """Map an axis value back to its bin index in ``[0, n_bins)``.

    Categorical axes ignore ``n_bins`` and use ``len(values)`` directly:
    each declared value is its own bin (the index in the ``values``
    list). Continuous axes invert :func:`_axis_value_from_unit` —
    affine to ``[0, 1)`` then floor to a bin index, clamped so the
    inclusive ``high`` endpoint maps to the last bin.
    """
    if spec.values is not None:
        # Categorical: snap to the index of the closest declared value.
        # Float equality after a JSON round-trip is fragile; closest-
        # match keeps the inversion robust to ``Report.model_validate``
        # serialising-deserialising 1.0 as 1.0 vs 1 etc.
        choices = [float(v) for v in spec.values]
        return int(min(range(len(choices)), key=lambda i: abs(choices[i] - value)))
    assert spec.low is not None
    assert spec.high is not None
    if spec.low == spec.high:
        return 0
    u = (value - spec.low) / (spec.high - spec.low)
    # Clamp to [0, 1] before binning so an inclusive high endpoint
    # (u == 1.0 exactly) maps to bin n_bins - 1, not n_bins.
    u_clamped = min(max(u, 0.0), 1.0 - 1e-12)
    return int(min(int(u_clamped * n_bins), n_bins - 1))


def _axis_value_from_unit(spec: AxisSpec, u: float) -> float:
    """Map a unit-cube draw onto the axis's value space.

    Identical contract to :func:`gauntlet.suite.sobol._axis_value_from_unit`
    and :func:`gauntlet.suite.lhs._axis_value_from_unit` — the three
    mappings stay in lock-step so a bin computed under one sampler can
    be inverted by any other.
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


class AdversarialSampler:
    """Thompson-sampling bandit sampler keyed off :attr:`Suite.n_samples`.

    Constructor args:
        pilot_report: Already-loaded :class:`Report` from a prior
            (uninformative) sweep. Its ``per_cell`` list is the
            evidence the bandit conditions on. Cells with axes the
            suite does not declare are dropped silently; cells missing
            any of the suite's axes raise :class:`ValueError` (silent
            skip would be exactly the cherry-picking footgun the
            anti-feature note warns about).
        bins_per_continuous: Bins per continuous axis. Default
            :data:`_BINS_PER_CONTINUOUS_AXIS` (= 4); raise for finer
            resolution at the cost of needing more pilot data.

    The ``rng`` argument on :meth:`sample` is the Beta-sampling
    entropy source. Two calls with two ``np.random.Generator``
    instances seeded with the same int produce identical lists.
    """

    def __init__(
        self,
        pilot_report: Report,
        *,
        bins_per_continuous: int = _BINS_PER_CONTINUOUS_AXIS,
    ) -> None:
        if bins_per_continuous < 2:
            raise ValueError(f"bins_per_continuous must be >= 2; got {bins_per_continuous}")
        self._pilot = pilot_report
        self._bins_per_continuous = bins_per_continuous

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        """Emit ``suite.n_samples`` adversarial cells.

        Raises:
            ValueError: when ``suite.n_samples is None`` or any pilot
                cell lacks a value for one of the suite's axes.
        """
        from gauntlet.suite.schema import SuiteCell

        n = suite.n_samples
        if n is None:
            raise ValueError(
                "AdversarialSampler requires Suite.n_samples; "
                "the schema validator should have caught this. "
                "Was the Suite constructed bypassing model_validate?",
            )
        # Loud anti-feature warning every time the sampler runs (the
        # loader emits one too at YAML-load time, but ``Suite.cells()``
        # is the path that actually biases the run).
        warnings.warn(ADVERSARIAL_WARNING, UserWarning, stacklevel=2)

        axis_names = tuple(suite.axes.keys())
        axis_specs = tuple(suite.axes.values())
        bin_counts = self._bin_counts(axis_specs)

        # Tally (failures, successes) per bin from the pilot report.
        # Using float arrays keeps the +1 Beta-prior addition cheap.
        fails, succs = self._tally_pilot(suite)

        # Pre-flatten to a single 1D index space so the per-sample
        # Beta draw is one numpy call, then unravel back to a tuple of
        # per-axis bin indices.
        flat_fails = fails.reshape(-1)
        flat_succs = succs.reshape(-1)

        out: list[SuiteCell] = []
        for row_idx in range(n):
            # Thompson sampling: one Beta draw per bin, pick argmax.
            # Beta(fails+1, succs+1) collapses to Beta(1, 1) =
            # Uniform[0, 1] for unobserved bins.
            sampled_rate = rng.beta(flat_fails + 1.0, flat_succs + 1.0)
            chosen = int(np.argmax(sampled_rate))
            bin_idx = np.unravel_index(chosen, bin_counts)
            mapping: dict[str, float] = {}
            for col_idx, (name, spec) in enumerate(zip(axis_names, axis_specs, strict=True)):
                k = int(bin_idx[col_idx])
                k_total = bin_counts[col_idx]
                # Uniform draw inside the chosen bin: u = (k + frac) / K
                # where frac is uniform [0, 1). For categorical axes
                # K = len(values) and the per-bin mapping degenerates
                # to "pick that exact value" via _axis_value_from_unit's
                # categorical branch.
                frac = float(rng.uniform())
                u = (k + frac) / k_total
                mapping[name] = _axis_value_from_unit(spec, u)
            out.append(SuiteCell(index=row_idx, values=mapping))
        return out

    # ------------------------------------------------------------------
    # Internals — pure helpers, broken out for unit-testability.
    # ------------------------------------------------------------------

    def _bin_counts(self, specs: tuple[AxisSpec, ...]) -> tuple[int, ...]:
        """Per-axis bin count tuple — categorical = len(values), else K."""
        out: list[int] = []
        for spec in specs:
            if spec.values is not None:
                out.append(len(spec.values))
            else:
                out.append(self._bins_per_continuous)
        return tuple(out)

    def _tally_pilot(self, suite: Suite) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Bucket pilot per-cell episodes into ``(fails, succs)`` arrays.

        The arrays have shape ``bin_counts`` (one axis per declared
        suite axis). Pilot cells whose ``perturbation_config`` is
        missing a suite axis raise :class:`ValueError` — silent
        coverage loss would be exactly the failure mode this sampler
        is meant to surface.
        """
        axis_names = tuple(suite.axes.keys())
        axis_specs = tuple(suite.axes.values())
        bin_counts = self._bin_counts(axis_specs)
        fails = np.zeros(bin_counts, dtype=np.float64)
        succs = np.zeros(bin_counts, dtype=np.float64)
        for cell in self._pilot.per_cell:
            cfg = cell.perturbation_config
            missing = [name for name in axis_names if name not in cfg]
            if missing:
                raise ValueError(
                    "adversarial sampler: pilot report cell "
                    f"{cell.cell_index} is missing axes {missing}; "
                    "the pilot must cover every suite axis. Drop the "
                    "axis from the suite or re-run the pilot with the "
                    "axis included.",
                )
            bin_key = tuple(
                _axis_value_to_bin(spec, cfg[name], bin_counts[col_idx])
                for col_idx, (name, spec) in enumerate(zip(axis_names, axis_specs, strict=True))
            )
            n_succ = cell.n_success
            n_fail = cell.n_episodes - cell.n_success
            succs[bin_key] += float(n_succ)
            fails[bin_key] += float(n_fail)
        return fails, succs
