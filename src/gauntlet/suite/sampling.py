"""Sampler protocol + concrete strategies for :class:`Suite.cells`.

See ``docs/polish-exploration-lhs-sampling.md`` for the design rationale.
This module is the dispatch layer behind :attr:`Suite.sampling`:

* ``"cartesian"`` (default) — :class:`CartesianSampler`, the existing
  :func:`itertools.product` enumeration. Byte-identical to the
  pre-LHS behaviour.
* ``"latin_hypercube"`` — :class:`LatinHypercubeSampler` (added in a
  follow-up step in this same task series).
* ``"sobol"`` — :class:`SobolSampler` (Joe-Kuo 6.21201
  low-discrepancy sequence; see ``docs/polish-exploration-sobol-sampler.md``).
* ``"adversarial"`` — :class:`AdversarialSampler` (B-07) — Thompson-
  sampling bandit over the perturbation hypercube, conditioned on a
  pilot run's :class:`gauntlet.report.schema.Report`. Biases coverage
  toward high-failure bins; do not use for benchmark reporting.

The :class:`Sampler` protocol is intentionally minimal: ``sample(suite,
rng)`` returns a list of :class:`SuiteCell` records. The Runner is
unaware of which sampler produced the list — every downstream consumer
keys off :attr:`SuiteCell.index` exactly as before.

The RNG passed to :meth:`Sampler.sample` is owned by the caller. For
the default :meth:`Suite.cells` entry point, it is seeded from
:attr:`Suite.seed` (or OS entropy when ``seed is None``) so two
:meth:`Suite.cells` calls on the same suite produce the same cells.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from gauntlet.suite.schema import Suite, SuiteCell

__all__ = [
    "CartesianSampler",
    "Sampler",
    "build_sampler",
]


@runtime_checkable
class Sampler(Protocol):
    """Strategy: turn a :class:`Suite` into a flat list of grid cells.

    Implementations must be pure (no I/O, no global state) and
    deterministic for a fixed ``rng`` — re-running with the same
    :class:`numpy.random.Generator` (seeded with the same int) must
    yield the same list.

    Cartesian sampling ignores ``rng`` because the grid is fully
    determined by per-axis enumerations; LHS / Sobol consume it.
    """

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        """Return every cell this strategy emits for ``suite``."""
        ...


class CartesianSampler:
    """The pre-LHS Cartesian-grid enumeration, factored out as a strategy.

    Wraps the existing :func:`itertools.product` over per-axis
    :meth:`AxisSpec.enumerate` calls. Every aspect of the output is
    byte-identical to the historical :meth:`Suite.cells` body — same
    axis order, same rightmost-axis-varies-fastest sequence, same
    zero-based contiguous :attr:`SuiteCell.index` values.

    ``rng`` is unused (cartesian sampling is deterministic from the
    suite alone) but kept on the signature so this class satisfies
    :class:`Sampler`.
    """

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        """Enumerate every axis-value combination as a flat cell list.

        ``rng`` is accepted for protocol conformance and ignored.
        """
        # Imported lazily to avoid a circular import: schema imports
        # this module to dispatch ``Suite.cells``, and this method is
        # only called from inside that dispatch path.
        from gauntlet.suite.schema import SuiteCell

        del rng  # cartesian sampling does not consume entropy
        axis_names = tuple(suite.axes.keys())
        per_axis_values = tuple(spec.enumerate() for spec in suite.axes.values())
        out: list[SuiteCell] = []
        for index, combo in enumerate(itertools.product(*per_axis_values)):
            mapping: dict[str, float] = dict(zip(axis_names, combo, strict=True))
            out.append(SuiteCell(index=index, values=mapping))
        return out


def build_sampler(mode: str, *, suite: Suite | None = None) -> Sampler:
    """Return the :class:`Sampler` for a given :attr:`Suite.sampling` value.

    The optional ``suite`` argument is only consulted by the
    ``"adversarial"`` branch — it needs the suite's
    :attr:`Suite.pilot_report` path to load the pilot
    :class:`gauntlet.report.schema.Report`. Every other branch ignores
    it and the kwarg defaults to ``None`` so existing callers (CLI,
    tests, downstream wrappers) keep their two-line signature.

    Raises:
        ValueError: if ``mode`` is not one of the recognised
            strategies, or if ``mode == "adversarial"`` and ``suite``
            is None / missing :attr:`Suite.pilot_report`. The
            :class:`Suite` schema already restricts ``sampling`` to
            the literal set, so the unknown-mode branch is
            defence-in-depth for direct callers that bypass schema
            validation.
    """
    builtin_modes = ("cartesian", "latin_hypercube", "sobol", "adversarial")

    # Check for built-in / plugin collision BEFORE dispatching, so a
    # third-party sampler that re-registers a built-in name surfaces
    # the same RuntimeWarning the policy / env / axis registries emit.
    # Identity collisions stay silent — gauntlet's own dogfooded
    # ``cartesian`` entry point loads ``CartesianSampler`` itself.
    from gauntlet.plugins import (
        SAMPLER_ENTRY_POINT_GROUP,
        discover_sampler_plugins,
        warn_on_collision,
    )

    plugins = discover_sampler_plugins()
    if mode in builtin_modes and mode in plugins:
        # The plugin object the third party shipped vs the built-in
        # class we resolve below. ``warn_on_collision`` is identity-
        # based — same shape as the policy / env / axis registries —
        # so the dogfooded ``cartesian = "..:CartesianSampler"`` entry
        # point never warns.
        warn_on_collision(
            name=mode,
            group=SAMPLER_ENTRY_POINT_GROUP,
            builtin_obj=_builtin_sampler_class(mode),
            plugin_obj=plugins[mode],
            plugin_dist="<plugin>",
        )

    if mode == "cartesian":
        return CartesianSampler()
    if mode == "latin_hypercube":
        # Local import dodges the schema <-> sampling cycle: schema
        # imports this module to dispatch ``Suite.cells``, and the
        # sampler subclasses each defer their ``SuiteCell`` import to
        # call time for the same reason.
        from gauntlet.suite.lhs import LatinHypercubeSampler

        return LatinHypercubeSampler()
    if mode == "sobol":
        # Same lazy-import rationale as LHS above.
        from gauntlet.suite.sobol import SobolSampler

        return SobolSampler()
    if mode == "adversarial":
        # Same lazy-import rationale.
        from gauntlet.suite.adversarial import AdversarialSampler, load_pilot_report

        if suite is None or suite.pilot_report is None:
            raise ValueError(
                "build_sampler(mode='adversarial') requires a Suite "
                "with pilot_report set; the schema validator should "
                "have caught this. Was the Suite constructed bypassing "
                "model_validate?",
            )
        return AdversarialSampler(load_pilot_report(suite.pilot_report))
    # Plugin fallthrough — third-party samplers registered under the
    # ``gauntlet.samplers`` entry-point group.
    if mode in plugins:
        return plugins[mode]()
    raise ValueError(
        f"unknown sampling mode {mode!r}; expected one of "
        f"{set(builtin_modes)} or a registered plugin "
        f"(installed plugins: {sorted(plugins)})",
    )


def _builtin_sampler_class(mode: str) -> type[Sampler]:
    """Return the class object backing built-in sampling ``mode``.

    Identity-based collision detection in :func:`build_sampler` needs
    to compare the third-party plugin's loaded class against the same
    object the built-in dispatch will instantiate. Each branch lazy-
    imports the matching class so this helper does NOT widen the
    module's import cost — the same lazy-import discipline the main
    dispatcher follows.
    """
    if mode == "cartesian":
        return CartesianSampler
    if mode == "latin_hypercube":
        from gauntlet.suite.lhs import LatinHypercubeSampler

        return LatinHypercubeSampler
    if mode == "sobol":
        from gauntlet.suite.sobol import SobolSampler

        return SobolSampler
    if mode == "adversarial":
        from gauntlet.suite.adversarial import AdversarialSampler

        return AdversarialSampler
    raise ValueError(f"_builtin_sampler_class: not a built-in mode: {mode!r}")
