"""Suite + AxisSpec Pydantic models.

See ``GAUNTLET_SPEC.md`` §3 (a Suite is "a collection of
:class:`PerturbationAxis` objects + their grid specification, loaded from
YAML") and §6 (type everything; reproducibility is non-negotiable).

A :class:`Suite` is pure data:

* ``name``, ``env``, ``episodes_per_cell`` and an optional master
  ``seed`` set the experiment identity.
* ``axes`` maps each canonical axis name (see
  :data:`gauntlet.env.perturbation.AXIS_NAMES`) to an :class:`AxisSpec`.
* :meth:`Suite.cells` enumerates the Cartesian product of axis values in
  stable, insertion-order, deterministic-across-runs sequence — the
  Runner (Task 6) consumes this as ``(cell_id, perturbation_values)``.

Two axis spec shapes are accepted (one or the other, never both):

* **Continuous / int**: ``{low, high, steps}`` — ``steps`` evenly spaced
  inclusive points (midpoint when ``steps == 1``).
* **Categorical**: ``{values: [...]}`` — explicit enumeration.

Validation is firm — extras are forbidden, axis names must be canonical,
``low <= high``, ``steps >= 1``, ``episodes_per_cell >= 1``, ``env`` must
match either a currently-registered backend or one of the lazy-import
keys in :data:`BUILTIN_BACKEND_IMPORTS` (``tabletop``,
``tabletop-pybullet``, ``tabletop-genesis``, ``tabletop-isaac``), and
the axes mapping must be non-empty.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gauntlet.env.perturbation import AXIS_NAMES
from gauntlet.env.registry import registered_envs

__all__ = [
    "BUILTIN_BACKEND_IMPORTS",
    "SAMPLING_MODES",
    "AxisSpec",
    "SamplingMode",
    "Suite",
    "SuiteCell",
]


# Public type alias so the loader / Sampler dispatch / docs all key off
# the same string set. Adding a new mode requires touching this literal
# and ``gauntlet.suite.sampling.build_sampler`` together — by design.
SamplingMode = Literal["cartesian", "latin_hypercube", "sobol"]

# Tuple form of the same set, used by validators that need a runtime
# ``in`` check (Pydantic narrows ``Literal`` in the model but the
# loader's defence-in-depth path benefits from a plain tuple).
SAMPLING_MODES: Final[tuple[SamplingMode, ...]] = (
    "cartesian",
    "latin_hypercube",
    "sobol",
)


# Backends that register themselves on import but do NOT import at package
# init time (because they live behind an optional extra). The Suite loader
# imports the value when a YAML names the matching key — see
# ``gauntlet.suite.loader._validate`` and RFC-005 §11.2. Schema validation
# needs this mapping too: ``_env_supported`` accepts these keys before the
# backend has been imported, and the loader converts a failed import into
# a user-facing install-hint error.
BUILTIN_BACKEND_IMPORTS: Final[dict[str, str]] = {
    "tabletop-pybullet": "gauntlet.env.pybullet",
    "tabletop-genesis": "gauntlet.env.genesis",
    "tabletop-isaac": "gauntlet.env.isaac",
}


@dataclass(frozen=True)
class SuiteCell:
    """One point in the suite grid.

    Attributes:
        index: Stable zero-based ordinal — assigned in
            :meth:`Suite.cells` enumeration order. The Runner uses this as
            the cell_id for parallelization and output filenames.
        values: Mapping ``{axis_name: value}`` covering every axis
            declared on the parent :class:`Suite`. Values are floats
            even for integer / categorical axes, matching the
            ``AxisSampler`` protocol return type.
    """

    index: int
    values: Mapping[str, float]


class AxisSpec(BaseModel):
    """Grid specification for a single perturbation axis.

    Exactly one of two shapes is accepted (validated in
    :meth:`_check_shape_exclusive`):

    * Continuous / int: ``low``, ``high``, ``steps`` (all required together).
    * Categorical: ``values`` (a non-empty list).

    Mixing the two raises a clear :class:`pydantic.ValidationError`.

    Attributes:
        low: Inclusive lower bound for the continuous shape.
        high: Inclusive upper bound for the continuous shape.
        steps: Number of evenly spaced points (``>= 1``). When ``steps``
            is ``1``, the single value emitted is the midpoint of
            ``[low, high]``.
        values: Explicit list of values for the categorical shape.
    """

    model_config = ConfigDict(extra="forbid")

    low: float | None = Field(default=None)
    high: float | None = Field(default=None)
    steps: int | None = Field(default=None)
    values: list[float] | None = Field(default=None)

    # ------------------------------------------------------------------
    # Field-level checks. Bounds compatibility (low <= high) is checked
    # in the model-level validator so the error wins even when the
    # individual fields are technically valid in isolation.
    # ------------------------------------------------------------------

    @field_validator("steps")
    @classmethod
    def _steps_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"steps must be >= 1; got {v}")
        return v

    @field_validator("values")
    @classmethod
    def _values_nonempty(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) == 0:
            raise ValueError("values must be a non-empty list when provided")
        return v

    @model_validator(mode="after")
    def _check_shape_exclusive(self) -> AxisSpec:
        has_continuous = any(x is not None for x in (self.low, self.high, self.steps))
        has_categorical = self.values is not None
        if has_continuous and has_categorical:
            raise ValueError(
                "axis spec: cannot mix {low, high, steps} with {values}; "
                "use one shape or the other",
            )
        if not has_continuous and not has_categorical:
            raise ValueError(
                "axis spec: must specify either {low, high, steps} or {values}",
            )
        if has_continuous:
            # ``low`` and ``high`` are always required together for the
            # continuous shape — there is no sensible default for a
            # missing endpoint. ``steps`` is optional at the AxisSpec
            # layer (the parent :class:`Suite` enforces it for
            # ``sampling == "cartesian"``); this lets a non-cartesian
            # YAML write ``{low: 0.3, high: 1.5}`` without dummy steps.
            endpoint_missing = [
                name for name, val in (("low", self.low), ("high", self.high)) if val is None
            ]
            if endpoint_missing:
                raise ValueError(
                    "axis spec: continuous shape requires low, high, steps; "
                    f"missing {endpoint_missing}",
                )
            # Type-narrowing for mypy: validated above.
            assert self.low is not None
            assert self.high is not None
            if self.low > self.high:
                raise ValueError(
                    f"axis spec: low must be <= high; got low={self.low}, high={self.high}",
                )
        return self

    # ------------------------------------------------------------------
    # Enumeration. Returns a tuple so callers can iterate it more than
    # once (Cartesian product needs random access via itertools.product).
    # ------------------------------------------------------------------

    def enumerate(self) -> tuple[float, ...]:
        """Return the ordered grid values for this axis.

        For the categorical shape, this is the ``values`` list (cast to
        float) in declared order.

        For the continuous shape:

        * ``steps == 1`` → single midpoint ``(low + high) / 2``.
        * ``steps >= 2`` → ``steps`` evenly spaced inclusive endpoints,
          ``low + i * (high - low) / (steps - 1)`` for ``i`` in
          ``range(steps)``.
        """
        if self.values is not None:
            return tuple(float(v) for v in self.values)
        # Continuous: validated above to have all three set.
        assert self.low is not None
        assert self.high is not None
        assert self.steps is not None
        steps = self.steps
        lo = self.low
        hi = self.high
        if steps == 1:
            return (0.5 * (lo + hi),)
        denom = float(steps - 1)
        return tuple(lo + i * (hi - lo) / denom for i in range(steps))


class Suite(BaseModel):
    """A named perturbation grid plus run-time knobs.

    The Suite is pure data — no RNG, no I/O, no side effects. The Runner
    (Task 6) derives per-cell seeds from :attr:`seed` to keep
    reproducibility a single-seed concern.

    Attributes:
        name: Non-empty human-readable identifier (becomes part of the
            report filename and surfaces in the HTML output).
        env: Environment slug. Must match a registered backend or one of
            the lazy-import built-ins (:data:`BUILTIN_BACKEND_IMPORTS`).
            Unknown names are rejected with a list of accepted keys.
        episodes_per_cell: How many rollouts to run per grid cell. Must
            be ``>= 1``.
        seed: Optional master seed. ``None`` means "no fixed seed; the
            Runner picks one and records it." When provided, the Runner
            derives per-cell seeds deterministically from this value.
        axes: Mapping ``{axis_name: AxisSpec}``. Insertion order
            (preserved by PyYAML and Pydantic v2) is the axis order used
            by :meth:`cells` and downstream by Runner / Report. Must be
            non-empty and every key must be in
            :data:`gauntlet.env.perturbation.AXIS_NAMES`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    env: str
    episodes_per_cell: int
    seed: int | None = None
    axes: dict[str, AxisSpec]
    # See ``docs/polish-exploration-lhs-sampling.md``. Default preserves
    # the historical Cartesian-grid enumeration for every existing YAML.
    sampling: SamplingMode = "cartesian"
    # Required for ``sampling != "cartesian"`` and forbidden for
    # ``sampling == "cartesian"`` — enforced by ``_check_sampling_inputs``
    # below. Sample budget for LHS / Sobol; ignored (must be unset) for
    # the Cartesian path because the cell count is the product of
    # per-axis ``steps`` there.
    n_samples: int | None = None

    # ------------------------------------------------------------------
    # Field-level checks.
    # ------------------------------------------------------------------

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("name must be a non-empty string")
        return v

    @field_validator("env")
    @classmethod
    def _env_supported(cls, v: str) -> str:
        # Registry-backed check: a name is "known" if it is either
        # currently registered or present in the lazy-import dispatch
        # table (the loader will import the matching subpackage on
        # demand before dispatching to the Runner). This keeps the
        # schema simulator-agnostic per RFC-005 §3.1.
        known = registered_envs() | BUILTIN_BACKEND_IMPORTS.keys()
        if v not in known:
            listed = ", ".join(sorted(known))
            raise ValueError(
                f"env: unknown env {v!r}; known envs are: {{{listed}}}",
            )
        return v

    @field_validator("episodes_per_cell")
    @classmethod
    def _episodes_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"episodes_per_cell must be >= 1; got {v}")
        return v

    @field_validator("axes")
    @classmethod
    def _axes_nonempty_and_canonical(cls, v: dict[str, AxisSpec]) -> dict[str, AxisSpec]:
        if len(v) == 0:
            raise ValueError("axes: at least one axis must be defined")
        unknown = [name for name in v if name not in AXIS_NAMES]
        if unknown:
            legal = ", ".join(AXIS_NAMES)
            raise ValueError(
                f"axes: unknown axis name(s) {unknown}; legal names are: {legal}",
            )
        return v

    @field_validator("n_samples")
    @classmethod
    def _n_samples_positive(cls, v: int | None) -> int | None:
        # Field-level: only that the integer is positive when provided.
        # The cross-field "required for non-cartesian, forbidden for
        # cartesian" rule lives in ``_check_sampling_inputs``.
        if v is not None and v < 1:
            raise ValueError(f"n_samples must be >= 1; got {v}")
        return v

    # ------------------------------------------------------------------
    # Cross-field validation (sampling mode <-> per-axis steps and
    # n_samples). Runs after the per-field validators so error messages
    # mention only one axis at a time.
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_sampling_inputs(self) -> Suite:
        if self.sampling == "cartesian":
            # n_samples is meaningless for Cartesian (cell count is the
            # axis-step product). Reject explicitly so a user who ports
            # an LHS YAML to cartesian without removing n_samples sees
            # a clear error rather than silent ignore.
            if self.n_samples is not None:
                raise ValueError(
                    "sampling=cartesian: n_samples must be omitted "
                    "(grid size is the product of per-axis steps); "
                    f"got n_samples={self.n_samples}",
                )
            # Every continuous axis must specify ``steps`` for the
            # Cartesian grid to be well-defined. The :class:`AxisSpec`
            # layer permits ``{low, high}`` without ``steps`` so
            # non-cartesian YAMLs can omit the redundant field; the
            # parent Suite enforces it back here for the cartesian
            # path. Categorical axes (``values``) are unaffected.
            for axis_name, spec in self.axes.items():
                is_continuous = spec.values is None
                if is_continuous and spec.steps is None:
                    raise ValueError(
                        f"axis spec: continuous shape requires low, high, "
                        f"steps; missing ['steps'] on axis {axis_name!r} "
                        "(required for sampling=cartesian)",
                    )
        else:
            # LHS / Sobol: n_samples is required and steps is forbidden
            # (an LHS YAML that names ``steps`` is confused — the LHS
            # budget is ``n_samples``, not the per-axis grid count).
            if self.n_samples is None:
                raise ValueError(
                    f"sampling={self.sampling}: n_samples is required "
                    "(it sets the sample budget; the cell count of an "
                    "LHS / Sobol suite is independent of per-axis steps)",
                )
            for axis_name, spec in self.axes.items():
                if spec.steps is not None:
                    raise ValueError(
                        f"axis spec: sampling={self.sampling} forbids "
                        f"per-axis steps; got steps={spec.steps} on axis "
                        f"{axis_name!r}. Drop the steps key — LHS / Sobol "
                        "sample size is set by the suite-level n_samples.",
                    )
        return self

    # ------------------------------------------------------------------
    # Cell enumeration. Cartesian product over per-axis enumerations,
    # in YAML axis-declaration order.
    # ------------------------------------------------------------------

    def cells(self) -> Iterator[SuiteCell]:
        """Yield every grid point as a :class:`SuiteCell`.

        Ordering is deterministic and stable across runs:

        * Axes vary in YAML insertion order (preserved by PyYAML +
          Pydantic v2 dict ordering).
        * Within axes, values follow :meth:`AxisSpec.enumerate` order.
        * The rightmost axis varies fastest (standard
          :func:`itertools.product` ordering).

        Each yielded :class:`SuiteCell` carries a zero-based ``index``
        the Runner uses as the cell id.
        """
        axis_names = tuple(self.axes.keys())
        per_axis_values = tuple(spec.enumerate() for spec in self.axes.values())
        for index, combo in enumerate(itertools.product(*per_axis_values)):
            mapping: dict[str, float] = dict(zip(axis_names, combo, strict=True))
            yield SuiteCell(index=index, values=mapping)

    def num_cells(self) -> int:
        """Return the total number of grid cells (product of per-axis sizes)."""
        total = 1
        for spec in self.axes.values():
            total *= len(spec.enumerate())
        return total
