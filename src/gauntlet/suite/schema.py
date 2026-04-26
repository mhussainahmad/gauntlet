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
* **Categorical**: ``{values: [...]}`` — explicit enumeration. ``values``
  may be a list of floats (the default) or a list of strings; the
  string shape is accepted only on the ``instruction_paraphrase`` axis
  (B-05) where each entry is a natural-language paraphrasing of the
  task. The cell channel stays float-valued — string lists enumerate
  to ``(0.0, 1.0, ..., len-1)`` indices and the
  :class:`gauntlet.env.instruction.InstructionWrapper` performs the
  index → string lookup at apply time.

Validation is firm — extras are forbidden, axis names must be canonical,
``low <= high``, ``steps >= 1``, ``episodes_per_cell >= 1``, ``env`` must
match either a currently-registered backend or one of the lazy-import
keys in :data:`BUILTIN_BACKEND_IMPORTS` (``tabletop``,
``tabletop-pybullet``, ``tabletop-genesis``, ``tabletop-isaac``), and
the axes mapping must be non-empty.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gauntlet.env.perturbation import AXIS_NAMES
from gauntlet.env.registry import registered_envs

__all__ = [
    "BUILTIN_BACKEND_IMPORTS",
    "SAMPLING_MODES",
    "AxisSpec",
    "ExtrinsicsRange",
    "ExtrinsicsValue",
    "SamplingMode",
    "Suite",
    "SuiteCell",
]


# Public type alias so the loader / Sampler dispatch / docs all key off
# the same string set. Adding a new mode requires touching this literal
# and ``gauntlet.suite.sampling.build_sampler`` together — by design.
SamplingMode = Literal["cartesian", "latin_hypercube", "sobol", "adversarial"]

# Tuple form of the same set, used by validators that need a runtime
# ``in`` check (Pydantic narrows ``Literal`` in the model but the
# loader's defence-in-depth path benefits from a plain tuple).
SAMPLING_MODES: Final[tuple[SamplingMode, ...]] = (
    "cartesian",
    "latin_hypercube",
    "sobol",
    "adversarial",
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


class ExtrinsicsValue(BaseModel):
    """One enumerated camera-extrinsics entry (B-42).

    Carries a structured 6-D pose delta applied to the env's render
    camera at episode start: ``translation`` is a length-3 ``[dx, dy,
    dz]`` in metres, ``rotation`` is a length-3 ``[drx, dry, drz]`` in
    radians (XYZ Euler, MuJoCo / PyBullet camera convention). Both
    fields are required — there is no implicit zero default — so a
    YAML that wants the "no rotation" baseline writes
    ``rotation: [0, 0, 0]`` explicitly.

    The schema layer enumerates a list of these to indices
    ``(0.0, 1.0, ..., len-1)`` for the cell-value channel; the env's
    :meth:`gauntlet.env.tabletop.TabletopEnv.set_camera_extrinsics_list`
    setter rebinds the index → 6-tuple mapping at suite-load time.
    """

    model_config = ConfigDict(extra="forbid")

    translation: list[float] = Field(...)
    rotation: list[float] = Field(...)

    @field_validator("translation", "rotation")
    @classmethod
    def _length_three(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError(
                f"extrinsics entry: translation / rotation must be length 3 "
                f"(x, y, z); got length {len(v)}",
            )
        return [float(x) for x in v]


class ExtrinsicsRange(BaseModel):
    """Continuous-range camera-extrinsics shape (B-42).

    Sobol-friendly six-dimensional bounds: ``translation`` is a length-3
    list of ``[lo, hi]`` pairs (metres) and ``rotation`` is a length-3
    list of ``[lo, hi]`` pairs (radians, XYZ Euler). The schema layer
    pre-resolves the range into ``n_samples`` enumerated entries at
    suite-load time using the 6-D Joe-Kuo Sobol sequence (deterministic,
    no entropy consumed) so the cell-value channel stays uniformly
    float-valued and downstream samplers (Cartesian / LHS / Sobol /
    adversarial) see the axis as a plain categorical with N entries.

    Anti-feature note: SO(3) rotations don't compose linearly so the
    closed-form Saltelli decomposition over the resulting per-axis
    indices is biased — the report module emits a :class:`UserWarning`
    whenever this axis is present in a Sobol report.
    """

    model_config = ConfigDict(extra="forbid")

    translation: list[list[float]] = Field(...)
    rotation: list[list[float]] = Field(...)

    @field_validator("translation", "rotation")
    @classmethod
    def _three_pairs(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3:
            raise ValueError(
                f"extrinsics range: translation / rotation must each be a "
                f"list of three [lo, hi] pairs; got length {len(v)}",
            )
        out: list[list[float]] = []
        for i, pair in enumerate(v):
            if len(pair) != 2:
                raise ValueError(
                    f"extrinsics range: dim {i} must be a [lo, hi] pair; got {pair!r}",
                )
            lo, hi = float(pair[0]), float(pair[1])
            if lo > hi:
                raise ValueError(
                    f"extrinsics range: dim {i} lo must be <= hi; got [{lo}, {hi}]",
                )
            out.append([lo, hi])
        return out


class AxisSpec(BaseModel):
    """Grid specification for a single perturbation axis.

    Exactly one of two shapes is accepted (validated in
    :meth:`_check_shape_exclusive`):

    * Continuous / int: ``low``, ``high``, ``steps`` (all required together).
    * Categorical: ``values`` (a non-empty list).

    On the ``camera_extrinsics`` axis (B-42) the categorical shape is
    additionally satisfied by either ``extrinsics_values`` (an
    enumerated list of structured 6-D pose deltas) or ``extrinsics_range``
    (Sobol-friendly per-dim ``[lo, hi]`` bounds, pre-resolved at
    suite-load time). Both fields are forbidden on every other axis.

    Mixing the two top-level shapes raises a clear
    :class:`pydantic.ValidationError`.

    Attributes:
        low: Inclusive lower bound for the continuous shape.
        high: Inclusive upper bound for the continuous shape.
        steps: Number of evenly spaced points (``>= 1``). When ``steps``
            is ``1``, the single value emitted is the midpoint of
            ``[low, high]``.
        values: Explicit list of values for the categorical shape.
        extrinsics_values: Enumerated list of structured 6-D pose
            deltas for the ``camera_extrinsics`` axis (B-42).
        extrinsics_range: Continuous-range shape for the
            ``camera_extrinsics`` axis (B-42); pre-resolved at suite-
            load time into N enumerated entries.
    """

    model_config = ConfigDict(extra="forbid")

    low: float | None = Field(default=None)
    high: float | None = Field(default=None)
    steps: int | None = Field(default=None)
    # B-05 — accept either a float list (the historical categorical
    # shape) or a string list (the instruction_paraphrase shape; the
    # parent ``Suite`` validator restricts string lists to the
    # ``instruction_paraphrase`` axis name). The discriminated union is
    # safe under Pydantic v2: float list candidates win on numeric input,
    # string list candidates win on string input, mixed inputs are
    # rejected by the per-arm coercion and surface a clear validation
    # error.
    values: list[float] | list[str] | None = Field(default=None)
    # B-32 — OOD prior for the ``initial_state_ood`` axis. Both fields
    # are length-3 ``(x, y, z)`` lists. They are forbidden on every
    # other axis (cross-axis check lives on the parent ``Suite``).
    prior_mean: list[float] | None = Field(default=None)
    prior_std: list[float] | None = Field(default=None)
    # B-42 — structured shapes for the ``camera_extrinsics`` axis. Both
    # fields are forbidden on every other axis (cross-axis check lives
    # on the parent ``Suite``). Exactly one of the two is allowed at
    # the AxisSpec layer; the parent Suite resolves
    # ``extrinsics_range`` into a concrete ``extrinsics_values`` list
    # at load time using a 6-D Sobol pre-expansion seeded from
    # ``Suite.seed`` (deterministic; the Sobol algorithm consumes no
    # entropy).
    extrinsics_values: list[ExtrinsicsValue] | None = Field(default=None)
    extrinsics_range: ExtrinsicsRange | None = Field(default=None)

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
    def _values_nonempty(cls, v: list[float] | list[str] | None) -> list[float] | list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("values must be a non-empty list when provided")
        return v

    @field_validator("prior_mean", "prior_std")
    @classmethod
    def _ood_prior_length_three(cls, v: list[float] | None, info: object) -> list[float] | None:
        # B-32. ``prior_mean`` / ``prior_std`` describe the (x, y, z)
        # training-pose distribution for the OOD axis. The
        # axis-name-specific "only on initial_state_ood" check lives on
        # ``Suite`` (the AxisSpec is name-agnostic). Here we just shape-
        # check the lists themselves.
        if v is None:
            return v
        if len(v) != 3:
            raise ValueError(
                f"axis spec: prior_mean / prior_std must be length 3 (x, y, z); got length {len(v)}"
            )
        return v

    @model_validator(mode="after")
    def _check_shape_exclusive(self) -> AxisSpec:
        has_continuous = any(x is not None for x in (self.low, self.high, self.steps))
        has_categorical = self.values is not None
        # B-42 — ``extrinsics_values`` and ``extrinsics_range`` are
        # alternative satisfiers of the categorical shape on the
        # ``camera_extrinsics`` axis. The parent Suite restricts them
        # to that axis name; here we enforce mutual exclusion plus the
        # "no other shape alongside" rule.
        has_extrinsics_values = self.extrinsics_values is not None
        has_extrinsics_range = self.extrinsics_range is not None
        has_extrinsics = has_extrinsics_values or has_extrinsics_range
        if has_extrinsics_values and has_extrinsics_range:
            raise ValueError(
                "axis spec: cannot mix extrinsics_values with extrinsics_range; "
                "use one shape or the other",
            )
        if has_extrinsics and (has_continuous or has_categorical):
            raise ValueError(
                "axis spec: cannot mix extrinsics_values / extrinsics_range with "
                "{low, high, steps} or {values}; use one shape only",
            )
        # B-32 — ``prior_mean`` / ``prior_std`` are aux fields on the
        # categorical shape (the OOD axis declares its sigma multipliers
        # under ``values``). They are forbidden on the continuous shape;
        # the parent Suite further restricts them to the
        # ``initial_state_ood`` axis name.
        has_ood_prior = self.prior_mean is not None or self.prior_std is not None
        if has_ood_prior and has_continuous:
            raise ValueError(
                "axis spec: prior_mean / prior_std are only valid alongside the "
                "categorical {values} shape; drop {low, high, steps} or drop the "
                "prior fields",
            )
        if has_continuous and has_categorical:
            raise ValueError(
                "axis spec: cannot mix {low, high, steps} with {values}; "
                "use one shape or the other",
            )
        if not has_continuous and not has_categorical and not has_extrinsics:
            raise ValueError(
                "axis spec: must specify either {low, high, steps} or {values} "
                "(or extrinsics_values / extrinsics_range on the "
                "'camera_extrinsics' axis)",
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
        float) in declared order. For the
        :class:`instruction_paraphrase`-style string shape (B-05) where
        ``values`` is a list of natural-language paraphrases, the
        enumeration emits the *indices* ``(0.0, 1.0, ..., len-1)`` —
        :class:`gauntlet.env.instruction.InstructionWrapper` performs
        the index → string lookup at apply time so the
        :class:`SuiteCell.values` mapping stays uniformly float-valued.

        For the B-42 ``extrinsics_values`` shape, the enumeration emits
        indices ``(0.0, 1.0, ..., len-1)``; the env's
        :meth:`gauntlet.env.tabletop.TabletopEnv.set_camera_extrinsics_list`
        performs the index → 6-tuple lookup at apply time. The
        ``extrinsics_range`` shape is pre-resolved into
        ``extrinsics_values`` at suite-load time before this method
        runs (the parent :class:`Suite` performs the expansion); a
        bare ``extrinsics_range`` reaching this method is a programmer
        error and raises.

        For the continuous shape:

        * ``steps == 1`` → single midpoint ``(low + high) / 2``.
        * ``steps >= 2`` → ``steps`` evenly spaced inclusive endpoints,
          ``low + i * (high - low) / (steps - 1)`` for ``i`` in
          ``range(steps)``.
        """
        if self.extrinsics_values is not None:
            # B-42 — enumerated structured pose deltas; the cell value
            # channel is the integer index into the list.
            return tuple(float(i) for i in range(len(self.extrinsics_values)))
        if self.extrinsics_range is not None:
            raise ValueError(
                "axis spec: extrinsics_range must be pre-resolved into "
                "extrinsics_values before enumerate(); the suite loader "
                "performs the expansion. This call path is unreachable from "
                "the public loader.",
            )
        if self.values is not None:
            # B-05 — string-valued ``values`` enumerates as indices; the
            # wrapper resolves each index back to its paraphrase string.
            if len(self.values) > 0 and isinstance(self.values[0], str):
                return tuple(float(i) for i in range(len(self.values)))
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

    def extrinsics_entries(self) -> tuple[ExtrinsicsValue, ...] | None:
        """Return the structured 6-D extrinsics entries (B-42).

        Helper companion to :meth:`enumerate` for the
        ``camera_extrinsics`` axis. Returns ``None`` for every other
        axis shape (so the runner can skip the
        :meth:`set_camera_extrinsics_list` rebind on cells that don't
        carry this axis); otherwise returns the validated
        :class:`ExtrinsicsValue` tuple in declared order.

        Callers MUST ensure the parent :class:`Suite` has resolved any
        ``extrinsics_range`` into a concrete ``extrinsics_values`` list
        first — the bare-range path raises from :meth:`enumerate` for
        the same reason.
        """
        if self.extrinsics_values is None:
            return None
        return tuple(self.extrinsics_values)

    def paraphrases(self) -> tuple[str, ...] | None:
        """Return the natural-language strings on a string-valued axis.

        B-05 helper: returns ``None`` if ``values`` is not a string list
        (i.e. every other axis shape). Otherwise returns the strings in
        declared order so
        :class:`gauntlet.env.instruction.InstructionWrapper` can map
        each :meth:`enumerate` index back to its paraphrase.
        """
        if self.values is None or len(self.values) == 0:
            return None
        if not isinstance(self.values[0], str):
            return None
        return tuple(str(s) for s in self.values)


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
    # below. Sample budget for LHS / Sobol / adversarial; ignored
    # (must be unset) for the Cartesian path because the cell count is
    # the product of per-axis ``steps`` there.
    n_samples: int | None = None
    # B-07 — adversarial sampler only. Path to a pilot run's
    # ``report.json``; required when ``sampling == "adversarial"`` and
    # forbidden for every other mode (a stale pilot reference is the
    # exact footgun the anti-feature warning calls out). Resolved
    # relative to the suite YAML's parent directory by ``load_suite``;
    # absolute paths and string-loaded suites pass through unchanged.
    pilot_report: str | None = None

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
        # Built-ins are checked first; an unknown name then falls
        # through to the third-party ``gauntlet.axes`` entry-point group
        # so a plugin axis can be referenced from a suite YAML without
        # touching the canonical AXIS_NAMES tuple. Lazy import — see
        # ``gauntlet.policy.registry`` for the same cycle-avoidance
        # rationale (the loader is on the YAML hot path).
        unknown = [name for name in v if name not in AXIS_NAMES]
        if unknown:
            from gauntlet.plugins import discover_axis_plugins

            plugin_axes = set(discover_axis_plugins().keys())
            still_unknown = [name for name in unknown if name not in plugin_axes]
            if still_unknown:
                legal = ", ".join(AXIS_NAMES)
                plugin_legal = ", ".join(sorted(plugin_axes)) or "<none installed>"
                raise ValueError(
                    f"axes: unknown axis name(s) {still_unknown}; legal "
                    f"built-in names are: {legal}; legal plugin names are: "
                    f"{plugin_legal}",
                )
        # B-32 — ``prior_mean`` / ``prior_std`` are only meaningful for
        # the ``initial_state_ood`` axis. Reject them on every other
        # axis so a confused YAML fails loudly instead of silently
        # ignoring the fields.
        for axis_name, spec in v.items():
            if axis_name == "initial_state_ood":
                continue
            if spec.prior_mean is not None or spec.prior_std is not None:
                raise ValueError(
                    f"axes[{axis_name!r}]: prior_mean / prior_std are only "
                    "valid on the 'initial_state_ood' axis",
                )
        # B-05 / B-06 — string-valued ``values`` (paraphrase strings or
        # object-swap class names) are only legal on the
        # ``instruction_paraphrase`` and ``object_swap`` axes. Every
        # other axis uses the float-coded value channel; a string list
        # there would silently misroute through the runner since the
        # cell value is consumed as a float.
        _string_value_axes = {"instruction_paraphrase", "object_swap"}
        for axis_name, spec in v.items():
            if axis_name in _string_value_axes:
                continue
            if spec.values is not None and len(spec.values) > 0 and isinstance(spec.values[0], str):
                raise ValueError(
                    f"axes[{axis_name!r}]: string-valued 'values' lists are "
                    "only valid on the 'instruction_paraphrase' (B-05) and "
                    "'object_swap' (B-06) axes",
                )
        # B-42 — ``extrinsics_values`` and ``extrinsics_range`` are
        # only meaningful for the ``camera_extrinsics`` axis. Reject
        # them on every other axis so a confused YAML fails loudly
        # instead of silently ignoring the structured shape.
        for axis_name, spec in v.items():
            if axis_name == "camera_extrinsics":
                continue
            if spec.extrinsics_values is not None or spec.extrinsics_range is not None:
                raise ValueError(
                    f"axes[{axis_name!r}]: extrinsics_values / "
                    "extrinsics_range are only valid on the "
                    "'camera_extrinsics' (B-42) axis",
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
    def _check_camera_extrinsics_shape(self) -> Suite:
        """B-42 — reject extrinsics_range on the Cartesian path.

        ``extrinsics_range`` is the Sobol-friendly continuous shape; the
        loader pre-expands it into N enumerated entries at suite-load
        time using the suite-level ``n_samples`` budget. The Cartesian
        sampler has no ``n_samples`` (cell count is the per-axis steps
        product), so the range form is not meaningful there. A YAML
        that wants enumerated extrinsics on Cartesian sampling writes
        ``extrinsics_values`` directly.
        """
        spec = self.axes.get("camera_extrinsics")
        if spec is None or spec.extrinsics_range is None:
            return self
        if self.sampling == "cartesian":
            raise ValueError(
                "axes['camera_extrinsics']: extrinsics_range is only "
                "valid on sampling=latin_hypercube / sobol / adversarial "
                "(the range is pre-expanded into N entries via 6-D Sobol "
                "using the suite n_samples budget). For sampling=cartesian, "
                "use extrinsics_values with an explicit list of pose deltas.",
            )
        return self

    @model_validator(mode="after")
    def _check_sampling_inputs(self) -> Suite:
        # B-07 — pilot_report is exclusive to adversarial. Surface the
        # mismatch first so a typo'd ``sampling: sobol`` paired with
        # ``pilot_report:`` fails with the right message.
        if self.sampling != "adversarial" and self.pilot_report is not None:
            raise ValueError(
                f"sampling={self.sampling}: pilot_report is only valid "
                "for sampling=adversarial; drop the pilot_report field "
                "or switch to sampling=adversarial.",
            )
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
                # B-42 — the ``extrinsics_values`` shape is a categorical
                # ad an alternative satisfier of "this axis is not
                # continuous". Treat it like ``values is not None``.
                is_continuous = spec.values is None and spec.extrinsics_values is None
                if is_continuous and spec.steps is None:
                    raise ValueError(
                        f"axis spec: continuous shape requires low, high, "
                        f"steps; missing ['steps'] on axis {axis_name!r} "
                        "(required for sampling=cartesian)",
                    )
        else:
            # LHS / Sobol / adversarial: n_samples is required and
            # steps is forbidden (an LHS YAML that names ``steps`` is
            # confused — the budget is ``n_samples``, not the per-axis
            # grid count).
            if self.n_samples is None:
                raise ValueError(
                    f"sampling={self.sampling}: n_samples is required "
                    "(it sets the sample budget; the cell count of an "
                    "LHS / Sobol / adversarial suite is independent of "
                    "per-axis steps)",
                )
            for axis_name, spec in self.axes.items():
                if spec.steps is not None:
                    raise ValueError(
                        f"axis spec: sampling={self.sampling} forbids "
                        f"per-axis steps; got steps={spec.steps} on axis "
                        f"{axis_name!r}. Drop the steps key — LHS / Sobol "
                        "/ adversarial sample size is set by the suite-"
                        "level n_samples.",
                    )
            # B-07 — adversarial needs a pilot report to bias against.
            # Without one the bandit has no posterior to sample from
            # and the mode degenerates to a confused uniform sampler.
            if self.sampling == "adversarial" and self.pilot_report is None:
                raise ValueError(
                    "sampling=adversarial: pilot_report is required "
                    "(adversarial sampling concentrates new draws in "
                    "the pilot's high-failure regions). Run a Sobol or "
                    "LHS pilot first and point this field at its "
                    "report.json.",
                )
        return self

    # ------------------------------------------------------------------
    # Cell enumeration. Dispatches through the :class:`Sampler` selected
    # by :attr:`sampling`:
    #
    # * ``"cartesian"`` (default) — :class:`CartesianSampler`,
    #   byte-identical to the historical itertools.product behaviour.
    # * ``"latin_hypercube"`` — :class:`LatinHypercubeSampler`,
    #   ``n_samples`` rows, RNG seeded from :attr:`seed`.
    # * ``"sobol"`` — :class:`SobolSampler`, the Joe-Kuo low-discrepancy
    #   sequence; ``n_samples`` rows, deterministic from
    #   :attr:`seed`-independent (Sobol is fully deterministic; the
    #   sampler ignores the RNG it receives). See
    #   ``docs/polish-exploration-sobol-sampler.md``.
    # * ``"adversarial"`` — :class:`AdversarialSampler`, Thompson-
    #   sampling bandit over per-bin Beta posteriors fitted from
    #   :attr:`pilot_report`. ANTI-FEATURE: biases coverage toward
    #   known-failure regions; resulting report is NOT a fair sample.
    #   The :class:`Suite` loader prints a loud :class:`UserWarning`
    #   whenever a YAML opts in. See ``docs/backlog.md`` B-07.
    # ------------------------------------------------------------------

    def cells(self) -> Iterator[SuiteCell]:
        """Yield every grid point as a :class:`SuiteCell`.

        Cartesian (default) ordering is deterministic and stable across
        runs:

        * Axes vary in YAML insertion order (preserved by PyYAML +
          Pydantic v2 dict ordering).
        * Within axes, values follow :meth:`AxisSpec.enumerate` order.
        * The rightmost axis varies fastest (standard
          :func:`itertools.product` ordering).

        Non-cartesian sampling (LHS / Sobol) returns ``n_samples``
        cells whose ordering is also deterministic — the underlying
        :class:`numpy.random.Generator` is seeded from :attr:`seed`
        (or OS entropy when ``seed is None``), so the same Suite
        produces the same cells across iterations.

        Each yielded :class:`SuiteCell` carries a zero-based ``index``
        the Runner uses as the cell id.
        """
        # Local import to dodge the schema <-> sampling cycle: the
        # sampling module imports SuiteCell at TYPE_CHECKING time and
        # CartesianSampler imports it lazily inside ``sample``.
        from gauntlet.suite.sampling import build_sampler

        sampler = build_sampler(self.sampling, suite=self)
        rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        yield from sampler.sample(self, rng)

    def num_cells(self) -> int:
        """Return the total number of cells :meth:`cells` will yield.

        For cartesian sampling, this is the product of per-axis grid
        sizes (no RNG / sampler call needed). For LHS / Sobol it is
        ``n_samples`` — the schema validator guarantees that field is
        set whenever ``sampling != "cartesian"``.
        """
        if self.sampling == "cartesian":
            total = 1
            for spec in self.axes.values():
                total *= len(spec.enumerate())
            return total
        # Non-cartesian: the validator guarantees ``n_samples`` is set.
        assert self.n_samples is not None
        return self.n_samples
