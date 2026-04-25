"""Suite linter — author-time footgun detection (B-25).

``gauntlet suite check <suite.yaml>`` lints a suite for the common
authoring footguns the schema validators do *not* reject — knobs that
are technically legal but almost always wrong.

The pure-Python entry point is :func:`lint_suite`, which takes an
already-validated :class:`Suite` and returns a list of
:class:`LintFinding` records (severity + rule id + message). It performs
no I/O and no pydantic re-validation; the loader has already ensured the
``Suite`` is well-typed.

Lint rules (see ``docs/backlog.md`` B-25 for the design):

* :data:`RULE_UNUSED_AXIS` — every axis whose realised value set
  collapses to a single point (categorical of length 1, all-identical
  ``values``, continuous ``steps == 1``, or LHS / Sobol with
  ``low == high``). Warning.
* :data:`RULE_CARTESIAN_EXPLOSION` — ``sampling: cartesian`` with a
  Cartesian product exceeding :data:`CARTESIAN_EXPLOSION_THRESHOLD`
  cells (defaults to ``10_000``). Warning.
* :data:`RULE_VISUAL_ONLY_ON_ISAAC` — any axis name that appears in the
  resolved backend's ``VISUAL_ONLY_AXES`` ClassVar. Error (the loader
  already rejects the *all-cosmetic* subset; this surfaces the
  one-bad-axis-out-of-many case earlier than the runner). Reuses
  :func:`gauntlet.suite.loader._visual_only_axes_of` so the unwrap-
  partial / missing-attr / non-frozenset branches stay in one place.
* :data:`RULE_INSUFFICIENT_EPISODES` — ``episodes_per_cell`` below the
  minimum required to hit a target Wilson 95% CI half-width on a
  single cell. Worst-case (``p == 0.5``) is ``1.96 · 0.5 / sqrt(N) ≈
  0.98 / sqrt(N) ≈ 1 / sqrt(N)``. Warning. The closed form keeps this
  rule independent of B-03 (``report.cell_ci_*``) shipping first.
* :data:`RULE_EMPTY_SUITE` — zero axes declared. The schema rejects
  the literal empty case at load time; the linter keeps the rule for
  defence-in-depth (and to surface a friendly message if a future
  caller hands :func:`lint_suite` a hand-constructed :class:`Suite`).
  Warning.

The public surface (:class:`LintFinding`, :func:`lint_suite`) is
re-exported from :mod:`gauntlet.suite` so downstream callers can
script off the linter without depending on the CLI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

from gauntlet.env.registry import get_env_factory
from gauntlet.suite.loader import _visual_only_axes_of
from gauntlet.suite.schema import AxisSpec, Suite

__all__ = [
    "CARTESIAN_EXPLOSION_THRESHOLD",
    "EPISODES_FOR_HALF_WIDTH_15",
    "EPISODES_FOR_HALF_WIDTH_20",
    "EPISODES_PER_CELL_WARN_BELOW",
    "RULE_CARTESIAN_EXPLOSION",
    "RULE_EMPTY_SUITE",
    "RULE_INSUFFICIENT_EPISODES",
    "RULE_UNUSED_AXIS",
    "RULE_VISUAL_ONLY_ON_ISAAC",
    "LintFinding",
    "LintSeverity",
    "lint_suite",
]


LintSeverity = Literal["warning", "error"]

# Rule identifier strings — exposed so callers (CLI, tests, downstream
# wrappers) can key off a stable name rather than substring-matching the
# message body.
RULE_UNUSED_AXIS: Final[str] = "unused-axis"
RULE_CARTESIAN_EXPLOSION: Final[str] = "cartesian-explosion"
RULE_VISUAL_ONLY_ON_ISAAC: Final[str] = "visual-only-axis-on-state-only-backend"
RULE_INSUFFICIENT_EPISODES: Final[str] = "insufficient-episodes-per-cell"
RULE_EMPTY_SUITE: Final[str] = "empty-suite"

# Cartesian-explosion threshold — runs above this almost always indicate
# the author intended an LHS / Sobol sweep and forgot to switch
# ``sampling``. Tuned at 10_000 cells per the backlog spec.
CARTESIAN_EXPLOSION_THRESHOLD: Final[int] = 10_000

# Wilson worst-case approximation thresholds for the
# ``episodes_per_cell`` rule. Worst-case half-width at p=0.5 is
# ``1.96 · 0.5 / sqrt(N) ≈ 0.98 / sqrt(N)``. Solving for the canonical
# half-widths the warning cites:
#
# * half_width = 0.30 → N ≈ (0.98/0.30)^2 ≈ 10.67  (warn under 10)
# * half_width = 0.20 → N ≈ (0.98/0.20)^2 ≈ 24.01  (recommend N ≥ 20)
# * half_width = 0.15 → N ≈ (0.98/0.15)^2 ≈ 42.68  (recommend N ≥ 40)
#
# Numbers stay in sync with the message body below so changing one in
# isolation flips the test that asserts on the recommendation text.
EPISODES_PER_CELL_WARN_BELOW: Final[int] = 10
EPISODES_FOR_HALF_WIDTH_20: Final[int] = 20
EPISODES_FOR_HALF_WIDTH_15: Final[int] = 40

# z-score for a 95% CI (standard normal). Lifted to a constant so the
# closed-form approximation in :func:`_wilson_worst_case_half_width`
# stays self-documenting.
_Z_95: Final[float] = 1.96


@dataclass(frozen=True)
class LintFinding:
    """One linter issue.

    Attributes:
        severity: ``"warning"`` (printed but does not fail the run) or
            ``"error"`` (prints and forces a non-zero exit).
        rule: Stable identifier (one of the ``RULE_*`` constants). Lets
            scripted callers grep / filter.
        message: Human-readable explanation. Begins with a lower-case
            verb ("axis foo declared but ...") so it composes cleanly
            into the CLI's ``warning: <message>`` envelope.
    """

    severity: LintSeverity
    rule: str
    message: str


def lint_suite(suite: Suite) -> list[LintFinding]:
    """Run every lint rule on ``suite`` and return the findings list.

    The list is ordered: empty-suite first (it tends to make every
    other rule moot), then per-axis rules in YAML insertion order, then
    suite-wide rules (cartesian explosion, episodes_per_cell). Tests key
    off the order to assert "rule X comes before rule Y" without having
    to filter.

    The function is pure (no I/O, no global mutation). It does call
    :func:`gauntlet.env.registry.get_env_factory` to resolve the
    backend's ``VISUAL_ONLY_AXES``; the schema validator has already
    ensured the env is registered or imports through the lazy-import
    table.
    """
    findings: list[LintFinding] = []

    # Rule 5 (empty suite) — surfaces first because every downstream
    # rule degenerates on an axis-less suite.
    if not suite.axes:
        findings.append(
            LintFinding(
                severity="warning",
                rule=RULE_EMPTY_SUITE,
                message=(
                    "no axes declared — every cell will be identical. "
                    "Add at least one perturbation axis or delete the suite."
                ),
            )
        )

    # Rule 1 (unused axis) — per-axis, YAML insertion order.
    for axis_name, spec in suite.axes.items():
        collapsed = _axis_collapsed_to_one_value(spec)
        if collapsed is None:
            continue
        findings.append(
            LintFinding(
                severity="warning",
                rule=RULE_UNUSED_AXIS,
                message=(
                    f"axis {axis_name!r} declared but only one value ({collapsed}) "
                    "— drop the axis or add variation."
                ),
            )
        )

    # Rule 3 (visual-only axes on a state-only backend). Errors so the
    # CLI exits non-zero, mirroring the loader's runtime rejection of
    # the all-cosmetic case but firing for the mixed case too. Best-
    # effort: a backend whose factory is something exotic returns an
    # empty frozenset and the rule becomes a no-op.
    visual_only = _backend_visual_only_axes(suite.env)
    if visual_only:
        for axis_name in suite.axes:
            if axis_name in visual_only:
                findings.append(
                    LintFinding(
                        severity="error",
                        rule=RULE_VISUAL_ONLY_ON_ISAAC,
                        message=(
                            f"axis {axis_name!r} is in {suite.env!r}'s "
                            f"VISUAL_ONLY_AXES — the backend cannot observe "
                            "this perturbation in state-only mode and will "
                            "reject it at runtime. Drop the axis or wait for "
                            "the rendering follow-up RFC."
                        ),
                    )
                )

    # Rule 2 (cartesian explosion) — suite-wide, cheap once per call.
    if suite.sampling == "cartesian" and suite.axes:
        n_cells = suite.num_cells()
        if n_cells > CARTESIAN_EXPLOSION_THRESHOLD:
            findings.append(
                LintFinding(
                    severity="warning",
                    rule=RULE_CARTESIAN_EXPLOSION,
                    message=(
                        f"cartesian product yields {n_cells:,} cells "
                        f"(>{CARTESIAN_EXPLOSION_THRESHOLD:,}) — consider "
                        "`sampling: sobol` with `n_samples: 256` for joint "
                        "axis coverage at a fraction of the cost."
                    ),
                )
            )

    # Rule 4 (insufficient episodes_per_cell) — suite-wide.
    n = suite.episodes_per_cell
    if n < EPISODES_PER_CELL_WARN_BELOW:
        half_width = _wilson_worst_case_half_width(n)
        findings.append(
            LintFinding(
                severity="warning",
                rule=RULE_INSUFFICIENT_EPISODES,
                message=(
                    f"episodes_per_cell={n} yields +/-{half_width:.2f} "
                    f"Wilson 95% CIs on per-cell success — consider "
                    f"N>={EPISODES_FOR_HALF_WIDTH_20} for +/-0.20 or "
                    f"N>={EPISODES_FOR_HALF_WIDTH_15} for +/-0.15."
                ),
            )
        )

    return findings


# ──────────────────────────────────────────────────────────────────────
# Internal helpers — pure, no I/O.
# ──────────────────────────────────────────────────────────────────────


def _axis_collapsed_to_one_value(spec: AxisSpec) -> float | None:
    """Return the collapsed value of an axis with no variation, else ``None``.

    Covers the four shapes that all amount to "this axis does nothing":

    * Categorical with ``len(values) == 1``.
    * Categorical with all entries equal (e.g. ``[1.0, 1.0, 1.0]``).
    * Continuous Cartesian with ``steps == 1``.
    * Continuous LHS / Sobol with ``low == high`` (no ``steps`` field;
      the LHS / Sobol sampler will draw the same point every time).

    The schema permits all four; only the linter catches them.
    """
    if spec.values is not None:
        # Categorical shape. Empty lists are rejected by the schema, so
        # we can dereference [0] safely on length 1, and ``set`` on a
        # multi-element list answers the all-identical question without
        # depending on float equality being transitive.
        if len(spec.values) == 1:
            return float(spec.values[0])
        unique = set(spec.values)
        if len(unique) == 1:
            return float(spec.values[0])
        return None

    # Continuous shape. ``low``, ``high`` are guaranteed not-None when
    # ``values`` is None (model validator ``_check_shape_exclusive``).
    assert spec.low is not None
    assert spec.high is not None
    if spec.steps == 1:
        return float(0.5 * (spec.low + spec.high))
    if spec.low == spec.high:
        return float(spec.low)
    return None


def _backend_visual_only_axes(env_name: str) -> frozenset[str]:
    """Resolve the backend's ``VISUAL_ONLY_AXES`` set, defaulting to empty.

    The schema validator guarantees ``env_name`` is either currently
    registered or one of the lazy-import keys. The lazy-import keys may
    not have been imported yet (the linter does *not* trigger an
    import — that would defeat ``gauntlet suite check`` as a quick
    static lint on a machine without the backend's extras installed).
    A name we cannot resolve through :func:`get_env_factory` falls back
    to the empty frozenset and the rule becomes a no-op.
    """
    try:
        factory = get_env_factory(env_name)
    except (KeyError, ValueError):
        return frozenset()
    return _visual_only_axes_of(factory)


def _wilson_worst_case_half_width(n: int) -> float:
    """Closed-form approximation to the 95% Wilson interval half-width at p=0.5.

    The exact Wilson interval at ``p == 0.5`` (the worst case) is

        half_width = (z * sqrt(0.25 / n + z^2 / (4 n^2))) / (1 + z^2 / n)

    For the cell sizes this rule warns about (``N < 10``) the
    second-order term is small, and for the message we want a stable
    closed form independent of B-03's full Wilson implementation.
    Using the leading term ``z * 0.5 / sqrt(n) ≈ 0.98 / sqrt(n)`` gives
    an upper bound that matches the canonical "1/sqrt(N)" rule of
    thumb the spec asks for. ``n == 0`` is impossible (schema
    enforces ``episodes_per_cell >= 1``); we guard anyway so the
    function is callable from tests directly.
    """
    if n <= 0:
        return float("inf")
    return _Z_95 * 0.5 / math.sqrt(n)
