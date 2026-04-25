"""Suite linter tests — B-25 (``gauntlet suite check``).

Covers each lint rule end-to-end:

* :data:`RULE_UNUSED_AXIS` — categorical-of-1, all-identical-categorical,
  cartesian ``steps==1``, LHS ``low==high``.
* :data:`RULE_CARTESIAN_EXPLOSION` — fires above
  :data:`CARTESIAN_EXPLOSION_THRESHOLD` cells, silent below.
* :data:`RULE_VISUAL_ONLY_ON_ISAAC` — fires per axis, errors out the
  CLI, fires on the mixed (1 visual + 1 state-effecting) case the
  loader's all-cosmetic guard misses.
* :data:`RULE_INSUFFICIENT_EPISODES` — fires below the warn threshold,
  silent at / above it.
* :data:`RULE_EMPTY_SUITE` — defence-in-depth (the schema rejects the
  literal empty case at load time, so the rule is exercised against a
  hand-constructed :class:`Suite` rather than a YAML).

Plus the CLI integration: clean / warn-only / error / load-failure exit
codes; bundled example YAMLs lint clean (or only-warning).

Tests use :func:`load_suite_from_string` to keep YAML inline and avoid
filesystem fixtures except where the CLI itself needs a path.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.env.registry import _REGISTRY
from gauntlet.suite import (
    AxisSpec,
    LintFinding,
    Suite,
    lint_suite,
    load_suite_from_string,
)
from gauntlet.suite.linter import (
    CARTESIAN_EXPLOSION_THRESHOLD,
    EPISODES_PER_CELL_WARN_BELOW,
    RULE_CARTESIAN_EXPLOSION,
    RULE_EMPTY_SUITE,
    RULE_INSUFFICIENT_EPISODES,
    RULE_UNUSED_AXIS,
    RULE_VISUAL_ONLY_ON_ISAAC,
    _wilson_worst_case_half_width,
)

# ──────────────────────────────────────────────────────────────────────
# Synthetic visual-only backend — same pattern as
# ``tests/test_coverage_suite.py``. Lets the rule fire without
# depending on the optional Isaac extra being installed in CI.
# ──────────────────────────────────────────────────────────────────────


class _SyntheticVisualOnlyEnv:
    """Minimal env with a non-empty ``VISUAL_ONLY_AXES`` ClassVar.

    Mirrors :class:`tests.test_coverage_suite._SyntheticVisualOnlyEnv`.
    The class never gets instantiated; the linter only reads the
    ``VISUAL_ONLY_AXES`` attribute via :func:`getattr`.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"object_texture", "lighting_intensity", "object_initial_pose_x"}
    )
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset({"object_texture", "lighting_intensity"})

    observation_space: Any = None
    action_space: Any = None

    def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
        raise NotImplementedError

    def step(self, action: Any) -> Any:
        raise NotImplementedError

    def set_perturbation(self, name: str, value: float) -> None:
        raise NotImplementedError

    def restore_baseline(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


@pytest.fixture
def visual_only_backend() -> Any:
    """Register a synthetic visual-only backend; teardown removes it."""
    name = "linter-visual-only"
    _REGISTRY[name] = _SyntheticVisualOnlyEnv
    try:
        yield name
    finally:
        _REGISTRY.pop(name, None)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ──────────────────────────────────────────────────────────────────────
# Rule 1: unused-axis. Four shapes.
# ──────────────────────────────────────────────────────────────────────


def _by_rule(findings: list[LintFinding], rule: str) -> list[LintFinding]:
    return [f for f in findings if f.rule == rule]


def test_unused_axis_categorical_length_one() -> None:
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: unused-cat-1
            env: tabletop
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                values: [0.5]
            """
        )
    )
    matches = _by_rule(lint_suite(suite), RULE_UNUSED_AXIS)
    assert len(matches) == 1
    assert "lighting_intensity" in matches[0].message
    assert matches[0].severity == "warning"


def test_unused_axis_categorical_all_identical() -> None:
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: unused-cat-dup
            env: tabletop
            episodes_per_cell: 20
            axes:
              object_texture:
                values: [1.0, 1.0, 1.0]
            """
        )
    )
    matches = _by_rule(lint_suite(suite), RULE_UNUSED_AXIS)
    assert len(matches) == 1
    assert "object_texture" in matches[0].message


def test_unused_axis_continuous_steps_one() -> None:
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: unused-cont-1
            env: tabletop
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                low: 0.3
                high: 1.5
                steps: 1
            """
        )
    )
    matches = _by_rule(lint_suite(suite), RULE_UNUSED_AXIS)
    assert len(matches) == 1
    assert "lighting_intensity" in matches[0].message


def test_unused_axis_lhs_low_equals_high() -> None:
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: unused-lhs
            env: tabletop
            sampling: latin_hypercube
            n_samples: 8
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                low: 0.5
                high: 0.5
              camera_offset_x:
                low: -0.05
                high: 0.05
            """
        )
    )
    matches = _by_rule(lint_suite(suite), RULE_UNUSED_AXIS)
    assert len(matches) == 1
    assert "lighting_intensity" in matches[0].message


def test_unused_axis_no_warning_when_axis_varies() -> None:
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: varying-axis
            env: tabletop
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                low: 0.3
                high: 1.5
                steps: 3
            """
        )
    )
    assert _by_rule(lint_suite(suite), RULE_UNUSED_AXIS) == []


# ──────────────────────────────────────────────────────────────────────
# Rule 2: cartesian explosion.
# ──────────────────────────────────────────────────────────────────────


def test_cartesian_explosion_fires_above_threshold() -> None:
    # 25 * 25 * 25 = 15_625 cells > 10_000.
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: explode
            env: tabletop
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                low: 0.3
                high: 1.5
                steps: 25
              camera_offset_x:
                low: -0.05
                high: 0.05
                steps: 25
              camera_offset_y:
                low: -0.05
                high: 0.05
                steps: 25
            """
        )
    )
    matches = _by_rule(lint_suite(suite), RULE_CARTESIAN_EXPLOSION)
    assert len(matches) == 1
    assert "15,625" in matches[0].message
    assert "sobol" in matches[0].message.lower()
    # Threshold mention keeps the hint actionable.
    assert f"{CARTESIAN_EXPLOSION_THRESHOLD:,}" in matches[0].message


def test_cartesian_explosion_silent_below_threshold() -> None:
    # Two-axis 6-cell suite is well under the threshold.
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: small-cart
            env: tabletop
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                low: 0.3
                high: 1.5
                steps: 3
              object_texture:
                values: [0, 1]
            """
        )
    )
    assert _by_rule(lint_suite(suite), RULE_CARTESIAN_EXPLOSION) == []


def test_cartesian_explosion_silent_for_sobol() -> None:
    # Sobol with a small budget; no cartesian explosion possible.
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: sobol-mode
            env: tabletop
            sampling: sobol
            n_samples: 32
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                low: 0.3
                high: 1.5
              camera_offset_x:
                low: -0.05
                high: 0.05
            """
        )
    )
    assert _by_rule(lint_suite(suite), RULE_CARTESIAN_EXPLOSION) == []


# ──────────────────────────────────────────────────────────────────────
# Rule 3: VISUAL_ONLY axes on a state-only backend.
# ──────────────────────────────────────────────────────────────────────


def test_visual_only_axis_fires_for_each_offending_axis(
    visual_only_backend: str,
) -> None:
    """The mixed case: one visual + one state-effecting axis. The
    loader's ``_reject_purely_visual_suites`` does NOT fire (not every
    axis is cosmetic), but the linter does — once per offending axis."""
    suite = load_suite_from_string(
        textwrap.dedent(
            f"""\
            name: mixed-visual
            env: {visual_only_backend}
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                values: [0.3, 1.5]
              object_initial_pose_x:
                low: -0.05
                high: 0.05
                steps: 3
            """
        )
    )
    matches = _by_rule(lint_suite(suite), RULE_VISUAL_ONLY_ON_ISAAC)
    assert len(matches) == 1
    assert matches[0].severity == "error"
    assert "lighting_intensity" in matches[0].message
    assert visual_only_backend in matches[0].message


def test_visual_only_axis_silent_on_mujoco() -> None:
    """``tabletop`` (MuJoCo) declares an empty ``VISUAL_ONLY_AXES``."""
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: mujoco-axes
            env: tabletop
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                values: [0.3, 1.5]
              object_texture:
                values: [0, 1]
            """
        )
    )
    assert _by_rule(lint_suite(suite), RULE_VISUAL_ONLY_ON_ISAAC) == []


def test_visual_only_axis_fires_per_axis(visual_only_backend: str) -> None:
    """Two cosmetic + one state-effecting → two error findings."""
    suite = load_suite_from_string(
        textwrap.dedent(
            f"""\
            name: many-visual
            env: {visual_only_backend}
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                values: [0.3, 1.5]
              object_texture:
                values: [0, 1]
              object_initial_pose_x:
                low: -0.05
                high: 0.05
                steps: 3
            """
        )
    )
    matches = _by_rule(lint_suite(suite), RULE_VISUAL_ONLY_ON_ISAAC)
    assert len(matches) == 2
    axes_named = {m.message.split("'")[1] for m in matches}
    assert axes_named == {"lighting_intensity", "object_texture"}


# ──────────────────────────────────────────────────────────────────────
# Rule 4: insufficient episodes_per_cell for tight CIs.
# ──────────────────────────────────────────────────────────────────────


def test_insufficient_episodes_fires_below_threshold() -> None:
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: thin-cells
            env: tabletop
            episodes_per_cell: 5
            axes:
              lighting_intensity:
                values: [0.3, 1.5]
            """
        )
    )
    matches = _by_rule(lint_suite(suite), RULE_INSUFFICIENT_EPISODES)
    assert len(matches) == 1
    msg = matches[0].message
    assert "episodes_per_cell=5" in msg
    # Recommendation thresholds N>=20 / N>=40 are part of the contract.
    assert "N>=20" in msg
    assert "N>=40" in msg


def test_insufficient_episodes_silent_at_warn_threshold() -> None:
    suite = load_suite_from_string(
        textwrap.dedent(
            f"""\
            name: enough-cells
            env: tabletop
            episodes_per_cell: {EPISODES_PER_CELL_WARN_BELOW}
            axes:
              lighting_intensity:
                values: [0.3, 1.5]
            """
        )
    )
    assert _by_rule(lint_suite(suite), RULE_INSUFFICIENT_EPISODES) == []


def test_wilson_worst_case_monotone_decreasing() -> None:
    """Larger N → tighter CIs (sanity for the closed-form approximation)."""
    assert _wilson_worst_case_half_width(5) > _wilson_worst_case_half_width(20)
    assert _wilson_worst_case_half_width(20) > _wilson_worst_case_half_width(40)
    # 1/sqrt(N) bound — within numerical tolerance of the spec text.
    half_width_5 = _wilson_worst_case_half_width(5)
    assert 0.40 < half_width_5 < 0.50  # ~0.438


# ──────────────────────────────────────────────────────────────────────
# Rule 5: empty suite (defence-in-depth — schema rejects load-time).
# ──────────────────────────────────────────────────────────────────────


def test_empty_suite_warning_fires_when_axes_dict_empty() -> None:
    """The schema rejects a literal empty axes dict at load time, so we
    construct the :class:`Suite` directly. Confirms the rule still fires
    on a hand-constructed suite (e.g. a future programmatic builder
    that bypasses YAML)."""
    suite = Suite.model_construct(
        name="empty",
        env="tabletop",
        episodes_per_cell=20,
        seed=None,
        axes={},
        sampling="cartesian",
        n_samples=None,
    )
    matches = _by_rule(lint_suite(suite), RULE_EMPTY_SUITE)
    assert len(matches) == 1
    assert "no axes declared" in matches[0].message


# ──────────────────────────────────────────────────────────────────────
# Multi-rule interaction.
# ──────────────────────────────────────────────────────────────────────


def test_multiple_rules_all_surface_in_one_pass() -> None:
    """A deliberately-broken suite should surface every applicable rule."""
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: deliberately-broken
            env: tabletop
            episodes_per_cell: 3
            axes:
              lighting_intensity:
                values: [0.5]
              camera_offset_x:
                low: -0.05
                high: 0.05
                steps: 5
            """
        )
    )
    findings = lint_suite(suite)
    rules = {f.rule for f in findings}
    assert RULE_UNUSED_AXIS in rules
    assert RULE_INSUFFICIENT_EPISODES in rules


# ──────────────────────────────────────────────────────────────────────
# CLI integration — `gauntlet suite check`.
# ──────────────────────────────────────────────────────────────────────


def _write_yaml(tmp_path: Path, body: str, *, name: str = "suite.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_cli_check_clean_suite_exits_zero(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = _write_yaml(
        tmp_path,
        """\
        name: clean
        env: tabletop
        episodes_per_cell: 20
        axes:
          lighting_intensity:
            low: 0.3
            high: 1.5
            steps: 3
          object_texture:
            values: [0, 1]
        """,
    )
    result = runner.invoke(app, ["suite", "check", str(suite_path)])
    assert result.exit_code == 0, result.stderr
    assert "no lint issues" in result.stderr


def test_cli_check_warning_only_exits_zero(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = _write_yaml(
        tmp_path,
        """\
        name: warn-only
        env: tabletop
        episodes_per_cell: 5
        axes:
          lighting_intensity:
            low: 0.3
            high: 1.5
            steps: 3
        """,
    )
    result = runner.invoke(app, ["suite", "check", str(suite_path)])
    assert result.exit_code == 0, result.stderr
    assert "warning:" in result.stderr
    assert RULE_INSUFFICIENT_EPISODES in result.stderr
    assert "ok with warnings" in result.stderr


def test_cli_check_error_exits_one(
    runner: CliRunner,
    tmp_path: Path,
    visual_only_backend: str,
) -> None:
    suite_path = _write_yaml(
        tmp_path,
        f"""\
        name: visual-error
        env: {visual_only_backend}
        episodes_per_cell: 20
        axes:
          lighting_intensity:
            values: [0.3, 1.5]
          object_initial_pose_x:
            low: -0.05
            high: 0.05
            steps: 3
        """,
    )
    result = runner.invoke(app, ["suite", "check", str(suite_path)])
    assert result.exit_code == 1
    assert "error:" in result.stderr
    assert RULE_VISUAL_ONLY_ON_ISAAC in result.stderr
    assert "failed" in result.stderr


def test_cli_check_missing_file_exits_one(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["suite", "check", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "suite file not found" in result.stderr


def test_cli_check_invalid_yaml_exits_one(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = _write_yaml(
        tmp_path,
        """\
        name: broken
        env: tabletop
        episodes_per_cell: 1
        axes: {}
        """,
    )
    result = runner.invoke(app, ["suite", "check", str(suite_path)])
    # Schema-rejected (empty axes dict) → load failure → exit 1.
    assert result.exit_code == 1
    assert "invalid suite YAML" in result.stderr


def test_cli_check_help_lists_subcommand(runner: CliRunner) -> None:
    result = runner.invoke(app, ["suite", "--help"])
    assert result.exit_code == 0
    assert "check" in result.stdout


def test_cli_top_level_help_lists_suite_group(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "suite" in result.stdout


# ──────────────────────────────────────────────────────────────────────
# Bundled examples — none of them should error; warnings are OK.
# ──────────────────────────────────────────────────────────────────────


def test_bundled_tabletop_smoke_lints_clean() -> None:
    """``examples/suites/tabletop-smoke.yaml`` has length-2 categorical
    + 3-step continuous and ``episodes_per_cell: 4`` — that triggers the
    insufficient-episodes warning but no errors."""
    repo_root = Path(__file__).parent.parent
    smoke = repo_root / "examples" / "suites" / "tabletop-smoke.yaml"
    suite = load_suite_from_string(smoke.read_text(encoding="utf-8"))
    findings = lint_suite(suite)
    assert all(f.severity != "error" for f in findings)


def test_lint_finding_is_frozen() -> None:
    """Findings are immutable — callers can stash them in sets / dicts."""
    from dataclasses import FrozenInstanceError

    finding = LintFinding(severity="warning", rule="r", message="m")
    with pytest.raises(FrozenInstanceError):
        finding.severity = "error"  # type: ignore[misc]


def test_axis_spec_round_trip_through_linter() -> None:
    """Direct :class:`AxisSpec` construction (no YAML) still lints."""
    suite = Suite(
        name="programmatic",
        env="tabletop",
        episodes_per_cell=2,
        seed=0,
        axes={"lighting_intensity": AxisSpec(values=[0.5])},
        sampling="cartesian",
    )
    rules = {f.rule for f in lint_suite(suite)}
    assert RULE_UNUSED_AXIS in rules
    assert RULE_INSUFFICIENT_EPISODES in rules
