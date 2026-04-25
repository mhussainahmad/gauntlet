"""Tests for the B-41 statistical-power calculator.

Three layers:

* Pure formula — :func:`gauntlet.report.wilson.required_episodes` and
  :func:`gauntlet.report.wilson.required_episodes_paired`. Exercises
  textbook reference values, monotonicity, the paired-CRN reduction,
  and every input validation branch.
* Linter rule — :data:`gauntlet.suite.linter.RULE_LOW_STATISTICAL_POWER`
  fires when ``episodes_per_cell`` is below the conservative paired
  threshold; silent at / above it.
* CLI subcommand — ``gauntlet suite plan`` table output, the
  ``--baseline-report`` override, and the loud anti-feature footer.

Reference values are computed once from the closed-form formula in
:func:`gauntlet.report.wilson.required_episodes` and pinned exactly so
a regression in the math (sign flip, wrong z constant, missing
``ceil``) trips an ``==`` test.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.report.wilson import (
    DEFAULT_PAIRED_RHO,
    required_episodes,
    required_episodes_paired,
)
from gauntlet.suite import lint_suite, load_suite_from_string
from gauntlet.suite.linter import (
    LOW_POWER_DEFAULT_DELTA,
    LOW_POWER_DEFAULT_RHO,
    RULE_LOW_STATISTICAL_POWER,
)

# ────────────────────────────────────────────────────────────────────────
# Pure formula — required_episodes (independent samples).
# ────────────────────────────────────────────────────────────────────────


def test_textbook_value_p_half_versus_p_four_tenths() -> None:
    """0.5 vs 0.4, alpha=0.05, power=0.8 → 389 episodes per arm.

    Computed from the spec's pooled-variance formula
    ``((z_{a/2} + z_b) * sqrt(2 p_avg q_avg))^2 / delta^2`` with
    ``z_{0.025} = 1.95996...`` and ``z_{0.8} = 0.84162...``. The raw
    value is 388.519; ``math.ceil`` rounds to 389. Pinning the exact
    integer catches a regression in either constant or the rounding.
    """
    assert required_episodes(0.5, 0.4) == 389


def test_paired_crn_halves_sample_at_default_rho() -> None:
    """Paired-CRN with rho=0.5 (default) is roughly half the independent answer.

    Closed form: ``ceil(n_indep * (1 - rho))``. For 389 independent at
    rho=0.5 the paired answer is ``ceil(389 * 0.5) = 195``.
    """
    n_indep = required_episodes(0.5, 0.4)
    n_paired = required_episodes_paired(0.5, 0.4)
    assert n_paired == 195
    assert n_paired == n_indep // 2 + (1 if n_indep % 2 else 0)


def test_paired_crn_zero_rho_recovers_independent() -> None:
    """``rho = 0`` means no pairing benefit — answer matches the independent path."""
    n_indep = required_episodes(0.5, 0.4)
    assert required_episodes_paired(0.5, 0.4, rho=0.0) == n_indep


def test_paired_crn_high_rho_shrinks_sample() -> None:
    """Stronger correlation → fewer samples needed; rho=0.9 keeps ~10% of indep."""
    n_indep = required_episodes(0.5, 0.4)
    n_paired = required_episodes_paired(0.5, 0.4, rho=0.9)
    # ceil(389 * 0.1) = 39
    assert n_paired == 39
    assert n_paired < n_indep / 5


def test_default_paired_rho_constant_is_half() -> None:
    """The empirical default for shared-seed CRN is 0.5 — pinned by spec."""
    assert DEFAULT_PAIRED_RHO == 0.5


def test_smaller_delta_needs_more_samples() -> None:
    """Monotonicity: tighter effect size → strictly more samples.

    The (p1 - p2)^2 denominator drives this. At a constant midpoint of
    0.5 the formula scales as ``1 / delta^2`` so the ratio of sample
    sizes must be roughly the inverse-square of the delta ratio.
    """
    n_big_gap = required_episodes(0.5, 0.4)  # delta=0.1
    n_small_gap = required_episodes(0.5, 0.45)  # delta=0.05
    n_tiny_gap = required_episodes(0.5, 0.49)  # delta=0.01
    assert n_big_gap < n_small_gap < n_tiny_gap
    # delta halves -> samples roughly 4x.
    assert 3.5 < n_small_gap / n_big_gap < 4.5
    # delta /10 -> samples roughly 100x.
    assert 90.0 < n_tiny_gap / n_big_gap < 110.0


def test_higher_power_needs_more_samples() -> None:
    """power=0.9 is strictly larger sample than power=0.8 at same alpha."""
    n_low = required_episodes(0.5, 0.4, power=0.8)
    n_high = required_episodes(0.5, 0.4, power=0.9)
    assert n_high > n_low


def test_tighter_alpha_needs_more_samples() -> None:
    """alpha=0.01 is strictly larger sample than alpha=0.05."""
    n_loose = required_episodes(0.5, 0.4, alpha=0.05)
    n_tight = required_episodes(0.5, 0.4, alpha=0.01)
    assert n_tight > n_loose


def test_symmetric_in_p1_p2() -> None:
    """The formula uses ``|p1 - p2|`` so swapping rates is a no-op."""
    assert required_episodes(0.5, 0.4) == required_episodes(0.4, 0.5)


def test_rejects_zero_delta() -> None:
    """``p1 == p2`` is degenerate (infinite samples); must raise."""
    with pytest.raises(ValueError, match="must differ"):
        required_episodes(0.5, 0.5)


def test_rejects_p_out_of_range() -> None:
    """Rates outside ``[0, 1]`` are nonsensical and rejected."""
    with pytest.raises(ValueError, match="p1 must be in"):
        required_episodes(-0.1, 0.4)
    with pytest.raises(ValueError, match="p1 must be in"):
        required_episodes(1.1, 0.4)
    with pytest.raises(ValueError, match="p2 must be in"):
        required_episodes(0.5, -0.1)
    with pytest.raises(ValueError, match="p2 must be in"):
        required_episodes(0.5, 1.5)


def test_rejects_alpha_power_out_of_range() -> None:
    """alpha and power both must be strictly in ``(0, 1)``."""
    with pytest.raises(ValueError, match="alpha must be in"):
        required_episodes(0.5, 0.4, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        required_episodes(0.5, 0.4, alpha=1.0)
    with pytest.raises(ValueError, match="power must be in"):
        required_episodes(0.5, 0.4, power=0.0)
    with pytest.raises(ValueError, match="power must be in"):
        required_episodes(0.5, 0.4, power=1.0)


def test_paired_rejects_rho_out_of_range() -> None:
    """rho < 0 or rho >= 1 are both invalid for the variance reduction."""
    with pytest.raises(ValueError, match="rho must be in"):
        required_episodes_paired(0.5, 0.4, rho=-0.1)
    with pytest.raises(ValueError, match="rho must be in"):
        required_episodes_paired(0.5, 0.4, rho=1.0)
    with pytest.raises(ValueError, match="rho must be in"):
        required_episodes_paired(0.5, 0.4, rho=1.5)


def test_floor_at_one_for_extreme_separation() -> None:
    """Boundary: huge effect size still floors at 1 (never returns 0)."""
    # Even a 99-percentage-point gap with the most pessimistic params
    # cannot produce ``n < 1`` because the formula and ceil floor both
    # bottom at one.
    assert required_episodes(0.99, 0.0) >= 1
    assert required_episodes_paired(0.99, 0.0, rho=0.99) >= 1


# ────────────────────────────────────────────────────────────────────────
# Linter rule — gauntlet.suite.linter.RULE_LOW_STATISTICAL_POWER (B-41).
# ────────────────────────────────────────────────────────────────────────


def _by_rule(findings: list, rule: str) -> list:
    return [f for f in findings if f.rule == rule]


def test_linter_low_power_fires_on_undersized_suite() -> None:
    """A suite with episodes_per_cell well below the paired threshold warns.

    The conservative threshold the linter uses is
    :func:`required_episodes_paired` at the canonical 0.5 vs 0.5 -
    :data:`gauntlet.suite.linter.LOW_POWER_DEFAULT_DELTA` effect size
    with rho :data:`gauntlet.suite.linter.LOW_POWER_DEFAULT_RHO`. At
    delta=0.1 / rho=0.5 / alpha=0.05 / power=0.8 the floor is 195
    episodes per cell; ``episodes_per_cell: 20`` is far below.
    """
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: thin-cells
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
    matches = _by_rule(lint_suite(suite), RULE_LOW_STATISTICAL_POWER)
    assert len(matches) == 1
    msg = matches[0].message
    assert "episodes_per_cell=20" in msg
    # Threshold is computed from the paired formula at the documented
    # defaults; the warning message must echo the floor so readers know
    # the recommended value.
    expected_floor = required_episodes_paired(
        0.5,
        0.5 - LOW_POWER_DEFAULT_DELTA,
        rho=LOW_POWER_DEFAULT_RHO,
    )
    assert str(expected_floor) in msg
    assert matches[0].severity == "warning"


def test_linter_low_power_silent_at_threshold() -> None:
    """When episodes_per_cell hits the paired threshold the rule is silent."""
    floor = required_episodes_paired(
        0.5,
        0.5 - LOW_POWER_DEFAULT_DELTA,
        rho=LOW_POWER_DEFAULT_RHO,
    )
    suite = load_suite_from_string(
        textwrap.dedent(
            f"""\
            name: enough-cells
            env: tabletop
            episodes_per_cell: {floor}
            axes:
              lighting_intensity:
                low: 0.3
                high: 1.5
                steps: 3
            """
        )
    )
    assert _by_rule(lint_suite(suite), RULE_LOW_STATISTICAL_POWER) == []


def test_linter_low_power_message_carries_anti_feature_warning() -> None:
    """The B-41 anti-feature note must be loud in the lint message itself.

    Per spec: the power calc tempts users to run *only* the minimum
    and miss the long failure tail. The warning text has to call this
    out so the linter cannot be used as cover for thin sweeps.
    """
    suite = load_suite_from_string(
        textwrap.dedent(
            """\
            name: thin-cells-anti
            env: tabletop
            episodes_per_cell: 20
            axes:
              lighting_intensity:
                values: [0.3, 1.5]
            """
        )
    )
    [finding] = _by_rule(lint_suite(suite), RULE_LOW_STATISTICAL_POWER)
    msg = finding.message.lower()
    # Anti-feature surface: "minimum / floor" + "long tail / failure tail"
    # in the same message body.
    assert "minimum" in msg or "floor" in msg
    assert "failure" in msg or "long tail" in msg


# ────────────────────────────────────────────────────────────────────────
# CLI subcommand — `gauntlet suite plan`.
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


def _write_yaml(tmp_path: Path, body: str, *, name: str = "suite.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_cli_plan_default_baseline_prints_per_cell_table(runner: CliRunner, tmp_path: Path) -> None:
    """`gauntlet suite plan` prints a per-cell table on stdout."""
    suite_path = _write_yaml(
        tmp_path,
        """\
        name: plan-default
        env: tabletop
        episodes_per_cell: 20
        axes:
          lighting_intensity:
            values: [0.3, 1.5]
        """,
    )
    result = runner.invoke(app, ["suite", "plan", str(suite_path)])
    assert result.exit_code == 0, result.stderr
    # Header columns required by the spec.
    out = result.stdout
    assert "cell" in out
    assert "current" in out
    assert "required" in out
    assert "gap" in out
    # Two cells (length-2 categorical), so two data rows.
    # Also expect the floor for rho=0.5/delta=0.1 default ~195 to appear.
    expected_floor = required_episodes_paired(0.5, 0.4)
    assert str(expected_floor) in out
    # Anti-feature warning emitted on stderr (loud, separate from
    # the machine-readable stdout table).
    assert "minimum" in result.stderr.lower() or "floor" in result.stderr.lower()


def test_cli_plan_uses_baseline_report_per_cell_rates(runner: CliRunner, tmp_path: Path) -> None:
    """``--baseline-report`` overrides the default p1 with each cell's real rate."""
    suite_path = _write_yaml(
        tmp_path,
        """\
        name: plan-with-baseline
        env: tabletop
        episodes_per_cell: 50
        axes:
          lighting_intensity:
            values: [0.3, 1.5]
        """,
    )
    # Construct a minimal report.json with two cells. The schema is
    # tolerant — a hand-built dict with the fields ``per_cell`` needs
    # round-trips via ``Report.model_validate``.
    report_payload = {
        "suite_name": "plan-with-baseline",
        "n_episodes": 100,
        "n_success": 75,
        "per_axis": [],
        "per_cell": [
            {
                "cell_index": 0,
                "perturbation_config": {"lighting_intensity": 0.3},
                "n_episodes": 50,
                "n_success": 45,  # rate 0.9
                "success_rate": 0.9,
            },
            {
                "cell_index": 1,
                "perturbation_config": {"lighting_intensity": 1.5},
                "n_episodes": 50,
                "n_success": 30,  # rate 0.6
                "success_rate": 0.6,
            },
        ],
        "failure_clusters": [],
        "heatmap_2d": {},
        "overall_success_rate": 0.75,
        "overall_failure_rate": 0.25,
        "cluster_multiple": 2.0,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "suite",
            "plan",
            str(suite_path),
            "--baseline-report",
            str(report_path),
            "--detect-delta",
            "0.1",
        ],
    )
    assert result.exit_code == 0, result.stderr
    # Cell 0 baseline=0.9 → required ~ paired(0.9, 0.8). Cell 1
    # baseline=0.6 → required ~ paired(0.6, 0.5). Both should show up.
    expected_cell0 = required_episodes_paired(0.9, 0.8)
    expected_cell1 = required_episodes_paired(0.6, 0.5)
    assert str(expected_cell0) in result.stdout
    assert str(expected_cell1) in result.stdout


def test_cli_plan_rejects_zero_delta(runner: CliRunner, tmp_path: Path) -> None:
    """``--detect-delta 0`` is degenerate; CLI must exit 1 with a clear message."""
    suite_path = _write_yaml(
        tmp_path,
        """\
        name: zero-delta
        env: tabletop
        episodes_per_cell: 20
        axes:
          lighting_intensity:
            values: [0.3, 1.5]
        """,
    )
    result = runner.invoke(
        app,
        ["suite", "plan", str(suite_path), "--detect-delta", "0"],
    )
    assert result.exit_code == 1
    assert "delta" in result.stderr.lower()


def test_cli_plan_anti_feature_warning_is_loud(runner: CliRunner, tmp_path: Path) -> None:
    """The anti-feature footer must be on stderr regardless of cell count.

    Three placement points per spec: docstring, linter message, CLI
    footer. This test pins the CLI placement; the others are covered
    by ``test_linter_low_power_message_carries_anti_feature_warning``
    and the docstring on :func:`required_episodes`.
    """
    suite_path = _write_yaml(
        tmp_path,
        """\
        name: plan-anti-feature
        env: tabletop
        episodes_per_cell: 1000
        axes:
          lighting_intensity:
            values: [0.3, 1.5]
        """,
    )
    result = runner.invoke(app, ["suite", "plan", str(suite_path)])
    assert result.exit_code == 0, result.stderr
    err = result.stderr.lower()
    # Loud, multi-keyword anti-feature note — the failure-tail framing
    # is the whole point of B-41's anti-feature.
    assert "long tail" in err or "failure" in err
    assert "minimum" in err or "floor" in err
