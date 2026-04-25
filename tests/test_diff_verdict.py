"""Unit tests for B-20 — regression-vs-noise verdict tagging on diff cell flips.

The differ now tags every :class:`~gauntlet.diff.CellFlip` with one of
``regressed`` / ``improved`` / ``within_noise`` based on whichever CI
evidence is available (paired CRN bracket from B-08, independent Wilson
brackets from B-03, or — last resort — the unsafe binary point delta on
legacy reports). The renderer suppresses within-noise flips by default
and the new ``gauntlet diff --show-noise`` flag surfaces them.

These tests are intentionally pure-Python: synthetic Reports + paired
artefacts hand-built so the suite stays fast and focused on the verdict
semantics.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.diff import (
    VERDICT_IMPROVED,
    VERDICT_REGRESSED,
    VERDICT_WITHIN_NOISE,
    CellFlip,
    McNemarResult,
    PairedCellDelta,
    PairedComparison,
    diff_reports,
    render_text,
)
from gauntlet.report import (
    CellBreakdown,
    Report,
)

# ──────────────────────────────────────────────────────────────────────
# Fixtures.
# ──────────────────────────────────────────────────────────────────────


def _cell(
    *,
    cell_index: int,
    perturbation_config: dict[str, float],
    success_rate: float,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> CellBreakdown:
    """Build a :class:`CellBreakdown` with optional Wilson CIs."""
    n_episodes = 10
    n_success = round(success_rate * n_episodes)
    return CellBreakdown(
        cell_index=cell_index,
        perturbation_config=dict(perturbation_config),
        n_episodes=n_episodes,
        n_success=n_success,
        success_rate=success_rate,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _report(*, per_cell: list[CellBreakdown], overall: float = 0.8) -> Report:
    return Report(
        suite_name="synthetic",
        n_episodes=100,
        n_success=int(overall * 100),
        per_axis=[],
        per_cell=per_cell,
        failure_clusters=[],
        heatmap_2d={},
        overall_success_rate=overall,
        overall_failure_rate=1.0 - overall,
        cluster_multiple=2.0,
    )


def _paired_cell(
    *,
    cell_index: int,
    perturbation_config: dict[str, float],
    delta: float,
    delta_ci_low: float,
    delta_ci_high: float,
    p_value: float,
    n_paired: int = 20,
) -> PairedCellDelta:
    """Build a :class:`PairedCellDelta` with explicit CI + McNemar p."""
    a_rate = 0.5
    b_rate = a_rate + delta
    return PairedCellDelta(
        cell_index=cell_index,
        perturbation_config=dict(perturbation_config),
        n_paired=n_paired,
        a_success_rate=a_rate,
        b_success_rate=b_rate,
        delta=delta,
        delta_ci_low=delta_ci_low,
        delta_ci_high=delta_ci_high,
        mcnemar=McNemarResult(b=4, c=0, statistic=4.0, p_value=p_value, exact=False),
    )


def _paired_comparison(cells: list[PairedCellDelta]) -> PairedComparison:
    return PairedComparison(
        paired=True,
        master_seed=42,
        suite_name="synthetic",
        n_cells=len(cells),
        n_paired_episodes=sum(c.n_paired for c in cells),
        cells=cells,
    )


# ──────────────────────────────────────────────────────────────────────
# 1 — paired-CRN verdict path (preferred).
# ──────────────────────────────────────────────────────────────────────


def test_paired_ci_brackets_zero_is_within_noise() -> None:
    """Paired CI straddling 0 → within_noise even though delta crossed threshold."""
    cfg = {"x": 0.5}
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.8)
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.65)
    paired = _paired_comparison(
        [
            _paired_cell(
                cell_index=0,
                perturbation_config=cfg,
                delta=-0.15,
                delta_ci_low=-0.30,
                delta_ci_high=0.05,  # straddles 0
                p_value=0.20,
            )
        ]
    )

    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]), paired_comparison=paired)
    assert len(diff.cell_flips) == 1
    flip = diff.cell_flips[0]
    assert flip.paired is True
    assert flip.verdict == VERDICT_WITHIN_NOISE


def test_paired_ci_clear_regression_is_regressed() -> None:
    """Paired CI entirely below 0 + low p-value → regressed."""
    cfg = {"x": 0.5}
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.8)
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.55)
    paired = _paired_comparison(
        [
            _paired_cell(
                cell_index=0,
                perturbation_config=cfg,
                delta=-0.25,
                delta_ci_low=-0.40,
                delta_ci_high=-0.10,  # entirely negative
                p_value=0.001,
            )
        ]
    )

    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]), paired_comparison=paired)
    assert diff.cell_flips[0].verdict == VERDICT_REGRESSED


def test_paired_high_pvalue_forces_within_noise() -> None:
    """McNemar p > 0.05 forces within_noise even when CI is clear of zero."""
    cfg = {"x": 0.5}
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.8)
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.6)
    # CI entirely negative, but p high → ambiguous discordant pairs.
    paired = _paired_comparison(
        [
            _paired_cell(
                cell_index=0,
                perturbation_config=cfg,
                delta=-0.20,
                delta_ci_low=-0.30,
                delta_ci_high=-0.05,  # CI clear of zero
                p_value=0.40,  # but p high
            )
        ]
    )

    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]), paired_comparison=paired)
    assert diff.cell_flips[0].verdict == VERDICT_WITHIN_NOISE


# ──────────────────────────────────────────────────────────────────────
# 2 — unpaired Wilson-CI path (fallback when no CRN payload).
# ──────────────────────────────────────────────────────────────────────


def test_unpaired_independent_wilson_overlap_is_within_noise() -> None:
    """A.ci_high overlaps B.ci_low → independent-Wilson delta CI brackets 0."""
    cfg = {"x": 0.5}
    # delta = -0.15. Worst-case delta CI = [b.low - a.high, b.high - a.low]
    # = [0.45 - 0.95, 0.85 - 0.55] = [-0.50, +0.30] → straddles 0.
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.75, ci_low=0.55, ci_high=0.95)
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.60, ci_low=0.45, ci_high=0.85)

    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]))
    assert diff.cell_flips[0].paired is False
    assert diff.cell_flips[0].verdict == VERDICT_WITHIN_NOISE


def test_unpaired_independent_wilson_no_overlap_is_regressed() -> None:
    """Tight CIs that don't overlap → unpaired delta CI clear of zero."""
    cfg = {"x": 0.5}
    # a.ci=[0.85, 0.95], b.ci=[0.55, 0.70].
    # delta CI = [0.55 - 0.95, 0.70 - 0.85] = [-0.40, -0.15] → all negative.
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.90, ci_low=0.85, ci_high=0.95)
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.62, ci_low=0.55, ci_high=0.70)

    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]))
    assert diff.cell_flips[0].verdict == VERDICT_REGRESSED


# ──────────────────────────────────────────────────────────────────────
# 3 — legacy fallback when CIs missing.
# ──────────────────────────────────────────────────────────────────────


def test_legacy_no_ci_falls_back_to_point_delta() -> None:
    """Pre-B-03 CellBreakdowns (no CIs) → verdict mirrors point delta."""
    cfg = {"x": 0.5}
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.80)  # no CIs
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.60)

    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]))
    flip = diff.cell_flips[0]
    assert flip.verdict == VERDICT_REGRESSED  # delta < 0


def test_legacy_no_ci_improvement_path() -> None:
    cfg = {"x": 0.5}
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.50)
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.75)

    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]))
    assert diff.cell_flips[0].verdict == VERDICT_IMPROVED


# ──────────────────────────────────────────────────────────────────────
# 4 — render filtering + show_noise toggle.
# ──────────────────────────────────────────────────────────────────────


def test_render_text_hides_noise_by_default() -> None:
    cfg = {"x": 0.5}
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.75, ci_low=0.55, ci_high=0.95)
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.60, ci_low=0.45, ci_high=0.85)
    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]))

    text = render_text(diff)
    # The verdict marker should NOT appear in the rendered diff because
    # the noise flip is suppressed; only the explanatory line is visible.
    assert "[NOISE]" not in text
    assert "suppressed as within-noise" in text
    assert "--show-noise" in text


def test_render_text_show_noise_surfaces_flip() -> None:
    cfg = {"x": 0.5}
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.75, ci_low=0.55, ci_high=0.95)
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.60, ci_low=0.45, ci_high=0.85)
    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]))

    text = render_text(diff, show_noise=True)
    assert "[NOISE]" in text
    assert "cell 0" in text


def test_render_text_marks_regressed_and_improved() -> None:
    cfg_reg = {"x": 0.0}
    cfg_imp = {"x": 1.0}
    a = _report(
        per_cell=[
            _cell(
                cell_index=0,
                perturbation_config=cfg_reg,
                success_rate=0.90,
                ci_low=0.85,
                ci_high=0.95,
            ),
            _cell(
                cell_index=1,
                perturbation_config=cfg_imp,
                success_rate=0.40,
                ci_low=0.30,
                ci_high=0.50,
            ),
        ]
    )
    b = _report(
        per_cell=[
            _cell(
                cell_index=0,
                perturbation_config=cfg_reg,
                success_rate=0.62,
                ci_low=0.55,
                ci_high=0.70,
            ),
            _cell(
                cell_index=1,
                perturbation_config=cfg_imp,
                success_rate=0.78,
                ci_low=0.70,
                ci_high=0.85,
            ),
        ]
    )
    diff = diff_reports(a, b)
    text = render_text(diff)
    assert "[REGRESSED]" in text
    assert "[IMPROVED]" in text


# ──────────────────────────────────────────────────────────────────────
# 5 — backwards-compat: legacy diff JSON without verdict still loads.
# ──────────────────────────────────────────────────────────────────────


def test_cell_flip_json_includes_verdict_field() -> None:
    """JSON schema check: serialised CellFlip carries verdict."""
    cfg = {"x": 0.5}
    a = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.80)
    b = _cell(cell_index=0, perturbation_config=cfg, success_rate=0.50)
    diff = diff_reports(_report(per_cell=[a]), _report(per_cell=[b]))

    payload = json.loads(diff.model_dump_json())
    assert payload["cell_flips"][0]["verdict"] == VERDICT_REGRESSED


def test_legacy_cell_flip_json_without_verdict_round_trips() -> None:
    """Old diff.json (pre-B-20) loads with verdict=None — backwards-compat."""
    legacy = {
        "cell_index": 0,
        "perturbation_config": {"x": 0.5},
        "a_success_rate": 0.8,
        "b_success_rate": 0.6,
        "direction": "regressed",
    }
    flip = CellFlip.model_validate(legacy)
    assert flip.verdict is None
    assert flip.paired is False


# ──────────────────────────────────────────────────────────────────────
# 6 — CLI integration: --show-noise flag wires through.
# ──────────────────────────────────────────────────────────────────────


def _write_report_json(path, report: Report) -> None:
    path.write_text(report.model_dump_json(indent=2))


def test_cli_diff_show_noise_flag_surfaces_noise(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = {"x": 0.5}
    a = _report(
        per_cell=[
            _cell(
                cell_index=0,
                perturbation_config=cfg,
                success_rate=0.75,
                ci_low=0.55,
                ci_high=0.95,
            )
        ]
    )
    b = _report(
        per_cell=[
            _cell(
                cell_index=0,
                perturbation_config=cfg,
                success_rate=0.60,
                ci_low=0.45,
                ci_high=0.85,
            )
        ]
    )
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    _write_report_json(a_path, a)
    _write_report_json(b_path, b)

    runner = CliRunner()

    # Default: noise suppressed.
    result_default = runner.invoke(app, ["diff", str(a_path), str(b_path)])
    assert result_default.exit_code == 0, result_default.output
    assert "[NOISE]" not in result_default.output
    assert "suppressed as within-noise" in result_default.output

    # With --show-noise: surfaced.
    result_show = runner.invoke(app, ["diff", str(a_path), str(b_path), "--show-noise"])
    assert result_show.exit_code == 0, result_show.output
    assert "[NOISE]" in result_show.output

    # JSON output always carries the verdict regardless of --show-noise.
    # The CLI emits the JSON payload via ``typer.echo`` (stdout) and the
    # one-line summary via ``_echo_err`` (stderr / Console). With the
    # default CliRunner, the summary lives in ``result.output`` alongside
    # stdout, so isolate the JSON block by braces.
    result_json = runner.invoke(app, ["diff", str(a_path), str(b_path), "--json"])
    assert result_json.exit_code == 0, result_json.output
    out = result_json.output
    payload = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert payload["cell_flips"][0]["verdict"] == VERDICT_WITHIN_NOISE
