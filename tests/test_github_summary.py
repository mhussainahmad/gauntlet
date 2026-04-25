"""Tests for :mod:`gauntlet.compare.github_summary` — backlog B-24."""

from __future__ import annotations

import re
from typing import Any

from gauntlet.compare import to_github_summary


def _payload(
    *,
    regressions: list[dict[str, Any]] | None = None,
    improvements: list[dict[str, Any]] | None = None,
    delta: float = 0.0,
) -> dict[str, Any]:
    return {
        "a": {"name": "run-a", "overall_success_rate": 0.8, "n_episodes": 100},
        "b": {"name": "run-b", "overall_success_rate": 0.8 + delta, "n_episodes": 100},
        "delta_success_rate": delta,
        "threshold": 0.1,
        "min_cell_size": 5,
        "paired": False,
        "regressions": regressions or [],
        "improvements": improvements or [],
    }


def _row(
    *,
    axes: dict[str, float],
    rate_a: float,
    rate_b: float,
    delta: float,
) -> dict[str, Any]:
    return {
        "axis_combination": axes,
        "rate_a": rate_a,
        "rate_b": rate_b,
        "delta": delta,
        "n_episodes_a": 10,
        "n_episodes_b": 10,
    }


def test_empty_diff_leads_with_no_regressions() -> None:
    md = to_github_summary(_payload())
    assert md.startswith("## No regressions"), md
    # No tables when both lists empty.
    assert "### Regressions" not in md
    assert "### Improvements" not in md


def test_regressions_only_leads_with_count_and_worst_axes() -> None:
    regs = [
        _row(axes={"lighting": 0.3}, rate_a=0.9, rate_b=0.4, delta=-0.5),
        _row(axes={"lighting": 1.5}, rate_a=0.8, rate_b=0.6, delta=-0.2),
    ]
    md = to_github_summary(_payload(regressions=regs, delta=-0.3))
    # Heading carries the count + the worst-cell axes (first regression).
    assert md.startswith("## 2 regression(s)")
    assert "lighting=0.3" in md.splitlines()[0]
    # Regressions table is present, improvements table is NOT.
    assert "### Regressions" in md
    assert "### Improvements" not in md
    # Each regression row shows up.
    assert "lighting=0.3" in md
    assert "lighting=1.5" in md
    # Delta column shows signed percent.
    assert "-50.0%" in md
    assert "-20.0%" in md


def test_mixed_diff_emits_both_tables_and_overall_delta() -> None:
    regs = [_row(axes={"a": 1.0}, rate_a=0.9, rate_b=0.5, delta=-0.4)]
    imps = [_row(axes={"b": 2.0}, rate_a=0.4, rate_b=0.7, delta=0.3)]
    md = to_github_summary(_payload(regressions=regs, improvements=imps, delta=-0.05))
    assert "### Regressions" in md
    assert "### Improvements" in md
    # Overall delta surfaces.
    assert "-5.0%" in md
    # Regressions table appears before improvements table.
    assert md.index("### Regressions") < md.index("### Improvements")


def test_markdown_table_rows_parse_as_pipe_delimited() -> None:
    regs = [_row(axes={"x": 0.1, "y": 2.0}, rate_a=0.9, rate_b=0.6, delta=-0.3)]
    md = to_github_summary(_payload(regressions=regs, delta=-0.3))
    # Find the regressions header row + its data rows.
    lines = md.splitlines()
    header_idx = lines.index("### Regressions")
    table_lines = [ln for ln in lines[header_idx:] if ln.startswith("|") and ln.endswith("|")]
    assert len(table_lines) >= 3  # header, separator, at least one data row
    # Header has 6 columns (Cell, Rate A, Rate B, Delta, n A, n B).
    assert table_lines[0].count("|") == 7
    # Separator row uses dashes.
    assert re.match(r"^(\| --- )+\|$", table_lines[1]) is not None
    # Data row mentions the axis combination (sorted alphabetically).
    assert "x=0.1" in table_lines[2]
    assert "y=2.0" in table_lines[2]
