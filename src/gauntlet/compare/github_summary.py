"""GitHub Actions step-summary markdown for ``gauntlet compare`` — B-24.

Consumes the ``compare.json`` payload built by
:func:`gauntlet.cli._build_compare` (a JSON-serialisable mapping with
keys ``a``, ``b``, ``delta_success_rate``, ``regressions``,
``improvements``, ``threshold``, ``min_cell_size``) and renders a
markdown blob suitable for piping into ``$GITHUB_STEP_SUMMARY``.

Failure-first per PRODUCT.md — the heading leads with the regression
count and worst-cell axis combination; improvements are reported in a
separate, secondary table.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["to_github_summary"]


def _as_mapping(obj: object) -> Mapping[str, object]:
    """Narrow an arbitrary JSON value to a ``str``-keyed mapping or empty."""
    if isinstance(obj, Mapping):
        # Defensive: mapping keys in JSON-decoded payloads are always str.
        return {str(k): v for k, v in obj.items()}
    return {}


def _as_list_of_mappings(obj: object) -> list[Mapping[str, object]]:
    """Narrow an arbitrary JSON value to a list of ``str``-keyed mappings."""
    if not isinstance(obj, list):
        return []
    return [_as_mapping(item) for item in obj]


def _as_float(obj: object, default: float = 0.0) -> float:
    """Coerce a JSON value to ``float`` with a default for ``None`` / wrong type."""
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    return default


def _as_int(obj: object, default: int = 0) -> int:
    """Coerce a JSON value to ``int`` with a default for ``None`` / wrong type."""
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return int(obj)
    return default


def _fmt_axes(axis_combination: Mapping[str, object]) -> str:
    """Render an axis_combination dict as a stable, terse string."""
    if not axis_combination:
        return "(baseline)"
    return ", ".join(f"{k}={v}" for k, v in sorted(axis_combination.items()))


def _fmt_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _fmt_signed_pct(delta: float) -> str:
    return f"{delta * 100:+.1f}%"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _render_table(rows: list[Mapping[str, object]], heading: str) -> list[str]:
    """Render a regressions / improvements table block."""
    lines: list[str] = [heading, ""]
    lines.append(_row(["Cell", "Rate A", "Rate B", "Delta", "n A", "n B"]))
    lines.append(_row(["---", "---", "---", "---", "---", "---"]))
    for r in rows:
        axes = _as_mapping(r.get("axis_combination"))
        lines.append(
            _row(
                [
                    f"`{_fmt_axes(axes)}`",
                    _fmt_pct(_as_float(r.get("rate_a"))),
                    _fmt_pct(_as_float(r.get("rate_b"))),
                    _fmt_signed_pct(_as_float(r.get("delta"))),
                    str(_as_int(r.get("n_episodes_a"))),
                    str(_as_int(r.get("n_episodes_b"))),
                ]
            )
        )
    lines.append("")
    return lines


def to_github_summary(compare_result: Mapping[str, object]) -> str:
    """Render a compare payload as a GitHub-step-summary markdown blob.

    ``compare_result`` is the payload returned by
    :func:`gauntlet.cli._build_compare` (also written to
    ``compare.json``). The output is plain CommonMark and can be
    written directly to ``$GITHUB_STEP_SUMMARY``.
    """
    a = _as_mapping(compare_result.get("a"))
    b = _as_mapping(compare_result.get("b"))
    regressions = _as_list_of_mappings(compare_result.get("regressions"))
    improvements = _as_list_of_mappings(compare_result.get("improvements"))
    delta = _as_float(compare_result.get("delta_success_rate"))

    lines: list[str] = []
    if regressions:
        worst_axes = _fmt_axes(_as_mapping(regressions[0].get("axis_combination")))
        lines.append(f"## {len(regressions)} regression(s); worst cell: `{worst_axes}`")
    else:
        lines.append("## No regressions")
    lines.append("")
    lines.append(
        f"- Run A: `{a.get('name', '?')}` "
        f"success {_fmt_pct(_as_float(a.get('overall_success_rate')))} "
        f"({_as_int(a.get('n_episodes'))} episodes)"
    )
    lines.append(
        f"- Run B: `{b.get('name', '?')}` "
        f"success {_fmt_pct(_as_float(b.get('overall_success_rate')))} "
        f"({_as_int(b.get('n_episodes'))} episodes)"
    )
    lines.append(f"- Overall delta: **{_fmt_signed_pct(delta)}**")
    lines.append(
        f"- Threshold: {_as_float(compare_result.get('threshold'))}, "
        f"min cell size: {_as_int(compare_result.get('min_cell_size'))}"
    )
    lines.append("")

    if regressions:
        lines.extend(_render_table(regressions, "### Regressions"))
    if improvements:
        lines.extend(_render_table(improvements, "### Improvements"))

    return "\n".join(lines).rstrip() + "\n"
