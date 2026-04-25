"""``git diff``-style plain-text rendering of a :class:`ReportDiff`.

The renderer emits ANSI-free plain text — color is applied at the CLI
layer where ``rich.Console`` is already wired (and where ``NO_COLOR`` /
non-TTY detection already lives). This keeps the rendering pure and
trivially testable: tests assert on substrings, not on escape codes.

Format mirrors the visual grammar of ``git diff``:

* Header lines prefixed ``---`` / ``+++`` carry the labels.
* Single-character signed prefixes ``+`` / ``-`` (and ``=`` for the
  zero-delta case) anchor every numeric row.
* Whole-section headers are wrapped in ``@@ ... @@`` to match hunk
  delimiters from unified-diff output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gauntlet.diff.diff import AxisDelta, CellFlip, ClusterDelta, ReportDiff
    from gauntlet.report import FailureCluster

__all__ = ["render_text"]


def _sign(delta: float) -> str:
    """Single-char prefix for a signed delta (``+`` / ``-`` / ``=``)."""
    if delta > 0:
        return "+"
    if delta < 0:
        return "-"
    return "="


def _fmt_pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _fmt_signed_pct(delta: float) -> str:
    return f"{delta * 100:+6.1f}%"


def _fmt_config(config: dict[str, float]) -> str:
    """Stable render of ``{axis: value}`` (sorted by axis name)."""
    items = sorted(config.items())
    return "{" + ", ".join(f"{k}={v:g}" for k, v in items) + "}"


def _render_axis(axis_delta: AxisDelta) -> list[str]:
    lines: list[str] = []
    if not axis_delta.rate_deltas:
        return lines
    lines.append(f"  axis: {axis_delta.name}")
    for value in sorted(axis_delta.rate_deltas.keys()):
        delta = axis_delta.rate_deltas[value]
        lines.append(f"    {_sign(delta)} {axis_delta.name}={value:g}: {_fmt_signed_pct(delta)}")
    return lines


def _render_flip(flip: CellFlip) -> str:
    sign = "-" if flip.direction == "regressed" else "+"
    delta = flip.b_success_rate - flip.a_success_rate
    head = (
        f"  {sign} cell {flip.cell_index} {_fmt_config(flip.perturbation_config)}: "
        f"{_fmt_pct(flip.a_success_rate)} -> {_fmt_pct(flip.b_success_rate)} "
        f"({_fmt_signed_pct(delta)}, {flip.direction})"
    )
    # Paired CRN attribution (B-08): when ``--paired`` ran, the delta
    # carries a much tighter Wald CI plus a McNemar p-value. Surface
    # both inline so the user sees "regressed by 8% with paired CI
    # [-12%, -4%], p=0.01" instead of just the point estimate. The
    # unpaired path keeps the legacy single-line render.
    if flip.paired and flip.delta_ci_low is not None and flip.delta_ci_high is not None:
        suffix_parts = [
            f"paired CI [{_fmt_signed_pct(flip.delta_ci_low)}, "
            f"{_fmt_signed_pct(flip.delta_ci_high)}]"
        ]
        if flip.mcnemar_p_value is not None:
            suffix_parts.append(f"McNemar p={flip.mcnemar_p_value:.3f}")
        return head + " [" + "; ".join(suffix_parts) + "]"
    return head


def _render_cluster(cluster: FailureCluster, *, sign: str) -> str:
    return (
        f"  {sign} cluster {_fmt_config(cluster.axes)}: "
        f"failure_rate={_fmt_pct(cluster.failure_rate)} lift={cluster.lift:.2f}x "
        f"(n={cluster.n_episodes})"
    )


def _render_intensified(delta: ClusterDelta) -> str:
    return (
        f"  ! cluster {_fmt_config(delta.axes)}: "
        f"lift {delta.a_lift:.2f}x -> {delta.b_lift:.2f}x "
        f"(+{delta.delta:.2f}x)"
    )


def render_text(diff: ReportDiff) -> str:
    """Render a :class:`ReportDiff` as a plain-text ``git diff``-style block.

    Returned string ends with a newline and contains no ANSI escape
    sequences. The shape is stable (and substring-tested in
    ``tests/test_diff.py``):

    .. code-block:: text

        --- a: <a_label> (<a_suite_name>)
        +++ b: <b_label> (<b_suite_name>)
        @@ overall @@
          <sign> overall_success_rate: <signed_pct>
          <sign> n_episodes: <signed_int>
        @@ axes @@
          axis: <name>
            <sign> <name>=<value>: <signed_pct>
            ...
        @@ cell flips @@
          <sign> cell <i> {axis=value, ...}: <pct_a> -> <pct_b> (<signed_pct>, <direction>)
        @@ failure clusters @@
          - cluster {...}: failure_rate=<pct> lift=<x>x (n=<n>)
          + cluster {...}: failure_rate=<pct> lift=<x>x (n=<n>)
          ! cluster {...}: lift <x>x -> <y>x (+<d>x)
    """
    lines: list[str] = []
    lines.append(f"--- a: {diff.a_label} ({diff.a_suite_name})")
    lines.append(f"+++ b: {diff.b_label} ({diff.b_suite_name})")

    # Paired CRN tag — single line so a piped consumer can grep "paired:"
    # to know whether deltas carry the variance-reduced CI bracket
    # (B-08). Rendered above the @@ overall @@ hunk so it's the first
    # thing the human sees, mirroring how ``--no-color`` git diff puts
    # mode-change lines above the header.
    if diff.paired and diff.paired_comparison is not None:
        lines.append(
            f"paired: true (master_seed={diff.paired_comparison.master_seed}, "
            f"n_paired_episodes={diff.paired_comparison.n_paired_episodes})"
        )

    # Overall hunk.
    lines.append("@@ overall @@")
    overall_sign = _sign(diff.overall_success_rate_delta)
    lines.append(
        f"  {overall_sign} overall_success_rate: {_fmt_signed_pct(diff.overall_success_rate_delta)}"
    )
    n_sign = _sign(diff.n_episodes_delta)
    lines.append(f"  {n_sign} n_episodes: {diff.n_episodes_delta:+d}")

    # Per-axis hunk.
    axis_lines: list[str] = []
    for axis_name in sorted(diff.axis_deltas.keys()):
        axis_lines.extend(_render_axis(diff.axis_deltas[axis_name]))
    if axis_lines:
        lines.append("@@ axes @@")
        lines.extend(axis_lines)

    # Cell-flip hunk.
    if diff.cell_flips:
        lines.append("@@ cell flips @@")
        for flip in diff.cell_flips:
            lines.append(_render_flip(flip))

    # Failure-cluster hunk.
    if diff.cluster_added or diff.cluster_removed or diff.cluster_intensified:
        lines.append("@@ failure clusters @@")
        for cluster in diff.cluster_removed:
            lines.append(_render_cluster(cluster, sign="-"))
        for cluster in diff.cluster_added:
            lines.append(_render_cluster(cluster, sign="+"))
        for delta in diff.cluster_intensified:
            lines.append(_render_intensified(delta))

    # If literally nothing moved, surface the headline so the output is
    # never confusing-empty.
    moved = bool(
        axis_lines
        or diff.cell_flips
        or diff.cluster_added
        or diff.cluster_removed
        or diff.cluster_intensified
        or diff.overall_success_rate_delta != 0.0
        or diff.n_episodes_delta != 0
    )
    if not moved:
        lines.append("  (no differences above the configured thresholds)")

    return "\n".join(lines) + "\n"
