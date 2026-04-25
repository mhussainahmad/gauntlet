"""Print a structured diff between two ``report.json`` (or ``episodes.json``) runs.

Mirrors the ``gauntlet diff`` CLI subcommand but uses the Python API
directly so you can hook the result into a notebook, a CI step, or a
custom dashboard.

Usage::

    uv run python examples/diff_two_runs.py \\
        --a out_a/report.json --b out_b/report.json

    # Or feed episodes.json directly — same auto-detect as the CLI:
    uv run python examples/diff_two_runs.py \\
        --a out_a/episodes.json --b out_b/episodes.json --json

The script auto-detects ``episodes.json`` (top-level list) vs
``report.json`` (top-level dict) the same way ``gauntlet diff`` does.
``--json`` emits the full :class:`gauntlet.diff.ReportDiff` payload to
stdout for downstream consumption; the default emits the human-readable
``git diff``-style rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gauntlet.diff import diff_reports, render_text
from gauntlet.report import Report, build_report
from gauntlet.runner import Episode


def _load_report(path: Path) -> Report:
    """Auto-detect ``episodes.json`` vs ``report.json`` and return a Report."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        episodes = [Episode.model_validate(item) for item in raw]
        return build_report(episodes)
    if isinstance(raw, dict):
        return Report.model_validate(raw)
    raise SystemExit(
        f"{path}: top-level JSON must be a list (episodes) or dict (report); "
        f"got {type(raw).__name__}"
    )


def main(
    *,
    a_path: Path,
    b_path: Path,
    json_output: bool,
    cell_flip_threshold: float,
    cluster_intensify_threshold: float,
) -> None:
    report_a = _load_report(a_path)
    report_b = _load_report(b_path)

    diff = diff_reports(
        report_a,
        report_b,
        a_label=str(a_path),
        b_label=str(b_path),
        cell_flip_threshold=cell_flip_threshold,
        cluster_intensify_threshold=cluster_intensify_threshold,
    )

    if json_output:
        print(diff.model_dump_json(indent=2))
    else:
        # render_text already terminates with a newline.
        sys.stdout.write(render_text(diff))

    # Headline summary on stderr — survives stdout redirection.
    print(
        f"diffed {a_path} -> {b_path}: "
        f"overall {diff.overall_success_rate_delta * 100:+.1f}%, "
        f"cell flips: {len(diff.cell_flips)}, "
        f"clusters +{len(diff.cluster_added)}/-{len(diff.cluster_removed)}/"
        f"!{len(diff.cluster_intensified)}",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Print a structured diff between two gauntlet runs.",
    )
    parser.add_argument(
        "--a",
        type=Path,
        required=True,
        help="First run: episodes.json or report.json.",
    )
    parser.add_argument(
        "--b",
        type=Path,
        required=True,
        help="Second run: episodes.json or report.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit ReportDiff.model_dump_json(indent=2) instead of text.",
    )
    parser.add_argument(
        "--cell-flip-threshold",
        type=float,
        default=0.10,
        help="Inclusive minimum |delta success_rate| to surface a per-cell flip. Default 0.10.",
    )
    parser.add_argument(
        "--cluster-intensify-threshold",
        type=float,
        default=0.5,
        help="Inclusive minimum (b.lift - a.lift) for cluster intensification. Default 0.5.",
    )
    args = parser.parse_args()
    main(
        a_path=args.a,
        b_path=args.b,
        json_output=args.json,
        cell_flip_threshold=args.cell_flip_threshold,
        cluster_intensify_threshold=args.cluster_intensify_threshold,
    )
