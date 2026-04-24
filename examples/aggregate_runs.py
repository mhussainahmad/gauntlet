"""Aggregate every ``report.json`` under a directory into a fleet report.

Mirrors the ``gauntlet aggregate`` CLI subcommand but uses the Python
API directly so you can hook the result into a notebook, a CI step,
or a custom dashboard.

Usage:
    uv run python examples/aggregate_runs.py \
        --runs ./runs \
        --out ./fleet-out \
        --persistence-threshold 0.5

Reads every file literally named ``report.json`` recursively under
``--runs`` (the layout ``gauntlet run --out <run-dir>`` produces),
aggregates them via :func:`gauntlet.aggregate.aggregate_directory`,
and writes ``fleet_report.json`` plus ``fleet_report.html`` into
``--out``. Set ``--no-html`` to skip the HTML render.

The script also prints a short summary to stderr — the same one the
CLI emits — so you can wire it into shell pipelines without parsing
the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gauntlet.aggregate import aggregate_directory, write_fleet_html


def main(
    *,
    runs_dir: Path,
    out_dir: Path,
    persistence_threshold: float,
    write_html: bool,
) -> None:
    fleet = aggregate_directory(
        runs_dir,
        persistence_threshold=persistence_threshold,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fleet_json = out_dir / "fleet_report.json"
    fleet_json.write_text(
        json.dumps(fleet.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {fleet_json}", file=sys.stderr)
    if write_html:
        fleet_html = out_dir / "fleet_report.html"
        write_fleet_html(fleet, fleet_html)
        print(f"Wrote {fleet_html}", file=sys.stderr)
    print(
        f"  fleet: {fleet.n_runs} runs / {fleet.n_total_episodes} episodes "
        f"(mean success: {fleet.mean_success_rate * 100:.1f}%)",
        file=sys.stderr,
    )
    print(
        f"  persistent failure clusters: {len(fleet.persistent_failure_clusters)} "
        f"(threshold >={persistence_threshold:.2f})",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Aggregate every report.json under --runs into a fleet meta-report.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Directory to recursively scan for report.json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory; created if missing.",
    )
    parser.add_argument(
        "--persistence-threshold",
        type=float,
        default=0.5,
        help="Cluster appears in >= this fraction of runs to be flagged. Default 0.5.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip the HTML render; write JSON only.",
    )
    args = parser.parse_args()
    main(
        runs_dir=args.runs,
        out_dir=args.out,
        persistence_threshold=args.persistence_threshold,
        write_html=not args.no_html,
    )
