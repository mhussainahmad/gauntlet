"""Build a self-contained dashboard SPA for a synthetic fleet of runs.

Usage:
    uv run python examples/build_fleet_dashboard.py [--out OUT_DIR]

Demonstrates :func:`gauntlet.dashboard.build_dashboard` end-to-end
without needing a real ``gauntlet run`` invocation: synthesises a tiny
fleet of three :class:`Report` objects (one healthy, one with a single
failing cell, one with a different failing cell), writes them into the
on-disk layout that ``gauntlet run --out <dir>`` produces (one
``run-*/report.json`` per run under a common parent), then materialises
the dashboard SPA into ``--out``.

The SPA is openable from a ``file://`` path with no web server (per
RFC 020 §3 — all run data is embedded inline as a JSON literal so the
HTML works even under Chromium's strict same-origin file CORS).

The script prints an ``open file://...`` hint pointing at
``index.html`` so the user can copy-paste straight into a browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gauntlet.dashboard import build_dashboard
from gauntlet.report import Report, build_report
from gauntlet.runner import Episode

__all__ = ["main"]


_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_OUT: Path = _REPO_ROOT / "out-dashboard"


def _synthetic_episode(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
    seed: int,
    suite_name: str,
) -> Episode:
    """Hand-build one :class:`Episode` without touching MuJoCo / pybullet.

    Mirrors the helper used by ``tests/test_dashboard.py`` so the
    example stays in sync with how the dashboard's input is shaped in
    the unit tests.
    """
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=seed,
        perturbation_config=dict(config),
        success=success,
        terminated=success,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
    )


def _synthetic_report(
    *,
    suite_name: str,
    failing_cell: tuple[float, float] | None,
) -> Report:
    """3 x 3 grid x 4 episodes per cell = 36 episodes per report.

    ``failing_cell`` is the ``(lighting, texture)`` pair that should
    fail every episode; ``None`` makes every episode succeed.
    """
    eps: list[Episode] = []
    cell_idx = 0
    for lighting in (0.3, 0.6, 0.9):
        for texture in (0.0, 1.0, 2.0):
            for ep_i in range(4):
                cfg = {"lighting_intensity": lighting, "object_texture": texture}
                fails = failing_cell is not None and (lighting, texture) == failing_cell
                eps.append(
                    _synthetic_episode(
                        cell_index=cell_idx,
                        episode_index=ep_i,
                        success=not fails,
                        config=cfg,
                        seed=cell_idx * 100 + ep_i,
                        suite_name=suite_name,
                    )
                )
            cell_idx += 1
    return build_report(eps, suite_env="tabletop")


def _write_run(base: Path, report: Report, *, run_name: str) -> Path:
    """Mirror the on-disk layout produced by ``gauntlet run --out``.

    Each run gets its own subdirectory named ``run_name`` containing a
    single ``report.json`` written with ``allow_nan=False`` to match
    the CLI's NaN-safe encoder.
    """
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    report_json = run_dir / "report.json"
    report_json.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report_json


def main(*, out_dir: Path = _DEFAULT_OUT) -> None:
    """Build the synthetic fleet, render the dashboard, print the open hint.

    Args:
        out_dir: directory the SPA is materialised into. Created if
            missing. ``out_dir / "index.html"`` is the entry point.
            Re-runs over the same directory overwrite the three SPA
            files in place; other files in ``out_dir`` are left
            untouched.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    if runs_dir.exists():
        # Clean stale fleet from a previous invocation so the dashboard
        # reflects exactly the three runs this script writes.
        for child in runs_dir.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(runs_dir.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        runs_dir.rmdir()
    runs_dir.mkdir(parents=True, exist_ok=False)

    # Three runs: healthy, one cluster failing on (0.3, 0.0), one
    # cluster failing on (0.9, 2.0). Enough variety for the dashboard
    # to show non-trivial summary scalars and per-axis curves.
    fleet: list[tuple[str, tuple[float, float] | None]] = [
        ("run-healthy", None),
        ("run-fail-low-light", (0.3, 0.0)),
        ("run-fail-high-light", (0.9, 2.0)),
    ]
    for run_name, failing in fleet:
        report = _synthetic_report(suite_name="dashboard-demo", failing_cell=failing)
        _write_run(runs_dir, report, run_name=run_name)

    spa_dir = out_dir / "spa"
    build_dashboard(runs_dir, spa_dir, title="Gauntlet Demo Fleet")

    index_html = (spa_dir / "index.html").resolve()
    print(f"Wrote {len(fleet)} synthetic runs under {runs_dir}")
    print(f"Built dashboard SPA into {spa_dir}")
    print(f"  index.html: {index_html}")
    print(f"  dashboard.js: {(spa_dir / 'dashboard.js').resolve()}")
    print(f"  dashboard.css: {(spa_dir / 'dashboard.css').resolve()}")
    print(f"open file://{index_html} in your browser to view the dashboard.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained dashboard SPA for a synthetic fleet.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory (default: {_DEFAULT_OUT}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(out_dir=args.out)
