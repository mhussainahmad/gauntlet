"""Run :class:`RandomPolicy` against the LHS smoke suite.

Usage:
    uv run python examples/evaluate_random_policy_lhs.py [--out OUT_DIR]
                                                         [--suite SUITE_YAML]
                                                         [--n-workers N]

The companion to ``evaluate_random_policy.py`` for the Latin Hypercube
Sampling code path. Drives the bundled
``examples/suites/tabletop-lhs-smoke.yaml`` — a five-axis sweep that
would be 3,125 cells under the historical cartesian grid (5 ** 5) but
collapses to 32 LHS samples while keeping every axis covered at 32
distinct strata.

Outputs the same three artefacts as the cartesian example
(``episodes.json``, ``report.json``, ``report.html``) so a researcher
can A/B the cartesian and LHS coverage of the same policy by running
the two scripts side-by-side and diffing the per-axis breakdowns.

NaN-safe JSON: empty heatmap cells round-trip as ``null`` rather than
the literal ``NaN`` Pydantic v2 would emit by default — matches what
``gauntlet run`` writes from the CLI.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from gauntlet.env import TabletopEnv
from gauntlet.policy import RandomPolicy
from gauntlet.report import Report, build_report, write_html
from gauntlet.report.html import _nan_to_none
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main"]


_TABLETOP_ACTION_DIM: int = 7
_SMOKE_MAX_STEPS: int = 20

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-lhs-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out-lhs"


def _build_env_factory(max_steps: int) -> Callable[[], TabletopEnv]:
    """Picklable env factory capping rollout length for the smoke run."""
    return partial(TabletopEnv, max_steps=max_steps)


def _build_policy_factory(action_dim: int) -> Callable[[], RandomPolicy]:
    """Picklable factory producing a fresh :class:`RandomPolicy` per episode."""
    return partial(RandomPolicy, action_dim=action_dim)


def _episodes_to_json(episodes: list[Episode]) -> str:
    payload: list[dict[str, Any]] = [ep.model_dump(mode="json") for ep in episodes]
    cleaned = _nan_to_none(payload)
    return json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _report_to_json(report: Report) -> str:
    cleaned = _nan_to_none(report.model_dump(mode="json"))
    return json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main(
    *,
    suite_path: Path = _DEFAULT_SUITE,
    out_dir: Path = _DEFAULT_OUT,
    n_workers: int = 2,
    max_steps: int = _SMOKE_MAX_STEPS,
) -> None:
    """Run the LHS evaluation pipeline and write the three artefacts.

    Args:
        suite_path: YAML suite to load. Defaults to the LHS smoke suite
            shipped in this repo.
        out_dir: Output directory; created if missing. Receives
            ``episodes.json``, ``report.json``, ``report.html``.
        n_workers: Worker processes. ``1`` triggers the in-process fast
            path; ``>= 2`` uses the multiprocessing pool.
        max_steps: Hard cap on per-episode env steps. Lowered from the
            ``TabletopEnv`` default of 200 so the smoke run finishes in
            seconds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suite: Suite = load_suite(suite_path)

    runner = Runner(
        n_workers=n_workers,
        env_factory=_build_env_factory(max_steps),
    )
    episodes: list[Episode] = runner.run(
        policy_factory=_build_policy_factory(_TABLETOP_ACTION_DIM),
        suite=suite,
    )

    report: Report = build_report(episodes)

    (out_dir / "episodes.json").write_text(_episodes_to_json(episodes), encoding="utf-8")
    (out_dir / "report.json").write_text(_report_to_json(report), encoding="utf-8")
    write_html(report, out_dir / "report.html")

    # The headline number that motivates this script existing: how
    # many cells did we save vs the equivalent cartesian sweep?
    n_axes = len(suite.axes)
    cart_5x_equivalent = 5**n_axes
    print(
        f"Wrote {len(episodes)} episodes / {len(report.per_cell)} LHS cells "
        f"-> {out_dir} (success: {report.overall_success_rate * 100:.1f}%). "
        f"A 5-step cartesian sweep over the same {n_axes} axes would have "
        f"required {cart_5x_equivalent} cells."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RandomPolicy against the tabletop LHS smoke suite.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=_DEFAULT_SUITE,
        help=f"Suite YAML to evaluate (default: {_DEFAULT_SUITE}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=2,
        help="Worker processes (default: 2).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_SMOKE_MAX_STEPS,
        help=f"Per-episode step cap (default: {_SMOKE_MAX_STEPS}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        suite_path=args.suite,
        out_dir=args.out,
        n_workers=args.n_workers,
        max_steps=args.max_steps,
    )
