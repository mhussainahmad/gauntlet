"""Run :class:`RandomPolicy` against the tabletop smoke suite.

Usage:
    uv run python examples/evaluate_random_policy.py [--out OUT_DIR]
                                                     [--suite SUITE_YAML]
                                                     [--n-workers N]

This is the README quickstart's CLI invocation translated to the public
Python API — the script a researcher writing notebook code would crib
from to learn what ``gauntlet`` looks like under the hood. Three
artefacts land in ``--out`` (default ``out/``):

* ``episodes.json`` — one record per rollout (consume with
  ``Episode.model_validate`` or feed straight back to
  ``gauntlet compare``).
* ``report.json`` — the analysed :class:`Report` (per-axis breakdowns,
  per-cell aggregates, failure clusters, 2D heatmaps).
* ``report.html`` — self-contained HTML rendering of the above. Open in
  any browser; no server / build step required.

Both JSON files are NaN-safe: empty heatmap cells (no episodes for that
2D combination) are emitted as ``null`` rather than the literal ``NaN``
that Pydantic v2 would otherwise produce, so they round-trip through
strict-mode :func:`json.loads`. This mirrors the same trick the CLI
applies in :mod:`gauntlet.cli` — see ``_nan_to_none``.
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


# Action dimension exposed by TabletopEnv (see env.tabletop module
# docstring: [dx, dy, dz, drx, dry, drz, gripper]).
_TABLETOP_ACTION_DIM: int = 7

# Cap rollout length. The default is 200 — fine for a real eval, but for
# this smoke example we want each episode to wrap up in a fraction of a
# second so the whole loop finishes in well under a minute.
_SMOKE_MAX_STEPS: int = 20

# Default suite + output paths, resolved relative to the repo root so the
# script Just Works from any working directory.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out"


def _build_env_factory(max_steps: int) -> Callable[[], TabletopEnv]:
    """Return a picklable env factory that caps rollout length.

    ``functools.partial`` over the class is the canonical
    spawn-friendly factory — closures defined inside ``main`` would
    refuse to pickle under ``n_workers >= 2``.
    """
    return partial(TabletopEnv, max_steps=max_steps)


def _build_policy_factory(action_dim: int) -> Callable[[], RandomPolicy]:
    """Return a picklable factory that builds a fresh :class:`RandomPolicy`.

    The Runner re-seeds the policy per episode via ``policy.reset(rng)``
    (when the policy implements :class:`ResettablePolicy`), so the
    constructor seed argument is irrelevant — every rollout still gets a
    deterministic, decorrelated stream from the master seed.
    """
    return partial(RandomPolicy, action_dim=action_dim)


def _episodes_to_json(episodes: list[Episode]) -> str:
    """Serialise *episodes* to a NaN-safe, indent=2 JSON string.

    Mirrors :func:`gauntlet.cli._write_json` so the output of this script
    is byte-comparable with what ``gauntlet run`` writes.
    """
    payload: list[dict[str, Any]] = [ep.model_dump(mode="json") for ep in episodes]
    cleaned = _nan_to_none(payload)
    return json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _report_to_json(report: Report) -> str:
    """Serialise *report* to a NaN-safe, indent=2 JSON string.

    The ``heatmap_2d`` payload contains ``float('nan')`` for empty cells;
    plain ``model_dump_json`` would emit literal ``NaN`` (invalid JSON).
    """
    cleaned = _nan_to_none(report.model_dump(mode="json"))
    return json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main(
    *,
    suite_path: Path = _DEFAULT_SUITE,
    out_dir: Path = _DEFAULT_OUT,
    n_workers: int = 2,
    max_steps: int = _SMOKE_MAX_STEPS,
) -> None:
    """Run the full evaluation pipeline and write the three artefacts.

    Args:
        suite_path: YAML suite to load. Defaults to the smoke suite that
            ships with this repo.
        out_dir: Output directory; created if missing. Receives
            ``episodes.json``, ``report.json``, and ``report.html``.
        n_workers: Worker processes. ``1`` triggers the in-process fast
            path; ``>= 2`` uses the multiprocessing pool. Defaults to
            ``2`` — enough to demonstrate parallelism without thrashing.
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

    episodes_path = out_dir / "episodes.json"
    report_json_path = out_dir / "report.json"
    report_html_path = out_dir / "report.html"

    episodes_path.write_text(_episodes_to_json(episodes), encoding="utf-8")
    report_json_path.write_text(_report_to_json(report), encoding="utf-8")
    write_html(report, report_html_path)

    print(
        f"Wrote {len(episodes)} episodes / {len(report.per_cell)} cells "
        f"-> {out_dir} (success: {report.overall_success_rate * 100:.1f}%)"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RandomPolicy against the tabletop smoke suite.",
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
