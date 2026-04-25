"""Run :class:`RandomPolicy` against the Sobol smoke suite.

Usage:
    uv run python examples/evaluate_random_policy_sobol.py [--out OUT_DIR]
                                                           [--suite SUITE_YAML]
                                                           [--n-workers N]

The companion to ``evaluate_random_policy_lhs.py`` for the Sobol
low-discrepancy code path. Drives the bundled
``examples/suites/tabletop-sobol-smoke.yaml`` — the same five-axis
perturbation surface as the LHS demo, but drawn from the Joe-Kuo
6.21201 Sobol sequence.

When does Sobol win over LHS? LHS only guarantees marginal (1-D)
coverage — every axis is sampled at ``n_samples`` distinct strata,
but the joint distribution of any two axes can still exhibit
clumping. Sobol additionally bounds the discrepancy of every 2-D
(and higher) projection of the point set. That extra guarantee
matters when the failure mode you are hunting lives on a *joint* of
two perturbation axes (e.g. high lighting AND offset camera together)
rather than either axis alone. Sobol also has the practical bonus of
being fully deterministic — re-running with a different ``seed``
produces byte-identical cells, which simplifies regression bisection.

Outputs the same three artefacts as the LHS example
(``episodes.json``, ``report.json``, ``report.html``) so a researcher
can A/B the LHS and Sobol coverage of the same policy by running the
two scripts side-by-side and diffing the per-axis breakdowns.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

from gauntlet.env import TabletopEnv
from gauntlet.env.base import GauntletEnv
from gauntlet.policy import RandomPolicy
from gauntlet.report import Report, build_report, write_html
from gauntlet.report.html import _nan_to_none
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main"]


_TABLETOP_ACTION_DIM: int = 7
_SMOKE_MAX_STEPS: int = 20

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-sobol-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out-sobol"


def _build_env_factory(max_steps: int) -> Callable[[], GauntletEnv]:
    """Picklable env factory capping rollout length for the smoke run.

    Cast through :func:`typing.cast` because :class:`functools.partial`
    over ``TabletopEnv`` types as ``partial[TabletopEnv]``, and
    :class:`Runner` declares ``env_factory: Callable[[], GauntletEnv]``
    (invariant in the return type). The cast is sound at runtime —
    ``TabletopEnv`` is a :class:`GauntletEnv` subclass.
    """
    return cast(Callable[[], GauntletEnv], partial(TabletopEnv, max_steps=max_steps))


def _build_policy_factory(action_dim: int) -> Callable[[], RandomPolicy]:
    """Picklable factory producing a fresh :class:`RandomPolicy` per episode."""
    return partial(RandomPolicy, action_dim=action_dim)


def _episodes_to_json(episodes: list[Episode]) -> str:
    payload: list[dict[str, Any]] = [ep.model_dump(mode="json") for ep in episodes]
    # ``_nan_to_none`` types its argument as the recursive ``_JsonValue``
    # alias; ``list`` is invariant under it, so a ``cast`` is the
    # least-noisy way to bridge the variance gap inside an example.
    cleaned = _nan_to_none(cast(Any, payload))
    return json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _report_to_json(report: Report) -> str:
    cleaned = _nan_to_none(cast(Any, report.model_dump(mode="json")))
    return json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main(
    *,
    suite_path: Path = _DEFAULT_SUITE,
    out_dir: Path = _DEFAULT_OUT,
    n_workers: int = 2,
    max_steps: int = _SMOKE_MAX_STEPS,
) -> None:
    """Run the Sobol evaluation pipeline and write the three artefacts.

    Args:
        suite_path: YAML suite to load. Defaults to the Sobol smoke
            suite shipped in this repo.
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

    # The headline numbers: how many cells did we save vs the full
    # cartesian sweep, and how does Sobol's joint-projection bound
    # compare to LHS's marginal-only guarantee.
    n_axes = len(suite.axes)
    cart_5x_equivalent = 5**n_axes
    print(
        f"Wrote {len(episodes)} episodes / {len(report.per_cell)} Sobol cells "
        f"-> {out_dir} (success: {report.overall_success_rate * 100:.1f}%). "
        f"A 5-step cartesian sweep over the same {n_axes} axes would have "
        f"required {cart_5x_equivalent} cells; LHS would also have used "
        f"{len(report.per_cell)} cells but only with a marginal-coverage "
        f"guarantee. Sobol additionally bounds every 2-D projection — "
        f"useful when failures live on joints of axes."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RandomPolicy against the tabletop Sobol smoke suite.",
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
