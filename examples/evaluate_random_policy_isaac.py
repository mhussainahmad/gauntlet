"""Run :class:`RandomPolicy` against the tabletop-isaac smoke suite.

Usage:
    uv sync --extra isaac --group isaac-dev
    uv run python examples/evaluate_random_policy_isaac.py [--out OUT_DIR]
                                                            [--suite SUITE_YAML]
                                                            [--n-workers N]

The Isaac Sim counterpart of :mod:`examples.evaluate_random_policy_genesis`.
Mirrors the former's public-Python-API shape, swapping only the env
factory — same Suite schema, same Runner, same Report artefacts.

Notes specific to Isaac Sim (RFC-009):

* **GPU required.** ``isaacsim`` wraps NVIDIA Omniverse Kit; even
  constructing the env spins up Kit which segfaults without a CUDA
  RTX-class GPU. Run this on a developer GPU workstation, not on
  ``ubuntu-latest`` CI.
* **State-only first cut.** Cosmetic axes are not observable on
  ``obs`` (no ``image`` key); the Suite must declare at least one
  state-affecting axis (``object_initial_pose_x/_y`` /
  ``distractor_count``) or it is rejected at load time.
* **Per-process construction cost.** First instantiation of
  ``IsaacSimTabletopEnv`` pays the Kit boot cost (~20-30 s on most
  workstations) plus scene load. With ``n_workers=2`` that is paid
  twice (each worker has its own SimulationApp); subsequent episodes
  in the worker reuse the loaded scene. Defaults to ``n_workers=1``
  for the smoke run.
* **Same-process determinism** (seed + action sequence -> same obs)
  is exact-to-fp-noise. Cross-backend numerical parity is an explicit
  non-goal — a given seed on ``tabletop`` vs ``tabletop-pybullet`` vs
  ``tabletop-genesis`` vs ``tabletop-isaac`` produces four different
  trajectories (RFC-009 §7.3). The report schema is backend-agnostic;
  comparing across backends with ``gauntlet compare --allow-cross-backend``
  measures simulator drift, not policy regression.

Importable when ``isaacsim`` is absent — the heavy import is deferred
to :func:`_build_env_factory` so ``--help`` works on a CPU-only laptop.
A runtime ``ImportError`` is raised the moment the user actually runs
the smoke (which then needs the extra installed and a GPU).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gauntlet.policy import RandomPolicy
from gauntlet.report import Report, build_report, write_html
from gauntlet.report.html import _nan_to_none
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

if TYPE_CHECKING:
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

__all__ = ["main"]


_TABLETOP_ACTION_DIM: int = 7

# Keep each episode short so the smoke run finishes in a couple
# minutes on a GPU workstation. Isaac Sim's `world.step` is cheap per
# tick; Kit boot dominates.
_SMOKE_MAX_STEPS: int = 20

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-isaac-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out-isaac"


def _build_env_factory(max_steps: int) -> Callable[[], IsaacSimTabletopEnv]:
    """Return a picklable env factory that caps rollout length.

    The ``import gauntlet.env.isaac`` lives inside this function (not
    at module top) so the example file IMPORTS cleanly when the
    ``[isaac]`` extra is absent — only invoking the smoke (which
    calls this factory) triggers the install-hint ImportError.

    ``functools.partial`` over the class is the canonical
    spawn-friendly factory pattern shared with the other example
    scripts.
    """
    from gauntlet.env.isaac import IsaacSimTabletopEnv

    return partial(IsaacSimTabletopEnv, max_steps=max_steps)


def _build_policy_factory(action_dim: int) -> Callable[[], RandomPolicy]:
    """Fresh :class:`RandomPolicy` per worker. Runner re-seeds per episode."""
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
    n_workers: int = 1,
    max_steps: int = _SMOKE_MAX_STEPS,
) -> None:
    """Run a RandomPolicy smoke eval on the tabletop-isaac backend.

    Args:
        suite_path: YAML suite to load. Defaults to
            ``tabletop-isaac-smoke.yaml`` (state-only sweep).
        out_dir: Output directory; created if missing. Receives
            ``episodes.json``, ``report.json``, ``report.html``.
        n_workers: Worker processes. Defaults to ``1`` because every
            extra worker pays the Kit boot cost on its first episode.
        max_steps: Hard cap on per-episode env steps.
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
        description="Run RandomPolicy against the tabletop-isaac backend.",
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
        default=1,
        help=(
            "Worker processes (default: 1 — every extra worker pays "
            "the Kit boot cost on its first episode)."
        ),
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
