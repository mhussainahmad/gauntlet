"""Run :class:`RandomPolicy` against the tabletop-genesis smoke suite.

Usage:
    uv sync --extra genesis --group genesis-dev
    uv run python examples/evaluate_random_policy_genesis.py [--out OUT_DIR]
                                                              [--suite SUITE_YAML]
                                                              [--n-workers N]

The Genesis counterpart of :mod:`examples.evaluate_random_policy`
(MuJoCo) and :mod:`examples.evaluate_smolvla_pybullet` (PyBullet VLA).
Mirrors the former's public-Python-API shape, swapping only the env
factory — same Suite schema, same Runner, same Report artefacts.

Notes specific to Genesis (RFC-007):

* Env construction is ~40 s on first instantiation per worker process
  (torch import + ``gs.init`` kernel compile + ``scene.build``). With
  ``n_workers=2`` that is a ~40 s startup, amortised across every
  episode the worker handles thereafter. ``scene.build`` inside one
  process is cached after the first call.
* The first cut is state-only (RFC-007 §2): no image obs, so
  ``RandomPolicy`` and any state-conditioned policy work; VLA
  adapters (``HuggingFacePolicy``, ``LeRobotPolicy``) need the
  follow-up rendering RFC (RFC-008).
* Same-process determinism (seed + action sequence -> same obs) is
  exact-to-fp-noise (~1e-11). Cross-backend numerical parity is an
  explicit non-goal — a given seed on ``tabletop`` vs
  ``tabletop-pybullet`` vs ``tabletop-genesis`` produces three
  different trajectories (RFC-007 §7.3). The report schema is
  backend-agnostic; comparing across backends with
  ``gauntlet compare --allow-cross-backend`` measures simulator
  drift, not policy regression.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from gauntlet.policy import RandomPolicy
from gauntlet.report import Report, build_report, write_html
from gauntlet.report.html import _nan_to_none
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main"]


_TABLETOP_ACTION_DIM: int = 7

# Keep each episode short so the smoke run finishes in a minute on
# CPU. Genesis's ``scene.step`` is cheap per tick; startup dominates.
_SMOKE_MAX_STEPS: int = 20

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-genesis-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out-genesis"


def _build_env_factory(max_steps: int) -> Callable[[], Any]:
    """Picklable env factory for the Genesis backend.

    Deferred ``from gauntlet.env.genesis import GenesisTabletopEnv``
    inside the factory keeps the parent process torch/genesis-free
    until the worker actually spawns.
    """

    def _factory() -> Any:
        from gauntlet.env.genesis import GenesisTabletopEnv

        return GenesisTabletopEnv(max_steps=max_steps)

    return _factory


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
    """Run a RandomPolicy smoke eval on the tabletop-genesis backend.

    Args:
        suite_path: YAML suite to load. Defaults to the shared smoke suite
            (``tabletop-smoke.yaml``) — the Suite schema is backend-neutral.
            Point at a suite that names ``env: tabletop-genesis`` to use
            the Genesis factory via the registry dispatch, or keep this
            factory-override path if reusing the generic smoke suite
            that declares ``env: tabletop``.
        out_dir: Output directory; created if missing. Receives
            ``episodes.json``, ``report.json``, ``report.html``.
        n_workers: Worker processes. Defaults to ``1`` because every
            extra worker pays the ~40 s Genesis init cost on its first
            episode — the sweet spot for a smoke is one worker unless
            the suite is large.
        max_steps: Hard cap on per-episode env steps. Default 20 keeps
            each episode sub-second after the first scene build.
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
        description="Run RandomPolicy against the tabletop-genesis backend.",
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
            "the ~40 s Genesis init cost on its first episode)."
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
