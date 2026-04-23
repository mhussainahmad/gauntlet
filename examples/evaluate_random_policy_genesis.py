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

Notes specific to Genesis (RFC-007, RFC-008):

* Env construction is ~40 s on first instantiation per worker process
  (torch import + ``gs.init`` kernel compile + ``scene.build``). With
  ``n_workers=2`` that is a ~40 s startup, amortised across every
  episode the worker handles thereafter. ``scene.build`` inside one
  process is cached after the first call.
* Image observations are available via ``--render-in-obs``
  (RFC-008). The flag flips the env factory to
  ``partial(GenesisTabletopEnv, render_in_obs=True, render_size=...)``
  so ``obs["image"]`` is a uint8 ``(H, W, 3)`` array; shape / dtype /
  bounds match MuJoCo and PyBullet. Without the flag the default
  state-only obs (five keys, no image) is emitted.
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

from gauntlet.env.genesis import GenesisTabletopEnv
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


def _build_env_factory(
    max_steps: int,
    render_in_obs: bool = False,
    render_size: tuple[int, int] = (224, 224),
) -> Callable[[], GenesisTabletopEnv]:
    """Return a picklable env factory that caps rollout length.

    ``functools.partial`` over the class is the canonical spawn-friendly
    factory — matches ``examples/evaluate_random_policy.py`` and
    ``examples/evaluate_smolvla_pybullet.py``. A closure nested inside
    this function would refuse to pickle under ``n_workers >= 2`` and
    silently work only on the in-process ``n_workers == 1`` path.

    ``render_in_obs=True`` emits ``obs["image"]`` (RFC-008); off by
    default so the state-only path stays byte-identical to the
    pre-RFC-008 contract.

    :class:`GenesisTabletopEnv` is imported at module scope in this
    example; the "parent-process-torch-free" rule applies to
    ``gauntlet.core``, not to example scripts that run post
    ``uv sync --extra genesis``.
    """
    return partial(
        GenesisTabletopEnv,
        max_steps=max_steps,
        render_in_obs=render_in_obs,
        render_size=render_size,
    )


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
    render_in_obs: bool = False,
    render_size: tuple[int, int] = (224, 224),
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
        render_in_obs: When true (RFC-008), emit ``obs["image"]`` as a
            uint8 ``(H, W, 3)`` array. Default off to keep the smoke
            path state-only.
        render_size: ``(height, width)`` of the emitted image. Ignored
            when ``render_in_obs=False``. Default ``(224, 224)``
            matches MuJoCo and the VLA input convention.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suite: Suite = load_suite(suite_path)

    runner = Runner(
        n_workers=n_workers,
        env_factory=_build_env_factory(
            max_steps, render_in_obs=render_in_obs, render_size=render_size
        ),
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
    parser.add_argument(
        "--render-in-obs",
        action="store_true",
        help="Emit obs['image'] (uint8 HxWx3) — RFC-008. Default off.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=[224, 224],
        help="(height, width) of obs['image']; ignored without --render-in-obs.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        suite_path=args.suite,
        out_dir=args.out,
        n_workers=args.n_workers,
        max_steps=args.max_steps,
        render_in_obs=bool(args.render_in_obs),
        render_size=(int(args.render_size[0]), int(args.render_size[1])),
    )
