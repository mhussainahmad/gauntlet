"""Evaluate π0 (Physical Intelligence) against the tabletop smoke suite.

**Honesty caveat — read before judging zero-shot numbers.**
``lerobot/pi0`` is pretrained on a DROID-style mixture; its action
embodiment does NOT match Gauntlet's ``TabletopEnv`` (7-D EE-twist +
gripper). Expect zero-shot success on the smoke suite to be very low.
Beyond that, π0 weights are gated by Physical Intelligence's licensing —
``PI0Policy.from_pretrained("lerobot/pi0")`` may fail with an HTTP 401
on a fresh HF login until you accept the model card. The
:class:`~gauntlet.policy.pi0.Pi0Policy` adapter ships anyway so the
evaluator becomes runnable the moment licensing clears.

This file is **not** executed by the test suite. Install the extras and
run::

    uv sync --extra pi0
    uv run python examples/evaluate_pi0.py --out out-pi0/
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

from gauntlet.env import TabletopEnv
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main", "make_pi0_policy"]


_MODEL_ID: str = "lerobot/pi0"
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out-pi0"


def make_pi0_policy() -> object:
    """Module-level factory — picklable, spawn-safe (spec §6)."""
    from gauntlet.policy.pi0 import Pi0Policy

    return Pi0Policy(model_id=_MODEL_ID, device="cpu", action_horizon=8)


def _build_env_factory() -> Callable[[], TabletopEnv]:
    return partial(TabletopEnv, render_in_obs=True, render_size=(224, 224))


def main(
    *,
    suite_path: Path = _DEFAULT_SUITE,
    out_dir: Path = _DEFAULT_OUT,
    n_workers: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suite: Suite = load_suite(suite_path)
    runner = Runner(n_workers=n_workers, env_factory=_build_env_factory())
    episodes: list[Episode] = runner.run(policy_factory=make_pi0_policy, suite=suite)
    print(f"Completed {len(episodes)} episodes -> {out_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run π0 against the tabletop smoke suite.")
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--n-workers", type=int, default=1)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(suite_path=args.suite, out_dir=args.out, n_workers=args.n_workers)
