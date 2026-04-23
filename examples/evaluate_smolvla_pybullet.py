"""Evaluate SmolVLA against the tabletop smoke suite — PyBullet backend.

Mirror of :mod:`examples.evaluate_smolvla` but dispatches to the PyBullet
``tabletop-pybullet`` backend via an ``env_factory`` that flips on
``render_in_obs=True`` (RFC-006). The policy is identical; only the
simulator changes.

**Honesty caveat — read before judging zero-shot numbers.** Same 6-D ←→
7-D embodiment mismatch as the MuJoCo example — ``lerobot/smolvla_base``
is pretrained on SO-100 joint positions whereas Gauntlet's tabletop env
has a 7-D EE-twist + gripper action space. See
``examples/evaluate_smolvla.py`` for the full discussion; zero-shot
success ≈ 0% by construction.

**Cross-backend caveat.** Same policy on MuJoCo vs PyBullet produces
numerically different trajectories — different solvers, different
contact models (RFC-005 §7.4). ``gauntlet compare`` across backends
measures simulator drift, not policy regression.

This file is **not** executed by the test suite (it downloads real
weights). It documents the reference wiring: a ≤20-line env-factory
swap from MuJoCo to PyBullet with rendering on.

Install the extras first, then run::

    uv sync --extra lerobot --extra pybullet
    uv run python examples/evaluate_smolvla_pybullet.py --out out-smolvla-pybullet/
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main", "make_smolvla_policy"]


_INSTRUCTION: str = "pick up the red cube and place it on the target"
_REPO_ID: str = "lerobot/smolvla_base"

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-pybullet-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out-smolvla-pybullet"


def make_smolvla_policy() -> object:
    """Module-level factory — picklable, spawn-safe (spec §6)."""
    from gauntlet.policy import LeRobotPolicy

    return LeRobotPolicy(repo_id=_REPO_ID, instruction=_INSTRUCTION, dtype="bfloat16")


def _build_env_factory() -> Callable[[], object]:
    """PyBullet env factory with rendering on. Size matches the MuJoCo example."""
    # Import lazily so loading the module without the [pybullet] extra
    # installed does not blow up; the factory call is what triggers import.
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    return partial(PyBulletTabletopEnv, render_in_obs=True, render_size=(512, 512))


def main(
    *,
    suite_path: Path = _DEFAULT_SUITE,
    out_dir: Path = _DEFAULT_OUT,
    n_workers: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suite: Suite = load_suite(suite_path)
    runner = Runner(n_workers=n_workers, env_factory=_build_env_factory())
    episodes: list[Episode] = runner.run(
        policy_factory=make_smolvla_policy,
        suite=suite,
    )
    print(f"Completed {len(episodes)} episodes -> {out_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SmolVLA-base against the tabletop-pybullet smoke suite.",
    )
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--n-workers", type=int, default=1)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(suite_path=args.suite, out_dir=args.out, n_workers=args.n_workers)
