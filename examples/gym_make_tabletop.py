"""Construct the MuJoCo Tabletop env via the standard ``gymnasium.make`` API.

Usage:
    uv run python examples/gym_make_tabletop.py [--steps N] [--seed S]

The whole point of this script is to demonstrate the new affordance —
``gym.make("gauntlet/Tabletop-v0")`` — that a downstream RL/IL pipeline
(stable-baselines3, RLlib, CleanRL, TorchRL, ...) expects from any
environment library. Compare to ``examples/evaluate_random_policy_lhs.py``,
which constructs ``TabletopEnv`` by direct class import — that path
keeps working, but library users typically reach for ``gym.make`` first.

What the script does:

1. ``import gauntlet`` — registers all four shipped backend ids with
   gymnasium's global registry on import. The three heavy backends
   (``TabletopPyBullet-v0``, ``TabletopGenesis-v0``, ``TabletopIsaac-v0``)
   are registered with string entry_points so this import does NOT pull
   in pybullet / genesis / isaacsim — a user without those extras can
   still use the MuJoCo backend.
2. ``gym.make("gauntlet/Tabletop-v0")`` — wraps :class:`TabletopEnv` in
   the standard :class:`gymnasium.wrappers.TimeLimit` (capped at
   ``max_episode_steps=200`` to match the adapter's own default).
3. Roll out a fixed-seed zero-action episode and print a one-line summary
   so the script has an observable side effect when run end-to-end.

This file is the demonstration, not a benchmark. It deliberately uses
zero actions so the rollout finishes in milliseconds and produces no
artefacts on disk.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import gauntlet  # noqa: F401  # import for side effect (registers gym ids)

__all__ = ["main"]

_DEFAULT_STEPS: int = 20
_DEFAULT_SEED: int = 0
_TABLETOP_ID: str = "gauntlet/Tabletop-v0"


def main(*, steps: int = _DEFAULT_STEPS, seed: int = _DEFAULT_SEED) -> None:
    """Construct the env via gym.make and roll out a zero-action episode.

    Args:
        steps: Number of zero-action steps to take. Capped from above by
            the env's own ``max_episode_steps=200`` (the
            :class:`gymnasium.wrappers.TimeLimit` wrapper installed by
            ``gym.make``).
        seed: Reset seed. Same value across runs yields bit-identical
            observations — the env's only entropy source is the seed.
    """
    env = gym.make(_TABLETOP_ID)
    try:
        _obs, _info = env.reset(seed=seed)
        shape = env.action_space.shape
        assert shape is not None  # narrows for static type checkers
        zero_action = np.zeros(shape, dtype=np.float64)

        terminated = truncated = False
        actual_steps = 0
        for _ in range(steps):
            _obs, _reward, terminated, truncated, _info = env.step(zero_action)
            actual_steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()

    print(
        f"gym.make({_TABLETOP_ID!r}) -> rolled out {actual_steps} steps "
        f"(terminated={terminated}, truncated={truncated})."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate gymnasium.make() for the MuJoCo tabletop env. "
            "Shows the standard ecosystem affordance that wraps the same "
            "underlying TabletopEnv used by the rest of gauntlet."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=_DEFAULT_STEPS,
        help=f"Number of zero-action env steps (default: {_DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Reset seed for reproducibility (default: {_DEFAULT_SEED}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(steps=args.steps, seed=args.seed)
