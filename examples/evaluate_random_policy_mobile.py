"""Smoke-evaluate a random policy against the mobile-base env (B-13).

Usage:
    uv run python examples/evaluate_random_policy_mobile.py [--steps N]

Drives a uniform-random 10-D action through :class:`MobileTabletopEnv`
for ``--steps`` control steps and prints the final base SE(2) pose.
The point is to demonstrate the new ``pose`` observation key + the
extended action space — not to land a successful nav-pick rollout.
"""

from __future__ import annotations

import argparse

import numpy as np

from gauntlet.env import MobileTabletopEnv


def main(*, steps: int = 50, seed: int = 0) -> None:
    env = MobileTabletopEnv(max_steps=steps)
    rng = np.random.default_rng(seed)
    obs, _info = env.reset(seed=seed)
    successes = 0
    for _ in range(steps):
        action = rng.uniform(-1.0, 1.0, size=10).astype(np.float64)
        obs, _reward, terminated, truncated, info = env.step(action)
        if info.get("success"):
            successes += 1
        if terminated or truncated:
            break
    print(
        f"final pose=(x={obs['pose'][0]:+.3f}, y={obs['pose'][1]:+.3f}, "
        f"theta={obs['pose'][2]:+.3f}) successes={successes}",
    )
    env.close()


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(steps=args.steps, seed=args.seed)
