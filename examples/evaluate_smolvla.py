"""Evaluate SmolVLA against the tabletop smoke suite (illustrative).

**Honesty caveat — read before judging zero-shot numbers.**
``lerobot/smolvla_base`` is pretrained on SO-100 / SO-101 follower arms
with **6-D joint-position** actions, whereas Gauntlet's ``TabletopEnv``
has a **7-D EE-twist + gripper** action space. The embodiments do NOT
match. The default :class:`~gauntlet.policy.lerobot.LeRobotPolicy`
``action_remap`` pads 6→7 with a zero gripper and warns once per
instance, but that is a *honest bridge* — NOT a correctness fix. Expect
zero-shot success on the smoke suite to be ~0%. Meaningful evaluation
requires a SmolVLA fine-tune on TabletopEnv-compatible data; users who
have one should pass an explicit ``action_remap`` and (if their
fine-tune changed the camera layout) override ``camera_keys``.

This file is **not** executed by the test suite — it downloads real
weights (~3 GB via lerobot) and wants a GPU with bfloat16 support. It
lives here as the reference wiring for the ``LeRobotPolicy`` adapter,
showing the ≤20-line policy factory that satisfies the Runner's pickle
contract (see docs/phase2-rfc-002-lerobot-smolvla.md §4).

Install the extras first, then run::

    uv sync --extra lerobot
    uv run python examples/evaluate_smolvla.py --out out-smolvla/

See ``https://huggingface.co/docs/lerobot/smolvla`` for fine-tuning.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

from gauntlet.env import TabletopEnv
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main", "make_smolvla_policy"]


_INSTRUCTION: str = "pick up the red cube and place it on the target"
_REPO_ID: str = "lerobot/smolvla_base"

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out-smolvla"


def make_smolvla_policy() -> object:
    """Module-level factory — picklable, spawn-safe (spec §6)."""
    from gauntlet.policy import LeRobotPolicy

    return LeRobotPolicy(repo_id=_REPO_ID, instruction=_INSTRUCTION, dtype="bfloat16")


def _build_env_factory() -> Callable[[], TabletopEnv]:
    # SmolVLA's default resize is 512x512; preserve pixels where we can.
    return partial(TabletopEnv, render_in_obs=True, render_size=(512, 512))


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
        description="Run SmolVLA-base against the tabletop smoke suite.",
    )
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--n-workers", type=int, default=1)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(suite_path=args.suite, out_dir=args.out, n_workers=args.n_workers)
