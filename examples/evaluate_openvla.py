"""Evaluate OpenVLA-7B against the tabletop smoke suite (illustrative).

This file is **not** executed by the test suite — it downloads real
weights (~15 GB) and wants a GPU with bfloat16 support. It lives here
as the reference wiring for the ``HuggingFacePolicy`` adapter, showing
the ≤20-line policy factory that satisfies the Runner's pickle contract
(see ``docs/phase2-rfc-001-huggingface-policy.md`` §4).

Install the extras first, then run:

    uv sync --extra hf
    uv run python examples/evaluate_openvla.py --out out-openvla/

You can feed the same factory to the CLI:

    gauntlet run examples/suites/tabletop-smoke.yaml \\
        --policy examples.evaluate_openvla:make_openvla_policy \\
        --out out-openvla/
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

from gauntlet.env import TabletopEnv
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main", "make_openvla_policy"]


# Tabletop action layout is [dx, dy, dz, drx, dry, drz, gripper] — the same
# 7-DoF shape OpenVLA emits, so no action-space adaptation is needed here.
_INSTRUCTION: str = "pick up the red cube and place it on the target"
_REPO_ID: str = "openvla/openvla-7b"
_UNNORM_KEY: str = "bridge_orig"

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out-openvla"


def make_openvla_policy() -> object:
    """Module-level factory — picklable, spawn-safe (spec §6).

    We import ``HuggingFacePolicy`` inside the function body so the
    factory itself pickles torch-free; the heavy imports (torch,
    transformers, PIL) only happen inside worker processes.
    """
    from gauntlet.policy import HuggingFacePolicy

    return HuggingFacePolicy(
        repo_id=_REPO_ID,
        instruction=_INSTRUCTION,
        unnorm_key=_UNNORM_KEY,
        dtype="bfloat16",
        image_obs_key="image",
    )


def _build_env_factory() -> Callable[[], TabletopEnv]:
    """OpenVLA needs a rendered frame — flip on ``render_in_obs``."""
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
    episodes: list[Episode] = runner.run(
        policy_factory=make_openvla_policy,
        suite=suite,
    )
    print(f"Completed {len(episodes)} episodes -> {out_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenVLA-7B against the tabletop smoke suite.",
    )
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--n-workers", type=int, default=1)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(suite_path=args.suite, out_dir=args.out, n_workers=args.n_workers)
