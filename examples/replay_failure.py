"""Load a prior run's ``episodes.json`` and replay one episode off-grid.

Usage:
    uv run python examples/replay_failure.py \
        --episodes out/episodes.json \
        --suite examples/suites/tabletop-smoke.yaml

Picks the first failing Episode (or the first Episode when no failure
exists), replays it once without overrides to demonstrate bit-identity,
then replays it a second time with ``lighting_intensity`` bumped up to
1.2 to show how a developer can poke at a single axis off the original
grid without re-running the whole suite.

The factory is module-level so it pickles cleanly if you ever call
``Runner`` yourself; replay runs in-process so this is merely
good hygiene.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from gauntlet.policy import ScriptedPolicy
from gauntlet.replay import replay_one
from gauntlet.runner import Episode
from gauntlet.suite import load_suite

_SCRIPTED_POLICY = ScriptedPolicy


def main(*, episodes_path: Path, suite_path: Path) -> None:
    suite = load_suite(suite_path)

    raw = json.loads(episodes_path.read_text(encoding="utf-8"))
    episodes = [Episode.model_validate(item) for item in raw]
    target = next((ep for ep in episodes if not ep.success), episodes[0])
    print(f"Replaying episode {target.cell_index}:{target.episode_index}")

    # Zero-override replay — bit-identical to the original run.
    same = replay_one(target=target, suite=suite, policy_factory=partial(_SCRIPTED_POLICY))
    print(f"  bit-identical: {same.model_dump() == target.model_dump()}")

    # One-axis override — re-simulate with a brighter light.
    if "lighting_intensity" in target.perturbation_config:
        tweaked = replay_one(
            target=target,
            suite=suite,
            policy_factory=partial(_SCRIPTED_POLICY),
            overrides={"lighting_intensity": 1.2},
        )
        print(
            f"  with lighting_intensity=1.2: success={tweaked.success} "
            f"steps={tweaked.step_count} reward={tweaked.total_reward:.4f}"
        )


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Replay one episode with an axis override.")
    parser.add_argument("--episodes", type=Path, required=True, help="Path to episodes.json.")
    parser.add_argument("--suite", type=Path, required=True, help="Path to suite YAML.")
    args = parser.parse_args()
    main(episodes_path=args.episodes, suite_path=args.suite)
