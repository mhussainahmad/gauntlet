"""Runner ``--record-trajectories`` / ``trajectory_dir`` contract.

Torch-free. These tests run in the default CI job and lock in the load-
bearing promise RFC §6 makes: when the Runner is built with
``trajectory_dir=None`` its output is byte-identical to Phase 1, and
when a :class:`~pathlib.Path` is supplied exactly ``num_episodes`` NPZ
sidecars land in that directory with the expected keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gauntlet.policy.random import RandomPolicy
from gauntlet.runner import Episode, Runner
from gauntlet.runner.worker import trajectory_path_for
from gauntlet.suite.schema import AxisSpec, Suite

# ----------------------------------------------------------------------------
# Module-level factories (pickle-safe under spawn, mirrors test_runner.py).
# ----------------------------------------------------------------------------


_ACTION_DIM: int = 7


def _make_random_policy() -> RandomPolicy:
    """Fresh :class:`RandomPolicy` per episode; Runner re-seeds via ``reset``."""
    return RandomPolicy(action_dim=_ACTION_DIM, seed=None)


def _make_fast_env() -> Any:
    """Short-rollout tabletop env so the suite wraps up in a second or two."""
    from gauntlet.env.tabletop import TabletopEnv

    return TabletopEnv(max_steps=10)


def _make_tiny_suite() -> Suite:
    """1 axis x 2 steps x 2 eps = 4 episodes — smallest that has
    independent seeds we can verify uniqueness against."""
    return Suite(
        name="trajectory-smoke",
        env="tabletop",
        seed=4242,
        episodes_per_cell=2,
        axes={"lighting_intensity": AxisSpec(low=0.5, high=1.0, steps=2)},
    )


# ----------------------------------------------------------------------------
# Default path: trajectory_dir=None => zero disk writes, byte-identical Episodes.
# ----------------------------------------------------------------------------


def test_trajectory_dir_none_is_default_and_writes_nothing(tmp_path: Path) -> None:
    """Without the kwarg the Runner must not create any files.

    The ``tmp_path`` fixture is passed in just so we can assert later
    that nothing near it changed; the Runner is called with no
    ``trajectory_dir`` at all.
    """
    suite = _make_tiny_suite()
    runner = Runner(n_workers=1, env_factory=_make_fast_env)
    episodes = runner.run(policy_factory=_make_random_policy, suite=suite)

    # Correct episode count + the tmp dir we didn't pass is untouched.
    assert len(episodes) == suite.num_cells() * suite.episodes_per_cell
    assert list(tmp_path.iterdir()) == []


def test_trajectory_dir_set_produces_byte_identical_episodes(tmp_path: Path) -> None:
    """Setting trajectory_dir must not perturb the Episode list.

    This is the load-bearing test. If it regresses, the determinism
    contract of the Runner is broken for anyone using
    ``--record-trajectories`` and the Phase 1 reproducibility story
    silently ends.
    """
    suite = _make_tiny_suite()

    without = Runner(n_workers=1, env_factory=_make_fast_env).run(
        policy_factory=_make_random_policy,
        suite=suite,
    )
    traj_dir = tmp_path / "trajectories"
    with_traj = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        trajectory_dir=traj_dir,
    ).run(policy_factory=_make_random_policy, suite=suite)

    # Same count + bit-for-bit equality on every field.
    assert len(without) == len(with_traj)
    for a, b in zip(without, with_traj, strict=True):
        assert a.model_dump(mode="json") == b.model_dump(mode="json")


# ----------------------------------------------------------------------------
# NPZ sidecar existence + filename scheme + content.
# ----------------------------------------------------------------------------


def test_trajectory_dir_writes_one_npz_per_episode(tmp_path: Path) -> None:
    """Exactly ``num_episodes`` NPZs appear with the expected filename."""
    suite = _make_tiny_suite()
    traj_dir = tmp_path / "trajectories"
    runner = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        trajectory_dir=traj_dir,
    )
    episodes = runner.run(policy_factory=_make_random_policy, suite=suite)

    npz_files = sorted(traj_dir.glob("*.npz"))
    assert len(npz_files) == len(episodes)

    # Every Episode maps to exactly one NPZ with the canonical name.
    for ep in episodes:
        expected = trajectory_path_for(traj_dir, ep.cell_index, ep.episode_index)
        assert expected.exists(), f"missing NPZ for cell={ep.cell_index} ep={ep.episode_index}"


def test_trajectory_npz_contains_expected_keys_and_shapes(tmp_path: Path) -> None:
    """NPZ payload matches the RFC §6 spec: obs_<key>, action, scalar identity."""
    # One cell x one ep keeps this fast; we only need one NPZ.
    suite = Suite(
        name="trajectory-one",
        env="tabletop",
        seed=17,
        episodes_per_cell=1,
        axes={"lighting_intensity": AxisSpec(low=0.7, high=0.7, steps=1)},
    )
    traj_dir = tmp_path / "trajectories"
    runner = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        trajectory_dir=traj_dir,
    )
    episodes = runner.run(policy_factory=_make_random_policy, suite=suite)
    assert len(episodes) == 1
    ep: Episode = episodes[0]
    path = trajectory_path_for(traj_dir, ep.cell_index, ep.episode_index)

    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
        # Obs keys (TabletopEnv._build_obs).
        for k in ("cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"):
            assert f"obs_{k}" in keys
        assert "action" in keys
        assert "seed" in keys
        assert "cell_index" in keys
        assert "episode_index" in keys

        # Shapes: obs is (T, ...), action is (T, 7), identity is scalar int64.
        T = ep.step_count
        assert npz["action"].shape == (T, 7)
        assert npz["obs_cube_pos"].shape == (T, 3)
        assert npz["obs_cube_quat"].shape == (T, 4)
        assert npz["obs_ee_pos"].shape == (T, 3)
        assert npz["obs_gripper"].shape == (T, 1)
        assert npz["obs_target_pos"].shape == (T, 3)
        assert int(npz["seed"]) == ep.seed
        assert int(npz["cell_index"]) == ep.cell_index
        assert int(npz["episode_index"]) == ep.episode_index
        assert npz["seed"].dtype == np.int64
        assert npz["action"].dtype == np.float64


def test_trajectory_rerun_overwrites_in_place(tmp_path: Path) -> None:
    """Running twice with the same inputs must not double the file count.

    Deterministic filename derivation from ``(cell_index, episode_index)``
    is the RFC-required property; numpy's ``savez_compressed`` writes
    atomically-by-POSIX, so the second run simply overwrites.
    """
    suite = _make_tiny_suite()
    traj_dir = tmp_path / "trajectories"
    runner = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        trajectory_dir=traj_dir,
    )

    runner.run(policy_factory=_make_random_policy, suite=suite)
    first_set = sorted(traj_dir.glob("*.npz"))

    runner.run(policy_factory=_make_random_policy, suite=suite)
    second_set = sorted(traj_dir.glob("*.npz"))

    assert first_set == second_set
    assert len(second_set) == suite.num_cells() * suite.episodes_per_cell


# ----------------------------------------------------------------------------
# Filename helper.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell_index", "episode_index", "expected"),
    [
        (0, 0, "cell_0000_ep_0000.npz"),
        (3, 7, "cell_0003_ep_0007.npz"),
        (1234, 5678, "cell_1234_ep_5678.npz"),
    ],
)
def test_trajectory_path_for_uses_rfc_scheme(
    tmp_path: Path, cell_index: int, episode_index: int, expected: str
) -> None:
    """Filename pattern matches RFC §6: ``cell_NNNN_ep_NNNN.npz``."""
    assert trajectory_path_for(tmp_path, cell_index, episode_index) == tmp_path / expected
