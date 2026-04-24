"""Direct unit tests for ``gauntlet.runner.worker``.

Phase 2.5 Task 11. The end-to-end Runner tests in ``tests/test_runner.py``
already cover the ``execute_one`` happy paths (single-worker and
multi-worker), but they do NOT exercise:

* ``pool_initializer`` called directly with a ``WorkerInitArgs`` —
  the in-process branch sets ``_WORKER_STATE`` from the args.
* ``run_work_item`` called from the main process after a manual
  ``pool_initializer`` — covers the env / policy_factory cache reads
  and the ``trajectory_dir`` default-None path.
* ``run_work_item`` raising ``RuntimeError`` when ``_WORKER_STATE`` was
  never populated (defence-in-depth for a Runner bug).
* ``trajectory_path_for`` filename derivation.
* ``write_trajectory_npz`` schema (per-key obs arrays + scalar identity
  arrays + parent dir auto-mkdir).

All tests run in the default torch-free job and are pure unit checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.env.tabletop import TabletopEnv
from gauntlet.policy.scripted import ScriptedPolicy
from gauntlet.runner.episode import Episode
from gauntlet.runner.worker import (
    _WORKER_STATE,
    WorkerInitArgs,
    WorkItem,
    extract_env_seed,
    pool_initializer,
    run_work_item,
    trajectory_path_for,
    write_trajectory_npz,
)


def _make_fast_env() -> Any:
    return TabletopEnv(max_steps=8)


def _make_scripted() -> ScriptedPolicy:
    return ScriptedPolicy()


def _make_work_item(
    *,
    cell_index: int = 0,
    episode_index: int = 0,
    seed: int = 7,
) -> WorkItem:
    master = np.random.SeedSequence(seed)
    cell_node = master.spawn(1)[0]
    episode_node = cell_node.spawn(1)[0]
    return WorkItem(
        suite_name="worker-direct",
        cell_index=cell_index,
        episode_index=episode_index,
        perturbation_values={},
        episode_seq=episode_node,
        master_seed=seed,
        n_cells=1,
        episodes_per_cell=1,
    )


# ──────────────────────────────────────────────────────────────────────
# pool_initializer + run_work_item — direct in-process call.
# ──────────────────────────────────────────────────────────────────────


def test_pool_initializer_populates_worker_state() -> None:
    """``pool_initializer`` stashes env / policy_factory / trajectory_dir
    in the module-global ``_WORKER_STATE``."""
    args = WorkerInitArgs(
        env_factory=_make_fast_env,
        policy_factory=_make_scripted,
        trajectory_dir=None,
    )
    pool_initializer(args)
    try:
        assert "env" in _WORKER_STATE
        assert _WORKER_STATE["policy_factory"] is _make_scripted
        assert _WORKER_STATE["trajectory_dir"] is None
    finally:
        env = _WORKER_STATE.pop("env", None)
        if env is not None and hasattr(env, "close"):
            env.close()
        _WORKER_STATE.pop("policy_factory", None)
        _WORKER_STATE.pop("trajectory_dir", None)


def test_run_work_item_returns_episode_after_initializer() -> None:
    """After a manual ``pool_initializer``, ``run_work_item`` produces a
    valid Episode for a constructed WorkItem."""
    args = WorkerInitArgs(
        env_factory=_make_fast_env,
        policy_factory=_make_scripted,
        trajectory_dir=None,
    )
    pool_initializer(args)
    try:
        item = _make_work_item(cell_index=2, episode_index=1, seed=21)
        episode = run_work_item(item)
        assert isinstance(episode, Episode)
        assert episode.suite_name == "worker-direct"
        assert episode.cell_index == 2
        assert episode.episode_index == 1
        # Seed echoes the SeedSequence-derived uint32.
        assert episode.seed == extract_env_seed(item.episode_seq)
    finally:
        env = _WORKER_STATE.pop("env", None)
        if env is not None and hasattr(env, "close"):
            env.close()
        _WORKER_STATE.pop("policy_factory", None)
        _WORKER_STATE.pop("trajectory_dir", None)


def test_run_work_item_raises_runtime_error_when_state_missing() -> None:
    """If ``_WORKER_STATE`` is empty (initializer never ran), the entry
    point raises a clear RuntimeError naming the Runner bug surface."""
    # Defensively wipe the state in case a sibling test left residue.
    _WORKER_STATE.pop("env", None)
    _WORKER_STATE.pop("policy_factory", None)
    _WORKER_STATE.pop("trajectory_dir", None)

    item = _make_work_item()
    with pytest.raises(RuntimeError, match="pool_initializer was not called"):
        run_work_item(item)


def test_run_work_item_with_trajectory_dir_writes_npz(tmp_path: Path) -> None:
    """When ``trajectory_dir`` is set in the init args, the worker emits
    one NPZ per episode keyed by ``cell_NNNN_ep_NNNN.npz``."""
    traj_dir = tmp_path / "trajs"
    args = WorkerInitArgs(
        env_factory=_make_fast_env,
        policy_factory=_make_scripted,
        trajectory_dir=traj_dir,
    )
    pool_initializer(args)
    try:
        item = _make_work_item(cell_index=3, episode_index=2, seed=33)
        run_work_item(item)
        expected = traj_dir / "cell_0003_ep_0002.npz"
        assert expected.is_file()
    finally:
        env = _WORKER_STATE.pop("env", None)
        if env is not None and hasattr(env, "close"):
            env.close()
        _WORKER_STATE.pop("policy_factory", None)
        _WORKER_STATE.pop("trajectory_dir", None)


# ──────────────────────────────────────────────────────────────────────
# trajectory_path_for — pure function.
# ──────────────────────────────────────────────────────────────────────


def test_trajectory_path_for_uses_zero_padded_indices(tmp_path: Path) -> None:
    p = trajectory_path_for(tmp_path, cell_index=5, episode_index=12)
    assert p.name == "cell_0005_ep_0012.npz"
    assert p.parent == tmp_path


def test_trajectory_path_for_handles_zero_indices(tmp_path: Path) -> None:
    p = trajectory_path_for(tmp_path, cell_index=0, episode_index=0)
    assert p.name == "cell_0000_ep_0000.npz"


# ──────────────────────────────────────────────────────────────────────
# write_trajectory_npz — direct schema check, including parent mkdir.
# ──────────────────────────────────────────────────────────────────────


def test_write_trajectory_npz_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trajs" / "cell_0000_ep_0000.npz"
    obs_arrays: dict[str, NDArray[np.float64]] = {
        "cube_pos": np.zeros((4, 3), dtype=np.float64),
        "ee_pos": np.ones((4, 3), dtype=np.float64),
    }
    actions = np.linspace(-1.0, 1.0, 4 * 7, dtype=np.float64).reshape(4, 7)

    write_trajectory_npz(
        path,
        obs_arrays=obs_arrays,
        actions=actions,
        seed=42,
        cell_index=7,
        episode_index=11,
    )
    # Auto-mkdir the parent.
    assert path.is_file()

    with np.load(path) as npz:
        np.testing.assert_array_equal(npz["obs_cube_pos"], obs_arrays["cube_pos"])
        np.testing.assert_array_equal(npz["obs_ee_pos"], obs_arrays["ee_pos"])
        np.testing.assert_array_equal(npz["action"], actions)
        assert int(npz["seed"]) == 42
        assert int(npz["cell_index"]) == 7
        assert int(npz["episode_index"]) == 11


def test_write_trajectory_npz_zero_step_episode(tmp_path: Path) -> None:
    """An empty-but-well-formed NPZ — zero-row obs arrays + zero-row
    action array — must round-trip without raising."""
    path = tmp_path / "empty.npz"
    obs_arrays: dict[str, NDArray[np.float64]] = {
        "cube_pos": np.zeros((0, 3), dtype=np.float64),
    }
    actions = np.zeros((0, 7), dtype=np.float64)
    write_trajectory_npz(
        path,
        obs_arrays=obs_arrays,
        actions=actions,
        seed=0,
        cell_index=0,
        episode_index=0,
    )
    with np.load(path) as npz:
        assert npz["action"].shape == (0, 7)
        assert npz["obs_cube_pos"].shape == (0, 3)


# ──────────────────────────────────────────────────────────────────────
# extract_env_seed — uint32 stability + per-spawn uniqueness.
# ──────────────────────────────────────────────────────────────────────


def test_extract_env_seed_returns_uint32_range_value() -> None:
    seq = np.random.SeedSequence(123)
    seed = extract_env_seed(seq)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


def test_extract_env_seed_distinct_for_distinct_spawns() -> None:
    """Two siblings spawned from the same parent get distinct env seeds —
    the canonical bug ``SeedSequence.entropy`` would have hit."""
    parent = np.random.SeedSequence(2026)
    children = parent.spawn(2)
    s0 = extract_env_seed(children[0])
    s1 = extract_env_seed(children[1])
    assert s0 != s1
