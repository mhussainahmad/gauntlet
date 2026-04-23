"""PyBullet backend determinism tests — RFC-005 §9.1 cases 3 and 4.

The determinism contract is load-bearing: the Runner's bit-determinism
guarantee requires ``(env_seed, fixed action sequence)`` to reproduce
byte-identical trajectories. These tests pin that at the backend seam.

All tests are marked ``@pytest.mark.pybullet`` and only run under:
    uv run --extra pybullet pytest -m pybullet
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.pybullet


def _reset_obs_snapshot(env: object, seed: int) -> dict[str, np.ndarray]:
    """Reset ``env`` and capture a defensive-copy snapshot of every obs key."""
    obs, _ = env.reset(seed=seed)  # type: ignore[attr-defined]
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in obs.items()}


def _assert_obs_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"key {k!r}: {a[k]!r} != {b[k]!r}"


def test_reset_is_bit_deterministic_for_same_seed() -> None:
    """RFC-005 §9.1 case 3. ``env.reset(seed=42)`` twice must yield the
    same observation dict byte-for-byte — the scene-rebuild +
    RNG-from-seed path must not leak state between resets.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        first = _reset_obs_snapshot(env, seed=42)
        second = _reset_obs_snapshot(env, seed=42)
        _assert_obs_equal(first, second)
    finally:
        env.close()


def test_different_seeds_produce_different_observations() -> None:
    """Negative sanity check for the seed contract — different seeds
    must produce different initial cube/target poses (otherwise the
    RNG was not actually consumed).
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        s1 = _reset_obs_snapshot(env, seed=1)
        s2 = _reset_obs_snapshot(env, seed=2)
    finally:
        env.close()

    # At least one of cube_pos / target_pos must differ across seeds.
    cube_diff = not np.array_equal(s1["cube_pos"], s2["cube_pos"])
    target_diff = not np.array_equal(s1["target_pos"], s2["target_pos"])
    assert cube_diff or target_diff


def test_step_sequence_bit_identical_for_same_seed() -> None:
    """RFC-005 §9.1 case 4. A fixed, seed-derived action sequence
    driven through 20 ``step`` calls must produce byte-identical
    observations on two independent reset+step runs.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    def _rollout() -> list[dict[str, np.ndarray]]:
        env = PyBulletTabletopEnv()
        try:
            env.reset(seed=2024)
            # Action stream is seeded independently from the env seed —
            # the point is to have a reproducible, non-trivial input.
            rng = np.random.default_rng(999)
            snapshots: list[dict[str, np.ndarray]] = []
            for _ in range(20):
                action = rng.uniform(-1.0, 1.0, size=(7,))
                obs, _, _, _, _ = env.step(action)
                snapshots.append(
                    {k: np.asarray(v, dtype=np.float64).copy() for k, v in obs.items()}
                )
            return snapshots
        finally:
            env.close()

    a = _rollout()
    b = _rollout()
    assert len(a) == len(b) == 20
    for i, (sa, sb) in enumerate(zip(a, b, strict=True)):
        _assert_obs_equal(sa, sb)
        # Extra defensive check — no NaNs leaking out of the solver.
        for k, v in sa.items():
            assert np.all(np.isfinite(v)), f"step {i} key {k!r} not finite"


def test_action_and_observation_space_parity_with_mujoco_tabletop() -> None:
    """RFC-005 §9.1 case 5. Shape + dtype + bounds parity with
    :class:`gauntlet.env.tabletop.TabletopEnv` so the same policy can
    run on either backend unmodified.
    """
    import numpy as np
    from gymnasium import spaces

    from gauntlet.env.pybullet import PyBulletTabletopEnv
    from gauntlet.env.tabletop import TabletopEnv

    pb = PyBulletTabletopEnv()
    mj = TabletopEnv()
    try:
        # Action space: Box((7,), float64, [-1, 1]).
        assert isinstance(pb.action_space, spaces.Box)
        assert isinstance(mj.action_space, spaces.Box)
        assert pb.action_space.shape == mj.action_space.shape
        assert pb.action_space.dtype == mj.action_space.dtype
        assert np.array_equal(pb.action_space.low, mj.action_space.low)
        assert np.array_equal(pb.action_space.high, mj.action_space.high)

        # Observation space: Dict with identical keys.
        assert isinstance(pb.observation_space, spaces.Dict)
        assert isinstance(mj.observation_space, spaces.Dict)
        assert set(pb.observation_space.spaces) == set(mj.observation_space.spaces)
        for key in mj.observation_space.spaces:
            pb_sub = pb.observation_space.spaces[key]
            mj_sub = mj.observation_space.spaces[key]
            assert pb_sub.shape == mj_sub.shape, key
            assert pb_sub.dtype == mj_sub.dtype, key
    finally:
        pb.close()
        mj.close()


def test_cube_quat_returns_wxyz_not_pybullet_xyzw() -> None:
    """RFC-005 §7.3. ``obs["cube_quat"]`` MUST use MuJoCo wxyz order —
    the PyBullet-native xyzw order must be converted inside the
    backend. On a fresh reset the cube sits with identity orientation;
    the obs must have ``1.0`` in slot 0 (w) and ``0.0`` in slot 3 (the
    PyBullet-native w position), not the other way around.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        obs, _ = env.reset(seed=7)
        q = np.asarray(obs["cube_quat"], dtype=np.float64)
        assert q.shape == (4,)
        # Identity quat in wxyz = (1, 0, 0, 0).
        assert q[0] == pytest.approx(1.0, abs=1e-9)
        assert q[1] == pytest.approx(0.0, abs=1e-9)
        assert q[2] == pytest.approx(0.0, abs=1e-9)
        assert q[3] == pytest.approx(0.0, abs=1e-9)
    finally:
        env.close()
