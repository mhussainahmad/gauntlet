"""Runner integration tests for the rollout-video recording path.

Marked ``@pytest.mark.video`` — runs in the dedicated ``video-tests`` CI
job. The default torch-/extras-free job deselects the marker.

These tests use a hand-rolled fake env that:

* Satisfies :class:`gauntlet.env.base.GauntletEnv` structurally.
* Exposes ``obs["image"]`` as a 32x32 uint8 RGB array — the public
  contract the Runner checks (NEVER inspects ``env._render_in_obs``).
* Lets the test toggle off ``obs["image"]`` to verify the
  ``record_video=True`` + missing-image error path.

The fake env keeps tests fast (no MuJoCo init, no offscreen renderer)
while exercising every code path through the new
:class:`VideoConfig`, :class:`VideoWriter`, and the per-step frame
buffer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gauntlet.env.base import GauntletEnv
from gauntlet.runner import Runner
from gauntlet.runner.video import video_path_for
from gauntlet.suite.schema import AxisSpec, Suite

pytestmark = pytest.mark.video


# ────────────────────────────────────────────────────────────────────────
# Fake env + factories — module-level so they pickle under spawn.
# ────────────────────────────────────────────────────────────────────────

_ACTION_DIM = 7
_IMG_H = 32
_IMG_W = 32

# Shared, mutable, module-level toggles. Tests reach in to flip them
# before calling Runner(); the in-process fast path runs in the same
# interpreter so the toggles propagate. ALL behaviour the multi-worker
# path relies on is module-level so spawn can pickle.
_FAKE_EXPOSES_IMAGE = [True]
_FAKE_FAILURE_PATTERN: list[Any] = [None]  # ``None`` -> always succeed


class _FakeImageEnv:
    """Protocol-conformant env that produces deterministic frames.

    The frames are a tiny gradient that drifts across the image as the
    rollout progresses — enough signal for the encoder to produce a
    non-trivial MP4.
    """

    AXIS_NAMES = frozenset({"distractor_count"})
    VISUAL_ONLY_AXES: frozenset[str] = frozenset()

    def __init__(self) -> None:
        from gymnasium import spaces

        obs_spaces: dict[str, spaces.Space[Any]] = {
            "cube_pos": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64),
        }
        if _FAKE_EXPOSES_IMAGE[0]:
            obs_spaces["image"] = spaces.Box(
                low=0, high=255, shape=(_IMG_H, _IMG_W, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float64)
        self._step_count = 0
        self._pending: dict[str, float] = {}
        self._cell_index_seen = 0
        self._fail_this_episode = False

    def _build_obs(self) -> dict[str, np.ndarray[Any, Any]]:
        obs: dict[str, np.ndarray[Any, Any]] = {
            "cube_pos": np.zeros(3, dtype=np.float64),
        }
        if _FAKE_EXPOSES_IMAGE[0]:
            frame = np.full((_IMG_H, _IMG_W, 3), fill_value=20, dtype=np.uint8)
            x = (self._step_count * 3) % (_IMG_W - 4)
            y = (self._step_count * 3) % (_IMG_H - 4)
            frame[y : y + 4, x : x + 4, :] = 240
            obs["image"] = frame
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        self._step_count = 0
        # Deterministic per-seed failure decision for record_only_failures
        # tests: even seeds fail, odd seeds succeed (when _FAKE_FAILURE_PATTERN
        # is the magic string "even"). When None, always succeed.
        if _FAKE_FAILURE_PATTERN[0] == "even":
            self._fail_this_episode = (seed or 0) % 2 == 0
        else:
            self._fail_this_episode = False
        self._pending.clear()
        return self._build_obs(), {"seed_echo": seed}

    def step(
        self,
        action: np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        terminated = self._step_count >= 4  # short rollouts to keep tests fast
        success = terminated and not self._fail_this_episode
        return (
            self._build_obs(),
            1.0 if success else 0.0,
            terminated,
            False,
            {"success": success, "step_count": self._step_count},
        )

    def set_perturbation(self, name: str, value: float) -> None:
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self._pending[name] = value

    def restore_baseline(self) -> None:
        self._pending.clear()
        self._step_count = 0

    def close(self) -> None:
        return None


def _fake_image_env_factory() -> Any:
    """Module-level so spawn can pickle it."""
    return _FakeImageEnv()


def _make_random_policy() -> Any:
    """Random policy from the public registry; module-level for spawn."""
    from gauntlet.policy.random import RandomPolicy

    return RandomPolicy(action_dim=_ACTION_DIM, seed=None)


def _suite(*, episodes_per_cell: int = 2) -> Suite:
    return Suite(
        name="video-test-suite",
        env="tabletop",  # registered by gauntlet.env.__init__
        seed=42,
        episodes_per_cell=episodes_per_cell,
        axes={"distractor_count": AxisSpec(values=[0.0, 1.0])},
    )


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def test_default_runner_does_not_record_video(tmp_path: Path) -> None:
    """Opt-out: ``record_video=False`` (default) keeps Episodes video-free."""
    _FAKE_EXPOSES_IMAGE[0] = True
    _FAKE_FAILURE_PATTERN[0] = None

    runner = Runner(n_workers=1, env_factory=_fake_image_env_factory)
    eps = runner.run(policy_factory=_make_random_policy, suite=_suite())

    assert len(eps) == 4  # 2 cells * 2 eps
    for ep in eps:
        assert ep.video_path is None
    # No video directory was written to ``tmp_path`` because we never
    # opted in. (Belt-and-braces: confirm no MP4s anywhere under tmp.)
    assert not any(tmp_path.rglob("*.mp4"))


def test_record_video_writes_one_mp4_per_episode(tmp_path: Path) -> None:
    """Opt-in: every episode produces an MP4 at the canonical path."""
    _FAKE_EXPOSES_IMAGE[0] = True
    _FAKE_FAILURE_PATTERN[0] = None

    video_dir = tmp_path / "videos"
    runner = Runner(
        n_workers=1,
        env_factory=_fake_image_env_factory,
        record_video=True,
        video_dir=video_dir,
        video_fps=10,
    )
    eps = runner.run(policy_factory=_make_random_policy, suite=_suite())

    assert len(eps) == 4
    # Every Episode carries a relative video_path AND the file exists.
    for ep in eps:
        assert ep.video_path is not None
        # Path is relative to ``video_dir.parent`` (tmp_path) so the
        # default HTML embed Just Works.
        assert ep.video_path.startswith("videos/")
        absolute = tmp_path / ep.video_path
        assert absolute.exists(), f"missing MP4 at {absolute}"
        assert absolute.stat().st_size > 0
        # Filename matches video_path_for() — same source of truth.
        expected = video_path_for(
            video_dir,
            cell_index=ep.cell_index,
            episode_index=ep.episode_index,
            seed=ep.seed,
        )
        assert absolute == expected


def test_record_only_failures_skips_successes(tmp_path: Path) -> None:
    """``record_only_failures=True`` writes MP4s only for failed episodes."""
    _FAKE_EXPOSES_IMAGE[0] = True
    _FAKE_FAILURE_PATTERN[0] = "even"  # even-seeded episodes fail

    video_dir = tmp_path / "videos"
    runner = Runner(
        n_workers=1,
        env_factory=_fake_image_env_factory,
        record_video=True,
        video_dir=video_dir,
        video_fps=10,
        record_only_failures=True,
    )
    eps = runner.run(policy_factory=_make_random_policy, suite=_suite())

    assert len(eps) == 4
    failures = [ep for ep in eps if not ep.success]
    successes = [ep for ep in eps if ep.success]
    assert failures, "fake env should produce at least one failure under 'even' pattern"

    for ep in failures:
        assert ep.video_path is not None, "failed episode missing video"
        assert (tmp_path / ep.video_path).exists()
    for ep in successes:
        assert ep.video_path is None, "successful episode unexpectedly has video"

    # Disk economy: only as many MP4s as failures.
    mp4s = list(video_dir.glob("*.mp4"))
    assert len(mp4s) == len(failures)


def test_record_video_without_render_in_obs_raises(tmp_path: Path) -> None:
    """Missing ``obs['image']`` -> documented ValueError on the worker side."""
    _FAKE_EXPOSES_IMAGE[0] = False  # disable the image key
    _FAKE_FAILURE_PATTERN[0] = None

    runner = Runner(
        n_workers=1,
        env_factory=_fake_image_env_factory,
        record_video=True,
        video_dir=tmp_path / "videos",
        video_fps=10,
    )
    with pytest.raises(ValueError, match=r"render_in_obs=True"):
        runner.run(policy_factory=_make_random_policy, suite=_suite(episodes_per_cell=1))

    # Reset the toggle so subsequent tests aren't affected by ordering.
    _FAKE_EXPOSES_IMAGE[0] = True


def test_record_video_default_video_dir_under_trajectory_dir(tmp_path: Path) -> None:
    """Default ``video_dir`` is ``trajectory_dir/"videos"`` for one-stop output."""
    _FAKE_EXPOSES_IMAGE[0] = True
    _FAKE_FAILURE_PATTERN[0] = None

    traj = tmp_path / "run-out"
    runner = Runner(
        n_workers=1,
        env_factory=_fake_image_env_factory,
        trajectory_dir=traj,
        record_video=True,
        # video_dir omitted -> picks trajectory_dir / "videos"
        video_fps=10,
    )
    eps = runner.run(policy_factory=_make_random_policy, suite=_suite(episodes_per_cell=1))

    expected_video_dir = traj / "videos"
    assert expected_video_dir.is_dir()
    for ep in eps:
        assert ep.video_path is not None
        assert (traj / ep.video_path).exists()


def test_record_video_validates_fps(tmp_path: Path) -> None:
    """Invalid ``video_fps`` is rejected at construction time."""
    with pytest.raises(ValueError, match="video_fps"):
        Runner(
            n_workers=1,
            env_factory=_fake_image_env_factory,
            record_video=True,
            video_dir=tmp_path / "videos",
            video_fps=0,
        )


def test_fake_env_satisfies_protocol() -> None:
    """Sanity: the fake env actually conforms to the Runner's Protocol."""
    _FAKE_EXPOSES_IMAGE[0] = True
    env = _FakeImageEnv()
    assert isinstance(env, GauntletEnv)
    env.close()
