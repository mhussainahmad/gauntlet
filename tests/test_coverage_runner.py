"""Branch-coverage backfill for :mod:`gauntlet.runner`.

Phase 2.5 Task 11. Targets the missed-branch lines under the default
test selection (``not (hf or lerobot or monitor or pybullet or genesis
or ros2 or isaac or video or render)``):

* ``runner.py``
  * L211 — ``record_video=True`` with non-positive / non-int ``video_fps``
  * L510-516 — ``_resolve_video_config`` defaulting branches:
    explicit ``video_dir`` wins, ``trajectory_dir / "videos"`` is the
    default, ``Path("videos")`` is the fallback when both are unset.
  * L301 — video_config mkdir is exercised on a fresh path.
* ``worker.py``
  * L406 — ``record_video=True`` against an env that does NOT expose
    ``obs["image"]`` raises ``ValueError`` with the install hint.
  * L429, L450-465, L517-523 — frame-buffer append, relative-path
    derivation, and the actual ``VideoWriter.write`` call. ``imageio``
    is mocked at the module boundary so the test runs in the default
    torch-/extras-free job (no ``[video]`` extra installed).

The tests use a small in-module fake env that satisfies
:class:`gauntlet.env.base.GauntletEnv` structurally and optionally
emits an ``image`` observation. No MuJoCo / no PyBullet / no genuine
imageio at import time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.env.base import GauntletEnv
from gauntlet.runner import Runner
from gauntlet.runner.worker import (
    VideoConfig,
    WorkItem,
    execute_one,
    extract_env_seed,
)

_ACTION_DIM = 7


class _FakeImageEnv:
    """Minimal :class:`GauntletEnv` that emits ``obs["image"]`` per step.

    Deterministic, succeeds on step 1. Used to drive the Runner's
    record-video path end-to-end without MuJoCo or PyBullet.
    """

    AXIS_NAMES = frozenset({"distractor_count"})
    VISUAL_ONLY_AXES: frozenset[str] = frozenset()

    def __init__(self, *, render: bool = True) -> None:
        from gymnasium import spaces

        self._render = render
        self.observation_space = spaces.Dict(
            {
                "cube_pos": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float64)
        self._pending: dict[str, float] = {}

    def _obs(self) -> dict[str, NDArray[Any]]:
        out: dict[str, NDArray[Any]] = {"cube_pos": np.zeros(3, dtype=np.float64)}
        if self._render:
            out["image"] = np.zeros((4, 4, 3), dtype=np.uint8)
        return out

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        self._pending.clear()
        return self._obs(), {"seed_echo": seed}

    def step(
        self,
        action: NDArray[Any],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        return self._obs(), 1.0, True, False, {"success": True}

    def set_perturbation(self, name: str, value: float) -> None:
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown axis {name!r}")
        self._pending[name] = value

    def restore_baseline(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        return None


def _fake_image_env_factory() -> Any:
    return _FakeImageEnv(render=True)


def _fake_state_only_env_factory() -> Any:
    return _FakeImageEnv(render=False)


class _NoopPolicy:
    def act(self, obs: Any) -> NDArray[np.float64]:
        return np.zeros(_ACTION_DIM, dtype=np.float64)


def _noop_policy_factory() -> _NoopPolicy:
    return _NoopPolicy()


def _make_work_item(*, cell_index: int = 0, episode_index: int = 0) -> WorkItem:
    master = np.random.SeedSequence(7)
    cell = master.spawn(1)[0]
    ep = cell.spawn(1)[0]
    return WorkItem(
        suite_name="cov-runner",
        cell_index=cell_index,
        episode_index=episode_index,
        perturbation_values={},
        episode_seq=ep,
        master_seed=7,
        n_cells=1,
        episodes_per_cell=1,
    )


# ----------------------------------------------------------------------------
# Runner.__init__ — record_video + bad fps validation (L211).
# ----------------------------------------------------------------------------


def test_runner_rejects_non_positive_video_fps_when_record_video() -> None:
    """Runner.__init__ L211: ``record_video=True`` + non-positive ``video_fps``."""
    with pytest.raises(ValueError, match="video_fps"):
        Runner(record_video=True, video_fps=0)
    with pytest.raises(ValueError, match="video_fps"):
        Runner(record_video=True, video_fps=-1)


def test_runner_rejects_non_int_video_fps_when_record_video() -> None:
    """``video_fps`` must be an int — float is rejected."""
    with pytest.raises(ValueError, match="video_fps"):
        Runner(record_video=True, video_fps=24.0)  # type: ignore[arg-type]


def test_runner_accepts_invalid_video_fps_when_record_video_disabled() -> None:
    """When ``record_video=False`` (the default), ``video_fps`` is unused
    so a 0 / float must not raise — the field is plumbed only when video
    recording is enabled. Mirrors the short-circuit in the validator."""
    Runner(record_video=False, video_fps=0)
    Runner(record_video=False, video_fps=24.0)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Runner._resolve_video_config — three-way default branch coverage (L510-516).
# ----------------------------------------------------------------------------


def test_resolve_video_config_returns_none_when_record_video_false() -> None:
    runner = Runner()
    assert runner._resolve_video_config() is None


def test_resolve_video_config_uses_explicit_video_dir(tmp_path: Path) -> None:
    """Explicit ``video_dir`` wins over both other branches."""
    video_dir = tmp_path / "explicit"
    runner = Runner(record_video=True, video_dir=video_dir, trajectory_dir=tmp_path)
    config = runner._resolve_video_config()
    assert isinstance(config, VideoConfig)
    assert config.video_dir == video_dir


def test_resolve_video_config_defaults_under_trajectory_dir(tmp_path: Path) -> None:
    """No explicit ``video_dir`` + ``trajectory_dir`` set → ``trajectory_dir / 'videos'``."""
    runner = Runner(record_video=True, trajectory_dir=tmp_path)
    config = runner._resolve_video_config()
    assert config is not None
    assert config.video_dir == tmp_path / "videos"


def test_resolve_video_config_falls_back_to_videos_when_no_dirs_set() -> None:
    """No ``video_dir``, no ``trajectory_dir`` → ``Path("videos")``."""
    runner = Runner(record_video=True)
    config = runner._resolve_video_config()
    assert config is not None
    assert config.video_dir == Path("videos")


def test_resolve_video_config_propagates_fps_and_record_only_failures(
    tmp_path: Path,
) -> None:
    runner = Runner(
        record_video=True,
        video_dir=tmp_path,
        video_fps=15,
        record_only_failures=True,
    )
    config = runner._resolve_video_config()
    assert config is not None
    assert config.fps == 15
    assert config.record_only_failures is True


# ----------------------------------------------------------------------------
# worker.execute_one — record_video without obs['image'] (worker.py L406).
# ----------------------------------------------------------------------------


def test_execute_one_record_video_requires_image_obs(tmp_path: Path) -> None:
    """worker.py L406: ``record_video=True`` against a state-only env raises."""
    env = _fake_state_only_env_factory()
    item = _make_work_item()
    config = VideoConfig(video_dir=tmp_path / "videos", fps=10)

    with pytest.raises(ValueError, match="render_in_obs=True"):
        execute_one(
            env,
            _noop_policy_factory,
            item,
            trajectory_dir=None,
            video_config=config,
        )

    env.close()


# ----------------------------------------------------------------------------
# worker.execute_one — full record_video path with mocked imageio.
#
# Mocks imageio.v3 at the module boundary so VideoWriter.write succeeds
# without the [video] extra installed. Verifies the frame buffer append
# (L429), the relative-path derivation (L450-465), and the encode call
# (L517-523) all run.
# ----------------------------------------------------------------------------


@pytest.fixture
def _mock_imageio(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``imageio.v3.imwrite`` with a MagicMock.

    The ``[video]`` extra (or a transitive dependency) may already have
    pulled real ``imageio`` into ``sys.modules``; in that case
    intercepting at ``sys.modules`` is too late because
    ``VideoWriter.write``'s ``import imageio.v3 as iio`` will rebind
    to the real module already cached. Instead we patch
    ``imageio.v3.imwrite`` directly so the call observes a mock,
    regardless of whether imageio is real or absent.
    """
    mock = MagicMock()
    # Make sure imageio + imageio.v3 are importable (mock-substituting if
    # the [video] extra is missing) so the lazy import inside
    # VideoWriter.write resolves cleanly.
    if "imageio" not in sys.modules:
        monkeypatch.setitem(sys.modules, "imageio", MagicMock())
    if "imageio.v3" not in sys.modules:
        v3 = MagicMock()
        monkeypatch.setitem(sys.modules, "imageio.v3", v3)
    monkeypatch.setattr("imageio.v3.imwrite", mock)
    return mock


def test_execute_one_record_video_writes_via_mocked_imageio(
    tmp_path: Path, _mock_imageio: MagicMock
) -> None:
    """Full happy path: env yields an image, video_config drives the write."""
    env = _fake_image_env_factory()
    item = _make_work_item(cell_index=4, episode_index=2)
    video_dir = tmp_path / "videos"
    config = VideoConfig(video_dir=video_dir, fps=12)

    episode = execute_one(
        env,
        _noop_policy_factory,
        item,
        trajectory_dir=None,
        video_config=config,
    )

    # The video_path on the Episode is the path RELATIVE to video_dir.parent.
    assert episode.video_path is not None
    assert episode.video_path.startswith("videos/")
    # imageio.v3.imwrite was called with the absolute path + frames.
    assert _mock_imageio.call_count == 1
    written_path = _mock_imageio.call_args.args[0]
    assert written_path.parent == video_dir
    assert written_path.name.startswith("episode_cell0004_ep0002_seed")
    assert written_path.name.endswith(".mp4")
    env.close()


def test_execute_one_record_video_record_only_failures_skips_write_on_success(
    tmp_path: Path, _mock_imageio: MagicMock
) -> None:
    """``record_only_failures=True`` + a successful rollout → no write."""
    env = _fake_image_env_factory()
    item = _make_work_item()
    config = VideoConfig(video_dir=tmp_path / "videos", fps=10, record_only_failures=True)

    episode = execute_one(
        env,
        _noop_policy_factory,
        item,
        trajectory_dir=None,
        video_config=config,
    )

    # Episode is successful (the fake env always succeeds). Write skipped.
    assert episode.success is True
    assert episode.video_path is None
    assert _mock_imageio.call_count == 0
    env.close()


def test_execute_one_record_video_relative_to_falls_back_to_absolute(
    tmp_path: Path, _mock_imageio: MagicMock
) -> None:
    """worker.py L460-465: when ``video_dir`` is not under ``relative_to``,
    the relative-path call raises ``ValueError`` and we fall back to the
    absolute path string. Triggered by an unrelated anchor directory.
    """
    env = _fake_image_env_factory()
    item = _make_work_item()
    video_dir = tmp_path / "videos"
    # Anchor the relative path against an unrelated tree so
    # ``Path.relative_to`` raises and we hit the fallback branch.
    unrelated = tmp_path.parent
    config = VideoConfig(video_dir=video_dir, fps=10, relative_to=unrelated / "elsewhere")

    episode = execute_one(
        env,
        _noop_policy_factory,
        item,
        trajectory_dir=None,
        video_config=config,
    )

    assert episode.video_path is not None
    # Absolute path written into the Episode because relative_to failed.
    assert Path(episode.video_path).is_absolute()
    env.close()


# ----------------------------------------------------------------------------
# Runner.run — exercises the parent-process video_config mkdir (runner.py L301).
# ----------------------------------------------------------------------------


def test_runner_run_creates_video_dir_on_entry(tmp_path: Path, _mock_imageio: MagicMock) -> None:
    """runner.py L300-301: when video is enabled, the parent process
    mkdirs ``video_config.video_dir`` exactly once on entry, before any
    worker can race against it. Verified by pointing video_dir at a
    not-yet-existing path and observing it post-run.
    """
    from gauntlet.suite.schema import AxisSpec, Suite

    suite = Suite(
        name="cov-vid",
        env="tabletop",  # accepted by schema; we override env_factory below.
        seed=1,
        episodes_per_cell=1,
        axes={"distractor_count": AxisSpec(values=[0.0])},
    )

    video_dir = tmp_path / "fresh" / "videos"
    assert not video_dir.exists()

    runner = Runner(
        n_workers=1,
        env_factory=_fake_image_env_factory,
        record_video=True,
        video_dir=video_dir,
        video_fps=10,
    )
    episodes = runner.run(policy_factory=_noop_policy_factory, suite=suite)

    assert video_dir.is_dir()
    assert len(episodes) == 1
    # imageio was driven via the worker for the single rollout.
    assert _mock_imageio.call_count == 1


# ----------------------------------------------------------------------------
# Runner.run — exercises the parent-process trajectory_dir mkdir
# (runner.py L293-294). No existing test in tests/test_runner.py drives
# the runner with ``trajectory_dir`` set; this fills that branch.
# ----------------------------------------------------------------------------


def test_runner_run_creates_trajectory_dir_on_entry(tmp_path: Path) -> None:
    """runner.py L293-294: ``trajectory_dir`` is mkdir-ed exactly once on
    the parent process before any worker can race against it."""
    from gauntlet.suite.schema import AxisSpec, Suite

    suite = Suite(
        name="cov-traj",
        env="tabletop",
        seed=1,
        episodes_per_cell=1,
        axes={"distractor_count": AxisSpec(values=[0.0])},
    )

    traj_dir = tmp_path / "fresh" / "trajs"
    assert not traj_dir.exists()

    runner = Runner(
        n_workers=1,
        env_factory=_fake_image_env_factory,
        trajectory_dir=traj_dir,
    )
    episodes = runner.run(policy_factory=_noop_policy_factory, suite=suite)

    assert traj_dir.is_dir()
    assert len(episodes) == 1
    # The worker also wrote the per-episode NPZ under that dir.
    npz_files = list(traj_dir.glob("*.npz"))
    assert len(npz_files) == 1


# ----------------------------------------------------------------------------
# extract_env_seed — uint32 round-trip sanity (already in test_worker_direct
# but a duplicate keeps the runner-coverage file self-contained for future
# reviewers tracing one branch report at a time).
# ----------------------------------------------------------------------------


def test_extract_env_seed_uint32_range() -> None:
    seq = np.random.SeedSequence(99)
    seed = extract_env_seed(seq)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


# ----------------------------------------------------------------------------
# Sanity: the fake env satisfies the GauntletEnv Protocol (also exercises
# the runtime_checkable Protocol body in env.base).
# ----------------------------------------------------------------------------


def test_fake_image_env_satisfies_gauntlet_env_protocol() -> None:
    env = _fake_image_env_factory()
    try:
        assert isinstance(env, GauntletEnv)
    finally:
        env.close()
