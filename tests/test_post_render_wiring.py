"""Runner wiring for the post-render / observation-overlay axes.

``image_attack``, ``color_shift_synthetic`` and ``instruction_paraphrase``
are implemented as env wrappers rather than backend axes. These tests
pin the wiring that lets a suite declaring them run through
:class:`gauntlet.runner.Runner` (and therefore ``gauntlet run``):

* wrappers stack without losing the inner backend's axes;
* the factory is picklable (spawned workers unpickle it);
* the wrapper order is fixed;
* an active image axis on an env that emits no image fails loudly;
* registry-built envs get ``render_in_obs=True`` when an image axis is
  present;
* a Runner sweep over all three axes completes.

A fake image env stands in for MuJoCo so no GL context is needed.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.env.base import GauntletEnv
from gauntlet.env.color_attack import ColorShiftWrapper
from gauntlet.env.image_attack import ImageAttackWrapper
from gauntlet.env.instruction import InstructionWrapper
from gauntlet.env.post_render import (
    WrappedEnvFactory,
    suite_needs_render,
    wrap_env_factory,
)
from gauntlet.env.registry import register_env
from gauntlet.policy import RandomPolicy
from gauntlet.runner import Runner
from gauntlet.suite import load_suite_from_string


class _FakeImageEnv:
    """Minimal GauntletEnv emitting a mid-grey image (or none)."""

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset({"lighting_intensity"})
    last_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, render_in_obs: bool = True, max_steps: int = 3) -> None:
        type(self).last_kwargs = {"render_in_obs": render_in_obs, "max_steps": max_steps}
        self._render = render_in_obs
        self._max_steps = max_steps
        spaces: dict[str, gym.spaces.Space[Any]] = {
            "ee_pos": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float64)
        }
        if render_in_obs:
            spaces["image"] = gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float64)
        self.lighting: float | None = None
        self._t = 0

    def _obs(self) -> dict[str, NDArray[Any]]:
        obs: dict[str, NDArray[Any]] = {"ee_pos": np.zeros(3)}
        if self._render:
            obs["image"] = np.full((8, 8, 3), 128, dtype=np.uint8)
        return obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        self._t = 0
        return self._obs(), {"success": False}

    def step(
        self, action: NDArray[Any]
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        self._t += 1
        return self._obs(), 0.0, False, self._t >= self._max_steps, {"success": False}

    def set_perturbation(self, name: str, value: float) -> None:
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self.lighting = float(value)

    def restore_baseline(self) -> None:
        self.lighting = None

    def close(self) -> None:
        pass


def _g(env: object) -> GauntletEnv:
    return cast(GauntletEnv, env)


_FAKE_FACTORY = cast("Callable[[], GauntletEnv]", _FakeImageEnv)

_FAKE_ENV_NAME = "fake-post-render-image"
register_env(_FAKE_ENV_NAME, _FakeImageEnv)  # type: ignore[arg-type]


def _suite_yaml(*, env: str = _FAKE_ENV_NAME) -> str:
    return f"""
name: post-render-wiring
env: {env}
seed: 3
episodes_per_cell: 1
axes:
  lighting_intensity:
    values: [0.5, 1.0]
  image_attack:
    values: [0, 2]
  color_shift_synthetic:
    values: [0, 5]
  instruction_paraphrase:
    values: ["pick up the cube", "grab the block"]
"""


def test_wrappers_stack_without_losing_inner_axes() -> None:
    env = InstructionWrapper(
        _g(ImageAttackWrapper(_g(ColorShiftWrapper(_g(_FakeImageEnv()))))), ("a", "b")
    )
    assert {
        "lighting_intensity",
        "image_attack",
        "color_shift_synthetic",
        "instruction_paraphrase",
    } <= set(env.AXIS_NAMES)
    # Routed through three wrappers down to the backend.
    env.set_perturbation("lighting_intensity", 0.7)
    assert env.lighting == 0.7


def test_wrapped_factory_order_and_pickle() -> None:
    factory = WrappedEnvFactory(
        base=_FAKE_FACTORY,
        image_attack=True,
        color_shift=True,
        paraphrases=("a", "b"),
    )
    clone = pickle.loads(pickle.dumps(factory))
    env = clone()
    # Outermost first: instruction -> image attack -> colour shift -> backend.
    assert isinstance(env, InstructionWrapper)
    assert isinstance(env._inner, ImageAttackWrapper)
    assert isinstance(env._inner._inner, ColorShiftWrapper)
    assert isinstance(env._inner._inner._inner, _FakeImageEnv)


def test_active_image_attack_without_image_fails_loudly() -> None:
    env = ImageAttackWrapper(_g(_FakeImageEnv(render_in_obs=False)))
    env.set_perturbation("image_attack", 2.0)
    with pytest.raises(ValueError, match="render_in_obs"):
        env.reset(seed=0)
    shifted = ColorShiftWrapper(_g(_FakeImageEnv(render_in_obs=False)))
    shifted.set_perturbation("color_shift_synthetic", 5.0)
    with pytest.raises(ValueError, match="render_in_obs"):
        shifted.reset(seed=0)


def test_baseline_attack_without_image_is_allowed() -> None:
    env = ImageAttackWrapper(_g(_FakeImageEnv(render_in_obs=False)))
    env.set_perturbation("image_attack", 0.0)
    obs, _ = env.reset(seed=0)
    assert "image" not in obs


def test_registry_factory_gets_render_in_obs_for_image_axes() -> None:
    suite = load_suite_from_string(_suite_yaml())
    assert suite_needs_render(suite)
    factory = wrap_env_factory(suite, None)
    factory()
    assert _FakeImageEnv.last_kwargs["render_in_obs"] is True


def test_no_post_render_axes_leaves_factory_untouched() -> None:
    suite = load_suite_from_string(
        f"""
name: plain
env: {_FAKE_ENV_NAME}
episodes_per_cell: 1
axes:
  lighting_intensity:
    values: [0.5]
"""
    )
    assert not suite_needs_render(suite)
    assert wrap_env_factory(suite, _FAKE_FACTORY) is _FAKE_FACTORY


def test_runner_sweeps_all_post_render_axes() -> None:
    suite = load_suite_from_string(_suite_yaml())
    episodes = Runner(n_workers=1).run(
        policy_factory=partial(RandomPolicy, action_dim=7),
        suite=suite,
    )
    assert len(episodes) == 2 * 2 * 2 * 2
    seen = {
        (
            ep.perturbation_config["image_attack"],
            ep.perturbation_config["color_shift_synthetic"],
            ep.perturbation_config["instruction_paraphrase"],
        )
        for ep in episodes
    }
    assert len(seen) == 8


def test_replay_reproduces_episode_with_wrapper_axes() -> None:
    """``replay_one`` builds its env through the same wrapper wiring."""
    from gauntlet.replay import replay_one

    suite = load_suite_from_string(_suite_yaml())
    episodes = Runner(n_workers=1).run(
        policy_factory=partial(RandomPolicy, action_dim=7),
        suite=suite,
    )
    target = next(ep for ep in episodes if ep.perturbation_config["image_attack"] == 2.0)
    replayed = replay_one(
        target=target,
        suite=suite,
        policy_factory=partial(RandomPolicy, action_dim=7),
    )
    assert replayed.perturbation_config == target.perturbation_config
    assert replayed.step_count == target.step_count
