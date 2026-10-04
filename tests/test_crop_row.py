"""``env: crop-row`` — camera-guided row following (B-47)."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from gauntlet.env.base import GauntletEnv
from gauntlet.env.crop_row import CROP_ROW_AXES, CropRowEnv
from gauntlet.env.registry import get_env_factory
from gauntlet.policy.crop_row import CropRowClassicalPolicy, row_steering
from gauntlet.policy.registry import resolve_policy_factory
from gauntlet.runner import Runner
from gauntlet.suite import load_suite_from_string

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "crop_row"


def _first_frame(**axes: float) -> np.ndarray:
    env = CropRowEnv()
    for k, v in axes.items():
        env.set_perturbation(k, v)
    obs, _ = env.reset(seed=4)
    return obs["image"]


def _rollout(policy: object, seed: int, **axes: float) -> dict[str, object]:
    env = CropRowEnv()
    for k, v in axes.items():
        env.set_perturbation(k, v)
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(policy.act(obs))  # type: ignore[attr-defined]
        done = term or trunc
    return info


# ----- contract ----------------------------------------------------------------


def test_protocol_registry_and_gym_id() -> None:
    env = CropRowEnv()
    assert isinstance(env, GauntletEnv)
    assert get_env_factory("crop-row") is CropRowEnv
    assert gym.make("gauntlet/CropRow-v0").unwrapped.__class__ is CropRowEnv
    assert resolve_policy_factory("crop-row-classical") is CropRowClassicalPolicy


def test_observation_is_image_only_and_truth_is_in_info() -> None:
    env = CropRowEnv(render_size=(48, 64))
    obs, info = env.reset(seed=0)
    assert set(obs) == {"image"}
    assert obs["image"].shape == (48, 64, 3) and obs["image"].dtype == np.uint8
    assert env.observation_space.contains(obs)
    assert {"lateral_error", "heading_error", "success"} <= set(info)
    obs, *_ = env.step(np.zeros(1))
    assert env.observation_space.contains(obs)


def test_same_seed_same_frames_different_seed_different_frames() -> None:
    a = CropRowEnv()
    b = CropRowEnv()
    fa = [a.reset(seed=3)[0]["image"]] + [a.step(np.array([0.3]))[0]["image"] for _ in range(5)]
    fb = [b.reset(seed=3)[0]["image"]] + [b.step(np.array([0.3]))[0]["image"] for _ in range(5)]
    for x, y in zip(fa, fb, strict=True):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a.reset(seed=4)[0]["image"], fa[0])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("dust_density", 0.8),
        ("glare_intensity", 0.8),
        ("motion_blur", 0.8),
        ("weed_density", 0.8),
        ("row_curvature", 0.1),
        ("lighting_intensity", 0.4),
    ],
)
def test_every_axis_changes_the_image(name: str, value: float) -> None:
    base = _first_frame().astype(np.float64)
    pert = _first_frame(**{name: value}).astype(np.float64)
    assert np.abs(base - pert).mean() > 2.0


def _exg_spread(image: np.ndarray) -> float:
    im = image.astype(np.float64)
    return float((2 * im[..., 1] - im[..., 0] - im[..., 2]).std())


def test_dust_washes_out_vegetation_and_glare_brightens() -> None:
    base = _first_frame()
    assert _exg_spread(_first_frame(dust_density=1.0)) < 0.25 * _exg_spread(base)
    assert _first_frame(glare_intensity=1.0).mean() > base.astype(np.float64).mean() + 30


def test_perturbations_are_queued_and_cleared_by_restore_baseline() -> None:
    env = CropRowEnv()
    env.set_perturbation("dust_density", 1.0)
    dusty = env.reset(seed=4)[0]["image"]
    np.testing.assert_array_equal(env.reset(seed=4)[0]["image"], dusty)  # persists
    env.restore_baseline()
    np.testing.assert_array_equal(env.reset(seed=4)[0]["image"], _first_frame())


def test_axis_validation() -> None:
    env = CropRowEnv()
    assert frozenset(env.AXIS_NAMES) == CROP_ROW_AXES
    with pytest.raises(ValueError, match="unknown perturbation axis"):
        env.set_perturbation("distractor_count", 1.0)
    with pytest.raises(ValueError, match=r"dust_density must be in"):
        env.set_perturbation("dust_density", 1.5)
    with pytest.raises(ValueError, match="must be in"):
        env.set_perturbation("weed_density", float("nan"))


def test_step_validation() -> None:
    env = CropRowEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError, match=r"shape \(1,\)"):
        env.step(np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        env.step(np.array([np.nan]))


def test_leaving_the_corridor_terminates_as_failure() -> None:
    env = CropRowEnv()
    env.reset(seed=0)
    term = False
    for _ in range(100):
        _, _, term, trunc, info = env.step(np.array([1.0]))
        if term or trunc:
            break
    assert term and not info["success"]
    assert abs(info["lateral_error"]) >= CropRowEnv.CORRIDOR


def test_ground_truth_controller_succeeds() -> None:
    """The task is solvable: steering on the true errors always succeeds."""

    class Oracle:
        def __init__(self, env: CropRowEnv) -> None:
            self.env = env

        def act(self, obs: object) -> np.ndarray:
            return row_steering(self.env._lateral_error(), self.env._heading_error())

    for seed in range(5):
        env = CropRowEnv()
        env.set_perturbation("row_curvature", 0.1)
        obs, info = env.reset(seed=seed)
        oracle = Oracle(env)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(oracle.act(obs))
            done = term or trunc
        assert info["success"]


def test_ground_coords_geometry() -> None:
    on, fwd, left = CropRowEnv.ground_coords((96, 128))
    assert on[-1].all() and not on[0].all()  # bottom row on the ground, top row sky
    assert fwd[-1, 64] > 0 and abs(left[-1, 63] + left[-1, 64]) < 1e-9
    assert left[-1, 0] > 0 > left[-1, -1]  # image left = vehicle left


# ----- reference policy ----------------------------------------------------------


def test_classical_policy_estimates_and_follows_clean_rows() -> None:
    policy = CropRowClassicalPolicy()
    env = CropRowEnv()
    obs, info = env.reset(seed=1)
    est = policy.estimate(obs["image"])
    assert est is not None
    assert abs(est[0] - info["lateral_error"]) < 0.02
    assert abs(est[1] - info["heading_error"]) < 0.02
    wins = sum(bool(_rollout(CropRowClassicalPolicy(), s)["success"]) for s in range(10))
    assert wins == 10


def test_classical_policy_reports_no_row_in_dense_dust() -> None:
    assert CropRowClassicalPolicy().estimate(_first_frame(dust_density=1.0)) is None


# ----- runner integration ----------------------------------------------------------

_SUITE = """
name: crop-row-test
env: crop-row
seed: 3
episodes_per_cell: 2
axes:
  dust_density:
    values: [0.0, 1.0]
  inference_delay_jitter:
    values: [0, 200]
"""


def test_runs_from_a_suite_with_latency() -> None:
    suite = load_suite_from_string(_SUITE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # control_dt is exposed: no fallback warning
        episodes = Runner(n_workers=1).run(policy_factory=CropRowClassicalPolicy, suite=suite)
    assert len(episodes) == 8
    clean = [
        e
        for e in episodes
        if e.perturbation_config == {"dust_density": 0.0, "inference_delay_jitter": 0.0}
    ]
    assert len(clean) == 2 and all(e.success for e in clean)
    assert all(e.step_count > 0 for e in episodes)


# ----- CNN example (numpy inference, no torch) -----------------------------------


def _cnn_policy() -> Any:
    spec = importlib.util.spec_from_file_location("cnn_policy", _EXAMPLE_DIR / "cnn_policy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_policy()


def test_cnn_example_follows_clean_rows() -> None:
    policy = _cnn_policy()
    env = CropRowEnv()
    obs, info = env.reset(seed=2)
    e, h = policy.estimate(obs["image"])
    assert abs(e - info["lateral_error"]) < 0.02 and abs(h - info["heading_error"]) < 0.02
    assert all(_rollout(_cnn_policy(), s)["success"] for s in range(3))
