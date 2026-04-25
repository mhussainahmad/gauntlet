"""Tests for ``gauntlet.env.perturbation`` and the env-side dispatch hook.

Covers:

* ``AXIS_NAMES`` registry contents and ordering (stable downstream contract).
* Per-axis construction + apply (one parametric test per scalar axis name).
* Observable env-state changes for each axis.
* Baseline restoration after applying a perturbation.
* Reset implicitly clears perturbation effects (via ``restore_baseline``).
* Determinism: perturbed envs with identical inputs produce bit-identical obs.
* Sampler reproducibility under fixed RNG seed.
* Validation of unknown axis names + DistractorCount edge cases.

These tests must run headless (no GL): no ``env.render()`` calls.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from gauntlet.env import N_DISTRACTOR_SLOTS, TabletopEnv
from gauntlet.env.perturbation import (
    AXIS_KIND_CATEGORICAL,
    AXIS_KIND_CONTINUOUS,
    AXIS_KIND_INT,
    AXIS_NAMES,
    PerturbationAxis,
    axis_for,
    camera_offset_x,
    camera_offset_y,
    distractor_count,
    initial_state_ood,
    lighting_intensity,
    object_initial_pose_x,
    object_initial_pose_y,
    object_texture,
)

# A fixed value per axis that produces a clearly observable model-state
# change versus baseline. Chosen inside DEFAULT_BOUNDS where possible.
_APPLY_VALUE: dict[str, float] = {
    "lighting_intensity": 0.25,  # baseline diffuse=1.0
    "camera_offset_x": 0.07,
    "camera_offset_y": -0.05,
    "object_texture": 1.0,  # flip to alt colour
    "object_initial_pose_x": 0.123,
    "object_initial_pose_y": -0.087,
    "distractor_count": 3.0,
    "initial_state_ood": 1.0,  # 1-sigma OOD displacement (B-32)
    # B-31 — image_attack is a *post-render* axis; it is in the
    # canonical AXIS_NAMES registry (so the suite YAML accepts it) but
    # the inner backend env does NOT implement it. The
    # ImageAttackWrapper handles dispatch instead. Tests that exercise
    # backend-direct ``set_perturbation`` filter this name out.
    "image_attack": 0.0,
}

# B-31 — axes whose dispatch lives outside the inner backend env.
# Filtered out of any test that drives ``TabletopEnv.set_perturbation``
# directly (the env legitimately rejects them).
_BACKEND_DIRECT_AXES: tuple[str, ...] = tuple(name for name in AXIS_NAMES if name != "image_attack")


def _zero_action() -> np.ndarray:
    return np.zeros(7, dtype=np.float64)


@pytest.fixture
def env() -> Iterator[TabletopEnv]:
    """Fresh TabletopEnv per test, closed on teardown."""
    e = TabletopEnv()
    try:
        yield e
    finally:
        e.close()


# ---------------------------------------------------------------- registry


class TestAxisNamesRegistry:
    def test_axis_names_are_canonical(self) -> None:
        # Order is the stable downstream contract — see AXIS_NAMES
        # docstring in env/perturbation/__init__.py.
        assert AXIS_NAMES == (
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_texture",
            "object_initial_pose_x",
            "object_initial_pose_y",
            "distractor_count",
            "initial_state_ood",
            "image_attack",
        )

    def test_axis_names_is_tuple(self) -> None:
        # tuple is immutable; downstream code keys off both contents AND order.
        assert isinstance(AXIS_NAMES, tuple)

    def test_no_duplicate_axis_names(self) -> None:
        assert len(set(AXIS_NAMES)) == len(AXIS_NAMES)


# ----------------------------------------------------------- axis factories


class TestAxisFactories:
    @pytest.mark.parametrize("name", AXIS_NAMES)
    def test_axis_for_returns_axis_with_matching_name(self, name: str) -> None:
        axis = axis_for(name)
        assert isinstance(axis, PerturbationAxis)
        assert axis.name == name

    def test_axis_for_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown perturbation axis"):
            axis_for("not_a_real_axis")

    def test_individual_constructors_match_axis_for(self) -> None:
        # axis_for is a registry over the public constructors.
        from gauntlet.env.perturbation import image_attack

        ctors = {
            "lighting_intensity": lighting_intensity,
            "camera_offset_x": camera_offset_x,
            "camera_offset_y": camera_offset_y,
            "object_texture": object_texture,
            "object_initial_pose_x": object_initial_pose_x,
            "object_initial_pose_y": object_initial_pose_y,
            "distractor_count": distractor_count,
            "initial_state_ood": initial_state_ood,
            "image_attack": image_attack,
        }
        for name, ctor in ctors.items():
            assert ctor().name == name
            assert axis_for(name).name == name

    def test_axis_kinds(self) -> None:
        from gauntlet.env.perturbation import image_attack

        assert lighting_intensity().kind == AXIS_KIND_CONTINUOUS
        assert camera_offset_x().kind == AXIS_KIND_CONTINUOUS
        assert camera_offset_y().kind == AXIS_KIND_CONTINUOUS
        assert object_initial_pose_x().kind == AXIS_KIND_CONTINUOUS
        assert object_initial_pose_y().kind == AXIS_KIND_CONTINUOUS
        assert object_texture().kind == AXIS_KIND_CATEGORICAL
        assert distractor_count().kind == AXIS_KIND_INT
        assert initial_state_ood().kind == AXIS_KIND_CONTINUOUS
        assert image_attack().kind == AXIS_KIND_CATEGORICAL

    def test_continuous_factory_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="low must be <= high"):
            lighting_intensity(low=1.0, high=0.0)

    def test_distractor_count_factory_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="within \\[0, 10\\]"):
            distractor_count(low=0, high=11)
        with pytest.raises(ValueError, match="within \\[0, 10\\]"):
            distractor_count(low=-1, high=5)


# ------------------------------------------------------------ env dispatch


class TestSetPerturbationDispatch:
    """The env-side hook every axis ultimately calls into.

    ``image_attack`` is excluded from the parametrize set: its dispatch
    lives in :class:`gauntlet.env.image_attack.ImageAttackWrapper`, not
    on the inner backend env, so the inner env legitimately rejects it.
    See ``tests/test_perturbation_image_attack.py`` for the wrapper
    coverage.
    """

    @pytest.mark.parametrize("name", _BACKEND_DIRECT_AXES)
    def test_known_axis_accepted_and_queued(self, env: TabletopEnv, name: str) -> None:
        env.set_perturbation(name, _APPLY_VALUE[name])
        # Queued, not yet applied; reset() consumes the queue.
        assert name in env._pending_perturbations
        env.reset(seed=0)
        assert env._pending_perturbations == {}

    def test_unknown_axis_name_raises_clear_error(self, env: TabletopEnv) -> None:
        with pytest.raises(ValueError, match="unknown perturbation axis"):
            env.set_perturbation("not_a_real_axis", 1.0)

    def test_image_attack_rejected_by_backend_directly(self, env: TabletopEnv) -> None:
        # image_attack is in the canonical AXIS_NAMES registry but the
        # inner backend env does not implement it (the wrapper does).
        # Routing it directly to the backend must fail loudly.
        with pytest.raises(ValueError, match="unknown perturbation axis"):
            env.set_perturbation("image_attack", 0.0)


# -------------------------------------------------- observable state changes


class TestObservableEffects:
    """Each axis must visibly change a model field after reset()."""

    def test_lighting_intensity_changes_light_diffuse(self, env: TabletopEnv) -> None:
        env.reset(seed=0)
        baseline = np.array(env._model.light_diffuse[0], dtype=np.float64).copy()
        env.set_perturbation("lighting_intensity", 0.25)
        env.reset(seed=0)
        after = np.array(env._model.light_diffuse[0], dtype=np.float64)
        np.testing.assert_array_equal(after, np.array([0.25, 0.25, 0.25]))
        assert not np.array_equal(after, baseline)

    def test_camera_offset_x_changes_cam_pos(self, env: TabletopEnv) -> None:
        env.reset(seed=0)
        base = np.array(env._model.cam_pos[env._main_cam_id], dtype=np.float64).copy()
        env.set_perturbation("camera_offset_x", 0.1)
        env.reset(seed=0)
        after = np.array(env._model.cam_pos[env._main_cam_id], dtype=np.float64)
        assert after[0] == pytest.approx(base[0] + 0.1)
        assert after[1] == pytest.approx(base[1])
        assert after[2] == pytest.approx(base[2])

    def test_camera_offset_y_changes_cam_pos(self, env: TabletopEnv) -> None:
        env.reset(seed=0)
        base = np.array(env._model.cam_pos[env._main_cam_id], dtype=np.float64).copy()
        env.set_perturbation("camera_offset_y", -0.08)
        env.reset(seed=0)
        after = np.array(env._model.cam_pos[env._main_cam_id], dtype=np.float64)
        assert after[0] == pytest.approx(base[0])
        assert after[1] == pytest.approx(base[1] - 0.08)
        assert after[2] == pytest.approx(base[2])

    def test_object_texture_flips_cube_material(self, env: TabletopEnv) -> None:
        env.reset(seed=0)
        baseline_matid = int(env._model.geom_matid[env._cube_geom_id])
        assert baseline_matid == env._cube_material_default_id

        env.set_perturbation("object_texture", 1.0)
        env.reset(seed=0)
        assert int(env._model.geom_matid[env._cube_geom_id]) == env._cube_material_alt_id

        # The alt material's RGB has more green than red.
        mat_rgba = env._model.mat_rgba[env._cube_material_alt_id]
        assert float(mat_rgba[1]) > float(mat_rgba[0])

        # Flipping back to 0 returns to baseline.
        env.set_perturbation("object_texture", 0.0)
        env.reset(seed=0)
        assert int(env._model.geom_matid[env._cube_geom_id]) == baseline_matid

    def test_object_initial_pose_x_overrides_cube_qpos(self, env: TabletopEnv) -> None:
        env.set_perturbation("object_initial_pose_x", 0.123)
        obs, _ = env.reset(seed=0)
        assert float(obs["cube_pos"][0]) == pytest.approx(0.123, abs=1e-9)

    def test_object_initial_pose_y_overrides_cube_qpos(self, env: TabletopEnv) -> None:
        env.set_perturbation("object_initial_pose_y", -0.087)
        obs, _ = env.reset(seed=0)
        assert float(obs["cube_pos"][1]) == pytest.approx(-0.087, abs=1e-9)

    def test_object_initial_pose_x_and_y_compose(self, env: TabletopEnv) -> None:
        env.set_perturbation("object_initial_pose_x", 0.05)
        env.set_perturbation("object_initial_pose_y", -0.04)
        obs, _ = env.reset(seed=0)
        assert float(obs["cube_pos"][0]) == pytest.approx(0.05, abs=1e-9)
        assert float(obs["cube_pos"][1]) == pytest.approx(-0.04, abs=1e-9)

    def test_distractor_count_enables_first_n(self, env: TabletopEnv) -> None:
        env.set_perturbation("distractor_count", 4)
        env.reset(seed=0)
        for i, gid in enumerate(env._distractor_geom_ids):
            alpha = float(env._model.geom_rgba[gid][3])
            if i < 4:
                assert alpha == pytest.approx(1.0), f"distractor {i} should be visible"
                assert int(env._model.geom_contype[gid]) == 1
                assert int(env._model.geom_conaffinity[gid]) == 1
            else:
                assert alpha == pytest.approx(0.0), f"distractor {i} should be hidden"
                assert int(env._model.geom_contype[gid]) == 0
                assert int(env._model.geom_conaffinity[gid]) == 0


# ------------------------------------------------------------- restoration


class TestRestoreBaseline:
    def test_restore_baseline_undoes_lighting(self, env: TabletopEnv) -> None:
        env.reset(seed=0)
        baseline = np.array(env._model.light_diffuse[0], dtype=np.float64).copy()
        env._model.light_diffuse[0] = np.array([0.1, 0.2, 0.3])
        assert not np.array_equal(env._model.light_diffuse[0], baseline)
        env.restore_baseline()
        np.testing.assert_array_equal(env._model.light_diffuse[0], baseline)

    def test_restore_baseline_undoes_cam_pos(self, env: TabletopEnv) -> None:
        env.reset(seed=0)
        baseline = np.array(env._model.cam_pos[env._main_cam_id], dtype=np.float64).copy()
        env._model.cam_pos[env._main_cam_id] = baseline + np.array([1.0, 1.0, 1.0])
        env.restore_baseline()
        np.testing.assert_array_equal(env._model.cam_pos[env._main_cam_id], baseline)

    def test_restore_baseline_undoes_cube_material(self, env: TabletopEnv) -> None:
        env.reset(seed=0)
        baseline_mat = int(env._model.geom_matid[env._cube_geom_id])
        baseline_rgba = np.array(env._model.geom_rgba[env._cube_geom_id], dtype=np.float64).copy()
        env._model.geom_matid[env._cube_geom_id] = env._cube_material_alt_id
        env._model.geom_rgba[env._cube_geom_id] = np.array([0.0, 1.0, 0.0, 1.0])
        env.restore_baseline()
        assert int(env._model.geom_matid[env._cube_geom_id]) == baseline_mat
        np.testing.assert_array_equal(env._model.geom_rgba[env._cube_geom_id], baseline_rgba)

    def test_restore_baseline_undoes_distractor_state(self, env: TabletopEnv) -> None:
        env.reset(seed=0)
        gid0 = env._distractor_geom_ids[0]
        base_rgba = np.array(env._model.geom_rgba[gid0], dtype=np.float64).copy()
        base_contype = int(env._model.geom_contype[gid0])
        base_conaff = int(env._model.geom_conaffinity[gid0])
        env._model.geom_rgba[gid0] = np.array([1.0, 1.0, 1.0, 1.0])
        env._model.geom_contype[gid0] = 1
        env._model.geom_conaffinity[gid0] = 1
        env.restore_baseline()
        np.testing.assert_array_equal(env._model.geom_rgba[gid0], base_rgba)
        assert int(env._model.geom_contype[gid0]) == base_contype
        assert int(env._model.geom_conaffinity[gid0]) == base_conaff


class TestResetClearsPerturbations:
    """An unperturbed reset() following a perturbed one must return to baseline."""

    def test_lighting_pert_then_clean_reset_restores_baseline(self, env: TabletopEnv) -> None:
        env.reset(seed=0)
        baseline = np.array(env._model.light_diffuse[0], dtype=np.float64).copy()
        env.set_perturbation("lighting_intensity", 0.25)
        env.reset(seed=0)
        assert not np.array_equal(env._model.light_diffuse[0], baseline)
        # Next reset with no new perturbation must wipe the effect.
        env.reset(seed=0)
        np.testing.assert_array_equal(env._model.light_diffuse[0], baseline)

    def test_distractor_pert_then_clean_reset_hides_all(self, env: TabletopEnv) -> None:
        env.set_perturbation("distractor_count", 5)
        env.reset(seed=0)
        visible = [float(env._model.geom_rgba[gid][3]) > 0.5 for gid in env._distractor_geom_ids]
        assert sum(visible) == 5
        env.reset(seed=0)
        visible_after = [
            float(env._model.geom_rgba[gid][3]) > 0.5 for gid in env._distractor_geom_ids
        ]
        assert sum(visible_after) == 0


# --------------------------------------------------------- distractor edge


class TestDistractorCountEdges:
    def test_zero_hides_all(self, env: TabletopEnv) -> None:
        env.set_perturbation("distractor_count", 0)
        env.reset(seed=0)
        for gid in env._distractor_geom_ids:
            assert float(env._model.geom_rgba[gid][3]) == pytest.approx(0.0)

    def test_max_enables_all(self, env: TabletopEnv) -> None:
        env.set_perturbation("distractor_count", N_DISTRACTOR_SLOTS)
        env.reset(seed=0)
        for gid in env._distractor_geom_ids:
            assert float(env._model.geom_rgba[gid][3]) == pytest.approx(1.0)
            assert int(env._model.geom_contype[gid]) == 1
            assert int(env._model.geom_conaffinity[gid]) == 1

    def test_overflow_raises(self, env: TabletopEnv) -> None:
        with pytest.raises(ValueError, match="distractor_count must be in"):
            env.set_perturbation("distractor_count", N_DISTRACTOR_SLOTS + 1)

    def test_negative_raises(self, env: TabletopEnv) -> None:
        with pytest.raises(ValueError, match="distractor_count must be in"):
            env.set_perturbation("distractor_count", -1)


# ----------------------------------------------------------- determinism


class TestDeterminism:
    """Spec §6 hard rule extended to perturbed envs."""

    @staticmethod
    def _rollout(
        env: TabletopEnv,
        seed: int,
        perts: dict[str, float],
        actions: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        for name, value in perts.items():
            env.set_perturbation(name, value)
        obs, _ = env.reset(seed=seed)
        trace = [obs]
        for action in actions:
            obs, _, _, _, _ = env.step(action)
            trace.append(obs)
        return trace

    def test_perturbed_envs_with_same_inputs_produce_bit_identical_obs(self) -> None:
        action_rng = np.random.default_rng(2026)
        actions = action_rng.uniform(-1.0, 1.0, size=(15, 7)).astype(np.float64)
        perts = {
            "lighting_intensity": 0.7,
            "camera_offset_x": 0.04,
            "camera_offset_y": -0.03,
            "object_texture": 1.0,
            # Note: ``initial_state_ood`` is mutually exclusive with the
            # explicit ``object_initial_pose_*`` axes since both write
            # the cube qpos. The OOD axis is exercised in detail in
            # ``tests/test_perturbation_initial_state_ood.py``.
            "object_initial_pose_x": 0.08,
            "object_initial_pose_y": -0.06,
            "distractor_count": 4,
        }

        env_a = TabletopEnv()
        env_b = TabletopEnv()
        try:
            trace_a = self._rollout(env_a, seed=99, perts=perts, actions=actions)
            trace_b = self._rollout(env_b, seed=99, perts=perts, actions=actions)
        finally:
            env_a.close()
            env_b.close()

        assert len(trace_a) == len(trace_b)
        for i, (oa, ob) in enumerate(zip(trace_a, trace_b, strict=True)):
            for key in oa:
                np.testing.assert_array_equal(
                    oa[key], ob[key], err_msg=f"diverged at step {i}, key {key}"
                )

    def test_same_env_two_perturbed_resets_with_same_inputs_match(self, env: TabletopEnv) -> None:
        """Catches state leaks between consecutive episodes on one env."""
        env.set_perturbation("lighting_intensity", 0.4)
        env.set_perturbation("object_initial_pose_x", 0.1)
        obs1, _ = env.reset(seed=7)
        for _ in range(4):
            env.step(_zero_action())
        env.set_perturbation("lighting_intensity", 0.4)
        env.set_perturbation("object_initial_pose_x", 0.1)
        obs2, _ = env.reset(seed=7)
        for key in obs1:
            np.testing.assert_array_equal(obs1[key], obs2[key])


# ------------------------------------------------------- sampler reproduce


class TestSamplerReproducibility:
    @pytest.mark.parametrize("name", AXIS_NAMES)
    def test_same_seed_same_samples(self, name: str) -> None:
        axis = axis_for(name)
        rng_a = np.random.default_rng(2026)
        rng_b = np.random.default_rng(2026)
        samples_a = [axis.sample(rng_a) for _ in range(20)]
        samples_b = [axis.sample(rng_b) for _ in range(20)]
        assert samples_a == samples_b

    def test_different_seeds_likely_diverge(self) -> None:
        # Continuous axis; categorical/int may collide by chance.
        axis = lighting_intensity()
        s_a = [axis.sample(np.random.default_rng(1)) for _ in range(5)]
        s_b = [axis.sample(np.random.default_rng(2)) for _ in range(5)]
        assert s_a != s_b

    @pytest.mark.parametrize("name", AXIS_NAMES)
    def test_samples_lie_within_axis_bounds(self, name: str) -> None:
        axis = axis_for(name)
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = axis.sample(rng)
            assert axis.low - 1e-9 <= v <= axis.high + 1e-9

    def test_object_texture_only_yields_zero_or_one(self) -> None:
        axis = object_texture()
        rng = np.random.default_rng(2026)
        seen = {axis.sample(rng) for _ in range(50)}
        assert seen.issubset({0.0, 1.0})

    def test_distractor_count_only_yields_integer_values(self) -> None:
        axis = distractor_count()
        rng = np.random.default_rng(2026)
        for _ in range(50):
            v = axis.sample(rng)
            assert v == float(int(v))
            assert 0 <= int(v) <= N_DISTRACTOR_SLOTS


# ----------------------------------------------------------- sampler factories
# Phase 2.5 Task 11 — direct unit tests for the low-level make_*_sampler
# error / int-rounding paths in ``env.perturbation.base``.


class TestSamplerFactories:
    """Direct tests for the make_continuous / make_int / make_categorical
    factories. The high-level axis constructors (``lighting_intensity`` etc.)
    already exercise the happy paths — this fills in the inverted-bound
    rejections and the ``int`` rounding behaviour."""

    def test_make_continuous_sampler_rejects_inverted_bounds(self) -> None:
        from gauntlet.env.perturbation.base import make_continuous_sampler

        with pytest.raises(ValueError, match="low must be <= high"):
            make_continuous_sampler(low=2.0, high=1.0)

    def test_make_continuous_sampler_returns_value_in_range(self) -> None:
        from gauntlet.env.perturbation.base import make_continuous_sampler

        sampler = make_continuous_sampler(low=-0.5, high=0.5)
        rng = np.random.default_rng(0)
        for _ in range(30):
            v = sampler(rng)
            assert -0.5 <= v <= 0.5

    def test_make_int_sampler_rejects_inverted_bounds(self) -> None:
        from gauntlet.env.perturbation.base import make_int_sampler

        with pytest.raises(ValueError, match="low must be <= high"):
            make_int_sampler(low=10, high=5)

    def test_make_int_sampler_returns_floats_with_integer_values(self) -> None:
        from gauntlet.env.perturbation.base import make_int_sampler

        sampler = make_int_sampler(low=0, high=3)
        rng = np.random.default_rng(0)
        for _ in range(40):
            v = sampler(rng)
            assert v == float(int(v))
            assert 0 <= int(v) <= 3

    def test_make_int_sampler_inclusive_high(self) -> None:
        from gauntlet.env.perturbation.base import make_int_sampler

        sampler = make_int_sampler(low=0, high=2)
        rng = np.random.default_rng(0)
        seen = {int(sampler(rng)) for _ in range(200)}
        # All three values reachable; high is inclusive.
        assert seen == {0, 1, 2}

    def test_make_categorical_sampler_rejects_empty(self) -> None:
        from gauntlet.env.perturbation.base import make_categorical_sampler

        with pytest.raises(ValueError, match="at least one"):
            make_categorical_sampler(())

    def test_make_categorical_sampler_returns_one_of_choices(self) -> None:
        from gauntlet.env.perturbation.base import make_categorical_sampler

        sampler = make_categorical_sampler((1.5, 2.5, 3.5))
        rng = np.random.default_rng(0)
        seen = {sampler(rng) for _ in range(60)}
        assert seen.issubset({1.5, 2.5, 3.5})

    def test_perturbation_axis_sample_delegates_to_sampler(self) -> None:
        """The ``PerturbationAxis.sample`` convenience wrapper just calls
        the bound sampler — verify with a tiny custom sampler that
        returns a constant."""
        from gauntlet.env.perturbation.base import (
            AXIS_KIND_CONTINUOUS,
            PerturbationAxis,
        )

        def _const_sampler(rng: np.random.Generator) -> float:
            return 7.0

        axis = PerturbationAxis(
            name="x",
            kind=AXIS_KIND_CONTINUOUS,
            sampler=_const_sampler,
            low=0.0,
            high=10.0,
        )
        assert axis.sample(np.random.default_rng(0)) == 7.0
