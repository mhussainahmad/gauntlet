"""Tests for the ``camera_extrinsics`` render-camera perturbation axis (B-42).

The axis applies a structured 6-D pose delta
``{translation: [dx, dy, dz], rotation: [drx, dry, drz]}`` (radians) to
the env's render camera at episode start. Operates on the env's
*render* camera — not on the policy's internal camera state — so VLA
adapters that read ``obs["image"]`` see the perturbed view.

Coverage:

* axis registration in the canonical AXIS_NAMES tuple,
* the YAML loader accepts the enumerated and Sobol-friendly continuous
  range shapes, including the schema rejections for misuse on other
  axes / cartesian sampling,
* MuJoCo cam_pos / cam_quat mutation under translation and rotation,
  including baseline restoration across resets,
* index-range validation at ``set_perturbation`` queue time,
* backend-restriction warning on Genesis / Isaac (defensive: the suite
  loader rejects mixing at YAML-load time but a direct caller bypass
  still gets a loud warning),
* SO(3) Sobol warning emitted by
  :func:`gauntlet.report.sobol_indices.compute_sobol_indices` when the
  axis is present in the episode set,
* determinism of the range-form pre-resolution (Sobol is deterministic
  so the same YAML resolves to bit-identical entries across loads).

These tests run headless (no GL): no ``env.render()`` calls.

References: RoboView-Bias (arXiv 2509.22356), "Do You Know Where Your
Camera Is?" (arXiv 2510.02268), LIBERO-Plus (arXiv 2510.13626).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
import pytest
from pydantic import ValidationError

from gauntlet.env.perturbation import (
    AXIS_KIND_CATEGORICAL,
    AXIS_NAMES,
    PerturbationAxis,
    axis_for,
    camera_extrinsics,
)
from gauntlet.env.tabletop import TabletopEnv
from gauntlet.report.sobol_indices import (
    CAMERA_EXTRINSICS_SO3_WARNING,
    compute_sobol_indices,
)
from gauntlet.runner.episode import Episode
from gauntlet.suite.loader import load_suite_from_string


@pytest.fixture
def env() -> Iterator[TabletopEnv]:
    """Fresh TabletopEnv per test, closed on teardown."""
    e = TabletopEnv()
    try:
        yield e
    finally:
        e.close()


# --------------------------------------------------------------- registry


class TestCameraExtrinsicsAxisRegistration:
    def test_axis_in_canonical_registry(self) -> None:
        assert "camera_extrinsics" in AXIS_NAMES

    def test_axis_for_returns_categorical_axis(self) -> None:
        axis = axis_for("camera_extrinsics")
        assert isinstance(axis, PerturbationAxis)
        assert axis.name == "camera_extrinsics"
        assert axis.kind == AXIS_KIND_CATEGORICAL

    def test_default_constructor_returns_placeholder(self) -> None:
        # The default registry has 1 entry (index 0 baseline) so the
        # axis can be constructed without a YAML; real suites override
        # via the env's setter.
        axis = camera_extrinsics()
        assert axis.name == "camera_extrinsics"
        assert axis.kind == AXIS_KIND_CATEGORICAL


# ---------------------------------------------------------- yaml loader


class TestSuiteYamlAcceptsExtrinsicsValues:
    def test_yaml_loads_enumerated_extrinsics_values(self) -> None:
        yaml_text = (
            "name: extr-suite\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "axes:\n"
            "  camera_extrinsics:\n"
            "    extrinsics_values:\n"
            "      - {translation: [0.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0]}\n"
            "      - {translation: [0.05, 0.0, 0.0], rotation: [0.0, 0.0, 0.0]}\n"
            "      - {translation: [0.0, 0.0, 0.0], rotation: [0.1, 0.0, 0.0]}\n"
        )
        suite = load_suite_from_string(yaml_text)
        spec = suite.axes["camera_extrinsics"]
        # Each entry enumerates as its index; the env resolves the
        # index back to the structured 6-tuple at apply time.
        assert spec.enumerate() == (0.0, 1.0, 2.0)
        cells = list(suite.cells())
        assert [c.values["camera_extrinsics"] for c in cells] == [0.0, 1.0, 2.0]
        # The structured payload survives the schema layer.
        entries = spec.extrinsics_entries()
        assert entries is not None
        assert len(entries) == 3
        assert entries[0].translation == [0.0, 0.0, 0.0]
        assert entries[1].translation == [0.05, 0.0, 0.0]
        assert entries[2].rotation == [0.1, 0.0, 0.0]

    def test_yaml_loads_extrinsics_range_with_sobol(self) -> None:
        yaml_text = (
            "name: extr-range\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "sampling: sobol\n"
            "n_samples: 8\n"
            "axes:\n"
            "  camera_extrinsics:\n"
            "    extrinsics_range:\n"
            "      translation: [[-0.1, 0.1], [-0.1, 0.1], [-0.05, 0.05]]\n"
            "      rotation: [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]\n"
        )
        suite = load_suite_from_string(yaml_text)
        spec = suite.axes["camera_extrinsics"]
        # Range form is pre-expanded into N enumerated entries via 6-D
        # Joe-Kuo Sobol at suite-load time.
        entries = spec.extrinsics_entries()
        assert entries is not None
        assert len(entries) == 8
        # Every translation entry must lie in the declared bounds.
        for e in entries:
            assert -0.1 <= e.translation[0] <= 0.1
            assert -0.1 <= e.translation[1] <= 0.1
            assert -0.05 <= e.translation[2] <= 0.05
            for r in e.rotation:
                assert -0.1 <= r <= 0.1

    def test_range_resolution_is_deterministic(self) -> None:
        yaml_text = (
            "name: extr-range-det\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "sampling: sobol\n"
            "n_samples: 4\n"
            "axes:\n"
            "  camera_extrinsics:\n"
            "    extrinsics_range:\n"
            "      translation: [[-0.1, 0.1], [-0.1, 0.1], [-0.05, 0.05]]\n"
            "      rotation: [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]\n"
        )
        # Sobol is fully deterministic — the same YAML must resolve to
        # bit-identical entries across loads regardless of suite seed.
        s1 = load_suite_from_string(yaml_text)
        s2 = load_suite_from_string(yaml_text)
        e1 = s1.axes["camera_extrinsics"].extrinsics_entries()
        e2 = s2.axes["camera_extrinsics"].extrinsics_entries()
        assert e1 is not None
        assert e2 is not None
        assert len(e1) == len(e2)
        for a, b in zip(e1, e2, strict=True):
            assert a.translation == b.translation
            assert a.rotation == b.rotation


class TestSuiteYamlRejectsMisuse:
    def test_extrinsics_values_on_other_axis_rejected(self) -> None:
        yaml_text = (
            "name: bad\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "axes:\n"
            "  lighting_intensity:\n"
            "    extrinsics_values:\n"
            "      - {translation: [0.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0]}\n"
        )
        with pytest.raises(ValidationError, match="camera_extrinsics"):
            load_suite_from_string(yaml_text)

    def test_extrinsics_range_on_cartesian_rejected(self) -> None:
        yaml_text = (
            "name: bad\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "axes:\n"
            "  camera_extrinsics:\n"
            "    extrinsics_range:\n"
            "      translation: [[-0.1, 0.1], [-0.1, 0.1], [-0.05, 0.05]]\n"
            "      rotation: [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]\n"
        )
        # Cartesian has no n_samples budget; the range form is
        # exclusive to LHS / Sobol / adversarial.
        with pytest.raises(ValidationError, match="extrinsics_range"):
            load_suite_from_string(yaml_text)

    def test_mixing_values_and_range_rejected(self) -> None:
        yaml_text = (
            "name: bad\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "sampling: sobol\n"
            "n_samples: 4\n"
            "axes:\n"
            "  camera_extrinsics:\n"
            "    extrinsics_values:\n"
            "      - {translation: [0.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0]}\n"
            "    extrinsics_range:\n"
            "      translation: [[-0.1, 0.1], [-0.1, 0.1], [-0.05, 0.05]]\n"
            "      rotation: [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]\n"
        )
        with pytest.raises(ValidationError, match="cannot mix"):
            load_suite_from_string(yaml_text)

    def test_translation_wrong_length_rejected(self) -> None:
        yaml_text = (
            "name: bad\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "axes:\n"
            "  camera_extrinsics:\n"
            "    extrinsics_values:\n"
            "      - {translation: [0.0, 0.0], rotation: [0.0, 0.0, 0.0]}\n"
        )
        with pytest.raises(ValidationError, match="length 3"):
            load_suite_from_string(yaml_text)


# --------------------------------------------------------- mujoco apply


class TestMuJoCoApply:
    def test_set_camera_extrinsics_list_rebinds(self, env: TabletopEnv) -> None:
        # Default registry has 1 entry (baseline).
        assert len(env.camera_extrinsics_list) == 1
        env.set_camera_extrinsics_list(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.05, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        )
        assert len(env.camera_extrinsics_list) == 2
        # Sanitised to plain floats.
        assert env.camera_extrinsics_list[1] == (0.05, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_set_camera_extrinsics_list_rejects_empty(self, env: TabletopEnv) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            env.set_camera_extrinsics_list(())

    def test_set_camera_extrinsics_list_rejects_wrong_arity(self, env: TabletopEnv) -> None:
        with pytest.raises(ValueError, match="6-tuple"):
            env.set_camera_extrinsics_list(
                ((0.0, 0.0, 0.0),),  # type: ignore[arg-type]
            )

    def test_set_perturbation_index_out_of_range_rejected(self, env: TabletopEnv) -> None:
        # Default registry has 1 entry; index 1 is out of range.
        with pytest.raises(ValueError, match="out of range"):
            env.set_perturbation("camera_extrinsics", 1.0)

    def test_translation_mutates_cam_pos(self, env: TabletopEnv) -> None:
        env.set_camera_extrinsics_list(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.07, -0.03, 0.02, 0.0, 0.0, 0.0),
            )
        )
        cam = env._main_cam_id
        baseline_pos = np.array(env._baseline["cam_pos_main"], dtype=np.float64).copy()
        env.set_perturbation("camera_extrinsics", 1.0)
        env.reset(seed=42)
        # Translation is added to the baseline; cam_quat stays at baseline.
        np.testing.assert_allclose(
            env._model.cam_pos[cam],
            baseline_pos + np.array([0.07, -0.03, 0.02]),
        )
        np.testing.assert_allclose(
            env._model.cam_quat[cam],
            env._baseline["cam_quat_main"],
        )

    def test_rotation_mutates_cam_quat(self, env: TabletopEnv) -> None:
        env.set_camera_extrinsics_list(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.1, 0.0, 0.0),
            )
        )
        cam = env._main_cam_id
        baseline_pos = np.array(env._baseline["cam_pos_main"], dtype=np.float64).copy()
        baseline_quat = np.array(env._baseline["cam_quat_main"], dtype=np.float64).copy()
        env.set_perturbation("camera_extrinsics", 1.0)
        env.reset(seed=42)
        # Pure rotation leaves cam_pos at baseline; cam_quat is the
        # baseline composed with the delta.
        np.testing.assert_allclose(env._model.cam_pos[cam], baseline_pos)
        # The new quat must be unit-norm (we explicitly normalise).
        n = float(np.linalg.norm(env._model.cam_quat[cam]))
        assert abs(n - 1.0) < 1e-9
        # And it must differ from the baseline (the delta rotation is
        # non-zero).
        assert not np.allclose(env._model.cam_quat[cam], baseline_quat)

    def test_baseline_restored_across_resets(self, env: TabletopEnv) -> None:
        # B-42 — the most load-bearing test: queue a non-zero
        # extrinsics, reset, then queue the baseline, reset again, and
        # assert cam_pos / cam_quat are bit-identical to the original
        # baseline. Exercises both the snapshot capture (cam_quat
        # baseline) and the restore_baseline copy-back.
        env.set_camera_extrinsics_list(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.05, 0.05, 0.05, 0.1, 0.1, 0.1),
            )
        )
        cam = env._main_cam_id
        baseline_pos = np.array(env._baseline["cam_pos_main"], dtype=np.float64).copy()
        baseline_quat = np.array(env._baseline["cam_quat_main"], dtype=np.float64).copy()

        # Apply non-zero extrinsics first.
        env.set_perturbation("camera_extrinsics", 1.0)
        env.reset(seed=42)
        assert not np.allclose(env._model.cam_pos[cam], baseline_pos)
        assert not np.allclose(env._model.cam_quat[cam], baseline_quat)

        # Then queue the baseline (index 0). The reset path must
        # restore cam_pos / cam_quat to their original baseline before
        # applying the (no-op) baseline delta.
        env.set_perturbation("camera_extrinsics", 0.0)
        env.reset(seed=42)
        np.testing.assert_allclose(env._model.cam_pos[cam], baseline_pos)
        np.testing.assert_allclose(env._model.cam_quat[cam], baseline_quat)

    def test_apply_branch_validates_index(self, env: TabletopEnv) -> None:
        # Defence in depth: even if the validation queue gets bypassed,
        # the apply branch refuses out-of-range indices.
        # Direct injection of a bad value into _pending_perturbations
        # bypasses set_perturbation's check.
        env._pending_perturbations["camera_extrinsics"] = 99.0
        with pytest.raises(ValueError, match="out of range"):
            env.reset(seed=42)


# --------------------------------------------------- backend restriction


class TestBackendRestrictionWarning:
    """Genesis + Isaac declare the axis in VISUAL_ONLY_AXES (suite loader
    rejects mixing at YAML time) AND emit a defensive UserWarning if a
    direct caller queues the axis via set_perturbation. The Genesis /
    Isaac envs require their respective extras to instantiate; tests
    that touch them are skipped without the extra.
    """

    def test_genesis_emits_runtime_warning(self) -> None:
        pytest.importorskip("genesis")
        from gauntlet.env.genesis.tabletop_genesis import GenesisTabletopEnv

        e = GenesisTabletopEnv()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                e.set_perturbation("camera_extrinsics", 0.0)
            messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            assert any("not yet supported on the Genesis" in m for m in messages)
        finally:
            e.close()

    def test_isaac_emits_runtime_warning(self) -> None:
        pytest.importorskip("isaacsim")
        from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

        e = IsaacSimTabletopEnv()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                e.set_perturbation("camera_extrinsics", 0.0)
            messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            assert any("not yet supported on the Isaac" in m for m in messages)
        finally:
            e.close()

    def test_genesis_declares_axis_in_visual_only(self) -> None:
        pytest.importorskip("genesis")
        from gauntlet.env.genesis.tabletop_genesis import GenesisTabletopEnv

        assert "camera_extrinsics" in GenesisTabletopEnv.VISUAL_ONLY_AXES
        assert "camera_extrinsics" in GenesisTabletopEnv.AXIS_NAMES

    def test_isaac_declares_axis_in_visual_only(self) -> None:
        pytest.importorskip("isaacsim")
        from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

        assert "camera_extrinsics" in IsaacSimTabletopEnv.VISUAL_ONLY_AXES
        assert "camera_extrinsics" in IsaacSimTabletopEnv.AXIS_NAMES


# ----------------------------------------------- pybullet (real impl)


class TestPyBulletApply:
    """PyBullet ships a real impl on the single-cam render path; the
    apply branch stores the index on a shadow attribute that
    ``_render_obs_image`` consumes via ``_apply_extrinsics_to_basis``.
    Tests are gated on the ``pybullet`` extra.
    """

    def test_pybullet_declares_axis_outside_visual_only(self) -> None:
        pytest.importorskip("pybullet")
        from gauntlet.env.pybullet.tabletop_pybullet import PyBulletTabletopEnv

        assert "camera_extrinsics" in PyBulletTabletopEnv.AXIS_NAMES
        assert "camera_extrinsics" not in PyBulletTabletopEnv.VISUAL_ONLY_AXES

    def test_pybullet_render_observes_extrinsics_delta(self) -> None:
        pytest.importorskip("pybullet")
        from gauntlet.env.pybullet.tabletop_pybullet import PyBulletTabletopEnv

        e = PyBulletTabletopEnv(render_in_obs=True)
        try:
            e.set_camera_extrinsics_list(
                (
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (0.05, 0.0, 0.0, 0.0, 0.0, 0.0),
                )
            )
            e.set_perturbation("camera_extrinsics", 0.0)
            obs0, _ = e.reset(seed=42)
            img0 = obs0["image"].copy()

            e.set_perturbation("camera_extrinsics", 1.0)
            obs1, _ = e.reset(seed=42)
            img1 = obs1["image"]

            # Translation is observable on the rendered frame.
            assert not np.array_equal(img0, img1)
        finally:
            e.close()


# ----------------------------------------------------- sobol so(3) warning


class TestSobolSO3Warning:
    def _make_episode(
        self,
        cell_index: int,
        success: bool,
        config: dict[str, float],
    ) -> Episode:
        # Mirrors the helper in tests/test_sobol_indices.py — only the
        # fields compute_sobol_indices reads (success,
        # perturbation_config) carry semantic weight; the rest are
        # sentinel values that satisfy the pydantic ``extra="forbid"``
        # schema.
        return Episode(
            suite_name="extr-test",
            cell_index=cell_index,
            episode_index=0,
            seed=0,
            perturbation_config=dict(config),
            success=success,
            terminated=success,
            truncated=False,
            step_count=0,
            total_reward=1.0 if success else 0.0,
        )

    def test_warning_emitted_when_axis_present(self) -> None:
        # Two-bucket camera_extrinsics axis, both buckets show non-zero
        # variance so the closed-form decomposition is well-defined.
        episodes = [
            self._make_episode(0, True, {"camera_extrinsics": 0.0}),
            self._make_episode(1, False, {"camera_extrinsics": 0.0}),
            self._make_episode(2, False, {"camera_extrinsics": 1.0}),
            self._make_episode(3, True, {"camera_extrinsics": 1.0}),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = compute_sobol_indices(episodes, ("camera_extrinsics",))
        assert "camera_extrinsics" in out
        # Exactly one SO(3) warning was emitted.
        so3_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "SO(3)" in str(w.message)
        ]
        assert len(so3_warnings) == 1
        # The full canonical warning message is surfaced verbatim so
        # downstream consumers (HTML report) can render it.
        assert CAMERA_EXTRINSICS_SO3_WARNING in str(so3_warnings[0].message)

    def test_warning_not_emitted_when_axis_absent(self) -> None:
        episodes = [
            self._make_episode(0, True, {"lighting_intensity": 0.5}),
            self._make_episode(1, False, {"lighting_intensity": 1.0}),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_sobol_indices(episodes, ("lighting_intensity",))
        # No SO(3) warning when the axis is not in axis_names.
        so3_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "SO(3)" in str(w.message)
        ]
        assert len(so3_warnings) == 0


# ------------------------------------------------ helper basis math (unit)


class TestExtrinsicsBasisHelper:
    """The PyBullet helper ``_apply_extrinsics_to_basis`` is pure numpy
    (no PyBullet calls inside) but lives in a module that imports
    pybullet at top-level. Skip locally when pybullet is missing.
    """

    def test_zero_delta_returns_input(self) -> None:
        pytest.importorskip("pybullet")
        from gauntlet.env.pybullet.tabletop_pybullet import _apply_extrinsics_to_basis

        eye, target, up = (0.6, -0.6, 0.8), (0.0, 0.0, 0.42), (0.0, 0.0, 1.0)
        e1, t1, u1 = _apply_extrinsics_to_basis(eye, target, up, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(e1, eye)
        np.testing.assert_allclose(t1, target)
        np.testing.assert_allclose(u1, up)

    def test_pure_translation_shifts_eye(self) -> None:
        pytest.importorskip("pybullet")
        from gauntlet.env.pybullet.tabletop_pybullet import _apply_extrinsics_to_basis

        eye, target, up = (0.6, -0.6, 0.8), (0.0, 0.0, 0.42), (0.0, 0.0, 1.0)
        e1, t1, u1 = _apply_extrinsics_to_basis(
            eye,
            target,
            up,
            (0.05, -0.03, 0.0, 0.0, 0.0, 0.0),
        )
        # Eye shifts by the translation; up stays unchanged.
        np.testing.assert_allclose(e1, (0.65, -0.63, 0.8))
        np.testing.assert_allclose(u1, up)
        # Target also shifts by the same delta (rigid translation —
        # camera mount moved, look direction unchanged).
        np.testing.assert_allclose(t1, (0.05, -0.03, 0.42))


# ------------------------------------------------- private API existence


def test_quat_helpers_round_trip() -> None:
    """The MuJoCo apply branch composes ``baseline_quat * delta_quat``;
    a zero delta must produce the baseline itself (load-bearing for
    the index-0 baseline-preserving path).
    """
    from gauntlet.env.tabletop import _quat_from_xyz_euler, _quat_mul_wxyz

    # Zero rotation → identity quat.
    q = _quat_from_xyz_euler(0.0, 0.0, 0.0)
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0])

    # Multiplying by identity is a no-op.
    a = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    out = _quat_mul_wxyz(a, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(out, a)
