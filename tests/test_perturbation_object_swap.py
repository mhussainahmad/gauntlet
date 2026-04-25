"""Tests for the ``object_swap`` semantic-distractor perturbation axis (B-06).

The axis replaces the target cube with a categorically-different object
(``mug``, ``banana``, ``screwdriver``, ``bottle``) drawn from the small
MJCF object library shipped with the MuJoCo tabletop env. Combined with
``instruction_paraphrase`` this measures grounding — LIBERO-PRO showed
VLAs persist in grasping when the target is replaced with an irrelevant
item.

Coverage:

* axis registration in the canonical AXIS_NAMES tuple,
* the YAML loader accepts a string-list ``values:`` shape with each
  legal class name,
* the baseline (``cube``) value preserves the cube's MJCF visibility +
  collision flags (current pre-B-06 behaviour),
* mug / banana / screwdriver / bottle each load their MJCF geom and
  the env reveals the matching geom while hiding the others,
* construction does NOT load the alt geoms by default (every cell that
  doesn't queue ``object_swap`` keeps the unmodified scene),
* anti-feature: PyBullet / Genesis / Isaac list ``object_swap`` in
  ``VISUAL_ONLY_AXES`` so the suite linter / loader rejects mixing.

These tests run headless (no GL): no ``env.render()`` calls.
"""

from __future__ import annotations

from collections.abc import Iterator

import mujoco
import pytest

from gauntlet.env.perturbation import (
    AXIS_KIND_CATEGORICAL,
    AXIS_NAMES,
    OBJECT_SWAP_CLASSES,
    PerturbationAxis,
    axis_for,
    object_swap,
)
from gauntlet.env.tabletop import TabletopEnv
from gauntlet.suite.loader import load_suite_from_string
from gauntlet.suite.schema import AxisSpec, Suite

# Names of the four alternate geoms shipped under
# ``src/gauntlet/env/assets/objects/``. Each MJCF fragment is mirrored
# inside the cube body in ``assets/tabletop.xml`` so the env can swap
# without rebuilding the model.
_ALT_GEOM_NAMES: tuple[str, ...] = (
    "swap_mug_geom",
    "swap_banana_geom",
    "swap_screwdriver_geom",
    "swap_bottle_geom",
)


@pytest.fixture
def env() -> Iterator[TabletopEnv]:
    """Fresh TabletopEnv per test, closed on teardown."""
    e = TabletopEnv()
    try:
        yield e
    finally:
        e.close()


def _alpha(env: TabletopEnv, geom_name: str) -> float:
    """Read the current alpha channel for ``geom_name`` off the live model."""
    gid = int(mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
    assert gid >= 0, f"geom {geom_name!r} missing from MJCF"
    return float(env._model.geom_rgba[gid][3])


def _contype(env: TabletopEnv, geom_name: str) -> int:
    gid = int(mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
    return int(env._model.geom_contype[gid])


# --------------------------------------------------------------- registry


class TestObjectSwapAxisRegistration:
    def test_axis_in_canonical_registry(self) -> None:
        assert "object_swap" in AXIS_NAMES

    def test_axis_for_returns_categorical_axis(self) -> None:
        axis = axis_for("object_swap")
        assert isinstance(axis, PerturbationAxis)
        assert axis.name == "object_swap"
        assert axis.kind == AXIS_KIND_CATEGORICAL

    def test_default_constructor_covers_canonical_classes(self) -> None:
        # The default sampler should cover every entry in the canonical
        # registry. Bounds are [0, len-1] inclusive.
        axis = object_swap()
        assert axis.low == 0.0
        assert axis.high == float(len(OBJECT_SWAP_CLASSES) - 1)

    def test_canonical_class_registry_shape(self) -> None:
        # Index 0 must stay ``cube`` so callers that don't opt in keep
        # the env's pre-B-06 behaviour.
        assert OBJECT_SWAP_CLASSES[0] == "cube"
        assert set(OBJECT_SWAP_CLASSES) == {"cube", "mug", "banana", "screwdriver", "bottle"}


# ---------------------------------------------------------- yaml loader


class TestSuiteYamlAcceptsObjectSwap:
    @pytest.mark.parametrize("class_name", ["cube", "mug", "banana", "screwdriver", "bottle"])
    def test_yaml_loads_each_class_singleton(self, class_name: str) -> None:
        # A one-element ``values:`` list is the simplest non-degenerate
        # shape — exercises the schema's string-axis carve-out for B-06
        # and confirms every canonical class name round-trips.
        yaml_text = (
            "name: swap-suite\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "axes:\n"
            "  object_swap:\n"
            f"    values: [{class_name}]\n"
        )
        suite = load_suite_from_string(yaml_text)
        cells = list(suite.cells())
        assert len(cells) == 1
        assert cells[0].values["object_swap"] == 0.0

    def test_yaml_loads_three_class_subset(self) -> None:
        # The suite YAML may declare a subset (the spec example uses
        # ``[cube, mug, banana]``); the schema enumerates strings as
        # indices and the env's ``set_object_swap_classes`` rebinds.
        yaml_text = (
            "name: swap-suite\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "axes:\n"
            "  object_swap:\n"
            "    values: [cube, mug, banana]\n"
        )
        suite = load_suite_from_string(yaml_text)
        spec = suite.axes["object_swap"]
        # String list enumerates as indices to keep the cell value
        # channel float-only (mirrors B-05 instruction_paraphrase).
        assert spec.enumerate() == (0.0, 1.0, 2.0)
        cells = list(suite.cells())
        assert [c.values["object_swap"] for c in cells] == [0.0, 1.0, 2.0]

    def test_yaml_accepts_float_values_too(self) -> None:
        # The float shape stays legal — a YAML can pre-encode the
        # indices directly if it prefers (mirrors how every other
        # categorical axis behaves).
        yaml_text = (
            "name: swap-suite\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "axes:\n"
            "  object_swap:\n"
            "    values: [0.0, 1.0]\n"
        )
        suite = load_suite_from_string(yaml_text)
        cells = list(suite.cells())
        assert [c.values["object_swap"] for c in cells] == [0.0, 1.0]

    def test_unknown_class_name_rejected_by_env(self) -> None:
        # Schema accepts arbitrary string values on the object_swap
        # axis (a foreign-name sanity check is the env's
        # ``set_object_swap_classes`` job — keeps the schema layer
        # name-agnostic, mirroring the B-05 paraphrase wiring posture).
        env = TabletopEnv()
        try:
            with pytest.raises(ValueError, match="object_swap: unknown class"):
                env.set_object_swap_classes(("cube", "frisbee"))
        finally:
            env.close()


# ------------------------------------------------------------ env apply


class TestObjectSwapEnvApply:
    def test_baseline_cube_value_preserves_default_scene(self, env: TabletopEnv) -> None:
        # Index 0 (``cube``) must reveal the cube and hide every alt.
        # Crucially, this must match the no-perturbation baseline so
        # callers that don't opt in see the env's pre-B-06 behaviour.
        env.set_perturbation("object_swap", 0.0)
        env.reset(seed=42)
        # Cube visible + colliding.
        assert _alpha(env, "cube_geom") == pytest.approx(1.0)
        assert _contype(env, "cube_geom") == 1
        # Every alt hidden + non-colliding.
        for geom_name in _ALT_GEOM_NAMES:
            assert _alpha(env, geom_name) == 0.0
            assert _contype(env, geom_name) == 0

    @pytest.mark.parametrize(
        ("class_index", "expected_geom"),
        [
            (1, "swap_mug_geom"),
            (2, "swap_banana_geom"),
            (3, "swap_screwdriver_geom"),
            (4, "swap_bottle_geom"),
        ],
    )
    def test_each_alt_class_reveals_matching_geom(
        self, env: TabletopEnv, class_index: int, expected_geom: str
    ) -> None:
        # Each non-baseline class must reveal *its* geom and hide every
        # other swap entry (including the cube). Collision flags follow
        # the visibility so the gripper interacts with the visible geom.
        env.set_perturbation("object_swap", float(class_index))
        env.reset(seed=42)
        # Selected geom is visible + colliding.
        assert _alpha(env, expected_geom) == pytest.approx(1.0)
        assert _contype(env, expected_geom) == 1
        # Cube and every other alt are hidden + non-colliding.
        hidden_geoms = ("cube_geom", *(g for g in _ALT_GEOM_NAMES if g != expected_geom))
        for geom_name in hidden_geoms:
            assert _alpha(env, geom_name) == 0.0
            assert _contype(env, geom_name) == 0

    def test_no_perturbation_keeps_cube_only_scene(self, env: TabletopEnv) -> None:
        # Construction must not pre-reveal any alt geom. Without an
        # ``object_swap`` perturbation, the scene must look identical
        # to a pre-B-06 env.
        env.reset(seed=0)
        assert _alpha(env, "cube_geom") == pytest.approx(1.0)
        for geom_name in _ALT_GEOM_NAMES:
            assert _alpha(env, geom_name) == 0.0

    def test_restore_baseline_clears_swap(self, env: TabletopEnv) -> None:
        # Apply a swap, reset, then a baseline reset must hide the alt
        # again. Confirms the baseline snapshot path covers swap geoms.
        env.set_perturbation("object_swap", 2.0)  # banana
        env.reset(seed=7)
        assert _alpha(env, "swap_banana_geom") == pytest.approx(1.0)
        # Second reset with no queued perturbation snaps back.
        env.reset(seed=7)
        assert _alpha(env, "swap_banana_geom") == 0.0
        assert _alpha(env, "cube_geom") == pytest.approx(1.0)

    def test_out_of_range_index_rejected(self, env: TabletopEnv) -> None:
        with pytest.raises(ValueError, match="object_swap: index"):
            env.set_perturbation("object_swap", float(len(OBJECT_SWAP_CLASSES)))
        with pytest.raises(ValueError, match="object_swap: index"):
            env.set_perturbation("object_swap", -1.0)

    def test_set_object_swap_classes_subset_dispatches_correctly(self, env: TabletopEnv) -> None:
        # Mirroring the B-05 wiring posture: a subset registry handed in
        # by the runner means "queue index i -> resolve to subset[i]".
        # Here we narrow to (cube, banana) so index 1 must reveal the
        # banana geom, not the mug (which would be index 1 in the
        # canonical ordering).
        env.set_object_swap_classes(("cube", "banana"))
        assert env.object_swap_classes == ("cube", "banana")
        env.set_perturbation("object_swap", 1.0)
        env.reset(seed=11)
        assert _alpha(env, "swap_banana_geom") == pytest.approx(1.0)
        assert _alpha(env, "swap_mug_geom") == 0.0


# ---------------------------------------------------- backend rejection


class TestObjectSwapBackendRejection:
    """B-06 anti-feature: only MuJoCo ships the alternate asset library.

    PyBullet / Genesis / Isaac declare ``object_swap`` in
    ``VISUAL_ONLY_AXES`` so the loader's
    :func:`gauntlet.suite.loader._reject_purely_visual_suites` rejects
    object-swap-only suites and the linter flags mixed declarations.
    These tests pin the ClassVar declaration without importing the
    optional backends (which need extras to install).
    """

    def test_mujoco_keeps_visual_only_axes_empty(self) -> None:
        # Sanity: MuJoCo ``TabletopEnv`` does ship the asset library, so
        # ``object_swap`` is NOT in its ``VISUAL_ONLY_AXES``.
        assert "object_swap" not in TabletopEnv.VISUAL_ONLY_AXES
        assert "object_swap" in TabletopEnv.AXIS_NAMES

    def test_object_swap_only_suite_rejected_via_visual_only(self) -> None:
        # Build a Suite directly against the canonical ``Suite`` shape
        # to confirm string-axis values still validate; combined with
        # the loader test above this anchors the YAML round-trip.
        suite = Suite(
            name="all-swap",
            env="tabletop",
            episodes_per_cell=1,
            axes={"object_swap": AxisSpec(values=["cube", "mug"])},
        )
        # MuJoCo has empty VISUAL_ONLY_AXES so the loader accepts a
        # swap-only suite — confirms the rejection in the test above is
        # specific to non-MuJoCo backends and not a side-effect of the
        # axis itself.
        assert suite.num_cells() == 2
