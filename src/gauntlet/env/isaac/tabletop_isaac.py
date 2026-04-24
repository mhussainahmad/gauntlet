"""Isaac Sim tabletop backend — state-only first cut (RFC-009).

Parity with :class:`gauntlet.env.tabletop.TabletopEnv`,
:class:`gauntlet.env.pybullet.tabletop_pybullet.PyBulletTabletopEnv`,
and :class:`gauntlet.env.genesis.tabletop_genesis.GenesisTabletopEnv`
(state-only ``render_in_obs=False`` shape) at the observation /
action / perturbation-axis interface level (RFC-009 §3 + §6), with
the deliberate differences RFC-009 §7 documents:

* **Numerical non-parity across backends** (§7.3). Same seed + same
  policy -> numerically different trajectories on MuJoCo vs PyBullet
  vs Genesis vs Isaac Sim. Semantically similar, not numerically
  identical. ``gauntlet compare`` across backends measures simulator
  drift.
* **State-only first cut** (§6). Cosmetic axes
  (``lighting_intensity``, ``object_texture``, ``camera_offset_{x,y}``)
  queue + validate + store on shadow attributes and are members of
  :attr:`VISUAL_ONLY_AXES` pending a follow-up rendering RFC.
* **GPU-runtime requirement.** ``isaacsim`` wraps NVIDIA Omniverse
  Kit; constructing this env on a CPU-only machine raises a Kit
  bootstrap error from inside :class:`isaacsim.SimulationApp`.
  RFC-009 §4.4 / §8 documents the test strategy: CI uses a fake
  ``isaacsim`` namespace injected into ``sys.modules`` so the
  contract is exercised without ever launching real Kit.

Scene layout (RFC-009 §5): plane + fixed table-top + dynamic cube +
kinematic-by-teleport EE + visual target marker + 10 pre-allocated
teleport-away distractors. No URDF / USD robot asset — the EE is a
small ``VisualCuboid`` whose pose is overwritten via
``set_world_pose`` each control step. Same shape Genesis (RFC-007 §5)
and PyBullet (RFC-005 §7.1) chose.

Step 4 (this file): scaffold only — constants, ``__init__``,
``observation_space`` / ``action_space``, ``AXIS_NAMES`` /
``VISUAL_ONLY_AXES``, and method stubs that raise
:class:`NotImplementedError`. Steps 5 and 6 land the body and the
seven perturbation branches.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import isaacsim  # bootstrap Kit (lazy via SimulationApp on construct)
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

__all__ = ["IsaacSimTabletopEnv"]


# Scene / control constants mirror the other backends exactly so a
# given (seed, axis_config) produces semantically comparable rollouts
# across simulators even if the floating-point trajectories diverge
# (RFC-009 §7.3).
_TABLE_TOP_Z: float = 0.42
_CUBE_HALF: float = 0.025
_CUBE_REST_Z: float = _TABLE_TOP_Z + _CUBE_HALF

_TABLE_HALF_X: float = 0.5
_TABLE_HALF_Y: float = 0.5
_TABLE_HALF_Z: float = 0.02

_CUBE_INIT_HALFRANGE: float = 0.15
_TARGET_HALFRANGE: float = 0.2
_EE_REST_OFFSET_Z: float = 0.15
_EE_VISUAL_HALF: float = 0.01

_N_DISTRACTOR_SLOTS: int = 10
_DISTRACTOR_HALF: float = 0.02

# Positions match the Genesis and PyBullet adapters so the
# "three-distractor ring" that ``distractor_count`` = 3 produces is
# at identical (x, y) across backends. Disabled distractors are
# teleported to ``z = -10.0`` (below the ground plane and out of every
# physically-plausible EE reach) — same teleport-away semantic the
# Genesis adapter uses.
_DISTRACTOR_BASELINE_XY: NDArray[np.float64] = np.array(
    [
        (0.30, 0.30),
        (-0.30, 0.30),
        (0.30, -0.30),
        (-0.30, -0.30),
        (0.35, 0.00),
        (-0.35, 0.00),
        (0.00, 0.35),
        (0.00, -0.35),
        (0.25, 0.10),
        (-0.25, -0.10),
    ],
    dtype=np.float64,
)
_DISTRACTOR_REST_Z: float = _TABLE_TOP_Z + _DISTRACTOR_HALF
_DISTRACTOR_HIDDEN_Z: float = -10.0  # below the plane, out of reach


class IsaacSimTabletopEnv:
    """Isaac Sim state-only tabletop pick-and-place env.

    Scene (RFC-009 §5.1): ground plane + fixed table + dynamic cube +
    kinematic EE + visual target marker + 10 pre-allocated
    teleport-away distractors.

    Action / observation spaces are shape-compatible with
    :class:`gauntlet.env.tabletop.TabletopEnv` (state-only); no
    kinematic arm, no IK — the EE is a kinematic prim teleported via
    ``set_world_pose`` each control step.

    Attributes
    ----------
    MAX_LINEAR_STEP, MAX_ANGULAR_STEP, GRASP_RADIUS, TARGET_RADIUS,
    GRIPPER_OPEN, GRIPPER_CLOSED :
        Control / grasp / success constants (parity with the other
        backends).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_texture",
            "object_initial_pose_x",
            "object_initial_pose_y",
            "distractor_count",
        }
    )
    # State-only first cut (RFC-009 §6.6). The four cosmetic axes
    # store on shadow attributes during ``set_perturbation`` /
    # ``_apply_pending_perturbations`` and produce no obs-dict delta
    # until a follow-up rendering RFC lands. The Suite loader's
    # ``_reject_purely_visual_suites`` guard fires off this set, so a
    # YAML naming only cosmetic axes on ``tabletop-isaac`` is
    # rejected at load time. Same shape Genesis had pre-RFC-008 and
    # PyBullet had pre-RFC-006.
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset(
        {
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_texture",
        }
    )

    MAX_LINEAR_STEP: float = 0.05
    MAX_ANGULAR_STEP: float = 0.1
    GRASP_RADIUS: float = 0.05
    TARGET_RADIUS: float = 0.05
    GRIPPER_OPEN: float = 1.0
    GRIPPER_CLOSED: float = -1.0

    observation_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]

    def __init__(
        self,
        *,
        max_steps: int = 200,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")

        self._max_steps = max_steps

        # ---- spaces (5-key state obs, identical to Genesis state-only) ----
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        obs_spaces: dict[str, gym.spaces.Space[Any]] = {
            "cube_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "cube_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "gripper": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            "target_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # ---- Kit bootstrap + scene construction ----
        # Step 5 lands the body. The scaffold instantiates SimulationApp
        # (headless), pulls World, and registers the prim handles so
        # subsequent reset/step calls find them. Done in __init__ so the
        # ~5-s Kit boot is paid once per worker process (matches the
        # Genesis ``gs.init`` amortisation pattern, RFC-009 §4.6).
        self._kit_app: Any = isaacsim.SimulationApp({"headless": True})

        # Imported here, not at module top, because importing them
        # before SimulationApp is constructed raises inside Kit. The
        # adapter pattern matches Isaac Sim's official examples.
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import (
            DynamicCuboid,
            FixedCuboid,
            VisualCuboid,
        )

        self._world: Any = World()

        # ---- Scene primitives ----
        # Table-top: kinematic FixedCuboid sized to match the other
        # backends.
        self._table: Any = self._world.scene.add(
            FixedCuboid(
                prim_path="/World/table",
                position=np.array([0.0, 0.0, _TABLE_TOP_Z - _TABLE_HALF_Z], dtype=np.float64),
                size=np.array(
                    [2.0 * _TABLE_HALF_X, 2.0 * _TABLE_HALF_Y, 2.0 * _TABLE_HALF_Z],
                    dtype=np.float64,
                ),
            )
        )
        # The cube — DynamicCuboid (rigid body, mass-driven).
        self._cube: Any = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/cube",
                position=np.array([0.0, 0.0, _CUBE_REST_Z], dtype=np.float64),
                size=np.array(
                    [2.0 * _CUBE_HALF, 2.0 * _CUBE_HALF, 2.0 * _CUBE_HALF], dtype=np.float64
                ),
            )
        )
        # Visual target marker — VisualCuboid (no collision, no mass).
        self._target: Any = self._world.scene.add(
            VisualCuboid(
                prim_path="/World/target",
                position=np.array([0.0, 0.0, _TABLE_TOP_Z + 0.001], dtype=np.float64),
                size=np.array(
                    [2.0 * self.TARGET_RADIUS, 2.0 * self.TARGET_RADIUS, 0.002],
                    dtype=np.float64,
                ),
            )
        )
        # EE — VisualCuboid (kinematic, teleported each step).
        self._ee: Any = self._world.scene.add(
            VisualCuboid(
                prim_path="/World/ee",
                position=np.array([0.0, 0.0, _CUBE_REST_Z + _EE_REST_OFFSET_Z], dtype=np.float64),
                size=np.array(
                    [2.0 * _EE_VISUAL_HALF, 2.0 * _EE_VISUAL_HALF, 2.0 * _EE_VISUAL_HALF],
                    dtype=np.float64,
                ),
            )
        )
        # Distractors — pre-allocate all 10 at hidden Z; the
        # ``distractor_count`` axis teleports the first ``count`` to
        # rest_z in ``_apply_pending_perturbations``.
        self._distractors: list[Any] = []
        for i in range(_N_DISTRACTOR_SLOTS):
            xy = _DISTRACTOR_BASELINE_XY[i]
            d = self._world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/distractor_{i}",
                    position=np.array(
                        [float(xy[0]), float(xy[1]), _DISTRACTOR_HIDDEN_Z], dtype=np.float64
                    ),
                    size=np.array(
                        [2.0 * _DISTRACTOR_HALF, 2.0 * _DISTRACTOR_HALF, 2.0 * _DISTRACTOR_HALF],
                        dtype=np.float64,
                    ),
                )
            )
            self._distractors.append(d)

        # ---- pending perturbations queue (drained by reset) ----
        self._pending_perturbations: dict[str, float] = {}

        # ---- per-episode runtime state ----
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._step_count: int = 0
        self._grasped: bool = False
        self._gripper_state: float = self.GRIPPER_OPEN
        self._success: bool = False
        self._target_pos: NDArray[np.float64] = np.zeros(3, dtype=np.float64)

        # Visual-axis shadows — set_perturbation stores here even on the
        # state-only first cut so the follow-up rendering RFC can read
        # them without a second pass over the adapter (matches the same
        # shadow-pattern Genesis pre-RFC-008 used).
        self._light_intensity: float = 1.0
        self._cam_offset: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._texture_choice: int = 0

        self._closed: bool = False

    # ----------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Deterministic reset — body lands in step 5."""
        del seed, options
        raise NotImplementedError("IsaacSimTabletopEnv.reset lands in step 5 of RFC-009 §12")

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        """Advance one control step — body lands in step 5."""
        del action
        raise NotImplementedError("IsaacSimTabletopEnv.step lands in step 5 of RFC-009 §12")

    def set_perturbation(self, name: str, value: float) -> None:
        """Queue an axis-value pair — full validation lands in step 6."""
        del value
        raise NotImplementedError(
            f"IsaacSimTabletopEnv.set_perturbation({name!r}, ...) lands in step 6 of RFC-009 §12"
        )

    def restore_baseline(self) -> None:
        """Restore observable baseline state — body lands in step 6."""
        raise NotImplementedError(
            "IsaacSimTabletopEnv.restore_baseline lands in step 6 of RFC-009 §12"
        )

    def close(self) -> None:
        """Release Kit + simulation resources. Idempotent.

        Best-effort drop. ``SimulationApp`` exposes ``.close()`` from
        Isaac Sim 4.x onwards; we guard against double-close with a
        flag because ``SimulationApp`` is a process-global singleton
        and a second close on the same handle can wedge Kit.
        """
        import contextlib

        if self._closed:
            return
        # Kit may raise opaque C errors during shutdown — best-effort
        # drop matches the PyBullet adapter's contextlib.suppress
        # pattern.
        with contextlib.suppress(Exception):
            self._kit_app.close()
        self._closed = True
