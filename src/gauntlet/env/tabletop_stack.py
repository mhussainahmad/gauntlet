"""Long-horizon 3-step stacking env (B-09 stub).

Three coloured cubes on a table top — red ``A``, green ``B``, blue
``C``. The policy's job is a 3-subtask skill chain:

* **subtask 0** — cube A stacked on cube B: ``A.xy ≈ B.xy`` (within
  :attr:`TabletopStackEnv.STACK_XY_TOL`) AND ``A.z`` strictly above
  ``B.z`` (within :attr:`TabletopStackEnv.STACK_Z_BAND`).
* **subtask 1** — cube C grasped: the end-effector is within
  :attr:`TabletopStackEnv.GRASP_RADIUS` of cube C *and* the gripper is
  closed. Latched once; staying ``True`` for the rest of the rollout
  even after the policy releases C onto the stack.
* **subtask 2** — cube C stacked on the A-on-B pile: ``C.xy ≈ A.xy``
  AND ``C.z > A.z``. Final-success predicate.

The list ``info["subtask_completion"]`` is monotonically
non-decreasing — once a subtask flips to ``True`` it stays that way for
the remainder of the rollout, even if the policy then knocks the stack
over. This matches the credit-accounting framing in
:class:`gauntlet.env.base.SubtaskMilestone`: per-step latching is
simpler than computing "the policy at one point reached subtask N",
and policies that bobble a milestone still earn the credit they
already got.

Episode-level success (``info["success"]``) is the last subtask
flipping to ``True`` (subtask 2 above). ``terminated`` follows
``success``; ``truncated`` fires at :attr:`max_steps`.

Action / observation surface
----------------------------
The action and observation surfaces deliberately mirror
:class:`gauntlet.env.tabletop.TabletopEnv` so a scripted policy
written for the source env carries over without retraining its
control conventions:

* Action: ``Box(shape=(7,), float64, [-1, 1])`` —
  ``[dx, dy, dz, drx, dry, drz, gripper]``. The first six are
  per-step end-effector twist commands scaled by
  :attr:`TabletopStackEnv.MAX_LINEAR_STEP` /
  :attr:`TabletopStackEnv.MAX_ANGULAR_STEP`. The gripper command
  snaps to ``open (+1) / closed (-1)``.
* Observation: ``Dict`` with per-cube ``cube_a_pos`` / ``cube_a_quat``
  / ``cube_b_pos`` / ``cube_b_quat`` / ``cube_c_pos`` /
  ``cube_c_quat`` slots, plus ``ee_pos`` and ``gripper``. There is
  no ``target_pos`` slot — stacking is the target.

Perturbation surface (deliberately empty in this stub)
------------------------------------------------------
``AXIS_NAMES = frozenset()`` — the stub env intentionally publishes
NO perturbation axes. Wiring the canonical 8 axes for a 3-cube scene
is scope-creep for the partial-completion PR (the dispatch table
would need per-cube object_swap, per-cube initial-pose, and a
multi-cube collision-budget treatment that the source TabletopEnv's
single-cube apply table does not have). Suite YAMLs that target
``tabletop-stack`` therefore run a *single-cell* perturbation grid
(the baseline) until the B-09 follow-up wires the axes.

Reproducibility
---------------
Same hard rule as :class:`TabletopEnv`: ``reset(seed=...)`` is the
sole entropy source. Two ``reset(seed=s)`` calls produce identical
initial observations.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, ClassVar, Final

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from gauntlet.env.base import Observation

__all__ = ["TabletopStackEnv"]


_ASSET_PATH: Path = Path(__file__).parent / "assets" / "tabletop_stack.xml"

# Cube body / geom name → slot index. Order matters: the per-step
# subtask predicates index this tuple ("A", "B", "C") to look up cube
# poses. Keeping the names as a tuple of strings rather than three
# scalars makes the body-id resolution loop a single comprehension.
_CUBE_NAMES: Final[tuple[str, str, str]] = ("a", "b", "c")

# Number of subtasks the skill-chain publishes. Frozen at 3 by the
# scene's three-cube geometry — see the module docstring.
_N_SUBTASKS: Final[int] = 3


_ObsType = dict[str, NDArray[Any]]
_ActType = NDArray[np.float64]


class TabletopStackEnv(gym.Env[_ObsType, _ActType]):
    """Three-cube stacking env satisfying ``GauntletEnv`` + ``SubtaskMilestone``.

    See the module docstring for the 3-subtask skill chain, action
    surface, and reproducibility contract.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}  # noqa: RUF012

    # Stub env: no perturbation axes are dispatched. Listed as a
    # ClassVar so the Suite loader can introspect axis support without
    # constructing the env (mirrors the GauntletEnv contract that
    # TabletopEnv satisfies). VISUAL_ONLY_AXES is the empty set on the
    # same grounds — there are no axes to declare visual-only against.
    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset()

    # SubtaskMilestone.n_subtasks — fixed for the env lifetime by the
    # 3-cube scene. Class-level so the runner can size aggregation
    # buffers before constructing the env (matches the SubtaskMilestone
    # Protocol's "MUST NOT vary between resets" rule).
    n_subtasks: ClassVar[int] = _N_SUBTASKS

    # Per-step action scaling. Same conventions as TabletopEnv so a
    # scripted policy can re-use the velocity calibration.
    MAX_LINEAR_STEP: float = 0.05  # metres per unit action
    MAX_ANGULAR_STEP: float = 0.1  # radians per unit action

    # Grasp / stacking tolerances.
    GRASP_RADIUS: float = 0.05
    # Stacking XY tolerance: the upper cube's centre projected to the
    # table plane must be within this radius of the lower cube's
    # centre. Matches the cube half-extent (0.025 m) doubled — i.e.
    # any meaningful overlap qualifies. Generous on purpose: the
    # success predicate is "stacked enough that physics will hold",
    # not "perfectly aligned".
    STACK_XY_TOL: float = 0.05
    # Stacking Z band: the upper cube's z minus the lower's must lie
    # in this open interval ``(STACK_Z_MIN, STACK_Z_MAX)``. The lower
    # bound rejects cubes co-planar with their support; the upper
    # bound rejects "floating above the support" (e.g. still in the
    # gripper, hovering). 1.5x cube half-extent gives the policy
    # comfortable slack on top.
    STACK_Z_MIN: float = 0.025  # cube half-extent
    STACK_Z_MAX: float = 0.075  # 3x half-extent

    # Gripper snap states. Mirror TabletopEnv literals.
    GRIPPER_OPEN: float = 1.0
    GRIPPER_CLOSED: float = -1.0

    # Scene constants (must agree with assets/tabletop_stack.xml).
    _TABLE_TOP_Z: float = 0.42  # = table.pos.z (0.4) + table_top.size.z (0.02)
    _CUBE_HALF: float = 0.025  # cube_*_geom.size along each axis
    _CUBE_REST_Z: float = _TABLE_TOP_Z + _CUBE_HALF
    _TABLE_HALF_X: float = 0.5
    _TABLE_HALF_Y: float = 0.5

    # Default cube starting poses. Per-axis Y offsets keep them
    # disjoint enough that the grasp predicate never gets ambiguous
    # ("which cube did the EE just close on?"). The reset() routine
    # may add small per-seed XY jitter on top.
    _DEFAULT_CUBE_XY: ClassVar[tuple[tuple[float, float], ...]] = (
        (0.0, -0.2),  # cube A
        (0.0, 0.0),  # cube B
        (0.0, 0.2),  # cube C
    )
    # Per-seed XY jitter half-range. Tight on purpose so the scripted
    # tests in tests/test_tabletop_stack.py can rely on the cubes
    # staying in their nominal columns even with seed variation.
    _CUBE_INIT_HALFRANGE: float = 0.03

    def __init__(
        self,
        *,
        max_steps: int = 600,
        n_substeps: int = 5,
    ) -> None:
        """Construct the env.

        ``max_steps`` is generous (600) compared to TabletopEnv (200)
        because the 3-subtask chain takes substantially more steps
        than a single pick-and-place. ``n_substeps`` defaults to the
        same physics ratio as the source env.
        """
        super().__init__()
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")
        if n_substeps <= 0:
            raise ValueError(f"n_substeps must be positive; got {n_substeps}")

        self._max_steps = max_steps
        self._n_substeps = n_substeps

        self._model: Any = mujoco.MjModel.from_xml_path(str(_ASSET_PATH))
        self._data: Any = mujoco.MjData(self._model)

        # Cached mujoco indices, in cube_a / cube_b / cube_c order.
        self._cube_body_ids: tuple[int, ...] = ()
        self._cube_qpos_adrs: tuple[int, ...] = ()
        self._cube_qvel_adrs: tuple[int, ...] = ()
        self._ee_body_id: int = 0
        self._ee_mocap_id: int = 0
        self._main_cam_id: int = 0
        self._cache_indices()

        # Empty pending-perturbation queue: AXIS_NAMES is empty so
        # set_perturbation always raises. The dict still exists so the
        # GauntletEnv contract's "queue cleared on reset" property is
        # honoured trivially.
        self._pending_perturbations: dict[str, float] = {}

        # Per-episode RNG; rebuilt in reset() from the seed.
        self._rng: np.random.Generator = np.random.default_rng()
        self._step_count: int = 0
        # ``int`` slot index into self._cube_body_ids of the cube the
        # gripper currently has hold of, or ``None`` when the gripper
        # is open / not near any cube. Cached across step boundaries
        # because a closed gripper that drifts slightly off-centre
        # should not silently lose its grip — the snap-to-EE pull-back
        # in :meth:`_snap_grasped_cube_to_ee` keeps the cube in place.
        self._grasped_cube_idx: int | None = None
        self._gripper_state: float = self.GRIPPER_OPEN
        # Latched per-subtask credit. Index aligned with the subtask
        # chain documented at module scope. ``_subtask_done`` is the
        # monotonic-latched view :meth:`_build_info` reads to
        # populate ``info["subtask_completion"]``.
        self._subtask_done: list[bool] = [False] * _N_SUBTASKS
        self._success: bool = False

        self.action_space: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        obs_spaces: dict[str, spaces.Space[Any]] = {}
        for letter in _CUBE_NAMES:
            obs_spaces[f"cube_{letter}_pos"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
            )
            obs_spaces[f"cube_{letter}_quat"] = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float64
            )
        obs_spaces["ee_pos"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
        obs_spaces["gripper"] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        self.observation_space: spaces.Dict = spaces.Dict(obs_spaces)

    # ------------------------------------------------------------------ setup

    def _cache_indices(self) -> None:
        """Resolve name → id lookups against the loaded model."""
        m = self._model
        body_ids: list[int] = []
        qpos_adrs: list[int] = []
        qvel_adrs: list[int] = []
        for letter in _CUBE_NAMES:
            body_name = f"cube_{letter}"
            bid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name))
            if bid < 0:
                raise RuntimeError(f"body {body_name!r} missing from MJCF")
            body_ids.append(bid)
            jnt_adr = int(m.body_jntadr[bid])
            qpos_adrs.append(int(m.jnt_qposadr[jnt_adr]))
            qvel_adrs.append(int(m.jnt_dofadr[jnt_adr]))
        self._cube_body_ids = tuple(body_ids)
        self._cube_qpos_adrs = tuple(qpos_adrs)
        self._cube_qvel_adrs = tuple(qvel_adrs)

        self._ee_body_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee"))
        self._ee_mocap_id = int(m.body_mocapid[self._ee_body_id])
        if self._ee_mocap_id < 0:
            raise RuntimeError("ee body is not a mocap body in the loaded MJCF")
        self._main_cam_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "main"))
        if self._main_cam_id < 0:
            raise RuntimeError("camera 'main' not found in loaded MJCF")

    # ------------------------------------------------------ GauntletEnv API

    def restore_baseline(self) -> None:
        """Stub satisfies the GauntletEnv contract — nothing to restore.

        The env exposes no perturbation axes (``AXIS_NAMES`` is empty),
        so there is no per-episode model mutation that could leak
        across resets. The method exists so the structural
        :class:`isinstance(env, GauntletEnv)` check passes — that's
        the protocol surface, not a no-op accident.
        """

    def set_perturbation(self, name: str, value: float) -> None:
        """Stub: this env publishes no axes; every queued name is rejected.

        See the module docstring "Perturbation surface" section. The
        runner's pre-flight check should reject these YAMLs at suite-load
        time; this fail-loud here is the runtime backstop.
        """
        del value
        raise ValueError(
            f"unknown perturbation axis: {name!r} "
            f"(tabletop-stack stub publishes no axes — AXIS_NAMES is empty)"
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Deterministic reset.

        ``seed`` is the only entropy source. Calling ``reset(seed=s)``
        twice yields identical initial observations.
        """
        del options
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # No baseline mutation today — restore_baseline is a no-op —
        # but mj_resetData wipes joint state so a stale grasp from
        # a prior episode does not leak.
        self.restore_baseline()
        mujoco.mj_resetData(self._model, self._data)

        # Lay out the cubes at their default rows with small per-seed
        # XY jitter inside ``_CUBE_INIT_HALFRANGE``. Identity quaternion
        # (wxyz = 1, 0, 0, 0) keeps cubes axis-aligned at rest.
        qpos = self._data.qpos
        qvel = self._data.qvel
        for slot, (default_xy, qpos_adr, qvel_adr) in enumerate(
            zip(self._DEFAULT_CUBE_XY, self._cube_qpos_adrs, self._cube_qvel_adrs, strict=True)
        ):
            jitter = self._rng.uniform(
                low=-self._CUBE_INIT_HALFRANGE,
                high=self._CUBE_INIT_HALFRANGE,
                size=2,
            ).astype(np.float64)
            qpos[qpos_adr + 0] = float(default_xy[0]) + float(jitter[0])
            qpos[qpos_adr + 1] = float(default_xy[1]) + float(jitter[1])
            qpos[qpos_adr + 2] = self._CUBE_REST_Z
            qpos[qpos_adr + 3] = 1.0
            qpos[qpos_adr + 4] = 0.0
            qpos[qpos_adr + 5] = 0.0
            qpos[qpos_adr + 6] = 0.0
            for k in range(6):
                qvel[qvel_adr + k] = 0.0
            del slot

        # Reset end-effector to a centred hover pose above the table.
        # Identity quat keeps mocap orientation stable across runs.
        self._data.mocap_pos[self._ee_mocap_id] = np.array(
            [0.0, 0.0, self._TABLE_TOP_Z + 0.25], dtype=np.float64
        )
        self._data.mocap_quat[self._ee_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Pending-perturbation queue is always empty (AXIS_NAMES is
        # empty), but clearing it satisfies the GauntletEnv contract
        # ordering (apply queued → clear queue) without a special case.
        self._pending_perturbations = {}

        self._step_count = 0
        self._grasped_cube_idx = None
        self._gripper_state = self.GRIPPER_OPEN
        self._subtask_done = [False] * _N_SUBTASKS
        self._success = False

        # Populate body_xpos / cached arrays for the obs.
        mujoco.mj_forward(self._model, self._data)

        return self._build_obs(), self._build_info()

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Advance one control step.

        Pipeline: clip → mocap pose update → grasp tracking →
        ``mj_step`` → snap grasped cube to EE → update subtask
        latches → build obs / reward / flags.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (7,):
            raise ValueError(f"action must have shape (7,); got {a.shape}")
        a = np.clip(a, -1.0, 1.0).astype(np.float64, copy=False)

        # 1. End-effector twist.
        self._apply_ee_command(a[0:3], a[3:6])

        # 2. Gripper command + grasp tracking.
        self._gripper_state = self.GRIPPER_OPEN if a[6] > 0.0 else self.GRIPPER_CLOSED
        self._update_grasp_state()

        # 3. Physics — no actuators, so we just step. Mocap-driven scene
        # has ``model.nu == 0``; matches TabletopEnv's stub behaviour.
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        # 4. If a cube is grasped, snap it back to the EE pose so
        # contact-induced drift cannot yank it free. Mirror
        # TabletopEnv._snap_cube_to_ee semantics.
        if self._grasped_cube_idx is not None:
            self._snap_grasped_cube_to_ee()
            mujoco.mj_forward(self._model, self._data)

        # 5. Bookkeeping + subtask latches + success.
        self._step_count += 1
        self._update_subtask_latches()
        self._success = self._subtask_done[_N_SUBTASKS - 1]

        terminated = self._success
        truncated = (not terminated) and self._step_count >= self._max_steps
        # Reward is shaped as the fraction of milestones latched so far —
        # gives the report a non-binary signal even before the worker
        # forwards the full info["subtask_completion"] list.
        reward = float(sum(self._subtask_done)) / float(_N_SUBTASKS)
        return self._build_obs(), reward, terminated, truncated, self._build_info()

    def render(self) -> NDArray[np.uint8]:
        """Render an RGB frame (debug only — tests must not call this)."""
        renderer = mujoco.Renderer(self._model)
        renderer.update_scene(self._data)
        pixels = renderer.render()
        renderer.close()
        return np.asarray(pixels, dtype=np.uint8)

    def close(self) -> None:
        """Drop references to the MuJoCo model / data. Idempotent."""
        with contextlib.suppress(Exception):
            self._data = None
        with contextlib.suppress(Exception):
            self._model = None

    # ------------------------------------------------------ SubtaskMilestone

    def is_subtask_done(self, idx: int, obs: Observation) -> bool:
        """Pure predicate: is the ``idx``-th subtask satisfied by ``obs``?

        Reads cube positions out of the standard observation slots
        (``cube_a_pos`` / ``cube_b_pos`` / ``cube_c_pos``) and the
        gripper state (``gripper``). Does NOT consult the env's
        latched flags — this is the un-latched view, intended for
        suite-loaders / scripted-policy authors who want to query the
        predicate without a step. The latched view lives in
        ``info["subtask_completion"]`` from the most recent step.

        Raises:
            IndexError: if ``idx`` is outside ``[0, n_subtasks)``.
            KeyError: if ``obs`` is missing a required slot.
        """
        if idx < 0 or idx >= _N_SUBTASKS:
            raise IndexError(
                f"subtask idx {idx} out of range [0, {_N_SUBTASKS}) for tabletop-stack"
            )
        a = np.asarray(obs["cube_a_pos"], dtype=np.float64)
        b = np.asarray(obs["cube_b_pos"], dtype=np.float64)
        c = np.asarray(obs["cube_c_pos"], dtype=np.float64)
        gripper = float(np.asarray(obs["gripper"], dtype=np.float64).reshape(-1)[0])
        ee_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
        if idx == 0:
            return self._is_stacked(upper=a, lower=b)
        if idx == 1:
            # Subtask 1 — C grasped right now. The latched view in
            # info["subtask_completion"] sticks once True; this
            # predicate is the un-latched "is C currently in the
            # gripper" query.
            ee_to_c = float(np.linalg.norm(ee_pos - c))
            closed = gripper <= 0.0
            return closed and ee_to_c <= self.GRASP_RADIUS
        # idx == 2 — final stack: C on top of A.
        return self._is_stacked(upper=c, lower=a)

    # --------------------------------------------------------------- helpers

    def _apply_ee_command(
        self,
        linear: NDArray[np.float64],
        angular: NDArray[np.float64],
    ) -> None:
        """Translate + rotate the mocap body by small deltas. Mirror TabletopEnv."""
        cur_pos = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
        new_pos = cur_pos + linear * self.MAX_LINEAR_STEP
        self._data.mocap_pos[self._ee_mocap_id] = new_pos

        axis_angle = angular * self.MAX_ANGULAR_STEP
        angle = float(np.linalg.norm(axis_angle))
        if angle > 0.0:
            axis = axis_angle / angle
            dq = np.zeros(4, dtype=np.float64)
            mujoco.mju_axisAngle2Quat(dq, axis, angle)
            cur_quat = np.array(self._data.mocap_quat[self._ee_mocap_id], dtype=np.float64)
            new_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(new_quat, dq, cur_quat)
            mujoco.mju_normalize4(new_quat)
            self._data.mocap_quat[self._ee_mocap_id] = new_quat

    def _update_grasp_state(self) -> None:
        """Snap ``_grasped_cube_idx`` based on gripper command + EE-cube proximity.

        Closing the gripper near a cube selects that cube as the
        grasp target. The cube-id is *latched* across steps while the
        gripper stays closed (so a slight EE drift does not silently
        drop the cube), and re-opens release the cube. If multiple
        cubes are simultaneously inside ``GRASP_RADIUS``, the lowest
        index wins (deterministic — matches the cube enumeration
        order).
        """
        if self._gripper_state == self.GRIPPER_OPEN:
            self._grasped_cube_idx = None
            return
        # Already grasping a cube this step? Stay latched even if drift
        # nudges the cube outside the radius (the post-physics snap
        # call below will pull it back). This is the same "sticky"
        # rule TabletopEnv uses on its single cube.
        if self._grasped_cube_idx is not None:
            return
        ee_pos = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
        # Find the closest cube within the grasp radius. The
        # tie-break (lowest index wins) is implicit in the iteration
        # order; we record both the closest distance and its slot to
        # short-circuit when nothing is in range.
        closest_idx: int | None = None
        closest_dist: float = float("inf")
        for slot, body_id in enumerate(self._cube_body_ids):
            cube_pos = np.array(self._data.xpos[body_id], dtype=np.float64)
            dist = float(np.linalg.norm(ee_pos - cube_pos))
            if dist <= self.GRASP_RADIUS and dist < closest_dist:
                closest_idx = slot
                closest_dist = dist
        self._grasped_cube_idx = closest_idx

    def _snap_grasped_cube_to_ee(self) -> None:
        """Overwrite the grasped cube's freejoint qpos / qvel to track the EE."""
        if self._grasped_cube_idx is None:  # pragma: no cover — caller-guarded
            return
        idx = self._grasped_cube_idx
        ee_pos = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
        ee_quat = np.array(self._data.mocap_quat[self._ee_mocap_id], dtype=np.float64)
        qpos = self._data.qpos
        qpos_adr = self._cube_qpos_adrs[idx]
        qpos[qpos_adr + 0] = ee_pos[0]
        qpos[qpos_adr + 1] = ee_pos[1]
        qpos[qpos_adr + 2] = ee_pos[2]
        qpos[qpos_adr + 3] = ee_quat[0]
        qpos[qpos_adr + 4] = ee_quat[1]
        qpos[qpos_adr + 5] = ee_quat[2]
        qpos[qpos_adr + 6] = ee_quat[3]
        qvel = self._data.qvel
        qvel_adr = self._cube_qvel_adrs[idx]
        for k in range(6):
            qvel[qvel_adr + k] = 0.0

    def _update_subtask_latches(self) -> None:
        """Recompute per-subtask predicates and latch True flags forever.

        Latching is monotonic non-decreasing (SubtaskMilestone
        contract): once a subtask flips to ``True``, it stays that
        way for the remainder of the rollout, even if a later step
        invalidates the predicate (stack toppled, cube released).
        """
        a_pos = np.array(self._data.xpos[self._cube_body_ids[0]], dtype=np.float64)
        b_pos = np.array(self._data.xpos[self._cube_body_ids[1]], dtype=np.float64)
        c_pos = np.array(self._data.xpos[self._cube_body_ids[2]], dtype=np.float64)

        # Subtask 0 — A stacked on B.
        if not self._subtask_done[0] and self._is_stacked(upper=a_pos, lower=b_pos):
            self._subtask_done[0] = True

        # Subtask 1 — C grasped at any point. We require subtask 0 to
        # have fired first so the natural order ("A on B, then pick
        # C") is enforced; this avoids the policy gaming the
        # latching by closing on C immediately and getting credit
        # for skipping the stack. Matches the LAMBDA / RoboCerebra
        # framing where mid-rollout drift is the failure surface.
        if not self._subtask_done[1] and self._subtask_done[0] and self._grasped_cube_idx == 2:
            self._subtask_done[1] = True

        # Subtask 2 — final stack: C on A. Likewise gated on subtask 1
        # so a policy that lucky-drops C on the pile without ever
        # picking it up does not earn the final credit.
        if (
            not self._subtask_done[2]
            and self._subtask_done[1]
            and self._is_stacked(upper=c_pos, lower=a_pos)
        ):
            self._subtask_done[2] = True

    def _is_stacked(
        self,
        *,
        upper: NDArray[np.float64],
        lower: NDArray[np.float64],
    ) -> bool:
        """``upper`` cube is resting on top of ``lower`` cube.

        Two predicates: XY centres aligned within ``STACK_XY_TOL``
        and the Z gap is in ``(STACK_Z_MIN, STACK_Z_MAX)`` so the
        upper cube is genuinely sitting on the lower (not coplanar,
        not floating).
        """
        xy_gap = float(np.linalg.norm(upper[:2] - lower[:2]))
        z_gap = float(upper[2] - lower[2])
        return xy_gap <= self.STACK_XY_TOL and self.STACK_Z_MIN < z_gap < self.STACK_Z_MAX

    def _build_obs(self) -> dict[str, NDArray[Any]]:
        obs: dict[str, NDArray[Any]] = {}
        for letter, body_id in zip(_CUBE_NAMES, self._cube_body_ids, strict=True):
            obs[f"cube_{letter}_pos"] = np.array(self._data.xpos[body_id], dtype=np.float64)
            obs[f"cube_{letter}_quat"] = np.array(self._data.xquat[body_id], dtype=np.float64)
        obs["ee_pos"] = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
        obs["gripper"] = np.array([self._gripper_state], dtype=np.float64)
        return obs

    def _build_info(self) -> dict[str, Any]:
        # Defensive copy — consumers should never mutate the env's
        # internal latch state by writing into the info dict.
        return {
            "success": self._success,
            "step": self._step_count,
            "grasped_cube_idx": self._grasped_cube_idx,
            "subtask_completion": list(self._subtask_done),
        }
