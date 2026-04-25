"""Mobile-base navigation + pick env (B-13).

Composition over an inner :class:`gauntlet.env.tabletop.TabletopEnv`.
The "wheeled chassis" is **kinematic-only** for this phase: we integrate
a SE(2) base pose forwards from a base-velocity action, but the inner
MJCF stays static. This is the deliberate phase-1 scope called out in
the B-13 spec — a real freejoint-mounted arm + collision rebake is a
follow-up. The point of phase 1 is to expose the action / observation
plumbing so downstream nav-aware policies have something to consume.

Action layout (``shape=(10,), dtype=float64, bounds=[-1, 1]``)::

    [dx, dy, dz, drx, dry, drz, gripper, base_vx, base_vy, base_omega_z]
     └─────────── inner tabletop EE twist ───────────┘ └─ base velocity ─┘

The first 7 entries are forwarded verbatim to the inner env's existing
mocap-driven step. The trailing 3 entries are integrated kinematically:

    base_pose <- base_pose + (vx, vy, omega) * BASE_DT

Observation: every key the inner env publishes, plus a single new
``pose: Box(shape=(3,), float64)`` carrying the integrated SE(2) base
pose ``(x, y, theta)``.

Task: the inner env spawns the cube + target inside the table half-
extents; this wrapper places a *base-frame target table* at a fixed
distance ``>= 1m`` from the initial base origin. Success is the AND of
two phase flags:

* ``nav_done`` — base XY within :attr:`NAV_RADIUS` of the target table.
* ``picked``  — inner env reports ``info["grasped"] is True``.

Reward is shaped on the sum of the two distances so a policy gets
useful gradient before either subgoal lands.

The wrapper does **not** forward perturbations to the inner env on the
phase-1 path: :attr:`AXIS_NAMES` is empty and
:meth:`set_perturbation` raises :class:`ValueError` on every key. This
keeps the SE(2) pose plumbing free of axis-coupling surprises until
the real freejoint mount lands.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from gauntlet.env.base import Action, Observation
from gauntlet.env.tabletop import TabletopEnv

__all__ = ["MobileTabletopEnv"]


# Type aliases re-exported from ``gauntlet.env.base`` so this module
# does not introduce its own explicit-``Any`` carriers (the FFI seam
# carve-out lives on ``base.py``; see pyproject.toml mypy override).
_ObsType = Observation
_ActType = Action
# Info dicts cross the gymnasium Protocol seam. We keep the value type
# as ``object`` (not ``Any``) on the wrapper's *own* signatures so the
# ``disallow_any_explicit`` rule stays satisfied without a per-module
# pyproject carve-out. The inner :class:`TabletopEnv` returns
# ``dict[str, Any]`` (it lives in the FFI override), and we widen that
# at the seam via the ``Mapping[str, object]`` parameter type.
_InfoMap = Mapping[str, object]
_Info = dict[str, object]

# Inner-env action width: see TabletopEnv module docstring.
_INNER_ACTION_DIM: int = 7
# Trailing base-velocity slots: (vx, vy, omega_z).
_BASE_ACTION_DIM: int = 3
_TOTAL_ACTION_DIM: int = _INNER_ACTION_DIM + _BASE_ACTION_DIM


class MobileTabletopEnv(gym.Env[_ObsType, _ActType]):
    """Wheeled-base wrapper around :class:`TabletopEnv` (kinematic phase 1).

    See module docstring for the action / observation contract and the
    deliberate "no MJCF rebake yet" scope decision.

    Attributes
    ----------
    BASE_DT:
        Per-step base integration interval (seconds). One env step
        advances the base pose by ``velocity * BASE_DT``.
    MAX_BASE_LINEAR:
        Per-step base linear-velocity cap (m/s). Action of ``1.0`` on
        ``vx``/``vy`` commands this magnitude.
    MAX_BASE_ANGULAR:
        Per-step base angular-velocity cap (rad/s).
    NAV_RADIUS:
        Distance from the target table at which ``nav_done`` flips True.
    TARGET_TABLE_OFFSET:
        Initial base-frame XY of the "target table" the base must
        navigate to. ``(1.5, 0.0)`` clears the spec's ``>= 1m`` floor.
    """

    metadata: dict[str, list[str]] = {"render_modes": ["rgb_array"]}  # noqa: RUF012

    # Phase-1 anti-feature: the wrapper does NOT forward perturbations to
    # the inner env. set_perturbation rejects every key. The intent is
    # to ship the SE(2) plumbing first; coupling axes through to the
    # inner env arrives with the real freejoint mount.
    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset()

    # Base motion model constants. Tight enough that 50 random steps
    # cannot teleport the base across the room; loose enough that a
    # scripted ``vx=+1`` policy reaches the target table in well under
    # the default ``max_steps``.
    BASE_DT: float = 0.1
    MAX_BASE_LINEAR: float = 1.0  # m/s
    MAX_BASE_ANGULAR: float = 1.0  # rad/s

    # Nav success tolerance + initial target placement.
    NAV_RADIUS: float = 0.3
    TARGET_TABLE_OFFSET: tuple[float, float] = (1.5, 0.0)

    def __init__(
        self,
        *,
        max_steps: int = 200,
        n_substeps: int = 5,
        inner: TabletopEnv | None = None,
    ) -> None:
        """Construct the wrapper, optionally injecting an inner env.

        ``inner`` is exposed for tests that want to swap a smaller /
        rendered inner env in. The default constructs a vanilla
        :class:`TabletopEnv` with the supplied ``max_steps`` /
        ``n_substeps`` so the registry factory call (``MobileTabletopEnv()``)
        Just Works.
        """
        super().__init__()
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")
        if n_substeps <= 0:
            raise ValueError(f"n_substeps must be positive; got {n_substeps}")

        self._inner: TabletopEnv = inner or TabletopEnv(
            max_steps=max_steps,
            n_substeps=n_substeps,
        )

        # SE(2) base pose: (x, y, theta). World-frame metres / radians.
        self._base_pose: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        # Snapshot of the target-table XY in world frame. Static for
        # phase 1 — the spec calls for "≥ 1m from initial base pose"
        # and we honour it with a fixed offset rather than randomising
        # (kept boring on purpose; a future axis can perturb it).
        self._target_table_xy: NDArray[np.float64] = np.array(
            self.TARGET_TABLE_OFFSET,
            dtype=np.float64,
        )
        self._nav_done: bool = False
        self._success: bool = False
        self._step_count: int = 0
        self._max_steps: int = max_steps

        self.action_space: spaces.Box = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(_TOTAL_ACTION_DIM,),
            dtype=np.float64,
        )
        # Compose obs space: every inner key, plus our pose key. The
        # ``cast`` keeps mypy on ``--strict`` with ``disallow_any_explicit``
        # happy: gymnasium's ``Dict.spaces`` returns ``dict[str, Space[Any]]``
        # at the FFI seam; we widen the value to ``spaces.Space[object]``
        # locally because we never read from these spaces, only forward
        # them into a fresh ``spaces.Dict`` constructor.
        inner_spaces: dict[str, spaces.Space[object]] = dict(
            self._inner.observation_space.spaces.items(),
        )
        inner_spaces["pose"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float64,
        )
        self.observation_space: spaces.Dict = spaces.Dict(inner_spaces)

    # --------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[_ObsType, _Info]:
        """Restore baseline, reset inner env, zero the base pose."""
        del options
        super().reset(seed=seed)
        inner_obs, inner_info = self._inner.reset(seed=seed)
        self._base_pose = np.zeros(3, dtype=np.float64)
        self._nav_done = False
        self._success = False
        self._step_count = 0
        return self._compose_obs(inner_obs), self._compose_info(inner_info)

    def step(
        self,
        action: _ActType,
    ) -> tuple[_ObsType, float, bool, bool, _Info]:
        """Forward inner action, integrate base velocity, recompute success."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (_TOTAL_ACTION_DIM,):
            raise ValueError(
                f"action must have shape ({_TOTAL_ACTION_DIM},); got {a.shape}",
            )
        a = np.clip(a, -1.0, 1.0).astype(np.float64, copy=False)

        inner_action = a[:_INNER_ACTION_DIM]
        base_action = a[_INNER_ACTION_DIM:]
        inner_obs, inner_reward, _terminated, _truncated, inner_info = self._inner.step(
            inner_action,
        )

        # Kinematic SE(2) integration. ``vx``/``vy`` are world-frame in
        # phase 1 — the chassis-relative twist→world rotation comes
        # with the freejoint mount. Keeping it world-frame here means
        # the scripted nav-then-pick test can drive ``vx=+1`` without
        # caring about ``theta``.
        vx = float(base_action[0]) * self.MAX_BASE_LINEAR
        vy = float(base_action[1]) * self.MAX_BASE_LINEAR
        omega = float(base_action[2]) * self.MAX_BASE_ANGULAR
        self._base_pose = self._base_pose + np.array(
            [vx * self.BASE_DT, vy * self.BASE_DT, omega * self.BASE_DT],
            dtype=np.float64,
        )

        # Phase-1 success: nav-done AND inner grasp. We deliberately do
        # NOT propagate the inner env's pick-and-place success — B-13's
        # task is nav-then-pick, not nav-then-place.
        nav_dist = float(
            np.linalg.norm(self._base_pose[:2] - self._target_table_xy),
        )
        self._nav_done = nav_dist <= self.NAV_RADIUS
        picked = bool(inner_info.get("grasped", False))
        self._success = self._nav_done and picked

        self._step_count += 1
        terminated = self._success
        truncated = (not terminated) and self._step_count >= self._max_steps

        # Reward: navigation-progress + (inner shaping once nav is done).
        # ``inner_reward`` is the inner env's negative-distance shaping;
        # gating it on ``nav_done`` keeps the gradient single-purpose
        # during the navigation phase.
        reward = -nav_dist + (inner_reward if self._nav_done else 0.0)
        return (
            self._compose_obs(inner_obs),
            float(reward),
            terminated,
            truncated,
            self._compose_info(inner_info),
        )

    def set_perturbation(self, name: str, value: float) -> None:
        """Reject every axis on the phase-1 path (anti-feature: see module docstring)."""
        del value
        raise ValueError(
            f"unknown perturbation axis: {name!r} (MobileTabletopEnv exposes no axes in phase 1)",
        )

    def restore_baseline(self) -> None:
        """Delegate to the inner env; the wrapper holds no model state."""
        self._inner.restore_baseline()
        self._base_pose = np.zeros(3, dtype=np.float64)
        self._nav_done = False
        self._success = False

    def close(self) -> None:
        """Release the inner env. Idempotent."""
        self._inner.close()

    # --------------------------------------------------------------- helpers

    def _compose_obs(self, inner_obs: _ObsType) -> _ObsType:
        out: _ObsType = dict(inner_obs)
        out["pose"] = self._base_pose.copy()
        return out

    def _compose_info(self, inner_info: _InfoMap) -> _Info:
        info: _Info = dict(inner_info)
        info["success"] = self._success
        info["nav_done"] = self._nav_done
        info["base_pose"] = self._base_pose.copy()
        return info
