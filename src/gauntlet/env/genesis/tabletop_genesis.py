"""Genesis tabletop backend — scaffold (RFC-007 §12 step 5).

Scaffold only: declares the class, action / observation spaces, the
two canonical ``AXIS_NAMES`` / ``VISUAL_ONLY_AXES`` frozensets, and
stubs every :class:`gauntlet.env.base.GauntletEnv` method with
:class:`NotImplementedError`. Behaviour lands in follow-up commits per
RFC-007 §12 (step 6 wires :mod:`genesis` construction into ``__init__``;
step 7 adds ``reset`` / ``step`` / ``_build_obs``; steps 8-10 add
perturbations).

This scaffold commit's acceptance criterion is:

* :class:`GenesisTabletopEnv` is importable (via ``uv sync --extra
  genesis`` only — RFC-007 §2 keeps the core torch/genesis-free).
* :func:`isinstance` against :class:`~gauntlet.env.base.GauntletEnv`
  returns True (the Protocol is ``runtime_checkable``; satisfied by
  the attribute surface this module exposes).
* The Suite loader accepts ``env: tabletop-genesis`` and triggers
  subpackage import on first use (wired via
  :data:`gauntlet.suite.schema.BUILTIN_BACKEND_IMPORTS`).

Non-goals for this commit:

* Any :mod:`genesis` API call. ``__init__`` deliberately does not call
  ``gs.init`` or construct a scene — the heavy construction path lands
  in the follow-up step so the scaffold's type-check / registration
  plumbing is reviewable in isolation.
* Any observation-dict content. ``_build_obs`` lands with the full
  scene wiring.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

__all__ = ["GenesisTabletopEnv"]


# Scene / control constants mirror the MuJoCo reference values exactly
# so baseline semantics line up with TabletopEnv (numerics differ
# per-backend — RFC-005 §7.4, RFC-007 §7.3). The constants are declared
# at module scope in the scaffold so the follow-up "real body" commit
# touches only the __init__ / reset / step / perturbation code paths,
# not the constants.
_TABLE_TOP_Z: float = 0.42
_CUBE_HALF: float = 0.025
_CUBE_REST_Z: float = _TABLE_TOP_Z + _CUBE_HALF
_EE_REST_OFFSET_Z: float = 0.15
_EE_VISUAL_HALF: float = 0.01

_N_DISTRACTOR_SLOTS: int = 10


class GenesisTabletopEnv:
    """Genesis state-only tabletop pick-and-place env — scaffold.

    Satisfies :class:`gauntlet.env.base.GauntletEnv` structurally: same
    :attr:`AXIS_NAMES` as :class:`gauntlet.env.tabletop.TabletopEnv` and
    :class:`gauntlet.env.pybullet.tabletop_pybullet.PyBulletTabletopEnv`,
    same 7-D action space, same state-obs keys.

    :attr:`VISUAL_ONLY_AXES` is the four cosmetic axes
    (``lighting_intensity``, ``camera_offset_x``, ``camera_offset_y``,
    ``object_texture``) — same shape PyBullet had pre-RFC-006, deferred
    to a follow-up rendering RFC (RFC-007 §9).

    Attributes
    ----------
    MAX_LINEAR_STEP, MAX_ANGULAR_STEP, GRASP_RADIUS, TARGET_RADIUS,
    GRIPPER_OPEN, GRIPPER_CLOSED :
        Control / grasp / success constants — same values as the
        MuJoCo / PyBullet reference backends so a given action roughly
        moves the EE by the same delta across backends.
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
    # The four cosmetic axes — state-only first cut. Empties once the
    # follow-up rendering RFC (RFC-008) wires ``render_in_obs=True``
    # through :mod:`genesis` cameras, same shape RFC-006 used for
    # PyBullet.
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
        n_substeps: int = 5,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")
        if n_substeps <= 0:
            raise ValueError(f"n_substeps must be positive; got {n_substeps}")

        self._max_steps = max_steps
        self._n_substeps = n_substeps

        # Byte-compatible with TabletopEnv and PyBulletTabletopEnv (no
        # ``image`` key — state-only first cut). The follow-up rendering
        # RFC will add the ``image`` key conditionally on a
        # ``render_in_obs`` kwarg.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        obs_spaces: dict[str, gym.spaces.Space[Any]] = {
            "cube_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "cube_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "gripper": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            "target_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # Pending-perturbation queue — the RFC-005 §3.2 four-step
        # ordering (restore_baseline → randomise → apply queue →
        # clear) applies here too. The full body lands in step 7+.
        self._pending_perturbations: dict[str, float] = {}

        # Per-episode runtime state — populated by reset() in step 7.
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._target_pos: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Reset the scene. Body lands in RFC-007 §12 step 7."""
        raise NotImplementedError(
            "GenesisTabletopEnv.reset — scaffold only; follow-up commit "
            "(RFC-007 §12 step 7) wires scene construction + reset."
        )

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[dict[str, NDArray[np.float64]], float, bool, bool, dict[str, Any]]:
        """Advance one control step. Body lands in RFC-007 §12 step 7."""
        raise NotImplementedError(
            "GenesisTabletopEnv.step — scaffold only; follow-up commit "
            "(RFC-007 §12 step 7) wires the control loop."
        )

    def set_perturbation(self, name: str, value: float) -> None:
        """Queue an axis-value pair for the next reset.

        Validation lands in RFC-007 §12 step 9. The scaffold accepts
        everything so the Suite loader's dry-run path (which constructs
        a backend to inspect ``VISUAL_ONLY_AXES``) does not raise.
        """
        raise NotImplementedError(
            "GenesisTabletopEnv.set_perturbation — scaffold only; "
            "follow-up commit (RFC-007 §12 step 9) wires axis validation."
        )

    def restore_baseline(self) -> None:
        """Restore the scene to its post-__init__ state.

        Body lands with the scene-construction commit (RFC-007 §12
        step 7) since ``restore_baseline`` is called from inside
        ``reset``.
        """
        raise NotImplementedError(
            "GenesisTabletopEnv.restore_baseline — scaffold only; "
            "follow-up commit (RFC-007 §12 step 7) wires scene reset."
        )

    def close(self) -> None:
        """Release any Genesis-held resources. Idempotent.

        The scaffold has no resources to release — ``__init__`` does
        not construct a scene. This no-op satisfies the Protocol; the
        real cleanup lands in step 6 alongside the ``gs.Scene``
        construction.
        """
