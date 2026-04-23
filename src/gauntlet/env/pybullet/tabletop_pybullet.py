"""PyBullet tabletop backend — stub (RFC-005 §5, step 9 fills the body).

This module lands in step 7 of the RFC-005 §13 checklist so the
subpackage ``__init__.py`` has a concrete class to register under the
``tabletop-pybullet`` key. Every method currently raises
:class:`NotImplementedError` — step 9 swaps the stubs for the real
constraint-based mocap EE, scene construction, deterministic solver
setup, and `_build_obs`/`_snap_cube_to_ee` logic.

The class already declares the two :class:`~typing.ClassVar` attributes
that the Suite loader and Protocol introspection need:

* :data:`AXIS_NAMES` — the canonical 7 perturbation axes (parity with
  :class:`gauntlet.env.tabletop.TabletopEnv`).
* :data:`VISUAL_ONLY_AXES` — ``{lighting_intensity, object_texture}``
  for state-only obs (RFC-005 §6.2). The Suite loader will reject
  purely-cosmetic sweeps against this set in step 12.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

__all__ = ["PyBulletTabletopEnv"]


_STUB_MESSAGE = (
    "PyBulletTabletopEnv stub — RFC-005 §13 step 7 ships this placeholder so "
    "the subpackage import + registry round-trip can be tested; step 9 lands "
    "the real backend body. Do not construct this class in tests that are "
    "not already marked @pytest.mark.pybullet."
)


class PyBulletTabletopEnv:
    """Placeholder — the real class body lands in step 9.

    Declares the class-level metadata (:data:`AXIS_NAMES`,
    :data:`VISUAL_ONLY_AXES`) that the registry, Suite loader, and
    Protocol introspection already depend on. Instance methods raise
    :class:`NotImplementedError` so a misconfigured suite cannot
    silently run against the stub.
    """

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
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset(
        {"lighting_intensity", "object_texture"}
    )

    observation_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]

    def __init__(self) -> None:
        # Spaces are populated so static callers (e.g. mypy + the
        # Runner's in-process fast path) can at least introspect the
        # shape without constructing the full scene. The behavioural
        # guarantees only land in step 9.
        self.observation_space = spaces.Dict(
            {
                "cube_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "cube_quat": spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float64
                ),
                "ee_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "gripper": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float64
                ),
                "target_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float64
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        raise NotImplementedError(_STUB_MESSAGE)

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[
        dict[str, NDArray[np.float64]], float, bool, bool, dict[str, Any]
    ]:
        raise NotImplementedError(_STUB_MESSAGE)

    def set_perturbation(self, name: str, value: float) -> None:
        raise NotImplementedError(_STUB_MESSAGE)

    def restore_baseline(self) -> None:
        raise NotImplementedError(_STUB_MESSAGE)

    def close(self) -> None:
        return None
