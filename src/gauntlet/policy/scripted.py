"""Open-loop scripted pick-and-place policy."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

__all__ = ["DEFAULT_PICK_AND_PLACE_TRAJECTORY", "ScriptedPolicy"]


# Canonical 7-DoF pick-and-place stub trajectory:
#   [x_delta, y_delta, z_delta, rx, ry, rz, gripper]
# gripper: +1 = open, -1 = close.
# Phases: open → approach above → descend → close → lift → translate → set
# down → release. Values are nominal stubs; the tabletop env in Task 3 will
# define the exact action scaling.
DEFAULT_PICK_AND_PLACE_TRAJECTORY: NDArray[np.float64] = np.array(
    [
        [0.00, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0],
        [0.10, 0.00, 0.05, 0.0, 0.0, 0.0, 1.0],
        [0.00, 0.00, -0.08, 0.0, 0.0, 0.0, 1.0],
        [0.00, 0.00, 0.00, 0.0, 0.0, 0.0, -1.0],
        [0.00, 0.00, 0.10, 0.0, 0.0, 0.0, -1.0],
        [0.15, 0.10, 0.00, 0.0, 0.0, 0.0, -1.0],
        [0.00, 0.00, -0.05, 0.0, 0.0, 0.0, -1.0],
        [0.00, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class ScriptedPolicy:
    """Open-loop scripted trajectory playback.

    Emits actions from a pre-recorded ``(T, action_dim)`` sequence, one per
    ``act()`` call. After exhaustion the last action is held, unless
    ``loop=True``, in which case the sequence repeats. ``reset()`` rewinds
    to step 0. Observations are ignored by design — scripted policies are
    open-loop.
    """

    def __init__(
        self,
        trajectory: NDArray[np.float64] | None = None,
        *,
        loop: bool = False,
    ) -> None:
        traj = (
            DEFAULT_PICK_AND_PLACE_TRAJECTORY.copy()
            if trajectory is None
            else np.asarray(trajectory, dtype=np.float64)
        )
        if traj.ndim != 2:
            raise ValueError(f"trajectory must be 2D (steps, action_dim); got shape {traj.shape}")
        if traj.shape[0] == 0:
            raise ValueError("trajectory must have at least one step")
        self._trajectory: NDArray[np.float64] = traj
        self.loop = loop
        self._step = 0

    @property
    def action_dim(self) -> int:
        """Width of one action vector (the trajectory's column count)."""
        return int(self._trajectory.shape[1])

    @property
    def length(self) -> int:
        """Number of pre-recorded steps in the trajectory."""
        return int(self._trajectory.shape[0])

    def act(self, obs: Observation) -> Action:
        """Return the next pre-recorded action and advance the playback cursor.

        The observation is ignored — scripted playback is open-loop.
        After the trajectory is exhausted, the last frame is held
        indefinitely unless ``loop=True`` was passed to ``__init__``,
        in which case the cursor wraps around to step 0.
        """
        del obs  # open-loop: scripted policy does not read observations
        idx = self._step % self.length if self.loop else min(self._step, self.length - 1)
        self._step += 1
        # cast: numpy stubs lose the dtype parameter on __getitem__, so a
        # runtime-safe slice shows up to mypy as Any. We know the dtype.
        return cast("Action", self._trajectory[idx].copy())

    def reset(self, rng: np.random.Generator) -> None:
        """Rewind to step 0. The RNG is accepted for protocol conformance."""
        del rng  # scripted trajectory is deterministic
        self._step = 0
