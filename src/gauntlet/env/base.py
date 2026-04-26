"""Env-agnostic Protocol that the Runner + Suite loader dispatch through.

This module is import-cheap on purpose: no simulator imports live here, so
``gauntlet.env.base`` stays safe to import from ``gauntlet.core`` (see
``GAUNTLET_SPEC.md`` §6 "small deps"). Backend-specific modules
(``gauntlet.env.tabletop`` for MuJoCo, ``gauntlet.env.pybullet`` for PyBullet)
register themselves via :mod:`gauntlet.env.registry` and satisfy this Protocol
structurally — there is no base class to inherit from.

See ``docs/phase2-rfc-005-pybullet-adapter.md`` §3 for the full rationale.
"""

from __future__ import annotations

from typing import Any, ClassVar, NamedTuple, Protocol, runtime_checkable

import gymnasium as gym
from numpy.typing import NDArray

# Observation / action type aliases shared by every backend. Kept here (not in
# ``gauntlet.env.tabletop``) so second-party backends can type against the
# Protocol without importing MuJoCo.
#
# ``Any`` inside ``NDArray`` mirrors ``gauntlet.policy.base`` — externally
# every backend returns float64 arrays, but the policy Protocol was declared
# liberally to absorb minor dtype drift at the policy/env seam.
Observation = dict[str, NDArray[Any]]
Action = NDArray[Any]


class CameraSpec(NamedTuple):
    """Declarative description of a single render camera.

    Used by the optional ``cameras=[...]`` kwarg on backend envs (see
    :class:`GauntletEnv` and :class:`gauntlet.env.tabletop.TabletopEnv`).
    When at least one ``CameraSpec`` is supplied, the env's observation
    grows an ``obs["images"]: dict[str, NDArray[uint8]]`` mapping
    ``spec.name -> rendered frame``. The legacy ``obs["image"]`` key
    is also populated, aliased to the **first** camera's frame, so
    consumers that read the single-camera surface keep working
    unchanged. See ``docs/polish-exploration-multi-camera.md`` §2 for
    the full contract.

    Attributes
    ----------
    name:
        Key under which this camera's frame appears in
        ``obs["images"][name]``. Must be non-empty and unique within
        the supplied list. Recommended values follow LeRobot
        conventions: ``"wrist"``, ``"top"``, ``"side"``, ``"front"``.
    pose:
        ``(x, y, z, rx, ry, rz)`` — world-frame position in **metres**
        and **MuJoCo-convention XYZ Euler angles in radians**. The
        camera looks along its local ``-Z`` axis with local ``+Y`` as
        up (standard MJCF ``<camera>`` semantics). Quaternions were
        rejected because the multi-cam use case is humans authoring
        camera positions in a config file; quats are awful by hand
        (RFC §2).
    size:
        ``(H, W)`` — rendered image shape. Height-first to match the
        existing ``render_size`` convention and MuJoCo's
        ``Renderer(model, height=H, width=W)`` constructor.
    """

    name: str
    pose: tuple[float, float, float, float, float, float]
    size: tuple[int, int]


@runtime_checkable
class GauntletEnv(Protocol):
    """Structural env contract the Runner and Suite-loader dispatch through.

    Any object that exposes this surface — including but not limited to
    :class:`gauntlet.env.tabletop.TabletopEnv` — is accepted by
    :mod:`gauntlet.runner.worker`. The Protocol is intentionally a **subset**
    of :class:`gymnasium.Env`: ``render`` is deliberately omitted so state-only
    backends (first-cut PyBullet) do not need to implement it. A separate
    ``RenderableGauntletEnv`` sub-Protocol is planned for the image-rendering
    follow-up RFC.

    Behavioural contract
    --------------------
    ``set_perturbation(name, value)``
        **Queues** a named scalar on the env — it does not take effect
        immediately. Raises :class:`ValueError` for
        ``name not in type(self).AXIS_NAMES``. Per-backend validation rules
        (integer bounds on ``distractor_count``, etc.) apply on top of the
        name check.

    ``restore_baseline()``
        Makes the env **observationally** equivalent to its post-``__init__``
        state — not bit-identical internal C state (that is a MuJoCo-only
        property). Called by the Runner between episodes
        (see ``runner/worker.py::execute_one``). Does **not** clear the
        queued perturbations — those are the input to the next reset.

    ``reset(seed=...)``
        MUST, in order:

        1. call ``restore_baseline()`` internally,
        2. apply seed-driven state randomisation,
        3. apply every queued perturbation on top,
        4. clear the pending-perturbation queue.

        The Runner already relies on this ordering
        (``TabletopEnv.reset`` implements it today). Reversing steps 2 and 3
        silently changes determinism semantics; backends MUST NOT do that.
        ``seed`` is the only entropy source for a reset.

    ``close()``
        Releases any simulator-held resources. Idempotent.

    ``AXIS_NAMES``
        Class-level :class:`frozenset` of axis names this backend accepts via
        ``set_perturbation``. Exposed as a :class:`~typing.ClassVar` so the
        Suite loader can introspect axis support per backend without
        constructing the env. Every backend that implements the canonical 7
        axes declares the same frozenset.
    """

    AXIS_NAMES: ClassVar[frozenset[str]]

    observation_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Initialise a new episode and return the first observation + info.

        See the class docstring "Behavioural contract" section for the
        mandatory four-step ordering (restore baseline -> seed-driven
        randomisation -> apply queued perturbations -> clear queue) every
        backend MUST implement to satisfy the Runner's determinism
        guarantee.
        """
        ...

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one control action and return ``(obs, reward, terminated, truncated, info)``.

        The five-tuple matches :class:`gymnasium.Env`. ``info["success"]``
        is the canonical truth flag the Runner reads when materialising
        :attr:`gauntlet.runner.Episode.success`.
        """
        ...

    def set_perturbation(self, name: str, value: float) -> None:
        """Queue a named scalar perturbation for the next :meth:`reset`.

        Does NOT take effect until the following ``reset()``. Raises
        :class:`ValueError` for ``name not in type(self).AXIS_NAMES``.
        Per-backend validation (integer bounds on ``distractor_count``,
        categorical-set membership on ``object_texture``, etc.) applies
        on top of the name check.
        """
        ...

    def restore_baseline(self) -> None:
        """Make the env observationally equivalent to its post-``__init__`` state.

        Called by the Runner between episodes. Does NOT clear the
        pending-perturbation queue — those values are the input to the
        next reset. Bit-identical internal state is a MuJoCo-only
        property; the contract is *observational* equivalence.
        """
        ...

    def close(self) -> None:
        """Release simulator-held resources. Idempotent."""
        ...


@runtime_checkable
class SubtaskMilestone(Protocol):
    """Optional opt-in seam for envs that publish per-subtask credit (B-09).

    Long-horizon envs (skill-chaining: stack, pour, multi-step assembly)
    can satisfy this Protocol IN ADDITION TO :class:`GauntletEnv` to
    expose milestone credit beyond the binary
    ``info["success"]`` flag. The Runner / Report layer reads
    ``info["subtask_completion"]: list[bool]`` directly off each step's
    ``info`` dict (length must equal :attr:`n_subtasks`, monotonic
    non-decreasing — once a subtask flips to ``True`` it stays
    ``True``); :meth:`is_subtask_done` is the *introspection* hook
    suite-loaders / scripted-policy authors call when they need to
    query the per-subtask predicate from outside the rollout loop
    (e.g. to assert the canned trajectory hits every milestone).

    Why a separate Protocol from :class:`GauntletEnv`
    -------------------------------------------------
    The original env Protocol was deliberately kept as the *minimum*
    surface every backend must support (RFC-005 §3). Forcing every env
    — including state-only PyBullet, the single-step pick-and-place
    Tabletop, and the third-party plugin envs — to declare a subtask
    list would either lie (``n_subtasks=1, subtask_0=success``) or
    explode the plugin contract. Splitting milestone publication into
    an *opt-in* Protocol keeps the GauntletEnv surface stable while
    giving the long-horizon envs a typed, ``isinstance``-checkable
    way to advertise the extra capability.

    Anti-feature note (deliberately documented)
    -------------------------------------------
    Subtask milestones are inherently env-specific — what counts as
    "stacked" on ``tabletop-stack`` is not the same predicate as
    "poured" on ``tabletop-pour``. The Protocol is shape-only:
    backends own the predicates; the schema only fixes the
    ``list[bool]`` payload and the monotonic-non-decreasing rule.
    Reports / aggregation that compare across envs must not assume
    common semantics for matching indices.

    Behavioural contract
    --------------------
    ``n_subtasks``
        Class- or instance-level integer ``>= 1``. Fixed for the env
        lifetime — a backend MUST NOT vary the count between resets.
        The runner uses this to size per-cell aggregation buffers
        before the first ``step()``.

    ``is_subtask_done(idx, obs)``
        Pure predicate: does the most-recent observation satisfy the
        ``idx``-th subtask's geometric / state condition? Called by
        suite-loaders and test scaffolding from outside the rollout
        loop, so the implementation MUST NOT mutate env state. The
        live per-step ``info["subtask_completion"]`` list is the
        canonical source of truth during a rollout (it carries the
        latched-monotonic semantics); ``is_subtask_done`` is the
        un-latched predicate.

        ``idx`` MUST be in ``[0, n_subtasks)``; backends raise
        :class:`IndexError` (or :class:`ValueError`) for out-of-range
        indices. ``obs`` is the same dict the env returns from
        :meth:`GauntletEnv.step`; backends key off the standard
        observation slots they already populate.

    Future-extension caveat
    -----------------------
    A full schema migration on :class:`gauntlet.runner.Episode`
    (subtask names registry, per-subtask reward decomposition, etc.)
    is deferred to the B-09 follow-up. This Protocol is the seam
    that follow-up will plug additional fields into without churning
    the GauntletEnv contract again.
    """

    n_subtasks: int

    def is_subtask_done(self, idx: int, obs: Observation) -> bool:
        """Return whether the ``idx``-th subtask's predicate is currently satisfied.

        Pure predicate — implementations MUST NOT mutate env state.
        See the class docstring for index-range and ``obs`` semantics.
        """
        ...


__all__ = [
    "Action",
    "CameraSpec",
    "GauntletEnv",
    "Observation",
    "SubtaskMilestone",
]
