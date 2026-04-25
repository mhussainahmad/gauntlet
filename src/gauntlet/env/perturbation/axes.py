"""Concrete perturbation axes for the tabletop env.

See ``GAUNTLET_SPEC.md`` §5 task 4. Five logical perturbation categories
expand into seven base scalar axes plus one OOD-shift axis plus one
post-render image-attack axis:

* :func:`lighting_intensity` — single scalar (one-channel grayscale level).
* :func:`camera_offset_x` / :func:`camera_offset_y` — two scalars covering
  the ``CameraOffset`` category.
* :func:`object_texture` — categorical 0/1 swap of cube colour.
* :func:`object_initial_pose_x` / :func:`object_initial_pose_y` — two
  scalars covering the in-distribution ``ObjectInitialPose`` category.
* :func:`distractor_count` — non-negative integer in [0, 10].
* :func:`initial_state_ood` — OOD sigma multiplier (B-32). The axis value
  is a unitless multiplier ``V``; the env interprets it as a displacement
  of ``V * prior_std * sign_per_dim`` from ``prior_mean`` (LIBERO-PRO /
  LIBERO-Plus framing — see backlog B-32). The prior is configured on the
  env via :meth:`gauntlet.env.tabletop.TabletopEnv.set_initial_state_ood_prior`.
* :func:`image_attack` — categorical, integer-coded sensor-attack id
  (B-31). Operates **post-render** via
  :class:`gauntlet.env.image_attack.ImageAttackWrapper`; backends do not
  consume this axis directly. See that module for the legal id set.

Each constructor returns a :class:`PerturbationAxis` with sensible
default bounds. Callers that load axes from YAML (Task 5) override the
bounds via the keyword arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, TypeAlias

from gauntlet.env.perturbation.base import (
    AXIS_KIND_CATEGORICAL,
    AXIS_KIND_CONTINUOUS,
    AXIS_KIND_INT,
    PerturbationAxis,
    make_categorical_sampler,
    make_continuous_sampler,
    make_int_sampler,
)

# Zero-arg factory producing a default-configured axis. Used by the
# ``axis_for`` registry below so callers can iterate all 7 axes by name.
AxisCtor: TypeAlias = Callable[[], PerturbationAxis]

__all__ = [
    "DEFAULT_BOUNDS",
    "axis_for",
    "camera_offset_x",
    "camera_offset_y",
    "distractor_count",
    "image_attack",
    "initial_state_ood",
    "lighting_intensity",
    "object_initial_pose_x",
    "object_initial_pose_y",
    "object_texture",
]


# Default sampling bounds per axis. Phase 1 chooses ranges that exercise
# the perturbation surface without leaving the env in a physically
# unrealistic state (cube on the table, camera roughly facing the scene).
# ``initial_state_ood`` bounds are unitless sigma multipliers — 0 means
# "at the prior mean" and larger magnitudes push further into the OOD
# tail. A ``5`` ceiling is generous; in practice users supply their own
# values via the YAML categorical shape.
DEFAULT_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "lighting_intensity": (0.3, 1.5),
    "camera_offset_x": (-0.1, 0.1),
    "camera_offset_y": (-0.1, 0.1),
    "object_texture": (0.0, 1.0),  # categorical
    "object_initial_pose_x": (-0.15, 0.15),
    "object_initial_pose_y": (-0.15, 0.15),
    "distractor_count": (0.0, 10.0),  # int
    "initial_state_ood": (0.0, 5.0),  # unitless sigma multiplier
    # B-31 — categorical attack id; legal ids enumerated by
    # ``gauntlet.env.image_attack.ATTACK_IDS``. Range is informational
    # for the categorical sampler; the wrapper rounds to the nearest
    # legal id at apply time and rejects anything else.
    "image_attack": (0.0, 5.0),
}


def lighting_intensity(*, low: float | None = None, high: float | None = None) -> PerturbationAxis:
    """Diffuse light intensity (single scalar applied to RGB equally)."""
    lo, hi = _resolve_bounds("lighting_intensity", low, high)
    return PerturbationAxis(
        name="lighting_intensity",
        kind=AXIS_KIND_CONTINUOUS,
        sampler=make_continuous_sampler(lo, hi),
        low=lo,
        high=hi,
    )


def camera_offset_x(*, low: float | None = None, high: float | None = None) -> PerturbationAxis:
    """Delta from baseline camera X position (metres)."""
    lo, hi = _resolve_bounds("camera_offset_x", low, high)
    return PerturbationAxis(
        name="camera_offset_x",
        kind=AXIS_KIND_CONTINUOUS,
        sampler=make_continuous_sampler(lo, hi),
        low=lo,
        high=hi,
    )


def camera_offset_y(*, low: float | None = None, high: float | None = None) -> PerturbationAxis:
    """Delta from baseline camera Y position (metres)."""
    lo, hi = _resolve_bounds("camera_offset_y", low, high)
    return PerturbationAxis(
        name="camera_offset_y",
        kind=AXIS_KIND_CONTINUOUS,
        sampler=make_continuous_sampler(lo, hi),
        low=lo,
        high=hi,
    )


def object_texture() -> PerturbationAxis:
    """Categorical cube colour swap (0 = baseline red, 1 = alternate green).

    No keyword arguments — the choice set is intrinsic to the asset.
    """
    lo, hi = DEFAULT_BOUNDS["object_texture"]
    return PerturbationAxis(
        name="object_texture",
        kind=AXIS_KIND_CATEGORICAL,
        sampler=make_categorical_sampler((0.0, 1.0)),
        low=lo,
        high=hi,
    )


def object_initial_pose_x(
    *, low: float | None = None, high: float | None = None
) -> PerturbationAxis:
    """Absolute initial cube X position (metres, table-relative)."""
    lo, hi = _resolve_bounds("object_initial_pose_x", low, high)
    return PerturbationAxis(
        name="object_initial_pose_x",
        kind=AXIS_KIND_CONTINUOUS,
        sampler=make_continuous_sampler(lo, hi),
        low=lo,
        high=hi,
    )


def object_initial_pose_y(
    *, low: float | None = None, high: float | None = None
) -> PerturbationAxis:
    """Absolute initial cube Y position (metres, table-relative)."""
    lo, hi = _resolve_bounds("object_initial_pose_y", low, high)
    return PerturbationAxis(
        name="object_initial_pose_y",
        kind=AXIS_KIND_CONTINUOUS,
        sampler=make_continuous_sampler(lo, hi),
        low=lo,
        high=hi,
    )


def initial_state_ood(*, low: float | None = None, high: float | None = None) -> PerturbationAxis:
    """Initial-state OOD shift axis (B-32).

    The axis value ``V`` is a *unitless sigma multiplier*. Per the
    LIBERO-PRO (arXiv 2510.03827) and LIBERO-Plus (arXiv 2510.13626)
    framing, the env interprets ``V`` as a per-dim displacement of
    ``V * prior_std * sign`` from ``prior_mean``, where ``sign`` is
    drawn deterministically per-dim from the env's seed (see
    :meth:`gauntlet.env.tabletop.TabletopEnv._apply_one_perturbation`).

    Convention: ``V == 0`` collapses to the prior mean (in-distribution
    centre); ``V == 1`` lands at the 1-sigma boundary; ``V >= 2``
    probes the OOD tail. The prior itself (``prior_mean`` and
    ``prior_std``) is configured on the env via
    :meth:`gauntlet.env.tabletop.TabletopEnv.set_initial_state_ood_prior`
    or via the YAML ``prior_mean`` / ``prior_std`` axis fields.
    """
    lo, hi = _resolve_bounds("initial_state_ood", low, high)
    return PerturbationAxis(
        name="initial_state_ood",
        kind=AXIS_KIND_CONTINUOUS,
        sampler=make_continuous_sampler(lo, hi),
        low=lo,
        high=hi,
    )


def image_attack() -> PerturbationAxis:
    """Categorical post-render image-attack axis (B-31).

    Values are integer-coded attack ids — see
    :data:`gauntlet.env.image_attack.ATTACK_IDS` for the legal set
    (``0`` = none / baseline, ``1`` = gaussian-low, ..., ``5`` =
    dropout-one-camera). The float form survives the suite YAML
    ``values:`` shape and is rounded back to int by the wrapper.

    The axis is *backend-agnostic*: the inner :class:`GauntletEnv`
    backends do not consume it. Apply only via
    :class:`gauntlet.env.image_attack.ImageAttackWrapper` which
    intercepts ``set_perturbation("image_attack", ...)`` and operates
    on the rendered ``obs["image"]`` / ``obs["images"][name]`` arrays.
    Routing into a raw backend env will raise ``ValueError`` from the
    backend's ``set_perturbation`` since ``"image_attack"`` is not in
    its ``AXIS_NAMES`` ClassVar.
    """
    # Lazy import: image_attack pulls Pillow only at JPEG-attack apply
    # time; the registry import path stays headless and dep-free.
    from gauntlet.env.image_attack import ATTACK_IDS

    lo, hi = DEFAULT_BOUNDS["image_attack"]
    return PerturbationAxis(
        name="image_attack",
        kind=AXIS_KIND_CATEGORICAL,
        sampler=make_categorical_sampler(tuple(float(i) for i in ATTACK_IDS)),
        low=lo,
        high=hi,
    )


def distractor_count(*, low: int | None = None, high: int | None = None) -> PerturbationAxis:
    """Number of visible distractor objects (integer, [0, 10])."""
    lo_f, hi_f = _resolve_bounds("distractor_count", low, high)
    lo, hi = int(lo_f), int(hi_f)
    if lo < 0 or hi > 10:
        raise ValueError(f"distractor_count bounds must lie within [0, 10]; got [{lo}, {hi}]")
    return PerturbationAxis(
        name="distractor_count",
        kind=AXIS_KIND_INT,
        sampler=make_int_sampler(lo, hi),
        low=float(lo),
        high=float(hi),
    )


# Registry: axis_name -> zero-arg constructor producing the default axis.
# Lets callers (tests, runner, YAML loader) iterate over all 7 axes
# without hard-coding the list.
_DEFAULT_CONSTRUCTORS: Final[dict[str, AxisCtor]] = {
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


def axis_for(name: str) -> PerturbationAxis:
    """Build an axis with default bounds for the given canonical name.

    Raises:
        ValueError: if ``name`` is not one of the registered axis names.
    """
    try:
        ctor = _DEFAULT_CONSTRUCTORS[name]
    except KeyError as exc:
        raise ValueError(f"unknown perturbation axis name: {name!r}") from exc
    return ctor()


# ----- internals -------------------------------------------------------------


def _resolve_bounds(
    name: str,
    low: float | int | None,
    high: float | int | None,
) -> tuple[float, float]:
    """Apply defaults from :data:`DEFAULT_BOUNDS` for any unset bound."""
    default_lo, default_hi = DEFAULT_BOUNDS[name]
    lo = float(low) if low is not None else default_lo
    hi = float(high) if high is not None else default_hi
    if lo > hi:
        raise ValueError(f"{name}: low must be <= high; got low={lo}, high={hi}")
    return lo, hi
