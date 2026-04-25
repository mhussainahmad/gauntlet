"""Perturbation axes — see ``GAUNTLET_SPEC.md`` §3 and §5 task 4.

Public surface:

* :data:`AXIS_NAMES` — canonical ordered tuple of the seven scalar axis
  names. Stable contract for downstream code (Suite YAML, Runner, Report
  all key off this).
* :class:`PerturbationAxis` — frozen data class: name + kind + sampler +
  bounds. See :mod:`gauntlet.env.perturbation.base` for the design notes.
* Per-axis constructors (:func:`lighting_intensity`, :func:`camera_offset_x`,
  …, :func:`distractor_count`) — build a :class:`PerturbationAxis` with
  optional bound overrides.
* :func:`axis_for` — registry lookup: name -> default-configured axis.

The environment side of the contract lives on
:meth:`gauntlet.env.tabletop.TabletopEnv.set_perturbation`; runners feed
the sampled value into that method before calling ``reset(seed=...)``.
"""

from __future__ import annotations

from typing import Final

from gauntlet.env.perturbation.axes import (
    DEFAULT_BOUNDS as DEFAULT_BOUNDS,
)
from gauntlet.env.perturbation.axes import (
    OBJECT_SWAP_CLASSES as OBJECT_SWAP_CLASSES,
)
from gauntlet.env.perturbation.axes import (
    axis_for as axis_for,
)
from gauntlet.env.perturbation.axes import (
    camera_extrinsics as camera_extrinsics,
)
from gauntlet.env.perturbation.axes import (
    camera_offset_x as camera_offset_x,
)
from gauntlet.env.perturbation.axes import (
    camera_offset_y as camera_offset_y,
)
from gauntlet.env.perturbation.axes import (
    color_shift_synthetic as color_shift_synthetic,
)
from gauntlet.env.perturbation.axes import (
    distractor_count as distractor_count,
)
from gauntlet.env.perturbation.axes import (
    image_attack as image_attack,
)
from gauntlet.env.perturbation.axes import (
    initial_state_ood as initial_state_ood,
)
from gauntlet.env.perturbation.axes import (
    instruction_paraphrase as instruction_paraphrase,
)
from gauntlet.env.perturbation.axes import (
    lighting_intensity as lighting_intensity,
)
from gauntlet.env.perturbation.axes import (
    object_initial_pose_x as object_initial_pose_x,
)
from gauntlet.env.perturbation.axes import (
    object_initial_pose_y as object_initial_pose_y,
)
from gauntlet.env.perturbation.axes import (
    object_swap as object_swap,
)
from gauntlet.env.perturbation.axes import (
    object_texture as object_texture,
)
from gauntlet.env.perturbation.base import (
    AXIS_KIND_CATEGORICAL as AXIS_KIND_CATEGORICAL,
)
from gauntlet.env.perturbation.base import (
    AXIS_KIND_CONTINUOUS as AXIS_KIND_CONTINUOUS,
)
from gauntlet.env.perturbation.base import (
    AXIS_KIND_INT as AXIS_KIND_INT,
)
from gauntlet.env.perturbation.base import (
    AxisKind as AxisKind,
)
from gauntlet.env.perturbation.base import (
    AxisSampler as AxisSampler,
)
from gauntlet.env.perturbation.base import (
    PerturbationAxis as PerturbationAxis,
)
from gauntlet.env.perturbation.base import (
    make_categorical_sampler as make_categorical_sampler,
)
from gauntlet.env.perturbation.base import (
    make_continuous_sampler as make_continuous_sampler,
)
from gauntlet.env.perturbation.base import (
    make_int_sampler as make_int_sampler,
)

# Canonical scalar axis names. Order is stable contract — downstream
# YAML keys, Runner cell ids, and Report axis labels all key off this.
AXIS_NAMES: Final[tuple[str, ...]] = (
    "lighting_intensity",
    "camera_offset_x",
    "camera_offset_y",
    "object_texture",
    "object_initial_pose_x",
    "object_initial_pose_y",
    "distractor_count",
    "initial_state_ood",
    "image_attack",
    "instruction_paraphrase",
    "object_swap",
    "camera_extrinsics",
    "color_shift_synthetic",
)


__all__ = [
    "AXIS_KIND_CATEGORICAL",
    "AXIS_KIND_CONTINUOUS",
    "AXIS_KIND_INT",
    "AXIS_NAMES",
    "DEFAULT_BOUNDS",
    "OBJECT_SWAP_CLASSES",
    "AxisKind",
    "AxisSampler",
    "PerturbationAxis",
    "axis_for",
    "camera_extrinsics",
    "camera_offset_x",
    "camera_offset_y",
    "color_shift_synthetic",
    "distractor_count",
    "image_attack",
    "initial_state_ood",
    "instruction_paraphrase",
    "lighting_intensity",
    "make_categorical_sampler",
    "make_continuous_sampler",
    "make_int_sampler",
    "object_initial_pose_x",
    "object_initial_pose_y",
    "object_swap",
    "object_texture",
]
