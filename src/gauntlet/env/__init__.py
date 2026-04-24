"""Parameterized MuJoCo environments — see ``GAUNTLET_SPEC.md`` §3.

This package also hosts the env-agnostic registry (RFC-005 §3.4). The
MuJoCo ``tabletop`` backend registers itself at package-import time; the
PyBullet ``tabletop-pybullet`` backend lives in ``gauntlet.env.pybullet``
and registers on demand (the Suite loader imports it only when a YAML
targets that backend — step 5 / step 7 of RFC-005 §13).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from gauntlet.env.base import CameraSpec, GauntletEnv
from gauntlet.env.perturbation import AXIS_NAMES, PerturbationAxis, axis_for
from gauntlet.env.registry import register_env
from gauntlet.env.tabletop import N_DISTRACTOR_SLOTS, TabletopEnv

# TabletopEnv satisfies GauntletEnv structurally (runtime isinstance check
# in tests/test_env.py), but mypy treats ``type[TabletopEnv]`` and
# ``Callable[..., GauntletEnv]`` as incomparable. The cast documents the
# deliberate static widening at the registration seam; the runtime check
# stays intact because the registry calls the factory unchanged.
register_env(
    "tabletop",
    cast(Callable[..., GauntletEnv], TabletopEnv),
)

__all__ = [
    "AXIS_NAMES",
    "N_DISTRACTOR_SLOTS",
    "CameraSpec",
    "GauntletEnv",
    "PerturbationAxis",
    "TabletopEnv",
    "axis_for",
]
