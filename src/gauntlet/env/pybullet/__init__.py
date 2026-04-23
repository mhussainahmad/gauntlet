"""PyBullet backend subpackage — registers ``tabletop-pybullet`` on import.

Loading order:

1. Guard-import :mod:`pybullet` and :mod:`pybullet_data`. On
   :class:`ImportError` (the ``[pybullet]`` extra is not installed),
   re-raise with a clear :class:`ImportError` message carrying the
   ``uv sync --extra pybullet`` /
   ``pip install 'gauntlet[pybullet]'`` install hints. The Suite loader
   catches this and translates it into the same user-facing message
   (see :func:`gauntlet.suite.loader._ensure_backend_registered`).
2. Import :class:`PyBulletTabletopEnv` from
   :mod:`gauntlet.env.pybullet.tabletop_pybullet` and call
   :func:`gauntlet.env.registry.register_env` under the
   ``"tabletop-pybullet"`` key.

This mirrors the RFC-001/002/003 pattern (``[hf]``, ``[lerobot]``,
``[monitor]`` extras each live behind an ImportError-guarded
subpackage). Core code must not import this module at init time — that
is the entire point of the optional extra (``GAUNTLET_SPEC.md`` §6,
"small deps"). The Suite loader is the canonical entry point; direct
``import gauntlet.env.pybullet`` is also supported for users who want
to reach into the backend class without going through a YAML.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

try:
    import pybullet_data  # noqa: F401  # import for side effect (install-check)

    import pybullet  # noqa: F401  # import for side effect (install-check)
except ImportError as exc:
    raise ImportError(
        "gauntlet.env.pybullet: the [pybullet] extra is not installed.\n"
        "Install with:\n"
        "    uv sync --extra pybullet\n"
        "or, for a plain pip env:\n"
        "    pip install 'gauntlet[pybullet]'"
    ) from exc

from gauntlet.env.base import GauntletEnv
from gauntlet.env.pybullet.tabletop_pybullet import PyBulletTabletopEnv
from gauntlet.env.registry import register_env

# PyBulletTabletopEnv satisfies the GauntletEnv Protocol structurally;
# mypy needs the cast for the same reason TabletopEnv's registration
# does (type[T] vs Callable[..., T]).
register_env(
    "tabletop-pybullet",
    cast(Callable[..., GauntletEnv], PyBulletTabletopEnv),
)

__all__ = ["PyBulletTabletopEnv"]
