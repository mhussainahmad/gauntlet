"""Isaac Sim backend subpackage — registers ``tabletop-isaac`` on import.

Single-import guard mirroring the RFC-005 PyBullet template (Isaac
Sim does not have an undeclared transitive runtime dep the way
Genesis has torch — see ``docs/phase2-rfc-009-isaac-sim-adapter.md``
§4.1).

1. Guard-import :mod:`isaacsim`. On :class:`ImportError`, re-raise
   with the canonical install hint pointing at the ``[isaac]`` extra.
2. Import :class:`IsaacSimTabletopEnv` from
   :mod:`gauntlet.env.isaac.tabletop_isaac` and call
   :func:`gauntlet.env.registry.register_env` under the
   ``"tabletop-isaac"`` key.

Honesty caveat: a successful ``import isaacsim`` does NOT guarantee
the adapter will run end-to-end. ``isaacsim`` wraps NVIDIA Omniverse
Kit which requires a CUDA-capable RTX-class GPU at runtime; the
PyPI wheel resolves on CPU-only machines but most of the runtime
surface segfaults probing for a GPU. The guard catches the absent
extra; GPU-runtime requirements surface as a separate error inside
``IsaacSimTabletopEnv.__init__``. RFC-009 §4.4 documents this.

Core code must not import this module at init time — that is the
entire point of the optional extra (``GAUNTLET_SPEC.md`` §6, "small
deps"). The Suite loader is the canonical entry point; direct
``import gauntlet.env.isaac`` is also supported for users who want
to reach into the backend class without going through a YAML.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

try:
    import isaacsim  # noqa: F401  # import for side effect (install-check)
except ImportError as exc:
    raise ImportError(
        "gauntlet.env.isaac: the [isaac] extra is not installed.\n"
        "Install with:\n"
        "    uv sync --extra isaac\n"
        "or, for a plain pip env:\n"
        "    pip install 'gauntlet[isaac]'\n"
        "Note: isaacsim requires a CUDA-capable NVIDIA RTX-class GPU "
        "at runtime; the PyPI wheel resolves on CPU-only machines but "
        "the Kit bootstrap inside __init__ will fail without a GPU. "
        "See docs/phase2-rfc-009-isaac-sim-adapter.md §4.4."
    ) from exc

from gauntlet.env.base import GauntletEnv
from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv
from gauntlet.env.registry import register_env

# IsaacSimTabletopEnv satisfies the GauntletEnv Protocol structurally;
# mypy needs the cast for the same reason TabletopEnv's registration
# does (type[T] vs Callable[..., T]).
register_env(
    "tabletop-isaac",
    cast(Callable[..., GauntletEnv], IsaacSimTabletopEnv),
)

__all__ = ["IsaacSimTabletopEnv"]
