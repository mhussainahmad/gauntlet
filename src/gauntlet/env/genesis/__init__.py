"""Genesis backend subpackage — registers ``tabletop-genesis`` on import.

Two-part load guard — mirrors the RFC-005 PyBullet template but extends
it to cover the **two** possible missing dependencies unique to Genesis
(see ``docs/phase2-rfc-007-genesis-adapter.md`` §4.1):

1. Guard-import :mod:`torch`. Genesis imports torch at package scope
   but does NOT declare it in ``install_requires`` (upstream delegates
   the CPU-vs-CUDA wheel choice to users), so a fresh
   ``pip install genesis-world`` without an explicit torch install
   gives :class:`ModuleNotFoundError` at :mod:`genesis` import. The
   ``[genesis]`` extra in ``pyproject.toml`` declares both
   ``genesis-world`` and ``torch`` — this guard mirrors that contract
   so the install-hint error names the extra the user needs, not the
   individual sub-dep.
2. Guard-import :mod:`genesis`. On :class:`ImportError` (the
   ``[genesis]`` extra itself is not installed), re-raise with the
   same ``uv sync --extra genesis`` / ``pip install 'gauntlet[genesis]'``
   install hints.
3. Import :class:`GenesisTabletopEnv` from
   :mod:`gauntlet.env.genesis.tabletop_genesis` and call
   :func:`gauntlet.env.registry.register_env` under the
   ``"tabletop-genesis"`` key.

This mirrors the RFC-001/002/003/005 pattern (``[hf]``, ``[lerobot]``,
``[monitor]``, ``[pybullet]`` extras each live behind an
ImportError-guarded subpackage). Core code must not import this module
at init time — that is the entire point of the optional extra
(``GAUNTLET_SPEC.md`` §6, "small deps"). The Suite loader is the
canonical entry point; direct ``import gauntlet.env.genesis`` is also
supported for users who want to reach into the backend class without
going through a YAML.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

# Torch is imported before Genesis because the latter imports the former
# at package scope — failing here gives the user a targeted error
# pointing at the extras name, not a surprise ``ModuleNotFoundError``
# from deep inside a third-party package.
try:
    import torch  # noqa: F401  # import for side effect (install-check)
except ImportError as exc:
    raise ImportError(
        "gauntlet.env.genesis: torch is required by genesis-world but is "
        "not installed.\n"
        "The [genesis] extra declares both genesis-world and torch; "
        "install with:\n"
        "    uv sync --extra genesis\n"
        "or, for a plain pip env:\n"
        "    pip install 'gauntlet[genesis]'"
    ) from exc

try:
    import genesis  # noqa: F401  # import for side effect (install-check)
except ImportError as exc:
    raise ImportError(
        "gauntlet.env.genesis: the [genesis] extra is not installed.\n"
        "Install with:\n"
        "    uv sync --extra genesis\n"
        "or, for a plain pip env:\n"
        "    pip install 'gauntlet[genesis]'"
    ) from exc

from gauntlet.env.base import GauntletEnv
from gauntlet.env.genesis.tabletop_genesis import GenesisTabletopEnv
from gauntlet.env.registry import register_env

# GenesisTabletopEnv satisfies the GauntletEnv Protocol structurally;
# mypy needs the cast for the same reason TabletopEnv's registration
# does (type[T] vs Callable[..., T]).
register_env(
    "tabletop-genesis",
    cast(Callable[..., GauntletEnv], GenesisTabletopEnv),
)

__all__ = ["GenesisTabletopEnv"]
