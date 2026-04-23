"""Runtime drift detector — RFC 003.

The public surface is bimodal:

* **Torch-free**: :class:`PerEpisodeDrift`, :class:`DriftReport`, and
  :func:`action_entropy` are re-exported eagerly. They only need
  numpy + pydantic, so ``from gauntlet.monitor import DriftReport``
  works on the default (no-extras) install.
* **Torch-backed**: :class:`StateAutoencoder`, :func:`train_ae`, and
  :func:`score_drift` are lazily resolved via module-level
  ``__getattr__``. On a torch-free install the import path fires only
  when the user asks for them, and raises a clean :class:`ImportError`
  that points at ``uv sync --extra monitor``.

This keeps ``import gauntlet.monitor`` cheap and safe on the default
install path while still letting users write
``from gauntlet.monitor import StateAutoencoder`` once they've opted in
to the ``[monitor]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gauntlet.monitor.entropy import ActionEntropyStats as ActionEntropyStats
from gauntlet.monitor.entropy import action_entropy as action_entropy
from gauntlet.monitor.schema import DriftReport as DriftReport
from gauntlet.monitor.schema import PerEpisodeDrift as PerEpisodeDrift

if TYPE_CHECKING:
    # These symbols exist only when torch is installed. The stubs here
    # make static type-checkers happy regardless of whether the runtime
    # extras are present.
    from gauntlet.monitor.ae import StateAutoencoder as StateAutoencoder
    from gauntlet.monitor.score import score_drift as score_drift
    from gauntlet.monitor.train import train_ae as train_ae

__all__ = [
    "ActionEntropyStats",
    "DriftReport",
    "PerEpisodeDrift",
    "StateAutoencoder",
    "action_entropy",
    "score_drift",
    "train_ae",
]


_MONITOR_INSTALL_HINT = (
    "gauntlet.monitor.{attr} requires the 'monitor' extra. Install with:\n"
    "    uv sync --extra monitor\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[monitor]'"
)


# Lazy re-exports for the torch-backed symbols. Importing
# ``gauntlet.monitor.ae`` / ``.train`` / ``.score`` raises the
# install-hint ``ImportError`` when torch is missing; routing those
# imports through ``__getattr__`` keeps a bare ``import gauntlet.monitor``
# torch-free.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "StateAutoencoder": ("gauntlet.monitor.ae", "StateAutoencoder"),
    "train_ae": ("gauntlet.monitor.train", "train_ae"),
    "score_drift": ("gauntlet.monitor.score", "score_drift"),
}


def __getattr__(name: str) -> Any:
    """Route torch-requiring symbols to their lazy-import module."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        try:
            import importlib

            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(_MONITOR_INSTALL_HINT.format(attr=name)) from exc
        return getattr(module, attr_name)
    raise AttributeError(f"module 'gauntlet.monitor' has no attribute {name!r}")
