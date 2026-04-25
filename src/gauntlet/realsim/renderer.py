"""Renderer extension point for the real-to-sim stub.

Defines the structural :class:`RealSimRenderer` Protocol that a future
gaussian-splatting (or any other) renderer plugin implements, plus a
small module-local registry of factories.

The registry is intentionally **local** rather than going through
:mod:`gauntlet.plugins`. Two reasons (see RFC §4.6):

1. The :mod:`gauntlet.plugins` public surface is pinned by an existing
   test (``tests/test_plugins.py::test_module_public_surface``). Bumping
   that surface for an extension point with zero in-tree consumers is
   the wrong trade-off.
2. The realsim renderer sits at a different abstraction level from
   policy / env plugins (it owns a *scene*, not a sweep / episode), so
   bundling them in one entry-point group conflates concerns.

When the first concrete renderer ships, a follow-up RFC decides whether
to promote the registry into :mod:`gauntlet.plugins` (one new
entry-point group) or keep it local. Either way the
:class:`RealSimRenderer` Protocol stays put.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover -- typing-only import
    import numpy as np

    from gauntlet.realsim.schema import CameraIntrinsics, Pose, Scene

__all__ = [
    "RealSimRenderer",
    "RendererFactory",
    "RendererRegistryError",
    "get_renderer",
    "list_renderers",
    "register_renderer",
]


# ---------------------------------------------------------------------------
# Protocol.
# ---------------------------------------------------------------------------


@runtime_checkable
class RealSimRenderer(Protocol):
    """Structural Protocol for an in-tree or third-party scene renderer.

    A renderer takes a :class:`Scene`, a virtual camera viewpoint
    (:class:`Pose`), and the intrinsics for that viewpoint, and returns
    an HxWx3 ``uint8`` RGB image as a NumPy array.

    The Protocol is :func:`runtime_checkable` so a coordinator can
    ``isinstance(obj, RealSimRenderer)``-test a candidate object before
    invoking ``render`` — useful when wiring a renderer factory through
    the registry below without an upfront type assertion.

    Returning ``np.ndarray`` (not a ``torch.Tensor``) keeps the contract
    backend-agnostic. A torch-backed renderer converts at its own
    boundary; a CPU rasteriser used in tests doesn't pay the torch
    import cost.

    The Protocol *does not* fix exact dtype / shape — a CPU stub may
    return a tiny placeholder ``np.ndarray((0, 0, 3), dtype=np.uint8)``,
    while a real renderer returns a full-resolution image. Stronger
    invariants (dimensions match ``intrinsics.width`` / ``height``)
    belong in a downstream "renderer test harness" RFC, not here.
    """

    def render(
        self,
        scene: Scene,
        viewpoint: Pose,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:  # pragma: no cover -- structural; implementations supply behaviour
        """Render *scene* from *viewpoint* and return an HxWx3 ``uint8`` RGB image."""
        ...


#: A zero-arg factory that materialises a :class:`RealSimRenderer`.
#: Stored in the registry rather than bare instances so factories can
#: hold lazy state (model weights, GPU contexts) and skip construction
#: when the renderer is registered but never used.
RendererFactory = Callable[[], RealSimRenderer]


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


class RendererRegistryError(ValueError):
    """Raised on registry name collisions or unknown lookups.

    Subclasses :class:`ValueError` so existing CLI ``except ValueError``
    handlers (e.g. ``gauntlet.cli._fail``) catch it without a code
    change.
    """


_REGISTRY: dict[str, RendererFactory] = {}


def register_renderer(name: str, factory: RendererFactory) -> None:
    """Register *factory* under *name* in the module-local registry.

    Re-registering the same name with the *same* factory object is a
    no-op (idempotent — matches the
    :func:`gymnasium.envs.registration.register` precedent). Re-
    registering with a *different* factory raises
    :class:`RendererRegistryError`; it would otherwise silently shadow
    a previous registration, which is exactly the failure mode the
    plugin-system collision warning was designed to prevent
    (see :func:`gauntlet.plugins.warn_on_collision`).

    Args:
        name: registry key. Must be a non-empty string.
        factory: zero-argument callable returning an object that
            satisfies :class:`RealSimRenderer`. Validation is *not*
            performed here — the Protocol is structural and the
            renderer may legitimately defer construction (GPU init,
            checkpoint download) until first :meth:`render`.

    Raises:
        RendererRegistryError: on empty *name* or a colliding
            registration with a different factory.
    """
    if not isinstance(name, str) or not name:
        raise RendererRegistryError("renderer name must be a non-empty string")
    if name in _REGISTRY and _REGISTRY[name] is not factory:
        raise RendererRegistryError(
            f"renderer {name!r} is already registered with a different factory; "
            f"call unregister_renderer({name!r}) first if you mean to replace it",
        )
    _REGISTRY[name] = factory


def get_renderer(name: str) -> RealSimRenderer:
    """Materialise the renderer registered under *name*.

    The factory is invoked once per call — callers that want a singleton
    should cache the returned instance themselves. Renderers are not
    process-shared by default (the registry holds factories, not
    instances) so a multi-process sweep can spin up fresh GPU contexts
    per worker without a cross-process handle dance.

    Args:
        name: registry key. Must have been previously registered via
            :func:`register_renderer`.

    Returns:
        A fresh renderer instance.

    Raises:
        RendererRegistryError: when *name* is not registered. The
            message lists the registered names so the user can spot
            typos.
    """
    if name not in _REGISTRY:
        raise RendererRegistryError(
            f"unknown renderer {name!r}; registered: {sorted(_REGISTRY) or '[]'}",
        )
    return _REGISTRY[name]()


def list_renderers() -> list[str]:
    """Return the sorted list of currently-registered renderer names.

    Useful for CLI ``--help`` listings and for plugin authors verifying
    their factory landed in the registry. The returned list is a fresh
    copy — mutating it does not affect the registry.
    """
    return sorted(_REGISTRY)


def _clear_registry_for_tests() -> None:
    """Internal: drop every registration. Tests use this to isolate.

    Underscore-prefixed so it does not appear in :data:`__all__`. Call
    in test fixture teardown after ``register_renderer`` mutations to
    avoid leaking state into other tests.
    """
    _REGISTRY.clear()
