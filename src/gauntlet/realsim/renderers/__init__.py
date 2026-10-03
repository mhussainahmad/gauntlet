"""In-tree :class:`gauntlet.realsim.RealSimRenderer` plugins.

Importing this package registers every shipped renderer in the
:mod:`gauntlet.realsim.renderer` module-local registry. Registration is
idempotent (re-registering the same factory is a no-op) so import order
across multiple call sites does not surface as a collision error.

Public surface
--------------
* :class:`NearestFrameRenderer` — zero-dep CPU renderer. Returns the
  RGB image from the training :class:`CameraFrame` whose camera pose is
  closest to the requested viewpoint. Registered under ``nearest-frame``.
* :class:`GaussianSplatRenderer` — gaussian-splatting plugin scaffold.
  Lazy-imports the optional ``[realsim-gsplat]`` extra at first render
  call; raises a clean install-hint :class:`ImportError` if missing.
  Registered under ``gsplat``.

Third-party renderers register themselves by calling
:func:`gauntlet.realsim.register_renderer` at import time or via the
forthcoming entry-point group (a follow-up RFC, see
``src/gauntlet/realsim/renderer.py`` docstring §4.6).
"""

from __future__ import annotations

from gauntlet.realsim.renderer import register_renderer
from gauntlet.realsim.renderers.gsplat import GaussianSplatRenderer
from gauntlet.realsim.renderers.nearest_frame import NearestFrameRenderer

__all__ = [
    "GaussianSplatRenderer",
    "NearestFrameRenderer",
]


# Register the shipped renderers. ``register_renderer`` is idempotent
# under same-factory re-registration, so duplicate calls from worker
# imports do not raise.
register_renderer("nearest-frame", NearestFrameRenderer)
register_renderer("gsplat", GaussianSplatRenderer)
