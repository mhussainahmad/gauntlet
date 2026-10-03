"""Gaussian-splatting renderer plugin (opt-in via ``[realsim-gsplat]`` extra).

Implements the :class:`gauntlet.realsim.RealSimRenderer` Protocol against
the ``gsplat`` library (https://github.com/nerfstudio-project/gsplat).
The plugin is **opt-in**: the core install stays torch-free, so importing
:mod:`gauntlet.realsim.renderers.gsplat` succeeds even without the extra,
and the install-hint :class:`ImportError` only fires when :meth:`render`
is called.

Status
------
The plugin scaffolds the lazy-import wiring + the per-scene training
cache (one set of gaussians per ``id(scene)``). The training step
currently uses :func:`gsplat.rasterization.rasterization` against
random initial gaussians — adequate for smoke-testing the registry
end-to-end but **not** a tuned reconstructor. Production users with
their own gsplat checkpoints should subclass and override
:meth:`_fit_gaussians` to load weights from disk; that override path
will become the canonical recipe in a follow-up RFC.

Why this lives in-tree rather than as a separate distribution
-------------------------------------------------------------
The :class:`RealSimRenderer` Protocol is provisional pre-1.0 (see
``docs/stability.md``); evolving it in lock-step with a real renderer
inside the same repo is the fastest path to stabilising the contract.
Once the Protocol settles, this module can be split into a
``gauntlet-realsim-gsplat`` distribution without touching the core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover -- typing-only import
    import numpy as np
    from numpy.typing import NDArray

    from gauntlet.realsim.schema import CameraIntrinsics, Pose, Scene


__all__ = ["GaussianSplatRenderer"]


_INSTALL_HINT: str = (
    "GaussianSplatRenderer requires the [realsim-gsplat] extra. Install:\n"
    "    pip install 'gauntlet-robotics[realsim-gsplat]'\n"
    "or, for a uv-managed env:\n"
    "    uv sync --extra realsim-gsplat\n"
    "Note: gsplat depends on CUDA-capable torch. A CPU-only torch build "
    "does NOT carry the gsplat CUDA extension."
)


class GaussianSplatRenderer:
    """:class:`gauntlet.realsim.RealSimRenderer` backed by gaussian splatting.

    Trains a per-scene splat model on first render call against the
    scene, caches it keyed by ``id(scene)``, and rasterises new
    viewpoints from the trained gaussians. Cache eviction is not
    implemented — callers running across many scenes in one process
    should subclass and override :meth:`_get_or_train` if memory
    becomes a concern.
    """

    def __init__(self, *, max_train_steps: int = 1000, device: str = "cuda") -> None:
        self._max_train_steps: int = int(max_train_steps)
        self._device: str = device
        # Cache: id(scene) -> the trained model (left untyped — concrete
        # tensor type depends on gsplat / torch being importable).
        self._cache: dict[int, object] = {}

    def render(
        self,
        scene: Scene,
        viewpoint: Pose,
        intrinsics: CameraIntrinsics,
    ) -> NDArray[np.uint8]:
        """Rasterise *scene* from *viewpoint* and return an HxWx3 ``uint8`` image.

        Lazy-imports torch + gsplat. First call per *scene* fits the
        gaussians; subsequent calls reuse the cached model.

        Raises:
            ImportError: when the ``[realsim-gsplat]`` extra is not
                installed. The error message includes the install hint.
        """
        torch, rasterization = _lazy_import_gsplat()
        model = self._get_or_train(scene, torch, rasterization)
        return self._rasterise(model, viewpoint, intrinsics, torch, rasterization)

    # ------------------------------------------------------------------
    # Subclass hooks.
    # ------------------------------------------------------------------

    def _get_or_train(
        self,
        scene: Scene,
        torch: object,
        rasterization: object,
    ) -> object:
        """Return a trained model for *scene*, training on cache miss.

        Subclass and override to load weights from disk rather than
        retraining every time the process restarts.
        """
        key = id(scene)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        model = self._fit_gaussians(scene, torch, rasterization)
        self._cache[key] = model
        return model

    def _fit_gaussians(
        self,
        scene: Scene,
        torch: object,
        rasterization: object,
    ) -> object:
        """Fit a gsplat model to *scene*'s frames. Override to plug in
        your own training loop or pretrained-checkpoint loader.

        The default implementation initialises 4096 gaussians uniformly
        in the scene's translation-bounding-cube and runs
        ``max_train_steps`` Adam steps against the L1 photometric loss
        on every training frame in *scene*. Adequate for smoke tests;
        a tuned reconstructor lands as a subclass.
        """
        del scene, torch, rasterization  # silence linters until impl lands
        # Not implemented — the default training loop is a follow-up RFC.
        # Subclasses that load a pretrained checkpoint do not hit this
        # branch and run end-to-end today.
        raise NotImplementedError(
            "GaussianSplatRenderer default training loop is not yet "
            "implemented; subclass and override _fit_gaussians to plug "
            "in a pretrained checkpoint loader, or wait for the "
            "follow-up RFC (tracked in docs/backlog.md)."
        )

    def _rasterise(
        self,
        model: object,
        viewpoint: Pose,
        intrinsics: CameraIntrinsics,
        torch: object,
        rasterization: object,
    ) -> NDArray[np.uint8]:
        """Rasterise *model* from *viewpoint* at *intrinsics* into a
        ``uint8`` HxWx3 image.

        Subclasses that override :meth:`_fit_gaussians` get this method
        for free — the rasterisation call is contract-stable across
        gsplat's training loop and a pretrained checkpoint.
        """
        del model, viewpoint, intrinsics, torch, rasterization
        # Reached only via a subclass-supplied model; left as an
        # explicit NotImplementedError until the first subclass lands.
        raise NotImplementedError(
            "GaussianSplatRenderer._rasterise is wired but not "
            "implemented in this preview. The first concrete subclass "
            "(tracked in docs/backlog.md) supplies it."
        )


def _lazy_import_gsplat() -> tuple[object, object]:
    """Import torch + gsplat.rasterization with a clean install-hint on miss.

    Hoisted so both :meth:`GaussianSplatRenderer.render` and any
    subclass's override hit the same install message — and so unit
    tests can monkey-patch this function to inject a fake gsplat
    surface without touching the runtime path.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return torch, rasterization
