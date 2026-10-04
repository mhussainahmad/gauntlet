"""Gaussian-splatting renderer (opt-in via the ``[realsim-gsplat]`` extra).

Implements :class:`gauntlet.realsim.RealSimRenderer`: on the first
:meth:`GaussianSplatRenderer.render` call for a scene it fits a set of 3D
gaussians to the scene's frames, caches them, and rasterises any
requested viewpoint from that fit.

Smoke-test reconstructor
------------------------
The default fit is deliberately small: a few thousand isotropic-ish
gaussians, uniform initialisation, plain Adam on an L1 photometric loss,
no densification or pruning, no SH view-dependence. It is enough to
check that a scene, its poses and its intrinsics are consistent and to
get a recognisable novel view. It is **not** a production
reconstructor; benchmark numbers against it say nothing about gaussian
splatting. Pipelines with their own trained splats should subclass and
override :meth:`GaussianSplatRenderer._fit_gaussians` to load them;
:meth:`_rasterise` then renders them unchanged.

Backends
--------
``backend="gsplat"`` uses ``gsplat.rasterization`` (CUDA). It compiles
its kernels on first use, which needs a CUDA toolkit (``nvcc``).
``backend="torch"`` uses :func:`rasterize_torch`, a dense PyTorch
implementation of the same EWA splatting model that runs on CPU or GPU.
It is O(gaussians x pixels), so keep fits small. ``backend="auto"``
(default) tries gsplat with a tiny probe call and falls back to torch on
*any* failure, including a missing compiler at first call, not just a
missing import.

Camera convention
-----------------
:class:`~gauntlet.realsim.schema.Pose` is camera-to-world. By default it
is read in the NeRFStudio / OpenGL convention (x right, y up, the camera
looking down -z), as in ``transforms.json`` exports. Pass
``camera_convention="opencv"`` for COLMAP-style poses (x right, y down,
looking down +z). Rasterisation itself is done in the OpenCV convention,
which is what gsplat expects.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover -- typing-only import
    from gauntlet.realsim.schema import CameraIntrinsics, Pose, Scene


__all__ = ["GaussianSplatRenderer", "SplatModel", "rasterize_torch"]


_INSTALL_HINT: str = (
    "GaussianSplatRenderer needs PyTorch and Pillow. Install:\n"
    "    pip install 'gauntlet-robotics[realsim-gsplat]'\n"
    "or, for a uv-managed env:\n"
    "    uv sync --extra realsim-gsplat\n"
    "The extra also pulls gsplat for the CUDA fast path; the pure-PyTorch "
    "backend (backend='torch') works without it."
)

_SMOKE_TEST_WARNING: str = (
    "GaussianSplatRenderer's default fit is a smoke-test reconstructor "
    "(small gaussian count, no densification, L1 only). Use it to sanity-check "
    "a scene, not to judge reconstruction quality; override _fit_gaussians "
    "to load a trained model."
)

# OpenGL camera axes -> OpenCV camera axes (flip y and z).
_GL_TO_CV: NDArray[np.float64] = np.diag([1.0, -1.0, -1.0, 1.0])

Backend = Literal["auto", "gsplat", "torch"]
CameraConvention = Literal["opengl", "opencv"]


class SplatModel:
    """A fitted set of gaussians, in raw (pre-activation) parameters.

    Attributes:
        means: ``[N, 3]`` world-space centres.
        quats: ``[N, 4]`` rotations, ``wxyz``, not necessarily normalised.
        log_scales: ``[N, 3]`` log of per-axis standard deviations.
        opacity_logits: ``[N]`` pre-sigmoid opacities.
        color_logits: ``[N, 3]`` pre-sigmoid RGB in ``[0, 1]``.
        background: ``[3]`` RGB composited behind the gaussians.
    """

    def __init__(
        self,
        means: Any,
        quats: Any,
        log_scales: Any,
        opacity_logits: Any,
        color_logits: Any,
        background: Any,
    ) -> None:
        self.means = means
        self.quats = quats
        self.log_scales = log_scales
        self.opacity_logits = opacity_logits
        self.color_logits = color_logits
        self.background = background

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])


def _quat_to_rotmat(torch: Any, quats: Any) -> Any:
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def rasterize_torch(
    means: Any,
    quats: Any,
    scales: Any,
    opacities: Any,
    colors: Any,
    viewmat: Any,
    K: Any,
    width: int,
    height: int,
    *,
    background: Any = None,
    near_plane: float = 0.01,
    eps2d: float = 0.3,
    pixel_chunk: int = 4096,
) -> Any:
    """Render gaussians to an ``[H, W, 3]`` image with plain PyTorch.

    Same model and argument conventions as ``gsplat.rasterization`` for
    a single camera: activated ``scales`` / ``opacities`` / ``colors``,
    ``wxyz`` quaternions, ``viewmat`` world-to-camera in the OpenCV
    convention, ``K`` the 3x3 pinhole matrix, and a 2D dilation of
    ``eps2d`` pixels added to each projected covariance. Gaussians are
    alpha-composited front to back. Differentiable; dense over
    gaussians x pixels, evaluated ``pixel_chunk`` pixels at a time.
    """
    import torch

    device, dtype = means.device, means.dtype
    rot = viewmat[:3, :3]
    p_cam = means @ rot.T + viewmat[:3, 3]
    z = p_cam[:, 2]
    keep = z > near_plane
    if background is None:
        background = torch.zeros(3, device=device, dtype=dtype)
    if not bool(keep.any()):
        return background.expand(height, width, 3).clone()

    p_cam, z = p_cam[keep], z[keep]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * p_cam[:, 0] / z + cx
    v = fy * p_cam[:, 1] / z + cy

    r = _quat_to_rotmat(torch, quats[keep])
    s = scales[keep]
    cov_world = r @ torch.diag_embed(s * s) @ r.transpose(-1, -2)
    cov_cam = rot @ cov_world @ rot.T
    zeros = torch.zeros_like(z)
    jac = torch.stack(
        [
            torch.stack([fx / z, zeros, -fx * p_cam[:, 0] / (z * z)], dim=-1),
            torch.stack([zeros, fy / z, -fy * p_cam[:, 1] / (z * z)], dim=-1),
        ],
        dim=-2,
    )
    cov2d = jac @ cov_cam @ jac.transpose(-1, -2)
    a = cov2d[:, 0, 0] + eps2d
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + eps2d
    det = (a * c - b * b).clamp_min(1e-12)
    inv_a, inv_b, inv_c = c / det, -b / det, a / det

    order = torch.argsort(z)
    u, v = u[order], v[order]
    inv_a, inv_b, inv_c = inv_a[order], inv_b[order], inv_c[order]
    opac = opacities[keep][order]
    col = colors[keep][order]

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    px, py = xs.reshape(-1), ys.reshape(-1)
    out = []
    for start in range(0, px.shape[0], pixel_chunk):
        dx = px[start : start + pixel_chunk, None] - u[None, :]
        dy = py[start : start + pixel_chunk, None] - v[None, :]
        power = -0.5 * (inv_a * dx * dx + 2.0 * inv_b * dx * dy + inv_c * dy * dy)
        alpha = (opac * torch.exp(power.clamp(max=0.0))).clamp(max=0.99)
        alpha = torch.where(alpha < 1.0 / 255.0, torch.zeros_like(alpha), alpha)
        trans = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1]], dim=1), dim=1
        )
        weights = alpha * trans
        rgb = weights @ col
        rgb = rgb + (1.0 - weights.sum(dim=1, keepdim=True)) * background
        out.append(rgb)
    return torch.cat(out, dim=0).reshape(height, width, 3)


class GaussianSplatRenderer:
    """:class:`gauntlet.realsim.RealSimRenderer` backed by gaussian splatting.

    Args:
        scene_root: directory :attr:`CameraFrame.path` is resolved
            against. Defaults to the process cwd, matching
            :class:`~gauntlet.realsim.renderers.NearestFrameRenderer`.
        max_train_steps: Adam steps for the default fit.
        num_gaussians: gaussians in the default fit.
        train_max_side: training frames are downsampled so their longer
            side is at most this many pixels (intrinsics scaled to match).
        backend: ``"auto"``, ``"gsplat"`` or ``"torch"`` (see module docs).
        device: ``"auto"`` (CUDA when available), ``"cuda"`` or ``"cpu"``.
        camera_convention: how to read :class:`Pose` matrices.
        seed: seed for the initialisation and frame sampling.

    Fits are cached per scene object (``id(scene)``); there is no
    eviction. Override :meth:`_get_or_train` to change that.
    """

    def __init__(
        self,
        *,
        scene_root: Path | str | None = None,
        max_train_steps: int = 1000,
        num_gaussians: int = 4096,
        train_max_side: int = 64,
        backend: Backend = "auto",
        device: str = "auto",
        camera_convention: CameraConvention = "opengl",
        seed: int = 0,
    ) -> None:
        if camera_convention not in ("opengl", "opencv"):
            raise ValueError(
                f"camera_convention must be 'opengl' or 'opencv'; got {camera_convention!r}"
            )
        if backend not in ("auto", "gsplat", "torch"):
            raise ValueError(f"backend must be 'auto', 'gsplat' or 'torch'; got {backend!r}")
        self._scene_root = Path(scene_root) if scene_root is not None else Path.cwd()
        self._max_train_steps = int(max_train_steps)
        self._num_gaussians = int(num_gaussians)
        self._train_max_side = int(train_max_side)
        self._backend_request: Backend = backend
        self._device_request = device
        self._convention: CameraConvention = camera_convention
        self._seed = int(seed)
        self._cache: dict[int, SplatModel] = {}
        self._resolved: tuple[str, Any] | None = None
        self.last_backend: str | None = None
        """Backend actually used by the most recent render ("gsplat" / "torch")."""

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def render(
        self,
        scene: Scene,
        viewpoint: Pose,
        intrinsics: CameraIntrinsics,
    ) -> NDArray[np.uint8]:
        """Rasterise *scene* from *viewpoint*; returns ``[H, W, 3]`` ``uint8``.

        The first call per scene fits the gaussians (slow); later calls
        reuse the fit.

        Raises:
            ImportError: torch or Pillow is missing (message carries the
                install hint).
            ValueError: the scene has no frames to fit.
        """
        torch = _lazy_import_torch()
        model = self._get_or_train(scene, torch)
        return self._rasterise(model, viewpoint, intrinsics, torch)

    # ------------------------------------------------------------------
    # Subclass hooks.
    # ------------------------------------------------------------------

    def _get_or_train(self, scene: Scene, torch: Any) -> SplatModel:
        key = id(scene)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        model = self._fit_gaussians(scene, torch)
        self._cache[key] = model
        return model

    def _fit_gaussians(self, scene: Scene, torch: Any) -> SplatModel:
        """Fit gaussians to *scene*'s frames (smoke-test quality; see module docs)."""
        if not scene.frames:
            raise ValueError("GaussianSplatRenderer: scene has zero frames; nothing to fit")
        warnings.warn(_SMOKE_TEST_WARNING, UserWarning, stacklevel=3)
        device = self._device(torch)
        gen = torch.Generator(device="cpu").manual_seed(self._seed)

        views = []
        for frame in scene.frames:
            intr = scene.intrinsics[frame.intrinsics_id]
            image = _load_image(self._scene_root / frame.path)
            image, k = _downsample(image, intr, self._train_max_side)
            views.append(
                (
                    torch.as_tensor(image, dtype=torch.float32, device=device) / 255.0,
                    torch.as_tensor(k, dtype=torch.float32, device=device),
                    torch.as_tensor(self._viewmat(frame.pose), dtype=torch.float32, device=device),
                )
            )

        centres = np.array([self._camera_to_world_cv(f.pose)[:3, 3] for f in scene.frames])
        forwards = np.array([self._camera_to_world_cv(f.pose)[:3, 2] for f in scene.frames])
        params = _init_params(torch, gen, centres, forwards, self._num_gaussians)
        background = torch.stack([v[0].reshape(-1, 3).median(dim=0).values for v in views]).mean(0)
        tensors = {name: value.to(device).requires_grad_(True) for name, value in params.items()}
        extent = float(params["extent"])
        tensors.pop("extent")
        optimiser = torch.optim.Adam(
            [
                {"params": [tensors["means"]], "lr": 1e-2 * extent},
                {"params": [tensors["quats"]], "lr": 1e-2},
                {"params": [tensors["log_scales"]], "lr": 5e-3},
                {"params": [tensors["opacity_logits"]], "lr": 5e-2},
                {"params": [tensors["color_logits"]], "lr": 5e-2},
            ]
        )
        rasterize = self._rasterizer(torch)
        order = torch.randint(0, len(views), (self._max_train_steps,), generator=gen)
        for step in range(self._max_train_steps):
            image, k, viewmat = views[int(order[step])]
            pred = rasterize(
                tensors["means"],
                tensors["quats"],
                torch.exp(tensors["log_scales"]),
                torch.sigmoid(tensors["opacity_logits"]),
                torch.sigmoid(tensors["color_logits"]),
                viewmat,
                k,
                int(image.shape[1]),
                int(image.shape[0]),
                background,
            )
            loss = (pred - image).abs().mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

        return SplatModel(
            means=tensors["means"].detach(),
            quats=tensors["quats"].detach(),
            log_scales=tensors["log_scales"].detach(),
            opacity_logits=tensors["opacity_logits"].detach(),
            color_logits=tensors["color_logits"].detach(),
            background=background.detach(),
        )

    def _rasterise(
        self,
        model: SplatModel,
        viewpoint: Pose,
        intrinsics: CameraIntrinsics,
        torch: Any,
    ) -> NDArray[np.uint8]:
        """Render *model* from *viewpoint* at full *intrinsics* resolution."""
        device = model.means.device
        k = torch.as_tensor(_k_matrix(intrinsics), dtype=torch.float32, device=device)
        viewmat = torch.as_tensor(self._viewmat(viewpoint), dtype=torch.float32, device=device)
        rasterize = self._rasterizer(torch)
        with torch.no_grad():
            image = rasterize(
                model.means,
                model.quats,
                torch.exp(model.log_scales),
                torch.sigmoid(model.opacity_logits),
                torch.sigmoid(model.color_logits),
                viewmat,
                k,
                int(intrinsics.width),
                int(intrinsics.height),
                model.background,
            )
        out = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        return np.ascontiguousarray(out)

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _camera_to_world_cv(self, pose: Pose) -> NDArray[np.float64]:
        c2w = np.asarray(pose.matrix, dtype=np.float64)
        u, _, vt = np.linalg.svd(c2w[:3, :3])  # re-orthonormalise drifted poses
        c2w = c2w.copy()
        c2w[:3, :3] = u @ vt
        return c2w @ _GL_TO_CV if self._convention == "opengl" else c2w

    def _viewmat(self, pose: Pose) -> NDArray[np.float64]:
        return np.asarray(np.linalg.inv(self._camera_to_world_cv(pose)), dtype=np.float64)

    def _device(self, torch: Any) -> Any:
        if self._device_request == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device_request)

    def _rasterizer(self, torch: Any) -> Any:
        """Return ``f(means, quats, scales, opac, colors, viewmat, K, W, H, bg) -> [H,W,3]``."""
        if self._resolved is None:
            self._resolved = self._resolve_backend(torch)
        name, fn = self._resolved
        self.last_backend = name
        return fn

    def _resolve_backend(self, torch: Any) -> tuple[str, Any]:
        def torch_backend(
            means: Any,
            quats: Any,
            scales: Any,
            opac: Any,
            colors: Any,
            viewmat: Any,
            k: Any,
            width: int,
            height: int,
            background: Any,
        ) -> Any:
            return rasterize_torch(
                means,
                quats,
                scales,
                opac,
                colors,
                viewmat,
                k,
                width,
                height,
                background=background,
            )

        if self._backend_request == "torch":
            return "torch", torch_backend
        try:
            gsplat_fn = _gsplat_backend(torch, self._device(torch))
        except Exception as exc:
            if self._backend_request == "gsplat":
                raise
            warnings.warn(
                f"gsplat unavailable ({type(exc).__name__}: {exc}); using the pure-PyTorch "
                "rasterizer (slower, same model).",
                UserWarning,
                stacklevel=4,
            )
            return "torch", torch_backend
        return "gsplat", gsplat_fn


def _gsplat_backend(torch: Any, device: Any) -> Any:
    """Import gsplat and prove it runs on *device* with a one-gaussian probe."""
    if device.type != "cuda":
        raise RuntimeError("gsplat needs a CUDA device")
    from gsplat import rasterization

    def gsplat_backend(
        means: Any,
        quats: Any,
        scales: Any,
        opac: Any,
        colors: Any,
        viewmat: Any,
        k: Any,
        width: int,
        height: int,
        background: Any,
    ) -> Any:
        colors_out, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opac,
            colors=colors,
            viewmats=viewmat[None],
            Ks=k[None],
            width=width,
            height=height,
            backgrounds=background[None],
        )
        return colors_out[0]

    one = torch.ones(1, 3, device=device)
    eye = torch.eye(4, device=device)
    eye[2, 3] = 2.0
    gsplat_backend(
        torch.zeros(1, 3, device=device),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        one * 0.1,
        torch.ones(1, device=device) * 0.5,
        one * 0.5,
        eye,
        torch.tensor([[8.0, 0, 4], [0, 8.0, 4], [0, 0, 1]], device=device),
        8,
        8,
        torch.zeros(3, device=device),
    )
    return gsplat_backend


def _init_params(
    torch: Any,
    gen: Any,
    centres: NDArray[np.float64],
    forwards: NDArray[np.float64],
    n: int,
) -> dict[str, Any]:
    """Uniform initialisation covering the cameras and what they look at.

    Half the gaussians fill the (padded) bounding box of the camera
    centres, which contains the subject when cameras surround it; the
    other half sit in front of random cameras along their view
    direction, which covers forward-facing captures.
    """
    lo, hi = centres.min(axis=0), centres.max(axis=0)
    extent = max(float(np.linalg.norm(hi - lo)), 1.0)
    pad = 0.25 * extent
    lo_t = torch.as_tensor(lo - pad, dtype=torch.float32)
    hi_t = torch.as_tensor(hi + pad, dtype=torch.float32)
    n_box = n // 2
    box = lo_t + (hi_t - lo_t) * torch.rand(n_box, 3, generator=gen)
    cam = torch.randint(0, len(centres), (n - n_box,), generator=gen)
    depth = (0.3 + 1.2 * torch.rand(n - n_box, 1, generator=gen)) * extent
    jitter = (torch.rand(n - n_box, 3, generator=gen) - 0.5) * 0.5 * extent
    c = torch.as_tensor(centres, dtype=torch.float32)[cam]
    f = torch.as_tensor(forwards, dtype=torch.float32)[cam]
    front = c + f * depth + jitter
    means = torch.cat([box, front], dim=0)
    spacing = extent / max(n, 1) ** (1.0 / 3.0)
    return {
        "means": means,
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=-1),
        "log_scales": torch.full((n, 3), math.log(spacing)),
        "opacity_logits": torch.full((n,), -2.0),
        "color_logits": torch.zeros(n, 3),
        "extent": torch.tensor(extent),
    }


def _k_matrix(intr: CameraIntrinsics) -> NDArray[np.float64]:
    return np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]])


def _downsample(
    image: NDArray[np.uint8], intr: CameraIntrinsics, max_side: int
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Area-downsample *image* so its longer side <= *max_side*; scale K to match."""
    h, w = image.shape[:2]
    factor = max(1, math.ceil(max(h, w) / max_side))
    k = _k_matrix(intr)
    if factor == 1:
        return image, k
    from PIL import Image

    new_w, new_h = w // factor, h // factor
    small = np.asarray(
        Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BOX), dtype=np.uint8
    )
    k = k.copy()
    k[0, :] *= new_w / w
    k[1, :] *= new_h / h
    return small, k


def _load_image(path: Path) -> NDArray[np.uint8]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover -- covered by the torch check in practice
        raise ImportError(_INSTALL_HINT) from exc
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _lazy_import_torch() -> Any:
    """Import torch (and check Pillow) with a clean install hint on miss.

    Hoisted so tests can monkey-patch it to simulate a missing extra.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    try:
        import PIL  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return torch
