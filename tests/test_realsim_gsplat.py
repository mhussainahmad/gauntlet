"""GaussianSplatRenderer: projection, camera conventions, and the default fit.

Needs torch + Pillow, so it runs in the ``monitor`` CI job (the
``[monitor]`` extra installs exactly those). Everything here uses the
pure-PyTorch backend on CPU with tiny scenes; the gsplat CUDA path is
not exercised in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")

from gauntlet.realsim.renderers.gsplat import (  # noqa: E402
    GaussianSplatRenderer,
    rasterize_torch,
)
from gauntlet.realsim.schema import (  # noqa: E402
    CameraFrame,
    CameraIntrinsics,
    Pose,
    Scene,
)

pytestmark = pytest.mark.monitor

_SIZE = 32
_INTR = CameraIntrinsics(fx=30.0, fy=30.0, cx=_SIZE / 2, cy=_SIZE / 2, width=_SIZE, height=_SIZE)


def _look_at_gl(eye: np.ndarray, target: np.ndarray) -> Pose:
    """Camera-to-world pose in the OpenGL convention (camera looks down -z)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    z_axis = -forward
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z_axis))) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x_axis, y_axis, z_axis, eye
    return Pose(matrix=m.tolist())


def _single_blob() -> dict[str, object]:
    return {
        "means": torch.zeros(1, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "scales": torch.full((1, 3), 0.1),
        "opacities": torch.tensor([0.9]),
        "colors": torch.tensor([[1.0, 1.0, 1.0]]),
    }


def _render_blob(renderer: GaussianSplatRenderer, pose: Pose) -> np.ndarray:
    blob = _single_blob()
    k = torch.tensor([[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]])
    viewmat = torch.as_tensor(renderer._viewmat(pose), dtype=torch.float32)
    image = rasterize_torch(
        blob["means"],
        blob["quats"],
        blob["scales"],
        blob["opacities"],
        blob["colors"],
        viewmat,
        k,
        _SIZE,
        _SIZE,
    )
    return np.asarray(image.numpy(), dtype=np.float32)


def test_blob_projects_to_principal_point_opengl() -> None:
    renderer = GaussianSplatRenderer(camera_convention="opengl")
    image = _render_blob(renderer, _look_at_gl(np.array([0.0, -2.0, 0.0]), np.zeros(3)))
    lum = image.sum(axis=2)
    y, x = np.unravel_index(int(lum.argmax()), lum.shape)
    assert abs(x + 0.5 - 16) <= 1.0 and abs(y + 0.5 - 16) <= 1.0
    assert lum.max() > 1.0


def test_camera_facing_away_sees_nothing() -> None:
    renderer = GaussianSplatRenderer(camera_convention="opengl")
    image = _render_blob(
        renderer, _look_at_gl(np.array([0.0, -2.0, 0.0]), np.array([0.0, -4.0, 0.0]))
    )
    assert float(image.max()) == 0.0


def test_opencv_convention_looks_down_positive_z() -> None:
    renderer = GaussianSplatRenderer(camera_convention="opencv")
    m = np.eye(4)
    m[2, 3] = -2.0  # OpenCV camera at z=-2 with identity rotation looks toward +z.
    image = _render_blob(renderer, Pose(matrix=m.tolist()))
    assert image.sum(axis=2).max() > 1.0


def test_off_centre_blob_moves_the_right_way() -> None:
    renderer = GaussianSplatRenderer(camera_convention="opencv")
    blob = _single_blob()
    blob["means"] = torch.tensor([[0.3, 0.0, 0.0]])  # +x in the world = right in the image
    k = torch.tensor([[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]])
    m = np.eye(4)
    m[2, 3] = -2.0
    viewmat = torch.as_tensor(renderer._viewmat(Pose(matrix=m.tolist())), dtype=torch.float32)
    image = rasterize_torch(
        blob["means"],
        blob["quats"],
        blob["scales"],
        blob["opacities"],
        blob["colors"],
        viewmat,
        k,
        _SIZE,
        _SIZE,
    ).numpy()
    _, x = np.unravel_index(int(image.sum(axis=2).argmax()), image.shape[:2])
    assert x + 0.5 == pytest.approx(16 + 30 * 0.3 / 2.0, abs=1.0)


def _ground_truth_scene(tmp_path: Path) -> tuple[Scene, list[np.ndarray]]:
    """Five coloured blobs viewed by eight cameras on a ring; frames saved as PNG."""
    gen = torch.Generator().manual_seed(1)
    means = (torch.rand(5, 3, generator=gen) - 0.5) * 0.8
    colors = torch.tensor(
        [[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0], [1.0, 1.0, 0.2], [0.8, 0.2, 0.8]]
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(5, 1)
    scales = torch.full((5, 3), 0.15)
    opac = torch.full((5,), 0.9)
    renderer = GaussianSplatRenderer(camera_convention="opengl")
    k = torch.tensor([[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]])
    frames, images = [], []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        eye = np.array([2.5 * np.cos(angle), 2.5 * np.sin(angle), 0.5])
        pose = _look_at_gl(eye, np.zeros(3))
        viewmat = torch.as_tensor(renderer._viewmat(pose), dtype=torch.float32)
        img = rasterize_torch(means, quats, scales, opac, colors, viewmat, k, _SIZE, _SIZE)
        arr = (img.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        PIL_Image.fromarray(arr).save(tmp_path / f"f{i}.png")
        frames.append(
            CameraFrame(path=f"f{i}.png", timestamp=float(i), intrinsics_id="cam", pose=pose)
        )
        images.append(arr)
    return Scene(intrinsics={"cam": _INTR}, frames=frames), images


def test_default_fit_reduces_held_out_error(tmp_path: Path) -> None:
    scene, images = _ground_truth_scene(tmp_path)
    held_out = scene.frames[3]
    train_scene = Scene(
        intrinsics=scene.intrinsics, frames=[f for i, f in enumerate(scene.frames) if i != 3]
    )

    def held_out_l1(steps: int) -> float:
        renderer = GaussianSplatRenderer(
            scene_root=tmp_path,
            max_train_steps=steps,
            num_gaussians=200,
            train_max_side=_SIZE,
            backend="torch",
            device="cpu",
        )
        with pytest.warns(UserWarning, match="smoke-test"):
            out = renderer.render(train_scene, held_out.pose, _INTR)
        assert out.shape == (_SIZE, _SIZE, 3) and out.dtype == np.uint8
        return float(np.abs(out.astype(float) - images[3].astype(float)).mean())

    untrained = held_out_l1(0)
    trained = held_out_l1(300)
    assert trained < 0.6 * untrained


def test_auto_backend_falls_back_to_torch_on_cpu(tmp_path: Path) -> None:
    scene, _ = _ground_truth_scene(tmp_path)
    renderer = GaussianSplatRenderer(
        scene_root=tmp_path, max_train_steps=2, num_gaussians=16, device="cpu"
    )
    with pytest.warns(UserWarning):
        renderer.render(scene, scene.frames[0].pose, _INTR)
    assert renderer.last_backend == "torch"


def test_fit_is_cached_per_scene(tmp_path: Path) -> None:
    scene, _ = _ground_truth_scene(tmp_path)
    renderer = GaussianSplatRenderer(
        scene_root=tmp_path, max_train_steps=2, num_gaussians=16, backend="torch", device="cpu"
    )
    with pytest.warns(UserWarning):
        first = renderer.render(scene, scene.frames[0].pose, _INTR)
    second = renderer.render(scene, scene.frames[0].pose, _INTR)
    np.testing.assert_array_equal(first, second)
