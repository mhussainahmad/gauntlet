"""Tests for the in-tree :class:`RealSimRenderer` plugins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Pillow ships only with the [hf] / [lerobot] / [pi0] / [groot] / [rdt]
# / [monitor] extras (see pyproject [project.optional-dependencies]).
# The realsim renderer tests need PIL to write PNGs onto the fake scene;
# skip the whole module on the torch-free job rather than fail at
# collection.
pytest.importorskip("PIL")

from gauntlet.realsim import (
    GaussianSplatRenderer,
    NearestFrameRenderer,
    get_renderer,
    list_renderers,
)
from gauntlet.realsim.schema import CameraFrame, CameraIntrinsics, Pose, Scene

# ----------------------------------------------------------------------
# Registration / discovery.
# ----------------------------------------------------------------------


def test_registry_lists_shipped_renderers() -> None:
    names = list_renderers()
    assert "nearest-frame" in names
    assert "gsplat" in names


def test_registry_zero_arg_factories_construct() -> None:
    nf = get_renderer("nearest-frame")
    assert isinstance(nf, NearestFrameRenderer)
    gs = get_renderer("gsplat")
    assert isinstance(gs, GaussianSplatRenderer)


# ----------------------------------------------------------------------
# NearestFrameRenderer behaviour.
# ----------------------------------------------------------------------


def _identity_pose() -> Pose:
    return Pose(
        matrix=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _translated_pose(x: float, y: float, z: float) -> Pose:
    return Pose(
        matrix=[
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _write_png(path: Path, *, height: int, width: int, fill: int) -> None:
    from PIL import Image

    arr = np.full((height, width, 3), fill, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


def _build_scene(tmp_path: Path) -> Scene:
    """Two-frame scene at distinct translations, distinguishable by fill colour."""
    intrinsics = CameraIntrinsics(
        fx=400.0,
        fy=400.0,
        cx=200.0,
        cy=150.0,
        width=400,
        height=300,
    )
    _write_png(tmp_path / "frame_0.png", height=300, width=400, fill=10)
    _write_png(tmp_path / "frame_1.png", height=300, width=400, fill=200)
    return Scene(
        intrinsics={"cam": intrinsics},
        frames=[
            CameraFrame(
                path="frame_0.png",
                timestamp=0.0,
                intrinsics_id="cam",
                pose=_translated_pose(0.0, 0.0, 0.0),
            ),
            CameraFrame(
                path="frame_1.png",
                timestamp=1.0,
                intrinsics_id="cam",
                pose=_translated_pose(5.0, 0.0, 0.0),
            ),
        ],
    )


def test_nearest_frame_round_trip_returns_matching_image(tmp_path: Path) -> None:
    scene = _build_scene(tmp_path)
    renderer = NearestFrameRenderer(scene_root=tmp_path)
    intrinsics = scene.intrinsics["cam"]
    # Viewpoint at exactly the second frame's pose — must return frame 1.
    image = renderer.render(
        scene=scene,
        viewpoint=scene.frames[1].pose,
        intrinsics=intrinsics,
    )
    assert image.shape == (300, 400, 3)
    assert image.dtype == np.uint8
    # Frame 1 was filled with 200 — every pixel must be 200.
    assert int(image[0, 0, 0]) == 200


def test_nearest_frame_resizes_to_requested_intrinsics(tmp_path: Path) -> None:
    scene = _build_scene(tmp_path)
    renderer = NearestFrameRenderer(scene_root=tmp_path)
    # Half-resolution intrinsics.
    intrinsics = CameraIntrinsics(
        fx=200.0,
        fy=200.0,
        cx=100.0,
        cy=75.0,
        width=200,
        height=150,
    )
    image = renderer.render(
        scene=scene,
        viewpoint=_identity_pose(),
        intrinsics=intrinsics,
    )
    assert image.shape == (150, 200, 3)


def test_nearest_frame_empty_scene_raises(tmp_path: Path) -> None:
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1)
    scene = Scene(intrinsics={"cam": intrinsics}, frames=[])
    renderer = NearestFrameRenderer(scene_root=tmp_path)
    with pytest.raises(ValueError, match="zero frames"):
        renderer.render(scene=scene, viewpoint=_identity_pose(), intrinsics=intrinsics)


# ----------------------------------------------------------------------
# GaussianSplatRenderer plugin wiring.
# ----------------------------------------------------------------------


def test_gsplat_renderer_missing_extra_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When gsplat / torch are not installed, render() raises ImportError
    with the canonical install hint — NOT a bare ModuleNotFoundError."""
    scene = _build_scene(tmp_path)
    intrinsics = scene.intrinsics["cam"]
    renderer = GaussianSplatRenderer()

    # Force the lazy import to fail regardless of host installation.
    from gauntlet.realsim.renderers import gsplat as gsplat_mod

    def _raise_install_hint() -> tuple[object, object]:
        raise ImportError(gsplat_mod._INSTALL_HINT)

    monkeypatch.setattr(gsplat_mod, "_lazy_import_gsplat", _raise_install_hint)

    with pytest.raises(ImportError, match=r"realsim-gsplat"):
        renderer.render(scene=scene, viewpoint=_identity_pose(), intrinsics=intrinsics)
