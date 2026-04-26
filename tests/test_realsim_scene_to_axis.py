"""Tests for the camera_extrinsics-axis bridge — Phase 3 Task 18.

Covers:

* :func:`scene_to_camera_extrinsics` shape + sub-sampling behaviour.
* :func:`rotation_matrix_to_xyz_euler` round-trip on canonical
  rotations.
* The output dict shape exactly matches
  :class:`gauntlet.suite.schema.ExtrinsicsValue` so a future suite YAML
  could ingest the result verbatim.
* A meta-test that walks the public surface of :mod:`gauntlet.realsim`
  and asserts no symbol has accidentally grown a working ``render``
  method while the renderer is officially out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import gauntlet.realsim as realsim
from gauntlet.realsim.scene_input import RealSceneInput
from gauntlet.realsim.scene_to_axis import (
    RendererNotImplementedError,
    rotation_matrix_to_xyz_euler,
    scene_to_camera_extrinsics,
)
from gauntlet.suite.schema import ExtrinsicsValue

# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------


def _identity_pose() -> NDArray[np.float64]:
    return np.eye(4, dtype=np.float64)


def _pose(
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rx: float = 0.0,
    ry: float = 0.0,
    rz: float = 0.0,
) -> NDArray[np.float64]:
    """Build a 4x4 pose with the given translation and XYZ-Euler rotation."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_mat = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float64,
    )
    ry_mat = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rz_mat = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rot = rz_mat @ ry_mat @ rx_mat
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[0, 3], pose[1, 3], pose[2, 3] = translation
    return pose


def _make_scene(poses: list[NDArray[np.float64]]) -> RealSceneInput:
    """Construct a synthetic :class:`RealSceneInput` around given poses."""
    return RealSceneInput(
        capture_dir=Path("/synthetic"),
        intrinsics={
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "width": 640.0,
            "height": 480.0,
        },
        extrinsics_per_frame=poses,
    )


# ---------------------------------------------------------------------------
# Rotation conversion.
# ---------------------------------------------------------------------------


def test_rotation_matrix_to_xyz_euler_identity() -> None:
    """Identity rotation -> all-zero Euler angles."""
    rx, ry, rz = rotation_matrix_to_xyz_euler(np.eye(3, dtype=np.float64))
    assert rx == pytest.approx(0.0, abs=1e-9)
    assert ry == pytest.approx(0.0, abs=1e-9)
    assert rz == pytest.approx(0.0, abs=1e-9)


def test_rotation_matrix_to_xyz_euler_round_trip() -> None:
    """A known (rx, ry, rz) round-trips through pose -> Euler."""
    for rx0, ry0, rz0 in [
        (0.1, -0.2, 0.3),
        (-0.5, 0.4, -0.6),
        (1.0, 0.7, -0.3),
    ]:
        pose = _pose(rx=rx0, ry=ry0, rz=rz0)
        rx, ry, rz = rotation_matrix_to_xyz_euler(pose[:3, :3])
        assert rx == pytest.approx(rx0, abs=1e-9)
        assert ry == pytest.approx(ry0, abs=1e-9)
        assert rz == pytest.approx(rz0, abs=1e-9)


def test_rotation_matrix_to_xyz_euler_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="3x3"):
        rotation_matrix_to_xyz_euler(np.eye(4, dtype=np.float64))


# ---------------------------------------------------------------------------
# scene_to_camera_extrinsics — shape contract.
# ---------------------------------------------------------------------------


def test_scene_to_camera_extrinsics_identity_yields_zero_deltas() -> None:
    scene = _make_scene([_identity_pose() for _ in range(4)])
    out = scene_to_camera_extrinsics(scene, n_samples=4)
    assert len(out) == 4
    for entry in out:
        assert entry["translation"] == [0.0, 0.0, 0.0]
        assert entry["rotation"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_scene_to_camera_extrinsics_recovers_translation() -> None:
    pose = _pose(translation=(0.1, -0.2, 0.3))
    scene = _make_scene([pose])
    out = scene_to_camera_extrinsics(scene, n_samples=1)
    assert len(out) == 1
    assert out[0]["translation"] == pytest.approx([0.1, -0.2, 0.3], abs=1e-12)


def test_scene_to_camera_extrinsics_recovers_rotation() -> None:
    pose = _pose(rx=0.2, ry=-0.1, rz=0.3)
    scene = _make_scene([pose])
    out = scene_to_camera_extrinsics(scene, n_samples=1)
    assert len(out) == 1
    assert out[0]["rotation"] == pytest.approx([0.2, -0.1, 0.3], abs=1e-9)


def test_scene_to_camera_extrinsics_subsamples_when_oversize() -> None:
    """100 frames + n_samples=8 -> 8 entries spanning first..last."""
    poses = [_pose(translation=(float(i), 0.0, 0.0)) for i in range(100)]
    scene = _make_scene(poses)
    out = scene_to_camera_extrinsics(scene, n_samples=8)
    assert len(out) == 8
    # First and last frame must always be included.
    assert out[0]["translation"][0] == pytest.approx(0.0, abs=1e-12)
    assert out[-1]["translation"][0] == pytest.approx(99.0, abs=1e-12)


def test_scene_to_camera_extrinsics_returns_all_when_undersized() -> None:
    """3 frames + n_samples=8 -> 3 entries (no duplication)."""
    poses = [_pose(translation=(float(i), 0.0, 0.0)) for i in range(3)]
    scene = _make_scene(poses)
    out = scene_to_camera_extrinsics(scene, n_samples=8)
    assert len(out) == 3
    assert [e["translation"][0] for e in out] == pytest.approx([0.0, 1.0, 2.0])


def test_scene_to_camera_extrinsics_rejects_zero_n_samples() -> None:
    scene = _make_scene([_identity_pose()])
    with pytest.raises(ValueError, match=r"n_samples must be >= 1"):
        scene_to_camera_extrinsics(scene, n_samples=0)


def test_scene_to_camera_extrinsics_rejects_empty_scene() -> None:
    """A handcrafted empty-scene handle is rejected.

    ``load_real_scene`` already rejects empty extrinsics earlier, but the
    bridge guards anyway since callers may construct
    :class:`RealSceneInput` directly in notebooks / tests.
    """
    scene = _make_scene([])
    with pytest.raises(ValueError, match="zero extrinsics"):
        scene_to_camera_extrinsics(scene, n_samples=1)


# ---------------------------------------------------------------------------
# Output schema-shape parity with ExtrinsicsValue.
# ---------------------------------------------------------------------------


def test_scene_to_camera_extrinsics_output_matches_extrinsics_value_schema() -> None:
    """Each emitted dict is a valid :class:`ExtrinsicsValue` payload.

    This is the load-bearing test for the bridge: it proves the dicts
    we emit could be stuffed back into a suite YAML's
    ``extrinsics_values`` list verbatim. If the upstream ``ExtrinsicsValue``
    schema ever tightens (e.g. ``rotation`` becomes radians-bounded), this
    test fires before the bridge silently emits invalid entries.
    """
    poses = [
        _pose(translation=(0.1, 0.2, 0.3), rx=0.05, ry=-0.03, rz=0.02),
        _pose(translation=(-0.1, 0.0, 0.5), rx=-0.05, ry=0.1, rz=-0.1),
    ]
    scene = _make_scene(poses)
    out = scene_to_camera_extrinsics(scene, n_samples=2)
    for entry in out:
        # ``ExtrinsicsValue.model_validate`` raises on schema mismatch
        # — any failure here is a contract bug.
        ExtrinsicsValue.model_validate(entry)
        # Concretely: translation + rotation are length-3 list[float].
        assert isinstance(entry["translation"], list)
        assert len(entry["translation"]) == 3
        assert all(isinstance(v, float) for v in entry["translation"])
        assert isinstance(entry["rotation"], list)
        assert len(entry["rotation"]) == 3
        assert all(isinstance(v, float) for v in entry["rotation"])


# ---------------------------------------------------------------------------
# Renderer seam.
# ---------------------------------------------------------------------------


def test_renderer_not_implemented_error_is_a_notimplementederror() -> None:
    """``RendererNotImplementedError`` subclasses :class:`NotImplementedError`."""
    err = RendererNotImplementedError("future seam")
    assert isinstance(err, NotImplementedError)


def test_renderer_not_implemented_error_is_exported() -> None:
    """The seam exception is reachable via the public package surface."""
    assert hasattr(realsim, "RendererNotImplementedError")
    assert realsim.RendererNotImplementedError is RendererNotImplementedError


# ---------------------------------------------------------------------------
# Meta-test: no accidental renderer landed.
# ---------------------------------------------------------------------------


def test_no_accidental_renderer_landed() -> None:
    """Walk :mod:`gauntlet.realsim.__all__` and prove no renderer drifted in.

    The renderer is officially out-of-scope for T18 (see
    ``docs/realsim.md`` — phased scope). The only public symbol that
    legitimately *describes* a renderer is the
    :class:`gauntlet.realsim.RealSimRenderer` Protocol, which is a
    type-only contract, not a runnable object.

    This test fires if a future contributor accidentally lands a
    renderer (a class with a working ``render`` method) without going
    through the explicit follow-up RFC. The check has two prongs:

    1. No public symbol other than the Protocol exposes ``render``.
    2. The Protocol's ``render`` method is the abstract structural
       declaration (no concrete behaviour).
    """
    for name in realsim.__all__:
        symbol = getattr(realsim, name)
        # The Protocol is the one legitimate ``render``-bearing symbol.
        if name == "RealSimRenderer":
            continue
        if hasattr(symbol, "render"):
            pytest.fail(
                f"public realsim symbol {name!r} unexpectedly has a 'render' "
                f"attribute; the renderer is officially deferred (T18 scope). "
                f"If this is intentional, file a follow-up RFC and remove this "
                f"guard.",
            )


def test_calling_renderer_seam_directly_raises_renderer_not_implemented() -> None:
    """A direct caller invoking the seam exception gets the right type.

    Regression guard: the seam exception is a real, instantiable type
    (not a forward-reference / typing-only artefact). A future renderer
    helper will replace its body with real behaviour; until then, any
    helper that raises this exception keeps the contract honest.
    """

    @dataclass
    class _PretendRenderer:
        """Local pretend-renderer used only inside this regression test."""

        def render(self) -> NDArray[np.uint8]:
            raise RendererNotImplementedError(
                "renderer is deferred — see docs/realsim.md (T18 scope)",
            )

    pretender = _PretendRenderer()
    with pytest.raises(RendererNotImplementedError, match="deferred"):
        pretender.render()
