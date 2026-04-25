"""Renderer Protocol + registry tests for the real-to-sim stub.

The renderer is module-local (RFC §4.6) — it does *not* go through
:mod:`gauntlet.plugins`. These tests pin:

* the structural :class:`RealSimRenderer` Protocol accepts a tiny
  in-test fake;
* the registry's idempotent re-registration, name collision detection,
  and unknown-name lookup error path.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from gauntlet.realsim import (
    CameraIntrinsics,
    Pose,
    RealSimRenderer,
    RendererRegistryError,
    Scene,
    get_renderer,
    list_renderers,
    register_renderer,
)
from gauntlet.realsim.renderer import _clear_registry_for_tests


@pytest.fixture(autouse=True)
def _clear_renderer_registry() -> Iterator[None]:
    """Drop every registration before + after each test for isolation."""
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


# ---------------------------------------------------------------------------
# Tiny in-test fake renderer.
# ---------------------------------------------------------------------------


class _StubRenderer:
    """A no-op renderer used only to prove the Protocol is satisfied.

    Returns a 1x1 black image so the CPU cost is negligible. The full
    contract (HxWx3 uint8 matching ``intrinsics``) is documented in the
    Protocol but not enforced — that belongs in a downstream
    "renderer test harness" RFC, not the stub.
    """

    def render(
        self,
        scene: Scene,
        viewpoint: Pose,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        return np.zeros((1, 1, 3), dtype=np.uint8)


def _factory() -> RealSimRenderer:
    return _StubRenderer()


def _identity_pose() -> Pose:
    return Pose(
        matrix=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _good_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)


# ---------------------------------------------------------------------------
# Protocol.
# ---------------------------------------------------------------------------


def test_stub_satisfies_protocol() -> None:
    """A no-op renderer satisfies the runtime-checkable Protocol."""
    stub = _StubRenderer()
    assert isinstance(stub, RealSimRenderer)


def test_stub_render_returns_ndarray() -> None:
    """The Protocol is structural; the stub honours the return-type contract."""
    stub = _StubRenderer()
    scene = Scene(intrinsics={"wrist": _good_intrinsics()}, frames=[])
    out = stub.render(scene, _identity_pose(), _good_intrinsics())
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 1, 3)
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


def test_register_and_get_renderer_round_trip() -> None:
    register_renderer("stub", _factory)
    materialised = get_renderer("stub")
    assert isinstance(materialised, _StubRenderer)
    assert isinstance(materialised, RealSimRenderer)


def test_register_renderer_idempotent_for_same_factory() -> None:
    """Re-registering with the *same* factory object is a no-op (no raise)."""
    register_renderer("stub", _factory)
    register_renderer("stub", _factory)
    assert list_renderers() == ["stub"]


def test_register_renderer_collision_with_different_factory() -> None:
    register_renderer("stub", _factory)

    def _other() -> RealSimRenderer:
        return _StubRenderer()

    with pytest.raises(RendererRegistryError, match=r"already registered"):
        register_renderer("stub", _other)


def test_register_renderer_rejects_empty_name() -> None:
    with pytest.raises(RendererRegistryError, match=r"non-empty string"):
        register_renderer("", _factory)


def test_get_renderer_unknown_name_lists_registered() -> None:
    register_renderer("stub", _factory)
    with pytest.raises(RendererRegistryError) as exc:
        get_renderer("missing")
    msg = str(exc.value)
    assert "missing" in msg
    assert "stub" in msg


def test_list_renderers_is_sorted_copy() -> None:
    register_renderer("zeta", _factory)
    register_renderer("alpha", _factory)
    listed = list_renderers()
    assert listed == ["alpha", "zeta"]
    # Mutating the returned list does not affect the registry.
    listed.append("ghost")
    assert list_renderers() == ["alpha", "zeta"]


def test_get_renderer_returns_fresh_instance_each_call() -> None:
    """Factories run per-call so multi-process workers get independent state."""
    register_renderer("stub", _factory)
    a = get_renderer("stub")
    b = get_renderer("stub")
    assert a is not b


def test_clear_registry_helper_resets_state() -> None:
    """The fixture above relies on _clear_registry_for_tests; pin its behaviour."""
    register_renderer("stub", _factory)
    assert list_renderers() == ["stub"]
    _clear_registry_for_tests()
    assert list_renderers() == []
