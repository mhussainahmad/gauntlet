"""Fake ``isaacsim`` / ``omni.isaac.core`` namespace for Isaac Sim adapter tests.

RFC-009 §8 — Isaac Sim's runtime requires a CUDA GPU and ~15 GB of
Kit binaries, neither of which is available in CI. So the
``tests/isaac/`` suite uses a ``sys.modules``-injected fake to
exercise the adapter contract (spaces, axis dispatch, reset/step
ordering, per-axis prim-call shape) without ever touching real Kit.

The fake covers exactly the surface the adapter touches (RFC-009
§Q4.1, §Q6) — about 10 symbols. Each prim stub returns deterministic
numpy arrays from ``get_world_pose`` so the adapter's ``_build_obs``
sees real arrays (not ``MagicMock``) and downstream numpy ops work.

The fixture is module-scoped autouse — every test file in this
directory inherits a clean fake namespace, and pytest restores the
original ``sys.modules`` at module teardown via
:meth:`pytest.MonkeyPatch.setitem` semantics. Tests must NOT modify
``sys.modules`` further; per-test prim-state injection happens via
the helper accessors documented inline below.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

# Names of every fake module installed; restored at teardown via
# monkeypatch to avoid leaking into other test files.
_FAKE_MODULE_NAMES: tuple[str, ...] = (
    "isaacsim",
    "isaacsim.core",
    "isaacsim.core.api",
    "isaacsim.core.api.objects",
    "omni",
    "omni.isaac",
    "omni.isaac.core",
)


class _FakePrim:
    """Stub for ``omni.isaac.core.objects.{Dynamic,Fixed,Visual}Cuboid``.

    Tracks every ``set_world_pose`` call (so per-axis tests can
    assert on dispatch) and returns whatever was last set from
    ``get_world_pose`` — so the adapter's ``_build_obs`` reads real
    numpy arrays back, not ``MagicMock`` values that would break
    downstream numpy maths.

    Attributes
    ----------
    prim_path : str
        The USD path the adapter passed in. Mock surface uses this
        as the prim identity.
    set_world_pose_calls : list[tuple[NDArray | None, NDArray | None]]
        Append-only log of (position, orientation) pairs the adapter
        has set. Tests inspect this directly to verify per-axis
        branches reach the right prim with the right values.
    """

    def __init__(
        self,
        *,
        prim_path: str,
        position: NDArray[np.float64] | None = None,
        orientation: NDArray[np.float64] | None = None,
        size: NDArray[np.float64] | None = None,
    ) -> None:
        self.prim_path = prim_path
        self.size = size
        # Default orientation is identity wxyz — matches Isaac Sim 5.x
        # `prim.get_world_pose()` return convention (RFC-009 §7.5).
        self._position: NDArray[np.float64] = (
            np.zeros(3, dtype=np.float64)
            if position is None
            else np.asarray(position, dtype=np.float64)
        )
        self._orientation: NDArray[np.float64] = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if orientation is None
            else np.asarray(orientation, dtype=np.float64)
        )
        self.set_world_pose_calls: list[
            tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]
        ] = []

    def set_world_pose(
        self,
        position: NDArray[np.float64] | None = None,
        orientation: NDArray[np.float64] | None = None,
    ) -> None:
        if position is not None:
            self._position = np.asarray(position, dtype=np.float64).copy()
        if orientation is not None:
            self._orientation = np.asarray(orientation, dtype=np.float64).copy()
        self.set_world_pose_calls.append(
            (
                None if position is None else np.asarray(position, dtype=np.float64).copy(),
                None if orientation is None else np.asarray(orientation, dtype=np.float64).copy(),
            )
        )

    def get_world_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._position.copy(), self._orientation.copy()


class _FakeScene:
    """Stub for ``World.scene`` — collects every added prim and
    returns the same instance from ``add``."""

    def __init__(self) -> None:
        self.added: list[_FakePrim] = []

    def add(self, prim: _FakePrim) -> _FakePrim:
        self.added.append(prim)
        return prim


class _FakeWorld:
    """Stub for ``omni.isaac.core.World`` — exposes ``.scene`` and
    no-op ``.reset()`` / ``.step()`` methods."""

    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.reset_calls: int = 0
        self.step_calls: int = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def step(self, render: bool = True) -> None:
        del render
        self.step_calls += 1


class _FakeSimulationApp:
    """Stub for ``isaacsim.SimulationApp`` — no-op constructor +
    ``.close()`` so the adapter's bootstrap and teardown both work
    under the mock without launching Kit."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.closed: bool = False

    def close(self) -> None:
        self.closed = True


def _build_fake_modules() -> dict[str, types.ModuleType]:
    """Construct the seven-module fake namespace ready for sys.modules
    injection. Each module exposes only what the adapter touches —
    no broader surface, so a future API drift in the real `isaacsim`
    fails loudly during a manual GPU-workstation smoke instead of
    silently passing in CI."""

    isaacsim_mod = types.ModuleType("isaacsim")
    isaacsim_mod.SimulationApp = _FakeSimulationApp  # type: ignore[attr-defined]

    isaacsim_core_mod = types.ModuleType("isaacsim.core")
    isaacsim_core_api_mod = types.ModuleType("isaacsim.core.api")
    isaacsim_core_api_mod.World = _FakeWorld  # type: ignore[attr-defined]

    isaacsim_core_api_objects_mod = types.ModuleType("isaacsim.core.api.objects")
    # All three primitive cuboid factories return the same _FakePrim
    # under the mock — semantic differences (dynamic vs fixed vs
    # visual) only matter for real PhysX behaviour, which the tests
    # do not assert on.
    isaacsim_core_api_objects_mod.DynamicCuboid = _FakePrim  # type: ignore[attr-defined]
    isaacsim_core_api_objects_mod.FixedCuboid = _FakePrim  # type: ignore[attr-defined]
    isaacsim_core_api_objects_mod.VisualCuboid = _FakePrim  # type: ignore[attr-defined]

    # The omni.isaac.core hierarchy is reachable via the legacy import
    # path that some Isaac Sim 4.x examples used. The current adapter
    # imports through `isaacsim.core.api`, but we expose the omni.*
    # path for forward compatibility / future tests that simulate
    # transition between API generations.
    omni_mod = types.ModuleType("omni")
    omni_isaac_mod = types.ModuleType("omni.isaac")
    omni_isaac_core_mod = types.ModuleType("omni.isaac.core")
    omni_isaac_core_mod.World = _FakeWorld  # type: ignore[attr-defined]

    return {
        "isaacsim": isaacsim_mod,
        "isaacsim.core": isaacsim_core_mod,
        "isaacsim.core.api": isaacsim_core_api_mod,
        "isaacsim.core.api.objects": isaacsim_core_api_objects_mod,
        "omni": omni_mod,
        "omni.isaac": omni_isaac_mod,
        "omni.isaac.core": omni_isaac_core_mod,
    }


@pytest.fixture(autouse=True)
def _install_fake_isaacsim(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Inject the fake namespace + flush any cached adapter import.

    The adapter (``gauntlet.env.isaac.tabletop_isaac``) imports
    ``isaacsim`` at module scope — once Python caches that import
    against the (real-or-mocked) ``isaacsim`` in ``sys.modules``,
    swapping the cached value out doesn't undo the bind in the
    adapter module's globals. So we delete both ``gauntlet.env.isaac``
    and ``gauntlet.env.isaac.tabletop_isaac`` from ``sys.modules``
    BEFORE every test, then install the fakes; the next test-level
    ``import gauntlet.env.isaac`` then re-runs the adapter module
    body against the fake.
    """
    # Flush any prior adapter import so a fresh `import gauntlet.env.isaac`
    # re-runs the module body with the fake namespace in place.
    for mod in list(sys.modules):
        if mod.startswith("gauntlet.env.isaac") or mod in _FAKE_MODULE_NAMES:
            monkeypatch.delitem(sys.modules, mod, raising=False)

    fakes = _build_fake_modules()
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)

    # The env registry is a process-global dict; re-importing
    # ``gauntlet.env.isaac`` after the cache flush above re-runs
    # ``register_env("tabletop-isaac", ...)`` which raises if the key
    # is already present. Pop the key now so the next import installs
    # cleanly. ``monkeypatch.setitem`` on a key the dict doesn't have
    # yet would still set it, so we use direct dict manipulation +
    # an addfinalizer to restore the registry at fixture teardown.
    from gauntlet.env.registry import _REGISTRY

    saved = _REGISTRY.pop("tabletop-isaac", None)

    def _restore() -> None:
        _REGISTRY.pop("tabletop-isaac", None)
        if saved is not None:
            _REGISTRY["tabletop-isaac"] = saved

    yield
    _restore()


# Public helpers re-exposed so individual test files import them
# without re-walking ``sys.modules``.

__all__ = [
    "_FakePrim",
    "_FakeScene",
    "_FakeSimulationApp",
    "_FakeWorld",
]
