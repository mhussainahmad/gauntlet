"""Import guard tests for the Isaac Sim adapter — RFC-009 §8.

The adapter subpackage (`gauntlet.env.isaac`) raises an ImportError
with the canonical install hint when the `[isaac]` extra is absent.
This codifies the contract from `__init__.py` so an accidental
removal of the guard surfaces in the default torch-free CI job.

The conftest in this directory injects a fake `isaacsim` namespace
before every test (so the rest of the test suite can construct the
adapter under the mock); these tests temporarily uninstall the fake
to simulate the "extra not installed" path.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _flush_isaac_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every cached `gauntlet.env.isaac*` and `isaacsim*` /
    `omni*` module so the next `import gauntlet.env.isaac` re-runs
    the guard against whatever is in `sys.modules` now."""
    for mod in list(sys.modules):
        if (
            mod.startswith("gauntlet.env.isaac")
            or mod.startswith("isaacsim")
            or mod == "isaacsim"
            or mod.startswith("omni")
            or mod == "omni"
        ):
            monkeypatch.delitem(sys.modules, mod, raising=False)


def test_import_raises_install_hint_when_isaacsim_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`import gauntlet.env.isaac` -> ImportError naming `--extra isaac`."""
    _flush_isaac_modules(monkeypatch)
    # Setting sys.modules["isaacsim"] = None causes the next `import
    # isaacsim` to raise ModuleNotFoundError, regardless of whether
    # the package is actually present in the host venv. Same trick
    # used by `tests/test_import_guards.py` for torch / lerobot.
    monkeypatch.setitem(sys.modules, "isaacsim", None)

    with pytest.raises(ImportError, match=r"uv sync --extra isaac"):
        importlib.import_module("gauntlet.env.isaac")


def test_import_error_mentions_pip_install_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The install-hint covers both `uv sync` and the `pip install
    'gauntlet[isaac]'` paths so a non-uv user gets useful guidance
    too."""
    _flush_isaac_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "isaacsim", None)

    with pytest.raises(ImportError, match=r"pip install 'gauntlet\[isaac\]'"):
        importlib.import_module("gauntlet.env.isaac")


def test_import_error_mentions_gpu_runtime_caveat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard message names the GPU-runtime caveat so users on CPU
    workstations don't waste a 15 GB download installing an extra
    that won't run for them. RFC-009 §4.4."""
    _flush_isaac_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "isaacsim", None)

    with pytest.raises(ImportError, match=r"GPU"):
        importlib.import_module("gauntlet.env.isaac")
