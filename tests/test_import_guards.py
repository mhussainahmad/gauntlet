"""Import-guard contract for extras-gated modules.

Torch-free. These tests run in the default CI job and verify two
user-visible promises:

1. The torch-free surface of :mod:`gauntlet.monitor` (schema + entropy +
   :class:`DriftReport`) is importable without the ``[monitor]`` extra.
2. The torch-backed symbols raise a clean ``ImportError`` pointing at
   ``uv sync --extra monitor`` when torch is absent — no mystery
   tracebacks, no swallowed errors.

We exercise both paths by monkey-patching ``sys.modules["torch"] = None``
to simulate a fresh env without the extra, which works regardless of
whether the host venv actually has torch installed. Reserved HF /
lerobot guards live next to each other once those RFCs land; this file
is the shared home for all three.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator

import pytest


@pytest.fixture
def torch_absent(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Simulate ``torch`` not being installed.

    Setting ``sys.modules["torch"] = None`` makes future ``import torch``
    calls raise ``ImportError`` without actually uninstalling the
    package — safe across CI jobs that may have torch in their venv.
    We also flush the three lazy monitor modules from the module cache
    so the guard's ``try/except`` block re-runs on next import.
    """
    to_flush = [
        "torch",
        "gauntlet.monitor.ae",
        "gauntlet.monitor.train",
        "gauntlet.monitor.score",
    ]
    for name in to_flush:
        if name in sys.modules:
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    yield


def test_monitor_torch_free_imports_always_work() -> None:
    """DriftReport + PerEpisodeDrift + action_entropy never touch torch."""
    from gauntlet.monitor import DriftReport, PerEpisodeDrift, action_entropy
    from gauntlet.monitor.entropy import ActionEntropyStats
    from gauntlet.monitor.schema import DriftReport as DriftReport2
    from gauntlet.monitor.schema import PerEpisodeDrift as PerEpisodeDrift2

    # These are just sanity checks that the re-export arms match.
    assert DriftReport is DriftReport2
    assert PerEpisodeDrift is PerEpisodeDrift2
    assert callable(action_entropy)
    assert ActionEntropyStats.__name__ == "ActionEntropyStats"


def test_state_autoencoder_raises_install_hint_when_torch_absent(
    torch_absent: None,
) -> None:
    """``from gauntlet.monitor import StateAutoencoder`` fails loudly."""
    # The ``gauntlet.monitor`` package itself is still imported (and
    # still torch-free).
    monitor = importlib.import_module("gauntlet.monitor")
    with pytest.raises(ImportError, match="uv sync --extra monitor"):
        monitor.StateAutoencoder  # noqa: B018


def test_train_ae_raises_install_hint_when_torch_absent(
    torch_absent: None,
) -> None:
    monitor = importlib.import_module("gauntlet.monitor")
    with pytest.raises(ImportError, match="uv sync --extra monitor"):
        monitor.train_ae  # noqa: B018


def test_score_drift_raises_install_hint_when_torch_absent(
    torch_absent: None,
) -> None:
    monitor = importlib.import_module("gauntlet.monitor")
    with pytest.raises(ImportError, match="uv sync --extra monitor"):
        monitor.score_drift  # noqa: B018


def test_cli_monitor_train_help_exits_zero() -> None:
    """``gauntlet monitor train --help`` must not require torch.

    Typer renders the help without importing the lazy-loaded
    ``gauntlet.monitor.train`` module, which is why a torch-free install
    still prints usable help text — a cheap sanity check the subcommand
    wiring is right.
    """
    result = subprocess.run(
        [sys.executable, "-m", "gauntlet.cli", "monitor", "train", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "trajectory_dir" in result.stdout.lower() or "TRAJECTORY_DIR" in result.stdout


def test_cli_monitor_score_help_exits_zero() -> None:
    """Sibling sanity-check on the ``score`` subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "gauntlet.cli", "monitor", "score", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "--ae" in result.stdout
