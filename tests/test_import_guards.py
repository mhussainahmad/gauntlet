"""Import-guard contract for extras-gated modules.

Torch-/lerobot-/pybullet-free. These tests run in the default CI job
and verify three user-visible promises:

1. The torch-free surface of :mod:`gauntlet.monitor` (schema + entropy +
   :class:`DriftReport`) is importable without the ``[monitor]`` extra.
2. The torch-backed symbols (StateAutoencoder, train_ae, score_drift)
   raise a clean ``ImportError`` pointing at ``uv sync --extra monitor``
   when torch is absent — no mystery tracebacks.
3. ``HuggingFacePolicy`` / ``LeRobotPolicy`` raise a clean
   ``ImportError`` with the right install hint when torch / lerobot is
   absent, both directly and via the ``gauntlet.policy.__getattr__``
   re-export.

We exercise torch/lerobot absence by monkey-patching
``sys.modules[<name>] = None`` to simulate a fresh env without the
extra, which works regardless of whether the host venv actually has
those packages installed.

See docs/phase2-rfc-001-huggingface-policy.md §6 cases 1 & 2 for HF,
docs/phase2-rfc-002-lerobot-smolvla.md §6 cases 1 & 2 for lerobot, and
docs/phase2-rfc-003-drift-detector.md §6 for the monitor surface.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator

import pytest

# ──────────────────────────────────────────────────────────────────────
# HuggingFacePolicy / LeRobotPolicy extras-absent guards
# ──────────────────────────────────────────────────────────────────────


class TestImportGuards:
    def test_import_guard_raises_install_hint_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``HuggingFacePolicy(...)`` must raise ``ImportError`` naming
        ``uv sync --extra hf`` when torch is unavailable (RFC §6 case 1)."""
        monkeypatch.setitem(sys.modules, "torch", None)
        import gauntlet.policy.huggingface as hf_mod

        try:
            importlib.reload(hf_mod)
            with pytest.raises(ImportError, match="uv sync --extra hf"):
                hf_mod.HuggingFacePolicy(repo_id="dummy/repo", instruction="pick up the red cube")
        finally:
            monkeypatch.delitem(sys.modules, "torch", raising=False)
            importlib.reload(hf_mod)

    def test_reexport_raises_install_hint_at_attribute_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``from gauntlet.policy import HuggingFacePolicy`` must fail loudly
        at attribute-access time when torch is missing — NOT at
        ``import gauntlet.policy`` time (RFC §6 case 2).
        """
        monkeypatch.setitem(sys.modules, "torch", None)
        import gauntlet.policy as pkg

        try:
            assert pkg.RandomPolicy is not None
            with pytest.raises(ImportError, match="uv sync --extra hf"):
                pkg.__getattr__("HuggingFacePolicy")
        finally:
            monkeypatch.delitem(sys.modules, "torch", raising=False)

    def test_lerobot_import_guard_raises_install_hint_when_lerobot_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``LeRobotPolicy(...)`` must raise ``ImportError`` naming
        ``uv sync --extra lerobot`` when lerobot is unavailable
        (RFC-002 §6 case 1)."""
        monkeypatch.setitem(sys.modules, "lerobot", None)
        import gauntlet.policy.lerobot as lr_mod

        try:
            importlib.reload(lr_mod)
            with pytest.raises(ImportError, match="uv sync --extra lerobot"):
                lr_mod.LeRobotPolicy(
                    repo_id="dummy/repo",
                    instruction="pick up the red cube",
                )
        finally:
            monkeypatch.delitem(sys.modules, "lerobot", raising=False)
            importlib.reload(lr_mod)

    def test_lerobot_reexport_raises_install_hint_at_attribute_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``from gauntlet.policy import LeRobotPolicy`` must fail loudly
        at attribute-access time when lerobot is missing (RFC-002 §6 case 2)."""
        monkeypatch.setitem(sys.modules, "lerobot", None)
        import gauntlet.policy as pkg

        try:
            assert pkg.RandomPolicy is not None
            with pytest.raises(ImportError, match="uv sync --extra lerobot"):
                pkg.__getattr__("LeRobotPolicy")
        finally:
            monkeypatch.delitem(sys.modules, "lerobot", raising=False)


# ──────────────────────────────────────────────────────────────────────
# Drift-detector (``[monitor]`` extra) guards
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def torch_absent(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Simulate ``torch`` not being installed for monitor guards.

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

    assert DriftReport is DriftReport2
    assert PerEpisodeDrift is PerEpisodeDrift2
    assert callable(action_entropy)
    assert ActionEntropyStats.__name__ == "ActionEntropyStats"


def test_state_autoencoder_raises_install_hint_when_torch_absent(
    torch_absent: None,
) -> None:
    """``from gauntlet.monitor import StateAutoencoder`` fails loudly."""
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
    """``gauntlet monitor train --help`` must not require torch."""
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


# ──────────────────────────────────────────────────────────────────────
# Phase 2.5 Task 11 — gauntlet.monitor.__getattr__ unknown-name branch.
# ──────────────────────────────────────────────────────────────────────


def test_monitor_unknown_attribute_raises_attribute_error() -> None:
    """``gauntlet.monitor.<not-a-known-name>`` raises a real AttributeError
    rather than ImportError — the lazy router only intercepts the three
    documented torch-backed symbols (line 80 branch in monitor/__init__.py)."""
    import gauntlet.monitor as monitor

    with pytest.raises(AttributeError, match="no attribute 'definitely_not_there'"):
        monitor.definitely_not_there  # type: ignore[attr-defined]  # noqa: B018


# ──────────────────────────────────────────────────────────────────────
# Phase 2.5 Task 11 — gauntlet.policy.__getattr__ unknown-name branch.
# ──────────────────────────────────────────────────────────────────────


def test_policy_unknown_attribute_raises_attribute_error() -> None:
    """The lazy ``gauntlet.policy.__getattr__`` raises AttributeError on
    names other than ``HuggingFacePolicy`` / ``LeRobotPolicy`` (line 70 branch)."""
    import gauntlet.policy as policy_pkg

    with pytest.raises(AttributeError, match="no attribute"):
        policy_pkg.__getattr__("definitely_not_there")


# ──────────────────────────────────────────────────────────────────────
# Phase 2.5 Task 11 — gauntlet.ros2.__getattr__ install-hint + unknown
# ──────────────────────────────────────────────────────────────────────


def test_ros2_publisher_attribute_raises_install_hint_when_rclpy_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from gauntlet.ros2 import Ros2EpisodePublisher`` triggers the
    lazy import inside ``gauntlet.ros2.__getattr__``; when rclpy is
    absent that surfaces as a clean ``ImportError`` pointing at the
    apt / Docker install path (lines 67-78 in ros2/__init__.py)."""
    monkeypatch.setitem(sys.modules, "rclpy", None)
    monkeypatch.delitem(sys.modules, "gauntlet.ros2.publisher", raising=False)

    import gauntlet.ros2 as ros2_pkg

    with pytest.raises(ImportError, match="apt install"):
        ros2_pkg.__getattr__("Ros2EpisodePublisher")


def test_ros2_recorder_attribute_raises_install_hint_when_rclpy_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rclpy", None)
    monkeypatch.delitem(sys.modules, "gauntlet.ros2.recorder", raising=False)

    import gauntlet.ros2 as ros2_pkg

    with pytest.raises(ImportError, match="osrf/ros"):
        ros2_pkg.__getattr__("Ros2RolloutRecorder")


def test_ros2_unknown_attribute_raises_attribute_error() -> None:
    """Names other than the two lazy-routed symbols raise plain
    AttributeError (no spurious ImportError)."""
    import gauntlet.ros2 as ros2_pkg

    with pytest.raises(AttributeError, match="no attribute"):
        ros2_pkg.__getattr__("definitely_not_there")
