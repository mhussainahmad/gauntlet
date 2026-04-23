"""Import-guard tests for :mod:`gauntlet.policy.huggingface` and
:mod:`gauntlet.policy.lerobot`.

These tests verify the extras-absent contract: even when the heavy deps
(torch for HF, lerobot for the SmolVLA adapter) are missing from the
environment, importing :mod:`gauntlet.policy` must not blow up, and
accessing :class:`HuggingFacePolicy` / :class:`LeRobotPolicy` must raise
a clean ``ImportError`` with the right install hint.

They are **unmarked** — they run in ALL three CI jobs (torch-/lerobot-free
default job, ``hf-tests``, and ``lerobot-tests``). The default job
enforces the contract in the environment the contract is actually about;
the hf/lerobot jobs enforce it via ``monkeypatch.setitem(sys.modules,
<mod>, None)`` so the same code path is exercised on a machine that
happens to have those deps installed.

See docs/phase2-rfc-001-huggingface-policy.md §6 cases 1 & 2 for the HF
guards and docs/phase2-rfc-002-lerobot-smolvla.md §6 cases 1 & 2 for the
lerobot analogues.
"""

from __future__ import annotations

import importlib
import sys

import pytest


class TestImportGuards:
    def test_import_guard_raises_install_hint_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``HuggingFacePolicy(...)`` must raise ``ImportError`` naming
        ``uv sync --extra hf`` when torch is unavailable (RFC §6 case 1)."""
        # Putting ``None`` in ``sys.modules`` is the documented sentinel that
        # turns the next ``import torch`` into ImportError — we use it so this
        # test exercises the guard regardless of whether torch is actually
        # installed in the current env.
        monkeypatch.setitem(sys.modules, "torch", None)
        import gauntlet.policy.huggingface as hf_mod

        try:
            importlib.reload(hf_mod)
            with pytest.raises(ImportError, match="uv sync --extra hf"):
                hf_mod.HuggingFacePolicy(repo_id="dummy/repo", instruction="pick up the red cube")
        finally:
            # Restore normal module state even on failure, so a broken run
            # doesn't poison ``sys.modules`` for every subsequent test in
            # this process.
            monkeypatch.delitem(sys.modules, "torch", raising=False)
            importlib.reload(hf_mod)

    def test_reexport_raises_install_hint_at_attribute_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``from gauntlet.policy import HuggingFacePolicy`` must fail loudly
        at attribute-access time when torch is missing — NOT at
        ``import gauntlet.policy`` time, which would break the torch-free
        install promise for every other policy user (RFC §6 case 2).
        """
        monkeypatch.setitem(sys.modules, "torch", None)
        import gauntlet.policy as pkg

        try:
            # ``from gauntlet.policy import RandomPolicy`` must still work.
            assert pkg.RandomPolicy is not None

            # Attribute access (which is what ``from pkg import HuggingFacePolicy``
            # performs) is the point of failure.
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
            # Restore normal module state so a broken run can't poison
            # ``sys.modules`` for every subsequent test.
            monkeypatch.delitem(sys.modules, "lerobot", raising=False)
            importlib.reload(lr_mod)

    def test_lerobot_reexport_raises_install_hint_at_attribute_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``from gauntlet.policy import LeRobotPolicy`` must fail loudly
        at attribute-access time when lerobot is missing — NOT at
        ``import gauntlet.policy`` time (RFC-002 §6 case 2)."""
        monkeypatch.setitem(sys.modules, "lerobot", None)
        import gauntlet.policy as pkg

        try:
            # Non-lerobot policies must still be reachable.
            assert pkg.RandomPolicy is not None
            with pytest.raises(ImportError, match="uv sync --extra lerobot"):
                pkg.__getattr__("LeRobotPolicy")
        finally:
            monkeypatch.delitem(sys.modules, "lerobot", raising=False)
