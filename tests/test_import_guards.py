"""Import-guard tests for :mod:`gauntlet.policy.huggingface`.

These tests verify the torch-absent contract: even when torch is missing
from the environment, importing :mod:`gauntlet.policy` must not blow up,
and accessing :class:`HuggingFacePolicy` must raise a clean ``ImportError``
with the ``uv sync --extra hf`` install hint.

They are **unmarked** — they run in BOTH CI jobs (torch-free default job
and ``hf-tests`` job). The default job enforces the contract in the
environment the contract is actually about; the hf job enforces it via
``monkeypatch.setitem(sys.modules, "torch", None)`` so the same code path
is exercised on a machine that happens to have torch installed.

See docs/phase2-rfc-001-huggingface-policy.md §6 cases 1 & 2.
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
