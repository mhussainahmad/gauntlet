"""Smoke test — verifies the package is importable and exposes a version."""

from __future__ import annotations

import gauntlet


def test_version_is_present() -> None:
    assert isinstance(gauntlet.__version__, str)
    assert gauntlet.__version__.count(".") >= 2


def test_version_matches_pep440_prefix() -> None:
    major, minor, *_ = gauntlet.__version__.split(".")
    assert major.isdigit()
    assert minor.isdigit()
