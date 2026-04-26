"""Centralised security helpers — Phase 2.5 Task 16.

This package collects the small set of input-boundary helpers the
harness uses to defend against the documented threat classes in
:doc:`docs/security` (path traversal at sink / cache boundaries; unsafe
YAML deserialisation). The contract is deliberately narrow:

* :func:`safe_join` — sandboxed path join with explicit traversal
  rejection. Use at boundaries where a user-supplied component is
  joined under a trusted root.
* :func:`safe_yaml_load` — the canonical YAML entry point. Wraps
  :func:`yaml.safe_load` so every call-site flows through one place;
  the CI grep gate verifies that PyYAML's unsafe-loader entry points
  never re-enter ``src/``.

Errors:

* :class:`PathTraversalError` — subclass of :class:`ValueError`, raised
  by :func:`safe_join` on attempted escape.
* :class:`YamlSecurityError` — subclass of :class:`ValueError`, reserved
  for future strict-mode rejections layered on top of
  :func:`yaml.safe_load`. The current wrapper does not raise it — every
  unsafe-tag rejection still flows through :class:`yaml.YAMLError` so
  callers (and the existing regression tests in
  ``tests/test_security_yaml.py``) keep working unchanged.
"""

from __future__ import annotations

from gauntlet.security.paths import PathTraversalError, safe_join
from gauntlet.security.yaml_guard import YamlSecurityError, safe_yaml_load

__all__ = [
    "PathTraversalError",
    "YamlSecurityError",
    "safe_join",
    "safe_yaml_load",
]
