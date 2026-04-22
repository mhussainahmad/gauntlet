"""Suite schema and YAML loader — see ``GAUNTLET_SPEC.md`` §5 task 5.

Public surface:

* :class:`Suite` — Pydantic model for a named perturbation grid.
* :class:`AxisSpec` — per-axis grid specification (continuous or
  categorical).
* :class:`SuiteCell` — one point in the grid (consumed by the Runner).
* :func:`load_suite` / :func:`load_suite_from_string` — YAML loaders.
* :data:`SUPPORTED_ENVS` — set of env slugs Phase 1 recognises.

``no_implicit_reexport`` is enabled project-wide (see ``pyproject.toml``
``[tool.mypy]``), so every re-export uses the explicit ``X as X`` form.
"""

from __future__ import annotations

from gauntlet.suite.loader import load_suite as load_suite
from gauntlet.suite.loader import load_suite_from_string as load_suite_from_string
from gauntlet.suite.schema import SUPPORTED_ENVS as SUPPORTED_ENVS
from gauntlet.suite.schema import AxisSpec as AxisSpec
from gauntlet.suite.schema import Suite as Suite
from gauntlet.suite.schema import SuiteCell as SuiteCell

__all__ = [
    "SUPPORTED_ENVS",
    "AxisSpec",
    "Suite",
    "SuiteCell",
    "load_suite",
    "load_suite_from_string",
]
