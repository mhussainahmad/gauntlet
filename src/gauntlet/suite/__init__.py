"""Suite schema and YAML loader — see ``GAUNTLET_SPEC.md`` §5 task 5.

Public surface:

* :class:`Suite` — Pydantic model for a named perturbation grid.
* :class:`AxisSpec` — per-axis grid specification (continuous or
  categorical).
* :class:`SuiteCell` — one point in the grid (consumed by the Runner).
* :func:`load_suite` / :func:`load_suite_from_string` — YAML loaders.
* :data:`BUILTIN_BACKEND_IMPORTS` — env-slug → Python-module map used by
  the loader to lazy-import backends that live behind optional extras
  (RFC-005 §11.2). Phase 1 left this empty (MuJoCo-only); Phase 2
  Task 5 seeds ``"tabletop-pybullet"``.

``no_implicit_reexport`` is enabled project-wide (see ``pyproject.toml``
``[tool.mypy]``), so every re-export uses the explicit ``X as X`` form.
"""

from __future__ import annotations

from gauntlet.suite.linter import LintFinding as LintFinding
from gauntlet.suite.linter import LintSeverity as LintSeverity
from gauntlet.suite.linter import lint_suite as lint_suite
from gauntlet.suite.loader import load_suite as load_suite
from gauntlet.suite.loader import load_suite_from_string as load_suite_from_string
from gauntlet.suite.schema import BUILTIN_BACKEND_IMPORTS as BUILTIN_BACKEND_IMPORTS
from gauntlet.suite.schema import SAMPLING_MODES as SAMPLING_MODES
from gauntlet.suite.schema import AxisSpec as AxisSpec
from gauntlet.suite.schema import SamplingMode as SamplingMode
from gauntlet.suite.schema import Suite as Suite
from gauntlet.suite.schema import SuiteCell as SuiteCell

__all__ = [
    "BUILTIN_BACKEND_IMPORTS",
    "SAMPLING_MODES",
    "AxisSpec",
    "LintFinding",
    "LintSeverity",
    "SamplingMode",
    "Suite",
    "SuiteCell",
    "lint_suite",
    "load_suite",
    "load_suite_from_string",
]
