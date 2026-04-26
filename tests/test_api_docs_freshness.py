"""docs/api.md freshness check — Phase 2.5 T14.

Walks every public sub-package's ``__all__`` and asserts that each
exported symbol is referenced in ``docs/api.md``. The reference need
not be a heading: a backtick mention inside an import block, a
parameter name, or a prose sentence is enough — the goal is "if a
symbol exists in the public surface, the API reference talks about
it somewhere", not "every symbol gets its own H3 heading".

Why coverage-style instead of diff-against-generator
----------------------------------------------------
The original Phase 2.5 plan called for a generator + diff pattern
(``scripts/gen_api_docs.py`` + ``diff -q`` against the checked-in
``docs/api.md``). PR #44 deliberately chose hand-curated prose with
RFC cross-links and runnable examples, and the polish-loop PRs since
have only added more curated context. A generator would either
destroy this voice (boilerplate lists with no anti-feature framing)
or mimic it (overkill — a templating language for one document).
This coverage check enforces what we actually care about — a new
public symbol forces an api.md edit — without dictating prose style
or being whitespace-flaky.

Marked ``slow`` per the Phase 2.5 plan: it walks every sub-package
on import, which pulls in the optional-extra lazy-import seams (and
in CI configurations where lerobot / pi0 / groot wheels are slow to
resolve, the import alone can dominate a default ``pytest`` budget).
The default profile skips it; ``pytest -m slow`` opts in.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

# Sub-packages whose ``__all__`` defines the documented public surface.
# Mirrors the import block at the top of ``src/gauntlet/__init__.py``'s
# module docstring and the per-section headings in ``docs/api.md``.
_PUBLIC_SUB_PACKAGES: tuple[str, ...] = (
    "gauntlet",
    "gauntlet.aggregate",
    "gauntlet.bisect",
    "gauntlet.compare",
    "gauntlet.dashboard",
    "gauntlet.diff",
    "gauntlet.env",
    "gauntlet.monitor",
    "gauntlet.policy",
    "gauntlet.realsim",
    "gauntlet.replay",
    "gauntlet.report",
    "gauntlet.ros2",
    "gauntlet.runner",
    "gauntlet.security",
    "gauntlet.suite",
)


# Symbols that the API reference deliberately covers as a *type alias*
# inside a paragraph rather than as a standalone heading or runnable
# example. Adding to this set is fine — but each entry needs a one-line
# comment explaining why it does not warrant its own heading. Keep
# small; prefer adding a heading to growing this set.
_DOC_COVERAGE_ALLOWLIST: frozenset[str] = frozenset()


_REPO_ROOT = Path(__file__).resolve().parents[1]
_API_DOC_PATH = _REPO_ROOT / "docs" / "api.md"


def _load_api_doc_text() -> str:
    """Read ``docs/api.md`` from the repo root.

    Lifted to a helper so the slow-marked test below stays focused on
    the assertion and the (small but non-zero) cost of opening the
    file is paid once.
    """
    if not _API_DOC_PATH.is_file():
        msg = f"docs/api.md not found at {_API_DOC_PATH}; T14 deliverable is missing"
        raise AssertionError(msg)
    return _API_DOC_PATH.read_text(encoding="utf-8")


def _public_symbols() -> list[tuple[str, str]]:
    """Enumerate every ``(module, symbol)`` pair from each sub-package's ``__all__``.

    Underscored symbols are filtered out — they would never appear in
    the public reference even if listed in ``__all__`` (the project
    enforces the underscore prefix convention via the linter).
    """
    pairs: list[tuple[str, str]] = []
    for module_name in _PUBLIC_SUB_PACKAGES:
        module = importlib.import_module(module_name)
        for sym in getattr(module, "__all__", ()):
            if sym.startswith("_"):
                continue
            pairs.append((module_name, sym))
    return pairs


def _is_referenced(symbol: str, text: str) -> bool:
    """Return ``True`` if *symbol* appears as a word-boundary match in *text*.

    Word-boundary rather than backtick-only because the reference is
    sometimes inside a multi-line import block (``    PairedComparison,``)
    where the trailing punctuation is a comma, not a backtick. Matching
    any non-identifier boundary catches every realistic prose form.
    """
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(symbol) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


@pytest.mark.slow
def test_api_doc_covers_every_public_symbol() -> None:
    """Every name in every sub-package's ``__all__`` is referenced in api.md.

    Failure surface is the symbol list itself — pytest prints the
    missing entries directly so the fix-it diff for the contributor
    is "open docs/api.md, add a one-paragraph mention next to the
    relevant section".
    """
    text = _load_api_doc_text()
    missing: list[tuple[str, str]] = []
    for module_name, sym in _public_symbols():
        if sym in _DOC_COVERAGE_ALLOWLIST:
            continue
        if not _is_referenced(sym, text):
            missing.append((module_name, sym))
    if missing:
        rendered = "\n".join(f"  - {mod}.{sym}" for mod, sym in missing)
        msg = (
            f"docs/api.md is missing references to {len(missing)} public "
            f"symbol(s); add a one-paragraph mention next to the relevant "
            f"section:\n{rendered}"
        )
        raise AssertionError(msg)
