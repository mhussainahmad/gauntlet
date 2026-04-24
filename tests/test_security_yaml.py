"""Security regression: YAML loader rejects unsafe deserialization tags.

Phase 2.5 Task 16 — pins the existing safe-by-default behaviour of
:func:`gauntlet.suite.loader.load_suite` /
:func:`gauntlet.suite.loader.load_suite_from_string`. Both funnel
through :func:`yaml.safe_load`, which raises
:class:`yaml.constructor.ConstructorError` (a :class:`yaml.YAMLError`
subclass) on PyYAML's unsafe tags such as ``!!python/object/apply``,
``!!python/object/new``, and ``!!python/name``. If somebody ever swaps
``yaml.safe_load`` for ``yaml.load`` without specifying ``Loader=``,
these tests catch the regression — an attacker-controlled suite YAML
must NEVER execute arbitrary Python on parse.

The tests double-assert: (a) the loader raises a documented error type
on the malicious payload, and (b) the side effect the payload would
have produced (mutation of a module-level sentinel) did NOT happen.
That second arm is what makes this an honest security pin instead of a
stylistic one — even if some future loader masked the exception, the
sentinel check would still flag.

References:
* https://pyyaml.org/wiki/PyYAMLDocumentation#loading-yaml — the
  ``yaml.load`` vs ``yaml.safe_load`` distinction.
* https://nvd.nist.gov/vuln/detail/CVE-2017-18342 — the canonical
  PyYAML arbitrary-code-execution class CVE that motivates this pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gauntlet.suite.loader import load_suite, load_suite_from_string

# Module-level sentinel. A successful arbitrary-code-execution attack via
# ``!!python/object/apply:tests.test_security_yaml._mark_executed`` would
# flip this flag. Every test asserts it stays False.
_SHOULD_NOT_RUN = False


def _mark_executed() -> bool:
    """Sentinel target — must NEVER be invoked by the YAML loader.

    The unsafe-tag attack strings reference this symbol so that, in the
    counterfactual world where the loader is wired to ``yaml.load`` with
    no Loader argument, parsing the payload would set
    :data:`_SHOULD_NOT_RUN` to True. Tests assert the flag stays False.
    """
    global _SHOULD_NOT_RUN
    _SHOULD_NOT_RUN = True
    return True


# Documented loader error envelope (mirrors tests/test_property_suite_loader.py).
# ``yaml.constructor.ConstructorError`` is a ``yaml.YAMLError`` subclass.
_DOCUMENTED_ERRORS = (ValueError, yaml.YAMLError)


# ----- !!python/object/apply ------------------------------------------------


_APPLY_PAYLOAD = (
    "name: attack-via-apply\n"
    "env: tabletop\n"
    "episodes_per_cell: !!python/object/apply:tests.test_security_yaml._mark_executed []\n"
    "axes:\n"
    "  lighting_intensity:\n"
    "    low: 0.3\n"
    "    high: 0.9\n"
    "    steps: 2\n"
)


def test_python_object_apply_tag_is_rejected_from_string() -> None:
    """``!!python/object/apply`` (RCE class) MUST raise on safe_load."""
    global _SHOULD_NOT_RUN
    _SHOULD_NOT_RUN = False
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite_from_string(_APPLY_PAYLOAD)
    assert _SHOULD_NOT_RUN is False, (
        "yaml loader executed _mark_executed() — RCE attack succeeded; "
        "loader is no longer using yaml.safe_load"
    )


def test_python_object_apply_tag_is_rejected_from_file(tmp_path: Path) -> None:
    """File-based loader path: same payload, same rejection contract."""
    global _SHOULD_NOT_RUN
    _SHOULD_NOT_RUN = False
    suite_path = tmp_path / "attack.yaml"
    suite_path.write_text(_APPLY_PAYLOAD, encoding="utf-8")
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite(suite_path)
    assert _SHOULD_NOT_RUN is False


# ----- !!python/object/new --------------------------------------------------


_NEW_PAYLOAD = (
    "name: attack-via-new\n"
    "env: tabletop\n"
    "episodes_per_cell: !!python/object/new:tests.test_security_yaml._mark_executed []\n"
    "axes:\n"
    "  lighting_intensity:\n"
    "    low: 0.3\n"
    "    high: 0.9\n"
    "    steps: 2\n"
)


def test_python_object_new_tag_is_rejected() -> None:
    """``!!python/object/new`` is a sibling RCE vector — must also raise."""
    global _SHOULD_NOT_RUN
    _SHOULD_NOT_RUN = False
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite_from_string(_NEW_PAYLOAD)
    assert _SHOULD_NOT_RUN is False


# ----- !!python/name --------------------------------------------------------


_NAME_PAYLOAD = (
    "name: attack-via-name\n"
    "env: tabletop\n"
    "episodes_per_cell: !!python/name:os.system\n"
    "axes:\n"
    "  lighting_intensity:\n"
    "    low: 0.3\n"
    "    high: 0.9\n"
    "    steps: 2\n"
)


def test_python_name_tag_is_rejected() -> None:
    """``!!python/name`` (resolves to a Python attribute) — must also raise.

    This tag does not invoke anything by itself, but on an unsafe loader
    it would resolve ``os.system`` into the parsed document, giving a
    downstream caller (or a clever Pydantic validator) a primitive that
    could be turned into RCE. ``yaml.safe_load`` rejects it outright.
    """
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite_from_string(_NAME_PAYLOAD)


# ----- !!python/object (bare) -----------------------------------------------


_OBJECT_PAYLOAD = (
    "!!python/object:builtins.dict\n"
    "name: attack\n"
    "env: tabletop\n"
    "episodes_per_cell: 1\n"
    "axes:\n"
    "  lighting_intensity:\n"
    "    low: 0.3\n"
    "    high: 0.9\n"
    "    steps: 2\n"
)


def test_python_object_top_level_tag_is_rejected() -> None:
    """A top-level ``!!python/object`` document tag must also be rejected."""
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite_from_string(_OBJECT_PAYLOAD)


# ----- positive control -----------------------------------------------------


def test_safe_load_baseline_still_accepts_well_formed_yaml() -> None:
    """Positive control: the same loader still parses a clean suite.

    Catches the case where someone over-corrects the safe-load path and
    breaks the happy path entirely (e.g. stripping the YAML loader to a
    JSON-only parser). If this regresses we'll know the protection went
    too far, not just that the attack tests still pass."""
    yaml_text = (
        "name: baseline-clean\n"
        "env: tabletop\n"
        "episodes_per_cell: 1\n"
        "axes:\n"
        "  lighting_intensity:\n"
        "    low: 0.3\n"
        "    high: 0.9\n"
        "    steps: 2\n"
    )
    suite = load_suite_from_string(yaml_text)
    assert suite.name == "baseline-clean"
    assert suite.env == "tabletop"
