"""Security regression: ``--override`` parser does not shell out.

Phase 2.5 Task 16 — pins the existing behaviour of
:func:`gauntlet.replay.parse_override`. The parser splits a string on
exactly one ``=``, treats the LHS as an axis-name string, and passes
the RHS to :func:`float`. There is no shell invocation, no
``eval`` / ``exec``, and no template expansion.

Threat model — the honest claim:

* The parser does not "defend against shell injection" because the
  parser does not shell out in the first place. The only way an
  attacker-controlled override string can produce an effect is by
  shifting through ``parse_override`` -> :func:`validate_overrides`
  -> :class:`Suite.axes` membership check, then through
  :class:`gauntlet.runner.runner.Runner` as a perturbation value.

* The pin is therefore: shell-metachar payloads either parse cleanly
  (axis name = the verbatim LHS, value = a float; downstream axis-name
  validation will then reject because ``"axis; rm -rf /"`` is not in
  :data:`gauntlet.env.perturbation.AXIS_NAMES`) or raise the documented
  :class:`OverrideError`. Anything else — :class:`subprocess.CalledProcessError`,
  :class:`PermissionError`, an actual file write — is a real defect.

These tests are concrete-string regression pins on top of the
property-based fuzzing in ``tests/test_property_replay_overrides.py``;
they exist so the next CI run after a parser refactor surfaces a
specific named attack the new parser missed.

References:
* https://owasp.org/www-community/attacks/Command_Injection — class.
* https://cwe.mitre.org/data/definitions/78.html — CWE-78.
"""

from __future__ import annotations

import pytest

from gauntlet.replay import OverrideError, parse_override

# Concrete shell-metachar attack strings. Each MUST either parse cleanly
# or raise ``OverrideError``; nothing else may escape.
_SHELL_METACHAR_STRINGS: list[str] = [
    # Command separators on the LHS — axis name absorbs the metacharacter
    # but the RHS still must parse as a float. ``"axis; rm -rf /"`` has a
    # ``=`` later; we want strings WITHOUT a valid float RHS too.
    "axis;rm -rf /=1",
    "axis;rm /tmp/x=1.0",
    "axis|cat /etc/passwd=1",
    "axis&whoami=1",
    "axis`whoami`=1",
    "axis$(whoami)=1",
    "axis${HOME}=1",
    "axis>/tmp/out=1",
    "axis<input.txt=1",
    "axis\\nrm /tmp/x=1",
    # Newline injection (would be lethal in a logger that interpolates).
    "axis\n=1",
    # Carriage return.
    "axis\r=1",
    # Argument-injection-shaped payload: an extra "=" implies double
    # assignment on the LHS, which the parser refuses outright.
    "axis=1; cat /etc/passwd",
    "axis=$(whoami)",
    "axis=`id`",
    "axis=${HOME}",
    # Path-traversal in the value.
    "axis=../../etc/passwd",
    # Null byte.
    "axis\x00=1",
    "axis=\x00",
    # Unicode that some parsers normalise to '='. FULLWIDTH EQUALS
    # SIGN (U+FF1D) is distinct from ASCII '='; we build the literal
    # via ``chr`` so ruff's RUF001 (ambiguous chars) does not flag.
    f"axis{chr(0xFF1D)}1",
    # Long string (DoS shape — must complete in <100ms).
    "a" * 10_000 + "=1",
]


@pytest.mark.parametrize("payload", _SHELL_METACHAR_STRINGS)
def test_shell_metachar_payload_does_not_escape_envelope(payload: str) -> None:
    """Each payload either parses cleanly or raises ``OverrideError``.

    Anything else (e.g. ``subprocess`` raising a ``FileNotFoundError``,
    a real shell command running) is a security regression. Note: the
    parser strips whitespace, so a trailing ``\\n`` may parse as an
    empty axis-name -> ``OverrideError``.
    """
    try:
        name, value = parse_override(payload)
    except OverrideError:
        return  # acceptable failure mode
    # If it parsed, the contract is exact — name is a string, value
    # is a float. The axis-name validation step (in validate_overrides
    # against the suite) is what catches the ``"axis;rm -rf /"`` case
    # downstream; here we only assert the parser itself stayed in its
    # documented envelope.
    assert isinstance(name, str)
    assert isinstance(value, float)


def test_known_clean_payload_still_parses() -> None:
    """Positive control: a clean ``axis=1.0`` still parses.

    Mirrors :mod:`tests.test_property_replay_overrides` but as a
    standalone regression pin so this file's tests are self-contained
    even if the property suite is skipped (e.g. on a hypothesis-stripped
    minimal install)."""
    name, value = parse_override("lighting_intensity=0.5")
    assert name == "lighting_intensity"
    assert value == 0.5


def test_no_subprocess_module_imported_during_parse() -> None:
    """Defence in depth: ``parse_override`` MUST NOT import :mod:`subprocess`.

    The parser path is pure-Python string slicing + ``float()``. If a
    refactor ever adds a shell-out for "convenience" (e.g. expanding
    ``$VAR`` via ``subprocess.check_output``), this test catches it
    immediately. We sniff ``sys.modules`` before/after to detect a
    lazy import that would otherwise stay invisible.
    """
    import sys

    before = "subprocess" in sys.modules
    parse_override("lighting_intensity=0.5")
    after = "subprocess" in sys.modules
    # If subprocess was already imported by the test harness (typer's
    # CliRunner uses it transitively) we cannot assert it's absent
    # afterwards — but we CAN assert the call did not newly import it.
    assert before == after or before is True, (
        "parse_override newly imported subprocess; security model broken"
    )
