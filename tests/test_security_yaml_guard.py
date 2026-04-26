"""Security regression: gauntlet.security.yaml_guard wrapper + no-yaml.load gate.

Phase 2.5 Task 16 — pins the canonical-YAML-entry-point contract.

Two layers:

1. **Behavioural** — ``safe_yaml_load`` must round-trip a clean dict and
   reject every PyYAML unsafe-tag class (``!!python/object/apply``,
   ``!!python/object/new``, ``!!python/name``, top-level ``!!python/
   object``). The contract here mirrors
   ``tests/test_security_yaml.py``, which pins the same envelope at
   the suite-loader call-site; this module pins it at the helper
   itself so a future caller that imports ``safe_yaml_load`` directly
   is also covered.

2. **Static / grep gate** — no occurrence of the substring
   ``yaml.load(`` (without ``safe_``) may appear under
   ``src/gauntlet`` outside ``src/gauntlet/security/yaml_guard.py``.
   Equivalent to a ruff custom check, but a subprocess test is
   simpler and runs in the same default-job pytest sweep.

References:
* https://pyyaml.org/wiki/PyYAMLDocumentation#loading-yaml — the
  ``yaml.load`` vs ``yaml.safe_load`` distinction.
* https://nvd.nist.gov/vuln/detail/CVE-2017-18342 — PyYAML
  arbitrary-code-execution class CVE.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from gauntlet.security import YamlSecurityError, safe_yaml_load

# ----- module sentinel — same shape as tests/test_security_yaml.py ----------


_SHOULD_NOT_RUN = False


def _mark_executed() -> bool:
    """Sentinel target. Must NEVER be invoked by safe_yaml_load."""
    global _SHOULD_NOT_RUN
    _SHOULD_NOT_RUN = True
    return True


# Documented loader error envelope. ``yaml.constructor.ConstructorError``
# is a ``yaml.YAMLError`` subclass; YamlSecurityError is included for
# forward compatibility (the current wrapper does not raise it).
_DOCUMENTED_ERRORS = (ValueError, yaml.YAMLError, YamlSecurityError)


# ----- 1. Behavioural — round-trip + unsafe-tag rejection -------------------


def test_safe_yaml_load_round_trips_simple_dict() -> None:
    """Positive control: clean dict YAML round-trips through the wrapper."""
    yaml_text = "name: clean\nepisodes_per_cell: 1\n"
    parsed = safe_yaml_load(yaml_text)
    assert parsed == {"name": "clean", "episodes_per_cell": 1}


def test_safe_yaml_load_round_trips_nested_structure() -> None:
    """Nested mappings + sequences also round-trip."""
    yaml_text = "axes:\n  lighting_intensity:\n    low: 0.3\n    high: 0.9\n"
    parsed = safe_yaml_load(yaml_text)
    assert parsed == {"axes": {"lighting_intensity": {"low": 0.3, "high": 0.9}}}


def test_safe_yaml_load_handles_file_like_input(tmp_path: Path) -> None:
    """A text file handle is accepted — same surface as yaml.safe_load."""
    p = tmp_path / "doc.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    with p.open("r", encoding="utf-8") as fh:
        parsed = safe_yaml_load(fh)
    assert parsed == {"a": 1}


_APPLY_PAYLOAD = "engine: !!python/object/apply:tests.test_security_yaml_guard._mark_executed []\n"

_NEW_PAYLOAD = "engine: !!python/object/new:tests.test_security_yaml_guard._mark_executed []\n"

_NAME_PAYLOAD = "engine: !!python/name:os.system\n"

_OBJECT_PAYLOAD = "!!python/object:builtins.dict\nname: attack\n"


@pytest.mark.parametrize(
    ("payload", "label"),
    [
        (_APPLY_PAYLOAD, "!!python/object/apply"),
        (_NEW_PAYLOAD, "!!python/object/new"),
        (_NAME_PAYLOAD, "!!python/name"),
        (_OBJECT_PAYLOAD, "top-level !!python/object"),
    ],
)
def test_safe_yaml_load_rejects_unsafe_tags(payload: str, label: str) -> None:
    """Every PyYAML unsafe tag must raise — regardless of which YAML
    construction the attacker chose.

    The sentinel ``_SHOULD_NOT_RUN`` is reset before each call so even
    a future bug that masked the exception would be caught by the
    side-effect check.
    """
    global _SHOULD_NOT_RUN
    _SHOULD_NOT_RUN = False
    with pytest.raises(_DOCUMENTED_ERRORS):
        safe_yaml_load(payload)
    assert _SHOULD_NOT_RUN is False, (
        f"safe_yaml_load executed _mark_executed() on payload tagged with "
        f"{label!r} — wrapper is no longer routing through yaml.safe_load"
    )


# ----- 2. Static / grep gate — no bare yaml.load( in src/gauntlet ----------


# The wrapper module is the ONLY file allowed to mention the literal
# ``yaml.load(`` substring. Even there, the live call uses
# ``yaml.safe_load(...)``; the documented-exemption clause is for
# in-file commentary about the gate itself.
_WRAPPER_RELPATH = Path("src/gauntlet/security/yaml_guard.py")


def _repo_root() -> Path:
    """Resolve the repository root regardless of pytest invocation cwd."""
    here = Path(__file__).resolve()
    # tests/ is one level under repo root.
    return here.parent.parent


def test_no_unsafe_yaml_load_in_src() -> None:
    """``yaml.load(`` (without ``safe_``) MUST NOT appear under
    ``src/gauntlet`` outside ``src/gauntlet/security/yaml_guard.py``.

    Implementation: ``grep -rnE`` with ``yaml\\.load\\(`` (escaped dot,
    anchored open-paren). The regex MUST NOT match ``yaml.safe_load(`` —
    verified by the assertion in
    :func:`test_grep_pattern_does_not_match_safe_load`.

    Returncode contract:
    * 0 → grep found at least one match → fail (a real bare yaml.load).
    * 1 → grep found zero matches → pass (gate intact).
    * >1 → grep itself errored → surface to the test runner.
    """
    grep = shutil.which("grep")
    if grep is None:  # pragma: no cover - grep is in coreutils on every CI image
        pytest.skip("grep not available on this platform")

    src_dir = _repo_root() / "src" / "gauntlet"
    assert src_dir.is_dir(), f"src/gauntlet not found at {src_dir}"

    result = subprocess.run(
        [grep, "-rnE", r"yaml\.load\(", str(src_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        # grep found matches — strip the wrapper file itself, since it
        # is permitted to mention the gate target in commentary.
        offending: list[str] = []
        for line in result.stdout.splitlines():
            # Each line is ``<path>:<lineno>:<content>``.
            path_part, _, _ = line.partition(":")
            rel = Path(path_part).resolve().relative_to(_repo_root())
            if rel == _WRAPPER_RELPATH:
                continue
            offending.append(line)
        assert not offending, (
            "Unsafe yaml.load( occurrences detected under src/gauntlet "
            "outside the canonical wrapper. Replace with "
            "gauntlet.security.safe_yaml_load. Findings:\n" + "\n".join(offending)
        )
    elif result.returncode == 1:
        # No matches at all — gate intact.
        return
    else:
        pytest.fail(
            f"grep exited with code {result.returncode}; stderr={result.stderr!r}",
        )


def test_grep_pattern_does_not_match_safe_load() -> None:
    """Sanity: the pattern ``yaml\\.load\\(`` must NOT match ``yaml.safe_load(``.

    Without this check, a bug in the regex (forgetting to escape the
    dot, or using ``.*load`` instead of ``.load``) would cause the gate
    to false-positive on every legitimate ``yaml.safe_load(...)`` call
    — and we would silently weaken the contract by adding allowlist
    exemptions to compensate.
    """
    grep = shutil.which("grep")
    if grep is None:  # pragma: no cover - grep is in coreutils on every CI image
        pytest.skip("grep not available on this platform")
    sample = "result = yaml.safe_load(stream)\n"
    result = subprocess.run(
        [grep, "-E", r"yaml\.load\("],
        input=sample,
        capture_output=True,
        text=True,
        check=False,
    )
    # Returncode 1 = grep found nothing; that is the contract.
    assert result.returncode == 1, (
        f"Pattern incorrectly matches yaml.safe_load(. stdout={result.stdout!r}"
    )


def test_yaml_security_error_is_a_value_error() -> None:
    """:class:`YamlSecurityError` is a :class:`ValueError` subclass.

    Pinned for forward compatibility — when a strict-mode policy
    eventually raises this on top of the safe-load envelope, callers
    that catch ``ValueError`` already work without a churn-only PR.
    """
    assert issubclass(YamlSecurityError, ValueError)
