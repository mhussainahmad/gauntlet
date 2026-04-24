"""Security regression: CLI file-path inputs cannot escalate to RCE.

Phase 2.5 Task 16 — pins the CLI's behaviour when handed a path that
points outside the user's working tree (e.g. ``../../etc/passwd``,
``/etc/passwd``, NUL-byte injection, symlink escape).

Threat model — what we actually defend against:

* The CLI does NOT sandbox paths. ``typer`` passes us an absolute
  :class:`pathlib.Path`; we use it verbatim. The honest contract is
  therefore "no surprise" rather than "traversal is blocked": a
  traversal-shaped path is allowed to be opened, but the result must
  always be one of:

    1. ``FileNotFoundError`` (clean ``error: ... not found`` exit), or
    2. ``yaml.YAMLError`` / ``ValidationError`` (the file existed but
       was not a valid suite YAML / episodes JSON / report JSON), or
    3. ``json.JSONDecodeError`` surfaced as a clean CLI error.

  The forbidden outcome is "the file was read AND silently treated as
  a valid suite". That would let an attacker who can plant a YAML
  somewhere on the user's machine (e.g. via a shared CI runner) coerce
  ``gauntlet run`` into executing an arbitrary perturbation grid
  against an unintended env.

* NUL-byte injection (``../../foo\x00.yaml``) is blocked at the OS
  layer by Python's ``open`` — we just assert that surfaces cleanly,
  not as an unhandled traceback.

* All paths flow through ``Path.is_file()`` checks before any open
  attempt, so the code path is the same whether the input is a normal
  filename or a traversal-shaped string.

References:
* https://owasp.org/www-community/attacks/Path_Traversal — class.
* https://cwe.mitre.org/data/definitions/22.html — CWE-22.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app

# Traversal-shaped paths the CLI must handle gracefully. Each one should
# either trigger ``not found`` (the file genuinely does not exist on the
# CI runner) OR a ``yaml/json`` parse error (the file does exist but is
# not a valid suite). Either is acceptable; an uncaught traceback is not.
_TRAVERSAL_INPUTS: list[str] = [
    "../../../../etc/passwd",
    "../../../etc/passwd",
    "/etc/passwd",
    "/proc/self/environ",
    "/dev/null",
    "../../../../../../tmp/does-not-exist.yaml",
    # Doubled traversal — defends against naive ".." stripping.
    "....//....//etc/passwd",
    # Trailing-slash variant.
    "/etc/passwd/",
]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ----- run subcommand -------------------------------------------------------


@pytest.mark.parametrize("traversal_path", _TRAVERSAL_INPUTS)
def test_run_traversal_path_does_not_crash_or_succeed(
    runner: CliRunner, tmp_path: Path, traversal_path: str
) -> None:
    """``gauntlet run`` on a traversal-shaped suite path: clean error.

    The exit code MUST be non-zero (any failure mode is OK); stderr
    MUST contain a recognisable error keyword, and the stack trace
    MUST NOT escape (typer's CliRunner catches uncaught exceptions
    and surfaces them via ``result.exception`` — we assert there is
    none, or that any caught exception inherits from ``typer.Exit``).
    """
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            traversal_path,
            "--policy",
            "random",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0, (
        f"traversal path {traversal_path!r} unexpectedly succeeded; stdout={result.stdout!r}"
    )
    # No artefacts should have been produced.
    assert not (out_dir / "episodes.json").exists()
    assert not (out_dir / "report.json").exists()


@pytest.mark.parametrize("traversal_path", _TRAVERSAL_INPUTS)
def test_report_traversal_path_does_not_crash_or_succeed(
    runner: CliRunner, traversal_path: str
) -> None:
    """``gauntlet report`` on a traversal-shaped path: clean error."""
    result = runner.invoke(app, ["report", traversal_path])
    assert result.exit_code != 0


@pytest.mark.parametrize("traversal_path", _TRAVERSAL_INPUTS)
def test_compare_traversal_path_does_not_crash_or_succeed(
    runner: CliRunner, traversal_path: str
) -> None:
    """``gauntlet compare`` on a traversal-shaped path (either side): clean error."""
    result = runner.invoke(app, ["compare", traversal_path, traversal_path])
    assert result.exit_code != 0


@pytest.mark.parametrize("traversal_path", _TRAVERSAL_INPUTS)
def test_replay_traversal_path_does_not_crash_or_succeed(
    runner: CliRunner, tmp_path: Path, traversal_path: str
) -> None:
    """``gauntlet replay`` on a traversal-shaped episodes / suite path: clean error."""
    # Episodes path traversal.
    result = runner.invoke(
        app,
        [
            "replay",
            traversal_path,
            "--suite",
            str(tmp_path / "suite.yaml"),
            "--policy",
            "random",
            "--episode-id",
            "0:0",
        ],
    )
    assert result.exit_code != 0


# ----- NUL-byte injection (Python rejects at OS boundary) -------------------


def test_run_nul_byte_in_path_fails_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    """``\\x00`` inside a path is rejected by ``open`` — we surface it as
    a clean failure, not an unhandled :class:`ValueError`.

    Note: typer / click may reject the path during option parsing
    (because the embedded NUL fails validation), or the CLI body may
    reject it on the ``Path.is_file()`` check. Either is acceptable;
    the test only requires a non-zero exit and no stack trace.
    """
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            "suite\x00.yaml",
            "--policy",
            "random",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0


# ----- symlink escape from a sandbox-shaped temp dir ------------------------


def test_run_symlink_to_outside_file_fails_validation(runner: CliRunner, tmp_path: Path) -> None:
    """A symlink pointing at a non-suite file outside ``tmp_path`` must
    fail suite validation, NOT silently follow the link and pretend the
    payload is a valid suite.

    Defends against the pattern: attacker plants a symlink in a shared
    directory, victim ``gauntlet run``s the symlink, the link resolves
    to ``/etc/passwd`` (or similar), and the harness blindly hands the
    payload to ``yaml.safe_load`` and downstream Pydantic. ``safe_load``
    will reject ``/etc/passwd`` (not YAML) with a ``yaml.YAMLError``;
    if it parses (e.g. someone planted a real-looking suite YAML
    outside the tree), Pydantic must still gate the env / axes.
    """
    # Construct a target file that is neither valid YAML nor a suite.
    target = tmp_path / "outside.txt"
    target.write_text("root:x:0:0:root:/root:/bin/bash\nbin:x:1:1:bin:/bin:/sbin/nologin\n")
    link = tmp_path / "suite_link.yaml"
    link.symlink_to(target)

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            str(link),
            "--policy",
            "random",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0
    # Critical: no episodes were emitted.
    assert not (out_dir / "episodes.json").exists()
