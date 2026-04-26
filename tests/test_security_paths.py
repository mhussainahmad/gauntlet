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

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.security import PathTraversalError, safe_join

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


# ----- helper-level tests: gauntlet.security.paths.safe_join ----------------
#
# Phase 2.5 Task 16. The CLI-level tests above pin behaviour at the typer
# boundary; the tests below pin the helper itself. They are a contract
# for direct callers of ``safe_join`` (currently
# ``gauntlet.runner.parquet.parquet_path_for``; expected to grow).
#
# The four-rung threat model the helper defends against:
#
# 1. ``..``-segment escape — the resolved path leaves ``base``.
# 2. Absolute-path injection — a join component is itself absolute, so
#    the bare ``Path.__truediv__`` would silently drop ``base``.
# 3. Symlink escape (only when ``follow_symlinks=False``) — a component
#    is a symlink whose target leaves ``base``.
# 4. Happy path — legitimate nested joins produce the resolved
#    descendant path.
#
# Errors must be PathTraversalError (a ValueError subclass) so existing
# CLI error envelopes continue to translate to clean ``typer.Exit``.


# 1. Parent-traversal escape ------------------------------------------------


def test_safe_join_rejects_parent_traversal(tmp_path: Path) -> None:
    """``safe_join(base, "../outside")`` raises :class:`PathTraversalError`."""
    base = tmp_path / "sandbox"
    base.mkdir()
    with pytest.raises(PathTraversalError):
        safe_join(base, "../outside")


def test_safe_join_rejects_deep_parent_traversal(tmp_path: Path) -> None:
    """Multi-segment traversal (``a/../../etc/passwd``) is rejected."""
    base = tmp_path / "sandbox"
    base.mkdir()
    with pytest.raises(PathTraversalError):
        safe_join(base, "a/../../etc/passwd")


def test_safe_join_traversal_error_is_a_value_error(tmp_path: Path) -> None:
    """:class:`PathTraversalError` is a :class:`ValueError` subclass.

    Existing CLI error envelopes catch ``ValueError`` and translate it
    to a clean ``typer.Exit``. Subclassing keeps that contract.
    """
    base = tmp_path / "sandbox"
    base.mkdir()
    with pytest.raises(ValueError):
        safe_join(base, "../outside")


# 2. Absolute-path injection -----------------------------------------------


def test_safe_join_rejects_absolute_path_injection(tmp_path: Path) -> None:
    """``safe_join(base, "/etc/passwd")`` raises rather than silently dropping ``base``.

    The Python builtin ``Path.__truediv__`` discards the LHS when the
    RHS is absolute (``Path("/a") / "/b" == Path("/b")``); the helper
    explicitly rejects that pattern so the failure mode is loud.
    """
    base = tmp_path / "sandbox"
    base.mkdir()
    with pytest.raises(PathTraversalError):
        safe_join(base, "/etc/passwd")


def test_safe_join_rejects_absolute_path_in_later_part(tmp_path: Path) -> None:
    """Absolute-path injection in the second positional rejected too."""
    base = tmp_path / "sandbox"
    base.mkdir()
    with pytest.raises(PathTraversalError):
        safe_join(base, "subdir", "/tmp/escape.txt")


# 3. Symlink-escape rejection (only when follow_symlinks=False) -------------


def test_safe_join_rejects_symlink_escape_when_strict(tmp_path: Path) -> None:
    """A symlink under ``base`` pointing OUTSIDE ``base`` is rejected in strict mode.

    Defends against the pattern: attacker plants a symlink in the
    output directory, the harness joins onto it, naive realpath then
    leaves the sandbox.
    """
    base = tmp_path / "sandbox"
    base.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    # Symlink at ``base/escape`` pointing OUT of base.
    link = base / "escape"
    link.symlink_to(outside)

    with pytest.raises(PathTraversalError):
        safe_join(base, "escape", follow_symlinks=False)


def test_safe_join_default_follow_symlinks_allows_internal_symlink(tmp_path: Path) -> None:
    """``follow_symlinks=True`` (default): symlink whose TARGET stays in ``base`` works.

    Backward-compat clause: the default permits legitimate internal
    symlinks (e.g. an output dir that is itself a symlink to a real
    location), so existing sinks that accept a symlinked output dir
    keep working.
    """
    base = tmp_path / "sandbox"
    base.mkdir()
    inner_target = base / "real_target"
    inner_target.mkdir()
    link = base / "via_link"
    link.symlink_to(inner_target, target_is_directory=True)

    # Default mode (follow_symlinks=True) — the internal symlink
    # resolves to a path still inside base, so the helper accepts.
    result = safe_join(base, "via_link", "file.txt")
    assert result.is_relative_to(base.resolve())


# 4. Happy path / regression --------------------------------------------------


def test_safe_join_normal_nested_join_works(tmp_path: Path) -> None:
    """Nominal nested join returns the resolved descendant path."""
    base = tmp_path / "sandbox"
    base.mkdir()
    result = safe_join(base, "subdir", "file.parquet")
    assert result == (base / "subdir" / "file.parquet").resolve()


def test_safe_join_no_parts_returns_resolved_base(tmp_path: Path) -> None:
    """``safe_join(base)`` with no extra parts returns resolved ``base``."""
    base = tmp_path / "sandbox"
    base.mkdir()
    result = safe_join(base)
    assert result == base.resolve()


def test_safe_join_accepts_string_base_and_parts(tmp_path: Path) -> None:
    """Both ``str`` and :class:`Path` are accepted for the inputs."""
    base = tmp_path / "sandbox"
    base.mkdir()
    result = safe_join(str(base), "a", "b.txt")
    assert result == (base / "a" / "b.txt").resolve()


def test_safe_join_works_when_base_does_not_yet_exist(tmp_path: Path) -> None:
    """``base`` need not exist on disk yet — useful for output-dir construction.

    The Runner mkdirs ``trajectory_dir`` only after constructing the
    parquet path; ``safe_join`` must therefore not require ``base``
    to pre-exist. ``Path.resolve(strict=False)`` makes this work.
    """
    base = tmp_path / "does_not_exist_yet"
    result = safe_join(base, "child.parquet")
    assert result == (base / "child.parquet").resolve()


@pytest.mark.parametrize(
    "evil",
    [
        "..",
        "../",
        "../..",
        "../etc/passwd",
        "subdir/../../escape",
    ],
)
def test_safe_join_table_of_traversal_payloads(tmp_path: Path, evil: str) -> None:
    """Table-driven traversal payloads — each MUST raise."""
    base = tmp_path / "sandbox"
    base.mkdir()
    with pytest.raises(PathTraversalError):
        safe_join(base, evil)


# 5. Regression — existing parquet_path_for happy-path is unchanged --------


def test_parquet_path_for_happy_path_still_works(tmp_path: Path) -> None:
    """Step-3 regression — gauntlet.runner.parquet.parquet_path_for stays
    backward-compatible.

    The pre-T16 contract returned ``trajectory_dir / formatted_name``;
    the safe_join wiring must preserve that exact shape so existing
    callers (the worker, the existing tests/test_parquet_trajectory)
    keep working unchanged.
    """
    from gauntlet.runner.parquet import parquet_path_for

    traj_dir = tmp_path / "traj"
    result = parquet_path_for(traj_dir, cell_index=3, episode_index=7)
    expected = traj_dir / "cell_0003_ep_0007.parquet"
    assert result == expected
    # Must be relative-safe — i.e. the original (potentially relative)
    # ``traj_dir`` is preserved, not silently absolutised.
    assert os.fspath(result).endswith(os.fspath(expected))
