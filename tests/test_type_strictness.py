"""``mypy --strict`` regression gate for ``src/gauntlet`` — Phase 2.5 T15.

Spawns ``python -m mypy --strict src/gauntlet`` in a subprocess and
asserts a clean exit. The project-wide ``[tool.mypy]`` block in
``pyproject.toml`` enables ``disallow_any_explicit = true`` (Phase 2.5
T15, PR #45) with two narrowly scoped per-module override buckets —
pydantic schema modules (where the v2 mypy plugin synthesises
``__mypy-replace(**kwargs: Any)``) and FFI seam modules (mujoco /
gymnasium / torch / lerobot / pybullet / genesis / isaacsim / rclpy /
imageio / pyarrow / yaml / importlib.metadata). Anything outside those
buckets must remain ``Any``-free.

Why a subprocess smoke test
---------------------------
The full mypy invocation is the contract. Re-implementing the audit in
Python (e.g. ``ast.parse`` + walking for ``Name(id="Any")``) would
silently drift from what mypy actually enforces — operator-typed
generic specialisations, ``cast(Any, ...)`` calls, ``**kwargs: Any``
synthesised by plugins, all of which mypy understands and a custom
walker would not. The subprocess approach is the boring, durable
shape: if mypy passes, the rule is enforced; if a future PR
re-introduces a top-level ``Any`` in a non-carved-out module, mypy
fails and so does this test.

Marked ``slow`` because mypy --strict on the full ``src/gauntlet``
tree takes several seconds in cold-cache CI runs — too long to live in
the default per-PR pytest budget. The same ``-m slow`` opt-in already
gates ``test_bench_smoke.py`` and ``test_api_docs_freshness.py``; this
test joins that bucket so the per-PR strictness ratchet is checked in
the dedicated slow-suite job rather than on every fast-loop run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TARGET = _REPO_ROOT / "src" / "gauntlet"


@pytest.mark.slow
def test_mypy_strict_clean_on_src_gauntlet() -> None:
    """``mypy --strict src/gauntlet`` exits 0 on every PR.

    The exit code is the contract; the captured stdout/stderr is
    surfaced verbatim on failure so the developer's fix-it loop is
    "open the file mypy named, look at the line mypy named, narrow
    the type or move the symbol behind an existing per-module
    carve-out". No new ``# type: ignore`` lines and no new top-level
    explicit ``Any`` literals — see ``docs/api.md`` "Type strictness"
    for the policy contract.
    """
    if not _TARGET.is_dir():
        msg = f"src/gauntlet not found at {_TARGET}; repo layout drifted"
        raise AssertionError(msg)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(_TARGET)],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    if result.returncode != 0:
        rendered = (
            f"mypy --strict src/gauntlet failed with exit {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
        raise AssertionError(rendered)
