"""Smoke tests for the Phase 2.5 T12 performance benchmark scripts.

Each test invokes one ``scripts/perf/bench_*.py`` CLI with the
smallest possible budget (``--episodes 2`` / ``--steps 5`` /
``--cells 10`` / ``--workers 1``) under a tmp-path-scoped ``--output``
and asserts the resulting JSON sidecar has the spec'd shape.

Marked ``@pytest.mark.slow`` so the default pytest run skips them — they
spawn subprocesses (each invocation is a fresh Python process that
imports gauntlet, so wall is hundreds of ms even at smallest budgets)
and the project's testing-policy carve-out keeps the default suite
laptop-friendly. Opt in with ``pytest -m slow tests/test_bench_smoke.py``
or ``-m "slow and not render"`` when the host has no offscreen GL.

Why subprocess.run instead of importing each module's ``main()``:
    1. We exercise the actual CLI argument parser the docs document.
    2. Each bench writes to a JSON sidecar; subprocess isolation means a
       crash in one bench does not poison the test process for the
       next.
    3. The render bench needs a fresh process to retry the
       offscreen GL probe — once a MuJoCo Renderer fails inside a
       process the failure tends to stick (cached in module state).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_PERF_DIR: Path = _REPO_ROOT / "scripts" / "perf"


pytestmark = pytest.mark.slow


def _run_bench(
    *,
    script: str,
    args: list[str],
    output_path: Path,
) -> dict[str, object]:
    """Invoke one bench script and return the parsed JSON sidecar.

    Each bench writes the same content (a) as the trailing JSON line on
    stdout and (b) as the JSON sidecar file. We assert both — the CLI
    contract is "either source of the JSON is valid" and a regression
    in only one of them is exactly what this smoke test exists to
    catch.
    """
    script_path = _PERF_DIR / script
    full_args = [sys.executable, str(script_path), "--output", str(output_path), *args]
    result = subprocess.run(
        full_args,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script} returned {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert output_path.is_file(), f"{script} did not write {output_path}"
    sidecar = json.loads(output_path.read_text(encoding="utf-8"))
    # Belt-and-braces: the trailing line on stdout MUST be valid JSON
    # (the docs promise ``tail -n 1 | jq`` works). Capture the last
    # non-empty line and parse it; abort on any drift.
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert stdout_lines, f"{script} produced no stdout"
    trailing = json.loads(stdout_lines[-1])
    # The trailing line and the sidecar agree on the headline fields
    # (the sidecar is pretty-printed with newlines so byte-identity is
    # not the contract; field equality is).
    for key in ("name", "version", "timestamp", "git_commit"):
        assert trailing.get(key) == sidecar.get(key), (
            f"{script}: stdout/sidecar disagree on {key!r} "
            f"(stdout={trailing.get(key)!r}, sidecar={sidecar.get(key)!r})"
        )
    return dict(sidecar)


def _assert_provenance_fields(payload: dict[str, object]) -> None:
    """Every T12 sidecar carries version / timestamp / git_commit."""
    assert "version" in payload
    assert "timestamp" in payload
    assert "git_commit" in payload  # may be None on a non-git checkout
    # version is a non-empty string (gauntlet.__version__).
    version = payload["version"]
    assert isinstance(version, str) and version, "version must be a non-empty str"
    # timestamp is the YYYY-MM-DDTHH:MM:SSZ shape from utc_iso8601_now().
    timestamp = payload["timestamp"]
    assert isinstance(timestamp, str), "timestamp must be a str"
    assert timestamp.endswith("Z"), f"timestamp not UTC-suffixed: {timestamp!r}"
    # partial flag is required and defaults False on a clean run.
    assert payload.get("partial") is False, "smoke run should land partial=false"


def test_bench_rollout_smoke(tmp_path: Path) -> None:
    """``scripts/perf/bench_rollout.py --episodes 2`` produces a valid sidecar."""
    out = tmp_path / "rollout.json"
    sidecar = _run_bench(
        script="bench_rollout.py",
        args=["--backend", "mujoco", "--episodes", "2"],
        output_path=out,
    )
    assert sidecar["name"] == "bench_rollout"
    assert sidecar["backend"] == "mujoco"
    assert sidecar["episodes"] == 2
    assert sidecar.get("skipped") is False
    # Throughput / latency keys are spec'd by the T12 task.
    for key in (
        "episodes_per_sec",
        "step_mean_ms",
        "step_p50_ms",
        "step_p95_ms",
        "step_p99_ms",
    ):
        assert key in sidecar, f"missing {key!r} in rollout sidecar"
        assert isinstance(sidecar[key], (int, float))
    _assert_provenance_fields(sidecar)


def test_bench_render_smoke(tmp_path: Path) -> None:
    """``scripts/perf/bench_render.py --steps 5`` produces a valid sidecar.

    Skip-not-fail: when the headless-GL probe inside the bench cannot
    construct an offscreen renderer, the bench writes a sidecar with
    ``skipped=True`` and a populated ``skip_reason``. Both states are
    valid; we only check shape.
    """
    out = tmp_path / "render.json"
    sidecar = _run_bench(
        script="bench_render.py",
        args=["--steps", "5"],
        output_path=out,
    )
    assert sidecar["name"] == "bench_render"
    assert sidecar["steps"] == 5
    skipped = sidecar.get("skipped")
    assert skipped in (True, False), f"skipped must be bool, got {skipped!r}"
    if skipped is False:
        for key in (
            "frames_per_sec",
            "render_step_mean_ms",
            "render_step_p50_ms",
            "render_step_p95_ms",
            "render_step_p99_ms",
        ):
            assert key in sidecar, f"missing {key!r} in render sidecar"
            assert isinstance(sidecar[key], (int, float))
    else:
        skip_reason = sidecar.get("skip_reason")
        assert isinstance(skip_reason, str) and skip_reason, (
            f"skipped sidecar must populate skip_reason; got {skip_reason!r}"
        )
    _assert_provenance_fields(sidecar)


def test_bench_suite_loader_smoke(tmp_path: Path) -> None:
    """``scripts/perf/bench_suite_loader.py --cells 10`` produces a valid sidecar."""
    out = tmp_path / "suite_loader.json"
    sidecar = _run_bench(
        script="bench_suite_loader.py",
        args=["--cells", "10", "--repetitions", "3"],
        output_path=out,
    )
    assert sidecar["name"] == "bench_suite_loader"
    assert sidecar["cells_requested"] == 10
    assert sidecar["cells"] == 10
    assert sidecar["reps"] == 3
    for key in ("load_time_ms", "ast_hash_time_ms"):
        assert key in sidecar, f"missing {key!r} in loader sidecar"
        value = sidecar[key]
        assert isinstance(value, (int, float)), f"{key} must be numeric"
        assert value >= 0.0, f"{key} must be non-negative"
    _assert_provenance_fields(sidecar)


def test_bench_runner_scaling_smoke(tmp_path: Path) -> None:
    """``scripts/perf/bench_runner_scaling.py --workers 1`` produces a valid sidecar."""
    out = tmp_path / "scaling.json"
    sidecar = _run_bench(
        script="bench_runner_scaling.py",
        args=["--workers", "1", "--episodes-per-cell", "1", "--max-steps", "5"],
        output_path=out,
    )
    assert sidecar["name"] == "bench_runner_scaling"
    walls = sidecar.get("walls_ms")
    assert isinstance(walls, dict), "walls_ms must be a dict"
    assert "1" in walls, "expected n_workers=1 entry"
    speedups = sidecar.get("speedup_vs_n1")
    assert isinstance(speedups, dict), "speedup_vs_n1 must be a dict"
    assert speedups.get("1") == 1.0, "n=1 baseline should report speedup 1.0"
    serial = sidecar.get("amdahl_serial_frac")
    parallel = sidecar.get("amdahl_parallel_frac")
    assert isinstance(serial, (int, float))
    assert isinstance(parallel, (int, float))
    assert 0.0 <= serial <= 1.0, f"amdahl_serial_frac out of [0,1]: {serial}"
    assert abs((serial + parallel) - 1.0) < 1e-6, (
        f"serial+parallel should sum to 1.0, got {serial + parallel}"
    )
    _assert_provenance_fields(sidecar)
