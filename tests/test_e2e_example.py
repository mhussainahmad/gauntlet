"""End-to-end test for ``examples/evaluate_random_policy.py``.

Exercises the example script the README quickstart references — the
"smoke" suite (``examples/suites/tabletop-smoke.yaml``) is run through
the full pipeline and the resulting artefacts are inspected. This is the
single integration test for Phase 1 Task 10.

The script is loaded via :func:`runpy.run_path` so we don't need to make
``examples/`` an importable package (it is intentionally not on the mypy
``files`` list and not in ``[tool.pytest.ini_options].testpaths``). The
returned namespace's ``main`` is cast to a typed callable to keep
``mypy --strict`` happy.
"""

from __future__ import annotations

import json
import runpy
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from gauntlet.runner.episode import Episode
from gauntlet.suite import load_suite

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_EXAMPLE_SCRIPT: Path = _REPO_ROOT / "examples" / "evaluate_random_policy.py"
_SMOKE_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"

# Soft wallclock cap. The smoke suite is sized for ~15s on a laptop with
# n_workers=2; 30s leaves headroom for slower CI shapes without making
# the test useless as a perf canary.
_MAX_RUNTIME_SECONDS: float = 30.0

# Type alias for the script entry point.
_MainFn = Callable[..., None]


@pytest.fixture(scope="module")
def example_main() -> _MainFn:
    """Load ``examples/evaluate_random_policy.py`` and return its ``main``.

    ``runpy.run_path`` executes the module in a fresh namespace; we pull
    ``main`` out and cast it to a typed callable so the rest of the test
    body type-checks under ``mypy --strict``.
    """
    namespace: dict[str, Any] = runpy.run_path(str(_EXAMPLE_SCRIPT))
    main_obj = namespace.get("main")
    assert main_obj is not None, "examples/evaluate_random_policy.py must export main()"
    assert callable(main_obj), "exported main must be callable"
    return cast(_MainFn, main_obj)


def test_e2e_example_writes_all_artefacts(
    example_main: _MainFn,
    tmp_path: Path,
) -> None:
    """Run the example script and assert the three artefacts land + parse."""
    out_dir = tmp_path / "out"

    start = time.monotonic()
    example_main(
        suite_path=_SMOKE_SUITE,
        out_dir=out_dir,
        n_workers=1,  # in-process fast path; matches CLI's default for the test gate
    )
    elapsed = time.monotonic() - start

    # ── Wallclock budget. Soft cap: meaningful as a perf canary, not a
    #    correctness assertion. If it ever flakes on CI, raise the cap
    #    before shrinking the suite.
    assert elapsed < _MAX_RUNTIME_SECONDS, (
        f"e2e example exceeded soft wallclock budget: {elapsed:.2f}s > {_MAX_RUNTIME_SECONDS}s"
    )

    # ── episodes.json: list of Episode-shaped dicts, length matches the
    #    suite's grid * episodes_per_cell.
    episodes_path = out_dir / "episodes.json"
    assert episodes_path.is_file(), f"missing artefact: {episodes_path}"
    raw_episodes: Any = json.loads(episodes_path.read_text(encoding="utf-8"))
    assert isinstance(raw_episodes, list), "episodes.json must be a JSON list"
    assert len(raw_episodes) > 0, "episodes.json must be non-empty"
    expected_keys = {
        "suite_name",
        "cell_index",
        "episode_index",
        "seed",
        "perturbation_config",
        "success",
        "terminated",
        "truncated",
        "step_count",
        "total_reward",
        "metadata",
    }
    for i, ep_raw in enumerate(raw_episodes):
        assert isinstance(ep_raw, dict), f"episode {i} not a dict"
        assert expected_keys.issubset(ep_raw.keys()), (
            f"episode {i} missing keys: {expected_keys - ep_raw.keys()}"
        )
        # Validates against the strict pydantic schema (extra="forbid"),
        # so it also catches any silent field-name drift.
        Episode.model_validate(ep_raw)

    # Episode count matches the suite shape exactly.
    suite = load_suite(_SMOKE_SUITE)
    expected_episodes = suite.num_cells() * suite.episodes_per_cell
    assert len(raw_episodes) == expected_episodes, (
        f"expected {expected_episodes} episodes ({suite.num_cells()} cells x "
        f"{suite.episodes_per_cell}/cell); got {len(raw_episodes)}"
    )

    # ── report.json: dict with the spec-mandated breakdown keys.
    report_path = out_dir / "report.json"
    assert report_path.is_file(), f"missing artefact: {report_path}"
    raw_report: Any = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(raw_report, dict), "report.json must be a JSON object"
    for key in ("overall_success_rate", "per_axis", "per_cell"):
        assert key in raw_report, f"report.json missing required key: {key!r}"
    assert isinstance(raw_report["overall_success_rate"], (int, float))
    assert isinstance(raw_report["per_axis"], list)
    assert isinstance(raw_report["per_cell"], list)
    # Axis breakdown count must match the YAML axis count (the spec
    # guarantees one AxisBreakdown per declared axis — drift here would
    # mean the Report builder silently dropped or duplicated an axis).
    assert len(raw_report["per_axis"]) == len(suite.axes)

    # ── report.html: a real HTML document carrying the suite name.
    html_path = out_dir / "report.html"
    assert html_path.is_file(), f"missing artefact: {html_path}"
    html = html_path.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>"), (
        f"report.html does not start with <!DOCTYPE html>; got: {html[:64]!r}"
    )
    assert suite.name in html, f"report.html does not mention suite name {suite.name!r}"
