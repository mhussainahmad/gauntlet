"""B-26 Optuna-style early-stop pruning tests.

Covers the Runner-side knobs (``prune_after_cells`` / ``prune_min_success``),
the Wilson-CI prune predicate, the ``sampling: sobol`` force-disable
warning, the "both flags must be supplied together" error, and the
``Report.pruned_at_cell`` schema field.

The tests use a deterministic fake env keyed off a perturbation value
(success = ``distractor_count`` axis value above a per-test threshold)
so each scenario is bit-stable and the Wilson CI bounds are knowable
ahead of time. The pattern mirrors :class:`tests.test_runner._FakeProtocolEnv`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.policy.scripted import ScriptedPolicy
from gauntlet.report.schema import Report
from gauntlet.runner import Runner
from gauntlet.suite.schema import AxisSpec, Suite

# ----------------------------------------------------------------------
# Fake env: success is keyed off the ``distractor_count`` perturbation
# so each test can build a suite that is "clearly passing" or "clearly
# failing" deterministically. The :data:`_PASS_THRESHOLD` global is the
# tuning knob — values strictly above it succeed.
# ----------------------------------------------------------------------


_PASS_THRESHOLD: float = 0.5


class _ThresholdEnv:
    """Fake :class:`gauntlet.env.base.GauntletEnv` keyed off a perturbation.

    Succeeds iff the most-recently-applied ``distractor_count`` axis
    value is strictly greater than :data:`_PASS_THRESHOLD`. Pure data
    + a step counter; no MuJoCo, no GL, no torch.
    """

    AXIS_NAMES = frozenset({"distractor_count"})
    VISUAL_ONLY_AXES: frozenset[str] = frozenset()

    def __init__(self) -> None:
        from gymnasium import spaces

        self.observation_space = spaces.Dict(
            {"cube_pos": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64)}
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        self._pending: dict[str, float] = {}
        self._last_value: float = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        # Latch the perturbation that was most recently applied; the
        # axis value drives the success flag returned by ``step``.
        self._last_value = float(self._pending.get("distractor_count", 0.0))
        return (
            {"cube_pos": np.zeros(3, dtype=np.float64)},
            {},
        )

    def step(
        self,
        action: np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        success = self._last_value > _PASS_THRESHOLD
        return (
            {"cube_pos": np.zeros(3, dtype=np.float64)},
            1.0 if success else 0.0,
            True,  # always terminate on step 1 — fast tests
            False,
            {"success": success},
        )

    def set_perturbation(self, name: str, value: float) -> None:
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self._pending[name] = value

    def restore_baseline(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        return None


def _threshold_env_factory() -> Any:
    return _ThresholdEnv()


def _make_scripted() -> ScriptedPolicy:
    return ScriptedPolicy()


def _make_suite(
    *,
    values: list[float],
    episodes_per_cell: int = 4,
    sampling: str = "cartesian",
) -> Suite:
    """Build a 1-axis suite with one cell per supplied value."""
    return Suite(
        name="prune-test",
        env="tabletop",  # registry name; the fake env_factory overrides dispatch
        seed=1234,
        episodes_per_cell=episodes_per_cell,
        axes={"distractor_count": AxisSpec(values=values)},
        sampling=sampling,  # type: ignore[arg-type]
    )


# ----------------------------------------------------------------------
# 1 — pruning kicks in when the CI clearly PASSES the bar.
# ----------------------------------------------------------------------


def test_prune_when_ci_clearly_passes() -> None:
    """20 all-success cells with threshold 0.5 -> prune within the first chunk."""
    # Every cell value (1.0, 1.0, ...) is above _PASS_THRESHOLD = 0.5,
    # so every episode succeeds. Wilson 95% CI for k=n=8 is
    # roughly [0.67, 1.0] which is strictly above 0.5 — the very first
    # post-chunk check trips the pass branch.
    suite = _make_suite(values=[1.0] * 20, episodes_per_cell=2)
    runner = Runner(n_workers=1, env_factory=_threshold_env_factory)
    episodes = runner.run(
        policy_factory=_make_scripted,
        suite=suite,
        prune_after_cells=4,
        prune_min_success=0.5,
    )
    assert runner.last_pruned_at_cell is not None, "expected a pass-prune"
    # First chunk = 4 cells * 2 eps = 8 episodes; pruning fires after.
    assert runner.last_pruned_at_cell == 4
    assert len(episodes) == 8
    # And every observed episode was indeed a success.
    assert all(ep.success for ep in episodes)


# ----------------------------------------------------------------------
# 2 — pruning kicks in when the CI clearly FAILS the bar.
# ----------------------------------------------------------------------


def test_prune_when_ci_clearly_fails() -> None:
    """20 all-fail cells with threshold 0.5 -> prune within the first chunk."""
    # Values strictly <= _PASS_THRESHOLD (0.5) so no episode succeeds.
    # Wilson 95% CI for k=0/n=8 is roughly [0.0, 0.32] — strictly
    # below 0.5, the fail branch fires after the first chunk.
    suite = _make_suite(values=[0.0] * 20, episodes_per_cell=2)
    runner = Runner(n_workers=1, env_factory=_threshold_env_factory)
    episodes = runner.run(
        policy_factory=_make_scripted,
        suite=suite,
        prune_after_cells=4,
        prune_min_success=0.5,
    )
    assert runner.last_pruned_at_cell == 4
    assert len(episodes) == 8
    assert not any(ep.success for ep in episodes)


# ----------------------------------------------------------------------
# 3 — no prune when the CI brackets the threshold.
# ----------------------------------------------------------------------


def test_no_prune_when_ci_straddles_threshold() -> None:
    """4 cells alternating success/fail; CI for ~50% straddles 0.5 -> full run."""
    # Values: [1.0, 0.0, 1.0, 0.0] -> alternating success/fail; running
    # success rate is exactly 0.5 across the run, and the Wilson 95%
    # CI for k=4 / n=8 (or k=2 / n=4) brackets 0.5 generously, so the
    # pruner never fires and every cell completes.
    suite = _make_suite(values=[1.0, 0.0, 1.0, 0.0], episodes_per_cell=2)
    runner = Runner(n_workers=1, env_factory=_threshold_env_factory)
    episodes = runner.run(
        policy_factory=_make_scripted,
        suite=suite,
        prune_after_cells=2,
        prune_min_success=0.5,
    )
    assert runner.last_pruned_at_cell is None
    # Full run: 4 cells * 2 eps = 8 episodes.
    assert len(episodes) == 8


# ----------------------------------------------------------------------
# 4 — sampling: sobol force-disables pruning, with a UserWarning.
# ----------------------------------------------------------------------


def test_sobol_sampling_force_disables_pruning() -> None:
    """B-19 needs the full sample structure -> warn + run all cells."""
    # Tiny Sobol suite: even though every cell would prune (all-pass),
    # the Runner must complete the full sweep and emit a warning.
    suite = Suite(
        name="prune-sobol",
        env="tabletop",
        seed=42,
        episodes_per_cell=2,
        axes={"distractor_count": AxisSpec(low=0.6, high=1.0)},
        sampling="sobol",
        n_samples=8,
    )
    runner = Runner(n_workers=1, env_factory=_threshold_env_factory)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        episodes = runner.run(
            policy_factory=_make_scripted,
            suite=suite,
            prune_after_cells=1,
            prune_min_success=0.5,
        )
    # Warning surfaced and the run completed every cell.
    assert any(
        issubclass(w.category, UserWarning) and "sobol" in str(w.message).lower() for w in caught
    ), [str(w.message) for w in caught]
    assert runner.last_pruned_at_cell is None
    assert len(episodes) == 8 * 2  # n_samples * episodes_per_cell


# ----------------------------------------------------------------------
# 5 — both flags must be supplied together (XOR error).
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("after_cells", "min_success"),
    [(5, None), (None, 0.5)],
    ids=["only-after-cells", "only-min-success"],
)
def test_prune_flags_must_travel_together(
    after_cells: int | None,
    min_success: float | None,
) -> None:
    suite = _make_suite(values=[0.0, 1.0], episodes_per_cell=1)
    runner = Runner(n_workers=1, env_factory=_threshold_env_factory)
    with pytest.raises(ValueError, match="must be supplied together"):
        runner.run(
            policy_factory=_make_scripted,
            suite=suite,
            prune_after_cells=after_cells,
            prune_min_success=min_success,
        )


# ----------------------------------------------------------------------
# 6 — Report.pruned_at_cell schema field round-trips through CLI.
# ----------------------------------------------------------------------


def test_cli_run_stamps_pruned_at_cell_on_report(tmp_path: Path) -> None:
    """End-to-end: gauntlet run with prune flags -> report.json carries the field."""
    cli_runner = CliRunner()
    suite_yaml = (
        "name: prune-cli-suite\n"
        "env: tabletop\n"
        "episodes_per_cell: 2\n"
        "seed: 7\n"
        "axes:\n"
        "  lighting_intensity:\n"
        "    values: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]\n"
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(suite_yaml, encoding="utf-8")
    out_dir = tmp_path / "out"

    # The scripted policy on the real tabletop env with max_steps=5 is
    # a near-guaranteed failure (the trajectory needs more steps to
    # complete a pick-and-place). With ``--prune-min-success 0.95`` the
    # Wilson 95% CI upper bound for a clearly-failing run drops below
    # 0.95 inside the first chunk, tripping the fail-prune branch and
    # populating the schema field on report.json.
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "--policy",
            "scripted",
            "--out",
            str(out_dir),
            "--env-max-steps",
            "5",
            "--no-html",
            "--prune-after-cells",
            "2",
            "--prune-min-success",
            "0.95",
        ],
    )
    assert result.exit_code == 0, result.stderr
    import json

    report_payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert "pruned_at_cell" in report_payload, report_payload.keys()
    # Pruner fires inside the first chunk for a clearly-failing run;
    # the field carries the cell count at which the run aborted.
    assert isinstance(report_payload["pruned_at_cell"], int)
    assert report_payload["pruned_at_cell"] < 10


# ----------------------------------------------------------------------
# 7 — CLI rejects single-flag invocations cleanly.
# ----------------------------------------------------------------------


def test_cli_rejects_only_one_prune_flag(tmp_path: Path) -> None:
    cli_runner = CliRunner()
    suite_yaml = (
        "name: prune-cli-half\n"
        "env: tabletop\n"
        "episodes_per_cell: 1\n"
        "seed: 7\n"
        "axes:\n"
        "  lighting_intensity:\n"
        "    values: [0.5]\n"
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(suite_yaml, encoding="utf-8")
    out_dir = tmp_path / "out"

    result = cli_runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "--policy",
            "scripted",
            "--out",
            str(out_dir),
            "--env-max-steps",
            "5",
            "--no-html",
            "--prune-after-cells",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "must be supplied together" in result.stderr


# ----------------------------------------------------------------------
# 8 — Report schema accepts pruned_at_cell as None and as an int.
# ----------------------------------------------------------------------


def test_report_schema_pruned_at_cell_field() -> None:
    """Direct schema check: backwards-compat (omitted) and explicit int both validate."""
    base_payload = {
        "suite_name": "x",
        "n_episodes": 0,
        "n_success": 0,
        "per_axis": [],
        "per_cell": [],
        "failure_clusters": [],
        "heatmap_2d": {},
        "overall_success_rate": 0.0,
        "overall_failure_rate": 0.0,
        "cluster_multiple": 2.0,
    }
    # Omitted field -> None default (backwards-compat with pre-B-26 JSON).
    rep_default = Report.model_validate(base_payload)
    assert rep_default.pruned_at_cell is None
    # Explicit int -> stored verbatim.
    rep_pruned = Report.model_validate({**base_payload, "pruned_at_cell": 4})
    assert rep_pruned.pruned_at_cell == 4
    # JSON round-trip preserves the field.
    round_tripped = Report.model_validate_json(rep_pruned.model_dump_json())
    assert round_tripped.pruned_at_cell == 4
