"""Tests for B-19 per-axis Sobol sensitivity indices.

Pure-math cases hit :func:`compute_sobol_indices` directly; the
integration cases drive :func:`build_report` so the schema tagging
(``approximate=True`` on cartesian, ``False`` on Sobol) is exercised
end-to-end. The JSON round-trip case asserts that an old report.json
written before B-19 (i.e. without the ``sensitivity_indices`` key)
still validates — the field default is ``None`` for that reason.
"""

from __future__ import annotations

import math

from gauntlet.report import Report, SensitivityIndex, build_report
from gauntlet.report.sobol_indices import compute_sobol_indices
from gauntlet.runner.episode import Episode

_SUITE = "test-suite-b19"


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
) -> Episode:
    return Episode(
        suite_name=_SUITE,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=0,
        perturbation_config=dict(config),
        success=success,
        terminated=success,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
    )


def _grid_episodes(
    *,
    success_fn: object,  # callable[[float, float], bool]
    a_values: list[float],
    b_values: list[float],
    reps: int = 1,
) -> list[Episode]:
    """Build a fully populated 2D ``axis_a`` by ``axis_b`` grid."""
    eps: list[Episode] = []
    cell = 0
    for a in a_values:
        for b in b_values:
            for r in range(reps):
                eps.append(
                    _ep(
                        cell_index=cell,
                        episode_index=r,
                        success=bool(success_fn(a, b)),  # type: ignore[operator]
                        config={"axis_a": a, "axis_b": b},
                    )
                )
            cell += 1
    return eps


# 1. Single-value axis is pinned at (0.0, 0.0) regardless of outcome variance.
def test_single_value_axis_indices_are_zero() -> None:
    # axis_a varies (0/1 outcome split), axis_b is pinned at 0.0.
    eps: list[Episode] = []
    eps.append(
        _ep(cell_index=0, episode_index=0, success=True, config={"axis_a": 0.0, "axis_b": 0.0})
    )
    eps.append(
        _ep(cell_index=0, episode_index=1, success=True, config={"axis_a": 0.0, "axis_b": 0.0})
    )
    eps.append(
        _ep(cell_index=1, episode_index=0, success=False, config={"axis_a": 1.0, "axis_b": 0.0})
    )
    eps.append(
        _ep(cell_index=1, episode_index=1, success=False, config={"axis_a": 1.0, "axis_b": 0.0})
    )
    out = compute_sobol_indices(eps, ("axis_a", "axis_b"))
    assert out["axis_b"] == (0.0, 0.0)
    # axis_a fully explains the outcome → first-order ≈ 1.0.
    s_a, st_a = out["axis_a"]
    assert s_a is not None and st_a is not None
    assert math.isclose(s_a, 1.0, abs_tol=1e-9)
    assert math.isclose(st_a, 1.0, abs_tol=1e-9)


# 2. All-success run has Var(Y) == 0; both indices are None per spec.
def test_all_success_run_returns_none_indices() -> None:
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"axis_a": 0.0, "axis_b": 0.0}),
        _ep(cell_index=1, episode_index=0, success=True, config={"axis_a": 1.0, "axis_b": 0.0}),
        _ep(cell_index=2, episode_index=0, success=True, config={"axis_a": 0.0, "axis_b": 1.0}),
        _ep(cell_index=3, episode_index=0, success=True, config={"axis_a": 1.0, "axis_b": 1.0}),
    ]
    out = compute_sobol_indices(eps, ("axis_a", "axis_b"))
    assert out["axis_a"] == (None, None)
    assert out["axis_b"] == (None, None)


# 3. Normal additive case: outcome perfectly determined by axis_a;
#    axis_b is irrelevant → S_a ≈ 1, S_b ≈ 0 (no interaction either).
def test_normal_case_axis_a_dominates() -> None:
    eps = _grid_episodes(
        success_fn=lambda a, _b: a >= 0.5,
        a_values=[0.0, 1.0],
        b_values=[0.0, 0.5, 1.0],
        reps=2,
    )
    out = compute_sobol_indices(eps, ("axis_a", "axis_b"))
    s_a, st_a = out["axis_a"]
    s_b, st_b = out["axis_b"]
    assert s_a is not None and st_a is not None
    assert s_b is not None and st_b is not None
    assert math.isclose(s_a, 1.0, abs_tol=1e-9)
    assert math.isclose(st_a, 1.0, abs_tol=1e-9)
    assert math.isclose(s_b, 0.0, abs_tol=1e-9)
    assert math.isclose(st_b, 0.0, abs_tol=1e-9)


# 4. Pure XOR-style interaction: marginals are flat (S_i ≈ 0) but the
#    total-order indices are 1.0 — the variance lives entirely in the
#    interaction term.
def test_xor_interaction_lifts_total_order() -> None:
    eps = _grid_episodes(
        success_fn=lambda a, b: (a > 0.5) ^ (b > 0.5),
        a_values=[0.0, 1.0],
        b_values=[0.0, 1.0],
        reps=3,
    )
    out = compute_sobol_indices(eps, ("axis_a", "axis_b"))
    s_a, st_a = out["axis_a"]
    s_b, st_b = out["axis_b"]
    assert s_a is not None and st_a is not None
    assert s_b is not None and st_b is not None
    # Marginals carry no signal under perfect XOR.
    assert math.isclose(s_a, 0.0, abs_tol=1e-9)
    assert math.isclose(s_b, 0.0, abs_tol=1e-9)
    # Total-order picks up the interaction.
    assert math.isclose(st_a, 1.0, abs_tol=1e-9)
    assert math.isclose(st_b, 1.0, abs_tol=1e-9)


# 5. build_report integration: indices populate the Report and are
#    keyed by axis name.
def test_build_report_populates_sensitivity_indices() -> None:
    eps = _grid_episodes(
        success_fn=lambda a, _b: a >= 0.5,
        a_values=[0.0, 1.0],
        b_values=[0.0, 1.0],
        reps=2,
    )
    report = build_report(eps)
    assert report.sensitivity_indices is not None
    assert set(report.sensitivity_indices.keys()) == {"axis_a", "axis_b"}
    a_idx = report.sensitivity_indices["axis_a"]
    assert isinstance(a_idx, SensitivityIndex)
    assert a_idx.first_order is not None
    assert a_idx.total_order is not None


# 6. ``approximate`` flag flips with the suite sampling mode.
def test_approximate_flag_on_cartesian_vs_sobol() -> None:
    eps = _grid_episodes(
        success_fn=lambda a, _b: a >= 0.5,
        a_values=[0.0, 1.0],
        b_values=[0.0, 1.0],
        reps=2,
    )
    cartesian = build_report(eps, sampling="cartesian")
    sobol = build_report(eps, sampling="sobol")
    none_default = build_report(eps)
    assert cartesian.sensitivity_indices is not None
    assert sobol.sensitivity_indices is not None
    assert none_default.sensitivity_indices is not None
    assert cartesian.sensitivity_indices["axis_a"].approximate is True
    assert sobol.sensitivity_indices["axis_a"].approximate is False
    # Default (no sampling kwarg) is approximate — old call-sites that
    # haven't opted in remain conservative.
    assert none_default.sensitivity_indices["axis_a"].approximate is True


# 7. JSON round-trip without the new field still loads — backwards-compat
#    contract for report.json files written before B-19.
def test_old_report_json_without_sensitivity_indices_still_validates() -> None:
    eps = _grid_episodes(
        success_fn=lambda a, _b: a >= 0.5,
        a_values=[0.0, 1.0],
        b_values=[0.0, 1.0],
        reps=2,
    )
    report = build_report(eps, sampling="sobol")
    dumped = report.model_dump(mode="json")
    # Strip the new field so the dict matches the pre-B-19 shape.
    dumped.pop("sensitivity_indices", None)
    restored = Report.model_validate(dumped)
    assert restored.sensitivity_indices is None
    # And a fresh dump → validate cycle preserves the populated field.
    again = Report.model_validate(report.model_dump(mode="json"))
    assert again.sensitivity_indices is not None
    assert again.sensitivity_indices["axis_a"].approximate is False
