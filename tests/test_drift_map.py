"""Unit tests for B-29 cross-backend embodiment-transfer drift map.

Coverage:

* :func:`gauntlet.compare.compute_drift_map` — happy path with
  populated per-axis-value rows; identical-backend reports raise
  :class:`DriftMapError`; missing ``suite_env`` raises; mismatched
  ``suite_name`` raises.
* :func:`gauntlet.compare.top_axis_drifts` — top-N sorting orders by
  ``abs_delta`` descending then axis name then value.
* :class:`gauntlet.compare.DriftMap` — JSON round-trip is byte-stable
  across ``model_dump_json`` -> ``model_validate_json``.
* CLI ``gauntlet compare --drift-map ...`` — happy path on
  cross-backend episodes.json pair emits a populated drift_map.json
  and prints the top-5 table; rejects without
  ``--allow-cross-backend`` and without ``--drift-map-policy-label``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.compare import (
    AxisDrift,
    DriftMap,
    DriftMapError,
    compute_drift_map,
    top_axis_drifts,
)
from gauntlet.report import AxisBreakdown, Report

# ──────────────────────────────────────────────────────────────────────
# Report fixture helpers.
# ──────────────────────────────────────────────────────────────────────


def _axis(
    name: str,
    rates: dict[float, float],
    counts: dict[float, int] | None = None,
) -> AxisBreakdown:
    """Build a minimal :class:`AxisBreakdown` for drift-map tests."""
    counts = counts or dict.fromkeys(rates, 5)
    successes = {v: round(rate * counts[v]) for v, rate in rates.items()}
    return AxisBreakdown(
        name=name,
        rates=rates,
        counts=counts,
        successes=successes,
    )


def _report(
    *,
    suite_env: str,
    suite_name: str = "synthetic",
    per_axis: list[AxisBreakdown] | None = None,
    overall_success_rate: float = 0.5,
) -> Report:
    """Build a minimal :class:`Report` for drift-map tests."""
    per_axis = per_axis or []
    return Report(
        suite_name=suite_name,
        suite_env=suite_env,
        n_episodes=10,
        n_success=round(overall_success_rate * 10),
        per_axis=per_axis,
        per_cell=[],
        failure_clusters=[],
        heatmap_2d={},
        overall_success_rate=overall_success_rate,
        overall_failure_rate=1.0 - overall_success_rate,
        cluster_multiple=2.0,
    )


# ──────────────────────────────────────────────────────────────────────
# 1 — Validation contracts.
# ──────────────────────────────────────────────────────────────────────


def test_compute_drift_map_rejects_identical_backend() -> None:
    """Same-backend drift map is a category error — raise loudly."""
    a = _report(suite_env="tabletop", per_axis=[_axis("texture", {0.0: 0.8, 1.0: 0.6})])
    b = _report(suite_env="tabletop", per_axis=[_axis("texture", {0.0: 0.8, 1.0: 0.6})])
    with pytest.raises(DriftMapError, match="different backends"):
        compute_drift_map(a, b, policy_label="scripted", suite_hash="abc")


def test_compute_drift_map_rejects_missing_suite_env() -> None:
    """Pre-RFC-005 reports without env slug cannot anchor cross-backend identity."""
    a = _report(suite_env="tabletop", per_axis=[_axis("texture", {0.0: 0.8})])
    b = Report(
        suite_name="synthetic",
        suite_env=None,
        n_episodes=10,
        n_success=5,
        per_axis=[_axis("texture", {0.0: 0.7})],
        per_cell=[],
        failure_clusters=[],
        heatmap_2d={},
        overall_success_rate=0.5,
        overall_failure_rate=0.5,
        cluster_multiple=2.0,
    )
    with pytest.raises(DriftMapError, match="suite_env"):
        compute_drift_map(a, b, policy_label="scripted", suite_hash="abc")


def test_compute_drift_map_rejects_mismatched_suite_name() -> None:
    """Different suites cannot establish a meaningful drift map even cross-backend."""
    a = _report(suite_env="tabletop", suite_name="suite-a")
    b = _report(suite_env="tabletop-pybullet", suite_name="suite-b")
    with pytest.raises(DriftMapError, match="suite_name"):
        compute_drift_map(a, b, policy_label="scripted", suite_hash="abc")


# ──────────────────────────────────────────────────────────────────────
# 2 — Happy path: populated drift map.
# ──────────────────────────────────────────────────────────────────────


def test_compute_drift_map_populates_rows_with_correct_deltas() -> None:
    """Each shared (axis, value) gets a row with delta = b - a."""
    a = _report(
        suite_env="tabletop",
        per_axis=[
            _axis("texture", {0.0: 0.9, 1.0: 0.5, 2.0: 0.3}),
            _axis("lighting", {0.0: 0.8, 1.0: 0.4}),
        ],
    )
    b = _report(
        suite_env="tabletop-pybullet",
        per_axis=[
            _axis("texture", {0.0: 0.6, 1.0: 0.5, 2.0: 0.0}),
            _axis("lighting", {0.0: 0.7, 1.0: 0.4}),
        ],
    )
    dmap = compute_drift_map(a, b, policy_label="scripted", suite_hash="hash-xyz")

    assert dmap.backend_a == "tabletop"
    assert dmap.backend_b == "tabletop-pybullet"
    assert dmap.policy_label == "scripted"
    assert dmap.suite_hash == "hash-xyz"
    assert set(dmap.axes.keys()) == {"texture", "lighting"}

    texture_rows = {r.value: r for r in dmap.axes["texture"]}
    assert texture_rows[0.0].delta == pytest.approx(-0.3)
    assert texture_rows[0.0].abs_delta == pytest.approx(0.3)
    assert texture_rows[0.0].relative_delta == pytest.approx(-0.3 / 0.9)
    assert texture_rows[1.0].delta == pytest.approx(0.0)
    assert texture_rows[1.0].abs_delta == pytest.approx(0.0)
    assert texture_rows[2.0].delta == pytest.approx(-0.3)

    # total_drift is the mean abs_delta across every row in every axis:
    # texture: 0.3 + 0.0 + 0.3 = 0.6
    # lighting: 0.1 + 0.0 = 0.1
    # mean over 5 rows = 0.7 / 5 = 0.14
    assert dmap.total_drift == pytest.approx(0.14)

    # Per-axis rows are sorted by abs_delta descending.
    assert dmap.axes["texture"][0].abs_delta >= dmap.axes["texture"][-1].abs_delta


def test_compute_drift_map_handles_zero_baseline_relative_delta() -> None:
    """``relative_delta`` is None when ``rate_a == 0.0`` to avoid +inf in JSON."""
    a = _report(suite_env="tabletop", per_axis=[_axis("texture", {0.0: 0.0, 1.0: 0.5})])
    b = _report(
        suite_env="tabletop-pybullet",
        per_axis=[_axis("texture", {0.0: 0.4, 1.0: 0.5})],
    )
    dmap = compute_drift_map(a, b, policy_label="scripted", suite_hash="abc")
    rows = {r.value: r for r in dmap.axes["texture"]}
    assert rows[0.0].relative_delta is None
    assert rows[1.0].relative_delta == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────
# 3 — top_axis_drifts ranking.
# ──────────────────────────────────────────────────────────────────────


def test_top_axis_drifts_sorts_by_abs_delta_descending() -> None:
    """Top-N pulls the most-divergent rows across every axis."""
    a = _report(
        suite_env="tabletop",
        per_axis=[
            _axis("texture", {0.0: 0.9, 1.0: 0.5}),
            _axis("lighting", {0.0: 0.5, 1.0: 0.5, 2.0: 0.5}),
        ],
    )
    b = _report(
        suite_env="tabletop-pybullet",
        per_axis=[
            _axis("texture", {0.0: 0.1, 1.0: 0.45}),
            _axis("lighting", {0.0: 0.0, 1.0: 0.5, 2.0: 0.4}),
        ],
    )
    dmap = compute_drift_map(a, b, policy_label="scripted", suite_hash="abc")
    top = top_axis_drifts(dmap, limit=3)
    assert len(top) == 3
    # texture@0.0: |0.1 - 0.9| = 0.8 — top.
    assert top[0].axis == "texture"
    assert top[0].value == 0.0
    assert top[0].abs_delta == pytest.approx(0.8)
    # lighting@0.0: |0.0 - 0.5| = 0.5 — second.
    assert top[1].axis == "lighting"
    assert top[1].value == 0.0
    # lighting@2.0: |0.4 - 0.5| = 0.1 — third.
    assert top[2].axis == "lighting"
    assert top[2].value == 2.0


# ──────────────────────────────────────────────────────────────────────
# 4 — JSON round-trip.
# ──────────────────────────────────────────────────────────────────────


def test_drift_map_json_round_trip_is_byte_stable() -> None:
    """``DriftMap.model_dump_json -> model_validate_json`` returns equal."""
    a = _report(
        suite_env="tabletop",
        per_axis=[_axis("texture", {0.0: 0.8, 1.0: 0.0})],
    )
    b = _report(
        suite_env="tabletop-pybullet",
        per_axis=[_axis("texture", {0.0: 0.5, 1.0: 0.0})],
    )
    dmap = compute_drift_map(a, b, policy_label="scripted", suite_hash="abc")
    payload = dmap.model_dump_json()
    restored = DriftMap.model_validate_json(payload)
    assert restored == dmap
    # And the inner AxisDrift is frozen / equal-on-fields.
    assert isinstance(restored.axes["texture"][0], AxisDrift)
    assert restored.axes["texture"][0].relative_delta is None or isinstance(
        restored.axes["texture"][0].relative_delta, float
    )


# ──────────────────────────────────────────────────────────────────────
# 5 — CLI integration.
# ──────────────────────────────────────────────────────────────────────


def _write_episodes_json(
    path: Path,
    *,
    suite_name: str,
    suite_env: str,
    success_by_cell: list[bool],
    suite_hash: str = "shared-hash",
    master_seed: int = 42,
) -> None:
    """Hand-build an episodes.json with provenance + suite_env metadata.

    Each entry in ``success_by_cell`` becomes one episode in its own
    cell; the per-cell perturbation value is the cell index so the
    rebuilt report's per_axis carries deterministic float keys.
    """
    eps: list[dict[str, object]] = []
    for cell_idx, success in enumerate(success_by_cell):
        eps.append(
            {
                "suite_name": suite_name,
                "cell_index": cell_idx,
                "episode_index": 0,
                "seed": 1000 + cell_idx,
                "perturbation_config": {"texture": float(cell_idx)},
                "success": success,
                "terminated": success,
                "truncated": not success,
                "step_count": 5,
                "total_reward": 0.0,
                "metadata": {
                    "master_seed": master_seed,
                    "suite_env": suite_env,
                    "n_cells": len(success_by_cell),
                    "episodes_per_cell": 1,
                },
                "suite_hash": suite_hash,
            }
        )
    path.write_text(json.dumps(eps), encoding="utf-8")


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner()


def test_cli_compare_drift_map_happy_path(cli_runner: CliRunner, tmp_path: Path) -> None:
    """End-to-end: cross-backend episodes.json pair → drift_map.json on disk."""
    a_path = tmp_path / "a_episodes.json"
    b_path = tmp_path / "b_episodes.json"
    # Two cells; cell 0 succeeds in MuJoCo, fails in PyBullet — drift
    # axis-value (0.0) magnitude = 1.0. Cell 1 agrees.
    _write_episodes_json(
        a_path,
        suite_name="synthetic",
        suite_env="tabletop",
        success_by_cell=[True, False],
    )
    _write_episodes_json(
        b_path,
        suite_name="synthetic",
        suite_env="tabletop-pybullet",
        success_by_cell=[False, False],
    )
    out_path = tmp_path / "compare.json"
    drift_path = tmp_path / "drift_map.json"
    result = cli_runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--out",
            str(out_path),
            "--allow-cross-backend",
            "--drift-map",
            str(drift_path),
            "--drift-map-policy-label",
            "scripted",
            "--min-cell-size",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert drift_path.exists()
    payload = json.loads(drift_path.read_text(encoding="utf-8"))
    assert payload["backend_a"] == "tabletop"
    assert payload["backend_b"] == "tabletop-pybullet"
    assert payload["policy_label"] == "scripted"
    assert payload["suite_hash"] == "shared-hash"
    assert "texture" in payload["axes"]
    # Top row should be cell 0 (1.0 -> 0.0) with abs_delta = 1.0.
    top = payload["axes"]["texture"][0]
    assert top["value"] == 0.0
    assert top["abs_delta"] == pytest.approx(1.0)


def test_cli_compare_drift_map_requires_allow_cross_backend(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """``--drift-map`` without ``--allow-cross-backend`` is a hard error."""
    a_path = tmp_path / "a_episodes.json"
    b_path = tmp_path / "b_episodes.json"
    _write_episodes_json(
        a_path,
        suite_name="synthetic",
        suite_env="tabletop",
        success_by_cell=[True],
    )
    _write_episodes_json(
        b_path,
        suite_name="synthetic",
        suite_env="tabletop-pybullet",
        success_by_cell=[False],
    )
    drift_path = tmp_path / "drift_map.json"
    result = cli_runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--drift-map",
            str(drift_path),
            "--drift-map-policy-label",
            "scripted",
            "--min-cell-size",
            "1",
        ],
    )
    # The cross-backend gate fires first (before --drift-map is even
    # checked); both error paths are equally valid for this test —
    # what matters is the CLI rejects the combo and does NOT write
    # drift_map.json.
    assert result.exit_code != 0
    assert not drift_path.exists()


def test_cli_compare_drift_map_requires_policy_label(cli_runner: CliRunner, tmp_path: Path) -> None:
    """``--drift-map`` without ``--drift-map-policy-label`` is a hard error."""
    a_path = tmp_path / "a_episodes.json"
    b_path = tmp_path / "b_episodes.json"
    _write_episodes_json(
        a_path,
        suite_name="synthetic",
        suite_env="tabletop",
        success_by_cell=[True],
    )
    _write_episodes_json(
        b_path,
        suite_name="synthetic",
        suite_env="tabletop-pybullet",
        success_by_cell=[False],
    )
    drift_path = tmp_path / "drift_map.json"
    result = cli_runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--allow-cross-backend",
            "--drift-map",
            str(drift_path),
            "--min-cell-size",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "drift-map-policy-label" in result.output
    assert not drift_path.exists()
