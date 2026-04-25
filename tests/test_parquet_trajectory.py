"""B-23 — Parquet trajectory dump tests.

Marked ``@pytest.mark.parquet`` — runs in the dedicated parquet-tests
job (which installs the ``[parquet]`` extra). On the default torch-
/extras-free job these tests are deselected by marker. The
``pytest.importorskip("pyarrow")`` at module scope is a belt-and-
braces guard so an accidental no-marker run on a pyarrow-less
interpreter still skips cleanly instead of erroring at import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from gauntlet.runner import Runner  # noqa: E402
from gauntlet.runner.parquet import (  # noqa: E402
    TrajectoryDict,
    parquet_path_for,
    write_parquet,
)
from gauntlet.suite.schema import AxisSpec, Suite  # noqa: E402

pytestmark = pytest.mark.parquet


# ──────────────────────────────────────────────────────────────────────
# Fixtures / factories.
# ──────────────────────────────────────────────────────────────────────


def _make_trajectory(n_steps: int = 5, action_dim: int = 7) -> TrajectoryDict:
    """Build a known-shape trajectory with deterministic values."""
    rng = np.random.default_rng(0)
    return TrajectoryDict(
        observations={
            "qpos": rng.normal(size=(n_steps, 4)).astype(np.float64),
            "qvel": rng.normal(size=(n_steps, 4)).astype(np.float64),
            # Higher-rank obs (image-shaped) — should be skipped by the
            # writer rather than blowing up the parquet schema.
            "image": np.zeros((n_steps, 8, 8, 3), dtype=np.float64),
        },
        actions=rng.normal(size=(n_steps, action_dim)).astype(np.float64),
        rewards=np.linspace(-1.0, 1.0, n_steps, dtype=np.float64),
        terminated=np.array([False] * (n_steps - 1) + [True], dtype=np.bool_),
        truncated=np.zeros(n_steps, dtype=np.bool_),
    )


def _make_fast_env() -> Any:
    from gauntlet.env.tabletop import TabletopEnv

    return TabletopEnv(max_steps=4)


def _make_scripted_policy() -> Any:
    from gauntlet.policy.scripted import ScriptedPolicy

    return ScriptedPolicy()


def _make_suite() -> Suite:
    return Suite(
        name="b23-parquet-test",
        env="tabletop",
        seed=1234,
        episodes_per_cell=1,
        axes={"lighting_intensity": AxisSpec(low=0.5, high=1.0, steps=2)},
    )


# ──────────────────────────────────────────────────────────────────────
# 1. write_parquet — writer round-trips a known trajectory.
# ──────────────────────────────────────────────────────────────────────


def test_write_parquet_round_trips_known_trajectory(tmp_path: Path) -> None:
    """Writer + pyarrow reader recover every per-step value."""
    trajectory = _make_trajectory(n_steps=5)
    out = tmp_path / "ep.parquet"
    written = write_parquet(out, trajectory)
    assert written == out
    assert out.exists()

    table = pq.read_table(out)
    df = table.to_pandas()

    # Step column is 0..T-1.
    assert df["step"].tolist() == [0, 1, 2, 3, 4]

    # Per-step rewards round-trip exactly.
    np.testing.assert_array_equal(df["reward"].to_numpy(), trajectory.rewards)
    np.testing.assert_array_equal(df["terminated"].to_numpy(), trajectory.terminated)
    np.testing.assert_array_equal(df["truncated"].to_numpy(), trajectory.truncated)

    # Action columns match.
    for i in range(7):
        np.testing.assert_array_equal(
            df[f"action_{i}"].to_numpy(),
            trajectory.actions[:, i],
        )


# ──────────────────────────────────────────────────────────────────────
# 2. Schema — declared columns match expectation.
# ──────────────────────────────────────────────────────────────────────


def test_write_parquet_columns_match_expected_schema(tmp_path: Path) -> None:
    """Column set is the documented schema. Image obs is skipped."""
    trajectory = _make_trajectory(n_steps=3)
    out = tmp_path / "schema.parquet"
    write_parquet(out, trajectory)

    table = pq.read_table(out)
    columns = set(table.column_names)

    expected = {
        "step",
        "reward",
        "terminated",
        "truncated",
        # 4-dim qpos -> 4 cols.
        "observation.qpos.0",
        "observation.qpos.1",
        "observation.qpos.2",
        "observation.qpos.3",
        # 4-dim qvel -> 4 cols.
        "observation.qvel.0",
        "observation.qvel.1",
        "observation.qvel.2",
        "observation.qvel.3",
        # 7-DoF action -> 7 cols.
        *(f"action_{i}" for i in range(7)),
    }
    assert columns == expected, f"unexpected columns: {columns ^ expected}"

    # Image columns explicitly absent (skipped by design).
    assert not any("image" in c for c in columns)


# ──────────────────────────────────────────────────────────────────────
# 3. ImportError when [parquet] extra is missing (mocked).
# ──────────────────────────────────────────────────────────────────────


def test_write_parquet_raises_clean_import_error_when_extra_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing [parquet] extra surfaces a clean ImportError with install hint."""
    # Force the lazy import inside ``write_parquet`` to fail by hiding
    # ``pyarrow`` from the import system. We cannot just ``del
    # sys.modules["pyarrow"]`` — once it's been imported at module load
    # the cached entry persists and the lazy import succeeds. Instead
    # we install a meta-path finder that refuses pyarrow lookups.
    import importlib.abc
    import importlib.machinery

    class _RefusePyarrow(importlib.abc.MetaPathFinder):
        def find_spec(
            self,
            name: str,
            path: Any,
            target: Any = None,
        ) -> importlib.machinery.ModuleSpec | None:
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError(f"simulated missing pyarrow ({name})")
            return None

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    monkeypatch.setattr(sys, "meta_path", [_RefusePyarrow(), *sys.meta_path])

    with pytest.raises(ImportError, match=r"\[parquet\] extra"):
        write_parquet(tmp_path / "x.parquet", _make_trajectory(n_steps=1))


# ──────────────────────────────────────────────────────────────────────
# 4. Runner integration — "parquet" mode emits parquet only.
# ──────────────────────────────────────────────────────────────────────


def test_runner_parquet_mode_emits_parquet_only(tmp_path: Path) -> None:
    """``trajectory_format='parquet'`` writes parquet and skips NPZ."""
    suite = _make_suite()
    traj_dir = tmp_path / "traj"
    runner = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        trajectory_dir=traj_dir,
    )
    episodes = runner.run(
        policy_factory=_make_scripted_policy,
        suite=suite,
        trajectory_format="parquet",
    )
    assert len(episodes) == 2  # 2 cells x 1 episode
    parquets = sorted(traj_dir.glob("*.parquet"))
    npzs = sorted(traj_dir.glob("*.npz"))
    assert len(parquets) == 2
    assert npzs == []
    # Parquet files match the canonical naming.
    expected = {parquet_path_for(traj_dir, ep.cell_index, ep.episode_index) for ep in episodes}
    assert set(parquets) == expected


# ──────────────────────────────────────────────────────────────────────
# 5. Runner integration — "both" mode emits both sidecars.
# ──────────────────────────────────────────────────────────────────────


def test_runner_both_mode_emits_both_sidecars(tmp_path: Path) -> None:
    """``trajectory_format='both'`` writes one NPZ AND one Parquet per episode."""
    suite = _make_suite()
    traj_dir = tmp_path / "traj"
    runner = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        trajectory_dir=traj_dir,
    )
    episodes = runner.run(
        policy_factory=_make_scripted_policy,
        suite=suite,
        trajectory_format="both",
    )
    assert len(episodes) == 2
    parquets = sorted(traj_dir.glob("*.parquet"))
    npzs = sorted(traj_dir.glob("*.npz"))
    assert len(parquets) == 2
    assert len(npzs) == 2
    # Pair matches by basename (mod suffix).
    assert {p.stem for p in parquets} == {n.stem for n in npzs}


# ──────────────────────────────────────────────────────────────────────
# 6. Runner integration — default ("npz") leaves no parquet.
# ──────────────────────────────────────────────────────────────────────


def test_runner_default_mode_emits_npz_only(tmp_path: Path) -> None:
    """The default (``trajectory_format='npz'``) is byte-identical to pre-B-23."""
    suite = _make_suite()
    traj_dir = tmp_path / "traj"
    runner = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        trajectory_dir=traj_dir,
    )
    runner.run(policy_factory=_make_scripted_policy, suite=suite)
    npzs = sorted(traj_dir.glob("*.npz"))
    parquets = sorted(traj_dir.glob("*.parquet"))
    assert len(npzs) == 2
    assert parquets == []


# ──────────────────────────────────────────────────────────────────────
# 7. Parquet content from runner — DuckDB-friendly shape.
# ──────────────────────────────────────────────────────────────────────


def test_runner_parquet_files_are_queryable(tmp_path: Path) -> None:
    """Parquet files emitted by the Runner load and contain per-step data."""
    suite = _make_suite()
    traj_dir = tmp_path / "traj"
    runner = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        trajectory_dir=traj_dir,
    )
    episodes = runner.run(
        policy_factory=_make_scripted_policy,
        suite=suite,
        trajectory_format="parquet",
    )
    # Load every file via pyarrow and assert basic shape invariants.
    for ep in episodes:
        path = parquet_path_for(traj_dir, ep.cell_index, ep.episode_index)
        table = pq.read_table(path)
        names = set(table.column_names)
        assert {"step", "reward", "terminated", "truncated"}.issubset(names)
        # At least one action column exists.
        assert any(c.startswith("action_") for c in names)
        # Row count == ep.step_count (one row per env step).
        assert table.num_rows == ep.step_count
