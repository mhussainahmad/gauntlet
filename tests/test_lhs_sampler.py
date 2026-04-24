"""Latin Hypercube Sampler tests — see ``src/gauntlet/suite/lhs.py``.

Properties pinned:

* Determinism: same RNG seed → same list of ``SuiteCell`` rows.
* Stratification: every axis covers all ``n_samples`` strata exactly
  once (the McKay 1979 guarantee).
* Range: every continuous value lies in ``[low, high]``.
* Categorical handling: ``object_texture`` (values: [0, 1]) only emits
  the declared values; over many samples each is hit roughly equally.
* Domain win: 5 axes x ``n_samples = 64`` LHS gives strictly more
  marginal coverage than the 5x5x5x5x5 = 3125-cell cartesian grid
  while running 49x fewer rollouts.
"""

from __future__ import annotations

import numpy as np
import pytest

from gauntlet.suite import AxisSpec, Suite, load_suite_from_string
from gauntlet.suite.lhs import LatinHypercubeSampler, lhs_unit_cube

_LHS_FOUR_AXES = """
name: lhs-four-axes
env: tabletop
seed: 42
sampling: latin_hypercube
n_samples: 32
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.5
  camera_offset_x:
    low: -0.05
    high: 0.05
  camera_offset_y:
    low: -0.05
    high: 0.05
  object_initial_pose_x:
    low: -0.15
    high: 0.15
"""


_LHS_WITH_CATEGORICAL = """
name: lhs-with-categorical
env: tabletop
seed: 42
sampling: latin_hypercube
n_samples: 50
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.5
  object_texture:
    values: [0, 1]
"""


_LHS_WITH_INT = """
name: lhs-with-int
env: tabletop
seed: 7
sampling: latin_hypercube
n_samples: 24
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.5
  distractor_count:
    low: 0
    high: 5
"""


# ============================================================ unit-cube helper


class TestLhsUnitCube:
    def test_shape_is_correct(self) -> None:
        out = lhs_unit_cube(16, 3, np.random.default_rng(0))
        assert out.shape == (16, 3)

    def test_values_in_unit_interval(self) -> None:
        out = lhs_unit_cube(16, 3, np.random.default_rng(0))
        assert (out >= 0.0).all()
        assert (out < 1.0).all()

    def test_each_stratum_hit_exactly_once_per_axis(self) -> None:
        # The McKay guarantee: for each axis, the n samples land one in
        # each of the n strata [i/n, (i+1)/n). Bin every column and
        # assert the histogram is all ones.
        n = 25
        d = 4
        out = lhs_unit_cube(n, d, np.random.default_rng(123))
        bins = np.linspace(0.0, 1.0, n + 1)
        for axis in range(d):
            counts, _ = np.histogram(out[:, axis], bins=bins)
            assert counts.tolist() == [1] * n, f"axis {axis} not stratified: {counts}"

    def test_determinism_from_seed(self) -> None:
        a = lhs_unit_cube(32, 4, np.random.default_rng(99))
        b = lhs_unit_cube(32, 4, np.random.default_rng(99))
        assert np.array_equal(a, b)

    def test_different_seeds_give_different_matrices(self) -> None:
        a = lhs_unit_cube(32, 4, np.random.default_rng(1))
        b = lhs_unit_cube(32, 4, np.random.default_rng(2))
        # Astronomically unlikely to collide bit-for-bit; even one
        # element differing is enough to falsify "fixed output".
        assert not np.array_equal(a, b)

    def test_bad_n_samples_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            lhs_unit_cube(0, 3, np.random.default_rng(0))

    def test_bad_n_axes_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_axes must be >= 1"):
            lhs_unit_cube(8, 0, np.random.default_rng(0))


# ====================================================== sampler-level behaviour


def _sample(text: str, seed: int = 42) -> tuple[Suite, list[dict[str, float]]]:
    suite = load_suite_from_string(text)
    cells = LatinHypercubeSampler().sample(suite, np.random.default_rng(seed))
    return suite, [dict(c.values) for c in cells]


class TestSamplerOutputShape:
    def test_emits_exactly_n_samples_cells(self) -> None:
        _, rows = _sample(_LHS_FOUR_AXES)
        assert len(rows) == 32

    def test_each_cell_covers_every_axis(self) -> None:
        suite, rows = _sample(_LHS_FOUR_AXES)
        expected_keys = set(suite.axes.keys())
        for row in rows:
            assert set(row.keys()) == expected_keys

    def test_cell_indices_contiguous_from_zero(self) -> None:
        suite = load_suite_from_string(_LHS_FOUR_AXES)
        cells = LatinHypercubeSampler().sample(suite, np.random.default_rng(42))
        assert [c.index for c in cells] == list(range(len(cells)))


class TestSamplerDeterminism:
    def test_same_seed_same_list(self) -> None:
        _, rows_a = _sample(_LHS_FOUR_AXES, seed=42)
        _, rows_b = _sample(_LHS_FOUR_AXES, seed=42)
        assert rows_a == rows_b

    def test_different_seed_different_list(self) -> None:
        _, rows_a = _sample(_LHS_FOUR_AXES, seed=1)
        _, rows_b = _sample(_LHS_FOUR_AXES, seed=2)
        assert rows_a != rows_b


class TestSamplerRanges:
    def test_continuous_values_in_declared_range(self) -> None:
        suite, rows = _sample(_LHS_FOUR_AXES)
        for row in rows:
            for axis_name, value in row.items():
                spec = suite.axes[axis_name]
                assert spec.low is not None
                assert spec.high is not None
                # LHS draws from [low, low + u*(high-low)) with u in [0,1).
                # The upper bound is therefore strict; we accept inclusive
                # at low and inclusive-at-high-with-tiny-epsilon for
                # floating-point. ``u in [0, 1)`` rules out hitting
                # exactly ``high`` so a strict ``<`` is correct.
                assert spec.low <= value < spec.high or value == pytest.approx(spec.high)

    def test_categorical_emits_only_declared_values(self) -> None:
        suite, rows = _sample(_LHS_WITH_CATEGORICAL)
        legal = set(suite.axes["object_texture"].values or [])
        observed = {row["object_texture"] for row in rows}
        assert observed <= legal

    def test_categorical_balanced_at_n_samples_multiple_of_K(self) -> None:
        # n_samples=50, K=2 -> exactly 25 of each (every stratum hit
        # once, bottom-half of [0,1) maps to value[0], top-half to
        # value[1]).
        _, rows = _sample(_LHS_WITH_CATEGORICAL)
        counts = {0.0: 0, 1.0: 0}
        for row in rows:
            counts[row["object_texture"]] += 1
        assert counts == {0.0: 25, 1.0: 25}, f"counts = {counts}"

    def test_integer_axis_values_in_range(self) -> None:
        suite, rows = _sample(_LHS_WITH_INT)
        spec = suite.axes["distractor_count"]
        assert spec.low is not None
        assert spec.high is not None
        for row in rows:
            v = row["distractor_count"]
            assert spec.low <= v < spec.high or v == pytest.approx(spec.high)


class TestSamplerMarginalStratification:
    """Cross-check the LHS marginal-coverage guarantee at the Suite layer.

    Even though the actual axis values are mapped onto declared ranges,
    every axis must still cover all ``n_samples`` *strata*. Bin each
    axis into ``n_samples`` equal-width strata over its declared range
    and confirm one sample per bin.
    """

    def test_every_axis_hits_every_stratum_exactly_once(self) -> None:
        suite, rows = _sample(_LHS_FOUR_AXES)
        n = suite.n_samples
        assert n is not None
        for axis_name, spec in suite.axes.items():
            if spec.values is not None:
                continue  # categorical axes have a different stratification
            assert spec.low is not None
            assert spec.high is not None
            bins = np.linspace(spec.low, spec.high, n + 1)
            # ``np.histogram`` with right-open bins (default) matches
            # the LHS contract: u in [i/n, (i+1)/n). The final bin's
            # right edge is inclusive only on np.histogram's last bin,
            # which matches u in [(n-1)/n, 1.0]. ``u`` is always < 1.0
            # so the last bin getting exactly one sample is guaranteed.
            values = np.array([row[axis_name] for row in rows], dtype=float)
            counts, _ = np.histogram(values, bins=bins)
            assert counts.tolist() == [1] * n, (
                f"axis {axis_name!r} not stratified: counts={counts.tolist()}"
            )


class TestSamplerErrorPaths:
    def test_missing_n_samples_raises_clear_error(self) -> None:
        # Bypass the loader by constructing a Suite-like object that
        # somehow reaches the sampler with n_samples=None. The schema
        # validator catches this at the top level, but the sampler
        # has its own defence-in-depth check we want to cover.
        from gauntlet.suite.schema import Suite

        # Construct via model_construct to skip validation entirely
        # — this is the only way to land in the sampler with bad
        # state, and matches what a buggy direct caller would do.
        bad = Suite.model_construct(
            name="x",
            env="tabletop",
            episodes_per_cell=1,
            axes={
                "lighting_intensity": AxisSpec(low=0.3, high=1.5),
            },
            sampling="latin_hypercube",
            n_samples=None,
        )
        with pytest.raises(ValueError, match="n_samples"):
            LatinHypercubeSampler().sample(bad, np.random.default_rng(0))


# ===================================================== domain-win demonstration


class TestDomainWinHeadlineNumber:
    """The pitch: 5 axes x 5 steps cartesian = 3,125 cells; LHS at
    n_samples=64 covers the same hypercube with strictly higher
    marginal stratification (64 strata per axis vs the cartesian 5).

    This test pins the headline number that goes in the PR description.
    """

    def test_5x5x5x5x5_cartesian_vs_lhs64(self) -> None:
        # The cartesian sweep we are saving the user from running:
        cart_text = """
name: five-axes-cart
env: tabletop
sampling: cartesian
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.5
    steps: 5
  camera_offset_x:
    low: -0.05
    high: 0.05
    steps: 5
  camera_offset_y:
    low: -0.05
    high: 0.05
    steps: 5
  object_initial_pose_x:
    low: -0.15
    high: 0.15
    steps: 5
  object_initial_pose_y:
    low: -0.15
    high: 0.15
    steps: 5
"""
        cart = load_suite_from_string(cart_text)
        cart_cells = list(cart.cells())
        assert len(cart_cells) == 3125  # = 5 ** 5

        lhs_text = cart_text.replace(
            "sampling: cartesian", "sampling: latin_hypercube\nn_samples: 64"
        )
        # Strip per-axis ``steps`` lines (LHS forbids them).
        lhs_text = "\n".join(
            line for line in lhs_text.splitlines() if not line.strip().startswith("steps:")
        )
        lhs = load_suite_from_string(lhs_text)
        lhs_cells = LatinHypercubeSampler().sample(lhs, np.random.default_rng(0))
        assert len(lhs_cells) == 64

        # Sanity: LHS gives 64 distinct strata per axis vs cartesian's 5.
        for axis in lhs.axes:
            distinct = {round(c.values[axis], 6) for c in lhs_cells}
            assert len(distinct) == 64, (
                f"axis {axis!r} expected 64 distinct LHS strata; got {len(distinct)}"
            )

        # Headline ratio: 3125 / 64 ≈ 49x fewer rollouts.
        assert len(cart_cells) // len(lhs_cells) >= 48
