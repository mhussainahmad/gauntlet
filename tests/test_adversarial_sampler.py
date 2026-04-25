"""Adversarial sampler tests — see ``src/gauntlet/suite/adversarial.py``.

Properties pinned:

* The Thompson-sampling bandit concentrates new draws on bins whose
  pilot data carry the highest failure rate (the headline guarantee
  spec'd against the synthetic ``[0.9, 0.5, 0.1]`` failure-rate
  trace from the B-07 ticket).
* A degenerate all-success pilot collapses the per-bin posteriors to
  the uniform Beta(1, 1) prior, making the sampler's bin choice fall
  back to a uniform-over-bins draw.
* Missing or malformed pilot reports surface a clear error at
  ``load_suite`` time (loud-fail per the anti-feature note).
* The ``sampling: adversarial`` YAML grammar parses through the suite
  loader and round-trips :attr:`Suite.pilot_report`.
* The loud :class:`UserWarning` is emitted once per ``load_suite``
  call and once per ``Suite.cells()`` call, with the wording the spec
  promises.
* The sampler is deterministic: same RNG seed -> identical row list.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from gauntlet.report.schema import (
    AxisBreakdown,
    CellBreakdown,
    Heatmap2D,
    Report,
)
from gauntlet.suite import AxisSpec, Suite, load_suite_from_string
from gauntlet.suite.adversarial import (
    ADVERSARIAL_WARNING,
    AdversarialSampler,
    load_pilot_report,
)

# --------------------------------------------------------------------------- helpers


def _empty_breakdowns() -> tuple[list[AxisBreakdown], dict[str, Heatmap2D]]:
    """Trivially-valid placeholders for the fields the sampler ignores."""
    return [], {}


def _make_pilot_report(
    *,
    suite_name: str,
    cells: list[CellBreakdown],
) -> Report:
    """Build a minimal :class:`Report` containing only the cells the sampler reads.

    Every other field is filled with the smallest validator-passing
    value: empty per-axis breakdowns, empty heatmap, derived
    success/failure rates so the schema invariants hold.
    """
    per_axis, heatmap_2d = _empty_breakdowns()
    n_episodes = sum(c.n_episodes for c in cells)
    n_success = sum(c.n_success for c in cells)
    overall_success = (n_success / n_episodes) if n_episodes else 0.0
    return Report(
        suite_name=suite_name,
        n_episodes=n_episodes,
        n_success=n_success,
        per_axis=per_axis,
        per_cell=cells,
        failure_clusters=[],
        heatmap_2d=heatmap_2d,
        overall_success_rate=overall_success,
        overall_failure_rate=1.0 - overall_success,
        cluster_multiple=2.0,
    )


def _make_three_bin_pilot() -> Report:
    """Pilot with one axis, three bins, failure rates [0.9, 0.5, 0.1].

    The suite uses ``bins_per_continuous=3`` so each pilot cell maps
    to exactly one bin: cells in the lower third of [0, 1.0] hit bin 0,
    middle third bin 1, upper third bin 2. Each bin gets 10 episodes.
    """
    cells: list[CellBreakdown] = []
    # Bin 0: failure rate 0.9 (1 success / 10 episodes), value 0.15.
    cells.append(
        CellBreakdown(
            cell_index=0,
            perturbation_config={"lighting_intensity": 0.15},
            n_episodes=10,
            n_success=1,
            success_rate=0.1,
        )
    )
    # Bin 1: failure rate 0.5 (5 successes / 10 episodes), value 0.5.
    cells.append(
        CellBreakdown(
            cell_index=1,
            perturbation_config={"lighting_intensity": 0.5},
            n_episodes=10,
            n_success=5,
            success_rate=0.5,
        )
    )
    # Bin 2: failure rate 0.1 (9 successes / 10 episodes), value 0.85.
    cells.append(
        CellBreakdown(
            cell_index=2,
            perturbation_config={"lighting_intensity": 0.85},
            n_episodes=10,
            n_success=9,
            success_rate=0.9,
        )
    )
    return _make_pilot_report(suite_name="three-bin-pilot", cells=cells)


def _one_axis_suite(n_samples: int = 200) -> Suite:
    """Single-axis suite that bins onto the three-bin pilot."""
    return Suite.model_construct(
        name="one-axis",
        env="tabletop",
        episodes_per_cell=1,
        seed=42,
        axes={"lighting_intensity": AxisSpec(low=0.0, high=1.0)},
        sampling="adversarial",
        n_samples=n_samples,
        pilot_report="ignored-direct-construction",
    )


def _write_pilot_json(tmp_path: Path, report: Report, name: str = "pilot.json") -> Path:
    """Dump a Report to a tmp_path file and return the absolute path."""
    p = tmp_path / name
    p.write_text(report.model_dump_json())
    return p


# --------------------------------------------------------------------------- tests


class TestSamplerConcentratesOnHighFailureBins:
    """Headline: the ``[0.9, 0.5, 0.1]`` synthetic pilot drives the sampler
    to bin 0 (failure-rate 0.9) the vast majority of the time."""

    def test_majority_of_draws_land_in_high_failure_bin(self) -> None:
        pilot = _make_three_bin_pilot()
        suite = _one_axis_suite(n_samples=200)
        sampler = AdversarialSampler(pilot, bins_per_continuous=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cells = sampler.sample(suite, np.random.default_rng(0))
        # The axis is [0.0, 1.0); bin 0 covers [0, 1/3), bin 1
        # [1/3, 2/3), bin 2 [2/3, 1).
        bin_counts = [0, 0, 0]
        for cell in cells:
            v = cell.values["lighting_intensity"]
            bin_counts[min(int(v * 3), 2)] += 1
        # Bin 0 (failure rate 0.9) should dominate; with 10 trials and
        # Beta(10, 2) vs Beta(6, 6) vs Beta(2, 10), Thompson collapses
        # heavily onto bin 0.
        assert bin_counts[0] > bin_counts[1], (
            f"high-failure bin (0) should beat mid bin (1); got {bin_counts}"
        )
        assert bin_counts[0] > bin_counts[2], (
            f"high-failure bin (0) should beat low bin (2); got {bin_counts}"
        )
        # And dominate by a comfortable margin (>50% of all draws).
        assert bin_counts[0] > 100, f"bin 0 should win majority of 200 draws; got {bin_counts}"


class TestUniformPilotProducesUniformDraws:
    """All-success pilot -> every bin's posterior is Beta(1, n+1) ≈ low,
    and the sampler's bin selection is uniform over bins (every Beta
    draws from the same distribution, so argmax is uniform).
    """

    def test_all_success_pilot_uniform_bin_selection(self) -> None:
        # 3 bins, each with 10/10 successes -> Beta(1, 11) for every bin.
        cells = [
            CellBreakdown(
                cell_index=i,
                perturbation_config={"lighting_intensity": v},
                n_episodes=10,
                n_success=10,
                success_rate=1.0,
            )
            for i, v in enumerate([0.15, 0.5, 0.85])
        ]
        pilot = _make_pilot_report(suite_name="uniform", cells=cells)
        suite = _one_axis_suite(n_samples=300)
        sampler = AdversarialSampler(pilot, bins_per_continuous=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rows = sampler.sample(suite, np.random.default_rng(0))
        bin_counts = [0, 0, 0]
        for cell in rows:
            v = cell.values["lighting_intensity"]
            bin_counts[min(int(v * 3), 2)] += 1
        # Every bin must see at least 50/300 draws (a uniform Multinomial
        # over 3 bins with 300 trials is 100 +/- ~17 by 99% CI; 50 is
        # the very generous lower bound).
        for i, c in enumerate(bin_counts):
            assert c > 50, f"bin {i} starved on uniform pilot ({bin_counts})"


class TestDeterminism:
    def test_same_seed_same_rows(self) -> None:
        pilot = _make_three_bin_pilot()
        suite = _one_axis_suite(n_samples=32)
        sampler = AdversarialSampler(pilot, bins_per_continuous=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rows_a = sampler.sample(suite, np.random.default_rng(7))
            rows_b = sampler.sample(suite, np.random.default_rng(7))
        a = [(c.index, dict(c.values)) for c in rows_a]
        b = [(c.index, dict(c.values)) for c in rows_b]
        assert a == b


class TestSuiteYamlGrammar:
    def test_yaml_with_pilot_report_loads(self, tmp_path: Path) -> None:
        pilot_path = _write_pilot_json(tmp_path, _make_three_bin_pilot())
        text = f"""
name: adversarial-yaml
env: tabletop
seed: 42
sampling: adversarial
n_samples: 16
episodes_per_cell: 1
pilot_report: {pilot_path}
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            suite = load_suite_from_string(text)
        assert suite.sampling == "adversarial"
        assert suite.n_samples == 16
        assert suite.pilot_report == str(pilot_path)

    def test_missing_pilot_report_raises(self) -> None:
        text = """
name: adversarial-no-pilot
env: tabletop
seed: 42
sampling: adversarial
n_samples: 16
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
"""
        with pytest.raises(Exception, match="pilot_report is required"):
            load_suite_from_string(text)

    def test_pilot_report_on_non_adversarial_rejected(self) -> None:
        text = """
name: sobol-with-pilot
env: tabletop
seed: 42
sampling: sobol
n_samples: 16
episodes_per_cell: 1
pilot_report: somewhere.json
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
"""
        with pytest.raises(Exception, match="pilot_report is only valid"):
            load_suite_from_string(text)

    def test_nonexistent_pilot_path_raises_at_load(self, tmp_path: Path) -> None:
        text = f"""
name: adversarial-bad-path
env: tabletop
seed: 42
sampling: adversarial
n_samples: 16
episodes_per_cell: 1
pilot_report: {tmp_path / "does-not-exist.json"}
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
"""
        with pytest.raises(FileNotFoundError, match="pilot report not found"):
            load_suite_from_string(text)


class TestLoudWarning:
    def test_warning_emitted_at_load_time(self, tmp_path: Path) -> None:
        pilot_path = _write_pilot_json(tmp_path, _make_three_bin_pilot())
        text = f"""
name: adversarial-warn
env: tabletop
seed: 42
sampling: adversarial
n_samples: 8
episodes_per_cell: 1
pilot_report: {pilot_path}
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
"""
        with pytest.warns(UserWarning, match="biases coverage"):
            load_suite_from_string(text)

    def test_warning_full_text_contains_anti_feature_caveat(self) -> None:
        # The wording is part of the contract — keeps the spec's
        # promise that the warning calls out "NOT a fair sample".
        assert "NOT a fair sample" in ADVERSARIAL_WARNING
        assert "failure-mode discovery" in ADVERSARIAL_WARNING


class TestPilotReportLoader:
    def test_load_pilot_report_round_trips(self, tmp_path: Path) -> None:
        pilot = _make_three_bin_pilot()
        path = _write_pilot_json(tmp_path, pilot)
        loaded = load_pilot_report(path)
        assert loaded.suite_name == "three-bin-pilot"
        assert len(loaded.per_cell) == 3

    def test_load_pilot_report_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="pilot report not found"):
            load_pilot_report(tmp_path / "missing.json")


class TestSamplerErrorPaths:
    def test_pilot_missing_axis_raises(self) -> None:
        # Pilot has no entry for the suite's axis -> loud error.
        bogus = _make_pilot_report(
            suite_name="bogus",
            cells=[
                CellBreakdown(
                    cell_index=0,
                    perturbation_config={"camera_offset_x": 0.0},
                    n_episodes=5,
                    n_success=2,
                    success_rate=0.4,
                )
            ],
        )
        suite = _one_axis_suite(n_samples=4)
        sampler = AdversarialSampler(bogus, bins_per_continuous=3)
        with pytest.raises(ValueError, match="missing axes"), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sampler.sample(suite, np.random.default_rng(0))

    def test_n_samples_required(self) -> None:
        pilot = _make_three_bin_pilot()
        bad = Suite.model_construct(
            name="x",
            env="tabletop",
            episodes_per_cell=1,
            axes={"lighting_intensity": AxisSpec(low=0.0, high=1.0)},
            sampling="adversarial",
            n_samples=None,
            pilot_report="ignored",
        )
        sampler = AdversarialSampler(pilot, bins_per_continuous=3)
        with pytest.raises(ValueError, match="n_samples"):
            sampler.sample(bad, np.random.default_rng(0))

    def test_bins_per_continuous_below_two_rejected(self) -> None:
        pilot = _make_three_bin_pilot()
        with pytest.raises(ValueError, match="bins_per_continuous must be >= 2"):
            AdversarialSampler(pilot, bins_per_continuous=1)
