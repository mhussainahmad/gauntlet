"""Tests for ``gauntlet.suite`` — schema + YAML loader.

Covers:

* Round-trip parse of the spec example (with canonical axis names).
* Parse of the shipped ``examples/suites/tabletop-basic-v1.yaml``.
* Firm validation: unknown axis name, bad bounds, bad steps,
  non-positive ``episodes_per_cell``, unsupported ``env``, empty axes,
  and shape-mixing (``{low, high, steps}`` + ``{values}``).
* Categorical axis enumeration fidelity.
* ``Suite.cells()`` grid size, determinism, and ordering.
* Continuous axis midpoint when ``steps == 1``.
* Optional ``seed`` field.
* ``model_dump`` round-trip.
* Loader path-does-not-exist behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gauntlet.suite import (
    AxisSpec,
    Suite,
    SuiteCell,
    load_suite,
    load_suite_from_string,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_SUITE = REPO_ROOT / "examples" / "suites" / "tabletop-basic-v1.yaml"


# A minimal-but-complete suite used by most happy-path tests.
_VALID_YAML = """
name: tabletop-basic-v1
env: tabletop
seed: 42
episodes_per_cell: 10
axes:
  lighting_intensity:
    low: 0.3
    high: 1.5
    steps: 4
  camera_offset_x:
    low: -0.05
    high: 0.05
    steps: 3
"""


# --------------------------------------------------------------------- parse


class TestHappyPath:
    def test_parse_spec_example_with_canonical_names(self) -> None:
        suite = load_suite_from_string(_VALID_YAML)
        assert suite.name == "tabletop-basic-v1"
        assert suite.env == "tabletop"
        assert suite.seed == 42
        assert suite.episodes_per_cell == 10
        assert set(suite.axes.keys()) == {"lighting_intensity", "camera_offset_x"}
        li = suite.axes["lighting_intensity"]
        assert li.low == 0.3
        assert li.high == 1.5
        assert li.steps == 4
        assert li.values is None

    def test_parse_shipped_example_file(self) -> None:
        assert EXAMPLE_SUITE.is_file(), f"example suite missing at {EXAMPLE_SUITE}"
        suite = load_suite(EXAMPLE_SUITE)
        assert suite.name == "tabletop-basic-v1"
        assert suite.env == "tabletop"
        assert suite.episodes_per_cell == 10
        assert "lighting_intensity" in suite.axes
        assert "object_texture" in suite.axes
        # Sanity: cell count matches 4 * 3 * 2 * 6.
        assert suite.num_cells() == 4 * 3 * 2 * 6

    def test_seed_null_parses(self) -> None:
        text = _VALID_YAML.replace("seed: 42", "seed: null")
        suite = load_suite_from_string(text)
        assert suite.seed is None

    def test_seed_omitted_defaults_to_none(self) -> None:
        text = "\n".join(line for line in _VALID_YAML.splitlines() if not line.startswith("seed:"))
        suite = load_suite_from_string(text)
        assert suite.seed is None


# ----------------------------------------------------------------- validation


class TestValidationErrors:
    def test_unknown_axis_name(self) -> None:
        bad = """
name: x
env: tabletop
episodes_per_cell: 1
axes:
  bogus_axis:
    low: 0.0
    high: 1.0
    steps: 2
"""
        with pytest.raises(ValidationError, match="bogus_axis"):
            load_suite_from_string(bad)

    def test_low_greater_than_high(self) -> None:
        bad = """
name: x
env: tabletop
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 2.0
    high: 1.0
    steps: 3
"""
        with pytest.raises(ValidationError, match="low must be <= high"):
            load_suite_from_string(bad)

    def test_steps_less_than_one(self) -> None:
        bad = """
name: x
env: tabletop
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
    steps: 0
"""
        with pytest.raises(ValidationError, match="steps must be >= 1"):
            load_suite_from_string(bad)

    def test_episodes_per_cell_non_positive(self) -> None:
        bad = """
name: x
env: tabletop
episodes_per_cell: 0
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
    steps: 2
"""
        with pytest.raises(ValidationError, match="episodes_per_cell"):
            load_suite_from_string(bad)

    def test_unsupported_env(self) -> None:
        bad = """
name: x
env: not_tabletop
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
    steps: 2
"""
        with pytest.raises(ValidationError, match="tabletop"):
            load_suite_from_string(bad)

    def test_empty_axes(self) -> None:
        bad = """
name: x
env: tabletop
episodes_per_cell: 1
axes: {}
"""
        with pytest.raises(ValidationError, match="at least one axis"):
            load_suite_from_string(bad)

    def test_mixed_continuous_and_categorical_shapes(self) -> None:
        bad = """
name: x
env: tabletop
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
    steps: 2
    values: [0.0, 1.0]
"""
        with pytest.raises(ValidationError, match="cannot mix"):
            load_suite_from_string(bad)

    def test_partial_continuous_shape_rejected(self) -> None:
        bad = """
name: x
env: tabletop
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
"""
        with pytest.raises(ValidationError, match="requires low, high, steps"):
            load_suite_from_string(bad)

    def test_empty_axis_spec_rejected(self) -> None:
        bad = """
name: x
env: tabletop
episodes_per_cell: 1
axes:
  lighting_intensity: {}
"""
        with pytest.raises(ValidationError, match="must specify"):
            load_suite_from_string(bad)

    def test_extra_top_level_field_rejected(self) -> None:
        bad = """
name: x
env: tabletop
episodes_per_cell: 1
surprise: yes
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
    steps: 2
"""
        with pytest.raises(ValidationError):
            load_suite_from_string(bad)

    def test_empty_name_rejected(self) -> None:
        bad = """
name: "   "
env: tabletop
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
    steps: 2
"""
        with pytest.raises(ValidationError, match="non-empty"):
            load_suite_from_string(bad)


# --------------------------------------------------------------- enumeration


class TestCellEnumeration:
    def test_categorical_axis_enumerates_exact_values(self) -> None:
        spec = AxisSpec(values=[0.0, 1.0])
        assert spec.enumerate() == (0.0, 1.0)

    def test_continuous_axis_steps_one_is_midpoint(self) -> None:
        spec = AxisSpec(low=0.3, high=1.5, steps=1)
        (only,) = spec.enumerate()
        assert only == pytest.approx(0.9)

    def test_continuous_axis_inclusive_endpoints(self) -> None:
        spec = AxisSpec(low=0.0, high=1.0, steps=5)
        assert spec.enumerate() == pytest.approx((0.0, 0.25, 0.5, 0.75, 1.0))

    def test_grid_size_is_product_of_axes(self) -> None:
        suite = load_suite_from_string(_VALID_YAML)
        # lighting_intensity steps=4, camera_offset_x steps=3 → 12 cells.
        cells = list(suite.cells())
        assert len(cells) == 12
        assert suite.num_cells() == 12

    def test_cells_are_deterministic_across_iterations(self) -> None:
        suite = load_suite_from_string(_VALID_YAML)
        first = [(c.index, dict(c.values)) for c in suite.cells()]
        second = [(c.index, dict(c.values)) for c in suite.cells()]
        assert first == second

    def test_cells_preserve_yaml_axis_order(self) -> None:
        # Swap declaration order in the YAML; the cell value mappings
        # must reflect the YAML order (lighting first vs camera first).
        text_a = _VALID_YAML
        text_b = """
name: tabletop-basic-v1
env: tabletop
seed: 42
episodes_per_cell: 10
axes:
  camera_offset_x:
    low: -0.05
    high: 0.05
    steps: 3
  lighting_intensity:
    low: 0.3
    high: 1.5
    steps: 4
"""
        suite_a = load_suite_from_string(text_a)
        suite_b = load_suite_from_string(text_b)
        assert tuple(suite_a.axes.keys()) == ("lighting_intensity", "camera_offset_x")
        assert tuple(suite_b.axes.keys()) == ("camera_offset_x", "lighting_intensity")
        # First cell in both suites pulls the first value of each axis
        # in declared order — mapping keys are ordered, and the Runner
        # can depend on that.
        first_a: SuiteCell = next(iter(suite_a.cells()))
        first_b: SuiteCell = next(iter(suite_b.cells()))
        assert tuple(first_a.values.keys()) == ("lighting_intensity", "camera_offset_x")
        assert tuple(first_b.values.keys()) == ("camera_offset_x", "lighting_intensity")

    def test_cell_indices_are_contiguous_from_zero(self) -> None:
        suite = load_suite_from_string(_VALID_YAML)
        indices = [c.index for c in suite.cells()]
        assert indices == list(range(len(indices)))

    def test_rightmost_axis_varies_fastest(self) -> None:
        # 2 x 3 grid -- with rightmost varying fastest, the first 3 cells
        # all share the first lighting value.
        text = """
name: x
env: tabletop
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
    steps: 2
  camera_offset_x:
    low: -0.1
    high: 0.1
    steps: 3
"""
        suite = load_suite_from_string(text)
        cells = list(suite.cells())
        assert len(cells) == 6
        lighting_vals = [c.values["lighting_intensity"] for c in cells]
        assert lighting_vals[0] == lighting_vals[1] == lighting_vals[2]
        assert lighting_vals[3] == lighting_vals[4] == lighting_vals[5]


# ----------------------------------------------------------- round-trip / io


class TestRoundTripAndIo:
    def test_model_dump_round_trip(self) -> None:
        suite = load_suite_from_string(_VALID_YAML)
        dumped = suite.model_dump()
        reparsed = Suite.model_validate(dumped)
        assert reparsed == suite
        # Same cell sequence too.
        assert [dict(c.values) for c in reparsed.cells()] == [dict(c.values) for c in suite.cells()]

    def test_load_suite_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_suite(missing)

    def test_load_suite_accepts_string_path(self, tmp_path: Path) -> None:
        target = tmp_path / "suite.yaml"
        target.write_text(_VALID_YAML, encoding="utf-8")
        suite = load_suite(str(target))
        assert suite.name == "tabletop-basic-v1"
