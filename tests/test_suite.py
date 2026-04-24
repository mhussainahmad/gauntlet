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


class TestBackendLazyImport:
    """Phase 2 Task 5 step 5 — schema-accepts-known + loader-imports-on-demand
    paths. The schema is simulator-agnostic; the loader raises a clear,
    user-facing install-hint error when the matching extra is missing.
    """

    _PYBULLET_YAML = """
name: pybullet-smoke
env: tabletop-pybullet
episodes_per_cell: 1
axes:
  object_initial_pose_x:
    low: -0.05
    high: 0.05
    steps: 2
"""

    def test_schema_accepts_tabletop_pybullet_even_before_backend_import(self) -> None:
        """The validator must accept any key in BUILTIN_BACKEND_IMPORTS
        independent of whether the subpackage has been imported yet.
        """
        from gauntlet.suite.schema import BUILTIN_BACKEND_IMPORTS, Suite

        assert "tabletop-pybullet" in BUILTIN_BACKEND_IMPORTS
        # Direct Suite.model_validate — no loader-side import triggered.
        suite = Suite.model_validate(
            {
                "name": "x",
                "env": "tabletop-pybullet",
                "episodes_per_cell": 1,
                "axes": {
                    "object_initial_pose_x": {
                        "low": 0.0,
                        "high": 0.1,
                        "steps": 2,
                    }
                },
            }
        )
        assert suite.env == "tabletop-pybullet"

    def test_loader_surfaces_install_hint_for_missing_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Importing the backend subpackage at load time must be converted
        into the documented install-hint error when it raises.

        Canonical simulation per RFC-005 §9.1 case 12: monkeypatch
        ``sys.modules['pybullet']`` to ``None`` so the subpackage's own
        ``import pybullet`` guard raises ``ImportError`` on re-import.
        The loader catches that and re-raises as the user-facing
        ``ValueError`` with the install hint.
        """
        import sys

        from gauntlet.env.registry import _REGISTRY

        # Simulate "the [pybullet] extra is not installed":
        # block pybullet itself and unload any cached gauntlet.env.pybullet
        # so the subpackage's __init__ runs again and hits its ImportError.
        monkeypatch.setitem(sys.modules, "pybullet", None)
        monkeypatch.setitem(sys.modules, "pybullet_data", None)
        for mod in list(sys.modules):
            if mod == "gauntlet.env.pybullet" or mod.startswith("gauntlet.env.pybullet."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        # If a prior test successfully registered the backend, drop the
        # entry for the duration of this test so the loader actually
        # attempts the import.
        if "tabletop-pybullet" in _REGISTRY:
            saved = _REGISTRY.pop("tabletop-pybullet")
            monkeypatch.setitem(_REGISTRY, "tabletop-pybullet", saved)
            del _REGISTRY["tabletop-pybullet"]

        with pytest.raises(ValueError) as excinfo:
            load_suite_from_string(self._PYBULLET_YAML)
        msg = str(excinfo.value)
        assert "tabletop-pybullet" in msg
        assert "extra is not installed" in msg
        assert "uv sync --extra pybullet" in msg
        assert "pip install 'gauntlet[pybullet]'" in msg

    def test_subpackage_import_raises_install_hint_when_pybullet_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RFC-005 §9.1 case 12 — direct import of the subpackage must
        surface the ``uv sync --extra pybullet`` hint when
        :mod:`pybullet` itself is unavailable.
        """
        import importlib
        import sys

        monkeypatch.setitem(sys.modules, "pybullet", None)
        monkeypatch.setitem(sys.modules, "pybullet_data", None)
        for mod in list(sys.modules):
            if mod == "gauntlet.env.pybullet" or mod.startswith("gauntlet.env.pybullet."):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        with pytest.raises(ImportError) as excinfo:
            importlib.import_module("gauntlet.env.pybullet")
        msg = str(excinfo.value)
        assert "uv sync --extra pybullet" in msg
        assert "pip install 'gauntlet[pybullet]'" in msg

    def test_schema_rejects_unknown_env_names(self) -> None:
        """Anything outside registered_envs() | BUILTIN_BACKEND_IMPORTS
        must still be rejected at schema-validation time.

        ``tabletop-mystery`` is an obviously-fake env name picked so it
        cannot collide with any current or future built-in backend
        (``tabletop-isaac`` joined ``BUILTIN_BACKEND_IMPORTS`` in
        RFC-009, which was the previous canary value here).
        """
        bad = """
name: x
env: tabletop-mystery
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.0
    high: 1.0
    steps: 2
"""
        with pytest.raises(ValidationError) as excinfo:
            load_suite_from_string(bad)
        msg = str(excinfo.value)
        assert "tabletop-mystery" in msg
        # Error includes the sorted list of known keys.
        assert "tabletop" in msg
        assert "tabletop-pybullet" in msg
        assert "tabletop-isaac" in msg

    _GENESIS_YAML = """
name: genesis-smoke
env: tabletop-genesis
episodes_per_cell: 1
axes:
  object_initial_pose_x:
    low: -0.05
    high: 0.05
    steps: 2
"""

    def test_schema_accepts_tabletop_genesis_even_before_backend_import(self) -> None:
        """RFC-007 §8 counterpart of the PyBullet test above. The validator
        must accept ``tabletop-genesis`` even when the subpackage has not
        been imported.
        """
        from gauntlet.suite.schema import BUILTIN_BACKEND_IMPORTS, Suite

        assert "tabletop-genesis" in BUILTIN_BACKEND_IMPORTS
        suite = Suite.model_validate(
            {
                "name": "x",
                "env": "tabletop-genesis",
                "episodes_per_cell": 1,
                "axes": {
                    "object_initial_pose_x": {
                        "low": 0.0,
                        "high": 0.1,
                        "steps": 2,
                    }
                },
            }
        )
        assert suite.env == "tabletop-genesis"

    def test_loader_surfaces_install_hint_for_missing_genesis_extra(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RFC-007 §8 counterpart of the PyBullet install-hint test.
        Simulate "the [genesis] extra is not installed" by blocking
        :mod:`torch` and :mod:`genesis` in :data:`sys.modules` and
        unloading cached copies of the ``gauntlet.env.genesis``
        subpackage. The loader must catch the resulting ImportError
        and surface the install hint.
        """
        import sys

        from gauntlet.env.registry import _REGISTRY

        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "genesis", None)
        for mod in list(sys.modules):
            if mod == "gauntlet.env.genesis" or mod.startswith("gauntlet.env.genesis."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        if "tabletop-genesis" in _REGISTRY:
            saved = _REGISTRY.pop("tabletop-genesis")
            monkeypatch.setitem(_REGISTRY, "tabletop-genesis", saved)
            del _REGISTRY["tabletop-genesis"]

        with pytest.raises(ValueError) as excinfo:
            load_suite_from_string(self._GENESIS_YAML)
        msg = str(excinfo.value)
        assert "tabletop-genesis" in msg
        assert "extra is not installed" in msg
        assert "uv sync --extra genesis" in msg
        assert "pip install 'gauntlet[genesis]'" in msg

    def test_subpackage_import_raises_install_hint_when_genesis_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Direct ``import gauntlet.env.genesis`` must surface the
        ``uv sync --extra genesis`` hint when :mod:`genesis` is
        unavailable. Covers both guard arms — torch-present (so the
        test actually exercises the genesis arm, not the torch arm)
        on a real torch-free install.
        """
        import importlib
        import sys

        monkeypatch.setitem(sys.modules, "genesis", None)
        for mod in list(sys.modules):
            if mod == "gauntlet.env.genesis" or mod.startswith("gauntlet.env.genesis."):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        with pytest.raises(ImportError) as excinfo:
            importlib.import_module("gauntlet.env.genesis")
        msg = str(excinfo.value)
        assert "uv sync --extra genesis" in msg
        assert "pip install 'gauntlet[genesis]'" in msg

    def test_subpackage_import_raises_install_hint_when_torch_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Genesis imports :mod:`torch` at package scope but does not
        declare it in install_requires (RFC-007 §4.1). The subpackage
        guard catches the torch-missing case first and points at the
        ``[genesis]`` extra (which declares torch explicitly)."""
        import importlib
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        for mod in list(sys.modules):
            if mod == "gauntlet.env.genesis" or mod.startswith("gauntlet.env.genesis."):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        with pytest.raises(ImportError) as excinfo:
            importlib.import_module("gauntlet.env.genesis")
        msg = str(excinfo.value)
        # Either message is acceptable as long as it points users to
        # the [genesis] extra; torch-first guard says "torch is
        # required by genesis-world".
        assert "uv sync --extra genesis" in msg
        assert "pip install 'gauntlet[genesis]'" in msg


# --------------------------------------------------------------------- loader
# Phase 2.5 Task 11 — top-level YAML shape + defence-in-depth.


class TestLoaderEdgeCases:
    """Direct coverage for the loader's ``_validate`` /
    ``_ensure_backend_registered`` / ``_visual_only_axes_of`` branches
    that the schema-driven happy-path tests above do not reach.
    """

    def test_load_suite_empty_yaml_rejected(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_suite(empty)

    def test_load_suite_top_level_scalar_rejected(self, tmp_path: Path) -> None:
        scalar = tmp_path / "scalar.yaml"
        scalar.write_text("just-a-scalar\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_suite(scalar)

    def test_load_suite_top_level_list_rejected(self, tmp_path: Path) -> None:
        ls = tmp_path / "list.yaml"
        ls.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_suite(ls)

    def test_load_suite_from_string_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            load_suite_from_string("")

    def test_load_suite_from_string_top_level_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            load_suite_from_string("- entry\n")

    def test_ensure_backend_registered_unknown_env_defence_in_depth(self) -> None:
        """Bypassing the schema validator and feeding an unknown env to
        ``_ensure_backend_registered`` directly must raise ValueError
        (schema would have rejected it; this is belt-and-braces)."""
        from gauntlet.suite.loader import _ensure_backend_registered

        with pytest.raises(ValueError, match="unknown env"):
            _ensure_backend_registered("definitely-not-an-env")

    def test_visual_only_axes_of_handles_partial(self) -> None:
        """``_visual_only_axes_of`` unwraps :class:`functools.partial`
        and reads the underlying class's ``VISUAL_ONLY_AXES``."""
        import functools

        from gauntlet.suite.loader import _visual_only_axes_of

        class _SentinelEnv:
            VISUAL_ONLY_AXES = frozenset({"a", "b"})

        wrapped = functools.partial(_SentinelEnv)
        assert _visual_only_axes_of(wrapped) == frozenset({"a", "b"})

    def test_visual_only_axes_of_handles_missing_attr(self) -> None:
        """A factory that does not declare ``VISUAL_ONLY_AXES`` returns
        an empty frozenset rather than blowing up."""
        from gauntlet.suite.loader import _visual_only_axes_of

        def _bare_factory() -> object:
            return object()

        assert _visual_only_axes_of(_bare_factory) == frozenset()

    def test_visual_only_axes_of_handles_non_frozenset_attr(self) -> None:
        """A factory whose ``VISUAL_ONLY_AXES`` is the wrong type (eg a
        list) is treated as if it had no such attribute."""
        from gauntlet.suite.loader import _visual_only_axes_of

        class _BadAttr:
            VISUAL_ONLY_AXES = ["a", "b"]  # not a frozenset

        assert _visual_only_axes_of(_BadAttr) == frozenset()
