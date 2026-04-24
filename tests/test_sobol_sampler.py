"""Sobol sampler — deferral contract tests.

This polish PR ships LHS only. The :class:`SobolSampler` placeholder
exists so the YAML grammar is forward-compatible: a YAML written today
with ``sampling: sobol`` validates and round-trips through Pydantic.
The actual draw raises :class:`NotImplementedError` with a clear
"planned for follow-up PR; LHS is supported" message.

These tests pin the deferral contract so a follow-up PR that ships
real Sobol can drop them as a unit and replace them with the actual
algorithm tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from gauntlet.suite import load_suite_from_string
from gauntlet.suite.sampling import SobolSampler, build_sampler

_SOBOL_YAML = """
name: sobol-deferred
env: tabletop
seed: 42
sampling: sobol
n_samples: 16
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.5
  camera_offset_x:
    low: -0.05
    high: 0.05
"""


class TestSobolSchemaForwardCompatibility:
    """The schema must accept ``sampling: sobol`` even though the sampler
    is deferred — a YAML written today must continue to load when the
    real Sobol implementation lands.
    """

    def test_yaml_loads(self) -> None:
        suite = load_suite_from_string(_SOBOL_YAML)
        assert suite.sampling == "sobol"
        assert suite.n_samples == 16

    def test_n_samples_required_for_sobol(self) -> None:
        from pydantic import ValidationError

        bad = _SOBOL_YAML.replace("n_samples: 16\n", "")
        with pytest.raises(ValidationError, match="n_samples is required"):
            load_suite_from_string(bad)


class TestSobolDeferralContract:
    """The placeholder raises a clear error pointing users at LHS."""

    def test_sample_raises_not_implemented(self) -> None:
        suite = load_suite_from_string(_SOBOL_YAML)
        with pytest.raises(NotImplementedError) as excinfo:
            SobolSampler().sample(suite, np.random.default_rng(0))
        msg = str(excinfo.value)
        assert "follow-up" in msg.lower()
        # The error must point users at the supported alternative so
        # they don't bounce on a dead end.
        assert "lhs" in msg.lower() or "latin_hypercube" in msg.lower()

    def test_suite_cells_propagates_not_implemented(self) -> None:
        # Going through the public ``Suite.cells`` path must surface
        # the same error — the dispatch in ``Suite.cells`` does not
        # swallow it.
        suite = load_suite_from_string(_SOBOL_YAML)
        with pytest.raises(NotImplementedError, match="follow-up"):
            list(suite.cells())

    def test_build_sampler_returns_placeholder_instance(self) -> None:
        # The dispatch table must return a real SobolSampler instance
        # (not e.g. raise eagerly at construction). Construction-time
        # raising would prevent the schema from validating ``sampling:
        # sobol`` YAMLs, breaking forward compatibility.
        sampler = build_sampler("sobol")
        assert isinstance(sampler, SobolSampler)
