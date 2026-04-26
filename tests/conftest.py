"""Shared pytest configuration.

Phase 2.5 Task 13 — registers two Hypothesis settings profiles for the
``test_property_*.py`` files added by T13:

* ``"default"`` — overrides the library default with ``max_examples=200``
  and ``deadline=None``. The deadline is unset because some property
  tests construct a :class:`gauntlet.env.tabletop.TabletopEnv` (MuJoCo
  model load is ~50 ms cold), and a per-example deadline would flake
  on slower CI runners.
* ``"ci"`` — same shape but ``max_examples=50`` so CI wall-time stays
  bounded. Loaded automatically when the environment variable
  ``GAUNTLET_HYPOTHESIS_PROFILE=ci`` is set; otherwise the default
  profile applies.

These profiles are global — Hypothesis picks them up at import time so
every test file under ``tests/`` shares the same budget knobs without
each file having to re-declare them. Per-test ``@settings(...)``
decorators still take precedence when present (the existing
``test_fuzz_*`` and ``test_property_*`` files set their own per-test
budgets and remain unchanged).
"""

from __future__ import annotations

import os

from hypothesis import settings

# Default profile: 200 examples, no deadline. The deadline is unset
# because property tests that touch TabletopEnv pay a ~50 ms MuJoCo
# model load on first reset; a per-example deadline tuned for the
# fast no-env path would cause flakes on slower CI runners.
settings.register_profile("default", max_examples=200, deadline=None)

# CI profile: 50 examples, no deadline. The CI workflow opts in via
# GAUNTLET_HYPOTHESIS_PROFILE=ci so a fast PR run does not pay the
# full 200-example budget on every push.
settings.register_profile("ci", max_examples=50, deadline=None)

settings.load_profile(os.environ.get("GAUNTLET_HYPOTHESIS_PROFILE", "default"))
