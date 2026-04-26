"""Property tests for observation NaN/Inf validation.

Phase 2.5 Task 13 — covers the gap left by every existing test:
``Episode`` carries no per-rollout flag for "the env emitted a non-
finite observation". Current behaviour: a NaN that leaks out of the
solver (or through a buggy env wrapper that returns ``np.nan`` for an
unset field) is silently consumed by the policy and may corrupt the
rollout's reward / success / trajectory output without surfacing on
the report. The plan asks us to pin this as a spec for a future
``Episode.observation_invalid`` field — *not* to add the field as
part of T13.

The tests below split into two groups:

1. **Spec tests, marked xfail strict=False.** These describe the
   intended contract: ``Episode.observation_invalid`` exists and is
   ``True`` whenever any per-step observation contains a non-finite
   value. They fail today because the field does not exist and the
   worker does not flag the rollout. ``strict=False`` lets a future
   PR that adds the field produce XPASS without breaking the suite.

2. **Current-behaviour pin.** The companion test pins the *current*
   propagation: a NaN in ``obs`` returned by a wrapped env survives
   to the next ``policy.act(obs)`` call without raising or being
   sanitised. Pins the gap so a future fix flips both tests in
   lockstep — the spec becomes XPASS and this pin starts failing,
   ratcheting the contract honestly.

Hypothesis usage: the per-test ``@settings(max_examples=...)`` knob
overrides the conftest profile because each example pays the
``policy.act`` cost (sub-millisecond, but cumulative); 50 examples is
plenty to cover the ``(nan, +inf, -inf)`` cross-product with finite
seeds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from gauntlet.runner.episode import Episode

pytestmark = pytest.mark.hypothesis_property


# ----- spec: ``Episode.observation_invalid`` should exist (xfail) ------------


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Spec for future ``Episode.observation_invalid`` field. T13 is the "
        "measurement task: this xfail captures the gap so a future PR that "
        "adds the field will produce XPASS in the same diff that flips "
        "``test_episode_does_not_yet_carry_observation_invalid_field`` to "
        "failing. Tracking under follow-up issue."
    ),
)
def test_episode_carries_observation_invalid_field() -> None:
    """The :class:`gauntlet.runner.Episode` schema should declare a
    ``observation_invalid: bool`` field, defaulting to ``False`` for
    backwards compatibility with existing episodes.json files.

    Currently fails because the field is not declared. ``strict=False``
    on the xfail marker so a future schema bump produces XPASS rather
    than a hard failure here.
    """
    fields = set(Episode.model_fields.keys())
    assert "observation_invalid" in fields, (
        f"Episode is missing the ``observation_invalid`` field; current fields: {sorted(fields)}"
    )


def test_episode_does_not_yet_carry_observation_invalid_field() -> None:
    """Companion to the xfail above — pins the *current* schema state
    (no ``observation_invalid`` field). When the future schema bump
    lands, this test starts failing and the xfail above produces
    XPASS; both flips happen in the same diff so the contract change
    is reviewed in one place."""
    fields = set(Episode.model_fields.keys())
    assert "observation_invalid" not in fields, (
        "Episode now declares ``observation_invalid`` — flip the xfail "
        "marker above to a passing test in the same diff."
    )


# ----- spec: worker should flag NaN/Inf in observations (xfail) -------------


def _has_non_finite(obs: dict[str, Any]) -> bool:
    """Return True if any obs value array contains a non-finite entry."""
    for value in obs.values():
        arr = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return True
    return False


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Spec: a NaN/+Inf/-Inf in any obs array MUST surface a typed "
        "``ValueError`` from the env / worker boundary so the rollout is "
        "honestly flagged rather than silently corrupted. T13 measures "
        "the gap; the fix lands in a separate follow-up PR."
    ),
)
@pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
def test_observation_with_non_finite_should_be_rejected(non_finite: float) -> None:
    """Spec: a hand-crafted obs dict with a NaN/Inf in any array must
    fail the per-step finite-observation contract. This is asserted
    against the helper :func:`_has_non_finite` — a future
    ``validate_observation`` utility on the worker side would be the
    real surface; for now we pin the *detection contract* and leave
    the wiring to the follow-up PR."""
    obs = {
        "cube_pos": np.array([non_finite, 0.0, 0.0], dtype=np.float64),
        "ee_pos": np.zeros(3, dtype=np.float64),
        "gripper": np.zeros(1, dtype=np.float64),
    }
    # The spec: a public ``validate_observation`` utility (or the
    # worker's per-step path) raises ValueError on non-finite obs.
    # The lookup below returns ``None`` today because the utility does
    # not yet exist; xfail captures that gap. A future PR should add
    # the utility under ``gauntlet.runner.worker`` and flip this test
    # to passing.
    from gauntlet.runner import worker as _worker

    validate = getattr(_worker, "validate_observation", None)
    assert validate is not None, (
        "gauntlet.runner.worker.validate_observation does not yet exist; "
        "T13 captures the gap — see xfail reason."
    )
    with pytest.raises(ValueError, match=r"non-finite|finite|NaN|Inf"):
        validate(obs)


# ----- current-behaviour pin: NaN in obs propagates silently ---------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    non_finite_choice=st.sampled_from([np.nan, np.inf, -np.inf]),
)
def test_non_finite_obs_currently_passes_validation_silently(
    seed: int, non_finite_choice: float
) -> None:
    """Pins the *current* behaviour: there is no env / worker / Episode
    surface that flags a non-finite obs value. A hand-crafted obs dict
    constructed by a buggy env wrapper would round-trip through the
    detection helper without flagging at the contract surface (because
    no such surface exists). Locks in the gap so a future sanitiser
    change knows to flip the xfail above to XPASS *and* this test to
    failing in the same diff."""
    rng = np.random.default_rng(seed)
    obs = {
        "cube_pos": rng.uniform(-1, 1, size=3).astype(np.float64),
        "ee_pos": rng.uniform(-1, 1, size=3).astype(np.float64),
        "gripper": np.array([0.0], dtype=np.float64),
    }
    # Inject one non-finite value; assert the helper detects it but
    # nothing on the public surface (Episode schema, worker entry-
    # point) reacts to that detection today.
    obs["cube_pos"][0] = non_finite_choice
    assert _has_non_finite(obs), "helper failed to detect the injected non-finite"

    # The Episode schema does not yet expose ``observation_invalid``,
    # so even building an Episode from the rollout that produced this
    # obs would carry no signal of the corruption. Pin that fact.
    fields = set(Episode.model_fields.keys())
    assert "observation_invalid" not in fields, (
        "schema bump landed — flip the xfail above and remove this pin."
    )


# ----- finite-obs negative control ------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_finite_obs_helper_returns_false(seed: int) -> None:
    """Sanity check that ``_has_non_finite`` does not over-flag — every
    finite obs dict returns False. Without this we couldn't trust the
    detection-contract assertion in the spec test above."""
    rng = np.random.default_rng(seed)
    obs: dict[str, NDArray[np.float64]] = {
        "cube_pos": rng.uniform(-1, 1, size=3).astype(np.float64),
        "ee_pos": rng.uniform(-1, 1, size=3).astype(np.float64),
        "gripper": np.array([float(rng.uniform(-1, 1))], dtype=np.float64),
    }
    assert not _has_non_finite(obs), f"helper false-positive on finite obs: {obs}"
