"""Backwards-compat regression for the rollout-video Polish feature.

Lives at the repo top level (no ``video`` marker) so the default
torch-/extras-free CI job runs it on every commit. The contract
under test: ``Runner(record_video=False)`` (the default) produces
Episodes that are byte-identical to the pre-PR shape *except* for the
new ``video_path`` field, which MUST default to ``None``.

If a future change accidentally turns video recording on by default,
or starts polluting the Episode schema with non-None values when the
user has not opted in, this test fails loudly.

The test deliberately exercises every Runner path that touches the
new fields:

* :class:`Runner.__init__` with no video kwargs.
* :meth:`Runner.run` end-to-end on an in-process worker.
* :meth:`Episode.model_dump(mode="json")` round-trip through
  ``model_validate`` to guarantee the new field is JSON-safe.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from gauntlet.env.base import GauntletEnv
from gauntlet.runner import Episode, Runner
from gauntlet.suite.schema import AxisSpec, Suite

# ────────────────────────────────────────────────────────────────────────
# Fake env — module-level so it pickles under spawn (this test is
# in-process only, but module-level keeps future expansion painless).
# ────────────────────────────────────────────────────────────────────────

_ACTION_DIM = 7


class _NoImageFakeEnv:
    """Backwards-compat fake — does NOT expose ``obs['image']``.

    Mirrors the pre-PR happy path (state-only env, deterministic
    success) so the regression catches any drift in the
    ``record_video=False`` code path. If the runner ever accidentally
    requires ``obs['image']`` regardless of opt-in, the very first
    rollout here would raise ``ValueError``.
    """

    AXIS_NAMES = frozenset({"distractor_count"})
    VISUAL_ONLY_AXES: frozenset[str] = frozenset()

    def __init__(self) -> None:
        from gymnasium import spaces

        self.observation_space = spaces.Dict(
            {"cube_pos": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64)}
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float64)
        self._pending: dict[str, float] = {}
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        self._step_count = 0
        self._pending.clear()
        return ({"cube_pos": np.zeros(3, dtype=np.float64)}, {"seed_echo": seed})

    def step(
        self,
        action: np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        terminated = self._step_count >= 3
        return (
            {"cube_pos": np.zeros(3, dtype=np.float64)},
            1.0,
            terminated,
            False,
            {"success": True, "step_count": self._step_count},
        )

    def set_perturbation(self, name: str, value: float) -> None:
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self._pending[name] = value

    def restore_baseline(self) -> None:
        self._pending.clear()
        self._step_count = 0

    def close(self) -> None:
        return None


def _no_image_factory() -> Any:
    return _NoImageFakeEnv()


def _make_random_policy() -> Any:
    from gauntlet.policy.random import RandomPolicy

    return RandomPolicy(action_dim=_ACTION_DIM, seed=None)


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def test_default_runner_keeps_existing_episode_schema() -> None:
    """``Runner(record_video=False)`` preserves the day-1 Episode contract.

    Back-compat-only intent: the fields that the original
    rollout-video PR promised to keep (``suite_name``, ``cell_index``,
    ``episode_index``, ``seed``, ``perturbation_config``, ``success``,
    ``terminated``, ``truncated``, ``step_count``, ``total_reward``,
    ``metadata``, ``video_path``) MUST still be present on
    :class:`Episode`, and the ones that were required on day 1 MUST
    still be required (no default added). Optional-since-day-1
    fields (``metadata``, ``video_path``) only need to be present —
    their default semantics are checked separately at runtime.

    The schema is allowed to GROW: new optional metric columns
    (e.g. ``action_variance``, ``failure_score``, ``n_collisions``,
    behavioural metrics, etc.) are intentional polish additions and
    must NOT make this test fail. Do not reintroduce a literal
    full-field-set equality check here — that pattern goes stale on
    every polish PR. Derive expectations from
    :attr:`Episode.model_fields` instead.

    The runtime invariant — ``video_path`` is ``None`` on the
    opt-out path — is a separate contract and is still asserted
    below.
    """
    suite = Suite(
        name="backcompat-suite",
        env="tabletop",
        seed=99,
        episodes_per_cell=2,
        axes={"distractor_count": AxisSpec(values=[0.0])},
    )

    runner = Runner(n_workers=1, env_factory=_no_image_factory)
    episodes = runner.run(policy_factory=_make_random_policy, suite=suite)

    assert len(episodes) == 2
    # Day-1 required fields — promised never to gain a default.
    historical_required: frozenset[str] = frozenset(
        {
            "suite_name",
            "cell_index",
            "episode_index",
            "seed",
            "perturbation_config",
            "success",
            "terminated",
            "truncated",
            "step_count",
            "total_reward",
        }
    )
    # Day-1 optional fields — promised to stay present (default semantics
    # may evolve, but the field name is a public surface).
    historical_optional: frozenset[str] = frozenset({"metadata", "video_path"})

    fields = type(episodes[0]).model_fields
    for name in historical_required:
        assert name in fields, f"day-1 required field {name!r} dropped from Episode"
        assert fields[name].is_required(), (
            f"day-1 required field {name!r} silently gained a default; "
            "back-compat says it MUST remain required"
        )
    for name in historical_optional:
        assert name in fields, f"day-1 optional field {name!r} dropped from Episode"

    for ep in episodes:
        # Runtime contract — opt-out path leaves video_path at the default.
        assert ep.video_path is None


def test_default_runner_episode_json_round_trip_is_lossless() -> None:
    """``Episode.model_dump_json`` round-trips through ``model_validate_json``.

    Pinning the JSON serialization ensures the schema-addition does not
    silently break ``episodes.json`` consumers (replay, ros2 publish).
    """
    suite = Suite(
        name="backcompat-suite-json",
        env="tabletop",
        seed=7,
        episodes_per_cell=1,
        axes={"distractor_count": AxisSpec(values=[0.0])},
    )

    runner = Runner(n_workers=1, env_factory=_no_image_factory)
    episodes = runner.run(policy_factory=_make_random_policy, suite=suite)
    assert len(episodes) == 1

    payload = json.dumps([ep.model_dump(mode="json") for ep in episodes])
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    assert parsed[0]["video_path"] is None  # JSON null is the default

    # Re-validate the JSON dict back into an Episode and assert
    # complete equality with the source.
    reloaded = Episode.model_validate(parsed[0])
    assert reloaded == episodes[0]


def test_runner_constructs_with_no_video_kwargs_and_passes_protocol_check() -> None:
    """Runner() (no video kwargs) constructs and never raises ``ValueError``."""
    Runner(n_workers=1, env_factory=_no_image_factory)
    # Belt-and-braces: the env-factory side of the contract is also
    # uninvoked at construction (no env is built until ``run``).


def test_no_image_env_runs_without_video_opt_in() -> None:
    """An env that does NOT expose ``obs['image']`` runs cleanly under the default.

    This is the hard backwards-compat invariant: the runner MUST NOT
    require ``render_in_obs=True`` for the default opt-out path.
    """
    env = _NoImageFakeEnv()
    assert isinstance(env, GauntletEnv)
    obs, _ = env.reset(seed=0)
    assert "image" not in obs
    env.close()
