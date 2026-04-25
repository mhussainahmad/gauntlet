"""Tests for :class:`gauntlet.runner.cache.EpisodeCache` and Runner integration.

See ``docs/polish-exploration-incremental-cache.md`` for the design.
The cache is opt-in: ``Runner(cache_dir=None)`` (the default) MUST be
byte-identical to pre-PR behaviour. The Runner-level integration tests
live in this same module so the no-cache regression test sits next to
the construction-on-demand assertion.

All factory helpers are module-level so they pickle cleanly under the
``spawn`` start method (the multi-worker tests do not need it, but the
shared helpers stay multi-worker-safe by default).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from gauntlet.policy.scripted import ScriptedPolicy
from gauntlet.runner import Episode, Runner
from gauntlet.runner.cache import CACHE_SCHEMA_VERSION, EpisodeCache
from gauntlet.suite.schema import AxisSpec, Suite

# ----------------------------------------------------------------------------
# Module-level Runner-test factories — picklable for spawn-based pools.
# ----------------------------------------------------------------------------


def _make_scripted_policy() -> ScriptedPolicy:
    """Pickle-friendly ScriptedPolicy factory."""
    return ScriptedPolicy()


def _make_fast_env() -> Any:
    """Tabletop env with a small max_steps so the test is sub-second."""
    from gauntlet.env.tabletop import TabletopEnv

    return TabletopEnv(max_steps=20)


# Match _make_fast_env above; threaded into the Runner cache key.
_TEST_MAX_STEPS = 20

# ----------------------------------------------------------------------------
# Suite builder shared across cache-key + Runner-integration tests.
# ----------------------------------------------------------------------------


def _make_suite(
    *,
    name: str = "cache-test-suite",
    seed: int | None = 42,
    episodes_per_cell: int = 1,
    axes: dict[str, AxisSpec] | None = None,
) -> Suite:
    if axes is None:
        # Single-axis 2-step grid -> 2 cells x episodes_per_cell.
        axes = {"lighting_intensity": AxisSpec(low=0.5, high=1.0, steps=2)}
    return Suite(
        name=name,
        env="tabletop",
        seed=seed,
        episodes_per_cell=episodes_per_cell,
        axes=axes,
    )


# ----------------------------------------------------------------------------
# 1. Key stability — same inputs -> same hex digest, deterministically.
# ----------------------------------------------------------------------------


def test_make_key_is_stable_across_calls() -> None:
    suite = _make_suite()
    k1 = EpisodeCache.make_key(
        suite,
        axis_config={"lighting_intensity": 0.5},
        seed=123,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="ScriptedPolicy",
    )
    k2 = EpisodeCache.make_key(
        suite,
        axis_config={"lighting_intensity": 0.5},
        seed=123,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="ScriptedPolicy",
    )
    assert k1 == k2
    assert len(k1) == 64
    assert all(c in "0123456789abcdef" for c in k1)


def test_make_key_axis_config_order_independent() -> None:
    """Iteration order of axis_config dict must NOT affect the key."""
    suite = _make_suite(
        axes={
            "lighting_intensity": AxisSpec(low=0.5, high=1.0, steps=1),
            "camera_offset_x": AxisSpec(low=-0.02, high=0.02, steps=1),
        }
    )
    k_a = EpisodeCache.make_key(
        suite,
        axis_config={"lighting_intensity": 0.75, "camera_offset_x": 0.0},
        seed=1,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="P",
    )
    k_b = EpisodeCache.make_key(
        suite,
        # Reversed insertion order — should still hash to the same value.
        axis_config={"camera_offset_x": 0.0, "lighting_intensity": 0.75},
        seed=1,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="P",
    )
    assert k_a == k_b


# ----------------------------------------------------------------------------
# 2. Key sensitivity — one bit different in any field -> different hex.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,changed",
    [
        ("axis_config", {"lighting_intensity": 0.501}),  # axis value bump
        ("seed", 124),
        ("episodes_per_cell", 2),
        ("max_steps", 21),
        ("env_name", "tabletop-pybullet"),
        ("policy_id", "RandomPolicy"),
    ],
)
def test_make_key_changes_when_any_field_changes(field: str, changed: Any) -> None:
    suite = _make_suite()
    base_kwargs: dict[str, Any] = {
        "axis_config": {"lighting_intensity": 0.5},
        "seed": 123,
        "episodes_per_cell": 1,
        "max_steps": 20,
        "env_name": "tabletop",
        "policy_id": "ScriptedPolicy",
    }
    base = EpisodeCache.make_key(suite, **base_kwargs)
    bumped = {**base_kwargs, field: changed}
    after = EpisodeCache.make_key(suite, **bumped)
    assert base != after, f"changing {field!r} did not change the cache key"


def test_make_key_changes_when_suite_payload_changes() -> None:
    """A different Suite (different name) -> different key."""
    suite_a = _make_suite(name="suite-a")
    suite_b = _make_suite(name="suite-b")
    kwargs: dict[str, Any] = {
        "axis_config": {"lighting_intensity": 0.5},
        "seed": 123,
        "episodes_per_cell": 1,
        "max_steps": 20,
        "env_name": "tabletop",
        "policy_id": "P",
    }
    assert EpisodeCache.make_key(suite_a, **kwargs) != EpisodeCache.make_key(suite_b, **kwargs)


def test_make_key_changes_when_suite_axis_payload_changes() -> None:
    """Editing the Suite's axes (silent suite tweak) MUST invalidate the key."""
    suite_a = _make_suite()
    # Bump steps from 2 to 3 — same name, same env, same other fields, but
    # the Suite JSON differs and so must the suite_hash.
    suite_b = _make_suite(
        axes={
            "lighting_intensity": AxisSpec(low=0.5, high=1.0, steps=3),
        }
    )
    kwargs: dict[str, Any] = {
        "axis_config": {"lighting_intensity": 0.5},
        "seed": 123,
        "episodes_per_cell": 1,
        "max_steps": 20,
        "env_name": "tabletop",
        "policy_id": "P",
    }
    assert EpisodeCache.make_key(suite_a, **kwargs) != EpisodeCache.make_key(suite_b, **kwargs)


def test_make_key_includes_schema_version() -> None:
    """The key MUST embed CACHE_SCHEMA_VERSION so a bump invalidates everything."""
    suite = _make_suite()
    # Recompute the canonical JSON the way EpisodeCache.make_key does and
    # look for the schema_version field in the canonical payload.
    import hashlib

    from gauntlet.runner.provenance import compute_suite_provenance_hash

    # B-40: ``suite_hash`` is now the 16-char blake2b suite-provenance
    # hash (not the legacy full-sha256). The canonical JSON below
    # mirrors ``EpisodeCache._compose_key`` byte-for-byte.
    suite_hash = compute_suite_provenance_hash(suite)
    canonical = {
        "suite_name": suite.name,
        "suite_hash": suite_hash,
        "env_name": "tabletop",
        "policy_id": "P",
        "axis_config": {"lighting_intensity": 0.5},
        "seed": 1,
        "episodes_per_cell": 1,
        "max_steps": 20,
        "schema_version": CACHE_SCHEMA_VERSION,
    }
    expected = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    actual = EpisodeCache.make_key(
        suite,
        axis_config={"lighting_intensity": 0.5},
        seed=1,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="P",
    )
    assert expected == actual


# ----------------------------------------------------------------------------
# 3. Storage layout — git-style 2-char sharding under <root>/<key[:2]>/.
# ----------------------------------------------------------------------------


def _sample_episode(suite_name: str = "cache-test-suite") -> Episode:
    return Episode(
        suite_name=suite_name,
        cell_index=0,
        episode_index=0,
        seed=12345,
        perturbation_config={"lighting_intensity": 0.5},
        success=True,
        terminated=True,
        truncated=False,
        step_count=10,
        total_reward=1.5,
        metadata={"master_seed": 42},
        video_path=None,
    )


def test_put_writes_sharded_path(tmp_path: Path) -> None:
    cache = EpisodeCache(root=tmp_path)
    key = "abcd" + "0" * 60
    cache.put(key, _sample_episode())
    expected = tmp_path / "ab" / f"{key}.json"
    assert expected.is_file(), f"expected sharded write at {expected}"


def test_get_returns_none_when_root_missing(tmp_path: Path) -> None:
    """Construction must not touch the FS; a get on an empty root is a miss."""
    cache_root = tmp_path / "does-not-exist"
    cache = EpisodeCache(root=cache_root)
    # Construction did not create the root.
    assert not cache_root.exists()
    assert cache.get("a" * 64) is None
    assert cache.stats() == {"hits": 0, "misses": 1, "puts": 0}


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = EpisodeCache(root=tmp_path)
    ep = _sample_episode()
    key = "f" * 64
    cache.put(key, ep)
    fetched = cache.get(key)
    assert fetched is not None
    assert fetched.model_dump() == ep.model_dump()
    assert cache.stats() == {"hits": 1, "misses": 0, "puts": 1}


def test_put_overwrites_existing_entry(tmp_path: Path) -> None:
    """A second put with the same key MUST overwrite (cache is content-addressed)."""
    cache = EpisodeCache(root=tmp_path)
    key = "1" * 64
    cache.put(key, _sample_episode())
    overwrite = _sample_episode().model_copy(update={"step_count": 99})
    cache.put(key, overwrite)
    fetched = cache.get(key)
    assert fetched is not None
    assert fetched.step_count == 99


def test_get_treats_corrupt_json_as_miss(tmp_path: Path) -> None:
    cache = EpisodeCache(root=tmp_path)
    key = "9" * 64
    # Hand-write a corrupt file at the expected path.
    bad_path = tmp_path / "99" / f"{key}.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("not-json{{{", encoding="utf-8")
    assert cache.get(key) is None
    assert cache.stats()["misses"] == 1


def test_get_treats_schema_mismatch_as_miss(tmp_path: Path) -> None:
    cache = EpisodeCache(root=tmp_path)
    key = "8" * 64
    bad_path = tmp_path / "88" / f"{key}.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    # Valid JSON but missing required Episode fields.
    bad_path.write_text(json.dumps({"some_other_schema": True}), encoding="utf-8")
    assert cache.get(key) is None
    assert cache.stats()["misses"] == 1


def test_put_uses_atomic_rename(tmp_path: Path) -> None:
    """A put leaves no .tmp sibling once it completes."""
    cache = EpisodeCache(root=tmp_path)
    key = "7" * 64
    cache.put(key, _sample_episode())
    shard_dir = tmp_path / "77"
    children = sorted(p.name for p in shard_dir.iterdir())
    # Exactly one file: the final json, no leftover .tmp.
    assert children == [f"{key}.json"]


def test_episode_round_trip_preserves_non_finite_floats(tmp_path: Path) -> None:
    """Episode hotfix #18: non-finite floats must survive cache round-trip.

    The cache stores Episodes via Pydantic's ``ser_json_inf_nan="strings"``
    serialiser; a NaN reward must come back as NaN (not a load failure).
    """
    cache = EpisodeCache(root=tmp_path)
    ep = _sample_episode().model_copy(update={"total_reward": float("nan")})
    key = "5" * 64
    cache.put(key, ep)
    fetched = cache.get(key)
    assert fetched is not None
    assert math.isnan(fetched.total_reward)


# ----------------------------------------------------------------------------
# 4. Runner integration — the load-bearing tests.
# ----------------------------------------------------------------------------


def test_no_cache_default_byte_identical(tmp_path: Path) -> None:
    """Runner(cache_dir=None) MUST produce byte-identical Episodes to today.

    Two back-to-back fixed-seed runs with no cache_dir set; every Pydantic
    field on every Episode must match. This pins the no-cache opt-out path.
    """
    suite = _make_suite(seed=999, episodes_per_cell=1)
    runner = Runner(n_workers=1, env_factory=_make_fast_env)
    a = runner.run(policy_factory=_make_scripted_policy, suite=suite)
    b = runner.run(policy_factory=_make_scripted_policy, suite=suite)
    assert len(a) == len(b)
    for ea, eb in zip(a, b, strict=True):
        assert ea.model_dump() == eb.model_dump()


def test_runner_no_cache_does_not_construct_episodecache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When cache_dir=None the Runner must NEVER construct EpisodeCache.

    Monkeypatch the constructor to raise; if the assertion fires the test
    fails. This pins the "byte-identical hot path" promise stronger than
    behaviour comparison alone.
    """
    constructed: list[bool] = []

    def _exploding_init(self: EpisodeCache, *args: Any, **kwargs: Any) -> None:
        constructed.append(True)
        raise AssertionError("EpisodeCache must NOT be constructed when cache_dir=None")

    monkeypatch.setattr(EpisodeCache, "__init__", _exploding_init)

    suite = _make_suite(seed=11, episodes_per_cell=1)
    runner = Runner(n_workers=1, env_factory=_make_fast_env)  # cache_dir defaults to None
    episodes = runner.run(policy_factory=_make_scripted_policy, suite=suite)
    assert len(episodes) == suite.num_cells()
    assert constructed == []  # double-check: no construction recorded


def test_runner_cache_hit_on_second_run(tmp_path: Path) -> None:
    """The second invocation against the same cache_dir must be 100% hits.

    Two cells x 1 episode = 2 episodes. First run: 0 hits, 2 misses, 2 puts.
    Second run: 2 hits, 0 misses, 0 puts. Episode payloads must match the
    no-cache path bit-for-bit.
    """
    suite = _make_suite(seed=2024, episodes_per_cell=1)
    cache_dir = tmp_path / "cache"

    runner_a = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        cache_dir=cache_dir,
        max_steps=_TEST_MAX_STEPS,
        policy_id="ScriptedPolicy-fixture",
    )
    first = runner_a.run(policy_factory=_make_scripted_policy, suite=suite)
    stats_first = runner_a.cache_stats()
    assert stats_first == {"hits": 0, "misses": len(first), "puts": len(first)}

    runner_b = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        cache_dir=cache_dir,
        max_steps=_TEST_MAX_STEPS,
        policy_id="ScriptedPolicy-fixture",
    )
    second = runner_b.run(policy_factory=_make_scripted_policy, suite=suite)
    stats_second = runner_b.cache_stats()
    assert stats_second == {"hits": len(second), "misses": 0, "puts": 0}

    assert len(first) == len(second)
    for ea, eb in zip(first, second, strict=True):
        assert ea.model_dump() == eb.model_dump()


def test_runner_cache_matches_no_cache_path(tmp_path: Path) -> None:
    """Cached Episode list must be bit-equal to the no-cache Episode list."""
    suite = _make_suite(seed=314, episodes_per_cell=1)
    cache_dir = tmp_path / "cache"

    no_cache = Runner(n_workers=1, env_factory=_make_fast_env).run(
        policy_factory=_make_scripted_policy, suite=suite
    )
    cached_runner = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        cache_dir=cache_dir,
        max_steps=_TEST_MAX_STEPS,
        policy_id="ScriptedPolicy-fixture",
    )
    cached = cached_runner.run(policy_factory=_make_scripted_policy, suite=suite)

    assert len(no_cache) == len(cached)
    for ea, eb in zip(no_cache, cached, strict=True):
        assert ea.model_dump() == eb.model_dump()


def test_runner_cache_invalidates_on_suite_edit(tmp_path: Path) -> None:
    """A Suite edit (different axes) must produce a clean miss on the next run."""
    cache_dir = tmp_path / "cache"

    suite_a = _make_suite(seed=1, episodes_per_cell=1)
    runner_a = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        cache_dir=cache_dir,
        max_steps=_TEST_MAX_STEPS,
        policy_id="ScriptedPolicy-fixture",
    )
    runner_a.run(policy_factory=_make_scripted_policy, suite=suite_a)
    assert runner_a.cache_stats()["misses"] == suite_a.num_cells()

    # Edit the suite: bump steps from 2 to 3 -> different suite_hash.
    suite_b = _make_suite(
        seed=1,
        episodes_per_cell=1,
        axes={"lighting_intensity": AxisSpec(low=0.5, high=1.0, steps=3)},
    )
    runner_b = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        cache_dir=cache_dir,
        max_steps=_TEST_MAX_STEPS,
        policy_id="ScriptedPolicy-fixture",
    )
    runner_b.run(policy_factory=_make_scripted_policy, suite=suite_b)
    # Every cell of suite_b must miss because suite_hash differs.
    stats = runner_b.cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == suite_b.num_cells()


def test_runner_cache_invalidates_on_policy_id_change(tmp_path: Path) -> None:
    """Different policy_id MUST produce a clean miss against the same cache."""
    cache_dir = tmp_path / "cache"
    suite = _make_suite(seed=7, episodes_per_cell=1)

    runner_a = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        cache_dir=cache_dir,
        max_steps=_TEST_MAX_STEPS,
        policy_id="policy-v1",
    )
    runner_a.run(policy_factory=_make_scripted_policy, suite=suite)

    runner_b = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        cache_dir=cache_dir,
        max_steps=_TEST_MAX_STEPS,
        policy_id="policy-v2",
    )
    runner_b.run(policy_factory=_make_scripted_policy, suite=suite)
    stats = runner_b.cache_stats()
    # policy-v2 has not been seen before -> all misses.
    assert stats["hits"] == 0
    assert stats["misses"] == suite.num_cells()


def test_runner_requires_max_steps_with_cache_dir(tmp_path: Path) -> None:
    """Setting cache_dir without max_steps must error (key would be incomplete)."""
    with pytest.raises(ValueError, match="max_steps"):
        Runner(
            n_workers=1,
            env_factory=_make_fast_env,
            cache_dir=tmp_path / "cache",
            # max_steps deliberately omitted
        )


def test_runner_policy_id_defaults_to_class_name(tmp_path: Path) -> None:
    """When policy_id is omitted, the Runner must derive it from the policy class."""
    cache_dir = tmp_path / "cache"
    suite = _make_suite(seed=8, episodes_per_cell=1)

    # First run with implicit policy_id.
    runner_a = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        cache_dir=cache_dir,
        max_steps=_TEST_MAX_STEPS,
    )
    runner_a.run(policy_factory=_make_scripted_policy, suite=suite)

    # Second run with the SAME implicit policy_id (same class) -> 100% hits.
    runner_b = Runner(
        n_workers=1,
        env_factory=_make_fast_env,
        cache_dir=cache_dir,
        max_steps=_TEST_MAX_STEPS,
    )
    runner_b.run(policy_factory=_make_scripted_policy, suite=suite)
    assert runner_b.cache_stats()["hits"] == suite.num_cells()


def test_cache_stats_zero_when_no_cache_dir() -> None:
    """cache_stats() returns zeros without a cache so wrappers can call it freely."""
    runner = Runner(n_workers=1, env_factory=_make_fast_env)
    assert runner.cache_stats() == {"hits": 0, "misses": 0, "puts": 0}


# ----------------------------------------------------------------------------
# 5. Regression pin — Episode shape on the no-cache path matches the
# pre-PR schema, field-for-field. Catches any accidental Episode-schema
# drift introduced by the cache integration even before the byte-equality
# check above can fire.
# ----------------------------------------------------------------------------


def test_no_cache_episode_shape_pinned() -> None:
    """The no-cache path must keep the day-1 Episode contract intact.

    Back-compat-only intent: the pre-cache field set
    (``suite_name``, ``cell_index``, ``episode_index``, ``seed``,
    ``perturbation_config``, ``success``, ``terminated``,
    ``truncated``, ``step_count``, ``total_reward``, ``metadata``,
    ``video_path``) MUST still exist on :class:`Episode`, and the
    fields that were required on day 1 MUST still be required (no
    default added). Optional-since-day-1 fields are checked for
    presence only.

    The schema is allowed to GROW: subsequent polish work added
    optional metric columns (B-21 actuator energy, B-18
    ``action_variance``, B-30 safety violations, B-04
    ``failure_score`` / ``failure_alarm``, B-02 behavioural metrics,
    B-22 provenance trio, B-28 ``source``, ...). Those additions are
    intentional and must NOT fail this test. Do not reintroduce a
    literal full-field-set equality check here — that pattern goes
    stale on every polish PR. Derive expectations from
    :attr:`Episode.model_fields` instead, and bump
    ``CACHE_SCHEMA_VERSION`` in :mod:`gauntlet.runner.cache` only
    when a CACHED Episode payload would deserialise into the WRONG
    fields under the new schema (i.e. an actually-breaking change).

    Runtime invariants — ``video_path is None`` on the no-cache,
    no-video path, and the canonical no-opt-in ``metadata`` echo —
    are separate contracts and are still asserted below.
    """
    suite = _make_suite(seed=12345, episodes_per_cell=1)
    runner = Runner(n_workers=1, env_factory=_make_fast_env)
    episodes = runner.run(policy_factory=_make_scripted_policy, suite=suite)
    assert len(episodes) == suite.num_cells()

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
    # Day-1 optional fields — promised to stay present.
    historical_optional: frozenset[str] = frozenset({"metadata", "video_path"})

    fields = Episode.model_fields
    for name in historical_required:
        assert name in fields, (
            f"day-1 required field {name!r} dropped from Episode; if you "
            "intentionally renamed it, bump CACHE_SCHEMA_VERSION in "
            "src/gauntlet/runner/cache.py to invalidate older cache entries."
        )
        assert fields[name].is_required(), (
            f"day-1 required field {name!r} silently gained a default; "
            "back-compat says it MUST remain required."
        )
    for name in historical_optional:
        assert name in fields, f"day-1 optional field {name!r} dropped from Episode"

    for ep in episodes:
        dumped = ep.model_dump()
        # video_path stays None on the no-cache, no-video path.
        assert dumped["video_path"] is None
        # master_seed echo is the only metadata key the Runner writes
        # without an explicit user opt-in.
        assert dumped["metadata"] == {
            "master_seed": 12345,
            "n_cells": suite.num_cells(),
            "episodes_per_cell": 1,
        }
