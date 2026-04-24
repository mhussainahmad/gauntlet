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

from gauntlet.runner import Episode
from gauntlet.runner.cache import CACHE_SCHEMA_VERSION, EpisodeCache
from gauntlet.suite.schema import AxisSpec, Suite

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

    suite_hash = hashlib.sha256(suite.model_dump_json(round_trip=True).encode("utf-8")).hexdigest()
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
