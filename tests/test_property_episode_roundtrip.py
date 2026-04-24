"""Property-based tests for :class:`Episode` JSON round-trip identity.

Phase 2.5 Task 13 — covers the contract that
``Episode.model_validate_json(Episode.model_dump_json())`` is the
identity function over hypothesis-generated valid Episodes. The CLI
(``gauntlet run`` writes ``episodes.json``; ``gauntlet replay`` reads
the same file) and the Phase 2 trajectory tooling both depend on this
property.

Scope: finite floats only. Pydantic 2's JSON codec serialises ``NaN``
and ``inf`` to JSON ``null``, but the strict validator on
``model_validate_json`` rejects ``null`` for float fields — so NaN/inf
episodes are not bit-reproducible across the JSON boundary. That is a
genuine asymmetry in the schema; documented in the PR body as a
follow-up. Real Runner-produced Episodes never carry NaN/inf, so this
property covers the production contract today.

Hypothesis budget: ``max_examples=50`` per test, well under 1s wall-time
because no env or runner is involved.
"""

from __future__ import annotations

from datetime import timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from gauntlet.runner.episode import Episode

# Bound floats to a realistic range and forbid NaN/inf — the production
# Runner emits float64 reward / config in this envelope. See module
# docstring for the NaN/inf carve-out.
_FINITE = st.floats(
    min_value=-1e9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)
# Axis names match the canonical registry naming scheme; we don't tie
# to the registry directly so the round-trip property covers any string
# the schema would accept.
_AXIS_KEY = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20)


@st.composite
def _episode_strategy(draw: st.DrawFn) -> Episode:
    """Generate a structurally-valid :class:`Episode` for round-trip tests."""
    n_axes = draw(st.integers(min_value=0, max_value=4))
    config_keys = draw(
        st.lists(_AXIS_KEY, min_size=n_axes, max_size=n_axes, unique=True),
    )
    config_vals = draw(st.lists(_FINITE, min_size=n_axes, max_size=n_axes))
    perturbation_config = dict(zip(config_keys, config_vals, strict=True))

    # Metadata permits float / int / str / bool. We exercise all four.
    n_meta = draw(st.integers(min_value=0, max_value=3))
    meta_keys = draw(
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
            min_size=n_meta,
            max_size=n_meta,
            unique=True,
        )
    )
    meta_vals = draw(
        st.lists(
            st.one_of(
                _FINITE,
                st.integers(min_value=-(2**62), max_value=2**62),
                st.text(max_size=20),
                st.booleans(),
            ),
            min_size=n_meta,
            max_size=n_meta,
        )
    )
    metadata: dict[str, float | int | str | bool] = dict(zip(meta_keys, meta_vals, strict=True))

    return Episode(
        suite_name=draw(st.text(min_size=1, max_size=30)),
        cell_index=draw(st.integers(min_value=0, max_value=10_000)),
        episode_index=draw(st.integers(min_value=0, max_value=10_000)),
        seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
        perturbation_config=perturbation_config,
        success=draw(st.booleans()),
        terminated=draw(st.booleans()),
        truncated=draw(st.booleans()),
        step_count=draw(st.integers(min_value=0, max_value=10_000)),
        total_reward=draw(_FINITE),
        metadata=metadata,
    )


@given(episode=_episode_strategy())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_episode_json_round_trip_is_identity(episode: Episode) -> None:
    """``Episode.model_validate_json(Episode.model_dump_json())`` returns
    a model that compares equal to the original."""
    blob = episode.model_dump_json()
    restored = Episode.model_validate_json(blob)
    assert restored == episode


@given(episode=_episode_strategy())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_episode_dict_round_trip_is_identity(episode: Episode) -> None:
    """The Python-dict round-trip is a separate code path inside pydantic
    (``model_dump`` -> ``model_validate``); also exercised because the
    CLI's HTML-report writer goes through dicts on the way to Jinja."""
    payload = episode.model_dump()
    restored = Episode.model_validate(payload)
    assert restored == episode


@given(
    episodes=st.lists(_episode_strategy(), min_size=1, max_size=8),
)
@settings(max_examples=30, deadline=timedelta(seconds=2))
def test_episode_list_json_array_round_trip_is_identity(
    episodes: list[Episode],
) -> None:
    """The ``episodes.json`` artefact is a JSON array of Episode payloads;
    list round-trip mirrors that on-disk format."""
    import json

    payload = [ep.model_dump(mode="json") for ep in episodes]
    blob = json.dumps(payload)
    restored = [Episode.model_validate(item) for item in json.loads(blob)]
    assert restored == episodes
