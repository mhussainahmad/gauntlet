"""File-based Episode cache for incremental rerun acceleration.

See ``docs/polish-exploration-incremental-cache.md`` for the design and
the open-question rationale.

Cache key composition (B-40)::

    key_dict = {
        "suite_name":       suite.name,
        "suite_hash":       compute_suite_provenance_hash(suite),  # 16-char blake2b
        "env_name":         suite.env,
        "policy_id":        <caller-supplied or class-name fallback>,
        "axis_config":      {axis_name: float},
        "seed":             env_seed (per-episode uint32, NOT suite.seed),
        "episodes_per_cell": int,
        "max_steps":        int,
        "schema_version":   "1",
    }
    key = sha256(
        json.dumps(key_dict, sort_keys=True, separators=(",", ":"))
    ).hexdigest()

The ``suite_hash`` field is the B-40 suite-level provenance hash —
a 16-char blake2b digest that bakes in ``gauntlet.__version__`` and
the env asset SHAs. Two suites that differ only in YAML key order or
whitespace cache-hit; a gauntlet version bump or asset edit
correctly invalidates. ANTI-FEATURE: a behaviour-changing release
silently invalidates every cache entry. See
:mod:`gauntlet.runner.provenance` for the design rationale and the
graceful "stale cache, re-run" path keyed off
``SUITE_PROVENANCE_HASH_VERSION``.

Storage layout::

    <cache_dir>/<key[:2]>/<key>.json

The two-character sharding mirrors Git's object store and keeps any
single directory listing under ~256 sibling files even at extreme
scale.

Backwards-compatibility contract:

* The Runner constructs an :class:`EpisodeCache` only when a caller
  passes ``cache_dir is not None``. In every other code path the cache
  module is a pure import — no filesystem access, no work.
* Atomic-rename writes (``os.replace`` on a sibling ``.tmp`` file)
  guarantee no partial JSON ever exists at a key path. A SIGINT mid-
  write leaves the cache in a consistent state.
* Corrupt cache files (``JSONDecodeError`` / ``ValidationError``)
  count as misses; the file is overwritten on the next ``put``.
* B-40 cache-key migration: :meth:`EpisodeCache.make_legacy_key`
  reproduces the pre-B-40 derivation (full-sha256 ``compute_suite_hash``
  in place of the new blake2b hash). The Runner consults it on miss
  for one release so a developer's existing cache directory keeps
  serving hits across the upgrade boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from gauntlet.runner.episode import Episode
from gauntlet.runner.provenance import compute_suite_hash, compute_suite_provenance_hash
from gauntlet.suite.schema import Suite

__all__ = ["CACHE_SCHEMA_VERSION", "EpisodeCache"]


# Bumping invalidates every cache entry. Bump when the Episode schema
# gains / drops fields, when the key composition changes, or when the
# storage format changes. Stays a string so a future date-based scheme
# (e.g. "2026-05-01") is forward-compatible.
CACHE_SCHEMA_VERSION = "1"


@dataclass
class _Stats:
    """In-process hit / miss / put counters for an :class:`EpisodeCache`."""

    hits: int = 0
    misses: int = 0
    puts: int = 0


@dataclass
class EpisodeCache:
    """File-per-Episode cache keyed by content-addressed SHA256.

    Construction does NOT touch the filesystem. The first ``put`` call
    creates the shard directory; ``get`` calls on a non-existent root
    return ``None`` (a miss) without any side effect. This keeps the
    "construct then never use" code path free for the hot test paths
    that monkeypatch the class.

    Attributes:
        root: Cache root directory. Created lazily on the first
            ``put``; ``get`` calls before any ``put`` return ``None``.
        _stats: In-process hit / miss / put counters. Reset to zero on
            construction; advance under ``get`` / ``put`` calls. Read
            via :meth:`stats` for end-of-run reporting.
    """

    root: Path
    _stats: _Stats = field(default_factory=_Stats)

    # ------------------------------------------------------------------
    # Key derivation. Pure: no filesystem, no RNG.
    # ------------------------------------------------------------------

    @staticmethod
    def _compose_key(
        *,
        suite: Suite,
        suite_hash: str,
        axis_config: Mapping[str, float],
        seed: int,
        episodes_per_cell: int,
        max_steps: int,
        env_name: str,
        policy_id: str,
    ) -> str:
        """Compose the canonical-JSON key around a caller-supplied ``suite_hash``.

        Shared between :meth:`make_key` (B-40 hash) and
        :meth:`make_legacy_key` (pre-B-40 sha256). The split lets the
        Runner derive both digests off a single Suite without
        re-walking the env asset tree per cell — call ``make_key``
        once for puts, fall back to ``make_legacy_key`` once on miss.
        """
        # Defensive ``dict(...)`` copy + value coercion to plain floats:
        # numpy scalar floats would otherwise serialise differently across
        # Python versions and break key stability.
        axis_payload = {k: float(v) for k, v in dict(axis_config).items()}
        key_dict = {
            "suite_name": suite.name,
            "suite_hash": suite_hash,
            "env_name": env_name,
            "policy_id": policy_id,
            "axis_config": axis_payload,
            "seed": int(seed),
            "episodes_per_cell": int(episodes_per_cell),
            "max_steps": int(max_steps),
            "schema_version": CACHE_SCHEMA_VERSION,
        }
        canonical = json.dumps(key_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def make_key(
        suite: Suite,
        *,
        axis_config: Mapping[str, float],
        seed: int,
        episodes_per_cell: int,
        max_steps: int,
        env_name: str,
        policy_id: str,
    ) -> str:
        """Derive the SHA256 cache key for one (suite, cell, episode).

        The ``seed`` parameter is the per-episode env seed (the
        ``uint32`` derived via :func:`gauntlet.runner.worker.extract_env_seed`),
        NOT ``suite.seed``. Each cell yields ``episodes_per_cell``
        distinct episodes, each with its own env seed; the cache stores
        one Episode per key.

        B-40: the ``suite_hash`` component is now
        :func:`gauntlet.runner.provenance.compute_suite_provenance_hash`
        — a 16-char blake2b digest that bakes in
        ``gauntlet.__version__`` and the env asset SHAs. Two suites
        with identical semantics but reordered YAML keys hash to the
        same suite_hash and therefore the same cache key; a gauntlet
        version bump invalidates by design. The pre-B-40 derivation
        is preserved in :meth:`make_legacy_key` for one-release
        back-compat (the Runner consults it on miss).

        Args:
            suite: The originating :class:`Suite`. The B-40 provenance
                hash is computed from this — silent suite edits, env
                asset changes, and gauntlet version bumps all
                invalidate the cache.
            axis_config: ``{axis_name: float}`` for this cell. Defensively
                copied before serialisation.
            seed: Per-episode env seed (uint32).
            episodes_per_cell: Echo of :attr:`Suite.episodes_per_cell`.
                Included independently so a suite edit that bumps it
                invalidates even cells whose ``axis_config`` is unchanged.
            max_steps: Per-episode env step cap. Caller-supplied
                because every backend bakes ``max_steps`` into its
                constructor and :class:`GauntletEnv` does not expose a
                public getter.
            env_name: Echo of :attr:`Suite.env`.
            policy_id: Caller-supplied policy identifier. Defaults at
                the Runner level to ``policy_factory().__class__.__name__``.

        Returns:
            64-char lowercase hex SHA256 digest of the canonical-JSON
            key. Stable across Python invocations.
        """
        suite_hash = compute_suite_provenance_hash(suite)
        return EpisodeCache._compose_key(
            suite=suite,
            suite_hash=suite_hash,
            axis_config=axis_config,
            seed=seed,
            episodes_per_cell=episodes_per_cell,
            max_steps=max_steps,
            env_name=env_name,
            policy_id=policy_id,
        )

    @staticmethod
    def make_legacy_key(
        suite: Suite,
        *,
        axis_config: Mapping[str, float],
        seed: int,
        episodes_per_cell: int,
        max_steps: int,
        env_name: str,
        policy_id: str,
    ) -> str:
        """Pre-B-40 cache key derivation, kept for one-release back-compat.

        Reproduces the historical key composition that used
        :func:`gauntlet.runner.provenance.compute_suite_hash` (full
        sha256 over the Suite payload, no gauntlet version, no env
        asset SHAs). Called on cache *miss* by the Runner so an
        existing on-disk cache directory written before the B-40
        upgrade keeps serving hits across the boundary.

        DO NOT call ``put`` under a legacy key — writes go through
        :meth:`make_key` so a future read finds the new digest. The
        legacy entries age out organically as suites change; the
        next major release can drop this method entirely and treat
        any pre-existing legacy entries as a clean miss + re-roll.
        """
        suite_hash = compute_suite_hash(suite)
        return EpisodeCache._compose_key(
            suite=suite,
            suite_hash=suite_hash,
            axis_config=axis_config,
            seed=seed,
            episodes_per_cell=episodes_per_cell,
            max_steps=max_steps,
            env_name=env_name,
            policy_id=policy_id,
        )

    # ------------------------------------------------------------------
    # Get / put — the hot path.
    # ------------------------------------------------------------------

    def _path_for(self, key: str) -> Path:
        """Map a key to the on-disk shard path.

        Two-char sharding (Git-style) keeps directory listings under
        ~256 entries even at million-key scale.
        """
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Episode | None:
        """Return the cached Episode for ``key``, or ``None`` on miss.

        A ``JSONDecodeError`` or pydantic ``ValidationError`` is treated
        as a miss; the corrupt file is left in place (a future
        ``gauntlet cache verify`` could sweep it) and overwritten on
        the next :meth:`put`.

        ``hits`` is incremented only on a successful hit; ``misses`` is
        incremented on every cache lookup that doesn't resolve.
        """
        path = self._path_for(key)
        if not path.is_file():
            self._stats.misses += 1
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            episode = Episode.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError):
            # Corrupt or schema-mismatched cache entry — treat as a miss.
            # Do not delete: a future "cache verify" tool can clean up,
            # and a re-roll will overwrite via atomic-rename put().
            self._stats.misses += 1
            return None
        self._stats.hits += 1
        return episode

    def has(self, key: str) -> bool:
        """Return whether a cache entry exists at *key*.

        Side-effect-free: does NOT touch the hit / miss counters.
        Used by the Runner's B-40 back-compat path to check whether a
        legacy-keyed entry is worth reading without double-counting
        the lookup against the new-key miss the caller already
        recorded.
        """
        return self._path_for(key).is_file()

    def get_legacy(self, key: str) -> Episode | None:
        """Read a pre-B-40 cache entry without touching the miss counter.

        The Runner's back-compat path probes :meth:`has` first to
        confirm the legacy file exists, then calls this. A
        ``ValidationError`` / ``JSONDecodeError`` here is silently
        swallowed — corrupt legacy entries fall through to the
        normal "re-roll and overwrite under the new key" flow. The
        ``hits`` counter is incremented on a successful read so the
        end-of-run report reflects "an uncached work item was
        served from disk", consistent with the new-key path.
        """
        path = self._path_for(key)
        if not path.is_file():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            episode = Episode.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError):
            return None
        self._stats.hits += 1
        # Re-classify the new-key miss the caller already recorded:
        # the public counter promises "uncached work items", not
        # "filesystem lookups", and a successful legacy hit means
        # this work item was in fact cached.
        if self._stats.misses > 0:
            self._stats.misses -= 1
        return episode

    def put(self, key: str, episode: Episode) -> None:
        """Atomically write ``episode`` to the cache under ``key``.

        Writes to a sibling ``.tmp`` path then ``os.replace``-renames
        to the final shard path. ``os.replace`` is atomic on POSIX and
        on Windows >= Server 2003 — a SIGINT mid-write leaves either
        the previous state or the new state, never a partial JSON.

        The ``puts`` counter is incremented on every successful write.
        """
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise via Pydantic JSON mode so non-finite floats round-
        # trip through the Episode's ``ser_json_inf_nan="strings"``
        # config (matches gauntlet hotfix #18).
        payload = episode.model_dump_json()
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
        self._stats.puts += 1

    # ------------------------------------------------------------------
    # Stats — read-only end-of-run snapshot.
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return a copy of the in-process hit / miss / put counters.

        Returned as a plain ``dict`` so callers can mutate / serialise
        without affecting the live counters.
        """
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "puts": self._stats.puts,
        }
