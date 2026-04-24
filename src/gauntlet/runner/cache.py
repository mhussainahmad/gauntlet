"""File-based Episode cache for incremental rerun acceleration.

See ``docs/polish-exploration-incremental-cache.md`` for the design and
the open-question rationale.

Cache key composition::

    key_dict = {
        "suite_name":       suite.name,
        "suite_hash":       sha256(Suite.model_dump_json(round_trip=True)),
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

        Args:
            suite: The originating :class:`Suite`. ``model_dump_json``
                is hashed to derive the ``suite_hash`` field of the
                key — silent suite edits invalidate the cache.
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
        suite_json = suite.model_dump_json(round_trip=True)
        suite_hash = hashlib.sha256(suite_json.encode("utf-8")).hexdigest()
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
