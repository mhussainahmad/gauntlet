# Polish exploration: incremental rollout caching

Status: exploration / pre-implementation
Owner: polish/incremental-cache branch

## 1. Why this matters (the domain win)

Iterative policy development repeatedly re-runs the same
`(suite, axis_config, seed)` cells. Today every `gauntlet run`
invocation re-rolls every cell from scratch. For the MuJoCo Tabletop
backend (~1.5 ms / step, ~30 ms / episode) a 5-axis x 5-step x
10-episode grid (1,250 episodes) finishes in well under a minute and
caching is a "nice to have".

For VLA policies (`evaluate_openvla.py`, `evaluate_smolvla.py`) the
math inverts hard:

* PyTorch model load: 5-15 s (one-time per worker; already amortised).
* Per-rollout time on CPU MuJoCo at 200 steps: 30+ s once you add
  policy inference latency.
* 1,250-episode reference sweep: ~10 hours.

After a single tweak — switching the suite seed, adding one axis value,
or rerunning to confirm no regression — the user re-pays the entire
10 hours. Caching `(suite_name, suite_hash, env_name, policy_id,
axis_config, seed)` -> `Episode` collapses the second invocation to
disk-IO time. **A 10-hour rerun becomes a sub-second cache hit.**

This is the single biggest "iterative dev superpower" upgrade the
harness can ship in Phase 2 polish without changing any other
contract.

## 2. Public API decision

`Runner.__init__` gains three optional kwargs. Defaults preserve
byte-identical pre-PR behaviour.

```python
Runner(
    n_workers=1,
    env_factory=...,
    trajectory_dir=None,
    record_video=False,
    # NEW (this PR):
    cache_dir=None,        # path or None — no cache when None
    policy_id=None,        # caller-supplied ID; defaults to policy class name
    max_steps=None,        # opt-in cache-key input; required iff cache_dir set
)
```

* `cache_dir=None` is the byte-identical opt-out. The Runner does not
  construct an `EpisodeCache`, does not call `cache.get`, does not
  call `cache.put`. The hot path is unchanged.
* `policy_id` is the caller's responsibility. We default it to
  `policy_factory().__class__.__name__` when `cache_dir` is set so
  most users get a sensible default; users who swap weights inside
  the same policy class (e.g. SmolVLA checkpoint A vs B) MUST pass
  an explicit `policy_id` or the cache will silently return stale
  rollouts. This is documented in the docstring with a worked
  example.
* `max_steps` is required when `cache_dir` is set because every
  backend bakes `max_steps` into its constructor (e.g.
  `partial(TabletopEnv, max_steps=20)`) and `GauntletEnv` does not
  expose a public getter. Constructing the env to introspect it
  defeats the caching purpose for the VLA case (env construction is
  not free either). Requiring an explicit value keeps the cache key
  honest. The CLI threads this from the existing `--env-max-steps`
  flag.

`Runner.run` signature is unchanged — the new kwargs configure the
*how* of execution, mirroring the existing `n_workers` /
`env_factory` / `trajectory_dir` shape.

## 3. Cache key design

```python
key_dict = {
    "suite_name": suite.name,
    "suite_hash": sha256(Suite.model_dump_json(round_trip=True)).hexdigest(),
    "env_name":   suite.env,
    "policy_id":  "<resolved by Runner>",
    "axis_config": dict(cell.values),     # {axis_name: float}
    "seed":       env_seed,               # uint32 from extract_env_seed
    "episodes_per_cell": suite.episodes_per_cell,
    "max_steps":  max_steps,
    "schema_version": "1",
}
key = sha256(
    json.dumps(key_dict, sort_keys=True, separators=(",", ":"))
).hexdigest()
```

Notes:

* **`seed` is the per-episode `env_seed`**, not `suite.seed`. Each
  cell yields `episodes_per_cell` distinct Episodes, each with its
  own `env_seed` derived from the spawn tree. The cache stores one
  Episode per key, so two episodes within the same cell must collide
  on every key field except `seed` and end up at distinct paths.
* **`suite_hash` includes the full Suite JSON** (not just `.name`)
  so silent suite edits — adding an axis value, changing
  `episodes_per_cell`, swapping `sampling: cartesian` for
  `latin_hypercube` — invalidate the cache. Pydantic v2 preserves
  field declaration order in `model_dump_json`, so the hash is
  stable across Python invocations.
* **`policy_id` is caller-supplied** because the harness has no way
  to fingerprint a torch checkpoint without forcing every user to
  load weights. Default-to-class-name covers the random / scripted /
  built-in case; VLA users pass the checkpoint hash or git SHA.
* **`schema_version: "1"`** so a future cache-format change (e.g. a
  new Episode field) can invalidate every entry without scanning the
  filesystem. Bump and the next run is a clean miss.
* **`json.dumps(..., sort_keys=True, separators=(",", ":"))`** is the
  canonical-form: deterministic ordering, no whitespace drift, no
  trailing newline. Stable across Python versions and OS.
* **`extra="forbid"` on Episode** means any field added to Episode
  in a future PR will fail to load from a cache written by an older
  PR. Bumping `schema_version` mitigates this. Documented in the
  `EpisodeCache` docstring as the standard schema-evolution caveat.

## 4. Storage layout

File-per-Episode under `<cache_dir>/<key[:2]>/<key>.json`. The two-char
sharding mirrors Git's object store and keeps any single directory
listing under ~256 sibling files even at extreme scale (a 1M-episode
cache shards to ~3.9k files per dir).

Atomic writes: `cache.put` writes to `<key>.json.tmp` and then
`os.replace`s to the final path. A SIGINT mid-write leaves no
partial JSON to be re-read as a "hit" by the next run.

Read path: `cache.get(key)` opens the file, parses JSON, validates as
`Episode.model_validate(...)`. A `JSONDecodeError` or `ValidationError`
counts as a miss (cache is treated as a hint, not a source of truth)
and the file is left in place — a future "cache verify" tool can
clean it up.

## 5. Runner integration

Per cell:

1. Build the WorkItem as today (this gives us `episode_seq`,
   perturbation_values, etc.).
2. Compute `env_seed = extract_env_seed(item.episode_seq)` in the
   parent process. (`extract_env_seed` is already pure; no change.)
3. Compute the cache key from the WorkItem + Runner-level fields.
4. `cache.get(key)` — if hit, append the cached Episode to the result
   list and skip dispatch.
5. Otherwise, dispatch the WorkItem to the executor (in-process or
   pool path — neither path needs to know about the cache).
6. After the executor returns the freshly-rolled Episode, call
   `cache.put(key, episode)` and append to the result list.

The existing `episodes.sort(key=lambda ep: (ep.cell_index,
ep.episode_index))` at the end of `Runner.run` already handles the
mixed (cached, fresh) ordering — no change needed there.

`cache_dir is None` short-circuits the entire flow: no
`EpisodeCache` is constructed, no `cache.get`/`cache.put` calls
happen, no per-cell key computation runs. The hot path is byte-
identical to today.

## 6. CLI integration

`gauntlet run` gains three flags:

* `--cache-dir PATH` — enable caching, store under PATH.
* `--no-cache` — explicit opt-out (overrides `--cache-dir`,
  documented for users who set a default in a wrapper script).
* `--cache-stats` — print `hits=N misses=N puts=N` to stderr after
  the run.

The CLI threads `policy_id` from a defaulted-to-`policy_spec_string`
value (the `--policy` argument is already a stable identifier for
the random / scripted / module-spec case). The CLI threads
`max_steps` from the existing `--env-max-steps` flag.

When `--cache-dir` is passed without `--env-max-steps`, the CLI
errors out with a clear hint: caching needs `max_steps` to be
explicit because the cache key depends on it.

## 7. Backwards-compatibility strategy

Three layered protections:

1. **Default-off opt-in.** `cache_dir=None` is the default at every
   layer (CLI, Runner). Every existing call path is byte-identical
   to pre-PR behaviour.
2. **No new module-level imports.** `EpisodeCache` lives in
   `src/gauntlet/runner/cache.py`; the Runner imports it at module
   scope but constructs it only when `cache_dir is not None`. The
   class itself uses only stdlib (`hashlib`, `json`, `pathlib`) plus
   `gauntlet.runner.episode.Episode` — no new wheel dependencies.
3. **Regression test.** `tests/test_cache_runner.py::test_no_cache_default_byte_identical`
   monkeypatches `EpisodeCache.__init__` to raise `AssertionError`,
   then runs a fixed-seed suite with `Runner(cache_dir=None)`. A
   construction would fire the assertion; the test confirms the
   no-cache path never touches the class.

## 8. Open questions (answered as part of this PR)

1. *Should `suite_hash` include the full Suite JSON or just `.name`?*
   Full JSON. Silent suite edits (adding an axis value, changing
   `episodes_per_cell`) MUST invalidate the cache; otherwise a user
   who tweaks the suite and reruns will silently get stale Episodes
   for the cells whose `(name, env, axis_config, seed)` happens to
   collide with the pre-tweak run. Hashing the full JSON is the
   minimum safety bar.

2. *What is `policy_id` if the user doesn't pass it?*
   The Runner falls back to `policy_factory().__class__.__name__`.
   This is constructed once per `Runner.run` call (NOT per cell —
   construction may have side-effects for VLA policies) by calling
   the factory once on the parent process and reading the class
   name. Documented as: "if you swap weights inside the same policy
   class, you MUST pass an explicit `policy_id` or you will get
   stale Episodes back".

3. *What if the cache file is corrupt?*
   `cache.get` treats `JSONDecodeError` and `ValidationError` as
   misses. The corrupt file is left in place; a future
   `gauntlet cache verify` could sweep them. The episode is
   re-rolled and the corrupt file is overwritten on `cache.put`.

4. *What if `policy_factory()` is expensive (e.g. loads a torch
   checkpoint) and we have a 100% cache hit rate?*
   The Runner only calls `policy_factory()` to derive
   `policy_id` when `cache_dir is set` AND `policy_id is None`.
   Users with expensive factories and known cache hits should pass
   an explicit `policy_id` to skip the introspection call. Documented
   in the docstring.

5. *What about the existing `trajectory_dir` / `record_video` side
   effects? Are they replayed on a cache hit?*
   No — a cache hit returns only the Episode object. Trajectory
   NPZs and MP4 videos are NOT rewritten on a hit. This is the
   correct behaviour for the VLA "I tweaked one axis, rerun the
   whole grid" workflow: the user wants the Episode summary fast,
   not a re-render of every video. Documented in the Runner
   docstring as a tradeoff. Users who need every artefact should
   not opt into caching.

6. *Concurrency: two `gauntlet run` processes pointing at the same
   `--cache-dir`?*
   The atomic-rename `put` is safe for the "two writers race to
   write the same key" case (`os.replace` is atomic on POSIX, and
   Windows >= Server 2003). The pathological case — two writers
   producing different Episodes for the same key — cannot happen
   because the key is content-addressed: identical inputs produce
   identical keys, and the Episode payloads should also be
   identical (modulo the determinism contract). Documented as
   "concurrent runs against the same cache_dir are safe".
