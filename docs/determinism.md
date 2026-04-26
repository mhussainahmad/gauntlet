# Determinism contract

Phase 2.5 Task 17 nailed the determinism contract every backend RFC
promises but which previously was only enforced at the env or Runner
layer in isolation. This page is the single source of truth.

## The contract, in one sentence

For a fixed `(backend, seed, action sequence)`, the backend produces
**byte-identical** state-only observations across reset cycles and
across processes. Cross-backend deltas are not byte-identical — they
are bounded by an explicit waiver list.

## What "state-only" means

State-only observations are the 5-key Dict every backend exposes when
`render_in_obs=False`:

* `cube_pos` — float64, shape `(3,)`
* `cube_quat` — float64, shape `(4,)` (MuJoCo wxyz order)
* `ee_pos` — float64, shape `(3,)`
* `gripper` — float64, shape `(1,)`, `-1` closed / `+1` open
* `target_pos` — float64, shape `(3,)`

Image bytes (`obs["image"]`, `obs["images"]`) are **out of scope** for
the determinism contract. Different GL stacks / driver versions render
visually identical but byte-different frames; chasing those at the bit
level is yak-shaving and the renderer-rendering RFCs (RFC-006, RFC-008)
explicitly carve them out.

## What the audit proves

The audit lives in three test files that layer onto each other:

| File | Layer | Cases |
|------|-------|-------|
| `tests/test_determinism_mujoco.py` | env, MuJoCo | 1, 4, 6 |
| `tests/test_determinism_cross_backend.py` | env, all backends | 2, 3 |
| `tests/test_determinism_hash.py` | digest contract | helper unit + 5, 7 |
| `tests/test_determinism_runner_workers.py` | runner | 5 (Runner) |

The seven cases pinned across the four files:

1. **Within-backend reset** — two `reset(seed=S)` calls produce
   byte-identical state-only obs (env layer + digest layer).
2. **State-shape parity** — every backend exposes the same 5-key
   `observation_space` (env layer).
3. **Action-space parity** — every backend exposes the same 7-D
   `action_space` (env layer).
4. **Perturbation determinism** — same `(axis, value)` queued at the
   same seed produces byte-identical post-reset state for every
   axis (env layer, per backend).
5. **Cross-process determinism** — Runner with `processes=1` and
   `processes=N` produces the same `list[Episode]` after the
   `(cell_index, episode_index)` sort (Runner layer); a hermetic
   subprocess running the same canned rollout produces the same
   rollout digest (digest layer).
6. **Restore-baseline idempotency** — `apply_perturbation` then
   `restore_baseline` returns the env to byte-identical baseline
   state for every axis (env layer).
7. **Replay episode-hash stability** — `replay_one(target=ep)` of an
   already-recorded `Episode` reproduces `episode_hash(ep)` exactly
   (digest layer; companion to `test_replay.py::
   test_zero_override_bit_identity`, which is full-Episode equality).

## The digest helpers

Three public helpers live under `gauntlet.runner` and are re-exported
from the package root:

* `obs_state_hash(obs) -> str` — SHA-256 over the canonical state-only
  observation. Image keys are excluded; remaining values are coerced
  to `float64` and serialised in a stable lexicographic key order.
* `rollout_hash(initial_obs, actions, terminal_obs, success, length)
  -> str` — SHA-256 over a rollout summary. Subprocess tests can
  print one digest and the parent compares.
* `episode_hash(episode: Episode) -> str` — SHA-256 over the
  deterministic identity / outcome fields of an `Episode`. Wall-
  clock-noisy fields (`inference_latency_ms_*`, `video_path`,
  `git_commit`, `gauntlet_version`, `metadata`, all backend-
  asymmetric telemetry) are excluded so the digest is invariant
  across replay in the same checkout.

Plus one diagnostic helper:

* `assert_byte_identical(left, right)` — focussed-diff helper that
  names the offending key, the byte offset of the first mismatch,
  and prints both byte sequences in hex. Use this instead of bare
  `assert obs == obs2` (which raises `ValueError("ambiguous truth
  value")` on ndarray values).

The digest helpers exclude image keys (`image`, `images`) by
construction — see `IMAGE_OBS_KEYS` and `STATE_OBS_KEYS` for the
canonical key sets.

## Cross-backend deltas — the waiver list

Cross-backend determinism is not byte-identity. MuJoCo, PyBullet,
Genesis, and Isaac use different solver defaults, contact models, and
floating-point tolerances; the same canned rollout produces obs that
agree to solver tolerance, not to bit equality.

The single source of truth for "what tolerance is acceptable for which
pair" is:

```
tests/data/cross_backend_deltas.json
```

The schema:

```json
{
  "version": 1,
  "doc": "...",
  "schema": { "...": "..." },
  "pairs": {
    "<lo>__<hi>": {
      "max_abs_delta_per_key": {
        "cube_pos": 1e-06,
        "cube_quat": 1e-06,
        "ee_pos": 0.0,
        "gripper": 0.0,
        "target_pos": 0.0
      },
      "rationale": "one-sentence justification"
    }
  }
}
```

* **Pair key**: the two backend names sorted ascending and joined by
  `__`. Each pair has exactly one canonical key — `mujoco__pybullet`,
  not `pybullet__mujoco`.
* **`max_abs_delta_per_key`**: per state-obs key, the max absolute
  float delta the audit may observe across the two backends after a
  fixed canned rollout. Missing key means `0.0` (the test asserts
  byte-identity for that key).
* **`rationale`**: human-readable single sentence; cite the RFC or
  upstream solver-version note when possible. The test asserts the
  rationale is non-empty, which is what makes the JSON the *waiver
  list* — adding a delta is gated on writing the rationale.

## Adding or upgrading a backend

1. Add a within-backend byte-identity test in
   `tests/test_determinism_hash.py` modelled on
   `test_mujoco_within_backend_byte_identical_across_resets` —
   import-skip on the relevant extra; assert per-step
   `obs_state_hash` matches across two independent resets at the
   same seed.
2. Add cross-backend pair entries to
   `tests/data/cross_backend_deltas.json` for every pair the new
   backend forms with existing backends. Pair keys are
   `lo__hi`-sorted; missing pairs cause
   `test_cross_backend_allowed_delta_documented` to fail with a
   pointed error.
3. If a backend upgrade legitimately changes solver tolerance, edit
   the relevant `max_abs_delta_per_key` and update the `rationale`
   field with the upstream version note. The test does not accept
   widened bounds without a rationale change — the diff is the
   review surface.
4. Update `tests/test_determinism_cross_backend.py`'s parity tests
   if the new backend changes the canonical 5-key state schema —
   that change is loud by construction (`STATE_OBS_KEYS` is a
   `frozenset` constant pinned in the test).

## Anti-features (deliberate)

* **Image-byte determinism is not asserted.** Renderer parity is the
  rendering RFCs' job (RFC-006, RFC-008); the digest helpers
  exclude images on purpose so a driver upgrade does not break the
  determinism gate.
* **Cross-backend byte-identity is not asserted.** The waiver JSON
  is the explicit answer — every pair gets a tolerance and a
  rationale. A "we tightened the bound to zero everywhere" PR is
  welcome but is a separate engineering job from the audit
  contract.
* **Wall-clock fields are excluded from `episode_hash`.** A run on a
  fast machine and a run on a slow machine produce different
  `inference_latency_ms_*` values; both are equally correct under
  the determinism contract. The digest catches the deterministic
  drift; the timing drift is caught by the budget gate
  (`Runner.max_inference_ms`).
