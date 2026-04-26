"""State-only observation hashing — Phase 2.5 Task 17 helpers.

The determinism contract every backend RFC promises is "fixed seed +
fixed action sequence -> bit-identical state-only obs". Pinning that at
the env layer (``tests/test_determinism_mujoco.py``) and at the space-
parity layer (``tests/test_determinism_cross_backend.py``) is necessary
but not sufficient: the gap was a portable, byte-precise *digest* that

* reduces a step / rollout / Episode down to a single comparable string,
* round-trips through subprocess boundaries (no ndarray pickling),
* refuses to silently accept a renderer-only delta (image bytes are
  excluded by construction — this is a state-only hash).

Public surface:

* :func:`obs_state_hash` — SHA-256 over the canonical state-only
  observation. Image keys are excluded; remaining values are coerced
  to ``float64`` and serialised in a stable key order.
* :func:`rollout_hash` — SHA-256 over a ``(initial_obs, actions,
  terminal_obs, success, length)`` rollout summary. Used by the
  cross-process determinism tests so the subprocess only needs to
  print one hex string.
* :func:`episode_hash` — SHA-256 over the deterministic identity
  fields of a :class:`gauntlet.runner.Episode` (the bits replay is
  required to reproduce). Wall-clock-noisy fields
  (``inference_latency_ms_*``, ``video_path``, ``git_commit``,
  ``gauntlet_version``, ``metadata``) are excluded so the hash is
  invariant across re-runs in the same checkout.
* :func:`assert_byte_identical` — focussed-diff helper that names the
  first differing key and prints both byte sequences in hex.

Anti-feature, deliberately documented: this module is the *contract*,
not the proof. Two backends that disagree on a single bit will produce
different digests; the test layer (``tests/test_determinism_hash.py``)
is what surfaces that and routes it through ``cross_backend_deltas.json``
for an explicit waiver. Adding a new state key to the obs dict (a 6th
key ever) reaches every backend's ``_build_obs`` and is loud by
construction — the digest changes, the within-backend tests fail until
every backend is updated in lockstep.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from gauntlet.runner.episode import Episode

__all__ = [
    "IMAGE_OBS_KEYS",
    "STATE_OBS_KEYS",
    "assert_byte_identical",
    "episode_hash",
    "obs_state_hash",
    "rollout_hash",
]


# Canonical 5-key state-obs schema, mirrored from
# ``tests/test_determinism_cross_backend.py::_EXPECTED_STATE_OBS_KEYS``.
# Any backend whose ``_build_obs`` returns a different key set is
# either a renderer key (filtered out below) or a real schema drift
# (the digest will differ, the test layer will surface it).
STATE_OBS_KEYS: frozenset[str] = frozenset(
    {"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"}
)

# Renderer outputs that the determinism contract intentionally excludes.
# Image bytes are out of scope — different GL stacks / driver versions
# produce visually identical but byte-different frames, and the state-
# only hash is the level at which "same seed -> same bits" is enforced.
IMAGE_OBS_KEYS: frozenset[str] = frozenset({"image", "images"})


def _canonical_state_obs_payload(obs: Mapping[str, ArrayLike]) -> bytes:
    """Serialise *obs* to a canonical byte string.

    Steps:
        1. Drop image keys (``image``, ``images``).
        2. Sort remaining keys lexicographically (stable order across
           Python ``dict`` literal orderings).
        3. For each (key, value) pair, append:
             * ``key.encode("utf-8")``
             * a single ``\x00`` separator
             * ``np.asarray(value, dtype=np.float64).tobytes()`` —
               coercing dtype is what makes ``int`` distractor counts
               and ``float32`` PyBullet outputs hash to the same bytes
               as the reference MuJoCo ``float64`` arrays.
             * a single ``\x00`` separator

    The ``\x00`` separators are belt-and-braces: with sorted keys and
    fixed-size payloads the sequence is already unambiguous, but the
    separators make a future addition of a variable-width key (a
    string-valued obs entry, say) safe by default.
    """
    parts: list[bytes] = []
    for key in sorted(obs):
        if key in IMAGE_OBS_KEYS:
            continue
        value = obs[key]
        arr: NDArray[np.float64] = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        parts.append(key.encode("utf-8"))
        parts.append(b"\x00")
        # ``shape`` is part of the payload so a flatten / reshape cannot
        # silently produce the same hash as a different layout.
        parts.append(",".join(str(d) for d in arr.shape).encode("utf-8"))
        parts.append(b"\x00")
        parts.append(arr.tobytes())
        parts.append(b"\x00")
    return b"".join(parts)


def obs_state_hash(obs: Mapping[str, ArrayLike]) -> str:
    """Return the SHA-256 hex digest of the state-only observation.

    Excludes image keys (``image``, ``images``) by construction;
    remaining values are coerced to ``float64`` and serialised in a
    stable lexicographic key order. Two observations that produce the
    same digest are byte-identical at the state level; image bytes
    may still differ.

    The function is safe to call on a partial obs dict (e.g. only
    ``{"cube_pos": ...}``) — it digests whatever state keys are
    present, in sorted order. The within-backend determinism tests
    pass full 5-key dicts; the cross-backend delta test inspects
    per-key arrays directly and does not rely on subset hashing.
    """
    digest = hashlib.sha256()
    digest.update(_canonical_state_obs_payload(obs))
    return digest.hexdigest()


def rollout_hash(
    *,
    initial_obs: Mapping[str, ArrayLike],
    actions: Iterable[ArrayLike],
    terminal_obs: Mapping[str, ArrayLike],
    success: bool,
    length: int,
) -> str:
    """Return the SHA-256 hex digest of a rollout summary.

    The digest covers:

    * ``obs_state_hash(initial_obs)`` — the post-reset state.
    * ``obs_state_hash(per-step action)`` for each action — actions
      are serialised exactly like obs values (``float64`` ``tobytes``
      with the array's shape) so a stray dtype mismatch cannot pass.
    * ``obs_state_hash(terminal_obs)`` — the final post-step state.
    * ``success`` and ``length`` — included so two rollouts with the
      same obs / action prefix but different termination behaviour
      hash to different values.

    Used by the subprocess cross-process determinism test: the child
    process prints exactly this string, the parent compares it to its
    own. ``length`` is also asserted explicitly by the test so a
    truncated child rollout is loud.
    """
    digest = hashlib.sha256()
    digest.update(b"initial:")
    digest.update(obs_state_hash(initial_obs).encode("ascii"))
    digest.update(b"\x00")
    for index, action in enumerate(actions):
        arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64))
        digest.update(f"action[{index}]:".encode("ascii"))
        digest.update(",".join(str(d) for d in arr.shape).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(arr.tobytes())
        digest.update(b"\x00")
    digest.update(b"terminal:")
    digest.update(obs_state_hash(terminal_obs).encode("ascii"))
    digest.update(b"\x00")
    digest.update(b"success:")
    digest.update(b"1" if success else b"0")
    digest.update(b"\x00")
    digest.update(b"length:")
    digest.update(str(int(length)).encode("ascii"))
    return digest.hexdigest()


# Episode fields carried for replay bit-identity. Strictly the
# fields :func:`gauntlet.replay.replay_one` is required to reproduce
# under zero overrides — wall-clock / environment / video-path noise
# is excluded so the hash is invariant across runs.
_EPISODE_HASH_FIELDS: tuple[str, ...] = (
    "suite_name",
    "cell_index",
    "episode_index",
    "seed",
    "success",
    "terminated",
    "truncated",
    "step_count",
    "total_reward",
)


def episode_hash(episode: Episode) -> str:
    """Return the SHA-256 hex digest of *episode*'s deterministic fields.

    Hashed fields:
        * ``suite_name`` / ``cell_index`` / ``episode_index`` / ``seed``
          — identity tuple. Two episodes with different identity
          tuples are different rollouts by definition.
        * ``perturbation_config`` — the axis values that drove the
          rollout. Serialised in sorted-key order.
        * ``success`` / ``terminated`` / ``truncated`` /
          ``step_count`` / ``total_reward`` — outcome. The first four
          are exact; ``total_reward`` is a ``float`` and is round-
          tripped through ``float.hex()`` so a 1-ulp difference
          surfaces.

    Excluded fields (intentionally):
        * ``video_path`` — disk path with the run output directory.
        * ``gauntlet_version`` / ``git_commit`` / ``suite_hash`` —
          provenance, not rollout outcome.
        * ``metadata`` — runner stuffs ``master_seed`` / topology
          here; covered by ``seed`` already.
        * ``inference_latency_ms_*`` — wall-clock dependent.
        * ``actuator_*`` / ``time_to_success`` / ``path_length_ratio``
          / ``jerk_rms`` / ``peak_force`` / ``near_collision_count``
          / safety counters — backend-asymmetric and float-noisy
          (see Episode docstring caveats).
        * ``failure_score`` / ``failure_alarm`` — derived from
          ``action_variance`` which is itself sampling-order-noisy
          unless the policy is deterministic.

    The exclusion list is conservative: the test layer asserts the
    digest is stable across replay, so any field that breaks that is
    a defect either in this hash or in the replay path. Adding more
    fields is loud by construction — the cross-backend / cross-replay
    tests fail.
    """
    digest = hashlib.sha256()
    for field_name in _EPISODE_HASH_FIELDS:
        value = getattr(episode, field_name)
        digest.update(field_name.encode("utf-8"))
        digest.update(b"\x00")
        if isinstance(value, bool):
            digest.update(b"1" if value else b"0")
        elif isinstance(value, int):
            digest.update(str(value).encode("ascii"))
        elif isinstance(value, float):
            # ``float.hex`` is round-trip-exact, unlike ``str(float)``
            # which depends on the host's libc. Catches a 1-ulp drift
            # that ``str`` representations would silently truncate.
            digest.update(value.hex().encode("ascii"))
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\x00")
    # ``perturbation_config`` is a ``dict[str, float]`` with no fixed
    # ordering guarantee in the schema. Iterate sorted to guarantee
    # the digest is invariant under dict-insertion order.
    digest.update(b"perturbation_config\x00")
    for axis_name in sorted(episode.perturbation_config):
        axis_value = episode.perturbation_config[axis_name]
        digest.update(axis_name.encode("utf-8"))
        digest.update(b":")
        digest.update(float(axis_value).hex().encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def assert_byte_identical(
    left: Mapping[str, ArrayLike],
    right: Mapping[str, ArrayLike],
    *,
    allow_keys: set[str] | None = None,
) -> None:
    """Assert two state-only obs dicts are byte-identical at every key.

    Compares the *raw bytes* of each value (after the same
    ``float64``-coercion the hash uses) so a 1-ulp dtype-promoted
    difference is caught. On the first mismatch raises
    :class:`AssertionError` with:

    * the offending key,
    * the byte offset of the first disagreement,
    * a short hex slice (8 bytes) around the disagreement on each side.

    ``allow_keys``: optional set of keys to skip. Image keys
    (:data:`IMAGE_OBS_KEYS`) are always skipped — the determinism
    contract is state-only by construction; pixel parity is out of
    scope (RFC-007 §7.3, RFC-008 §8 Q2).

    Use this in tests instead of a bare ``assert obs == obs2``: a
    plain dict equality on numpy arrays raises ``ValueError``
    ("ambiguous truth value"), and a ``np.array_equal`` at the top
    level discards which key disagreed.
    """
    skip = IMAGE_OBS_KEYS | (allow_keys or set())
    left_keys = set(left) - skip
    right_keys = set(right) - skip
    if left_keys != right_keys:
        only_left = sorted(left_keys - right_keys)
        only_right = sorted(right_keys - left_keys)
        raise AssertionError(f"obs key sets differ: only_left={only_left} only_right={only_right}")
    for key in sorted(left_keys):
        left_arr: NDArray[np.float64] = np.ascontiguousarray(
            np.asarray(left[key], dtype=np.float64)
        )
        right_arr: NDArray[np.float64] = np.ascontiguousarray(
            np.asarray(right[key], dtype=np.float64)
        )
        if left_arr.shape != right_arr.shape:
            raise AssertionError(
                f"key {key!r}: shape mismatch {left_arr.shape} vs {right_arr.shape}"
            )
        left_bytes = left_arr.tobytes()
        right_bytes = right_arr.tobytes()
        if left_bytes == right_bytes:
            continue
        # Find first byte that differs.
        offset = next(
            (i for i, (a, b) in enumerate(zip(left_bytes, right_bytes, strict=True)) if a != b),
            -1,
        )
        # Hex slice around the disagreement (8 bytes either side, capped).
        start = max(offset - 8, 0)
        end = min(offset + 8, len(left_bytes))
        left_hex = left_bytes[start:end].hex()
        right_hex = right_bytes[start:end].hex()
        raise AssertionError(
            f"key {key!r}: byte mismatch at offset {offset}; "
            f"left={left_hex!r} right={right_hex!r}; "
            f"left_values={left_arr.flatten().tolist()!r} "
            f"right_values={right_arr.flatten().tolist()!r}"
        )
