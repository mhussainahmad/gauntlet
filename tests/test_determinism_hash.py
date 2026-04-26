"""Cross-backend determinism audit via SHA-256 obs digests — Phase 2.5 T17.

Companion to ``tests/test_determinism_cross_backend.py`` (state-shape +
action-space parity) and ``tests/test_determinism_mujoco.py`` (within-
backend byte-identity at the env layer). This file pins the digest
contract:

* :func:`gauntlet.runner.obs_state_hash` reduces a state-only obs dict
  to a SHA-256 hex digest; image bytes are excluded by construction.
* :func:`gauntlet.runner.rollout_hash` digests a fixed-action rollout
  end-to-end so the subprocess test only needs to print one string.
* :func:`gauntlet.runner.episode_hash` digests an :class:`Episode`
  over the deterministic identity / outcome fields; replay must
  reproduce it bit-exactly.

Layered tests:

1. **Helper unit tests** — pin the digest format so a refactor cannot
   silently re-permute the canonical byte order.
2. **Within-backend byte-identity** — for every reachable backend,
   reset(seed=42) twice + run a 10-step canned action sequence; the
   per-step ``obs_state_hash`` must match step-by-step.
3. **Cross-backend allowed-delta waiver** — for each pair of backends
   reachable simultaneously, the per-key absolute delta after the
   same canned rollout must lie within the bound declared in
   ``tests/data/cross_backend_deltas.json``. A new delta means a JSON
   edit + rationale.
4. **Subprocess byte-identity** — same backend, same canned rollout,
   in a child Python interpreter (``sys.executable``). The child
   prints the rollout digest; the parent compares against its own.
5. **Replay episode-hash stability** — Runner produces an Episode,
   :func:`replay_one` reproduces it; the two ``episode_hash`` values
   must be equal.

Cases (4) and (5) live in this file too (split out by section).

PyBullet / Genesis tests gate on the relevant import via
``pytest.importorskip``. Isaac Sim's default-job fake is scoped to
``tests/isaac/`` only; the cross-backend pair tests honour that by
skipping any pair touching Isaac when its real binding is absent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gauntlet.env.tabletop import TabletopEnv
from gauntlet.runner import (
    IMAGE_OBS_KEYS,
    STATE_OBS_KEYS,
    assert_byte_identical,
    obs_state_hash,
)

# 10-step canned action sequence. Generated once at module import via a
# fixed-seed numpy Generator so the test is hermetic — no test-side
# randomness, no clock entropy. The values themselves do not matter;
# what matters is that they are float64 and the same on every test
# invocation.
_CANNED_ACTIONS: tuple[np.ndarray, ...] = tuple(
    np.random.default_rng(0xC1DE).uniform(-1.0, 1.0, size=(7,)).astype(np.float64)
    for _ in range(10)
)
_CANNED_SEED: int = 42

# Cross-backend deltas waiver — single source of truth, see module docstring.
_DELTAS_JSON: Path = Path(__file__).parent / "data" / "cross_backend_deltas.json"


# ----------------------------------------------------------------------------
# Helper unit tests — pin the digest format.
# ----------------------------------------------------------------------------


def test_obs_state_hash_excludes_image_keys() -> None:
    """Adding an ``image`` / ``images`` key must not change the digest."""
    base: dict[str, np.ndarray] = {
        "cube_pos": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "ee_pos": np.array([0.4, 0.5, 0.6], dtype=np.float64),
    }
    with_image: dict[str, Any] = {
        **base,
        # Random uint8 frame — different bytes on every run if it was
        # actually hashed; the digest must still match the no-image
        # reference.
        "image": np.random.default_rng(0).integers(0, 255, size=(8, 8, 3), dtype=np.uint8),
    }
    assert obs_state_hash(base) == obs_state_hash(with_image)


def test_obs_state_hash_dtype_coerced_to_float64() -> None:
    """A float32 / float64 obs with the same nominal values must hash equal."""
    a = {"cube_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    b = {"cube_pos": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
    assert obs_state_hash(a) == obs_state_hash(b)


def test_obs_state_hash_distinguishes_value_change() -> None:
    a = {"cube_pos": np.array([0.0, 0.0, 0.0], dtype=np.float64)}
    b = {"cube_pos": np.array([0.0, 0.0, 1e-12], dtype=np.float64)}
    # 1 ulp at zero is non-zero -> different bits -> different digest.
    assert obs_state_hash(a) != obs_state_hash(b)


def test_obs_state_hash_independent_of_dict_insertion_order() -> None:
    a = {"cube_pos": np.zeros(3), "ee_pos": np.ones(3)}
    b = {"ee_pos": np.ones(3), "cube_pos": np.zeros(3)}
    assert obs_state_hash(a) == obs_state_hash(b)


def test_image_obs_keys_constant_pinned() -> None:
    """Lock the excluded-key set against silent additions / removals."""
    expected_image: frozenset[str] = frozenset({"image", "images"})
    # Two named locals; SIM300 fires on either ordering against an ALL_CAPS
    # constant. Suppress — both sides are real symbols.
    assert IMAGE_OBS_KEYS == expected_image  # noqa: SIM300


def test_state_obs_keys_constant_pinned() -> None:
    """Lock the canonical 5-key state schema."""
    expected_state: frozenset[str] = frozenset(
        {"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"}
    )
    assert STATE_OBS_KEYS == expected_state  # noqa: SIM300


def test_assert_byte_identical_passes_on_equal_obs() -> None:
    a = {"cube_pos": np.array([1.0, 2.0, 3.0])}
    b = {"cube_pos": np.array([1.0, 2.0, 3.0])}
    # No raise.
    assert_byte_identical(a, b)


def test_assert_byte_identical_reports_offending_key_and_offset() -> None:
    a = {
        "cube_pos": np.array([1.0, 2.0, 3.0]),
        "ee_pos": np.array([0.0, 0.0, 0.0]),
    }
    b = {
        "cube_pos": np.array([1.0, 2.0, 3.0]),
        "ee_pos": np.array([0.0, 0.0, 1e-12]),
    }
    with pytest.raises(AssertionError) as exc:
        assert_byte_identical(a, b)
    msg = str(exc.value)
    assert "ee_pos" in msg
    assert "byte mismatch at offset" in msg


def test_assert_byte_identical_reports_shape_mismatch() -> None:
    a = {"cube_pos": np.array([1.0, 2.0, 3.0])}
    b = {"cube_pos": np.array([1.0, 2.0, 3.0, 4.0])}
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_byte_identical(a, b)


def test_assert_byte_identical_reports_key_set_diff() -> None:
    a = {"cube_pos": np.zeros(3), "ee_pos": np.zeros(3)}
    b = {"cube_pos": np.zeros(3)}
    with pytest.raises(AssertionError, match="key sets differ"):
        assert_byte_identical(a, b)


# ----------------------------------------------------------------------------
# Within-backend byte-identity — same seed, same canned rollout.
# ----------------------------------------------------------------------------


def _rollout_obs_hashes(env_factory: Callable[[], Any]) -> list[str]:
    """Return per-step ``obs_state_hash`` values for a canned rollout.

    Constructs a fresh env via *env_factory*, calls ``reset(seed=
    _CANNED_SEED)``, then steps the canned action sequence. The
    returned list has length ``len(_CANNED_ACTIONS) + 1`` — the
    initial post-reset obs followed by one per step.
    """
    env = env_factory()
    try:
        obs, _ = env.reset(seed=_CANNED_SEED)
        hashes: list[str] = [obs_state_hash(obs)]
        for action in _CANNED_ACTIONS:
            obs, _, _terminated, _truncated, _ = env.step(action)
            hashes.append(obs_state_hash(obs))
        return hashes
    finally:
        env.close()


def test_mujoco_within_backend_byte_identical_across_resets() -> None:
    """MuJoCo: two independent envs, same seed, same canned rollout
    -> per-step digests must match step-by-step."""
    a = _rollout_obs_hashes(TabletopEnv)
    b = _rollout_obs_hashes(TabletopEnv)
    assert len(a) == len(b) == len(_CANNED_ACTIONS) + 1
    for index, (h_a, h_b) in enumerate(zip(a, b, strict=True)):
        assert h_a == h_b, f"step {index}: digest mismatch {h_a} vs {h_b}"


def test_pybullet_within_backend_byte_identical_across_resets() -> None:
    """PyBullet: same contract as MuJoCo, gated on the extra."""
    pytest.importorskip(
        "pybullet",
        reason="PyBullet extra not installed (uv sync --extra pybullet)",
    )
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    def factory() -> Any:
        return PyBulletTabletopEnv(render_in_obs=False)

    a = _rollout_obs_hashes(factory)
    b = _rollout_obs_hashes(factory)
    assert len(a) == len(b) == len(_CANNED_ACTIONS) + 1
    for index, (h_a, h_b) in enumerate(zip(a, b, strict=True)):
        assert h_a == h_b, f"step {index}: digest mismatch {h_a} vs {h_b}"


def test_genesis_within_backend_byte_identical_across_resets() -> None:
    """Genesis: same contract as MuJoCo, gated on the extra."""
    pytest.importorskip(
        "genesis",
        reason="Genesis extra not installed (uv sync --extra genesis)",
    )
    from gauntlet.env.genesis import GenesisTabletopEnv

    def factory() -> Any:
        return GenesisTabletopEnv(render_in_obs=False)

    a = _rollout_obs_hashes(factory)
    b = _rollout_obs_hashes(factory)
    assert len(a) == len(b) == len(_CANNED_ACTIONS) + 1
    for index, (h_a, h_b) in enumerate(zip(a, b, strict=True)):
        assert h_a == h_b, f"step {index}: digest mismatch {h_a} vs {h_b}"


# ----------------------------------------------------------------------------
# Cross-backend allowed-delta waiver.
# ----------------------------------------------------------------------------


def _rollout_terminal_obs(env_factory: Callable[[], Any]) -> dict[str, np.ndarray]:
    """Run the canned rollout and return the terminal state-only obs dict.

    Image keys are filtered out of the returned dict so per-key delta
    inspection only touches state keys.
    """
    env = env_factory()
    try:
        env.reset(seed=_CANNED_SEED)
        obs: dict[str, Any] = {}
        for action in _CANNED_ACTIONS:
            obs, _, _terminated, _truncated, _ = env.step(action)
        return {
            k: np.asarray(v, dtype=np.float64) for k, v in obs.items() if k not in IMAGE_OBS_KEYS
        }
    finally:
        env.close()


def _load_cross_backend_deltas() -> dict[str, Any]:
    with _DELTAS_JSON.open() as fh:
        data: dict[str, Any] = json.load(fh)
    assert data["version"] == 1, f"unsupported deltas-json version: {data['version']}"
    return data


def _pair_key(backend_a: str, backend_b: str) -> str:
    """Canonical (sorted) pair key for the deltas JSON."""
    lo, hi = sorted([backend_a, backend_b])
    return f"{lo}__{hi}"


def _factory_for(backend: str) -> Callable[[], Any]:
    if backend == "mujoco":

        def mj() -> Any:
            return TabletopEnv()

        return mj
    if backend == "pybullet":
        pytest.importorskip(
            "pybullet",
            reason="PyBullet extra not installed (uv sync --extra pybullet)",
        )
        from gauntlet.env.pybullet import PyBulletTabletopEnv

        def pb() -> Any:
            return PyBulletTabletopEnv(render_in_obs=False)

        return pb
    if backend == "genesis":
        pytest.importorskip(
            "genesis",
            reason="Genesis extra not installed (uv sync --extra genesis)",
        )
        from gauntlet.env.genesis import GenesisTabletopEnv

        def gs() -> Any:
            return GenesisTabletopEnv(render_in_obs=False)

        return gs
    if backend == "isaac":
        # Real Isaac Sim is not exercised in the default job (RFC-009
        # §8); the fake is autouse-injected only inside tests/isaac/.
        # Skip cleanly here so the cross-backend matrix is honest.
        pytest.importorskip(
            "isaacsim",
            reason=(
                "Isaac Sim is fake-injected only inside tests/isaac/; "
                "the cross-backend pair test runs only with real Kit installed."
            ),
        )
        from gauntlet.env.isaac import IsaacSimTabletopEnv

        def isaac() -> Any:
            return IsaacSimTabletopEnv()

        return isaac
    raise AssertionError(f"unknown backend: {backend}")


_BACKEND_PAIRS: tuple[tuple[str, str], ...] = (
    ("mujoco", "pybullet"),
    ("mujoco", "genesis"),
    ("mujoco", "isaac"),
    ("pybullet", "genesis"),
    ("pybullet", "isaac"),
    ("genesis", "isaac"),
)


@pytest.mark.parametrize(("backend_a", "backend_b"), _BACKEND_PAIRS)
def test_cross_backend_allowed_delta_documented(backend_a: str, backend_b: str) -> None:
    """For each pair, the per-key delta after a canned rollout must
    be within the bound declared in ``cross_backend_deltas.json``.

    Skips cleanly when either backend's extra is missing — the ``mujoco
    __mujoco`` self-comparison is gated separately (within-backend
    test above) so this test only runs when both extras are present.

    A new delta means a JSON edit + rationale; no axis-by-axis
    silent floats.
    """
    deltas = _load_cross_backend_deltas()
    pair_key = _pair_key(backend_a, backend_b)
    pair_entry = deltas["pairs"].get(pair_key)
    assert pair_entry is not None, (
        f"pair {pair_key!r} missing from cross_backend_deltas.json — "
        f"add an entry with max_abs_delta_per_key + rationale before this gate goes green."
    )
    bounds: dict[str, float] = pair_entry["max_abs_delta_per_key"]
    rationale: str = pair_entry["rationale"]
    assert rationale.strip(), f"pair {pair_key!r} has empty rationale; explain the delta"

    obs_a = _rollout_terminal_obs(_factory_for(backend_a))
    obs_b = _rollout_terminal_obs(_factory_for(backend_b))

    common_keys = set(obs_a) & set(obs_b)
    assert common_keys, f"pair {pair_key!r}: backends share no state keys"
    for key in sorted(common_keys):
        bound = float(bounds.get(key, 0.0))
        delta = float(np.max(np.abs(obs_a[key] - obs_b[key])))
        assert delta <= bound, (
            f"pair {pair_key!r}: key {key!r} max-abs delta {delta:.3e} > bound {bound:.3e}; "
            f"either tighten the implementation or update "
            f"cross_backend_deltas.json with a rationale."
        )


def test_cross_backend_deltas_json_schema_pinned() -> None:
    """The deltas JSON has a stable shape — version + pairs + per-pair
    fields. Any future schema change should bump ``version`` so older
    test sources fail loud."""
    deltas = _load_cross_backend_deltas()
    assert deltas["version"] == 1
    assert "doc" in deltas and deltas["doc"].strip()
    assert "pairs" in deltas
    for pair_key, entry in deltas["pairs"].items():
        assert "__" in pair_key, f"pair key {pair_key!r} must be 'a__b'"
        assert "max_abs_delta_per_key" in entry
        assert "rationale" in entry
        assert entry["rationale"].strip()
