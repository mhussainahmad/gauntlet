"""Tests for the B-40 suite-level provenance hash.

The hash is a 16-char blake2b digest over a canonical JSON payload of
``(suite model dump, gauntlet version, hash_version, env asset SHAs)``.
These tests pin the four properties the spec calls out:

* **order independence** — reordering the ``axes`` mapping (or any
  other dict-keyed field) produces the same digest.
* **version sensitivity** — a different ``gauntlet_version`` produces
  a different digest. Same for ``hash_version`` (an
  internal-canonicalisation bump).
* **ASCII canonicalisation** — non-ASCII suite metadata flows
  deterministically through the digest.
* **episodes_per_cell delta** — bumping the field produces a
  different digest (the cache must invalidate).

Plus a ``gauntlet suite hash`` CLI smoke test and a few integration
checks against ``EpisodeCache.make_key`` so the cache-key migration
behaves as documented.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.runner.cache import EpisodeCache
from gauntlet.runner.provenance import (
    SUITE_PROVENANCE_HASH_VERSION,
    compute_env_asset_shas,
    compute_suite_hash,
    compute_suite_provenance_hash,
    default_assets_root,
)
from gauntlet.suite.schema import AxisSpec, Suite

# ----------------------------------------------------------------------------
# Suite builders. Each test gets a fresh suite — no module-level state.
# ----------------------------------------------------------------------------


def _two_axis_suite(
    *,
    name: str = "b40-suite",
    episodes_per_cell: int = 1,
    axis_order: tuple[str, str] = ("lighting_intensity", "camera_offset_x"),
) -> Suite:
    """Suite with exactly two axes, in the requested insertion order."""
    spec_for = {
        "lighting_intensity": AxisSpec(low=0.5, high=1.0, steps=2),
        "camera_offset_x": AxisSpec(low=-0.02, high=0.02, steps=2),
    }
    axes = {name: spec_for[name] for name in axis_order}
    return Suite(
        name=name,
        env="tabletop",
        seed=42,
        episodes_per_cell=episodes_per_cell,
        axes=axes,
    )


def _empty_assets_root(tmp_path: Path) -> Path:
    """An empty asset tree — keeps tests independent of the in-tree assets."""
    root = tmp_path / "assets"
    root.mkdir()
    return root


# ----------------------------------------------------------------------------
# 1. Output shape — 16-char lowercase hex.
# ----------------------------------------------------------------------------


def test_provenance_hash_is_16_char_lowercase_hex(tmp_path: Path) -> None:
    suite = _two_axis_suite()
    digest = compute_suite_provenance_hash(
        suite, gauntlet_version="t", assets_root=_empty_assets_root(tmp_path)
    )
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


def test_provenance_hash_is_deterministic(tmp_path: Path) -> None:
    suite = _two_axis_suite()
    assets = _empty_assets_root(tmp_path)
    a = compute_suite_provenance_hash(suite, gauntlet_version="t", assets_root=assets)
    b = compute_suite_provenance_hash(suite, gauntlet_version="t", assets_root=assets)
    assert a == b


# ----------------------------------------------------------------------------
# 2. Order independence — axis-name reordering must not change the digest.
# ----------------------------------------------------------------------------


def test_axis_dict_order_does_not_affect_hash(tmp_path: Path) -> None:
    """Two suites that only differ in ``axes`` insertion order must hash identically."""
    assets = _empty_assets_root(tmp_path)
    suite_a = _two_axis_suite(axis_order=("lighting_intensity", "camera_offset_x"))
    suite_b = _two_axis_suite(axis_order=("camera_offset_x", "lighting_intensity"))
    a = compute_suite_provenance_hash(suite_a, gauntlet_version="t", assets_root=assets)
    b = compute_suite_provenance_hash(suite_b, gauntlet_version="t", assets_root=assets)
    assert a == b, (
        "axis-dict reorder changed the suite-provenance hash; "
        "the canonicaliser must sort dict keys recursively"
    )


def test_axis_value_order_DOES_affect_hash(tmp_path: Path) -> None:
    """Categorical ``values`` lists carry index meaning — reorder MUST change the hash.

    B-05 (``instruction_paraphrase``) and B-06 (``object_swap``) both
    consume the index of the value; sorting the list silently would
    misroute the runner. The canonicaliser deliberately leaves list
    order alone.
    """
    assets = _empty_assets_root(tmp_path)
    suite_a = Suite(
        name="paraphrase-test",
        env="tabletop",
        seed=42,
        episodes_per_cell=1,
        axes={
            "instruction_paraphrase": AxisSpec(values=["pick the cup", "grasp the mug"]),
        },
    )
    suite_b = suite_a.model_copy(
        update={
            "axes": {
                "instruction_paraphrase": AxisSpec(values=["grasp the mug", "pick the cup"]),
            }
        }
    )
    a = compute_suite_provenance_hash(suite_a, gauntlet_version="t", assets_root=assets)
    b = compute_suite_provenance_hash(suite_b, gauntlet_version="t", assets_root=assets)
    assert a != b


# ----------------------------------------------------------------------------
# 3. Field-level sensitivity — every input change must move the digest.
# ----------------------------------------------------------------------------


def test_episodes_per_cell_change_changes_hash(tmp_path: Path) -> None:
    assets = _empty_assets_root(tmp_path)
    a = compute_suite_provenance_hash(
        _two_axis_suite(episodes_per_cell=1),
        gauntlet_version="t",
        assets_root=assets,
    )
    b = compute_suite_provenance_hash(
        _two_axis_suite(episodes_per_cell=2),
        gauntlet_version="t",
        assets_root=assets,
    )
    assert a != b


def test_gauntlet_version_change_changes_hash(tmp_path: Path) -> None:
    assets = _empty_assets_root(tmp_path)
    suite = _two_axis_suite()
    a = compute_suite_provenance_hash(suite, gauntlet_version="0.1.0", assets_root=assets)
    b = compute_suite_provenance_hash(suite, gauntlet_version="0.2.0", assets_root=assets)
    assert a != b, (
        "a gauntlet version bump must invalidate the suite-provenance "
        "hash by design (anti-feature documented in B-40)"
    )


def test_unknown_version_is_hashed_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``capture_gauntlet_version`` returns ``None`` the digest is still well-defined.

    The default-arg path threads ``"unknown"`` into the canonical
    payload — a sys.path checkout (no installed distribution) still
    produces a stable hash.
    """
    assets = _empty_assets_root(tmp_path)
    # Force the package-version lookup to fail so the function takes
    # the "unknown" fallback branch.
    from gauntlet.runner import provenance as provenance_mod

    monkeypatch.setattr(provenance_mod, "capture_gauntlet_version", lambda: None)
    suite = _two_axis_suite()
    fallback = compute_suite_provenance_hash(suite, assets_root=assets)
    explicit = compute_suite_provenance_hash(suite, gauntlet_version="unknown", assets_root=assets)
    assert fallback == explicit


def test_env_asset_change_changes_hash(tmp_path: Path) -> None:
    """Edit any file under the assets root -> different hash.

    Mirrors the documented invalidation contract: a simulator-asset
    edit silently changes rollouts, so the cache must miss.
    """
    suite = _two_axis_suite()
    assets = _empty_assets_root(tmp_path)
    (assets / "a.xml").write_text("<mujoco/>", encoding="utf-8")
    a = compute_suite_provenance_hash(suite, gauntlet_version="t", assets_root=assets)
    (assets / "a.xml").write_text("<mujoco edited='true'/>", encoding="utf-8")
    b = compute_suite_provenance_hash(suite, gauntlet_version="t", assets_root=assets)
    assert a != b


def test_missing_assets_root_yields_empty_dict(tmp_path: Path) -> None:
    """A non-existent assets root must NOT raise — empty mapping is the contract."""
    missing = tmp_path / "does-not-exist"
    assert compute_env_asset_shas(missing) == {}
    # And the suite hash is still well-defined.
    suite = _two_axis_suite()
    digest = compute_suite_provenance_hash(suite, gauntlet_version="t", assets_root=missing)
    assert len(digest) == 16


# ----------------------------------------------------------------------------
# 4. ASCII canonicalisation — unicode suite text hashes deterministically.
# ----------------------------------------------------------------------------


def test_unicode_suite_name_hashes_deterministically(tmp_path: Path) -> None:
    """Non-ASCII characters must encode the same way every run / locale."""
    assets = _empty_assets_root(tmp_path)
    base = _two_axis_suite()
    unicode_suite = base.model_copy(update={"name": "café-suite-é"})
    a = compute_suite_provenance_hash(unicode_suite, gauntlet_version="t", assets_root=assets)
    b = compute_suite_provenance_hash(unicode_suite, gauntlet_version="t", assets_root=assets)
    assert a == b
    # And the unicode name is genuinely a different suite from an ASCII alternative.
    ascii_suite = base.model_copy(update={"name": "cafe-suite-e"})
    c = compute_suite_provenance_hash(ascii_suite, gauntlet_version="t", assets_root=assets)
    assert a != c


# ----------------------------------------------------------------------------
# 5. ``hash_version`` constant — bumping it invalidates everything.
# ----------------------------------------------------------------------------


def test_hash_version_is_an_integer_constant() -> None:
    """``SUITE_PROVENANCE_HASH_VERSION`` is a plain int — the YAML cannot override it."""
    assert isinstance(SUITE_PROVENANCE_HASH_VERSION, int)
    assert SUITE_PROVENANCE_HASH_VERSION >= 1


def test_hash_version_is_in_the_canonical_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bumping ``SUITE_PROVENANCE_HASH_VERSION`` MUST change the digest.

    Reaches into the module and bumps the constant — confirms the
    canonicaliser actually folds the value in. Restored by the
    monkeypatch.
    """
    assets = _empty_assets_root(tmp_path)
    suite = _two_axis_suite()
    before = compute_suite_provenance_hash(suite, gauntlet_version="t", assets_root=assets)

    from gauntlet.runner import provenance as provenance_mod

    monkeypatch.setattr(provenance_mod, "SUITE_PROVENANCE_HASH_VERSION", 999)
    after = compute_suite_provenance_hash(suite, gauntlet_version="t", assets_root=assets)
    assert before != after


# ----------------------------------------------------------------------------
# 6. Cache-key integration — ``EpisodeCache.make_key`` consumes the new hash.
# ----------------------------------------------------------------------------


def test_cache_key_uses_new_provenance_hash() -> None:
    """The new B-40 hash MUST appear in the canonical key payload.

    Cross-check: ``make_key`` and ``make_legacy_key`` must produce
    different digests when the underlying suite hashes differ. They
    only collide if both are the empty string, which can't happen.
    """
    suite = _two_axis_suite()
    axis_config: dict[str, float] = {"lighting_intensity": 0.5, "camera_offset_x": 0.0}
    new_key = EpisodeCache.make_key(
        suite,
        axis_config=axis_config,
        seed=1,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="P",
    )
    legacy_key = EpisodeCache.make_legacy_key(
        suite,
        axis_config=axis_config,
        seed=1,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="P",
    )
    # Different ``suite_hash`` -> different final SHA256.
    assert new_key != legacy_key


def test_legacy_key_matches_pre_b40_derivation() -> None:
    """``make_legacy_key`` reproduces the historical suite_hash component."""
    import hashlib
    import json

    suite = _two_axis_suite()
    legacy_key = EpisodeCache.make_legacy_key(
        suite,
        axis_config={"lighting_intensity": 0.5, "camera_offset_x": 0.0},
        seed=1,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="P",
    )
    canonical = {
        "suite_name": suite.name,
        "suite_hash": compute_suite_hash(suite),
        "env_name": "tabletop",
        "policy_id": "P",
        "axis_config": {"lighting_intensity": 0.5, "camera_offset_x": 0.0},
        "seed": 1,
        "episodes_per_cell": 1,
        "max_steps": 20,
        # Mirror ``CACHE_SCHEMA_VERSION`` directly — keeping this test
        # decoupled from the constant so a future bump fails this
        # test loudly.
        "schema_version": "1",
    }
    expected = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert legacy_key == expected


def test_cache_key_invariant_to_axis_dict_order() -> None:
    """``make_key`` is order-invariant on the underlying Suite axes too."""
    suite_a = _two_axis_suite(axis_order=("lighting_intensity", "camera_offset_x"))
    suite_b = _two_axis_suite(axis_order=("camera_offset_x", "lighting_intensity"))
    axis_config: dict[str, float] = {"lighting_intensity": 0.5, "camera_offset_x": 0.0}
    key_a = EpisodeCache.make_key(
        suite_a,
        axis_config=axis_config,
        seed=1,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="P",
    )
    key_b = EpisodeCache.make_key(
        suite_b,
        axis_config=axis_config,
        seed=1,
        episodes_per_cell=1,
        max_steps=20,
        env_name="tabletop",
        policy_id="P",
    )
    assert key_a == key_b


# ----------------------------------------------------------------------------
# 7. CLI — ``gauntlet suite hash <suite.yaml>`` smoke test.
# ----------------------------------------------------------------------------


def test_cli_suite_hash_prints_provenance_digest(tmp_path: Path) -> None:
    """``gauntlet suite hash`` prints the 16-char digest on stdout."""
    yaml_path = tmp_path / "suite.yaml"
    yaml_path.write_text(
        "name: cli-hash-test\n"
        "env: tabletop\n"
        "seed: 1\n"
        "episodes_per_cell: 1\n"
        "axes:\n"
        "  lighting_intensity:\n"
        "    low: 0.5\n"
        "    high: 1.0\n"
        "    steps: 2\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["suite", "hash", str(yaml_path)])
    assert result.exit_code == 0, result.output
    printed = result.stdout.strip()
    assert len(printed) == 16
    assert all(c in "0123456789abcdef" for c in printed)
    # Re-derive in-process — must match what the CLI printed.
    from gauntlet.suite import load_suite

    suite = load_suite(yaml_path)
    expected = compute_suite_provenance_hash(suite)
    assert printed == expected


def test_cli_suite_hash_missing_file_fails_cleanly(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["suite", "hash", str(tmp_path / "no-such.yaml")])
    assert result.exit_code != 0


def test_cli_suite_hash_invalid_yaml_fails_cleanly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    # Missing required ``axes`` field -> validation error.
    bad.write_text("name: bad\nenv: tabletop\nepisodes_per_cell: 1\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["suite", "hash", str(bad)])
    assert result.exit_code != 0


# ----------------------------------------------------------------------------
# 8. Default assets root — sanity check on the in-tree shipped assets.
# ----------------------------------------------------------------------------


def test_default_assets_root_resolves_to_a_real_directory() -> None:
    """The shipped ``src/gauntlet/env/assets`` tree exists and contains files."""
    root = default_assets_root()
    assert root.is_dir(), f"expected env/assets to ship in-tree at {root}"
    shas = compute_env_asset_shas(root)
    assert len(shas) > 0, "no env assets discovered — did the install strip them?"
    # Every value is a 64-char sha256 hex.
    for path, digest in shas.items():
        assert len(digest) == 64, f"asset {path}: digest must be 64-char sha256 hex"
        assert all(c in "0123456789abcdef" for c in digest)
