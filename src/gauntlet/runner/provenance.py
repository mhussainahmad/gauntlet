"""Provenance capture for ``repro.json`` (B-22) and suite-level cache keys (B-40).

Every :class:`gauntlet.runner.Episode` produced by :class:`Runner` is
stamped with three provenance fields so the run can be re-executed
bit-identically on a fresh checkout:

* ``gauntlet_version`` — the installed distribution version.
* ``suite_hash`` — sha256 of the canonical Pydantic-serialised Suite.
* ``git_commit`` — ``git rev-parse HEAD`` if the working copy is in a
  git checkout, else ``None``.

The capture is intentionally fail-soft: a missing distribution, a
non-git checkout, or a misconfigured ``git`` binary all degrade to
``None`` so a non-installed test runner or a packaged tarball can still
emit a valid ``repro.json`` (with the matching field empty).

B-40 adds a *suite-level provenance hash* — a deterministic 16-char
blake2b digest over ``(canonical Suite payload, gauntlet version,
``hash_version``, env asset SHAs)``. It is used as the on-disk cache
key so two suites that differ only in YAML formatting (key order,
whitespace) cache-hit, while a gauntlet version bump or an asset edit
correctly invalidates. See the ``vla-eval`` paper (arxiv 2603.13966) for
the design precedent: content-addressed config hashes are the standard
fix for cross-paper irreproducibility, and gauntlet's per-episode B-22
fingerprint is too noisy (sha256, full digest) to use as a result-cache
key on its own.

ANTI-FEATURE: the hash bakes in ``gauntlet.__version__`` *by design*,
so a version bump invalidates every cached run. This is the price of
honesty — a silently-stale cache after a behaviour-changing release is
exactly the cross-paper failure mode the paper warns about. The
``hash_version`` integer (bumped only on canonicalisation-format
changes) gives us a graceful "stale cache, re-run" path: a cache entry
written under a different ``hash_version`` simply misses, the runner
re-rolls and overwrites under the new key, and no operator
intervention is required.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from gauntlet.suite.schema import Suite

__all__ = [
    "SUITE_PROVENANCE_HASH_VERSION",
    "capture_gauntlet_version",
    "capture_git_commit",
    "compute_env_asset_shas",
    "compute_suite_hash",
    "compute_suite_provenance_hash",
    "default_assets_root",
]


# Bumped only when the canonicalisation format itself changes (e.g. a
# new field is folded into the digest payload, or the encoding changes
# from JSON to something else). Bumping here invalidates every cache
# entry written under the old version, which is the intended graceful
# "stale cache, re-run" path called out in the docstring above.
#
# This is deliberately a module-level constant — *not* a Pydantic field
# on :class:`gauntlet.suite.schema.Suite`. A user-editable YAML field
# would be a footgun (a YAML could declare an old ``hash_version`` and
# silently stay compatible while the canonicalisation actually
# changed).
SUITE_PROVENANCE_HASH_VERSION = 1


def default_assets_root() -> Path:
    """Return the on-disk path to ``src/gauntlet/env/assets``.

    Resolved relative to ``gauntlet.env`` so an editable install, a
    packaged wheel, and a sys.path checkout all locate the same tree.
    Returns the bare ``Path`` even if the directory does not exist —
    callers (e.g. :func:`compute_env_asset_shas`) tolerate a missing
    root and emit an empty mapping.
    """
    from gauntlet import env as _env_pkg

    env_pkg_path = Path(_env_pkg.__file__).resolve().parent
    return env_pkg_path / "assets"


def capture_gauntlet_version() -> str | None:
    """Return the installed ``gauntlet`` distribution version, or ``None``.

    ``None`` is returned when the package is not importable as an
    installed distribution (e.g. a sys.path checkout without
    ``pip install -e .``). Wrapped in a try/except so a partially-set-up
    test environment never breaks ``Runner.run``.
    """
    try:
        return version("gauntlet")
    except PackageNotFoundError:
        return None


def compute_suite_hash(suite: Suite) -> str:
    """SHA-256 of the canonical Suite payload.

    Hashes :meth:`Suite.model_dump_json` with ``sort_keys`` semantics
    (Pydantic emits the declared field order; ``sort_keys=True`` on the
    encoded JSON makes the result invariant to future field-order
    changes). Stable across YAML reformatting because the input is the
    validated Pydantic model, not the raw file bytes.
    """
    # ``model_dump_json`` already produces a deterministic byte string
    # for a given model (Pydantic emits keys in declaration order).
    payload = suite.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_git_commit(cwd: Path | None = None) -> str | None:
    """Return ``git rev-parse HEAD`` in *cwd*, or ``None``.

    Fail-soft: returns ``None`` when ``git`` is missing, when ``cwd`` is
    not a git checkout, when the subprocess times out, or when the
    return code is non-zero. Never raises.

    A 2-second timeout is intentionally short — a healthy ``git
    rev-parse`` returns in milliseconds; anything slower means a
    locked index or a network-mounted ``.git`` and is not worth
    blocking the rollout for.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def compute_env_asset_shas(assets_root: Path | None = None) -> dict[str, str]:
    """SHA-256 every regular file under *assets_root*, keyed by relative POSIX path.

    The returned mapping is built by walking the asset tree in sorted
    order (``rglob('*')`` results are sorted before hashing), so the
    output is deterministic across filesystems whose iteration order is
    not. Symlinks and non-regular files are skipped — only what the
    simulator actually reads off disk.

    Tradeoff: every asset under :func:`default_assets_root` is hashed,
    not just the subset a given suite actually loads. This keeps the
    implementation simple (no need to plumb suite -> asset list
    through every backend) and means an unrelated asset edit (say, a
    new ``objects/cup.xml``) invalidates every suite's cache. The
    invalidation is the *intentional* behaviour: a full simulator
    rebuild can change the loaded scene even when the suite YAML is
    untouched, and the alternative — silently re-using stale cached
    rollouts — is the cross-paper failure mode B-40 exists to prevent.

    Returns an empty dict (NOT a raise) when *assets_root* does not
    exist or is empty, so a packaged wheel that strips the assets tree
    still produces a well-defined hash.

    Args:
        assets_root: Directory to walk. Defaults to
            :func:`default_assets_root` (the in-tree
            ``src/gauntlet/env/assets`` directory).

    Returns:
        ``{relative_posix_path: sha256_hex}`` for every regular file
        under *assets_root*, sorted by key.
    """
    root = assets_root if assets_root is not None else default_assets_root()
    if not root.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def compute_suite_provenance_hash(
    suite: Suite,
    *,
    gauntlet_version: str | None = None,
    assets_root: Path | None = None,
) -> str:
    """16-char blake2b hash of ``(canonical Suite, gauntlet version, asset SHAs)``.

    The B-40 suite-level fingerprint. Used as the on-disk cache key in
    :class:`gauntlet.runner.cache.EpisodeCache` so two suites with
    identical semantics (just different YAML key ordering or
    whitespace) cache-hit, while a gauntlet version bump or an asset
    edit correctly invalidates.

    The canonical payload is built as::

        {
            "hash_version": SUITE_PROVENANCE_HASH_VERSION,
            "gauntlet_version": <version or "unknown">,
            "suite": <suite.model_dump(mode="json")>,
            "env_asset_shas": <{rel_path: sha256_hex}>,
        }

    and serialised via ``json.dumps(..., sort_keys=True,
    separators=(",", ":"), ensure_ascii=True)``. Sorting keys recursively
    flattens any YAML key-ordering differences (Pydantic preserves
    declared field order, but the recursive ``sort_keys`` of the
    encoded ``model_dump`` payload makes the digest invariant to
    nested dict insertion order too — including the per-axis spec
    fields and the ``axes`` mapping itself). ASCII-only encoding
    sidesteps locale-dependent unicode normalisation.

    The digest uses ``blake2b(digest_size=8)`` -> 16-hex-char output.
    blake2b is faster than sha256 and gives plenty of collision
    resistance for a per-suite cache key (the alternative — full
    sha256 — is what :func:`compute_suite_hash` already provides for
    the per-episode B-22 provenance field, where a longer digest is
    appropriate). Truncating sha256 would invite confusion with the
    B-22 hash; using a different algorithm makes the two functions
    visually distinguishable in logs and JSON output.

    The ``axis values sorted`` clause from the B-40 spec is satisfied
    *only* at the dict-key level (axis names): per-axis ``values``
    lists are NOT sorted, because the index semantics of B-05
    (``instruction_paraphrase``) and B-06 (``object_swap``) attach
    meaning to position. Two suites that differ only in axis-name
    insertion order hash identically; two that differ in
    ``values: [a, b]`` vs ``values: [b, a]`` do NOT (and should not).

    Args:
        suite: Validated :class:`Suite` to fingerprint.
        gauntlet_version: Override the captured gauntlet version.
            Defaults to :func:`capture_gauntlet_version`'s result, or
            the literal ``"unknown"`` when the package is not
            installed (keeps the digest well-defined for sys.path
            checkouts).
        assets_root: Override the env asset directory. Defaults to
            :func:`default_assets_root`. Pass an explicit path in
            tests that fixture-up a controlled asset tree.

    Returns:
        16-character lowercase hex digest, deterministic across
        Python invocations and platforms.
    """
    version_str = gauntlet_version if gauntlet_version is not None else capture_gauntlet_version()
    payload: dict[str, object] = {
        "hash_version": SUITE_PROVENANCE_HASH_VERSION,
        "gauntlet_version": version_str if version_str is not None else "unknown",
        "suite": suite.model_dump(mode="json"),
        "env_asset_shas": compute_env_asset_shas(assets_root),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()
