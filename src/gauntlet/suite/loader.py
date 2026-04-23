"""YAML → :class:`Suite` loaders.

Two entry points:

* :func:`load_suite` — read a YAML file from disk.
* :func:`load_suite_from_string` — parse YAML already in memory (used by
  tests and any future CLI ``--suite-string`` flag).

Both funnel through Pydantic's :meth:`BaseModel.model_validate`, which
turns schema violations into :class:`pydantic.ValidationError`. The
``Any`` returned by :func:`yaml.safe_load` is contained at the boundary
— we validate the raw mapping and hand a typed :class:`Suite` back to
callers.

For backends that live behind optional extras (RFC-005 §11.2), the
loader triggers the canonical module import after pydantic validation
and converts an :class:`ImportError` / :class:`ModuleNotFoundError`
into a user-facing install-hint error. The schema already accepts the
matching ``env:`` key via :data:`BUILTIN_BACKEND_IMPORTS` so the user
sees "extra not installed" rather than "unknown env".
"""

from __future__ import annotations

import functools
import importlib
from pathlib import Path
from typing import Any, cast

import yaml

from gauntlet.env.registry import get_env_factory, registered_envs
from gauntlet.suite.schema import BUILTIN_BACKEND_IMPORTS, Suite

__all__ = [
    "load_suite",
    "load_suite_from_string",
]


_EXTRA_FOR_MODULE: dict[str, str] = {
    "gauntlet.env.pybullet": "pybullet",
    "gauntlet.env.genesis": "genesis",
}


def _visual_only_axes_of(factory: Any) -> frozenset[str]:
    """Best-effort lookup of a backend's VISUAL_ONLY_AXES ClassVar.

    Handles the two common factory shapes:
    * a class (``register_env("tabletop", TabletopEnv)``) — read the attr
      directly off the type.
    * a :func:`functools.partial` over a class (the CLI's
      ``--env-max-steps`` path) — unwrap via ``partial.func``.

    Anything more exotic (a random ``Callable`` with no static hook)
    degrades to an empty frozenset so the check becomes a no-op rather
    than a false positive.
    """
    if isinstance(factory, functools.partial):
        return _visual_only_axes_of(factory.func)
    attr = getattr(factory, "VISUAL_ONLY_AXES", None)
    if isinstance(attr, frozenset):
        return cast(frozenset[str], attr)
    return frozenset()


def load_suite(path: Path | str) -> Suite:
    """Load and validate a suite from a YAML file.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        pydantic.ValidationError: if the YAML contents fail validation.
        yaml.YAMLError: if the file is not valid YAML.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"suite file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    return _validate(raw, source=str(p))


def load_suite_from_string(yaml_text: str) -> Suite:
    """Parse and validate a suite from a YAML string.

    Raises:
        pydantic.ValidationError: if the YAML contents fail validation.
        yaml.YAMLError: if the string is not valid YAML.
    """
    raw: Any = yaml.safe_load(yaml_text)
    return _validate(raw, source="<string>")


def _validate(raw: Any, *, source: str) -> Suite:
    """Convert the ``Any`` from :func:`yaml.safe_load` into a typed Suite.

    The top-level YAML document must be a mapping — anything else
    (scalar, list, null) is rejected before Pydantic ever sees it so the
    error mentions the source path rather than field-level noise.
    """
    if raw is None:
        raise ValueError(f"suite YAML is empty: {source}")
    if not isinstance(raw, dict):
        raise ValueError(
            f"suite YAML must be a mapping at the top level; got {type(raw).__name__} in {source}",
        )
    # At this point the runtime shape is dict, but the value types are
    # still Any under the hood (YAML can produce anything). Pydantic's
    # validation pipeline narrows everything field-by-field, so we cast
    # once here to hand off a typed mapping.
    data = cast(dict[str, Any], raw)
    suite = Suite.model_validate(data)
    _ensure_backend_registered(suite.env)
    _reject_purely_visual_suites(suite)
    return suite


def _reject_purely_visual_suites(suite: Suite) -> None:
    """Reject a suite whose every axis is in the backend's VISUAL_ONLY_AXES.

    RFC-005 §6.2 / §12 Q1 — a state-only backend cannot produce any
    observable change from axes that only mutate rendered scene
    content (e.g. ``lighting_intensity``, ``object_texture`` on the
    PyBullet first cut). Running such a sweep would emit
    pairwise-identical cell results and silently look like a broken
    harness; rejecting at load time is loud by design.

    The check is a no-op on backends that declare an empty
    ``VISUAL_ONLY_AXES`` (e.g. MuJoCo ``TabletopEnv`` — the renderer
    consumes those axes through ``render_in_obs`` adapters).
    """
    factory = get_env_factory(suite.env)
    visual_only = _visual_only_axes_of(factory)
    if not visual_only:
        return
    declared = set(suite.axes.keys())
    if declared and declared <= visual_only:
        raise ValueError(
            f"suite {suite.name!r} (env={suite.env!r}): every declared axis "
            f"is cosmetic on a state-only backend. These axes mutate the "
            f"PyBullet scene but do not change state-only observations, so "
            f"every cell would report identical success rates. Declared: "
            f"{sorted(declared)}; cosmetic (VISUAL_ONLY) on this backend: "
            f"{sorted(visual_only)}. Add at least one state-effecting axis "
            f"(e.g. object_initial_pose_x / _y, distractor_count) or wait "
            f"for the image-rendering follow-up RFC."
        )


def _ensure_backend_registered(env_name: str) -> None:
    """Import the subpackage that registers ``env_name`` if needed.

    The Suite schema accepts any key in :data:`BUILTIN_BACKEND_IMPORTS`
    as a valid ``env:`` — those backends register themselves when their
    subpackage is imported. We trigger that import here, ONCE, at load
    time so worker processes inherit the populated registry via fork.

    Behaviour:

    * Already registered → no-op.
    * Known built-in → attempt the matching import. On success, expect
      the module's ``__init__`` to call ``register_env`` and confirm.
    * :class:`ImportError` / :class:`ModuleNotFoundError` → re-raise as
      a clear user-facing error with the ``uv sync --extra X`` hint.
    * Unknown name after a successful import → raise a generic
      ``unknown env`` error listing the currently-registered set
      (defence-in-depth; schema should have rejected this already).
    """
    if env_name in registered_envs():
        return

    module_path = BUILTIN_BACKEND_IMPORTS.get(env_name)
    if module_path is None:
        # Defence in depth — schema validation should have rejected this
        # already, but a misuse (direct Suite() construction bypassing
        # validate) must still surface a clear error.
        raise ValueError(
            f"unknown env {env_name!r}; registered: {sorted(registered_envs())}",
        )

    extra = _EXTRA_FOR_MODULE.get(module_path, env_name)
    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"env {env_name!r}: the matching extra is not installed.\n"
            f"Install with:\n"
            f"    uv sync --extra {extra}\n"
            f"or, for a plain pip env:\n"
            f"    pip install 'gauntlet[{extra}]'",
        ) from exc

    if env_name not in registered_envs():
        raise ValueError(
            f"env {env_name!r}: backend module {module_path!r} imported "
            f"but did not call register_env({env_name!r}, ...). "
            f"This is a backend-packaging bug.",
        )
