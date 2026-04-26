"""YAML → :class:`Suite` loaders.

Two entry points:

* :func:`load_suite` — read a YAML file from disk.
* :func:`load_suite_from_string` — parse YAML already in memory (used by
  tests and any future CLI ``--suite-string`` flag).

Both funnel through Pydantic's :meth:`BaseModel.model_validate`, which
turns schema violations into :class:`pydantic.ValidationError`. The
``Any`` returned by :func:`gauntlet.security.safe_yaml_load` (a thin
alias of :func:`yaml.safe_load`) is contained at the boundary — we
validate the raw mapping and hand a typed :class:`Suite` back to
callers. Routing every read through :mod:`gauntlet.security.yaml_guard`
gives the CI grep gate a single canonical call-site to verify.

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
import warnings
from pathlib import Path
from typing import Any, cast

from gauntlet.env.registry import get_env_factory, registered_envs
from gauntlet.security import safe_yaml_load
from gauntlet.suite.schema import BUILTIN_BACKEND_IMPORTS, Suite

__all__ = [
    "load_suite",
    "load_suite_from_string",
]


_EXTRA_FOR_MODULE: dict[str, str] = {
    "gauntlet.env.pybullet": "pybullet",
    "gauntlet.env.genesis": "genesis",
    "gauntlet.env.isaac": "isaac",
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
        raw: Any = safe_yaml_load(fh)
    return _validate(raw, source=str(p))


def load_suite_from_string(yaml_text: str) -> Suite:
    """Parse and validate a suite from a YAML string.

    Raises:
        pydantic.ValidationError: if the YAML contents fail validation.
        yaml.YAMLError: if the string is not valid YAML.
    """
    raw: Any = safe_yaml_load(yaml_text)
    return _validate(raw, source="<string>")


def _validate(raw: Any, *, source: str) -> Suite:
    """Convert the ``Any`` from :func:`safe_yaml_load` into a typed Suite.

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
    suite = _resolve_and_check_pilot_report(suite, source=source)
    suite = _resolve_camera_extrinsics_range(suite)
    return suite


def _resolve_camera_extrinsics_range(suite: Suite) -> Suite:
    """B-42 — pre-expand ``extrinsics_range`` into ``extrinsics_values``.

    The range form is a Sobol-friendly six-dimensional continuous
    shape; the loader resolves it here at suite-load time using the
    Joe-Kuo 6.21201 sequence so downstream samplers see the
    ``camera_extrinsics`` axis as a plain categorical with N entries.
    The Sobol sequence is fully deterministic, so the resolution is
    bit-identical across runs of the same YAML regardless of
    ``Suite.seed``.

    The schema validator already enforces:
    * ``extrinsics_range`` is forbidden on every axis except
      ``camera_extrinsics``;
    * ``extrinsics_range`` is forbidden on ``sampling=cartesian``.

    The remaining contract is that ``Suite.n_samples`` must be set
    (it is, on every non-cartesian sampling mode — that validator
    already runs).

    No-op when no ``camera_extrinsics`` axis carries a range.
    """
    spec = suite.axes.get("camera_extrinsics")
    if spec is None or spec.extrinsics_range is None:
        return suite

    # Local imports keep the loader cheap on the common case.
    import numpy as np

    from gauntlet.suite.schema import AxisSpec, ExtrinsicsValue
    from gauntlet.suite.sobol import sobol_unit_cube

    n = suite.n_samples
    # The schema validator pairs ``extrinsics_range`` with non-cartesian
    # sampling, and the cross-field validator ensures n_samples is set
    # on every non-cartesian sampling. Defence in depth: assert it.
    assert n is not None, "extrinsics_range requires Suite.n_samples"

    rng_box = spec.extrinsics_range
    # Build the per-dim [lo, hi] table in the order:
    # translation x, y, z, rotation x, y, z. This is the same order
    # the env's ``set_camera_extrinsics_list`` consumes — keep it
    # stable across the codebase.
    bounds: list[list[float]] = list(rng_box.translation) + list(rng_box.rotation)
    unit = sobol_unit_cube(n, 6, skip=1)
    entries: list[ExtrinsicsValue] = []
    for row_idx in range(n):
        scaled = np.empty(6, dtype=np.float64)
        for col_idx, (lo, hi) in enumerate(bounds):
            scaled[col_idx] = lo + float(unit[row_idx, col_idx]) * (hi - lo)
        entries.append(
            ExtrinsicsValue(
                translation=[float(scaled[0]), float(scaled[1]), float(scaled[2])],
                rotation=[float(scaled[3]), float(scaled[4]), float(scaled[5])],
            )
        )

    new_spec = AxisSpec.model_validate(
        {
            "extrinsics_values": [e.model_dump() for e in entries],
        }
    )
    new_axes = dict(suite.axes)
    new_axes["camera_extrinsics"] = new_spec
    return suite.model_copy(update={"axes": new_axes})


def _resolve_and_check_pilot_report(suite: Suite, *, source: str) -> Suite:
    """B-07 — resolve a relative ``pilot_report`` and emit the loud warning.

    When ``sampling == "adversarial"``:

    * If ``pilot_report`` is a relative path and ``source`` points at
      a real file, resolve it against the YAML's parent directory.
      String-loaded suites (``source == "<string>"``) treat the path
      as cwd-relative — documented behaviour, not a bug.
    * Verify the file exists and parses as a
      :class:`gauntlet.report.schema.Report`. Loading at suite-load
      time (rather than first ``cells()`` call) makes
      ``gauntlet suite check`` catch broken pilots without running
      the harness.
    * Emit :class:`UserWarning` with :data:`ADVERSARIAL_WARNING` so
      every consumer (CLI, scripts, tests) sees the same anti-feature
      caveat once per load.

    Returns a Suite with ``pilot_report`` rewritten to the resolved
    absolute path so downstream callers (the sampler) do not have to
    repeat the resolution.
    """
    if suite.sampling != "adversarial":
        return suite

    # Local import keeps the loader's startup cost flat for the
    # cartesian/LHS/Sobol common case.
    from gauntlet.suite.adversarial import ADVERSARIAL_WARNING, load_pilot_report

    assert suite.pilot_report is not None  # validator enforces this.
    pilot_path = Path(suite.pilot_report)
    if not pilot_path.is_absolute() and source != "<string>":
        pilot_path = (Path(source).parent / pilot_path).resolve()

    # Loud anti-feature warning — emitted at load time so every entry
    # path (CLI, suite check, direct scripts) carries the caveat.
    warnings.warn(ADVERSARIAL_WARNING, UserWarning, stacklevel=2)

    # Load-time parse: surfaces a malformed / missing pilot at
    # ``gauntlet suite check`` time, not three cells into a 200-cell
    # run. The Report itself is discarded; the sampler re-loads it on
    # ``Suite.cells()`` so the load result is not pinned to memory.
    load_pilot_report(pilot_path)

    return suite.model_copy(update={"pilot_report": str(pilot_path)})


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
