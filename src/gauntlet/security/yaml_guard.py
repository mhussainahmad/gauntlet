"""Canonical YAML entry point — Phase 2.5 Task 16.

PyYAML exposes both a safe loader (:func:`yaml.safe_load`) and a default
:func:`yaml.load` that, without an explicit ``Loader=`` argument, used
to default to the unsafe full loader. The unsafe loader honours tags
like ``!!python/object/apply`` which materialise arbitrary Python
objects on parse — a textbook RCE class (CVE-2017-18342).

Every YAML read in ``src/gauntlet`` MUST flow through
:func:`safe_yaml_load` so the codebase has exactly one place to audit
for loader policy. The CI grep gate (``tests/test_security_yaml_guard
.py::test_no_unsafe_yaml_load_in_src``) verifies the contract by
asserting that no occurrence of ``yaml.load(`` (without ``safe_``)
exists under ``src/gauntlet`` outside this module.

The wrapper is intentionally a thin pass-through: it calls
:func:`yaml.safe_load` directly and lets every exception
(:class:`yaml.YAMLError` and its subclasses) escape unchanged. The
existing regression tests in ``tests/test_security_yaml.py`` rely on
that envelope and would silently break if the wrapper started catching
and re-wrapping. :class:`YamlSecurityError` is defined here for
symmetry with :class:`gauntlet.security.paths.PathTraversalError` and
as a forward-compatible hook for any strict-mode policy we layer on
top of :func:`yaml.safe_load` in future tasks.
"""

from __future__ import annotations

import re
from typing import IO, Any

import yaml

__all__ = [
    "YamlSecurityError",
    "safe_yaml_load",
]


class YamlSecurityError(ValueError):
    """Raised by future strict-mode YAML policy on top of safe_load.

    Currently unused — :func:`safe_yaml_load` only relays
    :class:`yaml.YAMLError`. Reserved so call-sites can pre-emptively
    catch ``(yaml.YAMLError, YamlSecurityError)`` without a churn-only
    PR when a stricter policy lands.
    """


class _SafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that also reads ``1e-05`` / ``2E3`` as floats.

    PyYAML follows YAML 1.1, whose float pattern needs a decimal point,
    so a suite value written ``1e-05`` (Python's own ``repr`` of 0.00001)
    loaded as the *string* ``"1e-05"`` and failed validation with a
    misleading "string-valued values" error. YAML 1.2 reads it as a
    float; this resolver does the same. Construction is unchanged:
    still only the safe tag set.
    """


# types-PyYAML leaves add_implicit_resolver and dispose unannotated.
_SafeLoader.add_implicit_resolver(  # type: ignore[no-untyped-call]
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def safe_yaml_load(stream: str | bytes | IO[str] | IO[bytes]) -> Any:
    """Parse ``stream`` with PyYAML's safe constructor.

    Equivalent to :func:`yaml.safe_load` except that exponent floats
    without a decimal point (``1e-05``) resolve to ``float`` as in
    YAML 1.2 (see :class:`_SafeLoader`). The wrapper exists so every
    YAML read in ``src/gauntlet`` flows through one place — the CI
    grep gate then asserts no caller side-steps the wrapper by
    importing :func:`yaml.load` directly.

    The return type is :class:`typing.Any` because YAML can produce any
    Python value (mapping, sequence, scalar, ``None``). Callers are
    expected to validate the shape (e.g. via Pydantic) before use; see
    :func:`gauntlet.suite.loader._validate` for the canonical pattern.

    Args:
        stream: A YAML source. Accepts ``str``, ``bytes``, or any
            text / binary file-like object — exactly the surface
            :func:`yaml.safe_load` accepts.

    Returns:
        The parsed Python object. Same shape as
        :func:`yaml.safe_load` would have returned.

    Raises:
        yaml.YAMLError: On any parse failure. Includes
            :class:`yaml.constructor.ConstructorError` for unsafe tags
            (``!!python/object/apply``, ``!!python/object/new``,
            ``!!python/name``).
    """
    # Same steps as ``yaml.safe_load``, with the float-aware loader. The
    # grep gate matches ``yaml.load(`` and stays clean.
    loader = _SafeLoader(stream)
    try:
        result: Any = loader.get_single_data()
    finally:
        loader.dispose()  # type: ignore[no-untyped-call]
    return result
