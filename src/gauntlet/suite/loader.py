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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from gauntlet.suite.schema import Suite

__all__ = [
    "load_suite",
    "load_suite_from_string",
]


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
    return Suite.model_validate(data)
