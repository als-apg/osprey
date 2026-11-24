"""The facility file's schema and the pydantic model generated from it.

``Facility`` validates one whole facility file; a key the schema does not
declare fails validation.

:func:`published_schema` hands out the published JSON Schema the package
ships: the current one, or the floor of an earlier format version. A tool
that checks a facility.json without the framework reads it from here.
"""

from __future__ import annotations

import warnings
from importlib.resources import files
from importlib.resources.abc import Traversable

# The root class declares the facility file's ``schema`` header slot, which
# shadows pydantic's ``BaseModel.schema`` method; that one warning is expected.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r'Field name "schema" in "Facility" shadows an attribute in parent',
    )
    from osprey_facility.schema._generated import core

Facility = core.Facility

#: The installed package's directory of generated schema files.
_GENERATED = "osprey_facility.schema._generated"


def published_schema(version: int | None = None) -> Traversable:
    """The published JSON Schema file of the facility format.

    ``None`` is the current schema (``facility.schema.json``); an integer is
    the frozen floor of that format version (``facility-<N>.schema.json``).
    Raises :class:`LookupError` for a version the package ships no schema for.
    """
    if version is None:
        name = "facility.schema.json"
    elif isinstance(version, int) and not isinstance(version, bool):
        name = f"facility-{version}.schema.json"
    else:
        name = ""
    schema = files(_GENERATED).joinpath(name) if name else None
    if schema is None or not schema.is_file():
        raise LookupError(f"osprey-facility ships no JSON Schema for format version {version!r}")
    return schema


__all__ = ["Facility", "core", "published_schema"]
