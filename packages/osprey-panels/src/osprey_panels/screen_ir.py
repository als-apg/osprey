"""Screen-IR: the version the panels write and the schema their screens must pass.

:data:`SCREEN_IR_VERSION` is the one place the engine names the Screen-IR version;
:mod:`~osprey_panels.layout` stamps it into every document it draws. :func:`schema`
is the Screen-IR JSON Schema of the pinned PyDMConverter, vendored beside this module
as ``screen-ir.schema.json`` and read as package data. :func:`problems` checks a
document against it.

The vendored file is byte-identical to ``pydmconverter/ir/data/screen-ir.schema.json``
at the PyDMConverter commit OSPREY's Canopy contract pins
(``osprey.integrations.canopy.proven.PROVEN_COMMITS["PyDMConverter"]``). It was taken,
in a clone of ``proven.MIRRORS["PyDMConverter"]``, with::

    git show 4b8183f39d0ff5c1c87a33f50fc438f9fd63e97f:pydmconverter/ir/data/screen-ir.schema.json \\
        > packages/osprey-panels/src/osprey_panels/screen-ir.schema.json

When the pin moves, take the file again the same way.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

# jsonschema ships no type information and the environment has no stub distribution for it.
from jsonschema.validators import validator_for  # type: ignore[import-untyped]

__all__ = ["SCREEN_IR_VERSION", "problems", "schema"]

#: The Screen-IR version of every document the panels write.
SCREEN_IR_VERSION = "1.1"

#: The vendored schema's file name inside the package.
_SCHEMA_FILE = "screen-ir.schema.json"


@cache
def _schema_text() -> str:
    return files("osprey_panels").joinpath(_SCHEMA_FILE).read_text(encoding="utf-8")


def schema() -> dict[str, Any]:
    """The vendored Screen-IR JSON Schema, a fresh copy on every call."""
    loaded: dict[str, Any] = json.loads(_schema_text())
    return loaded


@cache
def _validator() -> Any:
    document = schema()
    cls = validator_for(document)
    cls.check_schema(document)
    return cls(document)


def problems(doc: dict[str, Any]) -> list[str]:
    """What keeps *doc* from being a valid Screen-IR document; empty when it is valid.

    Validated with the validator of the draft the schema declares. Each problem is
    one line, the JSON path of the failing part then the schema's message, sorted.
    """
    return sorted(f"{error.json_path}: {error.message}" for error in _validator().iter_errors(doc))
