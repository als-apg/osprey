"""A JSDoc typedef that names a server schema matches that schema field for field.

The dashboard's browser code describes the bodies it receives as JSDoc
typedefs, which ``tsc`` holds the code to. The server owns those shapes, and
``openapi.json`` beside this file pins them. A typedef whose name is a schema
in that snapshot must carry the same property names, the same required set,
the same nullability per property and the same scalar type, so a field the
server adds or makes nullable fails here instead of reaching the page as an
untyped surprise.

Typedefs whose names are not schemas (callback bundles, browser-only shapes)
are not checked. A typedef that names a schema must use the ``{object}`` form
with ``@property`` lines, the only form this test can read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_SNAPSHOT = Path(__file__).with_name("openapi.json")
_JS_DIR = (
    Path(__file__).parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "lattice_dashboard"
    / "static"
    / "js"
)

_BLOCK_RE = re.compile(r"/\*\*(.*?)\*/", re.DOTALL)
_TYPEDEF_RE = re.compile(r"@typedef\s+\{(?P<type>.*?)\}\s+(?P<name>\w+)")
_PROPERTY_RE = re.compile(
    r"@property\s+\{(?P<type>[^}]*)\}\s+(?:\[(?P<optional>\w+)\]|(?P<required>\w+))"
)

# A JSDoc scalar type and the JSON-schema scalar type it restates.
_SCALARS = {"string": "string", "boolean": "boolean", "number": "number", "integer": "number"}


@dataclass(frozen=True)
class _Typedef:
    name: str
    form: str
    source: str
    properties: dict[str, str]
    required: frozenset[str]


def _schemas() -> dict[str, Any]:
    return json.loads(_SNAPSHOT.read_text())["components"]["schemas"]


def _typedefs() -> list[_Typedef]:
    found = []
    for path in sorted(_JS_DIR.glob("*.js")):
        for block in _BLOCK_RE.findall(path.read_text()):
            head = _TYPEDEF_RE.search(block)
            if head is None:
                continue
            properties: dict[str, str] = {}
            required = set()
            for prop in _PROPERTY_RE.finditer(block):
                name = prop["required"] or prop["optional"]
                properties[name] = prop["type"].strip()
                if prop["required"]:
                    required.add(name)
            found.append(
                _Typedef(
                    name=head["name"],
                    form=head["type"].strip(),
                    source=path.name,
                    properties=properties,
                    required=frozenset(required),
                )
            )
    return found


def _schema_alternatives(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return spec.get("anyOf", [spec])


def _js_alternatives(js_type: str) -> list[str]:
    return [part.strip() for part in js_type.split("|")]


_SCHEMA_NAMES = set(_schemas())
_MATCHED = [t for t in _typedefs() if t.form == "object" and t.name in _SCHEMA_NAMES]


@pytest.mark.parametrize("typedef", _MATCHED, ids=[t.name for t in _MATCHED])
def test_typedef_matches_schema(typedef: _Typedef) -> None:
    schema = _schemas()[typedef.name]
    where = f"{typedef.source} @typedef {typedef.name}"
    fields = schema.get("properties", {})

    assert set(typedef.properties) == set(fields), (
        f"{where}: properties differ from the server schema "
        f"(missing {sorted(set(fields) - set(typedef.properties))}, "
        f"extra {sorted(set(typedef.properties) - set(fields))})"
    )
    assert typedef.required == set(schema.get("required", [])), (
        f"{where}: required properties differ from the server schema "
        f"(bracket a property the server may omit, unbracket one it always sends)"
    )
    for name, spec in fields.items():
        js = _js_alternatives(typedef.properties[name])
        alternatives = _schema_alternatives(spec)
        schema_nullable = {"type": "null"} in alternatives
        assert ("null" in js) == schema_nullable, (
            f"{where}: property {name!r} is {'nullable' if schema_nullable else 'never null'} "
            f"on the server, the typedef says {typedef.properties[name]!r}"
        )
        schema_scalars = {
            alt["type"]
            for alt in alternatives
            if alt.get("type") in ("string", "boolean", "number", "integer") and "enum" not in alt
        }
        if not schema_scalars:
            continue
        js_scalars = {_SCALARS[part] for part in js if part in _SCALARS}
        assert js_scalars == {_SCALARS[s] for s in schema_scalars}, (
            f"{where}: property {name!r} has scalar type {sorted(schema_scalars)} on the server, "
            f"the typedef says {typedef.properties[name]!r}"
        )


def test_schema_named_typedefs_use_the_object_form() -> None:
    inline = [
        f"{t.source} @typedef {t.name}"
        for t in _typedefs()
        if t.name in _SCHEMA_NAMES and t.form != "object"
    ]
    assert not inline, (
        f"write these as '@typedef {{object}} Name' with @property lines so their fields "
        f"are checked against the server schema: {inline}"
    )


def test_at_least_one_typedef_is_checked() -> None:
    assert _MATCHED, "no JSDoc typedef names a server schema, so nothing was checked"
