# ruff: noqa: T201 -- the --check mode reports a stale file on stdout
"""Write the open JSON Schema of facility.json.

``osprey_facility/schema/_generated/facility.schema.json`` (current) is the JSON
Schema of the generated pydantic ``Facility`` model, opened up so a reader
accepts keys a later additive change introduces: every
``additionalProperties: false`` is removed, the ``linkml_meta`` annotations are
dropped, and the ``schema`` property is a ``const`` equal to ``FACILITY_HEADER``.
Keys are sorted, so the same model always writes the same bytes.

Current carries its identity: ``$schema`` names the JSON Schema dialect and
``$id`` the URL the docs serve it under, ending in ``<N>.schema.json`` for the
header's N. Since current only ever grows additively within one N, the ``$id``
names the newest additive form of N.

Beside current lies one floor per header version,
``facility-<N>.schema.json``: the schema as it stood when N was introduced,
without ``$schema`` or ``$id`` (a floor is a comparison base, not a published
identity). The script writes the floor for the current N only when it is
missing, and never rewrites or removes a floor; a test checks that current is
an additive extension of the floor for its N (``osprey_facility.schema.compat``).
At an N bump the current on disk, which still carries the old N's ``$id``, is
frozen as ``facility-<old N>.final.schema.json`` before the new current
replaces it, so the old URL keeps serving the last form of the old N.
``--check`` writes nothing and fails when current is stale or the floor for the
current N is missing.

Usage::

    uv run python -m osprey_facility.schema._generate           # write
    uv run python -m osprey_facility.schema._generate --check   # compare
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: The command that runs this module, named in the ``--check`` message.
COMMAND = "python -m osprey_facility.schema._generate"

OUTPUT_DIR = Path(__file__).resolve().parent / "_generated"

#: The file name of the current schema.
CURRENT = "facility.schema.json"

#: The JSON Schema dialect current declares in ``$schema``.
DIALECT = "https://json-schema.org/draft/2020-12/schema"

#: The URL the docs serve the schemas under; ``$id`` is ``<IDENTITY_BASE>/<N>.schema.json``.
IDENTITY_BASE = "https://als-apg.github.io/osprey/latest/schemas/facility"

#: The root keywords that give a schema its identity; floors carry neither, and
#: ``compat.classify`` skips them.
IDENTITY_KEYS = ("$schema", "$id")

#: Keywords whose value maps names to subschemas (names, not keywords, as keys).
_SCHEMA_MAPS = frozenset({"properties", "$defs"})

#: Keywords whose value is an instance, not a schema: passed through untouched.
_INSTANCE_VALUES = frozenset({"default", "const", "enum", "examples"})


def header_version(header: str) -> int:
    """Return N of a header ``osprey.facility.facility/<N>``."""
    version = header.rsplit("/", 1)[-1]
    if not version.isdigit():
        raise ValueError(f"header {header!r} does not end in /<N>")
    return int(version)


def floor_name(header: str) -> str:
    """Return the floor file name for a header ``osprey.facility.facility/<N>``."""
    return f"facility-{header_version(header)}.schema.json"


def final_name(version: int) -> str:
    """Return the file name of the last current of header version ``version``."""
    return f"facility-{version}.final.schema.json"


def schema_id(version: int) -> str:
    """Return the ``$id`` of the schema of header version ``version``."""
    return f"{IDENTITY_BASE}/{version}.schema.json"


def without_identity(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``schema`` without its root ``$schema`` and ``$id``."""
    return {key: value for key, value in schema.items() if key not in IDENTITY_KEYS}


def _open(node: Any) -> Any:
    """Return ``node`` with every closed object opened and ``linkml_meta`` dropped.

    Walks schema positions only: inside a ``properties`` or ``$defs`` map the keys
    are names, so a property called ``linkml_meta`` or ``additionalProperties``
    would be kept, and the value of ``default``, ``const``, ``enum`` or
    ``examples`` is an instance, which is kept as it is.
    """
    if isinstance(node, list):
        return [_open(item) for item in node]
    if not isinstance(node, dict):
        return node
    opened: dict[str, Any] = {}
    for key, value in node.items():
        if key == "linkml_meta" or (key == "additionalProperties" and value is False):
            continue
        if key in _INSTANCE_VALUES:
            opened[key] = value
        elif key in _SCHEMA_MAPS and isinstance(value, dict):
            opened[key] = {name: _open(sub) for name, sub in value.items()}
        else:
            opened[key] = _open(value)
    return opened


def build(header: str | None = None) -> dict[str, Any]:
    """Return the open JSON Schema of facility.json, with its ``$schema`` and ``$id``.

    Args:
        header: The ``schema`` const; ``None`` uses ``FACILITY_HEADER``.
    """
    from osprey_facility.schema import Facility

    if header is None:
        from osprey_facility.validate import FACILITY_HEADER

        header = FACILITY_HEADER
    schema: dict[str, Any] = _open(Facility.model_json_schema())
    schema["properties"]["schema"]["const"] = header
    schema["$schema"] = DIALECT
    schema["$id"] = schema_id(header_version(header))
    return schema


def render(schema: dict[str, Any]) -> str:
    """Serialise a schema deterministically, with a trailing newline."""
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write current and a missing floor, or with ``--check`` report whether both are in place.

    Before current is rewritten for a new N, the current on disk is frozen as
    the final of the N its ``$id`` names, unless that final already exists.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        0 when the files were written or are in place, 1 when ``--check`` finds
        current stale or the floor for the current N missing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of write")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="directory of the schema files"
    )
    args = parser.parse_args(argv)
    from osprey_facility.validate import FACILITY_HEADER

    schema = build(FACILITY_HEADER)
    text = render(schema)
    current = args.output_dir / CURRENT
    floor = args.output_dir / floor_name(FACILITY_HEADER)
    if args.check:
        stale = []
        if not current.exists() or current.read_text(encoding="utf-8") != text:
            stale.append(f"{current} is stale")
        if not floor.exists():
            stale.append(f"{floor} is missing")
        for line in stale:
            print(f"{line}; run `{COMMAND}`")
        return 1 if stale else 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _freeze_final(current, schema["$id"])
    current.write_text(text, encoding="utf-8")
    if not floor.exists():
        floor.write_text(render(without_identity(schema)), encoding="utf-8")
    return 0


def _freeze_final(current: Path, new_id: str) -> None:
    """Keep the current on disk as the final of its N when the new current names another N.

    A current without ``$id`` predates published identity and has no URL to
    keep serving, so nothing is frozen for it.
    """
    if not current.exists():
        return
    old_text = current.read_text(encoding="utf-8")
    old_id = json.loads(old_text).get("$id")
    if old_id is None or old_id == new_id:
        return
    prefix, suffix = f"{IDENTITY_BASE}/", ".schema.json"
    version = old_id.removeprefix(prefix).removesuffix(suffix)
    if not (old_id.startswith(prefix) and old_id.endswith(suffix) and version.isdigit()):
        raise ValueError(f"{current} has an $id outside {IDENTITY_BASE}: {old_id!r}")
    final = current.parent / final_name(int(version))
    if not final.exists():
        final.write_text(old_text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
