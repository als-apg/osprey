# ruff: noqa: T201 -- the --check mode reports a stale file on stdout
"""Write the open JSON Schema of facility.json.

``osprey_facility/schema/_generated/facility.schema.json`` (current) is the JSON
Schema of the generated pydantic ``Facility`` model, opened up so a reader
accepts keys a later additive change introduces: every
``additionalProperties: false`` is removed, the ``linkml_meta`` annotations are
dropped, and the ``schema`` property is a ``const`` equal to ``FACILITY_HEADER``.
Keys are sorted, so the same model always writes the same bytes.

Beside current lies one floor per header version,
``facility-<N>.schema.json``: the schema as it stood when N was introduced. The
script writes the floor for the current N only when it is missing, and never
rewrites or removes a floor; a test checks that current is an additive extension
of the floor for its N (``osprey_facility.schema.compat``). ``--check`` writes
nothing and fails when current is stale or the floor for the current N is
missing.

Usage::

    uv run python -m osprey_facility.schema._generate           # write
    uv run python -m osprey_facility.schema._generate --check   # compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: The command that runs this module, named in the ``--check`` message.
COMMAND = "python -m osprey_facility.schema._generate"

OUTPUT_DIR = Path(__file__).resolve().parent / "_generated"

#: The file name of the current schema.
CURRENT = "facility.schema.json"

#: Keywords whose value maps names to subschemas (names, not keywords, as keys).
_SCHEMA_MAPS = frozenset({"properties", "$defs"})

#: Keywords whose value is an instance, not a schema: passed through untouched.
_INSTANCE_VALUES = frozenset({"default", "const", "enum", "examples"})


def floor_name(header: str) -> str:
    """Return the floor file name for a header ``osprey.facility.facility/<N>``."""
    version = header.rsplit("/", 1)[-1]
    if not version.isdigit():
        raise ValueError(f"header {header!r} does not end in /<N>")
    return f"facility-{version}.schema.json"


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
    """Return the open JSON Schema of facility.json.

    Args:
        header: The ``schema`` const; ``None`` uses ``FACILITY_HEADER``.
    """
    from osprey_facility.schema import Facility

    if header is None:
        from osprey_facility.validate import FACILITY_HEADER

        header = FACILITY_HEADER
    schema: dict[str, Any] = _open(Facility.model_json_schema())
    schema["properties"]["schema"]["const"] = header
    return schema


def render(schema: dict[str, Any]) -> str:
    """Serialise a schema deterministically, with a trailing newline."""
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write current and a missing floor, or with ``--check`` report whether both are in place.

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

    text = render(build(FACILITY_HEADER))
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
    current.write_text(text, encoding="utf-8")
    if not floor.exists():
        floor.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
