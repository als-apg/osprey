"""Write the vocabulary table of the facility schema.

``src/osprey/facility/schema/_generated/vocabulary.json`` carries what
``vocabulary.yaml`` declares, in a form the standard library reads without
LinkML: the device-class tree (each class with its parent, whether it is
abstract, its aliases and IRI), the signal roles and the property names, each
with its aliases. Every list is sorted by name and every object by key, so the
same schema always writes the same bytes.

The script needs LinkML (the ``dev`` extra); nothing at runtime does.

Usage (from the repository root)::

    uv run python scripts/facility_schema/vocabulary.py           # write
    uv run python scripts/facility_schema/vocabulary.py --check   # compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "src" / "osprey" / "facility" / "schema"
OUTPUT = SCHEMA_DIR / "_generated" / "vocabulary.json"

#: The enum holding the signal roles a channel's ``signal`` names.
SIGNAL_ROLE_ENUM = "signal_role_enum"

#: The enum holding the property names a device's ``properties`` lists.
PROPERTY_NAME_ENUM = "property_name_enum"


def _permissible_values(view: Any, enum_name: str) -> list[dict[str, Any]]:
    """One record per permissible value of an enum, sorted by name.

    Args:
        view: A ``SchemaView`` over ``vocabulary.yaml``.
        enum_name: The enum to read.

    Returns:
        ``{name, description, aliases, unit}`` per value; ``unit`` is the
        value's ``unit`` annotation, or ``None``.
    """
    enum = view.get_enum(enum_name, strict=True)
    records = []
    for name, value in (enum.permissible_values or {}).items():
        unit = value.annotations.get("unit") if value.annotations else None
        records.append(
            {
                "name": name,
                "description": value.description,
                "aliases": list(value.aliases or []),
                "unit": unit.value if unit is not None else None,
            }
        )
    return sorted(records, key=lambda record: record["name"])


def _classes(view: Any) -> list[dict[str, Any]]:
    """The device-class tree, one record per class, sorted by name.

    Args:
        view: A ``SchemaView`` over ``vocabulary.yaml``.

    Returns:
        ``{name, parent, abstract, description, aliases, iri}`` per class;
        ``parent`` is ``None`` for the root.
    """
    records = []
    for name in view.all_classes(imports=False):
        cls = view.get_class(name, strict=True)
        records.append(
            {
                "name": name,
                "parent": cls.is_a,
                "abstract": bool(cls.abstract),
                "description": cls.description,
                "aliases": list(cls.aliases or []),
                "iri": view.get_uri(cls, expand=True),
            }
        )
    return sorted(records, key=lambda record: record["name"])


def render(schema_dir: Path = SCHEMA_DIR) -> str:
    """The vocabulary table as the text of ``vocabulary.json``.

    Args:
        schema_dir: The directory holding ``vocabulary.yaml``.

    Returns:
        The table as indented JSON with sorted keys and a final newline.
    """
    from linkml_runtime.utils.schemaview import SchemaView

    view = SchemaView(str(schema_dir / "vocabulary.yaml"))
    table = {
        "schema": view.schema.name,
        "classes": _classes(view),
        "signal_roles": [
            {key: value for key, value in record.items() if key != "unit"}
            for record in _permissible_values(view, SIGNAL_ROLE_ENUM)
        ],
        "properties": _permissible_values(view, PROPERTY_NAME_ENUM),
    }
    return json.dumps(table, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the table, or with ``--check`` report whether the committed one is current.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        0 when the table was written or is current, 1 when ``--check`` finds it stale.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of write")
    parser.add_argument(
        "--schema-dir", type=Path, default=SCHEMA_DIR, help="directory holding vocabulary.yaml"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT, help="the file to write")
    args = parser.parse_args(argv)
    text = render(args.schema_dir)
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != text:
            print(f"{args.output} is stale; run {Path(__file__).name}")
            return 1
        return 0
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
