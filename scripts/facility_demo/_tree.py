"""Reading and writing the YAML files of a facility tree, shared by the generators.

Every generated YAML file of the example tree opens with the bundled-origin
line. :func:`merge` replaces records in a file's top-level list by key: a
record already in the file keeps its position, a new one is appended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def origin_line() -> str:
    """The bundled-origin line every generated YAML file of the example tree opens with."""
    from osprey.facility.bundled import header_line

    return f"{header_line('example')}\n"


def dump(data: Any) -> str:
    import yaml

    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=float("inf"))
    return origin_line() + text


def load(path: Path) -> list[dict[str, Any]]:
    import yaml

    if not path.is_file():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def merge(path: Path, records: list[dict[str, Any]], key: str, *, ordered: bool) -> None:
    """Replace ``records`` in the YAML list at ``path`` by ``key``.

    A record already in the file keeps its position; a new one is appended.
    With ``ordered`` the list is sorted by ``key``.
    """
    new = {record[key]: record for record in records}
    merged = [new.pop(row[key], row) for row in load(path)]
    merged += new.values()
    if ordered:
        merged.sort(key=lambda row: row[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump(merged), encoding="utf-8")
