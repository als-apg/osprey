"""The ``--names`` table: rows into the ``scheme_names`` of records a relations layer creates.

One row is one name of a thing: ``name`` in ``scheme``, with an optional
``basis``/``rule`` and a citation that becomes the alias's ``source``. Layers are
never unioned (p5-r4 as amended), so the table names only what the layer itself
creates:

* a stub device of this layer (the relations table's ends);
* a place this layer creates that no other source has.

A row whose ``thing`` is a device or place the authored file or another
imported layer has is refused, since a second layer stating that record's
names would be a ``layer-conflict``; this includes a place this layer creates
that another imported layer also has. A group is refused too (``Group`` has no
``scheme_names`` slot), as is a thing nobody has. A flat place name matching
several places is refused listing them.

``scheme_names`` is a keyed set on ``(name, scheme)``: equal entries are one
entry, and one key stated with two contents is refused on every row stating it.
``basis: stated`` is the default and is not written; ``basis: inferred`` needs a
``rule``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from osprey.facility.layers.relations.facility import FacilityIndex
from osprey.facility.layers.relations.places import BASES
from osprey.facility.layers.relations.tables import Refusals, Row, citation
from osprey.facility.sources import keyed_order

__all__ = ["TABLE", "names"]

#: The file name the layer keeps its names table under, used in refusals.
TABLE = "names.csv"

_SLOT = "scheme_names"


def names(
    rows: Iterable[Row],
    index: FacilityIndex,
    refusals: Refusals,
    *,
    stubs: Mapping[str, dict[str, Any]],
    places: Iterable[dict[str, Any]],
    groups: Iterable[str] = (),
    table: str = TABLE,
) -> None:
    """Add the names table's aliases to the layer's stub devices and places.

    The records are changed in place: an accepted row's alias is added to the
    record's ``scheme_names`` (equal entries once, in the build's keyed order).
    A refused row contributes nothing.

    Args:
        rows: The names table's rows.
        index: The facility minus the layer being replaced.
        refusals: Where refused rows go.
        stubs: The layer's stub devices, id to device record.
        places: The place records the layer creates.
        groups: The ids of the groups the layer creates.
        table: The table's file name in refusals.
    """
    created = {record["id"]: record for record in places}
    own_groups = set(groups)
    # record id -> (name, scheme) -> [(line, canonical content, entry)]
    stated: dict[str, dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]]] = {}
    targets: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = _entry(row, refusals, table)
        record = _target(row, index, stubs, created, own_groups, refusals, table)
        if entry is None or record is None:
            continue
        key = (entry["name"], entry["scheme"])
        targets[record["id"]] = record
        stated.setdefault(record["id"], {}).setdefault(key, []).append(
            (row.line, _canonical(entry), entry)
        )

    for rid, keyed in stated.items():
        accepted: list[dict[str, Any]] = []
        for (name, scheme), entries in keyed.items():
            if len({content for _, content, _ in entries}) > 1:
                for line, _, _ in entries:
                    refusals.refuse(
                        table,
                        line,
                        record_kind="name",
                        value=f"{rid} {scheme} {name}",
                        problem="rows naming one thing in one scheme state different content",
                        remedy="make the rows agree, or keep one",
                    )
                continue
            accepted.append(entries[0][2])
        if accepted:
            record = targets[rid]
            record[_SLOT] = keyed_order(_SLOT, [*(record.get(_SLOT) or []), *accepted])


def _entry(row: Row, refusals: Refusals, table: str) -> dict[str, Any] | None:
    cells = row.cells
    basis = cells.get("basis")
    if basis is not None and basis not in BASES:
        refusals.refuse(
            table,
            row.line,
            record_kind="basis",
            value=basis,
            problem="not a basis",
            remedy="write `basis` as stated or inferred",
        )
        return None
    if basis == "inferred" and "rule" not in cells:
        refusals.refuse(
            table,
            row.line,
            record_kind="name",
            value=cells["name"],
            problem="`basis` is inferred and `rule` is missing",
            remedy="name the rule that derived the row in `rule`, or drop `basis: inferred`",
        )
        return None
    entry: dict[str, Any] = {"name": cells["name"], "scheme": cells["scheme"]}
    if basis == "inferred":
        entry["basis"] = basis
    if "rule" in cells:
        entry["rule"] = cells["rule"]
    source = citation(cells)
    if source is not None:
        entry["source"] = source
    return entry


def _target(
    row: Row,
    index: FacilityIndex,
    stubs: Mapping[str, dict[str, Any]],
    created: Mapping[str, dict[str, Any]],
    groups: set[str],
    refusals: Refusals,
    table: str,
) -> dict[str, Any] | None:
    thing = row.cells["thing"]

    def refuse(problem: str, remedy: str) -> None:
        refusals.refuse(
            table, row.line, record_kind="thing", value=thing, problem=problem, remedy=remedy
        )

    owned_remedy = (
        "state the name where that record is stated, or remove the row; "
        "this layer names only the stub devices and places it creates"
    )
    if thing in stubs:
        return stubs[thing]
    if thing in index.devices:
        refuse("a device another source has", owned_remedy)
        return None
    if thing in groups or thing in index.groups:
        refuse(
            "a group; a group has no `scheme_names`",
            "remove the row, or name a stub device or a place this layer creates",
        )
        return None
    found = index.resolve_place(thing, set(created))
    if isinstance(found, list):
        refuse(
            f"matches several places: {', '.join(found)}",
            "write the place's full `/`-path in `thing`",
        )
        return None
    if found is None:
        refuse(
            "unknown thing",
            "name a stub device of the relations table or a place this layer creates, "
            "or remove the row",
        )
        return None
    if found not in created or found in index.places:
        refuse("a place another source has", owned_remedy)
        return None
    return created[found]


def _canonical(entry: Mapping[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True)
