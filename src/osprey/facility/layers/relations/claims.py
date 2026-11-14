"""The relations table: rows into the claims and stub devices one relations layer writes.

One row is one claim: ``type`` names a relation type of the vocabulary,
``subject`` and ``object`` its two ends. ``subject_kind`` / ``object_kind`` say
what an end names (``device``, ``place`` or ``group``; ``device`` when empty),
``subject_port`` / ``object_port`` its port and ``object_unit`` the unit an
object place counts in. ``attrs`` holds the claim's attributes as ``key=value``
pairs; ``basis`` is ``stated`` (the default, not written) or ``inferred``, which
needs a ``rule``. The citation columns give the claim's ``source``.

A device end names a device of the facility exactly. One the facility does not
have is stubbed (p5-r2) only when

* a row states its class (``subject_class`` / ``object_class``), a class of the
  facility or the core; or
* the import is run with ``stub_unknown``, which stubs a class-less device as
  ``Unidentified``.

Otherwise the end is refused. A class column naming another class than the
facility's device has is refused, as is one stub given two classes; a class
column equal to the facility's class is accepted and writes nothing. A class
column on a place or group end is refused. A device a row stubs with a class is
the layer's device for every row naming it, with or without a class column.

Place and group ends are never stubbed. A place end resolves like a ``within``
cell, against every place of the facility (bar the replaced layer) and the
places the layer creates; a flat name matching several is refused listing them.
A group end names a group of the facility or one the layer creates.

A claim's id is :func:`~osprey.facility.layers.relations.ids.claim_id` (p5-r5).
Rows with one id and equal content (a symmetric row written reversed included)
are one claim; rows with one id and different content are refused. A refused
row contributes nothing. The importer never stamps the clock.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from osprey.facility.layers.relations.facility import FacilityIndex
from osprey.facility.layers.relations.ids import claim_id
from osprey.facility.layers.relations.places import BASES
from osprey.facility.layers.relations.tables import Refusals, Row, citation, parse_attrs
from osprey.facility.relations import END_KINDS, STUB_CLASS, shape_problems

__all__ = ["TABLE", "claims"]

#: The file name the layer keeps its relations table under, used in refusals.
TABLE = "relations.csv"

#: The ``kind`` an end has when its kind column is empty.
_DEFAULT_KIND = "device"

_SIDES: tuple[str, ...] = ("subject", "object")


@dataclass
class _End:
    """One end as its row states it."""

    kind: str
    cell: str
    port: str | None
    unit: int | None
    stated_class: str | None
    resolved: str | None = None


@dataclass
class _Parsed:
    """One row that passed the row-level checks."""

    line: int
    type: str
    ends: dict[str, _End]
    attributes: dict[str, str] | None
    basis: str | None
    rule: str | None
    source: dict[str, str] | None


def claims(
    rows: Iterable[Row],
    index: FacilityIndex,
    refusals: Refusals,
    *,
    layer: str,
    places: set[str],
    groups: set[str],
    stub_unknown: bool = False,
    types: Mapping[str, Mapping[str, Any]] | None = None,
    table: str = TABLE,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Translate the relations table into claims and stub devices.

    Args:
        rows: The table's rows, as ``read_table`` gives them.
        index: The facility minus the layer being replaced.
        refusals: Where refused rows go.
        layer: The layer's name, the first segment of every claim id.
        places: The place ids the layer itself creates.
        groups: The group ids the layer itself creates.
        stub_unknown: Stub a class-less device end the facility does not have
            as ``Unidentified`` instead of refusing it.
        types: The relation type rows keyed by name; ``relation_types()`` when
            not given.
        table: The table's file name in refusals.

    Returns:
        The claims sorted by id, and the stub devices: id to a ``{"id",
        "class", "citations"}`` device record (``citations`` only when a claim
        naming the stub cites a document, sorted and de-duplicated). The stub
        records are fresh dicts the caller may extend in place.
    """
    if types is None:
        from osprey.facility.validate import relation_types

        types = relation_types()
    parsed = [p for p in (_parse(row, index, refusals, types, table) for row in rows) if p]
    stub_classes, broken = _stub_classes(parsed, index, refusals, table)

    accepted: list[tuple[_Parsed, str, dict[str, Any]]] = []
    for entry in parsed:
        claim = _resolve(
            entry, index, refusals, places, groups, stub_classes, broken, stub_unknown, table
        )
        if claim is None:
            continue
        problems = shape_problems(claim, types)
        if problems:
            for problem, remedy in problems:
                refusals.refuse(
                    table,
                    entry.line,
                    record_kind="relation",
                    value=_describe(entry),
                    problem=problem,
                    remedy=remedy,
                )
            continue
        accepted.append((entry, claim_id(layer, claim, types), claim))

    by_id: dict[str, tuple[int, dict[str, Any], str]] = {}
    clashing: dict[str, list[int]] = {}
    for entry, cid, claim in accepted:
        content = _content(claim, types)
        held = by_id.get(cid)
        if held is None:
            by_id[cid] = (entry.line, claim, content)
        elif held[2] != content:
            lines = clashing.setdefault(cid, [held[0]])
            lines.append(entry.line)
    for cid, lines in clashing.items():
        for line in lines:
            refusals.refuse(
                table,
                line,
                record_kind="claim",
                value=cid,
                problem="rows with one claim id state different content",
                remedy=(
                    "make the rows agree, or keep one; the id ignores the citation's "
                    "version, locator, state and imported and every non-key attribute"
                ),
            )
        del by_id[cid]

    out: list[dict[str, Any]] = []
    stubs: dict[str, dict[str, Any]] = {}
    for cid in sorted(by_id):
        _, claim, _ = by_id[cid]
        out.append({"id": cid, **claim})
    for entry, cid, _ in accepted:
        if cid not in by_id:
            continue
        for end in entry.ends.values():
            if end.kind != "device" or end.resolved in index.devices:
                continue
            device = end.resolved or end.cell
            record = stubs.setdefault(
                device, {"id": device, "class": stub_classes.get(device, STUB_CLASS)}
            )
            if entry.source is not None:
                cites = record.setdefault("citations", [])
                if entry.source not in cites:
                    cites.append(dict(entry.source))
    for record in stubs.values():
        if "citations" in record:
            record["citations"].sort(key=_citation_key)
    return out, dict(sorted(stubs.items()))


def _parse(
    row: Row,
    index: FacilityIndex,
    refusals: Refusals,
    types: Mapping[str, Mapping[str, Any]],
    table: str,
) -> _Parsed | None:
    cells = row.cells
    good = True
    rtype = cells["type"]
    if rtype not in types:
        refusals.refuse(
            table,
            row.line,
            record_kind="type",
            value=rtype,
            problem="unknown relation type",
            remedy=f"name one of {', '.join(sorted(types))} in `type`",
        )
        good = False
    ends: dict[str, _End] = {}
    for side in _SIDES:
        kind = cells.get(f"{side}_kind", _DEFAULT_KIND)
        if kind not in END_KINDS:
            refusals.refuse(
                table,
                row.line,
                record_kind=f"{side}_kind",
                value=kind,
                problem="not an end kind",
                remedy=f"write device, place or group in `{side}_kind`, or leave it empty",
            )
            good = False
        unit: int | None = None
        unit_cell = cells.get("object_unit") if side == "object" else None
        if unit_cell is not None:
            try:
                unit = int(unit_cell)
            except ValueError:
                refusals.refuse(
                    table,
                    row.line,
                    record_kind="object_unit",
                    value=unit_cell,
                    problem="not an integer",
                    remedy="write `object_unit` as a whole number",
                )
                good = False
        stated_class = cells.get(f"{side}_class")
        if stated_class is not None and kind != "device":
            refusals.refuse(
                table,
                row.line,
                record_kind=f"{side}_class",
                value=stated_class,
                problem=f"a class on an end that names a {kind}",
                remedy=f"empty `{side}_class`; only a device end takes a class",
            )
            good = False
        elif stated_class is not None and stated_class not in index.classes:
            refusals.refuse(
                table,
                row.line,
                record_kind="class",
                value=stated_class,
                problem="unknown class",
                remedy="name a class of the facility or the core",
            )
            good = False
        ends[side] = _End(kind, cells[side], cells.get(f"{side}_port"), unit, stated_class)
    basis = cells.get("basis")
    if basis is not None and basis not in BASES:
        refusals.refuse(
            table,
            row.line,
            record_kind="basis",
            value=basis,
            problem="not a basis",
            remedy="write `stated` or `inferred` in `basis`, or leave it empty",
        )
        good = False
    attributes: dict[str, str] | None = None
    if "attrs" in cells:
        attributes = parse_attrs(cells["attrs"], refusals, table, row.line)
        if attributes is None:
            good = False
    if not good:
        return None
    return _Parsed(
        row.line,
        rtype,
        ends,
        attributes or None,
        None if basis in (None, "stated") else basis,
        cells.get("rule"),
        citation(cells),
    )


def _stub_classes(
    parsed: list[_Parsed], index: FacilityIndex, refusals: Refusals, table: str
) -> tuple[dict[str, str], set[str]]:
    """Check class columns: the class each stub is given, and the stubs given two."""
    stated: dict[str, dict[str, list[int]]] = {}
    for entry in parsed:
        for side, end in entry.ends.items():
            if end.kind != "device" or end.stated_class is None:
                continue
            if end.cell in index.devices:
                held = index.devices[end.cell]
                if held is not None and held != end.stated_class:
                    refusals.refuse(
                        table,
                        entry.line,
                        record_kind="device",
                        value=end.cell,
                        problem=f"`{side}_class` {end.stated_class} differs from its class {held}",
                        remedy=f"write {held} in `{side}_class`, or leave it empty",
                    )
                continue
            stated.setdefault(end.cell, {}).setdefault(end.stated_class, []).append(entry.line)
    classes: dict[str, str] = {}
    broken: set[str] = set()
    for device, by_class in stated.items():
        if len(by_class) == 1:
            classes[device] = next(iter(by_class))
            continue
        broken.add(device)
        for lines in by_class.values():
            for line in lines:
                refusals.refuse(
                    table,
                    line,
                    record_kind="device",
                    value=device,
                    problem=f"stubbed with two classes: {', '.join(sorted(by_class))}",
                    remedy="give the device one class in every row that states one",
                )
    return classes, broken


def _resolve(
    entry: _Parsed,
    index: FacilityIndex,
    refusals: Refusals,
    places: set[str],
    groups: set[str],
    stub_classes: dict[str, str],
    broken: set[str],
    stub_unknown: bool,
    table: str,
) -> dict[str, Any] | None:
    """Resolve the row's ends and build its claim (without an id)."""
    good = True
    for side, end in entry.ends.items():
        if end.kind == "device":
            if end.cell in index.devices:
                held = index.devices[end.cell]
                if end.stated_class is not None and held is not None and held != end.stated_class:
                    good = False
                    continue
                end.resolved = end.cell
            elif end.cell in broken:
                good = False
            elif end.cell in stub_classes or stub_unknown:
                end.resolved = end.cell
            else:
                refusals.refuse(
                    table,
                    entry.line,
                    record_kind="device",
                    value=end.cell,
                    problem="unknown device",
                    remedy=(
                        f"name a device of the facility, state its class in `{side}_class`, "
                        "or run with `--stub-unknown`"
                    ),
                )
                good = False
        elif end.kind == "place":
            found = index.resolve_place(end.cell, places)
            if isinstance(found, str):
                end.resolved = found
            elif isinstance(found, list):
                refusals.refuse(
                    table,
                    entry.line,
                    record_kind="place",
                    value=end.cell,
                    problem=f"matches several places: {', '.join(found)}",
                    remedy="name the place by its full path",
                )
                good = False
            else:
                refusals.refuse(
                    table,
                    entry.line,
                    record_kind="place",
                    value=end.cell,
                    problem="unknown place",
                    remedy="name a place of the facility or of the places table",
                )
                good = False
        elif end.cell in index.groups or end.cell in groups:
            end.resolved = end.cell
        else:
            refusals.refuse(
                table,
                entry.line,
                record_kind="group",
                value=end.cell,
                problem="unknown group",
                remedy="name a group of the facility or of the members table",
            )
            good = False
    if not good:
        return None
    claim: dict[str, Any] = {"type": entry.type}
    for side, end in entry.ends.items():
        record: dict[str, Any] = {end.kind: end.resolved}
        if end.port is not None:
            record["port"] = end.port
        if end.unit is not None:
            record["unit"] = end.unit
        claim[side] = record
    if entry.attributes:
        claim["attributes"] = dict(entry.attributes)
    if entry.basis is not None:
        claim["basis"] = entry.basis
    if entry.rule is not None:
        claim["rule"] = entry.rule
    if entry.source is not None:
        claim["source"] = dict(entry.source)
    return claim


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content(claim: Mapping[str, Any], types: Mapping[str, Mapping[str, Any]]) -> str:
    """The claim's full content, its ends in sorted order for a symmetric type."""
    content = dict(claim)
    if (types.get(str(claim.get("type"))) or {}).get("symmetric"):
        ends = sorted((_canonical(claim["subject"]), _canonical(claim["object"])))
        content["subject"], content["object"] = ends
    return _canonical(content)


def _describe(entry: _Parsed) -> str:
    return f"{entry.ends['subject'].cell} {entry.type} {entry.ends['object'].cell}"


def _citation_key(source: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        source.get(column, "") for column in ("document", "version", "locator", "state", "imported")
    )
