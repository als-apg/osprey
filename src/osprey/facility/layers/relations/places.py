"""The ``--places`` table: rows into the place records one relations layer creates.

One row places one place: ``place`` is a flat name or a ``/``-path, ``within``
names the place it sits in (a flat name or a path). The record's id is
``<within>/<place>``; a ``place`` cell holding ``/`` is the id itself, and a
``within`` beside it must name the path's parent.

``within`` resolves against the authored places and the places this layer's
own rows create, never against another imported layer's places (p5-r1): where
a layer puts a place depends on its own table and the authored file alone, not
on import order. A flat ``within`` matching several places is refused listing
them (CF-8).

Several rows for one flat name follow the chain rule: when their parents lie
on one chain (each an ancestor of the deepest) the place is one record under
the deepest parent carrying every row's citation; otherwise it is one record
per parent, and the build reports ``place-parents-disagree``. A row with no
``within`` places nothing: it is top-level when nothing else places the name,
cited when it carries a citation, and without ``citations`` otherwise (the
build reports ``place-unplaced``). ``basis`` and ``rule`` are checked, not
kept. A place the authored file has is not re-created (CF-9); one another
imported layer has is written all the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from osprey.facility.layers.relations.facility import FacilityIndex
from osprey.facility.layers.relations.tables import Refusals, Row, citation

__all__ = ["BASES", "places"]

#: The values a ``basis`` cell may hold (core ``basis_enum``).
BASES: tuple[str, ...] = ("stated", "inferred")


@dataclass(frozen=True)
class _Entry:
    """One accepted row: the flat name, the cell naming its parent, and for a
    path row the ``within`` cell written beside it (which must agree)."""

    line: int
    name: str
    within: str | None
    beside: str | None
    citation: dict[str, str] | None


#: :func:`_parent`'s answer when a path row's ``within`` names another parent.
_DISAGREES = object()


def places(
    rows: list[Row], index: FacilityIndex, refusals: Refusals, table: str = "places.csv"
) -> list[dict]:
    """Translate the places table into place records.

    Args:
        rows: The table's rows, as ``read_table`` gives them.
        index: The facility minus the layer being replaced.
        refusals: Where refused rows go; a refused row contributes nothing.
        table: The table's file name, for refusals.

    Returns:
        The place records the layer creates, sorted by id: ``{"id": ...}``
        plus ``citations`` (a sorted list) when any row cites the place.
    """
    entries = [entry for row in rows if (entry := _entry(row, refusals, table)) is not None]
    authored = index.authored_places

    # The layer's own ids are what the chain rule makes of the rows resolved so
    # far; `within` cells may name them, so resolve again until they settle.
    own: set[str] = set()
    for _ in range(len(entries) + 1):
        parents = {
            e.line: p for e in entries if isinstance(p := _parent(e, index, own, authored), str)
        }
        settled = set(_chain([e for e in entries if e.line in parents], parents))
        if settled == own:
            break
        own = settled

    placed: dict[int, str] = {}
    for entry in entries:
        parent = _parent(entry, index, own, authored)
        if isinstance(parent, str):
            placed[entry.line] = parent
            continue
        _refuse_within(entry, parent, refusals, table)

    live = [e for e in entries if e.line in placed]
    while True:
        records = _chain(live, placed)
        known = authored | set(records)
        stranded = [e for e in live if placed[e.line] and placed[e.line] not in known]
        if not stranded:
            break
        for entry in stranded:
            _refuse_within(entry, None, refusals, table)
        live = [e for e in live if e not in stranded]

    out: list[dict] = []
    for pid in sorted(records):
        if pid in authored:
            continue
        record: dict = {"id": pid}
        cites = records[pid]
        if cites:
            record["citations"] = cites
        out.append(record)
    return out


def _entry(row: Row, refusals: Refusals, table: str) -> _Entry | None:
    cells = row.cells
    place = cells["place"]
    basis = cells.get("basis")
    good = True
    if basis is not None and basis not in BASES:
        refusals.refuse(
            table,
            row.line,
            record_kind="basis",
            value=basis,
            problem="not a basis",
            remedy="write `basis` as stated or inferred",
        )
        good = False
    if basis == "inferred" and "rule" not in cells:
        refusals.refuse(
            table,
            row.line,
            record_kind="place",
            value=place,
            problem="`basis` is inferred and `rule` is missing",
            remedy="name the rule that derived the row in `rule`, or drop `basis: inferred`",
        )
        good = False
    segments = place.split("/")
    if any(not segment for segment in segments):
        refusals.refuse(
            table,
            row.line,
            record_kind="place",
            value=place,
            problem="a path with an empty segment",
            remedy="write `place` as a name or a path such as SR/S01",
        )
        good = False
    if not good:
        return None
    within = cells.get("within")
    if "/" in place:
        parent, _, name = place.rpartition("/")
        return _Entry(row.line, name, parent, within, citation(cells))
    return _Entry(row.line, place, within, None, citation(cells))


def _parent(entry: _Entry, index: FacilityIndex, own: set[str], authored: set[str]):
    """The entry's parent id ('' for top-level), its candidates, ``None`` when
    unknown, or :data:`_DISAGREES`."""
    if entry.within is None:
        return ""
    found = index.resolve_place(entry.within, own, base=authored)
    if (
        entry.beside is not None
        and isinstance(found, str)
        and index.resolve_place(entry.beside, own, base=authored) != found
    ):
        return _DISAGREES
    return found


def _refuse_within(entry: _Entry, found: object, refusals: Refusals, table: str) -> None:
    if found is _DISAGREES:
        refusals.refuse(
            table,
            entry.line,
            record_kind="place",
            value=_join(entry.within or "", entry.name),
            problem=f"`within` {entry.beside} is not the path's parent",
            remedy="empty `within`, or name the path's parent in it",
        )
        return
    cell = entry.within or ""
    if isinstance(found, list):
        refusals.refuse(
            table,
            entry.line,
            record_kind="place",
            value=cell,
            problem=f"matches several places: {', '.join(found)}",
            remedy="name the place by its full path",
        )
        return
    refusals.refuse(
        table,
        entry.line,
        record_kind="place",
        value=cell,
        problem="unknown place",
        remedy="name a place the authored records or this places table state",
    )


def _chain(entries: list[_Entry], placed: dict[int, str]) -> dict[str, list[dict[str, str]]]:
    """Apply the chain rule: place id to its sorted, de-duplicated citations."""
    by_name: dict[str, list[_Entry]] = {}
    for entry in entries:
        by_name.setdefault(entry.name, []).append(entry)
    records: dict[str, list[dict[str, str]]] = {}
    for name, group in by_name.items():
        parents = {placed[e.line] for e in group}
        deepest = max(parents, key=lambda p: (p.count("/") + bool(p), p))
        if all(_ancestor(p, deepest) for p in parents):
            targets = {e.line: deepest for e in group}
        else:
            # Off one chain: one record per parent. A row with no `within`
            # places nothing; uncited, it adds nothing either.
            targets = {
                e.line: placed[e.line] for e in group if placed[e.line] or e.citation is not None
            }
        for entry in group:
            if entry.line not in targets:
                continue
            cites = records.setdefault(_join(targets[entry.line], name), [])
            if entry.citation is not None and entry.citation not in cites:
                cites.append(entry.citation)
    for cites in records.values():
        cites.sort(key=_citation_key)
    return records


def _ancestor(parent: str, deepest: str) -> bool:
    return parent == "" or parent == deepest or deepest.startswith(parent + "/")


def _join(parent: str, name: str) -> str:
    return f"{parent}/{name}" if parent else name


def _citation_key(cite: dict[str, str]) -> tuple[str, ...]:
    return tuple(cite.get(k, "") for k in ("document", "version", "locator", "state", "imported"))
