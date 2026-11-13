"""The ``--members`` table: rows into the group records a relations layer writes.

One row is one member of a group. Members are devices, so ``member_kind`` is
``device`` or empty. ``group_kind`` is the kind of a group the layer creates,
``family`` or ``section``; a group no row gives one states none, and a row
giving a second kind for its group is refused. Group membership is a whole set per layer (p5-r3):

* a group no other source has is created in the layer's ``groups.yaml`` with
  every member its rows name;
* a row for a group the facility already has is accepted only if that group
  already holds the member, and is then written nowhere; otherwise it is
  refused, since a second layer stating the group would be a
  ``layer-conflict``.

A member must be a device of the facility or a stub device of this layer. A
group record carries no citation (``Group`` has no ``citations`` slot); the
citation of a member row is added to the member's ``citations`` only where the
member is a stub of this layer.

The groups are needed twice: claims may end on a group the layer creates, so
:func:`group_ids` names those groups before the claims are read, and
:func:`groups` builds the records after them, once the stubs are known.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from osprey.facility.layers.relations.facility import FacilityIndex
from osprey.facility.layers.relations.tables import Refusals, Row, citation

__all__ = ["TABLE", "group_ids", "groups"]

#: The file name the layer keeps its members table under, used in refusals.
TABLE = "members.csv"

#: The only ``member_kind`` a row may state.
_KIND = "device"

#: The ``group_kind`` values a row may state: the ``kind`` values of a ``Group``.
_GROUP_KINDS = ("family", "section")


def group_ids(rows: Iterable[Row], index: FacilityIndex) -> set[str]:
    """The groups the layer creates: those its rows name and no other source has.

    Args:
        rows: The members table's rows.
        index: The facility minus the layer being replaced.

    Returns:
        The ids of the groups :func:`groups` will write.
    """
    return {row.cells["group"] for row in rows if row.cells["group"] not in index.groups}


def groups(
    rows: Iterable[Row],
    index: FacilityIndex,
    stubs: Mapping[str, dict[str, Any]],
    refusals: Refusals,
    table: str = TABLE,
) -> list[dict[str, Any]]:
    """Build the group records of the layer.

    A refused row contributes nothing. The citation of an accepted row whose
    member is in ``stubs`` is added to that stub record's ``citations`` in
    place (equal citations once, sorted).

    Args:
        rows: The members table's rows.
        index: The facility minus the layer being replaced.
        stubs: The layer's stub devices, id to device record.
        refusals: Where refused rows go.
        table: The table's file name in refusals.

    Returns:
        One ``{"id", "members"}`` record per created group, sorted by id, its
        members sorted, with the ``kind`` its rows state between the two.
    """
    created: dict[str, set[str]] = {}
    kinds: dict[str, str] = {}
    cited: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        group = row.cells["group"]
        member = row.cells["member"]
        kind = row.cells.get("member_kind", _KIND)
        if kind != _KIND:
            refusals.refuse(
                table,
                row.line,
                record_kind="member_kind",
                value=kind,
                problem="not `device`; the members of a group are devices",
                remedy="write `device` in `member_kind` or leave it empty",
            )
            continue
        if row.cells.get("group_kind", _GROUP_KINDS[0]) not in _GROUP_KINDS:
            refusals.refuse(
                table,
                row.line,
                record_kind="group_kind",
                value=row.cells["group_kind"],
                problem="not `family` or `section`",
                remedy="write `family` or `section` in `group_kind` or leave it empty",
            )
            continue
        held = index.groups.get(group)
        if held is not None:
            if member not in held:
                refusals.refuse(
                    table,
                    row.line,
                    record_kind="member",
                    value=member,
                    problem=f"not a member of group `{group}`, which another source has",
                    remedy=(
                        f"add the member to `{group}` where that group is stated, or remove the row"
                    ),
                )
            continue
        if member not in index.devices and member not in stubs:
            refusals.refuse(
                table,
                row.line,
                record_kind="member",
                value=member,
                problem="unknown device",
                remedy="name a device of the facility or of the relations table, or remove the row",
            )
            continue
        group_kind = row.cells.get("group_kind")
        if group_kind is not None and group_kind != kinds.setdefault(group, group_kind):
            refusals.refuse(
                table,
                row.line,
                record_kind="group_kind",
                value=group_kind,
                problem=f"group `{group}` is a `{kinds[group]}` in an earlier row",
                remedy=f"give `{group}` one `group_kind`",
            )
            continue
        created.setdefault(group, set()).add(member)
        source = citation(row.cells)
        if member in stubs and source is not None:
            cited.setdefault(member, []).append(source)
    for member, sources in cited.items():
        _add_citations(stubs[member], sources)
    return [
        {
            "id": group,
            **({"kind": kinds[group]} if group in kinds else {}),
            "members": sorted(members),
        }
        for group, members in sorted(created.items())
    ]


def _add_citations(record: dict[str, Any], sources: list[dict[str, str]]) -> None:
    merged = list(record.get("citations") or [])
    for source in sources:
        if source not in merged:
            merged.append(source)
    record["citations"] = sorted(merged, key=_citation_key)


def _citation_key(source: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        source.get(column, "") for column in ("document", "version", "locator", "state", "imported")
    )
