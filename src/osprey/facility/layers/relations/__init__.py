"""The relations layer: relation tables imported as facility sources.

A relations import reads one table of claims and, optionally, three tables the
claims lean on. Each is UTF-8 CSV with a header row naming its columns in any
order; empty cells are absent. The columns, required first, are::

    relations  type, subject, object; subject_kind, object_kind,
               subject_port, object_port, object_unit, attrs, basis, rule,
               subject_class, object_class
    names      thing, name, scheme; basis, rule
    places     place; within, basis, rule
    members    group, member; member_kind

and every table also takes the citation columns ``document``, ``version``,
``state``, ``locator`` and ``imported``.

One ``relations`` row is one claim of a relation type between two ends, each a
device (the default), a place or a group. A device end the facility does not
have is stubbed only when its row states a class (``subject_class`` /
``object_class``) or the run passes ``stub_unknown``, which stubs it as
``Unidentified`` (p5-r2); place and group ends are never stubbed. A claim's id
hashes its type, its ends (order-free for a symmetric type), its ports, the
type's key attribute and the citation's document -- never the version, the
locator or the ``imported`` date (p5-r5); the import never reads the clock.

The ``places`` table creates places under the authored places and its own rows,
never under another imported layer's places (p5-r1). The ``members`` table
creates a group no other source has; a row naming a member another source's
group lacks makes the layer state that group too, and the group carries every
member any source states. The ``names`` table names only the stub devices and places
this layer creates (p5-r4). Every end and name resolves against the whole
facility bar the layer being replaced.

The layer ``<layer>`` is written to ``imported/<layer>/``: ``relations.yaml``,
``devices.yaml`` (the stubs), ``places.yaml`` and ``groups.yaml``, each sorted
by id and written only when it holds a record, and each input table copied
byte for byte as ``relations.csv``, ``names.csv``, ``places.csv`` and
``members.csv``. ``relations.csv`` marks the directory as a relations layer.
The directory is replaced whole, so a file of the previous import that this
one does not write is gone.

The layer name must match ``[a-z0-9][a-z0-9_-]*`` and is not ``authored`` nor
a format verb's own layer (``list``, ``mml``); a directory holding any entry
this import never writes is another format's layer. Each of these, and every
row any table refuses, is ``source-invalid``, every one reported, and nothing
is written. :mod:`.ids` computes each claim's id.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from osprey.facility.bundled import LAYER_ROOT
from osprey.facility.errors import FacilityBuildError
from osprey.facility.layers import LayerSourceInvalid
from osprey.facility.layers.relations import claims as _claims
from osprey.facility.layers.relations import members as _members
from osprey.facility.layers.relations import names as _names
from osprey.facility.layers.relations import places as _places
from osprey.facility.layers.relations.facility import FacilityIndex
from osprey.facility.layers.relations.tables import (
    MEMBERS,
    NAMES,
    PLACES,
    RELATIONS,
    Refusals,
    read_table,
)
from osprey.facility.sources import AUTHORED, RELATIONS_MARK
from osprey_connectors import yaml_loader

__all__ = ["LAYER_FILES", "RECORD_FILES", "RESERVED_LAYERS", "TABLE_FILES", "import_relations"]

#: The record files the layer writes, in the order they are written.
RECORD_FILES: tuple[str, ...] = ("relations.yaml", "devices.yaml", "places.yaml", "groups.yaml")

#: The names the input tables are copied under, keyed by table.
TABLE_FILES: dict[str, str] = {
    "relations": RELATIONS_MARK,
    "names": _names.TABLE,
    "places": "places.csv",
    "members": _members.TABLE,
}

#: Every entry a relations layer's directory may hold.
LAYER_FILES: frozenset[str] = frozenset(RECORD_FILES) | frozenset(TABLE_FILES.values())

#: Layer names a relations import never takes: the authored records and the
#: format verbs' own layers.
RESERVED_LAYERS: frozenset[str] = frozenset({AUTHORED, "list", "mml"})

_LAYER_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


def import_relations(
    table: Path,
    facility_dir: Path,
    layer: str,
    *,
    names: Path | None = None,
    places: Path | None = None,
    members: Path | None = None,
    stub_unknown: bool = False,
) -> list[Path]:
    """Write relation tables as the sources of one relations layer.

    Args:
        table: The relations table.
        facility_dir: The ``data/facility`` directory.
        layer: The layer's name; its directory is ``imported/<layer>/``.
        names: The names table, if any.
        places: The places table, if any.
        members: The members table, if any.
        stub_unknown: Stub a class-less device end the facility does not have
            as ``Unidentified`` instead of refusing it.

    Returns:
        Every file written: the record files, then the copied tables.

    Raises:
        LayerSourceInvalid: The layer name is refused or a table has rows the
            layer refuses; nothing is written.
    """
    target = facility_dir / LAYER_ROOT / layer
    guard = _guard(layer, target)
    if guard:
        raise LayerSourceInvalid(guard)

    refusals = Refusals()
    relation_rows = read_table(table, RELATIONS, refusals)
    name_rows = read_table(names, NAMES, refusals) if names else []
    place_rows = read_table(places, PLACES, refusals) if places else []
    member_rows = read_table(members, MEMBERS, refusals) if members else []

    index = FacilityIndex.load(facility_dir, layer)
    place_records = _places.places(place_rows, index, refusals, places.name if places else "")
    group_ids = _members.group_ids(member_rows, index)
    claim_records, stubs = _claims.claims(
        relation_rows,
        index,
        refusals,
        layer=layer,
        places={record["id"] for record in place_records},
        groups=group_ids,
        stub_unknown=stub_unknown,
        table=table.name,
    )
    group_records = _members.groups(
        member_rows, index, stubs, refusals, members.name if members else ""
    )
    _names.names(
        name_rows,
        index,
        refusals,
        stubs=stubs,
        places=place_records,
        groups=group_ids,
        table=names.name if names else "",
    )
    if refusals:
        raise LayerSourceInvalid(refusals.errors())

    records: dict[str, list[dict[str, Any]]] = {
        "relations.yaml": claim_records,
        "devices.yaml": [stubs[key] for key in sorted(stubs)],
        "places.yaml": sorted(place_records, key=lambda record: record["id"]),
        "groups.yaml": sorted(group_records, key=lambda record: record["id"]),
    }
    tables = {"relations": table, "names": names, "places": places, "members": members}
    return _replace(facility_dir, target, records, tables)


def _guard(layer: str, target: Path) -> list[FacilityBuildError]:
    def refuse(detail: str, remedy: str) -> list[FacilityBuildError]:
        return [
            FacilityBuildError(
                "source-invalid",
                layer or '""',
                [f"{LAYER_ROOT}/{layer}"],
                remedy,
                record_kind="layer",
                detail=detail,
            )
        ]

    if not _LAYER_NAME.fullmatch(layer):
        return refuse(
            "is not a layer name",
            "name the layer in lowercase letters, digits, '-' and '_', starting with a letter "
            "or digit",
        )
    if layer in RESERVED_LAYERS:
        return refuse("is reserved for another kind of source", "name the layer otherwise")
    if target.exists():
        foreign = (
            sorted(entry.name for entry in target.iterdir() if entry.name not in LAYER_FILES)
            if target.is_dir()
            else [target.name]
        )
        if foreign:
            return refuse(
                f"is another format's layer: it holds {', '.join(foreign)}",
                "name the relations layer otherwise",
            )
    return []


def _replace(
    facility_dir: Path,
    target: Path,
    records: dict[str, list[dict[str, Any]]],
    tables: dict[str, Path | None],
) -> list[Path]:
    """Stage the layer outside ``imported/``, then swap it in for the old one."""
    staging = Path(tempfile.mkdtemp(prefix=f".relations-{target.name}-", dir=facility_dir))
    try:
        new = staging / "new"
        new.mkdir()
        written: list[str] = []
        for name in RECORD_FILES:
            if records[name]:
                _dump(new / name, records[name])
                written.append(name)
        for key, source in tables.items():
            if source is not None:
                shutil.copyfile(source, new / TABLE_FILES[key])
                written.append(TABLE_FILES[key])

        target.parent.mkdir(parents=True, exist_ok=True)
        old = staging / "old"
        if target.exists():
            os.rename(target, old)
        try:
            os.rename(new, target)
        except BaseException:
            if old.exists() and not target.exists():
                os.rename(old, target)
            raise
        return [target / name for name in written]
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _dump(path: Path, rows: list[dict[str, Any]]) -> Path:
    text = yaml_loader.safe_dump(
        rows, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100
    )
    path.write_text(text, encoding="utf-8")
    return path
