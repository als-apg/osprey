"""The ``mapping.yaml`` skeleton ``map --init`` writes.

The skeleton pre-fills every slot of the mapping document the export has a fact
for and leaves the rest ``None``, so the agent reviewing the install edits one
stable file instead of writing it from scratch.

Rules:

* ``facility.token`` is ``AD.Machine`` folded to PN_LOCAL, ``None`` without AD;
  ``title`` is ``AD.Machine`` as written; ``description`` is ``derived``.
* ``systems`` are keyed by raw token in system order (``_import_order``, then
  the remaining keys sorted). A system's native ``_description`` is carried as
  ``imported``; otherwise prose from ``AD.Machine``/``SubMachine``/
  ``OperationalMode`` is ``derived``. ``name`` is pre-filled with the raw token.
* ``section_order`` lists each system's ``name`` in that same order, so it is a
  permutation of ``systems.*.name``.
* ``families`` are keyed by raw token, merged across systems in first-encounter
  order. ``class`` is pre-filled with the mapped token and ``branch`` is
  ``None``; a family with no channels carries neither key. ``rename`` and
  ``branches`` are never written. Descriptions are the native text
  (``imported``) or prose built from export facts only (``derived``).
* ``directions`` holds one ``<raw family>.<field>`` slot per signal group,
  filled from the vote as ``derived`` (``None`` when undecided), with
  ``override: false``.
* ``judgments`` is written last and only when a family pends one, keyed by raw
  family in the same merged order, then by kind, then by field and export
  order. Its slots are the union across the systems carrying the family, so a
  signal or ordinal pending in two systems is one slot and a family with any
  supply group gets one ``shared_pvs``. A family with nothing pending has no
  entry; nothing pending anywhere omits the block, as ``branches`` is omitted.

The ``virtual_accelerator`` block is built apart from the rest, by
:func:`va_block` from the verdicts
:func:`~osprey.services.mml.va.verdicts.propose` reaches, because it exists
only for a 2.0 export and is appended to a document that may already have been
reviewed. :func:`va_system` names the one system it describes and
:func:`dump_va_block` serialises it as the text ``map --init`` appends, each
open slot carrying its allowed answers on a comment line so the reviewer reads
the vocabulary beside the question.

Grain numbers come from :class:`~osprey.services.mml.family.FamilyView`. The
module has no I/O; :func:`dump_yaml` only serialises.
"""

from __future__ import annotations

import re
from collections.abc import Container
from typing import Any

import yaml

from osprey.services.mml.directions import Vote
from osprey.services.mml.family import FamilyView, family_views, system_bodies
from osprey.services.mml.judgments import pending_judgments
from osprey.services.mml.mapping.branches import is_pn_local
from osprey.services.mml.mapping.schema import (
    ATTYPE_KIND,
    ESCAPE_HATCH_KIND,
    ROWS_BEYOND_KIND,
    SHARED_FIELD_KIND,
    SHARED_KIND,
    UNBOUND_KIND,
    VAFamily,
)

__all__ = [
    "STORAGE_RING",
    "VA_ANSWERS",
    "build_skeleton",
    "count_judgment_slots",
    "count_va_slots",
    "dump_va_block",
    "dump_yaml",
    "section_order",
    "va_block",
    "va_system",
]

#: Provenance of prose and directions generated from export facts.
DERIVED = "derived"

#: Provenance of a description carried over from the export.
IMPORTED = "imported"

#: Characters a PN_LOCAL token may not contain, folded to ``_``.
_NOT_PN_LOCAL = re.compile(r"[^A-Za-z0-9_]+")

#: The ``MachineType`` a virtual accelerator is built for. An export naming
#: several systems narrows to the one system that states it.
STORAGE_RING = "StorageRing"

#: What a reviewer may answer each virtual-accelerator slot kind with, as the
#: comment beside the null slot spells it. The vocabularies themselves are
#: closed by
#: :func:`~osprey.services.mml.mapping.schema.parse_mapping`, which refuses a
#: word outside them by name; these are the same words written for a reader.
VA_ANSWERS: dict[str, str] = {
    ATTYPE_KIND: "latch, strength:<PolynomB|PolynomA>[<i>], kick:<0|1>, energy, rf, monitor:<x|y>",
    SHARED_FIELD_KIND: "owner:<family>, latch",
    ESCAPE_HATCH_KIND: "latch, ignore_hook",
}

#: The keys a coupled family writes, in document order. A key the verdict
#: leaves undecided is omitted rather than written null: the model needs every
#: one of them, so a null would be a decision nobody made.
_VA_FAMILY_KEYS = ("kind", "element_field", "calibration", "nominal_source", "reason")


def _text(value: Any) -> str | None:
    """Return a stripped string when ``value`` is non-blank text, else ``None``."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _one_unit(value: Any) -> str | None:
    """Return the one unit ``value`` states, else ``None``.

    ``HWUnits`` is a scalar or one entry per device. A per-device list states
    one unit only when its non-blank entries agree.
    """
    if isinstance(value, (list, tuple)):
        distinct = {u for item in value if (u := _text(item)) is not None}
        return distinct.pop() if len(distinct) == 1 else None
    return _text(value)


def _fold_token(machine: str) -> str | None:
    """Fold a machine name to a PN_LOCAL token, or ``None`` when nothing is left."""
    token = _NOT_PN_LOCAL.sub("_", machine).strip("_")
    if not token:
        return None
    if not token[0].isalpha():
        token = f"_{token}"
    return token if is_pn_local(token) else None


def _systems(ao: dict) -> list[str]:
    """Return the system tokens: ``_import_order`` first, then the rest sorted."""
    present = sorted(raw for raw, _ in system_bodies(ao))
    order = ao.get("_import_order")
    listed = [s for s in order if s in present] if isinstance(order, list) else []
    return list(dict.fromkeys([*listed, *present]))


def _ad_block(ad: dict | None, system: str) -> dict:
    if not isinstance(ad, dict):
        return {}
    block = ad.get(system)
    return block if isinstance(block, dict) else {}


def section_order(ao: dict, systems: dict[str, dict]) -> list[str]:
    """Return each system's mapped ``name`` in system order.

    Args:
        ao: The merged export; only its system keys and ``_import_order`` are read.
        systems: The ``systems`` block, keyed by raw token, each with a ``name``.

    Returns:
        The names, ordered by ``_import_order`` with unlisted systems sorted
        after it (all systems sorted when ``_import_order`` is absent).
    """
    return [systems[raw]["name"] for raw in _systems(ao) if raw in systems]


def _facility(ad: dict | None, systems: list[str]) -> dict:
    machine = next(
        (m for s in systems if (m := _text(_ad_block(ad, s).get("Machine"))) is not None),
        None,
    )
    if machine is None and isinstance(ad, dict):
        machine = next(
            (m for b in ad.values() if isinstance(b, dict) and (m := _text(b.get("Machine")))),
            None,
        )
    if machine is None:
        return {"token": None, "title": None, "description": None, "provenance": DERIVED}
    return {
        "token": _fold_token(machine),
        "title": machine,
        "description": f"The {machine} accelerator facility, as exported by its MATLAB Middle Layer.",
        "provenance": DERIVED,
    }


def _system(raw: str, body: dict, ad_block: dict) -> dict:
    native = _text(body.get("_description"))
    if native is not None:
        return {"name": raw, "description": native, "provenance": IMPORTED}
    machine = _text(ad_block.get("Machine"))
    sub = _text(ad_block.get("SubMachine"))
    mode = _text(ad_block.get("OperationalMode"))
    if machine is None and sub is None and mode is None:
        return {"name": raw, "description": None, "provenance": DERIVED}
    prose = f"The {sub or raw} system"
    if machine is not None:
        prose += f" of {machine}"
    prose += "."
    if mode is not None:
        prose += f" Operational mode: {mode}."
    return {"name": raw, "description": prose, "provenance": DERIVED}


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _family_prose(raw: str, views: list[FamilyView]) -> str:
    systems = ", ".join(view.system for view in views)
    devices = sum(view.n_devices for view in views)
    channels = sum(view.channel_count for view in views)
    fields = list(dict.fromkeys(name for view in views for name in view.fields))
    prose = (
        f"MML family {raw} in {systems}: {_plural(devices, 'device')}, "
        f"{_plural(channels, 'channel')}"
    )
    if fields:
        prose += f" across fields {', '.join(fields)}"
    return prose + "."


def _field_prose(name: str, views: list[FamilyView]) -> str:
    fields = [view.fields[name] for view in views if name in view.fields]
    keys = list(dict.fromkeys(key for fld in fields for key in fld.keys))
    channels = sum(fld.channel_count for fld in fields)
    prose = f"Field {name}: {_plural(channels, 'channel')} via {', '.join(keys)}"
    if any(fld.broadcast for fld in fields):
        prose += ", one channel broadcast to every device"
    units = list(dict.fromkeys(u for fld in fields if (u := _one_unit(fld.body.get("HWUnits")))))
    if len(units) == 1:
        prose += f", hardware units {units[0]}"
    return prose + "."


def _family(raw: str, views: list[FamilyView]) -> dict:
    channels = sum(view.channel_count for view in views)
    native = next((view.description for view in views if view.description is not None), None)
    entry: dict[str, Any] = {}
    if channels > 0:
        entry["branch"] = None
        entry["class"] = raw
    entry["aliases"] = [raw]
    if native is not None:
        entry["description"], entry["provenance"] = native[0].strip(), IMPORTED
    else:
        entry["description"], entry["provenance"] = _family_prose(raw, views), DERIVED
    entry["channels"] = channels

    fields: dict[str, dict] = {}
    for name in dict.fromkeys(n for view in views for n in view.fields):
        text = next(
            (
                t
                for view in views
                if name in view.fields and (t := _text(view.fields[name].description))
            ),
            None,
        )
        if text is not None:
            fields[name] = {"description": text, "provenance": IMPORTED}
        else:
            fields[name] = {"description": _field_prose(name, views), "provenance": DERIVED}
    entry["fields"] = fields
    return entry


def _judgments(views: dict[str, list[FamilyView]]) -> dict[str, dict]:
    """Return the pending judgment slots of every family, unioned over systems.

    Args:
        views: Raw family views keyed by raw family token, in the merged family
            order, each list holding the views of the systems carrying it.

    Returns:
        ``{raw family: {kind: slots}}`` in family order, then kind order, with
        every slot ``None``. A family pending nothing has no entry, so an
        export pending nothing yields an empty dict.
    """
    block: dict[str, dict] = {}
    for raw, family in views.items():
        pending = [pending_judgments(view) for view in family]
        rows: dict[str, dict[str, None]] = {}
        for item in pending:
            for row in item.rows_beyond:
                rows.setdefault(row.field, {})[row.signal] = None
        ordinals = sorted({ordinal for item in pending for ordinal in item.unbound_devices})
        entry: dict[str, Any] = {}
        if rows:
            entry[ROWS_BEYOND_KIND] = rows
        if ordinals:
            entry[UNBOUND_KIND] = dict.fromkeys(ordinals)
        if any(item.groups for item in pending):
            entry[SHARED_KIND] = None
        if entry:
            block[raw] = entry
    return block


def count_judgment_slots(document: dict) -> int:
    """Return how many judgment slots a skeleton asks its reviewer to answer.

    Args:
        document: A mapping document, e.g. from :func:`build_skeleton`. A
            document with no ``judgments`` block asks nothing.

    Returns:
        One per row signal, one per unbound ordinal and one per family with
        shared PVs -- the number of null slots the block was written with.
    """
    total = 0
    for entry in document.get("judgments", {}).values():
        total += sum(len(signals) for signals in entry.get(ROWS_BEYOND_KIND, {}).values())
        total += len(entry.get(UNBOUND_KIND, {}))
        total += 1 if SHARED_KIND in entry else 0
    return total


def va_system(ao: dict, ad: dict | None, available: Container[str] | None = None) -> str | None:
    """Return the one system a virtual-accelerator block describes.

    A facility exports one sub-machine per system and the model is built for
    one of them. A single system is that one whatever it is called; several
    narrow to the one the AD calls a :data:`STORAGE_RING`, because that is the
    ring a virtual accelerator stands in for. Anything else is the reviewer's
    choice, which the skeleton writes as a null slot.

    Args:
        ao: The merged export; only its system keys and ``_import_order`` are
            read.
        ad: AD blocks keyed by system, or ``None`` when the export had none.
        available: The systems a 2.0 block was imported for, or ``None`` to
            consider every system of the export. A system outside it is not a
            candidate, so an export whose ring alone was re-exported resolves
            to that ring.

    Returns:
        The system token, or ``None`` when the export does not name one.
    """
    pool = [raw for raw in _systems(ao) if available is None or raw in available]
    if len(pool) == 1:
        return pool[0]
    rings = [raw for raw in pool if _text(_ad_block(ad, raw).get("MachineType")) == STORAGE_RING]
    return rings[0] if len(rings) == 1 else None


def va_block(verdicts: dict[str, VAFamily], system_choice: str | None) -> dict:
    """Build the ``virtual_accelerator`` block of a mapping document.

    Args:
        verdicts: What the rules reached per family, as
            :func:`~osprey.services.mml.va.verdicts.propose` returns it.
        system_choice: The system the block describes, as :func:`va_system`
            resolved it; ``None`` leaves the slot for the reviewer.

    Returns:
        The block as plain dicts, in the order it should be written: the
        system, then every family sorted by name. A family writes its verdict
        and only the keys that verdict decides, and an open slot writes its
        kind, its question and a null answer.
    """
    families: dict[str, dict] = {}
    for name in sorted(verdicts):
        verdict = verdicts[name]
        entry: dict[str, Any] = {"verdict": verdict.verdict}
        for key in _VA_FAMILY_KEYS:
            value = getattr(verdict, key)
            if value is not None:
                entry[key] = value
        if verdict.slot is not None:
            entry["slot"] = {
                "kind": verdict.slot.kind,
                "question": verdict.slot.question,
                "answer": None,
            }
        families[name] = entry
    return {"system": system_choice, "families": families}


def count_va_slots(block: dict) -> int:
    """Return how many virtual-accelerator slots a block asks a reviewer to answer.

    Args:
        block: A block, e.g. from :func:`va_block`.

    Returns:
        One for an undecided system, plus one per family a rule left a slot
        on -- the number of null slots ``map --check`` refuses.
    """
    families = block.get("families", {})
    open_slots = sum(1 for entry in families.values() if entry.get("slot") is not None)
    return open_slots + (1 if block.get("system") is None else 0)


def dump_va_block(block: dict) -> str:
    """Serialise the block as the text ``map --init`` appends to a mapping.

    Every open slot is followed by a comment naming what it may be answered
    with, so the vocabulary reads beside the question rather than in the
    documentation. Comments are the only thing here that
    :func:`~osprey.services.mml.mapping.schema.parse_mapping` never sees.

    Args:
        block: The block, e.g. from :func:`va_block`.

    Returns:
        The YAML text of a document holding ``virtual_accelerator`` alone,
        ending in a newline.
    """
    lines: list[str] = []
    kind: str | None = None
    for line in dump_yaml({"virtual_accelerator": block}).splitlines():
        stated = line.strip()
        if stated.startswith("kind: "):
            spelled = stated.removeprefix("kind: ")
            kind = spelled if spelled in VA_ANSWERS else None
        lines.append(line)
        if kind is not None and stated == "answer: null":
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}# answers: {VA_ANSWERS[kind]}")
            kind = None
    return "".join(f"{line}\n" for line in lines)


def build_skeleton(ao: dict, ad: dict | None, votes: dict[tuple[str, str], Vote]) -> dict:
    """Build the ``mapping.yaml`` skeleton for a merged export.

    Args:
        ao: ``{system: {family: body}}`` with optional ``_import_order``;
            ``_``-prefixed and non-dict entries are skipped. Not modified.
        ad: AD blocks keyed by system, or ``None`` when the export had none.
        votes: Direction votes keyed ``(raw_family, field)``, as returned by
            :func:`~osprey.services.mml.directions.vote_directions`.

    Returns:
        The document as plain dicts and lists, in the order it should be
        written, with ``judgments`` last and only when something pends; it
        parses with
        :func:`~osprey.services.mml.mapping.schema.parse_mapping`.
    """
    order = _systems(ao)

    systems = {raw: _system(raw, ao[raw], _ad_block(ad, raw)) for raw in order}

    views: dict[str, list[FamilyView]] = {}
    for system in order:
        for view in family_views(system, ao[system]):
            views.setdefault(view.raw_name, []).append(view)

    families = {raw: _family(raw, family_views) for raw, family_views in views.items()}

    directions: dict[str, dict] = {}
    for raw, entry in families.items():
        for name in entry["fields"]:
            vote = votes.get((raw, name))
            directions[f"{raw}.{name}"] = {
                "direction": None if vote is None else vote.direction,
                "provenance": DERIVED,
                "override": False,
            }

    document: dict[str, Any] = {
        "facility": _facility(ad, order),
        "systems": systems,
        "section_order": section_order(ao, systems),
        "families": families,
        "directions": directions,
    }
    judgments = _judgments(views)
    if judgments:
        document["judgments"] = judgments
    return document


def dump_yaml(data: dict) -> str:
    """Serialise a mapping document as block-style YAML in insertion order.

    Args:
        data: The document, e.g. from :func:`build_skeleton`.

    Returns:
        The YAML text; ``None`` is written as ``null``.
    """
    return yaml.safe_dump(
        data, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100
    )
