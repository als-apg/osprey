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

Grain numbers come from :class:`~osprey.services.mml.family.FamilyView`. The
module has no I/O; :func:`dump_yaml` only serialises.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from osprey.services.mml.directions import Vote
from osprey.services.mml.family import FamilyView, family_views, system_bodies
from osprey.services.mml.judgments import pending_judgments
from osprey.services.mml.mapping.branches import is_pn_local
from osprey.services.mml.mapping.schema import ROWS_BEYOND_KIND, SHARED_KIND, UNBOUND_KIND

__all__ = ["build_skeleton", "count_judgment_slots", "dump_yaml", "section_order"]

#: Provenance of prose and directions generated from export facts.
DERIVED = "derived"

#: Provenance of a description carried over from the export.
IMPORTED = "imported"

#: Characters a PN_LOCAL token may not contain, folded to ``_``.
_NOT_PN_LOCAL = re.compile(r"[^A-Za-z0-9_]+")


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
