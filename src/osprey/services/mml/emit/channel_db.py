"""The channel database ``osprey mml emit`` writes as ``middle_layer.json``.

:func:`build_channel_db` turns the canonical ``ao.json`` and the parsed mapping
into the depth-3 dict ``MiddleLayerDatabase.load_database`` reads: systems,
then families, then fields.

Rules:

* Top-level system keys are the mapped system names, inserted in
  ``section_order``; ``list_systems()`` reads file order, so the serializer
  must keep ``sort_keys=False``. A system with no emitted family is left out.
* Families are keyed by their mapped token and sorted; a family whose mapping
  entry has ``channels: 0``, or that binds no channel in this system, is
  omitted.
* A family dict holds ``_description`` (the mapping's family prose), then
  ``setup``, then its fields sorted by name. ``setup`` carries ``DeviceList``
  normalised to Nx2 plus ``CommonNames``/``Status``/``Position`` when present;
  a family with no ``DeviceList`` carries only ``CommonNames``/``Status``. Such
  an array is carried only when it has one slot per device, the rule the corpus
  applies, so both paradigms name the same device the same way.
* A field dict carries exactly the channel keys the field has, ``Description``
  from the mapping, and the loader's metadata keys :data:`FIELD_METADATA_KEYS`;
  every other key is dropped. Blank slots are written ``""``, a 1-row list is
  broadcast to ``n_devices``, and short lists are kept as exported.
* System prose sits under ``_description``; one top-level ``_provenance``
  STRING names the inputs. It is a string, never a mapping, so the loader's
  system census skips it.
* Every value is copied, so the result never aliases ``ao``.

Pure: stdlib plus the family view, the mapping schema and the emit context.
"""

from __future__ import annotations

import copy
from typing import Any

from osprey.services.mml.emit.context import EmitContext
from osprey.services.mml.family import FamilyView, family_views, system_bodies
from osprey.services.mml.mapping.schema import Mapping

__all__ = ["FIELD_METADATA_KEYS", "PROVENANCE_KEY", "build_channel_db"]

#: Field metadata the middle-layer loader copies onto each channel, in order.
FIELD_METADATA_KEYS: tuple[str, ...] = (
    "DataType",
    "Mode",
    "Units",
    "HWUnits",
    "PhysicsUnits",
    "MemberOf",
    "Range",
    "Tolerance",
)

#: Top-level key of the provenance string.
PROVENANCE_KEY = "_provenance"

_DESCRIPTION_KEY = "_description"
_SETUP_KEY = "setup"

#: Family arrays ``setup`` carries beside a ``DeviceList``, and without one.
_SETUP_WITH_DEVICES: tuple[str, ...] = ("CommonNames", "Status", "Position")
_SETUP_WITHOUT_DEVICES: tuple[str, ...] = ("CommonNames", "Status")


def build_channel_db(ao: dict, mapping: Mapping, ctx: EmitContext) -> dict:
    """Build the middle-layer channel database of an MML install.

    Args:
        ao: The canonical ``ao.json`` document, keyed by raw system token.
        mapping: The parsed mapping; ``section_order`` holds mapped system
            names, ``systems`` and ``families`` are keyed by raw token.
        ctx: The provenance of this emit run.

    Returns:
        The database dict, ready for ``json.dumps(..., sort_keys=False)``.

    Raises:
        ValueError: ``section_order`` and ``ao`` disagree on the systems, the
            mapping does not name an ``ao`` family, or two families map to the
            same token within one system.
    """
    db: dict[str, Any] = {}
    present = [raw for raw, _ in system_bodies(ao)]
    for name, raw_systems in mapping.ordered_systems(present):
        families: dict[str, dict] = {}
        for raw_system in raw_systems:
            for view in family_views(raw_system, ao[raw_system]):
                entry = _family_entry(view, mapping)
                if entry is None:
                    continue
                token = mapping.mapped(view.raw_name)
                if token in families:
                    raise ValueError(
                        f"family token {token!r} is emitted twice in system {name!r}; "
                        "two families map to the same token"
                    )
                families[token] = entry
        if not families:
            continue

        system: dict[str, Any] = {}
        prose = next(
            (
                mapping.systems[raw].description
                for raw in raw_systems
                if mapping.systems[raw].description
            ),
            None,
        )
        if prose is not None:
            system[_DESCRIPTION_KEY] = prose
        for token in sorted(families):
            system[token] = families[token]
        db[name] = system

    db[PROVENANCE_KEY] = ctx.provenance_string
    return db


def _family_entry(view: FamilyView, mapping: Mapping) -> dict | None:
    """Return the family dict for ``view``, or ``None`` when it is omitted."""
    family = mapping.families.get(view.raw_name)
    if family is None:
        raise ValueError(
            f"mapping has no family {view.raw_name!r} (system {view.system!r} in ao.json)"
        )
    if family.channels == 0 or view.channel_count == 0:
        return None

    entry: dict[str, Any] = {}
    if family.description is not None:
        entry[_DESCRIPTION_KEY] = family.description
    setup = _setup(view)
    if setup:
        entry[_SETUP_KEY] = setup
    for name in sorted(view.fields):
        field_view = view.fields[name]
        out: dict[str, Any] = {key: _blank(field_view.slots(key)) for key in field_view.keys}
        mapped_field = family.fields.get(name)
        if mapped_field is not None and mapped_field.description is not None:
            out["Description"] = mapped_field.description
        for key in FIELD_METADATA_KEYS:
            if key in field_view.body:
                out[key] = copy.deepcopy(field_view.body[key])
        entry[name] = out
    return entry


def _setup(view: FamilyView) -> dict[str, Any]:
    """Return the ``setup`` block: DeviceList-bearing or names/status only."""
    device_list = view.device_rows
    keys = _SETUP_WITH_DEVICES if device_list is not None else _SETUP_WITHOUT_DEVICES
    setup: dict[str, Any] = {}
    if device_list is not None:
        setup["DeviceList"] = copy.deepcopy(device_list)
    for key in keys:
        if key not in view.arrays:
            continue
        value = view.arrays[key]
        if not isinstance(value, (list, tuple)):
            setup[key] = copy.deepcopy(value)
            continue
        if view.aligned(key) is None:
            continue
        setup[key] = _blank(value)
    return setup


def _blank(slots: list | tuple) -> list:
    """Return a copy of ``slots`` with every ``None`` written ``""``."""
    return ["" if slot is None else copy.deepcopy(slot) for slot in slots]
