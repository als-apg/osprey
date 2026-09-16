"""The MML source: a Middle Layer family grain turned into corpus nodes.

The demo grammar path (:func:`.model.build_model`) reads a six-token address
tree. An MML install has no such tree; its grain is the family view, and every
semantic decision (system names, family renames, prose, the facility token)
comes from a parsed mapping. This module builds :class:`.model.Device` nodes
from those two inputs directly.

Rules:

* One device per ``(system, family, index)`` for a family with at least one
  channel. ``index`` counts from 1 within the family.
* The address is ``(system, system, family, index)`` in mapped tokens, and the
  device's local name is ``{system}_{family}_{index}``.
* A per-device array is used only when the family view reports it aligned; a
  blank or unusable slot falls back for that device alone.
* ``rawType`` is the export's own word: the device's ``DeviceType`` slot, or
  the raw family token when the export states none. A rename never reaches it.
* ``s_position_m`` is always finite: a missing, string or non-finite position
  stands in the device's ordinal and marks the device ``positionSource:
  "ordinal"``.
* One binding per non-blank slot per channel key: ``ChannelNames`` binds as
  protocol ``ca`` with subfield ``val``, ``TangoNames`` as ``tango`` with
  subfield ``tango``. ``fullPv`` is the slot stripped, never the address.
* A 1-row list on a family of more than one device binds every device to the
  same PV and marks each binding ``broadcast: 1``; 0-length and partial lists
  bind the slots they have. A slot beyond the family's devices is refused.
* A binding's description is the mapping's field description. ``HWUnits`` and
  ``DataType`` ride on a binding only as scalars: slot i of a list of
  ``n_devices`` entries, or a non-blank string; any other shape adds nothing.
* One signal group per ``(family, field, subfield)`` in mapped tokens; the
  mapping's ``<raw family>.<field>`` direction directs every subfield, and the
  model refuses any undirected group and any repeated device IRI.

The module is pure and depends on the standard library, the family view and
the mapping schema; it imports no rdflib, neo4j or YAML.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from osprey.services.mml.family import FamilyView, FieldView, family_views, system_bodies
from osprey.services.mml.mapping.schema import Family, Mapping, System

from .model import (
    CONFIDENCE,
    Address,
    ChannelBinding,
    Device,
    GraphModel,
    SignalGroup,
    binding_id,
    binding_iri,
    device_id,
    device_iri,
    normalize_extra_properties,
    signal_iri,
    signal_name,
    source_section_id,
)

__all__ = [
    "BINDING_KEYS",
    "POSITION_SOURCE_ORDINAL",
    "bindings_for_family",
    "build_graph_model",
    "devices_for_family",
]

#: ``positionSource`` value on a device whose position is its ordinal stand-in.
POSITION_SOURCE_ORDINAL = "ordinal"

#: ``(protocol, subfield)`` a channel key binds as, in ``CHANNEL_KEYS`` order.
BINDING_KEYS: dict[str, tuple[str, str]] = {
    "ChannelNames": ("ca", "val"),
    "TangoNames": ("tango", "tango"),
}

#: Field keys copied onto a binding when their value is scalar for its device.
_SCALAR_FIELD_KEYS: tuple[str, ...] = ("HWUnits", "DataType")


def _text(value: Any) -> str | None:
    """Return the stripped string when ``value`` is a non-blank string."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _finite_number(value: Any) -> float | None:
    """Return ``value`` as a float when it is a finite, non-boolean number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    """Return ``value`` as an int when it is a boolean, int or integral float."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _slot(slots: list | None, index: int) -> Any:
    return None if slots is None else slots[index]


def _resolve(view: FamilyView, mapping: Mapping, iri: str) -> tuple[str, str, str, System, Family]:
    """Resolve the tokens and mapping entries every node of ``view`` is minted from.

    Args:
        view: The family's computed grain.
        mapping: The parsed mapping.
        iri: The kind of IRI the caller mints, named when the token is missing.

    Returns:
        ``(facility token, mapped system name, mapped family token, system
        entry, family entry)``.

    Raises:
        ValueError: The mapping has no facility token, or does not name the
            view's system or family.
    """
    facility = mapping.facility.token
    if not facility:
        raise ValueError(f"mapping has no facility token; every {iri} IRI embeds one")
    system_entry = mapping.systems.get(view.system)
    if system_entry is None:
        raise ValueError(f"mapping names no system {view.system!r}")
    family_entry = mapping.families.get(view.raw_name)
    if family_entry is None:
        raise ValueError(f"mapping names no family {view.raw_name!r} (system {view.system!r})")
    return facility, system_entry.name, mapping.mapped(view.raw_name), system_entry, family_entry


def devices_for_family(view: FamilyView, mapping: Mapping) -> list[Device]:
    """Build the devices of one family.

    Args:
        view: The family's computed grain.
        mapping: The parsed mapping naming the family's system, family token,
            prose and facility token.

    Returns:
        One :class:`Device` per index, in index order; empty when the family
        has no channel. Ordinals are the 1-based position within the family and
        ``binding_iris`` is empty, both for the model builder to restamp.

    Raises:
        ValueError: The mapping has no facility token, or does not name the
            view's system or family.
    """
    if view.channel_count == 0:
        return []
    facility, system, family, system_entry, family_entry = _resolve(view, mapping, "device")

    common_names = view.aligned("CommonNames")
    device_types = view.aligned("DeviceType")
    positions = view.aligned("Position")
    statuses = view.aligned("Status")
    elements = view.aligned("ElementList")
    rows = view.device_rows if view.device_rows is not None else view.aligned("DeviceList")

    devices: list[Device] = []
    for i in range(view.n_devices):
        ordinal = i + 1
        token = str(ordinal)
        name = f"{family}_{token}"
        extras: dict[str, str | int | float] = {}

        position = _finite_number(_slot(positions, i))
        if position is None:
            position = float(ordinal)
            extras["positionSource"] = POSITION_SOURCE_ORDINAL

        status = _integer(_slot(statuses, i))
        if status is not None:
            extras["status"] = status

        row = _slot(rows, i)
        if isinstance(row, (list, tuple)) and len(row) == 2:
            sector, number = _integer(row[0]), _integer(row[1])
            if sector is not None and number is not None:
                extras["sector"] = sector
                extras["device"] = number

        element = _integer(_slot(elements, i))
        if element is not None:
            extras["elementIndex"] = element

        devices.append(
            Device(
                ring=system,
                system=system,
                family=family,
                device=token,
                source_name=_text(_slot(common_names, i)) or f"{family}{token}",
                section_code=system,
                raw_type=_text(_slot(device_types, i)) or view.raw_name,
                ordinal_in_section=ordinal,
                ordinal_in_facility=ordinal,
                s_position_m=position,
                iri=device_iri(system, name, facility=facility),
                device_id=device_id(system, name, facility=facility),
                source_section_id=source_section_id(system, facility=facility),
                binding_iris=(),
                family_description=family_entry.description,
                system_description=system_entry.description,
                ring_description=system_entry.description,
                extra_properties=normalize_extra_properties(extras),
            )
        )
    return devices


def _field_scalar(field: FieldView, key: str, index: int, n_devices: int) -> str | None:
    """Return the scalar ``field.body[key]`` gives device ``index``, or ``None``.

    Slot ``index`` of a list of ``n_devices`` entries when it is a non-blank
    string, the value itself when it is a non-blank string, otherwise nothing.
    """
    value = field.body.get(key)
    if isinstance(value, (list, tuple)):
        return _text(value[index]) if len(value) == n_devices else None
    return _text(value)


def bindings_for_family(
    view: FamilyView, mapping: Mapping
) -> tuple[list[ChannelBinding], list[SignalGroup]]:
    """Build the channel bindings and signal groups of one family.

    Args:
        view: The family's computed grain.
        mapping: The parsed mapping naming the family's system, family token,
            field prose and facility token.

    Returns:
        The bindings in device order, then field order, then channel-key order;
        and one undirected :class:`SignalGroup` per ``(family, field,
        subfield)`` in first-binding order, its members the bindings' PVs. Both
        are empty for a family with no channel.

    Raises:
        ValueError: The mapping has no facility token or does not name the
            view's system or family, or a channel list is longer than the
            family has devices.
    """
    if view.channel_count == 0:
        return [], []
    facility, system, family, _, family_entry = _resolve(view, mapping, "binding")
    n_devices = view.n_devices

    per_device: list[list[ChannelBinding]] = [[] for _ in range(n_devices)]
    members: dict[tuple[str, str, str], list[tuple[int, str]]] = {}
    for field in view.fields.values():
        field_entry = family_entry.fields.get(field.name)
        description = None if field_entry is None else field_entry.description
        for key in field.keys:
            protocol, subfield = BINDING_KEYS[key]
            slots = field.slots(key)
            if len(slots) > n_devices:
                raise ValueError(
                    f"{view.system}.{view.raw_name}.{field.name}.{key} lists {len(slots)} "
                    f"channels for {n_devices} devices; a slot beyond the last device "
                    "has no device to bind to"
                )
            broadcast = len(slots) != len(field.raw_slots(key))
            group_key = (family, field.name, subfield)
            group_name = signal_name(*group_key)
            for i, slot in enumerate(slots):
                pv = _text(slot)
                if pv is None:
                    continue
                token = str(i + 1)
                name = f"{family}_{token}"
                extras: dict[str, str | int | float] = {}
                if broadcast:
                    extras["broadcast"] = 1
                for scalar_key in _SCALAR_FIELD_KEYS:
                    scalar = _field_scalar(field, scalar_key, i, n_devices)
                    if scalar is not None:
                        extras[scalar_key] = scalar
                address = Address(system, system, family, token, field.name, subfield)
                per_device[i].append(
                    ChannelBinding(
                        address=address,
                        full_pv=pv,
                        protocol=protocol,
                        confidence=CONFIDENCE,
                        iri=binding_iri(system, name, field.name, subfield, facility=facility),
                        binding_id=binding_id(
                            system, name, field.name, subfield, facility=facility
                        ),
                        device_iri=device_iri(system, name, facility=facility),
                        device_key=address.device_key,
                        signal_key=address.signal_key,
                        signal_name=group_name,
                        signal_iri=signal_iri(group_name),
                        description=description,
                        extra_properties=normalize_extra_properties(extras),
                    )
                )
                members.setdefault(group_key, []).append((i, pv))

    groups = [
        SignalGroup(
            family=group_family,
            field=group_field,
            subfield=group_subfield,
            name=signal_name(group_family, group_field, group_subfield),
            iri=signal_iri(signal_name(group_family, group_field, group_subfield)),
            members=tuple(pv for _, pv in sorted(entries, key=lambda entry: entry[0])),
        )
        for (group_family, group_field, group_subfield), entries in members.items()
    ]
    return [binding for bindings in per_device for binding in bindings], groups


def build_graph_model(ao: dict, mapping: Mapping, section_order: list[str]) -> GraphModel:
    """Build the complete, directed graph model of an MML install.

    Args:
        ao: The canonical ``ao.json`` document, keyed by raw system token.
        mapping: The parsed mapping; its ``directions`` are the only direction
            source.
        section_order: Mapped system names in the order the corpus lists them.

    Returns:
        A :class:`GraphModel` whose devices follow ``section_order``, then the
        ``ao.json`` family order (key-sorted), then index; whose bindings follow
        their devices;
        and whose signal groups are sorted by key and all directed.

    Raises:
        ValueError: The order and ``ao`` disagree on the systems, the mapping
            does not name a system or family, a group is left undirected (the
            first by key is named), or two devices share an IRI.
    """
    facility = mapping.facility.token
    if not facility:
        raise ValueError("mapping has no facility token; every IRI embeds one")

    devices: list[Device] = []
    bindings: list[ChannelBinding] = []
    members: dict[tuple[str, str, str], list[str]] = {}
    present = [raw for raw, _ in system_bodies(ao)]
    ordered = mapping.ordered_systems(present, section_order)
    for raw_system in (raw for _, raws in ordered for raw in raws):
        section_ordinal = 0
        for view in family_views(raw_system, ao[raw_system]):
            family_bindings, family_groups = bindings_for_family(view, mapping)
            by_device: dict[str, list[str]] = {}
            for binding in family_bindings:
                by_device.setdefault(binding.device_iri, []).append(binding.iri)
            for device in devices_for_family(view, mapping):
                section_ordinal += 1
                devices.append(
                    replace(
                        device,
                        ordinal_in_section=section_ordinal,
                        ordinal_in_facility=len(devices) + 1,
                        binding_iris=tuple(by_device.get(device.iri, ())),
                    )
                )
            bindings.extend(family_bindings)
            for group in family_groups:
                members.setdefault(group.key, []).extend(group.members)

    seen: set[str] = set()
    for device in devices:
        if device.iri in seen:
            raise ValueError(
                f"device IRI {device.iri!r} is minted twice; two families or systems map "
                "to the same tokens"
            )
        seen.add(device.iri)

    directions: dict[tuple[str, str, str], str] = {}
    for key, entry in mapping.directions.items():
        raw_family, _, field = key.partition(".")
        if entry.direction is None or raw_family not in mapping.families:
            continue
        family = mapping.mapped(raw_family)
        for _, subfield in BINDING_KEYS.values():
            directions[(family, field, subfield)] = entry.direction

    groups = tuple(
        SignalGroup(
            family=family,
            field=field,
            subfield=subfield,
            name=signal_name(family, field, subfield),
            iri=signal_iri(signal_name(family, field, subfield)),
            members=tuple(members[(family, field, subfield)]),
        )
        for family, field, subfield in sorted(members)
    )
    model = GraphModel(
        facility=facility,
        devices=tuple(devices),
        bindings=tuple(bindings),
        signal_groups=groups,
    ).with_directions(directions)
    for group in model.signal_groups:
        if group.direction is None:
            raise ValueError(
                f"signal group ({group.family}, {group.field}, {group.subfield}) has no "
                f"direction; set directions.<raw family>.{group.field} in the mapping"
            )
    return model
