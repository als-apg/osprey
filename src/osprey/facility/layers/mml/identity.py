"""The mml layer's identity model: which device each export slot is.

An address is never a device's identity: one supply may feed several magnets,
so one address may be bound at several slots that are distinct devices. The
rules, each read from the export alone:

* **CommonName by position.** Slot ``i`` of every field of a family is one
  device, named by the family's ``CommonNames`` entry at the same position.
  Its id is ``<model>/<local name>``: the model the device's system is
  imported as, and the name with a leading ``<system>_`` or ``<system> ``
  stripped and every run of other than letters and digits made one ``_``.
* **System of a slot.** A slot belongs to the system its addresses name: the
  first address whose leading ``:``-separated segment (else its segment
  before a run of three or more ``_``) is one of the export's system tokens,
  compared case-blind; otherwise the system the family sits under. Names
  prefixed with a system token align, in order, with that system's slots; the
  remaining names align, in order, with the slots left over.
* **Dedup by name.** Slots resolving to one id, within a family, across
  families or across systems, are one device.
* **The mapping decides what the export leaves out.** A family's ``devices``
  slot in the mapping says how its slots are identified: ``names`` is the
  rule above; a list gives each slot its local name, one per device;
  ``address`` names each slot by the device segment of its first address that
  has one (an address with a ``{...}`` group: the text through the group's
  closing brace, less a leading segment ending in a ``:`` before the brace;
  else the segment after the first ``:``; else after the first run of three
  or more ``_``), else ``<family>_<i>``, an id already taken gaining the
  first free ``_<n>``, ``n`` counting up from ``i``, the slot's 0-based
  position; ``{same_as: <family>}`` makes slot ``i`` the device slot ``i`` of
  the named family of the same system is. A family the mapping leaves the
  slot out for is identified by ``names``; one with a slot the export names
  no device for stops the import until the mapping decides
  (``mapping-undecided``), so no id is ever guessed.
* **Shared endpoints.** An address bound by several devices is one channel
  that names each of them (:func:`endpoints`).

No token, segment list or skipped family is a constant here: systems come
from the export, skipped families from the mapping.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from osprey.facility.layers.mml.mapping import (
    ENGINE_AXES,
    DeviceIdentity,
    ImportStop,
    SameAs,
)

if TYPE_CHECKING:  # the export services stay out of the import graph
    from osprey.services.mml.family import FamilyView

__all__ = ["axis_twins", "common_class", "device_ids", "endpoints", "stated_ids"]

#: Separators a name's leading system token may carry.
_PREFIX_SEPARATORS: tuple[str, ...] = ("_", " ")

#: A run of three or more ``_``: the segment separator of an underscore address.
_UNDERSCORE_SEPARATOR = re.compile(r"_{3,}")

#: Every run of characters other than letters and digits.
_NON_WORD = re.compile(r"[^0-9A-Za-z]+")

#: The same runs, kept, so an address splits into its words and separators.
_WORD_BREAK = re.compile(r"([^0-9A-Za-z]+)")

#: What a mapping without a ``devices`` answer is told to write.
_ANSWER = "write address, a list of names or {same_as: <family>}"


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _word(text: str) -> str:
    return _NON_WORD.sub("_", text).strip("_")


def _system_of_name(name: str, systems: Sequence[str]) -> str | None:
    """The system a name's leading token states, or ``None``."""
    upper = name.upper()
    for system in systems:
        if any(upper.startswith(f"{system.upper()}{sep}") for sep in _PREFIX_SEPARATORS):
            return system
    return None


def _system_of_slot(addresses: Sequence[str], systems: Sequence[str], default: str) -> str:
    """The system a slot's addresses name, else ``default``."""
    by_upper = {system.upper(): system for system in systems}
    for address in addresses:
        if ":" in address:
            found = by_upper.get(address.split(":", 1)[0].upper())
            if found is not None:
                return found
    for address in addresses:
        parts = _UNDERSCORE_SEPARATOR.split(address)
        if len(parts) >= 2:
            found = by_upper.get(parts[0].rstrip("_").upper())
            if found is not None:
                return found
    return default


def _local_name(name: str, system: str) -> str:
    """A name without its system prefix, as one URI-safe word."""
    for sep in _PREFIX_SEPARATORS:
        prefix = f"{system}{sep}"
        if name.upper().startswith(prefix.upper()):
            name = name[len(prefix) :]
            break
    return _word(name)


def _address_token(address: str) -> str:
    """The device segment of an address, or ``""``."""
    opening = address.find("{")
    closing = address.find("}", opening)
    if 0 <= opening < closing:
        head = address[: closing + 1]
        colon = head.find(":")
        if 0 <= colon < opening:
            head = head[colon + 1 :]
        if _word(head):
            return _word(head)
    if ":" in address:
        parts = address.split(":")
        if parts[1].strip():
            return _word(parts[1].strip())
    parts = _UNDERSCORE_SEPARATOR.split(address)
    if len(parts) >= 2 and parts[1].strip():
        return _word(parts[1].strip())
    return ""


def _slot_addresses(view: FamilyView, index: int) -> list[str]:
    """Every address bound at one slot, in field and key order."""
    found: list[str] = []
    for fld in view.fields.values():
        for key in fld.keys:
            slots = fld.slots(key)
            address = _text(slots[index]) if index < len(slots) else None
            if address is not None:
                found.append(address)
    return found


def _roster(view: FamilyView, systems: Sequence[str]) -> tuple[dict[str, list[str]], list[str]]:
    """A family's names: by the system each states, and the rest (blanks too) in order."""
    names = view.aligned("CommonNames") or []
    by_system: dict[str, list[str]] = {}
    rest: list[str] = []
    for value in names:
        name = "" if value is None else str(value).strip()
        system = _system_of_name(name, systems) if name else None
        if system is None:
            rest.append(name)
        else:
            by_system.setdefault(system, []).append(name)
    return by_system, rest


def stated_ids(
    view: FamilyView, systems: Sequence[str], models: Mapping[str, str]
) -> list[str | None]:
    """The id the export's ``CommonNames`` give each slot of a family.

    Args:
        view: One family of one system.
        systems: The export's raw system tokens, in import order.
        models: The model each system is imported as, keyed by raw token.

    Returns:
        One entry per slot: ``<model>/<local name>``, or ``None`` where the
        export names no device.
    """
    by_system, rest = _roster(view, systems)
    seen = dict.fromkeys(by_system, 0)
    unsectioned = 0
    ids: list[str | None] = []
    for index in range(view.n_devices):
        system = _system_of_slot(_slot_addresses(view, index), systems, view.system)
        name: str | None = None
        stated = by_system.get(system, [])
        if seen.get(system, 0) < len(stated):
            name = stated[seen[system]]
            seen[system] += 1
        elif unsectioned < len(rest):
            name = rest[unsectioned]
            unsectioned += 1
        local = _local_name(name, system) if name is not None else ""
        ids.append(f"{models[system]}/{local}" if local else None)
    return ids


def _address_ids(
    view: FamilyView, systems: Sequence[str], models: Mapping[str, str], taken: set[str]
) -> list[str]:
    """Each slot's id from its address, no id handed to two slots."""
    ids: list[str] = []
    for index in range(view.n_devices):
        addresses = _slot_addresses(view, index)
        system = _system_of_slot(addresses, systems, view.system)
        token = next((t for t in map(_address_token, addresses) if t), "")
        base = f"{models[system]}/{token or f'{_word(view.raw_name)}_{index}'}"
        device, suffix = base, index
        while device in taken:
            device = f"{base}_{suffix}"
            suffix += 1
        taken.add(device)
        ids.append(device)
    return ids


def _listed_ids(
    view: FamilyView, names: Sequence[str], systems: Sequence[str], models: Mapping[str, str]
) -> list[str]:
    """Each slot's id from the mapping's own list of local names."""
    ids: list[str] = []
    for index, name in enumerate(names):
        system = _system_of_slot(_slot_addresses(view, index), systems, view.system)
        ids.append(f"{models[system]}/{name}")
    return ids


def device_ids(
    views: Sequence[FamilyView],
    models: Mapping[str, str],
    devices: Mapping[str, DeviceIdentity] | None = None,
) -> list[list[str]]:
    """The device id of every slot of every family.

    Args:
        views: The imported families, in import order; each view's ``system``
            is a key of ``models``.
        models: The model each system is imported as, keyed by raw system
            token, in import order.
        devices: The mapping's ``devices`` answer of each family that has
            one, keyed by raw family token.

    Returns:
        One list per view, holding the id of each of its ``n_devices`` slots.

    Raises:
        ImportStop: ``mapping-undecided`` for a family without an answer
            whose export names no device at some slot; else
            ``mapping-invalid`` for an answer its export cannot carry: a
            ``names`` family with such a slot, a list of another length than
            the family's devices, or a ``same_as`` naming a family the system
            does not carry, one of another device count, or one not itself
            identified by names or a list.
    """
    systems = list(models)
    forms = devices or {}
    taken: set[str] = set()
    result: list[list[str] | None] = []
    undecided: dict[str, str] = {}
    invalid: list[str] = []
    for view in views:
        raw = view.raw_name
        key = f"families.{raw}.devices"
        form = forms.get(raw)
        ids: list[str] | None = None
        if isinstance(form, SameAs):
            result.append(None)
            continue
        if form == "address":
            ids = _address_ids(view, systems, models, taken)
        elif form is None or form == "names":
            stated = stated_ids(view, systems, models)
            named = [device for device in stated if device is not None]
            if len(named) == len(stated):
                ids = named
            elif form is None:
                undecided.setdefault(raw, f"{key}: {_ANSWER}")
            else:
                invalid.append(
                    f"{key}: {raw} in {view.system} names no device "
                    f"{stated.index(None) + 1}; {_ANSWER}"
                )
        elif len(form) != view.n_devices:
            invalid.append(
                f"{key}: lists {len(form)} names and {raw} has {view.n_devices} devices "
                f"in {view.system}"
            )
        else:
            ids = _listed_ids(view, form, systems, models)
        if ids is not None:
            taken.update(ids)
        result.append(ids)

    carried = {(view.system, view.raw_name): index for index, view in enumerate(views)}
    for index, view in enumerate(views):
        form = forms.get(view.raw_name)
        if not isinstance(form, SameAs):
            continue
        key = f"families.{view.raw_name}.devices"
        other = forms.get(form.family)
        found = carried.get((view.system, form.family))
        if isinstance(other, SameAs) or other == "address":
            invalid.append(
                f"{key}: {form.family} is not identified by names or a list; name a family that is"
            )
        elif found is None:
            invalid.append(f"{key}: {view.system} carries no family {form.family}")
        elif views[found].n_devices != view.n_devices:
            invalid.append(
                f"{key}: {view.raw_name} has {view.n_devices} devices in {view.system} "
                f"and {form.family} has {views[found].n_devices}"
            )
        else:
            result[index] = result[found]

    if undecided:
        raise ImportStop("mapping-undecided", list(undecided.values()))
    if invalid:
        raise ImportStop("mapping-invalid", invalid)
    return [list(ids or ()) for ids in result]


def _one_axis_apart(first: str, second: str) -> bool:
    """Whether two addresses differ in exactly one word, an axis on each side."""
    ours, theirs = _WORD_BREAK.split(first), _WORD_BREAK.split(second)
    if len(ours) != len(theirs):
        return False
    apart = [(a.lower(), b.lower()) for a, b in zip(ours, theirs, strict=True) if a != b]
    return len(apart) == 1 and set(apart[0]) == set(ENGINE_AXES)


def axis_twins(first: FamilyView, second: FamilyView) -> bool:
    """Whether two families of one system bind the two axes of the same devices.

    Args:
        first: One family.
        second: Another family of the same system.

    Returns:
        ``True`` when both have the same device count, share a field, every
        slot of every shared field binds either one address or two that
        differ in exactly one word, ``x`` on one side and ``y`` on the other,
        and at least one slot does differ.
    """
    if first.n_devices != second.n_devices:
        return False
    differs = False
    for name, fld in first.fields.items():
        other = second.fields.get(name)
        if other is None:
            continue
        if list(fld.keys) != list(other.keys):
            return False
        for key in fld.keys:
            ours, theirs = fld.slots(key), other.slots(key)
            if len(ours) != len(theirs):
                return False
            for mine, yours in zip(ours, theirs, strict=True):
                a, b = _text(mine), _text(yours)
                if a == b:
                    continue
                if a is None or b is None or not _one_axis_apart(a, b):
                    return False
                differs = True
    return differs


def endpoints(views: Iterable[FamilyView], ids: Sequence[Sequence[str]]) -> dict[str, list[str]]:
    """The devices binding each address, in first-bound order.

    An address with one device belongs to it (``on``); an address with
    several is a shared endpoint of each (``endpoint_of``).

    Args:
        views: The imported families, as handed to :func:`device_ids`.
        ids: What :func:`device_ids` returned for them.

    Returns:
        Each address bound at a slot, mapped to its distinct devices.
    """
    owners: dict[str, list[str]] = {}
    for view, slots_ids in zip(views, ids, strict=True):
        for fld in view.fields.values():
            for key in fld.keys:
                for index, slot in enumerate(fld.slots(key)[: view.n_devices]):
                    address = _text(slot)
                    if address is None:
                        continue
                    devices = owners.setdefault(address, [])
                    if slots_ids[index] not in devices:
                        devices.append(slots_ids[index])
    return owners


def _lineage(name: str, parents: Mapping[str, str | None]) -> list[str]:
    chain: list[str] = []
    current: str | None = name
    while current is not None and current not in chain:
        chain.append(current)
        current = parents.get(current)
    return chain


def common_class(
    first: str | None, second: str | None, branches: Mapping[str, str] | None = None
) -> str | None:
    """The nearest class both of one device's family classes descend from.

    Args:
        first: One family's class.
        second: Another family's class for the same device.
        branches: Facility-declared classes, each mapped to its parent.

    Returns:
        The class itself when both agree, their nearest common ancestor in
        the vocabulary (extended by ``branches``), else ``None``. ``None`` on
        either side yields ``None``, so folding a device's family classes in
        any order gives one answer.
    """
    if first is None or second is None:
        return None
    if first == second:
        return first
    from osprey.facility.layers.mml.mapping import _vocabulary

    parents: dict[str, str | None] = {**_vocabulary(), **(branches or {})}
    ancestors = _lineage(second, parents)
    return next((name for name in _lineage(first, parents) if name in ancestors), None)
