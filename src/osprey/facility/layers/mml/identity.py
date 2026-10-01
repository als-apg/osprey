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
* **Fallback.** A slot without a usable name is named by the device segment
  of its first address that has one (an address with a ``{...}`` group: the
  text through the group's closing brace, less a leading segment ending in a
  ``:`` before the brace; else the segment after the first ``:``; else after
  the first run of three or more ``_``), else ``<family>_<i>``. An id already
  taken gains the first free ``_<n>``, ``n`` counting up from ``i``, the
  slot's 0-based position, so no fallback id names two slots.
* **Shared endpoints.** An address bound by several devices is one channel
  that names each of them (:func:`endpoints`).

No token, segment list or skipped family is a constant here: systems come
from the export, skipped families from the mapping.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the export services stay out of the import graph
    from osprey.services.mml.family import FamilyView

__all__ = ["common_class", "device_ids", "endpoints"]

#: Separators a name's leading system token may carry.
_PREFIX_SEPARATORS: tuple[str, ...] = ("_", " ")

#: A run of three or more ``_``: the segment separator of an underscore address.
_UNDERSCORE_SEPARATOR = re.compile(r"_{3,}")

#: Every run of characters other than letters and digits.
_NON_WORD = re.compile(r"[^0-9A-Za-z]+")


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


def device_ids(views: Sequence[FamilyView], models: Mapping[str, str]) -> list[list[str]]:
    """The device id of every slot of every family.

    Args:
        views: The imported families, in import order; each view's ``system``
            is a key of ``models``.
        models: The model each system is imported as, keyed by raw system
            token, in import order.

    Returns:
        One list per view, holding the id of each of its ``n_devices`` slots.
    """
    systems = list(models)
    taken: set[str] = set()
    result: list[list[str]] = []
    for view in views:
        by_system, rest = _roster(view, systems)
        seen = dict.fromkeys(by_system, 0)
        unsectioned = 0
        ids: list[str] = []
        for index in range(view.n_devices):
            addresses = _slot_addresses(view, index)
            system = _system_of_slot(addresses, systems, view.system)
            name: str | None = None
            stated = by_system.get(system, [])
            if seen.get(system, 0) < len(stated):
                name = stated[seen[system]]
                seen[system] += 1
            elif unsectioned < len(rest):
                name = rest[unsectioned]
                unsectioned += 1
            local = _local_name(name, system) if name is not None else ""
            if local:
                device = f"{models[system]}/{local}"
            else:
                token = next((t for t in map(_address_token, addresses) if t), "")
                base = f"{models[system]}/{token or f'{_word(view.raw_name)}_{index}'}"
                device, suffix = base, index
                while device in taken:
                    device = f"{base}_{suffix}"
                    suffix += 1
            taken.add(device)
            ids.append(device)
        result.append(ids)
    return result


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
