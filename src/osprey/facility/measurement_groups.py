"""What the facility file says a model's measurement groups and tune references are.

``measurement_groups`` is the addresses each group role of a model's
measurement file stands for, taken over the wiring the model states;
``tune_references`` the horizontal and vertical references the ``tune``
instrument stands for. The build validates a model's measurement file against
these; the pyAML view writes the same arrays.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from osprey.facility.kinds import SHIPPED_SORTS
from osprey.facility.records import FacilityRecords
from osprey.facility.sources import group_families

__all__ = [
    "GROUP_ROLES",
    "ROLE_PLANES",
    "measurement_groups",
    "tune_references",
    "wiring",
]

#: The group roles the view writes an array for, in the order it writes them.
GROUP_ROLES: tuple[str, ...] = tuple(
    role for role, sort in SHIPPED_SORTS.items() if sort != "instrument"
)

#: The plane each corrector role steers; a role not named here takes every plane.
ROLE_PLANES: dict[str, str] = {"hcor": "x", "vcor": "y"}

#: A device whose record states no ``s`` sorts after every placed device.
_UNPLACED = math.inf


def _device_order(records: FacilityRecords, members: Iterable[str]) -> list[str]:
    """``members`` in ``(s, id)`` order; a device with no ``s`` after every placed one.

    Args:
        records: The facility file's records.
        members: Device ids, possibly repeated.

    Returns:
        Each device once, ordered by its ``s`` and then its id.
    """

    def key(device: str) -> tuple[float, str]:
        s = (records.record("device", device) or {}).get("s")
        return (float(s) if isinstance(s, int | float) else _UNPLACED, device)

    return sorted(dict.fromkeys(str(member) for member in members), key=key)


def wiring(records: FacilityRecords, model: str) -> dict[str, Mapping[str, Any]]:
    """The wiring records a model states, by address.

    Args:
        records: The facility file's records.
        model: The model's name.

    Returns:
        Address to the model's wiring entry for it.

    Raises:
        KeyError: The facility file has no model ``model``.
    """
    entries = records.require("model", model).get("wiring", ())
    return {str(entry["address"]): entry for entry in entries}


def measurement_groups(
    records: FacilityRecords, model: str, sorts: Mapping[str, str] | None = None
) -> dict[str, list[str]]:
    """The addresses each group role of a model's measurement file stands for.

    ``bpm`` is the readbacks the model wires as monitors, in both planes;
    ``hcor`` the setpoints the model wires whose plane its engine describes as
    ``x``, ``vcor`` those it describes as ``y``; ``quad`` and ``sext`` every
    setpoint the model wires; a role a registered kind declares the monitors
    or the setpoints the model wires, in every plane, as the kind sorts it.
    Each is taken over the member devices of every
    group the role names: the channels on a member device and the channels
    naming one as an endpoint, visiting the
    devices in ``(s, id)`` order, an address listed once at its first device and
    the addresses within a device sorted. A combined corrector, one device
    steered in both planes, keeps each setpoint in the role of its own plane, so
    ``hcor`` and ``vcor`` share no address.

    Args:
        records: The facility file's records.
        model: The model's name.
        sorts: Each member role's sort (:func:`osprey.facility.kinds.member_sorts`);
            the shipped roles' when omitted, and a role of no known sort is
            left out.

    Returns:
        Role to addresses, for each group role the measurement file names: the
        shipped ones in :data:`GROUP_ROLES` order, then the others by name.

    Raises:
        KeyError: The facility file has no model ``model``.
    """
    record = records.require("model", model)
    groups_named = (record.get("measurement") or {}).get("groups") or {}
    known = SHIPPED_SORTS if sorts is None else sorts
    wired = wiring(records, model)
    found: dict[str, list[str]] = {}
    roles = [*GROUP_ROLES, *sorted(set(groups_named) - set(GROUP_ROLES))]
    for role in roles:
        if groups_named.get(role) is None or known.get(role) in (None, "instrument"):
            continue
        wanted = "monitor" if known[role] == "monitors" else "setpoint"
        plane = ROLE_PLANES.get(role)
        members = [
            member
            for group_id in group_families(groups_named.get(role))
            for member in (records.record("group", group_id) or {}).get("members") or []
        ]
        addresses: dict[str, None] = {}
        for device in _device_order(records, members):
            for address in sorted({str(c["id"]) for c in records.channels_on(device)}):
                entry = wired.get(address)
                if entry is None or entry.get("engine_role") != wanted:
                    continue
                if plane is not None and entry.get("plane") != plane:
                    continue
                addresses.setdefault(address, None)
        found[role] = list(addresses)
    return found


def tune_references(records: FacilityRecords, model: str, address: str) -> tuple[str, str] | None:
    """The horizontal and vertical tune references the ``tune`` instrument stands for.

    A waveform reading the whole tune output is read by element: ``@0`` and
    ``@1``. A scalar reading one plane is paired with the model's reading on the
    other plane whose engine block differs from its own in one key alone, the
    one that names the plane.

    Args:
        records: The facility file's records.
        model: The model's name.
        address: The address the ``tune`` instrument reads.

    Returns:
        The horizontal and vertical references, or None when the model does not
        wire ``address`` or no single partner reading pairs with it.

    Raises:
        KeyError: The facility file has no model ``model``.
    """
    entries = records.require("model", model).get("wiring", ())
    entry = next((entry for entry in entries if str(entry["address"]) == address), None)
    if entry is None:
        return None
    plane = entry.get("plane")
    if plane is None:
        return f"{address}@0", f"{address}@1"
    block = entry.get("engine") or {}
    other = {"x": "y", "y": "x"}.get(str(plane))
    partners = [
        str(candidate["address"])
        for candidate in entries
        if candidate.get("engine_role") == entry.get("engine_role")
        and candidate.get("plane") == other
        and _differ_in_one_key(candidate.get("engine") or {}, block)
    ]
    if len(partners) != 1:
        return None
    return (address, partners[0]) if plane == "x" else (partners[0], address)


def _differ_in_one_key(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """True when two engine blocks name the same keys and differ in the value of one.

    Args:
        first: One engine block.
        second: The other engine block.

    Returns:
        Whether the blocks share their keys and differ in exactly one value.
    """
    if set(first) != set(second):
        return False
    differing = [key for key in first if first[key] != second[key]]
    return len(differing) == 1
