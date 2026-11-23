"""The limits view: the records of ``limits.yaml`` as the limits database.

Written to ``<render>/data/channel_limits.json``::

    {
      "_version": "4.0",
      "<address>": {"min_value": ..., "max_value": ..., "max_step": ...,
                    "writable": true|false, "confirm": true|false},
      ...
    }

The file holds one entry per limits record and nothing else: no ``defaults``
block, and no entry for a channel without a record, which
``control_system.limits_checking.mode`` decides at write time. Each entry
states ``writable`` and ``confirm`` explicitly. Only a setpoint is writable,
and never when its record says ``writable: false``. A ``float`` or ``int``
setpoint is writable when its record carries both ``min_value`` and
``max_value``. A setpoint of any other ``value_type`` is writable when its
record says ``writable: true``. ``confirm`` is the record's own value, else
true.
A bound the record does not state is not written. A facility with no
``limits.yaml`` gets the version line alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility.views import ViewInputs
from osprey_connectors.simulation.values import DEFAULT_VALUE_TYPE
from osprey_facility.records import FacilityRecords
from osprey_facility.validate import LIMIT_BOUNDS, record_missing, record_writable

__all__ = [
    "LIMITS_FILE",
    "LIMITS_VERSION",
    "limits_document",
    "limits_findings",
    "write_limits_view",
]

LIMITS_FILE = "channel_limits.json"
LIMITS_VERSION = "4.0"


def _entry(record: Mapping[str, Any], role: str, value_type: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        slot: record[slot] for slot in LIMIT_BOUNDS if record.get(slot) is not None
    }
    entry["writable"] = record_writable(record, role, value_type)
    confirm = record.get("confirm")
    entry["confirm"] = confirm if isinstance(confirm, bool) else True
    return entry


def limits_document(records: FacilityRecords) -> dict[str, Any]:
    """The limits database of one facility file.

    Args:
        records: The facility file's records.

    Returns:
        ``_version`` first, then one entry per limits record, sorted by address.
    """
    document: dict[str, Any] = {"_version": LIMITS_VERSION}
    for address, row, role, value_type in _rows(records):
        document[address] = _entry(row, role, value_type)
    return document


def _rows(records: FacilityRecords) -> list[tuple[str, Mapping[str, Any], str, str]]:
    """Each limits record with its channel's role and value type, sorted by address."""
    rows = (records.document.get("limits") or {}).get("records") or []
    resolved = []
    for row in sorted(rows, key=lambda row: str(row["address"])):
        address = str(row["address"])
        channel = records.record("channel", address) or {}
        value_type = str(channel.get("value_type") or DEFAULT_VALUE_TYPE)
        resolved.append((address, row, str(channel.get("role")), value_type))
    return resolved


#: How many addresses a finding lists unless every one is asked for.
FINDING_ADDRESSES = 10


def limits_findings(
    records: FacilityRecords, unlisted_writable: bool, *, every: bool = False
) -> list[str]:
    """What a reader of the limits should know, as report lines; it changes nothing.

    Two findings, each a count line followed by one line per address, sorted,
    the first :data:`FINDING_ADDRESSES` of them and a count of the rest unless
    *every*:
    the setpoints with no limits record, which the target serves writable with
    no range when *unlisted_writable* (its limits mode is ``optional``) and
    refuses otherwise; and the limits records that serve their setpoint not
    writable for want of a slot
    (:func:`~osprey_facility.validate.record_missing`), each line naming the
    slots its record states and the ones it lacks. A finding with no address
    is left out.

    Args:
        records: The facility file's records.
        unlisted_writable: Whether the target writes a setpoint with no record.
        every: List every address, not the first few.

    Returns:
        The lines, the open setpoints' first.
    """
    document = records.document
    rows = {str(row["address"]): row for row in (document.get("limits") or {}).get("records") or []}
    open_setpoints = sorted(
        str(channel["id"])
        for channel in document.get("channels") or []
        if channel.get("role") == "setpoint" and str(channel["id"]) not in rows
    )
    short = [
        (address, row, missing)
        for address, row, role, value_type in _rows(records)
        if (missing := record_missing(row, role, value_type))
    ]
    lines: list[str] = []
    if open_setpoints:
        served = (
            "the target serves them writable with no range"
            if unlisted_writable
            else "the target refuses writes to them"
        )
        lines.append(f"limits: {len(open_setpoints)} setpoints have no limits record; {served}")
        lines.extend(_listed([f"  {address}" for address in open_setpoints], every))
    if short:
        lines.append(
            f"limits: {len(short)} limits records serve their setpoint not writable; "
            "stating what each line lacks makes it writable"
        )
        lines.extend(
            _listed([_lacking(address, row, missing) for address, row, missing in short], every)
        )
    return lines


def _lacking(address: str, row: Mapping[str, Any], missing: tuple[str, ...]) -> str:
    stated = [f"{slot} {row[slot]}" for slot in LIMIT_BOUNDS if row.get(slot) is not None]
    lacks = ["no writable: true" if slot == "writable" else f"no {slot}" for slot in missing]
    return f"  {address}: " + ", ".join([*stated, *lacks])


def _listed(lines: list[str], every: bool) -> list[str]:
    if every or len(lines) <= FINDING_ADDRESSES:
        return lines
    more = len(lines) - FINDING_ADDRESSES
    return [*lines[:FINDING_ADDRESSES], f"  and {more} more; --all-limits lists every one"]


def write_limits_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the limits view into ``root``.

    Args:
        root: The render's ``data/`` directory.
        inputs: The render's view inputs.

    Returns:
        The file written.
    """
    from osprey_facility.render import json_bytes

    root.mkdir(parents=True, exist_ok=True)
    target = root / LIMITS_FILE
    target.write_bytes(json_bytes(limits_document(inputs.records), sort_keys=False))
    return [target]
