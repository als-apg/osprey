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

from osprey.facility.records import FacilityRecords
from osprey.facility.validate import record_writable
from osprey.facility.views import ViewInputs
from osprey_connectors.simulation.values import DEFAULT_VALUE_TYPE

__all__ = [
    "LIMITS_FILE",
    "LIMITS_VERSION",
    "limits_document",
    "write_limits_view",
]

LIMITS_FILE = "channel_limits.json"
LIMITS_VERSION = "4.0"

_BOUNDS = ("min_value", "max_value", "max_step")


def _entry(record: Mapping[str, Any], role: str, value_type: str) -> dict[str, Any]:
    entry: dict[str, Any] = {slot: record[slot] for slot in _BOUNDS if record.get(slot) is not None}
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
    rows = (records.document.get("limits") or {}).get("records") or []
    document: dict[str, Any] = {"_version": LIMITS_VERSION}
    for row in sorted(rows, key=lambda row: str(row["address"])):
        address = str(row["address"])
        channel = records.record("channel", address) or {}
        value_type = str(channel.get("value_type") or DEFAULT_VALUE_TYPE)
        document[address] = _entry(row, str(channel.get("role")), value_type)
    return document


def write_limits_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the limits view into ``root``.

    Args:
        root: The render's ``data/`` directory.
        inputs: The render's view inputs.

    Returns:
        The file written.
    """
    from osprey.facility.views import json_bytes

    root.mkdir(parents=True, exist_ok=True)
    target = root / LIMITS_FILE
    target.write_bytes(json_bytes(limits_document(inputs.records), sort_keys=False))
    return [target]
