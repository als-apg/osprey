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
states ``writable`` and ``confirm`` explicitly. ``writable`` is true only for a
setpoint whose record carries both ``min_value`` and ``max_value`` and does not
say ``writable: false``; ``confirm`` is the record's own value, else true.
A bound the record does not state is not written. A facility with no
``limits.yaml`` gets the version line alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility.views import ViewInputs

__all__ = ["LIMITS_FILE", "LIMITS_VERSION", "limits_document", "write_limits_view"]

LIMITS_FILE = "channel_limits.json"
LIMITS_VERSION = "4.0"

_BOUNDS = ("min_value", "max_value", "max_step")


def _entry(record: Mapping[str, Any], role: str) -> dict[str, Any]:
    entry: dict[str, Any] = {slot: record[slot] for slot in _BOUNDS if record.get(slot) is not None}
    bounded = "min_value" in entry and "max_value" in entry
    writable = record.get("writable")
    entry["writable"] = writable is not False and role == "setpoint" and bounded
    confirm = record.get("confirm")
    entry["confirm"] = confirm if isinstance(confirm, bool) else True
    return entry


def limits_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The limits database of one facility file.

    Args:
        doc: The facility file.

    Returns:
        ``_version`` first, then one entry per limits record, sorted by address.
    """
    roles = {str(channel["id"]): channel.get("role") for channel in doc.get("channels", [])}
    records = (doc.get("limits") or {}).get("records") or []
    document: dict[str, Any] = {"_version": LIMITS_VERSION}
    for record in sorted(records, key=lambda record: str(record["address"])):
        address = str(record["address"])
        document[address] = _entry(record, str(roles.get(address)))
    return document


def write_limits_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the limits view into ``root``.

    Args:
        root: The render's ``data/`` directory.
        inputs: The render's view inputs.

    Returns:
        The file written.
    """
    import json

    root.mkdir(parents=True, exist_ok=True)
    target = root / LIMITS_FILE
    text = json.dumps(limits_document(inputs.doc), indent=2, ensure_ascii=False, allow_nan=False)
    target.write_bytes((text + "\n").encode("utf-8"))
    return [target]
