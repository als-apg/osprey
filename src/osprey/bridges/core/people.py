"""Who is speaking and who is in the room, as the engine ships them to the agent.

The payload these shapes ride in is documented at
``docs/source/reference/contracts/bridge-dispatch.rst``. This module imports only the
standard library and :mod:`osprey.bridges.core.ports`, which keeps the core's
isolation rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["asker_of"]


def asker_of(entry: Mapping[str, Any]) -> dict[str, str | None] | None:
    """Return who asked the entry's question, as ``{"id", "name"}``, or ``None``.

    The one reading of "who asked", so the dispatch payload and the history turn can
    never disagree. Reads the persisted ``sender_id`` and ``sender_display``, keeping
    each only when it is a non-empty ``str``; with neither, the asker is unknown and
    this returns ``None``. Never raises.
    """
    sid = entry.get("sender_id")
    name = entry.get("sender_display")
    sid = sid if isinstance(sid, str) and sid else None
    name = name if isinstance(name, str) and name else None
    if sid is None and name is None:
        return None
    return {"id": sid, "name": name}
