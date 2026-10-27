"""A synthetic simulator view's channel records, and its scenario state."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def view_channel(address: str, owner: str = "texture", **fields: Any) -> dict[str, Any]:
    """One channel record, every field filled.

    A readback by default, owned by ``owner``; a setpoint pairs with itself
    unless ``pair`` says otherwise. Any other field in ``fields`` replaces its
    default or is added to the record.
    """
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": address if role == "setpoint" else None,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": False,
        "value_range": None,
        "owner": owner,
        "on": None,
        **fields,
    }


def activate(state: Path, *names: str) -> None:
    """Write ``names`` (``nominal`` when none) as the active set under ``state``.

    The file's mtime is moved a full second past its last one, so a reader
    that watches the mtime sees the change however soon it follows the last.
    """
    state.mkdir(parents=True, exist_ok=True)
    target = state / "active_scenarios"
    before = target.stat().st_mtime_ns if target.exists() else 0
    target.write_text("\n".join(names or ("nominal",)) + "\n", encoding="utf-8")
    os.utime(target, ns=(before + 1_000_000_000, before + 1_000_000_000))
