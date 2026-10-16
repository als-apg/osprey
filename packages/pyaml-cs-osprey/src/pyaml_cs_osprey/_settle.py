"""How close a setpoint's readback must come, as the render's facility file states it.

A setpoint record may name its readback (``pair``) and state a ``tolerance``;
:func:`settle_bound` turns the two into the readback address and the bound in
the setpoint's unit, by the rule :func:`osprey.facility.validate.tolerance_bound`
validates them with. The facility file is parsed once per path and modification
time.

An absent facility file states nothing. A facility file that is present but
cannot be read or parsed raises: a tolerance it states is never dropped
silently.

This module is stdlib-only at import time.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Held across every read and change of :data:`_PARSED`.
_LOCK = threading.Lock()

#: The last facility file parsed: its path, its ``st_mtime_ns`` and its
#: channel records by ``id``.
_PARSED: tuple[Path, int, dict[str, Mapping[str, Any]]] | None = None


def settle_bound(address: str) -> tuple[str, float] | None:
    """The readback address and tolerance the facility states for ``address``.

    Args:
        address: A setpoint address.

    Returns:
        ``(readback, bound)``: the channel's ``pair`` (``address`` itself when it
        names none) and its tolerance in its unit. ``None`` when the render has
        no facility file, ``address`` is not one of its channels, or the channel
        states no tolerance.

    Raises:
        OSError: The facility file exists but cannot be read.
        ValueError: The facility file is not JSON holding a ``channels`` list.
    """
    from osprey.channel_roster.sources import facility_file_path
    from osprey_connectors.config import get_full_configuration

    try:
        config = get_full_configuration()
    except FileNotFoundError:
        config = {}
    path = facility_file_path(config)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    channel = _channels(path, mtime_ns).get(address)
    if channel is None:
        return None

    from osprey.facility.validate import tolerance_bound

    bound = tolerance_bound(channel)
    if bound is None:
        return None
    return str(channel.get("pair") or address), bound


def _channels(path: Path, mtime_ns: int) -> dict[str, Mapping[str, Any]]:
    """The facility file's channel records by ``id``, parsed once per modification time."""
    global _PARSED
    with _LOCK:
        parsed = _PARSED
    if parsed is not None and parsed[0] == path and parsed[1] == mtime_ns:
        return parsed[2]
    document = json.loads(path.read_text(encoding="utf-8"))
    channels = document.get("channels") if isinstance(document, Mapping) else None
    if not isinstance(channels, list):
        raise ValueError(f"facility file {path} holds no 'channels' list")
    by_id = {
        str(channel["id"]): channel
        for channel in channels
        if isinstance(channel, Mapping) and "id" in channel
    }
    with _LOCK:
        _PARSED = (path, mtime_ns, by_id)
    return by_id
