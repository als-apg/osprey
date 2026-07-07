"""Shared device-connection helper for the ophyd-async device factories.

Both factories (``mock.py``, ``epics.py``) build a ``{name: device}`` mapping
whose entries exist only as dict values, then connect them. That connect step
is centralized here so its non-obvious rationale lives in exactly one place.

Imports ophyd-async, so (like the rest of ``devices/``) this module lives
behind the optional ``osprey-framework[bluesky-bridge]`` extra — keep it out of
the bridge lifecycle core's import path (``app.py``, ``runs.py``,
``scanner.py``, ``security.py``).
"""

from __future__ import annotations

from typing import Any

from ophyd_async.core import wait_for_connection


async def connect_all(devices: dict[str, Any]) -> dict[str, Any]:
    """Connect every device in ``devices`` concurrently, returning the same mapping.

    Connects each device explicitly via ``Device.connect()`` rather than
    wrapping construction in ``ophyd_async.core.init_devices()``: that context
    manager finds "new" devices by diffing the *caller's local variables* on
    enter/exit, so a device that only ever exists as a `dict` entry (as every
    device built from a variable-length ``names``/specs sequence necessarily
    does) is invisible to it and would silently never get connected — every
    signal would stay unconnected, failing on first read/write rather than at
    connect time.

    Raises:
        ophyd_async.core.NotConnectedError: If any device fails to connect
            (e.g. an EPICS IOC is unreachable).
    """
    await wait_for_connection(**{name: device.connect() for name, device in devices.items()})
    return devices
