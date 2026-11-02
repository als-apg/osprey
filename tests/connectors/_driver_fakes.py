"""Fake DOOCS and Tango drivers, and the seam their connectors are driven through.

Both connectors import their driver inside ``connect()`` and read a limits file
and the facility's timezone. :func:`driven_through` serves a fake driver under
the driver's import name, with no limits file and a UTC facility clock.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from typing import Any
from unittest.mock import MagicMock, patch

#: Each driver's import name, and the connector module that imports it.
_CONNECTOR_OF = {
    "doocs4py": "osprey_connectors.control_system._doocs_connector",
    "tango": "osprey_connectors.control_system._tango_connector",
}


@contextmanager
def driven_through(driver: str, fake: Any) -> Iterator[None]:
    """Serve *fake* as the *driver* module (``doocs4py`` or ``tango``) to its connector."""
    connector = _CONNECTOR_OF[driver]
    with (
        patch.dict(sys.modules, {driver: fake}),
        patch(f"{connector}.LimitsValidator.from_config", return_value=None),
        patch(f"{connector}.get_facility_timezone", return_value=UTC),
    ):
        yield


def eq_data(value: Any) -> MagicMock:
    """A mock EqData object as returned by ``doocs4py.get()``."""
    ts = MagicMock()
    ts.get_seconds_and_microseconds_since_epoch.return_value = (1_700_000_000, 500_000)

    eq = MagicMock()
    eq.get_data.return_value = value
    eq.macropulse = 12345
    eq.timestamp = ts
    return eq


def device_attribute(value: Any) -> MagicMock:
    """A mock DeviceAttribute as returned by ``DeviceProxy.read_attribute()``."""
    time_val = MagicMock()
    time_val.tv_sec = 1_700_000_000
    time_val.tv_usec = 500_000

    attr = MagicMock()
    attr.value = value
    attr.quality = None
    attr.time = time_val
    attr.type = "DevDouble"
    return attr
