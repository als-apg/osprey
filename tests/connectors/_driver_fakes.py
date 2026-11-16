"""Fake DOOCS and Tango drivers, and the seam their connectors are driven through.

Both connectors import their driver inside ``connect()`` and read a limits
file. :func:`driven_through` serves a fake driver under the driver's import
name, with no limits file.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

#: The drivers a fake can stand in for: the DOOCS and TANGO clients.
_DRIVERS = frozenset({"doocs4py", "tango"})


#: The limits validator's loader, which every connector's ``connect()`` calls.
_FROM_CONFIG = "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config"


@contextmanager
def driven_through(driver: str, fake: Any) -> Iterator[None]:
    """Serve *fake* as the *driver* module (``doocs4py`` or ``tango``) to its connector."""
    if driver not in _DRIVERS:
        raise KeyError(driver)
    with (
        patch.dict(sys.modules, {driver: fake}),
        patch(_FROM_CONFIG, return_value=None),
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
