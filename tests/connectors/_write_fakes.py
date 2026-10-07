"""Shared fakes for the write-path and write-posture tests.

One recording connector, one config-reader shape and one mocked
``EPICSConnector`` builder, so the files that pin the write guard do not each
carry their own byte-equivalent copy.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

from osprey.connectors.control_system.base import (
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey_connectors.types import WRITES_ENABLED_KEY


def writes_enabled_config(key: str, default: Any = None) -> Any:
    """``get_config_value`` stand-in: writes armed deployment-wide, all else defaulted."""
    if key == WRITES_ENABLED_KEY:
        return True
    return default


def config_reader(section: dict[str, Any]) -> Callable[..., Any]:
    """A ``get_config_value`` stand-in serving one ``control_system:`` section.

    Answers both paths the posture is read through, the way dot-path lookup
    would: the whole section, which a type-stamped connector keys its block on,
    and the deployment-wide key inside it, which an unstamped connector reads.
    """

    def _get(key: str, default: Any = None) -> Any:
        if key == "control_system":
            return section
        if key == WRITES_ENABLED_KEY:
            return section.get("writes_enabled", default)
        return default

    return _get


class RecordingConnector(ControlSystemConnector):
    """A connector that records the writes the base-class guard let through.

    The abstract signature verbatim (``confirm`` included): a stand-in whose
    signature has drifted from the base class is wrapped by the same guard but
    exercises a call shape no real connector has. Every write it receives
    comes back ``CONFIRMED``.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, Any]] = []

    async def connect(self, config: dict[str, Any]) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        raise NotImplementedError

    async def read_multiple_channels(self, channel_addresses, timeout=None):
        raise NotImplementedError

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,  # noqa: ARG002 - the control-system connector interface fixes this signature
        confirm: bool | None = None,  # noqa: ARG002 - the control-system connector interface fixes this signature
    ) -> ChannelWriteResult:
        self.writes.append((channel_address, value))
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED,
        )

    async def write_multiple_channels(self, operations, timeout=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return [await self.write_channel(addr, val) for addr, val in operations]

    async def subscribe(self, channel_address, callback):
        raise NotImplementedError

    async def unsubscribe(self, channel_address):
        raise NotImplementedError

    async def get_metadata(self, channel_address):
        raise NotImplementedError

    async def validate_channel(self, channel_address) -> bool:  # noqa: ARG002 - the control-system connector interface fixes this signature
        return True


def make_mock_epics_connector(
    *,
    validate_side_effect: Any = None,
    caput_side_effect: Any = None,
    caput_return: Any = True,
    ca_severity_exception: type[BaseException] | None = None,
) -> EPICSConnector:
    """An ``EPICSConnector`` wired with a mock epics module and limits validator.

    Bypasses ``connect()`` (which imports pyepics) by setting the attributes
    the write path depends on directly. ``ca_severity_exception``, when given,
    is attached at ``ca.CASeverityException`` on the mock module, which is
    where the connector resolves the access-denied class from.
    """
    connector = EPICSConnector()
    connector._epics = MagicMock()
    connector._epics.caput = MagicMock(side_effect=caput_side_effect, return_value=caput_return)
    if ca_severity_exception is not None:
        connector._epics.ca.CASeverityException = ca_severity_exception
    connector._limits_validator = MagicMock()
    connector._limits_validator.validate = MagicMock(side_effect=validate_side_effect)
    connector._timeout = 5.0
    connector._connected = True
    return connector
