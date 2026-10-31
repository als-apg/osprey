"""Every connector answers a batch read with a partial failure the same way.

``read_multiple_channels`` is how a caller reads several channels in one call,
and its failure contract is the thing a caller builds on: a channel whose read
*raised* is left out of the result, and every other channel comes back with its
value. ``osprey.runtime.read_channels`` turns that shape into one error naming
exactly the channels that are missing — so the shape has to be the same on
every connector, or the runtime would name the wrong channels on some of them.

Three addresses are requested and the middle one fails. Each connector reaches
that failure through its own seam, keyed by address so the other two reads go
through untouched:

- **Mock** — ``_reading``, the store lookup behind ``read_channel``.
- **EPICS** — an injected fake ``aioca`` whose channel read fails for one name.
- **DOOCS** — a fake ``doocs4py`` module whose ``get`` raises for one address.
- **TANGO** — a fake ``tango`` module whose ``DeviceProxy.read_attribute``
  raises for one address.

A connector may also answer a channel *present* with no value; ``read_channels``
treats a ``None`` value as a failed read too, which the runtime tests below pin
on a stub connector.

The write-confirmation parity lives in ``test_cross_connector_parity.py``; its
fakes answer every address with one object, which cannot express a batch where
one channel fails and the others do not.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aioca
import pytest

from osprey.runtime import read_channels
from osprey_connectors.control_system._base import ChannelValue
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.errors import ChannelReadFailedError
from osprey_connectors_tests._epics_fakes import ECA_TIMEOUT, FakeChannel, ca_connector, ca_reading
from tests.connectors._driver_fakes import device_attribute, driven_through, eq_data
from tests.facility.served_tree import in_process_config, served_tree

# The value each of the three requested channels holds; the middle one is the
# channel that fails.
VALUES = (1.5, 2.5, 3.5)
FAILING = 1

READ_ERROR = "read exploded"


# ---------------------------------------------------------------------------
# The expectation table — one canonical result per row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadScenario:
    """One way the middle read of a three-channel batch can fail.

    ``absent`` are the positions the connector leaves out of its result, the
    ones ``read_channels`` names. Every other position comes back with the
    value its channel holds.
    """

    name: str
    absent: tuple[int, ...]


MIDDLE_READ_RAISES = ReadScenario(name="middle_read_raises", absent=(FAILING,))


def _raise_or_value(addresses, scenario, address):
    """The per-address answer every fake shares: a value, or a raise."""
    position = addresses.index(address)
    if position in scenario.absent:
        raise RuntimeError(READ_ERROR)
    return VALUES[position]


# ---------------------------------------------------------------------------
# Mock — seam: _reading
# ---------------------------------------------------------------------------

MOCK_ADDRESSES = ["SIM:A:RB", "SIM:B:RB", "SIM:C:RB"]


async def _in_process_connector(
    scenario: ReadScenario, monkeypatch, tmp_path
) -> VAInProcessConnector:
    """A mock serving ``MOCK_ADDRESSES`` with ``VALUES``, whose middle read raises."""
    view = served_tree(tmp_path, readings=MOCK_ADDRESSES)
    connector = VAInProcessConnector()
    await connector.connect(in_process_config(view, response_delay_ms=0))

    def reading(address, *, held):  # noqa: ARG001 - VAInProcessConnector._reading signature
        value = _raise_or_value(MOCK_ADDRESSES, scenario, address)
        return ChannelValue(value=value, timestamp=datetime.now(UTC))

    monkeypatch.setattr(connector, "_reading", reading)
    return connector


async def _read_mock(scenario: ReadScenario, monkeypatch, tmp_path) -> tuple[list[str], dict]:
    connector = await _in_process_connector(scenario, monkeypatch, tmp_path)
    result = await connector.read_multiple_channels(MOCK_ADDRESSES)
    await connector.disconnect()
    return MOCK_ADDRESSES, result


# ---------------------------------------------------------------------------
# EPICS — seam: an injected fake aioca, one channel per name
# ---------------------------------------------------------------------------

EPICS_ADDRESSES = ["SR:A:RB", "SR:B:RB", "SR:C:RB"]


def _epics_connector(scenario: ReadScenario) -> EPICSConnector:
    """A connected EPICS connector whose fake aioca answers per channel name.

    connect() is skipped by injecting the runtime state it would have built.
    A raising row's channel answers its get with the timeout aioca reports
    when the channel does not answer in time.
    """
    channels = {}
    for address in EPICS_ADDRESSES:
        try:
            reading = ca_reading(_raise_or_value(EPICS_ADDRESSES, scenario, address))
        except RuntimeError:
            reading = aioca.CANothing(address, ECA_TIMEOUT)
        channels[address] = FakeChannel(reading)
    return ca_connector(channels=channels)


async def _read_epics(scenario: ReadScenario, _monkeypatch, _tmp_path) -> tuple[list[str], dict]:
    connector = _epics_connector(scenario)
    return EPICS_ADDRESSES, await connector.read_multiple_channels(EPICS_ADDRESSES)


# ---------------------------------------------------------------------------
# DOOCS — seam: a fake doocs4py module whose get is keyed by address
# ---------------------------------------------------------------------------

DOOCS_ADDRESSES = ["FAC/DEV/LOC/A", "FAC/DEV/LOC/B", "FAC/DEV/LOC/C"]


async def _read_doocs(scenario: ReadScenario, _monkeypatch, _tmp_path) -> tuple[list[str], dict]:
    d = MagicMock()
    d.__version__ = "2.0.0"
    d.names.return_value = [("FACILITY", "XFEL")]
    d.get.side_effect = lambda address: eq_data(_raise_or_value(DOOCS_ADDRESSES, scenario, address))

    with driven_through("doocs4py", d):
        from osprey_connectors.control_system._doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect({})
        result = await conn.read_multiple_channels(DOOCS_ADDRESSES)
        await conn.disconnect()

    return DOOCS_ADDRESSES, result


# ---------------------------------------------------------------------------
# TANGO — seam: a fake tango module whose read_attribute is keyed by address
# ---------------------------------------------------------------------------

TANGO_ADDRESSES = ["sr/ps/01/Current", "sr/ps/02/Current", "sr/ps/03/Current"]


async def _read_tango(scenario: ReadScenario, _monkeypatch, _tmp_path) -> tuple[list[str], dict]:
    def proxy_for(device_name, *_args, **_kwargs):
        proxy = MagicMock()
        proxy.read_attribute.side_effect = lambda attribute: device_attribute(
            _raise_or_value(TANGO_ADDRESSES, scenario, f"{device_name}/{attribute}")
        )
        return proxy

    t = MagicMock()
    t.__version__ = "10.0.0"
    t.DeviceProxy.side_effect = proxy_for
    database = MagicMock()
    database.get_info.return_value = "TANGO Database sys/database/2"
    t.Database.return_value = database

    with driven_through("tango", t):
        from osprey_connectors.control_system._tango_connector import TangoConnector

        conn = TangoConnector()
        await conn.connect({})
        result = await conn.read_multiple_channels(TANGO_ADDRESSES)
        await conn.disconnect()

    return TANGO_ADDRESSES, result


_DRIVERS = {
    "in_process": _read_mock,
    "epics": _read_epics,
    "doocs": _read_doocs,
    "tango": _read_tango,
}

_ROWS = [(name, MIDDLE_READ_RAISES) for name in _DRIVERS]


# ---------------------------------------------------------------------------
# The parity matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("connector_name", "scenario"),
    _ROWS,
    ids=[f"{name}-{scenario.name}" for name, scenario in _ROWS],
)
class TestReadMultipleParity:
    """The same partial failure gets the same result shape from every connector."""

    async def test_the_failed_channel_is_absent_and_the_others_carry_their_values(
        self, connector_name, scenario, monkeypatch, tmp_path
    ):
        addresses, result = await _DRIVERS[connector_name](scenario, monkeypatch, tmp_path)

        for position, address in enumerate(addresses):
            if position in scenario.absent:
                assert address not in result
            else:
                assert result[address].value == VALUES[position]

        assert set(result) == {
            address for position, address in enumerate(addresses) if position not in scenario.absent
        }


# ---------------------------------------------------------------------------
# The runtime: what read_channels makes of that shape
# ---------------------------------------------------------------------------


class _StubConnector:
    """A connector stub whose batch read serves a canned per-address table.

    An address absent from ``table`` is left out of the result, exactly the
    shape every connector gives a channel whose read raised.
    """

    def __init__(self, table):
        self.table = table

    async def read_channel(self, channel_address, timeout=None):  # noqa: ARG002 - ControlSystemConnector.read_channel signature; the stub never blocks
        if channel_address not in self.table:
            raise RuntimeError(READ_ERROR)
        return ChannelValue(value=self.table[channel_address], timestamp=datetime.now(UTC))

    async def read_multiple_channels(self, channel_addresses, timeout=None):  # noqa: ARG002 - ControlSystemConnector.read_multiple_channels signature; the stub never blocks
        return {
            address: ChannelValue(value=self.table[address], timestamp=datetime.now(UTC))
            for address in channel_addresses
            if address in self.table
        }


def _runtime_over(connector):
    """Route ``osprey.runtime``'s connector lookup to ``connector``."""
    return patch("osprey.runtime._get_connector", new=AsyncMock(return_value=connector))


STUB_ADDRESSES = ["STUB:A", "STUB:B", "STUB:C"]


class TestRuntimeReadChannels:
    """``read_channels`` over the connector result shape the matrix pins."""

    def test_a_missing_channel_raises_naming_exactly_that_channel(self):
        table = {
            address: value
            for position, (address, value) in enumerate(zip(STUB_ADDRESSES, VALUES, strict=True))
            if position != FAILING
        }

        with _runtime_over(_StubConnector(table)), pytest.raises(ChannelReadFailedError) as exc:
            read_channels(STUB_ADDRESSES)

        assert exc.value.addresses == [STUB_ADDRESSES[FAILING]]
        cause = exc.value.causes[STUB_ADDRESSES[FAILING]]
        assert isinstance(cause, RuntimeError) and str(cause) == READ_ERROR
        assert exc.value.__cause__ is cause

    def test_a_full_result_comes_back_in_request_order(self):
        table = dict(zip(STUB_ADDRESSES, VALUES, strict=True))
        requested = list(reversed(STUB_ADDRESSES))

        with _runtime_over(_StubConnector(table)):
            values = read_channels(requested)

        assert values == [table[address] for address in requested]

    def test_a_present_none_valued_channel_is_named_as_failed(self):
        """A channel that comes back present with no value is a failed read, not a value."""
        table = dict(zip(STUB_ADDRESSES, VALUES, strict=True))
        table[STUB_ADDRESSES[FAILING]] = None

        with _runtime_over(_StubConnector(table)), pytest.raises(ChannelReadFailedError) as exc:
            read_channels(STUB_ADDRESSES)

        assert exc.value.addresses == [STUB_ADDRESSES[FAILING]]


# ---------------------------------------------------------------------------
# A cancelled per-channel read is raised, never a value
# ---------------------------------------------------------------------------


def _connector_classes():
    from osprey_connectors.control_system._doocs_connector import DOOCSConnector
    from osprey_connectors.control_system._tango_connector import TangoConnector

    return {
        "in_process": VAInProcessConnector,
        "epics": EPICSConnector,
        "doocs": DOOCSConnector,
        "tango": TangoConnector,
    }


@pytest.mark.parametrize("connector_name", list(_DRIVERS))
async def test_a_cancelled_channel_read_is_raised_from_the_batch(connector_name):
    """A cancelled read never comes back as a reading.

    A channel whose read raised an ordinary exception is left out of the
    batch; a read that was cancelled cancels the batch, so the cancellation
    reaches the caller instead of a result.
    """
    cls = _connector_classes()[connector_name]
    connector = cls.__new__(cls)
    addresses = ["CX:A", "CX:B", "CX:C"]

    async def read_channel(address, timeout=None):  # noqa: ARG001 - read_channel signature
        position = addresses.index(address)
        if position == FAILING:
            raise asyncio.CancelledError
        return ChannelValue(value=VALUES[position], timestamp=datetime.now(UTC))

    connector.read_channel = read_channel

    with pytest.raises(asyncio.CancelledError):
        await cls.read_multiple_channels(connector, addresses)
