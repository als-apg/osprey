"""PVA lifecycle completion: disconnect teardown, metadata and validation.

Tasks 1.1-1.4 built the PVA read, write-refusal and monitor paths. What is left
is everything that happens around them:

* ``disconnect()`` now tears down two transports, and the order is load-bearing
  — a p4p ``Subscription`` belongs to the ``Context`` that opened it, so the
  monitors close first and the context second, with Channel Access last and
  unchanged. Closing the context at all is what keeps connector invalidation
  (a production path: a ``ConnectionError`` invalidates and rebuilds the
  connector) from leaking p4p worker threads on every retry.
* ``get_metadata`` and ``validate_channel`` are the two entry points that want
  a channel's *description*, not its payload. Over PVA those are asked for with
  a field-limited pvRequest, so a camera channel costs a handful of fields
  instead of a full frame per call — and, as a direct consequence, a frame in a
  codec the connector cannot decode no longer makes a reachable channel report
  itself as invalid.

Conventions follow the sibling PVA files: a fake ``p4p`` module rather than the
real one (p4p is not installed on every dev machine), strict hand-written fakes
rather than ``MagicMock`` for handles whose teardown spelling differs per
transport, and assertions on the concrete payload — the request string, the
mapped units, the call order — never merely that a call "didn't raise".
"""

import types
from unittest.mock import MagicMock

import pytest

from osprey.connectors.control_system.epics_connector import (
    EPICSConnector,
    _ChannelSubscription,
)

PVA_GLOB = "SR:CAM*:IMAGE"
PVA_ADDRESS = "SR:CAM1:IMAGE"
CA_ADDRESS = "SR:BEAM:CURRENT"

METADATA_REQUEST = "field(alarm,timeStamp,display)"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDisconnected(RuntimeError):
    """Stands in for p4p's Disconnected (a RuntimeError, not a ConnectionError)."""


class FakeRemoteError(RuntimeError):
    """Stands in for p4p's RemoteError."""


def _fake_p4p_module() -> types.ModuleType:
    """A ``p4p`` module exposing exactly what the connector looks up on it."""
    thread_mod = types.ModuleType("p4p.client.thread")
    thread_mod.Context = MagicMock(name="Context")
    thread_mod.Disconnected = FakeDisconnected
    thread_mod.RemoteError = FakeRemoteError
    thread_mod.TimeoutError = TimeoutError
    client_mod = types.ModuleType("p4p.client")
    client_mod.thread = thread_mod
    p4p_mod = types.ModuleType("p4p")
    p4p_mod.client = client_mod
    return p4p_mod


class FakeValue:
    """Minimal p4p ``Value`` stand-in: a struct id plus (possibly nested) fields."""

    def __init__(self, type_id: str, fields: dict):
        self._type_id = type_id
        self._fields = {
            key: FakeValue("", value) if isinstance(value, dict) else value
            for key, value in fields.items()
        }

    def getID(self) -> str:  # noqa: N802 - p4p's spelling
        return self._type_id

    def get(self, name, default=None):
        return self._fields.get(name, default)

    def __contains__(self, name) -> bool:
        return name in self._fields


class FakeSubscription:
    """A p4p ``Subscription`` stand-in: closeable, and recording when it closed."""

    def __init__(self, log: list[str], label: str = "subscription") -> None:
        self._log = log
        self._label = label
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self._log.append(self._label)


class FakeContext:
    """A p4p thread ``Context`` stand-in covering ``get`` and ``close``.

    ``get`` replays a queued reply (a value, or an exception to raise) and
    records the pvRequest it was called with — the metadata paths are only
    honest about their cost if that request is field-limited.
    """

    def __init__(self, reply=None, log: list[str] | None = None) -> None:
        self.reply = reply
        self.gets: list[dict] = []
        self.close_calls = 0
        self._log = log if log is not None else []

    def get(self, name, request=None, timeout=None):
        self.gets.append({"name": name, "request": request, "timeout": timeout})
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply

    def close(self) -> None:
        self.close_calls += 1
        self._log.append("context")


def _metadata_value(type_id: str = "epics:nt/NTScalar:1.0") -> FakeValue:
    """A metadata-only reply: exactly the fields the field-limited get asks for."""
    return FakeValue(
        type_id,
        {
            "alarm": {"severity": 1, "status": 3, "message": "HIGH_ALARM"},
            "timeStamp": {"secondsPastEpoch": 1_700_000_000, "nanoseconds": 500_000_000},
            "display": {
                "units": "mA",
                "precision": 3,
                "description": "Storage ring current",
                "limitLow": 0.0,
                "limitHigh": 500.0,
            },
        },
    )


def _pva_connector(context=None, globs=(PVA_GLOB,), epics=None) -> EPICSConnector:
    """A connector wired for PVA without going through connect()."""
    connector = EPICSConnector()
    connector._pva_channel_globs = list(globs)
    connector._p4p = _fake_p4p_module()
    connector._pva_context = context
    connector._timeout = 3.0
    connector._epics = epics if epics is not None else MagicMock()
    connector._connected = True
    return connector


# ---------------------------------------------------------------------------
# disconnect(): two transports, one order
# ---------------------------------------------------------------------------


class TestDisconnectClosesPvaContext:
    @pytest.mark.asyncio
    async def test_context_is_closed_exactly_once_and_nulled(self):
        """The context p4p's worker threads hang off is released, and not reused."""
        context = FakeContext()
        connector = _pva_connector(context=context)

        await connector.disconnect()

        assert context.close_calls == 1
        assert connector._pva_context is None
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_second_disconnect_is_a_no_op(self):
        """Nulling the attribute is what makes a repeated teardown idempotent."""
        context = FakeContext()
        connector = _pva_connector(context=context)

        await connector.disconnect()
        await connector.disconnect()

        assert context.close_calls == 1

    @pytest.mark.asyncio
    async def test_pure_ca_connector_has_no_context_to_close(self):
        """A connector that never configured PVA globs tears down as before."""
        pv = MagicMock()
        connector = _pva_connector(context=None, globs=())
        connector._pv_cache = {"A": pv}

        await connector.disconnect()

        assert connector._pva_context is None
        pv.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscriptions_close_before_the_context(self):
        """A p4p Subscription belongs to its Context, so monitors go first."""
        order: list[str] = []
        context = FakeContext(log=order)
        connector = _pva_connector(context=context)
        connector._subscriptions = {
            "sub-a": _ChannelSubscription("pva", FakeSubscription(order, "sub-a")),
            "sub-b": _ChannelSubscription("pva", FakeSubscription(order, "sub-b")),
        }

        await connector.disconnect()

        assert order == ["sub-a", "sub-b", "context"]
        assert connector._subscriptions == {}

    @pytest.mark.asyncio
    async def test_channel_access_teardown_still_runs_after_the_context(self):
        """CA teardown is unchanged and downstream of the PVA close."""
        order: list[str] = []
        context = FakeContext(log=order)
        cached_ok = MagicMock()
        cached_ok.disconnect.side_effect = lambda: order.append("ca-pv")
        cached_bad = MagicMock()
        cached_bad.disconnect.side_effect = RuntimeError("already gone")
        ca_pv = MagicMock()
        ca_pv.clear_callbacks.side_effect = lambda: order.append("ca-subscription")

        connector = _pva_connector(context=context)
        connector._subscriptions = {"sub1": _ChannelSubscription("ca", ca_pv)}
        connector._pv_cache = {"A": cached_ok, "B": cached_bad}

        await connector.disconnect()

        assert order == ["ca-subscription", "context", "ca-pv"]
        assert connector._pv_cache == {}
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_failing_context_close_is_swallowed_and_ca_teardown_continues(self):
        """Teardown is best effort: one dead handle cannot strand the rest."""
        context = MagicMock()
        context.close.side_effect = RuntimeError("context already dead")
        cached = MagicMock()
        connector = _pva_connector(context=context)
        connector._pv_cache = {"A": cached}

        await connector.disconnect()

        assert connector._pva_context is None
        cached.disconnect.assert_called_once()
        assert connector._connected is False


# ---------------------------------------------------------------------------
# get_metadata()
# ---------------------------------------------------------------------------


class TestGetMetadataOverPva:
    @pytest.mark.asyncio
    async def test_display_and_alarm_fields_are_mapped(self):
        context = FakeContext(reply=_metadata_value())
        connector = _pva_connector(context=context)

        metadata = await connector.get_metadata(PVA_ADDRESS)

        assert metadata.units == "mA"
        assert metadata.precision == 3
        assert metadata.description == "Storage ring current"
        assert metadata.display_low == 0.0
        assert metadata.display_high == 500.0
        assert metadata.alarm_status == "HIGH_ALARM"
        assert metadata.raw_metadata["severity"] == 1
        assert metadata.raw_metadata["status"] == 3
        assert metadata.raw_metadata["nt_id"] == "epics:nt/NTScalar:1.0"

    @pytest.mark.asyncio
    async def test_timestamp_comes_from_the_nt_timestamp_field(self):
        context = FakeContext(reply=_metadata_value())
        connector = _pva_connector(context=context)

        metadata = await connector.get_metadata(PVA_ADDRESS)

        assert metadata.timestamp is not None
        assert metadata.timestamp.timestamp() == pytest.approx(1_700_000_000.5)

    @pytest.mark.asyncio
    async def test_the_get_is_field_limited_and_uses_the_connector_timeout(self):
        """Asking for the whole structure would pull a full frame per lookup."""
        context = FakeContext(reply=_metadata_value())
        connector = _pva_connector(context=context)

        await connector.get_metadata(PVA_ADDRESS)

        assert context.gets == [{"name": PVA_ADDRESS, "request": METADATA_REQUEST, "timeout": 3.0}]

    @pytest.mark.asyncio
    async def test_ndarray_metadata_never_touches_the_payload(self):
        """An NTNDArray reply carrying no value/dimension/codec still maps."""
        context = FakeContext(reply=_metadata_value("epics:nt/NTNDArray:1.0"))
        connector = _pva_connector(context=context)

        metadata = await connector.get_metadata(PVA_ADDRESS)

        assert metadata.units == "mA"
        assert metadata.raw_metadata["nt_id"] == "epics:nt/NTNDArray:1.0"

    @pytest.mark.asyncio
    async def test_missing_display_structure_degrades_to_empty_metadata(self):
        """Servers that publish no display block still produce a ChannelMetadata."""
        context = FakeContext(reply=FakeValue("epics:nt/NTScalar:1.0", {}))
        connector = _pva_connector(context=context)

        metadata = await connector.get_metadata(PVA_ADDRESS)

        assert metadata.units == ""
        assert metadata.precision is None
        assert metadata.alarm_status is None
        assert metadata.timestamp is not None  # falls back to now()

    @pytest.mark.asyncio
    async def test_ca_address_on_a_pva_capable_connector_uses_pyepics(self):
        pv = MagicMock()
        pv.connected = True
        pv.get.return_value = 42.0
        pv.timestamp = 1_700_000_000.0
        pv.units = "mA"
        pv.precision = 2
        epics = MagicMock()
        epics.PV.return_value = pv
        context = FakeContext(reply=_metadata_value())
        connector = _pva_connector(context=context, epics=epics)

        metadata = await connector.get_metadata(CA_ADDRESS)

        assert context.gets == []  # PVA client untouched
        assert epics.PV.call_args.args == (CA_ADDRESS,)
        assert metadata.units == "mA"
        assert metadata.precision == 2

    @pytest.mark.asyncio
    async def test_without_a_context_it_is_a_connection_error(self):
        connector = _pva_connector(context=None)

        with pytest.raises(ConnectionError, match="no PVA client context"):
            await connector.get_metadata(PVA_ADDRESS)

    @pytest.mark.asyncio
    async def test_timeout_surfaces_as_timeout_error(self):
        context = FakeContext(reply=TimeoutError())
        connector = _pva_connector(context=context)

        with pytest.raises(TimeoutError, match=f"Failed to read PVA channel '{PVA_ADDRESS}'"):
            await connector.get_metadata(PVA_ADDRESS)

    @pytest.mark.asyncio
    async def test_disconnected_surfaces_as_connection_error(self):
        context = FakeContext(reply=FakeDisconnected("channel gone"))
        connector = _pva_connector(context=context)

        with pytest.raises(ConnectionError, match="Failed to connect to PVA channel"):
            await connector.get_metadata(PVA_ADDRESS)


# ---------------------------------------------------------------------------
# validate_channel()
# ---------------------------------------------------------------------------


class TestValidateChannelOverPva:
    @pytest.mark.asyncio
    async def test_reachable_channel_validates_true_with_a_short_timeout(self):
        context = FakeContext(reply=_metadata_value())
        connector = _pva_connector(context=context)

        assert await connector.validate_channel(PVA_ADDRESS) is True
        assert context.gets == [{"name": PVA_ADDRESS, "request": METADATA_REQUEST, "timeout": 2.0}]

    @pytest.mark.asyncio
    async def test_timeout_validates_false(self):
        context = FakeContext(reply=TimeoutError())
        connector = _pva_connector(context=context)

        assert await connector.validate_channel(PVA_ADDRESS) is False

    @pytest.mark.asyncio
    async def test_disconnected_validates_false(self):
        context = FakeContext(reply=FakeDisconnected("no such channel"))
        connector = _pva_connector(context=context)

        assert await connector.validate_channel(PVA_ADDRESS) is False

    @pytest.mark.asyncio
    async def test_missing_context_validates_false(self):
        connector = _pva_connector(context=None)

        assert await connector.validate_channel(PVA_ADDRESS) is False

    @pytest.mark.asyncio
    async def test_undecodable_frame_does_not_make_a_reachable_channel_invalid(self):
        """Reachability is a property of the channel, not of its payload codec.

        A compressed NTNDArray makes the *read* path raise; validation asks only
        for the metadata fields, so the channel reports itself as reachable.
        """
        context = FakeContext(reply=_metadata_value("epics:nt/NTNDArray:1.0"))
        connector = _pva_connector(context=context)

        assert await connector.validate_channel(PVA_ADDRESS) is True

    @pytest.mark.asyncio
    async def test_ca_address_validates_over_channel_access(self):
        pv = MagicMock()
        pv.connected = True
        pv.get.return_value = 1.0
        pv.timestamp = 1_700_000_000.0
        epics = MagicMock()
        epics.PV.return_value = pv
        context = FakeContext(reply=_metadata_value())
        connector = _pva_connector(context=context, epics=epics)

        assert await connector.validate_channel(CA_ADDRESS) is True
        assert context.gets == []
        assert epics.PV.call_args.args == (CA_ADDRESS,)

    @pytest.mark.asyncio
    async def test_unreachable_ca_address_validates_false(self):
        pv = MagicMock()
        pv.connected = False
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _pva_connector(context=FakeContext(), epics=epics)

        assert await connector.validate_channel(CA_ADDRESS) is False
