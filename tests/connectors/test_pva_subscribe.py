"""PVA subscriptions: monitor lifecycle, callback marshaling and sentinel filtering.

Task 1.1 decided WHICH transport an address uses and task 1.2 mapped a p4p
``Value`` onto :class:`ChannelValue`; this file covers the third path into that
mapping — a p4p monitor, which differs from a read in three ways that matter:

* its updates arrive on p4p's worker thread, so the subscriber callback has to
  be handed to the event loop rather than called where the update landed;
* p4p delivers connection state (``Disconnected`` and friends) to the same
  callback, *as a value*, so anything that is an exception instance has to be
  filtered before it can be mistaken for channel data;
* raising inside a monitor callback makes p4p close the subscription, so a
  value the mapping cannot decode (a compressed NTNDArray, say) must be dropped
  rather than propagated.

Teardown is asserted on both transports together: a p4p ``Subscription`` has no
``clear_callbacks`` and a pyepics ``PV`` has no ``close``, so the connector
stores every subscription behind one ``_ChannelSubscription`` wrapper and
dispatches there. The fakes below are deliberately strict — no ``MagicMock``
for the handles — so calling the wrong teardown method raises instead of
silently recording a call that would never work against a real library.
"""

import asyncio
import threading
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


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDisconnected(RuntimeError):
    """Stands in for p4p's Disconnected — delivered to the callback as a value."""


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


class FakeSubscription:
    """A p4p ``Subscription`` stand-in: it can be closed, and nothing else.

    Deliberately has NO ``clear_callbacks`` — that is the pyepics spelling, and
    calling it here must fail loudly rather than pass a mock assertion.
    """

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class FakeMonitorContext:
    """A p4p thread ``Context`` whose ``monitor`` records and hands back a handle."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.subscriptions: list[FakeSubscription] = []

    def monitor(self, name, cb, **kwargs):
        self.calls.append({"name": name, "cb": cb, **kwargs})
        subscription = FakeSubscription()
        self.subscriptions.append(subscription)
        return subscription

    def fire(self, update, index: int = 0) -> None:
        """Deliver one update the way p4p's worker thread would."""
        self.calls[index]["cb"](update)


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


def _scalar_update(value=1.5) -> FakeValue:
    """An NTScalar update carrying a value, units and a timestamp."""
    return FakeValue(
        "epics:nt/NTScalar:1.0",
        {
            "value": value,
            "timeStamp": {"secondsPastEpoch": 1_700_000_000, "nanoseconds": 0},
            "display": {"units": "mA"},
        },
    )


def _compressed_frame() -> FakeValue:
    """A codec-tagged NTNDArray — the mapping refuses these with a ValueError."""
    return FakeValue(
        "epics:nt/NTNDArray:1.0",
        {
            "codec": {"name": "jpeg"},
            "dimension": [FakeValue("", {"size": 64}), FakeValue("", {"size": 48})],
            "value": b"compressed-blob",
        },
    )


def _pva_connector(context=None, globs=(PVA_GLOB,), epics=None) -> EPICSConnector:
    """A connector wired for PVA subscriptions without touching connect()."""
    connector = EPICSConnector()
    connector._pva_channel_globs = list(globs)
    connector._p4p = _fake_p4p_module()
    connector._pva_context = context
    connector._timeout = 3.0
    connector._epics = epics if epics is not None else MagicMock()
    return connector


def _recording_loop(monkeypatch) -> list[tuple]:
    """Capture ``call_soon_threadsafe`` on the running loop instead of running it."""
    calls: list[tuple] = []
    loop = asyncio.get_event_loop()
    monkeypatch.setattr(loop, "call_soon_threadsafe", lambda fn, *args: calls.append((fn, args)))
    return calls


# ---------------------------------------------------------------------------
# Routing: which transport opens the subscription
# ---------------------------------------------------------------------------


class TestSubscribeRouting:
    @pytest.mark.asyncio
    async def test_pva_address_opens_a_monitor_and_no_ca_pv(self):
        context = FakeMonitorContext()
        epics = MagicMock()
        connector = _pva_connector(context=context, epics=epics)

        sub_id = await connector.subscribe(PVA_ADDRESS, lambda v: None)

        assert [call["name"] for call in context.calls] == [PVA_ADDRESS]
        epics.PV.assert_not_called()  # no Channel Access connection was made
        assert sub_id.startswith(f"{PVA_ADDRESS}_")
        assert connector._subscriptions[sub_id].kind == "pva"
        assert connector._subscriptions[sub_id].handle is context.subscriptions[0]

    @pytest.mark.asyncio
    async def test_ca_address_on_a_pva_capable_connector_still_uses_pyepics(self):
        context = FakeMonitorContext()
        pv = MagicMock()
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _pva_connector(context=context, epics=epics)

        sub_id = await connector.subscribe(CA_ADDRESS, lambda v: None)

        assert context.calls == []  # PVA client untouched
        assert epics.PV.call_args.args == (CA_ADDRESS,)
        assert connector._subscriptions[sub_id].kind == "ca"
        assert connector._subscriptions[sub_id].handle is pv

    @pytest.mark.asyncio
    async def test_subscribe_without_a_context_is_a_connection_error(self):
        connector = _pva_connector(context=None)

        with pytest.raises(ConnectionError) as excinfo:
            await connector.subscribe(PVA_ADDRESS, lambda v: None)

        assert PVA_ADDRESS in str(excinfo.value)
        assert "pva_channels" in str(excinfo.value)
        assert connector._subscriptions == {}


# ---------------------------------------------------------------------------
# Update delivery: marshaling onto the event loop
# ---------------------------------------------------------------------------


class TestUpdateMarshaling:
    @pytest.mark.asyncio
    async def test_update_is_handed_to_the_loop_not_called_inline(self, monkeypatch):
        """p4p's worker thread must never run the subscriber callback itself."""
        context = FakeMonitorContext()
        connector = _pva_connector(context=context)
        received = []
        subscriber = received.append  # bind once: a fresh bound method is never `is`-equal
        loop_calls = _recording_loop(monkeypatch)

        await connector.subscribe(PVA_ADDRESS, subscriber)
        context.fire(_scalar_update(1.5))

        assert received == []  # nothing ran on the monitor thread
        assert len(loop_calls) == 1
        scheduled_callback, args = loop_calls[0]
        assert scheduled_callback is subscriber
        assert args[0].value == 1.5
        assert args[0].metadata.units == "mA"

    @pytest.mark.asyncio
    async def test_update_fired_from_another_thread_reaches_the_subscriber(self):
        """End to end with the real loop: a worker-thread update is delivered."""
        context = FakeMonitorContext()
        connector = _pva_connector(context=context)
        received = []

        await connector.subscribe(PVA_ADDRESS, received.append)
        worker = threading.Thread(target=context.fire, args=(_scalar_update(2.5),))
        worker.start()
        worker.join()
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert [value.value for value in received] == [2.5]

    @pytest.mark.asyncio
    async def test_ca_subscription_still_marshals_through_the_loop(self, monkeypatch):
        """The CA callback path is unchanged by the shared wrapper."""
        pv = MagicMock()
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _pva_connector(epics=epics)
        received = []
        subscriber = received.append
        loop_calls = _recording_loop(monkeypatch)

        await connector.subscribe(CA_ADDRESS, subscriber)
        epics.PV.call_args.kwargs["callback"](pvname=CA_ADDRESS, value=7.0, units="A")

        assert received == []
        assert len(loop_calls) == 1
        assert loop_calls[0][0] is subscriber
        assert loop_calls[0][1][0].value == 7.0


# ---------------------------------------------------------------------------
# Filtering: sentinels and undecodable values never become a ChannelValue
# ---------------------------------------------------------------------------


class TestSentinelAndErrorFiltering:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sentinel",
        [FakeDisconnected("channel disconnected"), FakeRemoteError("server said no")],
        ids=["disconnected", "remote-error"],
    )
    async def test_exception_sentinels_never_reach_the_subscriber(self, monkeypatch, sentinel):
        """p4p reports connection state as a callback value — it is not data."""
        context = FakeMonitorContext()
        connector = _pva_connector(context=context)
        received = []
        loop_calls = _recording_loop(monkeypatch)

        await connector.subscribe(PVA_ADDRESS, received.append)
        context.fire(sentinel)

        assert received == []
        assert loop_calls == []  # nothing was even scheduled

    @pytest.mark.asyncio
    async def test_a_sentinel_does_not_end_the_subscription(self, monkeypatch):
        """A disconnect is transient: the next real update still gets through."""
        context = FakeMonitorContext()
        connector = _pva_connector(context=context)
        received = []
        loop_calls = _recording_loop(monkeypatch)

        await connector.subscribe(PVA_ADDRESS, received.append)
        context.fire(FakeDisconnected("gone"))
        context.fire(_scalar_update(3.5))

        assert [call[1][0].value for call in loop_calls] == [3.5]

    @pytest.mark.asyncio
    async def test_undecodable_value_is_dropped_without_raising(self, monkeypatch):
        """A compressed frame is logged and skipped — raising would kill the monitor."""
        context = FakeMonitorContext()
        connector = _pva_connector(context=context)
        received = []
        loop_calls = _recording_loop(monkeypatch)

        await connector.subscribe(PVA_ADDRESS, received.append)
        context.fire(_compressed_frame())  # must not propagate to p4p's dispatcher
        context.fire(_scalar_update(4.5))

        assert received == []
        assert [call[1][0].value for call in loop_calls] == [4.5]


# ---------------------------------------------------------------------------
# Teardown: one wrapper, two transports
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_closes_the_p4p_subscription(self):
        context = FakeMonitorContext()
        connector = _pva_connector(context=context)
        sub_id = await connector.subscribe(PVA_ADDRESS, lambda v: None)

        await connector.unsubscribe(sub_id)

        # FakeSubscription has no clear_callbacks: reaching for one would raise.
        assert context.subscriptions[0].close_calls == 1
        assert sub_id not in connector._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_clears_callbacks_on_a_ca_pv(self):
        pv = MagicMock(spec=["clear_callbacks"])  # a PV has no close()
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _pva_connector(epics=epics)
        sub_id = await connector.subscribe(CA_ADDRESS, lambda v: None)

        await connector.unsubscribe(sub_id)

        pv.clear_callbacks.assert_called_once()
        assert sub_id not in connector._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_twice_is_a_noop(self):
        context = FakeMonitorContext()
        connector = _pva_connector(context=context)
        sub_id = await connector.subscribe(PVA_ADDRESS, lambda v: None)

        await connector.unsubscribe(sub_id)
        await connector.unsubscribe(sub_id)

        assert context.subscriptions[0].close_calls == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_id_is_a_noop(self):
        connector = _pva_connector(context=FakeMonitorContext())

        await connector.unsubscribe("never-registered")


# ---------------------------------------------------------------------------
# The wrapper itself — the shape disconnect() teardown relies on
# ---------------------------------------------------------------------------


class TestChannelSubscriptionWrapper:
    def test_close_is_idempotent(self):
        handle = FakeSubscription()
        subscription = _ChannelSubscription("pva", handle)

        subscription.close()
        subscription.close()
        subscription.close()

        assert handle.close_calls == 1
        assert subscription.closed is True

    def test_close_dispatches_on_kind(self):
        pv = MagicMock(spec=["clear_callbacks"])
        _ChannelSubscription("ca", pv).close()

        pv.clear_callbacks.assert_called_once()

    def test_a_failing_handle_does_not_break_teardown(self):
        handle = MagicMock(spec=["close"])
        handle.close.side_effect = RuntimeError("already gone")
        subscription = _ChannelSubscription("pva", handle)

        subscription.close()  # best effort: swallowed so disconnect() can continue

        assert subscription.closed is True

    def test_handle_and_kind_are_readable(self):
        handle = FakeSubscription()
        subscription = _ChannelSubscription("pva", handle)

        assert subscription.kind == "pva"
        assert subscription.handle is handle
        assert subscription.closed is False
