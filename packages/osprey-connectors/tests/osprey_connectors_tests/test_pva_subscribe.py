"""PVA subscriptions: monitor lifecycle, callback marshaling and sentinel filtering.

Task 1.1 decided WHICH transport an address uses and task 1.2 mapped a p4p
``Value`` onto :class:`ChannelValue`; this file covers the third path into that
mapping — a p4p monitor, which differs from a read in three ways that matter:

* its updates arrive on p4p's worker thread, so the subscriber callback has to
  be handed to the event loop rather than called where the update landed;
* p4p delivers connection state (``Disconnected`` and friends) to the same
  callback, *as a value* — the monitor is opened with ``notify_disconnect=True``
  so every drop is reported — so anything that is an exception instance is a
  connection-down report, never channel data, and the monitor is never
  re-created because of one;
* raising inside a monitor callback makes p4p close the subscription, so a
  value the mapping cannot decode (a compressed NTNDArray, say) must be dropped
  rather than propagated.

Teardown is asserted on both transports together: both handles close with
``close()``, but an aioca monitor belongs to the CA loop and is closed there,
so the connector stores every subscription behind one ``_ChannelSubscription``
wrapper and dispatches there. The fakes below are deliberately strict — no ``MagicMock``
for the handles — so calling the wrong teardown method raises instead of
silently recording a call that would never work against a real library.
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from osprey_connectors.control_system._epics_connector import (
    _ChannelSubscription,
)
from osprey_connectors.errors import ConnectorUnavailableError
from osprey_connectors_tests._epics_fakes import (
    FakeAioca,
    FakeChannel,
    FakeDisconnected,
    FakeRemoteError,
    FakeValue,
    ca_reading,
)
from osprey_connectors_tests._epics_fakes import pva_connector as _pva_connector

PVA_GLOB = "SR:CAM*:IMAGE"
PVA_ADDRESS = "SR:CAM1:IMAGE"
CA_ADDRESS = "SR:BEAM:CURRENT"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSubscription:
    """A p4p ``Subscription`` stand-in: it can be closed, and nothing else.

    Deliberately has nothing but ``close`` — any other teardown spelling must
    fail loudly rather than pass a mock assertion.
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
        ca = FakeAioca()
        connector = _pva_connector(context=context, aioca=ca)

        sub_id = await connector.subscribe(PVA_ADDRESS, lambda v: None)

        assert [call["name"] for call in context.calls] == [PVA_ADDRESS]
        assert context.calls[0]["notify_disconnect"] is True  # drops are reported
        assert ca.calls == []  # no Channel Access connection was made
        assert sub_id.startswith(f"{PVA_ADDRESS}_")
        assert connector._subscriptions[sub_id].kind == "pva"
        assert connector._subscriptions[sub_id].handle is context.subscriptions[0]

    @pytest.mark.asyncio
    async def test_ca_address_on_a_pva_capable_connector_still_uses_channel_access(self):
        context = FakeMonitorContext()
        ca = FakeAioca({CA_ADDRESS: FakeChannel()})
        connector = _pva_connector(context=context, aioca=ca)

        sub_id = await connector.subscribe(CA_ADDRESS, lambda v: None)

        assert context.calls == []  # PVA client untouched
        ctrl, time_ = ca.monitors_of(CA_ADDRESS)  # the address's CA record opened both
        assert ctrl is not None and time_ is not None
        assert sub_id.startswith(f"{CA_ADDRESS}_")
        assert connector._subscriptions[sub_id].kind == "ca"
        assert connector._subscriptions[sub_id].handle.sub_id == sub_id

    @pytest.mark.asyncio
    async def test_subscribe_without_a_context_is_connector_unavailable(self):
        connector = _pva_connector(context=None)

        with pytest.raises(ConnectorUnavailableError) as excinfo:
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
        """A CA event from the CA loop is posted to the subscriber's loop, never delivered inline."""
        ca = FakeAioca({CA_ADDRESS: FakeChannel()})
        connector = _pva_connector(aioca=ca)
        received = []
        subscriber = received.append
        loop_calls = _recording_loop(monkeypatch)

        await connector.subscribe(CA_ADDRESS, subscriber)
        ctrl, time_ = ca.monitors_of(CA_ADDRESS)
        ctrl.fire(ca_reading(0.0, units="mA", precision=3))
        time_.fire(ca_reading(7.0))

        assert received == []
        assert len(loop_calls) == 1
        assert loop_calls[0][0] is subscriber
        assert loop_calls[0][1][0].value == 7.0
        assert loop_calls[0][1][0].metadata.units == "mA"  # the connection's CTRL data


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
        """p4p reports connection state as a callback value — it is not data.

        After a value, the exception update is a connection-down report: the
        subscriber's ``on_connection`` is told ``False`` and the value callback
        is never scheduled with it.
        """
        context = FakeMonitorContext()
        connector = _pva_connector(context=context)
        received = []
        states = []
        subscriber = received.append
        loop_calls = _recording_loop(monkeypatch)

        await connector.subscribe(PVA_ADDRESS, subscriber, on_connection=states.append)
        context.fire(_scalar_update(1.5))
        loop_calls.clear()
        context.fire(sentinel)

        assert received == []
        assert [(fn, args) for fn, args in loop_calls] == [(states.append, (False,))]
        assert not any(fn is subscriber for fn, _ in loop_calls)

    @pytest.mark.asyncio
    async def test_a_sentinel_does_not_end_the_subscription(self, monkeypatch):
        """A disconnect is transient: the next real update still gets through.

        The monitor is neither closed nor re-opened on a drop — p4p restores it
        — and the value after the drop is preceded by a connection-up report.
        """
        context = FakeMonitorContext()
        connector = _pva_connector(context=context)
        received = []
        states = []
        subscriber = received.append
        loop_calls = _recording_loop(monkeypatch)

        await connector.subscribe(PVA_ADDRESS, subscriber, on_connection=states.append)
        context.fire(_scalar_update(2.5))
        context.fire(FakeDisconnected("gone"))
        context.fire(_scalar_update(3.5))

        assert [call[1][0].value for call in loop_calls if call[0] is subscriber] == [2.5, 3.5]
        assert [call[1][0] for call in loop_calls if call[0] is not subscriber] == [
            True,
            False,
            True,
        ]
        assert len(context.calls) == 1  # no monitor re-creation
        assert context.subscriptions[0].close_calls == 0

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
    async def test_unsubscribe_releases_a_ca_subscription(self):
        ca = FakeAioca({CA_ADDRESS: FakeChannel()})
        connector = _pva_connector(aioca=ca)
        sub_id = await connector.subscribe(CA_ADDRESS, lambda v: None)
        assert len(ca.monitors) == 2  # the record's CTRL and TIME monitors

        await connector.unsubscribe(sub_id)

        assert all(monitor.closed for monitor in ca.monitors)
        assert ca.monitors_of(CA_ADDRESS) == (None, None)
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

    @pytest.mark.asyncio
    async def test_close_dispatches_on_kind(self):
        ca = FakeAioca({CA_ADDRESS: FakeChannel()})
        connector = _pva_connector(aioca=ca)
        sub_id = await connector.subscribe(CA_ADDRESS, lambda v: None)
        handle = connector._subscriptions.pop(sub_id).handle  # the record handle

        _ChannelSubscription("ca", handle).close()  # no CA loop: closed in place

        assert ca.monitors_of(CA_ADDRESS) == (None, None)  # the last subscriber left
