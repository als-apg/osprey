"""PVA subscriptions: monitor lifecycle, callback marshaling and undecodable updates.

Routing decided WHICH transport an address uses and the read mapping turned a
pvapy ``PvObject`` into a :class:`ChannelValue`; this file covers the third
path into that mapping — a pvapy monitor, which differs from a read in ways
that matter:

* it runs on a ``Channel`` of its own (a pvapy Channel runs one monitor), so a
  subscription never borrows the connector's shared read channel;
* its updates arrive on pvapy's monitor thread, so the subscriber callback has
  to be handed to the event loop rather than called where the update landed;
* a value the mapping cannot decode (a compressed NTNDArray, say) must be
  dropped with a log line rather than raised into pvapy's dispatcher.

Both transports are pvapy monitors now, torn down the same way: ``stopMonitor``
and then ``unsubscribe`` of the one named subscriber. The fake ``pvaccess``
module (``tests/connectors/_epics_fakes.py``) records both, so a teardown that
skipped either is visible.
"""

import asyncio
import threading

import numpy as np
import pytest

from osprey.connectors.control_system.epics_connector import (
    EPICSConnector,
    _ChannelSubscription,
)
from tests.connectors._epics_fakes import (
    CA_DISPLAY_REQUEST,
    CA_READ_REQUEST,
    PVA_READ_REQUEST,
    FakePvaccess,
    ndarray_record,
    pva_connector,
    record,
    timed_out,
)

PVA_ADDRESS = "SR:CAM1:IMAGE"
CA_ADDRESS = "SR:BEAM:CURRENT"


def _connector() -> EPICSConnector:
    """A PVA-capable connector serving one PVA and one CA address."""
    pvaccess = FakePvaccess()
    pvaccess.serve(PVA_ADDRESS, record(1.5, units="mA"))
    pvaccess.serve(CA_ADDRESS, record(0.5, units="A"))
    return pva_connector(pvaccess)


def _monitor(connector: EPICSConnector, sub_id: str):
    """The fake Channel a subscription's monitor runs on."""
    return connector._subscriptions[sub_id].channel


def _compressed_frame() -> dict:
    """A codec-tagged NTNDArray — the mapping refuses these with a ValueError."""
    return ndarray_record(np.zeros(7, dtype=np.uint8), [64, 48], member="ubyteValue", codec="jpeg")


def _recording_loop(monkeypatch) -> list[tuple]:
    """Capture ``call_soon_threadsafe`` on the running loop instead of running it.

    Installed only after ``subscribe`` has returned: the connector's own
    offload hands its result back through the same method.
    """
    calls: list[tuple] = []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "call_soon_threadsafe", lambda fn, *args: calls.append((fn, args)))
    return calls


# ---------------------------------------------------------------------------
# Routing: which transport opens the subscription
# ---------------------------------------------------------------------------


class TestSubscribeRouting:
    @pytest.mark.asyncio
    async def test_pva_address_opens_a_pva_monitor_and_no_ca_channel(self):
        connector = _connector()

        sub_id = await connector.subscribe(PVA_ADDRESS, lambda v: None)

        pvaccess = connector._pvaccess
        (started,) = pvaccess.calls("startMonitor")
        assert (started["address"], started["provider"]) == (PVA_ADDRESS, "PVA")
        assert started["request"] == PVA_READ_REQUEST
        assert pvaccess.calls(provider="CA") == []  # no Channel Access connection was made
        assert sub_id.startswith(f"{PVA_ADDRESS}_")
        assert _monitor(connector, sub_id).subscribers.keys() == {
            connector._subscriptions[sub_id].name
        }

    @pytest.mark.asyncio
    async def test_ca_address_on_a_pva_capable_connector_monitors_over_ca(self):
        connector = _connector()

        sub_id = await connector.subscribe(CA_ADDRESS, lambda v: None)

        pvaccess = connector._pvaccess
        (started,) = pvaccess.calls("startMonitor")
        assert (started["address"], started["provider"]) == (CA_ADDRESS, "CA")
        assert started["request"] == CA_READ_REQUEST
        # Its display metadata is fetched up front, never on pvapy's callback thread.
        assert len(pvaccess.calls("get", request=CA_DISPLAY_REQUEST)) == 1
        assert pvaccess.calls(provider="PVA") == []
        assert sub_id.startswith(f"{CA_ADDRESS}_")

    @pytest.mark.asyncio
    async def test_each_subscription_gets_its_own_channel(self):
        """A pvapy Channel runs one monitor: two subscriptions, two Channels."""
        connector = _connector()

        first = await connector.subscribe(PVA_ADDRESS, lambda v: None)
        second = await connector.subscribe(PVA_ADDRESS, lambda v: None)

        assert first != second
        assert _monitor(connector, first) is not _monitor(connector, second)

    @pytest.mark.asyncio
    async def test_subscribe_without_a_client_is_a_connection_error(self):
        connector = pva_connector()
        connector._pvaccess = None

        with pytest.raises(ConnectionError) as excinfo:
            await connector.subscribe(PVA_ADDRESS, lambda v: None)

        assert PVA_ADDRESS in str(excinfo.value)
        assert connector._subscriptions == {}

    @pytest.mark.asyncio
    async def test_a_monitor_that_cannot_start_is_a_connection_error_and_released(self):
        """An unreachable channel leaves no half-open monitor behind."""
        connector = _connector()
        connector._pvaccess.monitor_errors[PVA_ADDRESS] = timed_out(PVA_ADDRESS)

        with pytest.raises(ConnectionError, match=PVA_ADDRESS):
            await connector.subscribe(PVA_ADDRESS, lambda v: None)

        assert connector._subscriptions == {}
        (channel,) = connector._pvaccess.channels_for(PVA_ADDRESS)
        assert channel.subscribers == {}
        assert [entry["op"] for entry in connector._pvaccess.log[-2:]] == [
            "stopMonitor",
            "unsubscribe",
        ]


# ---------------------------------------------------------------------------
# Update delivery: marshaling onto the event loop
# ---------------------------------------------------------------------------


class TestUpdateMarshaling:
    @pytest.mark.asyncio
    async def test_the_first_update_is_the_current_value(self):
        connector = _connector()
        received = []

        await connector.subscribe(PVA_ADDRESS, received.append)
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert [value.value for value in received] == [1.5]
        assert received[0].metadata.units == "mA"
        assert received[0].metadata.raw_metadata["provider"] == "pva"

    @pytest.mark.asyncio
    async def test_update_is_handed_to_the_loop_not_called_inline(self, monkeypatch):
        """pvapy's monitor thread must never run the subscriber callback itself."""
        connector = _connector()
        received = []
        subscriber = received.append  # bind once: a fresh bound method is never `is`-equal
        sub_id = await connector.subscribe(PVA_ADDRESS, subscriber)
        await asyncio.sleep(0.01)
        received.clear()
        loop_calls = _recording_loop(monkeypatch)

        _monitor(connector, sub_id).fire(record(2.5, units="mA"))

        assert received == []  # nothing ran on the monitor thread
        assert len(loop_calls) == 1
        scheduled_callback, args = loop_calls[0]
        assert scheduled_callback is subscriber
        assert args[0].value == 2.5
        assert args[0].metadata.units == "mA"

    @pytest.mark.asyncio
    async def test_update_fired_from_another_thread_reaches_the_subscriber(self):
        """End to end with the real loop: a worker-thread update is delivered."""
        connector = _connector()
        received = []

        sub_id = await connector.subscribe(PVA_ADDRESS, received.append)
        worker = threading.Thread(target=_monitor(connector, sub_id).fire, args=(record(3.5),))
        worker.start()
        worker.join()
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert [value.value for value in received] == [1.5, 3.5]

    @pytest.mark.asyncio
    async def test_ca_subscription_marshals_through_the_loop_too(self, monkeypatch):
        """The CA monitor is the same pvapy monitor, delivered the same way."""
        connector = _connector()
        received = []
        subscriber = received.append
        sub_id = await connector.subscribe(CA_ADDRESS, subscriber)
        await asyncio.sleep(0.01)
        loop_calls = _recording_loop(monkeypatch)

        _monitor(connector, sub_id).fire(record(7.0, units="A"))

        assert len(loop_calls) == 1
        assert loop_calls[0][0] is subscriber
        assert loop_calls[0][1][0].value == 7.0
        assert loop_calls[0][1][0].metadata.units == "A"  # from the cached display


# ---------------------------------------------------------------------------
# Filtering: undecodable values never become a ChannelValue
# ---------------------------------------------------------------------------


class TestUndecodableUpdates:
    @pytest.mark.asyncio
    async def test_undecodable_value_is_dropped_without_raising(self, monkeypatch):
        """A compressed frame is logged and skipped; the next update still gets through."""
        connector = _connector()
        received = []
        sub_id = await connector.subscribe(PVA_ADDRESS, received.append)
        await asyncio.sleep(0.01)
        received.clear()
        loop_calls = _recording_loop(monkeypatch)
        monitor = _monitor(connector, sub_id)

        monitor.fire(_compressed_frame())  # must not propagate to pvapy's dispatcher
        monitor.fire(record(4.5))

        assert received == []
        assert [call[1][0].value for call in loop_calls] == [4.5]

    def test_an_update_after_the_loop_closed_is_dropped_quietly(self):
        """A monitor still running under a closed loop must not raise into pvapy."""
        connector = _connector()
        closed = asyncio.new_event_loop()
        closed.close()
        received = []

        # The first update is delivered inside startMonitor, onto the closed loop.
        sub_id = connector._start_monitor(PVA_ADDRESS, received.append, closed, True)
        _monitor(connector, sub_id).fire(record(9.0))  # would raise RuntimeError unguarded

        assert received == []
        connector._subscriptions.pop(sub_id).close()


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_stops_the_monitor_and_removes_the_subscriber(self):
        connector = _connector()
        sub_id = await connector.subscribe(PVA_ADDRESS, lambda v: None)
        monitor = _monitor(connector, sub_id)
        name = connector._subscriptions[sub_id].name

        await connector.unsubscribe(sub_id)

        assert monitor.monitoring is False
        assert monitor.subscribers == {}
        assert connector._pvaccess.calls("unsubscribe") == [
            {
                "op": "unsubscribe",
                "address": PVA_ADDRESS,
                "provider": "PVA",
                "request": name,
                "timeout": 3.0,
            }
        ]
        assert sub_id not in connector._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_twice_is_a_noop(self):
        connector = _connector()
        sub_id = await connector.subscribe(PVA_ADDRESS, lambda v: None)

        await connector.unsubscribe(sub_id)
        await connector.unsubscribe(sub_id)

        assert len(connector._pvaccess.calls("stopMonitor")) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_id_is_a_noop(self):
        connector = _connector()

        await connector.unsubscribe("never-registered")

        assert connector._pvaccess.log == []


# ---------------------------------------------------------------------------
# The wrapper itself — the shape disconnect() teardown relies on
# ---------------------------------------------------------------------------


class _Channel:
    """A monitor channel stand-in recording its teardown calls, in order."""

    def __init__(self, fail_stop: bool = False) -> None:
        self.calls: list[str] = []
        self._fail_stop = fail_stop

    def stopMonitor(self) -> None:  # pvapy's spelling
        self.calls.append("stopMonitor")
        if self._fail_stop:
            raise RuntimeError("already gone")

    def unsubscribe(self, name: str) -> None:
        self.calls.append(f"unsubscribe:{name}")


class TestChannelSubscriptionWrapper:
    def test_close_is_idempotent(self):
        channel = _Channel()
        subscription = _ChannelSubscription(channel, "osprey-1")

        subscription.close()
        subscription.close()
        subscription.close()

        assert channel.calls == ["stopMonitor", "unsubscribe:osprey-1"]
        assert subscription.closed is True

    def test_a_failing_step_does_not_skip_the_next_one(self):
        """Best effort: a dead monitor still has its subscriber removed."""
        channel = _Channel(fail_stop=True)
        subscription = _ChannelSubscription(channel, "osprey-1")

        subscription.close()  # swallowed so disconnect() can continue

        assert channel.calls == ["stopMonitor", "unsubscribe:osprey-1"]
        assert subscription.closed is True

    def test_channel_and_name_are_readable(self):
        channel = _Channel()
        subscription = _ChannelSubscription(channel, "osprey-1")

        assert subscription.channel is channel
        assert subscription.name == "osprey-1"
        assert subscription.closed is False
