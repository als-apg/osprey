"""PVA subscriptions report a lost channel and never deliver it as a value.

The monitor is opened with ``notify_disconnect=True``: p4p then hands the
callback a ``Disconnected`` update whenever the channel goes away, instead of
going quiet. Four rules follow, and this file pins each of them:

* an exception update is a connection-down report (``on_connection(False)``),
  never a value;
* the monitor is never re-created — p4p restores it when the server returns,
  and the next value is preceded by ``on_connection(True)``;
* the first value update after each (re)connection is the subscription's
  metadata source, so ``get_metadata`` of a subscribed address answers from it
  and opens no second request;
* ``get_metadata`` of a subscribed address that has delivered nothing awaits
  that first update for up to ``timeout_s``, then raises ``TimeoutError``.

The first half drives the connector with an injected monitor context, so each
rule is asserted on exact sequences. The second half runs the same rules
against a real in-process p4p server (the ``test_pva_live_fixture.py``
pattern: ``Server(isolate=True)`` on a random loopback port, reached through
the name server), closing and re-opening a served PV to make the wire drop.
"""

import asyncio
import gc
import threading
import time
from unittest.mock import MagicMock

import pytest

from osprey_connectors.control_system._base import ChannelMetadata
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.errors import ChannelUnreachableError

PVA_ADDRESS = "SR:PVA:CURRENT"


# ---------------------------------------------------------------------------
# Injected monitor context
# ---------------------------------------------------------------------------


class _Disconnected(RuntimeError):
    """Stands in for p4p's Disconnected."""


class _Value:
    """An NTScalar-shaped update with display and control limits."""

    def __init__(self, value: float, units: str = "mA", limits=(0.0, 500.0)):
        self._fields = {
            "value": value,
            "timeStamp": _Struct({"secondsPastEpoch": 1_700_000_000, "nanoseconds": 0}),
            "display": _Struct({"units": units, "limitLow": limits[0], "limitHigh": limits[1]}),
            "control": _Struct({"limitLow": limits[0], "limitHigh": limits[1]}),
        }

    def getID(self):
        return "epics:nt/NTScalar:1.0"

    def get(self, name, default=None):
        return self._fields.get(name, default)

    def __contains__(self, name):
        return name in self._fields


class _Struct:
    def __init__(self, fields):
        self._fields = fields

    def get(self, name, default=None):
        return self._fields.get(name, default)

    def __contains__(self, name):
        return name in self._fields


class _Subscription:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _MonitorContext:
    """A p4p thread ``Context`` with ``monitor`` only — a ``get`` would raise."""

    def __init__(self):
        self.calls: list[dict] = []
        self.subscriptions: list[_Subscription] = []

    def monitor(self, name, cb, **kwargs):
        self.calls.append({"name": name, "cb": cb, **kwargs})
        subscription = _Subscription()
        self.subscriptions.append(subscription)
        return subscription

    def fire(self, update, index: int = 0):
        """Deliver an update from a worker thread, the way p4p does."""
        worker = threading.Thread(target=self.calls[index]["cb"], args=(update,))
        worker.start()
        worker.join()


def _connector(context, timeout: float = 1.0) -> EPICSConnector:
    connector = EPICSConnector()
    connector._pva_channel_globs = ["SR:PVA:*"]
    connector._p4p = MagicMock()
    connector._pva_context = context
    connector._timeout = timeout
    connector._connected = True
    return connector


async def _settle():
    await asyncio.sleep(0.02)


class TestDisconnectReport:
    async def test_monitor_is_opened_with_notify_disconnect(self):
        context = _MonitorContext()
        connector = _connector(context)

        await connector.subscribe(PVA_ADDRESS, lambda v: None)

        assert context.calls[0]["notify_disconnect"] is True

    async def test_drop_reports_false_and_never_becomes_a_value(self):
        context = _MonitorContext()
        connector = _connector(context)
        events: list = []

        await connector.subscribe(
            PVA_ADDRESS, lambda v: events.append(v.value), on_connection=events.append
        )
        context.fire(_Value(1.0))
        context.fire(_Disconnected("channel lost"))
        context.fire(RuntimeError("remote error"))  # any exception update is "down"
        await _settle()

        assert events == [True, 1.0, False]

    async def test_resume_reports_true_again_on_the_same_monitor(self):
        context = _MonitorContext()
        connector = _connector(context)
        events: list = []

        await connector.subscribe(
            PVA_ADDRESS, lambda v: events.append(v.value), on_connection=events.append
        )
        context.fire(_Value(1.0))
        context.fire(_Disconnected("channel lost"))
        context.fire(_Value(2.0))
        await _settle()

        assert events == [True, 1.0, False, True, 2.0]
        assert len(context.calls) == 1  # the monitor was not re-created
        assert context.subscriptions[0].close_calls == 0

    async def test_a_channel_that_never_connects_reports_nothing(self):
        """p4p's initial Disconnected for an unreached channel is not a drop."""
        context = _MonitorContext()
        connector = _connector(context)
        events: list = []

        await connector.subscribe(PVA_ADDRESS, events.append, on_connection=events.append)
        context.fire(_Disconnected("not yet connected"))
        await _settle()

        assert events == []


class TestSubscribedMetadata:
    async def test_metadata_comes_from_the_first_update_without_a_get(self):
        context = _MonitorContext()  # has no get(): a one-shot request would raise
        connector = _connector(context)

        await connector.subscribe(PVA_ADDRESS, lambda v: None)
        context.fire(_Value(1.0, units="mA", limits=(0.0, 500.0)))
        context.fire(_Value(2.0, units="A", limits=(1.0, 2.0)))  # not the source

        metadata = await connector.get_metadata(PVA_ADDRESS)

        assert isinstance(metadata, ChannelMetadata)
        assert metadata.units == "mA"
        assert (metadata.control_low, metadata.control_high) == (0.0, 500.0)

    async def test_get_metadata_awaits_the_first_update(self):
        context = _MonitorContext()
        connector = _connector(context, timeout=2.0)

        await connector.subscribe(PVA_ADDRESS, lambda v: None)
        pending = asyncio.ensure_future(connector.get_metadata(PVA_ADDRESS))
        await _settle()
        assert not pending.done()  # parked on the monitor, not answered by a get

        context.fire(_Value(1.0, units="mA"))
        metadata = await asyncio.wait_for(pending, 1.0)

        assert metadata.units == "mA"

    async def test_get_metadata_is_the_channel_unreachable_when_no_update_arrives(self):
        context = _MonitorContext()
        connector = _connector(context, timeout=0.2)

        await connector.subscribe(PVA_ADDRESS, lambda v: None)
        started = time.monotonic()
        with pytest.raises(ChannelUnreachableError) as excinfo:
            await connector.get_metadata(PVA_ADDRESS)

        assert time.monotonic() - started < 1.0
        assert PVA_ADDRESS in str(excinfo.value)

    async def test_a_drop_empties_the_metadata_until_the_channel_returns(self):
        context = _MonitorContext()
        connector = _connector(context, timeout=0.2)

        await connector.subscribe(PVA_ADDRESS, lambda v: None)
        context.fire(_Value(1.0, units="mA"))
        context.fire(_Disconnected("channel lost"))

        with pytest.raises(ChannelUnreachableError):
            await connector.get_metadata(PVA_ADDRESS)

        context.fire(_Value(1.0, units="A"))  # first update of the new connection
        metadata = await connector.get_metadata(PVA_ADDRESS)
        assert metadata.units == "A"

    async def test_unsubscribed_address_keeps_the_metadata_only_get(self, monkeypatch):
        context = _MonitorContext()
        connector = _connector(context)
        expected = ChannelMetadata(units="V")
        calls = []

        def one_shot(address, timeout):
            calls.append((address, timeout))
            return expected

        monkeypatch.setattr(connector, "_read_metadata_pva", one_shot)

        assert await connector.get_metadata(PVA_ADDRESS) is expected
        assert calls == [(PVA_ADDRESS, 1.0)]

    async def test_unsubscribe_while_waiting_falls_back_to_a_get(self, monkeypatch):
        context = _MonitorContext()
        connector = _connector(context, timeout=2.0)
        expected = ChannelMetadata(units="V")
        monkeypatch.setattr(connector, "_read_metadata_pva", lambda address, timeout: expected)

        sub_id = await connector.subscribe(PVA_ADDRESS, lambda v: None)
        pending = asyncio.ensure_future(connector.get_metadata(PVA_ADDRESS))
        await _settle()
        await connector.unsubscribe(sub_id)

        assert await asyncio.wait_for(pending, 1.0) is expected
        assert connector._pva_metadata_slots == {}


# ---------------------------------------------------------------------------
# Live: a real p4p server in this process
# ---------------------------------------------------------------------------

_PVA_ENV_KEYS = ("EPICS_PVA_NAME_SERVERS", "EPICS_PVA_ADDR_LIST", "EPICS_PVA_AUTO_ADDR_LIST")
LIVE_PREFIX = "OSPREY:DISCONNECT:"
LIVE_DROP = f"{LIVE_PREFIX}DROP"
LIVE_META = f"{LIVE_PREFIX}META"
LIVE_UNSERVED = f"{LIVE_PREFIX}NOT:SERVED"
LIVE_TIMEOUT_S = 5.0
LIVE_LIMITS = (-10.0, 10.0)


def _live_nt():
    from p4p.nt import NTScalar

    return NTScalar("d", display=True, control=True)


def _live_initial(nt, value: float):
    return nt.wrap(
        {
            "value": value,
            "display": {"units": "mm", "limitLow": LIVE_LIMITS[0], "limitHigh": LIVE_LIMITS[1]},
            "control": {"limitLow": LIVE_LIMITS[0], "limitHigh": LIVE_LIMITS[1]},
        }
    )


@pytest.fixture(scope="module")
def live_pvs():
    """A localhost-only p4p server on a random port and the PVs it serves."""
    pytest.importorskip("p4p", reason="p4p is required to serve a live PVAccess fixture")
    from p4p.server import Server
    from p4p.server.thread import SharedPV

    nt = _live_nt()
    pvs = {
        LIVE_DROP: SharedPV(nt=nt, initial=_live_initial(nt, 1.0)),
        LIVE_META: SharedPV(nt=nt, initial=_live_initial(nt, 2.0)),
    }
    server = Server(providers=[pvs], isolate=True)
    try:
        yield server.conf(), pvs
    finally:
        server.stop()


@pytest.fixture(scope="module")
def live_connector(live_pvs):
    """One connector pointed at the live server, torn down with its env vars."""
    import os

    conf, _ = live_pvs
    saved_env = {key: os.environ.get(key) for key in _PVA_ENV_KEYS}
    connector = EPICSConnector()
    asyncio.run(
        connector.connect(
            {
                "timeout_s": LIVE_TIMEOUT_S,
                "pva_channels": [f"{LIVE_PREFIX}*"],
                "pva_gateway": {
                    "address": "127.0.0.1",
                    "port": int(conf["EPICS_PVAS_SERVER_PORT"]),
                    "use_name_server": True,
                },
            }
        )
    )
    try:
        yield connector
    finally:
        asyncio.run(connector.disconnect())
        gc.collect()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _wait_for(predicate, timeout: float = LIVE_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached before the deadline")
        await asyncio.sleep(0.02)


class TestLiveDisconnect:
    async def test_server_drop_reports_false_then_true_on_the_same_monitor(
        self, live_pvs, live_connector
    ):
        _, pvs = live_pvs
        pv = pvs[LIVE_DROP]
        events: list = []
        monitors: list = []
        context = live_connector._pva_context
        real_monitor = context.monitor

        def counting_monitor(*args, **kwargs):
            monitors.append(kwargs)
            return real_monitor(*args, **kwargs)

        context.monitor = counting_monitor
        try:
            sub_id = await live_connector.subscribe(
                LIVE_DROP, lambda v: events.append(v.value), on_connection=events.append
            )
        finally:
            del context.monitor
        try:
            await _wait_for(lambda: 1.0 in events)
            assert events[:2] == [True, 1.0]

            pv.close(destroy=True)  # the server drops every client of this PV
            await _wait_for(lambda: False in events)
            assert all(isinstance(e, bool) for e in events[events.index(False) :])

            pv.open(_live_initial(_live_nt(), 3.0))
            await _wait_for(lambda: 3.0 in events)

            after_drop = events[events.index(False) :]
            assert after_drop[after_drop.index(True) + 1 :][-1] == 3.0
            assert after_drop.index(True) < after_drop.index(3.0)
            assert len(monitors) == 1  # one monitor for the whole drop and resume
            assert monitors[0]["notify_disconnect"] is True
        finally:
            await live_connector.unsubscribe(sub_id)

    async def test_get_metadata_of_a_subscribed_address_awaits_its_first_update(
        self, live_connector
    ):
        sub_id = await live_connector.subscribe(LIVE_META, lambda v: None)
        try:
            metadata = await live_connector.get_metadata(LIVE_META)

            assert metadata.units == "mm"
            assert (metadata.display_low, metadata.display_high) == LIVE_LIMITS
            assert (metadata.control_low, metadata.control_high) == LIVE_LIMITS
        finally:
            await live_connector.unsubscribe(sub_id)

    async def test_get_metadata_of_a_subscribed_unserved_address_times_out(
        self, live_connector, monkeypatch
    ):
        monkeypatch.setattr(live_connector, "_timeout", 0.5)
        events: list = []
        sub_id = await live_connector.subscribe(
            LIVE_UNSERVED, events.append, on_connection=events.append
        )
        try:
            with pytest.raises(ChannelUnreachableError):
                await live_connector.get_metadata(LIVE_UNSERVED)
            assert events == []  # never connected: nothing reported, no value
        finally:
            await live_connector.unsubscribe(sub_id)
