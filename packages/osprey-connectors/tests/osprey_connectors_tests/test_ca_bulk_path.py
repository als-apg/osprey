"""The Channel Access subscription path (``_epics_bulk``), unit and over a real wire.

Every Channel Access subscriber of one address is served from one record on the
CA loop: a CTRL monitor (the record's metadata) and a TIME monitor (its values),
both opened with ``all_updates`` and ``notify_disconnect``. A TIME event that
arrives before its connection's CTRL is held and delivered right after it; "up"
is reported ahead of a connection's first value; a drop is believed only while
the channel is really down; a repoint reports the record down and opens a fresh
pair of monitors on the new context.

Two halves:

* **Unit** — the connector on a fake ``aioca`` (:class:`FakeAioca`) and an
  inline CA loop. Tests fire readings and disconnects into the monitors the
  record opened, the way aioca would on the CA loop.
* **Wire** — a soft IOC (``tests/connectors/_soft_ioc.py``) and a client in its
  own process and session: each scenario runs as
  ``python -m osprey_connectors_tests.test_ca_bulk_path <scenario> <json args>`` so its
  CA context is created from exactly the environment the scenario chose (the
  no-gateway UDP shape or the gateway name-server shape). The client asks this
  process to kill or restart the IOC over a line protocol on stdin/stdout and
  prints one JSON result.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import time
import zoneinfo
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey_connectors.control_system import _epics_bulk
from osprey_connectors.control_system._base import ChannelMetadata, ChannelValue
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.errors import ChannelUnreachableError
from osprey_connectors_tests._epics_fakes import (
    DBR_ENUM,
    FakeAioca,
    FakeChannel,
    FakeSubscription,
    InlineCaLoop,
    ca_connector,
    ca_reading,
)
from osprey_connectors_tests._paths import TESTS_ROOT

# xdist_group("soft-ioc"): the wire half starts real soft IOCs. Their ports are
# probed free and then bound by a child process a moment later, so two workers
# probing at once can pick the same port; the IOCs' CA/PVA client sockets and
# repeater also share the host's UDP space. One worker keeps the probe-then-bind
# window free of a second prober.
pytestmark = pytest.mark.xdist_group("soft-ioc")

REPO = Path(__file__).resolve().parents[2]

ALARMED = "BULK:TEST:ALARMED"
PLAIN = "BULK:TEST:PLAIN"
MODE = "BULK:TEST:MODE"

NAME = "SR:CH"

# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

CTRL_LIMITS = {
    "lower_disp_limit": 0.0,
    "upper_disp_limit": 500.0,
    "lower_ctrl_limit": 0.0,
    "upper_ctrl_limit": 450.0,
    "lower_warning_limit": 10.0,
    "upper_warning_limit": 400.0,
    "lower_alarm_limit": 5.0,
    "upper_alarm_limit": 420.0,
}


def ctrl_reading(*, units: str = "mA", precision: int = 2, **overrides: Any) -> Any:
    """A CTRL-form reading carrying units, precision and the eight limits."""
    reading = ca_reading(0.0, units=units, precision=precision)
    for attr, limit in {**CTRL_LIMITS, **overrides}.items():
        setattr(reading, attr, limit)
    return reading


def time_reading(value: Any, *, status: int = 0, severity: int = 0, **kwargs: Any) -> Any:
    """A TIME-form reading."""
    return ca_reading(value, status=status, severity=severity, **kwargs)


def bulk_connector(*, timeout: float = 1.0, channel: FakeChannel | None = None):
    """A connector on a fake aioca whose every name answers from one channel."""
    aioca = FakeAioca(default=channel if channel is not None else FakeChannel())
    return ca_connector(aioca=aioca, timeout=timeout), aioca


def event_counts(connector: EPICSConnector, address: str) -> dict[str, int]:
    """Cumulative CTRL and TIME events of ``address``'s subscription record; zeros without one."""
    if connector._bulk is None:
        return {"ctrl": 0, "time": 0}
    return connector._bulk.event_counts(address)


async def drain() -> None:
    """Let posted callbacks and deferred removals run."""
    for _ in range(5):
        await asyncio.sleep(0)


async def live(connector: EPICSConnector, aioca: FakeAioca, name: str = NAME, **kwargs: Any):
    """Subscribe and deliver the connection's CTRL event: the record is up."""
    received: list[ChannelValue] = []
    sub_id = await connector.subscribe(name, received.append, **kwargs)
    ctrl, _ = aioca.monitors_of(name)
    ctrl.fire(ctrl_reading())
    await drain()
    return sub_id, received


def drop(aioca: FakeAioca, monitor: FakeSubscription) -> None:
    """A real drop: the channel is down, then the monitor reports it."""
    aioca.default.connected = False
    monitor.drop()


def reconnect(aioca: FakeAioca, name: str = NAME, **ctrl_kwargs: Any) -> None:
    """The channel comes back: libca re-arms the CTRL monitor, which fires first."""
    aioca.default.connected = True
    ctrl, _ = aioca.monitors_of(name)
    ctrl.fire(ctrl_reading(**ctrl_kwargs))


def count_closes(monitors: list[FakeSubscription]) -> list[str]:
    """Record every ``close()`` of ``monitors`` (by name) in the returned list."""
    closes: list[str] = []
    for monitor in monitors:
        original = monitor.close

        def counted(_original=original, _monitor=monitor):
            closes.append(_monitor.name)
            _original()

        monitor.close = counted
    return closes


@pytest.fixture
def utc(monkeypatch):
    monkeypatch.setattr(
        "osprey_connectors.control_system._epics_connector.get_facility_timezone",
        lambda: zoneinfo.ZoneInfo("UTC"),
    )


@pytest.fixture
def bulk_logger(monkeypatch):
    """The _epics_bulk logger, replaced so a test can count its warnings."""
    fake = MagicMock(name="_epics_bulk.logger")
    monkeypatch.setattr(_epics_bulk, "logger", fake)
    return fake


# ---------------------------------------------------------------------------
# Unit: the monitors, CTRL then TIME
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("utc")
class TestMonitors:
    async def test_x_and_ca_scheme_x_share_one_record_and_two_monitors(self):
        connector, aioca = bulk_connector()
        a: list = []
        b: list = []
        await connector.subscribe(NAME, a.append)
        await connector.subscribe(f"ca://{NAME}", b.append)
        ctrl, time_ = aioca.monitors_of(NAME)
        ctrl.fire(ctrl_reading())
        time_.fire(time_reading(3.0))
        await drain()

        assert [m.name for m in aioca.monitors] == [NAME, NAME]
        assert [v.value for v in a] == [3.0]
        assert [v.value for v in b] == [3.0]

    async def test_the_monitors_are_opened_ctrl_first_with_their_own_events_and_formats(self):
        connector, aioca = bulk_connector()

        await connector.subscribe(NAME, lambda v: None)

        ctrl, time_ = aioca.monitors
        assert ctrl.kwargs == {
            "events": aioca.DBE_PROPERTY,
            "format": aioca.FORMAT_CTRL,
            "count": 1,
            "all_updates": True,
            "notify_disconnect": True,
        }
        assert time_.kwargs["events"] == aioca.DBE_VALUE | aioca.DBE_ALARM
        assert time_.kwargs["format"] == aioca.FORMAT_TIME
        assert time_.kwargs["all_updates"] is True
        assert time_.kwargs["notify_disconnect"] is True

    async def test_values_carry_the_ctrl_metadata_and_the_time_alarm(self):
        connector, aioca = bulk_connector()
        _, received = await live(connector, aioca)

        aioca.monitors_of(NAME)[1].fire(
            time_reading(7.0, status=4, severity=1, timestamp=1_750_000_000.0)
        )
        await drain()

        [value] = received
        assert value.value == 7.0
        assert value.timestamp.timestamp() == 1_750_000_000.0
        meta = value.metadata
        assert (meta.units, meta.precision) == ("mA", 2)
        assert (meta.display_low, meta.display_high) == (0.0, 500.0)
        assert (meta.control_low, meta.control_high) == (0.0, 450.0)
        assert (meta.warning_low, meta.warning_high) == (10.0, 400.0)
        assert (meta.alarm_low, meta.alarm_high) == (5.0, 420.0)
        assert meta.writable is True
        assert (meta.alarm_status, meta.alarm_severity) == ("HIGH", 1)
        assert (meta.raw_metadata["status"], meta.raw_metadata["severity"]) == (4, 1)

    async def test_write_access_comes_from_the_channel(self):
        connector, aioca = bulk_connector(channel=FakeChannel(write=False))
        _, received = await live(connector, aioca)

        aioca.monitors_of(NAME)[1].fire(time_reading(1.0))
        await drain()

        assert received[0].metadata.writable is False

    async def test_values_are_posted_never_delivered_inside_the_monitor_callback(self):
        connector, aioca = bulk_connector()
        _, received = await live(connector, aioca)

        aioca.monitors_of(NAME)[1].fire(time_reading(1.0))
        assert received == []
        await drain()

        assert [v.value for v in received] == [1.0]

    async def test_enum_values_carry_their_label(self):
        connector, aioca = bulk_connector()
        received: list[ChannelValue] = []
        await connector.subscribe("SR:MODE", received.append)
        ctrl, time_ = aioca.monitors_of("SR:MODE")
        ctrl.fire(ca_reading(0, datatype=DBR_ENUM, enums=["OFFLINE", "STANDBY", "FAULT"]))
        time_.fire(time_reading(2, datatype=DBR_ENUM))
        await drain()

        assert received[0].value == 2
        assert received[0].metadata.enum_label == "FAULT"
        assert received[0].metadata.enum_labels == ["OFFLINE", "STANDBY", "FAULT"]
        assert received[0].metadata.alarm_status == "NO_ALARM"

    async def test_an_enum_without_labels_still_delivers_the_index(self):
        connector, aioca = bulk_connector()
        received: list[ChannelValue] = []
        await connector.subscribe("SR:MODE", received.append)
        ctrl, time_ = aioca.monitors_of("SR:MODE")
        ctrl.fire(ca_reading(0, datatype=DBR_ENUM))
        time_.fire(time_reading(1, datatype=DBR_ENUM))
        await drain()

        assert received[0].value == 1
        assert received[0].metadata.enum_label is None
        assert received[0].metadata.enum_labels is None

    async def test_event_counts_report_every_ctrl_and_time_event(self):
        connector, aioca = bulk_connector()
        await live(connector, aioca)
        ctrl, time_ = aioca.monitors_of(NAME)
        time_.fire(time_reading(1.0))
        time_.fire(time_reading(2.0))
        ctrl.fire(ctrl_reading(units="A"))  # a property change
        await drain()

        assert event_counts(connector, NAME) == {"ctrl": 2, "time": 2}
        assert event_counts(connector, f"ca://{NAME}") == {"ctrl": 2, "time": 2}
        assert event_counts(connector, "SR:OTHER") == {"ctrl": 0, "time": 0}


# ---------------------------------------------------------------------------
# Unit: ordering within a connection
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("utc")
class TestOrdering:
    async def test_up_is_reported_before_the_first_value(self):
        connector, aioca = bulk_connector()
        events: list = []
        await connector.subscribe(
            NAME, lambda v: events.append(v.value), on_connection=events.append
        )
        ctrl, time_ = aioca.monitors_of(NAME)

        ctrl.fire(ctrl_reading())
        time_.fire(time_reading(1.0))
        await drain()

        assert events == [True, 1.0]

    async def test_a_time_before_ctrl_is_held_and_delivered_after_it_with_that_ctrl(self):
        connector, aioca = bulk_connector()
        events: list = []
        await connector.subscribe(NAME, events.append, on_connection=events.append)
        ctrl, time_ = aioca.monitors_of(NAME)

        time_.fire(time_reading(1.0))
        time_.fire(time_reading(2.0))  # only the latest held one is delivered
        await drain()
        assert events == []

        ctrl.fire(ctrl_reading(units="kV"))
        await drain()

        assert events[0] is True
        [value] = events[1:]
        assert value.value == 2.0
        assert value.metadata.units == "kV"
        assert event_counts(connector, NAME) == {"ctrl": 1, "time": 2}

    async def test_a_held_time_warns_once_per_connection_when_no_ctrl_arrives(self, bulk_logger):
        connector, aioca = bulk_connector(timeout=0.05)
        await connector.subscribe(NAME, lambda v: None)
        _, time_ = aioca.monitors_of(NAME)

        time_.fire(time_reading(1.0))
        time_.fire(time_reading(2.0))
        await asyncio.sleep(0.15)
        time_.fire(time_reading(3.0))
        await asyncio.sleep(0.15)

        warnings = [c for c in bulk_logger.warning.call_args_list if "no control data" in c[0][0]]
        assert len(warnings) == 1
        assert NAME in warnings[0][0]

    async def test_a_ctrl_event_cancels_the_held_warning(self, bulk_logger):
        connector, aioca = bulk_connector(timeout=0.05)
        await connector.subscribe(NAME, lambda v: None)
        ctrl, time_ = aioca.monitors_of(NAME)

        time_.fire(time_reading(1.0))
        ctrl.fire(ctrl_reading())
        await asyncio.sleep(0.15)

        bulk_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# Unit: a channel that drops
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("utc")
class TestDrops:
    async def test_a_real_drop_reports_down_and_clears_the_ctrl(self):
        connector, aioca = bulk_connector()
        events: list = []
        await connector.subscribe(NAME, events.append, on_connection=events.append)
        ctrl, time_ = aioca.monitors_of(NAME)
        ctrl.fire(ctrl_reading(units="mA"))
        time_.fire(time_reading(1.0))
        await drain()

        drop(aioca, time_)
        await drain()
        assert events[-1] is False

        # The next connection's first TIME arrives before its CTRL: it is held,
        # not delivered with the previous connection's control data.
        aioca.default.connected = True
        time_.fire(time_reading(2.0))
        await drain()
        assert len(events) == 3
        ctrl.fire(ctrl_reading(units="A"))
        await drain()

        assert [e if isinstance(e, bool) else e.value for e in events] == [
            True,
            1.0,
            False,
            True,
            2.0,
        ]
        assert events[-1].metadata.units == "A"
        assert len(aioca.monitors) == 2  # libca re-armed the pair; nothing re-opened

    async def test_a_drop_from_both_monitors_is_reported_once(self):
        connector, aioca = bulk_connector()
        states: list = []
        await live(connector, aioca, on_connection=states.append)
        ctrl, time_ = aioca.monitors_of(NAME)

        drop(aioca, ctrl)
        drop(aioca, time_)
        await drain()

        assert states == [True, False]

    async def test_a_stale_drop_is_ignored_while_the_channel_is_connected(self):
        connector, aioca = bulk_connector()
        events: list = []
        await connector.subscribe(NAME, events.append, on_connection=events.append)
        ctrl, time_ = aioca.monitors_of(NAME)
        ctrl.fire(ctrl_reading(units="mA"))

        # [CTRL_new, CANothing]: the drop reaches the loop after the next
        # connection's CTRL event, with the channel already up again.
        ctrl.fire(ctrl_reading(units="kV"))
        time_.drop()
        time_.fire(time_reading(5.0))
        await drain()

        assert events[0] is True
        assert False not in events
        [value] = events[1:]
        assert (value.value, value.metadata.units) == (5.0, "kV")

    async def test_a_drop_before_the_first_connection_posts_nothing(self):
        connector, aioca = bulk_connector()
        states: list = []
        await connector.subscribe(NAME, lambda v: None, on_connection=states.append)

        drop(aioca, aioca.monitors_of(NAME)[0])
        await drain()

        assert states == []

    async def test_a_monitor_aioca_closed_is_reopened_once_per_connection(self, bulk_logger):
        connector, aioca = bulk_connector()
        events: list = []
        await live(connector, aioca, on_connection=events.append)
        ctrl, time_ = aioca.monitors_of(NAME)

        time_.state = FakeSubscription.CLOSED  # aioca closed it on its own
        drop(aioca, ctrl)
        await drain()
        assert len(aioca.monitors) == 4
        assert ctrl.closed and time_.closed

        # A second closure in the same connection is not re-opened again.
        ctrl2, time2 = aioca.monitors_of(NAME)
        time2.state = FakeSubscription.CLOSED
        drop(aioca, ctrl2)
        await drain()
        assert len(aioca.monitors) == 4

        # The next connection's CTRL event re-arms the one re-open, and uses it
        # at once on the TIME monitor still closed.
        reconnect(aioca)
        await drain()

        assert len(aioca.monitors) == 6
        assert ctrl2.closed and time2.closed
        assert events == [True, False, True]
        reopen_warnings = [c for c in bulk_logger.warning.call_args_list if "re-opening" in c[0][0]]
        assert len(reopen_warnings) == 2

    async def test_the_re_open_waits_until_the_drop_has_been_dispatched(self):
        """aioca delivers a drop while walking the channel's set of subscriptions,
        and closing one removes it from that set: the re-open must come after."""
        connector, aioca = bulk_connector()
        await live(connector, aioca)
        ctrl, time_ = aioca.monitors_of(NAME)
        channel_subscriptions = {ctrl, time_}
        for monitor in (ctrl, time_):
            plain_close = monitor.close

            def close(monitor=monitor, plain_close=plain_close):
                channel_subscriptions.discard(monitor)
                plain_close()

            monitor.close = close
        time_.state = FakeSubscription.CLOSED  # aioca closed it on its own

        aioca.default.connected = False
        for monitor in channel_subscriptions:  # raises if the set changes under it
            monitor.drop()
        assert len(aioca.monitors) == 2
        await drain()

        assert len(aioca.monitors) == 4
        assert ctrl.closed and time_.closed

    async def test_a_second_closure_in_one_connection_warns_once(self, bulk_logger):
        connector, aioca = bulk_connector()
        await live(connector, aioca)
        for _ in range(3):
            ctrl, time_ = aioca.monitors_of(NAME)
            time_.state = FakeSubscription.CLOSED
            drop(aioca, ctrl)
            await drain()

        messages = [c[0][0] for c in bulk_logger.warning.call_args_list]
        assert len([m for m in messages if "re-opening" in m]) == 1
        assert len([m for m in messages if "closed again" in m]) == 1

    async def test_a_second_closure_without_a_drop_reports_the_record_down(self):
        connector, aioca = bulk_connector()
        events: list = []
        await live(connector, aioca, on_connection=events.append)
        ctrl, time_ = aioca.monitors_of(NAME)
        time_.state = FakeSubscription.CLOSED
        ctrl.fire(ctrl_reading())  # an event still arrives: the one re-open
        await drain()
        ctrl2, time2 = aioca.monitors_of(NAME)
        ctrl2.fire(ctrl_reading())
        await drain()
        assert events == [True]

        time2.state = FakeSubscription.CLOSED
        ctrl2.fire(ctrl_reading())  # a second closure in the same connection
        await drain()

        assert events == [True, False], "a record whose monitor cannot deliver is down"


def failing_camonitor(monkeypatch, aioca: FakeAioca, *, fail_on: set[int]) -> list[int]:
    """Make the ``n``-th ``camonitor`` call from now raise, for each ``n`` in ``fail_on``.

    Returns the running call count, so a test can clear ``fail_on`` to heal it.
    """
    calls = [0]
    plain = aioca.camonitor

    def camonitor(*args, **kwargs):
        calls[0] += 1
        if calls[0] in fail_on:
            raise RuntimeError("camonitor failed")
        return plain(*args, **kwargs)

    monkeypatch.setattr(aioca, "camonitor", camonitor)
    return calls


@pytest.mark.usefixtures("utc")
class TestFailedReopen:
    """A re-open that fails leaves the record down and re-openable, never wedged."""

    async def _dead(self, monkeypatch, connector, aioca, events):
        """Bring the record up, then have aioca close a monitor and the re-open fail."""
        await live(connector, aioca, on_connection=events.append)
        fail_on = {2}  # the re-open's CTRL opens, its TIME raises
        failing_camonitor(monkeypatch, aioca, fail_on=fail_on)
        ctrl, time_ = aioca.monitors_of(NAME)
        time_.state = FakeSubscription.CLOSED
        drop(aioca, ctrl)
        await drain()
        return fail_on

    async def test_both_handles_are_closed_and_subscribers_told_down_once(
        self, monkeypatch, bulk_logger
    ):
        connector, aioca = bulk_connector()
        events: list = []
        await self._dead(monkeypatch, connector, aioca, events)

        assert all(m.closed for m in aioca.monitors), "the half-open CTRL is closed too"
        assert aioca.monitors_of(NAME) == (None, None)
        assert events == [True, False]
        assert bulk_logger.error.call_count == 1

    async def test_a_later_subscriber_reopens_it(self, monkeypatch):
        connector, aioca = bulk_connector()
        first: list = []
        fail_on = await self._dead(monkeypatch, connector, aioca, first)
        fail_on.clear()

        joiner: list = []
        await connector.subscribe(NAME, joiner.append, on_connection=joiner.append)
        ctrl, time_ = aioca.monitors_of(NAME)
        assert ctrl is not None and time_ is not None
        reconnect(aioca)
        time_.fire(time_reading(7.0))
        await drain()

        assert first == [True, False, True], "the first subscriber is told up again"
        assert joiner[0] is True and joiner[1].value == 7.0

    async def test_a_subscriber_whose_reopen_fails_is_told_and_the_next_one_retries(
        self, monkeypatch
    ):
        connector, aioca = bulk_connector()
        fail_on = await self._dead(monkeypatch, connector, aioca, [])
        fail_on.update({3})  # the joiner's CTRL open raises

        with pytest.raises(RuntimeError, match="camonitor failed"):
            await connector.subscribe(NAME, lambda v: None)
        assert aioca.monitors_of(NAME) == (None, None)

        fail_on.clear()
        await connector.subscribe(NAME, lambda v: None)
        assert all(m is not None for m in aioca.monitors_of(NAME))

    async def test_get_metadata_falls_back_to_a_read(self, monkeypatch):
        connector, aioca = bulk_connector(timeout=5.0)
        expected = ChannelMetadata(units="kV")
        read = AsyncMock(return_value=ChannelValue(value=1.0, timestamp=None, metadata=expected))
        monkeypatch.setattr(connector, "read_channel", read)
        await self._dead(monkeypatch, connector, aioca, [])

        assert await asyncio.wait_for(connector.get_metadata(NAME), 1.0) is expected
        read.assert_awaited_once_with(NAME)

    async def test_a_waiter_pending_when_the_reopen_fails_falls_back_to_a_read(self, monkeypatch):
        connector, aioca = bulk_connector(timeout=5.0)
        expected = ChannelMetadata(units="kV")
        read = AsyncMock(return_value=ChannelValue(value=1.0, timestamp=None, metadata=expected))
        monkeypatch.setattr(connector, "read_channel", read)
        await connector.subscribe(NAME, lambda v: None)  # down: waiters wait for CTRL
        pending = asyncio.ensure_future(connector.get_metadata(NAME))
        await drain()
        assert not pending.done()

        failing_camonitor(monkeypatch, aioca, fail_on={1})
        connector._ca.repoint()  # its re-open fails

        assert await asyncio.wait_for(pending, 1.0) is expected

    async def test_a_failed_repoint_reopen_is_retried_by_the_next_repoint(
        self, monkeypatch, bulk_logger
    ):
        connector, aioca = bulk_connector()
        events: list = []
        await live(connector, aioca, on_connection=events.append)
        fail_on = {1}
        failing_camonitor(monkeypatch, aioca, fail_on=fail_on)

        connector._ca.repoint()
        record = connector._bulk._records[NAME]
        assert (record.ctrl_monitor, record.time_monitor) == (None, None)
        assert bulk_logger.error.call_count == 1

        fail_on.clear()
        connector._ca.repoint()
        reconnect(aioca)
        await drain()
        assert all(m is not None for m in aioca.monitors_of(NAME))
        assert events == [True, False, True]


# ---------------------------------------------------------------------------
# Unit: subscribers joining and leaving
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("utc")
class TestRecords:
    async def test_a_late_joiner_is_told_up_and_given_the_last_value(self):
        connector, aioca = bulk_connector()
        await live(connector, aioca)
        aioca.monitors_of(NAME)[1].fire(time_reading(4.0))
        await drain()

        events: list = []
        await connector.subscribe(
            NAME, lambda v: events.append(v.value), on_connection=events.append
        )
        assert events == []  # posted, not inline
        await drain()

        assert events == [True, 4.0]
        assert len(aioca.monitors) == 2  # no new monitor for the joiner

    async def test_a_joiner_before_any_value_is_told_up_only(self):
        connector, aioca = bulk_connector()
        await live(connector, aioca)

        events: list = []
        await connector.subscribe(NAME, events.append, on_connection=events.append)
        await drain()

        assert events == [True]

    async def test_a_joiner_on_a_record_that_is_down_is_told_nothing_yet(self):
        connector, aioca = bulk_connector()
        await connector.subscribe(NAME, lambda v: None)

        events: list = []
        await connector.subscribe(NAME, events.append, on_connection=events.append)
        await drain()

        assert events == []

    async def test_two_subscribers_each_get_every_value_once(self):
        connector, aioca = bulk_connector()
        first: list = []
        second: list = []
        await connector.subscribe(NAME, lambda v: first.append(v.value))
        await connector.subscribe(NAME, lambda v: second.append(v.value))
        ctrl, time_ = aioca.monitors_of(NAME)
        ctrl.fire(ctrl_reading())
        time_.fire(time_reading(1.0))
        time_.fire(time_reading(2.0))
        await drain()

        assert first == [1.0, 2.0]
        assert second == [1.0, 2.0]

    async def test_one_subscriber_leaving_keeps_the_monitors(self):
        connector, aioca = bulk_connector()
        sub_id, _ = await live(connector, aioca)
        others: list = []
        await connector.subscribe(NAME, lambda v: others.append(v.value))

        await connector.unsubscribe(sub_id)
        aioca.monitors_of(NAME)[1].fire(time_reading(3.0))
        await drain()

        assert not any(m.closed for m in aioca.monitors)
        assert others == [3.0]

    async def test_the_last_subscriber_leaving_closes_both_monitors_and_the_reopener(self):
        connector, aioca = bulk_connector()
        sub_id, _ = await live(connector, aioca)
        closes = count_closes(aioca.monitors)
        assert len(connector._ca.reopeners) == 1

        await connector.unsubscribe(sub_id)

        assert closes == [NAME, NAME]
        assert connector._ca.reopeners == []
        assert sub_id not in connector._subscriptions
        assert event_counts(connector, NAME) == {"ctrl": 0, "time": 0}

    async def test_subscribe_unsubscribe_subscribe_opens_a_fresh_pair(self):
        connector, aioca = bulk_connector()
        sub_id, _ = await live(connector, aioca)
        await connector.unsubscribe(sub_id)

        received: list = []
        await connector.subscribe(NAME, received.append)
        await drain()
        assert received == []  # the fresh record waits for its own CTRL event
        ctrl, time_ = aioca.monitors_of(NAME)
        ctrl.fire(ctrl_reading())
        time_.fire(time_reading(9.0))
        await drain()

        assert [v.value for v in received] == [9.0]
        assert len(aioca.monitors) == 4

    async def test_unsubscribe_unknown_id_is_a_noop(self):
        connector, _ = bulk_connector()
        await connector.unsubscribe("never-registered")

    async def test_unsubscribe_twice_is_a_noop(self):
        connector, aioca = bulk_connector()
        sub_id, _ = await live(connector, aioca)
        closes = count_closes(aioca.monitors)

        await connector.unsubscribe(sub_id)
        await connector.unsubscribe(sub_id)

        assert closes == [NAME, NAME]

    async def test_a_raising_value_factory_does_not_kill_the_record(self, monkeypatch, bulk_logger):
        connector, aioca = bulk_connector()
        real = EPICSConnector._ca_channel_value.__get__(connector)
        calls = iter([True])

        def flaky(reading, ctrl, writable=None):
            if next(calls, False):
                raise ValueError("unbuildable reading")
            return real(reading, ctrl, writable)

        monkeypatch.setattr(connector, "_ca_channel_value", flaky)
        _, received = await live(connector, aioca)
        _, time_ = aioca.monitors_of(NAME)

        time_.fire(time_reading(1.0))
        time_.fire(time_reading(2.0))
        await drain()

        assert [v.value for v in received] == [2.0]
        assert not any(m.closed for m in aioca.monitors)
        bulk_logger.warning.assert_called_once()

    async def test_a_raising_connection_callback_does_not_break_the_subscription(self):
        connector, aioca = bulk_connector()

        def boom(_up):
            raise RuntimeError("subscriber bug")

        _, received = await live(connector, aioca, on_connection=boom)
        aioca.monitors_of(NAME)[1].fire(time_reading(1.0))
        await drain()

        assert [v.value for v in received] == [1.0]


# ---------------------------------------------------------------------------
# Unit: a cancelled subscribe
# ---------------------------------------------------------------------------


class QueuedCaLoop(InlineCaLoop):
    """A CA loop that runs each coroutine as a task of its own and queues ``call_soon``.

    Work handed over is queued in order, as on the real loop thread. With
    ``shielded`` a cancelled caller leaves its coroutine running (the join was
    already queued or done when the cancel came); without it the cancel reaches
    the coroutine before it starts.
    """

    def __init__(self, *, shielded: bool) -> None:
        super().__init__()
        self.shielded = shielded

    async def run(self, coro):
        task = asyncio.ensure_future(coro)
        if self.shielded:
            return await asyncio.shield(task)
        return await task

    def call_soon(self, callback, *args):
        asyncio.get_running_loop().call_soon(callback, *args)


@pytest.mark.usefixtures("utc")
class TestCancelledSubscribe:
    @pytest.mark.parametrize("shielded", [True, False], ids=["join-ran", "join-never-ran"])
    async def test_a_cancel_during_subscribe_leaves_no_subscriber(self, shielded):
        connector, aioca = bulk_connector()
        connector._ca = QueuedCaLoop(shielded=shielded)
        received: list = []

        task = asyncio.ensure_future(connector.subscribe(NAME, received.append))
        await asyncio.sleep(0)  # the join is queued on the CA loop
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await drain()

        assert connector._subscriptions == {}
        assert all(m.closed for m in aioca.monitors)
        assert connector._ca.reopeners == []
        assert event_counts(connector, NAME) == {"ctrl": 0, "time": 0}
        if shielded:
            assert len(aioca.monitors) == 2  # opened by the join, closed by the release
        else:
            assert aioca.monitors == []


# ---------------------------------------------------------------------------
# Unit: a subscriber whose loop has closed
# ---------------------------------------------------------------------------


@pytest.fixture
def closed_loop():
    loop = asyncio.new_event_loop()
    loop.close()
    return loop


@pytest.mark.usefixtures("utc")
class TestClosedSubscriberLoop:
    async def test_a_subscriber_whose_loop_closed_is_dropped_and_the_rest_keep_receiving(
        self, closed_loop
    ):
        connector, aioca = bulk_connector()
        _, received = await live(connector, aioca)
        bulk = connector._bulk
        await bulk.subscribe(NAME, "gone", lambda v: None, closed_loop, None)
        _, time_ = aioca.monitors_of(NAME)

        time_.fire(time_reading(1.0))
        await drain()
        time_.fire(time_reading(2.0))
        await drain()

        assert [v.value for v in received] == [1.0, 2.0]
        assert "gone" not in bulk._records[NAME].subscribers
        assert not any(m.closed for m in aioca.monitors)

    async def test_the_last_closed_loop_subscriber_closes_the_monitors_once_and_never_reopens(
        self, closed_loop, bulk_logger
    ):
        connector, aioca = bulk_connector()
        bulk = connector._ca_bulk()
        await bulk.subscribe(NAME, "gone", lambda v: None, closed_loop, None)
        ctrl, time_ = aioca.monitors_of(NAME)
        closes = count_closes(aioca.monitors)
        ctrl.fire(ctrl_reading())

        # Two TIME events queued before the deferred removal runs.
        time_.fire(time_reading(1.0))
        time_.fire(time_reading(2.0))
        assert closes == []  # never removed inside the monitor callback
        await drain()

        assert closes == [NAME, NAME]
        assert connector._ca.reopeners == []
        connector._ca.repoint()
        assert len(aioca.monitors) == 2
        bulk_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# Unit: a repoint
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("utc")
class TestRepoint:
    async def test_a_repoint_reports_down_and_opens_a_fresh_pair(self):
        connector, aioca = bulk_connector()
        events: list = []
        await connector.subscribe(NAME, events.append, on_connection=events.append)
        old_ctrl, old_time = aioca.monitors_of(NAME)
        old_ctrl.fire(ctrl_reading())
        old_time.fire(time_reading(1.0))
        await drain()

        connector._ca.repoint()
        await drain()
        assert events[-1] is False
        new_ctrl, new_time = aioca.monitors_of(NAME)
        assert len(aioca.monitors) == 4
        assert (new_ctrl.kwargs, new_time.kwargs) == (old_ctrl.kwargs, old_time.kwargs)

        new_ctrl.fire(ctrl_reading(units="kV"))
        new_time.fire(time_reading(2.0))
        await drain()

        assert [e if isinstance(e, bool) else e.value for e in events] == [
            True,
            1.0,
            False,
            True,
            2.0,
        ]
        assert events[-1].metadata.units == "kV"
        assert len(connector._ca.reopeners) == 1

    async def test_a_repoint_settles_a_metadata_waiter_and_get_metadata_reads(self, monkeypatch):
        connector, _ = bulk_connector(timeout=5.0)
        expected = ChannelMetadata(units="kV")
        read = AsyncMock(return_value=ChannelValue(value=1.0, timestamp=None, metadata=expected))
        monkeypatch.setattr(connector, "read_channel", read)
        await connector.subscribe(NAME, lambda v: None)
        pending = asyncio.ensure_future(connector.get_metadata(NAME))
        await drain()

        connector._ca.repoint()

        assert await asyncio.wait_for(pending, 1.0) is expected
        read.assert_awaited_once_with(NAME)


# ---------------------------------------------------------------------------
# Unit: get_metadata of a subscribed address
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("utc")
class TestMetadata:
    async def test_metadata_awaits_the_ctrl_event(self):
        connector, aioca = bulk_connector()
        await connector.subscribe(NAME, lambda v: None)
        pending = asyncio.ensure_future(connector.get_metadata(NAME))
        await drain()
        assert not pending.done()

        aioca.monitors_of(NAME)[0].fire(ctrl_reading())
        metadata = await asyncio.wait_for(pending, 1.0)

        assert (metadata.units, metadata.precision) == ("mA", 2)
        assert (metadata.alarm_low, metadata.alarm_high) == (5.0, 420.0)
        assert metadata.writable is True

    async def test_metadata_after_a_value_is_that_values(self):
        connector, aioca = bulk_connector()
        await live(connector, aioca)
        aioca.monitors_of(NAME)[1].fire(time_reading(1.0, status=5, severity=2))
        await drain()

        metadata = await connector.get_metadata(f"ca://{NAME}")

        assert metadata.alarm_status == "LOLO"
        assert metadata.units == "mA"

    async def test_metadata_of_a_subscribed_address_reads_nothing(self, monkeypatch):
        connector, aioca = bulk_connector()
        await live(connector, aioca)
        read = AsyncMock()
        monkeypatch.setattr(connector, "read_channel", read)
        calls = len(aioca.calls)

        await connector.get_metadata(NAME)

        read.assert_not_awaited()
        assert len(aioca.calls) == calls

    async def test_no_ctrl_event_within_the_timeout_is_the_channel_unreachable(self):
        connector, _ = bulk_connector(timeout=0.05)
        await connector.subscribe(NAME, lambda v: None)

        with pytest.raises(ChannelUnreachableError) as excinfo:
            await connector.get_metadata(NAME)

        assert NAME in str(excinfo.value)

    async def test_an_unsubscribed_address_is_read_once(self, monkeypatch):
        connector, _ = bulk_connector()
        expected = ChannelMetadata(units="kV")
        read = AsyncMock(return_value=ChannelValue(value=1.0, timestamp=None, metadata=expected))
        monkeypatch.setattr(connector, "read_channel", read)

        assert await connector.get_metadata(NAME) is expected
        read.assert_awaited_once_with(NAME)

    async def test_the_last_subscriber_leaving_while_waiting_falls_back_to_a_read(
        self, monkeypatch
    ):
        connector, _ = bulk_connector(timeout=5.0)
        expected = ChannelMetadata(units="kV")
        read = AsyncMock(return_value=ChannelValue(value=1.0, timestamp=None, metadata=expected))
        monkeypatch.setattr(connector, "read_channel", read)
        sub_id = await connector.subscribe(NAME, lambda v: None)
        pending = asyncio.ensure_future(connector.get_metadata(NAME))
        await drain()

        await connector.unsubscribe(sub_id)

        assert await asyncio.wait_for(pending, 1.0) is expected


# ---------------------------------------------------------------------------
# Unit: warming the context
# ---------------------------------------------------------------------------


class TestWarm:
    async def test_warm_creates_the_context_and_opens_no_channel(self):
        connector, aioca = bulk_connector()
        asked: list = []
        aioca.get_channel_infos = lambda: asked.append(True) or []

        await connector.warm()

        assert asked == [True]
        assert aioca.calls == []
        assert aioca.monitors == []

    async def test_warm_before_connect_is_a_noop(self):
        await EPICSConnector().warm()


# ---------------------------------------------------------------------------
# Wire: a soft IOC and a client in its own process
# ---------------------------------------------------------------------------


def _records():
    from osprey_connectors_tests._soft_ioc import ai_record, enum_record

    return [
        # Outside HIHI, so it is delivered in a MAJOR alarm.
        ai_record(ALARMED, 12.5, alarm=(-10.0, 10.0), warning=(-5.0, 5.0)),
        ai_record(PLAIN, 2.5),
        enum_record(MODE, ["Off", "On", "Fault"], value=1),
    ]


@pytest.fixture
def ioc(tmp_path):
    from osprey_connectors_tests._soft_ioc import SoftIOC

    with SoftIOC(_records(), workdir=tmp_path) as running:
        yield running


def _client_env(ioc, *, gateway: bool) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("EPICS_")}
    # The gateway shape gets its name server from connect(); UDP from the environment.
    env.update(ioc.client_env(name_server=False))
    if gateway:
        env.pop("EPICS_CA_ADDR_LIST", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO), str(TESTS_ROOT), env.get("PYTHONPATH")])
    )
    return env


def run_client(ioc, scenario: str, *, gateway: bool, timeout: float = 90.0) -> dict[str, Any]:
    """Run one client scenario against ``ioc``; serve its kill/restart requests."""
    from osprey_connectors_tests._soft_ioc import kill_group

    args = {"port": ioc.ports.ca_server, "gateway": gateway}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "osprey_connectors_tests.test_ca_bulk_path",
            scenario,
            json.dumps(args),
        ],
        cwd=REPO,
        env=_client_env(ioc, gateway=gateway),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()
    stderr: list[str] = []

    def pump(stream, sink) -> None:
        for line in stream:
            sink(line)
        if sink is lines.put:
            lines.put(None)

    threading.Thread(target=pump, args=(proc.stdout, lines.put), daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, stderr.append), daemon=True).start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"client {scenario!r} timed out; stderr:\n{''.join(stderr)}")
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                proc.wait(timeout=10)
                raise AssertionError(
                    f"client {scenario!r} exited {proc.returncode} without a result; "
                    f"stderr:\n{''.join(stderr)}"
                )
            if not line.startswith("{"):
                continue
            message = json.loads(line)
            if "cmd" in message:
                getattr(ioc, message["cmd"])()
                proc.stdin.write("ok\n")
                proc.stdin.flush()
            elif "result" in message:
                proc.wait(timeout=10)
                assert proc.returncode == 0, f"client exited {proc.returncode}: {''.join(stderr)}"
                return message["result"]
    finally:
        kill_group(proc)


class TestOverTheWire:
    @pytest.mark.parametrize("gateway", [False, True], ids=["udp", "gateway"])
    def test_values_limits_labels_and_one_record_per_address(self, ioc, gateway):
        result = run_client(ioc, "shapes", gateway=gateway)

        alarmed = result["values"][ALARMED]
        assert alarmed["value"] == 12.5
        assert (alarmed["alarm_low"], alarmed["alarm_high"]) == (-10.0, 10.0)
        assert (alarmed["warning_low"], alarmed["warning_high"]) == (-5.0, 5.0)
        assert (alarmed["display_low"], alarmed["display_high"]) == (-100.0, 100.0)
        assert (alarmed["units"], alarmed["precision"]) == ("mm", 3)
        assert (alarmed["alarm_status"], alarmed["alarm_severity"]) == ("HIHI", 2)
        assert alarmed["writable"] is True
        plain = result["values"][PLAIN]
        assert plain["value"] == 2.5
        assert (plain["alarm_low"], plain["alarm_high"]) == (None, None)
        assert (plain["warning_low"], plain["warning_high"]) == (None, None)
        mode = result["values"][MODE]
        assert (mode["value"], mode["enum_label"]) == (1, "On")
        assert mode["enum_labels"] == ["Off", "On", "Fault"]

        for name in (ALARMED, PLAIN, MODE):
            assert result["counts"][name] == {"ctrl": 1, "time": 1}, name
            assert result["connection"][name] == [True], name
        assert result["metadata"]["alarm_high"] == 10.0
        assert result["records"] == 3  # MODE was subscribed as ca://MODE and MODE

    def test_ioc_kill_and_restart_keeps_one_pair_and_delivers_each_value_once(self, ioc):
        result = run_client(ioc, "restart", gateway=False)

        assert result["events"] == [True, 12.5, False, True, 12.5]
        assert result["counts"] == {"ctrl": 2, "time": 2}
        assert result["same_monitors"] is True  # libca re-armed them; nothing re-opened

    def test_subscribe_unsubscribe_subscribe_and_two_subscribers(self, ioc):
        result = run_client(ioc, "resubscribe", gateway=False)

        assert result["first"] == [2.5]
        assert result["again"] == [2.5]
        assert result["joiner"] == [True, 2.5]
        assert result["again_after_join"] == [2.5]  # the joiner caused no second value
        assert result["counts"] == {"ctrl": 1, "time": 1}  # the fresh record's own events

    def test_a_second_connector_on_another_route_repoints_the_first_and_it_comes_back(self, ioc):
        result = run_client(ioc, "repoint", gateway=True)

        assert result["first_events"] == [True, 12.5, False, True, 12.5]
        assert result["one_shot"] == 2.5
        assert result["second_values"] == [12.5]
        assert result["teardown"] == "ok"


# ---------------------------------------------------------------------------
# The client process (python -m osprey_connectors_tests.test_ca_bulk_path <scenario>)
# ---------------------------------------------------------------------------


async def _until(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.05)


async def _ask(cmd: str) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    await asyncio.to_thread(sys.stdin.readline)


async def _connected(args: dict[str, Any], *, name_server: bool = True) -> EPICSConnector:
    """A connector on the scenario's shape; ``name_server`` picks the gateway route."""
    config: dict[str, Any] = {"timeout_s": 3.0}
    if args["gateway"]:
        config["gateways"] = {
            "read_only": {
                "address": "127.0.0.1",
                "port": args["port"],
                "use_name_server": name_server,
            }
        }
    connector = EPICSConnector()
    await connector.connect(config)
    return connector


async def _on_ca_loop(connector: EPICSConnector, fn):
    """``fn()`` on the CA loop, where the subscription records live."""

    async def call():
        return fn()

    return await connector._ca.run(call())


def _value_json(value: ChannelValue) -> dict[str, Any]:
    meta = value.metadata
    out = {"value": value.value}
    for field in (
        "units",
        "precision",
        "alarm_status",
        "alarm_severity",
        "enum_label",
        "enum_labels",
        "display_low",
        "display_high",
        "warning_low",
        "warning_high",
        "alarm_low",
        "alarm_high",
        "writable",
    ):
        out[field] = getattr(meta, field)
    return out


async def _scenario_shapes(args):
    connector = await _connected(args)
    values: dict[str, list] = {ALARMED: [], PLAIN: [], MODE: []}
    connection: dict[str, list] = {ALARMED: [], PLAIN: [], MODE: []}
    for name, address in ((ALARMED, ALARMED), (PLAIN, PLAIN), (MODE, f"ca://{MODE}")):
        await connector.subscribe(
            address, values[name].append, on_connection=connection[name].append
        )
    await connector.subscribe(MODE, lambda v: None)
    await _until(lambda: all(values.values()))
    metadata = await connector.get_metadata(ALARMED)
    await asyncio.sleep(0.5)  # nothing more may arrive
    records = await _on_ca_loop(connector, connector._bulk.snapshot)
    return {
        "values": {name: _value_json(vals[0]) for name, vals in values.items()},
        "counts": {name: event_counts(connector, name) for name in values},
        "connection": connection,
        "metadata": {"alarm_high": metadata.alarm_high},
        "records": len(records),
    }


async def _scenario_restart(args):
    connector = await _connected(args)
    events: list = []
    await connector.subscribe(
        ALARMED, lambda v: events.append(v.value), on_connection=events.append
    )
    await _until(lambda: len(events) >= 2)

    def monitors():
        record = connector._bulk._records[ALARMED]
        return id(record.ctrl_monitor), id(record.time_monitor)

    before = await _on_ca_loop(connector, monitors)
    await _ask("restart")
    await _until(lambda: len(events) >= 5, timeout=45.0)
    await asyncio.sleep(1.5)  # a duplicated value would land here
    after = await _on_ca_loop(connector, monitors)
    return {
        "events": events,
        "counts": event_counts(connector, ALARMED),
        "same_monitors": before == after,
    }


async def _scenario_resubscribe(args):
    connector = await _connected(args)
    first: list = []
    sub_id = await connector.subscribe(PLAIN, lambda v: first.append(v.value))
    await _until(lambda: first)
    await connector.unsubscribe(sub_id)
    await asyncio.sleep(0.3)
    again: list = []
    await connector.subscribe(PLAIN, lambda v: again.append(v.value))
    await _until(lambda: again)
    joiner: list = []
    await connector.subscribe(PLAIN, lambda v: joiner.append(v.value), on_connection=joiner.append)
    await _until(lambda: len(joiner) >= 2)
    await asyncio.sleep(0.5)
    return {
        "first": first,
        "again": again,
        "joiner": joiner,
        "again_after_join": again,
        "counts": event_counts(connector, PLAIN),
    }


async def _scenario_repoint(args):
    first = await _connected(args, name_server=True)
    first_events: list = []
    sub_id = await first.subscribe(
        ALARMED, lambda v: first_events.append(v.value), on_connection=first_events.append
    )
    await _until(lambda: len(first_events) >= 2)

    # Another route to the same IOC: its connect() repoints the process's CA.
    second = await _connected(args, name_server=False)
    await _until(lambda: len(first_events) >= 5)
    one_shot = (await first.read_channel(PLAIN, timeout=3.0)).value
    await first.unsubscribe(sub_id)
    await first.disconnect()

    second_values: list = []
    await second.subscribe(ALARMED, lambda v: second_values.append(v.value))
    await _until(lambda: second_values)
    await asyncio.sleep(0.3)
    await second.disconnect()
    return {
        "first_events": first_events,
        "one_shot": one_shot,
        "second_values": second_values,
        "teardown": "ok",
    }


_SCENARIOS = {
    "shapes": _scenario_shapes,
    "restart": _scenario_restart,
    "resubscribe": _scenario_resubscribe,
    "repoint": _scenario_repoint,
}


def _client_main(argv: list[str]) -> None:
    scenario, args = argv[1], json.loads(argv[2])
    result = asyncio.run(_SCENARIOS[scenario](args))
    print(json.dumps({"result": result}), flush=True)
    # Skip interpreter teardown: libca's threads may still hold the GIL's attention.
    os._exit(0)


if __name__ == "__main__":
    _client_main(sys.argv)
