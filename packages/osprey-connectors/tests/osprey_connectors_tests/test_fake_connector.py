"""The test-only :class:`~tests._fake_connector.FakeConnector` keeps its promises.

Session and adapter tests drive a control system through it, so what they rely
on is pinned here: a connected channel reports ``True`` before its first value
(both inside ``subscribe``); a ``never_connects`` name reports nothing and cannot
be read; ``drop`` reports ``False`` and silences the channel; ``restore`` reports
``True`` and then the value it now holds; every change delivers the stored value
to the subscriptions of that channel only; ``wait_stale`` raises the lost
connector only after ``go_stale``.
"""

import asyncio

import pytest

from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError
from tests._fake_connector import FakeConnector

ADDR = "SR:FAKE:CURRENT"
OTHER = "SR:FAKE:OTHER"


@pytest.fixture
async def connector():
    conn = FakeConnector()
    await conn.connect({"never_connects": ["SR:FAKE:DEAD"]})
    yield conn
    await conn.disconnect()


def _recorder():
    events: list = []

    def on_value(value):
        events.append(("value", value.value))

    def on_connection(up):
        events.append(("up", up))

    return events, on_value, on_connection


def _set_verdict(monkeypatch, *, allowed: bool) -> None:
    """Answer the connector's write verdict: allowed, or refused as ``switch_off``."""
    from osprey_connectors.posture_store import WriteVerdict

    verdict = WriteVerdict(
        allowed=allowed, reason=None if allowed else "switch_off", target=None, kind=None
    )
    monkeypatch.setattr(
        "osprey_connectors.control_system._base._monitor_verdict", lambda _target: verdict
    )


class TestNeverConnects:
    async def test_a_subscription_reports_nothing(self, connector):
        events, on_value, on_connection = _recorder()

        await connector.subscribe("SR:FAKE:DEAD", on_value, on_connection=on_connection)
        connector.set_value("SR:FAKE:DEAD", 3.0)

        assert events == []

    async def test_a_read_and_a_metadata_lookup_raise_channel_unreachable(self, connector):
        with pytest.raises(ChannelUnreachableError):
            await connector.read_channel("SR:FAKE:DEAD")
        with pytest.raises(ChannelUnreachableError):
            await connector.get_metadata("SR:FAKE:DEAD")

    async def test_other_names_are_unaffected(self, connector):
        events, on_value, on_connection = _recorder()

        await connector.subscribe(ADDR, on_value, on_connection=on_connection)

        assert [kind for kind, _ in events] == ["up", "value"]
        assert events[0] == ("up", True)

    async def test_restore_connects_a_never_connects_name_late(self, connector):
        events, on_value, on_connection = _recorder()
        await connector.subscribe("SR:FAKE:DEAD", on_value, on_connection=on_connection)

        connector.set_value("SR:FAKE:DEAD", 3.0)
        connector.restore("SR:FAKE:DEAD")

        assert events == [("up", True), ("value", 3.0)]

    async def test_without_config_every_name_connects(self):
        conn = FakeConnector()
        await conn.connect({})
        events, on_value, on_connection = _recorder()

        await conn.subscribe("SR:FAKE:DEAD", on_value, on_connection=on_connection)

        assert events[0] == ("up", True)


class TestDropAndRestore:
    async def test_drop_reports_false_and_silences_values(self, connector):
        events, on_value, on_connection = _recorder()
        await connector.subscribe(ADDR, on_value, on_connection=on_connection)
        events.clear()

        connector.drop(ADDR)
        connector.set_value(ADDR, 7.0)

        assert events == [("up", False)]

    async def test_restore_reports_true_then_the_held_value(self, connector):
        events, on_value, on_connection = _recorder()
        await connector.subscribe(ADDR, on_value, on_connection=on_connection)
        connector.drop(ADDR)
        connector.set_value(ADDR, 7.0)
        events.clear()

        connector.restore(ADDR)

        assert events == [("up", True), ("value", 7.0)]

    async def test_values_flow_again_after_restore(self, connector):
        events, on_value, on_connection = _recorder()
        await connector.subscribe(ADDR, on_value, on_connection=on_connection)
        connector.drop(ADDR)
        connector.restore(ADDR)
        events.clear()

        connector.set_value(ADDR, 9.0)

        assert events == [("value", 9.0)]

    async def test_the_full_sequence(self, connector):
        events, on_value, on_connection = _recorder()
        connector.set_value(ADDR, 5.0)

        await connector.subscribe(ADDR, on_value, on_connection=on_connection)
        connector.set_value(ADDR, 6.0)
        connector.drop(ADDR)
        connector.restore(ADDR)

        assert events == [
            ("up", True),
            ("value", 5.0),
            ("value", 6.0),
            ("up", False),
            ("up", True),
            ("value", 6.0),
        ]

    async def test_drop_and_restore_are_idempotent(self, connector):
        events, on_value, on_connection = _recorder()
        await connector.subscribe(ADDR, on_value, on_connection=on_connection)
        events.clear()

        connector.drop(ADDR)
        connector.drop(ADDR)
        connector.restore(ADDR)
        connector.restore(ADDR)

        assert [e for e in events if e[0] == "up"] == [("up", False), ("up", True)]

    async def test_a_dropped_channel_cannot_be_read(self, connector):
        connector.drop(ADDR)
        with pytest.raises(ChannelUnreachableError):
            await connector.read_channel(ADDR)

        connector.restore(ADDR)
        assert (await connector.read_channel(ADDR)).value is not None

    async def test_subscribing_while_dropped_waits_for_restore(self, connector):
        events, on_value, on_connection = _recorder()
        connector.drop(ADDR)

        await connector.subscribe(ADDR, on_value, on_connection=on_connection)
        assert events == []

        connector.restore(ADDR)
        assert [kind for kind, _ in events] == ["up", "value"]

    async def test_drop_touches_only_its_address(self, connector):
        events, on_value, on_connection = _recorder()
        await connector.subscribe(OTHER, on_value, on_connection=on_connection)
        events.clear()

        connector.drop(ADDR)
        connector.set_value(OTHER, 1.0)

        assert events == [("value", 1.0)]

    async def test_every_subscriber_of_the_address_is_told(self, connector):
        first, on_value_1, on_connection_1 = _recorder()
        second, on_value_2, on_connection_2 = _recorder()
        await connector.subscribe(ADDR, on_value_1, on_connection=on_connection_1)
        await connector.subscribe(ADDR, on_value_2, on_connection=on_connection_2)

        connector.drop(ADDR)

        assert first[-1] == ("up", False)
        assert second[-1] == ("up", False)

    async def test_a_plain_subscriber_survives_drop_and_restore(self, connector):
        values: list = []
        await connector.subscribe(ADDR, lambda v: values.append(v.value))
        connector.set_value(ADDR, 2.0)

        connector.drop(ADDR)
        connector.restore(ADDR)

        assert values[-2:] == [2.0, 2.0]

    async def test_a_raising_connection_callback_does_not_stop_the_others(self, connector):
        def boom(up):  # noqa: ARG001 - the on_connection signature; it raises regardless
            raise RuntimeError("bad subscriber")

        events, on_value, on_connection = _recorder()
        await connector.subscribe(ADDR, lambda v: None, on_connection=boom)
        await connector.subscribe(ADDR, on_value, on_connection=on_connection)

        connector.drop(ADDR)

        assert events[-1] == ("up", False)

    async def test_unsubscribed_callbacks_hear_nothing(self, connector):
        events, on_value, on_connection = _recorder()
        sub_id = await connector.subscribe(ADDR, on_value, on_connection=on_connection)
        await connector.unsubscribe(sub_id)
        events.clear()

        connector.drop(ADDR)
        connector.restore(ADDR)

        assert events == []


class TestSubscribe:
    """A subscription delivers what the channel holds, and only when it changes."""

    @pytest.fixture
    def armed(self, monkeypatch):
        _set_verdict(monkeypatch, allowed=True)

    async def test_subscribe_delivers_the_stored_value_once(self, connector):
        connector.set_value(ADDR, 5.0)
        seen = []

        await connector.subscribe(ADDR, seen.append)
        await asyncio.sleep(0.05)

        assert [update.value for update in seen] == [5.0]

    @pytest.mark.usefixtures("armed")
    async def test_every_write_is_delivered(self, connector):
        seen = []
        await connector.subscribe(ADDR, seen.append)

        await connector.write_channel(ADDR, 1.0)
        await connector.write_channel(ADDR, 2.0)

        assert [update.value for update in seen[1:]] == [1.0, 2.0]

    async def test_set_value_notifies_without_a_write(self, connector):
        seen = []
        await connector.subscribe(ADDR, seen.append)

        connector.set_value(ADDR, 7.5)

        assert seen[-1].value == 7.5
        assert (await connector.read_channel(ADDR)).value == 7.5

    async def test_a_change_reaches_only_its_own_channel(self, connector):
        changed, bystander = [], []
        await connector.subscribe(ADDR, changed.append)
        await connector.subscribe(OTHER, bystander.append)

        connector.set_value(ADDR, 100.0)

        assert changed[-1].value == 100.0
        assert len(bystander) == 1

    async def test_one_callback_can_hold_two_subscriptions(self, connector):
        seen = []

        first = await connector.subscribe(ADDR, seen.append)
        second = await connector.subscribe(ADDR, seen.append)
        assert first != second

        await connector.unsubscribe(first)
        connector.set_value(ADDR, 3.0)

        assert [update.value for update in seen[2:]] == [3.0]

    async def test_nothing_is_delivered_after_unsubscribe(self, connector):
        seen = []
        sub_id = await connector.subscribe(ADDR, seen.append)

        await connector.unsubscribe(sub_id)
        connector.set_value(ADDR, 9.0)

        assert len(seen) == 1

    @pytest.mark.usefixtures("armed")
    async def test_a_raising_subscriber_fails_neither_the_write_nor_its_neighbours(self, connector):
        seen = []

        def broken(_update):
            raise RuntimeError("subscriber bug")

        await connector.subscribe(ADDR, broken)
        await connector.subscribe(ADDR, seen.append)

        result = await connector.write_channel(ADDR, 4.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert seen[-1].value == 4.0


class TestWrites:
    """The writes gate and the write outcomes are the base contract's."""

    async def test_a_write_is_refused_while_writes_are_off(self, connector, monkeypatch):
        _set_verdict(monkeypatch, allowed=False)

        result = await connector.write_channel(ADDR, 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert (await connector.read_channel(ADDR)).value == 0.0

    async def test_an_unrequested_confirmation_reads_nothing_back(self, connector, monkeypatch):
        _set_verdict(monkeypatch, allowed=True)

        result = await connector.write_channel(ADDR, 1.0, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed_value is None

    async def test_a_write_to_a_dropped_channel_raises_channel_unreachable(
        self, connector, monkeypatch
    ):
        _set_verdict(monkeypatch, allowed=True)
        connector.drop(ADDR)

        with pytest.raises(ChannelUnreachableError):
            await connector.write_channel(ADDR, 1.0)


class TestWaitStale:
    async def test_it_raises_the_lost_connector_only_after_go_stale(self, connector):
        waiter = asyncio.ensure_future(connector.wait_stale())
        await asyncio.sleep(0.05)
        assert not waiter.done()

        connector.go_stale()

        with pytest.raises(ConnectorUnavailableError):
            await asyncio.wait_for(waiter, timeout=1)
