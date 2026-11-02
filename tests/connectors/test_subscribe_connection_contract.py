"""``subscribe(channel_address, callback, *, on_connection=None)`` across connectors.

The base class declares it; EPICS (CA and PVA) and the in-process simulator
report connection state through it; DOOCS and TANGO accept and ignore it; the connector-host IPC
proxy still has no ``subscribe`` at all.
"""

import asyncio
import inspect
import types
from unittest.mock import MagicMock

import pytest

from osprey_connectors.control_system._base import ControlSystemConnector
from osprey_connectors.control_system._doocs_connector import DOOCSConnector
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._tango_connector import TangoConnector
from osprey_connectors.control_system._va_connector import VirtualAcceleratorConnector
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.ipc._proxy import ConnectorHostProxy
from osprey_connectors_tests._epics_fakes import FakeAioca, FakeChannel, ca_connector, ca_reading
from tests.facility.served_tree import in_process_config, served_tree

CONNECTORS = [
    ControlSystemConnector,
    EPICSConnector,
    VirtualAcceleratorConnector,
    VAInProcessConnector,
    DOOCSConnector,
    TangoConnector,
]


# ---------------------------------------------------------------------------
# The signature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CONNECTORS, ids=lambda c: c.__name__)
def test_on_connection_is_keyword_only_and_optional(cls):
    params = inspect.signature(cls.subscribe).parameters
    assert list(params)[:3] == ["self", "channel_address", "callback"]
    on_connection = params["on_connection"]
    assert on_connection.kind is inspect.Parameter.KEYWORD_ONLY
    assert on_connection.default is None


def test_the_ipc_proxy_still_has_no_subscribe():
    assert not hasattr(ConnectorHostProxy, "subscribe")


# ---------------------------------------------------------------------------
# EPICS Channel Access
# ---------------------------------------------------------------------------


async def drain() -> None:
    """Let the callbacks posted to this loop run."""
    for _ in range(5):
        await asyncio.sleep(0)


def _ca_subscribed_channel() -> tuple[EPICSConnector, FakeAioca, FakeChannel]:
    channel = FakeChannel(ca_reading(5.0), ctrl=ca_reading(0.0, units="mA", precision=3))
    aioca = FakeAioca({"SR:CH": channel})
    return ca_connector(aioca=aioca, timeout=1.0), aioca, channel


def _time_reading(value: float):
    return ca_reading(value, timestamp=1_750_000_000.0, status=0, severity=0)


class TestCaOnConnection:
    async def test_without_on_connection_values_still_flow(self):
        connector, aioca, channel = _ca_subscribed_channel()
        values: list = []

        await connector.subscribe("SR:CH", lambda v: values.append(v.value))
        ctrl, time_ = aioca.monitors_of("SR:CH")
        ctrl.fire(channel.ctrl)
        time_.fire(_time_reading(5.0))
        await drain()

        assert ctrl is not None and time_ is not None
        assert values == [5.0]

    async def test_connect_and_drop_reach_the_subscriber_on_the_loop(self):
        connector, aioca, channel = _ca_subscribed_channel()
        states: list[bool] = []

        await connector.subscribe("SR:CH", lambda v: None, on_connection=states.append)
        ctrl, _time = aioca.monitors_of("SR:CH")

        ctrl.fire(channel.ctrl)
        channel.connected = False
        ctrl.drop()
        assert states == []  # posted, not called inline
        await drain()

        assert states == [True, False]

    async def test_connection_state_is_posted_ahead_of_the_first_value(self):
        connector, aioca, channel = _ca_subscribed_channel()
        events: list = []

        await connector.subscribe(
            "SR:CH", lambda v: events.append(v.value), on_connection=events.append
        )
        ctrl, time_ = aioca.monitors_of("SR:CH")
        ctrl.fire(channel.ctrl)
        await drain()
        time_.fire(_time_reading(5.0))
        await drain()

        assert events == [True, 5.0]

    async def test_a_value_ahead_of_the_control_data_waits_behind_the_connection_report(self):
        connector, aioca, channel = _ca_subscribed_channel()
        events: list = []

        await connector.subscribe(
            "SR:CH", lambda v: events.append(v.value), on_connection=events.append
        )
        ctrl, time_ = aioca.monitors_of("SR:CH")
        time_.fire(_time_reading(5.0))
        await drain()
        assert events == []

        ctrl.fire(channel.ctrl)
        await drain()

        assert events == [True, 5.0]

    async def test_a_raising_connection_callback_does_not_break_the_subscription(self):
        connector, aioca, channel = _ca_subscribed_channel()
        values: list = []

        def boom(up):  # noqa: ARG001 - the on_connection signature; it raises regardless
            raise RuntimeError("bad subscriber")

        assert await connector.subscribe(
            "SR:CH", lambda v: values.append(v.value), on_connection=boom
        )
        ctrl, time_ = aioca.monitors_of("SR:CH")
        ctrl.fire(channel.ctrl)
        await drain()
        time_.fire(_time_reading(6.0))
        await drain()

        assert values == [6.0]
        assert not ctrl.closed and not time_.closed


# ---------------------------------------------------------------------------
# EPICS PVAccess
# ---------------------------------------------------------------------------


class _Value:
    def __init__(self, value):
        self._fields = {"value": value}

    def getID(self):
        return "epics:nt/NTScalar:1.0"

    def get(self, name, default=None):
        return self._fields.get(name, default)


class _MonitorContext:
    def __init__(self):
        self.callbacks = []

    def monitor(self, name, cb, **kwargs):  # noqa: ARG002 - p4p Context.monitor signature
        self.callbacks.append(cb)
        return types.SimpleNamespace(close=lambda: None)


def _pva_connector(context) -> EPICSConnector:
    connector = EPICSConnector()
    connector._pva_channel_globs = ["SR:PVA:*"]
    connector._p4p = MagicMock()
    connector._pva_context = context
    connector._timeout = 1.0
    connector._connected = True
    return connector


class TestPvaOnConnection:
    async def test_true_before_the_first_value_false_after_a_drop_true_again(self):
        context = _MonitorContext()
        connector = _pva_connector(context)
        events: list = []

        await connector.subscribe(
            "SR:PVA:I", lambda v: events.append(v.value), on_connection=events.append
        )
        cb = context.callbacks[0]
        cb(_Value(1.0))
        cb(_Value(2.0))
        cb(RuntimeError("Disconnected"))
        cb(RuntimeError("Disconnected"))  # a repeated event reports nothing new
        cb(_Value(3.0))
        await asyncio.sleep(0.01)

        assert events == [True, 1.0, 2.0, False, True, 3.0]

    async def test_without_on_connection_values_still_flow(self):
        context = _MonitorContext()
        connector = _pva_connector(context)
        values: list = []

        await connector.subscribe("SR:PVA:I", lambda v: values.append(v.value))
        context.callbacks[0](RuntimeError("Disconnected"))
        context.callbacks[0](_Value(4.0))
        await asyncio.sleep(0.01)

        assert values == [4.0]


# ---------------------------------------------------------------------------
# The in-process simulator
# ---------------------------------------------------------------------------


class TestInProcessOnConnection:
    @staticmethod
    async def _connected(tmp_path) -> VAInProcessConnector:
        view = served_tree(tmp_path, {"SR:CH:SP": None})
        connector = VAInProcessConnector()
        await connector.connect(in_process_config(view, response_delay_ms=0))
        return connector

    async def test_true_once_before_subscribe_returns(self, tmp_path):
        connector = await self._connected(tmp_path)
        states: list = []

        await connector.subscribe("SR:CH:SP", lambda v: None, on_connection=states.append)

        assert states == [True]
        await connector.disconnect()

    async def test_a_raising_connection_callback_does_not_fail_subscribe(self, tmp_path):
        connector = await self._connected(tmp_path)

        def boom(up):  # noqa: ARG001 - the on_connection signature; it raises regardless
            raise RuntimeError("bad subscriber")

        assert await connector.subscribe("SR:CH:SP", lambda v: None, on_connection=boom)
        await connector.disconnect()


# ---------------------------------------------------------------------------
# DOOCS and TANGO accept and ignore it
# ---------------------------------------------------------------------------


class TestAcceptAndIgnore:
    async def test_doocs_ignores_on_connection(self):
        connector = DOOCSConnector()
        connector._doocs4py = MagicMock()
        connector._subscriptions = {}
        states: list = []

        sub_id = await connector.subscribe(
            "FLASH.RF/A/B/C", lambda v: None, on_connection=states.append
        )

        assert sub_id
        assert states == []

    async def test_tango_ignores_on_connection(self, monkeypatch):
        connector = TangoConnector()
        proxy = MagicMock()
        proxy.subscribe_event.return_value = 7
        monkeypatch.setattr(connector, "_get_proxy", lambda device: proxy)
        connector._tango = MagicMock()
        connector._subscriptions = {}
        states: list = []

        sub_id = await connector.subscribe(
            "sys/tg_test/1/double_scalar", lambda v: None, on_connection=states.append
        )

        assert sub_id
        assert states == []
