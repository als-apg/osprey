"""``subscribe(channel_address, callback, *, on_connection=None)`` across connectors.

The base class declares it; EPICS (CA and PVA) reports connection state through
it; DOOCS and TANGO accept and ignore it; the connector-host IPC proxy still has
no ``subscribe`` at all. The in-process simulator's report is pinned beside the
framework's served tree, in the framework's connector tests.

The first callback of a connection carries the channel's current value, with no
change needed to send it: pinned here for the in-process simulator and, over a
real soft IOC, for the served simulator's connector. DOOCS and TANGO subscribe
through their own libraries' change events; that those send the current value
first is read from their documentation, not run here.
"""

import asyncio
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from osprey_connectors.control_system._base import ControlSystemConnector
from osprey_connectors.control_system._doocs_connector import DOOCSConnector
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._tango_connector import TangoConnector
from osprey_connectors.control_system._va_connector import VirtualAcceleratorConnector
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.ipc._proxy import ConnectorHostProxy
from osprey_connectors.simulation.view import SCHEMAS
from osprey_connectors_tests._epics_fakes import FakeAioca, FakeChannel, ca_connector, ca_reading
from osprey_connectors_tests._paths import TESTS_ROOT

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


# ---------------------------------------------------------------------------
# The first callback carries the current value
# ---------------------------------------------------------------------------

SETPOINT = "T:SP"
NOMINAL = 5.0


def _one_setpoint_view(root: Path) -> Path:
    """A simulator view serving one texture setpoint at ``NOMINAL``."""
    documents = {
        "served_models.json": {"models": ["texture"]},
        "addresses.json": {"channels": [SETPOINT], "status": []},
        "variables.json": {
            "code": "T",
            "models": [
                {
                    "name": "texture",
                    "engine": "texture",
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [],
                }
            ],
            "channels": [
                {
                    "address": SETPOINT,
                    "owner": "texture",
                    "role": "setpoint",
                    "value_type": "float",
                    "writable": True,
                    "pair": None,
                    "unit": None,
                    "description": None,
                    "value_range": None,
                    "on": None,
                }
            ],
        },
        "seeds.json": {"seeds": {SETPOINT: {"nominal": NOMINAL}}},
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    view = root / "simulator"
    view.mkdir(parents=True)
    for name, document in documents.items():
        (view / name).write_text(json.dumps({"schema": SCHEMAS[name], **document}))
    return view


class TestCurrentValueFirst:
    async def test_the_in_process_simulator_sends_the_current_value_before_subscribe_returns(
        self, tmp_path
    ):
        connector = VAInProcessConnector()
        await connector.connect(
            {"simulator_view": str(_one_setpoint_view(tmp_path)), "response_delay_ms": 0}
        )
        events: list = []

        await connector.subscribe(
            SETPOINT, lambda v: events.append(v.value), on_connection=events.append
        )

        assert events == [True, pytest.approx(NOMINAL)]
        await connector._notify()
        assert events == [True, pytest.approx(NOMINAL)], "an unchanged value is sent once"
        await connector.disconnect()

    async def test_an_awaitable_callback_is_awaited_before_subscribe_returns(self, tmp_path):
        connector = VAInProcessConnector()
        await connector.connect(
            {"simulator_view": str(_one_setpoint_view(tmp_path)), "response_delay_ms": 0}
        )
        values: list = []

        async def callback(value):
            values.append(value.value)

        await connector.subscribe(SETPOINT, callback)

        assert values == [pytest.approx(NOMINAL)]
        await connector.disconnect()

    # xdist_group("soft-ioc"): a soft IOC's ports are probed free, then bound by a
    # child process; one worker keeps that window free of a second prober.
    @pytest.mark.xdist_group("soft-ioc")
    def test_the_served_simulator_sends_the_current_value_over_the_wire(self, tmp_path):
        from osprey_connectors_tests._soft_ioc import SoftIOC, ai_record

        with SoftIOC([ai_record(WIRE_RECORD, WIRE_VALUE)], workdir=tmp_path) as ioc:
            result = _run_wire_client(ioc)

        assert result == {"first": WIRE_VALUE}


# ---------------------------------------------------------------------------
# Wire: the served simulator's connector against a soft IOC, in its own process
# ---------------------------------------------------------------------------

WIRE_RECORD = "SUBSCRIBE:CONTRACT:VALUE"
WIRE_VALUE = 1.5
#: The client's CA timeout; every wait of the wire test is a multiple of it.
WIRE_TIMEOUT_S = 3.0
REPO = Path(__file__).resolve().parents[2]


def _run_wire_client(ioc) -> dict:
    """Subscribe the served simulator's connector to ``WIRE_RECORD``; its first value."""
    from osprey_connectors_tests._soft_ioc import kill_group

    env = {key: value for key, value in os.environ.items() if not key.startswith("EPICS_")}
    env.update(ioc.client_env(name_server=False))
    env.pop("EPICS_CA_ADDR_LIST", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO), str(TESTS_ROOT), env.get("PYTHONPATH")])
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "osprey_connectors_tests.test_subscribe_connection_contract",
            str(ioc.ports.ca_server),
        ],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=20 * WIRE_TIMEOUT_S)
    finally:
        kill_group(proc)
    results = [line for line in stdout.splitlines() if line.startswith("{")]
    assert results, f"the client printed no result; stderr:\n{stderr}"
    return json.loads(results[-1])


async def _wire_client(port: int) -> dict:
    connector = VirtualAcceleratorConnector()
    await connector.connect(
        {
            "timeout_s": WIRE_TIMEOUT_S,
            "gateways": {
                "read_only": {"address": "127.0.0.1", "port": port, "use_name_server": True}
            },
        }
    )
    values: list = []
    arrived = asyncio.Event()

    def callback(value):
        values.append(value.value)
        arrived.set()

    await connector.subscribe(WIRE_RECORD, callback)
    await asyncio.wait_for(arrived.wait(), timeout=10 * WIRE_TIMEOUT_S)
    return {"first": values[0]}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_wire_client(int(sys.argv[1])))), flush=True)
    # libca's threads may still be running; skip interpreter teardown.
    os._exit(0)
