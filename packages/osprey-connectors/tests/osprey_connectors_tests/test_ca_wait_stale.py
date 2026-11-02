"""``EPICSConnector.wait_stale``: telling a stale CA transport from a dead IOC.

After an outage of a name server, libca may search for a channel it held again
only after a long back-off, while a channel it opens afresh connects at once.
``wait_stale`` reads aioca's own channel state on the CA loop (no CA call, no
connection of its own) and raises ``ConnectorUnavailableError`` -- the lost
connector, which its supervisor replaces -- once a name that connected in this
process is still down a probe after some channel of the connector connected
since it went down.

Two halves:

* **Unit** — a connector on a fake ``aioca`` and an inline CA loop, its real
  subscription records driven by the fake's monitors, its one-shot channels
  told by the fake's channel cache: the held-down rules, the evidence rule,
  the never-connected bound, the one-shot window and the never-returning
  shapes.
* **Wire** — a soft IOC (``tests/connectors/_soft_ioc.py``) and a client in its
  own process and session, as in ``test_ca_bulk_path.py``: each scenario runs as
  ``python -m osprey_connectors_tests.test_ca_wait_stale <scenario> <json args>`` and
  asks this process to kill or restart the IOC over stdin/stdout.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.control_system import _epics_connector as epics_connector_module
from osprey_connectors.control_system._base import ControlSystemConnector
from osprey_connectors.control_system._epics_bulk import normalise_address
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError
from osprey_connectors_tests._epics_fakes import (
    FakeAioca,
    FakeChannel,
    ca_connector,
    ca_reading,
    clean_epics_env,  # noqa: F401 - fixture, used by name
    install_fake_aioca,
    patch_writes_enabled,
)
from osprey_connectors_tests._paths import TESTS_ROOT
from osprey_connectors_tests._soft_ioc import READY_TIMEOUT_S

# xdist_group("soft-ioc"): the wire half starts real soft IOCs. Their ports are
# probed free and then bound by a child process a moment later, so two workers
# probing at once can pick the same port; the IOCs' CA/PVA client sockets and
# repeater also share the host's UDP space. One worker keeps the probe-then-bind
# window free of a second prober.
pytestmark = pytest.mark.xdist_group("soft-ioc")

REPO = Path(__file__).resolve().parents[2]

#: How long a watch that must never return is given to return anyway.
NEVER_S = 3.0

#: Liveness: the kill-and-restart client must see its subscription connect at all.
CONNECT_DEADLINE_S = 20.0

#: How long the connected client waits before it asks for the kill.
SETTLE_BEFORE_KILL_S = 1.5

#: How long the IOC stays dead: past the 3 s hold-down, with nobody to answer.
#: libca's search back-off then keeps the dropped channel down for seconds
#: after the restart -- the lag the watch cuts short.
OUTAGE_S = 4.5

#: The silent window the kill-only client runs with, instead of 30 s.
SILENT_S = 6.0

#: Liveness: after the restart the client must report stale at all.
RECONNECT_DEADLINE_S = 60.0

#: The speed claim: stale is reported this soon after the restart.
STALE_AFTER_RESTART_BUDGET_S = 10.0

#: Client interpreter start-up plus the IOC kill teardown (kill_group's two 5 s
#: waits and the 5 s reader join, 15 s at most).
CLIENT_MARGIN_S = 30.0

#: The outer bound on one client run: the sum of the longest scenario's parts,
#: the IOC restart included, so it never fails before a deadline inside it does.
CLIENT_TIMEOUT_S = (
    CONNECT_DEADLINE_S
    + SETTLE_BEFORE_KILL_S
    + OUTAGE_S
    + READY_TIMEOUT_S
    + RECONNECT_DEADLINE_S
    + CLIENT_MARGIN_S
)

SERVED = "WS:SERVED"
#: A second record on the same IOC, read once after the restart: a fresh channel.
OTHER = "WS:OTHER"
UNSERVED = ["WS:NOBODY:A", "WS:NOBODY:B"]


async def _goes_stale(connector, seconds: float = 5.0) -> None:
    """Assert ``connector.wait_stale()`` raises the lost connector within ``seconds``."""
    with pytest.raises(ConnectorUnavailableError):
        await asyncio.wait_for(connector.wait_stale(), timeout=seconds)


async def _never_returns(waiter: asyncio.Future, seconds: float = NEVER_S) -> None:
    """Assert ``waiter`` stays pending for ``seconds``, then cancel it."""
    await asyncio.sleep(seconds)
    assert not waiter.done()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter


# ---------------------------------------------------------------------------
# Unit: a fake aioca, the real subscription records and a fake name server
# ---------------------------------------------------------------------------

#: The hold-down and confirmation the unit half runs with, and how long a
#: record waits the hold-down out.
HOLD_S = 0.02
CONFIRM_S = 0.02
AGED_S = 0.06


@pytest.fixture
def fast_probe(monkeypatch):
    monkeypatch.setattr(epics_connector_module, "STALE_PROBE_INTERVAL_S", 0.01)
    monkeypatch.setattr(epics_connector_module, "STALE_HOLD_DOWN_S", HOLD_S)
    monkeypatch.setattr(epics_connector_module, "STALE_CONFIRM_S", CONFIRM_S)


def _watched(*, server="127.0.0.1:5075") -> tuple[EPICSConnector, FakeAioca]:
    """A connector on a fake aioca and an inline CA loop, told ``server``."""
    fake = FakeAioca()
    connector = ca_connector(aioca=fake, timeout=0.05)
    connector._name_server = server
    return connector, fake


async def _down_record(connector, fake, name: str, *, ever: bool = True) -> str:
    """Subscribe ``name`` and leave its record down: dropped, or never connected."""
    fake.channels[name] = FakeChannel(connected=ever)
    sub_id = await connector.subscribe(name, lambda value: None)
    if ever:
        _connect(fake, name)
        _drop(fake, name)
    return sub_id


async def _up_record(connector, fake, name: str) -> str:
    """Subscribe ``name`` and bring its record up: a channel connects."""
    fake.channels[name] = FakeChannel()
    sub_id = await connector.subscribe(name, lambda value: None)
    _connect(fake, name)
    return sub_id


def _connect(fake, name: str) -> None:
    """The record's channel connects: its CTRL event arrives."""
    fake.channels[name].connected = True
    fake.monitors_of(name)[0].fire(ca_reading(0.0, units="mA"))


def _drop(fake, name: str) -> None:
    """The record's channel drops."""
    fake.channels[name].connected = False
    fake.monitors_of(name)[1].drop()


async def _one_shot(connector, fake, name: str, *, ever: bool = True) -> None:
    """A one-shot read of ``name`` that fails to connect (after one that did, if ``ever``)."""
    if ever:
        channel = fake.channels[normalise_address(name)] = FakeChannel()
        await connector.read_channel(name)
        channel.connected = False
    with pytest.raises(ChannelUnreachableError):
        await connector.read_channel(name, timeout=0.01)


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.005)


class TestNeverReturns:
    @pytest.mark.usefixtures("fast_probe")
    async def test_without_a_name_server(self):
        connector, aioca = _watched(server=None)
        await _down_record(connector, aioca, "X")
        await asyncio.sleep(AGED_S)
        await _up_record(connector, aioca, "Y")  # what is stale with a name server
        await _never_returns(asyncio.ensure_future(connector.wait_stale()), 0.3)

    async def test_the_in_process_simulator(self):
        await _never_returns(asyncio.ensure_future(VAInProcessConnector().wait_stale()))

    async def test_the_base_class(self):
        # DOOCS and Tango inherit this default unchanged.
        await _never_returns(
            asyncio.ensure_future(ControlSystemConnector.wait_stale(VAInProcessConnector()))
        )


@pytest.mark.usefixtures("fast_probe")
class TestStaleRule:
    async def test_still_down_after_another_channel_connected_is_stale(self):
        connector, aioca = _watched()
        await _down_record(connector, aioca, "X")
        await asyncio.sleep(AGED_S)
        await _up_record(connector, aioca, "Y")
        await _goes_stale(connector)

    async def test_a_reconnect_of_another_record_is_evidence(self):
        connector, aioca = _watched()
        await _down_record(connector, aioca, "X")
        await _down_record(connector, aioca, "Y")
        await asyncio.sleep(AGED_S)
        _connect(aioca, "Y")  # libca brought Y back, X lags
        await _goes_stale(connector)

    async def test_a_new_one_shot_connect_is_evidence(self):
        connector, aioca = _watched()
        await _down_record(connector, aioca, "X")
        await asyncio.sleep(AGED_S)
        aioca.channels["FRESH"] = FakeChannel()
        await connector.read_channel("FRESH")
        await _goes_stale(connector)

    async def test_without_a_connect_since_the_drop_it_never_triggers(self):
        connector, aioca = _watched()
        await _up_record(connector, aioca, "Y")  # connected before X went down
        await _down_record(connector, aioca, "X")
        await asyncio.sleep(AGED_S)
        # A read of an already connected channel is no new connection.
        await connector.read_channel("Y")
        await connector.read_channel("Y")
        await _never_returns(asyncio.ensure_future(connector.wait_stale()), 0.3)

    async def test_a_name_that_recovers_within_the_confirmation_does_not_count(self, monkeypatch):
        monkeypatch.setattr(epics_connector_module, "STALE_CONFIRM_S", 0.2)
        connector, aioca = _watched()
        await _down_record(connector, aioca, "X")
        await asyncio.sleep(AGED_S)
        await _up_record(connector, aioca, "Y")
        waiter = asyncio.ensure_future(connector.wait_stale())
        await asyncio.sleep(0.05)
        assert not waiter.done(), "the evidence is not confirmed yet"
        _connect(aioca, "X")  # libca catches up within the confirmation
        await _never_returns(waiter, 0.4)

    async def test_down_the_silent_window_with_no_connect_is_stale(self, monkeypatch):
        monkeypatch.setattr(epics_connector_module, "STALE_SILENT_S", 0.3)
        connector, aioca = _watched()
        await _down_record(connector, aioca, "X")
        waiter = asyncio.ensure_future(connector.wait_stale())
        await asyncio.sleep(0.15)
        assert not waiter.done()
        with pytest.raises(ConnectorUnavailableError):
            await asyncio.wait_for(waiter, timeout=5)

    async def test_the_silent_window_never_counts_a_name_that_never_connected(self, monkeypatch):
        monkeypatch.setattr(epics_connector_module, "STALE_SILENT_S", 0.05)
        connector, aioca = _watched()
        await _down_record(connector, aioca, "NEVER", ever=False)
        await _never_returns(asyncio.ensure_future(connector.wait_stale()), 0.3)

    async def test_a_record_down_the_hold_down_or_less_is_not_held_down(self, monkeypatch):
        monkeypatch.setattr(epics_connector_module, "STALE_HOLD_DOWN_S", 3.0)
        connector, aioca = _watched()
        await _down_record(connector, aioca, "X")
        # A record whose last subscriber left is forgotten, however long it was down.
        await connector.unsubscribe(await _down_record(connector, aioca, "Y"))
        await asyncio.sleep(AGED_S)
        await _up_record(connector, aioca, "Z")
        await _never_returns(asyncio.ensure_future(connector.wait_stale()), 0.3)

    async def test_a_name_that_never_connected_never_counts(self):
        connector, aioca = _watched()
        await _down_record(connector, aioca, "NEVER", ever=False)
        await asyncio.sleep(AGED_S)
        await _up_record(connector, aioca, "Y")
        waiter = asyncio.ensure_future(connector.wait_stale())
        await _never_returns(waiter, 0.3)

    async def test_a_record_left_without_monitors_is_held_down(self, monkeypatch):
        connector, aioca = _watched()
        await _up_record(connector, aioca, "X")
        plain = aioca.camonitor

        def broken(*args, **kwargs):
            raise RuntimeError("camonitor failed")

        monkeypatch.setattr(aioca, "camonitor", broken)
        connector._ca.repoint()  # the re-open fails: X is down, no monitors
        monkeypatch.setattr(aioca, "camonitor", plain)
        await asyncio.sleep(AGED_S)
        await _up_record(connector, aioca, "Y")
        await _goes_stale(connector)


@pytest.mark.usefixtures("fast_probe")
class TestOneShotRule:
    async def _evidence(self, connector, aioca) -> None:
        aioca.channels["FRESH"] = FakeChannel()
        await connector.read_channel("FRESH")

    async def test_a_recent_failure_on_a_channel_that_connected_is_held_down(self):
        connector, aioca = _watched()
        await _one_shot(connector, aioca, "ca://ONE")
        await self._evidence(connector, aioca)
        await asyncio.sleep(CONFIRM_S * 2)
        assert set(await connector._stale_now()) == {"ONE"}, "by channel name, scheme stripped"
        await _goes_stale(connector)

    async def test_a_failure_of_a_channel_that_never_connected_is_not(self):
        connector, aioca = _watched()
        await _one_shot(connector, aioca, "ONE", ever=False)
        await self._evidence(connector, aioca)
        await _never_returns(asyncio.ensure_future(connector.wait_stale()), 0.3)

    async def test_a_failure_older_than_the_window_is_not(self, monkeypatch):
        monkeypatch.setattr(epics_connector_module, "STALE_ONE_SHOT_WINDOW_S", 0.02)
        connector, aioca = _watched()
        await _one_shot(connector, aioca, "ONE")
        await asyncio.sleep(AGED_S)
        await self._evidence(connector, aioca)
        await _never_returns(asyncio.ensure_future(connector.wait_stale()), 0.3)

    async def test_a_channel_connected_now_is_not(self):
        connector, aioca = _watched()
        await _one_shot(connector, aioca, "ONE")
        await self._evidence(connector, aioca)
        aioca.channels["ONE"] = FakeChannel()  # libca connected it since
        await _never_returns(asyncio.ensure_future(connector.wait_stale()), 0.3)

    async def test_a_channel_absent_from_the_channel_cache_is_not_held_down(self, monkeypatch):
        # A repoint purges the cache; a name it no longer holds is not down here.
        connector, aioca = _watched()
        await _one_shot(connector, aioca, "ONE")
        await self._evidence(connector, aioca)
        monkeypatch.setattr(aioca, "get_channel_infos", lambda: [], raising=False)
        await _never_returns(asyncio.ensure_future(connector.wait_stale()), 0.3)

    async def test_an_unreadable_channel_cache_holds_nothing_down(self, monkeypatch):
        connector, aioca = _watched()
        await _one_shot(connector, aioca, "ONE")
        await self._evidence(connector, aioca)

        def broken():
            raise RuntimeError("no CA context")

        monkeypatch.setattr(aioca, "get_channel_infos", broken, raising=False)
        await _never_returns(asyncio.ensure_future(connector.wait_stale()), 0.3)

    async def test_the_read_path_records_failures_connections_and_new_connects(self):
        connector, aioca = _watched()
        with pytest.raises(ChannelUnreachableError):
            await connector.read_channel("ca://DOWN", timeout=0.01)
        assert time.monotonic() - connector._connect_failures["DOWN"] < 5
        assert "DOWN" not in connector._one_shot_connected
        assert connector._one_shot_connect_at is None

        aioca.channels["UP"] = FakeChannel()
        await connector.read_channel("ca://UP")
        first = connector._one_shot_connect_at
        assert "UP" in connector._one_shot_connected and first is not None
        await connector.read_channel("UP")
        assert connector._one_shot_connect_at == first, "an open channel is no new connect"

        aioca.channels["DOWN"] = FakeChannel()  # it connects after its failure
        await connector.read_channel("DOWN")
        assert "DOWN" not in connector._connect_failures
        assert connector._one_shot_connect_at > first


@pytest.mark.usefixtures("clean_epics_env")
class TestNameServerStorage:
    async def test_connect_stores_the_name_server_and_only_with_one(self, monkeypatch):
        install_fake_aioca(monkeypatch)
        patch_writes_enabled(monkeypatch, False)

        connector = EPICSConnector()
        await connector.connect(
            {"gateways": {"read_only": {"address": "gw", "port": 5075, "use_name_server": True}}}
        )
        assert connector._name_server == "gw:5075"

        await connector.connect({"gateways": {"read_only": {"address": "gw", "port": 5064}}})
        assert connector._name_server is None

        await connector.connect({})
        assert connector._name_server is None


# ---------------------------------------------------------------------------
# Wire: a soft IOC and a client in its own process
# ---------------------------------------------------------------------------


@pytest.fixture
def ioc(tmp_path):
    from osprey_connectors_tests._soft_ioc import SoftIOC, ai_record

    with SoftIOC([ai_record(SERVED, 1.5), ai_record(OTHER, 2.5)], workdir=tmp_path) as running:
        yield running


def _client_env(ioc, *, gateway: bool) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("EPICS_")}
    env.update(ioc.client_env(name_server=False))
    if gateway:
        env.pop("EPICS_CA_ADDR_LIST", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO), str(TESTS_ROOT), env.get("PYTHONPATH")])
    )
    return env


def run_client(
    ioc, scenario: str, *, gateway: bool = True, timeout: float = CLIENT_TIMEOUT_S, **extra: Any
) -> dict[str, Any]:
    """Run one client scenario against ``ioc``; serve its kill/restart requests."""
    from osprey_connectors_tests._soft_ioc import kill_group

    args = {"port": ioc.ports.ca_server, "gateway": gateway, **extra}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "osprey_connectors_tests.test_ca_wait_stale",
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


class TestDeadlines:
    def test_the_client_timeout_covers_every_part_of_the_longest_scenario(self):
        timeout = inspect.signature(run_client).parameters["timeout"].default
        parts = (
            CONNECT_DEADLINE_S
            + SETTLE_BEFORE_KILL_S
            + OUTAGE_S
            + READY_TIMEOUT_S
            + RECONNECT_DEADLINE_S
        )
        assert timeout > parts

    def test_the_reconnect_wait_is_wider_than_the_claim(self):
        assert RECONNECT_DEADLINE_S > STALE_AFTER_RESTART_BUDGET_S


class TestOverTheWire:
    def test_names_the_ioc_does_not_serve_never_report_stale(self, ioc):
        result = run_client(ioc, "unserved")
        assert result["served_connected"] is True, "a channel connected after they went down"
        assert result["held"] == [], "a name that never connected is never held down"
        assert result["stale"] is False

    def test_a_kill_and_restart_reports_stale_exactly_once(self, ioc):
        result = run_client(ioc, "kill_restart")
        assert result["before_kill"] is False
        assert result["during_outage"] is False
        assert result["stale"] is True
        assert result["other_read"] is True, "a fresh channel connected at once"
        assert result["down_at_stale"] is True, "libca still held it down"
        assert result["seconds_after_restart"] < STALE_AFTER_RESTART_BUDGET_S

        # The successor (a fresh process and transport) connects and
        # does not report stale again: one outage, one replacement.
        successor = run_client(ioc, "successor")
        assert successor["connected"] is True
        assert successor["stale"] is False

    def test_a_server_that_stays_gone_reports_stale_once_after_the_silent_window(self, ioc):
        result = run_client(ioc, "kill_only")
        assert result["stale"] is True
        assert result["down_at_stale"] is True
        assert SILENT_S <= result["seconds_after_kill"] < SILENT_S + STALE_AFTER_RESTART_BUDGET_S

        # Its successor never connects to the gone server, so it never counts it.
        successor = run_client(ioc, "successor_while_gone")
        assert successor["connected"] is False
        assert successor["stale"] is False

    def test_a_one_shot_of_an_unknown_pv_on_an_idle_connector_is_not_stale(self, ioc):
        result = run_client(ioc, "one_shot_unknown")
        assert result["read"] == "ChannelUnreachableError"
        assert result["served_read"] is True, "a channel connected after the failure"
        assert result["held"] == [], "a name that never connected is never held down"
        assert result["stale"] is False

    def test_a_udp_search_connector_never_reports_stale(self, ioc):
        result = run_client(ioc, "udp", gateway=False)
        assert result["name_server"] is None
        assert result["stale"] is False


# ---------------------------------------------------------------------------
# Client side (runs in its own process)
# ---------------------------------------------------------------------------


async def _ask(cmd: str) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    await asyncio.to_thread(sys.stdin.readline)


async def _connected(args: dict[str, Any]) -> EPICSConnector:
    config: dict[str, Any] = {"timeout_s": 3.0}
    if args["gateway"]:
        config["gateways"] = {
            "read_only": {"address": "127.0.0.1", "port": args["port"], "use_name_server": True}
        }
    connector = EPICSConnector()
    await connector.connect(config)
    return connector


async def _held(connector) -> list[str]:
    """The names held down right now, asked on the CA loop."""
    return sorted(await connector._ca.run(connector._held_down()))


async def _record_down(connector, name: str) -> bool:
    """Whether ``name``'s subscription record exists and is down, asked on the CA loop."""

    async def ask() -> bool | None:
        return connector._bulk.is_down(name)

    return await connector._ca.run(ask()) is True


async def _read_within(connector, name: str, seconds: float) -> bool:
    """Read ``name`` until a read succeeds, for at most ``seconds``."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            await connector.read_channel(name, timeout=1.0)
            return True
        except ChannelUnreachableError:
            await asyncio.sleep(0.1)
    return False


async def _stale_within(connector, seconds: float) -> bool:
    """Whether ``wait_stale`` raised the lost connector within ``seconds``."""
    try:
        await asyncio.wait_for(connector.wait_stale(), timeout=seconds)
    except ConnectorUnavailableError:
        return True
    except TimeoutError:
        return False
    raise AssertionError("wait_stale returned without raising")


async def _scenario_unserved(args):
    connector = await _connected(args)
    for name in UNSERVED:
        await connector.subscribe(name, lambda v: None)
    await asyncio.sleep(3.5)  # past the hold-down
    served_connected = await _read_within(connector, SERVED, CONNECT_DEADLINE_S)
    waiter = asyncio.ensure_future(_stale_within(connector, 3.0))
    await asyncio.sleep(1.0)
    return {
        "served_connected": served_connected,
        "held": await _held(connector),
        "stale": await waiter,
    }


async def _scenario_kill_restart(args):
    connector = await _connected(args)
    states: list[bool] = []
    await connector.subscribe(SERVED, lambda v: None, on_connection=states.append)
    deadline = time.monotonic() + CONNECT_DEADLINE_S
    while states[-1:] != [True]:
        if time.monotonic() > deadline:
            raise TimeoutError("never connected")
        await asyncio.sleep(0.05)
    waiter = asyncio.ensure_future(connector.wait_stale())
    await asyncio.sleep(SETTLE_BEFORE_KILL_S)
    before_kill = waiter.done()

    await _ask("kill")
    await asyncio.sleep(OUTAGE_S)  # past the 3 s hold-down, with nobody to answer
    during_outage = waiter.done()
    await _ask("restart")
    restarted = time.monotonic()
    # A fresh channel to the same IOC connects at once; the dropped one lags.
    other_read = await _read_within(connector, OTHER, RECONNECT_DEADLINE_S)
    try:
        await asyncio.wait_for(asyncio.shield(waiter), timeout=RECONNECT_DEADLINE_S)
        raise AssertionError("wait_stale returned without raising")
    except ConnectorUnavailableError:
        stale = True
    except TimeoutError:
        stale = False
    return {
        "before_kill": before_kill,
        "during_outage": during_outage,
        "stale": stale,
        "other_read": other_read,
        "down_at_stale": await _record_down(connector, SERVED),
        "seconds_after_restart": time.monotonic() - restarted,
    }


async def _scenario_kill_only(args):
    epics_connector_module.STALE_SILENT_S = SILENT_S
    connector = await _connected(args)
    states: list[bool] = []
    await connector.subscribe(SERVED, lambda v: None, on_connection=states.append)
    deadline = time.monotonic() + CONNECT_DEADLINE_S
    while states[-1:] != [True]:
        if time.monotonic() > deadline:
            raise TimeoutError("never connected")
        await asyncio.sleep(0.05)
    waiter = asyncio.ensure_future(connector.wait_stale())
    killed = time.monotonic()  # at or before the drop, so the lower bound holds
    await _ask("kill")
    try:
        await asyncio.wait_for(waiter, timeout=SILENT_S + STALE_AFTER_RESTART_BUDGET_S + 5.0)
        raise AssertionError("wait_stale returned without raising")
    except ConnectorUnavailableError:
        stale = True
    except TimeoutError:
        stale = False
    return {
        "stale": stale,
        "down_at_stale": await _record_down(connector, SERVED),
        "seconds_after_kill": time.monotonic() - killed,
    }


async def _scenario_successor_while_gone(args):
    epics_connector_module.STALE_SILENT_S = SILENT_S
    connector = await _connected(args)
    states: list[bool] = []
    await connector.subscribe(SERVED, lambda v: None, on_connection=states.append)
    stale = await _stale_within(connector, SILENT_S + 4.0)
    return {"connected": True in states, "stale": stale}


async def _scenario_successor(args):
    connector = await _connected(args)
    states: list[bool] = []
    await connector.subscribe(SERVED, lambda v: None, on_connection=states.append)
    stale = await _stale_within(connector, 6.0)
    return {"connected": states[-1:] == [True], "stale": stale}


async def _scenario_one_shot_unknown(args):
    connector = await _connected(args)
    try:
        await connector.read_channel("WS:UNKNOWN", timeout=1.0)
        read = "ok"
    except ChannelUnreachableError as exc:
        read = type(exc).__name__
    served_read = await _read_within(connector, SERVED, CONNECT_DEADLINE_S)
    waiter = asyncio.ensure_future(_stale_within(connector, 3.0))
    await asyncio.sleep(1.0)
    return {
        "read": read,
        "served_read": served_read,
        "held": await _held(connector),
        "stale": await waiter,
    }


async def _scenario_udp(args):
    connector = await _connected(args)
    await connector.subscribe(UNSERVED[0], lambda v: None)
    return {
        "name_server": connector._name_server,
        "stale": await _stale_within(connector, 3.5),
    }


_SCENARIOS = {
    "unserved": _scenario_unserved,
    "kill_restart": _scenario_kill_restart,
    "successor": _scenario_successor,
    "kill_only": _scenario_kill_only,
    "successor_while_gone": _scenario_successor_while_gone,
    "one_shot_unknown": _scenario_one_shot_unknown,
    "udp": _scenario_udp,
}


def _client_main(argv: list[str]) -> None:
    scenario, args = argv[1], json.loads(argv[2])
    result = asyncio.run(_SCENARIOS[scenario](args))
    print(json.dumps({"result": result}), flush=True)
    # Skip interpreter teardown: libca's threads (the CA loop's) may still be running.
    os._exit(0)


if __name__ == "__main__":
    _client_main(sys.argv)
