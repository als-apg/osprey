"""Live: a scenario's ``stopped_servers`` on both wires, with held clients.

A simulated IOC a scenario stops withholds every one of its channels, as a
stopped IOC does: a held Channel Access client sees its channel disconnect
and a fresh one finds nothing, a PVAccess read times out -- both *no answer*,
never a refusal and never a frozen value. Removing the scenario serves them
again, without a restart. The in-process half is the connectors package's own ``test_stopped_servers.py``.

The server, its clients and the stub view are the model runner seam's
(``tests/va/test_lume_pva_seam.py``), imported so the two suites serve the
same runner; that module names this one's server port. Run by the live-CA
gate (``scripts/va/live_ca/gate.py``), which fails it on any skip.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.va._collection import assert_module_collected
from tests.va.test_lume_pva_seam import (  # the seam's harness and stub view
    CA_PORTS,
    CA_TIMEOUT_S,
    STUB_CODE,
    STUB_COUNT,
    STUB_RANGE,
    STUB_SETPOINT,
    STUB_TICK_S,
    _ca_connects,
    _on_ca_loop,
    _served,
    _Server,
    _wait_until,
    _write_stub_view,
)

#: Floor for this module's own test count.
MIN_COLLECTED_TESTS = 1

#: The scenario of the stop view that stops the stub child ``M``'s IOC.
STOP_SCENARIO = "m-stopped"
#: ``M``'s IOC uptime, the stop view's one IOC health address: a reading that
#: moves with the clock, so a value held from the stop shows as one that did not.
M_UPTIME = f"{STUB_CODE}:SIM:M:UPTIME"
#: A short read: a stopped IOC's channel answers nothing, so every read of it
#: runs to its deadline.
STOP_READ_S = 2.0
#: How long a held CA client may take to find a restarted IOC's channel again:
#: it searches for a lost channel with a growing backoff.
RESTART_TIMEOUT_S = 120.0

_CONNECTOR_OUTCOME = """
import asyncio, json, os, sys
from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

async def read(payload):
    register_builtin_connectors()
    connector = await ConnectorFactory.create_control_system_connector(
        payload["section"], control_target="va"
    )
    try:
        await connector.read_channel(payload["address"], timeout=payload["timeout"])
        return "answered"
    except Exception as exc:
        return type(exc).__name__
    finally:
        await connector.disconnect()

outcome = asyncio.run(read(json.load(sys.stdin)))
sys.stdout.write("\\nOSPREY_OUTCOME " + outcome + "\\n")
sys.stdout.flush()
os._exit(0)
"""


def _connector_outcome(server: str, address: str) -> str:
    """``answered``, or the class of what the ``virtual_accelerator`` connector raised."""
    gateway = {"address": "127.0.0.1", "port": int(CA_PORTS[server]), "use_name_server": False}
    section = {
        "type": "virtual_accelerator",
        "connector": {
            "virtual_accelerator": {"gateways": {"read_only": gateway, "write_access": gateway}}
        },
    }
    done = subprocess.run(
        [sys.executable, "-P", "-c", _CONNECTOR_OUTCOME],
        input=json.dumps({"section": section, "address": address, "timeout": STOP_READ_S}),
        capture_output=True,
        text=True,
        timeout=CA_TIMEOUT_S * 2,
        check=False,
    )
    for line in reversed(done.stdout.splitlines()):
        if line.startswith("OSPREY_OUTCOME "):
            return line.removeprefix("OSPREY_OUTCOME ")
    raise AssertionError(f"the connector read crashed: {(done.stdout + done.stderr)[-500:]}")


def _ca_try(address: str, timeout: float = CA_TIMEOUT_S) -> Any:
    """One CA read of ``address`` on the held client: the reading, ``ok`` or not.

    A held channel its server dropped is disconnected, and a read of it waits
    for a reconnect until ``timeout``: reads meant to find no answer pass the
    short ``STOP_READ_S``, or every one of them costs the module's whole deadline.
    """
    import aioca

    return _on_ca_loop(aioca.caget(address, timeout=timeout, throw=False))


def _ca_failed(address: str) -> str | None:
    """What a short CA read of ``address`` failed with, or ``None`` when it answered.

    The text, not the reading: a failed aioca reading is falsy, so a poll
    waiting for one would never see it arrive.
    """
    reading = _ca_try(address, STOP_READ_S)
    return None if reading.ok else str(reading)


def _no_answer(failure: str) -> bool:
    """Whether a CA read's failure is *no answer* -- a lost channel or a timeout, never a refusal."""
    text = failure.lower()
    return "disconnect" in text or "timeout" in text or "expired" in text


def _pva_try(server: _Server, address: str) -> str:
    """``answered``, or the class of what one short PVA read of ``address`` raised."""
    try:
        server.pva.get(address, timeout=STOP_READ_S)
    except Exception as exc:
        return type(exc).__name__
    return "answered"


def _activate(state_dir: Path, *names: str) -> None:
    """Write the active set, moving the file's modification time forward."""
    path = state_dir / "active_scenarios"
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    stamp = max(before + 1_000_000, path.stat().st_mtime_ns)
    os.utime(path, ns=(stamp, stamp))


@pytest.fixture(scope="class")
def stop_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Server]:
    root = tmp_path_factory.mktemp("stop")
    view_dir = _write_stub_view(root)
    scenarios = view_dir / "scenarios.json"
    document = json.loads(scenarios.read_text(encoding="utf-8"))
    document["scenarios"].append({"name": STOP_SCENARIO, "stopped_servers": ["sim-M"]})
    scenarios.write_text(json.dumps(document), encoding="utf-8")
    addresses = view_dir / "addresses.json"
    listed = json.loads(addresses.read_text(encoding="utf-8"))
    listed["server_health"] = [M_UPTIME]
    addresses.write_text(json.dumps(listed), encoding="utf-8")
    state_dir = root / "state"
    state_dir.mkdir()
    with _served(
        view_dir, "stop", state_dir=state_dir, tick_interval_s=STUB_TICK_S, stub=True
    ) as served:
        yield served


class TestStoppedServer:
    """A scenario's ``stopped_servers``: held clients on both wires get no answer, then answers."""

    def test_a_stopped_servers_channels_give_held_clients_no_answer_then_answer_again(
        self, stop_server: _Server
    ) -> None:
        assert stop_server.state_dir is not None
        stopped = (STUB_RANGE, STUB_SETPOINT, f"{STUB_CODE}:SIM:M:STATUS", M_UPTIME)
        # Connected on both wires first, at the module's own deadline: the
        # channels these clients hold are the ones the stop must silence.
        for address in (STUB_COUNT, *stopped):
            stop_server.pva_get(address)
            assert _ca_connects(address), f"{address} did not answer before the stop"
            assert _ca_try(address).ok
        uptime_before = float(_ca_try(M_UPTIME))
        read_at = time.monotonic()
        try:
            _activate(stop_server.state_dir, STOP_SCENARIO)
            for address in stopped:
                failure = _wait_until(lambda a=address: _ca_failed(a))
                assert failure is not None and _no_answer(failure), (
                    f"{address} on the held CA client: {failure}"
                )
                assert _pva_try(stop_server, address) == "TimeoutError"
            # Another IOC of the same server keeps answering on both wires.
            assert _ca_try(STUB_COUNT).ok
            assert _pva_try(stop_server, STUB_COUNT) == "answered"
            # A client that was not connected finds nothing either.
            assert _connector_outcome("stop", STUB_RANGE) == "ChannelUnreachableError"
        finally:
            _activate(stop_server.state_dir)
        restarted_after_s = time.monotonic() - read_at
        for address in stopped:
            # The held client searches again for its lost channel, on its own backoff.
            assert _wait_until(
                lambda a=address: _ca_try(a, STOP_READ_S).ok, timeout=RESTART_TIMEOUT_S
            ), f"{address} never answered again"
            assert _wait_until(lambda a=address: _pva_try(stop_server, a) == "answered")
        assert _connector_outcome("stop", STUB_RANGE) == "answered"
        # Answering again is the current value, on both wires: the uptime has
        # moved on by at least the time the IOC was down, never held from the stop.
        for uptime_after in (
            float(_ca_try(M_UPTIME)),
            float(stop_server.pva_get(M_UPTIME)["value"]),
        ):
            assert uptime_after >= uptime_before + restarted_after_s - 1.0, (
                f"{M_UPTIME} answered {uptime_after} after {restarted_after_s:.1f} s down, "
                f"from {uptime_before}"
            )


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
