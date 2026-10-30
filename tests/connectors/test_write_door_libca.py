"""The connector's write door over real libca, where the door opens.

The door opens in the process that holds the connector — the executor sandbox
and the connector-host child — and never in a notebook kernel, which loads no
client library at all. So this is the one real-libca proof of the door: a
child process installs the armed block, builds an ``EPICSConnector`` with
writes on against a soft IOC, and subscribes to the setpoint with ``aioca``.
A monitor update is delivered from the Channel Access thread into a fresh
context, so the put its callback makes finds the door closed and is refused
with ``RAW_CLIENT_WRITE``, while the connector's own put, which carries the
door into its thread hop, lands.

The block is cooperative attribution: it keeps honest code on the connector
path, and is not a sandbox (see the safety-chain page's cooperative bypasses).

This test process never loads a Channel Access client: the IOC and the client
each run in their own process.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import osprey_connectors
from osprey_connectors_tests._harness._control_context_fixtures import (
    VERDICT_ENV_VARS,
    pin_identity,
    write_control_context,
)
from osprey_connectors_tests.ipc._soft_ioc import SoftIOC

# Decided by lookup, never by import: importing either would load libca here.
pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("epicscorelibs", "aioca")),
    reason="needs epicscorelibs and aioca",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# The connectors under test, wherever they are installed: the workspace member
# here, the floor release in the framework-at-connectors-floor lane.
CONNECTORS_ROOT = Path(osprey_connectors.__file__).resolve().parents[1]
PYTHONPATH = os.pathsep.join([str(REPO_ROOT / "src"), str(CONNECTORS_ROOT)])

CA_TIMEOUT_S = 5.0
CHILD_TIMEOUT_S = 120.0

#: The child. It reads the setpoint's name from argv and its config from
#: ``CONFIG_FILE``, and prints one JSON line.
CHILD_SOURCE = """
import asyncio, json, os, sys, threading

setpoint = sys.argv[1]

from osprey.runtime import raw_put_block
from osprey.services.python_executor.execution.wrapper import armed_contract

raw_put_block.install("armed", **armed_contract(refuse_rpc=False))

import aioca
import yaml
from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors


async def main():
    with open(os.environ["CONFIG_FILE"]) as handle:
        section = yaml.safe_load(handle)["control_system"]
    register_builtin_connectors()
    connector = await ConnectorFactory.create_control_system_connector(
        section, control_target="live"
    )

    in_flight = threading.Event()
    first = asyncio.Event()
    landed = asyncio.Event()
    seen = []

    async def on_update(value):
        # Everything is caught here: an exception escaping an aioca callback
        # closes the subscription.
        try:
            await aioca.caput(setpoint, 99.0)
            entry = {"name": None, "reason": None}
        except BaseException as exc:
            entry = {"name": type(exc).__name__, "reason": getattr(exc, "reason", None)}
        entry["in_flight"] = in_flight.is_set()
        seen.append(entry)
        first.set()
        if float(value) == 3.0:
            landed.set()

    sub = aioca.camonitor(setpoint, on_update)
    await asyncio.wait_for(first.wait(), timeout=30)
    seen.clear()

    # The update the put causes is dispatched to this loop by libca's thread,
    # on no schedule tied to the write, and the confirming read can return
    # before it does. Holding that read until the update's callback has run
    # puts the callback inside the write every time.
    confirming_read = connector._confirming_read

    async def confirming_read_after_the_update(*args, **kwargs):
        await asyncio.wait_for(landed.wait(), timeout=30)
        return await confirming_read(*args, **kwargs)

    connector._confirming_read = confirming_read_after_the_update

    in_flight.set()
    try:
        result = await connector.write_channel(setpoint, 3.0, confirm=True)
    finally:
        in_flight.clear()
    await asyncio.sleep(0.5)
    sub.close()
    final = await aioca.caget(setpoint)
    print(json.dumps({
        "outcome": result.outcome.value,
        "error": result.error_message,
        "callbacks": seen,
        "final": float(final),
    }), flush=True)


asyncio.run(main())
"""


def _control_system(port: int) -> dict:
    """An ``epics`` section whose tier allows writes, both gateways at the soft IOC.

    No limits file: in the optional mode a channel with no record is written
    with no limits.
    """
    endpoint = {"address": "127.0.0.1", "port": port, "use_name_server": False}
    return {
        "type": "epics",
        "writes": "allowed",
        "connector": {
            "epics": {
                "timeout_s": CA_TIMEOUT_S,
                "gateways": {"read_only": dict(endpoint), "write_access": dict(endpoint)},
            },
        },
    }


#: Variables that would change the child's write verdict behind the test's back.
_VERDICT_ENV = (*VERDICT_ENV_VARS, "OSPREY_AGENT_DATA_ROOT")


def test_a_monitor_callbacks_put_is_refused_while_the_connectors_write_lands(
    tmp_path, ioc_prefix, monkeypatch
):
    setpoint = f"{ioc_prefix}:SP"
    # The login's switch on for the live machine the child's connector is
    # built for, under a scratch root the child is pointed at.
    pin_identity(monkeypatch)
    data_root = tmp_path / "data"
    write_control_context(data_root, target="live", owned_by=None, switch={"live": "on"})
    ioc = SoftIOC(tmp_path, "machine", ioc_prefix, 1.0)
    try:
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.safe_dump({"control_system": _control_system(ioc.port)}))
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("EPICS_CA", "EPICS_PVA")) and k not in _VERDICT_ENV
        }
        env.update(
            {
                "OSPREY_AGENT_DATA_ROOT": str(data_root),
                "PYTHONPATH": PYTHONPATH,
                "CONFIG_FILE": str(config_file),
                "EPICS_CA_ADDR_LIST": "127.0.0.1",
                "EPICS_CA_AUTO_ADDR_LIST": "NO",
                "EPICS_CA_SERVER_PORT": str(ioc.port),
            }
        )
        child = subprocess.run(
            [sys.executable, "-c", CHILD_SOURCE, setpoint],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=CHILD_TIMEOUT_S,
        )
    finally:
        ioc.stop()

    assert child.returncode == 0, f"child exited {child.returncode}:\n{child.stderr}"
    lines = [line for line in child.stdout.splitlines() if line.startswith("{")]
    assert lines, f"child printed no result:\nstdout: {child.stdout}\nstderr: {child.stderr}"
    seen = json.loads(lines[-1])

    assert seen["outcome"] == "confirmed", seen
    assert seen["error"] is None, seen
    # The confirming read is held until the put's own update has been handled,
    # so exactly that one callback runs, inside the write, and is refused.
    assert seen["callbacks"] == [
        {"name": "ChannelWriteBlockedError", "reason": "RAW_CLIENT_WRITE", "in_flight": True}
    ], seen
    assert seen["final"] == pytest.approx(3.0), seen
