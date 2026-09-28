"""The EPICS CA trigger source against a real soft IOC.

The unit tests in ``test_epics_ca_source.py`` fake ``pvaccess.Channel``; this
one checks the parts a fake cannot: that pvapy's CA provider delivers the
connect-time update the watcher suppresses, that a put crossing the threshold
reaches the fire callback, and that an enum record arrives as its index.

The IOC is ``python -m epicscorelibs.ioc`` (the pattern of
``tests/connectors/ipc/test_pool_soft_ioc.py``). The watcher runs in a child
process because pvapy fixes the ``EPICS_CA_*`` settings when the first CA
channel of a process is created: this test process must not pin them to one
throwaway IOC's port for every later test in the worker.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("pvaccess")
pytest.importorskip("epicscorelibs")

REPO_ROOT = Path(__file__).resolve().parents[2]
IOC_READY_TIMEOUT_S = 30.0
CHILD_TIMEOUT_S = 60.0

#: Run in the child: arm the source on two triggers, drive both PVs across
#: their thresholds with CA puts, and print what the fire callback received.
_CHILD = textwrap.dedent(
    """
    import asyncio, json, sys
    import pvaccess
    from osprey.dispatch.sources.epics_ca import EpicsCaSource
    from osprey.dispatch.trigger_config import TriggerConfig

    P = sys.argv[1]

    def trigger(name, pv):
        return TriggerConfig(
            name=name, source="epics_ca", action={"prompt": "x"},
            source_config={"pv": pv, "threshold": 1.0, "edge": "rising",
                           "cool_down_sec": 0.0},
        )

    async def main():
        fired = []
        async def fire(trig, payload):
            fired.append({"trigger": trig.name, **payload})
        source = EpicsCaSource()
        await source.start(
            [trigger("level", P + ":LEVEL"), trigger("state", P + ":STATE")], fire
        )
        await asyncio.sleep(2.0)  # connect; the initial updates are suppressed
        assert fired == [], fired
        pvaccess.Channel(P + ":LEVEL", pvaccess.CA).put(5.0)
        pvaccess.Channel(P + ":STATE", pvaccess.CA).putInt(1)
        for _ in range(100):
            if len(fired) >= 2:
                break
            await asyncio.sleep(0.1)
        await source.stop()
        print(json.dumps(fired))

    asyncio.run(main())
    """
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _ca_env(port: int) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EPICS_CA", "EPICS_PVA"))}
    env.update(
        {
            "EPICS_CA_SERVER_PORT": str(port),
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
        }
    )
    return env


@pytest.fixture
def soft_ioc(tmp_path: Path):
    """A soft IOC on a free port serving one ``ao`` and one ``bo`` record.

    The names carry a per-test prefix, so an IOC this test did not start (another
    xdist worker's) serves none of them and a mix-up fails as "never fired"
    rather than as somebody else's plausible value.
    """
    port = _free_port()
    prefix = f"DSP{uuid.uuid4().hex[:8].upper()}"
    database = tmp_path / "live.db"
    database.write_text(
        f'record(ao, "{prefix}:LEVEL") {{ field(VAL, "0") }}\n'
        f'record(bo, "{prefix}:STATE") {{ field(ZNAM, "OFF") field(ONAM, "ON") field(VAL, "0") }}\n'
    )
    log = tmp_path / "ioc.log"
    env = _ca_env(port) | {"EPICS_CAS_INTF_ADDR_LIST": "127.0.0.1"}
    # stdin stays open: the IOC's interactive shell exits the IOC on EOF.
    with log.open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "epicscorelibs.ioc", "-d", str(database)],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
        )
    try:
        deadline = time.monotonic() + IOC_READY_TIMEOUT_S
        while True:
            if process.poll() is not None or time.monotonic() > deadline:
                pytest.fail(f"soft IOC did not start:\n{log.read_text(errors='replace')}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.1)
        yield port, prefix
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_threshold_crossings_fire_over_real_channel_access(soft_ioc: tuple[int, str]) -> None:
    port, prefix = soft_ioc
    env = _ca_env(port) | {"EPICS_CA_ADDR_LIST": "127.0.0.1"}
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, prefix],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=CHILD_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stderr
    fired = {
        event["trigger"]: event for event in json.loads(result.stdout.strip().splitlines()[-1])
    }

    assert set(fired) == {"level", "state"}
    assert fired["level"]["value"] == 5.0
    assert fired["level"]["previous_value"] == 0.0
    # The bo record arrives as index + choices; the watcher thresholds the index.
    assert fired["state"]["value"] == 1.0
    assert fired["state"]["previous_value"] == 0.0
