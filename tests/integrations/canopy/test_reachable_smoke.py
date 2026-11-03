"""``OspreyPVConnector.reachable`` over a real ``EPICSConnector`` with no name server.

The adapter's statement comes from its own channels only: a drop it saw reads
``no`` after the hold-down, never ``unknown``, with no name server to ask. A
soft IOC (``osprey_connectors_tests._soft_ioc``) serves one record; a client in its own
process and session opens a control session over a real ``EPICSConnector``
searching by UDP, puts the adapter on it and subscribes, then asks this process
to kill the IOC and watches the statement until it reads ``no``.

The client runs as ``python -m tests.integrations.canopy.test_reachable_smoke
<json>`` and ends with ``os._exit`` (libca's threads may still be running).

No wall-clock budget is asserted: the waits are liveness deadlines, and the only
timing claim is a lower bound (the ``no`` comes no sooner than the hold-down
after the kill), which a loaded host can only make easier to meet.
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
from pathlib import Path
from typing import Any

# Marks the module as a CANOPY contract test: the mark is derived from imports.
# Its client process runs CANOPY's PV contract through the backend.
import canopy  # noqa: F401
import pytest

from osprey_connectors.control_system import _epics_connector as epics_connector_module

# xdist_group("soft-ioc"): its soft IOC probes ports free and binds them a moment
# later, so one worker keeps that window free of a second prober.
pytestmark = pytest.mark.xdist_group("soft-ioc")

REPO = Path(__file__).resolve().parents[3]

SERVED = "RS:SERVED"

#: Liveness: the subscription must connect at all.
CONNECT_DEADLINE_S = 20.0

#: Liveness: after the kill the statement must reach ``no`` at all.
NO_DEADLINE_S = 30.0

#: How long the client keeps reading after ``no``: it must not fall back.
AFTER_NO_S = 2.0

#: How often the client reads the statement.
POLL_S = 0.05

#: The outer bound on the client run: its parts plus interpreter start-up and teardown.
CLIENT_TIMEOUT_S = CONNECT_DEADLINE_S + NO_DEADLINE_S + AFTER_NO_S + 30.0


@pytest.fixture
def ioc(tmp_path):
    from osprey_connectors_tests._soft_ioc import SoftIOC, ai_record

    with SoftIOC([ai_record(SERVED, 1.5)], workdir=tmp_path) as running:
        yield running


def _client_env(ioc) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("EPICS_")}
    env.update(ioc.client_env(name_server=False))
    # The connectors' test package, as pytest's ``pythonpath`` gives it here.
    tests = REPO / "packages" / "osprey-connectors" / "tests"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO), str(tests), env.get("PYTHONPATH")])
    )
    return env


def run_client(ioc) -> dict[str, Any]:
    """Run the client against ``ioc``; serve its kill request; return its result."""
    from osprey_connectors_tests._soft_ioc import kill_group

    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.integrations.canopy.test_reachable_smoke", json.dumps({})],
        cwd=REPO,
        env=_client_env(ioc),
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
    deadline = time.monotonic() + CLIENT_TIMEOUT_S
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"client timed out; stderr:\n{''.join(stderr)}")
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                proc.wait(timeout=10)
                raise AssertionError(
                    f"client exited {proc.returncode} without a result; stderr:\n{''.join(stderr)}"
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


def test_a_killed_ioc_reads_no_after_the_hold_down_never_unknown(ioc):
    result = run_client(ioc)
    assert result["name_server"] is None, "UDP search: the connector has no name server"
    assert result["before_kill"] == ["yes"], "the soft IOC was reached"
    after = result["after_kill"]
    assert after[-1] == "no", f"the kill never read no: {after}"
    assert "unknown" not in after, f"a drop the adapter saw read unknown: {after}"
    assert set(after) <= {"yes", "no"}
    assert result["no_after_kill_s"] >= epics_connector_module.STALE_HOLD_DOWN_S
    assert isinstance(result["since"], float)
    assert result["killed_at"] <= result["since"] <= result["no_at"], "since is the drop"
    assert result["since_steady"], "since is stamped once"


# ---------------------------------------------------------------------------
# Client side (runs in its own process)
# ---------------------------------------------------------------------------


async def _ask(cmd: str) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    await asyncio.to_thread(sys.stdin.readline)


async def _client() -> dict[str, Any]:
    from osprey.integrations.canopy.backend import OspreyPVConnector
    from osprey.runtime.session import STATE_BOUND, Binding, ControlSession
    from osprey_connectors.control_system._epics_connector import EPICSConnector
    from osprey_connectors.posture_store import WriteVerdict

    connector = EPICSConnector()
    await connector.connect({"timeout_s": 3.0})
    session = ControlSession(
        surface="canopy",
        binding=Binding("ioc", 1, "epics", "Soft IOC", STATE_BOUND),
        connector=connector,
        record_key=None,
        verdict=WriteVerdict(allowed=False, reason="switch_off", target="ioc", kind=None),
    )
    adapter = OspreyPVConnector(connect_report_s=5.0)
    adapter._loop = asyncio.get_running_loop()
    adapter._session = session
    await adapter.subscribe(SERVED, lambda _event: None)

    deadline = time.monotonic() + CONNECT_DEADLINE_S
    while adapter.reachable["state"] != "yes" and time.monotonic() < deadline:
        await asyncio.sleep(POLL_S)
    before_kill = [adapter.reachable["state"]]

    # Taken before asking: the drop may be told before the kill is acknowledged.
    killed_mono, killed_at = time.monotonic(), time.time()
    await _ask("kill")
    after_kill: list[str] = []
    no_mono = no_at = since = None
    deadline = killed_mono + NO_DEADLINE_S
    while time.monotonic() < deadline:
        reachable = adapter.reachable
        if not after_kill or after_kill[-1] != reachable["state"]:
            after_kill.append(reachable["state"])
        if reachable["state"] == "no":
            no_mono, no_at, since = time.monotonic(), time.time(), reachable["since"]
            break
        await asyncio.sleep(POLL_S)
    sinces = set()
    watch_until = time.monotonic() + AFTER_NO_S
    while no_mono is not None and time.monotonic() < watch_until:
        reachable = adapter.reachable
        if after_kill[-1] != reachable["state"]:
            after_kill.append(reachable["state"])
        sinces.add(reachable["since"])
        await asyncio.sleep(POLL_S)
    return {
        "name_server": connector._name_server,
        "before_kill": before_kill,
        "after_kill": after_kill,
        "no_after_kill_s": None if no_mono is None else no_mono - killed_mono,
        "killed_at": killed_at,
        "no_at": no_at,
        "since": since,
        "since_steady": sinces == {since},
    }


def _client_main(argv: list[str]) -> None:
    json.loads(argv[1])
    result = asyncio.run(_client())
    print(json.dumps({"result": result}), flush=True)
    # Skip interpreter teardown: libca's threads (the CA loop's) may still be running.
    os._exit(0)


if __name__ == "__main__":
    _client_main(sys.argv)
