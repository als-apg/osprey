"""The EPICS connector over real Channel Access, against soft IOCs on this host.

Everything here runs pvapy against a real IOC: reads (value, units, precision,
timestamp, alarm, enum labels), the confirming write and its put-callback
wait, a put-callback that outlasts the write's timeout, an access-security
denial, an unreachable channel, and a monitor delivering onto the event loop.
It also pins the exact pvapy exception texts the connector's error classifier
matches on, so a pvapy release that rewords them fails here rather than
quietly turning a refusal into an unclassified error (or the reverse).

The IOCs are ``python -m epicscorelibs.ioc``, as in
``tests/connectors/ipc/test_pool_soft_ioc.py``: no container, no EPICS base
build.

Why every scenario runs in a subprocess
---------------------------------------
pvapy reads ``EPICS_CA_*`` once, when the first Channel Access channel of the
process is created, and never again. One process can therefore talk to one
IOC port only — and the test process is shared with every other test an xdist
worker runs. So this process never creates a channel: each scenario runs this
same file as a script (``python -m tests.connectors.test_epics_soft_ioc
<scenario> <prefix>``), in a fresh interpreter whose environment names only
the IOC under test, and prints its observations as one JSON line.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHONPATH = os.pathsep.join(
    [
        str(REPO_ROOT),
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "packages" / "osprey-connectors" / "src"),
    ]
)

IOC_READY_TIMEOUT_S = 30.0
CA_TIMEOUT_S = 5.0
#: DLY1 of the slow ``seq`` record: how long its put-callback takes to fire.
SLOW_DELAY_S = 1.5
SCENARIO_TIMEOUT_S = 60.0
#: How long a scenario's interpreter may take to exit once its connector has
#: disconnected. pvapy, on macOS, suspends forever any thread that used it and
#: then exits; the connector's never-exiting daemon workers are what keep that
#: from hanging the loop's shutdown or the interpreter's. A regression shows up
#: here as a process that outlives this budget (or the scenario timeout).
EXIT_BUDGET_S = 5.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _database(prefix: str) -> str:
    return (
        f'record(ao, "{prefix}:SP") {{ field(VAL, "1.5") field(PREC, "3") field(EGU, "mA")'
        f' field(DESC, "setpoint") field(HOPR, "10") field(LOPR, "-10") field(PINI, "YES") }}\n'
        # A constant input above HIHI: processed at init, it sits in a MAJOR HIHI alarm.
        f'record(ai, "{prefix}:RB") {{ field(INP, "7") field(PREC, "2") field(EGU, "A")'
        f' field(HIHI, "5") field(HHSV, "MAJOR") field(PINI, "YES") }}\n'
        f'record(mbbo, "{prefix}:MODE") {{ field(ZRST, "OFF") field(ONST, "ON")'
        f' field(TWST, "STANDBY") field(VAL, "1") field(PINI, "YES") }}\n'
        # An asynchronous record: its put-callback fires DLY1 seconds after the put.
        f'record(seq, "{prefix}:SLOW") {{ field(DLY1, "{SLOW_DELAY_S}")'
        f' field(DOL1, "{prefix}:SLOW.VAL") field(LNK1, "{prefix}:SLOWRB PP") }}\n'
        f'record(ao, "{prefix}:SLOWRB") {{ }}\n'
    )


class SoftIOC:
    """One ``epicscorelibs`` soft IOC on its own port, optionally behind a read-only ACF."""

    def __init__(self, directory: Path, name: str, prefix: str, *, read_only: bool) -> None:
        self.port = _free_port()
        database = directory / f"{name}.db"
        database.write_text(_database(prefix))
        command = [sys.executable, "-m", "epicscorelibs.ioc", "-d", str(database)]
        if read_only:
            # An access-security file must be loaded before iocInit, which the
            # stock launcher offers no flag for; wrap iocInit instead.
            acf = directory / f"{name}.acf"
            acf.write_text("ASG(DEFAULT) {\n  RULE(1, READ)\n}\n")
            launcher = (
                "import sys, epicscorelibs.ioc as m\n"
                "_init = m.iocInit\n"
                f"m.iocInit = lambda: (m.ioc('asSetFilename(\"{acf}\")'), _init())[1]\n"
                f"sys.argv = ['ioc', '-d', {str(database)!r}]\n"
                "m.main()\n"
            )
            command = [sys.executable, "-c", launcher]
        self.log = directory / f"{name}.log"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("EPICS_CA", "EPICS_PVA"))}
        env.update(
            {
                "EPICS_CA_SERVER_PORT": str(self.port),
                "EPICS_CAS_INTF_ADDR_LIST": "127.0.0.1",
                "EPICS_CA_AUTO_ADDR_LIST": "NO",
            }
        )
        # stdin stays open: the IOC's interactive shell exits the IOC on EOF.
        with self.log.open("wb") as output:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, env=env
            )
        self._wait_until_serving()

    def _wait_until_serving(self) -> None:
        deadline = time.monotonic() + IOC_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"soft IOC exited with {self.process.returncode}: "
                    f"{self.log.read_text(errors='replace')}"
                )
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.1)
        self.stop()
        raise RuntimeError(f"soft IOC never listened on 127.0.0.1:{self.port}")

    def stop(self) -> None:
        self.process.kill()
        self.process.wait(timeout=10)


@pytest.fixture(scope="module")
def prefix() -> str:
    """A unique prefix, so no other IOC on this host serves the names read here."""
    return f"SIOC{uuid.uuid4().hex[:8].upper()}"


@pytest.fixture(scope="module")
def ioc(tmp_path_factory, prefix):
    server = SoftIOC(tmp_path_factory.mktemp("ioc"), "rw", prefix, read_only=False)
    yield server
    server.stop()


@pytest.fixture(scope="module")
def read_only_ioc(tmp_path_factory, prefix):
    server = SoftIOC(tmp_path_factory.mktemp("ro_ioc"), "ro", prefix, read_only=True)
    yield server
    server.stop()


def _drive(tmp_path: Path, port: int, scenario: str, prefix: str) -> dict[str, Any]:
    """Run one scenario in a fresh interpreter pointed at ``port`` alone."""
    config = tmp_path / "config.yml"
    config.write_text(
        "control_system:\n"
        "  type: epics\n"
        "  writes_enabled: true\n"
        "  limits_checking:\n"
        "    enabled: false\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EPICS_CA", "EPICS_PVA"))}
    env.update(
        {
            "PYTHONPATH": PYTHONPATH,
            "CONFIG_FILE": str(config),
            "SOFT_IOC_PORT": str(port),
        }
    )
    env.pop("OSPREY_EXECUTION_MODE", None)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "tests.connectors.test_epics_soft_ioc", scenario, prefix],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=SCENARIO_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"scenario {scenario} never exited (a hung pvapy thread?); "
            f"it printed: {exc.stdout!r}\n{exc.stderr!r}"
        )
    returned_at = time.time()
    assert result.returncode == 0, f"scenario {scenario} failed:\n{result.stderr}"
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    assert lines, f"scenario {scenario} printed no result:\n{result.stdout}\n{result.stderr}"
    seen = json.loads(lines[-1])
    # The result is printed from inside the event loop, after disconnect() and
    # before asyncio.run() shuts the loop down; everything after it is teardown.
    exit_s = returned_at - seen.pop("disconnected_at")
    assert exit_s < EXIT_BUDGET_S, f"scenario {scenario} took {exit_s:.1f}s to exit"
    return seen


# ---------------------------------------------------------------------------
# Scenarios: run in the child interpreter, never in the pytest process.
# ---------------------------------------------------------------------------


async def _connector():
    from osprey_connectors.control_system.epics_connector import EPICSConnector

    connector = EPICSConnector()
    endpoint = {
        "address": "127.0.0.1",
        "port": int(os.environ["SOFT_IOC_PORT"]),
        "use_name_server": False,
    }
    await connector.connect(
        {
            "timeout": CA_TIMEOUT_S,
            "gateways": {"read_only": dict(endpoint), "write_access": dict(endpoint)},
        }
    )
    return connector


def _meta(value) -> dict[str, Any]:
    metadata = value.metadata
    return {
        "value": value.value,
        "units": metadata.units,
        "precision": metadata.precision,
        "description": metadata.description,
        "alarm_status": metadata.alarm_status,
        "alarm_severity": metadata.alarm_severity,
        "enum_label": metadata.enum_label,
        "enum_labels": metadata.enum_labels,
        "timestamp_age_s": time.time() - value.timestamp.timestamp(),
        "timestamp_tz": value.timestamp.tzinfo is not None,
    }


def _write(result, elapsed: float) -> dict[str, Any]:
    return {
        "outcome": str(result.outcome.value),
        "refusal_reason": result.refusal_reason,
        "observed_value": result.observed_value,
        "error_message": result.error_message,
        "elapsed_s": elapsed,
    }


async def scenario_reads(prefix: str) -> dict[str, Any]:
    connector = await _connector()
    try:
        return {
            "sp": _meta(await connector.read_channel(f"{prefix}:SP")),
            "rb": _meta(await connector.read_channel(f"{prefix}:RB")),
            "mode": _meta(await connector.read_channel(f"{prefix}:MODE")),
            "valid": await connector.validate_channel(f"{prefix}:SP"),
        }
    finally:
        await connector.disconnect()


async def scenario_writes(prefix: str) -> dict[str, Any]:
    connector = await _connector()
    out: dict[str, Any] = {}
    try:
        cases = [
            ("confirmed", f"{prefix}:SP", 2.5, True, CA_TIMEOUT_S),
            ("enum_label", f"{prefix}:MODE", "STANDBY", True, CA_TIMEOUT_S),
            ("unrequested", f"{prefix}:SP", 4.25, False, CA_TIMEOUT_S),
            ("slow", f"{prefix}:SLOW", 5.0, True, CA_TIMEOUT_S),
            ("slow_timeout", f"{prefix}:SLOW", 6.0, True, 0.5),
        ]
        for name, address, value, confirm, timeout in cases:
            start = time.monotonic()
            result = await connector.write_channel(address, value, timeout=timeout, confirm=confirm)
            out[name] = _write(result, time.monotonic() - start)
        await asyncio.sleep(SLOW_DELAY_S + 0.5)  # let the abandoned put-callback land
        out["sp_after"] = (await connector.read_channel(f"{prefix}:SP")).value
        out["slow_after"] = (await connector.read_channel(f"{prefix}:SLOW")).value
    finally:
        await connector.disconnect()
    return out


async def scenario_denied(prefix: str) -> dict[str, Any]:
    connector = await _connector()
    try:
        read = (await connector.read_channel(f"{prefix}:SP")).value
        out = {"read": read}
        for name, confirm in (("confirming", True), ("plain", False)):
            start = time.monotonic()
            result = await connector.write_channel(f"{prefix}:SP", 3.0, confirm=confirm)
            out[name] = _write(result, time.monotonic() - start)
        out["after"] = (await connector.read_channel(f"{prefix}:SP")).value
        return out
    finally:
        await connector.disconnect()


async def scenario_unreachable(prefix: str) -> dict[str, Any]:
    connector = await _connector()
    out: dict[str, Any] = {}
    try:
        for name, call in (
            ("read", lambda: connector.read_channel(f"{prefix}:NOPE", timeout=1.0)),
            ("write", lambda: connector.write_channel(f"{prefix}:NOPE", 1.0, timeout=1.0)),
        ):
            try:
                await call()
                out[name] = "no error"
            except Exception as exc:
                out[name] = type(exc).__name__
        out["valid"] = await connector.validate_channel(f"{prefix}:NOPE")
    finally:
        await connector.disconnect()
    return out


async def scenario_subscribe(prefix: str) -> dict[str, Any]:
    connector = await _connector()
    loop_thread = threading.get_ident()
    updates: list[dict[str, Any]] = []
    try:
        sub_id = await connector.subscribe(
            f"{prefix}:SP",
            lambda value: updates.append(
                {
                    "value": value.value,
                    "units": value.metadata.units,
                    "on_loop": threading.get_ident() == loop_thread,
                }
            ),
        )
        await asyncio.sleep(0.5)
        initial = len(updates)
        await connector.write_channel(f"{prefix}:SP", 7.75, confirm=False)
        await asyncio.sleep(1.0)
        await connector.unsubscribe(sub_id)
        stopped_at = len(updates)
        await connector.write_channel(f"{prefix}:SP", 8.5, confirm=False)
        await asyncio.sleep(0.5)
        return {
            "initial": initial,
            "updates": updates,
            "after_unsubscribe": len(updates) - stopped_at,
        }
    finally:
        await connector.disconnect()


async def scenario_messages(prefix: str) -> dict[str, Any]:
    """The raw pvapy texts the connector's classifier depends on."""
    import pvaccess

    from osprey_connectors.control_system.epics_connector import _classify_client_error

    await _connector()  # sets the CA environment exactly as production does
    out: dict[str, Any] = {}
    missing = pvaccess.Channel(f"{prefix}:NOPE", pvaccess.CA)
    missing.setTimeout(1.0)
    try:
        missing.get("field(value)")
    except pvaccess.PvaException as exc:
        out["timed_out"] = str(exc)
        out["timed_out_class"] = _classify_client_error(exc, pvaccess.PvaException)
    if os.environ.get("SOFT_IOC_READ_ONLY"):
        channel = pvaccess.Channel(f"{prefix}:SP", pvaccess.CA)
        channel.setTimeout(CA_TIMEOUT_S)
        try:
            channel.put(3.0, "record[block=true]field(value)")
        except pvaccess.PvaException as exc:
            out["denied"] = str(exc)
            out["denied_class"] = _classify_client_error(exc, pvaccess.PvaException)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reads_map_value_metadata_and_enum(ioc, prefix, tmp_path):
    seen = _drive(tmp_path, ioc.port, "reads", prefix)

    sp = seen["sp"]
    assert sp["value"] == 1.5
    assert sp["units"] == "mA"
    assert sp["precision"] == 3  # from display.format, e.g. "F9.3"
    # Channel Access carries no DESC: the display structure's description is
    # empty, which maps to "not reported" rather than to an empty string.
    assert sp["description"] is None
    assert sp["alarm_status"] == "NO_ALARM"
    assert sp["alarm_severity"] == 0
    assert sp["timestamp_tz"] is True
    assert 0 <= sp["timestamp_age_s"] < 120  # the record's own processing time, not 1970

    rb = seen["rb"]
    assert rb["value"] == 7.0
    assert rb["units"] == "A"
    assert rb["precision"] == 2
    assert rb["alarm_status"] == "HIHI"
    assert rb["alarm_severity"] == 2  # MAJOR

    mode = seen["mode"]
    assert mode["value"] == 1  # the index is the value
    assert mode["enum_label"] == "ON"
    assert mode["enum_labels"] == ["OFF", "ON", "STANDBY"]

    assert seen["valid"] is True


def test_writes_confirm_wait_for_the_callback_and_time_out_unconfirmed(ioc, prefix, tmp_path):
    seen = _drive(tmp_path, ioc.port, "writes", prefix)

    assert seen["confirmed"]["outcome"] == "confirmed"
    assert seen["confirmed"]["observed_value"] == 2.5

    # An enum written by label is confirmed against the label it now reads.
    assert seen["enum_label"]["outcome"] == "confirmed"
    assert seen["enum_label"]["observed_value"] == 2

    assert seen["unrequested"]["outcome"] == "unrequested"
    assert seen["unrequested"]["observed_value"] is None
    assert seen["sp_after"] == 4.25

    # The confirming put waited for the seq record's delayed put-callback.
    slow = seen["slow"]
    assert slow["outcome"] == "confirmed"
    assert slow["observed_value"] == 5.0
    assert slow["elapsed_s"] >= SLOW_DELAY_S - 0.05

    # A put-callback that outlasts the timeout: sent, never acknowledged in time.
    late = seen["slow_timeout"]
    assert late["outcome"] == "unconfirmed"
    assert "did not acknowledge" in late["error_message"]
    assert late["elapsed_s"] < SLOW_DELAY_S
    assert seen["slow_after"] == 6.0  # it was sent, and the IOC did take it


def test_access_security_denial_is_a_control_system_refusal(read_only_ioc, prefix, tmp_path):
    seen = _drive(tmp_path, read_only_ioc.port, "denied", prefix)

    assert seen["read"] == 1.5
    for name in ("confirming", "plain"):
        assert seen[name]["outcome"] == "refused", seen[name]
        assert seen[name]["refusal_reason"] == "CONTROL_SYSTEM_REFUSED"
    assert seen["after"] == 1.5


def test_unreachable_channel_is_a_connection_error(ioc, prefix, tmp_path):
    seen = _drive(tmp_path, ioc.port, "unreachable", prefix)

    assert seen == {"read": "ConnectionError", "write": "ConnectionError", "valid": False}


def test_subscribe_delivers_updates_on_the_loop(ioc, prefix, tmp_path):
    seen = _drive(tmp_path, ioc.port, "subscribe", prefix)

    assert seen["initial"] >= 1  # the monitor's first update is the current value
    assert all(update["on_loop"] for update in seen["updates"])
    assert 7.75 in [update["value"] for update in seen["updates"]]
    assert all(update["units"] == "mA" for update in seen["updates"])
    assert seen["after_unsubscribe"] == 0


def test_pvapy_message_texts_the_classifier_matches(read_only_ioc, prefix, tmp_path, monkeypatch):
    """A pvapy upgrade that rewords these must fail here, not in production."""
    monkeypatch.setenv("SOFT_IOC_READ_ONLY", "1")
    seen = _drive(tmp_path, read_only_ioc.port, "messages", prefix)

    assert seen["timed_out"] == f"Channel {prefix}:NOPE timed out."
    assert seen["timed_out_class"] == "unreachable"
    assert seen["denied"] == f"channel {prefix}:SP PvaClientPut::put Write access denied"
    assert seen["denied_class"] == "access_denied"


async def _main(scenario: str, prefix: str) -> None:
    """Run one scenario and print its result before the loop and process wind down."""
    result = await globals()[f"scenario_{scenario}"](prefix)
    result["disconnected_at"] = time.time()
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1], sys.argv[2]))
