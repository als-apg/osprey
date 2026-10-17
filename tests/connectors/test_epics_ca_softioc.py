"""The EPICS connector over real Channel Access, against a soft IOC.

The unit tests drive the connector through a fake aioca; this file holds it
to the real one. The IOC is ``python -m epicscorelibs.ioc``-equivalent — the
minimal soft IOC that ships with ``epicscorelibs`` — started with an access
security file, so a denied write is a real one. Every scenario runs the
connector in a child process (``_epics_ca_driver.py``): a process has one
Channel Access context, which each connect repoints, and what happens at
interpreter exit is part of what is tested.

The exit scenarios are the regression test for the crash that made aioca the
connector's only Channel Access client: with two clients in one process, the
one that used CA second adopted the other's context, which was then destroyed
under it at exit.
"""

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("aioca", reason="the EPICS connector's Channel Access client")
pytest.importorskip("epicscorelibs.ioc", reason="serves the soft IOC")

DRIVER = Path(__file__).with_name("_epics_ca_driver.py")
REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHONPATH = os.pathsep.join(
    [
        str(REPO_ROOT),
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "packages" / "osprey-connectors" / "src"),
    ]
)
RESULT_MARKER = "__RESULT__"
IOC_READY_TIMEOUT_S = 30.0
DRIVER_TIMEOUT_S = 120.0

_IOC_SCRIPT = """
import sys, time
import epicscorelibs.ioc as ioc
ioc.iocshRegisterCommon()
ioc.ioc('asSetFilename("' + sys.argv[1] + '")')
ioc.start_ioc(database=sys.argv[2])
while True:
    time.sleep(3600)
"""

_ACF = """
ASG(DEFAULT) { RULE(1, READ) RULE(1, WRITE) }
ASG(RO) { RULE(1, READ) }
"""


def _database(prefix: str) -> str:
    return f"""
record(ao, "{prefix}:AO") {{ field(VAL, "1.5") field(PREC, "3") field(EGU, "mA")
    field(HIHI, "8") field(HHSV, "MAJOR") }}
record(longout, "{prefix}:LONG") {{ field(VAL, "3") }}
record(mbbo, "{prefix}:MBBO") {{ field(ZRST, "Off") field(ONST, "On") field(VAL, "1") }}
record(stringout, "{prefix}:STR") {{ field(VAL, "hello") }}
record(ao, "{prefix}:SLOW") {{ field(FLNK, "{prefix}:SLOWCALC") }}
record(calcout, "{prefix}:SLOWCALC") {{ field(CALC, "A") field(INPA, "{prefix}:SLOW")
    field(ODLY, "3") }}
record(ao, "{prefix}:RO") {{ field(ASG, "RO") field(VAL, "2") }}
record(waveform, "{prefix}:WF") {{ field(FTVL, "CHAR") field(NELM, "64") }}
record(ao, "{prefix}:DISABLED") {{ field(DISV, "1") field(DISA, "1") field(DISP, "1") }}
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _ca_env(port: int) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EPICS_CA", "EPICS_PVA"))}
    env.update(
        {
            "EPICS_CA_SERVER_PORT": str(port),
            "EPICS_CA_ADDR_LIST": "127.0.0.1",
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
            "EPICS_CAS_INTF_ADDR_LIST": "127.0.0.1",
        }
    )
    return env


@pytest.fixture
def prefix() -> str:
    """One prefix per test, so no other worker's IOC can answer for these names."""
    return f"CA{uuid.uuid4().hex[:8].upper()}"


@pytest.fixture
def ioc(tmp_path, prefix):
    """A soft IOC with access security on a port of its own; yields its CA env."""
    port = _free_port()
    database = tmp_path / "test.db"
    database.write_text(_database(prefix))
    acf = tmp_path / "test.acf"
    acf.write_text(_ACF)
    log = tmp_path / "ioc.log"
    env = _ca_env(port)
    with log.open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, "-c", _IOC_SCRIPT, str(acf), str(database)],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
        )
    try:
        deadline = time.monotonic() + IOC_READY_TIMEOUT_S
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"soft IOC exited: {log.read_text(errors='replace')}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"soft IOC never listened on {port}") from None
                time.sleep(0.1)
        yield env
    finally:
        process.kill()
        process.wait(timeout=10)


def _drive(scenario: str, prefix: str, env: dict[str, str], marker: str = RESULT_MARKER) -> dict:
    """Run one driver scenario; assert a clean exit and return the line ``marker`` heads."""
    child_env = dict(env, PYTHONPATH=PYTHONPATH)
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(DRIVER), scenario, prefix],
        capture_output=True,
        text=True,
        env=child_env,
        timeout=DRIVER_TIMEOUT_S,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, (
        f"{scenario} exited {completed.returncode}\n{completed.stdout}\n{completed.stderr}"
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    assert lines, f"{scenario} printed no {marker}\n{completed.stdout}\n{completed.stderr}"
    return json.loads(lines[-1][len(marker) :])


def test_reads_writes_and_monitors_over_the_wire(ioc, prefix):
    result = _drive("contract", prefix, ioc)
    reads, writes = result["reads"], result["writes"]

    assert reads["AO"]["value"] == 1.5
    assert reads["AO"]["type"] == "float"
    assert reads["AO"]["units"] == "mA"
    assert reads["AO"]["precision"] == 3
    assert reads["LONG"]["value"] == 3
    assert reads["LONG"]["type"] == "int"
    assert reads["MBBO"]["value"] == 1
    assert reads["MBBO"]["enum_label"] == "On"
    assert reads["MBBO"]["enum_labels"] == ["Off", "On"]
    assert reads["STR"]["value"] == "hello"
    assert reads["NOPE"] == "ConnectionError"

    assert writes["confirmed"]["outcome"] == "CONFIRMED"
    assert writes["confirmed"]["observed"] == 2.25
    assert writes["into_alarm"]["outcome"] == "CONFIRMED"
    assert writes["into_alarm"]["alarm_status"] == "HIHI"
    assert writes["enum_label"]["outcome"] == "CONFIRMED"
    assert writes["enum_label"]["observed"] == 0
    assert writes["unrequested"]["outcome"] == "UNREQUESTED"
    assert writes["not_sent"]["outcome"] == "FAILED"
    assert "no value was sent" in writes["not_sent"]["error"]
    assert writes["unacknowledged"]["outcome"] == "UNCONFIRMED"
    assert writes["access_denied"]["outcome"] == "REFUSED"
    assert writes["access_denied"]["refusal_reason"] == "CONTROL_SYSTEM_REFUSED"

    # Text is read by the rule for the channel's own field type before it is
    # sent, never left to the IOC, which would store "1e3" in a longout as 1.
    assert writes["text_hex_to_long"]["outcome"] == "CONFIRMED"
    assert writes["text_hex_to_long"]["observed"] == 16
    assert writes["text_exponent_to_long"]["outcome"] == "REFUSED"
    assert writes["text_exponent_to_long"]["refusal_reason"] == "VALIDATION_ERROR"
    # A number the integer field cannot hold exactly is refused unsent, never
    # cut down by the IOC: 1.5 would land as 1, 2**31 as -2147483648.
    for refused in ("fraction_to_long", "overflow_to_long"):
        assert writes[refused]["outcome"] == "REFUSED", writes[refused]
        assert writes[refused]["refusal_reason"] == "VALIDATION_ERROR"
    assert result["long_after_refusal"] == 16
    assert writes["text_exponent_to_ao"]["outcome"] == "CONFIRMED"
    assert writes["text_exponent_to_ao"]["observed"] == 2.5
    assert writes["text_to_char_waveform"]["outcome"] == "CONFIRMED"
    assert writes["text_to_char_waveform"]["observed"] == "/data/run42"
    assert writes["disabled_record"]["outcome"] == "FAILED"

    # The monitor saw the write made while it was open, and nothing after close.
    assert 4.5 in result["monitor"]
    assert 5.5 not in result["monitor"]
    assert result["max_step"] == [5.5, None]


def test_a_fresh_loop_per_call_reuses_one_channel_cache(ioc, prefix):
    """osprey.runtime's asyncio.run per call must not reconnect or leak per call."""
    result = _drive("runtime_style", prefix, ioc)

    assert result["reads"] == 20
    assert result["channel_caches"] == 1


@pytest.mark.parametrize("scenario", ["aioca_first", "connector_first"])
def test_aioca_and_the_connector_share_one_context_and_exit_cleanly(ioc, prefix, scenario):
    result = _drive(scenario, prefix, ioc)

    assert result["write"] == "CONFIRMED"


def test_the_dispatch_source_and_the_connector_share_the_ca_loop(ioc, prefix):
    result = _drive("dispatch", prefix, ioc)

    assert result["fired"] == [[1.0, 6.0], [2.0, 7.5]]


def test_the_ca_loop_is_stopped_before_aioca_tears_channel_access_down(ioc, prefix):
    """aioca destroys the CA context from an ``atexit`` hook; a loop still running
    CA work under it crashed the process at exit, intermittently."""
    result = _drive("stopped_before_teardown", prefix, ioc, marker="__ATEXIT__")

    assert result == {"ca_loop_alive": False}


def test_an_unreachable_name_server_does_not_hold_the_exit(ioc, prefix):
    """ca_context_destroy waits out the circuit to a dead name server: 30 s of exit."""
    started = time.monotonic()
    result = _drive("unreachable_name_server", prefix, ioc)

    assert result == {"read": "ConnectionError"}
    assert time.monotonic() - started < 15


def test_a_connect_to_another_gateway_repoints_the_process(ioc, prefix):
    """A notebook kernel follows a target switch in-process: each connect moves CA."""
    result = _drive("repoint", prefix, ioc)

    assert result["moved"] == "ConnectionError", result
    assert result["write"] == "CONFIRMED", result
    assert result["after"] == 4.0, result
    # aioca closes every monitor with the context it drops; the connector
    # re-opens its own on the new one, so the subscriber keeps hearing writes.
    assert 4.0 in result["monitor_after_repoint"], result


def test_a_dispatch_trigger_stays_armed_across_a_repoint(ioc, prefix):
    """The dispatch source watches through the connector, so its monitor comes back too."""
    result = _drive("dispatch_repoint", prefix, ioc)

    assert result["fired"] == [[1.0, 6.0]], result


def test_the_archiver_recorder_samples_through_the_ca_loop(ioc, prefix):
    """A channel nobody serves costs only itself, within the cadence, and every
    sample, whatever loop it runs on, reuses the one channel cache of the CA loop."""
    result = _drive("recorder", prefix, ioc)

    for sample in result["samples"]:
        assert sample == {"AO": 1.5, "LONG": 3, "MBBO": 1}, result
    assert all(seconds < 2.0 for seconds in result["elapsed"]), result
    assert result["ca_loop_caches"] == [True], result
