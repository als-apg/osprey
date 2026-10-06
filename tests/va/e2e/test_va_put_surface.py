"""The put surface a served virtual accelerator offers a client, from the far side of the wire.

A client reaches the virtual accelerator through two transports and one model
RPC, and each refuses what the served view does not mark writable:

* a Channel Access put and a PVAccess put to a BPM readback, to a locked
  setpoint (the limits record that says ``writable: false``) and to a physics
  model's status address are each refused, and the value reads back as it was.
  A PVAccess put is completed with an error; Channel Access put completion
  carries no status, so there the refusal is the value that does not move;
* a fault is a variable of the model, never a served name: a Channel Access
  search for one finds no PV, though the model RPC reads the same name;
* the runner claims no control channel of its own: a search for ``reset``, the
  name its control PV would take under the served (empty) prefix, finds no PV
  on either transport;
* a Channel Access put above a setpoint's limits band is taken, and reads back
  clamped to the top of the band;
* a model RPC ``set`` on a server booted without ``VA_MODEL_WRITE_TOKEN`` is
  refused whatever token the client presents, and the model variable it named
  reads back as it was.

None of it is decidable without a served container, so this module boots one
``osprey-va-full`` container of its own over the demo view, publishing a
Channel Access port and a pvAccess port and setting no model write token. The
session container of ``conftest.py`` is not used: it publishes no pvAccess
port, and the clamp leg writes a corrector this module owns outright.

Every Channel Access operation happens in a subprocess that leaves through
``os._exit``, for the reasons ``conftest.py`` gives; this process never becomes
a Channel Access client. PVAccess operations run in this process through a p4p
client context built per call, after the environment naming the container's
port is in place.

The whole directory is opt-in behind ``OSPREY_VA_E2E_ENABLE=1``; the skip is
applied by ``conftest.pytest_collection_modifyitems``.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from tests.va.e2e import conftest as e2e_conftest

#: The image under test.
IMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", "osprey-va-full:latest")

#: Container-name prefix; ``_serving`` appends the run's own ephemeral port.
CONTAINER_PREFIX = "osprey-va-e2e-put-surface"

#: The instance the container serves as.
INSTANCE = "virtual_accelerator"

#: A local run on Apple Silicon is emulated.
BOOT_TIMEOUT_S = 180.0

#: What the readiness probe waits for.
PROBE_CHANNEL = "SR:MAG:HCM:01:CURRENT:RB"

#: Bound on a search this module expects to answer, and on one it expects to
#: go unanswered: a name the server holds is found well inside it.
SEARCH_TIMEOUT_S = 10.0

#: The prefix of the one line a Channel Access child reports on; the client
#: library writes notices of its own to the same stream.
REPORT_MARK = "REPORT "

#: Bound on one Channel Access subprocess.
CA_CHILD_TIMEOUT_S = 60.0

#: Bound on one PVAccess operation.
PVA_TIMEOUT_S = 30.0

# -- the served names --------------------------------------------------------

#: A BPM readback of the physics model.
BPM_READBACK = "SR:DIAG:BPM:03:POSITION:X"

#: The setpoint the demo's limits records lock with ``writable: false``.
LOCKED_SETPOINT = "SR:VAC:ION-PUMP:01:VOLTAGE:SP"

#: The status address of the demo's physics model.
MODEL_STATUS = "ca:SIM:SR:STATUS"

#: The three names no put moves, with the value each put carries: far from
#: anything the name reads, so a put that landed could not pass for one that
#: did not.
REFUSED_PUTS: dict[str, Any] = {
    BPM_READBACK: 0.25,
    LOCKED_SETPOINT: 4321.0,
    MODEL_STATUS: "failed",
}

#: BPM03's horizontal offset fault, as the model RPC names it.
FAULT_NAME = f"SR/{BPM_READBACK}/offset"

#: The name the runner's reset control PV would be served as. The served
#: prefix is empty, so it is the base name alone.
RESET_CONTROL_NAME = "reset"

#: The corrector whose limits record bands it to [-12, 12] A, a put above the
#: band, and the band's top.
BANDED_SETPOINT = "SR:MAG:HCM:01:CURRENT:SP"
OUT_OF_BAND_VALUE = 15.0
BAND_HIGH = 12.0

#: The value the refused model RPC ``set`` carries, in metres, and the token
#: the client presents with it.
FAULT_VALUE_M = 1e-3
PRESENTED_TOKEN = "any-token"


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _docker(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _require_image() -> None:
    """Fail loudly unless the image is present."""
    inspected = _docker("image", "inspect", IMAGE, "--format", "{{.Config.Cmd}}", timeout=60)
    if inspected.returncode != 0:
        pytest.fail(
            f"image {IMAGE!r} is not present. Build it with "
            f"scripts/va/build_and_boot_check.sh, or name another with OSPREY_VA_E2E_IMAGE."
        )


def _ca_environment(port: int) -> dict[str, str]:
    environment = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{port}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_SERVER_PORT", "EPICS_CAS_SERVER_PORT"):
        environment.pop(stale, None)
    return environment


def _ca_child(port: int, body: str) -> dict[str, Any]:
    """Run *body* as a Channel Access client of the container on *port*; its JSON report.

    *body* binds ``out``, a JSON-serialisable dict, which the child writes and
    flushes before leaving through ``os._exit``.
    """
    code = (
        "import json, os, sys, epics\n"
        "out = {}\n"
        f"{body}\n"
        f"sys.stdout.write('\\n' + {REPORT_MARK!r} + json.dumps(out) + '\\n')\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=CA_CHILD_TIMEOUT_S,
        env=_ca_environment(port),
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith(REPORT_MARK)]
    assert lines, f"the Channel Access child reported nothing:\n{run.stdout}\n{run.stderr}"
    report: dict[str, Any] = json.loads(lines[-1][len(REPORT_MARK) :])
    return report


def _served(port: int) -> bool:
    """Whether the container on *port* answers the probe channel, asked out of process."""
    try:
        report = _ca_child(
            port,
            f"out['value'] = epics.caget({PROBE_CHANNEL!r}, timeout=1.0, connection_timeout=1.0)",
        )
    except (subprocess.TimeoutExpired, AssertionError):
        return False
    return report.get("value") is not None


@dataclass(frozen=True)
class Served:
    """The container's two published ports."""

    ca: int
    pva: int


@contextlib.contextmanager
def _serving() -> Iterator[Served]:
    """Boot one container with no model write token; yield its two ports.

    The published ports and the server's own ports are the same numbers, since
    a search reply carries the server's own port; the Channel Access port also
    names the container.
    """
    port = _free_port()
    pva_port = _free_port()
    name = f"{CONTAINER_PREFIX}-{port}"
    _docker("rm", "-f", name, timeout=60)
    started = _docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        "-e",
        f"EPICS_PVAS_SERVER_PORT={pva_port}",
        "-e",
        f"VA_INSTANCE={INSTANCE}",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        "-p",
        f"127.0.0.1:{pva_port}:{pva_port}/tcp",
        *e2e_conftest.demo_data_run_args(),
        IMAGE,
    )
    if started.returncode != 0:
        raise RuntimeError(f"docker run failed: {started.stdout}\n{started.stderr}")
    try:
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if _served(port):
                break
            time.sleep(1.0)
        else:
            logs = _docker("logs", "--tail", "40", name, timeout=60)
            raise RuntimeError(
                f"{name} never served {PROBE_CHANNEL} within {BOOT_TIMEOUT_S}s.\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
        yield Served(ca=port, pva=pva_port)
    finally:
        _docker("rm", "-f", name, timeout=60)


@pytest.fixture(scope="module")
def served() -> Iterator[Served]:
    """The container, up and serving, for the life of this module."""
    _require_image()
    with _serving() as ports:
        yield ports


# ---------------------------------------------------------------------------
# The two transports
# ---------------------------------------------------------------------------


def _ca_put(port: int, address: str, value: Any) -> dict[str, Any]:
    """One Channel Access put to *address*, with the value read before and after.

    The report holds ``connected``, ``before``, ``put`` (what ``PV.put``
    returned) or ``error`` (what it raised), and ``after``, read uncached once
    the put completed. A text *value* reads the channel as text.
    """
    read = f"pv.get(timeout=5.0, use_monitor=False, as_string={isinstance(value, str)})"
    return _ca_child(
        port,
        f"pv = epics.PV({address!r}, connection_timeout={SEARCH_TIMEOUT_S})\n"
        f"out['connected'] = pv.wait_for_connection(timeout={SEARCH_TIMEOUT_S})\n"
        "if out['connected']:\n"
        f"    out['before'] = {read}\n"
        "    try:\n"
        f"        out['put'] = pv.put({value!r}, wait=True, timeout=10.0)\n"
        "    except Exception as exc:\n"
        "        out['error'] = f'{type(exc).__name__}: {exc}'\n"
        f"    out['after'] = {read}\n",
    )


def _ca_found(port: int, address: str) -> bool:
    """Whether a Channel Access search for *address* finds a PV within the bound."""
    report = _ca_child(
        port,
        f"pv = epics.PV({address!r}, connection_timeout={SEARCH_TIMEOUT_S})\n"
        f"out['connected'] = pv.wait_for_connection(timeout={SEARCH_TIMEOUT_S})\n",
    )
    return bool(report["connected"])


@contextlib.contextmanager
def _pva(port: int) -> Iterator[Any]:
    """A p4p client context that searches the container on *port* alone.

    p4p reads the environment when a context is built, so the context is built
    inside the patched environment.
    """
    from p4p.client.thread import Context

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("EPICS_PVA_NAME_SERVERS", f"127.0.0.1:{port}")
        patch.setenv("EPICS_PVA_AUTO_ADDR_LIST", "NO")
        patch.setenv("EPICS_PVA_ADDR_LIST", "")
        ctx = Context("pva")
        try:
            yield ctx
        finally:
            ctx.close()


def _pva_get(port: int, address: str) -> Any:
    """*address*'s value over PVAccess, unwrapped to a plain Python value."""
    with _pva(port) as ctx:
        value = ctx.get(address, timeout=PVA_TIMEOUT_S)
    return value.raw.value if hasattr(value, "raw") else value


def _rpc(port: int, verb: str, **fields: Any) -> Any:
    """One model RPC call; its result, or :class:`ModelRpcError` with the refusal."""
    from osprey.services.virtual_accelerator.serving.model_rpc import (
        RPC_PV,
        RPC_TIMEOUT_S,
        build_request,
        parse_reply,
    )

    with _pva(port) as ctx:
        return parse_reply(ctx.rpc(RPC_PV, build_request(verb, **fields), timeout=RPC_TIMEOUT_S))


def _unchanged(before: Any, after: Any) -> bool:
    """Whether a value read after a refused put is the one read before it.

    Every name is held exact: the demo view this module serves renders its
    monitors without their declared motion, so nothing moves between reads.
    """
    if isinstance(before, float) and isinstance(after, float):
        return after == pytest.approx(before, rel=1e-9, abs=1e-12)
    return bool(after == before)


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("address", sorted(REFUSED_PUTS))
def test_a_ca_put_to_a_name_the_view_does_not_write_is_refused(
    served: Served, address: str
) -> None:
    report = _ca_put(served.ca, address, REFUSED_PUTS[address])

    assert report["connected"], f"{address} is not served"
    assert report["after"] != REFUSED_PUTS[address], report
    assert _unchanged(report["before"], report["after"]), report


@pytest.mark.parametrize("address", sorted(REFUSED_PUTS))
def test_a_pva_put_to_a_name_the_view_does_not_write_is_refused(
    served: Served, address: str
) -> None:
    from p4p.client.thread import RemoteError

    before = _pva_get(served.pva, address)
    with _pva(served.pva) as ctx, pytest.raises(RemoteError):
        ctx.put(address, REFUSED_PUTS[address], timeout=PVA_TIMEOUT_S)
    after = _pva_get(served.pva, address)

    assert _unchanged(before, after), (before, after)


def test_a_fault_is_a_model_variable_and_no_ca_name(served: Served) -> None:
    held = _rpc(served.pva, "get", names=[FAULT_NAME])

    assert FAULT_NAME in held, held
    assert not _ca_found(served.ca, FAULT_NAME)


def test_no_reset_control_pv_is_served_on_ca(served: Served) -> None:
    assert _ca_found(served.ca, PROBE_CHANNEL)
    assert not _ca_found(served.ca, RESET_CONTROL_NAME)


def test_no_reset_control_pv_is_served_on_pva(served: Served) -> None:
    with _pva(served.pva) as ctx, pytest.raises(TimeoutError):
        ctx.get(RESET_CONTROL_NAME, timeout=SEARCH_TIMEOUT_S)


def test_an_out_of_band_ca_put_reads_back_clamped_to_the_band(served: Served) -> None:
    report = _ca_put(served.ca, BANDED_SETPOINT, OUT_OF_BAND_VALUE)

    assert report["connected"], f"{BANDED_SETPOINT} is not served"
    assert report.get("put") == 1, report
    assert report["after"] == pytest.approx(BAND_HIGH), report


def test_an_rpc_set_without_a_model_write_token_is_refused(served: Served) -> None:
    from osprey.services.virtual_accelerator.serving.model_rpc import ModelRpcError
    from osprey.services.virtual_accelerator.serving.model_surface import WRITES_DISABLED

    before = _rpc(served.pva, "get", names=[FAULT_NAME])[FAULT_NAME]
    with pytest.raises(ModelRpcError, match=WRITES_DISABLED):
        _rpc(
            served.pva,
            "set",
            values={FAULT_NAME: before + FAULT_VALUE_M},
            token=PRESENTED_TOKEN,
        )
    after = _rpc(served.pva, "get", names=[FAULT_NAME])[FAULT_NAME]

    assert after == before
