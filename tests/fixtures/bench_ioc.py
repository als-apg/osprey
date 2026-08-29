"""Boot the bench softIoc container, and talk Channel Access to it.

The bench IOC (``docker/bench-ioc/``) is the "real machine" opposite the
virtual accelerator in target-switch tests: a second Channel Access server, on
a second port, answering for the SAME channel names with DIFFERENT values. A
suite that cannot tell the two apart on the wire proves nothing about switching
between targets, which is why booting one is worth a shared module rather than
a per-suite copy.

Two things here look like a copy-paste of
``tests/va/e2e/test_target_switch.py``'s container helpers and are not:

* ``tests/va/e2e/test_target_switch.py`` keeps its own
  ``_free_port``/``_docker``/``_serving`` trio, predating this module and not
  yet folded into it. This module exists precisely so the *bench* IOC's
  version of that trio is spelled once and imported, so new callers get the
  boot contract (ephemeral port, port-suffixed name, stale cleanup, readiness
  gate, ``rm -f`` in ``finally``) rather than re-deriving it.
* The image precondition is different in kind. The virtual accelerator's
  ``_require_image()`` greps the image's ``Cmd`` for a variable name; this one
  asserts that the image can actually *run* ``softIoc``, because that -- not a
  string in a config blob -- is what the bench IOC is for.

The precondition fails rather than skips. A lane that skipped on a missing
image would report success having proved nothing, which is the failure mode the
qmd marker doctrine names. The image is never built implicitly either: an
implicit build turns a five-minute first run into an unexplained hang, so a
missing image is an error message naming the one command that fixes it.

Every Channel Access call runs out of process, through
``tests/e2e/_va_host_ca_op.py``. That is not indirection for its own sake: the
Channel Access client library latches its environment at first use, so an
in-process client would pin this pytest process to the first port it ever
talked to -- fatal for a suite whose entire subject is talking to two servers
in turn. The subprocess also carries the CA-teardown isolation that script's
docstring explains.
"""

from __future__ import annotations

import contextlib
import functools
import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# The image, and what it serves
# ---------------------------------------------------------------------------

#: The bench IOC image. Built by hand, never implicitly by this module.
IMAGE = "bench-ioc:latest"

#: The one command that produces IMAGE. ``-f`` is load-bearing: BuildKit only
#: auto-detects a file literally named ``Dockerfile``.
BUILD_COMMAND = (
    "docker build -f docker/bench-ioc/Containerfile -t bench-ioc:latest docker/bench-ioc"
)

#: Printed by ``iocInit`` once every record is loaded and the CA server is
#: listening. It goes to stderr (unbuffered), so it reaches ``docker logs``
#: immediately -- but a caller must read both streams to find it.
READY_MARKER = "iocRun: All initialization complete"

#: Default prefix for container names. The chosen port is appended, which is
#: what makes concurrent runs safe: a port is one run's alone by construction.
CONTAINER_PREFIX = "osprey-bench-ioc"

# The seeded record set, named here so callers assert against one spelling.
# Values are constants in docker/bench-ioc/bench.db and hold from the first
# read: the IOC has no autosave and no persistence, so every boot starts again
# from these and no run can inherit what an earlier run wrote.
PROBE_CHANNEL = "SR:MAG:HCM:01:CURRENT:RB"
PROBE_VALUE = 7.25
#: Writable setpoint -- accepts a caput and latches it (no readback echo: the
#: bench IOC is constants, not a simulation, so read this channel back, not RB).
WRITABLE_SP = "SR:MAG:HCM:01:CURRENT:SP"
#: Read-OK, write-DENIED by the IOC's access-security file. A caput here is
#: refused by the control system itself, not by any client-side check.
PROTECTED_SP = "SR:MAG:VCM:02:CURRENT:SP"
PROTECTED_VALUE = -3.5
#: An mbbi: the value is the state index, the label names the state it is in.
MODE_STATE = "SR:DIAG:STRIPLINE:01:MODE:STATE"
MODE_STATE_VALUE = 2
#: The label for MODE_STATE_VALUE, and the record's full state list in index
#: order -- both spelled in ``docker/bench-ioc/bench.db`` as ZRST..THST.
MODE_STATE_LABEL = "ACQUIRING"
MODE_STATE_LABELS = ("OFFLINE", "STANDBY", "ACQUIRING", "FAULT")
#: A calc record seeded consistently with its inputs, so it is deterministic
#: from boot rather than only after its first scan.
SCALED_AMPLITUDE = "SR:DIAG:STRIPLINE:01:AMPLITUDE:SCALED"
SCALED_AMPLITUDE_VALUE = 6.375

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

#: Boot deadline. Generous because the image is linux/amd64 and runs under
#: emulation on an arm64 host; the same suites' virtual-accelerator boots are
#: given 120-180s for the same reason.
BOOT_TIMEOUT_S = 180.0
#: Gap between readiness polls. Each CA poll spawns a process, so this is not
#: a tight loop.
POLL_INTERVAL_S = 1.0
#: Bound on one docker CLI call.
DOCKER_TIMEOUT_S = 180.0
#: Bound on one out-of-process CA op: process spawn, connector connect over the
#: name-server TCP circuit, and one read/write round trip.
CA_OP_TIMEOUT_S = 60.0
#: The connector's own per-operation timeout inside that subprocess.
CONNECTOR_TIMEOUT_S = 5.0

#: The out-of-process CA op, and its result marker. The marker is a local
#: literal for the reason ``tests/e2e/test_va_substrate_equivalence.py`` gives:
#: the script is a runnable protocol, and importing it to read one constant
#: would couple this module to that directory's collection.
CA_OP_SCRIPT = Path(__file__).resolve().parent.parent / "e2e" / "_va_host_ca_op.py"
CA_RESULT_MARKER = "__HOST_CA_RESULT__"


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------


def free_port() -> int:
    """An ephemeral TCP port, from the OS rather than from a guess."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def docker(*args: str, timeout: float = DOCKER_TIMEOUT_S) -> subprocess.CompletedProcess:
    """One docker CLI call, captured and never raising on a non-zero exit."""
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


@functools.cache
def require_bench_image() -> None:
    """Fail loudly unless the bench image exists and can run ``softIoc``.

    Presence alone is not the precondition. The image's whole job is to serve a
    record set through ``softIoc``, and the runtime base it builds on ships no
    ``dbd`` directory -- a build that lost the one copied file would still
    produce a perfectly inspectable image whose IOC dies at startup. Asking the
    binary to answer ``-h`` costs a second and turns that into a sentence.

    Fails rather than skips, deliberately: a skipped lane reports success having
    proved nothing about a switch it never observed.

    Cached per process, not per caller: the probe boots an emulated container,
    so a run with several consuming modules would otherwise pay for it once per
    module to re-answer a question about an image that cannot change mid-run.
    A failure is not cached -- ``functools.cache`` stores no result for a call
    that raised -- so every caller still fails, each with the full message.
    """
    inspected = docker("image", "inspect", IMAGE, "--format", "{{.Architecture}}", timeout=60)
    if inspected.returncode != 0:
        pytest.fail(
            f"image {IMAGE!r} is not present, and this fixture never builds it implicitly "
            f"(an implicit build turns a first run into an unexplained multi-minute hang). "
            f"Build it with:\n    {BUILD_COMMAND}"
        )
    probe = docker("run", "--rm", IMAGE, "softIoc", "-h", timeout=120)
    if probe.returncode != 0:
        pytest.fail(
            f"image {IMAGE!r} ({inspected.stdout.strip()}) is present but cannot run softIoc "
            f"(exit {probe.returncode}), so it can serve no records at all. Rebuild it with:\n"
            f"    {BUILD_COMMAND}\n"
            f"--- stdout ---\n{probe.stdout}\n--- stderr ---\n{probe.stderr}"
        )


def _logs(name: str, *, tail: str = "60") -> str:
    """Both streams of a container's log, joined.

    The readiness marker is written to stderr, so a caller reading only stdout
    would wait out the whole boot deadline on a container that came up fine.
    """
    result = docker("logs", "--tail", tail, name, timeout=60)
    return f"{result.stdout}\n{result.stderr}"


def _initialized(name: str) -> bool:
    """Whether the IOC has finished loading its database, per its own log."""
    return READY_MARKER in _logs(name)


def _serving(port: int) -> bool:
    """Whether the probe channel answers on *port*, asked out of process.

    The stronger of the two readiness questions: the log marker says the IOC
    initialized, this says a client on the host can actually reach it through
    the published port.
    """
    try:
        result = ca_op(port, read=PROBE_CHANNEL)
    except (AssertionError, subprocess.SubprocessError):
        return False
    return abs(float(result["read_value"]) - PROBE_VALUE) < 1e-6


@dataclass(frozen=True)
class BenchIOC:
    """One running bench IOC: the port it serves and the container serving it."""

    port: int
    container: str


@contextlib.contextmanager
def bench_ioc(
    *,
    prefix: str = CONTAINER_PREFIX,
    boot_timeout: float = BOOT_TIMEOUT_S,
) -> Iterator[BenchIOC]:
    """Boot one bench IOC and yield it once it answers, tearing it down after.

    The published port and the server's own port are the same number by
    construction -- a CA search reply carries the server's port, so a remap
    would hand every client a port it cannot reach, with no useful error -- and
    that number also names the container.

    Readiness is asked twice, in order, because the two failures need different
    sentences: an IOC that never printed its initialization marker is a database
    or startup problem, while an IOC that initialized but never answered on the
    port is a publishing or port-derivation problem.
    """
    port = free_port()
    name = f"{prefix}-{port}"
    # Stale cleanup only. The port belongs to this run, so this can name nothing
    # a concurrent run is using -- which is the point of the suffix.
    docker("rm", "-f", name, timeout=60)

    started = docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        IMAGE,
    )
    if started.returncode != 0:
        # Clean up before raising, which the test_target_switch.py trio this
        # mirrors does not do. A run that fails AFTER the name is claimed --
        # another process taking the port between the bind probe and here, a
        # daemon that dies mid-start -- leaves a Created container under a name
        # carrying this run's port. No later run picks that port, so no later
        # stale cleanup ever names it, and it accumulates until someone prunes
        # by hand.
        with contextlib.suppress(subprocess.SubprocessError):
            docker("rm", "-f", name, timeout=60)
        raise RuntimeError(f"docker run failed for {name}:\n{started.stdout}\n{started.stderr}")

    try:
        deadline = time.monotonic() + boot_timeout
        initialized = False
        while time.monotonic() < deadline:
            if not initialized:
                initialized = _initialized(name)
            if initialized and _serving(port):
                break
            time.sleep(POLL_INTERVAL_S)
        else:
            stage = (
                f"initialized but never answered {PROBE_CHANNEL} on port {port}"
                if initialized
                else f"never printed {READY_MARKER!r}"
            )
            raise RuntimeError(f"{name} {stage} within {boot_timeout}s.\n{_logs(name)}")
        yield BenchIOC(port=port, container=name)
    finally:
        # A wedged daemon makes this call time out, and a raise from a finally
        # block replaces whatever was already propagating -- turning a boot
        # failure that carries the container's own log into a bare
        # TimeoutExpired naming only docker. The cleanup is best effort; the
        # diagnosis is not.
        with contextlib.suppress(subprocess.SubprocessError):
            docker("rm", "-f", name, timeout=60)


# ---------------------------------------------------------------------------
# Channel Access
# ---------------------------------------------------------------------------


def connector_config(port: int, *, timeout: float = CONNECTOR_TIMEOUT_S) -> dict[str, Any]:
    """The connector config that reaches a bench IOC on *port*.

    A plain EPICS connector -- the bench IOC is a stock softIoc, not a virtual
    accelerator, and picking the connector by what is actually on the wire is
    the point of a control-system-agnostic interface.

    ``use_name_server`` puts the client on the name-server TCP circuit, the one
    host-to-container configuration proven to work across container runtimes
    (the image publishes TCP only and broadcasts nothing). The connector derives
    the client environment from this block itself; a caller that exported those
    variables by hand would be racing it.

    Both gateway roles name the same endpoint. There is one server here, and
    the connector routes writes through ``write_access`` only when writes are
    enabled -- so a read-only op still reaches the IOC, and the protected
    channel's refusal comes from the IOC's access-security rules rather than
    from a client that never asked.
    """
    gateway = {"address": "localhost", "port": port, "use_name_server": True}
    return {
        "type": "epics",
        "connector": {
            "epics": {
                "timeout": timeout,
                "gateways": {"read_only": dict(gateway), "write_access": dict(gateway)},
            }
        },
    }


def ca_op(
    port: int,
    *,
    read: str,
    write: dict[str, Any] | None = None,
    settle_read: bool = False,
    writes_enabled: bool = False,
    timeout: float = CA_OP_TIMEOUT_S,
) -> dict[str, Any]:
    """Run one ``connect -> optional write -> read`` against the bench IOC.

    Out of process, through ``tests/e2e/_va_host_ca_op.py``, for the reason this
    module's docstring gives: the CA client library latches its environment at
    first use, so a suite that talked to one port in this process could never
    talk to another. Returns that script's parsed result:
    ``read_value``, ``read_settled``, ``write_outcome`` and
    ``write_error_message`` -- the outcome is ``None`` when no write was asked
    for, otherwise the ``WriteOutcome`` word the connector reported.

    Limits checking stays off. The bench IOC has no channel-limits database, and
    a client-side refusal would mask the only refusal these suites care about --
    the one the control system itself issues.
    """
    spec = {
        "connector_config": connector_config(port),
        "config_overrides": {
            "control_system.writes_enabled": writes_enabled,
            "control_system.limits_checking.enabled": False,
        },
        "read": read,
        "write": write,
        "settle_read": settle_read,
    }
    proc = subprocess.run(
        [sys.executable, str(CA_OP_SCRIPT), json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"bench CA op failed (rc={proc.returncode}) on port {port}:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith(CA_RESULT_MARKER):
            return json.loads(line[len(CA_RESULT_MARKER) :])
    raise AssertionError(
        f"bench CA op produced no {CA_RESULT_MARKER} line on port {port}:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def caget(port: int, channel: str, *, timeout: float = CA_OP_TIMEOUT_S) -> float:
    """Read one channel from the bench IOC on *port*."""
    return float(ca_op(port, read=channel, timeout=timeout)["read_value"])


def caput(
    port: int,
    channel: str,
    value: float,
    *,
    read_back: str | None = None,
    settle_read: bool = False,
    timeout: float = CA_OP_TIMEOUT_S,
) -> dict[str, Any]:
    """Write one channel on the bench IOC, then read *read_back* (default: it).

    Returns the full result rather than a bare flag, because a refused write is
    a *result* here, not an exception: the IOC's access-security rules deny one
    channel by design, and that denial arrives as ``write_outcome == "refused"``
    with the read still answering the channel's unchanged value.

    ``settle_read`` polls the readback until it reflects the written value. The
    bench IOC's records are seeded constants with no setpoint-to-readback echo
    wired between them, so leave it off unless reading back a channel that does
    latch what was written to it.
    """
    return ca_op(
        port,
        read=read_back if read_back is not None else channel,
        write={"address": channel, "value": value},
        settle_read=settle_read,
        writes_enabled=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# pytest wiring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_endpoint() -> Iterator[BenchIOC]:
    """One bench IOC, up and answering, for the life of the importing module.

    Import it into a suite to use it (``from tests.fixtures.bench_ioc import
    bench_endpoint``). Module-scoped because booting an emulated container per
    test would dominate the run; the record set is reseeded on every boot, so
    the only state a later test can see is what an earlier one wrote -- write to
    the setpoints with that in mind, or take your own ``bench_ioc()`` context.
    """
    require_bench_image()
    with bench_ioc() as ioc:
        yield ioc
