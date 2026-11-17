"""PID 1 of the queue-server container: exit when the RE Manager is gone.

``start-re-manager`` is upstream's watchdog process; the manager and the
RunEngine worker are its children. A manager that dies before it has enabled
that watchdog (it enables it only after its Redis connection succeeded) is
neither restarted nor followed: the watchdog process stays alive with no
manager inside, the container stays ``running``, and a restart policy has
nothing to act on.

Run as ``python -m osprey.services.bluesky_bridge.qserver_supervisor --
start-re-manager <arguments>``, this module starts that command as a child
and:

- forwards ``SIGTERM`` and ``SIGINT`` to it, and exits with its status when
  it exits;
- after a start grace period asks the manager what the container's health
  check asks (``qserver ping``, with the keys already in the environment);
- when a ping goes unanswered, asks the kernel whether any process still
  holds the manager's control port: a TCP connect that is *refused* means
  none does;
- when both held :data:`MAX_FAILED_PROBES` times in a row, logs one line,
  stops the child and everything it started, and exits with
  :data:`UNRESPONSIVE_EXIT_STATUS` so the container's restart policy starts a
  fresh manager.

Stopping the child ends the RunEngine worker too, and with it a plan that may
be moving hardware. That is why an unanswered ping alone is not enough. A
manager that is alive but not answering (a stalled host, a control-socket key
pair that does not match) still holds its port: the connect is accepted or
times out, the count starts again, and the container stays as it is, unhealthy
by its health check. Only a manager whose process is gone leaves the port
closed, and in that state no plan can be paused or aborted through the manager
any more.

It reads the manager's public control socket only, and nothing of upstream's
internals. Standard library only: it must start in any image that can start
the manager.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger("osprey.services.bluesky_bridge.qserver_supervisor")

#: Seconds after the start in which the manager is not asked anything: it
#: waits on Redis and imports the bluesky stack before its socket answers.
START_GRACE_S = 60.0
#: Seconds between two probes.
PROBE_INTERVAL_S = 10.0
#: Consecutive probes that found the manager gone after which it is taken as
#: gone. Upstream's own watchdog replaces a manager that hung after it
#: connected in about 5 to 15 s, which is one or two probes and never this
#: many.
MAX_FAILED_PROBES = 6
#: Seconds one ping may take, the health check's own timeout.
PROBE_TIMEOUT_S = 10.0
#: Seconds one connect to the control port may take. A refusal is immediate;
#: a connect that needs longer is not one.
CONNECT_TIMEOUT_S = 3.0
#: Seconds the child gets to stop after ``SIGTERM`` before ``SIGKILL``.
STOP_TIMEOUT_S = 10.0
#: Exit status of a supervisor that stopped a child whose manager was gone
#: (``EX_SOFTWARE``): not zero, and not a status the manager exits with.
UNRESPONSIVE_EXIT_STATUS = 70

#: Where upstream's manager binds its control socket when nothing says
#: otherwise (``default_zmq_control_address_for_server``).
DEFAULT_CONTROL_ADDRESS = "tcp://*:60615"
#: The variable upstream's manager reads that address from.
CONTROL_ADDRESS_ENV = "QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER"
#: The manager's arguments that carry it; the second is upstream's deprecated
#: spelling of the first.
_CONTROL_ADDRESS_OPTIONS = ("--zmq-control-addr", "--zmq-addr")
#: Bind hosts that mean "every interface", reached here over loopback.
_WILDCARD_HOSTS = ("*", "0.0.0.0", "")

Probe = Callable[[], bool]
Clock = Callable[[], float]
WaitForExit = Callable[["subprocess.Popen[bytes]", float], int | None]


def ping_manager() -> bool:
    """Whether the manager answers on its control socket right now.

    The same question the container's health check asks, asked the same way.
    A ping that fails, does not return in time or cannot be run is no answer.
    """
    try:
        result = subprocess.run(
            ["qserver", "ping"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def control_endpoint(command: Sequence[str], environ: Mapping[str, str]) -> tuple[str, int] | None:
    """Where the manager started by ``command`` listens, as seen from here.

    Read from what the manager itself is given, in upstream's own order: the
    ``--zmq-control-addr`` argument, then the environment variable, then the
    default. An address in a manager config file is not read.

    Returns:
        ``(host, port)`` to connect to, or None when the address names no TCP
        port (the supervisor then never finds the manager gone).
    """
    address = environ.get(CONTROL_ADDRESS_ENV) or DEFAULT_CONTROL_ADDRESS
    for index, argument in enumerate(command):
        name, separator, value = argument.partition("=")
        if name not in _CONTROL_ADDRESS_OPTIONS:
            continue
        if separator:
            address = value
        elif index + 1 < len(command):
            address = command[index + 1]
        break

    scheme, _, location = address.partition("://")
    host, _, port = location.rpartition(":")
    if scheme != "tcp" or not port.isdecimal():
        return None
    return ("127.0.0.1" if host in _WILDCARD_HOSTS else host, int(port))


def control_port_refused(host: str, port: int) -> bool:
    """Whether a TCP connect to the manager's control port is refused.

    Refused means no process holds the port. A connect that is accepted (the
    kernel completes it for a process that never gets to answer), that times
    out, or that fails in any other way is not a refusal.
    """
    try:
        socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S).close()
    except ConnectionRefusedError:
        return True
    except OSError:
        return False
    return False


def start_child(command: Sequence[str]) -> subprocess.Popen[bytes]:
    """Start ``command`` as the leader of a process group of its own.

    The group is what lets :func:`stop_process_group` reach the manager and
    the worker, which are the command's children and not this process's.
    """
    return subprocess.Popen(list(command), start_new_session=True)


def stop_process_group(child: subprocess.Popen[bytes], *, timeout: float = STOP_TIMEOUT_S) -> None:
    """Stop ``child`` and every process in its group: terminate, then kill."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(timeout=timeout)
    # Unconditional: a descendant can outlive a child that stopped on request.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def _wait_for_exit(child: subprocess.Popen[bytes], seconds: float) -> int | None:
    try:
        return child.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return None


def _exit_status(returncode: int) -> int:
    """A child's return code as a shell reports it (128 + N for signal N)."""
    return 128 - returncode if returncode < 0 else returncode


def supervise(
    child: subprocess.Popen[bytes],
    *,
    port_refused: Probe,
    probe: Probe = ping_manager,
    clock: Clock = time.monotonic,
    wait_for_exit: WaitForExit = _wait_for_exit,
) -> int:
    """Follow ``child`` until it exits or its manager process is gone.

    Args:
        child: The running ``start-re-manager``, from :func:`start_child`.
        port_refused: Whether a connect to the manager's control port is
            refused; asked only after an unanswered ``probe``.
        probe: Whether the manager answers; :func:`ping_manager` by default.
        clock: Monotonic seconds.
        wait_for_exit: Waits up to the given seconds for ``child`` and
            returns its return code, or None while it is still running.

    Returns:
        The status this process should exit with: the child's own when it
        exited, :data:`UNRESPONSIVE_EXIT_STATUS` when it was stopped here.
    """
    started = clock()
    failures = 0
    while True:
        returncode = wait_for_exit(child, PROBE_INTERVAL_S)
        if returncode is not None:
            return _exit_status(returncode)
        if clock() - started < START_GRACE_S:
            continue
        # Gone is both, every time: one answer, or one probe that finds the
        # port still held, starts the count again.
        if probe() or not port_refused():
            failures = 0
            continue
        # The probes take time; a child that ended meanwhile speaks for itself.
        returncode = child.poll()
        if returncode is not None:
            return _exit_status(returncode)
        failures += 1
        if failures < MAX_FAILED_PROBES:
            continue
        logger.error(
            "the RE Manager process is gone: %d probes in a row (one every %.0f s, after a "
            "%.0f s start grace) it did not answer a ping and its control port refused a "
            "connection; stopping start-re-manager (pid %d) and its worker, and exiting %d "
            "so the container's restart policy starts a fresh manager",
            failures,
            PROBE_INTERVAL_S,
            START_GRACE_S,
            child.pid,
            UNRESPONSIVE_EXIT_STATUS,
        )
        stop_process_group(child)
        return UNRESPONSIVE_EXIT_STATUS


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command after ``--`` under supervision; return the exit status."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="[%(levelname).1s %(asctime)s] %(message)s"
    )
    command = list(sys.argv[1:] if argv is None else argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        logger.error("usage: python -m %s -- <command> [argument ...]", logger.name)
        return 2

    endpoint = control_endpoint(command, os.environ)
    if endpoint is None:
        logger.warning(
            "the manager's control address names no TCP port, so a manager that is gone "
            "cannot be told from one that is not answering; %s only passes signals and "
            "the exit status on",
            logger.name,
        )

    def port_refused() -> bool:
        return endpoint is not None and control_port_refused(*endpoint)

    # Installed before the child exists: PID 1 gets no default disposition, so
    # a stop arriving in the gap would otherwise be dropped, not deferred.
    children: list[subprocess.Popen[bytes]] = []

    def forward(signum: int, _frame: object) -> None:
        for running in children:
            with contextlib.suppress(ProcessLookupError):
                running.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward)

    try:
        child = start_child(command)
    except OSError as exc:
        logger.error("cannot start %s: %s", command[0], exc)
        return 127
    children.append(child)
    return supervise(child, port_refused=port_refused)


if __name__ == "__main__":
    sys.exit(main())
