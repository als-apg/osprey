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
- when the manager has not answered :data:`MAX_FAILED_PROBES` times in a row,
  logs one line, stops the child and everything it started, and exits with
  :data:`UNRESPONSIVE_EXIT_STATUS` so the container's restart policy starts a
  fresh manager.

It reads the manager's public control socket only, and nothing of upstream's
internals. Standard library only: it must start in any image that can start
the manager.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

logger = logging.getLogger("osprey.services.bluesky_bridge.qserver_supervisor")

#: Seconds after the start in which the manager is not asked anything: it
#: waits on Redis and imports the bluesky stack before its socket answers.
START_GRACE_S = 60.0
#: Seconds between two probes.
PROBE_INTERVAL_S = 10.0
#: Consecutive unanswered probes after which the manager is taken as gone.
#: Upstream's own watchdog replaces a manager that hung after it connected in
#: about 5 to 15 s, which is one or two probes and never this many.
MAX_FAILED_PROBES = 6
#: Seconds one probe may take, the health check's own timeout.
PROBE_TIMEOUT_S = 10.0
#: Seconds the child gets to stop after ``SIGTERM`` before ``SIGKILL``.
STOP_TIMEOUT_S = 10.0
#: Exit status of a supervisor that stopped an unresponsive manager
#: (``EX_SOFTWARE``): not zero, and not a status the manager exits with.
UNRESPONSIVE_EXIT_STATUS = 70

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
    probe: Probe = ping_manager,
    clock: Clock = time.monotonic,
    wait_for_exit: WaitForExit = _wait_for_exit,
) -> int:
    """Follow ``child`` until it exits or its manager stops answering.

    Args:
        child: The running ``start-re-manager``, from :func:`start_child`.
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
        if probe():
            failures = 0
            continue
        # The probe takes time; a child that ended meanwhile speaks for itself.
        returncode = child.poll()
        if returncode is not None:
            return _exit_status(returncode)
        failures += 1
        if failures < MAX_FAILED_PROBES:
            continue
        logger.error(
            "the RE Manager did not answer %d pings in a row (one every %.0f s, after a "
            "%.0f s start grace); stopping start-re-manager (pid %d) and its worker, and "
            "exiting %d so the container's restart policy starts a fresh manager",
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
    return supervise(child)


if __name__ == "__main__":
    sys.exit(main())
