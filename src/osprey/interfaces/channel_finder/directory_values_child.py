"""The process that reads current values for the Channels panel.

The panel process never opens a control session. This child does, under the
environment its supervisor renders, and serves one thing on the listening socket
it inherits: a batch read (:mod:`.directory_values_ipc`). The wire has no way to
ask for anything else, and raw client puts are refused process-wide before the
session opens.

**It is a read-only run**, whatever the operator's switch says. Nothing here
writes, so it says so in :data:`~osprey.audit.posture.POSTURE_ENV_VAR` before the
session opens: the connector stays on the read gateway, so the gateway too refuses
any put, a switch flip is not a move, and its audit records read ``sandbox``. A
block with only a write gateway has nothing to read through: the session is
unservable and every row reads unavailable.

**How it ends.** It serves until one of these, and its exit status is the whole
message to its supervisor:

* :data:`~osprey.runtime.session.EXIT_START_REFUSED` (78): the environment is one
  a session must not run in, or the raw-put block could not be armed. The last
  stderr line is :data:`~.directory_values_ipc.START_REFUSED_PREFIX` and the reason.
* :data:`~osprey.runtime.session.EXIT_BINDING_MOVED` (75): the record names
  another binding, or it can no longer be watched.
* :data:`~osprey.runtime.session.EXIT_TRANSPORT_STALE` (76): the connector's
  transport went stale.
* Its parent is gone. Channels runs under ``uvicorn.run`` in a daemon thread, so
  its process exit skips the lifespan stop, and this child is in its own session:
  a daemon thread polls ``os.getppid()`` and leaves through ``os._exit``.

**The socket outlives this process.** The supervisor holds it across respawns,
so the server here is never closed: closing one removes the socket's path on
Python 3.13. Every exit is ``os._exit``, which does not, and it is taken inside
the running loop: ``asyncio.run``'s teardown would join the default executor,
holding the exit for any PVA read in flight and leaving the listener open.

**Reads end in the connector.** The session's call bound is the per-channel cap,
so a failed connect finishes at the connector's own deadline, which records it
for ``wait_stale``, and a PVA read gives its worker thread back. The row's bound
sits just above: cancelling a read at the cap would stop the connector before it
records the failure, and exit 76 could never fire. A batch has one worker thread
per row, so a dead PVA channel holds up only its own.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from osprey.audit import posture
from osprey.audit.envelope import DECISION_ALLOWED
from osprey.audit.writer import record
from osprey.interfaces.channel_finder import directory_values_ipc as ipc
from osprey.interfaces.channel_finder.directory_values_ipc import ValueRow
from osprey.runtime.budget import (
    GRACE_MARGIN_S,
    MAX_RELAY_TIMEOUT_S,
    RELAY_MARGIN_S,
    WriteBudget,
)
from osprey.runtime.session import (
    EXIT_BINDING_MOVED,
    EXIT_START_REFUSED,
    EXIT_TRANSPORT_STALE,
    ControlSession,
    SessionUnservableError,
)
from osprey.runtime.session_env import SessionEnvironmentError, check_environment
from osprey_connectors.control_system import ChannelValue
from osprey_connectors.errors import ConnectorUnavailableError

logger = logging.getLogger(__name__)

#: The surface name this process's session and audit records go under.
SURFACE = "channels"

#: The audit ``subject`` of every batch.
READ_SUBJECT = "channel_values_read"

#: How long one channel gets to answer (D9): the connector's call bound here.
PV_READ_BOUND_S = 1.0

#: What a row waits past that, for a connector that does not keep its bound.
ROW_MARGIN_S = 0.5

#: How often the parent watcher looks at the parent.
PARENT_POLL_S = 1.0

#: The status of a child whose parent is gone. Nothing is left to read it.
_EXIT_ORPHANED = 0


def _exit_process(status: int) -> None:
    """End the process now with *status*, after flushing what was logged."""
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    os._exit(status)


def _start_parent_watcher(parent_pid: int, interval: float = PARENT_POLL_S) -> threading.Thread:
    """Start the daemon thread that ends this process once *parent_pid* is gone.

    A parent of init counts as gone even when it is the one recorded: a parent
    that died before the record was taken has already handed this child to init.
    """

    def poll() -> None:
        while True:
            parent = os.getppid()
            if parent != parent_pid or parent == 1:
                logger.warning("Parent %s is gone; leaving", parent_pid)
                _exit_process(_EXIT_ORPHANED)
            time.sleep(interval)

    thread = threading.Thread(target=poll, name="channel-values-parent-watch", daemon=True)
    thread.start()
    return thread


def _arm_raw_put_block() -> None:
    """Refuse every raw client put in this process, and rpc and Tango commands with them.

    This process only reads, so unlike the CANOPY backend it has no use for them.
    The executor's contract leaves PVAccess puts to its limits check, as a readwrite
    run's one PVAccess write route; here there is neither that check nor any write,
    so they are refused like the rest.
    """
    from osprey.runtime import raw_put_block
    from osprey.services.python_executor.execution.wrapper import armed_contract
    from osprey.services.python_executor.write_surface import _ARMED_CHECKED

    contract = armed_contract(refuse_rpc=True)
    contract["blocked_targets"] = (
        *contract["blocked_targets"],
        *((dotted, (attr,)) for dotted, attr in _ARMED_CHECKED),
    )
    contract["ca_provider_targets"] = ()
    raw_put_block.install("armed", **contract)


def _read_budget() -> WriteBudget:
    """The session's budget: every call bound to the per-channel cap. Nothing here writes."""
    bound = PV_READ_BOUND_S
    whole = 3 * bound
    return WriteBudget(
        T=bound,
        S=bound,
        D=whole,
        grace_s=whole + GRACE_MARGIN_S,
        relay_timeout_s=min(whole + RELAY_MARGIN_S, MAX_RELAY_TIMEOUT_S),
    )


def _refuse_start(reason: str) -> int:
    """Say why the start is refused; the status to exit with."""
    sys.stderr.write(f"{ipc.START_REFUSED_PREFIX}{reason}\n")
    sys.stderr.flush()
    return EXIT_START_REFUSED


def _audit_batch(session: ControlSession, count: int) -> None:
    """Record one batch: who read, on which target, how many channels. Never raises."""
    record(
        surface=SURFACE,
        posture=posture.posture(),
        posture_source=posture.posture_source(),
        session=f"{SURFACE}:{os.getpid()}",
        subject=READ_SUBJECT,
        decision=DECISION_ALLOWED,
        reason="batch_read",
        detail=f"target={session.binding.target} pv_count={count}",
    )


def _found_row(name: str, reading: ChannelValue) -> ValueRow:
    """The row of a channel that answered; a value too large to send is left out.

    The unit, precision and enum label ride beside the value; an empty unit is none.
    """
    value = ipc.wire_value(reading.value)
    too_large = None
    size = ipc.encoded_size(value)
    if size > ipc.MAX_VALUE_BYTES:
        value = None
        too_large = f"value of {size} bytes is too large to show"
    return ValueRow(
        name=name,
        outcome=ipc.OUTCOME_FOUND,
        value=value,
        timestamp=reading.timestamp.isoformat(),
        severity=ipc.wire_value(reading.metadata.alarm_severity),
        status=ipc.wire_value(reading.metadata.alarm_status),
        reason=too_large,
        units=reading.metadata.units or None,
        precision=reading.metadata.precision,
        enum_label=reading.metadata.enum_label,
    )


async def _read_row(session: ControlSession, name: str) -> ValueRow:
    """Read one channel; whatever goes wrong costs this row only."""
    try:
        reading = await asyncio.wait_for(session.read(name), PV_READ_BOUND_S + ROW_MARGIN_S)
        return _found_row(name, reading)
    except (SessionUnservableError, ConnectorUnavailableError) as exc:
        return ValueRow(name=name, outcome=ipc.OUTCOME_UNAVAILABLE, reason=str(exc))
    except TimeoutError:
        reason = f"no answer within {PV_READ_BOUND_S:g} s"
        return ValueRow(name=name, outcome=ipc.OUTCOME_NO_ANSWER, reason=reason)
    except Exception as exc:
        reason = str(exc) or type(exc).__name__
        return ValueRow(name=name, outcome=ipc.OUTCOME_NO_ANSWER, reason=reason)


async def read_batch(session: ControlSession, names: list[str]) -> list[ValueRow]:
    """One row per name, in order; each channel on its own bound, so a batch takes one."""
    if not names:
        return []
    _audit_batch(session, len(names))
    return list(await asyncio.gather(*(_read_row(session, name) for name in names)))


def _client_handler(session: ControlSession):
    """The connection handler: batch reads until the peer closes or breaks the wire."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                message = await ipc.read_frame(reader, limit=ipc.MAX_REQUEST_BYTES)
                if message is None:
                    return
                rows = await read_batch(session, ipc.decode_request(message))
                writer.write(ipc.encode_response(rows))
                await writer.drain()
        except ipc.FrameError as exc:
            logger.warning("Refused a value request: %s", exc)
        except ConnectionError:
            logger.debug("A value client left mid-request", exc_info=True)
        finally:
            writer.close()

    return handle


async def _wait_moved(session: ControlSession) -> int:
    """Wait for the binding to move; the exit releases the transport, so nothing is closed.

    A watch that fails is treated as a move. Closing first would answer a batch in
    that window with rows that stay unavailable on screen until the next Refresh.
    """
    try:
        await session.wait_moved()
        logger.warning("Control binding %s moved; leaving", session.binding.id)
    except Exception:
        logger.error("Watching the control record failed; leaving", exc_info=True)
    return EXIT_BINDING_MOVED


async def _wait_stale(session: ControlSession) -> int:
    """Wait for the transport to go stale; the exit releases it, so nothing is closed."""
    try:
        await session.wait_stale()
    except Exception:
        # A stale probe that fails says nothing about what is being served.
        logger.error("Watching the transport for staleness failed", exc_info=True)
        await asyncio.get_running_loop().create_future()
    logger.error("Stale transport; leaving")
    return EXIT_TRANSPORT_STALE


async def serve(listener: socket.socket) -> int:
    """Check, arm, open and serve on *listener*; the status to exit with."""
    try:
        check_environment(os.environ)
    except SessionEnvironmentError as exc:
        return _refuse_start("; ".join(exc.problems))

    os.environ[posture.POSTURE_ENV_VAR] = posture.SANDBOX_MODE
    _start_parent_watcher(os.getppid())

    try:
        _arm_raw_put_block()
    except Exception as exc:
        # Anything in a process without the block could put around the connector.
        logger.error("The raw-put block could not be armed", exc_info=True)
        return _refuse_start(f"the raw-put block could not be armed: {exc}")

    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=ipc.MAX_NAMES, thread_name_prefix="channel-values-read")
    )
    try:
        session = await ControlSession.open(surface=SURFACE, budget=_read_budget())
    except SessionEnvironmentError as exc:
        return _refuse_start("; ".join(exc.problems))

    # Never closed (module docstring); the loop's own listener callback holds it.
    await asyncio.start_unix_server(_client_handler(session), sock=listener)
    watchers = {
        asyncio.create_task(_wait_moved(session), name="channel-values-move-watch"),
        asyncio.create_task(_wait_stale(session), name="channel-values-stale-watch"),
    }
    done, pending = await asyncio.wait(watchers, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    return done.pop().result()


async def _serve_and_exit(listener: socket.socket) -> None:
    """Serve, then leave from inside the loop (module docstring)."""
    _exit_process(await serve(listener))


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m osprey.interfaces.channel_finder.directory_values_child``."""
    parser = argparse.ArgumentParser(description="Serve batch value reads for the Channels panel.")
    parser.add_argument(ipc.SOCKET_FD_ARG, type=int, required=True, dest="socket_fd")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        listener = socket.socket(fileno=args.socket_fd)
    except OSError as exc:
        _exit_process(_refuse_start(f"no socket on fd {args.socket_fd}: {exc}"))
        return
    asyncio.run(_serve_and_exit(listener))


if __name__ == "__main__":
    main()
