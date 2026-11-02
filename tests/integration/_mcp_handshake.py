"""Minimal synchronous MCP stdio client for boot-smoke tests.

Spawns an MCP server as a subprocess, performs the JSON-RPC initialize
handshake, then issues tools/list and returns the advertised tool names.

Used by tests/integration/test_build_boot.py.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Iterable
from queue import Empty, Queue

from osprey.deployment.subprocess_capture import observe_exit, stop_child

#: One completed startup phase, as ``osprey.mcp_server.startup.run_mcp_server``
#: reports it on stderr: ``[STARTUP-TIMING] <server> | <phase>: <n>ms``. The
#: first lands within milliseconds of the spawn, which is what makes the marker
#: usable as an announcement rather than only as a measurement.
_STARTUP_TIMING = re.compile(r"\[STARTUP-TIMING\][^|]*\|\s*(\w+):\s*[0-9.]+\s*ms")

#: The phase reported immediately before the server begins reading frames.
_SERVING_PHASE = "total_startup"

#: How long a server that has announced itself may take to finish starting.
#: Separate from the handshake budget on purpose: time spent importing a module
#: is not time the server was given to answer, and on a loaded machine the
#: control-system import alone runs to tens of seconds.
_STARTUP_BUDGET_S = 120.0

#: How many trailing stderr lines a failure report carries. A dying server's
#: stderr opens with startup-timing lines and ends with the traceback that
#: explains the exit, so the report keeps the tail; this is sized to hold a
#: deep Python traceback whole.
_STDERR_TAIL_LINES = 80

#: How long a drain thread, stdout or stderr, may take to reach end-of-file once
#: the server has exited, and how long that exit is waited for after a broken
#: pipe. The pipes close when the process exits, so this bounds only a pipe
#: still held open by a descendant; it guards against a hang and decides nothing.
_DRAIN_JOIN_S = 2.0

#: How long the server's group gets to exit on SIGTERM before it is killed.
_STOP_GRACE_S = 2.0


class MCPHandshakeError(RuntimeError):
    """Raised when the MCP server fails to spawn, parse, or respond."""


def _drain(stream, queue: Queue) -> None:
    for line in iter(stream.readline, ""):
        queue.put(line)
    stream.close()


def list_mcp_tools(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> list[str]:
    """Spawn an MCP stdio server and return the tool names it advertises.

    Sends JSON-RPC initialize, notifications/initialized, then tools/list,
    one frame per line. Reads stdout line-by-line, ignoring lines that do
    not parse as JSON-RPC (servers may emit log lines).

    ``timeout`` is how long the server gets to ANSWER, not how long it gets to
    exist. Before its first read a server spends time that belongs to no frame:
    interpreter start, dotenv, the module import, building the server. A budget
    measured from ``Popen`` is spent on that, and reports a slow start as an
    unanswered handshake. A server that reports its startup phases is therefore
    held to a separate startup allowance first, and only once it says it is
    serving does the handshake clock start. A server that reports nothing is
    indistinguishable from one that is stuck and keeps the literal timeout.

    Raises MCPHandshakeError on spawn failure, timeout, or protocol error.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    try:
        proc = subprocess.Popen(  # command comes from generated .mcp.json
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
            bufsize=1,
            process_group=0,
        )
    except FileNotFoundError as exc:
        raise MCPHandshakeError(f"command not found: {command}") from exc

    stdout_q: Queue = Queue()
    stderr_q: Queue = Queue()
    stdout_drain = threading.Thread(target=_drain, args=(proc.stdout, stdout_q), daemon=True)
    stdout_drain.start()
    stderr_drain = threading.Thread(target=_drain, args=(proc.stderr, stderr_q), daemon=True)
    stderr_drain.start()

    def _send(msg: dict) -> None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
        except BrokenPipeError as exc:
            stderr_drain.join(timeout=_DRAIN_JOIN_S)
            rc = observe_exit(proc, within=_DRAIN_JOIN_S)
            if rc is None:
                raise MCPHandshakeError(
                    f"server closed its stdin before reading id={msg.get('id')}{_stderr_suffix()}"
                ) from exc
            raise MCPHandshakeError(
                f"server exited rc={rc} before reading id={msg.get('id')}{_stderr_suffix()}"
            ) from exc

    stderr_seen: list[str] = []
    announced = False
    serving_at: float | None = None

    def _read_stderr() -> None:
        """Drain what the child has said, and note how far its startup has got."""
        nonlocal announced, serving_at
        try:
            while True:
                line = stderr_q.get_nowait()
                stderr_seen.append(line)
                phase = _STARTUP_TIMING.search(line)
                if phase:
                    announced = True
                    if phase.group(1) == _SERVING_PHASE and serving_at is None:
                        serving_at = time.monotonic()
        except Empty:
            pass

    def _stderr_tail() -> str:
        """The last lines the child wrote to stderr, marked when earlier ones are cut."""
        _read_stderr()
        omitted = len(stderr_seen) - _STDERR_TAIL_LINES
        tail = "".join(stderr_seen[-_STDERR_TAIL_LINES:])
        if omitted > 0:
            return f"({omitted} earlier lines omitted)\n{tail}"
        return tail

    def _stderr_suffix() -> str:
        """``; stderr: <tail>`` when the child wrote anything, else nothing."""
        tail = _stderr_tail()
        return f"; stderr: {tail}" if tail.strip() else ""

    def _deadline() -> float:
        """When the wait expires, given how far the child has got.

        Three cases, and the middle one is the point. A server that has
        announced its startup but not finished it is starting rather than
        stalling, so it is held to the startup allowance instead of the answer
        budget; once it says it is serving, the answer budget starts from
        there. A server that has announced nothing is indistinguishable from
        one that is stuck, and keeps the caller's budget exactly as given.
        """
        if serving_at is not None:
            return serving_at + timeout
        if announced:
            return max(spawn_deadline, spawned_at + _STARTUP_BUDGET_S)
        return spawn_deadline

    def _recv(target_id: int) -> dict:
        while True:
            _read_stderr()
            remaining = _deadline() - time.monotonic()
            if remaining <= 0:
                raise MCPHandshakeError(f"timeout waiting for id={target_id}{_stderr_suffix()}")
            rc = observe_exit(proc)
            if rc is not None:
                stderr_drain.join(timeout=_DRAIN_JOIN_S)
                raise MCPHandshakeError(
                    f"server exited rc={rc} before responding to id={target_id}{_stderr_suffix()}"
                )
            try:
                line = stdout_q.get(timeout=min(0.25, remaining))
            except Empty:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # log line, ignore
            if isinstance(msg, dict) and msg.get("id") == target_id:
                if "error" in msg:
                    raise MCPHandshakeError(f"JSON-RPC error: {msg['error']}")
                return msg

    spawned_at = time.monotonic()
    spawn_deadline = spawned_at + timeout
    failure: MCPHandshakeError | None = None
    other: BaseException | None = None

    try:
        _send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "osprey-boot-smoke", "version": "0"},
                },
            }
        )
        _recv(1)

        _send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        _send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        resp = _recv(2)
        tools = resp.get("result", {}).get("tools", [])
        return [t["name"] for t in tools if isinstance(t, dict) and "name" in t]
    except MCPHandshakeError as exc:
        failure = exc
        raise
    except BaseException as exc:
        other = exc
        raise
    finally:
        with contextlib.suppress(OSError):
            proc.stdin.close()
        outcome = stop_child(proc, grace=_STOP_GRACE_S)
        stdout_drain.join(timeout=_DRAIN_JOIN_S)
        stderr_drain.join(timeout=_DRAIN_JOIN_S)
        if not outcome.gone:
            left = f"server group {proc.pid}: {outcome.describe()}"
            if failure is not None:
                raise MCPHandshakeError(f"{failure}; {left}") from failure
            if other is not None:
                other.add_note(left)
            else:
                raise MCPHandshakeError(left)


def assert_tools_superset(
    server_name: str,
    actual: Iterable[str],
    expected: Iterable[str],
) -> None:
    """Helper for callers — assert ``actual`` ⊇ ``expected``, with a clear diff message."""
    actual_set = set(actual)
    missing = sorted(set(expected) - actual_set)
    if missing:
        raise AssertionError(
            f"MCP server {server_name!r} did not advertise expected tools.\n"
            f"  missing: {missing}\n"
            f"  actual:  {sorted(actual_set)}"
        )
