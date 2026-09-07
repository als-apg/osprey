"""Minimal synchronous MCP stdio client for boot-smoke tests.

Spawns an MCP server as a subprocess, performs the JSON-RPC initialize
handshake, then issues tools/list and returns the advertised tool names.

Used by tests/integration/test_build_boot.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Iterable
from queue import Empty, Queue


class MCPHandshakeError(RuntimeError):
    """Raised when the MCP server fails to spawn, parse, or respond."""


def _drain(stream, queue: Queue) -> None:
    for line in iter(stream.readline, ""):
        queue.put(line)
    stream.close()


#: How many times the inactivity budget a whole handshake may take before it is
#: abandoned regardless of how chatty the server is. The inactivity budget below
#: is the real liveness check; this only bounds the pathological case of a server
#: that talks forever and answers nothing.
_HARD_TIMEOUT_FACTOR = 6.0


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

    ``timeout`` is an **inactivity** budget, not a total one: it is the longest
    the server may go without writing anything at all. A total budget measures
    the machine rather than the server — these servers import the framework in
    the child process, which is seconds of work on an idle box and much more on
    one running the rest of the suite in parallel, so a wall-clock deadline
    started before the import turns a busy CI runner into a red test. Silence is
    the honest signal for a server that will never answer, and it is unaffected
    by load. :data:`_HARD_TIMEOUT_FACTOR` bounds the remaining pathological case.

    Raises MCPHandshakeError on spawn failure, timeout, or protocol error.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    try:
        proc = subprocess.Popen(  # noqa: S603 - command comes from generated .mcp.json
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise MCPHandshakeError(f"command not found: {command}") from exc

    stdout_q: Queue = Queue()
    stderr_q: Queue = Queue()
    threading.Thread(target=_drain, args=(proc.stdout, stdout_q), daemon=True).start()
    threading.Thread(target=_drain, args=(proc.stderr, stderr_q), daemon=True).start()

    #: Everything the server has written to stderr so far. Drained continuously
    #: rather than only on failure, because stderr traffic is what proves the
    #: server is still making progress.
    stderr_seen: list[str] = []

    def _send(msg: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def _drain_stderr() -> bool:
        """Move whatever stderr has produced into the buffer; True if any."""
        drained = _collect(stderr_q)
        if drained:
            stderr_seen.append(drained)
            return True
        return False

    def _recv(target_id: int, hard_deadline: float) -> dict:
        import time

        last_activity = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= hard_deadline:
                _drain_stderr()
                raise MCPHandshakeError(
                    f"gave up waiting for id={target_id}: the server kept writing for the "
                    f"whole {timeout * _HARD_TIMEOUT_FACTOR:g}s handshake budget without "
                    f"replying; stderr: {''.join(stderr_seen)[-600:]}"
                )
            if now - last_activity >= timeout:
                raise MCPHandshakeError(
                    f"timeout waiting for id={target_id}: no output for {timeout:g}s; "
                    f"stderr: {''.join(stderr_seen)[-600:]}"
                )
            if proc.poll() is not None:
                _drain_stderr()
                raise MCPHandshakeError(
                    f"server exited rc={proc.returncode} before responding to id={target_id}; "
                    f"stderr: {''.join(stderr_seen)[-600:]}"
                )
            if _drain_stderr():
                last_activity = time.monotonic()
            try:
                line = stdout_q.get(timeout=0.25)
            except Empty:
                continue
            last_activity = time.monotonic()
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

    import time

    deadline = time.monotonic() + timeout * _HARD_TIMEOUT_FACTOR

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
        _recv(1, deadline)

        _send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        _send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        resp = _recv(2, deadline)
        tools = resp.get("result", {}).get("tools", [])
        return [t["name"] for t in tools if isinstance(t, dict) and "name" in t]
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def _collect(q: Queue) -> str:
    parts: list[str] = []
    try:
        while True:
            parts.append(q.get_nowait())
    except Empty:
        pass
    return "".join(parts)


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
