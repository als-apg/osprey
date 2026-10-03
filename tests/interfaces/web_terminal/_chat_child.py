"""A stand-in for the Simple view's agent child.

A real process the runner holds, which starts the processes the tests then
look for: shell commands in sessions of their own, as the agent CLI runs them,
and a helper in the group the child is in — the server's own, the way an MCP
server or a python-executor sandbox is. ``ChildClient`` is the
``ClaudeSDKClient`` slice ``OperatorSession`` reaches through the agent
runner, so ``sdk_seam`` can hand a real child to a real session.
"""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from osprey.interfaces.web_terminal.operator_session import OperatorSession
from tests.interfaces.web_terminal._fakes import sdk_seam
from tests.interfaces.web_terminal._pty_child import (
    CHILD_HANG_CEILING,
    sleeper_script,
)

# How long the stand-in's close waits after each step, as the SDK's close does.
_CLOSE_WAIT_S = 2.0


def chat_child_script(
    pid_file: Path, *, scripts: Sequence[Path], helper_ignores_term: bool = False
) -> str:
    """Return the source of a chat child that starts processes the way the agent CLI does.

    Each path in *scripts* runs as a grandchild in a session (and process
    group) of its own, as the agent's shell commands do. One helper runs in
    the group the child is in, which is the server's own; it ignores SIGHUP
    and SIGTERM when *helper_ignores_term*. Every one of them has its stdio on
    ``DEVNULL``. The grandchildren's pids and then the helper's are written to
    *pid_file*, one per line, under a temporary name renamed into place. A
    daemon thread reaps the child's exited children, as the agent CLI reaps
    the commands it ran, so a command that exits does not linger as a zombie.
    The child then reads stdin until EOF and exits, as the agent CLI does when
    its runner closes.
    """
    helper = sleeper_script(
        pid_file.parent,
        "mcp_helper.py",
        ignore=(signal.SIGHUP, signal.SIGTERM) if helper_ignores_term else (),
    )
    return f"""
import os, subprocess, sys, threading, time
def spawn(path, new_session):
    return subprocess.Popen(
        [sys.executable, path],
        start_new_session=new_session,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
procs = [spawn(path, True) for path in {[str(p) for p in scripts]!r}]
procs.append(spawn({str(helper)!r}, False))
tmp = {str(pid_file) + ".tmp"!r}
with open(tmp, "w") as fh:
    fh.write("".join(f"{{p.pid}}\\n" for p in procs))
os.replace(tmp, {str(pid_file)!r})
def reap():
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            time.sleep(0.05)
threading.Thread(target=reap, daemon=True).start()
sys.stdin.buffer.read()
"""


class ChildHandle:
    """The slice of the SDK transport's process handle the runner reads."""

    def __init__(self, popen: subprocess.Popen[bytes]) -> None:
        self._popen = popen

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def returncode(self) -> int | None:
        return self._popen.poll()

    def kill(self) -> None:
        self._popen.kill()


class ChildClient:
    """A ``ClaudeSDKClient`` stand-in whose transport holds a real child."""

    def __init__(self, argv: Sequence[str]) -> None:
        self._argv = list(argv)
        self._popen: subprocess.Popen[bytes] | None = None
        self._transport: Any = None

    async def __aenter__(self) -> ChildClient:
        self._popen = subprocess.Popen(self._argv, stdin=subprocess.PIPE)
        self._transport = SimpleNamespace(_process=ChildHandle(self._popen))
        return self

    async def __aexit__(self, *exc: object) -> bool:
        popen = self._popen
        if popen is not None:
            if popen.stdin is not None:
                try:
                    popen.stdin.close()
                except OSError:
                    pass
            for signum in (signal.SIGTERM, signal.SIGKILL):
                if await _waited(popen, _CLOSE_WAIT_S):
                    break
                try:
                    popen.send_signal(signum)
                except ProcessLookupError:
                    break
            await _waited(popen, _CLOSE_WAIT_S)
        self._transport = None
        return False

    async def interrupt(self) -> None:
        return None


async def _waited(popen: subprocess.Popen[bytes], seconds: float) -> bool:
    try:
        await asyncio.to_thread(popen.wait, seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


def child_factory(
    tmp_path: Path,
    *,
    scripts: Sequence[str] = (),
    helper_ignores_term: bool = False,
    script_ignores: tuple[int, ...] = (),
) -> Callable[..., ChildClient]:
    """Return a client factory ``sdk_seam`` calls once per child it starts.

    Each call writes the *scripts* (sleepers named as given) and the child's
    source into ``tmp_path / f"c{n}"`` and returns a :class:`ChildClient` for
    it. ``factory.pid_files`` lists each child's pid file in creation order.
    """
    pid_files: list[Path] = []

    def factory(**_kwargs: Any) -> ChildClient:
        directory = tmp_path / f"c{len(pid_files)}"
        directory.mkdir()
        paths = [sleeper_script(directory, name, ignore=script_ignores) for name in scripts]
        pid_file = directory / "pids"
        pid_files.append(pid_file)
        source = chat_child_script(pid_file, scripts=paths, helper_ignores_term=helper_ignores_term)
        return ChildClient([sys.executable, "-c", source])

    factory.pid_files = pid_files  # type: ignore[attr-defined]
    factory.cwd = tmp_path  # type: ignore[attr-defined]
    return factory


def wait_for_chat_pids(session: OperatorSession, path: Path, count: int) -> list[int]:
    """Wait until *path* holds *count* pids, one per line, and return them."""
    ceiling = time.monotonic() + CHILD_HANG_CEILING
    while True:
        if path.exists():
            lines = path.read_text().split()
            if len(lines) >= count:
                return [int(line) for line in lines[:count]]
        if session.process_exited:
            raise AssertionError(
                f"the chat child exited before writing {count} pid(s) to {path.name}"
            )
        if time.monotonic() > ceiling:
            raise AssertionError(
                f"the chat child is alive but has not written {count} pid(s) to "
                f"{path.name} within the {CHILD_HANG_CEILING:g} s hang ceiling"
            )
        time.sleep(0.05)


async def start_chat(
    factory: Callable[..., ChildClient], *, session_key: str | None = None
) -> OperatorSession:
    """Return an ``OperatorSession`` started on a child *factory* builds."""
    cwd = getattr(factory, "cwd", None) or tempfile.gettempdir()
    session = OperatorSession(cwd=str(cwd), session_key=session_key)
    with sdk_seam(factory):
        await session.start()
    return session
