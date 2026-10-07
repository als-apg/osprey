"""Waits shared by the web-terminal tests that drive a real PTY child.

Each wait ends on an observation of the child: its answer arrived, or it is
dead without having answered. A dead child fails the wait at once with its exit
code and whatever it printed, so a broken child is never mistaken for a slow
one. ``CHILD_HANG_CEILING`` only ends a wait on a child that is alive and
silent; it never decides a pass. Some children start grandchildren of their
own, and every test that starts one kills them in ``finally``.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.interfaces.web_terminal.pty_manager import PtySession

#: Hard ceiling on any wait for a PTY child, whatever it is doing. A child that
#: has died fails the wait at once, so this only ends a wait on a child that is
#: alive and silent; it never decides a pass. It sits below pytest-timeout's
#: per-test cap (300 s in the fast gate, 600 s in CI) so the failure is this
#: module's message rather than a thread dump.
CHILD_HANG_CEILING = 120.0

#: What the shell prints once it has executed every line typed before it.
SENTINEL = b"__OSPREY_DONE__"

#: A PTY echoes typed input, so the typed line must not contain the sentinel's
#: bytes: the empty quotes split it, and only the shell executing the line joins
#: it back together.
_SENTINEL_COMMAND = b'echo __OSPREY_""DONE__\n'

_POLL_S = 0.02


def read_answer(session: PtySession, command: bytes) -> bytes:
    """Type *command* into the child's shell and return everything up to its answer.

    The wait ends when the shell prints ``SENTINEL`` (typed after *command*, so
    its output follows the command's), or fails at once when the child is dead
    without having printed it. The returned bytes include the PTY's echo of the
    typed lines, so callers assert with ``in``. A typed literal that the echo
    alone would satisfy proves nothing: split it with empty quotes, as the
    sentinel is, so only execution produces it.
    """
    try:
        session.write_input(command + b"\n" + _SENTINEL_COMMAND)
    except OSError:
        pass  # A dead child is reported below, with its exit code.

    output = b""
    ceiling = time.monotonic() + CHILD_HANG_CEILING
    while True:
        if SENTINEL in output:
            return output
        alive = session.is_alive
        readable, _, _ = select.select([session._master_fd], [], [], 0.1)
        chunk = b""
        if readable:
            try:
                chunk = os.read(session._master_fd, 4096)
            except OSError:
                chunk = b""
        if chunk:
            output += chunk
            continue
        if not alive:
            raise AssertionError(
                f"the PTY child exited with code {session.exit_code} before answering "
                f"{command!r}; it printed {output!r}"
            )
        if time.monotonic() > ceiling:
            raise AssertionError(
                f"the PTY child is alive but has not answered {command!r} within the "
                f"{CHILD_HANG_CEILING:g} s hang ceiling; it printed {output!r}"
            )


def _report_lines(report: Path) -> list[str]:
    return report.read_text().split() if report.exists() else []


def wait_for_report(report: Path, expected: int, child: PtySession) -> list[str]:
    """Wait until *report* holds *expected* lines, then return them.

    Takes the one child that owes the next line: earlier generations of the
    same report are dead by design (a respawn kills them), so only the owed
    child's death is a failure. A child that wrote its line and then exited
    passes, because the file is read once more after its death is seen.
    """
    ceiling = time.monotonic() + CHILD_HANG_CEILING
    while True:
        lines = _report_lines(report)
        if len(lines) >= expected:
            return lines
        if not child.is_alive:
            lines = _report_lines(report)
            if len(lines) >= expected:
                return lines
            raise AssertionError(
                f"the reporting child exited with code {child.exit_code} with {len(lines)} "
                f"of {expected} line(s) in {report.name}: {lines!r}"
            )
        if time.monotonic() > ceiling:
            raise AssertionError(
                f"the reporting child is alive but has not written {expected} line(s) to "
                f"{report.name} within the {CHILD_HANG_CEILING:g} s hang ceiling; "
                f"it wrote {lines!r}"
            )
        time.sleep(_POLL_S)


def wait_for_exit(session: PtySession) -> int | None:
    """Wait for the child to die and return its exit code."""
    ceiling = time.monotonic() + CHILD_HANG_CEILING
    while session.is_alive:
        if time.monotonic() > ceiling:
            raise AssertionError(
                f"the PTY child is still running after the {CHILD_HANG_CEILING:g} s hang ceiling"
            )
        time.sleep(_POLL_S)
    return session.exit_code


def sleeper_script(directory: Path, name: str, *, ignore: tuple[int, ...] = ()) -> Path:
    """Write to ``directory / name`` a program that ignores *ignore*, then sleeps.

    The program prints one line once its signal dispositions are set, so a
    parent that reads that line knows a signal sent afterwards meets them.
    Returns the path, so the process's command line ends in *name*.
    """
    path = directory / name
    path.write_text(
        "import signal, sys, time\n"
        f"for signum in {[int(s) for s in ignore]!r}:\n"
        "    signal.signal(signum, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(120)\n"
    )
    return path


def detaching_child_script(
    pid_file: Path,
    *,
    scripts: Sequence[Path],
    helper_ignores_hup: bool = False,
    input_file: Path | None = None,
) -> str:
    """Return the source of a PTY child that starts processes the way the agent CLI does.

    Each path in *scripts* runs as a grandchild in a session (and process
    group) of its own, as the agent's shell commands do. One helper runs in
    the child's own group, as the agent's MCP servers do; it ignores SIGHUP
    when *helper_ignores_hup*. Once every one of them has set its signal
    dispositions, the grandchildren's pids and then the helper's are written
    to *pid_file*, one per line, under a temporary name renamed into place.
    The child then reads stdin a byte at a time, appending to *input_file*
    when given. It keeps SIGHUP's default disposition, so the hang-up kills it.
    """
    helper = sleeper_script(
        pid_file.parent,
        "mcp_helper.py",
        ignore=(signal.SIGHUP,) if helper_ignores_hup else (),
    )
    return f"""
import os, subprocess, sys, tty
tty.setraw(0)
def spawn(path, new_session):
    proc = subprocess.Popen(
        [sys.executable, path],
        start_new_session=new_session,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    )
    proc.stdout.readline()
    return proc
procs = [spawn(path, True) for path in {[str(p) for p in scripts]!r}]
procs.append(spawn({str(helper)!r}, False))
tmp = {str(pid_file) + ".tmp"!r}
with open(tmp, "w") as fh:
    fh.write("".join(f"{{p.pid}}\\n" for p in procs))
os.replace(tmp, {str(pid_file)!r})
input_file = {str(input_file) if input_file is not None else None!r}
while True:
    try:
        byte = os.read(0, 1)
    except OSError:
        break
    if not byte:
        break
    if input_file is not None:
        with open(input_file, "ab") as fh:
            fh.write(byte)
"""


def wait_for_pids(session: PtySession, path: Path, count: int) -> list[int]:
    """Wait until *path* holds *count* pids, one per line, and return them."""
    ceiling = time.monotonic() + CHILD_HANG_CEILING
    while True:
        if path.exists():
            lines = path.read_text().split()
            if len(lines) >= count:
                return [int(line) for line in lines[:count]]
        if not session.is_alive:
            raise AssertionError(
                f"the PTY child exited with code {session.exit_code} before writing "
                f"{count} pid(s) to {path.name}"
            )
        if time.monotonic() > ceiling:
            raise AssertionError(
                f"the PTY child is alive but has not written {count} pid(s) to "
                f"{path.name} within the {CHILD_HANG_CEILING:g} s hang ceiling"
            )
        time.sleep(0.05)


def pid_gone(pid: int, within: float = 5.0) -> bool:
    """Return whether *pid* is gone, or a zombie nobody has reaped yet, within *within* s."""
    deadline = time.monotonic() + within
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        if not state or state.startswith("Z"):
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def kill_quietly(pids: Iterable[int]) -> None:
    """SIGKILL every pid in *pids*, ignoring those already gone."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
