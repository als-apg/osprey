"""The boot-smoke handshake budget measures the server, not the machine.

``list_mcp_tools`` spawns a real MCP server and waits for it to answer. Those
servers import the framework in the child process, which takes seconds on an
idle box and much longer on one running the rest of the suite in parallel — so
a budget started at ``Popen`` is spent on the import and reports a slow start
as an unanswered handshake.

The helper therefore reads the server's own startup phases from stderr: a server
that announces a phase is starting rather than stalling and is held to the
startup allowance; once it reports ``total_startup`` the answer budget begins.
A server that announces nothing is indistinguishable from one that is stuck and
keeps the literal budget. These tests pin both halves — a slow server that
reports its phases is waited for, a mute one is not — with stub servers rather
than the real ones, so they cost seconds: they start bare interpreters and no
service (the group-stop proof's stub also starts one sleeper child for the stop
to end), unlike the container-backed modules beside them. The two halves read
a real budget, so they carry ``wallclock`` and are timed alone; the others hold
the helper to its contract on any clock.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from osprey.deployment import subprocess_capture
from tests._live import wait_until
from tests._processes import kill_group, kill_quietly
from tests._scoped_stdlib import scoped
from tests.integration import _mcp_handshake
from tests.integration._mcp_handshake import MCPHandshakeError, list_mcp_tools

#: The helper leaves no pipe or child behind: a pipe or process left for the
#: collector warns from a finalizer, which pytest reports as unraisable, so both
#: are errors here.
pytestmark = [
    pytest.mark.filterwarnings("error::ResourceWarning"),
    pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning"),
]


@pytest.fixture(autouse=True)
def _earlier_garbage_frozen() -> Iterator[None]:
    """Leave what earlier tests on this worker made out of this test's collections.

    The tests here run ``gc.collect()`` so that a pipe or process the helper
    leaks fails them. Unfrozen, that call also walks the whole heap the worker
    built before this test, which on a loaded runner costs minutes, and
    finalizes any cycle an earlier test left behind (an unclosed event loop,
    say), blaming it on the helper. Frozen objects are left out of a
    collection, so it covers only what this test made.
    """
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


#: A stub MCP server. Given a ``gate`` directory it announces one startup phase
#: in the framework's own ``[STARTUP-TIMING]`` shape as soon as the interpreter
#: runs, marks that by creating ``gate/announced``, and closes with
#: ``total_startup`` only once ``gate/go`` exists. Without one it announces
#: nothing and sleeps ``mute`` seconds first. Then it speaks JSON-RPC.
_STUB = """\
import json, pathlib, sys, time

gate = {gate!r}
if gate:
    print("[STARTUP-TIMING] stub | import_0: 1ms", file=sys.stderr, flush=True)
    pathlib.Path(gate, "announced").touch()
    deadline = time.monotonic() + 30
    while not pathlib.Path(gate, "go").exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    print("[STARTUP-TIMING] stub | total_startup: 1ms", file=sys.stderr, flush=True)

time.sleep({mute})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if msg.get("method") == "initialize":
        reply = {{"jsonrpc": "2.0", "id": msg["id"], "result": {{"capabilities": {{}}}}}}
    elif msg.get("method") == "tools/list":
        reply = {{"jsonrpc": "2.0", "id": msg["id"],
                  "result": {{"tools": [{{"name": "stub_tool"}}]}}}}
    else:
        continue
    print(json.dumps(reply), flush=True)
"""

#: A stub MCP server that dies at startup the way a framework server does: a
#: run of ``[STARTUP-TIMING]`` lines first, then an uncaught exception, exit 1,
#: before it reads a frame. The explaining traceback is the END of its stderr.
_DYING_STUB = """\
import sys

for i in range(40):
    print(f"[STARTUP-TIMING] stub | import_{{i}}: 1ms", file=sys.stderr, flush=True)
raise RuntimeError("{marker}")
"""

#: The dying stub's exception message: present in the error text only if the
#: report carries the tail of stderr.
_DYING_MARKER = "stub-server-startup-failure-marker"

#: A stub MCP server that leaves a child holding its pipes: it records its
#: process group and its child's pid, starts a sleeper that inherits stdout and
#: stderr (same group, both pipes held open), announces one startup phase, then
#: dies before reading a frame.
_FORKING_STUB = """\
import pathlib, subprocess, sys, os

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pathlib.Path({record!r}).write_text(f"{{os.getpgid(0)}} {{child.pid}}")
print("[STARTUP-TIMING] stub | import_0: 1ms", file=sys.stderr, flush=True)
raise RuntimeError("{marker}")
"""


def _stub(tmp_path: Path, *, gate: Path | None = None, mute: float = 0.0) -> list[str]:
    script = tmp_path / "stub_server.py"
    gate_dir = None if gate is None else str(gate)
    script.write_text(textwrap.dedent(_STUB).format(gate=gate_dir, mute=mute))
    return [sys.executable, str(script)]


def _dying_server(tmp_path: Path) -> list[str]:
    script = tmp_path / "dying_server.py"
    script.write_text(_DYING_STUB.format(marker=_DYING_MARKER))
    return [str(script)]


def _on_spawn(monkeypatch: pytest.MonkeyPatch, hook: Callable[[subprocess.Popen], object]) -> None:
    """Run *hook* on each process the helper spawns, before ``Popen`` returns it."""
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        hook(process)
        return process

    scoped(monkeypatch, "tests.integration._mcp_handshake", "subprocess", Popen=popen)


def _handshake_failure(command: str, args: list[str]) -> str:
    """The handshake error's text, with no frame left holding the helper's child."""
    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=command, args=args, timeout=10.0)
    return str(excinfo.value)


#: How far the announcing-server test moves the helper's clock during the
#: stub's startup: far past a small multiple of the one-second budget, and
#: short of the 120 s startup allowance.
_ANNOUNCED_STARTUP_S = 100.0
#: How far each clock read moves it on meanwhile: more than the budget, so a
#: deadline taken at one read has passed by the next.
_STARTUP_STEP_S = 25.0


@pytest.mark.wallclock
def test_a_slow_but_announcing_server_is_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hundred seconds of announced startup against a one-second budget still passes.

    This is the shape that reds a loaded runner: the server is fine, just slow
    to reach its first reply. The spawn returns once the stub has printed its
    first phase. From the helper's first read of a phase, each read of its
    clock moves it on :data:`_STARTUP_STEP_S`, and the stub finishes starting
    only after :data:`_ANNOUNCED_STARTUP_S`. So a wait held to the budget or a
    few times it, from the spawn or from the announcement, or one that forgets
    the announcement between reads, gives up first.
    """
    gate = tmp_path / "gate"
    gate.mkdir()
    command, *args = _stub(tmp_path, gate=gate)
    real_phase = _mcp_handshake._STARTUP_TIMING

    class _Clock:
        """The helper's ``time`` and phase pattern: real time, moved on as above."""

        offset = 0.0
        starting = False
        released = False

        def __getattr__(self, name: str):
            return getattr(time, name)

        def search(self, line: str):
            found = real_phase.search(line)
            self.starting = self.starting or found is not None
            return found

        def monotonic(self) -> float:
            if self.starting and not self.released:
                self.offset += _STARTUP_STEP_S
                if self.offset >= _ANNOUNCED_STARTUP_S:
                    self.released = True
                    (gate / "go").touch()
            return time.monotonic() + self.offset

    clock = _Clock()
    _on_spawn(
        monkeypatch, lambda _: wait_until((gate / "announced").exists, timeout=30, interval=0.005)
    )
    monkeypatch.setattr(_mcp_handshake, "time", clock)
    monkeypatch.setattr(_mcp_handshake, "_STARTUP_TIMING", clock)

    assert list_mcp_tools(command=command, args=args, timeout=1.0) == ["stub_tool"]
    assert clock.released


@pytest.mark.wallclock
def test_a_mute_server_still_times_out(tmp_path: Path) -> None:
    """The half that must not weaken: a server that announces nothing keeps the budget.

    Against half a second of budget, the stub's 5 s of silence catches a wait
    ten times the budget, or 4.5 s past it.
    """
    command, *args = _stub(tmp_path, mute=5.0)

    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=command, args=args, timeout=0.5)

    assert "timeout waiting for" in str(excinfo.value)


def test_a_dying_server_reports_its_exit_code_and_traceback(tmp_path: Path) -> None:
    """A server that exits before answering is reported with the tail of its stderr.

    A framework server opens stderr with startup-timing lines, so the head of
    what it said is timing and the traceback that explains the exit is the tail.
    """
    message = _handshake_failure(sys.executable, _dying_server(tmp_path))

    assert "rc=1" in message
    assert "Traceback" in message
    assert _DYING_MARKER in message


def test_a_server_dead_before_the_first_message_reports_its_exit_code_and_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server already gone when the first frame is written is reported like any early exit.

    The spawn is held until the child has exited, so the first write meets a
    closed pipe every time rather than only when the machine is loaded.
    """

    def exited(process: subprocess.Popen) -> None:
        # Observed, not reaped: the helper's stop is the one reaper of its server.
        assert subprocess_capture.observe_exit(process, within=30.0) is not None

    _on_spawn(monkeypatch, exited)
    message = _handshake_failure(sys.executable, _dying_server(tmp_path))

    assert "rc=1" in message
    assert "Traceback" in message
    assert _DYING_MARKER in message


def test_a_silent_dying_server_is_reported_by_its_exit_code_alone(tmp_path: Path) -> None:
    """A server that exits having written nothing to stderr gets no empty stderr clause."""
    script = tmp_path / "silent_server.py"
    script.write_text("import sys, time\ntime.sleep(0.5)\nsys.exit(1)\n")

    message = _handshake_failure(sys.executable, [str(script)])

    assert message.endswith("rc=1 before responding to id=1")


def test_a_server_that_closes_its_stdin_and_keeps_running_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that stops reading but does not exit is a handshake failure, never a hang.

    The spawn returns once the child has closed its stdin, so the first write
    meets a closed pipe while the child is alive. The helper's two waits after
    the broken pipe are cut to 0.2 s and the child lives fifteen times that, so
    only a wait well past the constant sees it exit.
    """
    monkeypatch.setattr(_mcp_handshake, "_DRAIN_JOIN_S", 0.2)
    flag = tmp_path / "stdin_closed"
    script = tmp_path / "deaf_server.py"
    script.write_text(
        "import os, pathlib, sys, time\n"
        "os.close(0)\n"
        f"pathlib.Path({str(flag)!r}).touch()\n"
        f"time.sleep({15 * _mcp_handshake._DRAIN_JOIN_S})\n"
    )
    _on_spawn(monkeypatch, lambda _: wait_until(flag.exists, timeout=30))

    message = _handshake_failure(sys.executable, [str(script)])

    assert message.endswith("server closed its stdin before reading id=1")


def test_a_finished_handshake_leaves_the_childs_stdin_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper closes the pipe it writes to, however the handshake ends."""
    made: list[subprocess.Popen] = []
    _on_spawn(monkeypatch, made.append)

    _handshake_failure(sys.executable, _dying_server(tmp_path))

    assert len(made) == 1 and made[0].stdin is not None and made[0].stdin.closed


def test_a_server_whose_child_holds_its_pipes_leaves_no_group_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper ends the server's whole group, not only its leader."""
    record = tmp_path / "group"
    script = tmp_path / "forking_server.py"
    script.write_text(_FORKING_STUB.format(record=str(record), marker=_DYING_MARKER))
    leaders: list[int] = []
    _on_spawn(monkeypatch, lambda process: leaders.append(process.pid))
    try:
        message = _handshake_failure(sys.executable, [str(script)])
        gc.collect()
        pgid, _ = (int(field) for field in record.read_text().split())

        assert "rc=1" in message
        assert subprocess_capture.group_is_gone(pgid)
        assert leaders == [pgid]
    finally:
        # A red run must leave nothing: kill the server's group unless it is ours.
        if record.exists():
            pgid, grandchild = (int(field) for field in record.read_text().split())
            if pgid != os.getpgrp():
                kill_group(pgid)
            kill_quietly([grandchild])


def _stop_reporting(unproven: str, stopped: list[int]):
    """A ``stop_child`` that really stops the child, then answers it left ``unproven``.

    ``unproven`` is ``"survivors"`` (pid 424242 outlived SIGKILL) or
    ``"unknown"`` (the process table could not be read for the group).
    """
    real = subprocess_capture.stop_child

    def stop(process: subprocess.Popen, **kwargs) -> subprocess_capture.StopOutcome[int]:
        stopped.append(process.pid)
        real(process, **kwargs)
        held = (424242,) if unproven == "survivors" else (process.pid,)
        return subprocess_capture.StopOutcome(ended=(process.pid,), **{unproven: held})

    return stop


def test_a_survivor_is_named_in_the_handshake_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group still running after the stop is added to the error that ended the handshake."""
    stopped: list[int] = []
    monkeypatch.setattr(_mcp_handshake, "stop_child", _stop_reporting("survivors", stopped))
    message = _handshake_failure(sys.executable, _dying_server(tmp_path))
    gc.collect()

    assert "rc=1" in message
    assert _DYING_MARKER in message
    assert message.endswith(f"; server group {stopped[0]}: still running after SIGKILL: pid 424242")
    assert len(stopped) == 1


def test_a_survivor_after_a_good_handshake_fails_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handshake that answered still fails when the server's group outlives the stop."""
    stopped: list[int] = []
    monkeypatch.setattr(_mcp_handshake, "stop_child", _stop_reporting("survivors", stopped))
    command, *args = _stub(tmp_path)

    message = _handshake_failure(command, args)
    gc.collect()

    assert message == f"server group {stopped[0]}: still running after SIGKILL: pid 424242"


def test_an_unreadable_process_table_after_the_stop_fails_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop that could not look is never read as a group that is gone."""
    stopped: list[int] = []
    monkeypatch.setattr(_mcp_handshake, "stop_child", _stop_reporting("unknown", stopped))
    command, *args = _stub(tmp_path)

    message = _handshake_failure(command, args)
    gc.collect()

    assert message == (
        f"server group {stopped[0]}: "
        f"process table unreadable: group {stopped[0]} may still be running"
    )
