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
to end), unlike the container-backed modules beside them.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from osprey.deployment import subprocess_capture
from tests._processes import kill_group, kill_quietly
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
def _earlier_garbage_swept() -> None:
    """Finalize what earlier tests on this worker left for the collector, unreported.

    The tests here run ``gc.collect()`` so that a pipe or process the helper
    leaks fails them. The same call also finalizes any cycle an earlier test on
    the worker left behind (an unclosed event loop, say) and would blame it on
    the helper; sweeping first leaves only what the test itself made.
    """
    previous = sys.unraisablehook
    sys.unraisablehook = lambda _report: None
    try:
        gc.collect()
    finally:
        sys.unraisablehook = previous


#: A stub MCP server. With ``phases`` above zero it reports that many startup
#: phases to stderr in the framework's own ``[STARTUP-TIMING]`` shape, one every
#: ``gap`` seconds, closes with ``total_startup``, then speaks JSON-RPC. With
#: ``phases=0`` it announces nothing and sleeps ``mute`` seconds first.
_STUB = """\
import json, sys, time

for i in range({phases}):
    time.sleep({gap})
    print(f"[STARTUP-TIMING] stub | import_{{i}}: {{ {gap} * 1000:.0f}}ms", file=sys.stderr, flush=True)
if {phases}:
    print(f"[STARTUP-TIMING] stub | total_startup: {{ {phases} * {gap} * 1000:.0f}}ms", file=sys.stderr, flush=True)

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


def _stub(tmp_path: Path, *, phases: int = 0, gap: float = 0.0, mute: float = 0.0) -> list[str]:
    script = tmp_path / "stub_server.py"
    script.write_text(textwrap.dedent(_STUB).format(phases=phases, gap=gap, mute=mute))
    return [sys.executable, str(script)]


def test_a_slow_but_announcing_server_is_waited_for(tmp_path: Path) -> None:
    """Six seconds of announced startup against a three-second budget still passes.

    This is the shape that reds a loaded runner: the server is fine, it is just
    taking longer to reach its first reply than a budget started at spawn would
    allow. Its first phase has to land inside the budget — it does within a
    second even on a loaded box — and from there the startup allowance holds.
    """
    command, *args = _stub(tmp_path, phases=12, gap=0.5)

    assert list_mcp_tools(command=command, args=args, timeout=3.0) == ["stub_tool"]


def test_a_mute_server_still_times_out(tmp_path: Path) -> None:
    """The half that must not weaken: a server that announces nothing keeps the budget."""
    command, *args = _stub(tmp_path, mute=30.0)

    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=command, args=args, timeout=2.0)

    assert "timeout waiting for" in str(excinfo.value)


def test_a_dying_server_reports_its_exit_code_and_traceback(tmp_path: Path) -> None:
    """A server that exits before answering is reported with the tail of its stderr.

    A framework server opens stderr with startup-timing lines, so the head of
    what it said is timing and the traceback that explains the exit is the tail.
    """
    script = tmp_path / "dying_server.py"
    script.write_text(_DYING_STUB.format(marker=_DYING_MARKER))

    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=sys.executable, args=[str(script)], timeout=10.0)

    message = str(excinfo.value)
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
    real_popen = subprocess.Popen

    def popen_after_exit(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        # Observed, not reaped: the helper's stop is the one reaper of its server.
        assert subprocess_capture.observe_exit(proc, within=30.0) is not None
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen_after_exit)
    script = tmp_path / "dying_server.py"
    script.write_text(_DYING_STUB.format(marker=_DYING_MARKER))

    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=sys.executable, args=[str(script)], timeout=10.0)

    message = str(excinfo.value)
    assert "rc=1" in message
    assert "Traceback" in message
    assert _DYING_MARKER in message


def test_a_silent_dying_server_is_reported_by_its_exit_code_alone(tmp_path: Path) -> None:
    """A server that exits having written nothing to stderr gets no empty stderr clause."""
    script = tmp_path / "silent_server.py"
    script.write_text("import sys, time\ntime.sleep(0.5)\nsys.exit(1)\n")

    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=sys.executable, args=[str(script)], timeout=10.0)

    assert str(excinfo.value).endswith("rc=1 before responding to id=1")


def test_a_server_that_closes_its_stdin_and_keeps_running_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that stops reading but does not exit is a handshake failure, never a hang.

    The spawn is held until the child has closed its stdin, so the first write
    meets a closed pipe while the child is still alive.
    """
    flag = tmp_path / "stdin_closed"
    script = tmp_path / "deaf_server.py"
    script.write_text(
        "import os, pathlib, sys, time\n"
        "os.close(0)\n"
        f"pathlib.Path({str(flag)!r}).touch()\n"
        "time.sleep(30)\n"
    )
    real_popen = subprocess.Popen

    def popen_after_stdin_closed(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        deadline = time.monotonic() + 30
        while not flag.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen_after_stdin_closed)

    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=sys.executable, args=[str(script)], timeout=10.0)

    assert str(excinfo.value).endswith("server closed its stdin before reading id=1")


def test_a_finished_handshake_leaves_the_childs_stdin_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper closes the pipe it writes to, however the handshake ends."""
    real = subprocess.Popen
    made: list[subprocess.Popen] = []

    def spy(*a, **k):
        p = real(*a, **k)
        made.append(p)
        return p

    monkeypatch.setattr(subprocess, "Popen", spy)
    script = tmp_path / "dying_server.py"
    script.write_text(_DYING_STUB.format(marker=_DYING_MARKER))

    with pytest.raises(MCPHandshakeError):
        list_mcp_tools(command=sys.executable, args=[str(script)], timeout=10.0)

    assert len(made) == 1 and made[0].stdin is not None and made[0].stdin.closed


def _handshake_failure(command: str, args: list[str]) -> str:
    """The handshake error's text, with no frame left holding the helper's child."""
    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=command, args=args, timeout=10.0)
    return str(excinfo.value)


def test_a_server_whose_child_holds_its_pipes_leaves_no_group_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper ends the server's whole group, not only its leader."""
    record = tmp_path / "group"
    script = tmp_path / "forking_server.py"
    script.write_text(_FORKING_STUB.format(record=str(record), marker=_DYING_MARKER))
    real = subprocess.Popen
    leaders: list[int] = []

    def spy(*a, **k):
        p = real(*a, **k)
        leaders.append(p.pid)
        return p

    monkeypatch.setattr(subprocess, "Popen", spy)
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
    script = tmp_path / "dying_server.py"
    script.write_text(_DYING_STUB.format(marker=_DYING_MARKER))

    message = _handshake_failure(sys.executable, [str(script)])
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
