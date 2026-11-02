"""The process-tree reader ``PtySession.terminate`` uses, on real processes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import psutil
import pytest

from osprey.deployment import process_table
from osprey.deployment.subprocess_capture import group_is_gone
from osprey.interfaces.web_terminal import process_tree
from osprey.interfaces.web_terminal.process_tree import ProcessGroup
from osprey.interfaces.web_terminal.pty_manager import PtySession
from tests._live import wait_until
from tests._processes import kill_quietly
from tests.interfaces.web_terminal._chat_child import child_factory
from tests.interfaces.web_terminal._pty_child import (
    CHILD_HANG_CEILING,
    detaching_child_script,
    readable_rows,
    sleeper_script,
    wait_for_pids,
)

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="no process groups on Windows"),
]


def _own_row() -> process_tree.ProcessRow:
    return readable_rows(holding=[os.getpid()])[os.getpid()]


def test_a_started_group_and_the_roots_own_group_are_found(tmp_path):
    pid_file = tmp_path / "pids"
    script = sleeper_script(tmp_path, "magnet_scan.py")
    session = PtySession([sys.executable, "-c", detaching_child_script(pid_file, scripts=[script])])
    session.start()
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 2)
        grandchild, helper = pids
        child = session.pid
        assert child is not None

        rows = readable_rows(holding=[child, grandchild, helper])
        groups = {g.pgid: g for g in process_tree.tree_groups(child, rows)}

        assert set(groups) == {grandchild, os.getpgid(child)}
        assert groups[grandchild].label == "magnet_scan.py"
        own = {pid for pid, _ in groups[os.getpgid(child)].members}
        assert {child, helper} <= own
    finally:
        session.terminate()
        kill_quietly(pids)


def test_tree_groups_orders_same_second_starts_by_pgid(monkeypatch):
    """Groups whose earliest members started in the same second come back by pgid."""
    # Commands started within one second carry the same whole-second start.
    second = 1_700_000_000.0
    rows = {
        pid: process_tree.ProcessRow(pid, ppid, pgid, started, command, "S")
        for pid, ppid, pgid, started, command in [
            (4202, 1, 4202, second - 5.0, "agent"),
            (4208, 4202, 4208, second, "python orbit_poll.py"),
            (4201, 4202, 4201, second, "python magnet_scan.py"),
        ]
    }
    monkeypatch.setattr(process_tree.os, "getpgrp", lambda: 100)

    groups = process_tree.tree_groups(4202, rows)

    assert [g.pgid for g in groups] == [4202, 4201, 4208]
    assert [g.label for g in groups] == ["agent", "magnet_scan.py", "orbit_poll.py"]


# Every row of one look carries a fixed, round start.
_SECOND = 1_700_000_000.0
_AGENT = "/home/op/.local/bin/claude --setting-sources project --mcp-config=.mcp.json"
# The agent CLI's shell wrapper as the process table lists it, around the command it evaluates.
_WRAPPER_HEAD = (
    "/bin/zsh -c source /home/op/.claude/shell-snapshots/snapshot-zsh-1791044269505-8e7vp7.sh "
    "2>/dev/null || true && setopt NO_EXTENDED_GLOB NO_BARE_GLOB_QUAL 2>/dev/null || true && eval "
)
_WRAPPER_TAIL = " < /dev/null && pwd -P >| /tmp/claude-4202-cwd"
_BEAT = "bash -c 'while true; do date >> /tmp/beat; sleep 1; done'"
_BEAT_FG = "bash -c 'for i in $(seq 300); do date >> /tmp/beat-fg; sleep 1; done'"


def _wrapper(command: str) -> str:
    """The wrapper's command line for *command*, quoted the way the agent CLI quotes it."""
    return _WRAPPER_HEAD + "'" + command.replace("'", "'\"'\"'") + "'" + _WRAPPER_TAIL


def _table(*rows: tuple[int, int, int, float, str]) -> dict[int, process_tree.ProcessRow]:
    """A process table of ``(pid, ppid, pgid, started, command)`` rows, all running."""
    return {
        pid: process_tree.ProcessRow(pid, ppid, pgid, started, command, "S")
        for pid, ppid, pgid, started, command in rows
    }


# The agent, the wrapper it started for a loop, and the shell the loop runs in.
_LOOP = (
    (4202, 1, 4202, _SECOND - 5.0, _AGENT),
    (4210, 4202, 4210, _SECOND, _wrapper(_BEAT)),
    (4211, 4210, 4210, _SECOND, "bash -c while true; do date >> /tmp/beat; sleep 1; done"),
)


def test_a_started_loop_is_named_the_same_on_every_look(monkeypatch):
    """The group is named by its leader, not by whichever child of the loop is alive."""
    monkeypatch.setattr(process_tree.os, "getpgrp", lambda: 100)
    seen = []
    for leaf in [
        (4230, 4211, 4210, _SECOND + 7.0, "sleep 1"),
        (4231, 4211, 4210, _SECOND + 8.0, "date"),
    ]:
        rows = _table(*_LOOP, leaf)
        (group,) = [g for g in process_tree.tree_groups(4202, rows) if g.pgid == 4210]
        seen.append(group.to_json())

    assert seen[0] == seen[1]
    assert seen[0] == {"label": "bash -c 'while true; do date >> /tmp/be…", "command": _BEAT}


def test_two_different_loops_are_named_differently(monkeypatch):
    rows = _table(
        *_LOOP,
        (4230, 4211, 4210, _SECOND + 7.0, "sleep 1"),
        (4212, 4202, 4212, _SECOND, _wrapper(_BEAT_FG)),
        (
            4213,
            4212,
            4212,
            _SECOND,
            "bash -c for i in $(seq 300); do date >> /tmp/beat-fg; sleep 1; done",
        ),
        (4232, 4213, 4212, _SECOND + 7.0, "sleep 1"),
    )
    monkeypatch.setattr(process_tree.os, "getpgrp", lambda: 100)

    groups = {g.pgid: g for g in process_tree.tree_groups(4202, rows)}

    assert groups[4210].to_json() == {
        "label": "bash -c 'while true; do date >> /tmp/be…",
        "command": _BEAT,
    }
    assert groups[4212].to_json() == {
        "label": "bash -c 'for i in $(seq 300); do date >…",
        "command": _BEAT_FG,
    }


def test_a_leader_that_is_no_wrapper_names_the_group_as_it_is(monkeypatch):
    rows = _table(
        (4202, 1, 4202, _SECOND - 5.0, _AGENT),
        (4210, 4202, 4210, _SECOND, "/x/bin/python /x/magnet_scan.py --sector 3"),
        (4230, 4210, 4210, _SECOND + 7.0, "sleep 5"),
    )
    monkeypatch.setattr(process_tree.os, "getpgrp", lambda: 100)

    groups = {g.pgid: g for g in process_tree.tree_groups(4202, rows)}

    assert groups[4210].to_json() == {
        "label": "magnet_scan.py",
        "command": "/x/bin/python /x/magnet_scan.py --sector 3",
    }
    assert groups[4202].label == "claude"


@pytest.mark.parametrize(
    ("command", "launched"),
    [
        (_wrapper(_BEAT), _BEAT),
        (_wrapper("python scan.py --sector 3"), "python scan.py --sector 3"),
        (_WRAPPER_HEAD + "ls" + _WRAPPER_TAIL, "ls"),
        # A command that reads its input is evaluated without the wrapper's own redirection.
        (_WRAPPER_HEAD + "'sort < in.txt' && pwd -P >| /tmp/claude-1-cwd", "sort < in.txt"),
        (_WRAPPER_HEAD + "'x && eval y' && pwd -P >| /tmp/claude-1-cwd", "x && eval y"),
        ("bash -c while true; do date >> /tmp/beat; sleep 1; done", None),
        ("/x/bin/python /x/magnet_scan.py", None),
        ("/bin/zsh -c echo hi && pwd -P >| /tmp/x", None),
        ("-zsh", None),
        ("", None),
    ],
)
def test_agent_command(command, launched):
    assert process_tree.agent_command(command) == launched


@pytest.mark.parametrize(
    ("command", "label"),
    [
        ("python /x/magnet_scan.py --sector 3", "magnet_scan.py"),
        ("uv run python scan.py", "scan.py"),
        ("sleep 60", "sleep"),
        ("-zsh", "zsh"),
        ("", ""),
        ("FOO=1 python scan.py", "scan.py"),
        ("claude --mcp-config=.mcp.json", "claude"),
        ("cd /x && make", "cd /x && make"),
        ("FOO=1 make", "FOO=1 make"),
        ("bash -c 'while true; do sleep 1; done'", "bash -c 'while true; do sleep 1; done'"),
        (
            "while true; do date >> /tmp/beat; sleep 1; done",
            "while true; do date >> /tmp/beat; sleep…",
        ),
    ],
)
def test_label_for(command, label):
    assert process_tree.label_for(command) == label


@pytest.mark.parametrize(
    ("command", "name"),
    [
        ("python /x/magnet_scan.py --sector 3", "magnet_scan.py"),
        ("sleep 60", "sleep"),
        ("while true; do date >> /tmp/beat; sleep 1; done", "while"),
        ("bash -c 'while true; do sleep 1; done'", "bash"),
        ("curl -H 'Authorization: Bearer token' https://x", "curl"),
        ("", ""),
    ],
)
def test_name_for(command, name):
    assert process_tree.name_for(command) == name


def test_still_running_drops_a_group_whose_members_ended():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        row = process_table.snapshot_within(5.0)[proc.pid]
        group = ProcessGroup(row.pgid, ((row.pid, row.started),), "sleep", row.command)
        rows = process_table.snapshot_within(5.0)
        assert process_tree.still_running([group], rows) == [group], rows.get(row.pid)
    finally:
        proc.kill()
        proc.wait(timeout=30)
    assert group_is_gone(row.pgid, within=5.0), row
    rows = process_table.snapshot_within(5.0)
    assert process_tree.still_running([group], rows) == [], rows.get(row.pid)


def test_still_running_drops_a_group_whose_members_are_unreaped_zombies():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        rows = readable_rows(holding=[proc.pid])
        row = rows[proc.pid]
        group = ProcessGroup(row.pgid, ((row.pid, row.started),), "python", row.command)
        assert proc.stdin is not None
        proc.stdin.close()
        wait_until(
            lambda: psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE,
            timeout=10,
            what="the child to become a zombie",
        )

        assert process_tree.still_running([group], readable_rows()) == []
    finally:
        proc.kill()
        proc.wait(timeout=30)


def test_still_running_drops_a_reused_group_id():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        rows = readable_rows(holding=[proc.pid])
        row = rows[proc.pid]
        group = ProcessGroup(row.pgid, ((row.pid, row.started + 1.0),), "sleep", row.command)
        assert process_tree.still_running([group], rows) == []
    finally:
        proc.kill()
        proc.wait(timeout=30)


def test_end_groups_never_signals_the_servers_own_group(monkeypatch):
    row = _own_row()
    group = ProcessGroup(os.getpgrp(), ((row.pid, row.started),), "pytest", row.command)
    signalled: list[int] = []
    monkeypatch.setattr(process_tree, "END_TERM_WAIT_S", 0.0)
    monkeypatch.setattr(process_tree, "END_KILL_WAIT_S", 0.0)
    monkeypatch.setattr(process_tree.os, "killpg", lambda pgid, sig: signalled.append(pgid))

    process_tree.end_groups([group])

    assert os.getpgrp() not in signalled


def _failing_reads(exc: BaseException):
    """A ``process_iter`` that fails with *exc* once iterated."""

    def process_iter(*args, **kwargs):
        raise exc
        yield  # a generator, as ``psutil.process_iter`` is

    return process_iter


def test_no_process_table_is_unknown_to_every_tree_reader(monkeypatch):
    monkeypatch.setattr(process_table.psutil, "process_iter", _failing_reads(psutil.AccessDenied()))

    with pytest.raises(process_table.ProcessTableUnreadable) as caught:
        process_tree.read_while_alive(lambda: True)
    assert not caught.value.retryable
    with pytest.raises(process_table.ProcessTableUnreadable):
        process_tree.started_groups(os.getpid(), lambda: True)


def test_end_groups_signals_nothing_when_the_table_cannot_be_read(monkeypatch):
    monkeypatch.setattr(process_table.psutil, "process_iter", _failing_reads(psutil.AccessDenied()))
    group = ProcessGroup(4242, ((4242, 0.0),), "scan.py", "python scan.py")
    signalled: list[int] = []
    monkeypatch.setattr(process_tree.os, "killpg", lambda pgid, sig: signalled.append(pgid))

    ending = process_tree.end_groups([group])

    assert signalled == []
    assert ending == process_tree.Ending(unknown=(group,))


def _reads_fail_after_the_first(patch) -> None:
    """Let the first process-table read reach the table and make every later one fail."""
    real = process_table.snapshot
    reads: list[None] = []

    def first_read_only():
        reads.append(None)
        if len(reads) > 1:
            raise process_table.ProcessTableUnreadable(PermissionError("process table"))
        return real()

    patch.setattr(process_table, "snapshot", first_read_only)


def test_end_groups_reports_unknown_not_survivors_when_the_table_fails_after_sigterm(
    monkeypatch,
):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline() == b"ready\n"
        row = process_table.snapshot_within(5.0)[proc.pid]
        group = ProcessGroup(row.pgid, ((row.pid, row.started),), "python", row.command)
        signalled: list[tuple[int, int]] = []
        real_killpg = os.killpg

        def killpg(pgid: int, sig: int) -> None:
            signalled.append((pgid, sig))
            real_killpg(pgid, sig)

        # Scoped, not test-wide: ``os`` is the one module object, so a patch
        # left in place would also reach the ``proc.kill()`` below.
        with monkeypatch.context() as patch:
            patch.setattr(process_tree, "END_TERM_WAIT_S", 0.3)
            _reads_fail_after_the_first(patch)
            patch.setattr(process_tree.os, "killpg", killpg)
            ending = process_tree.end_groups([group])

        assert signalled == [(row.pgid, signal.SIGTERM)]
        assert ending.unknown == (group,)
        assert ending.survivors == ()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.kill()
        proc.wait(timeout=30)


def test_end_processes_signals_nothing_when_the_table_cannot_be_read(monkeypatch):
    monkeypatch.setattr(process_table.psutil, "process_iter", _failing_reads(psutil.AccessDenied()))
    row = process_tree.ProcessRow(4242, 1, 4242, 0.0, "python scan.py", "sleeping")
    signalled: list[int] = []
    monkeypatch.setattr(process_tree.os, "kill", lambda pid, sig: signalled.append(pid))

    ending = process_tree.end_processes([row])

    assert signalled == []
    assert ending == process_tree.Ending(unknown=(row,))


def test_started_groups_raises_when_the_table_cannot_be_read(monkeypatch):
    monkeypatch.setattr(process_table.psutil, "process_iter", _failing_reads(psutil.Error("x")))

    monkeypatch.setattr(process_tree, "READ_WAIT_S", 0.0)

    with pytest.raises(process_table.ProcessTableUnreadable):
        process_tree.started_groups(os.getpid(), lambda: True)


def test_describe_names_a_group_by_its_name_not_its_command_line():
    group = ProcessGroup(
        4242,
        ((4242, 0.0),),
        "curl -H 'Authorization: Bearer token' h…",
        "curl -H 'Authorization: Bearer token' https://x",
    )

    assert process_tree.describe([group]) == "pgid 4242 'curl' (pids 4242)"


def test_describe_escapes_names():
    text = process_tree.describe([ProcessGroup(4242, ((4242, 0.0),), "scan\x1b[2J", "scan\x1b[2J")])
    assert text.startswith("pgid 4242 ")
    assert "\x1b" not in text
    assert "\\x1b" in text


# ---------------------------------------------------------------------------
# A chat child: started groups and the server's own group
# ---------------------------------------------------------------------------


def _read_pids(path, count: int) -> list[int]:
    def pids() -> list[int] | None:
        lines = path.read_text().split() if path.exists() else []
        return [int(line) for line in lines[:count]] if len(lines) >= count else None

    return wait_until(pids, timeout=CHILD_HANG_CEILING, what=f"{count} pid(s) in {path.name}")


@pytest.mark.usefixtures("readable_process_table")
async def test_server_group_members_are_the_roots_descendants_in_the_servers_group(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    pids: list[int] = []
    async with factory() as client:
        root = client._transport._process.pid
        try:
            pids = _read_pids(factory.pid_files[0], 2)
            grandchild, helper = pids

            rows = readable_rows(holding=[root, *pids])
            members = {row.pid for row in process_tree.server_group_members(root, rows)}

            assert helper in members
            assert root not in members
            assert grandchild not in members
        finally:
            kill_quietly(pids)


@pytest.mark.usefixtures("readable_process_table")
async def test_started_groups_leave_out_the_roots_own_group(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    pids: list[int] = []
    async with factory() as client:
        root = client._transport._process.pid
        try:
            pids = _read_pids(factory.pid_files[0], 2)
            grandchild = pids[0]

            process = client._transport._process
            groups = process_tree.started_groups(root, lambda: process.returncode is None)

            assert [g.pgid for g in groups] == [grandchild]
            assert groups[0].label == "magnet_scan.py"
        finally:
            kill_quietly(pids)

    pid_file = tmp_path / "pty-pids"
    script = sleeper_script(tmp_path, "orbit_poll.py")
    session = PtySession([sys.executable, "-c", detaching_child_script(pid_file, scripts=[script])])
    session.start()
    pty_pids: list[int] = []
    try:
        pty_pids = wait_for_pids(session, pid_file, 2)
        child = session.pid
        assert child is not None

        groups = process_tree.started_groups(child, lambda: session.is_alive)
        pgids = {g.pgid for g in groups}

        assert pgids == {pty_pids[0]}
        assert os.getpgid(child) not in pgids
    finally:
        session.terminate()
        kill_quietly(pty_pids)


def test_end_processes_never_signals_the_server(monkeypatch):
    row = _own_row()
    signalled: list[int] = []
    monkeypatch.setattr(process_tree, "END_TERM_WAIT_S", 0.0)
    monkeypatch.setattr(process_tree, "END_KILL_WAIT_S", 0.0)
    monkeypatch.setattr(process_tree.os, "kill", lambda pid, sig: signalled.append(pid))

    process_tree.end_processes([row])

    assert os.getpid() not in signalled


def test_end_processes_skips_a_reused_pid(monkeypatch):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        live = readable_rows(holding=[proc.pid])[proc.pid]
        reused = process_tree.ProcessRow(
            live.pid, live.ppid, live.pgid, live.started + 1.0, live.command, live.state
        )
        signalled: list[int] = []
        # Scoped, not test-wide: ``os`` is the one module object, so a patch
        # left in place would also swallow the ``proc.kill()`` below and leave
        # ``proc.wait()`` sitting out the child's whole sleep.
        with monkeypatch.context() as patch:
            patch.setattr(process_tree.os, "kill", lambda pid, sig: signalled.append(pid))
            ending = process_tree.end_processes([reused])

        assert signalled == []
        assert ending.ended == () and ending.survivors == ()
    finally:
        proc.kill()
        proc.wait(timeout=30)


def _answering(monkeypatch, answers: list[dict | BaseException]) -> list[None]:
    """Answer each process-table read with the next of *answers*; return the log of reads.

    An exception in *answers* is a read that fails with it.
    """
    reads: list[None] = []

    def read():
        reads.append(None)
        answer = answers[len(reads) - 1]
        if isinstance(answer, BaseException):
            raise process_table.ProcessTableUnreadable(answer)
        return answer

    monkeypatch.setattr(process_table, "snapshot", read)
    monkeypatch.setattr(process_table, "_READ_RETRY_S", 0.0)
    return reads


def test_read_while_alive_returns_the_first_read_that_succeeds(monkeypatch):
    rows = {1: process_tree.ProcessRow(1, 0, 1, 0.0, "init", "sleeping")}
    reads = _answering(monkeypatch, [psutil.Error("busy"), rows, rows])

    assert process_tree.read_while_alive(lambda: True) is rows
    assert len(reads) == 2


def test_read_while_alive_stops_retrying_once_the_root_has_exited(monkeypatch):
    busy = psutil.Error("busy")
    reads = _answering(monkeypatch, [busy, busy, busy])

    with pytest.raises(process_tree.ProcessTableUnreadable) as caught:
        process_tree.read_while_alive(lambda: not reads)

    assert caught.value.cause is busy
    assert len(reads) == 1


def test_read_while_alive_takes_no_read_once_the_root_has_exited(monkeypatch):
    reads = _answering(monkeypatch, [{}])

    with pytest.raises(process_tree.ProcessTableUnreadable) as caught:
        process_tree.read_while_alive(lambda: False)

    assert isinstance(caught.value.cause, ProcessLookupError)
    assert not caught.value.retryable
    assert reads == []


@pytest.mark.parametrize(
    "missing", [FileNotFoundError("proc"), PermissionError("proc"), psutil.AccessDenied()]
)
def test_read_while_alive_does_not_retry_a_table_no_read_can_answer(monkeypatch, missing):
    reads = _answering(monkeypatch, [missing, {}])

    with pytest.raises(process_tree.ProcessTableUnreadable) as caught:
        process_tree.read_while_alive(lambda: True)

    assert caught.value.cause is missing
    assert len(reads) == 1


def test_read_while_alive_gives_up_once_its_wait_has_passed(monkeypatch):
    busy = psutil.Error("busy")
    reads = _answering(monkeypatch, [busy, busy, {}])
    monkeypatch.setattr(process_tree, "READ_WAIT_S", 0.0)

    with pytest.raises(process_tree.ProcessTableUnreadable) as caught:
        process_tree.read_while_alive(lambda: True)

    assert caught.value.cause is busy
    assert len(reads) == 1


def test_a_tree_read_from_a_given_table_takes_no_read_of_its_own(monkeypatch):
    own = _own_row()
    rows = {
        own.pid: own,
        7001: process_tree.ProcessRow(7001, own.pid, 7001, 5.0, "python scan.py", "sleeping"),
        7002: process_tree.ProcessRow(7002, own.pid, own.pgid, 6.0, "python helper.py", "sleeping"),
    }
    reads = _answering(monkeypatch, [])

    groups = process_tree.tree_groups(own.pid, rows)
    members = process_tree.server_group_members(own.pid, rows)

    assert [group.pgid for group in groups] == [7001]
    assert [row.pid for row in members] == [7002]
    assert reads == []


def test_describe_processes_escapes_labels():
    text = process_tree.describe_processes(
        [process_tree.ProcessRow(4242, 1, 4242, 0.0, "scan\x1b[2J", "S")]
    )
    assert text.startswith("pid 4242 ")
    assert "\x1b" not in text
    assert "\\x1b" in text
