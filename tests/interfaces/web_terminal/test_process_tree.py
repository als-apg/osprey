"""The process-tree reader ``PtySession.terminate`` uses, on real processes."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import pytest

from osprey.interfaces.web_terminal import process_tree
from osprey.interfaces.web_terminal.process_tree import ProcessGroup
from osprey.interfaces.web_terminal.pty_manager import PtySession
from tests.interfaces.web_terminal._chat_child import child_factory
from tests.interfaces.web_terminal._pty_child import (
    CHILD_HANG_CEILING,
    detaching_child_script,
    kill_quietly,
    pid_gone,
    sleeper_script,
    wait_for_pids,
)

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="no process groups on Windows"),
    pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps to read the process tree"),
]


def _own_row() -> process_tree.ProcessRow:
    rows = process_tree.snapshot()
    assert rows is not None
    return rows[os.getpid()]


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

        groups = {g.pgid: g for g in process_tree.tree_groups(child)}

        assert set(groups) == {grandchild, os.getpgid(child)}
        assert groups[grandchild].label == "magnet_scan.py"
        own = {pid for pid, _ in groups[os.getpgid(child)].members}
        assert {child, helper} <= own
    finally:
        session.terminate()
        kill_quietly(pids)


def test_tree_groups_orders_same_second_starts_by_pgid(monkeypatch):
    """Groups whose earliest members started in the same second come back by pgid."""
    # ``ps`` reports whole seconds, so commands started within one second compare equal.
    second = 1_700_000_000.0
    rows = {
        pid: process_tree.ProcessRow(pid, ppid, pgid, started, command, "S")
        for pid, ppid, pgid, started, command in [
            (4202, 1, 4202, second - 5.0, "agent"),
            (4208, 4202, 4208, second, "python orbit_poll.py"),
            (4201, 4202, 4201, second, "python magnet_scan.py"),
        ]
    }
    monkeypatch.setattr(process_tree, "snapshot", lambda: rows)
    monkeypatch.setattr(process_tree.os, "getpgrp", lambda: 100)

    groups = process_tree.tree_groups(4202)

    assert [g.pgid for g in groups] == [4202, 4201, 4208]
    assert [g.label for g in groups] == ["agent", "magnet_scan.py", "orbit_poll.py"]


@pytest.mark.parametrize(
    ("command", "label"),
    [
        ("python /x/magnet_scan.py --sector 3", "magnet_scan.py"),
        ("uv run python scan.py", "scan.py"),
        ("sleep 60", "sleep"),
        ("-zsh", "zsh"),
        ("", ""),
    ],
)
def test_label_for(command, label):
    assert process_tree.label_for(command) == label


def test_still_running_drops_a_group_whose_members_ended():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        rows = process_tree.snapshot()
        assert rows is not None
        row = rows[proc.pid]
        group = ProcessGroup(row.pgid, ((row.pid, row.started),), "sleep", row.command)
        assert process_tree.still_running([group]) == [group]
    finally:
        proc.kill()
        proc.wait()
    assert pid_gone(proc.pid)
    assert process_tree.still_running([group]) == []


def test_still_running_drops_a_group_whose_members_are_unreaped_zombies():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        rows = process_tree.snapshot()
        assert rows is not None
        row = rows[proc.pid]
        group = ProcessGroup(row.pgid, ((row.pid, row.started),), "python", row.command)
        assert proc.stdin is not None
        proc.stdin.close()
        deadline = time.monotonic() + 10
        state = ""
        while not state.startswith("Z"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(proc.pid)], capture_output=True, text=True
            ).stdout.strip()

        assert process_tree.still_running([group]) == []
    finally:
        proc.kill()
        proc.wait()


def test_still_running_drops_a_reused_group_id():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 10
        rows = process_tree.snapshot()
        while rows is None or proc.pid not in rows:
            assert time.monotonic() < deadline
            rows = process_tree.snapshot()
        row = rows[proc.pid]
        group = ProcessGroup(row.pgid, ((row.pid, row.started + 1.0),), "sleep", row.command)
        assert process_tree.still_running([group], rows) == []
    finally:
        proc.kill()
        proc.wait()


def test_end_groups_never_signals_the_servers_own_group(monkeypatch):
    row = _own_row()
    group = ProcessGroup(os.getpgrp(), ((row.pid, row.started),), "pytest", row.command)
    signalled: list[int] = []
    monkeypatch.setattr(process_tree, "END_TERM_WAIT_S", 0.0)
    monkeypatch.setattr(process_tree, "END_KILL_WAIT_S", 0.0)
    monkeypatch.setattr(process_tree.os, "killpg", lambda pgid, sig: signalled.append(pgid))

    process_tree.end_groups([group])

    assert os.getpgrp() not in signalled


def test_no_ps_finds_nothing_and_drops_nothing(monkeypatch):
    def no_ps(*args, **kwargs):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(process_tree.subprocess, "run", no_ps)
    groups = [ProcessGroup(4242, ((4242, 0.0),), "scan.py", "python scan.py")]

    assert process_tree.tree_groups(os.getpid()) == []
    assert process_tree.still_running(groups) == groups


def test_describe_escapes_labels():
    text = process_tree.describe([ProcessGroup(4242, ((4242, 0.0),), "a\nb", "a\nb")])
    assert "\n" not in text
    assert "\\n" in text


# ---------------------------------------------------------------------------
# A chat child: started groups and the server's own group
# ---------------------------------------------------------------------------


def _read_pids(path, count: int) -> list[int]:
    deadline = time.monotonic() + CHILD_HANG_CEILING
    while True:
        if path.exists():
            lines = path.read_text().split()
            if len(lines) >= count:
                return [int(line) for line in lines[:count]]
        assert time.monotonic() < deadline, f"no {count} pid(s) in {path.name}"
        time.sleep(0.05)


async def test_server_group_members_are_the_roots_descendants_in_the_servers_group(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    pids: list[int] = []
    async with factory() as client:
        root = client._transport._process.pid
        try:
            pids = _read_pids(factory.pid_files[0], 2)
            grandchild, helper = pids

            members = {row.pid for row in process_tree.server_group_members(root)}

            assert helper in members
            assert root not in members
            assert grandchild not in members
        finally:
            kill_quietly(pids)


async def test_started_groups_leave_out_the_roots_own_group(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    pids: list[int] = []
    async with factory() as client:
        root = client._transport._process.pid
        try:
            pids = _read_pids(factory.pid_files[0], 2)
            grandchild = pids[0]

            groups = process_tree.started_groups(root)

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

        pgids = {g.pgid for g in process_tree.started_groups(child)}

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
        deadline = time.monotonic() + 10
        rows = process_tree.snapshot()
        while rows is None or proc.pid not in rows:
            assert time.monotonic() < deadline
            rows = process_tree.snapshot()
        live = rows[proc.pid]
        reused = process_tree.ProcessRow(
            live.pid, live.ppid, live.pgid, live.started + 1.0, live.command, live.state
        )
        signalled: list[int] = []
        monkeypatch.setattr(process_tree.os, "kill", lambda pid, sig: signalled.append(pid))

        ended, survivors = process_tree.end_processes([reused])

        assert signalled == []
        assert ended == [] and survivors == []
    finally:
        proc.kill()
        proc.wait()


def test_no_ps_finds_no_server_group_members(monkeypatch):
    def no_ps(*args, **kwargs):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(process_tree.subprocess, "run", no_ps)

    assert process_tree.server_group_members(os.getpid()) == []


def test_describe_processes_escapes_labels():
    text = process_tree.describe_processes(
        [process_tree.ProcessRow(4242, 1, 4242, 0.0, "scan\x1b[2J", "S")]
    )
    assert text.startswith("pid 4242 ")
    assert "\x1b" not in text
    assert "\\x1b" in text
