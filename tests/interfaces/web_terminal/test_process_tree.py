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
from tests.interfaces.web_terminal._pty_child import (
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
